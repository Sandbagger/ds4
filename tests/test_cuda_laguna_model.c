/* Model-backed CUDA parity contract for poolside/Laguna-S-2.1-GGUF.
 *
 * This is intentionally outside `make test`: it needs the pinned 68 GB GGUF,
 * one pinned Poolside/llama.cpp oracle (vectors=4), and a CUDA machine.  The
 * test uses only the public engine/session API and stays red until resident
 * CUDA execution lands.
 *
 * Run with:
 *   DS4_TEST_MODEL=/path/to/laguna-s-2.1-Q4_K_M.gguf \
 *     make test-cuda-laguna-model
 */

#include "ds4.h"

#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define FIXTURE_DIR "tests/test-vectors/laguna-resident"
#define LAGUNA_VOCAB 100352
#define VECTOR_BYTES ((size_t)LAGUNA_VOCAB * sizeof(float))
#define CONTINUATION_COUNT 8
#define CUDA_CENTERED_RMS_MAX 0.04
#define CUDA_CENTERED_ABS_MAX 0.20
#define CUDA_TOP20_MIN 18

typedef struct {
    const char *id;
    float *poolside;
} oracle_case;

typedef struct {
    oracle_case cases[4];
    int continuation[CONTINUATION_COUNT];
} oracle_fixtures;

static const char *case_ids[] = {
    "short",
    "swa-513",
    "yarn-8193",
    "deep-32768",
};

static bool fail_message(const char *what, const char *detail) {
    fprintf(stderr, "FAIL: %s%s%s\n", what,
            detail && detail[0] ? ": " : "", detail && detail[0] ? detail : "");
    return false;
}

static uint32_t read_le_u32(const unsigned char *p) {
    return (uint32_t)p[0] |
           ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static unsigned char *read_exact_bytes(
        const char *path, size_t expected, bool nul_terminate) {
    struct stat st;
    if (stat(path, &st) != 0) {
        fprintf(stderr, "FAIL: stat %s: %s\n", path, strerror(errno));
        return NULL;
    }
    if (!S_ISREG(st.st_mode) || st.st_size < 0 || (uint64_t)st.st_size != expected) {
        fprintf(stderr, "FAIL: %s has %lld bytes, expected %zu regular-file bytes\n",
                path, (long long)st.st_size, expected);
        return NULL;
    }
    if (expected == SIZE_MAX && nul_terminate) {
        fprintf(stderr, "FAIL: %s is too large\n", path);
        return NULL;
    }

    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "FAIL: open %s: %s\n", path, strerror(errno));
        return NULL;
    }
    unsigned char *bytes = malloc(expected + (nul_terminate ? 1u : 0u));
    if (!bytes) {
        fclose(fp);
        fprintf(stderr, "FAIL: allocate %zu bytes for %s\n", expected, path);
        return NULL;
    }
    const size_t got = fread(bytes, 1, expected, fp);
    const int trailing = fgetc(fp);
    const int close_rc = fclose(fp);
    if (got != expected || trailing != EOF || close_rc != 0) {
        fprintf(stderr, "FAIL: exact read %s got=%zu expected=%zu\n",
                path, got, expected);
        free(bytes);
        return NULL;
    }
    if (nul_terminate) bytes[expected] = '\0';
    return bytes;
}

static char *read_text(const char *name) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s", FIXTURE_DIR, name);
    struct stat st;
    if (stat(path, &st) != 0) {
        fprintf(stderr, "FAIL: stat %s: %s\n", path, strerror(errno));
        return NULL;
    }
    if (!S_ISREG(st.st_mode) || st.st_size < 0 || (uint64_t)st.st_size >= SIZE_MAX) {
        fprintf(stderr, "FAIL: invalid text fixture %s\n", path);
        return NULL;
    }
    return (char *)read_exact_bytes(path, (size_t)st.st_size, true);
}

static float *read_vector(const char *case_id) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s.llama.f32",
             FIXTURE_DIR, case_id);
    unsigned char *bytes = read_exact_bytes(path, VECTOR_BYTES, false);
    if (!bytes) return NULL;

    float *values = malloc(VECTOR_BYTES);
    if (!values) {
        fprintf(stderr, "FAIL: allocate vector %s\n", path);
        free(bytes);
        return NULL;
    }
    for (int i = 0; i < LAGUNA_VOCAB; i++) {
        const uint32_t bits = read_le_u32(bytes + (size_t)i * 4u);
        memcpy(&values[i], &bits, sizeof(bits));
        if (!isfinite(values[i])) {
            fprintf(stderr, "FAIL: non-finite float in %s at vocab id %d\n",
                    path, i);
            free(values);
            free(bytes);
            return NULL;
        }
    }
    free(bytes);
    return values;
}

static void free_fixtures(oracle_fixtures *fixtures) {
    if (!fixtures) return;
    for (size_t i = 0; i < sizeof(fixtures->cases) / sizeof(fixtures->cases[0]); i++) {
        free(fixtures->cases[i].poolside);
        fixtures->cases[i].poolside = NULL;
    }
}

static bool load_fixtures(oracle_fixtures *fixtures) {
    memset(fixtures, 0, sizeof(*fixtures));
    for (size_t i = 0; i < sizeof(case_ids) / sizeof(case_ids[0]); i++) {
        fixtures->cases[i].id = case_ids[i];
        fixtures->cases[i].poolside = read_vector(case_ids[i]);
        if (!fixtures->cases[i].poolside) {
            free_fixtures(fixtures);
            return false;
        }
    }

    char path[512];
    snprintf(path, sizeof(path), "%s/yarn-8193.continuation.i32", FIXTURE_DIR);
    unsigned char *bytes = read_exact_bytes(
            path, CONTINUATION_COUNT * sizeof(int32_t), false);
    if (!bytes) {
        free_fixtures(fixtures);
        return false;
    }
    for (int i = 0; i < CONTINUATION_COUNT; i++) {
        const uint32_t raw = read_le_u32(bytes + (size_t)i * 4u);
        int32_t token = 0;
        memcpy(&token, &raw, sizeof(token));
        if (token < 0 || token >= LAGUNA_VOCAB) {
            fprintf(stderr, "FAIL: continuation token %d is out of range: %d\n",
                    i, token);
            free(bytes);
            free_fixtures(fixtures);
            return false;
        }
        fixtures->continuation[i] = token;
    }
    free(bytes);
    return true;
}

static int vector_argmax(const float *values) {
    int best_id = 0;
    for (int i = 1; i < LAGUNA_VOCAB; i++) {
        if (values[i] > values[best_id]) best_id = i;
    }
    return best_id;
}

static bool ranks_before(float value, int id, float other, int other_id) {
    return value > other || (value == other && id < other_id);
}

static void vector_top20(const float *values, int out[20]) {
    for (int i = 0; i < 20; i++) out[i] = -1;
    for (int id = 0; id < LAGUNA_VOCAB; id++) {
        int at = 0;
        while (at < 20 && out[at] >= 0 &&
               !ranks_before(values[id], id, values[out[at]], out[at])) {
            at++;
        }
        if (at == 20) continue;
        for (int j = 19; j > at; j--) out[j] = out[j - 1];
        out[at] = id;
    }
}

static int top20_overlap(const float *left, const float *right) {
    int left_ids[20];
    int right_ids[20];
    vector_top20(left, left_ids);
    vector_top20(right, right_ids);
    int overlap = 0;
    for (int i = 0; i < 20; i++) {
        for (int j = 0; j < 20; j++) {
            if (left_ids[i] == right_ids[j]) {
                overlap++;
                break;
            }
        }
    }
    return overlap;
}

static bool compare_one_oracle(
        const char *scenario,
        const char *oracle_name,
        const float *actual,
        const float *oracle,
        int session_argmax) {
    double actual_sum = 0.0;
    double oracle_sum = 0.0;
    for (int i = 0; i < LAGUNA_VOCAB; i++) {
        if (!isfinite(actual[i])) {
            fprintf(stderr, "FAIL: %s CUDA logit %d is non-finite\n", scenario, i);
            return false;
        }
        actual_sum += actual[i];
        oracle_sum += oracle[i];
    }
    const double actual_mean = actual_sum / LAGUNA_VOCAB;
    const double oracle_mean = oracle_sum / LAGUNA_VOCAB;
    double square_sum = 0.0;
    double max_abs = 0.0;
    for (int i = 0; i < LAGUNA_VOCAB; i++) {
        const double delta = ((double)actual[i] - actual_mean) -
                             ((double)oracle[i] - oracle_mean);
        square_sum += delta * delta;
        const double absolute = fabs(delta);
        if (absolute > max_abs) max_abs = absolute;
    }
    const double rms = sqrt(square_sum / LAGUNA_VOCAB);
    const int actual_argmax = vector_argmax(actual);
    const int oracle_argmax = vector_argmax(oracle);
    const int overlap = top20_overlap(actual, oracle);
    const bool ok = rms <= CUDA_CENTERED_RMS_MAX &&
                    max_abs <= CUDA_CENTERED_ABS_MAX &&
                    overlap >= CUDA_TOP20_MIN &&
                    actual_argmax == oracle_argmax &&
                    actual_argmax == session_argmax;
    fprintf(stderr,
            "%s oracle=%s centered_rms=%.9g centered_max=%.9g "
            "top20=%d argmax=%d/%d session_argmax=%d %s\n",
            scenario, oracle_name, rms, max_abs, overlap,
            actual_argmax, oracle_argmax, session_argmax,
            ok ? "PASS" : "FAIL");
    return ok;
}

static bool compare_session_oracle(
        ds4_session *session, const oracle_case *fixture, const char *scenario) {
    float *actual = malloc(VECTOR_BYTES);
    if (!actual) return fail_message("allocate CUDA logits", scenario);
    if (ds4_session_copy_logits(session, actual, LAGUNA_VOCAB) != LAGUNA_VOCAB) {
        free(actual);
        return fail_message("copy CUDA logits", scenario);
    }
    const int session_argmax = ds4_session_argmax(session);
    const bool ok = compare_one_oracle(
            scenario, "poolside", actual, fixture->poolside, session_argmax);
    free(actual);
    return ok;
}

static bool create_and_sync(
        ds4_engine *engine,
        const ds4_tokens *tokens,
        int context,
        ds4_session **out,
        const char *scenario) {
    char err[256] = {0};
    if (ds4_session_create(out, engine, context) != 0) {
        return fail_message("session create", scenario);
    }
    if (ds4_session_sync(*out, tokens, err, sizeof(err)) != 0) {
        fprintf(stderr, "FAIL: %s session sync: %s\n", scenario, err);
        ds4_session_free(*out);
        *out = NULL;
        return false;
    }
    return true;
}

static bool run_short(ds4_engine *engine, const oracle_case *fixture) {
    static const char expected[] =
        "Explain why a ring buffer wraps, in two sentences.\n";
    char *text = read_text("short.txt");
    if (!text) return false;
    if (strcmp(text, expected) != 0) {
        free(text);
        return fail_message("short.txt bytes changed", NULL);
    }
    ds4_tokens tokens = {0};
    ds4_encode_chat_prompt(engine, "", text, DS4_THINK_NONE, &tokens);
    free(text);
    ds4_session *session = NULL;
    const bool synced = create_and_sync(
            engine, &tokens, 1024, &session, "short");
    const bool ok = synced && compare_session_oracle(session, fixture, "short");
    ds4_session_free(session);
    ds4_tokens_free(&tokens);
    return ok;
}

static bool tokenize_raw_fixture(
        ds4_engine *engine,
        const char *file,
        int expected_tokens,
        ds4_tokens *tokens) {
    char *text = read_text(file);
    if (!text) return false;
    ds4_tokenize_rendered_chat(engine, text, tokens);
    free(text);
    if (tokens->len != expected_tokens) {
        fprintf(stderr, "FAIL: %s retokenized to %d IDs, expected %d\n",
                file, tokens->len, expected_tokens);
        ds4_tokens_free(tokens);
        return false;
    }
    return true;
}

static bool run_raw_frontier(
        ds4_engine *engine,
        const char *case_id,
        const char *prompt_file,
        int frontier,
        int context,
        const oracle_case *fixture,
        const int *continuation) {
    ds4_tokens tokens = {0};
    if (!tokenize_raw_fixture(engine, prompt_file, frontier, &tokens)) return false;
    ds4_session *session = NULL;
    if (!create_and_sync(engine, &tokens, context, &session, case_id)) {
        ds4_tokens_free(&tokens);
        return false;
    }
    bool ok = compare_session_oracle(session, fixture, case_id);
    if (ok && continuation) {
        char err[256] = {0};
        for (int step = 0; step < CONTINUATION_COUNT; step++) {
            const int actual = ds4_session_argmax(session);
            if (actual != continuation[step]) {
                fprintf(stderr,
                        "FAIL: %s teacher-forced step=%d argmax=%d expected=%d\n",
                        case_id, step, actual, continuation[step]);
                ok = false;
                break;
            }
            if (ds4_session_eval(
                    session, continuation[step], err, sizeof(err)) != 0) {
                fprintf(stderr, "FAIL: %s teacher-forced step=%d: %s\n",
                        case_id, step, err);
                ok = false;
                break;
            }
        }
    }
    ds4_session_free(session);
    ds4_tokens_free(&tokens);
    return ok;
}

static bool compare_sessions_bitwise(
        ds4_session *actual_session,
        ds4_session *control_session,
        const char *scenario,
        int row) {
    float *actual = malloc(VECTOR_BYTES);
    float *control = malloc(VECTOR_BYTES);
    if (!actual || !control) {
        free(actual);
        free(control);
        return fail_message("allocate batch comparison logits", scenario);
    }
    const bool copied =
        ds4_session_copy_logits(actual_session, actual, LAGUNA_VOCAB) == LAGUNA_VOCAB &&
        ds4_session_copy_logits(control_session, control, LAGUNA_VOCAB) == LAGUNA_VOCAB;
    if (!copied) {
        free(actual);
        free(control);
        return fail_message("copy batch comparison logits", scenario);
    }
    if (memcmp(actual, control, VECTOR_BYTES) != 0 ||
        ds4_session_argmax(actual_session) != ds4_session_argmax(control_session)) {
        int first = -1;
        for (int i = 0; i < LAGUNA_VOCAB; i++) {
            if (memcmp(&actual[i], &control[i], sizeof(float)) != 0) {
                first = i;
                break;
            }
        }
        fprintf(stderr,
                "FAIL: %s row=%d is not bit-exact first_id=%d values=%g/%g "
                "argmax=%d/%d\n",
                scenario, row, first,
                first >= 0 ? actual[first] : 0.0f,
                first >= 0 ? control[first] : 0.0f,
                ds4_session_argmax(actual_session),
                ds4_session_argmax(control_session));
        free(actual);
        free(control);
        return false;
    }
    free(actual);
    free(control);
    return true;
}

static bool load_decode_prompts(
        ds4_engine *engine, ds4_tokens prompts[2]) {
    char *short_text = read_text("short.txt");
    if (!short_text) return false;
    ds4_encode_chat_prompt(
            engine, "", short_text, DS4_THINK_NONE, &prompts[0]);
    free(short_text);
    if (!tokenize_raw_fixture(
            engine, "swa-513.prompt", 513, &prompts[1])) {
        ds4_tokens_free(&prompts[0]);
        return false;
    }
    return true;
}


static bool terminal_state_unchanged(
        ds4_session *session,
        const float *expected_logits,
        int expected_pos,
        const char *operation) {
    float *actual_logits = malloc(VECTOR_BYTES);
    if (!actual_logits) return fail_message("allocate terminal state logits", operation);
    if (ds4_session_copy_logits(session, actual_logits, LAGUNA_VOCAB) != LAGUNA_VOCAB) {
        free(actual_logits);
        return fail_message("copy terminal state logits", operation);
    }
    const int actual_pos = ds4_session_pos(session);
    const bool unchanged = actual_pos == expected_pos &&
                           memcmp(actual_logits, expected_logits, VECTOR_BYTES) == 0;
    if (!unchanged) {
        fprintf(stderr,
                "FAIL: terminal %s changed state pos=%d/%d logits=%s\n",
                operation, actual_pos, expected_pos,
                memcmp(actual_logits, expected_logits, VECTOR_BYTES) == 0 ?
                    "same" : "changed");
    }
    free(actual_logits);
    return unchanged;
}

static bool expect_terminal_rejection(
        ds4_session *session,
        const float *expected_logits,
        int expected_pos,
        int rc,
        const char *operation) {
    if (rc == 0) {
        fprintf(stderr, "FAIL: terminal %s unexpectedly succeeded\n", operation);
        return false;
    }
    return terminal_state_unchanged(
            session, expected_logits, expected_pos, operation);
}

static bool run_deep_exact_context(
        ds4_engine *engine, const oracle_case *fixture) {
    ds4_tokens tokens = {0};
    ds4_tokens control_prompt = {0};
    ds4_session *session = NULL;
    ds4_session *control = NULL;
    float *baseline_logits = NULL;
    float *control_logits = NULL;
    char err[256] = {0};
    bool ok = false;

    if (!tokenize_raw_fixture(
            engine, "deep-32768.prompt", 32768, &tokens)) goto done;
    if (ds4_session_create(&session, engine, 32768) != 0) {
        fail_message("session create", "deep-32768");
        goto done;
    }
    baseline_logits = malloc(VECTOR_BYTES);
    if (!baseline_logits) {
        fail_message("allocate initial logits", "deep-32768");
        goto done;
    }
    const int initial_pos = ds4_session_pos(session);
    if (ds4_session_copy_logits(session, baseline_logits, LAGUNA_VOCAB) !=
            LAGUNA_VOCAB) {
        fail_message("copy initial logits", "deep-32768");
        goto done;
    }

    memset(err, 0, sizeof(err));
    const int ordinary_rc = ds4_session_sync(
            session, &tokens, err, sizeof(err));
    if (!expect_terminal_rejection(
            session, baseline_logits, initial_pos, ordinary_rc,
            "deep-32768 ordinary equal-context sync")) goto done;

    memset(err, 0, sizeof(err));
    if (ds4_session_sync_logits_only(
            session, &tokens, err, sizeof(err)) != 0) {
        fprintf(stderr, "FAIL: deep-32768 logits-only sync: %s\n", err);
        goto done;
    }
    const int terminal_pos = ds4_session_pos(session);
    if (terminal_pos != 32768 ||
        ds4_session_copy_logits(session, baseline_logits, LAGUNA_VOCAB) !=
            LAGUNA_VOCAB) {
        fail_message("capture deep-32768 terminal state", NULL);
        goto done;
    }
#ifdef DS4_TEST_HOOKS
    ds4_test_session_state terminal_state = {0};
    if (ds4_test_session_state_read(session, &terminal_state) != 0 ||
        !terminal_state.logits_only_terminal ||
        !terminal_state.checkpoint_valid ||
        terminal_state.pos != 32768) {
        fail_message("deep-32768 terminal hook state", NULL);
        goto done;
    }
#endif
    if (!compare_session_oracle(session, fixture, "deep-32768")) goto done;

    memset(err, 0, sizeof(err));
    const int direct_token = ds4_session_argmax(session);
    const int direct_rc = ds4_session_eval(
            session, direct_token, err, sizeof(err));
    if (!expect_terminal_rejection(
            session, baseline_logits, terminal_pos, direct_rc,
            "deep-32768 direct decode")) goto done;

    ds4_session_rewind(session, terminal_pos - 1);
    if (!terminal_state_unchanged(
            session, baseline_logits, terminal_pos,
            "deep-32768 rewind")) goto done;
    memset(err, 0, sizeof(err));
    const int rewind_rc = ds4_session_eval(
            session, direct_token, err, sizeof(err));
    if (!expect_terminal_rejection(
            session, baseline_logits, terminal_pos, rewind_rc,
            "deep-32768 rewind followed by decode")) goto done;

    memset(err, 0, sizeof(err));
    const ds4_session_rewrite_result rewrite_rc =
        ds4_session_rewrite_from_common(
            session, &tokens, 0, err, sizeof(err));
    if (!expect_terminal_rejection(
            session, baseline_logits, terminal_pos, rewrite_rc,
            "deep-32768 rewrite")) goto done;

    memset(err, 0, sizeof(err));
    const int repeat_sync_rc = ds4_session_sync(
            session, &tokens, err, sizeof(err));
    if (!expect_terminal_rejection(
            session, baseline_logits, terminal_pos, repeat_sync_rc,
            "deep-32768 ordinary repeat sync")) goto done;

    memset(err, 0, sizeof(err));
    const int repeat_logits_only_rc = ds4_session_sync_logits_only(
            session, &tokens, err, sizeof(err));
    if (!expect_terminal_rejection(
            session, baseline_logits, terminal_pos, repeat_logits_only_rc,
            "deep-32768 logits-only repeat sync")) goto done;

    ds4_session_snapshot snapshot = {0};
    memset(err, 0, sizeof(err));
    const int snapshot_rc = ds4_session_save_snapshot(
            session, &snapshot, err, sizeof(err));
    if (!expect_terminal_rejection(
            session, baseline_logits, terminal_pos, snapshot_rc,
            "deep-32768 snapshot save")) {
        ds4_session_snapshot_free(&snapshot);
        goto done;
    }
    if (snapshot.ptr || snapshot.len != 0 || snapshot.cap != 0) {
        fprintf(stderr, "FAIL: deep-32768 snapshot output changed on rejection\n");
        ds4_session_snapshot_free(&snapshot);
        goto done;
    }
    ds4_session_snapshot_free(&snapshot);

    ds4_session_payload_file payload = {0};
    memset(err, 0, sizeof(err));
    const int payload_rc = ds4_session_stage_payload(
            session, &payload, err, sizeof(err));
    if (!expect_terminal_rejection(
            session, baseline_logits, terminal_pos, payload_rc,
            "deep-32768 staged payload export")) {
        ds4_session_payload_file_free(&payload);
        goto done;
    }
    if (payload.path || payload.bytes != 0) {
        fprintf(stderr, "FAIL: deep-32768 payload output changed on rejection\n");
        ds4_session_payload_file_free(&payload);
        goto done;
    }
    ds4_session_payload_file_free(&payload);

    if (!tokenize_raw_fixture(
            engine, "swa-513.prompt", 513, &control_prompt)) goto done;
    if (!create_and_sync(
            engine, &control_prompt, 1024, &control,
            "deep-32768 batch control")) goto done;
    control_logits = malloc(VECTOR_BYTES);
    if (!control_logits) {
        fail_message("allocate batch control logits", "deep-32768");
        goto done;
    }
    const int control_pos = ds4_session_pos(control);
    if (ds4_session_copy_logits(control, control_logits, LAGUNA_VOCAB) !=
            LAGUNA_VOCAB) {
        fail_message("copy batch control logits", "deep-32768");
        goto done;
    }
    /* Keep the normal member first: terminal admission must scan every item. */
    ds4_decode_item items[2] = {
        {control, ds4_session_argmax(control)},
        {session, ds4_session_argmax(session)},
    };
    memset(err, 0, sizeof(err));
    const int batch_rc = ds4_sessions_eval_batch(
            items, 2, err, sizeof(err));
    if (!expect_terminal_rejection(
            session, baseline_logits, terminal_pos, batch_rc,
            "deep-32768 decode batch with terminal session")) goto done;
    if (!terminal_state_unchanged(
            control, control_logits, control_pos,
            "deep-32768 decode batch control")) goto done;

    ok = true;
done:
    ds4_session_free(control);
    ds4_tokens_free(&control_prompt);
    free(control_logits);
    ds4_session_free(session);
    ds4_tokens_free(&tokens);
    free(baseline_logits);
    return ok;
}

static bool run_decode_batch(ds4_engine *engine) {
    ds4_tokens prompts[2] = {{0}};
    ds4_session *batched[2] = {0};
    ds4_session *controls[2] = {0};
    bool ok = load_decode_prompts(engine, prompts);
    for (int i = 0; ok && i < 2; i++) {
        ok = create_and_sync(
                     engine, &prompts[i], 1024, &batched[i], "decode-batch") &&
             create_and_sync(
                     engine, &prompts[i], 1024, &controls[i], "decode-control");
    }
    int tokens[2] = {0};
    ds4_decode_item items[2];
    for (int i = 0; ok && i < 2; i++) {
        tokens[i] = ds4_session_argmax(batched[i]);
        if (tokens[i] != ds4_session_argmax(controls[i])) {
            ok = fail_message("decode batch/control initial argmax", NULL);
            break;
        }
        items[i] = (ds4_decode_item){batched[i], tokens[i]};
    }
    char err[256] = {0};
    if (ok && ds4_sessions_eval_batch(items, 2, err, sizeof(err)) != 0) {
        ok = fail_message("two-session batch eval", err);
    }
    for (int i = 0; ok && i < 2; i++) {
        if (ds4_session_eval(controls[i], tokens[i], err, sizeof(err)) != 0) {
            ok = fail_message("serialized decode control", err);
        }
    }
    for (int i = 0; ok && i < 2; i++) {
        ok = compare_sessions_bitwise(
                batched[i], controls[i], "decode-batch", i);
    }
    for (int i = 0; i < 2; i++) {
        ds4_session_free(controls[i]);
        ds4_session_free(batched[i]);
        ds4_tokens_free(&prompts[i]);
    }
    return ok;
}

static bool run_mixed_batch(ds4_engine *engine) {
    ds4_tokens target = {0};
    if (!tokenize_raw_fixture(
            engine, "swa-513.prompt", 513, &target)) return false;
    ds4_tokens base = target;
    base.len = 512;
    ds4_session *prefill_mixed = NULL;
    ds4_session *prefill_control = NULL;
    bool ok = create_and_sync(
                      engine, &base, 1024, &prefill_mixed, "mixed-prefill") &&
              create_and_sync(
                      engine, &base, 1024, &prefill_control, "mixed-prefill-control");

    ds4_tokens prompts[2] = {{0}};
    ds4_session *mixed[2] = {0};
    ds4_session *controls[2] = {0};
    if (ok) ok = load_decode_prompts(engine, prompts);
    for (int i = 0; ok && i < 2; i++) {
        ok = create_and_sync(
                     engine, &prompts[i], 1024, &mixed[i], "mixed-decode") &&
             create_and_sync(
                     engine, &prompts[i], 1024, &controls[i], "mixed-control");
    }

    ds4_decode_item items[2];
    int tokens[2] = {0};
    for (int i = 0; ok && i < 2; i++) {
        tokens[i] = ds4_session_argmax(mixed[i]);
        if (tokens[i] != ds4_session_argmax(controls[i])) {
            ok = fail_message("mixed batch/control initial argmax", NULL);
            break;
        }
        items[i] = (ds4_decode_item){mixed[i], tokens[i]};
    }
    char err[256] = {0};
    if (ok && ds4_sessions_eval_batch_with_prefill(
                      items, 2, prefill_mixed, &target,
                      err, sizeof(err)) != 0) {
        ok = fail_message("mixed prefill/decode batch", err);
    }
    if (ok && ds4_session_sync(
                      prefill_control, &target, err, sizeof(err)) != 0) {
        ok = fail_message("serialized resumed prefill", err);
    }
    for (int i = 0; ok && i < 2; i++) {
        if (ds4_session_eval(controls[i], tokens[i], err, sizeof(err)) != 0) {
            ok = fail_message("serialized mixed decode", err);
        }
    }
    if (ok) {
        ok = compare_sessions_bitwise(
                prefill_mixed, prefill_control, "mixed-prefill", -1);
    }
    for (int i = 0; ok && i < 2; i++) {
        ok = compare_sessions_bitwise(mixed[i], controls[i], "mixed-decode", i);
    }

    for (int i = 0; i < 2; i++) {
        ds4_session_free(controls[i]);
        ds4_session_free(mixed[i]);
        ds4_tokens_free(&prompts[i]);
    }
    ds4_session_free(prefill_control);
    ds4_session_free(prefill_mixed);
    ds4_tokens_free(&target);
    return ok;
}

int main(void) {
    const char *model = getenv("DS4_TEST_MODEL");
    if (!model || !model[0]) {
        fprintf(stderr, "FAIL: DS4_TEST_MODEL is not set\n");
        return 1;
    }

    oracle_fixtures fixtures;
    if (!load_fixtures(&fixtures)) return 1;

    const ds4_engine_options options = {
        .model_path = model,
        .backend = DS4_BACKEND_CUDA,
        .n_threads = 1,
    };
    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &options) != 0) {
        free_fixtures(&fixtures);
        fprintf(stderr, "FAIL: open pinned Laguna model on explicit CUDA backend\n");
        return 1;
    }
    if (ds4_engine_vocab_size(engine) != LAGUNA_VOCAB) {
        fprintf(stderr, "FAIL: Laguna vocabulary=%d expected=%d\n",
                ds4_engine_vocab_size(engine), LAGUNA_VOCAB);
        ds4_engine_close(engine);
        free_fixtures(&fixtures);
        return 1;
    }

    bool ok = run_short(engine, &fixtures.cases[0]);
    if (ok) {
        ok = run_raw_frontier(
                engine, "swa-513", "swa-513.prompt", 513, 1024,
                &fixtures.cases[1], NULL);
    }
    if (ok) {
        ok = run_raw_frontier(
                engine, "yarn-8193", "yarn-8193.prompt", 8193, 8202,
                &fixtures.cases[2], fixtures.continuation);
    }
    /* The exact-context session is terminal and is freed before multi-session scenarios. */
    if (ok) ok = run_deep_exact_context(engine, &fixtures.cases[3]);
    if (ok) ok = run_decode_batch(engine);
    if (ok) ok = run_mixed_batch(engine);

    ds4_engine_close(engine);
    free_fixtures(&fixtures);
    if (!ok) return 1;
    fprintf(stderr,
            "test_cuda_laguna_model PASS oracle=poolside cases=4 vectors=4 continuation=8 "
            "decode_batch=2 mixed=1+2\n");
    return 0;
}
