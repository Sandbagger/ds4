/* Model-independent policy contract for logits-only terminal sessions.
 *
 * The hook-created sessions are deliberately CPU-only and carry deterministic
 * private state.  This test never opens a model or touches an accelerator; it
 * verifies the admission matrix and the terminal read/mutation boundary.
 */

#ifndef DS4_TEST_HOOKS
#define DS4_TEST_HOOKS
#endif
#include "ds4.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* This existing engine entry point is intentionally not part of ds4.h; the
 * policy test calls it to cover the internal eval-argmax dispatch boundary. */
int ds4_session_eval_argmax(ds4_session *s, int token, char *err, size_t errlen);

/* Leave room for every supported vocabulary while asking the session for the
 * actual size through the public copy operation. */
#define TEST_LOGIT_CAP 200000
#define TEST_LOGIT_SENTINEL 1234567.0f
#define TEST_CTX 8
#define TERMINAL_ERROR "session is logits-only terminal"

enum {
    TEST_SYNC_INVALID = 0,
    TEST_SYNC_ORDINARY = 1,
    TEST_SYNC_EXACT = 2,
};

static int g_checks;
static int g_failures;

static void count_progress(void *ud, const char *event, int current, int total) {
    (void)event;
    (void)current;
    (void)total;
    (*(int *)ud)++;
}

static bool count_cancel(void *ud) {
    (*(int *)ud)++;
    return true;
}

#define CHECK(condition, message) do {                                      \
    g_checks++;                                                             \
    if (!(condition)) {                                                     \
        fprintf(stderr, "FAIL: %s (line %d)\n", (message), __LINE__);     \
        g_failures++;                                                       \
    }                                                                       \
} while (0)

#define CHECKF(condition, format, ...) do {                                 \
    g_checks++;                                                             \
    if (!(condition)) {                                                     \
        fprintf(stderr, "FAIL: " format " (line %d)\n",                  \
                __VA_ARGS__, __LINE__);                                    \
        g_failures++;                                                       \
    }                                                                       \
} while (0)

static bool state_equal(const ds4_test_session_state *a,
                        const ds4_test_session_state *b) {
    return a->pos == b->pos &&
           a->checkpoint_valid == b->checkpoint_valid &&
           a->logits_only_terminal == b->logits_only_terminal &&
           a->token_hash == b->token_hash &&
           a->logit_hash == b->logit_hash;
}

static bool capture_state(const char *where, const ds4_session *s,
                          ds4_test_session_state *out) {
    const int rc = ds4_test_session_state_read(s, out);
    CHECKF(rc == 0, "%s: state hook returned %d", where, rc);
    return rc == 0;
}

static void check_state_unchanged(const char *where, ds4_session *s,
                                  const ds4_test_session_state *before) {
    ds4_test_session_state after;
    if (!capture_state(where, s, &after)) return;
    CHECKF(state_equal(before, &after), "%s: private state changed", where);
}

static void check_terminal_result(const char *where, int rc, const char *err) {
    CHECKF(rc == 1, "%s: expected rejection sentinel 1, rc=%d", where, rc);
    CHECKF(err && strstr(err, TERMINAL_ERROR) != NULL,
           "%s: missing terminal error, got '%s'", where,
           err ? err : "(null)");
}

static void check_terminal_negative(const char *where, int rc, const char *err) {
    CHECKF(rc == -1, "%s: expected rejection sentinel -1, rc=%d", where, rc);
    CHECKF(err && strstr(err, TERMINAL_ERROR) != NULL,
           "%s: missing terminal error, got '%s'", where,
           err ? err : "(null)");
}

static ds4_tokens prompt_for_len(int *storage, int len, int seed) {
    ds4_tokens prompt = { storage, len, len };
    for (int i = 0; i < len; i++) storage[i] = seed + i;
    return prompt;
}

static int copy_logits_for_test(ds4_session *s, float **out) {
    float *logits = malloc((size_t)TEST_LOGIT_CAP * sizeof(*logits));
    CHECK(logits != NULL, "allocate conservative logits buffer");
    if (!logits) {
        *out = NULL;
        return 0;
    }
    for (int i = 0; i < TEST_LOGIT_CAP; i++) logits[i] = TEST_LOGIT_SENTINEL;
    const int n = ds4_session_copy_logits(s, logits, TEST_LOGIT_CAP);
    CHECKF(n > 0 && n <= TEST_LOGIT_CAP,
           "copy_logits returned invalid vocabulary size %d", n);
    bool all_overwritten = n > 0 && n <= TEST_LOGIT_CAP;
    if (all_overwritten) {
        for (int i = 0; i < n; i++) {
            if (logits[i] == TEST_LOGIT_SENTINEL) {
                all_overwritten = false;
                break;
            }
        }
    }
    CHECK(all_overwritten,
          "copy_logits did not overwrite every returned sentinel slot");
    *out = logits;
    return n > 0 && n <= TEST_LOGIT_CAP ? n : 0;
}

static bool g_fixture_hashes_set[2];
static uint64_t g_fixture_token_hash[2];
static uint64_t g_fixture_logit_hash[2];

static bool create_session(ds4_session **out, bool terminal) {
    const int rc = ds4_test_session_create_policy(out, TEST_CTX, terminal);
    CHECKF(rc == 0 && out && *out != NULL,
           "create %s session failed rc=%d",
           terminal ? "terminal" : "inert", rc);
    if (rc != 0 || !out || !*out) return false;

    ds4_test_session_state state;
    const int state_rc = ds4_test_session_state_read(*out, &state);
    CHECKF(state_rc == 0, "create %s: state hook returned %d",
           terminal ? "terminal" : "inert", state_rc);
    if (state_rc != 0) {
        ds4_session_free(*out);
        *out = NULL;
        return false;
    }
    const int ctx = ds4_session_ctx(*out);
    const bool valid = state.pos > 0 && state.pos < ctx &&
                       state.checkpoint_valid &&
                       state.logits_only_terminal == terminal &&
                       state.token_hash != 0 && state.logit_hash != 0;
    CHECKF(state.pos > 0 && state.pos < ctx,
           "create %s: fixture position %d is not strictly between zero and context %d",
           terminal ? "terminal" : "inert", state.pos, ctx);
    CHECKF(state.checkpoint_valid,
           "create %s: fixture checkpoint is not valid",
           terminal ? "terminal" : "inert");
    CHECKF(state.logits_only_terminal == terminal,
           "create %s: terminal bit mismatch",
           terminal ? "terminal" : "inert");
    CHECKF(state.token_hash != 0,
           "create %s: token fingerprint is zero",
           terminal ? "terminal" : "inert");
    CHECKF(state.logit_hash != 0,
           "create %s: logit fingerprint is zero",
           terminal ? "terminal" : "inert");
    const int kind = terminal ? 1 : 0;
    if (!g_fixture_hashes_set[kind]) {
        g_fixture_hashes_set[kind] = true;
        g_fixture_token_hash[kind] = state.token_hash;
        g_fixture_logit_hash[kind] = state.logit_hash;
    } else {
        CHECKF(state.token_hash == g_fixture_token_hash[kind],
               "create %s: token fingerprint is not deterministic",
               terminal ? "terminal" : "inert");
        CHECKF(state.logit_hash == g_fixture_logit_hash[kind],
               "create %s: logit fingerprint is not deterministic",
               terminal ? "terminal" : "inert");
    }
    if (!valid) {
        ds4_session_free(*out);
        *out = NULL;
        return false;
    }
    return true;
}

static void test_sync_mode_matrix(void) {
    fprintf(stderr, "RUN: logits-only eligibility matrix\n");
    const ds4_backend backends[] = {
        DS4_BACKEND_METAL,
        DS4_BACKEND_CUDA,
        DS4_BACKEND_CPU,
    };
    const int ctx = TEST_CTX;
    const int non_equal_lengths[] = { ctx - 1, ctx + 1 };
    int rows = 0;

    /* Every boolean/backend combination is checked at both ordinary lengths. */
    for (int native_cuda = 0; native_cuda <= 1; native_cuda++) {
        for (int laguna = 0; laguna <= 1; laguna++) {
            for (size_t bi = 0; bi < sizeof(backends) / sizeof(backends[0]); bi++) {
                for (int session_distributed = 0;
                     session_distributed <= 1; session_distributed++) {
                    for (int engine_distributed = 0;
                         engine_distributed <= 1; engine_distributed++) {
                        for (int transport_tp = 0; transport_tp <= 1; transport_tp++) {
                            for (int cuda_tp = 0; cuda_tp <= 1; cuda_tp++) {
                                for (size_t li = 0;
                                     li < sizeof(non_equal_lengths) /
                                         sizeof(non_equal_lengths[0]); li++) {
                                    const int mode = ds4_test_logits_only_sync_mode(
                                        native_cuda != 0,
                                        laguna != 0,
                                        backends[bi],
                                        session_distributed != 0,
                                        engine_distributed != 0,
                                        transport_tp != 0,
                                        cuda_tp != 0,
                                        non_equal_lengths[li],
                                        ctx);
                                    CHECKF(mode == TEST_SYNC_ORDINARY,
                                           "non-equal row %d selected mode %d",
                                           rows, mode);
                                    rows++;
                                }
                                const bool eligible =
                                    native_cuda && laguna &&
                                    backends[bi] == DS4_BACKEND_CUDA &&
                                    !session_distributed &&
                                    !engine_distributed &&
                                    !transport_tp && !cuda_tp;
                                const int equal_mode =
                                    ds4_test_logits_only_sync_mode(
                                        native_cuda != 0,
                                        laguna != 0,
                                        backends[bi],
                                        session_distributed != 0,
                                        engine_distributed != 0,
                                        transport_tp != 0,
                                        cuda_tp != 0,
                                        ctx,
                                        ctx);
                                CHECKF(equal_mode == (eligible ? TEST_SYNC_EXACT
                                                               : TEST_SYNC_INVALID),
                                       "equal row %d selected mode %d eligible=%d",
                                       rows, equal_mode, eligible ? 1 : 0);
                                rows++;
                            }
                        }
                    }
                }
            }
        }
    }

    /* Named negatives make the CPU-only and ROCm enum alias constraints clear. */
    CHECK(ds4_test_logits_only_sync_mode(
              false, true, DS4_BACKEND_CPU, false, false, false, false,
              ctx, ctx) == TEST_SYNC_INVALID,
          "CPU-only equality must not select exact mode");
    CHECK(ds4_test_logits_only_sync_mode(
              false, true, DS4_BACKEND_CUDA, false, false, false, false,
              ctx, ctx) == TEST_SYNC_INVALID,
          "ROCm-build equality must not select exact mode");
    CHECK(ds4_test_logits_only_sync_mode(
              false, true, DS4_BACKEND_METAL, false, false, false, false,
              ctx, ctx) == TEST_SYNC_INVALID,
          "Metal equality must not infer native CUDA");
    CHECK(rows == 576, "eligibility matrix did not cover all 192 combinations");
}

static void test_allowed_reads(void) {
    fprintf(stderr, "RUN: terminal approved reads\n");
    ds4_session *terminal = NULL;
    if (!create_session(&terminal, true)) return;

    ds4_test_session_state before;
    if (!capture_state("allowed reads before", terminal, &before)) {
        ds4_session_free(terminal);
        return;
    }
    float *logits = NULL;
    const int vocab = copy_logits_for_test(terminal, &logits);
    if (vocab <= 0 || !logits) {
        free(logits);
        ds4_session_free(terminal);
        return;
    }
    const int top = ds4_session_argmax(terminal);
    CHECK(top >= 0 && top < vocab, "terminal argmax did not work");
    const int second = ds4_session_argmax_excluding(terminal, top);
    CHECK(second >= 0 && second < vocab && second != top,
          "terminal argmax_excluding did not work");

    ds4_token_score scores[3];
    memset(scores, 0xa5, sizeof(scores));
    CHECK(ds4_session_top_logprobs(terminal, scores, 3) == 3,
          "terminal top_logprobs did not work");
    CHECK(scores[0].id >= 0 && scores[0].id < vocab,
          "terminal top_logprobs returned no token");

    ds4_token_score score;
    memset(&score, 0xa5, sizeof(score));
    CHECK(ds4_session_token_logprob(terminal, 0, &score) == 1,
          "terminal token_logprob did not work");
    CHECK(score.id == 0, "terminal token_logprob returned wrong token");

    free(logits);
    const int pos = ds4_session_pos(terminal);
    const int ctx = ds4_session_ctx(terminal);
    CHECK(pos >= 0 && pos < ctx, "terminal hook must start below context");
    CHECK(ctx == TEST_CTX, "terminal hook returned wrong context");
    check_state_unchanged("approved reads", terminal, &before);
    ds4_session_free(terminal);
}

static void test_scalar_rejections(void) {
    fprintf(stderr, "RUN: terminal scalar mutation/dispatch table\n");
    ds4_session *terminal = NULL;
    if (!create_session(&terminal, true)) return;

    int short_storage[TEST_CTX + 1];
    int equal_storage[TEST_CTX + 1];
    ds4_tokens short_prompt = prompt_for_len(short_storage, TEST_CTX - 1, 10);
    ds4_tokens equal_prompt = prompt_for_len(equal_storage, TEST_CTX, 20);
    char err[256];
    ds4_test_session_state before;
    CHECK(capture_state("scalar baseline", terminal, &before),
          "capture scalar baseline");
    float *vocab_probe = NULL;
    const int vocab = copy_logits_for_test(terminal, &vocab_probe);
    free(vocab_probe);
    if (vocab <= 0) {
        ds4_session_free(terminal);
        return;
    }

    memset(err, 0, sizeof(err));
    check_terminal_result("ordinary sync", ds4_session_sync(
                              terminal, &short_prompt, err, sizeof(err)), err);
    check_state_unchanged("ordinary sync", terminal, &before);
    memset(err, 0, sizeof(err));
    check_terminal_result("ordinary exact-context sync", ds4_session_sync(
                              terminal, &equal_prompt, err, sizeof(err)), err);
    check_state_unchanged("ordinary exact-context sync", terminal, &before);

    memset(err, 0, sizeof(err));
    check_terminal_result("non-equal logits-only delegates ordinary",
                          ds4_session_sync_logits_only(
                              terminal, &short_prompt, err, sizeof(err)), err);
    check_state_unchanged("non-equal logits-only delegates ordinary",
                           terminal, &before);
    memset(err, 0, sizeof(err));
    check_terminal_result("repeat logits-only sync", ds4_session_sync_logits_only(
                              terminal, &equal_prompt, err, sizeof(err)), err);
    check_state_unchanged("repeat logits-only sync", terminal, &before);

    memset(err, 0, sizeof(err));
    check_terminal_result("direct eval", ds4_session_eval(
                              terminal, 7, err, sizeof(err)), err);
    check_state_unchanged("direct eval", terminal, &before);

    memset(err, 0, sizeof(err));
    check_terminal_negative("internal eval-argmax", ds4_session_eval_argmax(
                                terminal, 7, err, sizeof(err)), err);
    check_state_unchanged("internal eval-argmax", terminal, &before);

    int accepted[4] = { 101, 102, 103, 104 };
    const int accepted_before[4] = { 101, 102, 103, 104 };
    memset(err, 0, sizeof(err));
    const int spec_rc = ds4_session_eval_speculative_argmax(
        terminal, 7, 3, 2, accepted, 4, err, sizeof(err));
    check_terminal_negative("speculative eval", spec_rc, err);
    CHECK(memcmp(accepted, accepted_before, sizeof(accepted)) == 0,
          "speculative eval changed caller output");
    check_state_unchanged("speculative eval", terminal, &before);

    int drafts[2] = { 7, 8 };
    memset(err, 0, sizeof(err));
    check_terminal_result("TP speculative cycle", ds4_session_tp_spec_cycle(
                              terminal, drafts, 2, err, sizeof(err)), err);
    check_state_unchanged("TP speculative cycle", terminal, &before);

    memset(err, 0, sizeof(err));
    CHECK(ds4_session_rewrite_from_common(
              terminal, &short_prompt, 0, err, sizeof(err)) == -1,
          "rewrite must return sentinel -1");
    CHECK(strstr(err, TERMINAL_ERROR) != NULL,
          "rewrite must name terminal state");
    check_state_unchanged("rewrite", terminal, &before);

    const int common = ds4_session_common_prefix(terminal, NULL);
    CHECK(common == 0,
          "common-prefix terminal result must be sentinel 0");
    check_state_unchanged("common prefix", terminal, &before);
    CHECK(ds4_session_prefill_cap(terminal) == 0,
          "prefill-cap terminal result must be an empty sentinel");
    CHECK(ds4_session_tokens(terminal) == NULL,
          "tokens terminal result must be an empty sentinel");
    check_state_unchanged("neutral accessors", terminal, &before);

    float *new_logits = malloc((size_t)vocab * sizeof(*new_logits));
    CHECK(new_logits != NULL, "allocate set_logits input");
    if (new_logits) {
        for (int i = 0; i < vocab; i++) new_logits[i] = 77.0f;
        float *input_copy = malloc((size_t)vocab * sizeof(*input_copy));
        CHECK(input_copy != NULL, "allocate set_logits input copy");
        if (input_copy) memcpy(input_copy, new_logits,
                               (size_t)vocab * sizeof(*input_copy));
        CHECK(ds4_session_set_logits(terminal, new_logits, vocab) == 1,
              "set_logits must reject terminal state with sentinel 1");
        if (input_copy) {
            CHECK(memcmp(new_logits, input_copy,
                         (size_t)vocab * sizeof(*new_logits)) == 0,
                  "set_logits changed caller input");
            free(input_copy);
        }
        free(new_logits);
    }
    check_state_unchanged("set_logits", terminal, &before);

    uint64_t rng = UINT64_C(0x1122334455667788);
    const uint64_t rng_before = rng;
    const int sample_rc = ds4_session_sample(
        terminal, 0.7f, 8, 0.9f, 0.05f, &rng);
    CHECK(sample_rc == -1, "sampling must return sentinel -1");
    CHECK(rng == rng_before, "sampling changed caller RNG");
    check_state_unchanged("sampling", terminal, &before);

    const int power_before = ds4_session_power(terminal);
    CHECK(ds4_session_set_power(terminal, 73) == 1,
          "set_power must reject terminal state with sentinel 1");
    CHECK(ds4_session_power(terminal) == power_before,
          "set_power changed engine power");
    check_state_unchanged("set_power", terminal, &before);

    int progress_calls = 0;
    int display_calls = 0;
    int cancel_calls = 0;
    ds4_session_set_progress(terminal, count_progress, &progress_calls);
    ds4_session_set_display_progress(terminal, count_progress, &display_calls);
    ds4_session_set_cancel(terminal, count_cancel, &cancel_calls);
    ds4_session_report_progress(terminal, "test", 1, 1);
    CHECK(progress_calls == 0,
          "progress callback was dispatched by terminal operations");
    CHECK(display_calls == 0,
          "display-progress callback was dispatched by terminal operations");
    memset(err, 0, sizeof(err));
    check_terminal_result("sync after callback setters", ds4_session_sync(
                              terminal, &short_prompt, err, sizeof(err)), err);
    CHECK(cancel_calls == 0,
          "cancel callback was dispatched by terminal operations");
    check_state_unchanged("progress/cancel setters", terminal, &before);

    ds4_session_gpu_warmup(terminal);
    check_state_unchanged("GPU warmup", terminal, &before);

    ds4_session_rewind(terminal, 0);
    check_state_unchanged("rewind", terminal, &before);
    ds4_session_invalidate(terminal);
    check_state_unchanged("invalidation", terminal, &before);

    ds4_session_free(terminal);
}

static void test_layer_rejections(void) {
    fprintf(stderr, "RUN: terminal layer operations\n");
    ds4_session *terminal = NULL;
    if (!create_session(&terminal, true)) return;
    ds4_test_session_state before;
    if (!capture_state("layer baseline", terminal, &before)) {
        ds4_session_free(terminal);
        return;
    }
    char err[256];
    memset(err, 0, sizeof(err));
    check_terminal_result("layer-slice reset", ds4_session_layer_slice_reset(
                              terminal, err, sizeof(err)), err);
    check_state_unchanged("layer-slice reset", terminal, &before);

    int token = 7;
    float input_hc[1] = { 31.0f };
    float output_hc[1] = { 41.0f };
    float logits[1] = { 51.0f };
    const float output_hc_before = output_hc[0];
    const float logits_before = logits[0];
    memset(err, 0, sizeof(err));
    check_terminal_result("layer evaluation", ds4_session_eval_layer_slice(
                              terminal, &token, 1, 0, 0, 0,
                              input_hc, output_hc, false, logits,
                              err, sizeof(err)), err);
    CHECK(output_hc[0] == output_hc_before,
          "layer evaluation changed caller hidden output");
    CHECK(logits[0] == logits_before,
          "layer evaluation changed caller logits output");
    check_state_unchanged("layer evaluation", terminal, &before);

    memset(err, 0, sizeof(err));
    check_terminal_result("output-head evaluation",
                          ds4_session_eval_output_head_from_hc(
                              terminal, input_hc, 1, logits, err, sizeof(err)), err);
    CHECK(logits[0] == logits_before,
          "output-head evaluation changed caller logits output");
    check_state_unchanged("output-head evaluation", terminal, &before);
    ds4_session_free(terminal);
}

static FILE *sentinel_file(long *offset_out) {
    FILE *fp = tmpfile();
    CHECK(fp != NULL, "create temporary sentinel file");
    if (!fp) return NULL;
    const unsigned char marker[] = { 0xde, 0xad, 0xbe, 0xef };
    CHECK(fwrite(marker, 1, sizeof(marker), fp) == sizeof(marker),
          "write temporary sentinel file");
    CHECK(fflush(fp) == 0, "flush temporary sentinel file");
    if (offset_out) *offset_out = ftell(fp);
    return fp;
}

static void test_payload_rejections(void) {
    fprintf(stderr, "RUN: terminal payload/snapshot rejection table\n");
    ds4_session *terminal = NULL;
    if (!create_session(&terminal, true)) return;
    ds4_test_session_state before;
    if (!capture_state("payload baseline", terminal, &before)) {
        ds4_session_free(terminal);
        return;
    }
    char err[256];

    CHECK(ds4_session_payload_bytes(terminal) == 0,
          "payload-size query must return empty sentinel");
    check_state_unchanged("payload-size query", terminal, &before);

    ds4_session_payload_file staged = {
        (char *)"caller-owned-path", UINT64_C(0x0102030405060708)
    };
    const ds4_session_payload_file staged_before = staged;
    memset(err, 0, sizeof(err));
    check_terminal_result("stage payload", ds4_session_stage_payload(
                              terminal, &staged, err, sizeof(err)), err);
    CHECK(memcmp(&staged, &staged_before, sizeof(staged)) == 0,
          "stage payload changed caller output before rejection");
    check_state_unchanged("stage payload", terminal, &before);

    long save_offset = -1;
    FILE *save_fp = sentinel_file(&save_offset);
    if (save_fp) {
        memset(err, 0, sizeof(err));
        check_terminal_result("save payload", ds4_session_save_payload(
                                  terminal, save_fp, err, sizeof(err)), err);
        CHECK(ftell(save_fp) == save_offset,
              "save payload changed file offset before rejection");
        fclose(save_fp);
    }
    check_state_unchanged("save payload", terminal, &before);

    long load_offset = -1;
    FILE *load_fp = sentinel_file(&load_offset);
    int token = 7;
    if (load_fp) {
        CHECK(fseek(load_fp, 0, SEEK_SET) == 0, "rewind payload load sentinel");
        load_offset = ftell(load_fp);
        memset(err, 0, sizeof(err));
        check_terminal_result("load payload", ds4_session_load_payload(
                                  terminal, load_fp, 4, err, sizeof(err)), err);
        CHECK(ftell(load_fp) == load_offset,
              "load payload advanced file before rejection");
        fclose(load_fp);
    }
    check_state_unchanged("load payload", terminal, &before);

    unsigned char snapshot_bytes[16];
    memset(snapshot_bytes, 0xa6, sizeof(snapshot_bytes));
    unsigned char snapshot_before[sizeof(snapshot_bytes)];
    memcpy(snapshot_before, snapshot_bytes, sizeof(snapshot_bytes));
    ds4_session_snapshot snapshot = {
        snapshot_bytes, sizeof(snapshot_bytes), sizeof(snapshot_bytes)
    };
    const ds4_session_snapshot snapshot_struct_before = snapshot;
    memset(err, 0, sizeof(err));
    check_terminal_result("save snapshot", ds4_session_save_snapshot(
                              terminal, &snapshot, err, sizeof(err)), err);
    CHECK(memcmp(&snapshot, &snapshot_struct_before, sizeof(snapshot)) == 0,
          "save snapshot changed caller descriptor before rejection");
    CHECK(memcmp(snapshot_bytes, snapshot_before, sizeof(snapshot_bytes)) == 0,
          "save snapshot changed caller buffer before rejection");
    check_state_unchanged("save snapshot", terminal, &before);

    memset(err, 0, sizeof(err));
    check_terminal_result("load snapshot", ds4_session_load_snapshot(
                              terminal, &snapshot, err, sizeof(err)), err);
    CHECK(memcmp(&snapshot, &snapshot_struct_before, sizeof(snapshot)) == 0,
          "load snapshot changed caller descriptor before rejection");
    CHECK(memcmp(snapshot_bytes, snapshot_before, sizeof(snapshot_bytes)) == 0,
          "load snapshot changed caller buffer before rejection");
    check_state_unchanged("load snapshot", terminal, &before);

    CHECK(ds4_session_layer_payload_bytes(terminal, 0, 0) == 0,
          "layer payload-size query must return empty sentinel");
    check_state_unchanged("layer payload-size query", terminal, &before);

    long layer_save_offset = -1;
    FILE *layer_save_fp = sentinel_file(&layer_save_offset);
    if (layer_save_fp) {
        memset(err, 0, sizeof(err));
        check_terminal_result("save layer payload",
                              ds4_session_save_layer_payload(
                                  terminal, layer_save_fp, 0, 0,
                                  err, sizeof(err)), err);
        CHECK(ftell(layer_save_fp) == layer_save_offset,
              "save layer payload changed file before rejection");
        fclose(layer_save_fp);
    }
    check_state_unchanged("save layer payload", terminal, &before);

    long layer_load_offset = -1;
    FILE *layer_load_fp = sentinel_file(&layer_load_offset);
    if (layer_load_fp) {
        CHECK(fseek(layer_load_fp, 0, SEEK_SET) == 0,
              "rewind layer payload load sentinel");
        layer_load_offset = ftell(layer_load_fp);
        memset(err, 0, sizeof(err));
        check_terminal_result("load layer payload",
                              ds4_session_load_layer_payload(
                                  terminal, layer_load_fp, 4, &token, 1, 0, 0,
                                  err, sizeof(err)), err);
        CHECK(ftell(layer_load_fp) == layer_load_offset,
              "load layer payload advanced file before rejection");
        fclose(layer_load_fp);
    }
    check_state_unchanged("load layer payload", terminal, &before);
    ds4_session_free(terminal);
}

static void test_batch_rejections(void) {
    fprintf(stderr, "RUN: all-or-nothing ordinary and mixed batches\n");
    char err[256];

    /* This must be rejected before the count == 1 shortcut. */
    ds4_session *single_terminal = NULL;
    if (create_session(&single_terminal, true)) {
        ds4_test_session_state single_before;
        capture_state("single-item batch baseline", single_terminal, &single_before);
        ds4_decode_item single_item = { single_terminal, 7 };
        memset(err, 0, sizeof(err));
        check_terminal_result("single-item terminal batch",
                              ds4_sessions_eval_batch(
                                  &single_item, 1, err, sizeof(err)), err);
        check_state_unchanged("single-item terminal batch",
                              single_terminal, &single_before);
    }
    ds4_session_free(single_terminal);

    ds4_session *normal = NULL;
    ds4_session *terminal = NULL;
    if (!create_session(&normal, false) || !create_session(&terminal, true)) {
        ds4_session_free(normal);
        ds4_session_free(terminal);
        return;
    }
    ds4_test_session_state normal_before;
    ds4_test_session_state terminal_before;
    capture_state("ordinary batch normal baseline", normal, &normal_before);
    capture_state("ordinary batch terminal baseline", terminal, &terminal_before);
    ds4_decode_item items[2] = {
        { normal, 7 },
        { terminal, 8 },
    };
    memset(err, 0, sizeof(err));
    check_terminal_result("ordinary batch terminal second",
                          ds4_sessions_eval_batch(items, 2, err, sizeof(err)), err);
    check_state_unchanged("ordinary batch normal first", normal, &normal_before);
    check_state_unchanged("ordinary batch terminal second", terminal, &terminal_before);
    ds4_session_free(normal);
    ds4_session_free(terminal);

    ds4_session *prefill = NULL;
    ds4_session *mixed_normal = NULL;
    ds4_session *mixed_terminal = NULL;
    if (!create_session(&prefill, false) ||
        !create_session(&mixed_normal, false) ||
        !create_session(&mixed_terminal, true)) {
        ds4_session_free(prefill);
        ds4_session_free(mixed_normal);
        ds4_session_free(mixed_terminal);
        return;
    }
    int prompt_storage[TEST_CTX + 1];
    /* The terminal scan must win even before ordinary prefill validation. */
    ds4_tokens prefill_prompt = prompt_for_len(
        prompt_storage, TEST_CTX - 1, 30);
    ds4_test_session_state prefill_before;
    ds4_test_session_state mixed_normal_before;
    ds4_test_session_state mixed_terminal_before;
    capture_state("mixed prefill baseline", prefill, &prefill_before);
    capture_state("mixed normal baseline", mixed_normal, &mixed_normal_before);
    capture_state("mixed terminal baseline", mixed_terminal, &mixed_terminal_before);
    ds4_decode_item mixed_items[2] = {
        { mixed_normal, 7 },
        { mixed_terminal, 8 },
    };
    memset(err, 0, sizeof(err));
    check_terminal_result("mixed batch terminal decode second",
                          ds4_sessions_eval_batch_with_prefill(
                              mixed_items, 2, prefill, &prefill_prompt,
                              err, sizeof(err)), err);
    check_state_unchanged("mixed batch prefill", prefill, &prefill_before);
    check_state_unchanged("mixed batch normal first", mixed_normal,
                           &mixed_normal_before);
    check_state_unchanged("mixed batch terminal second", mixed_terminal,
                           &mixed_terminal_before);

    /* The same scan must reject a terminal prefill before touching the decode. */
    ds4_session *terminal_prefill = NULL;
    ds4_session *prefill_normal = NULL;
    if (create_session(&terminal_prefill, true) &&
        create_session(&prefill_normal, false)) {
        ds4_test_session_state terminal_prefill_before;
        ds4_test_session_state prefill_normal_before;
        capture_state("terminal prefill baseline", terminal_prefill,
                      &terminal_prefill_before);
        capture_state("terminal-prefill decode baseline", prefill_normal,
                      &prefill_normal_before);
        ds4_decode_item one_item[1] = { { prefill_normal, 7 } };
        memset(err, 0, sizeof(err));
        check_terminal_result("mixed batch terminal prefill",
                              ds4_sessions_eval_batch_with_prefill(
                                  one_item, 1, terminal_prefill, &prefill_prompt,
                                  err, sizeof(err)), err);
        check_state_unchanged("terminal prefill", terminal_prefill,
                               &terminal_prefill_before);
        check_state_unchanged("terminal-prefill decode", prefill_normal,
                               &prefill_normal_before);
    }
    ds4_session_free(terminal_prefill);
    ds4_session_free(prefill_normal);
    ds4_session_free(prefill);
    ds4_session_free(mixed_normal);
    ds4_session_free(mixed_terminal);
}

int main(void) {
    test_sync_mode_matrix();
    test_allowed_reads();
    test_scalar_rejections();
    test_layer_rejections();
    test_payload_rejections();
    test_batch_rejections();
    fprintf(stderr, "%s: checks=%d failures=%d\n",
            g_failures == 0 ? "PASS" : "FAIL", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
