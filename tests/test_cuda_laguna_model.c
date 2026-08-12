/* Model-backed CUDA parity contract for poolside/Laguna-S-2.1-GGUF.
 *
 * This is intentionally outside `make test`: it needs the pinned 68 GB GGUF,
 * one pinned Poolside oracle, and a CUDA machine.  The test uses only the
 * public engine/session API.  It is expected to stay red at the Laguna CUDA
 * engine admission gate until resident CUDA execution lands.
 *
 * Run with:
 *   DS4_TEST_MODEL=/path/to/laguna-s-2.1-Q4_K_M.gguf \
 *     make test-cuda-laguna-model
 */

#include "ds4.h"
#include "ds4_gpu.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
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
#define STREAMED_CACHE_BYTES (UINT64_C(8) * 1024u * 1024u * 1024u)
#define LAGUNA_ROUTED_LAYERS 47u
#define LAGUNA_ROUTED_EXPERTS 10u
#define WARM_STABILITY_TOLERANCE_BYTES \
    (UINT64_C(64) * 1024u * 1024u)
#define WARM_STABILITY_CYCLES 3u

typedef enum {
    MODEL_MODE_RESIDENT,
    MODEL_MODE_STREAMED,
} model_mode;

typedef enum {
    MODEL_CASE_ALL,
    MODEL_CASE_SHORT,
    MODEL_CASE_SWA,
    MODEL_CASE_CONTINUATION,
    MODEL_CASE_PREFILL_8192,
    MODEL_CASE_WARM_STABILITY,
} model_case_selection;

typedef struct {
    const char *id;
    float *poolside;
} oracle_case;

typedef struct {
    int pos;
    float *logits;
} session_state;

typedef struct {
    oracle_case cases[4];
    int continuation[CONTINUATION_COUNT];
} oracle_fixtures;

typedef struct {
    int count;
    int current[4];
    int total[4];
    bool overflow;
} prefill_progress_capture;

static const char *case_ids[] = {
    "short",
    "swa-513",
    "yarn-8193",
    "deep-32768",
};

static const char *model_mode_name(model_mode mode) {
    return mode == MODEL_MODE_STREAMED ? "streamed" : "resident";
}

static const char *model_case_name(model_case_selection selected) {
    switch (selected) {
        case MODEL_CASE_SHORT: return "short";
        case MODEL_CASE_SWA: return "swa-513";
        case MODEL_CASE_CONTINUATION: return "continuation";
        case MODEL_CASE_PREFILL_8192: return "prefill-8192";
        case MODEL_CASE_WARM_STABILITY: return "warm-stability";
        case MODEL_CASE_ALL: return "all";
    }
    return "unknown";
}

static void usage(const char *program) {
    fprintf(stderr,
            "Usage: %s [--mode resident|streamed "
            "--case short|swa-513|continuation|prefill-8192|"
            "warm-stability|all]\n"
            "       --case prefill-8192 and warm-stability are streamed-only\n"
            "       --case all is currently resident-only\n",
            program);
}

static bool parse_selection(
        int argc,
        char **argv,
        model_mode *mode,
        model_case_selection *selected) {
    if (!mode || !selected) return false;
    *mode = MODEL_MODE_RESIDENT;
    *selected = MODEL_CASE_ALL;
    if (argc == 1) return true;
    if (argc != 5) return false;

    bool mode_seen = false;
    bool case_seen = false;
    for (int i = 1; i < argc; i += 2) {
        if (strcmp(argv[i], "--mode") == 0 && !mode_seen) {
            mode_seen = true;
            if (strcmp(argv[i + 1], "resident") == 0) {
                *mode = MODEL_MODE_RESIDENT;
            } else if (strcmp(argv[i + 1], "streamed") == 0) {
                *mode = MODEL_MODE_STREAMED;
            } else {
                return false;
            }
        } else if (strcmp(argv[i], "--case") == 0 && !case_seen) {
            case_seen = true;
            if (strcmp(argv[i + 1], "short") == 0) {
                *selected = MODEL_CASE_SHORT;
            } else if (strcmp(argv[i + 1], "swa-513") == 0) {
                *selected = MODEL_CASE_SWA;
            } else if (strcmp(argv[i + 1], "continuation") == 0) {
                *selected = MODEL_CASE_CONTINUATION;
            } else if (strcmp(argv[i + 1], "prefill-8192") == 0) {
                *selected = MODEL_CASE_PREFILL_8192;
            } else if (strcmp(argv[i + 1], "warm-stability") == 0) {
                *selected = MODEL_CASE_WARM_STABILITY;
            } else if (strcmp(argv[i + 1], "all") == 0) {
                *selected = MODEL_CASE_ALL;
            } else {
                return false;
            }
        } else {
            return false;
        }
    }
    return mode_seen && case_seen &&
        !(*mode == MODEL_MODE_STREAMED && *selected == MODEL_CASE_ALL) &&
        !(*mode == MODEL_MODE_RESIDENT &&
          (*selected == MODEL_CASE_PREFILL_8192 ||
           *selected == MODEL_CASE_WARM_STABILITY));
}

static bool fail_message(const char *what, const char *detail) {
    fprintf(stderr, "FAIL: %s%s%s\n", what,
            detail && detail[0] ? ": " : "", detail && detail[0] ? detail : "");
    return false;
}

static bool inherited_model_fd(int *out, bool *set) {
    const char *value = getenv("DS4_TEST_MODEL_FD");
    *out = -1;
    *set = false;
    if (!value) {
        return true;
    }
    errno = 0;
    char *end = NULL;
    const long parsed = strtol(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' ||
        parsed < 0 || parsed > INT_MAX || fcntl((int)parsed, F_GETFD) == -1) {
        return fail_message("DS4_TEST_MODEL_FD is not an open descriptor", value);
    }
    *out = (int)parsed;
    *set = true;
    return true;
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
    snprintf(path, sizeof(path), "%s/%s.llama.f32", FIXTURE_DIR, case_id);
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

static bool verify_streamed_cache_contract(
        ds4_engine *engine,
        const char *scenario,
        uint64_t routed_tokens) {
    ds4_gpu_laguna_compact_test_snapshot compact;
    ds4_gpu_laguna_routed_origin_test_snapshot origins;
    memset(&compact, 0, sizeof(compact));
    memset(&origins, 0, sizeof(origins));
    if (!ds4_gpu_test_laguna_compact_active_snapshot(&compact)) {
        return fail_message("streamed compact snapshot is unavailable", scenario);
    }
    if (!ds4_gpu_test_laguna_compact_routed_origin_snapshot(&origins)) {
        return fail_message("streamed routed-origin audit is unavailable", scenario);
    }
    ds4_runtime_snapshot runtime;
    ds4_runtime_allocation_record runtime_records[256];
    size_t runtime_required = 0;
    memset(&runtime, 0, sizeof(runtime));
    memset(runtime_records, 0, sizeof(runtime_records));
    if (!ds4_test_engine_laguna_runtime_snapshot(
            engine,
            &runtime,
            runtime_records,
            sizeof(runtime_records) / sizeof(runtime_records[0]),
            &runtime_required)) {
        return fail_message("streamed runtime snapshot is unavailable", scenario);
    }

    const bool no_fallback =
        compact.model_mapping_registered_bytes == 0 &&
        compact.whole_model_copied_bytes == 0 &&
        compact.opportunistic_range_allocated_bytes == 0 &&
        compact.legacy_model_range_count == 0 &&
        compact.legacy_model_arena_count == 0;
    const uint64_t expected_projection_requests =
        routed_tokens * LAGUNA_ROUTED_LAYERS *
        LAGUNA_ROUTED_EXPERTS * 3u;
    const bool origin_exact =
        expected_projection_requests != 0 &&
        origins.routed_projection_requests == expected_projection_requests &&
        origins.engine_slot_resolutions ==
            origins.routed_projection_requests &&
        origins.static_slab_resolutions == 0 &&
        origins.model_mapping_resolutions == 0 &&
        origins.managed_resolutions == 0 &&
        origins.per_request_resolutions == 0 &&
        origins.unknown_resolutions == 0;
    const bool cache_healthy =
        compact.cache_acquire_misses != 0 &&
        compact.cache_load_successes == compact.cache_acquire_misses &&
        compact.cache_load_failures == 0 &&
        !compact.cache_unsafe &&
        compact.routed_payload_bytes != 0 &&
        compact.routed_payload_bytes <= compact.cache_payload_bytes;
    const bool graph_accounting_exact =
        runtime.violation == DS4_RUNTIME_VIOLATION_NONE &&
        runtime_required == 22u && runtime.active_record_count == 22u &&
        runtime.category_current[DS4_RUNTIME_CATEGORY_GRAPH_SCRATCH] == 0 &&
        runtime.category_current[DS4_RUNTIME_CATEGORY_KV_STATE] == 0 &&
        runtime.category_peak[DS4_RUNTIME_CATEGORY_GRAPH_SCRATCH] ==
            UINT64_C(1537052680) &&
        runtime.category_peak[DS4_RUNTIME_CATEGORY_KV_STATE] ==
            UINT64_C(1686110208);
    const bool ok = no_fallback && origin_exact && cache_healthy &&
        graph_accounting_exact;
    fprintf(stderr,
            "%s cache-origin requests=%llu engine_slots=%llu "
            "static=%llu model_map=%llu managed=%llu request=%llu "
            "unknown=%llu hits=%llu misses=%llu loads=%llu/%llu "
            "graph_peak=%llu kv_peak=%llu live=%zu %s\n",
            scenario,
            (unsigned long long)origins.routed_projection_requests,
            (unsigned long long)origins.engine_slot_resolutions,
            (unsigned long long)origins.static_slab_resolutions,
            (unsigned long long)origins.model_mapping_resolutions,
            (unsigned long long)origins.managed_resolutions,
            (unsigned long long)origins.per_request_resolutions,
            (unsigned long long)origins.unknown_resolutions,
            (unsigned long long)compact.cache_acquire_hits,
            (unsigned long long)compact.cache_acquire_misses,
            (unsigned long long)compact.cache_load_successes,
            (unsigned long long)compact.cache_load_failures,
            (unsigned long long)runtime.category_peak[
                DS4_RUNTIME_CATEGORY_GRAPH_SCRATCH],
            (unsigned long long)runtime.category_peak[
                DS4_RUNTIME_CATEGORY_KV_STATE],
            runtime.active_record_count,
            ok ? "PASS" : "FAIL");
    return ok;
}

static bool create_and_sync(
        ds4_engine *engine,
        const ds4_tokens *tokens,
        int context,
        ds4_session **out,
        const char *scenario) {
    char err[256] = {0};
    const uint32_t configured_chunk = ds4_engine_prefill_chunk(engine);
    const int session_context = configured_chunk != 0 ? 32768 : context;
    if (ds4_session_create(out, engine, session_context) != 0) {
        return fail_message("session create", scenario);
    }
    const int expected_cap = configured_chunk != 0 ?
        (int)configured_chunk :
        (session_context < 16384 ? session_context : 16384);
    if (ds4_session_prefill_cap(*out) != expected_cap) {
        ds4_session_free(*out);
        *out = NULL;
        return fail_message("session prefill cap", scenario);
    }
    if (ds4_session_sync(*out, tokens, err, sizeof(err)) != 0) {
        ds4_gpu_laguna_compact_test_snapshot compact = {0};
        ds4_runtime_snapshot runtime = {0};
        ds4_runtime_allocation_record records[256];
        size_t required = 0;
        memset(records, 0, sizeof(records));
        const bool compact_captured =
            ds4_gpu_test_laguna_compact_active_snapshot(&compact);
        const bool runtime_captured =
            ds4_test_engine_laguna_runtime_snapshot(
                engine, &runtime, records,
                sizeof(records) / sizeof(records[0]), &required);
        fprintf(stderr, "FAIL: %s session sync: %s\n", scenario, err);
        if (compact_captured || runtime_captured) {
            fprintf(stderr,
                    "  compact=%d runtime=%d unsafe=%d violation=%d "
                    "source_current=%llu source_peak=%llu precharge=%llu "
                    "post=%llu touches=%llu/%llu "
                    "advice_failures=%llu\n",
                    compact_captured ? 1 : 0,
                    runtime_captured ? 1 : 0,
                    compact_captured ? compact.cache_unsafe : -1,
                    runtime_captured ? (int)runtime.violation : -1,
                    (unsigned long long)(runtime_captured ?
                        runtime.report_current[
                            DS4_RUNTIME_REPORT_MODEL_SOURCE_RESIDENT] : 0u),
                    (unsigned long long)(runtime_captured ?
                        runtime.report_peak[
                            DS4_RUNTIME_REPORT_MODEL_SOURCE_RESIDENT] : 0u),
                    (unsigned long long)(compact_captured ?
                        compact.page_advice_precharge_source_resident_bytes : 0u),
                    (unsigned long long)(compact_captured ?
                        compact.page_advice_post_source_resident_bytes : 0u),
                    (unsigned long long)(compact_captured ?
                        compact.page_advice_mapping_touch_pages : 0u),
                    (unsigned long long)(compact_captured ?
                        compact.page_advice_mapping_touch_bytes : 0u),
                    (unsigned long long)(compact_captured ?
                        compact.page_advice_failed_calls : 0u));
        }
        ds4_session_free(*out);
        *out = NULL;
        return false;
    }
    return true;
}

static bool capture_quiescent_runtime(
        ds4_engine *engine,
        ds4_runtime_snapshot *snapshot,
        const char *scenario) {
    ds4_runtime_allocation_record records[256];
    size_t required = 0;
    memset(snapshot, 0, sizeof(*snapshot));
    memset(records, 0, sizeof(records));
    if (!ds4_test_engine_laguna_runtime_snapshot(
            engine, snapshot, records,
            sizeof(records) / sizeof(records[0]), &required)) {
        return fail_message("quiescent runtime snapshot", scenario);
    }
    if (snapshot->violation != DS4_RUNTIME_VIOLATION_NONE ||
        required != snapshot->active_record_count ||
        required > sizeof(records) / sizeof(records[0]) ||
        snapshot->category_current[DS4_RUNTIME_CATEGORY_GRAPH_SCRATCH] != 0 ||
        snapshot->category_current[DS4_RUNTIME_CATEGORY_KV_STATE] != 0) {
        fprintf(stderr,
                "FAIL: %s is not quiescent violation=%d live=%zu/%zu "
                "graph=%llu kv=%llu\n",
                scenario,
                (int)snapshot->violation,
                snapshot->active_record_count,
                required,
                (unsigned long long)snapshot->category_current[
                    DS4_RUNTIME_CATEGORY_GRAPH_SCRATCH],
                (unsigned long long)snapshot->category_current[
                    DS4_RUNTIME_CATEGORY_KV_STATE]);
        return false;
    }
    return true;
}

static uint64_t distance_u64(uint64_t left, uint64_t right) {
    return left >= right ? left - right : right - left;
}

/* A cold/warm numerical and lifetime gate over the public session API.  The
 * first pass establishes accepted output and cold routed-I/O cost.  The
 * identical second pass must reuse cache state without changing output.  A
 * further three full session lifecycles then prove that request-owned graph
 * and KV state returns to the same engine-owned baseline after every free. */
static bool run_warm_stability(
        ds4_engine *engine,
        const oracle_case *fixture,
        uint64_t *routed_tokens) {
    static const char expected[] =
        "Explain why a ring buffer wraps, in two sentences.\n";
    char *text = read_text("short.txt");
    if (!text) return false;
    if (strcmp(text, expected) != 0) {
        free(text);
        return fail_message("short.txt bytes changed", "warm-stability");
    }

    ds4_tokens tokens = {0};
    ds4_encode_chat_prompt(engine, "", text, DS4_THINK_NONE, &tokens);
    free(text);
    if (tokens.len <= 0) {
        ds4_tokens_free(&tokens);
        return fail_message("empty warm-stability prompt", NULL);
    }
    if (routed_tokens) {
        *routed_tokens =
            (uint64_t)tokens.len * (2u + WARM_STABILITY_CYCLES) +
            WARM_STABILITY_CYCLES;
    }

    bool ok = true;
    ds4_gpu_laguna_compact_test_snapshot initial = {0};
    ds4_gpu_laguna_compact_test_snapshot after_cold = {0};
    ds4_gpu_laguna_compact_test_snapshot after_warm = {0};
    if (!ds4_gpu_test_laguna_compact_active_snapshot(&initial)) {
        ok = fail_message("initial compact snapshot", "warm-stability");
    }

    ds4_session *session = NULL;
    if (ok) {
        ok = create_and_sync(
            engine, &tokens, 1024, &session, "warm-stability-cold");
    }
    if (!ok) {
        fprintf(stderr,
                "  startup precharge=%llu post=%llu touches=%llu/%llu\n",
                (unsigned long long)
                    initial.page_advice_precharge_source_resident_bytes,
                (unsigned long long)
                    initial.page_advice_post_source_resident_bytes,
                (unsigned long long)initial.page_advice_mapping_touch_pages,
                (unsigned long long)initial.page_advice_mapping_touch_bytes);
    }
    if (ok) {
        ok = compare_session_oracle(
            session, fixture, "warm-stability-cold");
    }
    const int cold_argmax = ok ? ds4_session_argmax(session) : -1;
    ds4_session_free(session);
    session = NULL;
    if (ok && !ds4_gpu_test_laguna_compact_active_snapshot(&after_cold)) {
        ok = fail_message("post-cold compact snapshot", "warm-stability");
    }

    if (ok) {
        ok = create_and_sync(
            engine, &tokens, 1024, &session, "warm-stability-warm");
    }
    if (ok) {
        ok = compare_session_oracle(
            session, fixture, "warm-stability-warm");
    }
    if (ok && ds4_session_argmax(session) != cold_argmax) {
        ok = fail_message(
            "cold and warm accepted output differ", "warm-stability");
    }
    ds4_session_free(session);
    session = NULL;
    if (ok && !ds4_gpu_test_laguna_compact_active_snapshot(&after_warm)) {
        ok = fail_message("post-warm compact snapshot", "warm-stability");
    }

    uint64_t cold_read_bytes = 0;
    uint64_t warm_read_bytes = 0;
    uint64_t warm_hits = 0;
    if (ok && (after_cold.model_file_read_bytes <
                   initial.model_file_read_bytes ||
               after_warm.model_file_read_bytes <
                   after_cold.model_file_read_bytes ||
               after_warm.cache_acquire_hits <
                   after_cold.cache_acquire_hits)) {
        ok = fail_message("non-monotonic cache counters", "warm-stability");
    }
    if (ok) {
        cold_read_bytes = after_cold.model_file_read_bytes -
            initial.model_file_read_bytes;
        warm_read_bytes = after_warm.model_file_read_bytes -
            after_cold.model_file_read_bytes;
        warm_hits = after_warm.cache_acquire_hits -
            after_cold.cache_acquire_hits;
        if (cold_read_bytes == 0 || warm_hits == 0 ||
            warm_read_bytes > cold_read_bytes) {
            fprintf(stderr,
                    "FAIL: warm-stability cache reuse cold_read=%llu "
                    "warm_read=%llu warm_hits=%llu\n",
                    (unsigned long long)cold_read_bytes,
                    (unsigned long long)warm_read_bytes,
                    (unsigned long long)warm_hits);
            ok = false;
        }
    }
    ds4_runtime_snapshot baseline = {0};
    ds4_runtime_snapshot cycles[WARM_STABILITY_CYCLES];
    memset(cycles, 0, sizeof(cycles));
    if (ok) {
        ok = capture_quiescent_runtime(
            engine, &baseline, "first post-warm result");
    }
    for (size_t cycle = 0; ok && cycle < WARM_STABILITY_CYCLES; cycle++) {
        char scenario[64];
        snprintf(scenario, sizeof(scenario),
                 "warm-stability-cycle-%zu", cycle + 1u);
        ok = create_and_sync(engine, &tokens, 1024, &session, scenario);
        if (ok && ds4_session_argmax(session) != cold_argmax) {
            ok = fail_message("warm cycle changed accepted output", scenario);
        }
        char error[256] = {0};
        if (ok && ds4_session_eval(
                      session, cold_argmax, error, sizeof(error)) != 0) {
            ok = fail_message("warm cycle decode", error);
        }
        ds4_session_free(session);
        session = NULL;
        if (ok) {
            ok = capture_quiescent_runtime(engine, &cycles[cycle], scenario);
        }
    }

    for (size_t category = 0;
         ok && category < DS4_RUNTIME_OWNED_CATEGORY_COUNT; category++) {
        bool ever_increased = false;
        bool ever_decreased = false;
        uint64_t previous = baseline.category_current[category];
        for (size_t cycle = 0; cycle < WARM_STABILITY_CYCLES; cycle++) {
            const uint64_t current = cycles[cycle].category_current[category];
            if (distance_u64(
                    baseline.category_current[category], current) >
                WARM_STABILITY_TOLERANCE_BYTES) {
                fprintf(stderr,
                        "FAIL: warm category=%zu cycle=%zu baseline=%llu "
                        "current=%llu exceeds=%llu\n",
                        category, cycle + 1u,
                        (unsigned long long)baseline.category_current[category],
                        (unsigned long long)current,
                        (unsigned long long)WARM_STABILITY_TOLERANCE_BYTES);
                ok = false;
                break;
            }
            if (current > previous) ever_increased = true;
            if (current < previous) ever_decreased = true;
            previous = current;
        }
        if (ok && ever_increased && !ever_decreased) {
            fprintf(stderr,
                    "FAIL: owned category=%zu grows monotonically after warm-up\n",
                    category);
            ok = false;
        }
    }

    if (ok) {
        for (size_t cycle = 0; cycle < WARM_STABILITY_CYCLES; cycle++) {
            if (cycles[cycle].active_record_count !=
                    baseline.active_record_count ||
                distance_u64(cycles[cycle].owned_total_current,
                             baseline.owned_total_current) >
                    WARM_STABILITY_TOLERANCE_BYTES) {
                ok = fail_message(
                    "session lifecycle did not return to warm baseline",
                    "warm-stability");
                break;
            }
        }
    }

    ds4_session_free(session);
    ds4_tokens_free(&tokens);
    if (ok) {
        fprintf(stderr,
                "warm-stability cycles=%u cold_read=%llu warm_read=%llu "
                "warm_hits=%llu baseline=%llu live=%zu PASS\n",
                WARM_STABILITY_CYCLES,
                (unsigned long long)cold_read_bytes,
                (unsigned long long)warm_read_bytes,
                (unsigned long long)warm_hits,
                (unsigned long long)baseline.owned_total_current,
                baseline.active_record_count);
    }
    return ok;
}

static bool run_short(
        ds4_engine *engine,
        const oracle_case *fixture,
        uint64_t *routed_tokens) {
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
    if (routed_tokens) *routed_tokens = (uint64_t)tokens.len;

    const char *diag_dir_env = getenv("DS4_LAGUNA_DIAG_DIR");
    if (diag_dir_env && diag_dir_env[0]) {
        char *diag_dir = strdup(diag_dir_env);
        float *baseline_logits = malloc(VECTOR_BYTES);
        float *probed_logits = malloc(VECTOR_BYTES);
        ds4_session *baseline = NULL;
        ds4_session *probed = NULL;
        bool ok = diag_dir && baseline_logits && probed_logits;
        if (ok && unsetenv("DS4_LAGUNA_DIAG_DIR") != 0) {
            ok = fail_message("disable Laguna layer diagnostic", NULL);
        }
        if (ok) {
            ok = create_and_sync(
                    engine, &tokens, 1024, &baseline, "short-diag-baseline");
        }
        if (ok && ds4_session_copy_logits(
                baseline, baseline_logits, LAGUNA_VOCAB) != LAGUNA_VOCAB) {
            ok = fail_message("copy baseline diagnostic logits", NULL);
        }
        if (diag_dir && setenv("DS4_LAGUNA_DIAG_DIR", diag_dir, 1) != 0) {
            ok = fail_message("enable Laguna layer diagnostic", NULL);
        }
        if (ok) {
            ok = create_and_sync(
                    engine, &tokens, 1024, &probed, "short-diag-probed");
        }
        if (ok && ds4_session_copy_logits(
                probed, probed_logits, LAGUNA_VOCAB) != LAGUNA_VOCAB) {
            ok = fail_message("copy probed diagnostic logits", NULL);
        }
        if (unsetenv("DS4_LAGUNA_DIAG_DIR") != 0) {
            ok = fail_message("consume Laguna layer diagnostic", NULL);
        }
        if (ok && memcmp(baseline_logits, probed_logits, VECTOR_BYTES) != 0) {
            ok = fail_message("Laguna layer diagnostic perturbed logits", NULL);
        }
        if (ok) {
            fprintf(stderr,
                    "short-layer-diag PASS files=62 nonperturbing=bit-exact\n");
        }
        ds4_session_free(probed);
        ds4_session_free(baseline);
        free(probed_logits);
        free(baseline_logits);
        free(diag_dir);
        ds4_tokens_free(&tokens);
        return ok;
    }

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
        const int *continuation,
        bool require_continuation_cache_hits) {
    ds4_tokens tokens = {0};
    if (!tokenize_raw_fixture(engine, prompt_file, frontier, &tokens)) return false;
    ds4_session *session = NULL;
    if (!create_and_sync(engine, &tokens, context, &session, case_id)) {
        ds4_tokens_free(&tokens);
        return false;
    }
    bool ok = compare_session_oracle(session, fixture, case_id);
    uint64_t hits_before_continuation = 0;
    if (ok && continuation && require_continuation_cache_hits) {
        ds4_gpu_laguna_compact_test_snapshot before;
        memset(&before, 0, sizeof(before));
        if (!ds4_gpu_test_laguna_compact_active_snapshot(&before)) {
            ok = fail_message(
                    "continuation compact snapshot is unavailable", case_id);
        } else {
            hits_before_continuation = before.cache_acquire_hits;
        }
    }
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
    if (ok && continuation && require_continuation_cache_hits) {
        ds4_gpu_laguna_compact_test_snapshot after;
        memset(&after, 0, sizeof(after));
        if (!ds4_gpu_test_laguna_compact_active_snapshot(&after) ||
            after.cache_acquire_hits <= hits_before_continuation) {
            ok = fail_message(
                    "teacher-forced continuation produced no cache hit", case_id);
        }
    }
    ds4_session_free(session);
    ds4_tokens_free(&tokens);
    return ok;
}

static void capture_prefill_progress(
        void *userdata,
        const char *event,
        int current,
        int total) {
    prefill_progress_capture *capture = userdata;
    if (!capture || !event || strcmp(event, "prefill_chunk") != 0) return;
    if (capture->count >= (int)(sizeof(capture->current) /
                                sizeof(capture->current[0]))) {
        capture->overflow = true;
        return;
    }
    capture->current[capture->count] = current;
    capture->total[capture->count] = total;
    capture->count++;
}

static bool live_prefill_accounting_exact(
        ds4_engine *engine,
        ds4_runtime_snapshot *snapshot,
        ds4_runtime_allocation_record records[256],
        size_t *required_out) {
    memset(snapshot, 0, sizeof(*snapshot));
    memset(records, 0, 256u * sizeof(records[0]));
    *required_out = 0;
    return ds4_test_engine_laguna_runtime_snapshot(
               engine, snapshot, records, 256u, required_out) &&
        snapshot->violation == DS4_RUNTIME_VIOLATION_NONE &&
        *required_out == 146u && snapshot->active_record_count == 146u &&
        snapshot->category_current[DS4_RUNTIME_CATEGORY_GRAPH_SCRATCH] ==
            UINT64_C(1537052680) &&
        snapshot->category_current[DS4_RUNTIME_CATEGORY_KV_STATE] ==
            UINT64_C(1686110208) &&
        snapshot->category_peak[DS4_RUNTIME_CATEGORY_GRAPH_SCRATCH] ==
            UINT64_C(1537052680) &&
        snapshot->category_peak[DS4_RUNTIME_CATEGORY_KV_STATE] ==
            UINT64_C(1686110208);
}

static bool runtime_allocation_state_equal(
        const ds4_runtime_snapshot *before,
        const ds4_runtime_snapshot *after) {
    if (!before || !after ||
        memcmp(before->category_bounds, after->category_bounds,
               sizeof(before->category_bounds)) != 0 ||
        memcmp(before->category_current, after->category_current,
               sizeof(before->category_current)) != 0 ||
        memcmp(before->category_peak, after->category_peak,
               sizeof(before->category_peak)) != 0 ||
        memcmp(before->report_bounds, after->report_bounds,
               sizeof(before->report_bounds)) != 0 ||
        before->owned_total_bound_bytes != after->owned_total_bound_bytes ||
        before->owned_total_current != after->owned_total_current ||
        before->owned_total_peak != after->owned_total_peak ||
        before->qualification_total_bound_bytes !=
            after->qualification_total_bound_bytes ||
        before->violation != after->violation ||
        before->active_record_count != after->active_record_count) {
        return false;
    }
    const ds4_runtime_report physical_reports[] = {
        DS4_RUNTIME_REPORT_MODEL_SOURCE_RESIDENT,
        DS4_RUNTIME_REPORT_HOST_LIBRARY_UNATTRIBUTED,
        DS4_RUNTIME_REPORT_CUDA_LIBRARY_UNATTRIBUTED,
    };
    const ds4_runtime_snapshot *snapshots[] = { before, after };
    /* qualification_total_peak is a simultaneous high-water sample.  The
     * source report peak can predate session graph/KV ownership, so combining
     * that historical peak with current ownership would invent a footprint
     * that never existed. */
    for (size_t snapshot_index = 0;
         snapshot_index < sizeof(snapshots) / sizeof(snapshots[0]);
         snapshot_index++) {
        const ds4_runtime_snapshot *snapshot = snapshots[snapshot_index];
        uint64_t expected_qualification = snapshot->owned_total_current;
        for (size_t report_index = 0;
             report_index < sizeof(physical_reports) /
                 sizeof(physical_reports[0]);
             report_index++) {
            const uint64_t bytes =
                snapshot->report_current[physical_reports[report_index]];
            if (expected_qualification > UINT64_MAX - bytes) return false;
            expected_qualification += bytes;
        }
        if (snapshot->report_current[
                    DS4_RUNTIME_REPORT_MODEL_SOURCE_RESIDENT] >
                snapshot->report_bounds[
                    DS4_RUNTIME_REPORT_MODEL_SOURCE_RESIDENT] ||
            snapshot->report_peak[
                    DS4_RUNTIME_REPORT_MODEL_SOURCE_RESIDENT] <
                snapshot->report_current[
                    DS4_RUNTIME_REPORT_MODEL_SOURCE_RESIDENT] ||
            snapshot->qualification_total_current !=
                expected_qualification ||
            snapshot->qualification_total_peak <
                snapshot->qualification_total_current ||
            snapshot->qualification_total_peak >
                snapshot->qualification_total_bound_bytes) {
            return false;
        }
    }
    for (size_t i = 0; i < DS4_RUNTIME_REPORT_COUNT; i++) {
        if (i == DS4_RUNTIME_REPORT_MODEL_SOURCE_RESIDENT) continue;
        if (before->report_current[i] != after->report_current[i] ||
            before->report_peak[i] != after->report_peak[i]) {
            return false;
        }
    }
    return after->report_peak[DS4_RUNTIME_REPORT_MODEL_SOURCE_RESIDENT] >=
            before->report_peak[DS4_RUNTIME_REPORT_MODEL_SOURCE_RESIDENT] &&
        after->qualification_total_peak >= before->qualification_total_peak &&
        after->event_sequence >= before->event_sequence;
}

/* This is deliberately an allocation/scheduling proof, not a replacement
 * Poolside oracle. The existing YaRN numerical fixture remains authoritative;
 * this case isolates whether the fixed 4K graph can process two real chunks
 * without reallocating or changing the durable 32K KV ownership. */
static bool run_prefill_8192(
        ds4_engine *engine,
        uint64_t *routed_tokens) {
    ds4_tokens tokens = {0};
    if (!tokenize_raw_fixture(
            engine, "yarn-8193.prompt", 8193, &tokens)) {
        return false;
    }
    tokens.len = 8192;
    if (routed_tokens) *routed_tokens = (uint64_t)tokens.len;

    ds4_session *session = NULL;
    char err[256] = {0};
    bool ok = ds4_session_create(&session, engine, 32768) == 0;
    if (!ok) ok = fail_message("session create", "prefill-8192");
    if (ok && ds4_session_prefill_cap(session) != 4096) {
        ok = fail_message("session prefill cap", "prefill-8192");
    }

    ds4_runtime_snapshot before;
    ds4_runtime_snapshot after;
    ds4_runtime_allocation_record before_records[256];
    ds4_runtime_allocation_record after_records[256];
    size_t before_required = 0;
    size_t after_required = 0;
    if (ok && !live_prefill_accounting_exact(
                  engine, &before, before_records, &before_required)) {
        ok = fail_message("live prefill accounting before sync", "prefill-8192");
    }

    prefill_progress_capture progress = {0};
    if (ok) {
        ds4_session_set_progress(session, capture_prefill_progress, &progress);
        if (ds4_session_sync(session, &tokens, err, sizeof(err)) != 0) {
            ok = fail_message("two-chunk session sync", err);
        }
        ds4_session_set_progress(session, NULL, NULL);
    }
    if (ok && (progress.overflow || progress.count != 2 ||
               progress.current[0] != 4096 ||
               progress.current[1] != 8192 ||
               progress.total[0] != 8192 || progress.total[1] != 8192)) {
        ok = fail_message("two-chunk progress boundaries", "prefill-8192");
    }
    if (ok && ds4_session_pos(session) != 8192) {
        ok = fail_message("two-chunk checkpoint position", "prefill-8192");
    }
    if (ok && !live_prefill_accounting_exact(
                  engine, &after, after_records, &after_required)) {
        ok = fail_message("live prefill accounting after sync", "prefill-8192");
    }
    if (ok && (before_required != after_required ||
               !runtime_allocation_state_equal(&before, &after) ||
               memcmp(before_records, after_records,
                      before_required * sizeof(before_records[0])) != 0)) {
        ok = fail_message("prefill allocations changed across chunks",
                          "prefill-8192");
    }

    ds4_session_free(session);
    ds4_tokens_free(&tokens);
    if (ok) {
        fprintf(stderr,
                "prefill-8192 chunks=4096+4096 graph=%llu kv=%llu "
                "owners=146 PASS\n",
                (unsigned long long)before.category_current[
                    DS4_RUNTIME_CATEGORY_GRAPH_SCRATCH],
                (unsigned long long)before.category_current[
                    DS4_RUNTIME_CATEGORY_KV_STATE]);
    }
    return ok;
}

static void session_state_free(session_state *state) {
    if (!state) return;
    free(state->logits);
    memset(state, 0, sizeof(*state));
}

static bool session_state_capture(
        ds4_session *session, session_state *state, const char *scenario) {
    state->logits = malloc(VECTOR_BYTES);
    if (!state->logits) return fail_message("allocate session-state logits", scenario);
    if (ds4_session_copy_logits(
                session, state->logits, LAGUNA_VOCAB) != LAGUNA_VOCAB) {
        session_state_free(state);
        return fail_message("copy session-state logits", scenario);
    }
    state->pos = ds4_session_pos(session);
    return true;
}

static bool session_state_unchanged(
        ds4_session *session, const session_state *before, const char *scenario) {
    float *after = malloc(VECTOR_BYTES);
    if (!after) return fail_message("allocate unchanged-state logits", scenario);
    if (ds4_session_copy_logits(session, after, LAGUNA_VOCAB) != LAGUNA_VOCAB) {
        free(after);
        return fail_message("copy unchanged-state logits", scenario);
    }
    const int after_pos = ds4_session_pos(session);
    const bool ok = after_pos == before->pos &&
                    memcmp(after, before->logits, VECTOR_BYTES) == 0;
    if (!ok) {
        fprintf(stderr,
                "FAIL: %s changed terminal state pos=%d/%d logits_equal=%d\n",
                scenario, after_pos, before->pos,
                memcmp(after, before->logits, VECTOR_BYTES) == 0);
    }
    free(after);
    return ok;
}

static bool terminal_rejected(int rc, const char *operation, const char *err) {
    if (rc == 0) {
        fprintf(stderr, "FAIL: terminal %s unexpectedly succeeded\n", operation);
        return false;
    }
    if (!err || !strstr(err, "logits-only terminal")) {
        fprintf(stderr, "FAIL: terminal %s returned the wrong error: %s\n",
                operation, err ? err : "");
        return false;
    }
    return true;
}

static bool run_deep_exact_context(
        ds4_engine *engine, const oracle_case *fixture) {
    ds4_tokens tokens = {0};
    ds4_session *session = NULL;
    ds4_session *control = NULL;
    float *seed_logits = NULL;
    session_state fresh = {0};
    session_state terminal = {0};
    session_state control_before = {0};
    uint8_t snapshot_marker = UINT8_C(0xa5);
    char staged_marker[] = "unchanged-stage-output";
    ds4_session_snapshot snapshot = {
        .ptr = &snapshot_marker,
        .len = UINT64_C(17),
        .cap = UINT64_C(19),
    };
    ds4_session_payload_file staged = {
        .path = staged_marker,
        .bytes = UINT64_C(23),
    };
    char err[256] = {0};
    bool ok = tokenize_raw_fixture(
            engine, "deep-32768.prompt", 32768, &tokens);

    if (ok && ds4_session_create(&session, engine, 32768) != 0) {
        ok = fail_message("session create", "deep-32768");
    }
    if (ok && ds4_session_prefill_cap(session) !=
                  (ds4_engine_prefill_chunk(engine) != 0 ? 4096 : 16384)) {
        ok = fail_message("session prefill cap", "deep-32768");
    }
    if (ok) {
        seed_logits = calloc(LAGUNA_VOCAB, sizeof(*seed_logits));
        if (!seed_logits) ok = fail_message("allocate deep boundary logits", NULL);
    }
    if (ok && ds4_session_set_logits(
                      session, seed_logits, LAGUNA_VOCAB) != 0) {
        ok = fail_message("seed deep boundary logits", NULL);
    }
    free(seed_logits);
    seed_logits = NULL;
    if (ok) ok = session_state_capture(session, &fresh, "deep ordinary boundary");
    if (ok && ds4_session_sync(session, &tokens, err, sizeof(err)) == 0) {
        ok = fail_message("deep ordinary equality sync unexpectedly succeeded", NULL);
    }
    if (ok) {
        ok = session_state_unchanged(session, &fresh, "deep ordinary boundary");
    }

    memset(err, 0, sizeof(err));
    if (ok && ds4_session_sync_logits_only(
                      session, &tokens, err, sizeof(err)) != 0) {
        ok = fail_message("deep logits-only equality sync", err);
    }
    if (ok && ds4_session_pos(session) != 32768) {
        ok = fail_message("deep logits-only position", NULL);
    }
    if (ok) ok = compare_session_oracle(session, fixture, "deep-32768");
    if (ok) ok = session_state_capture(session, &terminal, "deep terminal");
    const int terminal_token = ok ? ds4_session_argmax(session) : 0;

    memset(err, 0, sizeof(err));
    if (ok) {
        ok = terminal_rejected(
                ds4_session_eval(session, terminal_token, err, sizeof(err)),
                "direct decode", err);
    }
    if (ok) ok = session_state_unchanged(session, &terminal, "terminal direct decode");

    memset(err, 0, sizeof(err));
    if (ok) {
        ds4_session_rewind(session, terminal.pos - 1);
        ok = terminal_rejected(
                ds4_session_eval(session, terminal_token, err, sizeof(err)),
                "rewind followed by decode", err);
    }
    if (ok) {
        ok = session_state_unchanged(
                session, &terminal, "terminal rewind followed by decode");
    }

    memset(err, 0, sizeof(err));
    if (ok) {
        ok = terminal_rejected(
                ds4_session_rewrite_from_common(
                        session, &tokens, tokens.len - 1, err, sizeof(err)),
                "rewrite", err);
    }
    if (ok) ok = session_state_unchanged(session, &terminal, "terminal rewrite");

    memset(err, 0, sizeof(err));
    if (ok) {
        ok = terminal_rejected(
                ds4_session_sync(session, &tokens, err, sizeof(err)),
                "ordinary repeat sync", err);
    }
    if (ok) {
        ok = session_state_unchanged(
                session, &terminal, "terminal ordinary repeat sync");
    }

    memset(err, 0, sizeof(err));
    if (ok) {
        ok = terminal_rejected(
                ds4_session_sync_logits_only(session, &tokens, err, sizeof(err)),
                "logits-only repeat sync", err);
    }
    if (ok) {
        ok = session_state_unchanged(
                session, &terminal, "terminal logits-only repeat sync");
    }

    memset(err, 0, sizeof(err));
    if (ok) {
        ok = terminal_rejected(
                ds4_session_save_snapshot(session, &snapshot, err, sizeof(err)),
                "snapshot save", err);
    }
    if (ok && (snapshot.ptr != &snapshot_marker ||
               snapshot.len != UINT64_C(17) ||
               snapshot.cap != UINT64_C(19))) {
        ok = fail_message("terminal snapshot save changed output", NULL);
    }
    if (ok) ok = session_state_unchanged(session, &terminal, "terminal snapshot save");

    memset(err, 0, sizeof(err));
    if (ok) {
        ok = terminal_rejected(
                ds4_session_stage_payload(session, &staged, err, sizeof(err)),
                "staged payload export", err);
    }
    if (ok && (staged.path != staged_marker || staged.bytes != UINT64_C(23))) {
        ok = fail_message("terminal staged payload changed output", NULL);
    }
    if (ok) {
        ok = session_state_unchanged(
                session, &terminal, "terminal staged payload export");
    }

    ds4_tokens control_prompt = tokens;
    control_prompt.len = 512;
    if (ok) {
        ok = create_and_sync(
                engine, &control_prompt, 1024, &control, "terminal batch control");
    }
    if (ok) {
        ok = session_state_capture(
                control, &control_before, "terminal batch control");
    }
    if (ok) {
        ds4_decode_item items[2] = {
            {control, ds4_session_argmax(control)},
            {session, terminal_token},
        };
        memset(err, 0, sizeof(err));
        ok = terminal_rejected(
                ds4_sessions_eval_batch(items, 2, err, sizeof(err)),
                "batch decode", err);
    }
    if (ok) {
        ok = session_state_unchanged(
                control, &control_before, "terminal batch nonterminal member");
    }
    if (ok) {
        ok = session_state_unchanged(
                session, &terminal, "terminal batch terminal member");
    }

    if (staged.path != staged_marker) ds4_session_payload_file_free(&staged);
    if (snapshot.ptr != &snapshot_marker) ds4_session_snapshot_free(&snapshot);
    session_state_free(&control_before);
    session_state_free(&terminal);
    session_state_free(&fresh);
    free(seed_logits);
    ds4_session_free(control);
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

static bool run_decode_batch(ds4_engine *engine) {
    const uint64_t fallback_before =
        ds4_test_laguna_decode_fallback_count();
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
    if (ok && ds4_test_laguna_decode_fallback_count() !=
                  fallback_before + 1u) {
        ok = fail_message(
                "two-session batch did not use the Laguna correctness fallback",
                NULL);
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
    const uint64_t fallback_before =
        ds4_test_laguna_mixed_fallback_count();
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
    if (ok && ds4_test_laguna_mixed_fallback_count() !=
                  fallback_before + 1u) {
        ok = fail_message(
                "mixed prefill/decode did not use the Laguna correctness fallback",
                NULL);
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

int main(int argc, char **argv) {
    model_mode mode;
    model_case_selection selected;
    if (!parse_selection(argc, argv, &mode, &selected)) {
        usage(argv[0]);
        return 2;
    }
    const char *model = getenv("DS4_TEST_MODEL");
    if (!model || !model[0]) {
        fprintf(stderr, "FAIL: DS4_TEST_MODEL is not set\n");
        return 1;
    }
    const char *diag_dir = getenv("DS4_LAGUNA_DIAG_DIR");
    const bool diagnostic_mode = diag_dir && diag_dir[0];
    if (diagnostic_mode &&
        (mode != MODEL_MODE_RESIDENT ||
         selected == MODEL_CASE_CONTINUATION)) {
        fprintf(stderr,
                "FAIL: DS4_LAGUNA_DIAG_DIR supports resident short only\n");
        return 2;
    }
    int model_fd = -1;
    bool model_fd_set = false;
    if (!inherited_model_fd(&model_fd, &model_fd_set)) return 1;

    oracle_fixtures fixtures;
    if (!load_fixtures(&fixtures)) return 1;

    const ds4_engine_options options = {
        .model_path = model,
        .backend = DS4_BACKEND_CUDA,
        .n_threads = 1,
        .context_size = mode == MODEL_MODE_STREAMED ? 32768 : 0,
        .prefill_chunk = mode == MODEL_MODE_STREAMED ? 4096u : 0,
        .ssd_streaming_cache_bytes =
            mode == MODEL_MODE_STREAMED ? STREAMED_CACHE_BYTES : 0,
        .ssd_streaming = mode == MODEL_MODE_STREAMED,
        .ssd_streaming_cache_bytes_set =
            mode == MODEL_MODE_STREAMED,
        .session_slots = mode == MODEL_MODE_STREAMED ? 1u : 0,
        .qualification_model_fd = model_fd,
        .qualification_model_fd_set = model_fd_set,
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

    bool ok = true;
    uint64_t streamed_routed_tokens = 0u;
    if (selected == MODEL_CASE_SHORT || selected == MODEL_CASE_ALL) {
        ok = run_short(
            engine, &fixtures.cases[0],
            mode == MODEL_MODE_STREAMED ? &streamed_routed_tokens : NULL);
    }
    if (ok && selected == MODEL_CASE_SWA) {
        ok = run_raw_frontier(
                engine, "swa-513", "swa-513.prompt", 513, 1024,
                &fixtures.cases[1], NULL, false);
        if (mode == MODEL_MODE_STREAMED) streamed_routed_tokens = 513u;
    }
    if (ok && selected == MODEL_CASE_CONTINUATION) {
        ok = run_raw_frontier(
                engine, "yarn-8193", "yarn-8193.prompt", 8193, 8202,
                &fixtures.cases[2], fixtures.continuation,
                mode == MODEL_MODE_STREAMED);
        if (mode == MODEL_MODE_STREAMED) {
            streamed_routed_tokens = 8193u + CONTINUATION_COUNT;
        }
    }
    if (ok && selected == MODEL_CASE_PREFILL_8192) {
        ok = run_prefill_8192(engine, &streamed_routed_tokens);
    }
    if (ok && selected == MODEL_CASE_WARM_STABILITY) {
        ok = run_warm_stability(
            engine, &fixtures.cases[0], &streamed_routed_tokens);
    }
    if (ok && selected == MODEL_CASE_ALL && !diagnostic_mode) {
        if (ok) {
            ok = run_raw_frontier(
                    engine, "swa-513", "swa-513.prompt", 513, 1024,
                    &fixtures.cases[1], NULL, false);
        }
        if (ok) {
            ok = run_raw_frontier(
                    engine, "yarn-8193", "yarn-8193.prompt", 8193, 8202,
                    &fixtures.cases[2], fixtures.continuation, false);
        }
        /* run_raw_frontier frees the 8202-token session before deep allocation. */
        if (ok) ok = run_deep_exact_context(engine, &fixtures.cases[3]);
        /* The deep session is freed before either multi-session scenario. */
        if (ok) ok = run_decode_batch(engine);
        if (ok) ok = run_mixed_batch(engine);
    }
    if (mode == MODEL_MODE_STREAMED) {
        const bool cache_contract =
            verify_streamed_cache_contract(
                engine, model_case_name(selected), streamed_routed_tokens);
        ok = cache_contract && ok;
    }

    ds4_engine_close(engine);
    free_fixtures(&fixtures);
    if (!ok) return 1;
    if (diagnostic_mode) return 0;
    if (selected != MODEL_CASE_ALL) {
        if (selected == MODEL_CASE_PREFILL_8192 ||
            selected == MODEL_CASE_WARM_STABILITY) {
            fprintf(stderr,
                    "test_cuda_laguna_model PASS "
                    "contract=%s mode=%s case=%s\n",
                    selected == MODEL_CASE_PREFILL_8192 ?
                        "allocation-schedule" : "warm-stability",
                    model_mode_name(mode), model_case_name(selected));
            return 0;
        }
        fprintf(stderr,
                "test_cuda_laguna_model PASS oracle=poolside mode=%s case=%s\n",
                model_mode_name(mode), model_case_name(selected));
        return 0;
    }
    fprintf(stderr,
            "test_cuda_laguna_model PASS oracle=poolside cases=4 vectors=4 continuation=8 "
            "decode_batch=2 mixed=1+2\n");
    return 0;
}
