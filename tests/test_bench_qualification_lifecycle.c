/*
 * Host-only test for the Task 19 benchmark qualification lifecycle.
 *
 * This harness includes the real ds4-bench translation unit with its CLI main
 * renamed.  The authenticated sequence parser, engine/session operations,
 * runtime request accounting, external checkpoint, runtime snapshot, and
 * qualification emitter are the only seams faked.  The parser returns one
 * literal, trusted sequence object; no model, CUDA, GPU, network, or service
 * is used.  Once the sequence branch grows its real lifecycle runner this
 * same executable is the executable call-order oracle.
 */

#include "ds4.h"
#include "ds4_gpu.h"
#include "ds4_bench_qualification.h"

#include <inttypes.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ARRAY_LEN(value) (sizeof(value) / sizeof((value)[0]))
#define QUALIFICATION_OUTPUT_TOKENS 512
#define LATE_DECODE_FAILURE_REPETITION 2
#define LATE_DECODE_FAILURE_INDEX 257
#define SHORT_NATIVE_STOP_REPETITION 2
#define SHORT_NATIVE_STOP_INDEX 3

/* Keep opaque production handles fake and never dereference them. */
typedef union {
    long double align;
    unsigned char bytes[64];
} fake_storage;

static fake_storage fake_engine_storage;
static fake_storage fake_session_storage[4];
static int fake_prompt_tokens[512];
#define LITERAL_RENDERED_SEQUENCE "task19 literal qualification prompt"
static const unsigned char literal_input[] = LITERAL_RENDERED_SEQUENCE;
/* The real strict parser owns raw, non-NUL input bytes. Keep a writable
 * sentinel immediately after this fake's logical bytes so the fake can reject
 * both a direct char * cast and an in-place terminator without an out-of-bounds
 * comparison. */
static unsigned char literal_sequence_input[] =
    LITERAL_RENDERED_SEQUENCE "\x7f";

#define RESIDENT_LITERAL_RENDERED_SEQUENCE "task19 resident qualification prompt"
static const unsigned char resident_literal_input[] =
    RESIDENT_LITERAL_RENDERED_SEQUENCE;
/* Resident uses a separate sentinel so a mode mix-up cannot accidentally
 * compare the streamed fixture's storage. */
static unsigned char resident_sequence_input[] =
    RESIDENT_LITERAL_RENDERED_SEQUENCE "\x7f";

static const uint32_t literal_prompt_tokens = 512u;
/* Deliberately nonzero: a hard-coded common EOS value must not pass the fake. */
static const int literal_eos_token = 17;
/* Distinct model-native stop control; fake generated tokens start at 20. */
static const int literal_native_stop_token = 19;
static const uint64_t literal_cache_bytes = UINT64_C(8589934592);
static const char literal_manifest_sha256[] =
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
static const char literal_sequence_sha256[] =
    "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
static const char literal_input_sha256[] =
    "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
static const char literal_instance_id[] =
    "123e4567-e89b-12d3-a456-426614174099";
static const char resident_manifest_sha256[] =
    "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
static const char resident_sequence_sha256[] =
    "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
static const char resident_input_sha256[] =
    "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";

static ds4_engine *const fake_engine =
    (ds4_engine *)(void *)&fake_engine_storage;

enum call_kind {
    CALL_SEQUENCE_PARSE,
    CALL_SEQUENCE_FREE,
    CALL_RESIDENT_SEQUENCE_PARSE,
    CALL_RESIDENT_SEQUENCE_FREE,
    CALL_ENGINE_OPEN,
    CALL_ENGINE_CLOSE,
    CALL_NVML_CAPTURE,
    CALL_TOKENIZE_RENDERED,
    CALL_SESSION_CREATE,
    CALL_SESSION_FREE,
    CALL_REQUEST_BEGIN,
    CALL_REQUEST_PROMPT,
    CALL_PREFILL_START,
    CALL_SESSION_SYNC_ATTRIBUTED,
    CALL_PREFILL_COMPLETE,
    CALL_TOKEN_CHOOSE,
    CALL_SESSION_EVAL_ATTRIBUTED,
    CALL_GENERATED,
    CALL_VISIBLE,
    CALL_FIRST_VISIBLE,
    CALL_REQUEST_BARRIER,
    CALL_REQUEST_FINISH,
    CALL_EXTERNAL_CHECKPOINT,
    CALL_RUNTIME_SNAPSHOT,
    CALL_EMIT,
    CALL_RESIDENT_EMIT,
    CALL_UNEXPECTED,
};

typedef struct {
    enum call_kind kind;
    int repetition;
    ds4_session *session;
    ds4_runtime_request_context *request;
    const ds4_bench_sequence *sequence;
    const ds4_bench_resident_sequence *resident_sequence;
    ds4_bench_qualification_event event;
    const ds4_runtime_request_metrics *metrics;
    uint64_t value;
    int token;
} call_record;

typedef struct {
    /* Four repetitions x 512 tokens x four per-token operations, plus
     * milestone/ownership calls.  Keep the full timeline, not a sampled log. */
    call_record calls[16384];
    size_t call_count;
    int contract_failures;
    int engine_open_count;
    int engine_close_count;
    int tokenize_rendered_count;
    int nvml_capture_count;
    int session_create_count;
    int session_free_count;
    int request_begin_count;
    int request_prompt_count;
    int prefill_start_count;
    int sync_count;
    int prefill_complete_count;
    int choose_count;
    int eval_count;
    int generated_count;
    int visible_count;
    int first_visible_count;
    int barrier_count;
    int finish_count;
    int checkpoint_count;
    int snapshot_count;
    int emit_count;
    int resident_emit_count;
    int sequence_free_count;
    int resident_sequence_free_count;
    int decode_attempt_count[4];
    int decode_success_count[4];
    int completed_metrics_count;
    ds4_runtime_request_metrics completed_metrics[4];
    bool resident_mode;
    bool inject_emitter_failure;
    bool inject_resident_parser_failure;
    bool reject_resident_cross_mode;
    bool reject_streamed_cross_mode;
    bool inject_snapshot_failure;
    bool inject_checkpoint_failure;
    bool inject_decode_failure;
    int decode_failure_repetition;
    int decode_failure_index;
    bool inject_native_stop;
    int native_stop_repetition;
    int native_stop_index;
    int native_stop_token;
    bool decode_failed;
    bool emitter_failed;
    bool snapshot_failed;
    bool checkpoint_failed;
    const ds4_bench_sequence *trusted_sequence;
    const ds4_bench_resident_sequence *trusted_resident_sequence;
    const ds4_gpu_nvml_inventory_snapshot *captured_pre_child;
    uint64_t last_emitted_monotonic_ns;
    ds4_runtime_request_context *requests[4];
    int request_repetition[4];
    ds4_session *sessions[4];
    int session_repetition[4];
} fake_state;

static fake_state state;
static int all_failures;

static void expected_request_id(int repetition, char *buffer, size_t capacity) {
    snprintf(buffer, capacity,
             "123e4567-e89b-12d3-a456-4266141740%02d", repetition + 1);
}

static void fail_contract(const char *message) {
    state.contract_failures++;
    fprintf(stderr, "FAIL: %s\n", message);
}

static void record_call(enum call_kind kind, int repetition) {
    if (state.call_count >= ARRAY_LEN(state.calls)) {
        fail_contract("fake call log capacity exhausted");
        return;
    }
    state.calls[state.call_count++] = (call_record){
        .kind = kind,
        .repetition = repetition,
        .session = NULL,
        .request = NULL,
        .sequence = NULL,
        .resident_sequence = NULL,
        .event = DS4_BENCH_QUALIFICATION_EVENT_REQUEST_ACCEPTED,
        .metrics = NULL,
        .value = 0u,
        .token = 0,
    };
}

static void record_pointer_call(enum call_kind kind,
                                int repetition,
                                ds4_session *session,
                                ds4_runtime_request_context *request) {
    record_call(kind, repetition);
    if (state.call_count == 0u) return;
    call_record *call = &state.calls[state.call_count - 1u];
    call->session = session;
    call->request = request;
}

static int session_index(const ds4_session *session) {
    for (int i = 0; i < state.session_create_count && i < 4; i++) {
        if (state.sessions[i] == session) return i;
    }
    return -1;
}

static int request_index(const ds4_runtime_request_context *request) {
    /* A runner may reuse one stack slot for each fresh request context.  The
     * most recent begin is the authoritative binding for that address. */
    const int count = state.request_begin_count < 4 ?
        state.request_begin_count : 4;
    for (int i = count - 1; i >= 0; i--) {
        if (state.requests[i] == request) return i;
    }
    return -1;
}

static int request_rep(const ds4_runtime_request_context *request) {
    const int index = request_index(request);
    return index < 0 ? -1 : state.request_repetition[index];
}

static int session_rep(const ds4_session *session) {
    const int index = session_index(session);
    return index < 0 ? -1 : state.session_repetition[index];
}

static bool check_repetition(int actual, int expected, const char *what) {
    if (actual == expected) return true;
    char message[160];
    snprintf(message, sizeof(message), "%s used repetition %d, expected %d",
             what, actual, expected);
    fail_contract(message);
    return false;
}

static bool fake_sequence_parse_file_trusted(
        const char *path,
        const char *expected_manifest_sha256,
        const char *expected_sequence_sha256,
        ds4_bench_sequence *out,
        char *error,
        size_t error_size) {
    (void)error;
    (void)error_size;
    record_call(CALL_SEQUENCE_PARSE, -1);
    const bool reject_cross_mode = state.reject_streamed_cross_mode;
    const char *expected_path = reject_cross_mode
        ? "/literal/resident-sequence.txt" : "/literal/sequence.txt";
    const char *expected_manifest = reject_cross_mode
        ? resident_manifest_sha256 : literal_manifest_sha256;
    const char *expected_sequence = reject_cross_mode
        ? resident_sequence_sha256 : literal_sequence_sha256;
    if (!path || strcmp(path, expected_path) != 0) {
        fail_contract("trusted parser did not receive the expected sequence path");
        return false;
    }
    if (!out || !expected_manifest_sha256 || !expected_sequence_sha256 ||
        strcmp(expected_manifest_sha256, expected_manifest) != 0 ||
        strcmp(expected_sequence_sha256, expected_sequence) != 0) {
        fail_contract("trusted parser did not receive the expected authenticated digests");
        return false;
    }
    state.trusted_sequence = out;
    if (reject_cross_mode) {
        snprintf(error, error_size, "streamed parser rejected resident fixture");
        return false;
    }
    ds4_bench_sequence literal = {0};
    memcpy(literal.manifest_sha256, literal_manifest_sha256,
           sizeof(literal.manifest_sha256));
    memcpy(literal.profile_id, "cache-8gib", sizeof("cache-8gib"));
    literal.cache_bytes = literal_cache_bytes;
    literal.prompt_order_index = 0u;
    memcpy(literal.prompt_id, "native-512", sizeof("native-512"));
    literal.prompt_tokens = literal_prompt_tokens;
    literal.input_size_bytes = sizeof(literal_input) - 1u;
    literal.input_size = sizeof(literal_input) - 1u;
    literal.input_bytes = literal_sequence_input;
    memcpy(literal.input_sha256, literal_input_sha256,
           sizeof(literal.input_sha256));
    memcpy(literal.sequence_sha256, literal_sequence_sha256,
           sizeof(literal.sequence_sha256));
    *out = literal;
    state.trusted_sequence = out;
    return true;
}

static void fake_sequence_free(ds4_bench_sequence *sequence) {
    record_call(CALL_SEQUENCE_FREE, -1);
    state.sequence_free_count++;
    if (sequence != state.trusted_sequence) {
        fail_contract("sequence free did not receive the exact trusted sequence pointer");
    }
    /* The input is static in this fake.  Do not free or scrub it. */
}

static bool fake_resident_sequence_parse_file_trusted(
        const char *path,
        const char *expected_manifest_sha256,
        const char *expected_sequence_sha256,
        ds4_bench_resident_sequence *out,
        char *error,
        size_t error_size) {
    record_call(CALL_RESIDENT_SEQUENCE_PARSE, -1);
    state.resident_mode = true;
    if (!out || !expected_manifest_sha256 || !expected_sequence_sha256 ||
        strcmp(path ? path : "", state.reject_resident_cross_mode
                                      ? "/literal/sequence.txt"
                                      : "/literal/resident-sequence.txt") != 0 ||
        strcmp(expected_manifest_sha256,
               state.reject_resident_cross_mode ? literal_manifest_sha256
                                                 : resident_manifest_sha256) != 0 ||
        strcmp(expected_sequence_sha256,
               state.reject_resident_cross_mode ? literal_sequence_sha256
                                                 : resident_sequence_sha256) != 0) {
        fail_contract("resident trusted parser did not receive its literal authenticated fixture");
        return false;
    }
    state.trusted_resident_sequence = out;
    if (state.reject_resident_cross_mode) {
        if (error && error_size != 0u) {
            snprintf(error, error_size,
                     "resident parser rejected streamed fixture");
        }
        return false;
    }
    if (state.inject_resident_parser_failure) {
        if (error && error_size != 0u) {
            snprintf(error, error_size, "injected resident parser failure");
        }
        return false;
    }
    if (!out) return false;
    ds4_bench_resident_sequence resident = {0};
    memcpy(resident.sequence.manifest_sha256, resident_manifest_sha256,
           sizeof(resident.sequence.manifest_sha256));
    memcpy(resident.sequence.profile_id, "resident", sizeof("resident"));
    resident.sequence.cache_bytes = 0u;
    resident.sequence.prompt_order_index = 0u;
    memcpy(resident.sequence.prompt_id, "native-512", sizeof("native-512"));
    resident.sequence.prompt_tokens = literal_prompt_tokens;
    resident.sequence.input_size_bytes = sizeof(resident_literal_input) - 1u;
    resident.sequence.input_size = sizeof(resident_literal_input) - 1u;
    resident.sequence.input_bytes = resident_sequence_input;
    memcpy(resident.sequence.input_sha256, resident_input_sha256,
           sizeof(resident.sequence.input_sha256));
    memcpy(resident.sequence.sequence_sha256, resident_sequence_sha256,
           sizeof(resident.sequence.sequence_sha256));
    *out = resident;
    return true;
}

static void fake_resident_sequence_free(
        ds4_bench_resident_sequence *sequence) {
    record_call(CALL_RESIDENT_SEQUENCE_FREE, -1);
    state.resident_sequence_free_count++;
    if (sequence != state.trusted_resident_sequence) {
        fail_contract("resident sequence free did not receive the exact trusted sequence pointer");
    }
    /* The input is static in this fake.  Do not free or scrub it. */
}

static int fake_gpu_nvml_inventory_capture(
        ds4_gpu_nvml_inventory_snapshot *out) {
    record_call(CALL_NVML_CAPTURE, -1);
    state.nvml_capture_count++;
    if (!out) {
        fail_contract("pre-engine NVML capture received a null output");
        return 1;
    }
    if (state.nvml_capture_count != 1 || state.engine_open_count != 0) {
        fail_contract("pre-child NVML inventory was not captured exactly once before engine open");
    }
    memset(out, 0, sizeof(*out));
    out->api_version = 77u;
    snprintf(out->api_identity, sizeof(out->api_identity), "task19-fake-nvml");
    snprintf(out->library_version, sizeof(out->library_version), "task19-fake-library");
    snprintf(out->device_uuid, sizeof(out->device_uuid), "GPU-task19-fake");
    out->process_count = 1u;
    out->processes[0].pid = 4242u;
    out->processes[0].used_bytes = UINT64_C(123456);
    out->processes[0].used_bytes_known = true;
    state.captured_pre_child = out;
    return 0;
}

static bool is_captured_pre_child_snapshot(
        const ds4_gpu_nvml_inventory_snapshot *snapshot) {
    return snapshot != NULL &&
           snapshot == state.captured_pre_child &&
           snapshot->api_version == 77u &&
           strcmp(snapshot->api_identity, "task19-fake-nvml") == 0 &&
           strcmp(snapshot->library_version, "task19-fake-library") == 0 &&
           strcmp(snapshot->device_uuid, "GPU-task19-fake") == 0 &&
           snapshot->process_count == 1u &&
           snapshot->processes[0].pid == 4242u &&
           snapshot->processes[0].used_bytes == UINT64_C(123456) &&
           snapshot->processes[0].used_bytes_known;
}

static bool pinned_engine_options(const ds4_engine_options *options) {
    if (!options) return false;
    return options->model_path &&
           strcmp(options->model_path, "/literal/fake.gguf") == 0 &&
           options->backend == DS4_BACKEND_CUDA &&
           options->context_size == 32768 &&
           options->prefill_chunk == 4096u &&
           options->session_slots == 1u &&
           options->ssd_streaming &&
           options->ssd_streaming_cache_bytes == literal_cache_bytes &&
           options->ssd_streaming_cache_bytes_set &&
           options->placement_ctx_hint == 32768 &&
           !options->quality &&
           !options->warm_weights &&
           !options->ssd_streaming_cold &&
           !options->ssd_streaming_cache_experts_set &&
           !options->ssd_streaming_full_layers_set &&
           !options->ssd_streaming_preload_experts &&
           !options->qualification_plan_path_set &&
           options->qualification_control_fd_set &&
           options->qualification_control_fd == 9;
}

static bool pinned_resident_engine_options(const ds4_engine_options *options) {
    if (!options) return false;
    return options->model_path &&
           strcmp(options->model_path, "/literal/fake.gguf") == 0 &&
           options->backend == DS4_BACKEND_CUDA &&
           options->context_size == 32768 &&
           options->prefill_chunk == 4096u &&
           options->session_slots == 1u &&
           !options->ssd_streaming &&
           options->ssd_streaming_cache_bytes == 0u &&
           !options->ssd_streaming_cache_bytes_set &&
           options->placement_ctx_hint == 32768 &&
           !options->quality &&
           !options->warm_weights &&
           !options->ssd_streaming_cold &&
           !options->ssd_streaming_cache_experts_set &&
           !options->ssd_streaming_full_layers_set &&
           !options->ssd_streaming_preload_experts &&
           !options->qualification_plan_path_set &&
           options->qualification_control_fd_set &&
           options->qualification_control_fd == 9;
}

static int fake_engine_open(ds4_engine **out, const ds4_engine_options *options) {
    record_call(CALL_ENGINE_OPEN, -1);
    state.engine_open_count++;
    if (state.nvml_capture_count != 1 || state.captured_pre_child == NULL) {
        fail_contract("engine open was not preceded by exactly one pre-child NVML capture");
    }
    const bool options_valid = state.resident_mode
        ? pinned_resident_engine_options(options)
        : pinned_engine_options(options);
    if (!out || !options_valid) {
        fail_contract("engine open did not receive the pinned qualification configuration");
        return 1;
    }
    if (state.engine_open_count != 1) {
        fail_contract("qualification lifecycle opened more than one engine");
    }
    *out = fake_engine;
    return 0;
}

static int fake_engine_create_with_gpu_config(
        ds4_engine **out,
        const ds4_engine_options *options,
        const struct ds4_gpu_config *gpu_config) {
    if (gpu_config != NULL) {
        fail_contract("qualification lifecycle probed a GPU layout instead of opening the pinned engine");
        return 1;
    }
    return fake_engine_open(out, options);
}

static void fake_engine_close(ds4_engine *engine) {
    record_call(CALL_ENGINE_CLOSE, -1);
    state.engine_close_count++;
    if (engine != fake_engine) fail_contract("wrong engine pointer closed");
    if (state.engine_close_count != 1) {
        fail_contract("qualification lifecycle closed the engine more than once");
    }
    if (state.session_free_count != state.session_create_count) {
        fail_contract("engine closed before every created session was freed");
    }
}

static int fake_engine_vocab_size(ds4_engine *engine) {
    (void)engine;
    return 512;
}

static uint32_t fake_engine_prefill_chunk(ds4_engine *engine) {
    (void)engine;
    return 4096u;
}

static int fake_engine_routed_quant_bits(ds4_engine *engine) {
    (void)engine;
    return 4;
}

static bool fake_engine_runtime_snapshot(
        ds4_engine *engine, ds4_runtime_wire_snapshot *out) {
    record_call(CALL_RUNTIME_SNAPSHOT, -1);
    state.snapshot_count++;
    if (engine != fake_engine || !out) {
        fail_contract("runtime snapshot received an invalid engine/output");
        return false;
    }
    if (state.inject_snapshot_failure && state.snapshot_count == 2) {
        state.snapshot_failed = true;
        return false;
    }
    memset(out, 0, sizeof(*out));
    snprintf(out->instance_id, sizeof(out->instance_id), "%s",
             literal_instance_id);
    /* Finalization consumes the intervening publication sequence before the
     * completion snapshot, matching the independent JSONL lifecycle oracle. */
    out->snapshot_seq = UINT64_C(99) + (uint64_t)state.snapshot_count +
        (uint64_t)(state.snapshot_count / 3);
    out->state = DS4_RUNTIME_WIRE_STATE_READY;

    /* Keep this an independent literal snapshot rather than borrowing the
     * production snapshot serializer or the standalone emitter fixture. */
    memcpy(out->build.revision,
           "1111111111111111111111111111111111111111", sizeof(out->build.revision));
    out->build.dirty = false;
    memcpy(out->build.backend, "cuda", sizeof("cuda"));
    memcpy(out->build.features[0], "laguna", sizeof("laguna"));
    /* Build features describe capabilities, not this run's selected mode. */
    memcpy(out->build.features[1], "ssd_streaming", sizeof("ssd_streaming"));
    out->build.feature_count = 2u;
    out->executable = (ds4_runtime_file_identity){
        .device = 1u, .inode = 2u, .size_bytes = 3u, .mtime_ns = 4u,
    };
    out->model = (ds4_runtime_file_identity){
        .device = 5u, .inode = 6u, .size_bytes = 7u, .mtime_ns = 8u,
    };
    memcpy(out->model_id, "laguna-s-2.1", sizeof("laguna-s-2.1"));
    memcpy(out->model_family, "laguna", sizeof("laguna"));

    out->configured_context_tokens = 32768u;
    out->configured_prefill_chunk_tokens = 4096u;
    out->configured_session_slots = 1u;
    out->configured_ssd_streaming = !state.resident_mode;
    out->configured_ssd_streaming_cache_bytes = state.resident_mode
        ? 0u : literal_cache_bytes;
    out->effective_context_tokens = 32768u;
    out->effective_prefill_chunk_tokens = 4096u;
    out->effective_session_slots = 1u;
    out->expert_cache_limit_bytes = state.resident_mode
        ? 0u : literal_cache_bytes;
    out->configured_prefill_rows = 4096u;
    out->allocated_prefill_rows = 4096u;

    /* Every measurement is finite and ordered current <= peak <= bound. */
    out->allocations.category_current[DS4_RUNTIME_CATEGORY_STATIC_WEIGHTS] = 256u;
    out->allocations.category_peak[DS4_RUNTIME_CATEGORY_STATIC_WEIGHTS] = 512u;
    out->allocations.category_bounds[DS4_RUNTIME_CATEGORY_STATIC_WEIGHTS] = 4096u;
    out->allocations.category_current[DS4_RUNTIME_CATEGORY_EXPERT_CACHE_PAYLOAD] =
        state.resident_mode ? 0u : 1000u;
    out->allocations.category_peak[DS4_RUNTIME_CATEGORY_EXPERT_CACHE_PAYLOAD] =
        state.resident_mode ? 0u : 2000u;
    out->allocations.category_bounds[DS4_RUNTIME_CATEGORY_EXPERT_CACHE_PAYLOAD] =
        state.resident_mode ? 0u : literal_cache_bytes;
    out->allocations.category_current[DS4_RUNTIME_CATEGORY_CACHE_METADATA_ADDRESS_TABLES] = 128u;
    out->allocations.category_peak[DS4_RUNTIME_CATEGORY_CACHE_METADATA_ADDRESS_TABLES] = 256u;
    out->allocations.category_bounds[DS4_RUNTIME_CATEGORY_CACHE_METADATA_ADDRESS_TABLES] = 4096u;
    out->allocations.category_current[DS4_RUNTIME_CATEGORY_KV_STATE] = 512u;
    out->allocations.category_peak[DS4_RUNTIME_CATEGORY_KV_STATE] = 1024u;
    out->allocations.category_bounds[DS4_RUNTIME_CATEGORY_KV_STATE] = 4096u;
    out->allocations.category_current[DS4_RUNTIME_CATEGORY_GRAPH_SCRATCH] = 128u;
    out->allocations.category_peak[DS4_RUNTIME_CATEGORY_GRAPH_SCRATCH] = 256u;
    out->allocations.category_bounds[DS4_RUNTIME_CATEGORY_GRAPH_SCRATCH] = 4096u;
    out->allocations.category_current[DS4_RUNTIME_CATEGORY_PINNED_STAGING] = 64u;
    out->allocations.category_peak[DS4_RUNTIME_CATEGORY_PINNED_STAGING] = 128u;
    out->allocations.category_bounds[DS4_RUNTIME_CATEGORY_PINNED_STAGING] = 4096u;
    out->allocations.category_current[DS4_RUNTIME_CATEGORY_OTHER_HOST] = 32u;
    out->allocations.category_peak[DS4_RUNTIME_CATEGORY_OTHER_HOST] = 64u;
    out->allocations.category_bounds[DS4_RUNTIME_CATEGORY_OTHER_HOST] = 4096u;
    out->allocations.category_current[DS4_RUNTIME_CATEGORY_OTHER_CUDA] = 32u;
    out->allocations.category_peak[DS4_RUNTIME_CATEGORY_OTHER_CUDA] = 64u;
    out->allocations.category_bounds[DS4_RUNTIME_CATEGORY_OTHER_CUDA] = 4096u;
    out->allocations.owned_total_current = state.resident_mode ? 1152u : 2152u;
    out->allocations.owned_total_peak = state.resident_mode ? 2304u : 4304u;
    out->allocations.owned_total_bound_bytes = state.resident_mode
        ? UINT64_C(28672) : UINT64_C(8589963264);

    out->allocations.report_current[DS4_RUNTIME_REPORT_MODEL_MAPPED_VIRTUAL] = 4096u;
    out->allocations.report_peak[DS4_RUNTIME_REPORT_MODEL_MAPPED_VIRTUAL] = 8192u;
    out->allocations.report_bounds[DS4_RUNTIME_REPORT_MODEL_MAPPED_VIRTUAL] = 16384u;
    out->allocations.report_current[DS4_RUNTIME_REPORT_MODEL_MAPPING_REGISTERED] = 4096u;
    out->allocations.report_peak[DS4_RUNTIME_REPORT_MODEL_MAPPING_REGISTERED] = 8192u;
    out->allocations.report_bounds[DS4_RUNTIME_REPORT_MODEL_MAPPING_REGISTERED] = 16384u;
    out->allocations.report_current[DS4_RUNTIME_REPORT_MODEL_SOURCE_RESIDENT] = 5000u;
    out->allocations.report_peak[DS4_RUNTIME_REPORT_MODEL_SOURCE_RESIDENT] = 6000u;
    out->allocations.report_bounds[DS4_RUNTIME_REPORT_MODEL_SOURCE_RESIDENT] = 8192u;
    out->allocations.report_current[DS4_RUNTIME_REPORT_HOST_LIBRARY_UNATTRIBUTED] = 100u;
    out->allocations.report_peak[DS4_RUNTIME_REPORT_HOST_LIBRARY_UNATTRIBUTED] = 200u;
    out->allocations.report_bounds[DS4_RUNTIME_REPORT_HOST_LIBRARY_UNATTRIBUTED] = 4096u;
    out->allocations.report_current[DS4_RUNTIME_REPORT_CUDA_LIBRARY_UNATTRIBUTED] = 200u;
    out->allocations.report_peak[DS4_RUNTIME_REPORT_CUDA_LIBRARY_UNATTRIBUTED] = 400u;
    out->allocations.report_bounds[DS4_RUNTIME_REPORT_CUDA_LIBRARY_UNATTRIBUTED] = 4096u;
    /* Mapping and registration are report-only: do not charge them again. */
    out->allocations.qualification_total_current = state.resident_mode
        ? 6452u : 7452u;
    out->allocations.qualification_total_peak = state.resident_mode
        ? 8904u : 10904u;
    out->allocations.qualification_total_bound_bytes = state.resident_mode
        ? UINT64_C(45056) : UINT64_C(8589979648);
    out->allocations.external_sample.host_library_unattributed_bytes = 100u;
    out->allocations.external_sample.cuda_library_unattributed_bytes = 200u;
    out->allocations.external_sample.unrelated_process_inventory_stable = true;

    out->counters.cache_acquire_hits = 1u;
    out->counters.cache_acquire_misses = 2u;
    out->counters.cache_evictions = 3u;
    out->counters.model_file_read_operations = 4u;
    out->counters.model_file_read_bytes = 5u;
    out->counters.model_file_read_ns = 6u;
    out->counters.host_to_device_bytes = 7u;
    out->counters.host_to_device_ns = 8u;
    out->counters.page_advice_attempts = 9u;
    out->counters.page_advice_bytes = 10u;
    out->counters.page_advice_failures = 11u;
    return true;
}

static ds4_runtime_status fake_engine_laguna_external_checkpoint(
        ds4_engine *engine,
        const ds4_gpu_nvml_inventory_snapshot *pre_child,
        const uint8_t expected_build_identity[DS4_RUNTIME_BUILD_IDENTITY_BYTES],
        ds4_engine_laguna_external_checkpoint_observation *out) {
    record_call(CALL_EXTERNAL_CHECKPOINT, -1);
    state.checkpoint_count++;
    bool build_identity_nonzero = false;
    if (expected_build_identity != NULL) {
        for (size_t i = 0; i < DS4_RUNTIME_BUILD_IDENTITY_BYTES; i++) {
            if (expected_build_identity[i] != 0u) {
                build_identity_nonzero = true;
                break;
            }
        }
    }
    if (engine != fake_engine || !out ||
        !is_captured_pre_child_snapshot(pre_child) ||
        !build_identity_nonzero) {
        fail_contract("external checkpoint did not use the captured pre-child inventory and trusted build identity");
    }
    if (!out) {
        return DS4_RUNTIME_STATUS_UNSAFE;
    }
    if (state.inject_checkpoint_failure && state.checkpoint_count == 2) {
        state.checkpoint_failed = true;
        return DS4_RUNTIME_STATUS_UNSAFE;
    }
    memset(out, 0, sizeof(*out));
    out->sample.unrelated_process_inventory_stable = true;
    return DS4_RUNTIME_STATUS_OK;
}

static int fake_session_create(ds4_session **out, ds4_engine *engine, int context_size) {
    record_call(CALL_SESSION_CREATE, state.session_create_count);
    const int repetition = state.session_create_count;
    state.session_create_count++;
    if (!out || engine != fake_engine || context_size != 32768 || repetition >= 4) {
        fail_contract("session create did not use the open engine and pinned context");
        return 1;
    }
    *out = (ds4_session *)(void *)&fake_session_storage[repetition];
    state.sessions[repetition] = *out;
    state.session_repetition[repetition] = repetition;
    return 0;
}

static void fake_session_free(ds4_session *session) {
    const int repetition = session_rep(session);
    record_pointer_call(CALL_SESSION_FREE, repetition, session, NULL);
    state.session_free_count++;
    if (repetition < 0 || repetition >= 4) {
        fail_contract("unknown session was freed");
        return;
    }
    for (int i = 0; i < repetition; i++) {
        if (state.sessions[i] == session && i != repetition) {
            fail_contract("session pointer was reused across repetitions");
        }
    }
    int free_count = 0;
    for (size_t i = 0; i + 1u < state.call_count; i++) {
        if (state.calls[i].kind == CALL_SESSION_FREE &&
            state.calls[i].session == session) free_count++;
    }
    if (free_count != 0) fail_contract("current session was freed more than once");
}

static void fill_fake_tokens(ds4_tokens *out) {
    if (!out) {
        fail_contract("tokenizer received a null output");
        return;
    }
    for (size_t i = 0; i < ARRAY_LEN(fake_prompt_tokens); i++) {
        fake_prompt_tokens[i] = (int)i + 1;
    }
    out->v = fake_prompt_tokens;
    out->len = (int)literal_prompt_tokens;
    out->cap = (int)literal_prompt_tokens;
}

static void fake_tokenize_text(ds4_engine *engine, const char *text, ds4_tokens *out) {
    (void)engine;
    (void)text;
    record_call(CALL_UNEXPECTED, -1);
    fail_contract("qualification lifecycle used ds4_tokenize_text instead of rendered sequence input");
    fill_fake_tokens(out);
}

static bool is_exact_rendered_sequence_input(const char *text) {
    /* The tokenizer API takes a C string. strcmp checks both the complete
     * already-rendered payload and its terminating NUL without reading a fixed
     * length from an arbitrary caller buffer. */
    if (!text) return false;
    if (state.resident_mode) {
        return resident_sequence_input[sizeof(resident_literal_input) - 1u] == 0x7fu &&
               text != (const char *)resident_sequence_input &&
               strcmp(text, (const char *)resident_literal_input) == 0;
    }
    return literal_sequence_input[sizeof(literal_input) - 1u] == 0x7fu &&
           text != (const char *)literal_sequence_input &&
           strcmp(text, (const char *)literal_input) == 0;
}

static void fake_tokenize_rendered_chat(ds4_engine *engine, const char *text,
                                        ds4_tokens *out) {
    record_call(CALL_TOKENIZE_RENDERED, -1);
    state.tokenize_rendered_count++;
    if (engine != fake_engine || !is_exact_rendered_sequence_input(text) ||
        state.tokenize_rendered_count != 1 || state.engine_open_count != 1 ||
        state.engine_close_count != 0 || state.session_create_count != 0) {
        fail_contract("rendered sequence tokenization was not the one exact post-open pre-session operation");
    }
    if (!out) {
        fail_contract("rendered sequence tokenizer received a null output");
        return;
    }
    fill_fake_tokens(out);
    if (out->len != (int)literal_prompt_tokens) {
        fail_contract("rendered sequence tokenizer did not return the canonical prompt token count");
    }
}

static void fake_encode_chat_prompt(ds4_engine *engine, const char *system,
                                    const char *prompt, ds4_think_mode mode,
                                    ds4_tokens *out) {
    (void)engine;
    (void)system;
    (void)prompt;
    (void)mode;
    record_call(CALL_UNEXPECTED, -1);
    fail_contract("qualification lifecycle used chat encoding instead of rendered sequence input");
    fill_fake_tokens(out);
}

static void fake_tokens_free(ds4_tokens *tokens) {
    (void)tokens;
}

static uint64_t fake_session_payload_bytes(ds4_session *session) {
    const int repetition = session_rep(session);
    return UINT64_C(4096) + (uint64_t)(repetition < 0 ? 0 : repetition);
}

static int fake_token_eos(ds4_engine *engine) {
    if (engine != fake_engine) {
        fail_contract("EOS lookup did not use the open fake engine");
    }
    return literal_eos_token;
}

static bool fake_token_is_stop(ds4_engine *engine, int token) {
    if (engine != fake_engine) {
        fail_contract("native-stop lookup did not use the open fake engine");
    }
    return token == literal_eos_token || token == literal_native_stop_token;
}

static int fake_decode_token(int repetition, int token_index) {
    /* Keep every fake token inside the 512-entry fake vocabulary, away from
     * both model-native stop ids, while making repetition/index visible
     * to the eval seam. */
    return 20 + ((repetition * 127 + token_index) % 492);
}

static bool fake_native_stop_at(int repetition, int token_index) {
    return state.inject_native_stop &&
        repetition == state.native_stop_repetition &&
        token_index == state.native_stop_index;
}

static int fake_session_argmax_excluding(ds4_session *session, int excluded) {
    (void)excluded;
    record_pointer_call(CALL_UNEXPECTED, session_rep(session), session, NULL);
    fail_contract("qualification lifecycle used EOS-excluding argmax instead of native-stop argmax");
    return -1;
}

static int fake_session_argmax(ds4_session *session) {
    const int repetition = session_rep(session);
    const int token_index = repetition >= 0 && repetition < 4
        ? state.decode_attempt_count[repetition] : -1;
    const int token = fake_native_stop_at(repetition, token_index)
        ? state.native_stop_token : fake_decode_token(repetition, token_index);
    record_pointer_call(CALL_TOKEN_CHOOSE, repetition, session, NULL);
    state.choose_count++;
    if (state.call_count != 0u) {
        call_record *call = &state.calls[state.call_count - 1u];
        call->value = token_index < 0 ? 0u : (uint64_t)token_index;
        call->token = token;
    }
    if (repetition < 0 || repetition >= 4 || session != state.sessions[repetition] ||
        token_index < 0 || token_index >= QUALIFICATION_OUTPUT_TOKENS ||
        (!fake_native_stop_at(repetition, token_index) &&
         token != fake_decode_token(repetition, token_index)) ||
        (fake_native_stop_at(repetition, token_index) &&
         token != literal_eos_token && token != literal_native_stop_token)) {
        fail_contract("native-stop chooser did not select the expected indexed token");
    }
    return token;
}

static int fake_session_sync_attributed(
        ds4_session *session, const ds4_tokens *prompt,
        ds4_runtime_request_context *request, char *error, size_t error_size) {
    (void)error;
    (void)error_size;
    const int repetition = session_rep(session);
    record_pointer_call(CALL_SESSION_SYNC_ATTRIBUTED, repetition, session, request);
    state.sync_count++;
    if (repetition < 0 || request_rep(request) != repetition || !prompt ||
        prompt->len != (int)literal_prompt_tokens) {
        fail_contract("attributed prefill was not bound to this repetition and canonical prompt count");
        return 1;
    }
    return 0;
}

static int fake_session_eval_attributed(
        ds4_session *session, int token,
        ds4_runtime_request_context *request, char *error, size_t error_size) {
    const int repetition = session_rep(session);
    const int token_index = repetition >= 0 && repetition < 4
        ? state.decode_attempt_count[repetition] : -1;
    record_pointer_call(CALL_SESSION_EVAL_ATTRIBUTED, repetition, session, request);
    state.eval_count++;
    if (state.call_count != 0u) {
        call_record *call = &state.calls[state.call_count - 1u];
        call->value = token_index < 0 ? 0u : (uint64_t)token_index;
        call->token = token;
    }
    if (fake_token_is_stop(fake_engine, token) ||
        token != fake_decode_token(repetition, token_index) ||
        repetition < 0 || request_rep(request) != repetition ||
        token_index < 0 || token_index >= QUALIFICATION_OUTPUT_TOKENS) {
        fail_contract("attributed decode did not evaluate the expected indexed non-EOS token");
        return 1;
    }
    state.decode_attempt_count[repetition]++;
    if (state.inject_decode_failure &&
        repetition == state.decode_failure_repetition &&
        token_index == state.decode_failure_index) {
        state.decode_failed = true;
        if (error && error_size != 0u) {
            snprintf(error, error_size,
                     "injected decode failure repetition=%d token_index=%d",
                     repetition, token_index);
        }
        return 1;
    }
    state.decode_success_count[repetition]++;
    return 0;
}

static int fake_session_request_barrier(
        ds4_session *session, ds4_runtime_request_context *request,
        char *error, size_t error_size) {
    (void)error;
    (void)error_size;
    const int repetition = session_rep(session);
    record_pointer_call(CALL_REQUEST_BARRIER, repetition, session, request);
    state.barrier_count++;
    if (repetition < 0 || request_rep(request) != repetition) {
        fail_contract("request barrier was not bound to the current session/request");
        return 1;
    }
    return 0;
}

static int fake_session_pos(ds4_session *session) {
    (void)session;
    return (int)literal_prompt_tokens;
}

static int fake_session_ctx(ds4_session *session) {
    (void)session;
    return 32768;
}

static int fake_session_prefill_cap(ds4_session *session) {
    (void)session;
    return 4096;
}

static bool fake_session_is_distributed(ds4_session *session) {
    (void)session;
    return false;
}

static void fake_session_gpu_warmup(ds4_session *session) {
    (void)session;
}

static int fake_session_sync(ds4_session *session, const ds4_tokens *prompt,
                             char *error, size_t error_size) {
    (void)session;
    (void)prompt;
    (void)error;
    (void)error_size;
    record_call(CALL_UNEXPECTED, -1);
    fail_contract("qualification lifecycle used the unattributed prefill API");
    return 1;
}

static int fake_session_eval(ds4_session *session, int token,
                             char *error, size_t error_size) {
    (void)session;
    (void)token;
    (void)error;
    (void)error_size;
    record_call(CALL_UNEXPECTED, -1);
    fail_contract("qualification lifecycle used the unattributed decode API");
    return 1;
}

static bool fake_request_begin(ds4_runtime_request_context *request,
                               uint64_t accepted_monotonic_ns) {
    const int repetition = state.request_begin_count;
    record_pointer_call(CALL_REQUEST_BEGIN, repetition, NULL, request);
    state.request_begin_count++;
    if (!request || repetition >= 4) {
        fail_contract("request begin received an invalid repetition");
        return false;
    }
    memset(request, 0, sizeof(*request));
    expected_request_id(repetition, request->request_id,
                        sizeof(request->request_id));
    snprintf(request->instance_id, sizeof(request->instance_id), "%s",
             literal_instance_id);
    request->accepted_monotonic_ns = accepted_monotonic_ns ? accepted_monotonic_ns :
                                     UINT64_C(1000) + (uint64_t)repetition;
    request->initialized = true;
    state.requests[repetition] = request;
    state.request_repetition[repetition] = repetition;
    return true;
}

static bool fake_request_set_prompt(ds4_runtime_request_context *request,
                                    uint64_t prompt_tokens) {
    const int repetition = request_rep(request);
    record_pointer_call(CALL_REQUEST_PROMPT, repetition, NULL, request);
    state.request_prompt_count++;
    if (prompt_tokens != literal_prompt_tokens || repetition < 0) {
        fail_contract("request context was not bound to canonical sequence prompt tokens");
        return false;
    }
    request->prompt_tokens = prompt_tokens;
    request->prompt_tokens_set = true;
    return true;
}

static bool fake_request_mark_prefill_started(
        ds4_runtime_request_context *request, uint64_t timestamp) {
    const int repetition = request_rep(request);
    record_pointer_call(CALL_PREFILL_START, repetition, NULL, request);
    state.prefill_start_count++;
    if (repetition < 0 || timestamp == 0u) {
        fail_contract("prefill start did not use a valid request context");
        return false;
    }
    request->prefill_started = true;
    request->prefill_started_monotonic_ns = timestamp;
    return true;
}

static bool fake_request_mark_prefill_complete(
        ds4_runtime_request_context *request, uint64_t timestamp) {
    const int repetition = request_rep(request);
    record_pointer_call(CALL_PREFILL_COMPLETE, repetition, NULL, request);
    state.prefill_complete_count++;
    if (repetition < 0 || !request->prefill_started || timestamp == 0u) {
        fail_contract("prefill completion did not follow prefill start");
        return false;
    }
    request->prefill_complete = true;
    request->prefill_complete_monotonic_ns = timestamp;
    return true;
}

static bool fake_request_add_generated(ds4_runtime_request_context *request,
                                       uint64_t delta) {
    const int repetition = request_rep(request);
    record_pointer_call(CALL_GENERATED, repetition, NULL, request);
    state.generated_count++;
    if (state.call_count != 0u) {
        state.calls[state.call_count - 1u].value = request ? request->generated_tokens : 0u;
    }
    if (delta != 1u || repetition < 0 || !request ||
        request->generated_tokens + delta !=
            (uint64_t)state.decode_success_count[repetition]) {
        fail_contract("request accounting did not add one token per successful decode");
        return false;
    }
    request->generated_tokens += delta;
    return true;
}

static bool fake_request_record_visible(ds4_runtime_request_context *request,
                                        uint64_t delta, uint64_t timestamp) {
    const int repetition = request_rep(request);
    record_pointer_call(CALL_VISIBLE, repetition, NULL, request);
    state.visible_count++;
    if (state.call_count != 0u) {
        state.calls[state.call_count - 1u].value = request
            ? request->visible_generated_tokens : 0u;
    }
    if (delta != 1u || timestamp == 0u || repetition < 0 || !request ||
        request->visible_generated_tokens + delta != request->generated_tokens) {
        fail_contract("request accounting did not expose one token per generated decode");
        return false;
    }
    if (!request->visible_decode_started) {
        request->first_visible_decode_monotonic_ns = timestamp;
    }
    request->last_visible_decode_monotonic_ns = timestamp;
    request->visible_generated_tokens += delta;
    request->visible_decode_started = true;
    return true;
}

static bool fake_request_publish_visible(
        ds4_runtime_request_context *request, uint64_t visible_tokens,
        uint64_t first_timestamp, uint64_t last_timestamp) {
    return fake_request_record_visible(request, visible_tokens, last_timestamp) &&
           first_timestamp != 0u;
}

static bool fake_request_first_visible(ds4_runtime_request_context *request,
                                       uint64_t timestamp) {
    const int repetition = request_rep(request);
    record_pointer_call(CALL_FIRST_VISIBLE, repetition, NULL, request);
    state.first_visible_count++;
    if (repetition < 0 || timestamp == 0u || !request ||
        !request->visible_decode_started || request->visible_generated_tokens != 1u ||
        request->first_visible_emitted) {
        fail_contract("first-visible emission was not bound to the first visible token");
        return false;
    }
    request->first_visible_emitted = true;
    request->first_visible_emitted_monotonic_ns = timestamp;
    return true;
}

static bool fake_request_add_counters(ds4_runtime_request_context *request,
                                      const ds4_runtime_wire_counters *delta) {
    (void)request;
    (void)delta;
    return true;
}

static bool fake_request_observe_page_advice(ds4_runtime_request_context *request,
                                             uint64_t timestamp) {
    (void)request;
    (void)timestamp;
    return true;
}

static bool fake_request_record_page_advice_complete(
        ds4_runtime_request_context *request, uint64_t timestamp) {
    (void)request;
    (void)timestamp;
    return true;
}

static bool fake_request_finish(
        ds4_runtime_request_context *request,
        ds4_runtime_request_terminal_status status,
        uint64_t timestamp,
        ds4_runtime_request_metrics *metrics) {
    const int repetition = request_rep(request);
    record_pointer_call(CALL_REQUEST_FINISH, repetition, NULL, request);
    state.finish_count++;
    if (repetition < 0 || status != DS4_RUNTIME_REQUEST_COMPLETED ||
        timestamp == 0u || !metrics || !request || !request->prompt_tokens_set ||
        request->generated_tokens == 0u ||
        request->visible_generated_tokens != request->generated_tokens ||
        !request->visible_decode_started || !request->first_visible_emitted) {
        fail_contract("request finish did not complete all accounted tokens");
        return false;
    }
    memset(metrics, 0, sizeof(*metrics));
    snprintf(metrics->request_id, sizeof(metrics->request_id), "%s",
             request->request_id);
    snprintf(metrics->instance_id, sizeof(metrics->instance_id), "%s",
             request->instance_id);
    metrics->prompt_tokens = request->prompt_tokens;
    metrics->generated_tokens = request->generated_tokens;
    metrics->wall_time_ns = timestamp - request->accepted_monotonic_ns;
    metrics->ttft_present = true;
    metrics->ttft_ns = request->first_visible_emitted_monotonic_ns -
        request->accepted_monotonic_ns;
    const uint64_t prefill_elapsed =
        request->prefill_complete_monotonic_ns -
        request->prefill_started_monotonic_ns;
    if (prefill_elapsed != 0u) {
        metrics->prefill_tokens_per_second =
            (double)request->prompt_tokens * 1000000000.0 /
            (double)prefill_elapsed;
    }
    const uint64_t decode_elapsed =
        request->last_visible_decode_monotonic_ns -
        request->first_visible_decode_monotonic_ns;
    if (decode_elapsed != 0u) {
        metrics->visible_decode_tokens_per_second =
            (double)(request->visible_generated_tokens - 1u) *
            1000000000.0 / (double)decode_elapsed;
    }
    metrics->counters = request->counters;
    metrics->snapshot_seq = UINT64_C(99) + (uint64_t)state.snapshot_count +
        (uint64_t)(state.snapshot_count / 3) + 1u;
    metrics->terminal_status = status;
    request->terminal = true;
    return true;
}

static bool fake_validate_emission(
        bool resident,
        const ds4_bench_sequence *sequence,
        const ds4_bench_resident_sequence *resident_sequence,
        ds4_bench_qualification_event event,
        const char *request_id,
        uint32_t repetition_index,
        uint64_t monotonic_ns,
        uint64_t session_payload_bytes,
        const ds4_runtime_wire_snapshot *runtime_snapshot,
        const ds4_runtime_request_metrics *metrics,
        char *error,
        size_t error_size) {
    const int emission_count = resident ? state.resident_emit_count : state.emit_count;
    const int event_index = emission_count - 1;
    const ds4_bench_qualification_event expected_event =
        (ds4_bench_qualification_event)(event_index % 3);
    const int expected_repetition = event_index / 3;
    char expected_request[DS4_RUNTIME_INSTANCE_ID_CAPACITY];
    expected_request_id(expected_repetition, expected_request,
                        sizeof(expected_request));
    ds4_session *current_session =
        expected_repetition >= 0 && expected_repetition < state.session_create_count
            ? state.sessions[expected_repetition] : NULL;
    const bool sequence_pointer_valid = resident
        ? resident_sequence != NULL &&
          resident_sequence == state.trusted_resident_sequence &&
          sequence == &resident_sequence->sequence
        : resident_sequence == NULL && sequence == state.trusted_sequence;
    if (emission_count > 12 || monotonic_ns == 0u ||
        monotonic_ns <= state.last_emitted_monotonic_ns ||
        (current_session != NULL &&
         session_payload_bytes != fake_session_payload_bytes(current_session)) ||
        current_session == NULL || !sequence_pointer_valid ||
        event != expected_event ||
        !check_repetition((int)repetition_index, expected_repetition,
                          resident ? "resident emitter" : "emitter") ||
        !request_id || strcmp(request_id, expected_request) != 0 ||
        !runtime_snapshot ||
        strcmp(runtime_snapshot->instance_id, literal_instance_id) != 0 ||
        runtime_snapshot->snapshot_seq !=
            UINT64_C(99) + (uint64_t)emission_count +
                (uint64_t)(emission_count / 3)) {
        fail_contract(resident
            ? "resident emitter did not receive the exact ordered typed lifecycle record"
            : "emitter did not receive the exact ordered lifecycle record");
    }
    if (resident && (!runtime_snapshot ||
        runtime_snapshot->configured_ssd_streaming ||
        runtime_snapshot->configured_ssd_streaming_cache_bytes != 0u ||
        runtime_snapshot->expert_cache_limit_bytes != 0u ||
        runtime_snapshot->configured_prefill_rows != 4096u ||
        runtime_snapshot->allocated_prefill_rows != 4096u ||
        runtime_snapshot->allocations.category_current[
            DS4_RUNTIME_CATEGORY_EXPERT_CACHE_PAYLOAD] != 0u ||
        runtime_snapshot->allocations.category_peak[
            DS4_RUNTIME_CATEGORY_EXPERT_CACHE_PAYLOAD] != 0u ||
        runtime_snapshot->allocations.category_bounds[
            DS4_RUNTIME_CATEGORY_EXPERT_CACHE_PAYLOAD] != 0u ||
        runtime_snapshot->allocations.category_current[
            DS4_RUNTIME_CATEGORY_STATIC_WEIGHTS] == 0u ||
        runtime_snapshot->allocations.category_current[
            DS4_RUNTIME_CATEGORY_KV_STATE] == 0u ||
        runtime_snapshot->allocations.category_current[
            DS4_RUNTIME_CATEGORY_GRAPH_SCRATCH] == 0u ||
        runtime_snapshot->allocations.report_current[
            DS4_RUNTIME_REPORT_MODEL_SOURCE_RESIDENT] == 0u ||
        runtime_snapshot->allocations.report_current[
            DS4_RUNTIME_REPORT_HOST_LIBRARY_UNATTRIBUTED] == 0u ||
        runtime_snapshot->allocations.report_current[
            DS4_RUNTIME_REPORT_CUDA_LIBRARY_UNATTRIBUTED] == 0u ||
        !runtime_snapshot->allocations.external_sample
             .unrelated_process_inventory_stable)) {
        fail_contract("resident snapshot did not preserve truthful physical evidence");
    }
    if (monotonic_ns > state.last_emitted_monotonic_ns) {
        state.last_emitted_monotonic_ns = monotonic_ns;
    }
    if (event != DS4_BENCH_QUALIFICATION_EVENT_REQUEST_COMPLETE &&
        metrics != NULL) {
        fail_contract("accepted/first-token record unexpectedly carried metrics");
    }
    if (event == DS4_BENCH_QUALIFICATION_EVENT_REQUEST_COMPLETE) {
        if (!metrics || !runtime_snapshot ||
            metrics->terminal_status != DS4_RUNTIME_REQUEST_COMPLETED ||
            strcmp(metrics->request_id, expected_request) != 0 ||
            strcmp(metrics->instance_id, literal_instance_id) != 0 ||
            metrics->snapshot_seq == UINT64_MAX ||
            metrics->snapshot_seq + 1u != runtime_snapshot->snapshot_seq ||
            metrics->prompt_tokens != literal_prompt_tokens ||
            metrics->generated_tokens == 0u ||
            !metrics->ttft_present || metrics->ttft_ns == 0u ||
            metrics->ttft_ns > metrics->wall_time_ns ||
            metrics->wall_time_ns == 0u ||
            !(metrics->prefill_tokens_per_second > 0.0) ||
            !(metrics->visible_decode_tokens_per_second > 0.0)) {
            fail_contract("completion metrics did not carry the full 512-token lifecycle");
        } else if (expected_repetition >= 0 && expected_repetition < 4) {
            state.completed_metrics[expected_repetition] = *metrics;
            state.completed_metrics_count++;
        }
    }
    if (state.inject_emitter_failure && !state.emitter_failed) {
        state.emitter_failed = true;
        if (error && error_size != 0u) {
            snprintf(error, error_size, "injected fake emitter failure");
        }
        return false;
    }
    return true;
}

static bool fake_emit_record(
        FILE *stream,
        const ds4_bench_qualification_record *record,
        char *error,
        size_t error_size) {
    (void)stream;
    const int repetition = record ? (int)record->repetition_index : -1;
    record_call(CALL_EMIT, repetition);
    state.emit_count++;
    if (state.call_count != 0u) {
        call_record *call = &state.calls[state.call_count - 1u];
        call->sequence = record ? record->sequence : NULL;
        call->event = record ? record->event : DS4_BENCH_QUALIFICATION_EVENT_REQUEST_ACCEPTED;
        call->metrics = record ? record->request_metrics : NULL;
        call->value = record && record->runtime_snapshot ?
            record->runtime_snapshot->snapshot_seq : 0u;
    }
    return fake_validate_emission(
        false,
        record ? record->sequence : NULL,
        NULL,
        record ? record->event : DS4_BENCH_QUALIFICATION_EVENT_REQUEST_ACCEPTED,
        record ? record->request_id : NULL,
        record ? record->repetition_index : 0u,
        record ? record->monotonic_ns : 0u,
        record ? record->session_payload_bytes : 0u,
        record ? record->runtime_snapshot : NULL,
        record ? record->request_metrics : NULL,
        error,
        error_size);
}

static bool fake_emit_resident_record(
        FILE *stream,
        const ds4_bench_resident_qualification_record *record,
        char *error,
        size_t error_size) {
    (void)stream;
    const int repetition = record ? (int)record->repetition_index : -1;
    record_call(CALL_RESIDENT_EMIT, repetition);
    state.resident_emit_count++;
    if (state.call_count != 0u) {
        call_record *call = &state.calls[state.call_count - 1u];
        call->sequence = record && record->sequence
            ? &record->sequence->sequence : NULL;
        call->resident_sequence = record ? record->sequence : NULL;
        call->event = record ? record->event : DS4_BENCH_QUALIFICATION_EVENT_REQUEST_ACCEPTED;
        call->metrics = record ? record->request_metrics : NULL;
        call->value = record && record->runtime_snapshot ?
            record->runtime_snapshot->snapshot_seq : 0u;
    }
    return fake_validate_emission(
        true,
        record && record->sequence ? &record->sequence->sequence : NULL,
        record ? record->sequence : NULL,
        record ? record->event : DS4_BENCH_QUALIFICATION_EVENT_REQUEST_ACCEPTED,
        record ? record->request_id : NULL,
        record ? record->repetition_index : 0u,
        record ? record->monotonic_ns : 0u,
        record ? record->session_payload_bytes : 0u,
        record ? record->runtime_snapshot : NULL,
        record ? record->request_metrics : NULL,
        error,
        error_size);
}

#ifdef DS4_BENCH_LIFECYCLE_REAL_EMITTER
/* Compose each structural observer with its linked typed production emitter.
 * The runner's exact record, stream, and error arguments pass through without
 * reconstruction.  The literal fake snapshots satisfy the production ABI. */
static bool lifecycle_real_emit_record(
        FILE *stream,
        const ds4_bench_qualification_record *record,
        char *error,
        size_t error_size) {
    if (!fake_emit_record(stream, record, error, error_size)) return false;
    return ds4_bench_qualification_emit_record(
        stream, record, error, error_size);
}

static bool lifecycle_real_resident_emit_record(
        FILE *stream,
        const ds4_bench_resident_qualification_record *record,
        char *error,
        size_t error_size) {
    if (!fake_emit_resident_record(stream, record, error, error_size)) return false;
    return ds4_bench_resident_qualification_emit_record(
        stream, record, error, error_size);
}
#endif

/* Macro substitution is deliberately limited to the lifecycle-facing APIs. */
#define main ds4_bench_test_cli_main
/* Keep the CUDA-only qualification branch testable in this host-only fake
 * backend while normal DS4_NO_GPU builds remain fail-closed. */
#define DS4_BENCH_QUALIFICATION_TEST_BACKEND 1
#define ds4_bench_sequence_parse_file_trusted fake_sequence_parse_file_trusted
#define ds4_bench_sequence_free fake_sequence_free
#define ds4_bench_resident_sequence_parse_file_trusted fake_resident_sequence_parse_file_trusted
#define ds4_bench_resident_sequence_free fake_resident_sequence_free
#define ds4_engine_open fake_engine_open
#define ds4_engine_create_with_gpu_config fake_engine_create_with_gpu_config
#define ds4_engine_close fake_engine_close
#define ds4_gpu_nvml_inventory_capture fake_gpu_nvml_inventory_capture
#define ds4_engine_vocab_size fake_engine_vocab_size
#define ds4_engine_prefill_chunk fake_engine_prefill_chunk
#define ds4_engine_routed_quant_bits fake_engine_routed_quant_bits
#define ds4_engine_runtime_snapshot fake_engine_runtime_snapshot
#define ds4_engine_laguna_external_checkpoint fake_engine_laguna_external_checkpoint
#define ds4_session_create fake_session_create
#define ds4_session_free fake_session_free
#define ds4_session_sync_attributed fake_session_sync_attributed
#define ds4_session_eval_attributed fake_session_eval_attributed
#define ds4_session_request_barrier fake_session_request_barrier
#define ds4_session_pos fake_session_pos
#define ds4_session_ctx fake_session_ctx
#define ds4_session_prefill_cap fake_session_prefill_cap
#define ds4_session_is_distributed fake_session_is_distributed
#define ds4_session_gpu_warmup fake_session_gpu_warmup
#define ds4_session_sync fake_session_sync
#define ds4_session_eval fake_session_eval
#define ds4_session_argmax_excluding fake_session_argmax_excluding
#define ds4_session_argmax fake_session_argmax
#define ds4_session_payload_bytes fake_session_payload_bytes
#define ds4_token_eos fake_token_eos
#define ds4_token_is_stop fake_token_is_stop
#define ds4_tokenize_text fake_tokenize_text
#define ds4_tokenize_rendered_chat fake_tokenize_rendered_chat
#define ds4_encode_chat_prompt fake_encode_chat_prompt
#define ds4_tokens_free fake_tokens_free
#define ds4_runtime_request_begin fake_request_begin
#define ds4_runtime_request_set_prompt_tokens fake_request_set_prompt
#define ds4_runtime_request_mark_prefill_started fake_request_mark_prefill_started
#define ds4_runtime_request_mark_prefill_complete fake_request_mark_prefill_complete
#define ds4_runtime_request_add_generated_tokens fake_request_add_generated
#define ds4_runtime_request_record_visible_decoded fake_request_record_visible
#define ds4_runtime_request_publish_visible_decode_window fake_request_publish_visible
#define ds4_runtime_request_mark_first_visible_emitted fake_request_first_visible
#define ds4_runtime_request_add_counters fake_request_add_counters
#define ds4_runtime_request_observe_page_advice fake_request_observe_page_advice
#define ds4_runtime_request_record_page_advice_complete fake_request_record_page_advice_complete
#define ds4_runtime_request_finish fake_request_finish
#ifdef DS4_BENCH_LIFECYCLE_REAL_EMITTER
#define ds4_bench_qualification_emit_record lifecycle_real_emit_record
#define ds4_bench_resident_qualification_emit_record lifecycle_real_resident_emit_record
#else
#define ds4_bench_qualification_emit_record fake_emit_record
#define ds4_bench_resident_qualification_emit_record fake_emit_resident_record
#endif
#include "../ds4_bench.c"
#undef main

static int first_call(enum call_kind kind, int repetition) {
    for (size_t i = 0; i < state.call_count; i++) {
        if (state.calls[i].kind == kind &&
            (repetition < 0 || state.calls[i].repetition == repetition)) {
            return (int)i;
        }
    }
    return -1;
}

static int nth_call(enum call_kind kind, int repetition, int ordinal) {
    for (size_t i = 0; i < state.call_count; i++) {
        if (state.calls[i].kind == kind &&
            (repetition < 0 || state.calls[i].repetition == repetition)) {
            if (ordinal-- == 0) return (int)i;
        }
    }
    return -1;
}

static void require_call_order(int before, int after, const char *message) {
    if (before < 0 || after < 0 || before >= after) {
        fail_contract(message);
    }
}

static void check_lifecycle_shape(bool resident) {
    const enum call_kind emit_kind = resident ? CALL_RESIDENT_EMIT : CALL_EMIT;
    const enum call_kind parse_kind = resident
        ? CALL_RESIDENT_SEQUENCE_PARSE : CALL_SEQUENCE_PARSE;
    const enum call_kind sequence_free_kind = resident
        ? CALL_RESIDENT_SEQUENCE_FREE : CALL_SEQUENCE_FREE;
    const enum call_kind other_parse_kind = resident
        ? CALL_SEQUENCE_PARSE : CALL_RESIDENT_SEQUENCE_PARSE;
    const int emission_count = resident ? state.resident_emit_count : state.emit_count;
    int event_count = 0;
    for (size_t i = 0; i < state.call_count; i++) {
        if (state.calls[i].kind != emit_kind) continue;
        if (event_count >= 12) {
            fail_contract(resident
                ? "more than twelve resident lifecycle emissions occurred"
                : "more than twelve lifecycle emissions occurred");
            break;
        }
        if (state.calls[i].event !=
                (ds4_bench_qualification_event)(event_count % 3) ||
            state.calls[i].repetition != event_count / 3) {
            fail_contract(resident
                ? "resident milestones were not emitted as accepted, first-token, complete per repetition"
                : "milestones were not emitted as accepted, first-token, complete per repetition");
        }
        if (resident && (state.calls[i].resident_sequence == NULL ||
                         state.calls[i].resident_sequence !=
                             state.trusted_resident_sequence)) {
            fail_contract("resident milestones did not retain the typed sequence owner");
        }
        if (!resident && state.calls[i].resident_sequence != NULL) {
            fail_contract("streamed milestone unexpectedly carried resident sequence metadata");
        }
        event_count++;
    }
    if (event_count != 12 || emission_count != 12) {
        fail_contract(resident
            ? "expected exactly twelve resident lifecycle emissions"
            : "expected exactly twelve lifecycle emissions");
    }
    if (state.resident_mode != resident) {
        fail_contract("fake lifecycle mode did not match the requested typed path");
    }
    if (state.engine_open_count != 1 || state.engine_close_count != 1) {
        fail_contract("expected exactly one engine open and close");
    }
    if (state.sequence_free_count != (resident ? 0 : 1) ||
        state.resident_sequence_free_count != (resident ? 1 : 0)) {
        fail_contract("expected exactly one cleanup of the selected typed sequence owner");
    }
    if (first_call(parse_kind, -1) < 0 || first_call(other_parse_kind, -1) >= 0) {
        fail_contract("lifecycle did not use only the parser for its explicit mode");
    }
    if (state.nvml_capture_count != 1 || state.tokenize_rendered_count != 1) {
        fail_contract("expected one pre-engine NVML capture and one rendered-input tokenization");
    }
    const int parse = first_call(parse_kind, -1);
    const int engine_open = first_call(CALL_ENGINE_OPEN, -1);
    const int nvml_capture = first_call(CALL_NVML_CAPTURE, -1);
    const int rendered_tokenize = first_call(CALL_TOKENIZE_RENDERED, -1);
    const int first_session = first_call(CALL_SESSION_CREATE, -1);
    const int engine_close = first_call(CALL_ENGINE_CLOSE, -1);
    const int last_free = nth_call(CALL_SESSION_FREE, -1, 3);
    const int sequence_free = first_call(sequence_free_kind, -1);
    require_call_order(parse, nvml_capture,
                       "typed sequence parse must precede pre-child NVML capture");
    require_call_order(nvml_capture, engine_open,
                       "pre-child NVML capture must precede engine open");
    require_call_order(engine_open, rendered_tokenize,
                       "engine must open before rendered sequence tokenization");
    require_call_order(rendered_tokenize, first_session,
                       "rendered sequence tokenization must precede the first qualification session");
    require_call_order(engine_open, first_session,
                       "engine must open before the first qualification session");
    require_call_order(last_free, engine_close,
                       "engine must remain open until the last qualification session is freed");
    require_call_order(engine_close, sequence_free,
                       "typed sequence cleanup must occur after engine close");
    if (state.session_create_count != 4 || state.session_free_count != 4) {
        fail_contract("expected four fresh sessions and four frees");
    }
    if (state.request_begin_count != 4 || state.request_prompt_count != 4) {
        fail_contract("expected one request context begin and prompt binding per repetition");
    }
    if (state.prefill_start_count != 4 || state.sync_count != 4 ||
        state.prefill_complete_count != 4 ||
        state.choose_count != 4 * QUALIFICATION_OUTPUT_TOKENS ||
        state.eval_count != 4 * QUALIFICATION_OUTPUT_TOKENS ||
        state.generated_count != 4 * QUALIFICATION_OUTPUT_TOKENS ||
        state.visible_count != 4 * QUALIFICATION_OUTPUT_TOKENS ||
        state.first_visible_count != 4 || state.barrier_count != 4 ||
        state.finish_count != 4 || state.checkpoint_count != 12 ||
        state.snapshot_count != 12 || state.completed_metrics_count != 4) {
        fail_contract("lifecycle operation counts did not match four 512-token repetitions");
    }
    for (int repetition = 0; repetition < 4; repetition++) {
        const ds4_runtime_request_metrics *metrics =
            &state.completed_metrics[repetition];
        if (metrics->prompt_tokens != literal_prompt_tokens ||
            metrics->generated_tokens != QUALIFICATION_OUTPUT_TOKENS ||
            !metrics->ttft_present || metrics->ttft_ns == 0u ||
            metrics->wall_time_ns == 0u || metrics->ttft_ns > metrics->wall_time_ns ||
            !(metrics->prefill_tokens_per_second > 0.0) ||
            !(metrics->visible_decode_tokens_per_second > 0.0) ||
            metrics->terminal_status != DS4_RUNTIME_REQUEST_COMPLETED) {
            fail_contract("full completion metrics were not retained for every repetition");
        }
        const int create = nth_call(CALL_SESSION_CREATE, repetition, 0);
        const int request = nth_call(CALL_REQUEST_BEGIN, repetition, 0);
        const int prompt = nth_call(CALL_REQUEST_PROMPT, repetition, 0);
        const int accepted_checkpoint = nth_call(CALL_EXTERNAL_CHECKPOINT, -1, repetition * 3);
        const int accepted_snapshot = nth_call(CALL_RUNTIME_SNAPSHOT, -1, repetition * 3);
        const int accepted = nth_call(emit_kind, repetition, 0);
        const int prefill_start = nth_call(CALL_PREFILL_START, repetition, 0);
        const int sync = nth_call(CALL_SESSION_SYNC_ATTRIBUTED, repetition, 0);
        const int prefill_complete = nth_call(CALL_PREFILL_COMPLETE, repetition, 0);
        const int choose = nth_call(CALL_TOKEN_CHOOSE, repetition, 0);
        const int eval = nth_call(CALL_SESSION_EVAL_ATTRIBUTED, repetition, 0);
        const int generated = nth_call(CALL_GENERATED, repetition, 0);
        const int visible = nth_call(CALL_VISIBLE, repetition, 0);
        const int first_visible = nth_call(CALL_FIRST_VISIBLE, repetition, 0);
        const int first_checkpoint = nth_call(CALL_EXTERNAL_CHECKPOINT, -1, repetition * 3 + 1);
        const int first_snapshot = nth_call(CALL_RUNTIME_SNAPSHOT, -1, repetition * 3 + 1);
        const int first = nth_call(emit_kind, repetition, 1);
        const int barrier = nth_call(CALL_REQUEST_BARRIER, repetition, 0);
        const int finish = nth_call(CALL_REQUEST_FINISH, repetition, 0);
        const int complete_checkpoint = nth_call(CALL_EXTERNAL_CHECKPOINT, -1, repetition * 3 + 2);
        const int complete_snapshot = nth_call(CALL_RUNTIME_SNAPSHOT, -1, repetition * 3 + 2);
        const int complete = nth_call(emit_kind, repetition, 2);
        const int free = nth_call(CALL_SESSION_FREE, repetition, 0);
        require_call_order(request, prompt, "request prompt binding must follow request begin");
        require_call_order(prompt, accepted_checkpoint, "accepted checkpoint must follow request binding");
        require_call_order(accepted_checkpoint, accepted_snapshot, "accepted checkpoint must precede its runtime snapshot");
        require_call_order(accepted_snapshot, accepted, "accepted emission must follow its coherent snapshot");
        if (accepted_snapshot < 0 || accepted != accepted_snapshot + 1 ||
            accepted_snapshot != accepted_checkpoint + 1) {
            fail_contract("accepted emission was not made from one just-captured checkpoint/snapshot pair");
        }
        require_call_order(accepted, prefill_start, "request acceptance must precede prefill work");
        require_call_order(prefill_start, sync, "prefill start must precede attributed sync");
        require_call_order(sync, prefill_complete, "attributed sync must precede prefill complete");
        require_call_order(prefill_complete, choose, "token choice must follow prefill complete");
        require_call_order(choose, eval, "token evaluation must follow token choice");
        require_call_order(eval, generated, "generated-token accounting must follow evaluation");
        require_call_order(generated, visible, "visible-token accounting must follow generated accounting");
        require_call_order(visible, first_visible, "first-visible emission must follow visible accounting");
        require_call_order(first_visible, first_checkpoint, "first-token checkpoint must follow first-visible accounting");
        if (state.decode_attempt_count[repetition] != QUALIFICATION_OUTPUT_TOKENS ||
            state.decode_success_count[repetition] != QUALIFICATION_OUTPUT_TOKENS) {
            fail_contract("repetition did not attempt and complete exactly 512 decodes");
        }
        for (int token_index = 0; token_index < QUALIFICATION_OUTPUT_TOKENS; token_index++) {
            const int indexed_choose = nth_call(CALL_TOKEN_CHOOSE, repetition, token_index);
            const int indexed_eval = nth_call(CALL_SESSION_EVAL_ATTRIBUTED, repetition, token_index);
            const int indexed_generated = nth_call(CALL_GENERATED, repetition, token_index);
            const int indexed_visible = nth_call(CALL_VISIBLE, repetition, token_index);
            if (indexed_choose < 0 || indexed_eval < 0 || indexed_generated < 0 ||
                indexed_visible < 0 ||
                state.calls[indexed_choose].value != (uint64_t)token_index ||
                state.calls[indexed_eval].value != (uint64_t)token_index ||
                state.calls[indexed_choose].token != fake_decode_token(repetition, token_index) ||
                state.calls[indexed_eval].token != fake_decode_token(repetition, token_index) ||
                state.calls[indexed_eval].token == literal_eos_token ||
                state.calls[indexed_generated].value != (uint64_t)token_index ||
                state.calls[indexed_visible].value != (uint64_t)token_index) {
                fail_contract("per-token decode/accounting timeline was not indexed and non-EOS");
            }
            require_call_order(indexed_choose, indexed_eval,
                               "indexed token evaluation must follow its choice");
            require_call_order(indexed_eval, indexed_generated,
                               "indexed generated accounting must follow evaluation");
            require_call_order(indexed_generated, indexed_visible,
                               "indexed visible accounting must follow generated accounting");
            if (token_index == 0) {
                require_call_order(prefill_complete, indexed_choose,
                                   "first decode must follow prefill completion");
            } else {
                const int previous_visible = nth_call(CALL_VISIBLE, repetition, token_index - 1);
                require_call_order(previous_visible, indexed_choose,
                                   "each decode must follow the previous visible token");
            }
        }
        const int last_choose = nth_call(CALL_TOKEN_CHOOSE, repetition,
                                         QUALIFICATION_OUTPUT_TOKENS - 1);
        const int last_eval = nth_call(CALL_SESSION_EVAL_ATTRIBUTED, repetition,
                                       QUALIFICATION_OUTPUT_TOKENS - 1);
        const int last_generated = nth_call(CALL_GENERATED, repetition,
                                            QUALIFICATION_OUTPUT_TOKENS - 1);
        const int last_visible = nth_call(CALL_VISIBLE, repetition,
                                          QUALIFICATION_OUTPUT_TOKENS - 1);
        require_call_order(first, last_choose,
                           "remaining 511 decodes must follow the first-token milestone");
        require_call_order(last_choose, last_eval,
                           "last token evaluation must follow its choice");
        require_call_order(last_eval, last_generated,
                           "last generated accounting must follow its evaluation");
        require_call_order(last_generated, last_visible,
                           "last visible accounting must follow its generated accounting");
        require_call_order(last_visible, barrier,
                           "request barrier must follow all 512 visible tokens");
        require_call_order(first_checkpoint, first_snapshot, "first-token checkpoint must precede its runtime snapshot");
        require_call_order(first_snapshot, first, "first-token emission must follow its coherent snapshot");
        if (first_snapshot < 0 || first != first_snapshot + 1 ||
            first_snapshot != first_checkpoint + 1) {
            fail_contract("first-token emission was not made from one just-captured checkpoint/snapshot pair");
        }
        require_call_order(first, barrier, "request barrier must follow first-token emission");
        require_call_order(barrier, finish, "request barrier must precede runtime request finish");
        require_call_order(finish, complete_checkpoint, "completion checkpoint must follow request finish");
        require_call_order(complete_checkpoint, complete_snapshot, "completion checkpoint must precede its runtime snapshot");
        require_call_order(complete_snapshot, complete, "completion emission must follow its coherent snapshot");
        if (complete_snapshot < 0 || complete != complete_snapshot + 1 ||
            complete_snapshot != complete_checkpoint + 1) {
            fail_contract("completion emission was not made from one just-captured checkpoint/snapshot pair");
        }
        require_call_order(complete, free, "session free must follow completion emission");
        if (repetition == 3) {
            require_call_order(complete, sequence_free,
                               "final completion event must precede typed sequence cleanup");
        }
        if (repetition > 0) {
            const int previous_free = nth_call(CALL_SESSION_FREE, repetition - 1, 0);
            require_call_order(previous_free, create,
                               "the next fresh session must follow the previous session free");
        }
        (void)create;
        (void)accepted;
        (void)first;
        (void)complete;
    }
    for (int i = 0; i < 4; i++) {
        for (int j = i + 1; j < 4; j++) {
            if (state.sessions[i] == state.sessions[j]) {
                fail_contract("repetition sessions were not distinct");
            }
        }
    }
}

static void reset_fake_state(bool inject_emitter_failure) {
    memset(&state, 0, sizeof(state));
    literal_sequence_input[sizeof(literal_input) - 1u] = 0x7fu;
    resident_sequence_input[sizeof(resident_literal_input) - 1u] = 0x7fu;
    state.inject_emitter_failure = inject_emitter_failure;
}

static int invoke_bench(void) {
    char *argv[] = {
        (char *)"ds4-bench",
        (char *)"--qualification-sequence", (char *)"/literal/sequence.txt",
        (char *)"--qualification-manifest-sha256", (char *)literal_manifest_sha256,
        (char *)"--qualification-sequence-sha256", (char *)literal_sequence_sha256,
        (char *)"--model", (char *)"/literal/fake.gguf",
        (char *)"--backend", (char *)"cuda",
        (char *)"--qualification-control-fd", (char *)"9",
        NULL,
    };
    return ds4_bench_test_cli_main((int)(ARRAY_LEN(argv) - 1u), argv);
}

static int invoke_bench_resident(void) {
    char *argv[] = {
        (char *)"ds4-bench",
        (char *)"--qualification-resident-sequence",
        (char *)"/literal/resident-sequence.txt",
        (char *)"--qualification-manifest-sha256", (char *)resident_manifest_sha256,
        (char *)"--qualification-sequence-sha256", (char *)resident_sequence_sha256,
        (char *)"--model", (char *)"/literal/fake.gguf",
        (char *)"--backend", (char *)"cuda",
        (char *)"--qualification-control-fd", (char *)"9",
        NULL,
    };
    return ds4_bench_test_cli_main((int)(ARRAY_LEN(argv) - 1u), argv);
}

static int invoke_bench_resident_streamed_fixture(void) {
    char *argv[] = {
        (char *)"ds4-bench",
        (char *)"--qualification-resident-sequence",
        (char *)"/literal/sequence.txt",
        (char *)"--qualification-manifest-sha256", (char *)literal_manifest_sha256,
        (char *)"--qualification-sequence-sha256", (char *)literal_sequence_sha256,
        (char *)"--model", (char *)"/literal/fake.gguf",
        (char *)"--backend", (char *)"cuda",
        (char *)"--qualification-control-fd", (char *)"9",
        NULL,
    };
    return ds4_bench_test_cli_main((int)(ARRAY_LEN(argv) - 1u), argv);
}

static int invoke_bench_streamed_resident_fixture(void) {
    char *argv[] = {
        (char *)"ds4-bench",
        (char *)"--qualification-sequence", (char *)"/literal/resident-sequence.txt",
        (char *)"--qualification-manifest-sha256", (char *)resident_manifest_sha256,
        (char *)"--qualification-sequence-sha256", (char *)resident_sequence_sha256,
        (char *)"--model", (char *)"/literal/fake.gguf",
        (char *)"--backend", (char *)"cuda",
        (char *)"--qualification-control-fd", (char *)"9",
        NULL,
    };
    return ds4_bench_test_cli_main((int)(ARRAY_LEN(argv) - 1u), argv);
}

static void check_native_stop_termination(
        bool resident, int stop_token, const char *label) {
    const int stopped_repetition = SHORT_NATIVE_STOP_REPETITION;
    const int stopped_output_tokens = SHORT_NATIVE_STOP_INDEX;
    const int expected_successes = 3 * QUALIFICATION_OUTPUT_TOKENS +
        stopped_output_tokens;
    const int expected_choices = expected_successes + 1;
    const int expected_emit_count = 4 * 3;
    const int expected_tokens[4] = {
        QUALIFICATION_OUTPUT_TOKENS, QUALIFICATION_OUTPUT_TOKENS,
        stopped_output_tokens, QUALIFICATION_OUTPUT_TOKENS,
    };

    reset_fake_state(false);
    state.inject_native_stop = true;
    state.native_stop_repetition = stopped_repetition;
    state.native_stop_index = SHORT_NATIVE_STOP_INDEX;
    state.native_stop_token = stop_token;

    const int rc = resident ? invoke_bench_resident() : invoke_bench();
    if (rc != 0) {
        fprintf(stderr, "RED: %s native-stop path returned %d\n", label, rc);
        fail_contract("model-native stop path did not complete the bounded run");
    }
    const int emit_count = resident ? state.resident_emit_count : state.emit_count;
    if (state.session_create_count != 4 || state.session_free_count != 4 ||
        state.request_begin_count != 4 || state.request_prompt_count != 4 ||
        state.prefill_start_count != 4 || state.sync_count != 4 ||
        state.prefill_complete_count != 4 || state.choose_count != expected_choices ||
        state.eval_count != expected_successes ||
        state.generated_count != expected_successes ||
        state.visible_count != expected_successes || state.first_visible_count != 4 ||
        state.barrier_count != 4 || state.finish_count != 4 ||
        state.checkpoint_count != expected_emit_count ||
        state.snapshot_count != expected_emit_count ||
        state.completed_metrics_count != 4 || emit_count != expected_emit_count ||
        state.emit_count != (resident ? 0 : expected_emit_count) ||
        state.resident_emit_count != (resident ? expected_emit_count : 0) ||
        state.engine_open_count != 1 || state.engine_close_count != 1 ||
        state.sequence_free_count != (resident ? 0 : 1) ||
        state.resident_sequence_free_count != (resident ? 1 : 0)) {
        fail_contract("native-stop path changed bounded lifecycle counts");
    }
    for (int repetition = 0; repetition < 4; repetition++) {
        if (state.decode_attempt_count[repetition] != expected_tokens[repetition] ||
            state.decode_success_count[repetition] != expected_tokens[repetition] ||
            state.completed_metrics[repetition].generated_tokens !=
                (uint64_t)expected_tokens[repetition]) {
            fail_contract("native-stop path did not report actual generated-token metrics");
        }
    }

    const int stop_choose = nth_call(
        CALL_TOKEN_CHOOSE, stopped_repetition, SHORT_NATIVE_STOP_INDEX);
    const int stop_eval = nth_call(
        CALL_SESSION_EVAL_ATTRIBUTED, stopped_repetition, SHORT_NATIVE_STOP_INDEX);
    const int stop_generated = nth_call(
        CALL_GENERATED, stopped_repetition, SHORT_NATIVE_STOP_INDEX);
    const int stop_visible = nth_call(
        CALL_VISIBLE, stopped_repetition, SHORT_NATIVE_STOP_INDEX);
    const int previous_visible = nth_call(
        CALL_VISIBLE, stopped_repetition, SHORT_NATIVE_STOP_INDEX - 1);
    const int barrier = nth_call(CALL_REQUEST_BARRIER, stopped_repetition, 0);
    if (stop_choose < 0 || state.calls[stop_choose].value !=
            (uint64_t)SHORT_NATIVE_STOP_INDEX ||
        state.calls[stop_choose].token != stop_token ||
        !fake_token_is_stop(fake_engine, state.calls[stop_choose].token) ||
        stop_eval >= 0 || stop_generated >= 0 || stop_visible >= 0) {
        fail_contract("model-native stop was evaluated or accounted as output");
    }
    require_call_order(previous_visible, stop_choose,
                       "model-native stop choice must follow the visible prefix");
    require_call_order(stop_choose, barrier,
                       "model-native stop choice must precede request barrier");
    if (nth_call(CALL_TOKEN_CHOOSE, stopped_repetition,
                 SHORT_NATIVE_STOP_INDEX + 1) >= 0) {
        fail_contract("model-native stop did not terminate before the next choice");
    }
    for (size_t i = stop_choose >= 0 ? (size_t)stop_choose + 1u : 0u;
         i < state.call_count; i++) {
        const call_record *call = &state.calls[i];
        if (call->repetition == stopped_repetition &&
            (call->kind == CALL_TOKEN_CHOOSE ||
             call->kind == CALL_SESSION_EVAL_ATTRIBUTED ||
             call->kind == CALL_GENERATED || call->kind == CALL_VISIBLE)) {
            fail_contract("model-native stop allowed token work after the stop choice");
        }
    }
    if (state.completed_metrics[stopped_repetition].generated_tokens !=
            (uint64_t)stopped_output_tokens ||
        state.completed_metrics[stopped_repetition].terminal_status !=
            DS4_RUNTIME_REQUEST_COMPLETED) {
        fail_contract("native EOS/stop completion metrics did not carry the actual prefix");
    }
    for (int i = 0; i < 4; i++) {
        for (int j = i + 1; j < 4; j++) {
            if (state.sessions[i] == state.sessions[j]) {
                fail_contract("native-stop repetitions reused a session");
            }
        }
    }
    all_failures += state.contract_failures;
}

static void check_native_stop_before_output(bool resident) {
    reset_fake_state(false);
    state.inject_native_stop = true;
    state.native_stop_repetition = 0;
    state.native_stop_index = 0;
    state.native_stop_token = literal_eos_token;
    const int rc = resident ? invoke_bench_resident() : invoke_bench();
    const int choice = nth_call(CALL_TOKEN_CHOOSE, 0, 0);
    if (rc == 0 || choice < 0 || state.calls[choice].token != literal_eos_token ||
        state.choose_count != 1 || state.eval_count != 0 ||
        state.generated_count != 0 || state.visible_count != 0 ||
        state.first_visible_count != 0 || state.finish_count != 0 ||
        state.barrier_count != 0 || state.checkpoint_count != 1 ||
        state.snapshot_count != 1 || state.completed_metrics_count != 0 ||
        state.session_create_count != 1 || state.session_free_count != 1 ||
        state.engine_open_count != 1 || state.engine_close_count != 1 ||
        state.emit_count != (resident ? 0 : 1) ||
        state.resident_emit_count != (resident ? 1 : 0) ||
        state.sequence_free_count != (resident ? 0 : 1) ||
        state.resident_sequence_free_count != (resident ? 1 : 0)) {
        fail_contract("native stop before output must refuse without a fabricated first token or completion");
    }
    all_failures += state.contract_failures;
}

static void check_happy_path(void) {
    reset_fake_state(false);
    const int rc = invoke_bench();
    if (rc != 0) {
        fprintf(stderr,
                "RED: authenticated qualification sequence still has no real lifecycle runner "
                "(valid fake sequence returned %d before engine/lifecycle calls)\n", rc);
    }
    if (state.emit_count == 0 && rc != 0) {
        /* Keep the deliberate pre-implementation failure explicit, but still
         * run all strict assertions once the runner exists. */
        fail_contract("happy-path fake lifecycle emitted no milestones");
    }
    check_lifecycle_shape(false);
    all_failures += state.contract_failures;
}

static void check_fail_closed_cleanup(void) {
    /* The injected failure is an emitter rejection of the first accepted
     * record, before any backend work.  It is therefore safe to require
     * current-session/engine teardown, but this oracle intentionally does not
     * invent a request barrier after an operation that never touched the
     * backend. */
    reset_fake_state(true);
    const int rc = invoke_bench();
    if (rc == 0) fail_contract("injected emitter failure unexpectedly returned success");
    if (!state.emitter_failed || state.emit_count != 1) {
        fail_contract("injected emitter failure did not stop at its first milestone");
    }
    if (state.session_create_count != 1 || state.session_free_count != 1) {
        fail_contract("injected emitter failure did not free the current session exactly once");
    }
    if (state.engine_open_count != 1 || state.engine_close_count != 1) {
        fail_contract("injected emitter failure did not close its one engine exactly once");
    }
    if (state.sequence_free_count != 1) {
        fail_contract("injected emitter failure did not free the trusted sequence exactly once");
    }
    int failed_emit = -1;
    for (size_t i = 0; i < state.call_count; i++) {
        if (state.calls[i].kind == CALL_EMIT) {
            if (failed_emit < 0) failed_emit = (int)i;
            else fail_contract("emission failure allowed a later milestone");
        }
    }
    if (failed_emit >= 0) {
        const int sequence_free = first_call(CALL_SEQUENCE_FREE, -1);
        require_call_order(failed_emit, sequence_free,
                           "injected emitter failure must precede trusted sequence cleanup");
        for (size_t i = (size_t)failed_emit + 1u; i < state.call_count; i++) {
            const enum call_kind kind = state.calls[i].kind;
            if (kind != CALL_SESSION_FREE && kind != CALL_ENGINE_CLOSE &&
                kind != CALL_SEQUENCE_FREE) {
                fail_contract("emission failure allowed lifecycle work after abort");
            }
        }
    }
    all_failures += state.contract_failures;
}

static void check_late_decode_failure(bool resident) {
    reset_fake_state(false);
    state.inject_decode_failure = true;
    state.decode_failure_repetition = LATE_DECODE_FAILURE_REPETITION;
    state.decode_failure_index = LATE_DECODE_FAILURE_INDEX;

    const int rc = resident ? invoke_bench_resident() : invoke_bench();
    if (rc == 0) fail_contract(resident
        ? "resident later-token decode failure unexpectedly returned success"
        : "streamed later-token decode failure unexpectedly returned success");
    if (!state.decode_failed) {
        fail_contract("later-token decode failure was not injected at its discriminating index");
    }

    const int expected_emit_count = 2 * 3 + 2;
    const int expected_decode_attempts =
        2 * QUALIFICATION_OUTPUT_TOKENS + LATE_DECODE_FAILURE_INDEX + 1;
    const int expected_decode_successes =
        2 * QUALIFICATION_OUTPUT_TOKENS + LATE_DECODE_FAILURE_INDEX;
    const int emit_count = resident ? state.resident_emit_count : state.emit_count;
    if (state.session_create_count != 3 || state.session_free_count != 3 ||
        state.request_begin_count != 3 || state.request_prompt_count != 3 ||
        state.prefill_start_count != 3 || state.sync_count != 3 ||
        state.prefill_complete_count != 3 || state.choose_count != expected_decode_attempts ||
        state.eval_count != expected_decode_attempts ||
        state.generated_count != expected_decode_successes ||
        state.visible_count != expected_decode_successes ||
        state.first_visible_count != 3 || state.barrier_count != 2 ||
        state.finish_count != 2 || state.checkpoint_count != expected_emit_count ||
        state.snapshot_count != expected_emit_count || state.engine_open_count != 1 ||
        state.engine_close_count != 1 ||
        state.sequence_free_count != (resident ? 0 : 1) ||
        state.resident_sequence_free_count != (resident ? 1 : 0) ||
        state.completed_metrics_count != 2 || emit_count != expected_emit_count ||
        state.emit_count != (resident ? 0 : expected_emit_count) ||
        state.resident_emit_count != (resident ? expected_emit_count : 0)) {
        fail_contract("later-token decode failure did not preserve the complete partial metric/timeline counts");
    }
    if (state.decode_attempt_count[0] != QUALIFICATION_OUTPUT_TOKENS ||
        state.decode_attempt_count[1] != QUALIFICATION_OUTPUT_TOKENS ||
        state.decode_attempt_count[2] != LATE_DECODE_FAILURE_INDEX + 1 ||
        state.decode_attempt_count[3] != 0 ||
        state.decode_success_count[0] != QUALIFICATION_OUTPUT_TOKENS ||
        state.decode_success_count[1] != QUALIFICATION_OUTPUT_TOKENS ||
        state.decode_success_count[2] != LATE_DECODE_FAILURE_INDEX ||
        state.decode_success_count[3] != 0) {
        fail_contract("later-token decode failure changed the successful-prefix counts");
    }

    const int failed_eval = nth_call(
        CALL_SESSION_EVAL_ATTRIBUTED,
        LATE_DECODE_FAILURE_REPETITION,
        LATE_DECODE_FAILURE_INDEX);
    if (failed_eval < 0 ||
        state.calls[failed_eval].value != (uint64_t)LATE_DECODE_FAILURE_INDEX ||
        state.calls[failed_eval].token !=
            fake_decode_token(LATE_DECODE_FAILURE_REPETITION, LATE_DECODE_FAILURE_INDEX)) {
        fail_contract("later-token failure did not identify the expected decode call");
    }
    const int failed_generated = nth_call(
        CALL_GENERATED, LATE_DECODE_FAILURE_REPETITION, LATE_DECODE_FAILURE_INDEX);
    const int failed_visible = nth_call(
        CALL_VISIBLE, LATE_DECODE_FAILURE_REPETITION, LATE_DECODE_FAILURE_INDEX);
    if (failed_generated >= 0 || failed_visible >= 0) {
        fail_contract("failed decode was accounted or exposed as a visible token");
    }
    const int failed_session_free = nth_call(
        CALL_SESSION_FREE, LATE_DECODE_FAILURE_REPETITION, 0);
    const int engine_close = first_call(CALL_ENGINE_CLOSE, -1);
    const int sequence_free = first_call(
        resident ? CALL_RESIDENT_SEQUENCE_FREE : CALL_SEQUENCE_FREE, -1);
    require_call_order(failed_eval, failed_session_free,
                       "failed decode must release its current session");
    require_call_order(failed_session_free, engine_close,
                       "failed decode session must be released before engine close");
    require_call_order(engine_close, sequence_free,
                       "failed decode engine must close before typed sequence cleanup");
    for (size_t i = failed_eval >= 0 ? (size_t)failed_eval + 1u : 0u;
         i < state.call_count; i++) {
        const enum call_kind kind = state.calls[i].kind;
        if (kind != CALL_SESSION_FREE && kind != CALL_ENGINE_CLOSE &&
            kind != (resident ? CALL_RESIDENT_SEQUENCE_FREE : CALL_SEQUENCE_FREE)) {
            fail_contract("later-token decode failure allowed work after the terminal failed eval");
        }
    }
    for (size_t i = 0; i < state.call_count; i++) {
        if ((state.calls[i].kind == CALL_EMIT ||
             state.calls[i].kind == CALL_RESIDENT_EMIT) &&
            state.calls[i].repetition >= LATE_DECODE_FAILURE_REPETITION) {
            if (state.calls[i].event == DS4_BENCH_QUALIFICATION_EVENT_REQUEST_COMPLETE) {
                fail_contract("later-token decode failure emitted a completion success");
            }
        }
    }
    all_failures += state.contract_failures;
}

static void check_resident_cross_mode(void) {
    reset_fake_state(false);
    state.reject_resident_cross_mode = true;
    const int resident_rc = invoke_bench_resident_streamed_fixture();
    if (resident_rc == 0) fail_contract("resident flag accepted a streamed fixture");
    if (first_call(CALL_RESIDENT_SEQUENCE_PARSE, -1) < 0 ||
        state.resident_sequence_free_count != 1 ||
        state.nvml_capture_count != 0 || state.engine_open_count != 0 ||
        state.session_create_count != 0 || state.resident_emit_count != 0) {
        fail_contract("resident cross-mode rejection did not stop before backend work");
    }
    all_failures += state.contract_failures;

    reset_fake_state(false);
    state.reject_streamed_cross_mode = true;
    const int streamed_rc = invoke_bench_streamed_resident_fixture();
    if (streamed_rc == 0) fail_contract("streamed flag accepted a resident fixture");
    if (first_call(CALL_SEQUENCE_PARSE, -1) < 0 ||
        state.sequence_free_count != 1 || state.nvml_capture_count != 0 ||
        state.engine_open_count != 0 || state.session_create_count != 0 ||
        state.emit_count != 0) {
        fail_contract("streamed cross-mode rejection did not stop before backend work");
    }
    all_failures += state.contract_failures;
}

static void check_resident_parser_failure(void) {
    reset_fake_state(false);
    state.inject_resident_parser_failure = true;
    const int rc = invoke_bench_resident();
    if (rc == 0) fail_contract("resident parser failure unexpectedly returned success");
    if (first_call(CALL_RESIDENT_SEQUENCE_PARSE, -1) < 0 ||
        state.resident_sequence_free_count != 1 ||
        state.sequence_free_count != 0 || state.nvml_capture_count != 0 ||
        state.engine_open_count != 0 || state.session_create_count != 0 ||
        state.resident_emit_count != 0) {
        fail_contract("resident parser failure performed backend work or missed typed cleanup");
    }
    const int parse = first_call(CALL_RESIDENT_SEQUENCE_PARSE, -1);
    const int sequence_free = first_call(CALL_RESIDENT_SEQUENCE_FREE, -1);
    require_call_order(parse, sequence_free,
                       "resident parser failure must precede resident sequence cleanup");
    for (size_t i = sequence_free >= 0 ? (size_t)sequence_free + 1u : 0u;
         i < state.call_count; i++) {
        fail_contract("resident parser failure allowed work after typed cleanup");
    }
    all_failures += state.contract_failures;
}

static void check_resident_emitter_failure(void) {
    reset_fake_state(true);
    const int rc = invoke_bench_resident();
    if (rc == 0) fail_contract("resident emitter failure unexpectedly returned success");
    if (!state.emitter_failed || state.resident_emit_count != 1 ||
        state.emit_count != 0 || state.session_create_count != 1 ||
        state.session_free_count != 1 || state.engine_open_count != 1 ||
        state.engine_close_count != 1 || state.resident_sequence_free_count != 1 ||
        state.sequence_free_count != 0) {
        fail_contract("resident emitter failure did not stop and clean the current typed run");
    }
    const int failed_emit = first_call(CALL_RESIDENT_EMIT, -1);
    const int sequence_free = first_call(CALL_RESIDENT_SEQUENCE_FREE, -1);
    require_call_order(failed_emit, sequence_free,
                       "resident emitter failure must precede resident sequence cleanup");
    for (size_t i = failed_emit >= 0 ? (size_t)failed_emit + 1u : 0u;
         i < state.call_count; i++) {
        const enum call_kind kind = state.calls[i].kind;
        if (kind != CALL_SESSION_FREE && kind != CALL_ENGINE_CLOSE &&
            kind != CALL_RESIDENT_SEQUENCE_FREE) {
            fail_contract("resident emitter failure allowed lifecycle work after abort");
        }
    }
    all_failures += state.contract_failures;
}

static void check_resident_midrun_failure(bool snapshot_failure) {
    reset_fake_state(false);
    state.inject_snapshot_failure = snapshot_failure;
    state.inject_checkpoint_failure = !snapshot_failure;
    const int rc = invoke_bench_resident();
    if (rc == 0) fail_contract("resident mid-run failure unexpectedly returned success");
    if (snapshot_failure && !state.snapshot_failed) {
        fail_contract("resident snapshot failure was not injected");
    }
    if (!snapshot_failure && !state.checkpoint_failed) {
        fail_contract("resident checkpoint failure was not injected");
    }
    if (state.resident_emit_count != 1 || state.emit_count != 0 ||
        state.session_create_count != 1 || state.session_free_count != 1 ||
        state.engine_open_count != 1 || state.engine_close_count != 1 ||
        state.resident_sequence_free_count != 1 || state.sequence_free_count != 0 ||
        state.barrier_count != 0 || state.finish_count != 0) {
        fail_contract("resident mid-run failure did not clean the current typed run");
    }
    if (snapshot_failure) {
        if (state.checkpoint_count != 2 || state.snapshot_count != 2) {
            fail_contract("resident snapshot failure did not stop at the second snapshot");
        }
    } else if (state.checkpoint_count != 2 || state.snapshot_count != 1) {
        fail_contract("resident checkpoint failure did not stop before the second snapshot");
    }
    const enum call_kind failed_kind = snapshot_failure
        ? CALL_RUNTIME_SNAPSHOT : CALL_EXTERNAL_CHECKPOINT;
    const int failed_call = nth_call(failed_kind, -1, 1);
    const int sequence_free = first_call(CALL_RESIDENT_SEQUENCE_FREE, -1);
    require_call_order(failed_call, sequence_free,
                       "resident mid-run failure must precede resident sequence cleanup");
    for (size_t i = failed_call >= 0 ? (size_t)failed_call + 1u : 0u;
         i < state.call_count; i++) {
        const enum call_kind kind = state.calls[i].kind;
        if (kind != CALL_SESSION_FREE && kind != CALL_ENGINE_CLOSE &&
            kind != CALL_RESIDENT_SEQUENCE_FREE) {
            fail_contract("resident mid-run failure allowed subsequent lifecycle work");
        }
    }
    all_failures += state.contract_failures;
}

static void check_resident_happy_path(void) {
    reset_fake_state(false);
    const int rc = invoke_bench_resident();
    if (rc != 0) {
        fprintf(stderr,
                "RED: resident qualification sequence has no typed lifecycle runner "
                "(valid fake resident sequence returned %d before engine/lifecycle calls)\n", rc);
    }
    if (state.resident_emit_count == 0 && rc != 0) {
        fail_contract("resident happy-path fake lifecycle emitted no milestones");
    }
    check_lifecycle_shape(true);
    all_failures += state.contract_failures;
}

#ifdef DS4_BENCH_LIFECYCLE_REAL_EMITTER
int main(int argc, char **argv) {
    /* Keep the established streamed stdout contract. The separate resident
     * target selects its typed contract explicitly, never from wire data. */
    const bool resident = argc == 2 && strcmp(argv[1], "--resident") == 0;
    if (argc != 1 && !resident) return 2;
    reset_fake_state(false);
    const int rc = resident ? invoke_bench_resident() : invoke_bench();
    if (state.contract_failures != 0 || rc != 0) return rc != 0 ? rc : 1;
    check_lifecycle_shape(resident);
    if (state.contract_failures != 0) {
        fprintf(stderr, "qualification real-emitter composition: %d failures\n",
                state.contract_failures);
        return 1;
    }
    return 0;
}
#else
int main(int argc, char **argv) {
    if (argc > 1) {
        if (strcmp(argv[1], "--probe-argv-rejection") != 0) return 2;
        reset_fake_state(false);
        const int rc = ds4_bench_test_cli_main(argc - 1, argv + 1);
        if (state.call_count != 0u) {
            fprintf(stderr, "argument rejection reached parser/backend\n");
            return 91;
        }
        return rc;
    }
    check_happy_path();
    check_native_stop_before_output(false);
    check_native_stop_termination(false, literal_eos_token, "EOS");
    check_native_stop_termination(false, literal_native_stop_token,
                                  "distinct model-native stop");
    check_late_decode_failure(false);
    check_fail_closed_cleanup();
    check_resident_cross_mode();
    check_resident_parser_failure();
    check_resident_emitter_failure();
    check_resident_midrun_failure(true);
    check_resident_midrun_failure(false);
    check_resident_happy_path();
    check_native_stop_before_output(true);
    check_native_stop_termination(true, literal_eos_token, "resident EOS");
    check_native_stop_termination(true, literal_native_stop_token,
                                  "resident distinct model-native stop");
    check_late_decode_failure(true);
    if (all_failures != 0) {
        fprintf(stderr, "qualification lifecycle fake backend: %d failures\n",
                all_failures);
        return 1;
    }
    puts("qualification lifecycle fake backend: PASS");
    return 0;
}
#endif
