/*
 * Host-only RED test for the Task 19 benchmark qualification lifecycle.
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

static const uint32_t literal_prompt_tokens = 512u;
/* Deliberately nonzero: a hard-coded common EOS value must not pass the fake. */
static const int literal_eos_token = 17;
static const uint64_t literal_cache_bytes = UINT64_C(8589934592);
static const char literal_manifest_sha256[] =
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
static const char literal_sequence_sha256[] =
    "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
static const char literal_input_sha256[] =
    "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
static const char literal_instance_id[] =
    "123e4567-e89b-12d3-a456-426614174099";

static ds4_engine *const fake_engine =
    (ds4_engine *)(void *)&fake_engine_storage;

enum call_kind {
    CALL_SEQUENCE_PARSE,
    CALL_SEQUENCE_FREE,
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
    CALL_UNEXPECTED,
};

typedef struct {
    enum call_kind kind;
    int repetition;
    ds4_session *session;
    ds4_runtime_request_context *request;
    const ds4_bench_sequence *sequence;
    ds4_bench_qualification_event event;
    const ds4_runtime_request_metrics *metrics;
    uint64_t value;
    int token;
} call_record;

typedef struct {
    call_record calls[256];
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
    int sequence_free_count;
    bool inject_emitter_failure;
    bool emitter_failed;
    const ds4_bench_sequence *trusted_sequence;
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
    if (!path || strcmp(path, "/literal/sequence.txt") != 0) {
        fail_contract("trusted parser did not receive the literal sequence path");
        return false;
    }
    if (!out || !expected_manifest_sha256 || !expected_sequence_sha256 ||
        strcmp(expected_manifest_sha256, literal_manifest_sha256) != 0 ||
        strcmp(expected_sequence_sha256, literal_sequence_sha256) != 0) {
        fail_contract("trusted parser did not receive the literal authenticated digests");
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

static int fake_engine_open(ds4_engine **out, const ds4_engine_options *options) {
    record_call(CALL_ENGINE_OPEN, -1);
    state.engine_open_count++;
    if (state.nvml_capture_count != 1 || state.captured_pre_child == NULL) {
        fail_contract("engine open was not preceded by exactly one pre-child NVML capture");
    }
    if (!out || !pinned_engine_options(options)) {
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
    memset(out, 0, sizeof(*out));
    snprintf(out->instance_id, sizeof(out->instance_id), "%s",
             literal_instance_id);
    out->snapshot_seq = UINT64_C(100) + (uint64_t)state.snapshot_count;
    out->state = DS4_RUNTIME_WIRE_STATE_READY;
    out->configured_context_tokens = 32768u;
    out->configured_prefill_chunk_tokens = 4096u;
    out->configured_session_slots = 1u;
    out->configured_ssd_streaming = true;
    out->configured_ssd_streaming_cache_bytes = literal_cache_bytes;
    out->effective_context_tokens = 32768u;
    out->effective_prefill_chunk_tokens = 4096u;
    out->effective_session_slots = 1u;
    out->expert_cache_limit_bytes = literal_cache_bytes;
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
    return text != NULL &&
           literal_sequence_input[sizeof(literal_input) - 1u] == 0x7fu &&
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

static int fake_session_argmax_excluding(ds4_session *session, int excluded) {
    const int repetition = session_rep(session);
    record_pointer_call(CALL_TOKEN_CHOOSE, repetition, session, NULL);
    state.choose_count++;
    if (repetition < 0 || repetition >= 4 || session != state.sessions[repetition] ||
        excluded != literal_eos_token) {
        fail_contract("non-EOS chooser did not exclude the exact fake engine EOS token");
    }
    return 42;
}

static int fake_session_argmax(ds4_session *session) {
    record_pointer_call(CALL_UNEXPECTED, session_rep(session), session, NULL);
    fail_contract("qualification lifecycle used ds4_session_argmax instead of ds4_session_argmax_excluding");
    return 42;
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
    (void)error;
    (void)error_size;
    const int repetition = session_rep(session);
    record_pointer_call(CALL_SESSION_EVAL_ATTRIBUTED, repetition, session, request);
    state.eval_count++;
    if (token == fake_token_eos(fake_engine) || token != 42 ||
        repetition < 0 || request_rep(request) != repetition) {
        fail_contract("attributed decode did not evaluate exactly one non-EOS token");
        return 1;
    }
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
    if (delta != 1u || repetition < 0) {
        fail_contract("request accounting did not add exactly one generated token");
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
    if (delta != 1u || timestamp == 0u || repetition < 0) {
        fail_contract("request accounting did not add exactly one visible token");
        return false;
    }
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
    if (repetition < 0 || timestamp == 0u || !request->visible_decode_started) {
        fail_contract("first-visible emission was not bound after visible accounting");
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
        timestamp == 0u || !metrics || !request->prompt_tokens_set ||
        request->generated_tokens != 1u || request->visible_generated_tokens != 1u) {
        fail_contract("request finish did not complete one accounted request");
        return false;
    }
    memset(metrics, 0, sizeof(*metrics));
    snprintf(metrics->request_id, sizeof(metrics->request_id), "%s",
             request->request_id);
    snprintf(metrics->instance_id, sizeof(metrics->instance_id), "%s",
             request->instance_id);
    metrics->prompt_tokens = literal_prompt_tokens;
    metrics->generated_tokens = 1u;
    metrics->snapshot_seq = UINT64_C(100) + (uint64_t)state.snapshot_count;
    metrics->terminal_status = status;
    request->terminal = true;
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
    const int event_index = state.emit_count - 1;
    const ds4_bench_qualification_event expected_event =
        (ds4_bench_qualification_event)(event_index % 3);
    const int expected_repetition = event_index / 3;
    char expected_request[DS4_RUNTIME_INSTANCE_ID_CAPACITY];
    expected_request_id(expected_repetition, expected_request,
                        sizeof(expected_request));
    ds4_session *current_session =
        expected_repetition >= 0 && expected_repetition < state.session_create_count ?
        state.sessions[expected_repetition] : NULL;
    if (!record || state.emit_count > 12 ||
        record->monotonic_ns == 0u ||
        record->monotonic_ns <= state.last_emitted_monotonic_ns ||
        (current_session != NULL &&
         record->session_payload_bytes != fake_session_payload_bytes(current_session)) ||
        current_session == NULL ||
        record->sequence != state.trusted_sequence ||
        record->event != expected_event ||
        !check_repetition((int)record->repetition_index, expected_repetition,
                          "emitter") ||
        !record->request_id || strcmp(record->request_id, expected_request) != 0 ||
        !record->runtime_snapshot ||
        strcmp(record->runtime_snapshot->instance_id, literal_instance_id) != 0 ||
        record->runtime_snapshot->snapshot_seq !=
            UINT64_C(100) + (uint64_t)state.emit_count) {
        fail_contract("emitter did not receive the exact ordered lifecycle record");
    }
    if (record && record->monotonic_ns > state.last_emitted_monotonic_ns) {
        state.last_emitted_monotonic_ns = record->monotonic_ns;
    }
    if (record && record->event != DS4_BENCH_QUALIFICATION_EVENT_REQUEST_COMPLETE &&
        record->request_metrics != NULL) {
        fail_contract("accepted/first-token record unexpectedly carried metrics");
    }
    if (record && record->event == DS4_BENCH_QUALIFICATION_EVENT_REQUEST_COMPLETE) {
        const ds4_runtime_request_metrics *metrics = record->request_metrics;
        if (!metrics || metrics->terminal_status != DS4_RUNTIME_REQUEST_COMPLETED ||
            strcmp(metrics->request_id, expected_request) != 0 ||
            strcmp(metrics->instance_id, literal_instance_id) != 0 ||
            metrics->snapshot_seq == UINT64_MAX ||
            metrics->snapshot_seq + 1u != record->runtime_snapshot->snapshot_seq ||
            metrics->prompt_tokens != literal_prompt_tokens ||
            metrics->generated_tokens != 1u) {
            fail_contract("completion metrics did not match the enclosing runtime snapshot");
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

/* Macro substitution is deliberately limited to the lifecycle-facing APIs. */
#define main ds4_bench_test_cli_main
/* Keep the CUDA-only qualification branch testable in this host-only fake
 * backend while normal DS4_NO_GPU builds remain fail-closed. */
#define DS4_BENCH_QUALIFICATION_TEST_BACKEND 1
#define ds4_bench_sequence_parse_file_trusted fake_sequence_parse_file_trusted
#define ds4_bench_sequence_free fake_sequence_free
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
#define ds4_bench_qualification_emit_record fake_emit_record
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

static void check_lifecycle_shape(void) {
    int event_count = 0;
    for (size_t i = 0; i < state.call_count; i++) {
        if (state.calls[i].kind != CALL_EMIT) continue;
        if (event_count >= 12) {
            fail_contract("more than twelve lifecycle emissions occurred");
            break;
        }
        if (state.calls[i].event !=
                (ds4_bench_qualification_event)(event_count % 3) ||
            state.calls[i].repetition != event_count / 3) {
            fail_contract("milestones were not emitted as accepted, first-token, complete per repetition");
        }
        event_count++;
    }
    if (event_count != 12) fail_contract("expected exactly twelve lifecycle emissions");
    if (state.engine_open_count != 1 || state.engine_close_count != 1) {
        fail_contract("expected exactly one engine open and close");
    }
    if (state.sequence_free_count != 1) {
        fail_contract("expected exactly one trusted sequence cleanup");
    }
    if (state.nvml_capture_count != 1 || state.tokenize_rendered_count != 1) {
        fail_contract("expected one pre-engine NVML capture and one rendered-input tokenization");
    }
    const int engine_open = first_call(CALL_ENGINE_OPEN, -1);
    const int nvml_capture = first_call(CALL_NVML_CAPTURE, -1);
    const int rendered_tokenize = first_call(CALL_TOKENIZE_RENDERED, -1);
    const int first_session = first_call(CALL_SESSION_CREATE, -1);
    const int engine_close = first_call(CALL_ENGINE_CLOSE, -1);
    const int last_free = nth_call(CALL_SESSION_FREE, -1, 3);
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
    if (state.session_create_count != 4 || state.session_free_count != 4) {
        fail_contract("expected four fresh sessions and four frees");
    }
    if (state.request_begin_count != 4 || state.request_prompt_count != 4) {
        fail_contract("expected one request context begin and prompt binding per repetition");
    }
    if (state.prefill_start_count != 4 || state.sync_count != 4 ||
        state.prefill_complete_count != 4 || state.choose_count != 4 ||
        state.eval_count != 4 || state.generated_count != 4 ||
        state.visible_count != 4 || state.first_visible_count != 4 ||
        state.barrier_count != 4 || state.finish_count != 4 ||
        state.checkpoint_count != 12 || state.snapshot_count != 12) {
        fail_contract("lifecycle operation counts did not match four one-token repetitions");
    }
    for (int repetition = 0; repetition < 4; repetition++) {
        const int create = nth_call(CALL_SESSION_CREATE, repetition, 0);
        const int request = nth_call(CALL_REQUEST_BEGIN, repetition, 0);
        const int prompt = nth_call(CALL_REQUEST_PROMPT, repetition, 0);
        const int accepted_checkpoint = nth_call(CALL_EXTERNAL_CHECKPOINT, -1, repetition * 3);
        const int accepted_snapshot = nth_call(CALL_RUNTIME_SNAPSHOT, -1, repetition * 3);
        const int accepted = nth_call(CALL_EMIT, repetition, 0);
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
        const int first = nth_call(CALL_EMIT, repetition, 1);
        const int barrier = nth_call(CALL_REQUEST_BARRIER, repetition, 0);
        const int finish = nth_call(CALL_REQUEST_FINISH, repetition, 0);
        const int complete_checkpoint = nth_call(CALL_EXTERNAL_CHECKPOINT, -1, repetition * 3 + 2);
        const int complete_snapshot = nth_call(CALL_RUNTIME_SNAPSHOT, -1, repetition * 3 + 2);
        const int complete = nth_call(CALL_EMIT, repetition, 2);
        const int free = nth_call(CALL_SESSION_FREE, repetition, 0);
        const int sequence_free = nth_call(CALL_SEQUENCE_FREE, -1, 0);
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
                               "final completion event must precede trusted sequence cleanup");
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
    check_lifecycle_shape();
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

int main(void) {
    check_happy_path();
    check_fail_closed_cleanup();
    if (all_failures != 0) {
        fprintf(stderr, "qualification lifecycle fake RED assertions failed: %d\n",
                all_failures);
        return 1;
    }
    puts("qualification lifecycle fake backend: PASS");
    return 0;
}
