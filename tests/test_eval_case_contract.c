/*
 * Host-only RED oracle for Task 19 stable ds4-eval case records.
 *
 * This executable includes the real ds4_eval.c translation unit with main
 * renamed.  The fake engine/session/request seams never dereference model or
 * accelerator state.  A valid invocation is expected to run four selected
 * cases and print only the frozen nine-field JSON records.  The current
 * production source deliberately fails this oracle at --case-id parsing;
 * that is the intended RED, not a compile/link fallback.
 */

#include "ds4.h"
#include "ds4_runtime.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#define ARRAY_LEN(value) (sizeof(value) / sizeof((value)[0]))
#define FAKE_EOS_TOKEN 777
#define FAKE_ANSWER_TOKEN_BASE 1000
#define MAX_CALLS 256
#define MAX_SESSIONS 4
#define MAX_CAPTURE 65536

static const int expected_case_indices[] = {0, 1, 2, 75};
static const char *const expected_case_ids[] = {
    "recNu3MXkvWUzHZr9",
    "001b51d76b4d422988f2c11f104a2c6c",
    "aime2025-01",
    "compsec-076",
};
static const char *const expected_answers[] = {"B", "C", "70", "17-20"};
static const char *const expected_request_ids[] = {
    "123e4567-e89b-12d3-a456-426614174001",
    "123e4567-e89b-12d3-a456-426614174002",
    "123e4567-e89b-12d3-a456-426614174003",
    "123e4567-e89b-12d3-a456-426614174004",
};
static const char *const context_request_ids[] = {
    "223e4567-e89b-12d3-a456-426614174001",
    "223e4567-e89b-12d3-a456-426614174002",
    "223e4567-e89b-12d3-a456-426614174003",
    "223e4567-e89b-12d3-a456-426614174004",
};
static const char expected_instance_id[] =
    "123e4567-e89b-12d3-a456-426614174099";
static const char context_instance_id[] =
    "223e4567-e89b-12d3-a456-426614174099";
static const char snapshot_instance_id[] =
    "323e4567-e89b-12d3-a456-426614174099";
static const char *const expected_snapshot_seq[] = {"1009", "2027", "4093", "8191"};

/* This is fixture data, not a call to any production digest helper.  The
 * evidence preimage is the exact UTF-8 byte stream beginning with
 * "ds4.eval.case.evidence/v1\n", then for each of the preceding eight fields
 * in order "FIELD_NAME=DECIMAL_UTF8_BYTE_LENGTH:EXACT_UTF8_VALUE\n".  Field
 * names and length digits are ASCII; the value length is UTF-8 bytes; the final
 * LF is included; evidence_sha256 and unrecorded model output are excluded.
 * The algorithm is independently implemented by tests/validate_eval_case_json.py. */
static const char expected_machine_stdout[] =
    "{\"schema\":\"ds4.eval.case/v1\",\"case_id\":\"recNu3MXkvWUzHZr9\",\"answer\":\"B\",\"grade\":\"passed\",\"terminal_status\":\"completed\",\"request_id\":\"123e4567-e89b-12d3-a456-426614174001\",\"instance_id\":\"123e4567-e89b-12d3-a456-426614174099\",\"snapshot_seq\":\"1009\",\"evidence_sha256\":\"efc5cc823d2c62525f7f2dce7ed0e89ed2a4831096d2774b9356659ffd62bb7f\"}\n"
    "{\"schema\":\"ds4.eval.case/v1\",\"case_id\":\"001b51d76b4d422988f2c11f104a2c6c\",\"answer\":\"C\",\"grade\":\"passed\",\"terminal_status\":\"completed\",\"request_id\":\"123e4567-e89b-12d3-a456-426614174002\",\"instance_id\":\"123e4567-e89b-12d3-a456-426614174099\",\"snapshot_seq\":\"2027\",\"evidence_sha256\":\"44e9fc74b8cccf730b77d5ce15fbe9ed291fed0e76796aad37f327bc677d2d24\"}\n"
    "{\"schema\":\"ds4.eval.case/v1\",\"case_id\":\"aime2025-01\",\"answer\":\"70\",\"grade\":\"passed\",\"terminal_status\":\"completed\",\"request_id\":\"123e4567-e89b-12d3-a456-426614174003\",\"instance_id\":\"123e4567-e89b-12d3-a456-426614174099\",\"snapshot_seq\":\"4093\",\"evidence_sha256\":\"4fc1d1f128b9b74a9bd29c0ad5784cc0e60c59bd0759eed70cd56b54ec194c44\"}\n"
    "{\"schema\":\"ds4.eval.case/v1\",\"case_id\":\"compsec-076\",\"answer\":\"17-20\",\"grade\":\"passed\",\"terminal_status\":\"completed\",\"request_id\":\"123e4567-e89b-12d3-a456-426614174004\",\"instance_id\":\"123e4567-e89b-12d3-a456-426614174099\",\"snapshot_seq\":\"8191\",\"evidence_sha256\":\"39ffdab240cb634f3c83238c81c98745163820f6ae9d2786f1042e6515d6adb6\"}\n";

typedef union {
    long double align;
    unsigned char bytes[256];
} fake_storage;

static fake_storage fake_engine_storage;
static fake_storage fake_session_storage[MAX_SESSIONS];
static int fake_prompt_tokens[16];
static ds4_engine *const fake_engine = (ds4_engine *)(void *)&fake_engine_storage;

enum call_kind {
    CALL_ENGINE_OPEN,
    CALL_ENGINE_CLOSE,
    CALL_SESSION_CREATE,
    CALL_SESSION_FREE,
    CALL_CASE_ENCODE,
    CALL_REQUEST_BEGIN,
    CALL_REQUEST_PROMPT,
    CALL_PREFILL_START,
    CALL_SYNC_ATTRIBUTED,
    CALL_PREFILL_COMPLETE,
    CALL_SAMPLE,
    CALL_EVAL_ATTRIBUTED,
    CALL_GENERATED,
    CALL_VISIBLE,
    CALL_FIRST_VISIBLE,
    CALL_REQUEST_BARRIER,
    CALL_REQUEST_FINISH,
    CALL_ORDINARY_SYNC,
    CALL_ORDINARY_EVAL,
    CALL_UNEXPECTED,
};

typedef struct {
    enum call_kind kind;
    int repetition;
    int case_index;
    ds4_session *session;
    ds4_runtime_request_context *request;
    int token;
} call_record;

typedef struct {
    call_record calls[MAX_CALLS];
    size_t call_count;
    int contract_failures;
    int engine_open_count;
    int engine_close_count;
    int session_create_count;
    int session_free_count;
    int prompt_encode_count;
    int actual_prompt_encode_count;
    int request_begin_count;
    int request_prompt_count;
    int prefill_start_count;
    int sync_count;
    int prefill_complete_count;
    int sample_count;
    int eval_count;
    int generated_count;
    int visible_count;
    int first_visible_count;
    int barrier_count;
    int finish_count;
    int ordinary_sync_count;
    int ordinary_eval_count;
    int active_case;
    int sample_phase;
    bool execution_started;
    bool legacy_mode;
    ds4_session *sessions[MAX_SESSIONS];
    int session_repetitions[MAX_SESSIONS];
    ds4_runtime_request_context *requests[MAX_SESSIONS];
    int request_repetitions[MAX_SESSIONS];
    int actual_case_indices[ARRAY_LEN(expected_case_indices)];
} fake_state;

static fake_state state;

static void fail_contract(const char *format, ...) {
    va_list ap;
    state.contract_failures++;
    fputs("case-contract fake: ", stderr);
    va_start(ap, format);
    vfprintf(stderr, format, ap);
    va_end(ap);
    fputc('\n', stderr);
}

static void record_call(enum call_kind kind, int repetition, int case_index,
                        ds4_session *session,
                        ds4_runtime_request_context *request,
                        int token) {
    if (state.call_count >= ARRAY_LEN(state.calls)) {
        fail_contract("call log capacity exhausted");
        return;
    }
    state.calls[state.call_count++] = (call_record){
        .kind = kind,
        .repetition = repetition,
        .case_index = case_index,
        .session = session,
        .request = request,
        .token = token,
    };
}

static int session_repetition(const ds4_session *session) {
    for (int index = 0; index < state.session_create_count && index < MAX_SESSIONS;
         index++) {
        if (state.sessions[index] == session) return state.session_repetitions[index];
    }
    return -1;
}

static int request_repetition(const ds4_runtime_request_context *request) {
    const int count = state.request_begin_count < MAX_SESSIONS ?
        state.request_begin_count : MAX_SESSIONS;
    /* Request contexts may be a single stack slot reused once per case. */
    for (int index = count - 1; index >= 0; index--) {
        if (state.requests[index] == request) return state.request_repetitions[index];
    }
    return -1;
}

static int nth_call(enum call_kind kind, int repetition, int ordinal) {
    for (size_t index = 0; index < state.call_count; index++) {
        const call_record *call = &state.calls[index];
        if (call->kind != kind ||
            (repetition >= 0 && call->repetition != repetition)) continue;
        if (ordinal-- == 0) return (int)index;
    }
    return -1;
}

static void require_order(int before, int after, const char *message) {
    if (before < 0 || after < 0 || before >= after) fail_contract("%s", message);
}

static void expected_request_id(int repetition, char *out, size_t capacity) {
    if (repetition < 0 || repetition >= (int)ARRAY_LEN(expected_request_ids)) {
        snprintf(out, capacity, "invalid-request");
        return;
    }
    snprintf(out, capacity, "%s", expected_request_ids[repetition]);
}

static int expected_case_for_repetition(int repetition) {
    return repetition >= 0 && repetition < (int)ARRAY_LEN(expected_case_indices) ?
        expected_case_indices[repetition] : -1;
}

static int answer_slot_for_token(int token) {
    const int slot = token - FAKE_ANSWER_TOKEN_BASE;
    return slot >= 0 && slot < (int)ARRAY_LEN(expected_answers) ? slot : -1;
}

static int case_index_from_prompt(const char *prompt) {
    if (!prompt) return -1;
    if (strstr(prompt, "An intelligent civilization in the Large Magellanic Cloud"))
        return 0;
    if (strstr(prompt, "In production, grass powder is often pressed into pellets"))
        return 1;
    if (strstr(prompt, "Find the sum of all integer bases $b>9"))
        return 2;
    /* This literal is in build_question_prompt()->question, not tc->title. */
    if (strstr(prompt, "static bool check_rpcsec_packet")) return 75;
    return -1;
}

/* ---- model-free fake engine/session/request operations ------------------ */

static int fake_engine_open(ds4_engine **out, const ds4_engine_options *options) {
    record_call(CALL_ENGINE_OPEN, -1, -1, NULL, NULL, 0);
    state.engine_open_count++;
    if (!out || !options || !options->model_path ||
        strcmp(options->model_path, "/literal/fake.gguf") != 0 ||
        options->backend != DS4_BACKEND_CPU || options->context_size != 32768) {
        fail_contract("engine open did not receive the host-only fake configuration");
        return 1;
    }
    if (state.engine_open_count != 1) fail_contract("engine opened more than once");
    *out = fake_engine;
    return 0;
}

static void fake_engine_close(ds4_engine *engine) {
    record_call(CALL_ENGINE_CLOSE, -1, -1, NULL, NULL, 0);
    state.engine_close_count++;
    if (engine != fake_engine || state.engine_close_count != 1)
        fail_contract("wrong engine close lifecycle");
}

static const char *fake_engine_model_name(ds4_engine *engine) {
    if (engine != fake_engine) fail_contract("model name used a non-fake engine");
    return "task19-fake-model";
}

static int fake_engine_vocab_size(ds4_engine *engine) {
    (void)engine;
    return 4096;
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
    if (engine != fake_engine || !out) {
        fail_contract("runtime snapshot used an invalid fake engine/output");
        return false;
    }
    memset(out, 0, sizeof(*out));
    snprintf(out->instance_id, sizeof(out->instance_id), "%s", snapshot_instance_id);
    /* Deliberately use a distinct valid instance and sequence.  A case record
     * must use the finalized request metrics, never this process snapshot as a
     * proxy. */
    out->snapshot_seq = UINT64_C(60001);
    out->state = DS4_RUNTIME_WIRE_STATE_READY;
    return true;
}

static ds4_runtime_status fake_engine_external_checkpoint(
        ds4_engine *engine,
        const ds4_gpu_nvml_inventory_snapshot *pre_child,
        const uint8_t expected_build_identity[DS4_RUNTIME_BUILD_IDENTITY_BYTES],
        ds4_engine_laguna_external_checkpoint_observation *out) {
    (void)pre_child;
    (void)expected_build_identity;
    (void)out;
    if (engine != fake_engine) fail_contract("external checkpoint used a non-fake engine");
    return DS4_RUNTIME_STATUS_OK;
}

static int fake_session_create(ds4_session **out, ds4_engine *engine, int context_size) {
    const int slot = state.session_create_count;
    record_call(CALL_SESSION_CREATE, -1, -1, NULL, NULL, 0);
    state.session_create_count++;
    if (!out || engine != fake_engine || context_size != 32768 || slot >= MAX_SESSIONS) {
        fail_contract("session create did not use the fake engine and context");
        return 1;
    }
    *out = (ds4_session *)(void *)&fake_session_storage[slot];
    state.sessions[slot] = *out;
    state.session_repetitions[slot] = slot;
    return 0;
}

static void fake_session_free(ds4_session *session) {
    record_call(CALL_SESSION_FREE, session_repetition(session), state.active_case,
                session, NULL, 0);
    state.session_free_count++;
    if (session_repetition(session) < 0)
        fail_contract("unknown fake session was freed");
}

static void fake_session_set_progress(ds4_session *session,
                                      ds4_session_progress_fn fn, void *ud) {
    (void)session;
    (void)fn;
    (void)ud;
}

static void fake_session_set_display_progress(ds4_session *session,
                                              ds4_session_progress_fn fn,
                                              void *ud) {
    (void)session;
    (void)fn;
    (void)ud;
}

static int fake_session_distributed_route_ready(ds4_session *session,
                                                char *err, size_t errlen) {
    (void)session;
    (void)err;
    (void)errlen;
    return 1;
}

static bool fake_session_is_distributed(ds4_session *session) {
    (void)session;
    return false;
}

static void fake_session_gpu_warmup(ds4_session *session) {
    (void)session;
}

static void fake_encode_chat_prompt(ds4_engine *engine, const char *system,
                                    const char *prompt, ds4_think_mode mode,
                                    ds4_tokens *out) {
    (void)system;
    (void)mode;
    state.prompt_encode_count++;
    const int prompt_case = case_index_from_prompt(prompt);
    if (engine != fake_engine || !out) {
        fail_contract("chat encoding did not use the fake engine/output");
        return;
    }
    if (state.session_create_count > 0 && !state.legacy_mode) {
        /* Context sizing occurs before session creation and legacy execution
         * prompts are deliberately outside this machine-case call log.  Once
         * a case-mode fake session exists, encoding is the execution prompt,
         * part of the request call log, and must follow the current request begin. */
        const int slot = state.actual_prompt_encode_count;
        const int repetition = state.request_begin_count - 1;
        record_call(CALL_CASE_ENCODE, repetition, prompt_case, NULL, NULL, 0);
        state.actual_prompt_encode_count++;
        if (state.actual_prompt_encode_count > (int)ARRAY_LEN(expected_case_indices))
            fail_contract("more than four selected prompt encodes");
        if (repetition < 0 || repetition >= (int)ARRAY_LEN(expected_case_indices)) {
            fail_contract("execution prompt encoded before request admission");
        } else if (prompt_case != expected_case_for_repetition(repetition)) {
            fail_contract("execution prompt did not match the admitted case order");
        }
        if (prompt_case < 0) {
            fail_contract("selected prompt was not one of the four canonical cases");
        } else if (slot < (int)ARRAY_LEN(expected_case_indices)) {
            state.actual_case_indices[slot] = prompt_case;
            state.active_case = prompt_case;
        }
        state.sample_phase = 0;
    }
    fake_prompt_tokens[0] = 1;
    fake_prompt_tokens[1] = 2;
    out->v = fake_prompt_tokens;
    out->len = 8;
    out->cap = 8;
}

static void fake_tokenize_text(ds4_engine *engine, const char *text, ds4_tokens *out) {
    (void)engine;
    (void)text;
    if (!out) return;
    fake_prompt_tokens[0] = FAKE_EOS_TOKEN;
    out->v = fake_prompt_tokens;
    out->len = 1;
    out->cap = 1;
}

static void fake_tokens_free(ds4_tokens *tokens) {
    (void)tokens;
}

static int fake_token_eos(ds4_engine *engine) {
    if (engine != fake_engine) fail_contract("EOS lookup used a non-fake engine");
    return FAKE_EOS_TOKEN;
}

static bool fake_token_is_stop(ds4_engine *engine, int token) {
    if (engine != fake_engine) fail_contract("stop check used a non-fake engine");
    return token == FAKE_EOS_TOKEN;
}

static char *fake_token_text(ds4_engine *engine, int token, size_t *length) {
    if (engine != fake_engine) fail_contract("token text used a non-fake engine");
    const int slot = answer_slot_for_token(token);
    const char *answer = slot >= 0 ? expected_answers[slot] : "";
    char text[64];
    const int written = snprintf(text, sizeof(text), "Answer: %s", answer);
    if (written < 0) return NULL;
    const size_t size = (size_t)written;
    char *copy = malloc(size + 1u);
    if (!copy) return NULL;
    memcpy(copy, text, size + 1u);
    if (length) *length = size;
    return copy;
}

static int fake_session_sync_attributed(
        ds4_session *session, const ds4_tokens *prompt,
        ds4_runtime_request_context *request, char *err, size_t errlen) {
    (void)err;
    (void)errlen;
    const int repetition = request_repetition(request);
    record_call(CALL_SYNC_ATTRIBUTED, repetition, state.active_case,
                session, request, 0);
    state.sync_count++;
    if (session_repetition(session) < 0 || !prompt || prompt->len != 8 ||
        repetition < 0 || state.active_case != expected_case_for_repetition(repetition)) {
        fail_contract("attributed prefill did not use the exact selected case/request");
        return 1;
    }
    return 0;
}

static int fake_session_eval_attributed(
        ds4_session *session, int token,
        ds4_runtime_request_context *request, char *err, size_t errlen) {
    (void)err;
    (void)errlen;
    const int repetition = request_repetition(request);
    record_call(CALL_EVAL_ATTRIBUTED, repetition, state.active_case,
                session, request, token);
    state.eval_count++;
    if (session_repetition(session) < 0 || repetition < 0 ||
        token != FAKE_ANSWER_TOKEN_BASE + repetition || token == FAKE_EOS_TOKEN) {
        fail_contract("attributed decode did not evaluate the canonical fake answer token");
        return 1;
    }
    return 0;
}

static int fake_session_sample(ds4_session *session, float temperature, int top_k,
                               float top_p, float min_p, uint64_t *rng) {
    (void)temperature;
    (void)top_k;
    (void)top_p;
    (void)min_p;
    (void)rng;
    const int repetition = state.request_begin_count > 0 ?
        state.request_begin_count - 1 : -1;
    record_call(CALL_SAMPLE, repetition, state.active_case, session, NULL,
                state.sample_phase == 0 ? FAKE_ANSWER_TOKEN_BASE + repetition : FAKE_EOS_TOKEN);
    state.sample_count++;
    if (session_repetition(session) < 0 || repetition < 0) {
        fail_contract("sample used an unknown session/request");
        return FAKE_EOS_TOKEN;
    }
    if (state.sample_phase == 0) {
        state.sample_phase = 1;
        return FAKE_ANSWER_TOKEN_BASE + repetition;
    }
    if (state.sample_phase == 1) {
        state.sample_phase = 2;
        return FAKE_EOS_TOKEN;
    }
    fail_contract("fake generation sampled after EOS");
    return FAKE_EOS_TOKEN;
}

static int fake_session_argmax(ds4_session *session) {
    record_call(CALL_UNEXPECTED, request_repetition(NULL), state.active_case,
                session, NULL, 0);
    fail_contract("machine case path used ordinary ds4_session_argmax");
    return FAKE_EOS_TOKEN;
}

static int fake_session_argmax_excluding(ds4_session *session, int excluded) {
    (void)excluded;
    record_call(CALL_UNEXPECTED, request_repetition(NULL), state.active_case,
                session, NULL, 0);
    fail_contract("machine case path used ordinary argmax instead of attributed sampling");
    return FAKE_EOS_TOKEN;
}

static int fake_session_top_logprobs(ds4_session *session, ds4_token_score *out, int k) {
    (void)session;
    (void)out;
    (void)k;
    return 0;
}

static int fake_session_sync(ds4_session *session, const ds4_tokens *prompt,
                             char *err, size_t errlen) {
    (void)prompt;
    (void)err;
    (void)errlen;
    state.ordinary_sync_count++;
    record_call(CALL_ORDINARY_SYNC, -1, state.active_case, session, NULL, 0);
    if (!state.legacy_mode)
        fail_contract("machine case path used ordinary ds4_session_sync");
    /* Controlled nonzero: stop legacy execution before any model work. */
    return 1;
}

static int fake_session_eval(ds4_session *session, int token, char *err, size_t errlen) {
    (void)err;
    (void)errlen;
    state.ordinary_eval_count++;
    record_call(CALL_ORDINARY_EVAL, -1, state.active_case, session, NULL, token);
    if (!state.legacy_mode)
        fail_contract("machine case path used ordinary ds4_session_eval");
    return 1;
}

static bool fake_request_begin(ds4_runtime_request_context *request,
                               uint64_t accepted_monotonic_ns) {
    const int repetition = state.request_begin_count;
    record_call(CALL_REQUEST_BEGIN, repetition, expected_case_for_repetition(repetition),
                NULL, request, 0);
    if (!request || repetition >= MAX_SESSIONS ||
        repetition >= (int)ARRAY_LEN(expected_case_indices)) {
        fail_contract("request begin received an invalid repetition");
        return false;
    }
    memset(request, 0, sizeof(*request));
    /* Deliberately differ from the finalized metrics and checked-in fixture.
     * The machine record must read identity only from fake_request_finish(). */
    snprintf(request->request_id, sizeof(request->request_id), "%s",
             context_request_ids[repetition]);
    snprintf(request->instance_id, sizeof(request->instance_id), "%s",
             context_instance_id);
    request->accepted_monotonic_ns = accepted_monotonic_ns ? accepted_monotonic_ns :
        UINT64_C(1000) + (uint64_t)repetition;
    request->initialized = true;
    state.requests[repetition] = request;
    state.request_repetitions[repetition] = repetition;
    state.request_begin_count++;
    state.execution_started = true;
    /* A prompt may be encoded before request admission.  Preserve that
     * selection so sync attribution validates the actual case, while still
     * providing the expected slot when admission precedes encoding. */
    if (state.actual_prompt_encode_count == repetition)
        state.active_case = expected_case_for_repetition(repetition);
    state.sample_phase = 0;
    return true;
}

static bool fake_request_set_prompt(ds4_runtime_request_context *request,
                                    uint64_t prompt_tokens) {
    const int repetition = request_repetition(request);
    record_call(CALL_REQUEST_PROMPT, repetition, state.active_case,
                NULL, request, 0);
    state.request_prompt_count++;
    if (repetition < 0 || prompt_tokens != 8u) {
        fail_contract("request prompt binding did not carry the fake prompt count");
        return false;
    }
    request->prompt_tokens = prompt_tokens;
    request->prompt_tokens_set = true;
    return true;
}

static bool fake_request_mark_prefill_started(
        ds4_runtime_request_context *request, uint64_t timestamp) {
    const int repetition = request_repetition(request);
    record_call(CALL_PREFILL_START, repetition, state.active_case,
                NULL, request, 0);
    state.prefill_start_count++;
    if (repetition < 0 || timestamp == 0u) {
        fail_contract("prefill start was not timestamped on a live request");
        return false;
    }
    request->prefill_started = true;
    request->prefill_started_monotonic_ns = timestamp;
    return true;
}

static bool fake_request_mark_prefill_complete(
        ds4_runtime_request_context *request, uint64_t timestamp) {
    const int repetition = request_repetition(request);
    record_call(CALL_PREFILL_COMPLETE, repetition, state.active_case,
                NULL, request, 0);
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
    const int repetition = request_repetition(request);
    record_call(CALL_GENERATED, repetition, state.active_case,
                NULL, request, 0);
    state.generated_count++;
    if (repetition < 0 || delta != 1u || !request->prefill_complete) {
        fail_contract("generated accounting did not follow attributed prefill");
        return false;
    }
    request->generated_tokens += delta;
    return true;
}

static bool fake_request_visible_common(ds4_runtime_request_context *request,
                                        uint64_t visible, uint64_t timestamp) {
    const int repetition = request_repetition(request);
    record_call(CALL_VISIBLE, repetition, state.active_case,
                NULL, request, 0);
    state.visible_count++;
    if (repetition < 0 || visible != 1u || timestamp == 0u ||
        request->generated_tokens != 1u) {
        fail_contract("visible accounting did not follow one generated token");
        return false;
    }
    request->visible_generated_tokens += visible;
    request->visible_decode_started = true;
    request->first_visible_decode_monotonic_ns = timestamp;
    request->last_visible_decode_monotonic_ns = timestamp;
    return true;
}

static bool fake_request_record_visible(ds4_runtime_request_context *request,
                                        uint64_t visible, uint64_t timestamp) {
    return fake_request_visible_common(request, visible, timestamp);
}

static bool fake_request_publish_visible(ds4_runtime_request_context *request,
                                         uint64_t visible,
                                         uint64_t first_timestamp,
                                         uint64_t last_timestamp) {
    if (first_timestamp == 0u) {
        fail_contract("published visible window lacked first timestamp");
        return false;
    }
    return fake_request_visible_common(request, visible, last_timestamp);
}

static bool fake_request_first_visible(ds4_runtime_request_context *request,
                                       uint64_t timestamp) {
    const int repetition = request_repetition(request);
    record_call(CALL_FIRST_VISIBLE, repetition, state.active_case,
                NULL, request, 0);
    state.first_visible_count++;
    if (repetition < 0 || timestamp == 0u ||
        request->visible_generated_tokens != 1u) {
        fail_contract("first-visible accounting did not follow visible output");
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

static bool fake_request_page_advice_complete(ds4_runtime_request_context *request,
                                              uint64_t timestamp) {
    (void)request;
    (void)timestamp;
    return true;
}

static int fake_session_request_barrier(ds4_session *session,
                                        ds4_runtime_request_context *request,
                                        char *err, size_t errlen) {
    (void)err;
    (void)errlen;
    const int repetition = request_repetition(request);
    record_call(CALL_REQUEST_BARRIER, repetition, state.active_case,
                session, request, 0);
    state.barrier_count++;
    if (session_repetition(session) < 0 || repetition < 0 ||
        !request->first_visible_emitted) {
        fail_contract("request barrier did not follow first-visible output");
        return 1;
    }
    return 0;
}

static bool fake_request_finish(ds4_runtime_request_context *request,
                                ds4_runtime_request_terminal_status status,
                                uint64_t timestamp,
                                ds4_runtime_request_metrics *metrics) {
    const int repetition = request_repetition(request);
    record_call(CALL_REQUEST_FINISH, repetition, state.active_case,
                NULL, request, 0);
    state.finish_count++;
    if (repetition < 0 || status != DS4_RUNTIME_REQUEST_COMPLETED ||
        timestamp == 0u || !metrics || !request->prompt_tokens_set ||
        !request->prefill_complete || !request->first_visible_emitted ||
        request->generated_tokens != 1u || request->visible_generated_tokens != 1u) {
        fail_contract("request finish did not finalize one complete attributed request");
        return false;
    }
    memset(metrics, 0, sizeof(*metrics));
    /* Only finalized request metrics carry the fixture identity.  Context and
     * engine-snapshot identities above are intentionally different oracles. */
    snprintf(metrics->request_id, sizeof(metrics->request_id), "%s",
             expected_request_ids[repetition]);
    snprintf(metrics->instance_id, sizeof(metrics->instance_id), "%s",
             expected_instance_id);
    metrics->snapshot_seq = (uint64_t)strtoull(expected_snapshot_seq[repetition], NULL, 10);
    metrics->prompt_tokens = 8u;
    metrics->generated_tokens = 1u;
    metrics->terminal_status = status;
    request->terminal = true;
    return true;
}

/* ---- include the real evaluator, substituting only model/runtime seams ---- */
#define main ds4_eval_test_cli_main
#define ds4_engine_open fake_engine_open
#define ds4_engine_close fake_engine_close
#define ds4_engine_model_name fake_engine_model_name
#define ds4_engine_vocab_size fake_engine_vocab_size
#define ds4_engine_prefill_chunk fake_engine_prefill_chunk
#define ds4_engine_routed_quant_bits fake_engine_routed_quant_bits
#define ds4_engine_runtime_snapshot fake_engine_runtime_snapshot
#define ds4_engine_laguna_external_checkpoint fake_engine_external_checkpoint
#define ds4_session_create fake_session_create
#define ds4_session_free fake_session_free
#define ds4_session_set_progress fake_session_set_progress
#define ds4_session_set_display_progress fake_session_set_display_progress
#define ds4_session_distributed_route_ready fake_session_distributed_route_ready
#define ds4_session_is_distributed fake_session_is_distributed
#define ds4_session_gpu_warmup fake_session_gpu_warmup
#define ds4_encode_chat_prompt fake_encode_chat_prompt
#define ds4_tokenize_text fake_tokenize_text
#define ds4_tokens_free fake_tokens_free
#define ds4_token_eos fake_token_eos
#define ds4_token_is_stop fake_token_is_stop
#define ds4_token_text fake_token_text
#define ds4_session_sync_attributed fake_session_sync_attributed
#define ds4_session_eval_attributed fake_session_eval_attributed
#define ds4_session_sample fake_session_sample
#define ds4_session_argmax fake_session_argmax
#define ds4_session_argmax_excluding fake_session_argmax_excluding
#define ds4_session_top_logprobs fake_session_top_logprobs
#define ds4_session_sync fake_session_sync
#define ds4_session_eval fake_session_eval
#define ds4_session_request_barrier fake_session_request_barrier
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
#define ds4_runtime_request_record_page_advice_complete fake_request_page_advice_complete
#define ds4_runtime_request_finish fake_request_finish
#include "../ds4_eval.c"
#undef main

#undef ds4_engine_open
#undef ds4_engine_close
#undef ds4_engine_model_name
#undef ds4_engine_vocab_size
#undef ds4_engine_prefill_chunk
#undef ds4_engine_routed_quant_bits
#undef ds4_engine_runtime_snapshot
#undef ds4_engine_laguna_external_checkpoint
#undef ds4_session_create
#undef ds4_session_free
#undef ds4_session_set_progress
#undef ds4_session_set_display_progress
#undef ds4_session_distributed_route_ready
#undef ds4_session_is_distributed
#undef ds4_session_gpu_warmup
#undef ds4_encode_chat_prompt
#undef ds4_tokenize_text
#undef ds4_tokens_free
#undef ds4_token_eos
#undef ds4_token_is_stop
#undef ds4_token_text
#undef ds4_session_sync_attributed
#undef ds4_session_eval_attributed
#undef ds4_session_sample
#undef ds4_session_argmax
#undef ds4_session_argmax_excluding
#undef ds4_session_top_logprobs
#undef ds4_session_sync
#undef ds4_session_eval
#undef ds4_session_request_barrier
#undef ds4_runtime_request_begin
#undef ds4_runtime_request_set_prompt_tokens
#undef ds4_runtime_request_mark_prefill_started
#undef ds4_runtime_request_mark_prefill_complete
#undef ds4_runtime_request_add_generated_tokens
#undef ds4_runtime_request_record_visible_decoded
#undef ds4_runtime_request_publish_visible_decode_window
#undef ds4_runtime_request_mark_first_visible_emitted
#undef ds4_runtime_request_add_counters
#undef ds4_runtime_request_observe_page_advice
#undef ds4_runtime_request_record_page_advice_complete
#undef ds4_runtime_request_finish

static void reset_fake_state(void) {
    memset(&state, 0, sizeof(state));
    state.active_case = -1;
    for (size_t index = 0; index < ARRAY_LEN(state.actual_case_indices); index++)
        state.actual_case_indices[index] = -1;
}

static void invalid_selector_atexit(void) {
    /* parse_options currently exits(2); keep the before-engine assertion alive
     * across that process exit instead of trusting only a returning main. */
    if (state.engine_open_count != 0) _exit(123);
}

static int check_machine_state(void) {
    int failures_before = state.contract_failures;
    if (state.engine_open_count != 1 || state.engine_close_count != 1)
        fail_contract("expected exactly one fake engine open and close");
    if (state.session_create_count < 1 ||
        state.session_free_count != state.session_create_count)
        fail_contract("expected every fake session to be freed exactly once");
    if (state.request_begin_count != 4 || state.request_prompt_count != 4 ||
        state.prefill_start_count != 4 || state.sync_count != 4 ||
        state.prefill_complete_count != 4 || state.sample_count != 8 ||
        state.eval_count != 4 || state.generated_count != 4 ||
        state.visible_count != 4 || state.first_visible_count != 4 ||
        state.barrier_count != 4 || state.finish_count != 4)
        fail_contract("attributed operation counts were not four requests/two samples each");
    if (state.ordinary_sync_count != 0 || state.ordinary_eval_count != 0)
        fail_contract("machine mode used ordinary sync/eval instead of attributed calls");
    if (state.actual_prompt_encode_count != 4)
        fail_contract("machine path did not encode exactly four selected prompts");
    for (size_t index = 0; index < ARRAY_LEN(expected_case_indices); index++) {
        if (state.actual_case_indices[index] != expected_case_indices[index]) {
            fail_contract("selected case %zu was global index %d, expected %d",
                          index, state.actual_case_indices[index],
                          expected_case_indices[index]);
        }
    }
    if (state.contract_failures != failures_before)
        return state.contract_failures;

    for (int repetition = 0; repetition < (int)ARRAY_LEN(expected_case_indices);
         repetition++) {
        const int request = nth_call(CALL_REQUEST_BEGIN, repetition, 0);
        const int encode = nth_call(CALL_CASE_ENCODE, repetition, 0);
        const int prompt = nth_call(CALL_REQUEST_PROMPT, repetition, 0);
        const int prefill_start = nth_call(CALL_PREFILL_START, repetition, 0);
        const int sync = nth_call(CALL_SYNC_ATTRIBUTED, repetition, 0);
        const int prefill_complete = nth_call(CALL_PREFILL_COMPLETE, repetition, 0);
        const int sample = nth_call(CALL_SAMPLE, repetition, 0);
        const int eval = nth_call(CALL_EVAL_ATTRIBUTED, repetition, 0);
        const int generated = nth_call(CALL_GENERATED, repetition, 0);
        const int visible = nth_call(CALL_VISIBLE, repetition, 0);
        const int first_visible = nth_call(CALL_FIRST_VISIBLE, repetition, 0);
        const int barrier = nth_call(CALL_REQUEST_BARRIER, repetition, 0);
        const int finish = nth_call(CALL_REQUEST_FINISH, repetition, 0);
        require_order(request, encode, "execution prompt encoding must follow request begin");
        require_order(encode, prompt, "request prompt binding must follow execution prompt encoding");
        require_order(prompt, prefill_start, "prefill start must follow request prompt binding");
        require_order(prefill_start, sync, "attributed sync must follow prefill start");
        require_order(sync, prefill_complete, "prefill complete must follow attributed sync");
        require_order(prefill_complete, sample, "sampling must follow prefill completion");
        require_order(sample, eval, "attributed eval must follow answer sampling");
        require_order(eval, generated, "generated accounting must follow attributed eval");
        require_order(generated, visible, "visible accounting must follow generated accounting");
        require_order(visible, first_visible, "first-visible accounting must follow visible accounting");
        require_order(first_visible, barrier, "request barrier must follow first-visible accounting");
        require_order(barrier, finish, "request finish must follow request barrier");
        const int eos = nth_call(CALL_SAMPLE, repetition, 1);
        require_order(sample, eos, "EOS sample must follow the answer sample");
        require_order(eos, barrier, "request barrier must follow generation termination");
    }
    for (size_t index = 0; index < state.call_count; index++) {
        if (state.calls[index].kind == CALL_UNEXPECTED ||
            state.calls[index].kind == CALL_ORDINARY_SYNC ||
            state.calls[index].kind == CALL_ORDINARY_EVAL)
            fail_contract("ordinary/unattributed operation was used in machine mode");
    }
    return state.contract_failures;
}

typedef struct {
    int wait_status;
    char stdout_bytes[MAX_CAPTURE];
    size_t stdout_len;
    char stderr_bytes[MAX_CAPTURE];
    size_t stderr_len;
} child_result;

static size_t read_fd(int fd, char *buffer, size_t capacity) {
    size_t length = 0;
    while (length < capacity) {
        ssize_t got = read(fd, buffer + length, capacity - length);
        if (got > 0) {
            length += (size_t)got;
            continue;
        }
        if (got == 0) break;
        if (errno == EINTR) continue;
        break;
    }
    return length;
}

static child_result run_cli_child(int argc, char **argv, bool valid_machine,
                                  bool legacy_smoke) {
    child_result result;
    memset(&result, 0, sizeof(result));
    int stdout_pipe[2];
    int stderr_pipe[2];
    if (pipe(stdout_pipe) != 0 || pipe(stderr_pipe) != 0) {
        perror("pipe");
        result.wait_status = -1;
        return result;
    }
    fflush(NULL);
    const pid_t pid = fork();
    if (pid == 0) {
        close(stdout_pipe[0]);
        close(stderr_pipe[0]);
        if (dup2(stdout_pipe[1], STDOUT_FILENO) < 0 ||
            dup2(stderr_pipe[1], STDERR_FILENO) < 0) _exit(190);
        close(stdout_pipe[1]);
        close(stderr_pipe[1]);
        reset_fake_state();
        state.legacy_mode = legacy_smoke;
        if (!valid_machine && !legacy_smoke) atexit(invalid_selector_atexit);
        const int rc = ds4_eval_test_cli_main(argc, argv);
        fflush(NULL);
        if (valid_machine) {
            if (rc != 0) {
                fprintf(stderr, "machine invocation returned %d\n", rc);
                _exit(120);
            }
            const int failures = check_machine_state();
            if (failures != 0) _exit(121);
            if (fflush(stdout) != 0) _exit(122);
            _exit(0);
        }
        if (legacy_smoke) {
            /* The fake engine/session are intentional here: this checks that
             * ordinary parsing still reaches the legacy sync boundary, while
             * the controlled ordinary-sync error prevents model work. */
            if (state.engine_open_count != 1 || state.ordinary_sync_count != 1 ||
                state.ordinary_eval_count != 0) {
                fprintf(stderr,
                        "legacy invocation did not reach exactly one ordinary sync boundary\n");
                _exit(123);
            }
            if (state.request_begin_count != 0 || state.request_prompt_count != 0 ||
                state.prefill_start_count != 0 || state.sync_count != 0 ||
                state.prefill_complete_count != 0 || state.sample_count != 0 ||
                state.eval_count != 0 || state.generated_count != 0 ||
                state.visible_count != 0 || state.first_visible_count != 0 ||
                state.barrier_count != 0 || state.finish_count != 0) {
                fprintf(stderr,
                        "legacy invocation used attributed request/model accounting\n");
                _exit(123);
            }
            _exit(rc == 0 ? 124 : rc);
        }
        /* Invalid selectors must never reach fake engine access even if the
         * production parser returns instead of calling exit(2). */
        if (state.engine_open_count != 0) {
            fprintf(stderr, "invalid selector reached fake engine\n");
            _exit(123);
        }
        _exit(rc == 0 ? 124 : rc);
    }
    if (pid < 0) {
        perror("fork");
        close(stdout_pipe[0]);
        close(stdout_pipe[1]);
        close(stderr_pipe[0]);
        close(stderr_pipe[1]);
        result.wait_status = -1;
        return result;
    }
    close(stdout_pipe[1]);
    close(stderr_pipe[1]);
    if (waitpid(pid, &result.wait_status, 0) < 0) result.wait_status = -1;
    result.stdout_len = read_fd(stdout_pipe[0], result.stdout_bytes, MAX_CAPTURE);
    result.stderr_len = read_fd(stderr_pipe[0], result.stderr_bytes, MAX_CAPTURE);
    close(stdout_pipe[0]);
    close(stderr_pipe[0]);
    return result;
}

static int child_exit_code(const child_result *result) {
    if (!result || result->wait_status < 0 || !WIFEXITED(result->wait_status)) return -1;
    return WEXITSTATUS(result->wait_status);
}

static void report_child_stderr(const child_result *result, const char *label) {
    if (!result || result->stderr_len == 0) return;
    fprintf(stderr, "%s stderr:\n%.*s", label, (int)result->stderr_len,
            result->stderr_bytes);
    if (result->stderr_bytes[result->stderr_len - 1] != '\n') fputc('\n', stderr);
}

static int invoke_valid_machine(void) {
    char *argv[] = {
        (char *)"ds4-eval",
        (char *)"--model", (char *)"/literal/fake.gguf",
        (char *)"--backend", (char *)"cpu",
        (char *)"--ctx", (char *)"32768",
        (char *)"--tokens", (char *)"8",
        (char *)"--nothink",
        (char *)"--case-id", (char *)"recNu3MXkvWUzHZr9",
        (char *)"--case-id", (char *)"001b51d76b4d422988f2c11f104a2c6c",
        (char *)"--case-id", (char *)"aime2025-01",
        (char *)"--case-id", (char *)"compsec-076",
        NULL,
    };
    child_result result = run_cli_child((int)ARRAY_LEN(argv) - 1, argv, true, false);
    const int rc = child_exit_code(&result);
    if (rc != 0) {
        fprintf(stderr,
                "RED: stable --case-id machine run was not accepted by production "
                "(child exit=%d; expected four fake records)\n", rc);
        report_child_stderr(&result, "valid --case-id");
        return 1;
    }
    if (result.stdout_len != strlen(expected_machine_stdout) ||
        memcmp(result.stdout_bytes, expected_machine_stdout, result.stdout_len) != 0) {
        fprintf(stderr,
                "case-contract: machine stdout was not exactly the four frozen JSON records\n");
        return 1;
    }
    return 0;
}

static int invoke_invalid(const char *label, char **argv, int argc) {
    child_result result = run_cli_child(argc, argv, false, false);
    const int rc = child_exit_code(&result);
    if (rc != 2) {
        fprintf(stderr, "case-contract: %s returned %d, expected selector rejection 2\n",
                label, rc);
        report_child_stderr(&result, label);
        return 1;
    }
    return 0;
}

static int check_invalid_selectors(void) {
    int failures = 0;
    char *unknown[] = {
        (char *)"ds4-eval", (char *)"--case-id", (char *)expected_case_ids[0],
        (char *)"--case-id", (char *)"not-a-case", NULL,
    };
    char *duplicate[] = {
        (char *)"ds4-eval", (char *)"--case-id", (char *)expected_case_ids[0],
        (char *)"--case-id", (char *)expected_case_ids[0],
        (char *)"--case-id", (char *)expected_case_ids[2],
        (char *)"--case-id", (char *)expected_case_ids[3], NULL,
    };
    char *incomplete[] = {
        (char *)"ds4-eval", (char *)"--case-id", (char *)expected_case_ids[0],
        (char *)"--case-id", (char *)expected_case_ids[1],
        (char *)"--case-id", (char *)expected_case_ids[2], NULL,
    };
    char *reordered[] = {
        (char *)"ds4-eval", (char *)"--case-id", (char *)expected_case_ids[1],
        (char *)"--case-id", (char *)expected_case_ids[0],
        (char *)"--case-id", (char *)expected_case_ids[2],
        (char *)"--case-id", (char *)expected_case_ids[3], NULL,
    };
    char *with_sequence[] = {
        (char *)"ds4-eval", (char *)"--case-id", (char *)expected_case_ids[0],
        (char *)"--case-id", (char *)expected_case_ids[1],
        (char *)"--case-id", (char *)expected_case_ids[2],
        (char *)"--case-id", (char *)expected_case_ids[3],
        (char *)"--case-sequence", (char *)"1,2,3,4", NULL,
    };
    char *with_questions[] = {
        (char *)"ds4-eval", (char *)"--case-id", (char *)expected_case_ids[0],
        (char *)"--case-id", (char *)expected_case_ids[1],
        (char *)"--case-id", (char *)expected_case_ids[2],
        (char *)"--case-id", (char *)expected_case_ids[3],
        (char *)"--questions", (char *)"4", NULL,
    };
    struct invalid_case {
        const char *label;
        char **argv;
        int argc;
    } cases[] = {
        {"unknown case ID", unknown, (int)ARRAY_LEN(unknown) - 1},
        {"duplicate case ID", duplicate, (int)ARRAY_LEN(duplicate) - 1},
        {"incomplete case selection", incomplete, (int)ARRAY_LEN(incomplete) - 1},
        {"reordered case selection", reordered, (int)ARRAY_LEN(reordered) - 1},
        {"case ID plus case sequence", with_sequence, (int)ARRAY_LEN(with_sequence) - 1},
        {"case ID plus questions", with_questions, (int)ARRAY_LEN(with_questions) - 1},
    };
    for (size_t index = 0; index < ARRAY_LEN(cases); index++) {
        if (invoke_invalid(cases[index].label, cases[index].argv, cases[index].argc) != 0)
            failures++;
    }
    return failures;
}

static int invoke_machine_mode_conflict(const char *label, char **argv, int argc) {
    child_result result = run_cli_child(argc, argv, false, false);
    const int rc = child_exit_code(&result);
    int failures = 0;
    if (rc != 2) {
        fprintf(stderr, "case-contract: %s returned %d, expected selector rejection 2\n",
                label, rc);
        report_child_stderr(&result, label);
        failures++;
    }
    /* Machine case records and legacy progress both use stdout.  A rejected
     * mode combination must not let either output channel leak before the
     * selector diagnostic, especially when --trace names /dev/stdout. */
    if (result.stdout_len != 0) {
        fprintf(stderr,
                "case-contract: %s wrote %zu stdout bytes before selector rejection\n",
                label, result.stdout_len);
        failures++;
    }
    return failures;
}

static int check_machine_mode_conflicts(void) {
    int failures = 0;
    char *with_self_test[] = {
        (char *)"ds4-eval",
        (char *)"--model", (char *)"/literal/fake.gguf",
        (char *)"--backend", (char *)"cpu",
        (char *)"--ctx", (char *)"32768",
        (char *)"--tokens", (char *)"8",
        (char *)"--nothink",
        (char *)"--case-id", (char *)"recNu3MXkvWUzHZr9",
        (char *)"--case-id", (char *)"001b51d76b4d422988f2c11f104a2c6c",
        (char *)"--case-id", (char *)"aime2025-01",
        (char *)"--case-id", (char *)"compsec-076",
        (char *)"--self-test-extractors", NULL,
    };
    char *with_regrade_trace[] = {
        (char *)"ds4-eval",
        (char *)"--model", (char *)"/literal/fake.gguf",
        (char *)"--backend", (char *)"cpu",
        (char *)"--ctx", (char *)"32768",
        (char *)"--tokens", (char *)"8",
        (char *)"--nothink",
        (char *)"--case-id", (char *)"recNu3MXkvWUzHZr9",
        (char *)"--case-id", (char *)"001b51d76b4d422988f2c11f104a2c6c",
        (char *)"--case-id", (char *)"aime2025-01",
        (char *)"--case-id", (char *)"compsec-076",
        /* /dev/null is a valid empty input: the early regrade path emits its
         * zero-case summary and returns 1, rather than failing to open a file
         * with the same exit code this selector conflict must use. */
        (char *)"--regrade-trace", (char *)"/dev/null", NULL,
    };
    char *with_trace_stdout[] = {
        (char *)"ds4-eval",
        (char *)"--model", (char *)"/literal/fake.gguf",
        (char *)"--backend", (char *)"cpu",
        (char *)"--ctx", (char *)"32768",
        (char *)"--tokens", (char *)"8",
        (char *)"--nothink",
        (char *)"--case-id", (char *)"recNu3MXkvWUzHZr9",
        (char *)"--case-id", (char *)"001b51d76b4d422988f2c11f104a2c6c",
        (char *)"--case-id", (char *)"aime2025-01",
        (char *)"--case-id", (char *)"compsec-076",
        (char *)"--trace", (char *)"/dev/stdout", NULL,
    };
    failures += invoke_machine_mode_conflict(
        "case IDs plus extractor self-test", with_self_test,
        (int)ARRAY_LEN(with_self_test) - 1);
    failures += invoke_machine_mode_conflict(
        "case IDs plus trace regrade", with_regrade_trace,
        (int)ARRAY_LEN(with_regrade_trace) - 1);
    failures += invoke_machine_mode_conflict(
        "case IDs plus stdout trace", with_trace_stdout,
        (int)ARRAY_LEN(with_trace_stdout) - 1);
    return failures;
}

static int check_legacy_smoke(void) {
    /* This invokes real argv parsing without a case selector.  The missing
     * model is an intentional post-parse boundary, proving legacy CLI reachability
     * without allocating a model or accelerator. */
    char *argv[] = {
        (char *)"ds4-eval", (char *)"--model", (char *)"/literal/fake.gguf",
        (char *)"--backend", (char *)"cpu", (char *)"--questions", (char *)"1",
        (char *)"--ctx", (char *)"32768", (char *)"--nothink", NULL,
    };
    child_result result = run_cli_child((int)ARRAY_LEN(argv) - 1, argv, false, true);
    const int rc = child_exit_code(&result);
    if (rc != 1 && rc != 2) {
        fprintf(stderr, "case-contract: legacy no-case-id smoke returned %d\n", rc);
        report_child_stderr(&result, "legacy smoke");
        return 1;
    }
    return 0;
}

int main(void) {
    int failures = 0;
    failures += check_invalid_selectors();
    failures += check_legacy_smoke();
    failures += invoke_valid_machine();
    failures += check_machine_mode_conflicts();
    if (failures != 0) {
        fprintf(stderr, "case selector contract RED assertions failed: %d\n", failures);
        return 1;
    }
    puts("case selector fake backend: PASS");
    return 0;
}
