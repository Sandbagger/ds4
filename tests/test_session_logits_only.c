/* Model-independent contract for exact-context logits-only terminal sessions.
 *
 * The test hooks construct deterministic CPU sessions.  They do not admit
 * exact-context execution in production; they only let this policy test prove
 * eligibility, terminal read access, rejection-before-mutation, and atomic
 * batch rejection without loading a model.
 */

#include "ds4.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TEST_CTX 16
#define TEST_VOCAB_SIZE 129280
#define TEST_FIRST_CHECKPOINT_TOKEN 7
#define ARRAY_LEN(a) (sizeof(a) / sizeof((a)[0]))

/* Internal decode primitive deliberately covered by the terminal policy. */
int ds4_session_eval_argmax(ds4_session *s, int token,
                            char *err, size_t errlen);

static int failures;

#define CHECK(condition, ...) do {                                           \
    if (!(condition)) {                                                      \
        fprintf(stderr, "FAIL: ");                                          \
        fprintf(stderr, __VA_ARGS__);                                        \
        fputc('\n', stderr);                                                 \
        failures++;                                                          \
    }                                                                        \
} while (0)

static ds4_test_session_state session_state(ds4_session *s,
                                             const char *label) {
    ds4_test_session_state state;
    memset(&state, 0, sizeof(state));
    CHECK(ds4_test_session_state_get(s, &state) == 0,
          "%s state capture", label);
    return state;
}

static bool states_equal(const ds4_test_session_state *a,
                         const ds4_test_session_state *b) {
    return a->pos == b->pos &&
           a->first_token == b->first_token &&
           a->checkpoint_valid == b->checkpoint_valid &&
           a->logits_only_terminal == b->logits_only_terminal &&
           a->token_hash == b->token_hash &&
           a->logit_hash == b->logit_hash &&
           a->progress_set == b->progress_set &&
           a->progress_ud_set == b->progress_ud_set &&
           a->display_progress_set == b->display_progress_set &&
           a->display_progress_ud_set == b->display_progress_ud_set &&
           a->cancel_set == b->cancel_set &&
           a->cancel_ud_set == b->cancel_ud_set &&
           a->progress_is_test_probe == b->progress_is_test_probe &&
           a->display_progress_is_test_probe ==
               b->display_progress_is_test_probe &&
           a->cancel_is_test_probe == b->cancel_is_test_probe &&
           a->progress_dispatches == b->progress_dispatches &&
           a->warmup_dispatches == b->warmup_dispatches &&
           a->layer_payload_dispatches == b->layer_payload_dispatches;
}

static void check_unchanged(ds4_session *s,
                            const ds4_test_session_state *before,
                            const char *label) {
    const ds4_test_session_state after = session_state(s, label);
    CHECK(states_equal(before, &after), "%s changed session state", label);
}

static void reset_error(char *err, size_t errlen) {
    memset(err, '?', errlen);
    if (errlen) err[errlen - 1] = '\0';
}

static void check_terminal_error(const char *label, int actual, int expected,
                                 const char *err) {
    CHECK(actual == expected, "%s returned %d, want %d",
          label, actual, expected);
    CHECK(err && strstr(err, "logits-only terminal") != NULL,
          "%s did not name logits-only terminal state: %s",
          label, err ? err : "(null)");
}

static ds4_session *create_policy_session(bool terminal, const char *label) {
    ds4_session *s = NULL;
    CHECK(ds4_test_session_create_policy(&s, TEST_CTX, terminal) == 0,
          "%s creation", label);
    CHECK(s != NULL, "%s is null", label);
    return s;
}

static int expected_sync_mode(bool native_cuda_build,
                              bool laguna,
                              ds4_backend backend,
                              bool session_distributed,
                              bool engine_distributed,
                              bool transport_tensor_parallel,
                              bool cuda_tensor_parallel,
                              int prompt_len,
                              int ctx_size) {
    if (prompt_len != ctx_size) return 0;
    return native_cuda_build && laguna && backend == DS4_BACKEND_CUDA &&
           !session_distributed && !engine_distributed &&
           !transport_tensor_parallel && !cuda_tensor_parallel
        ? 1 : -1;
}

static void check_sync_mode(const char *label,
                            bool native_cuda_build,
                            bool laguna,
                            ds4_backend backend,
                            bool session_distributed,
                            bool engine_distributed,
                            bool transport_tensor_parallel,
                            bool cuda_tensor_parallel,
                            int prompt_len,
                            int ctx_size) {
    const int want = expected_sync_mode(
        native_cuda_build, laguna, backend,
        session_distributed, engine_distributed,
        transport_tensor_parallel, cuda_tensor_parallel,
        prompt_len, ctx_size);
    const int got = ds4_test_logits_only_sync_mode(
        native_cuda_build, laguna, backend,
        session_distributed, engine_distributed,
        transport_tensor_parallel, cuda_tensor_parallel,
        prompt_len, ctx_size);
    CHECK(got == want,
          "%s mode=%d want=%d native=%d laguna=%d backend=%d "
          "session_dist=%d engine_dist=%d transport_tp=%d cuda_tp=%d "
          "prompt=%d ctx=%d",
          label, got, want, native_cuda_build, laguna, (int)backend,
          session_distributed, engine_distributed,
          transport_tensor_parallel, cuda_tensor_parallel,
          prompt_len, ctx_size);
}

static void test_eligibility_matrix(void) {
    static const int unequal_lengths[] = {-1, 0, 1, TEST_CTX - 1,
                                           TEST_CTX + 1, TEST_CTX * 2};
    static const ds4_backend backends[] = {
        DS4_BACKEND_METAL, DS4_BACKEND_CUDA, DS4_BACKEND_CPU,
    };

    for (int native = 0; native <= 1; native++) {
        for (int laguna = 0; laguna <= 1; laguna++) {
            for (size_t backend = 0; backend < ARRAY_LEN(backends); backend++) {
                for (int session_dist = 0; session_dist <= 1; session_dist++) {
                    for (int engine_dist = 0; engine_dist <= 1; engine_dist++) {
                        for (int transport_tp = 0; transport_tp <= 1; transport_tp++) {
                            for (int cuda_tp = 0; cuda_tp <= 1; cuda_tp++) {
                                check_sync_mode(
                                    "exact exhaustive matrix",
                                    native != 0, laguna != 0, backends[backend],
                                    session_dist != 0, engine_dist != 0,
                                    transport_tp != 0, cuda_tp != 0,
                                    TEST_CTX, TEST_CTX);
                                for (size_t len = 0;
                                     len < ARRAY_LEN(unequal_lengths); len++) {
                                    check_sync_mode(
                                        "non-equal delegates to ordinary",
                                        native != 0, laguna != 0,
                                        backends[backend], session_dist != 0,
                                        engine_dist != 0, transport_tp != 0,
                                        cuda_tp != 0, unequal_lengths[len],
                                        TEST_CTX);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    check_sync_mode("CPU-only build rejects exact context",
                    false, true, DS4_BACKEND_CPU,
                    false, false, false, false, TEST_CTX, TEST_CTX);
    check_sync_mode("ROCm build rejects CUDA-enum exact context",
                    false, true, DS4_BACKEND_CUDA,
                    false, false, false, false, TEST_CTX, TEST_CTX);
    check_sync_mode("native local Laguna CUDA accepts exact context",
                    true, true, DS4_BACKEND_CUDA,
                    false, false, false, false, TEST_CTX, TEST_CTX);
}

static void test_allowed_reads(void) {
    ds4_session *s = create_policy_session(true, "terminal read session");
    if (!s) return;

    const ds4_test_session_state before = session_state(s, "allowed reads before");
    CHECK(before.logits_only_terminal, "terminal read fixture is not terminal");
    CHECK(before.checkpoint_valid, "terminal read fixture lacks checkpoint");
    CHECK(before.pos > 0 && before.pos < TEST_CTX,
          "terminal read fixture position %d is not inside context", before.pos);

    const int argmax = ds4_session_argmax(s);
    CHECK(argmax >= 0 && argmax < TEST_VOCAB_SIZE,
          "argmax outside synthetic vocabulary: %d", argmax);
    const int excluding = ds4_session_argmax_excluding(s, argmax);
    CHECK(excluding >= 0 && excluding < TEST_VOCAB_SIZE && excluding != argmax,
          "argmax_excluding returned %d for excluded %d", excluding, argmax);

    ds4_token_score top[4];
    memset(top, 0xa5, sizeof(top));
    CHECK(ds4_session_top_logprobs(s, top, 4) == 4,
          "top_logprobs did not return four entries");
    CHECK(top[0].id == argmax,
          "top_logprobs first id=%d want argmax=%d", top[0].id, argmax);

    ds4_token_score score;
    memset(&score, 0xa5, sizeof(score));
    CHECK(ds4_session_token_logprob(s, argmax, &score) == 1,
          "token_logprob rejected terminal read");
    CHECK(score.id == argmax, "token_logprob id=%d want=%d", score.id, argmax);

    float *logits = malloc((size_t)TEST_VOCAB_SIZE * sizeof(*logits));
    CHECK(logits != NULL, "copy_logits allocation");
    if (logits) {
        for (int i = 0; i < TEST_VOCAB_SIZE; i++) logits[i] = -12345.0f;
        CHECK(ds4_session_copy_logits(s, logits, TEST_VOCAB_SIZE) ==
                  TEST_VOCAB_SIZE,
              "copy_logits did not return synthetic vocabulary size");
        CHECK(logits[argmax] == score.logit,
              "copy_logits argmax logit differs from token_logprob");
        free(logits);
    }

    CHECK(ds4_session_pos(s) == before.pos,
          "pos read=%d want=%d", ds4_session_pos(s), before.pos);
    CHECK(ds4_session_ctx(s) == TEST_CTX,
          "ctx read=%d want=%d", ds4_session_ctx(s), TEST_CTX);
    check_unchanged(s, &before, "allowed reads");
    ds4_session_free(s);
}

typedef struct {
    int progress_calls;
    int display_calls;
    int cancel_calls;
} callback_counts;

static void progress_callback(void *ud, const char *event,
                              int current, int total) {
    callback_counts *counts = ud;
    (void)event;
    (void)current;
    (void)total;
    counts->progress_calls++;
}

static void display_callback(void *ud, const char *event,
                             int current, int total) {
    callback_counts *counts = ud;
    (void)event;
    (void)current;
    (void)total;
    counts->display_calls++;
}

static bool cancel_callback(void *ud) {
    callback_counts *counts = ud;
    counts->cancel_calls++;
    return false;
}

static void test_scalar_mutation_rejections(void) {
    ds4_session *terminal = create_policy_session(true, "terminal mutation session");
    ds4_session *control = create_policy_session(false, "power control session");
    if (!terminal || !control) {
        ds4_session_free(terminal);
        ds4_session_free(control);
        return;
    }

    const ds4_test_session_state before =
        session_state(terminal, "scalar mutations before");
    CHECK(before.progress_set && before.progress_ud_set &&
              before.display_progress_set && before.display_progress_ud_set &&
              before.cancel_set && before.cancel_ud_set,
          "terminal fixture did not preseed callback probes");
    CHECK(before.progress_is_test_probe &&
              before.display_progress_is_test_probe &&
              before.cancel_is_test_probe,
          "terminal fixture callback identities are not the internal probes");
    CHECK(before.progress_dispatches == 0,
          "terminal fixture progress probe was already dispatched");
    CHECK(before.warmup_dispatches == 0,
          "terminal fixture warmup probe was already dispatched");
    CHECK(before.layer_payload_dispatches == 0,
          "terminal fixture layer-payload probe was already dispatched");
    CHECK(before.first_token == TEST_FIRST_CHECKPOINT_TOKEN,
          "terminal fixture first token=%d want deterministic token=%d",
          before.first_token, TEST_FIRST_CHECKPOINT_TOKEN);
    char err[256];
    int short_values[] = {TEST_FIRST_CHECKPOINT_TOKEN, 11, 13};
    ds4_tokens short_prompt = {
        .v = short_values,
        .len = (int)ARRAY_LEN(short_values),
        .cap = (int)ARRAY_LEN(short_values),
    };
    int exact_values[TEST_CTX];
    for (int i = 0; i < TEST_CTX; i++) exact_values[i] = i + 17;
    ds4_tokens exact_prompt = {
        .v = exact_values,
        .len = TEST_CTX,
        .cap = TEST_CTX,
    };

    reset_error(err, sizeof(err));
    check_terminal_error("ordinary sync",
                         ds4_session_sync(terminal, &short_prompt,
                                          err, sizeof(err)),
                         1, err);
    check_unchanged(terminal, &before, "ordinary sync");

    reset_error(err, sizeof(err));
    check_terminal_error("repeat logits-only sync",
                         ds4_session_sync_logits_only(terminal, &exact_prompt,
                                                      err, sizeof(err)),
                         1, err);
    check_unchanged(terminal, &before, "repeat logits-only sync");

    reset_error(err, sizeof(err));
    check_terminal_error("direct eval",
                         ds4_session_eval(terminal, 23, err, sizeof(err)),
                         1, err);
    check_unchanged(terminal, &before, "direct eval");

    reset_error(err, sizeof(err));
    check_terminal_error("internal eval-argmax",
                         ds4_session_eval_argmax(terminal, 29,
                                                 err, sizeof(err)),
                         -1, err);
    check_unchanged(terminal, &before, "internal eval-argmax");

    int accepted[] = {701, 702, 703};
    const int accepted_before[] = {701, 702, 703};
    reset_error(err, sizeof(err));
    check_terminal_error(
        "speculative eval",
        ds4_session_eval_speculative_argmax(
            terminal, 31, 3, TEST_VOCAB_SIZE - 1,
            accepted, (int)ARRAY_LEN(accepted), err, sizeof(err)),
        -1, err);
    CHECK(memcmp(accepted, accepted_before, sizeof(accepted)) == 0,
          "speculative eval changed accepted output");
    check_unchanged(terminal, &before, "speculative eval");

    const int drafts[] = {37, 41};
    reset_error(err, sizeof(err));
    check_terminal_error("TP speculative cycle",
                         ds4_session_tp_spec_cycle(
                             terminal, drafts, (int)ARRAY_LEN(drafts),
                             err, sizeof(err)),
                         1, err);
    check_unchanged(terminal, &before, "TP speculative cycle");

    reset_error(err, sizeof(err));
    check_terminal_error(
        "rewrite",
        ds4_session_rewrite_from_common(
            terminal, &short_prompt, 0, err, sizeof(err)),
        DS4_SESSION_REWRITE_ERROR, err);
    check_unchanged(terminal, &before, "rewrite");

    CHECK(ds4_session_common_prefix(terminal, &short_prompt) == 0,
          "terminal common-prefix access was not neutral");
    CHECK(ds4_session_tokens(terminal) == NULL,
          "terminal token access was not empty");
    CHECK(ds4_session_prefill_cap(terminal) == 0,
          "terminal prefill-cap access was not neutral");
    CHECK(ds4_session_power(control) != 100,
          "synthetic engine power must be non-neutral for terminal read test");
    CHECK(ds4_session_power(terminal) == 100,
          "terminal power access was not neutral");
    CHECK(!ds4_session_is_distributed(terminal),
          "terminal distributed-state access was not neutral");
    reset_error(err, sizeof(err));
    check_terminal_error("distributed route access",
                         ds4_session_distributed_route_ready(
                             terminal, err, sizeof(err)),
                         -1, err);
    check_unchanged(terminal, &before, "neutral accessors");

    uint64_t rng = UINT64_C(0x0123456789abcdef);
    const uint64_t rng_before = rng;
    CHECK(ds4_session_sample(terminal, 0.8f, 32, 0.9f, 0.05f, &rng) == -1,
          "terminal sampling did not return -1");
    CHECK(rng == rng_before, "terminal sampling changed RNG");
    check_unchanged(terminal, &before, "sampling");

    float *replacement = malloc((size_t)TEST_VOCAB_SIZE * sizeof(*replacement));
    CHECK(replacement != NULL, "replacement-logit allocation");
    if (replacement) {
        for (int i = 0; i < TEST_VOCAB_SIZE; i++) {
            replacement[i] = (float)(i % 97) / 97.0f;
        }
        CHECK(ds4_session_set_logits(terminal, replacement,
                                     TEST_VOCAB_SIZE) == 1,
              "terminal set_logits did not reject");
        check_unchanged(terminal, &before, "set_logits");
        free(replacement);
    }

    const int power_before = ds4_session_power(control);
    const int requested_power = power_before == 50 ? 60 : 50;
    CHECK(ds4_session_set_power(terminal, requested_power) == 1,
          "terminal set_power did not reject");
    CHECK(ds4_session_power(control) == power_before,
          "terminal set_power changed shared inert engine power");
    check_unchanged(terminal, &before, "set_power");

    callback_counts callbacks = {0};
    ds4_session_set_progress(terminal, progress_callback, &callbacks);
    ds4_session_set_display_progress(terminal, display_callback, &callbacks);
    ds4_session_set_cancel(terminal, cancel_callback, &callbacks);
    check_unchanged(terminal, &before, "callback setters");
    ds4_session_report_progress(terminal, "terminal", 1, 1);
    CHECK(callbacks.progress_calls == 0 && callbacks.display_calls == 0 &&
              callbacks.cancel_calls == 0,
          "terminal callback dispatched progress=%d display=%d cancel=%d",
          callbacks.progress_calls, callbacks.display_calls,
          callbacks.cancel_calls);
    check_unchanged(terminal, &before, "progress dispatch");

    callback_counts control_callbacks = {0};
    const ds4_test_session_state control_callbacks_before =
        session_state(control, "nonterminal callbacks before");
    CHECK(!control_callbacks_before.progress_set &&
              !control_callbacks_before.display_progress_set &&
              !control_callbacks_before.cancel_set,
          "nonterminal callback control was unexpectedly preseeded");
    ds4_session_set_progress(control, progress_callback, &control_callbacks);
    ds4_session_set_display_progress(
        control, display_callback, &control_callbacks);
    ds4_session_set_cancel(control, cancel_callback, &control_callbacks);
    const ds4_test_session_state control_callbacks_set =
        session_state(control, "nonterminal callbacks set");
    CHECK(control_callbacks_set.progress_set &&
              control_callbacks_set.progress_ud_set &&
              control_callbacks_set.display_progress_set &&
              control_callbacks_set.display_progress_ud_set &&
              control_callbacks_set.cancel_set &&
              control_callbacks_set.cancel_ud_set,
          "nonterminal callback setters did not install callbacks");
    CHECK(!control_callbacks_set.progress_is_test_probe &&
              !control_callbacks_set.display_progress_is_test_probe &&
              !control_callbacks_set.cancel_is_test_probe,
          "nonterminal user callbacks were mistaken for internal probes");
    CHECK(control_callbacks_set.progress_dispatches ==
              control_callbacks_before.progress_dispatches,
          "nonterminal callback setters dispatched the progress probe");
    ds4_session_report_progress(control, "control", 1, 1);
    CHECK(control_callbacks.progress_calls == 1 &&
              control_callbacks.display_calls == 0 &&
              control_callbacks.cancel_calls == 0,
          "nonterminal progress dispatch counts progress=%d display=%d cancel=%d",
          control_callbacks.progress_calls, control_callbacks.display_calls,
          control_callbacks.cancel_calls);
    const ds4_test_session_state control_callbacks_after =
        session_state(control, "nonterminal progress dispatched");
    CHECK(control_callbacks_after.progress_dispatches ==
              control_callbacks_before.progress_dispatches,
          "nonterminal user callback changed internal probe dispatch count");

    const ds4_test_session_state control_warmup_before =
        session_state(control, "nonterminal warmup before");
    ds4_session_gpu_warmup(control);
    ds4_test_session_state control_warmup_expected = control_warmup_before;
    control_warmup_expected.warmup_dispatches++;
    const ds4_test_session_state control_warmup_after =
        session_state(control, "nonterminal warmup after");
    CHECK(states_equal(&control_warmup_expected, &control_warmup_after),
          "nonterminal warmup did not cross the post-guard dispatch probe");

    ds4_session_gpu_warmup(terminal);
    check_unchanged(terminal, &before, "GPU warmup");

    reset_error(err, sizeof(err));
    check_terminal_error("layer-slice reset",
                         ds4_session_layer_slice_reset(
                             terminal, err, sizeof(err)),
                         1, err);
    check_unchanged(terminal, &before, "layer-slice reset");

    int layer_tokens[] = {43};
    const int layer_tokens_before[] = {43};
    float input_hc[8];
    float input_hc_before[8];
    float output_hc[8];
    float output_hc_before[8];
    float layer_logits[8];
    float layer_logits_before[8];
    memset(input_hc, 0x11, sizeof(input_hc));
    memset(output_hc, 0x22, sizeof(output_hc));
    memset(layer_logits, 0x33, sizeof(layer_logits));
    memcpy(input_hc_before, input_hc, sizeof(input_hc));
    memcpy(output_hc_before, output_hc, sizeof(output_hc));
    memcpy(layer_logits_before, layer_logits, sizeof(layer_logits));
    reset_error(err, sizeof(err));
    check_terminal_error(
        "layer evaluation",
        ds4_session_eval_layer_slice(
            terminal, layer_tokens, 1, (uint32_t)before.pos,
            0, 0, input_hc, output_hc, true, layer_logits,
            err, sizeof(err)),
        1, err);
    CHECK(memcmp(layer_tokens, layer_tokens_before, sizeof(layer_tokens)) == 0,
          "layer evaluation changed token input");
    CHECK(memcmp(input_hc, input_hc_before, sizeof(input_hc)) == 0,
          "layer evaluation changed hidden input");
    CHECK(memcmp(output_hc, output_hc_before, sizeof(output_hc)) == 0,
          "layer evaluation changed hidden output");
    CHECK(memcmp(layer_logits, layer_logits_before, sizeof(layer_logits)) == 0,
          "layer evaluation changed logits output");
    check_unchanged(terminal, &before, "layer evaluation");

    float head_logits[8];
    float head_logits_before[8];
    memset(head_logits, 0x44, sizeof(head_logits));
    memcpy(head_logits_before, head_logits, sizeof(head_logits));
    reset_error(err, sizeof(err));
    check_terminal_error(
        "output-head evaluation",
        ds4_session_eval_output_head_from_hc(
            terminal, input_hc, 1, head_logits, err, sizeof(err)),
        1, err);
    CHECK(memcmp(head_logits, head_logits_before, sizeof(head_logits)) == 0,
          "output-head evaluation changed logits output");
    check_unchanged(terminal, &before, "output-head evaluation");

    ds4_session_rewind(terminal, 0);
    check_unchanged(terminal, &before, "rewind");
    ds4_session_invalidate(terminal);
    check_unchanged(terminal, &before, "invalidation");

    ds4_session_free(control);
    ds4_session_free(terminal);
}

static const unsigned char file_sentinel[] = {
    0xd5, 0x34, 0xa1, 0x77, 0x09, 0xee, 0x51, 0x63,
    0x80, 0x22, 0xfa, 0xbc, 0x14, 0x95, 0x6d, 0x03,
};

static FILE *open_sentinel_file(const char *label, long offset) {
    FILE *fp = tmpfile();
    CHECK(fp != NULL, "%s tmpfile", label);
    if (!fp) return NULL;
    CHECK(fwrite(file_sentinel, 1, sizeof(file_sentinel), fp) ==
              sizeof(file_sentinel),
          "%s seed write", label);
    CHECK(fflush(fp) == 0, "%s seed flush", label);
    CHECK(fseek(fp, offset, SEEK_SET) == 0, "%s seed seek", label);
    return fp;
}

static void check_sentinel_file(FILE *fp, long expected_offset,
                                const char *label) {
    if (!fp) return;
    CHECK(ftell(fp) == expected_offset,
          "%s file offset=%ld want=%ld", label, ftell(fp), expected_offset);
    CHECK(fseek(fp, 0, SEEK_END) == 0, "%s seek end", label);
    CHECK(ftell(fp) == (long)sizeof(file_sentinel),
          "%s file size changed", label);
    CHECK(fseek(fp, 0, SEEK_SET) == 0, "%s seek start", label);
    unsigned char actual[sizeof(file_sentinel)];
    memset(actual, 0, sizeof(actual));
    CHECK(fread(actual, 1, sizeof(actual), fp) == sizeof(actual),
          "%s sentinel read", label);
    CHECK(memcmp(actual, file_sentinel, sizeof(actual)) == 0,
          "%s file contents changed", label);
    CHECK(fseek(fp, expected_offset, SEEK_SET) == 0,
          "%s restore offset", label);
}

static void test_payload_snapshot_rejections(void) {
    ds4_session *terminal = create_policy_session(true, "terminal payload session");
    ds4_session *control = create_policy_session(false, "payload control session");
    if (!terminal || !control) {
        ds4_session_free(control);
        ds4_session_free(terminal);
        return;
    }
    const ds4_test_session_state before =
        session_state(terminal, "payloads before");
    char err[256];

    CHECK(ds4_session_payload_bytes(terminal) == 0,
          "terminal payload-size query was not zero");
    check_unchanged(terminal, &before, "payload-size query");

    char staged_path_sentinel[] = "terminal-stage-sentinel";
    ds4_session_payload_file staged = {
        .path = staged_path_sentinel,
        .bytes = UINT64_C(0x123456789abcdef0),
    };
    reset_error(err, sizeof(err));
    check_terminal_error("stage payload",
                         ds4_session_stage_payload(
                             terminal, &staged, err, sizeof(err)),
                         1, err);
    CHECK(staged.path == staged_path_sentinel &&
              staged.bytes == UINT64_C(0x123456789abcdef0),
          "stage payload changed caller-owned sentinel");
    check_unchanged(terminal, &before, "stage payload");
    if (staged.path && staged.path != staged_path_sentinel) {
        ds4_session_payload_file_free(&staged);
    }

    FILE *save_fp = open_sentinel_file("save payload", 5);
    if (save_fp) {
        reset_error(err, sizeof(err));
        check_terminal_error("save payload",
                             ds4_session_save_payload(
                                 terminal, save_fp, err, sizeof(err)),
                             1, err);
        check_sentinel_file(save_fp, 5, "save payload");
        fclose(save_fp);
    }
    check_unchanged(terminal, &before, "save payload");

    FILE *load_fp = open_sentinel_file("load payload", 3);
    if (load_fp) {
        reset_error(err, sizeof(err));
        check_terminal_error("load payload",
                             ds4_session_load_payload(
                                 terminal, load_fp, sizeof(file_sentinel),
                                 err, sizeof(err)),
                             1, err);
        check_sentinel_file(load_fp, 3, "load payload");
        fclose(load_fp);
    }
    check_unchanged(terminal, &before, "load payload");

    uint8_t *save_snapshot_bytes = malloc(32);
    CHECK(save_snapshot_bytes != NULL, "save snapshot sentinel allocation");
    if (save_snapshot_bytes) {
        memset(save_snapshot_bytes, 0x5a, 32);
        uint8_t save_snapshot_before[32];
        memcpy(save_snapshot_before, save_snapshot_bytes, 32);
        ds4_session_snapshot snapshot = {
            .ptr = save_snapshot_bytes,
            .len = 7,
            .cap = 32,
        };
        reset_error(err, sizeof(err));
        check_terminal_error("save snapshot",
                             ds4_session_save_snapshot(
                                 terminal, &snapshot, err, sizeof(err)),
                             1, err);
        CHECK(snapshot.ptr == save_snapshot_bytes && snapshot.len == 7 &&
                  snapshot.cap == 32,
              "save snapshot changed pointer/length/capacity sentinel");
        if (snapshot.ptr == save_snapshot_bytes) {
            CHECK(memcmp(snapshot.ptr, save_snapshot_before, 32) == 0,
                  "save snapshot changed buffer sentinel");
            free(save_snapshot_bytes);
        } else {
            free(snapshot.ptr);
        }
    }
    check_unchanged(terminal, &before, "save snapshot");

    uint8_t *load_snapshot_bytes = malloc(32);
    CHECK(load_snapshot_bytes != NULL, "load snapshot sentinel allocation");
    if (load_snapshot_bytes) {
        memset(load_snapshot_bytes, 0x6b, 32);
        uint8_t load_snapshot_before[32];
        memcpy(load_snapshot_before, load_snapshot_bytes, 32);
        const ds4_session_snapshot snapshot = {
            .ptr = load_snapshot_bytes,
            .len = 17,
            .cap = 32,
        };
        reset_error(err, sizeof(err));
        check_terminal_error("load snapshot",
                             ds4_session_load_snapshot(
                                 terminal, &snapshot, err, sizeof(err)),
                             1, err);
        CHECK(memcmp(load_snapshot_bytes, load_snapshot_before, 32) == 0,
              "load snapshot changed input buffer sentinel");
        free(load_snapshot_bytes);
    }
    check_unchanged(terminal, &before, "load snapshot");

    const ds4_test_session_state control_layer_before =
        session_state(control, "nonterminal layer-payload size before");
    CHECK(ds4_session_layer_payload_bytes(control, 0, 0) == 0,
          "CPU control layer-payload-size query was not zero");
    ds4_test_session_state control_layer_expected = control_layer_before;
    control_layer_expected.layer_payload_dispatches++;
    const ds4_test_session_state control_layer_after =
        session_state(control, "nonterminal layer-payload size after");
    CHECK(states_equal(&control_layer_expected, &control_layer_after),
          "nonterminal layer-payload size did not cross dispatch probe");

    CHECK(ds4_session_layer_payload_bytes(terminal, 0, 0) == 0,
          "terminal layer-payload-size query was not zero");
    check_unchanged(terminal, &before, "layer-payload-size query");

    FILE *save_layer_fp = open_sentinel_file("save layer payload", 6);
    if (save_layer_fp) {
        reset_error(err, sizeof(err));
        check_terminal_error("save layer payload",
                             ds4_session_save_layer_payload(
                                 terminal, save_layer_fp, 0, 0,
                                 err, sizeof(err)),
                             1, err);
        check_sentinel_file(save_layer_fp, 6, "save layer payload");
        fclose(save_layer_fp);
    }
    check_unchanged(terminal, &before, "save layer payload");

    int load_tokens[] = {47, 53};
    const int load_tokens_before[] = {47, 53};
    FILE *load_layer_fp = open_sentinel_file("load layer payload", 2);
    if (load_layer_fp) {
        reset_error(err, sizeof(err));
        check_terminal_error("load layer payload",
                             ds4_session_load_layer_payload(
                                 terminal, load_layer_fp,
                                 sizeof(file_sentinel), load_tokens,
                                 (uint32_t)ARRAY_LEN(load_tokens), 0, 0,
                                 err, sizeof(err)),
                             1, err);
        check_sentinel_file(load_layer_fp, 2, "load layer payload");
        fclose(load_layer_fp);
    }
    CHECK(memcmp(load_tokens, load_tokens_before, sizeof(load_tokens)) == 0,
          "load layer payload changed token input");
    check_unchanged(terminal, &before, "load layer payload");

    ds4_session_free(control);
    ds4_session_free(terminal);
}

static ds4_tokens make_extension(ds4_session *s, const char *label) {
    ds4_tokens result = {0};
    const ds4_tokens *base = ds4_session_tokens(s);
    CHECK(base != NULL, "%s base token access", label);
    if (!base) return result;
    CHECK(base->len > 0 && base->len + 1 < ds4_session_ctx(s),
          "%s base len=%d ctx=%d cannot extend",
          label, base->len, ds4_session_ctx(s));
    if (base->len <= 0 || base->len + 1 >= ds4_session_ctx(s)) return result;
    result.len = base->len + 1;
    result.cap = result.len;
    result.v = malloc((size_t)result.len * sizeof(*result.v));
    CHECK(result.v != NULL, "%s extension allocation", label);
    if (!result.v) {
        result.len = 0;
        result.cap = 0;
        return result;
    }
    memcpy(result.v, base->v, (size_t)base->len * sizeof(*result.v));
    result.v[result.len - 1] = 59;
    return result;
}

static void test_batch_atomic_rejections(void) {
    char err[256];

    ds4_session *singleton =
        create_policy_session(true, "singleton terminal batch session");
    if (singleton) {
        const ds4_test_session_state singleton_before =
            session_state(singleton, "singleton batch before");
        ds4_decode_item singleton_item = {.session = singleton, .token = 61};
        reset_error(err, sizeof(err));
        check_terminal_error("singleton terminal batch",
                             ds4_sessions_eval_batch(
                                 &singleton_item, 1, err, sizeof(err)),
                             1, err);
        check_unchanged(singleton, &singleton_before,
                        "singleton terminal batch");
        ds4_session_free(singleton);
    }

    ds4_session *ordinary_control =
        create_policy_session(false, "ordinary batch control");
    ds4_session *ordinary_terminal =
        create_policy_session(true, "ordinary batch terminal");
    if (ordinary_control && ordinary_terminal) {
        const ds4_test_session_state control_before =
            session_state(ordinary_control, "ordinary control before");
        const ds4_test_session_state terminal_before =
            session_state(ordinary_terminal, "ordinary terminal before");
        ds4_decode_item items[] = {
            {.session = ordinary_control, .token = 67},
            {.session = ordinary_terminal, .token = 71},
        };
        reset_error(err, sizeof(err));
        check_terminal_error("ordinary batch terminal second",
                             ds4_sessions_eval_batch(
                                 items, (int)ARRAY_LEN(items),
                                 err, sizeof(err)),
                             1, err);
        check_unchanged(ordinary_control, &control_before,
                        "ordinary batch control");
        check_unchanged(ordinary_terminal, &terminal_before,
                        "ordinary batch terminal");
    }
    ds4_session_free(ordinary_terminal);
    ds4_session_free(ordinary_control);

    ds4_session *prefill = create_policy_session(false, "mixed prefill control");
    ds4_session *mixed_control =
        create_policy_session(false, "mixed decode control");
    ds4_session *mixed_terminal =
        create_policy_session(true, "mixed decode terminal");
    if (prefill && mixed_control && mixed_terminal) {
        ds4_tokens extension = make_extension(prefill, "mixed prefill");
        const ds4_test_session_state prefill_before =
            session_state(prefill, "mixed prefill before");
        const ds4_test_session_state control_before =
            session_state(mixed_control, "mixed control before");
        const ds4_test_session_state terminal_before =
            session_state(mixed_terminal, "mixed terminal before");
        ds4_decode_item items[] = {
            {.session = mixed_control, .token = 73},
            {.session = mixed_terminal, .token = 79},
        };
        if (extension.v) {
            reset_error(err, sizeof(err));
            check_terminal_error("mixed batch terminal decode second",
                                 ds4_sessions_eval_batch_with_prefill(
                                     items, (int)ARRAY_LEN(items),
                                     prefill, &extension, err, sizeof(err)),
                                 1, err);
            check_unchanged(prefill, &prefill_before,
                            "mixed decode-terminal prefill");
            check_unchanged(mixed_control, &control_before,
                            "mixed decode-terminal control");
            check_unchanged(mixed_terminal, &terminal_before,
                            "mixed decode-terminal terminal");
        }
        free(extension.v);
    }
    ds4_session_free(mixed_terminal);
    ds4_session_free(mixed_control);
    ds4_session_free(prefill);

    ds4_session *terminal_prefill =
        create_policy_session(true, "terminal mixed prefill");
    ds4_session *decode_control0 =
        create_policy_session(false, "terminal-prefill decode control zero");
    ds4_session *decode_control1 =
        create_policy_session(false, "terminal-prefill decode control one");
    if (terminal_prefill && decode_control0 && decode_control1) {
        const ds4_test_session_state prefill_before =
            session_state(terminal_prefill, "terminal prefill before");
        const ds4_test_session_state control0_before =
            session_state(decode_control0, "terminal-prefill control zero before");
        const ds4_test_session_state control1_before =
            session_state(decode_control1, "terminal-prefill control one before");
        int prompt_values[] = {83, 89, 97};
        ds4_tokens prompt = {
            .v = prompt_values,
            .len = (int)ARRAY_LEN(prompt_values),
            .cap = (int)ARRAY_LEN(prompt_values),
        };
        ds4_decode_item items[] = {
            {.session = decode_control0, .token = 101},
            {.session = decode_control1, .token = 103},
        };
        reset_error(err, sizeof(err));
        check_terminal_error("mixed batch terminal prefill",
                             ds4_sessions_eval_batch_with_prefill(
                                 items, (int)ARRAY_LEN(items),
                                 terminal_prefill, &prompt,
                                 err, sizeof(err)),
                             1, err);
        check_unchanged(terminal_prefill, &prefill_before,
                        "mixed terminal-prefill terminal");
        check_unchanged(decode_control0, &control0_before,
                        "mixed terminal-prefill control zero");
        check_unchanged(decode_control1, &control1_before,
                        "mixed terminal-prefill control one");
    }
    ds4_session_free(decode_control1);
    ds4_session_free(decode_control0);
    ds4_session_free(terminal_prefill);
}

static void test_exact_cache_session_safety(void) {
    int oversized_rc = -1;
    int first_rc = -1;
    int second_rc = -1;
    int reuse_rc = -1;
    CHECK(ds4_test_session_limit_lifecycle(
              true, true, 32768, 32768, 32769,
              &oversized_rc, &first_rc, &second_rc, &reuse_rc) == 0,
          "exact-cache lifecycle fixture");
    CHECK(oversized_rc == 2,
          "oversized exact-cache session returned %d, want 2", oversized_rc);
    CHECK(first_rc == 0,
          "first exact-cache session returned %d, want 0", first_rc);
    CHECK(second_rc == 2,
          "second live exact-cache session returned %d, want 2", second_rc);
    CHECK(reuse_rc == 0,
          "exact-cache slot after free returned %d, want 0", reuse_rc);

    oversized_rc = first_rc = second_rc = reuse_rc = -1;
    CHECK(ds4_test_session_limit_lifecycle(
              false, true, 0, 32768, 32769,
              &oversized_rc, &first_rc, &second_rc, &reuse_rc) == 0,
          "legacy-cache lifecycle fixture");
    CHECK(oversized_rc == 0 && first_rc == 0 && second_rc == 0 && reuse_rc == 0,
          "legacy sessions were restricted: oversized=%d first=%d second=%d reuse=%d",
          oversized_rc, first_rc, second_rc, reuse_rc);

    oversized_rc = first_rc = second_rc = reuse_rc = -1;
    CHECK(ds4_test_session_limit_lifecycle(
              true, false, 0, 32768, 32769,
              &oversized_rc, &first_rc, &second_rc, &reuse_rc) == 0,
          "non-graph exact-cache lifecycle fixture");
    CHECK(oversized_rc == 0 && first_rc == 0 && second_rc == 0 && reuse_rc == 0,
          "non-graph sessions were restricted: oversized=%d first=%d second=%d reuse=%d",
          oversized_rc, first_rc, second_rc, reuse_rc);

    oversized_rc = -1;
    int concurrent_rc = -1;
    reuse_rc = -1;
    int diagnostic_rc = -1;
    CHECK(ds4_test_direct_graph_limit_lifecycle(
              &oversized_rc, &concurrent_rc, &reuse_rc,
              &diagnostic_rc) == 0,
          "direct graph lifecycle fixture");
    CHECK(oversized_rc == 2,
          "oversized direct graph returned %d, want 2", oversized_rc);
    CHECK(concurrent_rc == 2,
          "direct graph alongside a live session returned %d, want 2",
          concurrent_rc);
    CHECK(reuse_rc == 0,
          "direct graph after session free returned %d, want 0", reuse_rc);
    CHECK(diagnostic_rc == 2,
          "exact-cache direct graph diagnostic returned %d, want 2",
          diagnostic_rc);
}

int main(void) {
    test_eligibility_matrix();
    test_allowed_reads();
    test_scalar_mutation_rejections();
    test_payload_snapshot_rejections();
    test_batch_atomic_rejections();
    test_exact_cache_session_safety();

    if (failures != 0) {
        fprintf(stderr, "test_session_logits_only FAIL failures=%d\n", failures);
        return 1;
    }
    puts("test_session_logits_only PASS");
    return 0;
}
