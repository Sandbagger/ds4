/* Frozen Laguna 512+1 behavioral probe for release DS4 CUDA objects.
 *
 * Starting from an exact 512-token prefix, both trajectories explicitly
 * evaluate token 3612 at position 512.  The resulting position-513 logits
 * are the first behavioral observation.  Greedy and Poolside-reference
 * teacher-forced trajectories run in separate, sequential sessions.
 */

#include "ds4.h"

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

enum {
    PREFIX_TOKENS = 512,
    RESUME_TOKEN = 3612,
    CONTEXT_TOKENS = 1024,
    VOCAB_SIZE = 100352,
    TOP_K = 20,
    MAX_STEPS = 32,
};

static const char SCHEMA[] = "laguna-ds4-behavior-probe/v1";
static const int32_t FROZEN_PREFIX_PATTERN[] = {
    785, 10068, 3612, 8473, 1077, 1857, 606, 330,
    2746, 22910, 1059, 4158, 7799, 83, 268,
};

typedef struct {
    const char *model;
    const char *tokens;
    const char *reference;
    const char *out;
    int steps;
} options;

typedef struct {
    int token;
    int top20[TOP_K];
} greedy_step;

typedef struct {
    int target_token;
    int top20[TOP_K];
    double target_logprob;
    double target_nll;
} teacher_step;

static void usage(FILE *stream, const char *program) {
    fprintf(stream,
            "usage: %s --model MODEL --tokens PREFIX-512.i32 "
            "--reference REFERENCE-CONTINUATION.i32 --steps 1..32 "
            "--out OUTPUT.json\n",
            program);
}

static bool parse_steps(const char *text, int *out) {
    char *end = NULL;
    errno = 0;
    const long value = strtol(text, &end, 10);
    if (errno != 0 || !end || end == text || *end != '\0' ||
        value < 1 || value > MAX_STEPS) {
        return false;
    }
    *out = (int)value;
    return true;
}

static bool parse_options(int argc, char **argv, options *out) {
    memset(out, 0, sizeof(*out));
    for (int i = 1; i < argc; i++) {
        const char *flag = argv[i];
        if (i + 1 >= argc) return false;
        const char *value = argv[++i];
        if (strcmp(flag, "--model") == 0 && !out->model) {
            out->model = value;
        } else if (strcmp(flag, "--tokens") == 0 && !out->tokens) {
            out->tokens = value;
        } else if (strcmp(flag, "--reference") == 0 && !out->reference) {
            out->reference = value;
        } else if (strcmp(flag, "--steps") == 0 && out->steps == 0) {
            if (!parse_steps(value, &out->steps)) return false;
        } else if (strcmp(flag, "--out") == 0 && !out->out) {
            out->out = value;
        } else {
            return false;
        }
    }
    return out->model && out->tokens && out->reference && out->out &&
           out->steps >= 1 && out->steps <= MAX_STEPS;
}

static bool validate_paths(const options *opt) {
    struct stat status;
    if (stat(opt->model, &status) != 0 || !S_ISREG(status.st_mode) ||
        status.st_size <= 0) {
        fprintf(stderr, "behavior: model is not a non-empty regular file: %s\n",
                opt->model);
        return false;
    }
    errno = 0;
    if (lstat(opt->out, &status) == 0) {
        fprintf(stderr, "behavior: output already exists: %s\n", opt->out);
        return false;
    }
    if (errno != ENOENT) {
        fprintf(stderr, "behavior: inspect output %s: %s\n",
                opt->out, strerror(errno));
        return false;
    }
    return true;
}

static bool read_i32_tokens_exact(
        const char *path,
        int count,
        int32_t *out,
        const char *label) {
    FILE *stream = fopen(path, "rb");
    if (!stream) {
        fprintf(stderr, "behavior: open %s %s: %s\n",
                label, path, strerror(errno));
        return false;
    }
    const size_t byte_count = (size_t)count * sizeof(int32_t);
    unsigned char *bytes = malloc(byte_count);
    if (!bytes) {
        fprintf(stderr, "behavior: allocate %s input\n", label);
        fclose(stream);
        return false;
    }
    const size_t got = fread(bytes, 1, byte_count, stream);
    const int trailing = fgetc(stream);
    const int close_rc = fclose(stream);
    if (got != byte_count || trailing != EOF || close_rc != 0) {
        fprintf(stderr, "behavior: %s %s is not exactly %zu bytes\n",
                label, path, byte_count);
        free(bytes);
        return false;
    }
    for (int i = 0; i < count; i++) {
        const unsigned char *p = bytes + (size_t)i * sizeof(int32_t);
        const uint32_t raw = (uint32_t)p[0] |
                             ((uint32_t)p[1] << 8u) |
                             ((uint32_t)p[2] << 16u) |
                             ((uint32_t)p[3] << 24u);
        int32_t token = 0;
        memcpy(&token, &raw, sizeof(token));
        if (token < 0 || token >= VOCAB_SIZE) {
            fprintf(stderr,
                    "behavior: %s token %d is out of range: %d\n",
                    label, i, token);
            free(bytes);
            return false;
        }
        out[i] = token;
    }
    free(bytes);
    return true;
}

static bool validate_frozen_prefix(const int32_t prefix[PREFIX_TOKENS]) {
    const int pattern_count =
            (int)(sizeof(FROZEN_PREFIX_PATTERN) /
                  sizeof(FROZEN_PREFIX_PATTERN[0]));
    for (int i = 0; i < PREFIX_TOKENS; i++) {
        const int32_t expected = FROZEN_PREFIX_PATTERN[i % pattern_count];
        if (prefix[i] != expected) {
            fprintf(stderr,
                    "behavior: prefix token %d=%d expected frozen token %d\n",
                    i, prefix[i], expected);
            return false;
        }
    }
    return true;
}

static bool create_resumed_session(
        ds4_engine *engine,
        const ds4_tokens *prefix,
        ds4_session **out,
        const char *name) {
    char error[256] = {0};
    if (ds4_session_create(out, engine, CONTEXT_TOKENS) != 0) {
        fprintf(stderr, "behavior: create %s session failed\n", name);
        return false;
    }
    if (ds4_session_sync(*out, prefix, error, sizeof(error)) != 0) {
        fprintf(stderr, "behavior: sync %s prefix: %s\n", name, error);
        ds4_session_free(*out);
        *out = NULL;
        return false;
    }
    if (ds4_session_pos(*out) != PREFIX_TOKENS) {
        fprintf(stderr, "behavior: %s prefix position=%d expected=%d\n",
                name, ds4_session_pos(*out), PREFIX_TOKENS);
        ds4_session_free(*out);
        *out = NULL;
        return false;
    }
    if (ds4_session_eval(*out, RESUME_TOKEN, error, sizeof(error)) != 0) {
        fprintf(stderr, "behavior: %s resume decode: %s\n", name, error);
        ds4_session_free(*out);
        *out = NULL;
        return false;
    }
    if (ds4_session_pos(*out) != PREFIX_TOKENS + 1) {
        fprintf(stderr, "behavior: %s resumed position=%d expected=%d\n",
                name, ds4_session_pos(*out), PREFIX_TOKENS + 1);
        ds4_session_free(*out);
        *out = NULL;
        return false;
    }
    return true;
}

static bool copy_finite_logits(
        ds4_session *session,
        float *logits,
        const char *trajectory,
        int step) {
    if (ds4_session_copy_logits(session, logits, VOCAB_SIZE) != VOCAB_SIZE) {
        fprintf(stderr, "behavior: copy %s logits at step %d failed\n",
                trajectory, step + 1);
        return false;
    }
    for (int id = 0; id < VOCAB_SIZE; id++) {
        if (!isfinite(logits[id])) {
            fprintf(stderr,
                    "behavior: %s logit %d at step %d is non-finite\n",
                    trajectory, id, step + 1);
            return false;
        }
    }
    return true;
}

static bool ranks_before(float value, int id, float other, int other_id) {
    return value > other || (value == other && id < other_id);
}

static void vector_top20(const float *values, int out[TOP_K]) {
    for (int i = 0; i < TOP_K; i++) out[i] = -1;
    for (int id = 0; id < VOCAB_SIZE; id++) {
        int at = 0;
        while (at < TOP_K && out[at] >= 0 &&
               !ranks_before(values[id], id, values[out[at]], out[at])) {
            at++;
        }
        if (at == TOP_K) continue;
        for (int j = TOP_K - 1; j > at; j--) out[j] = out[j - 1];
        out[at] = id;
    }
}

static double logsumexp_binary64(const float *logits) {
    double maximum = (double)logits[0];
    for (int id = 1; id < VOCAB_SIZE; id++) {
        const double value = (double)logits[id];
        if (value > maximum) maximum = value;
    }
    double total = 0.0;
    for (int id = 0; id < VOCAB_SIZE; id++) {
        total += exp((double)logits[id] - maximum);
    }
    return maximum + log(total);
}

static bool eval_token(
        ds4_session *session,
        int token,
        int expected_position,
        const char *trajectory,
        int step) {
    char error[256] = {0};
    if (ds4_session_eval(session, token, error, sizeof(error)) != 0) {
        fprintf(stderr, "behavior: %s eval step %d token %d: %s\n",
                trajectory, step + 1, token, error);
        return false;
    }
    if (ds4_session_pos(session) != expected_position) {
        fprintf(stderr,
                "behavior: %s position after step %d=%d expected=%d\n",
                trajectory, step + 1, ds4_session_pos(session),
                expected_position);
        return false;
    }
    return true;
}

static bool run_greedy_trajectory(
        ds4_engine *engine,
        const ds4_tokens *prefix,
        int steps,
        float *logits,
        greedy_step out[MAX_STEPS]) {
    ds4_session *session = NULL;
    if (!create_resumed_session(engine, prefix, &session, "greedy")) {
        return false;
    }
    bool ok = true;
    for (int step = 0; step < steps; step++) {
        if (!copy_finite_logits(session, logits, "greedy", step)) {
            ok = false;
            break;
        }
        vector_top20(logits, out[step].top20);
        out[step].token = out[step].top20[0];
        if (!eval_token(session, out[step].token,
                        PREFIX_TOKENS + 2 + step, "greedy", step)) {
            ok = false;
            break;
        }
    }
    ds4_session_free(session);
    return ok;
}

static bool run_teacher_trajectory(
        ds4_engine *engine,
        const ds4_tokens *prefix,
        const int32_t reference[MAX_STEPS],
        int steps,
        float *logits,
        teacher_step out[MAX_STEPS],
        double *nll_total) {
    ds4_session *session = NULL;
    if (!create_resumed_session(engine, prefix, &session, "teacher")) {
        return false;
    }
    bool ok = true;
    *nll_total = 0.0;
    for (int step = 0; step < steps; step++) {
        if (!copy_finite_logits(session, logits, "teacher", step)) {
            ok = false;
            break;
        }
        out[step].target_token = reference[step];
        vector_top20(logits, out[step].top20);
        const double log_normalizer = logsumexp_binary64(logits);
        out[step].target_logprob =
                (double)logits[out[step].target_token] - log_normalizer;
        out[step].target_nll = -out[step].target_logprob;
        if (!isfinite(out[step].target_logprob) ||
            !isfinite(out[step].target_nll)) {
            fprintf(stderr, "behavior: teacher NLL at step %d is non-finite\n",
                    step + 1);
            ok = false;
            break;
        }
        *nll_total += out[step].target_nll;
        if (!eval_token(session, out[step].target_token,
                        PREFIX_TOKENS + 2 + step, "teacher", step)) {
            ok = false;
            break;
        }
    }
    ds4_session_free(session);
    return ok;
}

static void write_id_array(FILE *stream, const int *ids, int count) {
    fputc('[', stream);
    for (int i = 0; i < count; i++) {
        if (i != 0) fputc(',', stream);
        fprintf(stream, "%d", ids[i]);
    }
    fputc(']', stream);
}

static bool write_json(
        const options *opt,
        const char *reduction,
        const greedy_step greedy[MAX_STEPS],
        const teacher_step teacher[MAX_STEPS],
        double teacher_nll_total) {
    int matching_prefix = 0;
    while (matching_prefix < opt->steps &&
           greedy[matching_prefix].token ==
                   teacher[matching_prefix].target_token) {
        matching_prefix++;
    }
    const int fd = open(opt->out, O_WRONLY | O_CREAT | O_EXCL, 0600);
    if (fd < 0) {
        fprintf(stderr, "behavior: create output %s: %s\n",
                opt->out, strerror(errno));
        return false;
    }
    FILE *stream = fdopen(fd, "w");
    if (!stream) {
        fprintf(stderr, "behavior: fdopen output %s: %s\n",
                opt->out, strerror(errno));
        close(fd);
        unlink(opt->out);
        return false;
    }

    fprintf(stream, "{\n");
    fprintf(stream, "  \"schema\": \"%s\",\n", SCHEMA);
    fprintf(stream, "  \"prefix_tokens\": %d,\n", PREFIX_TOKENS);
    fprintf(stream, "  \"resume_token\": %d,\n", RESUME_TOKEN);
    fprintf(stream, "  \"initial_position\": %d,\n", PREFIX_TOKENS + 1);
    fprintf(stream, "  \"steps\": %d,\n", opt->steps);
    fprintf(stream, "  \"mmvq_reduction\": \"%s\",\n", reduction);
    fprintf(stream, "  \"greedy_tokens\": ");
    fputc('[', stream);
    for (int step = 0; step < opt->steps; step++) {
        if (step != 0) fputc(',', stream);
        fprintf(stream, "%d", greedy[step].token);
    }
    fprintf(stream, "],\n  \"greedy_steps\": [\n");
    for (int step = 0; step < opt->steps; step++) {
        fprintf(stream,
                "    {\"step\":%d,\"position\":%d,"
                "\"greedy_token\":%d,\"top20\":",
                step + 1, PREFIX_TOKENS + 1 + step, greedy[step].token);
        write_id_array(stream, greedy[step].top20, TOP_K);
        fprintf(stream, "}%s\n", step + 1 == opt->steps ? "" : ",");
    }
    fprintf(stream, "  ],\n");
    fprintf(stream, "  \"greedy_matching_prefix\": %d,\n", matching_prefix);
    fprintf(stream, "  \"teacher_steps\": [\n");
    for (int step = 0; step < opt->steps; step++) {
        fprintf(stream,
                "    {\"step\":%d,\"position\":%d,"
                "\"target_token\":%d,\"top20\":",
                step + 1, PREFIX_TOKENS + 1 + step,
                teacher[step].target_token);
        write_id_array(stream, teacher[step].top20, TOP_K);
        fprintf(stream,
                ",\"target_logprob\":%.17g,\"target_nll\":%.17g}%s\n",
                teacher[step].target_logprob, teacher[step].target_nll,
                step + 1 == opt->steps ? "" : ",");
    }
    fprintf(stream, "  ],\n");
    fprintf(stream, "  \"teacher_nll_total\": %.17g,\n",
            teacher_nll_total);
    fprintf(stream, "  \"teacher_nll_avg\": %.17g\n",
            teacher_nll_total / (double)opt->steps);
    fprintf(stream, "}\n");

    bool ok = !ferror(stream);
    if (fclose(stream) != 0) ok = false;
    if (!ok) {
        fprintf(stderr, "behavior: write output %s failed\n", opt->out);
        unlink(opt->out);
    }
    return ok;
}

int main(int argc, char **argv) {
    options opt;
    if (!parse_options(argc, argv, &opt)) {
        usage(stderr, argv[0]);
        return 2;
    }
    if (!validate_paths(&opt)) return 2;

    const char *reduction_env = getenv("DS4_MM_VQ_REDUCTION");
    const char *reduction = "default";
    if (reduction_env && reduction_env[0] != '\0') {
        if (strcmp(reduction_env, "poolside") != 0) {
            fprintf(stderr,
                    "behavior: DS4_MM_VQ_REDUCTION must be unset or poolside\n");
            return 2;
        }
        reduction = "poolside";
    }

    int rc = 1;
    int32_t prefix_ids[PREFIX_TOKENS];
    int32_t reference[MAX_STEPS] = {0};
    ds4_tokens prefix = {0};
    ds4_engine *engine = NULL;
    float *logits = NULL;
    greedy_step greedy[MAX_STEPS] = {0};
    teacher_step teacher[MAX_STEPS] = {0};
    double teacher_nll_total = 0.0;

    if (!read_i32_tokens_exact(
                opt.tokens, PREFIX_TOKENS, prefix_ids, "prefix") ||
        !validate_frozen_prefix(prefix_ids) ||
        !read_i32_tokens_exact(
                opt.reference, opt.steps, reference, "reference")) {
        goto cleanup;
    }
    for (int i = 0; i < PREFIX_TOKENS; i++) {
        ds4_tokens_push(&prefix, prefix_ids[i]);
    }
    if (prefix.len != PREFIX_TOKENS) {
        fprintf(stderr, "behavior: construct prefix failed\n");
        goto cleanup;
    }

    const ds4_engine_options engine_options = {
        .model_path = opt.model,
        .backend = DS4_BACKEND_CUDA,
        .n_threads = 1,
    };
    if (ds4_engine_open(&engine, &engine_options) != 0) {
        fprintf(stderr, "behavior: open Laguna CUDA engine failed\n");
        goto cleanup;
    }
    if (ds4_engine_vocab_size(engine) != VOCAB_SIZE) {
        fprintf(stderr, "behavior: vocabulary mismatch\n");
        goto cleanup;
    }
    logits = malloc((size_t)VOCAB_SIZE * sizeof(*logits));
    if (!logits) {
        fprintf(stderr, "behavior: allocate logits failed\n");
        goto cleanup;
    }

    if (!run_greedy_trajectory(
                engine, &prefix, opt.steps, logits, greedy) ||
        !run_teacher_trajectory(
                engine, &prefix, reference, opt.steps, logits,
                teacher, &teacher_nll_total) ||
        !write_json(&opt, reduction, greedy, teacher, teacher_nll_total)) {
        goto cleanup;
    }

    fprintf(stderr,
            "probe_ds4_laguna_behavior PASS steps=%d prefix=512+1 "
            "reduction=%s teacher_nll=%.17g\n",
            opt.steps, reduction, teacher_nll_total);
    rc = 0;

cleanup:
    free(logits);
    ds4_engine_close(engine);
    ds4_tokens_free(&prefix);
    return rc;
}
