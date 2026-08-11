/* Direct DS4 512+1 decode capture for the Laguna layer-1 routed MoE.
 *
 * This producer runs the exact resident CUDA path twice from the same
 * 512-token prefix.  The first decode is the control; the second enables the
 * test-only tensor observer for token 513.  Their logits must remain bitwise
 * identical, so capture stores cannot silently become part of inference.
 */

#include "ds4.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

enum {
    PREFIX_TOKENS = 512,
    RESUME_TOKEN = 3612,
    CONTEXT_TOKENS = 1024,
    VOCAB_SIZE = 100352,
};

typedef struct {
    const char *model;
    const char *tokens;
    const char *out;
} options;

typedef struct {
    const char *name;
    uint64_t bytes;
} expected_artifact;

static const expected_artifact EXPECTED_ARTIFACTS[] = {
    {"layer-01-ffn-norm.f32", 3072u * 4u},
    {"layer-01-router-logits.f32", 256u * 4u},
    {"layer-01-router-selected.i32", 10u * 4u},
    {"layer-01-router-weights.f32", 10u * 4u},
    {"layer-01-ffn-moe-input.q8_1", 3456u},
    {"layer-01-ffn-moe-gate.f32", 10u * 1024u * 4u},
    {"layer-01-ffn-moe-up.f32", 10u * 1024u * 4u},
    {"layer-01-ffn-moe-swiglu.f32", 10u * 1024u * 4u},
    {"layer-01-ffn-moe-col-l2.f32", 10u * 4u},
    {"layer-01-ffn-moe-down-input.f32", 10u * 1024u * 4u},
    {"layer-01-ffn-moe-down-input.q8_1", 10u * 1024u / 32u * 36u},
    {"layer-01-ffn-moe-down.f32", 10u * 3072u * 4u},
    {"layer-01-ffn-moe-weighted.f32", 10u * 3072u * 4u},
    {"layer-01-ffn-moe-out.f32", 3072u * 4u},
    {"layer-01-ffn-shared-out.f32", 3072u * 4u},
    {"layer-01-ffn-out.f32", 3072u * 4u},
    {"layer-01.f32", 3072u * 4u},
};

static void usage(FILE *stream, const char *program) {
    fprintf(stream,
            "usage: %s --model MODEL --tokens PREFIX-512.i32 --out DIR\n",
            program);
}

static int parse_options(int argc, char **argv, options *out) {
    memset(out, 0, sizeof(*out));
    for (int i = 1; i < argc; i++) {
        if (i + 1 >= argc) return 0;
        const char *flag = argv[i++];
        const char *value = argv[i];
        if (strcmp(flag, "--model") == 0 && !out->model) {
            out->model = value;
        } else if (strcmp(flag, "--tokens") == 0 && !out->tokens) {
            out->tokens = value;
        } else if (strcmp(flag, "--out") == 0 && !out->out) {
            out->out = value;
        } else {
            return 0;
        }
    }
    return out->model && out->tokens && out->out;
}

static int read_prefix(const char *path, ds4_tokens *tokens) {
    FILE *stream = fopen(path, "rb");
    if (!stream) {
        fprintf(stderr, "capture: open %s: %s\n", path, strerror(errno));
        return 0;
    }
    unsigned char bytes[PREFIX_TOKENS * sizeof(int32_t)];
    const size_t got = fread(bytes, 1, sizeof(bytes), stream);
    const int trailing = fgetc(stream);
    const int close_rc = fclose(stream);
    if (got != sizeof(bytes) || trailing != EOF || close_rc != 0) {
        fprintf(stderr, "capture: %s is not exactly %zu bytes\n",
                path, sizeof(bytes));
        return 0;
    }
    for (uint32_t i = 0; i < PREFIX_TOKENS; i++) {
        const unsigned char *p = bytes + (uint64_t)i * sizeof(int32_t);
        const uint32_t raw = (uint32_t)p[0] |
                             ((uint32_t)p[1] << 8u) |
                             ((uint32_t)p[2] << 16u) |
                             ((uint32_t)p[3] << 24u);
        int32_t token = 0;
        memcpy(&token, &raw, sizeof(token));
        if (token < 0 || token >= VOCAB_SIZE) {
            fprintf(stderr, "capture: token %u is out of range: %d\n",
                    i, token);
            return 0;
        }
        ds4_tokens_push(tokens, token);
    }
    return tokens->len == PREFIX_TOKENS;
}

static int create_synced_session(
        ds4_engine *engine,
        const ds4_tokens *prefix,
        ds4_session **out,
        const char *name) {
    char error[256] = {0};
    if (ds4_session_create(out, engine, CONTEXT_TOKENS) != 0) {
        fprintf(stderr, "capture: create %s session failed\n", name);
        return 0;
    }
    if (ds4_session_sync(*out, prefix, error, sizeof(error)) != 0) {
        fprintf(stderr, "capture: sync %s session: %s\n", name, error);
        ds4_session_free(*out);
        *out = NULL;
        return 0;
    }
    return 1;
}

static int eval_resume(ds4_session *session, const char *name) {
    char error[256] = {0};
    if (ds4_session_eval(session, RESUME_TOKEN, error, sizeof(error)) != 0) {
        fprintf(stderr, "capture: %s resume decode: %s\n", name, error);
        return 0;
    }
    if (ds4_session_pos(session) != PREFIX_TOKENS + 1) {
        fprintf(stderr, "capture: %s position=%d expected=%d\n",
                name, ds4_session_pos(session), PREFIX_TOKENS + 1);
        return 0;
    }
    return 1;
}

static int copy_logits(ds4_session *session, float *logits, const char *name) {
    if (ds4_session_copy_logits(session, logits, VOCAB_SIZE) != VOCAB_SIZE) {
        fprintf(stderr, "capture: copy %s logits failed\n", name);
        return 0;
    }
    return 1;
}

static int verify_artifacts(const char *directory) {
    char path[4096];
    for (size_t i = 0;
         i < sizeof(EXPECTED_ARTIFACTS) / sizeof(EXPECTED_ARTIFACTS[0]);
         i++) {
        const expected_artifact *artifact = &EXPECTED_ARTIFACTS[i];
        const int length = snprintf(
                path, sizeof(path), "%s/%s", directory, artifact->name);
        if (length < 0 || (size_t)length >= sizeof(path)) return 0;
        struct stat status;
        if (stat(path, &status) != 0 || !S_ISREG(status.st_mode) ||
            status.st_size < 0 || (uint64_t)status.st_size != artifact->bytes) {
            fprintf(stderr,
                    "capture: artifact %s missing or wrong size (expected %llu)\n",
                    path, (unsigned long long)artifact->bytes);
            return 0;
        }
    }
    return 1;
}

int main(int argc, char **argv) {
    options opt;
    if (!parse_options(argc, argv, &opt)) {
        usage(stderr, argv[0]);
        return 2;
    }
    if (getenv("DS4_LAGUNA_DIAG_DIR") ||
        getenv("DS4_LAGUNA_DIAG_LAYER")) {
        fprintf(stderr, "capture: diagnostic environment must start unset\n");
        return 2;
    }
    if (mkdir(opt.out, 0700) != 0) {
        fprintf(stderr, "capture: create output %s: %s\n",
                opt.out, strerror(errno));
        return 2;
    }

    int rc = 1;
    ds4_tokens prefix = {0};
    ds4_engine *engine = NULL;
    ds4_session *control = NULL;
    ds4_session *captured = NULL;
    float *control_logits = NULL;
    float *captured_logits = NULL;
    if (!read_prefix(opt.tokens, &prefix)) goto cleanup;

    const ds4_engine_options engine_options = {
        .model_path = opt.model,
        .backend = DS4_BACKEND_CUDA,
        .n_threads = 1,
    };
    if (ds4_engine_open(&engine, &engine_options) != 0) {
        fprintf(stderr, "capture: open Laguna CUDA engine failed\n");
        goto cleanup;
    }
    if (ds4_engine_vocab_size(engine) != VOCAB_SIZE) {
        fprintf(stderr, "capture: vocabulary mismatch\n");
        goto cleanup;
    }
    if (!create_synced_session(engine, &prefix, &control, "control") ||
        !create_synced_session(engine, &prefix, &captured, "captured")) {
        goto cleanup;
    }

    control_logits = malloc((size_t)VOCAB_SIZE * sizeof(float));
    captured_logits = malloc((size_t)VOCAB_SIZE * sizeof(float));
    if (!control_logits || !captured_logits) goto cleanup;
    if (!eval_resume(control, "control") ||
        !copy_logits(control, control_logits, "control")) {
        goto cleanup;
    }

    if (setenv("DS4_LAGUNA_DIAG_DIR", opt.out, 1) != 0 ||
        setenv("DS4_LAGUNA_DIAG_LAYER", "1", 1) != 0) {
        fprintf(stderr, "capture: enable diagnostics failed\n");
        goto cleanup;
    }
    const int captured_ok = eval_resume(captured, "captured");
    const int unset_dir = unsetenv("DS4_LAGUNA_DIAG_DIR");
    const int unset_layer = unsetenv("DS4_LAGUNA_DIAG_LAYER");
    if (!captured_ok || unset_dir != 0 || unset_layer != 0 ||
        !copy_logits(captured, captured_logits, "captured")) {
        goto cleanup;
    }
    if (memcmp(control_logits,
               captured_logits,
               (size_t)VOCAB_SIZE * sizeof(float)) != 0) {
        fprintf(stderr, "capture: diagnostics perturbed token-513 logits\n");
        goto cleanup;
    }
    if (!verify_artifacts(opt.out)) goto cleanup;

    fprintf(stderr,
            "probe_ds4_laguna_moe PASS token=513 layer=1 "
            "resume_token=%d nonperturbing=bit-exact\n",
            RESUME_TOKEN);
    rc = 0;

cleanup:
    unsetenv("DS4_LAGUNA_DIAG_DIR");
    unsetenv("DS4_LAGUNA_DIAG_LAYER");
    free(captured_logits);
    free(control_logits);
    ds4_session_free(captured);
    ds4_session_free(control);
    ds4_engine_close(engine);
    ds4_tokens_free(&prefix);
    return rc;
}
