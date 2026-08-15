/* Direct DS4 512+1 decode capture for a selected Laguna routed-MoE layer.
 *
 * The release-control build emits token-513 logits using only release DS4 and
 * CUDA objects.  The hook build runs the same decode twice from the same
 * 512-token prefix, first with the observer disabled and then enabled, and
 * requires all three logit vectors to be bitwise identical.
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
    LAGUNA_LAYERS = 48,
};

typedef struct {
    const char *model;
    const char *tokens;
    const char *out;
    const char *release_logits;
    int detail_layer;
} options;

#define LOGITS_BYTES ((size_t)VOCAB_SIZE * sizeof(float))

#ifndef DS4_LAGUNA_RELEASE_CONTROL
typedef struct {
    const char *name;
    uint64_t bytes;
    uint64_t bytes_per_head;
    int routed_only;
} expected_artifact;

static const expected_artifact EXPECTED_ARTIFACTS[] = {
    {"layer-%02d-attn-norm.f32", 3072u * 4u, 0, 0},
    {"layer-%02d-q-proj.f32", 0, 128u * 4u, 0},
    {"layer-%02d-k-proj.f32", 1024u * 4u, 0, 0},
    {"layer-%02d-v-proj.f32", 1024u * 4u, 0, 0},
    {"layer-%02d-gate-proj.f32", 0, 4u, 0},
    {"layer-%02d-q-rope.f32", 0, 128u * 4u, 0},
    {"layer-%02d-k-rope.f32", 1024u * 4u, 0, 0},
    {"layer-%02d-attn-gated.f32", 0, 128u * 4u, 0},
    {"layer-%02d-attn-o-proj.f32", 3072u * 4u, 0, 0},
    {"layer-%02d-ffn-inp.f32", 3072u * 4u, 0, 0},
    {"layer-%02d-ffn-norm.f32", 3072u * 4u, 0, 0},
    {"layer-%02d-router-logits.f32", 256u * 4u, 0, 1},
    {"layer-%02d-router-selected.i32", 10u * 4u, 0, 1},
    {"layer-%02d-router-weights.f32", 10u * 4u, 0, 1},
    {"layer-%02d-ffn-moe-input.q8_1", 3456u, 0, 1},
    {"layer-%02d-ffn-moe-gate.f32", 10u * 1024u * 4u, 0, 1},
    {"layer-%02d-ffn-moe-up.f32", 10u * 1024u * 4u, 0, 1},
    {"layer-%02d-ffn-moe-swiglu.f32", 10u * 1024u * 4u, 0, 1},
    {"layer-%02d-ffn-moe-col-l2.f32", 10u * 4u, 0, 1},
    {"layer-%02d-ffn-moe-down-input.f32", 10u * 1024u * 4u, 0, 1},
    {"layer-%02d-ffn-moe-down-input.q8_1", 10u * 1024u / 32u * 36u, 0, 1},
    {"layer-%02d-ffn-moe-down.f32", 10u * 3072u * 4u, 0, 1},
    {"layer-%02d-ffn-moe-weighted.f32", 10u * 3072u * 4u, 0, 1},
    {"layer-%02d-ffn-moe-out.f32", 3072u * 4u, 0, 1},
    {"layer-%02d-ffn-shared-out.f32", 3072u * 4u, 0, 1},
    {"layer-%02d-ffn-out.f32", 3072u * 4u, 0, 0},
    {"layer-%02d.f32", 3072u * 4u, 0, 0},
};

static uint32_t detail_head_count(int detail_layer) {
    return detail_layer % 4 == 0 ? 48u : 72u;
}

static uint64_t artifact_bytes(
        const expected_artifact *artifact,
        int detail_layer) {
    return artifact->bytes +
           artifact->bytes_per_head * detail_head_count(detail_layer);
}
#endif

static void usage(FILE *stream, const char *program) {
#ifdef DS4_LAGUNA_RELEASE_CONTROL
    fprintf(stream,
            "usage: %s --model MODEL --tokens PREFIX-512.i32 "
            "--release-logits FILE [--detail-layer 0..47]\n",
            program);
#else
    fprintf(stream,
            "usage: %s --model MODEL --tokens PREFIX-512.i32 --out DIR "
            "--release-logits FILE [--detail-layer 0..47]\n",
            program);
#endif
}

static int parse_detail_layer(const char *value, int *out) {
    errno = 0;
    char *end = NULL;
    const long parsed = strtol(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' ||
        parsed < 0 || parsed >= LAGUNA_LAYERS) {
        return 0;
    }
    *out = (int)parsed;
    return 1;
}

static int parse_options(int argc, char **argv, options *out) {
    memset(out, 0, sizeof(*out));
    out->detail_layer = 1;
    int have_detail_layer = 0;
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
        } else if (strcmp(flag, "--release-logits") == 0 &&
                   !out->release_logits) {
            out->release_logits = value;
        } else if (strcmp(flag, "--detail-layer") == 0 &&
                   !have_detail_layer &&
                   parse_detail_layer(value, &out->detail_layer)) {
            have_detail_layer = 1;
        } else {
            return 0;
        }
    }
#ifdef DS4_LAGUNA_RELEASE_CONTROL
    return out->model && out->tokens && out->release_logits && !out->out;
#else
    return out->model && out->tokens && out->out && out->release_logits;
#endif
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

#ifdef DS4_LAGUNA_RELEASE_CONTROL
static int write_release_logits(const char *path, const float *logits) {
    FILE *stream = fopen(path, "wbx");
    if (!stream) {
        fprintf(stderr, "capture: create release logits %s: %s\n",
                path, strerror(errno));
        return 0;
    }
    const size_t written = fwrite(logits, 1, LOGITS_BYTES, stream);
    const int close_rc = fclose(stream);
    if (written != LOGITS_BYTES || close_rc != 0) {
        fprintf(stderr, "capture: write release logits %s failed\n", path);
        return 0;
    }
    return 1;
}
#endif

#ifndef DS4_LAGUNA_RELEASE_CONTROL
static int read_release_logits(const char *path, float *logits) {
    FILE *stream = fopen(path, "rb");
    if (!stream) {
        fprintf(stderr, "capture: open release logits %s: %s\n",
                path, strerror(errno));
        return 0;
    }
    const size_t got = fread(logits, 1, LOGITS_BYTES, stream);
    const int trailing = fgetc(stream);
    const int close_rc = fclose(stream);
    if (got != LOGITS_BYTES || trailing != EOF || close_rc != 0) {
        fprintf(stderr,
                "capture: %s is not exactly %zu bytes of release logits\n",
                path, LOGITS_BYTES);
        return 0;
    }
    return 1;
}
#endif

#ifndef DS4_LAGUNA_RELEASE_CONTROL
static int format_artifact_name(
        char *out,
        size_t out_size,
        const expected_artifact *artifact,
        int detail_layer) {
    const int length = snprintf(
            out, out_size, artifact->name, detail_layer);
    return length >= 0 && (size_t)length < out_size;
}

static int verify_artifacts(const char *directory, int detail_layer) {
    char path[4096];
    char name[128];
    for (size_t i = 0;
         i < sizeof(EXPECTED_ARTIFACTS) / sizeof(EXPECTED_ARTIFACTS[0]);
         i++) {
        const expected_artifact *artifact = &EXPECTED_ARTIFACTS[i];
        if (detail_layer == 0 && artifact->routed_only) continue;
        if (!format_artifact_name(
                name, sizeof(name), artifact, detail_layer)) return 0;
        const uint64_t expected_bytes = artifact_bytes(artifact, detail_layer);
        const int length = snprintf(
                path, sizeof(path), "%s/%s", directory, name);
        if (length < 0 || (size_t)length >= sizeof(path)) return 0;
        struct stat status;
        if (stat(path, &status) != 0 || !S_ISREG(status.st_mode) ||
            status.st_size < 0 || (uint64_t)status.st_size != expected_bytes) {
            fprintf(stderr,
                    "capture: artifact %s missing or wrong size (expected %llu)\n",
                    path, (unsigned long long)expected_bytes);
            return 0;
        }
    }
    return 1;
}
#endif

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
#ifndef DS4_LAGUNA_RELEASE_CONTROL
    if (mkdir(opt.out, 0700) != 0) {
        fprintf(stderr, "capture: create output %s: %s\n",
                opt.out, strerror(errno));
        return 2;
    }
#endif

    int rc = 1;
    ds4_tokens prefix = {0};
    ds4_engine *engine = NULL;
    ds4_session *control = NULL;
    ds4_session *captured = NULL;
    float *release_logits = NULL;
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
    if (!create_synced_session(engine, &prefix, &control,
#ifdef DS4_LAGUNA_RELEASE_CONTROL
                               "release")) {
#else
                               "hook-null") ||
        !create_synced_session(engine, &prefix, &captured, "hook-active")) {
#endif
        goto cleanup;
    }

    control_logits = malloc((size_t)VOCAB_SIZE * sizeof(float));
#ifndef DS4_LAGUNA_RELEASE_CONTROL
    release_logits = malloc((size_t)VOCAB_SIZE * sizeof(float));
    captured_logits = malloc((size_t)VOCAB_SIZE * sizeof(float));
    if (!release_logits || !control_logits || !captured_logits) goto cleanup;
    if (!read_release_logits(opt.release_logits, release_logits)) goto cleanup;
#else
    if (!control_logits) goto cleanup;
#endif
    if (!eval_resume(control,
#ifdef DS4_LAGUNA_RELEASE_CONTROL
                     "release") ||
        !copy_logits(control, control_logits, "release")) {
#else
                     "hook-null") ||
        !copy_logits(control, control_logits, "hook-null")) {
#endif
        goto cleanup;
    }

#ifdef DS4_LAGUNA_RELEASE_CONTROL
    if (!write_release_logits(opt.release_logits, control_logits)) goto cleanup;
    fprintf(stderr,
            "probe_ds4_laguna_moe PASS token=513 mode=release "
            "resume_token=%d logits_bytes=%zu\n",
            RESUME_TOKEN, LOGITS_BYTES);
    rc = 0;
#else
    if (memcmp(release_logits, control_logits, LOGITS_BYTES) != 0) {
        fprintf(stderr,
                "capture: release and hook-null token-513 logits differ\n");
        goto cleanup;
    }
    char detail_layer[16];
    const int detail_layer_length = snprintf(
            detail_layer, sizeof(detail_layer), "%d", opt.detail_layer);
    if (detail_layer_length < 0 ||
        (size_t)detail_layer_length >= sizeof(detail_layer) ||
        setenv("DS4_LAGUNA_DIAG_DIR", opt.out, 1) != 0 ||
        setenv("DS4_LAGUNA_DIAG_LAYER", detail_layer, 1) != 0) {
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
    if (memcmp(control_logits, captured_logits, LOGITS_BYTES) != 0) {
        fprintf(stderr, "capture: diagnostics perturbed token-513 logits\n");
        goto cleanup;
    }
    if (!verify_artifacts(opt.out, opt.detail_layer)) goto cleanup;

    fprintf(stderr,
            "probe_ds4_laguna_moe PASS token=513 layer=%d "
            "resume_token=%d release_vs_hook_null=bit-exact "
            "hook_null_vs_hook_active=bit-exact\n",
            opt.detail_layer, RESUME_TOKEN);
    rc = 0;
#endif

cleanup:
    unsetenv("DS4_LAGUNA_DIAG_DIR");
    unsetenv("DS4_LAGUNA_DIAG_LAYER");
    free(captured_logits);
    free(control_logits);
    free(release_logits);
    ds4_session_free(captured);
    ds4_session_free(control);
    ds4_engine_close(engine);
    ds4_tokens_free(&prefix);
    return rc;
}
