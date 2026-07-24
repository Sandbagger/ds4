#include "ds4_gpu.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float yarn_ramp(float low, float high, int i) {
    const float y = ((float)(i / 2) - low) / fmaxf(0.001f, high - low);
    return 1.0f - fminf(1.0f, fmaxf(0.0f, y));
}

static void yarn_corr_dims(uint32_t n_rot, uint32_t n_ctx_orig,
                           float freq_base, float beta_fast, float beta_slow,
                           float dims[2]) {
    const float denom = 2.0f * logf(freq_base);
    dims[0] = floorf((float)n_rot * logf((float)n_ctx_orig /
                         (beta_fast * 2.0f * (float)M_PI)) / denom);
    dims[1] = ceilf((float)n_rot * logf((float)n_ctx_orig /
                        (beta_slow * 2.0f * (float)M_PI)) / denom);
    dims[0] = fmaxf(0.0f, dims[0]);
    dims[1] = fminf((float)(n_rot - 1u), dims[1]);
}

static void reference_head_rms_rope(float *out, const float *in,
                                    const float *weights, uint32_t n_tokens,
                                    uint32_t n_head, uint32_t head_dim,
                                    uint32_t n_rot, uint32_t pos0,
                                    uint32_t n_ctx_orig, float freq_base,
                                    float freq_scale, float ext_factor,
                                    float attn_factor, float beta_fast,
                                    float beta_slow, float eps) {
    float corr_dims[2] = {0.0f, 0.0f};
    if (ext_factor != 0.0f) {
        yarn_corr_dims(n_rot, n_ctx_orig, freq_base, beta_fast, beta_slow,
                       corr_dims);
    }
    memcpy(out, in, (size_t)n_tokens * n_head * head_dim * sizeof(*out));
    for (uint32_t token = 0; token < n_tokens; token++) {
        for (uint32_t h = 0; h < n_head; h++) {
            float *head = out + ((uint64_t)token * n_head + h) * head_dim;
            double ss = 0.0;
            for (uint32_t i = 0; i < head_dim; i++) {
                ss += (double)head[i] * head[i];
            }
            const float rms = 1.0f / sqrtf((float)(ss / head_dim) + eps);
            for (uint32_t i = 0; i < head_dim; i++) {
                head[i] *= rms * weights[i];
            }
            const uint32_t half_rot = n_rot / 2u;
            for (uint32_t i = 0; i < half_rot; i++) {
                const uint32_t rel_i0 = i * 2u;
                const float theta_extrap = (float)(pos0 + token) *
                    powf(freq_base, -((float)rel_i0) / (float)n_rot);
                const float theta_interp = freq_scale * theta_extrap;
                float theta = theta_interp;
                float mscale = attn_factor;
                if (ext_factor != 0.0f) {
                    const float mix = yarn_ramp(corr_dims[0], corr_dims[1],
                                                 (int)rel_i0) * ext_factor;
                    theta = theta_interp * (1.0f - mix) + theta_extrap * mix;
                    mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
                }
                const float c = cosf(theta);
                const float s = sinf(theta);
                const float x0 = head[i];
                const float x1 = head[i + half_rot];
                head[i] = (x0 * c - x1 * s) * mscale;
                head[i + half_rot] = (x0 * s + x1 * c) * mscale;
            }
        }
    }
}

typedef struct {
    const char *name;
    uint32_t n_tokens;
    uint32_t n_head;
    uint32_t n_rot;
    uint32_t pos0;
    uint32_t n_ctx_orig;
    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;
} laguna_norm_rope_case;

static int run_norm_rope_case(const float *weights,
                              const laguna_norm_rope_case *c) {
    const uint32_t head_dim = 128;
    const uint64_t count = (uint64_t)c->n_tokens * c->n_head * head_dim;
    float *input = (float *)malloc((size_t)count * sizeof(*input));
    float *reference = (float *)malloc((size_t)count * sizeof(*reference));
    float *actual = (float *)malloc((size_t)count * sizeof(*actual));
    ds4_gpu_tensor *x = NULL;
    int rc = 1;

    if (!input || !reference || !actual) goto cleanup;
    for (uint64_t i = 0; i < count; i++) {
        input[i] = ((float)((int)(i % 29) - 14)) / 13.0f;
    }
    reference_head_rms_rope(reference, input, weights, c->n_tokens,
                            c->n_head, head_dim, c->n_rot, c->pos0,
                            c->n_ctx_orig, c->freq_base, c->freq_scale,
                            c->ext_factor, c->attn_factor, c->beta_fast,
                            c->beta_slow, 1.0e-6f);
    for (uint64_t i = 0; i < count; i++) {
        if (!isfinite(reference[i])) {
            fprintf(stderr, "norm-rope: independent reference is non-finite\n");
            goto cleanup;
        }
    }
    x = ds4_gpu_tensor_alloc(count * sizeof(*input));
    if (!x || !ds4_gpu_tensor_write(x, 0, input, count * sizeof(*input))) {
        fprintf(stderr, "norm-rope: synthetic tensor setup failed\n");
        goto cleanup;
    }
    const int wrapper_ok = ds4_gpu_laguna_head_rms_norm_rope_tensor(
        x, weights, head_dim * sizeof(*weights), 0, c->n_tokens, c->n_head,
        head_dim, c->n_rot, c->pos0, c->n_ctx_orig, c->freq_base,
        c->freq_scale, c->ext_factor, c->attn_factor, c->beta_fast,
        c->beta_slow, 1.0e-6f);
    const cudaError_t sync = cudaDeviceSynchronize();
    if (sync != cudaSuccess) {
        fprintf(stderr, "norm-rope: cudaDeviceSynchronize: %s\n",
                cudaGetErrorString(sync));
        goto cleanup;
    }
    if (wrapper_ok) {
        if (!ds4_gpu_tensor_read(x, 0, actual, count * sizeof(*actual))) {
            fprintf(stderr, "norm-rope: output read failed\n");
            goto cleanup;
        }
        float max_abs = 0.0f;
        double square_error = 0.0;
        for (uint64_t i = 0; i < count; i++) {
            if (!isfinite(actual[i])) {
                fprintf(stderr, "norm-rope/%s: non-finite output at %llu\n",
                        c->name, (unsigned long long)i);
                goto cleanup;
            }
            const float error = fabsf(actual[i] - reference[i]);
            max_abs = fmaxf(max_abs, error);
            square_error += (double)error * error;
        }
        const double rms_error = sqrt(square_error / (double)count);
        if (max_abs > 2.0e-4f || rms_error > 5.0e-5) {
            fprintf(stderr, "norm-rope/%s: parity max_abs=%g rms=%g exceeds tolerance\n",
                    c->name, (double)max_abs, rms_error);
            goto cleanup;
        }
        rc = 0;
        goto cleanup;
    }
    fprintf(stderr, "norm-rope/%s: expected red failure at Laguna CUDA stub\n",
            c->name);

cleanup:
    ds4_gpu_tensor_free(x);
    free(actual);
    free(reference);
    free(input);
    return rc;
}

static void usage(const char *program) {
    fprintf(stderr, "usage: %s --case norm-rope|all\n", program);
}

int main(int argc, char **argv) {
    if (argc != 3 || strcmp(argv[1], "--case") != 0 ||
        (strcmp(argv[2], "norm-rope") != 0 && strcmp(argv[2], "all") != 0)) {
        usage(argv[0]);
        return 2;
    }
    if (!ds4_gpu_init()) {
        fprintf(stderr, "norm-rope: ds4_gpu_init failed\n");
        return 1;
    }
    const uint32_t head_dim = 128;
    float *weights = (float *)malloc(head_dim * sizeof(*weights));
    if (!weights) {
        fprintf(stderr, "norm-rope: synthetic weights allocation failed\n");
        ds4_gpu_cleanup();
        return 1;
    }
    for (uint32_t i = 0; i < head_dim; i++) {
        weights[i] = 0.70f + 0.01f * (float)(i % 17u);
    }
    if (!ds4_gpu_set_model_map(weights, head_dim * sizeof(*weights))) {
        fprintf(stderr, "norm-rope: synthetic model map setup failed\n");
        ds4_gpu_cleanup();
        free(weights);
        return 1;
    }
    static const laguna_norm_rope_case cases[] = {
        { "global-pos0", 1, 48, 64, 0, 8192, 500000.0f,
          1.0f / 32.0f, 1.0f, 1.0f, 32.0f, 1.0f },
        { "global-yarn-frontier", 4, 48, 64, 8191, 8192, 500000.0f,
          1.0f / 32.0f, 1.0f, 1.0f, 32.0f, 1.0f },
        { "swa-ring-frontier", 4, 72, 128, 510, 262144, 10000.0f,
          1.0f, 0.0f, 1.0f, 0.0f, 0.0f },
    };
    int rc = 0;
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        if (run_norm_rope_case(weights, &cases[i]) != 0) rc = 1;
    }
    /* The model-map registration pins weights until GPU cleanup unregisters it. */
    ds4_gpu_cleanup();
    free(weights);
    return rc;
}
