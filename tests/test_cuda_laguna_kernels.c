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
        uint64_t max_index = 0;
        for (uint64_t i = 0; i < count; i++) {
            if (!isfinite(actual[i])) {
                fprintf(stderr, "norm-rope/%s: non-finite output at %llu\n",
                        c->name, (unsigned long long)i);
                goto cleanup;
            }
            const float error = fabsf(actual[i] - reference[i]);
            if (error > max_abs) {
                max_abs = error;
                max_index = i;
            }
            square_error += (double)error * error;
        }
        const double rms_error = sqrt(square_error / (double)count);
        if (max_abs > 2.0e-4f || rms_error > 5.0e-5) {
            fprintf(stderr,
                    "norm-rope/%s: parity max_abs=%g rms=%g index=%llu actual=%g reference=%g exceeds tolerance\n",
                    c->name, (double)max_abs, rms_error,
                    (unsigned long long)max_index, (double)actual[max_index],
                    (double)reference[max_index]);
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

static int run_qk_norm_rope_case(const float *q_weights,
                                 const float *k_weights) {
    const uint32_t n_tokens = 4;
    const uint32_t n_q_head = 48;
    const uint32_t n_k_head = 8;
    const uint32_t head_dim = 128;
    const uint32_t n_rot = 64;
    const uint32_t pos0 = 8191;
    const uint64_t q_count = (uint64_t)n_tokens * n_q_head * head_dim;
    const uint64_t k_count = (uint64_t)n_tokens * n_k_head * head_dim;
    const uint64_t weight_bytes = head_dim * sizeof(*q_weights);
    float *q_input = (float *)malloc((size_t)q_count * sizeof(*q_input));
    float *k_input = (float *)malloc((size_t)k_count * sizeof(*k_input));
    float *q_reference = (float *)malloc((size_t)q_count * sizeof(*q_reference));
    float *k_reference = (float *)malloc((size_t)k_count * sizeof(*k_reference));
    float *q_actual = (float *)malloc((size_t)q_count * sizeof(*q_actual));
    float *k_actual = (float *)malloc((size_t)k_count * sizeof(*k_actual));
    ds4_gpu_tensor *q = NULL;
    ds4_gpu_tensor *k = NULL;
    int rc = 1;

    if (!q_input || !k_input || !q_reference || !k_reference || !q_actual ||
        !k_actual) goto cleanup;
    for (uint64_t i = 0; i < q_count; i++) {
        q_input[i] = ((float)((int)((i * 13u) % 31u) - 15)) / 17.0f;
    }
    for (uint64_t i = 0; i < k_count; i++) {
        k_input[i] = ((float)((int)((i * 19u) % 37u) - 18)) / 19.0f;
    }
    reference_head_rms_rope(q_reference, q_input, q_weights, n_tokens,
                            n_q_head, head_dim, n_rot, pos0, 8192,
                            500000.0f, 1.0f / 32.0f, 1.0f, 1.0f, 32.0f,
                            1.0f, 1.0e-6f);
    reference_head_rms_rope(k_reference, k_input, k_weights, n_tokens,
                            n_k_head, head_dim, n_rot, pos0, 8192,
                            500000.0f, 1.0f / 32.0f, 1.0f, 1.0f, 32.0f,
                            1.0f, 1.0e-6f);
    q = ds4_gpu_tensor_alloc(q_count * sizeof(*q_input));
    k = ds4_gpu_tensor_alloc(k_count * sizeof(*k_input));
    if (!q || !k ||
        !ds4_gpu_tensor_write(q, 0, q_input, q_count * sizeof(*q_input)) ||
        !ds4_gpu_tensor_write(k, 0, k_input, k_count * sizeof(*k_input))) {
        fprintf(stderr, "qk-norm-rope: synthetic tensor setup failed\n");
        goto cleanup;
    }
    const int wrapper_ok = ds4_gpu_laguna_qk_head_rms_norm_rope_tensor(
        q, k, q_weights, 2u * weight_bytes, 0, weight_bytes, n_tokens,
        n_q_head, n_k_head, head_dim, n_rot, pos0, 8192, 500000.0f,
        1.0f / 32.0f, 1.0f, 1.0f, 32.0f, 1.0f, 1.0e-6f);
    const cudaError_t sync = cudaDeviceSynchronize();
    if (sync != cudaSuccess) {
        fprintf(stderr, "qk-norm-rope: cudaDeviceSynchronize: %s\n",
                cudaGetErrorString(sync));
        goto cleanup;
    }
    if (!wrapper_ok) {
        fprintf(stderr, "qk-norm-rope: expected red failure at Laguna CUDA stub\n");
        goto cleanup;
    }
    if (!ds4_gpu_tensor_read(q, 0, q_actual, q_count * sizeof(*q_actual)) ||
        !ds4_gpu_tensor_read(k, 0, k_actual, k_count * sizeof(*k_actual))) {
        fprintf(stderr, "qk-norm-rope: output read failed\n");
        goto cleanup;
    }
    double square_error = 0.0;
    float max_abs = 0.0f;
    uint64_t max_index = 0;
    for (uint64_t i = 0; i < q_count + k_count; i++) {
        const float actual = i < q_count ? q_actual[i] : k_actual[i - q_count];
        const float reference = i < q_count ? q_reference[i] :
            k_reference[i - q_count];
        if (!isfinite(actual)) {
            fprintf(stderr, "qk-norm-rope: non-finite output at %llu\n",
                    (unsigned long long)i);
            goto cleanup;
        }
        const float error = fabsf(actual - reference);
        if (error > max_abs) {
            max_abs = error;
            max_index = i;
        }
        square_error += (double)error * error;
    }
    const double rms_error = sqrt(square_error / (double)(q_count + k_count));
    if (max_abs > 2.0e-4f || rms_error > 5.0e-5) {
        const float actual = max_index < q_count ? q_actual[max_index] :
            k_actual[max_index - q_count];
        const float reference = max_index < q_count ? q_reference[max_index] :
            k_reference[max_index - q_count];
        fprintf(stderr,
                "qk-norm-rope: parity max_abs=%g rms=%g index=%llu actual=%g reference=%g exceeds tolerance\n",
                (double)max_abs, rms_error, (unsigned long long)max_index,
                (double)actual, (double)reference);
        goto cleanup;
    }
    rc = 0;

cleanup:
    ds4_gpu_tensor_free(k);
    ds4_gpu_tensor_free(q);
    free(k_actual);
    free(q_actual);
    free(k_reference);
    free(q_reference);
    free(k_input);
    free(q_input);
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
    float *weights = (float *)malloc(2u * head_dim * sizeof(*weights));
    if (!weights) {
        fprintf(stderr, "norm-rope: synthetic weights allocation failed\n");
        ds4_gpu_cleanup();
        return 1;
    }
    for (uint32_t i = 0; i < head_dim; i++) {
        weights[i] = 0.70f + 0.01f * (float)(i % 17u);
        weights[head_dim + i] = 0.55f + 0.01f * (float)((i * 3u) % 23u);
    }
    if (!ds4_gpu_set_model_map(weights, 2u * head_dim * sizeof(*weights))) {
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
    if (run_qk_norm_rope_case(weights, weights + head_dim) != 0) rc = 1;
    /* The model-map registration pins weights until GPU cleanup unregisters it. */
    ds4_gpu_cleanup();
    free(weights);
    return rc;
}
