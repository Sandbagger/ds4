#include "ds4_gpu.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* This is deliberately independent of CUDA's half helpers: the attention
 * reference owns the exact FP32-to-FP16 cache contract that decode reads. */
static uint16_t reference_f32_bits_to_f16(uint32_t bits) {
    const uint32_t sign = (bits >> 16) & 0x8000u;
    uint32_t exponent = (bits >> 23) & 0xffu;
    uint32_t mantissa = bits & 0x7fffffu;
    if (exponent == 0xffu) return (uint16_t)(sign | (mantissa ? 0x7e00u : 0x7c00u));
    if (exponent <= 112u) {
        if (exponent < 102u) return (uint16_t)sign;
        mantissa |= 0x800000u;
        const uint32_t shift = 126u - exponent;
        uint32_t rounded = mantissa >> shift;
        const uint32_t remainder = mantissa & ((1u << shift) - 1u);
        const uint32_t halfway = 1u << (shift - 1u);
        if (remainder > halfway || (remainder == halfway && (rounded & 1u))) {
            rounded++;
        }
        return (uint16_t)(sign | rounded);
    }
    exponent -= 112u;
    mantissa += 0x0fffu + ((mantissa >> 13) & 1u);
    if (mantissa & 0x00800000u) {
        mantissa = 0;
        exponent++;
    }
    if (exponent >= 31u) return (uint16_t)(sign | 0x7c00u);
    return (uint16_t)(sign | (exponent << 10) | (mantissa >> 13));
}

static uint16_t reference_f32_to_f16(float value) {
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    return reference_f32_bits_to_f16(bits);
}

static int run_f32_to_f16_reference_cases(void) {
    const struct {
        const char *name;
        uint32_t input_bits;
        uint16_t expected;
    } cases[] = {
        { "+zero", 0x00000000u, 0x0000u },
        { "-zero", 0x80000000u, 0x8000u },
        { "halfway-subnormal", 0x33000000u, 0x0000u },
        { "above-halfway-subnormal", 0x33000001u, 0x0001u },
        { "minimum-subnormal", 0x33800000u, 0x0001u },
        { "minimum-normal", 0x38800000u, 0x0400u },
        { "maximum-finite", 0x477fe000u, 0x7bffu },
        { "+infinity", 0x7f800000u, 0x7c00u },
        { "-infinity", 0xff800000u, 0xfc00u },
        { "quiet-nan", 0x7fc00000u, 0x7e00u },
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        const uint16_t actual = reference_f32_bits_to_f16(cases[i].input_bits);
        if (actual != cases[i].expected) {
            fprintf(stderr,
                    "f32-to-f16/%s: got 0x%04x expected 0x%04x\n",
                    cases[i].name, actual, cases[i].expected);
            return 1;
        }
    }
    return 0;
}

static float reference_f16_to_f32(uint16_t value) {
    const uint32_t sign = (uint32_t)(value & 0x8000u) << 16;
    uint32_t exponent = (value >> 10) & 0x1fu;
    uint32_t mantissa = value & 0x03ffu;
    uint32_t bits;
    if (exponent == 0u) {
        if (mantissa == 0u) {
            bits = sign;
        } else {
            exponent = 113u;
            while ((mantissa & 0x0400u) == 0u) {
                mantissa <<= 1u;
                exponent--;
            }
            bits = sign | (exponent << 23) | ((mantissa & 0x03ffu) << 13);
        }
    } else if (exponent == 31u) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
    }
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

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
    uint64_t weight_offset;
} laguna_norm_rope_case;

typedef struct {
    const char *name;
    const float *actual;
    const float *reference;
    uint64_t count;
} laguna_parity_span;

typedef struct {
    const char *name;
    const ds4_gpu_tensor *tensor;
    uint64_t bytes;
    void *snapshot;
} laguna_tensor_snapshot;

static int laguna_capture_tensor_snapshot(laguna_tensor_snapshot *snapshot) {
    snapshot->snapshot = malloc((size_t)snapshot->bytes);
    return snapshot->snapshot != NULL &&
        ds4_gpu_tensor_read(snapshot->tensor, 0, snapshot->snapshot,
                            snapshot->bytes);
}

static int laguna_tensor_matches_snapshot(const laguna_tensor_snapshot *snapshot,
                                          const char *case_name) {
    void *actual = malloc((size_t)snapshot->bytes);
    const int matches = actual != NULL &&
        ds4_gpu_tensor_read(snapshot->tensor, 0, actual, snapshot->bytes) &&
        memcmp(actual, snapshot->snapshot, (size_t)snapshot->bytes) == 0;
    if (!matches) {
        fprintf(stderr, "%s: rejected call changed %s\n", case_name,
                snapshot->name);
    }
    free(actual);
    return matches;
}

static int laguna_parity_spans_within_limits(
        const char *case_name, const laguna_parity_span *spans,
        size_t n_spans, int report) {
    for (size_t span = 0; span < n_spans; span++) {
        double square_error = 0.0;
        float max_abs = 0.0f;
        for (uint64_t i = 0; i < spans[span].count; i++) {
            if (!isfinite(spans[span].actual[i])) {
                if (report) {
                    fprintf(stderr, "%s/%s: non-finite output\n", case_name,
                            spans[span].name);
                }
                return 0;
            }
            const float error = fabsf(spans[span].actual[i] -
                                      spans[span].reference[i]);
            if (error > max_abs) max_abs = error;
            square_error += (double)error * error;
        }
        const double rms_error = sqrt(square_error / (double)spans[span].count);
        if (max_abs > 2.0e-4f || rms_error > 5.0e-5) {
            if (report) {
                fprintf(stderr,
                        "%s/%s: parity max_abs=%g rms=%g exceeds tolerance\n",
                        case_name, spans[span].name, (double)max_abs, rms_error);
            }
            return 0;
        }
    }
    return 1;
}

static int run_qk_metric_dilution_case(void) {
    static const float q_actual[6] = { 0 };
    static const float q_reference[6] = { 0 };
    static const float k_actual[1] = { 1.0e-4f };
    static const float k_reference[1] = { 0 };
    static const laguna_parity_span spans[] = {
        { "q", q_actual, q_reference, 6 },
        { "k", k_actual, k_reference, 1 },
    };
    if (laguna_parity_spans_within_limits("qk-metric-dilution", spans,
                                          sizeof(spans) / sizeof(spans[0]), 0)) {
        fprintf(stderr,
                "qk-metric-dilution: accepted a K-only RMS error above the limit\n");
        return 1;
    }
    return 0;
}

static int run_norm_rope_case(const float *model_map, uint64_t model_size,
                              const float *weights,
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
        x, model_map, model_size, c->weight_offset, c->n_tokens, c->n_head,
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
                const uint64_t row = i / head_dim;
                fprintf(stderr,
                        "norm-rope/%s: non-finite output at token=%llu head=%llu dim=%llu\n",
                        c->name, (unsigned long long)(row / c->n_head),
                        (unsigned long long)(row % c->n_head),
                        (unsigned long long)(i % head_dim));
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
            const uint64_t row = max_index / head_dim;
            fprintf(stderr,
                    "norm-rope/%s: parity max_abs=%g rms=%g token=%llu head=%llu dim=%llu actual=%g reference=%g exceeds tolerance\n",
                    c->name, (double)max_abs, rms_error,
                    (unsigned long long)(row / c->n_head),
                    (unsigned long long)(row % c->n_head),
                    (unsigned long long)(max_index % head_dim),
                    (double)actual[max_index],
                    (double)reference[max_index]);
            goto cleanup;
        }
        rc = 0;
        goto cleanup;
    }
    fprintf(stderr, "norm-rope/%s: CUDA wrapper returned failure\n", c->name);

cleanup:
    ds4_gpu_tensor_free(x);
    free(actual);
    free(reference);
    free(input);
    return rc;
}

static int run_qk_norm_rope_case(const float *q_weights, const float *k_weights,
                                 const laguna_norm_rope_case *c) {
    const uint32_t n_tokens = c->n_tokens;
    const uint32_t n_q_head = c->n_head;
    const uint32_t n_k_head = 8;
    const uint32_t head_dim = 128;
    const uint32_t n_rot = c->n_rot;
    const uint32_t pos0 = c->pos0;
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
                            n_q_head, head_dim, n_rot, pos0, c->n_ctx_orig,
                            c->freq_base, c->freq_scale, c->ext_factor,
                            c->attn_factor, c->beta_fast, c->beta_slow,
                            1.0e-6f);
    reference_head_rms_rope(k_reference, k_input, k_weights, n_tokens,
                            n_k_head, head_dim, n_rot, pos0, c->n_ctx_orig,
                            c->freq_base, c->freq_scale, c->ext_factor,
                            c->attn_factor, c->beta_fast, c->beta_slow,
                            1.0e-6f);
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
        n_q_head, n_k_head, head_dim, n_rot, pos0, c->n_ctx_orig,
        c->freq_base, c->freq_scale, c->ext_factor, c->attn_factor,
        c->beta_fast, c->beta_slow, 1.0e-6f);
    const cudaError_t sync = cudaDeviceSynchronize();
    if (sync != cudaSuccess) {
        fprintf(stderr, "qk-norm-rope: cudaDeviceSynchronize: %s\n",
                cudaGetErrorString(sync));
        goto cleanup;
    }
    if (!wrapper_ok) {
        fprintf(stderr, "qk-norm-rope/%s: CUDA wrapper returned failure\n",
                c->name);
        goto cleanup;
    }
    if (!ds4_gpu_tensor_read(q, 0, q_actual, q_count * sizeof(*q_actual)) ||
        !ds4_gpu_tensor_read(k, 0, k_actual, k_count * sizeof(*k_actual))) {
        fprintf(stderr, "qk-norm-rope: output read failed\n");
        goto cleanup;
    }
    const laguna_parity_span spans[] = {
        { "q", q_actual, q_reference, q_count },
        { "k", k_actual, k_reference, k_count },
    };
    if (!laguna_parity_spans_within_limits(c->name, spans,
                                           sizeof(spans) / sizeof(spans[0]), 1)) {
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

typedef struct {
    const char *family;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t cache_cap;
    uint32_t pos;
    uint32_t key_start;
    uint32_t key_count;
    float gate;
    bool mixed_gates;
} laguna_decode_attention_case;

static float reference_softplus(float value) {
    return value > 0.0f ? value + log1pf(expf(-value)) : log1pf(expf(value));
}

static int run_decode_attention_case(const laguna_decode_attention_case *c) {
    const uint32_t head_dim = 128u;
    const uint64_t q_count = (uint64_t)c->n_head * head_dim;
    const uint64_t kv_width = (uint64_t)c->n_head_kv * head_dim;
    const uint64_t cache_values = (uint64_t)c->cache_cap * kv_width;
    const float scale = 1.0f / sqrtf((float)head_dim);
    float *q_host = NULL;
    float *k_host = NULL;
    float *v_host = NULL;
    float *gate_host = NULL;
    float *scores = NULL;
    float *reference = NULL;
    float *actual = NULL;
    uint16_t *key_expected = NULL;
    uint16_t *value_expected = NULL;
    uint16_t *key_actual = NULL;
    uint16_t *value_actual = NULL;
    ds4_gpu_tensor *heads = NULL;
    ds4_gpu_tensor *key_cache = NULL;
    ds4_gpu_tensor *value_cache = NULL;
    ds4_gpu_tensor *q = NULL;
    ds4_gpu_tensor *k = NULL;
    ds4_gpu_tensor *v = NULL;
    ds4_gpu_tensor *gate = NULL;
    laguna_tensor_snapshot snapshots[7] = {0};
    int rc = 1;

    q_host = (float *)malloc((size_t)q_count * sizeof(*q_host));
    k_host = (float *)malloc((size_t)kv_width * sizeof(*k_host));
    v_host = (float *)malloc((size_t)kv_width * sizeof(*v_host));
    gate_host = (float *)malloc((size_t)c->n_head * sizeof(*gate_host));
    scores = (float *)malloc((size_t)c->key_count * sizeof(*scores));
    reference = (float *)malloc((size_t)q_count * sizeof(*reference));
    actual = (float *)malloc((size_t)q_count * sizeof(*actual));
    key_expected = (uint16_t *)malloc((size_t)cache_values * sizeof(*key_expected));
    value_expected = (uint16_t *)malloc((size_t)cache_values * sizeof(*value_expected));
    key_actual = (uint16_t *)malloc((size_t)cache_values * sizeof(*key_actual));
    value_actual = (uint16_t *)malloc((size_t)cache_values * sizeof(*value_actual));
    if (!q_host || !k_host || !v_host || !gate_host || !scores || !reference || !actual ||
        !key_expected || !value_expected || !key_actual || !value_actual) {
        fprintf(stderr, "decode-attention: host allocation failed\n");
        goto cleanup;
    }

    for (uint64_t i = 0; i < q_count; i++) {
        q_host[i] = (float)((int)((i * 17u) % 41u) - 20) / 23.0f;
    }
    for (uint64_t i = 0; i < kv_width; i++) {
        k_host[i] = (float)((int)((i * 11u) % 37u) - 18) / 29.0f;
        v_host[i] = (float)((int)((i * 23u) % 43u) - 21) / 31.0f;
    }
    for (uint64_t i = 0; i < cache_values; i++) {
        const float key_value = (float)((int)((i * 7u) % 47u) - 23) / 37.0f;
        const float value_value = (float)((int)((i * 29u) % 53u) - 26) / 41.0f;
        key_expected[i] = reference_f32_to_f16(key_value);
        value_expected[i] = reference_f32_to_f16(value_value);
    }
    memcpy(key_actual, key_expected, (size_t)cache_values * sizeof(*key_actual));
    memcpy(value_actual, value_expected,
           (size_t)cache_values * sizeof(*value_actual));
    const uint64_t current_base = (uint64_t)(c->pos % c->cache_cap) * kv_width;
    for (uint64_t i = 0; i < kv_width; i++) {
        key_expected[current_base + i] = reference_f32_to_f16(k_host[i]);
        value_expected[current_base + i] = reference_f32_to_f16(v_host[i]);
    }
    static const float mixed_gates[] = { -20.0f, -2.0f, 0.0f, 2.0f, 20.0f };
    for (uint32_t h = 0; h < c->n_head; h++) {
        gate_host[h] = c->mixed_gates ? mixed_gates[h % 5u] : c->gate;
    }

    const uint32_t heads_per_kv = c->n_head / c->n_head_kv;
    for (uint32_t h = 0; h < c->n_head; h++) {
        const uint32_t kv_head = h / heads_per_kv;
        float max_score = -INFINITY;
        for (uint32_t r = 0; r < c->key_count; r++) {
            const uint32_t row = (c->key_start + r) % c->cache_cap;
            const uint64_t base = (uint64_t)row * kv_width +
                (uint64_t)kv_head * head_dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) {
                dot += q_host[(uint64_t)h * head_dim + d] *
                    reference_f16_to_f32(key_expected[base + d]);
            }
            scores[r] = scale * dot;
            if (scores[r] > max_score) max_score = scores[r];
        }
        float sum = 0.0f;
        for (uint32_t r = 0; r < c->key_count; r++) {
            const uint32_t row = (c->key_start + r) % c->cache_cap;
            const uint64_t base = (uint64_t)row * kv_width +
                (uint64_t)kv_head * head_dim;
            (void)base;
            sum += expf(scores[r] - max_score);
        }
        const float gate_scale = reference_softplus(gate_host[h]);
        for (uint32_t d = 0; d < head_dim; d++) {
            float weighted_value = 0.0f;
            for (uint32_t r = 0; r < c->key_count; r++) {
                const uint32_t row = (c->key_start + r) % c->cache_cap;
                const uint64_t base = (uint64_t)row * kv_width +
                    (uint64_t)kv_head * head_dim;
                weighted_value += expf(scores[r] - max_score) *
                    reference_f16_to_f32(value_expected[base + d]);
            }
            reference[(uint64_t)h * head_dim + d] =
                weighted_value / sum * gate_scale;
        }
    }
    for (uint64_t i = 0; i < q_count; i++) {
        if (!isfinite(reference[i])) {
            fprintf(stderr, "decode-attention/%s: independent reference is non-finite\n",
                    c->family);
            goto cleanup;
        }
    }

    heads = ds4_gpu_tensor_alloc(q_count * sizeof(*actual));
    key_cache = ds4_gpu_tensor_alloc(cache_values * sizeof(*key_expected));
    value_cache = ds4_gpu_tensor_alloc(cache_values * sizeof(*value_expected));
    q = ds4_gpu_tensor_alloc(q_count * sizeof(*q_host));
    k = ds4_gpu_tensor_alloc(kv_width * sizeof(*k_host));
    v = ds4_gpu_tensor_alloc(kv_width * sizeof(*v_host));
    gate = ds4_gpu_tensor_alloc((uint64_t)c->n_head * sizeof(*gate_host));
    memset(actual, 0xa5, (size_t)q_count * sizeof(*actual));
    if (!heads || !key_cache || !value_cache || !q || !k || !v || !gate ||
        !ds4_gpu_tensor_write(heads, 0, actual, q_count * sizeof(*actual)) ||
        !ds4_gpu_tensor_write(key_cache, 0, key_actual,
                              cache_values * sizeof(*key_actual)) ||
        !ds4_gpu_tensor_write(value_cache, 0, value_actual,
                              cache_values * sizeof(*value_actual)) ||
        !ds4_gpu_tensor_write(q, 0, q_host, q_count * sizeof(*q_host)) ||
        !ds4_gpu_tensor_write(k, 0, k_host, kv_width * sizeof(*k_host)) ||
        !ds4_gpu_tensor_write(v, 0, v_host, kv_width * sizeof(*v_host)) ||
        !ds4_gpu_tensor_write(gate, 0, gate_host,
                              (uint64_t)c->n_head * sizeof(*gate_host))) {
        fprintf(stderr, "decode-attention/%s: synthetic tensor setup failed\n",
                c->family);
        goto cleanup;
    }

    const laguna_tensor_snapshot initial_snapshots[] = {
        { "heads", heads, q_count * sizeof(*actual), NULL },
        { "key cache", key_cache, cache_values * sizeof(*key_actual), NULL },
        { "value cache", value_cache, cache_values * sizeof(*value_actual), NULL },
        { "q", q, q_count * sizeof(*q_host), NULL },
        { "k", k, kv_width * sizeof(*k_host), NULL },
        { "v", v, kv_width * sizeof(*v_host), NULL },
        { "gate", gate, (uint64_t)c->n_head * sizeof(*gate_host), NULL },
    };
    memcpy(snapshots, initial_snapshots, sizeof(snapshots));
    for (size_t i = 0; i < sizeof(snapshots) / sizeof(snapshots[0]); i++) {
        if (!laguna_capture_tensor_snapshot(&snapshots[i])) {
            fprintf(stderr, "decode-attention/%s: snapshot setup failed\n",
                    c->family);
            goto cleanup;
        }
    }

    /* Rejections must happen before either production launch: exercise the
     * scalar/causal/null boundary against the same fully valid fixture. */
    const struct {
        const char *name;
        uint32_t key_start;
        uint32_t key_count;
        uint32_t n_head;
        uint32_t n_head_kv;
        float scale;
    } rejected[] = {
        { "non-integral-gqa", c->key_start, c->key_count, 50u, 8u, scale },
        { "zero-keys", c->key_start, 0u, c->n_head, c->n_head_kv, scale },
        { "too-many-keys", c->key_start, c->cache_cap + 1u, c->n_head,
          c->n_head_kv, scale },
        { "key-start-after-pos", c->pos + 1u, 1u, c->n_head,
          c->n_head_kv, scale },
        { "non-causal-range", c->key_start, c->key_count - 1u, c->n_head,
          c->n_head_kv, scale },
        { "zero-scale", c->key_start, c->key_count, c->n_head,
          c->n_head_kv, 0.0f },
        { "nan-scale", c->key_start, c->key_count, c->n_head,
          c->n_head_kv, NAN },
    };
    (void)cudaGetLastError();
    for (size_t i = 0; i < sizeof(rejected) / sizeof(rejected[0]); i++) {
        if (ds4_gpu_laguna_store_attention_tensor(
                heads, key_cache, value_cache, q, k, v, gate, c->pos,
                c->cache_cap, rejected[i].key_start, rejected[i].key_count,
                rejected[i].n_head, rejected[i].n_head_kv, head_dim,
                rejected[i].scale)) {
            fprintf(stderr, "decode-attention/%s: accepted %s\n", c->family,
                    rejected[i].name);
            goto cleanup;
        }
    }
    static const char *null_names[] = {
        "heads", "key cache", "value cache", "q", "k", "v", "gate",
    };
    for (size_t i = 0; i < sizeof(null_names) / sizeof(null_names[0]); i++) {
        if (ds4_gpu_laguna_store_attention_tensor(
                i == 0 ? NULL : heads, i == 1 ? NULL : key_cache,
                i == 2 ? NULL : value_cache, i == 3 ? NULL : q,
                i == 4 ? NULL : k, i == 5 ? NULL : v,
                i == 6 ? NULL : gate, c->pos, c->cache_cap, c->key_start,
                c->key_count, c->n_head, c->n_head_kv, head_dim, scale)) {
            fprintf(stderr, "decode-attention/%s: accepted null %s\n",
                    c->family, null_names[i]);
            goto cleanup;
        }
    }
    ds4_gpu_tensor *short_q = ds4_gpu_tensor_view(q, 0,
                                                    q_count * sizeof(*q_host) - 1u);
    if (!short_q || ds4_gpu_laguna_store_attention_tensor(
            heads, key_cache, value_cache, short_q, k, v, gate, c->pos,
            c->cache_cap, c->key_start, c->key_count, c->n_head,
            c->n_head_kv, head_dim, scale)) {
        fprintf(stderr, "decode-attention/%s: accepted undersized q view\n",
                c->family);
        ds4_gpu_tensor_free(short_q);
        goto cleanup;
    }
    ds4_gpu_tensor_free(short_q);
    ds4_gpu_tensor *short_heads = ds4_gpu_tensor_view(heads, 0, q_count * sizeof(*actual) - 1u);
    ds4_gpu_tensor *short_k = ds4_gpu_tensor_view(k, 0, kv_width * sizeof(*k_host) - 1u);
    ds4_gpu_tensor *short_v = ds4_gpu_tensor_view(v, 0, kv_width * sizeof(*v_host) - 1u);
    ds4_gpu_tensor *short_gate = ds4_gpu_tensor_view(gate, 0, (uint64_t)c->n_head * sizeof(*gate_host) - 1u);
    ds4_gpu_tensor *short_key = ds4_gpu_tensor_view(key_cache, 0, cache_values * sizeof(*key_expected) - 1u);
    ds4_gpu_tensor *short_value = ds4_gpu_tensor_view(value_cache, 0, cache_values * sizeof(*value_expected) - 1u);
    const int short_accepted = !short_heads || !short_k || !short_v || !short_gate || !short_key || !short_value ||
        ds4_gpu_laguna_store_attention_tensor(short_heads, key_cache, value_cache, q, k, v, gate, c->pos, c->cache_cap, c->key_start, c->key_count, c->n_head, c->n_head_kv, head_dim, scale) ||
        ds4_gpu_laguna_store_attention_tensor(heads, short_key, value_cache, q, k, v, gate, c->pos, c->cache_cap, c->key_start, c->key_count, c->n_head, c->n_head_kv, head_dim, scale) ||
        ds4_gpu_laguna_store_attention_tensor(heads, key_cache, short_value, q, k, v, gate, c->pos, c->cache_cap, c->key_start, c->key_count, c->n_head, c->n_head_kv, head_dim, scale) ||
        ds4_gpu_laguna_store_attention_tensor(heads, key_cache, value_cache, q, short_k, v, gate, c->pos, c->cache_cap, c->key_start, c->key_count, c->n_head, c->n_head_kv, head_dim, scale) ||
        ds4_gpu_laguna_store_attention_tensor(heads, key_cache, value_cache, q, k, short_v, gate, c->pos, c->cache_cap, c->key_start, c->key_count, c->n_head, c->n_head_kv, head_dim, scale) ||
        ds4_gpu_laguna_store_attention_tensor(heads, key_cache, value_cache, q, k, v, short_gate, c->pos, c->cache_cap, c->key_start, c->key_count, c->n_head, c->n_head_kv, head_dim, scale);
    ds4_gpu_tensor_free(short_value); ds4_gpu_tensor_free(short_key); ds4_gpu_tensor_free(short_gate);
    ds4_gpu_tensor_free(short_v); ds4_gpu_tensor_free(short_k); ds4_gpu_tensor_free(short_heads);
    if (short_accepted) { fprintf(stderr, "decode-attention/%s: accepted undersized view\n", c->family); goto cleanup; }
    if (cudaDeviceSynchronize() != cudaSuccess || cudaGetLastError() != cudaSuccess) {
        fprintf(stderr, "decode-attention/%s: rejection left CUDA error\n",
                c->family);
        goto cleanup;
    }
    for (size_t i = 0; i < sizeof(snapshots) / sizeof(snapshots[0]); i++) {
        if (!laguna_tensor_matches_snapshot(&snapshots[i], c->family)) {
            goto cleanup;
        }
    }

    const int wrapper_ok = ds4_gpu_laguna_store_attention_tensor(
        heads, key_cache, value_cache, q, k, v, gate, c->pos, c->cache_cap,
        c->key_start, c->key_count, c->n_head, c->n_head_kv, head_dim, scale);
    /* Each case submits one KV-store + decode-attention pipeline. */
    const cudaError_t sync = cudaDeviceSynchronize();
    if (sync != cudaSuccess) {
        fprintf(stderr, "decode-attention/%s: cudaDeviceSynchronize: %s\n",
                c->family, cudaGetErrorString(sync));
        goto cleanup;
    }
    if (!wrapper_ok) {
        fprintf(stderr,
                "decode-attention/%s: CUDA wrapper returned failure (pos=%u keys=%u gate=%g)\n",
                c->family, c->pos, c->key_count, (double)c->gate);
        goto cleanup;
    }
    if (!ds4_gpu_tensor_read(key_cache, 0, key_actual,
                             cache_values * sizeof(*key_actual)) ||
        !ds4_gpu_tensor_read(value_cache, 0, value_actual,
                             cache_values * sizeof(*value_actual)) ||
        !ds4_gpu_tensor_read(heads, 0, actual, q_count * sizeof(*actual))) {
        fprintf(stderr, "decode-attention/%s: output read failed\n", c->family);
        goto cleanup;
    }
    if (memcmp(key_actual, key_expected,
               (size_t)cache_values * sizeof(*key_actual)) != 0 ||
        memcmp(value_actual, value_expected,
               (size_t)cache_values * sizeof(*value_actual)) != 0) {
        fprintf(stderr,
                "decode-attention/%s: F32-to-F16 KV ring mapping differs (pos=%u keys=%u)\n",
                c->family, c->pos, c->key_count);
        goto cleanup;
    }
    float max_abs = 0.0f;
    double square_error = 0.0;
    uint64_t max_index = 0;
    for (uint64_t i = 0; i < q_count; i++) {
        if (!isfinite(actual[i])) {
            fprintf(stderr,
                    "decode-attention/%s: non-finite output head=%llu dim=%llu\n",
                    c->family, (unsigned long long)(i / head_dim),
                    (unsigned long long)(i % head_dim));
            goto cleanup;
        }
        const float error = fabsf(actual[i] - reference[i]);
        if (error > max_abs) {
            max_abs = error;
            max_index = i;
        }
        square_error += (double)error * error;
    }
    const double rms = sqrt(square_error / (double)q_count);
    if (max_abs > 1.0e-3f || rms > 2.0e-4) {
        fprintf(stderr,
                "decode-attention/%s: parity max_abs=%g rms=%g head=%llu dim=%llu actual=%g reference=%g exceeds tolerance\n",
                c->family, (double)max_abs, rms,
                (unsigned long long)(max_index / head_dim),
                (unsigned long long)(max_index % head_dim),
                (double)actual[max_index], (double)reference[max_index]);
        goto cleanup;
    }
    rc = 0;

cleanup:
    for (size_t i = 0; i < sizeof(snapshots) / sizeof(snapshots[0]); i++) {
        free(snapshots[i].snapshot);
    }
    ds4_gpu_tensor_free(gate);
    ds4_gpu_tensor_free(v);
    ds4_gpu_tensor_free(k);
    ds4_gpu_tensor_free(q);
    ds4_gpu_tensor_free(value_cache);
    ds4_gpu_tensor_free(key_cache);
    ds4_gpu_tensor_free(heads);
    free(value_actual);
    free(key_actual);
    free(value_expected);
    free(key_expected);
    free(actual);
    free(reference);
    free(gate_host);
    free(scores);
    free(v_host);
    free(k_host);
    free(q_host);
    return rc;
}

static int run_decode_attention_cases(void) {
    static const float gates[] = { -20.0f, -2.0f, 0.0f, 2.0f, 20.0f };
    const uint32_t global_counts[] = { 1u, 1023u, 1024u, 1025u };
    const uint32_t swa_positions[] = { 0u, 510u, 511u, 512u, 513u };
    int rc = 0;
    for (size_t count_i = 0; count_i < sizeof(global_counts) / sizeof(global_counts[0]); count_i++) {
        for (size_t gate_i = 0; gate_i < sizeof(gates) / sizeof(gates[0]); gate_i++) {
            const uint32_t key_count = global_counts[count_i];
            const laguna_decode_attention_case c = {
                "gqa6-global", 48u, 8u, 2048u, key_count - 1u, 0u,
                key_count, gates[gate_i], false,
            };
            if (run_decode_attention_case(&c) != 0) rc = 1;
        }
    }
    for (size_t pos_i = 0; pos_i < sizeof(swa_positions) / sizeof(swa_positions[0]); pos_i++) {
        for (size_t gate_i = 0; gate_i < sizeof(gates) / sizeof(gates[0]); gate_i++) {
            const uint32_t pos = swa_positions[pos_i];
            const uint32_t key_count = pos < 512u ? pos + 1u : 512u;
            const laguna_decode_attention_case c = {
                "gqa9-swa", 72u, 8u, 512u, pos, pos + 1u - key_count,
                key_count, gates[gate_i], false,
            };
            if (run_decode_attention_case(&c) != 0) rc = 1;
        }
    }
    const laguna_decode_attention_case mutation_cases[] = {
        { "gqa6-global-mixed-gates", 48u, 8u, 2048u, 16u, 0u, 17u,
          0.0f, true },
        { "gqa6-global-nonpow", 48u, 8u, 8202u, 16u, 0u, 17u,
          0.0f, true },
        { "gqa6-shifted-partial-ring", 48u, 8u, 17u, 20u, 18u, 3u,
          0.0f, true },
    };
    for (size_t i = 0; i < sizeof(mutation_cases) / sizeof(mutation_cases[0]); i++) {
        if (run_decode_attention_case(&mutation_cases[i]) != 0) rc = 1;
    }
    return rc;
}

static int run_prefill_attention_case(uint32_t n_tokens) {
    const uint32_t n_head = 48u, n_head_kv = 8u, head_dim = 128u;
    const uint32_t cache_cap = 4u;
    const uint64_t q_count = (uint64_t)n_tokens * n_head * head_dim;
    const uint64_t kv_count = (uint64_t)n_tokens * n_head_kv * head_dim;
    const uint64_t kv_width = (uint64_t)n_head_kv * head_dim;
    const uint64_t cache_count = (uint64_t)cache_cap * kv_width;
    const float scale = 1.0f / sqrtf((float)head_dim);
    float *q_host = calloc((size_t)q_count, sizeof(*q_host));
    float *k_host = calloc((size_t)kv_count, sizeof(*k_host));
    float *v_host = calloc((size_t)kv_count, sizeof(*v_host));
    float *gate_host = calloc((size_t)n_tokens * n_head, sizeof(*gate_host));
    float *reference = calloc((size_t)q_count, sizeof(*reference));
    float *actual = calloc((size_t)q_count, sizeof(*actual));
    uint16_t *key_expected = calloc((size_t)cache_count, sizeof(*key_expected));
    uint16_t *value_expected = calloc((size_t)cache_count, sizeof(*value_expected));
    uint16_t *key_actual = calloc((size_t)cache_count, sizeof(*key_actual));
    uint16_t *value_actual = calloc((size_t)cache_count, sizeof(*value_actual));
    ds4_gpu_tensor *heads = NULL, *key_cache = NULL, *value_cache = NULL;
    ds4_gpu_tensor *staged_key = NULL, *staged_value = NULL, *q = NULL, *k = NULL;
    ds4_gpu_tensor *v = NULL, *gate = NULL;
    int rc = 1;
    if (!q_host || !k_host || !v_host || !gate_host || !reference || !actual ||
        !key_expected || !value_expected || !key_actual || !value_actual) goto cleanup;
    for (uint64_t i = 0; i < q_count; i++) q_host[i] = ((int)(i * 17u % 37u) - 18) / 29.0f;
    for (uint64_t i = 0; i < kv_count; i++) {
        k_host[i] = ((int)(i * 11u % 41u) - 20) / 31.0f;
        v_host[i] = ((int)(i * 23u % 43u) - 21) / 37.0f;
    }
    static const float gates[] = { -2.0f, 0.0f, 2.0f, 20.0f, -20.0f };
    for (uint32_t t = 0; t < n_tokens; t++) for (uint32_t h = 0; h < n_head; h++)
        gate_host[(uint64_t)t * n_head + h] = gates[(3u * t + h) % 5u];
    for (uint32_t t = 0; t < n_tokens; t++) {
        const uint64_t row = (uint64_t)(t % cache_cap) * kv_width;
        for (uint64_t i = 0; i < kv_width; i++) {
            key_expected[row + i] = reference_f32_to_f16(k_host[(uint64_t)t * kv_width + i]);
            value_expected[row + i] = reference_f32_to_f16(v_host[(uint64_t)t * kv_width + i]);
        }
        for (uint32_t h = 0; h < n_head; h++) {
            const uint32_t kv_head = h / (n_head / n_head_kv);
            float scores[4], max_score = -INFINITY, sum = 0.0f;
            for (uint32_t r = 0; r <= t; r++) {
                const uint64_t base = (uint64_t)r * kv_width + (uint64_t)kv_head * head_dim;
                float dot = 0.0f;
                for (uint32_t d = 0; d < head_dim; d++) dot += q_host[((uint64_t)t * n_head + h) * head_dim + d] * reference_f16_to_f32(key_expected[base + d]);
                scores[r] = scale * dot;
                if (scores[r] > max_score) max_score = scores[r];
            }
            for (uint32_t r = 0; r <= t; r++) sum += expf(scores[r] - max_score);
            for (uint32_t d = 0; d < head_dim; d++) {
                float weighted = 0.0f;
                for (uint32_t r = 0; r <= t; r++) {
                    const uint64_t base = (uint64_t)r * kv_width + (uint64_t)kv_head * head_dim;
                    weighted += expf(scores[r] - max_score) * reference_f16_to_f32(value_expected[base + d]);
                }
                reference[((uint64_t)t * n_head + h) * head_dim + d] = weighted / sum * reference_softplus(gate_host[(uint64_t)t * n_head + h]);
            }
        }
    }
    heads = ds4_gpu_tensor_alloc(q_count * sizeof(*actual));
    key_cache = ds4_gpu_tensor_alloc(cache_count * sizeof(*key_actual));
    value_cache = ds4_gpu_tensor_alloc(cache_count * sizeof(*value_actual));
    staged_key = ds4_gpu_tensor_alloc(kv_count * sizeof(*key_actual));
    staged_value = ds4_gpu_tensor_alloc(kv_count * sizeof(*value_actual));
    q = ds4_gpu_tensor_alloc(q_count * sizeof(*q_host));
    k = ds4_gpu_tensor_alloc(kv_count * sizeof(*k_host));
    v = ds4_gpu_tensor_alloc(kv_count * sizeof(*v_host));
    gate = ds4_gpu_tensor_alloc((uint64_t)n_tokens * n_head * sizeof(*gate_host));
    if (!heads || !key_cache || !value_cache || !staged_key || !staged_value || !q || !k || !v || !gate ||
        !ds4_gpu_tensor_write(q, 0, q_host, q_count * sizeof(*q_host)) ||
        !ds4_gpu_tensor_write(k, 0, k_host, kv_count * sizeof(*k_host)) ||
        !ds4_gpu_tensor_write(v, 0, v_host, kv_count * sizeof(*v_host)) ||
        !ds4_gpu_tensor_write(key_cache, 0, key_actual, cache_count * sizeof(*key_actual)) ||
        !ds4_gpu_tensor_write(value_cache, 0, value_actual, cache_count * sizeof(*value_actual)) ||
        !ds4_gpu_tensor_write(gate, 0, gate_host, (uint64_t)n_tokens * n_head * sizeof(*gate_host))) goto cleanup;
    if (!ds4_gpu_laguna_attention_prefill_tensor(heads, key_cache, value_cache, staged_key, staged_value, q, k, v, gate, 0u, n_tokens, cache_cap, n_head, n_head_kv, head_dim, scale)) {
        fprintf(stderr, "prefill-attention/%u: CUDA prefill stub returned failure\n", n_tokens);
        goto cleanup;
    }
    if (cudaDeviceSynchronize() != cudaSuccess ||
        !ds4_gpu_tensor_read(heads, 0, actual, q_count * sizeof(*actual)) ||
        !ds4_gpu_tensor_read(key_cache, 0, key_actual, cache_count * sizeof(*key_actual)) ||
        !ds4_gpu_tensor_read(value_cache, 0, value_actual, cache_count * sizeof(*value_actual))) {
        fprintf(stderr, "prefill-attention/%u: synchronization or read failed\n", n_tokens);
        goto cleanup;
    }
    for (uint64_t i = 0; i < q_count; i++) {
        if (!isfinite(actual[i]) || fabsf(actual[i] - reference[i]) > 1.0e-3f) {
            fprintf(stderr, "prefill-attention/%u: output[%llu]=%g expected=%g\n", n_tokens,
                    (unsigned long long)i, (double)actual[i], (double)reference[i]);
            goto cleanup;
        }
    }
    if (memcmp(key_actual, key_expected, cache_count * sizeof(*key_actual)) ||
        memcmp(value_actual, value_expected, cache_count * sizeof(*value_actual))) {
        fprintf(stderr, "prefill-attention/%u: cache mismatch\n", n_tokens);
        goto cleanup;
    }
    rc = 0;
cleanup:
    ds4_gpu_tensor_free(gate); ds4_gpu_tensor_free(v); ds4_gpu_tensor_free(k); ds4_gpu_tensor_free(q);
    ds4_gpu_tensor_free(staged_value); ds4_gpu_tensor_free(staged_key); ds4_gpu_tensor_free(value_cache); ds4_gpu_tensor_free(key_cache); ds4_gpu_tensor_free(heads);
    free(value_actual); free(key_actual); free(value_expected); free(key_expected); free(actual); free(reference); free(gate_host); free(v_host); free(k_host); free(q_host);
    return rc;
}

static float prefill_fixture_value(uint32_t logical_pos, uint64_t element,
                                   uint32_t multiplier, uint32_t modulus,
                                   int32_t bias) {
    return (float)((int32_t)((logical_pos * multiplier + element * 17u) % modulus) + bias) /
        (float)(modulus - 1u);
}

static int run_prefill_ring_case(const char *name, uint32_t pos0,
                                 uint32_t n_tokens, uint32_t cache_cap) {
    const uint32_t n_head = 48u, n_head_kv = 8u, head_dim = 128u;
    const uint64_t q_count = (uint64_t)n_tokens * n_head * head_dim;
    const uint64_t kv_width = (uint64_t)n_head_kv * head_dim;
    const uint64_t kv_count = (uint64_t)n_tokens * kv_width;
    const uint64_t cache_count = (uint64_t)cache_cap * kv_width;
    const float scale = 1.0f / sqrtf((float)head_dim);
    float *q_host = calloc((size_t)q_count, sizeof(*q_host));
    float *k_host = calloc((size_t)kv_count, sizeof(*k_host));
    float *v_host = calloc((size_t)kv_count, sizeof(*v_host));
    float *gates = calloc((size_t)n_tokens * n_head, sizeof(*gates));
    float *expected = calloc((size_t)q_count, sizeof(*expected));
    float *actual = calloc((size_t)q_count, sizeof(*actual));
    float *scores = calloc((size_t)cache_cap, sizeof(*scores));
    uint16_t *key_expected = calloc((size_t)cache_count, sizeof(*key_expected));
    uint16_t *value_expected = calloc((size_t)cache_count, sizeof(*value_expected));
    uint16_t *key_initial = calloc((size_t)cache_count, sizeof(*key_initial));
    uint16_t *value_initial = calloc((size_t)cache_count, sizeof(*value_initial));
    uint16_t *key_actual = calloc((size_t)cache_count, sizeof(*key_actual));
    uint16_t *value_actual = calloc((size_t)cache_count, sizeof(*value_actual));
    ds4_gpu_tensor *heads = NULL, *key_cache = NULL, *value_cache = NULL;
    ds4_gpu_tensor *staged_key = NULL, *staged_value = NULL, *q = NULL, *k = NULL;
    ds4_gpu_tensor *v = NULL, *gate = NULL;
    int rc = 1;
    if (!q_host || !k_host || !v_host || !gates || !expected || !actual || !scores ||
        !key_expected || !value_expected || !key_initial || !value_initial ||
        !key_actual || !value_actual) goto cleanup;
    for (uint32_t p = 0; p < pos0; p++) for (uint64_t i = 0; i < kv_width; i++) {
        const uint64_t dst = (uint64_t)(p % cache_cap) * kv_width + i;
        key_expected[dst] = reference_f32_to_f16(prefill_fixture_value(p, i, 13u, 61u, -30));
        value_expected[dst] = reference_f32_to_f16(prefill_fixture_value(p, i, 19u, 67u, -33));
    }
    memcpy(key_initial, key_expected, cache_count * sizeof(*key_initial));
    memcpy(value_initial, value_expected, cache_count * sizeof(*value_initial));
    static const float mixed_gates[] = { -20.0f, -2.0f, 0.0f, 2.0f, 20.0f };
    for (uint32_t t = 0; t < n_tokens; t++) {
        const uint32_t p = pos0 + t;
        for (uint64_t i = 0; i < kv_width; i++) {
            k_host[(uint64_t)t * kv_width + i] = prefill_fixture_value(p, i, 13u, 61u, -30);
            v_host[(uint64_t)t * kv_width + i] = prefill_fixture_value(p, i, 19u, 67u, -33);
            const uint64_t dst = (uint64_t)(p % cache_cap) * kv_width + i;
            key_expected[dst] = reference_f32_to_f16(k_host[(uint64_t)t * kv_width + i]);
            value_expected[dst] = reference_f32_to_f16(v_host[(uint64_t)t * kv_width + i]);
        }
        for (uint32_t h = 0; h < n_head; h++) {
            gates[(uint64_t)t * n_head + h] = mixed_gates[(3u * t + h) % 5u];
            for (uint32_t d = 0; d < head_dim; d++) {
                q_host[((uint64_t)t * n_head + h) * head_dim + d] =
                    prefill_fixture_value(p + h, d, 7u, 59u, -29);
            }
        }
        const uint32_t key_start = p + 1u > cache_cap ? p + 1u - cache_cap : 0u;
        const uint32_t key_count = p - key_start + 1u;
        for (uint32_t h = 0; h < n_head; h++) {
            const uint32_t kv_head = h / (n_head / n_head_kv);
            float max_score = -INFINITY, sum = 0.0f;
            for (uint32_t r = 0; r < key_count; r++) {
                const uint64_t base = (uint64_t)((key_start + r) % cache_cap) * kv_width +
                    (uint64_t)kv_head * head_dim;
                float dot = 0.0f;
                for (uint32_t d = 0; d < head_dim; d++) dot +=
                    q_host[((uint64_t)t * n_head + h) * head_dim + d] * reference_f16_to_f32(key_expected[base + d]);
                scores[r] = scale * dot;
                if (scores[r] > max_score) max_score = scores[r];
            }
            for (uint32_t r = 0; r < key_count; r++) sum += expf(scores[r] - max_score);
            for (uint32_t d = 0; d < head_dim; d++) {
                float weighted = 0.0f;
                for (uint32_t r = 0; r < key_count; r++) {
                    const uint64_t base = (uint64_t)((key_start + r) % cache_cap) * kv_width +
                        (uint64_t)kv_head * head_dim;
                    weighted += expf(scores[r] - max_score) * reference_f16_to_f32(value_expected[base + d]);
                }
                expected[((uint64_t)t * n_head + h) * head_dim + d] =
                    weighted / sum * reference_softplus(gates[(uint64_t)t * n_head + h]);
            }
        }
    }
    heads = ds4_gpu_tensor_alloc(q_count * sizeof(*actual));
    key_cache = ds4_gpu_tensor_alloc(cache_count * sizeof(*key_actual));
    value_cache = ds4_gpu_tensor_alloc(cache_count * sizeof(*value_actual));
    staged_key = ds4_gpu_tensor_alloc(kv_count * sizeof(*key_actual));
    staged_value = ds4_gpu_tensor_alloc(kv_count * sizeof(*value_actual));
    q = ds4_gpu_tensor_alloc(q_count * sizeof(*q_host)); k = ds4_gpu_tensor_alloc(kv_count * sizeof(*k_host));
    v = ds4_gpu_tensor_alloc(kv_count * sizeof(*v_host)); gate = ds4_gpu_tensor_alloc((uint64_t)n_tokens * n_head * sizeof(*gates));
    if (!heads || !key_cache || !value_cache || !staged_key || !staged_value || !q || !k || !v || !gate ||
        !ds4_gpu_tensor_write(q, 0, q_host, q_count * sizeof(*q_host)) ||
        !ds4_gpu_tensor_write(k, 0, k_host, kv_count * sizeof(*k_host)) ||
        !ds4_gpu_tensor_write(v, 0, v_host, kv_count * sizeof(*v_host)) ||
        !ds4_gpu_tensor_write(key_cache, 0, key_initial, cache_count * sizeof(*key_initial)) ||
        !ds4_gpu_tensor_write(value_cache, 0, value_initial, cache_count * sizeof(*value_initial)) ||
        !ds4_gpu_tensor_write(gate, 0, gates, (uint64_t)n_tokens * n_head * sizeof(*gates))) goto cleanup;
    if (!ds4_gpu_laguna_attention_prefill_tensor(heads, key_cache, value_cache, staged_key, staged_value, q, k, v, gate, pos0, n_tokens, cache_cap, n_head, n_head_kv, head_dim, scale)) {
        fprintf(stderr, "prefill-attention/%s: CUDA prefill stub returned failure\n", name); goto cleanup;
    }
    if (cudaDeviceSynchronize() != cudaSuccess || !ds4_gpu_tensor_read(heads, 0, actual, q_count * sizeof(*actual)) ||
        !ds4_gpu_tensor_read(key_cache, 0, key_actual, cache_count * sizeof(*key_actual)) ||
        !ds4_gpu_tensor_read(value_cache, 0, value_actual, cache_count * sizeof(*value_actual))) goto cleanup;
    for (uint64_t i = 0; i < q_count; i++) if (!isfinite(actual[i]) || fabsf(actual[i] - expected[i]) > 1.0e-3f) goto cleanup;
    if (memcmp(key_actual, key_expected, cache_count * sizeof(*key_actual)) || memcmp(value_actual, value_expected, cache_count * sizeof(*value_actual))) goto cleanup;
    rc = 0;
cleanup:
    ds4_gpu_tensor_free(gate); ds4_gpu_tensor_free(v); ds4_gpu_tensor_free(k); ds4_gpu_tensor_free(q); ds4_gpu_tensor_free(staged_value); ds4_gpu_tensor_free(staged_key); ds4_gpu_tensor_free(value_cache); ds4_gpu_tensor_free(key_cache); ds4_gpu_tensor_free(heads);
    free(value_actual); free(key_actual); free(value_initial); free(key_initial); free(value_expected); free(key_expected); free(scores); free(actual); free(expected); free(gates); free(v_host); free(k_host); free(q_host);
    return rc;
}

static int run_prefill_causal_mutation_case(uint32_t first_changed) {
    const uint32_t n_head = 48u, n_head_kv = 8u, head_dim = 128u, n_tokens = 3u, cache_cap = 4u;
    const uint64_t q_count = (uint64_t)n_tokens * n_head * head_dim;
    const uint64_t kv_width = (uint64_t)n_head_kv * head_dim, kv_count = (uint64_t)n_tokens * kv_width;
    const uint64_t cache_count = (uint64_t)cache_cap * kv_width;
    const float scale = 1.0f / sqrtf((float)head_dim);
    float *q_host = calloc((size_t)q_count, sizeof(*q_host)), *k_a = calloc((size_t)kv_count, sizeof(*k_a)), *v_a = calloc((size_t)kv_count, sizeof(*v_a));
    float *k_b = calloc((size_t)kv_count, sizeof(*k_b)), *v_b = calloc((size_t)kv_count, sizeof(*v_b)), *gate_host = calloc((size_t)n_tokens * n_head, sizeof(*gate_host));
    float *out_a = calloc((size_t)q_count, sizeof(*out_a)), *out_b = calloc((size_t)q_count, sizeof(*out_b));
    ds4_gpu_tensor *heads[2] = {0}, *key_cache[2] = {0}, *value_cache[2] = {0}, *staged_key[2] = {0}, *staged_value[2] = {0}, *q[2] = {0}, *k[2] = {0}, *v[2] = {0}, *gate[2] = {0};
    int rc = 1;
    if (!q_host || !k_a || !v_a || !k_b || !v_b || !gate_host || !out_a || !out_b) goto cleanup;
    for (uint64_t i = 0; i < q_count; i++) q_host[i] = ((int)(i % 31u) - 15) / 17.0f;
    for (uint64_t i = 0; i < kv_count; i++) k_a[i] = k_b[i] = v_a[i] = v_b[i] = ((int)(i % 29u) - 14) / 19.0f;
    for (uint32_t h = 0; h < n_head; h++) for (uint32_t t = 0; t < n_tokens; t++) gate_host[(uint64_t)t * n_head + h] = (float)((int)((h + t) % 5u) - 2);
    for (uint64_t i = (uint64_t)first_changed * kv_width; i < kv_count; i++) { k_b[i] += 0.75f; v_b[i] -= 0.5f; }
    for (uint32_t arm = 0; arm < 2; arm++) {
        heads[arm] = ds4_gpu_tensor_alloc(q_count * sizeof(float)); key_cache[arm] = ds4_gpu_tensor_alloc(cache_count * sizeof(uint16_t)); value_cache[arm] = ds4_gpu_tensor_alloc(cache_count * sizeof(uint16_t)); staged_key[arm] = ds4_gpu_tensor_alloc(kv_count * sizeof(uint16_t)); staged_value[arm] = ds4_gpu_tensor_alloc(kv_count * sizeof(uint16_t)); q[arm] = ds4_gpu_tensor_alloc(q_count * sizeof(float)); k[arm] = ds4_gpu_tensor_alloc(kv_count * sizeof(float)); v[arm] = ds4_gpu_tensor_alloc(kv_count * sizeof(float)); gate[arm] = ds4_gpu_tensor_alloc((uint64_t)n_tokens * n_head * sizeof(float));
        if (!heads[arm] || !key_cache[arm] || !value_cache[arm] || !staged_key[arm] || !staged_value[arm] || !q[arm] || !k[arm] || !v[arm] || !gate[arm] || !ds4_gpu_tensor_write(q[arm], 0, q_host, q_count * sizeof(float)) || !ds4_gpu_tensor_write(k[arm], 0, arm ? k_b : k_a, kv_count * sizeof(float)) || !ds4_gpu_tensor_write(v[arm], 0, arm ? v_b : v_a, kv_count * sizeof(float)) || !ds4_gpu_tensor_write(gate[arm], 0, gate_host, (uint64_t)n_tokens * n_head * sizeof(float))) goto cleanup;
        if (!ds4_gpu_laguna_attention_prefill_tensor(heads[arm], key_cache[arm], value_cache[arm], staged_key[arm], staged_value[arm], q[arm], k[arm], v[arm], gate[arm], 0u, n_tokens, cache_cap, n_head, n_head_kv, head_dim, scale)) { fprintf(stderr, "prefill-attention/causal-mutation: CUDA prefill stub returned failure\n"); goto cleanup; }
        if (cudaDeviceSynchronize() != cudaSuccess || !ds4_gpu_tensor_read(heads[arm], 0, arm ? out_b : out_a, q_count * sizeof(float))) goto cleanup;
    }
    if (memcmp(out_a, out_b, (uint64_t)first_changed * n_head * head_dim * sizeof(float)) != 0) { fprintf(stderr, "prefill-attention/causal-mutation: later KV changed earlier output\n"); goto cleanup; }
    rc = 0;
cleanup:
    for (uint32_t arm = 0; arm < 2; arm++) { ds4_gpu_tensor_free(gate[arm]); ds4_gpu_tensor_free(v[arm]); ds4_gpu_tensor_free(k[arm]); ds4_gpu_tensor_free(q[arm]); ds4_gpu_tensor_free(staged_value[arm]); ds4_gpu_tensor_free(staged_key[arm]); ds4_gpu_tensor_free(value_cache[arm]); ds4_gpu_tensor_free(key_cache[arm]); ds4_gpu_tensor_free(heads[arm]); }
    free(out_b); free(out_a); free(gate_host); free(v_b); free(k_b); free(v_a); free(k_a); free(q_host); return rc;
}

static int run_prefill_rejection_cases(void) {
    const uint32_t n_head = 48u, n_head_kv = 8u, head_dim = 128u;
    const uint32_t cache_cap = 4u, n_tokens = 1u;
    const uint64_t q_bytes = (uint64_t)n_head * head_dim * sizeof(float);
    const uint64_t kv_bytes = (uint64_t)n_head_kv * head_dim * sizeof(float);
    const uint64_t cache_bytes = (uint64_t)cache_cap * n_head_kv * head_dim * sizeof(uint16_t);
    const uint64_t gate_bytes = (uint64_t)n_head * sizeof(float);
    ds4_gpu_tensor *t[9] = {0};
    laguna_tensor_snapshot snapshots[9] = {0};
    const uint64_t sizes[] = { q_bytes, cache_bytes, cache_bytes, kv_bytes / 2u, kv_bytes / 2u,
        q_bytes, kv_bytes, kv_bytes, gate_bytes };
    const char *names[] = { "heads", "key cache", "value cache", "staged key",
        "staged value", "q", "k", "v", "gate" };
    int rc = 1;
    for (size_t i = 0; i < 9; i++) {
        t[i] = ds4_gpu_tensor_alloc(sizes[i]);
        snapshots[i] = (laguna_tensor_snapshot){ names[i], t[i], sizes[i], NULL };
        if (!t[i] || !laguna_capture_tensor_snapshot(&snapshots[i])) goto cleanup;
    }
    const float scale = 1.0f / sqrtf((float)head_dim);
#define PREFILL_CALL(a, pos, ntok, cap, nh, nkh, dim, s) \
    ds4_gpu_laguna_attention_prefill_tensor((a)[0], (a)[1], (a)[2], (a)[3], (a)[4], \
        (a)[5], (a)[6], (a)[7], (a)[8], (pos), (ntok), (cap), (nh), (nkh), (dim), (s))
    const struct { const char *name; uint32_t pos, ntok, cap, nh, nkh, dim; float scale; } bad[] = {
        { "zero-tokens", 0u, 0u, cache_cap, n_head, n_head_kv, head_dim, scale },
        { "position-overflow", UINT32_MAX, 1u, cache_cap, n_head, n_head_kv, head_dim, scale },
        { "zero-cache", 0u, n_tokens, 0u, n_head, n_head_kv, head_dim, scale },
        { "zero-heads", 0u, n_tokens, cache_cap, 0u, n_head_kv, head_dim, scale },
        { "zero-kv-heads", 0u, n_tokens, cache_cap, n_head, 0u, head_dim, scale },
        { "nonintegral-gqa", 0u, n_tokens, cache_cap, 50u, n_head_kv, head_dim, scale },
        { "wrong-head-dim", 0u, n_tokens, cache_cap, n_head, n_head_kv, 64u, scale },
        { "zero-scale", 0u, n_tokens, cache_cap, n_head, n_head_kv, head_dim, 0.0f },
        { "nan-scale", 0u, n_tokens, cache_cap, n_head, n_head_kv, head_dim, NAN },
    };
    (void)cudaGetLastError();
    for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); i++) {
        if (PREFILL_CALL(t, bad[i].pos, bad[i].ntok, bad[i].cap, bad[i].nh,
                         bad[i].nkh, bad[i].dim, bad[i].scale)) {
            fprintf(stderr, "prefill-rejection: accepted %s\n", bad[i].name);
            goto cleanup;
        }
    }
    for (size_t i = 0; i < 9; i++) {
        ds4_gpu_tensor *null_args[9];
        memcpy(null_args, t, sizeof(t));
        null_args[i] = NULL;
        if (PREFILL_CALL(null_args, 0u, n_tokens, cache_cap, n_head, n_head_kv,
                         head_dim, scale)) {
            fprintf(stderr, "prefill-rejection: accepted null %s\n", names[i]);
            goto cleanup;
        }
    }
    for (size_t i = 0; i < 9; i++) {
        ds4_gpu_tensor *short_args[9];
        ds4_gpu_tensor *short_view = ds4_gpu_tensor_view(t[i], 0, sizes[i] - 1u);
        if (!short_view) goto cleanup;
        memcpy(short_args, t, sizeof(t));
        short_args[i] = short_view;
        const int accepted = PREFILL_CALL(short_args, 0u, n_tokens, cache_cap,
                                          n_head, n_head_kv, head_dim, scale);
        ds4_gpu_tensor_free(short_view);
        if (accepted) {
            fprintf(stderr, "prefill-rejection: accepted undersized %s\n", names[i]);
            goto cleanup;
        }
    }
    if (cudaDeviceSynchronize() != cudaSuccess || cudaGetLastError() != cudaSuccess) {
        fprintf(stderr, "prefill-rejection: rejection left CUDA error\n");
        goto cleanup;
    }
    for (size_t i = 0; i < 9; i++) {
        if (!laguna_tensor_matches_snapshot(&snapshots[i], "prefill-rejection")) goto cleanup;
    }
    rc = 0;
cleanup:
#undef PREFILL_CALL
    for (size_t i = 0; i < 9; i++) {
        free(snapshots[i].snapshot);
        ds4_gpu_tensor_free(t[i]);
    }
    return rc;
}

static int run_prefill_attention_cases(void) {
    int rc = run_prefill_rejection_cases();
    if (run_prefill_attention_case(1u) != 0) { fprintf(stderr, "prefill-attention/1: contract failure\n"); rc = 1; }
    if (run_prefill_attention_case(3u) != 0) { fprintf(stderr, "prefill-attention/3: contract failure\n"); rc = 1; }
    if (run_prefill_ring_case("resumed-global", 3u, 2u, 16u) != 0) { fprintf(stderr, "prefill-attention/resumed-global: contract failure\n"); rc = 1; }
    if (run_prefill_ring_case("swa-512-crossing", 509u, 4u, 512u) != 0) { fprintf(stderr, "prefill-attention/swa-512-crossing: contract failure\n"); rc = 1; }
    if (run_prefill_ring_case("multi-wrap-4", 0u, 9u, 4u) != 0) { fprintf(stderr, "prefill-attention/multi-wrap-4: contract failure\n"); rc = 1; }
    if (run_prefill_causal_mutation_case(1u) != 0) { fprintf(stderr, "prefill-attention/causal-mutation-1: contract failure\n"); rc = 1; }
    if (run_prefill_causal_mutation_case(2u) != 0) { fprintf(stderr, "prefill-attention/causal-mutation-2: contract failure\n"); rc = 1; }
    return rc;
}

/* Independent Q4_K/Q8_K routed-MoE oracle.  Both quantization boundaries
 * belong here: a float-only reference would be testing a different kernel. */
#define LAGUNA_QK_K 256u
#define LAGUNA_MOE_DIM 256u
#define LAGUNA_MOE_EXPERTS 16u
#define LAGUNA_MOE_USED 10u
#define LAGUNA_MOE_GUARD 8u

typedef struct { uint16_t d, dmin; uint8_t scales[12]; uint8_t qs[LAGUNA_QK_K / 2u]; } laguna_q4k_block;
typedef struct { float d; int8_t qs[LAGUNA_QK_K]; int16_t bsums[LAGUNA_QK_K / 16u]; } laguna_q8k_block;
typedef struct { ds4_gpu_tensor *base, *view; float *host; uint32_t count; } laguna_guarded_output;
typedef struct {
    uint64_t gate_offset, up_offset, down_offset;
    uint32_t gate_type, up_type, down_type;
    uint64_t gate_expert_bytes, gate_row_bytes;
    uint64_t up_expert_bytes, up_row_bytes;
    uint64_t down_expert_bytes, down_row_bytes;
} laguna_moe_desc;

static void laguna_q4k_scale_min(uint32_t group, const uint8_t *scales, uint8_t *scale, uint8_t *minimum) {
    if (group < 4u) { *scale = scales[group] & 63u; *minimum = scales[group + 4u] & 63u; }
    else { *scale = (scales[group + 4u] & 0x0fu) | ((scales[group - 4u] >> 6u) << 4u);
           *minimum = (scales[group + 4u] >> 4u) | ((scales[group] >> 6u) << 4u); }
}

static void laguna_quantize_q8k(laguna_q8k_block *out, const float *in) {
    uint32_t max_i = 0;
    for (uint32_t i = 1; i < LAGUNA_QK_K; i++) if (fabsf(in[i]) > fabsf(in[max_i])) max_i = i;
    const float maxv = in[max_i];
    if (maxv == 0.0f) { memset(out, 0, sizeof(*out)); return; }
    const float inverse_scale = -127.0f / maxv;
    for (uint32_t i = 0; i < LAGUNA_QK_K; i++) {
        long q = lrintf(inverse_scale * in[i]);
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        out->qs[i] = (int8_t)q;
    }
    for (uint32_t group = 0; group < LAGUNA_QK_K / 16u; group++) {
        int sum = 0; for (uint32_t i = 0; i < 16u; i++) sum += out->qs[group * 16u + i];
        out->bsums[group] = (int16_t)sum;
    }
    out->d = 1.0f / inverse_scale;
}

static float laguna_q4k_q8k_dot(const laguna_q4k_block *weights, const laguna_q8k_block *input) {
    int isum = 0, sum_min = 0;
    for (uint32_t group = 0; group < LAGUNA_QK_K / 32u; group++) {
        uint8_t scale, minimum;
        laguna_q4k_scale_min(group, weights->scales, &scale, &minimum);
        sum_min += (int)minimum * ((int)input->bsums[group * 2u] + (int)input->bsums[group * 2u + 1u]);
        const uint32_t byte_offset = (group >> 1u) * 32u, shift = (group & 1u) * 4u;
        for (uint32_t i = 0; i < 32u; i++)
            isum += (int)scale * (int)((weights->qs[byte_offset + i] >> shift) & 0x0fu) * (int)input->qs[group * 32u + i];
    }
    return input->d * reference_f16_to_f32(weights->d) * (float)isum - input->d * reference_f16_to_f32(weights->dmin) * (float)sum_min;
}

static void laguna_encode_q4k(laguna_q4k_block *out, uint32_t seed, float scale) {
    memset(out, 0, sizeof(*out)); out->d = reference_f32_to_f16(scale);
    for (uint32_t group = 0; group < 4u; group++) out->scales[group] = 1u;
    for (uint32_t group = 4u; group < 8u; group++) out->scales[group + 4u] = 1u;
    for (uint32_t group = 0; group < 8u; group++) {
        const uint32_t byte_offset = (group >> 1u) * 32u, shift = (group & 1u) * 4u;
        for (uint32_t i = 0; i < 32u; i++) {
            const uint8_t q = (uint8_t)(1u + ((seed + 17u * group + 7u * i) % 5u));
            out->qs[byte_offset + i] |= (uint8_t)(q << shift);
        }
    }
}

static void laguna_fill_q4k_matrix(laguna_q4k_block *matrix, uint32_t expert, uint32_t projection) {
    for (uint32_t row = 0; row < LAGUNA_MOE_DIM; row++)
        laguna_encode_q4k(&matrix[row], 1u + 1009u * projection + 257u * expert + row,
                           0.0020f + 0.0002f * (float)expert +
                           0.000005f * (float)((row + projection) % 11u));
}

static float laguna_silu(float value) { return value >= 0.0f ? value / (1.0f + expf(-value)) : expf(value) / (1.0f + expf(value)); }

static void laguna_reference_routed_moe(
        float *out,
        const unsigned char *model,
        const laguna_moe_desc *routed,
        const int32_t *selected,
        const float *router,
        const float *x,
        uint32_t n_tokens) {
    const laguna_q4k_block *routed_gate = (const laguna_q4k_block *)(model + routed->gate_offset);
    const laguna_q4k_block *routed_up = (const laguna_q4k_block *)(model + routed->up_offset);
    const laguna_q4k_block *routed_down = (const laguna_q4k_block *)(model + routed->down_offset);
    for (uint32_t token = 0; token < n_tokens; token++) {
        laguna_q8k_block xq, midq; float mid[LAGUNA_MOE_DIM];
        laguna_quantize_q8k(&xq, x + (uint64_t)token * LAGUNA_MOE_DIM);
        memset(out + (uint64_t)token * LAGUNA_MOE_DIM, 0, LAGUNA_MOE_DIM * sizeof(*out));
        for (uint32_t slot = 0; slot < LAGUNA_MOE_USED; slot++) {
            const uint32_t expert = (uint32_t)selected[(uint64_t)token * LAGUNA_MOE_USED + slot];
            const float weight = router[(uint64_t)token * LAGUNA_MOE_USED + slot];
            const laguna_q4k_block *gate = routed_gate + (uint64_t)expert * LAGUNA_MOE_DIM;
            const laguna_q4k_block *up = routed_up + (uint64_t)expert * LAGUNA_MOE_DIM;
            const laguna_q4k_block *down = routed_down + (uint64_t)expert * LAGUNA_MOE_DIM;
            for (uint32_t row = 0; row < LAGUNA_MOE_DIM; row++) {
                mid[row] = weight * laguna_silu(laguna_q4k_q8k_dot(gate + row, &xq)) *
                    laguna_q4k_q8k_dot(up + row, &xq);
            }
            laguna_quantize_q8k(&midq, mid);
            for (uint32_t row = 0; row < LAGUNA_MOE_DIM; row++) out[(uint64_t)token * LAGUNA_MOE_DIM + row] += laguna_q4k_q8k_dot(down + row, &midq);
        }
    }
}

static int laguna_reference_signal(const char *name, const float *values, uint32_t count) {
    float max_abs = 0.0f; double square_sum = 0.0;
    for (uint32_t i = 0; i < count; i++) {
        if (!isfinite(values[i]) || fabsf(values[i]) > 1.0f) { fprintf(stderr, "routed-moe/%s: non-finite or unbounded reference\n", name); return 0; }
        if (fabsf(values[i]) > max_abs) max_abs = fabsf(values[i]);
        square_sum += (double)values[i] * values[i];
    }
    if (sqrt(square_sum / (double)count) < 0.02 || max_abs < 0.05) { fprintf(stderr, "routed-moe/%s: fixture signal is too weak\n", name); return 0; }
    return 1;
}

static int laguna_guarded_output_init(laguna_guarded_output *output, uint32_t count) {
    output->count = count;
    output->host = (float *)malloc((count + 2u * LAGUNA_MOE_GUARD) * sizeof(float));
    output->base = ds4_gpu_tensor_alloc((uint64_t)(count + 2u * LAGUNA_MOE_GUARD) * sizeof(float));
    output->view = output->base ? ds4_gpu_tensor_view(output->base, LAGUNA_MOE_GUARD * sizeof(float), (uint64_t)count * sizeof(float)) : NULL;
    if (!output->host || !output->base || !output->view) return 0;
    for (uint32_t i = 0; i < LAGUNA_MOE_GUARD; i++) { output->host[i] = -12345.0f - (float)i; output->host[LAGUNA_MOE_GUARD + count + i] = 23456.0f + (float)i; }
    return 1;
}

static int laguna_guarded_output_prepare(laguna_guarded_output *output) {
    for (uint32_t i = 0; i < output->count; i++) output->host[LAGUNA_MOE_GUARD + i] = NAN;
    return ds4_gpu_tensor_write(output->base, 0, output->host, (uint64_t)(output->count + 2u * LAGUNA_MOE_GUARD) * sizeof(float));
}

static int laguna_guarded_output_check(
        const char *name,
        laguna_guarded_output *output,
        const float *reference,
        uint32_t n_tokens) {
    if (output->count != n_tokens * LAGUNA_MOE_DIM) {
        fprintf(stderr, "routed-moe/%s: unexpected output size\n", name);
        return 0;
    }
    if (!ds4_gpu_tensor_read(output->base, 0, output->host,
                             (uint64_t)(output->count + 2u * LAGUNA_MOE_GUARD) * sizeof(float))) {
        fprintf(stderr, "routed-moe/%s: output read failed\n", name);
        return 0;
    }
    for (uint32_t i = 0; i < LAGUNA_MOE_GUARD; i++) {
        if (output->host[i] != -12345.0f - (float)i ||
            output->host[LAGUNA_MOE_GUARD + output->count + i] != 23456.0f + (float)i) {
            fprintf(stderr, "routed-moe/%s: output guard changed\n", name);
            return 0;
        }
    }
    for (uint32_t token = 0; token < n_tokens; token++) {
        double square_error = 0.0;
        float max_abs = 0.0f;
        for (uint32_t row = 0; row < LAGUNA_MOE_DIM; row++) {
            const uint64_t index = (uint64_t)token * LAGUNA_MOE_DIM + row;
            const float actual = output->host[LAGUNA_MOE_GUARD + index];
            const float expected = reference[index];
            if (!isfinite(actual) || !isfinite(expected)) {
                fprintf(stderr, "routed-moe/%s: non-finite actual/reference\n", name);
                return 0;
            }
            const float error = fabsf(actual - expected);
            if (error > max_abs) max_abs = error;
            square_error += (double)error * error;
        }
        const double rms_error = sqrt(square_error / (double)LAGUNA_MOE_DIM);
        if (max_abs > 2.0e-3f || rms_error > 5.0e-4) {
            fprintf(stderr,
                    "routed-moe/%s/token-%u: max_abs=%g rms=%g exceeds tolerance\n",
                    name, token, (double)max_abs, rms_error);
            return 0;
        }
    }
    return 1;
}

static void laguna_guarded_output_free(laguna_guarded_output *output) { ds4_gpu_tensor_free(output->view); ds4_gpu_tensor_free(output->base); free(output->host); memset(output, 0, sizeof(*output)); }

static float laguna_max_delta(const float *a, const float *b, uint32_t count) {
    float maximum = 0.0f;
    for (uint32_t i = 0; i < count; i++) {
        const float delta = fabsf(a[i] - b[i]);
        if (delta > maximum) maximum = delta;
    }
    return maximum;
}

static int laguna_glm_routed_moe_q4_call(
        ds4_gpu_tensor *out,
        ds4_gpu_tensor *mid,
        const unsigned char *model,
        uint64_t model_size,
        uint64_t projection_bytes,
        uint64_t expert_bytes,
        uint64_t row_bytes,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *router,
        const ds4_gpu_tensor *x,
        uint32_t n_tokens,
        bool force_resident) {
    return ds4_gpu_glm_routed_moe_batch_tensor(
            out, mid, model, model_size,
            0u, projection_bytes, 2u * projection_bytes,
            12u, 12u, 12u,
            expert_bytes, row_bytes, expert_bytes, row_bytes,
            expert_bytes, row_bytes,
            LAGUNA_MOE_DIM, LAGUNA_MOE_DIM, LAGUNA_MOE_DIM,
            selected, router, LAGUNA_MOE_EXPERTS, LAGUNA_MOE_USED,
            0u, x, n_tokens, LAGUNA_MOE_USED * LAGUNA_MOE_DIM,
            force_resident);
}

static void laguna_fill_routed_case(
        float *input,
        int32_t *selected,
        float *router,
        uint32_t n_tokens,
        uint32_t seed) {
    for (uint32_t token = 0; token < n_tokens; token++) {
        for (uint32_t i = 0; i < LAGUNA_MOE_DIM; i++) {
            input[(uint64_t)token * LAGUNA_MOE_DIM + i] =
                0.045f + 0.00031f *
                (float)((seed * 17u + token * 29u + i * 13u) % 211u);
        }
        input[(uint64_t)token * LAGUNA_MOE_DIM + 251u - token] =
            0.205f + 0.002f * (float)(seed + token);
        for (uint32_t slot = 0; slot < LAGUNA_MOE_USED; slot++) {
            selected[(uint64_t)token * LAGUNA_MOE_USED + slot] =
                (int32_t)((seed * 3u + token * 5u + slot * 7u) % LAGUNA_MOE_EXPERTS);
            router[(uint64_t)token * LAGUNA_MOE_USED + slot] = slot == 0u ?
                0.48f + 0.01f * (float)((seed + token) % 5u) :
                0.090f + 0.005f * (float)((seed + token * 3u + slot) % 11u);
        }
    }
}

static int run_routed_moe_selected_validation_case(void) {
    const uint64_t row_bytes = sizeof(laguna_q4k_block);
    const uint64_t expert_bytes = LAGUNA_MOE_DIM * row_bytes;
    const uint64_t projection_bytes = LAGUNA_MOE_EXPERTS * expert_bytes;
    const uint64_t model_size = 3u * projection_bytes;
    unsigned char *model = (unsigned char *)calloc(1, (size_t)model_size);
    float input[LAGUNA_MOE_DIM];
    float router[LAGUNA_MOE_USED];
    int32_t selected[LAGUNA_MOE_USED];
    ds4_gpu_tensor *mid = NULL, *selected_t = NULL, *router_t = NULL, *x = NULL;
    laguna_guarded_output out = {0};
    laguna_tensor_snapshot snapshots[5] = {0};
    int validation_enabled = 0;
    int rc = 1;

    if (!model) goto cleanup;
    for (uint32_t projection = 0; projection < 3u; projection++) {
        laguna_q4k_block *matrix =
            (laguna_q4k_block *)(model + projection * projection_bytes);
        for (uint32_t expert = 0; expert < LAGUNA_MOE_EXPERTS; expert++) {
            laguna_fill_q4k_matrix(matrix + (uint64_t)expert * LAGUNA_MOE_DIM,
                                   expert, projection);
        }
    }
    for (uint32_t i = 0; i < LAGUNA_MOE_DIM; i++) {
        input[i] = 0.031f + 0.00029f * (float)((i * 11u) % 197u);
    }
    for (uint32_t slot = 0; slot < LAGUNA_MOE_USED; slot++) {
        selected[slot] = (int32_t)slot;
        router[slot] = 0.05f + 0.01f * (float)slot;
    }
    selected[4] = (int32_t)LAGUNA_MOE_EXPERTS;

    if (!ds4_gpu_set_model_map(model, model_size)) goto cleanup;
    mid = ds4_gpu_tensor_alloc((uint64_t)LAGUNA_MOE_USED * LAGUNA_MOE_DIM * sizeof(float));
    selected_t = ds4_gpu_tensor_alloc(sizeof(selected));
    router_t = ds4_gpu_tensor_alloc(sizeof(router));
    x = ds4_gpu_tensor_alloc(sizeof(input));
    if (!mid || !selected_t || !router_t || !x ||
        !ds4_gpu_tensor_write(selected_t, 0, selected, sizeof(selected)) ||
        !ds4_gpu_tensor_write(router_t, 0, router, sizeof(router)) ||
        !ds4_gpu_tensor_write(x, 0, input, sizeof(input)) ||
        !laguna_guarded_output_init(&out, LAGUNA_MOE_DIM) ||
        !laguna_guarded_output_prepare(&out)) goto cleanup;

    const laguna_tensor_snapshot initial_snapshots[] = {
        { "output", out.base,
          (uint64_t)(LAGUNA_MOE_DIM + 2u * LAGUNA_MOE_GUARD) * sizeof(float), NULL },
        { "mid", mid, (uint64_t)LAGUNA_MOE_USED * LAGUNA_MOE_DIM * sizeof(float), NULL },
        { "selected", selected_t, sizeof(selected), NULL },
        { "router", router_t, sizeof(router), NULL },
        { "input", x, sizeof(input), NULL },
    };
    memcpy(snapshots, initial_snapshots, sizeof(snapshots));
    for (size_t i = 0; i < sizeof(snapshots) / sizeof(snapshots[0]); i++) {
        if (!laguna_capture_tensor_snapshot(&snapshots[i])) goto cleanup;
    }
    if (setenv("DS4_CUDA_VALIDATE_SELECTED", "1", 1) != 0) goto cleanup;
    validation_enabled = 1;
    (void)cudaGetLastError();
    if (ds4_gpu_glm_routed_moe_batch_tensor(
            out.view, mid, model, model_size,
            0u, projection_bytes, 2u * projection_bytes,
            12u, 12u, 12u,
            expert_bytes, row_bytes, expert_bytes, row_bytes,
            expert_bytes, row_bytes,
            LAGUNA_MOE_DIM, LAGUNA_MOE_DIM, LAGUNA_MOE_DIM,
            selected_t, router_t, LAGUNA_MOE_EXPERTS, LAGUNA_MOE_USED,
            0u, x, 1u, LAGUNA_MOE_USED * LAGUNA_MOE_DIM, true)) {
        fprintf(stderr, "routed-moe/selected-validation: accepted invalid selected ID\n");
        goto cleanup;
    }
    if (cudaDeviceSynchronize() != cudaSuccess || cudaGetLastError() != cudaSuccess) {
        fprintf(stderr, "routed-moe/selected-validation: rejection left CUDA error\n");
        goto cleanup;
    }
    for (size_t i = 0; i < sizeof(snapshots) / sizeof(snapshots[0]); i++) {
        if (!laguna_tensor_matches_snapshot(&snapshots[i], "routed-moe/selected-validation")) {
            goto cleanup;
        }
    }
    rc = 0;
cleanup:
    if (validation_enabled) (void)unsetenv("DS4_CUDA_VALIDATE_SELECTED");
    for (size_t i = 0; i < sizeof(snapshots) / sizeof(snapshots[0]); i++) {
        free(snapshots[i].snapshot);
    }
    laguna_guarded_output_free(&out);
    ds4_gpu_tensor_free(x);
    ds4_gpu_tensor_free(router_t);
    ds4_gpu_tensor_free(selected_t);
    ds4_gpu_tensor_free(mid);
    free(model);
    return rc;
}

static int run_routed_moe_cases(void) {
    const uint64_t row_bytes = sizeof(laguna_q4k_block);
    const uint64_t expert_bytes = LAGUNA_MOE_DIM * row_bytes;
    const uint64_t projection_bytes = LAGUNA_MOE_EXPERTS * expert_bytes;
    const uint64_t model_size = 3u * projection_bytes;
    unsigned char *model = (unsigned char *)calloc(1, (size_t)model_size);
    float one_input[LAGUNA_MOE_DIM], growth_input[3u * LAGUNA_MOE_DIM];
    float reuse_input[LAGUNA_MOE_DIM];
    float one_reference[LAGUNA_MOE_DIM], growth_reference[3u * LAGUNA_MOE_DIM];
    float reuse_reference[LAGUNA_MOE_DIM];
    int32_t one_selected[LAGUNA_MOE_USED], growth_selected[3u * LAGUNA_MOE_USED];
    int32_t reuse_selected[LAGUNA_MOE_USED];
    float one_router[LAGUNA_MOE_USED], growth_router[3u * LAGUNA_MOE_USED];
    float reuse_router[LAGUNA_MOE_USED];
    ds4_gpu_tensor *mid = NULL, *selected_t = NULL, *router_t = NULL, *x = NULL;
    laguna_guarded_output one_output = {0}, growth_output = {0};
    laguna_tensor_snapshot snapshots[2] = {0};
    int rc = 1;
    if (run_routed_moe_selected_validation_case() != 0) return 1;
    if (!model) goto cleanup;
    laguna_moe_desc routed = {
        .gate_offset = 0u, .up_offset = projection_bytes,
        .down_offset = 2u * projection_bytes,
        .gate_type = 12u, .up_type = 12u, .down_type = 12u,
        .gate_expert_bytes = expert_bytes, .gate_row_bytes = row_bytes,
        .up_expert_bytes = expert_bytes, .up_row_bytes = row_bytes,
        .down_expert_bytes = expert_bytes, .down_row_bytes = row_bytes,
    };
    for (uint32_t projection = 0; projection < 3u; projection++) {
        laguna_q4k_block *matrix = (laguna_q4k_block *)(model + projection * projection_bytes);
        for (uint32_t expert = 0; expert < LAGUNA_MOE_EXPERTS; expert++) {
            laguna_fill_q4k_matrix(matrix + (uint64_t)expert * LAGUNA_MOE_DIM,
                                   expert, projection);
        }
    }
    laguna_fill_routed_case(one_input, one_selected, one_router, 1u, 1u);
    laguna_fill_routed_case(growth_input, growth_selected, growth_router, 3u, 7u);
    laguna_fill_routed_case(reuse_input, reuse_selected, reuse_router, 1u, 19u);
    laguna_reference_routed_moe(one_reference, model, &routed,
                                one_selected, one_router, one_input, 1u);
    laguna_reference_routed_moe(growth_reference, model, &routed,
                                growth_selected, growth_router, growth_input, 3u);
    laguna_reference_routed_moe(reuse_reference, model, &routed,
                                reuse_selected, reuse_router, reuse_input, 1u);
    if (!laguna_reference_signal("one", one_reference, LAGUNA_MOE_DIM) ||
        !laguna_reference_signal("growth", growth_reference,
                                 3u * LAGUNA_MOE_DIM) ||
        !laguna_reference_signal("reuse", reuse_reference, LAGUNA_MOE_DIM)) {
        goto cleanup;
    }
    if (!ds4_gpu_set_model_map(model, model_size)) goto cleanup;
    mid = ds4_gpu_tensor_alloc(3u * LAGUNA_MOE_USED * LAGUNA_MOE_DIM * sizeof(float));
    selected_t = ds4_gpu_tensor_alloc(3u * LAGUNA_MOE_USED * sizeof(int32_t));
    router_t = ds4_gpu_tensor_alloc(3u * LAGUNA_MOE_USED * sizeof(float));
    x = ds4_gpu_tensor_alloc(3u * LAGUNA_MOE_DIM * sizeof(float));
    if (!mid || !selected_t || !router_t || !x ||
        !laguna_guarded_output_init(&one_output, LAGUNA_MOE_DIM) ||
        !laguna_guarded_output_init(&growth_output, 3u * LAGUNA_MOE_DIM)) {
        goto cleanup;
    }
    if (!ds4_gpu_tensor_write(selected_t, 0, one_selected, sizeof(one_selected)) ||
        !ds4_gpu_tensor_write(router_t, 0, one_router, sizeof(one_router)) ||
        !ds4_gpu_tensor_write(x, 0, one_input, sizeof(one_input)) ||
        !laguna_guarded_output_prepare(&one_output) ||
        !laguna_glm_routed_moe_q4_call(one_output.view, mid, model, model_size,
                                       projection_bytes, expert_bytes, row_bytes,
                                       selected_t, router_t, x, 1u, true)) {
        fprintf(stderr, "routed-moe/one: public GLM bridge returned failure\n");
        goto cleanup;
    }
    if (cudaDeviceSynchronize() != cudaSuccess ||
        !laguna_guarded_output_check("one", &one_output, one_reference, 1u)) goto cleanup;

    if (!ds4_gpu_tensor_write(selected_t, 0, growth_selected, sizeof(growth_selected)) ||
        !ds4_gpu_tensor_write(router_t, 0, growth_router, sizeof(growth_router)) ||
        !ds4_gpu_tensor_write(x, 0, growth_input, sizeof(growth_input)) ||
        !laguna_guarded_output_prepare(&growth_output) ||
        !laguna_glm_routed_moe_q4_call(growth_output.view, mid, model, model_size,
                                       projection_bytes, expert_bytes, row_bytes,
                                       selected_t, router_t, x, 3u, true)) {
        fprintf(stderr, "routed-moe/growth: public GLM bridge returned failure\n");
        goto cleanup;
    }
    if (cudaDeviceSynchronize() != cudaSuccess ||
        !laguna_guarded_output_check("growth", &growth_output, growth_reference, 3u)) goto cleanup;

    if (!ds4_gpu_tensor_write(selected_t, 0, reuse_selected, sizeof(reuse_selected)) ||
        !ds4_gpu_tensor_write(router_t, 0, reuse_router, sizeof(reuse_router)) ||
        !ds4_gpu_tensor_write(x, 0, reuse_input, sizeof(reuse_input)) ||
        !laguna_guarded_output_prepare(&one_output) ||
        !laguna_glm_routed_moe_q4_call(one_output.view, mid, model, model_size,
                                       projection_bytes, expert_bytes, row_bytes,
                                       selected_t, router_t, x, 1u, true)) {
        fprintf(stderr, "routed-moe/reuse: public GLM bridge returned failure\n");
        goto cleanup;
    }
    if (cudaDeviceSynchronize() != cudaSuccess ||
        !laguna_guarded_output_check("reuse", &one_output, reuse_reference, 1u)) goto cleanup;

    {
        float mutation_input[LAGUNA_MOE_DIM];
        float base_reference[LAGUNA_MOE_DIM];
        float mutation_reference[LAGUNA_MOE_DIM];
        int32_t base_selected[LAGUNA_MOE_USED];
        int32_t mutation_selected[LAGUNA_MOE_USED];
        float mutation_router[LAGUNA_MOE_USED];
        laguna_fill_routed_case(mutation_input, base_selected, mutation_router,
                                1u, 31u);
        memcpy(mutation_selected, base_selected, sizeof(mutation_selected));
        mutation_selected[0] =
            (mutation_selected[0] + 8) % (int32_t)LAGUNA_MOE_EXPERTS;
        laguna_reference_routed_moe(base_reference, model, &routed,
                                    base_selected, mutation_router,
                                    mutation_input, 1u);
        laguna_reference_routed_moe(mutation_reference, model, &routed,
                                    mutation_selected, mutation_router,
                                    mutation_input, 1u);
        if (laguna_max_delta(base_reference, mutation_reference,
                             LAGUNA_MOE_DIM) < 0.02f) {
            fprintf(stderr,
                    "routed-moe/selected-mutation: reference delta is too weak\n");
            goto cleanup;
        }
        if (!laguna_reference_signal("selected-mutation", mutation_reference,
                                     LAGUNA_MOE_DIM) ||
            !ds4_gpu_tensor_write(selected_t, 0, mutation_selected,
                                  sizeof(mutation_selected)) ||
            !ds4_gpu_tensor_write(router_t, 0, mutation_router,
                                  sizeof(mutation_router)) ||
            !ds4_gpu_tensor_write(x, 0, mutation_input,
                                  sizeof(mutation_input)) ||
            !laguna_guarded_output_prepare(&one_output) ||
            !laguna_glm_routed_moe_q4_call(
                    one_output.view, mid, model, model_size,
                    projection_bytes, expert_bytes, row_bytes,
                    selected_t, router_t, x, 1u, true)) {
            fprintf(stderr,
                    "routed-moe/selected-mutation: public GLM bridge returned failure\n");
            goto cleanup;
        }
        if (cudaDeviceSynchronize() != cudaSuccess ||
            !laguna_guarded_output_check("selected-mutation", &one_output,
                                         mutation_reference, 1u)) goto cleanup;
    }

    {
        float mutation_input[LAGUNA_MOE_DIM];
        float base_reference[LAGUNA_MOE_DIM];
        float mutation_reference[LAGUNA_MOE_DIM];
        int32_t mutation_selected[LAGUNA_MOE_USED];
        float base_router[LAGUNA_MOE_USED];
        float mutation_router[LAGUNA_MOE_USED];
        laguna_fill_routed_case(mutation_input, mutation_selected, base_router,
                                1u, 43u);
        memcpy(mutation_router, base_router, sizeof(mutation_router));
        mutation_router[1] += 1.0f;
        laguna_reference_routed_moe(base_reference, model, &routed,
                                    mutation_selected, base_router,
                                    mutation_input, 1u);
        laguna_reference_routed_moe(mutation_reference, model, &routed,
                                    mutation_selected, mutation_router,
                                    mutation_input, 1u);
        if (laguna_max_delta(base_reference, mutation_reference,
                             LAGUNA_MOE_DIM) < 0.02f) {
            fprintf(stderr,
                    "routed-moe/router-mutation: reference delta is too weak\n");
            goto cleanup;
        }
        if (!laguna_reference_signal("router-mutation", mutation_reference,
                                     LAGUNA_MOE_DIM) ||
            !ds4_gpu_tensor_write(selected_t, 0, mutation_selected,
                                  sizeof(mutation_selected)) ||
            !ds4_gpu_tensor_write(router_t, 0, mutation_router,
                                  sizeof(mutation_router)) ||
            !ds4_gpu_tensor_write(x, 0, mutation_input,
                                  sizeof(mutation_input)) ||
            !laguna_guarded_output_prepare(&one_output) ||
            !laguna_glm_routed_moe_q4_call(
                    one_output.view, mid, model, model_size,
                    projection_bytes, expert_bytes, row_bytes,
                    selected_t, router_t, x, 1u, true)) {
            fprintf(stderr,
                    "routed-moe/router-mutation: public GLM bridge returned failure\n");
            goto cleanup;
        }
        if (cudaDeviceSynchronize() != cudaSuccess ||
            !laguna_guarded_output_check("router-mutation", &one_output,
                                         mutation_reference, 1u)) goto cleanup;
    }

    if (!laguna_guarded_output_prepare(&one_output)) goto cleanup;
    snapshots[0] = (laguna_tensor_snapshot){
        "output", one_output.base,
        (uint64_t)(LAGUNA_MOE_DIM + 2u * LAGUNA_MOE_GUARD) * sizeof(float), NULL };
    snapshots[1] = (laguna_tensor_snapshot){ "input", x, sizeof(reuse_input), NULL };
    for (size_t i = 0; i < sizeof(snapshots) / sizeof(snapshots[0]); i++) {
        if (!laguna_capture_tensor_snapshot(&snapshots[i])) goto cleanup;
    }
    (void)cudaGetLastError();
    if (laguna_glm_routed_moe_q4_call(one_output.view, mid, model, model_size,
                                      projection_bytes, expert_bytes, row_bytes,
                                      selected_t, router_t, x, 1u, false)) {
        fprintf(stderr, "routed-moe/non-resident: public GLM bridge accepted snapshot call\n");
        goto cleanup;
    }
    if (cudaDeviceSynchronize() != cudaSuccess || cudaGetLastError() != cudaSuccess) {
        fprintf(stderr, "routed-moe/non-resident: rejection left CUDA error\n");
        goto cleanup;
    }
    for (size_t i = 0; i < sizeof(snapshots) / sizeof(snapshots[0]); i++) {
        if (!laguna_tensor_matches_snapshot(&snapshots[i], "routed-moe/non-resident")) {
            goto cleanup;
        }
    }
    rc = 0;
cleanup:
    for (size_t i = 0; i < sizeof(snapshots) / sizeof(snapshots[0]); i++) {
        free(snapshots[i].snapshot);
    }
    laguna_guarded_output_free(&growth_output);
    laguna_guarded_output_free(&one_output);
    ds4_gpu_tensor_free(x);
    ds4_gpu_tensor_free(router_t);
    ds4_gpu_tensor_free(selected_t);
    ds4_gpu_tensor_free(mid);
    free(model);
    return rc;
}

static void usage(const char *program) {
    fprintf(stderr, "usage: %s --case norm-rope|decode-attention|prefill-attention|routed-moe|all\n", program);
}

int main(int argc, char **argv) {
    if (argc != 3 || strcmp(argv[1], "--case") != 0 ||
        (strcmp(argv[2], "norm-rope") != 0 &&
         strcmp(argv[2], "decode-attention") != 0 &&
         strcmp(argv[2], "prefill-attention") != 0 &&
         strcmp(argv[2], "routed-moe") != 0 &&
         strcmp(argv[2], "all") != 0)) {
        usage(argv[0]);
        return 2;
    }
    const bool run_norm = strcmp(argv[2], "norm-rope") == 0 ||
        strcmp(argv[2], "all") == 0;
    const bool run_decode = strcmp(argv[2], "decode-attention") == 0 ||
        strcmp(argv[2], "all") == 0;
    const bool run_prefill = strcmp(argv[2], "prefill-attention") == 0 ||
        strcmp(argv[2], "all") == 0;
    const bool run_routed_moe = strcmp(argv[2], "routed-moe") == 0 ||
        strcmp(argv[2], "all") == 0;
    if (run_f32_to_f16_reference_cases() != 0) return 1;
    if (!ds4_gpu_init()) {
        fprintf(stderr, "norm-rope: ds4_gpu_init failed\n");
        return 1;
    }
    float *weights = NULL;
    static const laguna_norm_rope_case cases[] = {
        { "global-pos0", 1, 48, 64, 0, 8192, 500000.0f,
          1.0f / 32.0f, 1.0f, 1.0f, 32.0f, 1.0f, 0u },
        { "global-yarn-frontier", 4, 48, 64, 8191, 8192, 500000.0f,
          1.0f / 32.0f, 1.0f, 1.0f, 32.0f, 1.0f, 0u },
        { "swa-ring-frontier", 4, 72, 128, 510, 262144, 10000.0f,
          1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0u },
        { "k-global-prefill-one", 1, 8, 64, 8193, 8192, 500000.0f,
          1.0f / 32.0f, 1.0f, 1.0f, 32.0f, 1.0f,
          128u * sizeof(float) },
        { "k-swa-prefill-many", 4, 8, 128, 510, 262144, 10000.0f,
          1.0f, 0.0f, 1.0f, 0.0f, 0.0f,
          128u * sizeof(float) },
    };
    int rc = 0;
    if (run_norm) {
        const uint32_t head_dim = 128;
        weights = (float *)malloc(2u * head_dim * sizeof(*weights));
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
        for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
            if (run_norm_rope_case(
                    weights, 2u * head_dim * sizeof(*weights),
                    weights + cases[i].weight_offset / sizeof(*weights),
                    &cases[i]) != 0) {
                rc = 1;
            }
        }
    static const laguna_norm_rope_case qk_cases[] = {
        { "qk-global-yarn-frontier", 4, 48, 64, 8191, 8192, 500000.0f,
          1.0f / 32.0f, 1.0f, 1.0f, 32.0f, 1.0f, 0u },
        { "qk-swa-ring-frontier", 4, 72, 128, 510, 262144, 10000.0f,
          1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0u },
    };
        for (size_t i = 0; i < sizeof(qk_cases) / sizeof(qk_cases[0]); i++) {
            if (run_qk_norm_rope_case(weights, weights + head_dim,
                                      &qk_cases[i]) != 0) {
                rc = 1;
            }
        }
        if (run_qk_metric_dilution_case() != 0) rc = 1;
    }
    if (run_decode && run_decode_attention_cases() != 0) {
        rc = 1;
    }
    if (run_prefill && run_prefill_attention_cases() != 0) {
        rc = 1;
    }
    if (run_routed_moe && run_routed_moe_cases() != 0) {
        rc = 1;
    }
    /* The model-map registration pins weights until GPU cleanup unregisters it. */
    ds4_gpu_cleanup();
    free(weights);
    return rc;
}
