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
} laguna_norm_rope_case;

typedef struct {
    const char *name;
    const float *actual;
    const float *reference;
    uint64_t count;
} laguna_parity_span;

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
    if (!heads || !key_cache || !value_cache || !q || !k || !v || !gate ||
        !ds4_gpu_tensor_write(heads, 0, q_host, q_count * sizeof(*q_host)) ||
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
    if (ds4_gpu_laguna_store_attention_tensor(
            NULL, key_cache, value_cache, q, k, v, gate, c->pos, c->cache_cap,
            c->key_start, c->key_count, c->n_head, c->n_head_kv, head_dim,
            scale) || cudaDeviceSynchronize() != cudaSuccess ||
        cudaGetLastError() != cudaSuccess) {
        fprintf(stderr, "decode-attention/%s: rejection left CUDA error\n",
                c->family);
        goto cleanup;
    }

    ds4_gpu_tensor *short_q = ds4_gpu_tensor_view(q, 0, q_count * sizeof(*q_host) - 1u);
    if (!short_q || ds4_gpu_laguna_store_attention_tensor(heads, key_cache, value_cache, short_q, k, v, gate, c->pos, c->cache_cap, c->key_start, c->key_count, c->n_head, c->n_head_kv, head_dim, scale)) {
        fprintf(stderr, "decode-attention/%s: accepted undersized q view\n", c->family); ds4_gpu_tensor_free(short_q); goto cleanup;
    }
    ds4_gpu_tensor_free(short_q);
    ds4_gpu_tensor *short_heads = ds4_gpu_tensor_view(heads, 0, q_count * sizeof(*actual) - 1u);
    ds4_gpu_tensor *short_key = ds4_gpu_tensor_view(key_cache, 0, cache_values * sizeof(*key_expected) - 1u);
    ds4_gpu_tensor *short_value = ds4_gpu_tensor_view(value_cache, 0, cache_values * sizeof(*value_expected) - 1u);
    if (!short_heads || !short_key || !short_value ||
        ds4_gpu_laguna_store_attention_tensor(short_heads, key_cache, value_cache, q, k, v, gate, c->pos, c->cache_cap, c->key_start, c->key_count, c->n_head, c->n_head_kv, head_dim, scale) ||
        ds4_gpu_laguna_store_attention_tensor(heads, short_key, value_cache, q, k, v, gate, c->pos, c->cache_cap, c->key_start, c->key_count, c->n_head, c->n_head_kv, head_dim, scale) ||
        ds4_gpu_laguna_store_attention_tensor(heads, key_cache, short_value, q, k, v, gate, c->pos, c->cache_cap, c->key_start, c->key_count, c->n_head, c->n_head_kv, head_dim, scale)) {
        fprintf(stderr, "decode-attention/%s: accepted undersized output/cache view\n", c->family);
        ds4_gpu_tensor_free(short_value); ds4_gpu_tensor_free(short_key); ds4_gpu_tensor_free(short_heads); goto cleanup;
    }
    ds4_gpu_tensor_free(short_value); ds4_gpu_tensor_free(short_key); ds4_gpu_tensor_free(short_heads);
    ds4_gpu_tensor *short_k = ds4_gpu_tensor_view(k, 0, kv_width * sizeof(*k_host) - 1u);
    ds4_gpu_tensor *short_v = ds4_gpu_tensor_view(v, 0, kv_width * sizeof(*v_host) - 1u);
    ds4_gpu_tensor *short_gate = ds4_gpu_tensor_view(gate, 0, (uint64_t)c->n_head * sizeof(*gate_host) - 1u);
    if (!short_k || !short_v || !short_gate ||
        ds4_gpu_laguna_store_attention_tensor(heads, key_cache, value_cache, q, short_k, v, gate, c->pos, c->cache_cap, c->key_start, c->key_count, c->n_head, c->n_head_kv, head_dim, scale) ||
        ds4_gpu_laguna_store_attention_tensor(heads, key_cache, value_cache, q, k, short_v, gate, c->pos, c->cache_cap, c->key_start, c->key_count, c->n_head, c->n_head_kv, head_dim, scale) ||
        ds4_gpu_laguna_store_attention_tensor(heads, key_cache, value_cache, q, k, v, short_gate, c->pos, c->cache_cap, c->key_start, c->key_count, c->n_head, c->n_head_kv, head_dim, scale)) {
        fprintf(stderr, "decode-attention/%s: accepted undersized input view\n", c->family);
        ds4_gpu_tensor_free(short_gate); ds4_gpu_tensor_free(short_v); ds4_gpu_tensor_free(short_k); goto cleanup;
    }
    ds4_gpu_tensor_free(short_gate); ds4_gpu_tensor_free(short_v); ds4_gpu_tensor_free(short_k);
    if (!ds4_gpu_tensor_read(heads, 0, actual, q_count * sizeof(*actual)) ||
        memcmp(actual, q_host, q_count * sizeof(*actual)) ||
        !ds4_gpu_tensor_read(q, 0, actual, q_count * sizeof(*actual)) ||
        memcmp(actual, q_host, q_count * sizeof(*actual)) ||
        !ds4_gpu_tensor_read(k, 0, actual, kv_width * sizeof(*actual)) ||
        memcmp(actual, k_host, kv_width * sizeof(*actual)) ||
        !ds4_gpu_tensor_read(v, 0, actual, kv_width * sizeof(*actual)) ||
        memcmp(actual, v_host, kv_width * sizeof(*actual)) ||
        !ds4_gpu_tensor_read(gate, 0, actual,
                             (uint64_t)c->n_head * sizeof(*actual)) ||
        memcmp(actual, gate_host, (uint64_t)c->n_head * sizeof(*actual)) ||
        !ds4_gpu_tensor_read(key_cache, 0, key_expected,
                             cache_values * sizeof(*key_expected)) ||
        memcmp(key_expected, key_actual, cache_values * sizeof(*key_expected)) ||
        !ds4_gpu_tensor_read(value_cache, 0, value_expected,
                             cache_values * sizeof(*value_expected)) ||
        memcmp(value_expected, value_actual, cache_values * sizeof(*value_expected))) {
        fprintf(stderr, "decode-attention/%s: rejection changed caller buffer\n", c->family);
        goto cleanup;
    }
    for (uint64_t i = 0; i < kv_width; i++) {
        key_expected[current_base + i] = reference_f32_to_f16(k_host[i]);
        value_expected[current_base + i] = reference_f32_to_f16(v_host[i]);
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

static void usage(const char *program) {
    fprintf(stderr, "usage: %s --case norm-rope|decode-attention|all\n", program);
}

int main(int argc, char **argv) {
    if (argc != 3 || strcmp(argv[1], "--case") != 0 ||
        (strcmp(argv[2], "norm-rope") != 0 &&
         strcmp(argv[2], "decode-attention") != 0 &&
         strcmp(argv[2], "all") != 0)) {
        usage(argv[0]);
        return 2;
    }
    const bool run_norm = strcmp(argv[2], "norm-rope") == 0 ||
        strcmp(argv[2], "all") == 0;
    const bool run_decode = strcmp(argv[2], "decode-attention") == 0 ||
        strcmp(argv[2], "all") == 0;
    if (run_f32_to_f16_reference_cases() != 0) return 1;
    if (!ds4_gpu_init()) {
        fprintf(stderr, "norm-rope: ds4_gpu_init failed\n");
        return 1;
    }
    float *weights = NULL;
    static const laguna_norm_rope_case cases[] = {
        { "global-pos0", 1, 48, 64, 0, 8192, 500000.0f,
          1.0f / 32.0f, 1.0f, 1.0f, 32.0f, 1.0f },
        { "global-yarn-frontier", 4, 48, 64, 8191, 8192, 500000.0f,
          1.0f / 32.0f, 1.0f, 1.0f, 32.0f, 1.0f },
        { "swa-ring-frontier", 4, 72, 128, 510, 262144, 10000.0f,
          1.0f, 0.0f, 1.0f, 0.0f, 0.0f },
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
            if (run_norm_rope_case(weights, &cases[i]) != 0) rc = 1;
        }
    static const laguna_norm_rope_case qk_cases[] = {
        { "qk-global-yarn-frontier", 4, 48, 64, 8191, 8192, 500000.0f,
          1.0f / 32.0f, 1.0f, 1.0f, 32.0f, 1.0f },
        { "qk-swa-ring-frontier", 4, 72, 128, 510, 262144, 10000.0f,
          1.0f, 0.0f, 1.0f, 0.0f, 0.0f },
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
    /* The model-map registration pins weights until GPU cleanup unregisters it. */
    ds4_gpu_cleanup();
    free(weights);
    return rc;
}
