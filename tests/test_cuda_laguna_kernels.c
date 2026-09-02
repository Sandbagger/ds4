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

static int8_t poolside_mmq_activation_q(uint32_t token, uint32_t block,
                                        uint32_t i) {
    if (i == 0u) return 127;
    if (i == 1u) return -127;
    uint32_t x = token * 0x9e3779b9u;
    x ^= block * 0x85ebca6bu;
    x ^= i * 0xc2b2ae35u;
    x ^= x >> 16u;
    return (int8_t)((int)(x % 255u) - 127);
}

static int8_t poolside_mmq_weight_q(uint32_t row, uint32_t block,
                                    uint32_t i) {
    uint32_t x = row * 0x27d4eb2du;
    x ^= block * 0x165667b1u;
    x ^= i * 0x9e3779b9u;
    x ^= x >> 15u;
    return (int8_t)((int)(x % 255u) - 127);
}

static uint16_t poolside_mmq_weight_scale_bits(uint32_t row,
                                                uint32_t block) {
    const int exponent = (int)((row * 7u + block * 11u) % 11u) - 7;
    const uint16_t mantissa = (uint16_t)(1u +
        ((row * 73u + block * 151u + 19u) % 1023u));
    return (uint16_t)(((uint16_t)(exponent + 15) << 10u) | mantissa);
}

static float poolside_mmq_activation_scale(uint32_t token,
                                            uint32_t block) {
    const int exponent = (int)((token * 5u + block * 7u) % 13u) - 9;
    return ldexpf(1.0f, exponent);
}

static int32_t poolside_mmq_dot(const int8_t *weight, uint32_t token,
                                uint32_t block) {
    int32_t dot = 0;
    for (uint32_t i = 0; i < 32u; i++) {
        dot += (int32_t)weight[i] *
               (int32_t)poolside_mmq_activation_q(token, block, i);
    }
    return dot;
}

static int run_poolside_q8_projection_case(unsigned char **retained_model) {
    const uint32_t in_dim = 3072u;
    const uint32_t rounding_out_dim = 64u;
    const uint32_t mmq_out_dim = 128u;
    const uint32_t n_tokens = 22u;
    const uint32_t q8_block = 32u;
    const uint32_t blocks = in_dim / q8_block;
    const uint64_t rounding_q8_bytes =
        (uint64_t)rounding_out_dim * blocks * 34u;
    const uint64_t mmq_weight_offset = rounding_q8_bytes;
    const uint64_t mmq_q8_bytes = (uint64_t)mmq_out_dim * blocks * 34u;
    const uint64_t rms_weight_offset = mmq_weight_offset + mmq_q8_bytes;
    const uint64_t n1_weight_offset =
        rms_weight_offset + (uint64_t)in_dim * sizeof(float);
    const uint64_t n1_q8_bytes = (uint64_t)mmq_out_dim * blocks * 34u;
    const uint64_t model_bytes = n1_weight_offset + n1_q8_bytes;
    const uint64_t input_count = (uint64_t)n_tokens * in_dim;
    const uint64_t rounding_output_count =
        (uint64_t)n_tokens * rounding_out_dim;
    const uint64_t output_count = (uint64_t)n_tokens * mmq_out_dim;
    const float amax = 0x1.36573ap-8f;
    const float selected = 0x1.0a5afep-9f;
    /* Poolside D4: d_inv=127/amax, q=round(x*d_inv), d=1/d_inv.
     * Pin the resulting float so -ffast-math cannot reassociate it. */
    const float expected = 0x1.0ccc8ep-9f;
    const float tolerance = 2.0e-7f;
    unsigned char *model = NULL;
    float *input = NULL;
    float *actual = NULL;
    float *rms_input = NULL;
    float *rms_actual = NULL;
    ds4_gpu_tensor *x = NULL;
    ds4_gpu_tensor *out = NULL;
    ds4_gpu_tensor *rms_x = NULL;
    ds4_gpu_tensor *rms_out = NULL;
    int registered = 0;
    int rc = 1;

    if (!retained_model || *retained_model) {
        fprintf(stderr, "poolside-q8: invalid retained-model slot\n");
        return 1;
    }
    model = (unsigned char *)calloc(1, (size_t)model_bytes);
    input = (float *)calloc((size_t)input_count, sizeof(*input));
    actual = (float *)malloc((size_t)output_count * sizeof(*actual));
    rms_input = (float *)malloc((size_t)in_dim * sizeof(*rms_input));
    rms_actual = (float *)malloc((size_t)in_dim * sizeof(*rms_actual));
    if (!model || !input || !actual || !rms_input || !rms_actual) {
        fprintf(stderr, "poolside-q8: synthetic allocation failed\n");
        goto cleanup;
    }

    const uint16_t unit_scale = reference_f32_to_f16(1.0f);
    for (uint32_t row = 0; row < rounding_out_dim; row++) {
        for (uint32_t block = 0; block < blocks; block++) {
            unsigned char *weight = model +
                ((uint64_t)row * blocks + block) * 34u;
            memcpy(weight, &unit_scale, sizeof(unit_scale));
            if (block == 0u) {
                ((int8_t *)(weight + sizeof(unit_scale)))[1] = 1;
            }
        }
    }
    for (uint32_t row = 0; row < mmq_out_dim; row++) {
        for (uint32_t block = 0; block < blocks; block++) {
            unsigned char *weight = model + mmq_weight_offset +
                ((uint64_t)row * blocks + block) * 34u;
            const uint16_t scale =
                poolside_mmq_weight_scale_bits(row, block);
            memcpy(weight, &scale, sizeof(scale));
            int8_t *values = (int8_t *)(weight + sizeof(scale));
            for (uint32_t i = 0; i < q8_block - 1u; i++) {
                values[i] = poolside_mmq_weight_q(row, block, i);
            }
            int found = 0;
            for (int candidate = -127; candidate <= 127 && !found;
                 candidate++) {
                values[q8_block - 1u] = (int8_t)candidate;
                found = 1;
                for (uint32_t token = 0; token < n_tokens; token++) {
                    if (poolside_mmq_dot(values, token, block) == 0) {
                        found = 0;
                        break;
                    }
                }
            }
            if (!found) {
                fprintf(stderr,
                        "poolside-q8: cannot make nonzero dot row=%u block=%u\n",
                        row, block);
                goto cleanup;
            }
        }
    }
    for (uint32_t i = 0; i < in_dim; i++) {
        const uint32_t sign = ((i * 13u) & 1u) << 31;
        const uint32_t exponent = 120u + ((i * 7u + 3u) % 15u);
        const uint32_t mantissa = (i * 2654435761u) & 0x007fffffu;
        const uint32_t input_bits = sign | (exponent << 23) | mantissa;
        const uint32_t weight_mantissa =
            (i * 2246822519u) & 0x003fffffu;
        const uint32_t weight_bits = (127u << 23) | weight_mantissa;
        memcpy(&rms_input[i], &input_bits, sizeof(input_bits));
        memcpy(model + rms_weight_offset + (uint64_t)i * sizeof(weight_bits),
               &weight_bits, sizeof(weight_bits));
    }
    for (uint32_t row = 0; row < mmq_out_dim; row++) {
        for (uint32_t block = 0; block < blocks; block++) {
            unsigned char *weight = model + n1_weight_offset +
                ((uint64_t)row * blocks + block) * 34u;
            const uint16_t scale =
                poolside_mmq_weight_scale_bits(row, block);
            memcpy(weight, &scale, sizeof(scale));
            int8_t *values = (int8_t *)(weight + sizeof(scale));
            for (uint32_t i = 0; i < q8_block; i++) {
                values[i] = poolside_mmq_weight_q(row, block, i);
            }
            if (poolside_mmq_dot(values, 0u, block) == 0) {
                values[0] += values[0] < 127 ? 1 : -1;
            }
        }
    }
    for (uint32_t token = 0; token < n_tokens; token++) {
        for (uint32_t block = 0; block < blocks; block++) {
            float *values = input + (uint64_t)token * in_dim + block * q8_block;
            values[0] = amax;
            values[1] = selected;
        }
    }

    if (!ds4_gpu_set_model_map(model, model_bytes)) {
        fprintf(stderr, "poolside-q8: synthetic model map setup failed\n");
        goto cleanup;
    }
    registered = 1;
    *retained_model = model;
    rms_x = ds4_gpu_tensor_alloc((uint64_t)in_dim * sizeof(*rms_input));
    rms_out = ds4_gpu_tensor_alloc((uint64_t)in_dim * sizeof(*rms_actual));
    if (!rms_x || !rms_out ||
        !ds4_gpu_tensor_write(rms_x, 0, rms_input,
                              (uint64_t)in_dim * sizeof(*rms_input)) ||
        !ds4_gpu_rms_norm_weight_rows_tensor(
            rms_out, rms_x, model, model_bytes, rms_weight_offset,
            in_dim, 1u, 1.0e-6f) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(rms_out, 0, rms_actual,
                             (uint64_t)in_dim * sizeof(*rms_actual))) {
        fprintf(stderr, "poolside-q8: CUDA RMSNorm setup failed\n");
        goto cleanup;
    }
    /* Pinned from Poolside llama.cpp 04b2b72 on GB10: its n=3072 fused
     * RMSNorm+weight kernel uses 1024 threads and a two-stage warp tree.
     * Exact bits matter because one ULP here can flip the following Q8_1
     * activation quantization at a rounding boundary. */
    static const struct {
        uint32_t index;
        uint32_t expected_bits;
    } rms_oracle[] = {
        { 0u,    0x3a919cd6u }, { 1u,    0xbe8c106cu },
        { 31u,   0xbe671747u }, { 32u,   0x3ac4fb45u },
        { 255u,  0xbb08f16bu }, { 256u,  0x3e99a2c0u },
        { 1023u, 0xbe140d8fu }, { 1024u, 0x3a2e2f42u },
        { 2047u, 0xbcdff3b8u }, { 2048u, 0x4086ea6eu },
        { 3071u, 0xbc23c9cau },
    };
    for (size_t i = 0; i < sizeof(rms_oracle) / sizeof(rms_oracle[0]); i++) {
        uint32_t actual_bits = 0;
        memcpy(&actual_bits, &rms_actual[rms_oracle[i].index],
               sizeof(actual_bits));
        if (actual_bits != rms_oracle[i].expected_bits) {
            fprintf(stderr,
                    "poolside-q8: RMSNorm[%u]=0x%08x expected=0x%08x\n",
                    rms_oracle[i].index, actual_bits,
                    rms_oracle[i].expected_bits);
            goto cleanup;
        }
    }
    x = ds4_gpu_tensor_alloc(input_count * sizeof(*input));
    out = ds4_gpu_tensor_alloc(output_count * sizeof(*actual));
    if (!x || !out ||
        !ds4_gpu_tensor_write(x, 0, input, input_count * sizeof(*input)) ||
        !ds4_gpu_matmul_q8_0_tensor(out, model, model_bytes, 0u,
                                    in_dim, rounding_out_dim, x, n_tokens) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(out, 0, actual,
                             rounding_output_count * sizeof(*actual))) {
        fprintf(stderr, "poolside-q8: CUDA projection failed\n");
        goto cleanup;
    }
    for (uint64_t i = 0; i < rounding_output_count; i++) {
        const float error = fabsf(actual[i] - expected);
        if (!isfinite(actual[i]) || error > tolerance) {
            fprintf(stderr,
                    "poolside-q8: output[%llu]=%.9g expected=%.9g "
                    "abs=%.9g tolerance=%.9g\n",
                    (unsigned long long)i, actual[i], expected, error,
                    tolerance);
            goto cleanup;
        }
    }

    for (uint32_t token = 0; token < n_tokens; token++) {
        for (uint32_t block = 0; block < blocks; block++) {
            const float scale =
                poolside_mmq_activation_scale(token, block);
            float *values = input +
                (uint64_t)token * in_dim + (uint64_t)block * q8_block;
            for (uint32_t i = 0; i < q8_block; i++) {
                values[i] = scale *
                    (float)poolside_mmq_activation_q(token, block, i);
            }
        }
    }
    if (!ds4_gpu_tensor_write(x, 0, input,
                              input_count * sizeof(*input)) ||
        !ds4_gpu_matmul_q8_0_poolside_tensor(
            out, model, model_bytes, mmq_weight_offset,
            in_dim, mmq_out_dim, x, n_tokens) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(out, 0, actual,
                             output_count * sizeof(*actual))) {
        fprintf(stderr, "poolside-q8: CUDA MMQ oracle projection failed\n");
        goto cleanup;
    }
    /* Actual Poolside llama.cpp 04b2b72 backend output on GB10 for this
     * K=3072, M=128, N=22 all-block-nonzero fixture. The result includes
     * MMQ tile/register order and stream-K fixup association; a natural
     * block-order or shared-tree reconstruction is not equivalent. */
    static const struct {
        uint32_t token;
        uint32_t row;
        uint32_t expected_bits;
    } mmq_oracle[] = {
        { 0u,  0u,   0xc5fed1aeu }, { 1u,  1u,   0xc908d915u },
        { 7u,  7u,   0xc82216b0u }, { 8u,  8u,   0xc5c2eb89u },
        { 15u, 63u,  0x467ca1d8u }, { 16u, 64u,  0xca0b5968u },
        { 21u, 127u, 0xcad7f502u },
    };
    for (size_t i = 0; i < sizeof(mmq_oracle) / sizeof(mmq_oracle[0]); i++) {
        const uint64_t index =
            (uint64_t)mmq_oracle[i].token * mmq_out_dim +
            mmq_oracle[i].row;
        uint32_t actual_bits = 0;
        memcpy(&actual_bits, &actual[index], sizeof(actual_bits));
        if (actual_bits != mmq_oracle[i].expected_bits) {
            fprintf(stderr,
                    "poolside-q8: MMQ token=%u row=%u got=0x%08x "
                    "expected=0x%08x\n",
                    mmq_oracle[i].token, mmq_oracle[i].row,
                    actual_bits, mmq_oracle[i].expected_bits);
            goto cleanup;
        }
    }
    uint64_t mmq_hash = UINT64_C(1469598103934665603);
    for (uint64_t i = 0; i < output_count; i++) {
        uint32_t bits = 0;
        memcpy(&bits, &actual[i], sizeof(bits));
        for (uint32_t byte = 0; byte < 4u; byte++) {
            mmq_hash ^= (bits >> (byte * 8u)) & 0xffu;
            mmq_hash *= UINT64_C(1099511628211);
        }
    }
    if (mmq_hash != UINT64_C(0xc7f239af7daa6dc7)) {
        fprintf(stderr,
                "poolside-q8: MMQ hash=0x%016llx expected=0xc7f239af7daa6dc7\n",
                (unsigned long long)mmq_hash);
        goto cleanup;
    }

    if (!ds4_gpu_matmul_q8_0_poolside_tensor(
            out, model, model_bytes, n1_weight_offset,
            in_dim, mmq_out_dim, x, 1u) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(out, 0, actual,
                             mmq_out_dim * sizeof(*actual))) {
        fprintf(stderr, "poolside-q8: CUDA MMVQ oracle projection failed\n");
        goto cleanup;
    }
    /* Actual Poolside llama.cpp 04b2b72 MMVQ output on GB10.  Nsight shows
     * that M=128 and Laguna's M=100352 output head use the same quantize_q8_1
     * and four-warp mul_mat_vec_q topology; only grid.x changes. */
    static const struct {
        uint32_t row;
        uint32_t expected_bits;
    } mmvq_oracle[] = {
        { 0u,   0xc5b97414u }, { 1u,   0xc9efea4cu },
        { 7u,   0xc93e1019u }, { 8u,   0x4684d66bu },
        { 15u,  0xc97f6f81u }, { 16u,  0xc6a0e206u },
        { 31u,  0xc9f6e8d4u }, { 32u,  0xc87f397fu },
        { 63u,  0x465786abu }, { 64u,  0x494a7e55u },
        { 95u,  0x48534e3au }, { 96u,  0xc7008d0eu },
        { 127u, 0x4b16b145u },
    };
    for (size_t i = 0; i < sizeof(mmvq_oracle) / sizeof(mmvq_oracle[0]); i++) {
        uint32_t actual_bits = 0;
        memcpy(&actual_bits, &actual[mmvq_oracle[i].row],
               sizeof(actual_bits));
        if (actual_bits != mmvq_oracle[i].expected_bits) {
            fprintf(stderr,
                    "poolside-q8: MMVQ row=%u got=0x%08x expected=0x%08x\n",
                    mmvq_oracle[i].row, actual_bits,
                    mmvq_oracle[i].expected_bits);
            goto cleanup;
        }
    }
    uint64_t mmvq_hash = UINT64_C(1469598103934665603);
    for (uint64_t i = 0; i < mmq_out_dim; i++) {
        uint32_t bits = 0;
        memcpy(&bits, &actual[i], sizeof(bits));
        for (uint32_t byte = 0; byte < 4u; byte++) {
            mmvq_hash ^= (bits >> (byte * 8u)) & 0xffu;
            mmvq_hash *= UINT64_C(1099511628211);
        }
    }
    if (mmvq_hash != UINT64_C(0xa4a437a74a744282)) {
        fprintf(stderr,
                "poolside-q8: MMVQ hash=0x%016llx "
                "expected=0xa4a437a74a744282\n",
                (unsigned long long)mmvq_hash);
        goto cleanup;
    }
    rc = 0;

cleanup:
    ds4_gpu_tensor_free(rms_out);
    ds4_gpu_tensor_free(rms_x);
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(x);
    free(rms_actual);
    free(rms_input);
    free(actual);
    free(input);
    if (!registered) free(model);
    return rc;
}

/* Model-shaped Q8_0 prefill coverage. Laguna's 22-token short case enters the
 * tensor-core path that scalar and tiny-batch kernel fixtures do not reach. */
typedef struct {
    uint16_t d;
    int8_t qs[32];
} laguna_q8_0_block;

_Static_assert(sizeof(laguna_q8_0_block) == 34u, "Q8_0 block layout changed");

static void laguna_fill_q8_0_block(
        laguna_q8_0_block *block, uint32_t row, uint32_t column_block) {
    const float scale = 0.0015f + 0.000031f * (float)((row * 7u + column_block * 11u) % 29u);
    block->d = reference_f32_to_f16(scale);
    for (uint32_t i = 0; i < 32u; i++) {
        block->qs[i] = (int8_t)((int32_t)((row * 17u + column_block * 13u + i * 19u) % 255u) - 127);
    }
}

static void laguna_quantize_q8_0_reference(
        int8_t *quantized, float *scales, const float *input,
        uint32_t n_tokens, uint32_t in_dim) {
    const uint32_t blocks = in_dim / 32u;
    for (uint32_t token = 0; token < n_tokens; token++) {
        for (uint32_t block = 0; block < blocks; block++) {
            const float *src = input + (uint64_t)token * in_dim + block * 32u;
            float max_abs = 0.0f;
            for (uint32_t i = 0; i < 32u; i++) {
                max_abs = fmaxf(max_abs, fabsf(src[i]));
            }
            const float id = max_abs != 0.0f ? 127.0f / max_abs : 0.0f;
            const float d = id != 0.0f ? 1.0f / id : 0.0f;
            scales[(uint64_t)token * blocks + block] = d;
            for (uint32_t i = 0; i < 32u; i++) {
                int value = (int)roundf(src[i] * id);
                if (value > 127) value = 127;
                if (value < -128) value = -128;
                quantized[((uint64_t)token * blocks + block) * 32u + i] = (int8_t)value;
            }
        }
    }
}

static void laguna_q8_0_matmul_reference(
        float *output, const laguna_q8_0_block *weights,
        const int8_t *quantized, const float *scales,
        uint32_t n_tokens, uint32_t in_dim, uint32_t out_dim) {
    const uint32_t blocks = in_dim / 32u;
    for (uint32_t token = 0; token < n_tokens; token++) {
        for (uint32_t row = 0; row < out_dim; row++) {
            float sum = 0.0f;
            for (uint32_t block = 0; block < blocks; block++) {
                const laguna_q8_0_block *weight = weights + (uint64_t)row * blocks + block;
                const int8_t *input = quantized + ((uint64_t)token * blocks + block) * 32u;
                int32_t dot = 0;
                for (uint32_t i = 0; i < 32u; i++) {
                    dot += (int32_t)weight->qs[i] * (int32_t)input[i];
                }
                sum += reference_f16_to_f32(weight->d) *
                    scales[(uint64_t)token * blocks + block] * (float)dot;
            }
            output[(uint64_t)token * out_dim + row] = sum;
        }
    }
}

static int run_q8_matmul_prefill_case(unsigned char **retained_model) {
    const uint32_t n_tokens = 22u;
    const uint32_t in_dim = 3072u;
    const uint32_t out_dim = 73u;
    const uint32_t blocks = in_dim / 32u;
    const uint64_t weight_count = (uint64_t)out_dim * blocks;
    const uint64_t input_count = (uint64_t)n_tokens * in_dim;
    const uint64_t output_count = (uint64_t)n_tokens * out_dim;
    laguna_q8_0_block *weights = calloc((size_t)weight_count, sizeof(*weights));
    float *input = malloc((size_t)input_count * sizeof(*input));
    int8_t *quantized = malloc((size_t)input_count * sizeof(*quantized));
    float *scales = malloc((size_t)n_tokens * blocks * sizeof(*scales));
    float *reference = malloc((size_t)output_count * sizeof(*reference));
    float *actual = malloc((size_t)output_count * sizeof(*actual));
    ds4_gpu_tensor *input_tensor = NULL;
    ds4_gpu_tensor *output_tensor = NULL;
    int registered = 0;
    int rc = 1;

    if (!retained_model || *retained_model) {
        fprintf(stderr, "q8-matmul: invalid retained-model slot\n");
        goto cleanup;
    }
    if (!weights || !input || !quantized || !scales || !reference || !actual) {
        fprintf(stderr, "q8-matmul: fixture allocation failed\n");
        goto cleanup;
    }
    for (uint32_t row = 0; row < out_dim; row++) {
        for (uint32_t block = 0; block < blocks; block++) {
            laguna_fill_q8_0_block(weights + (uint64_t)row * blocks + block,
                                   row, block);
        }
    }
    for (uint64_t i = 0; i < input_count; i++) {
        input[i] = (float)((int32_t)((i * 23u + (i / in_dim) * 41u) % 257u) - 128) /
            (31.0f + (float)(i % 7u));
    }
    laguna_quantize_q8_0_reference(quantized, scales, input, n_tokens, in_dim);
    laguna_q8_0_matmul_reference(reference, weights, quantized, scales,
                                 n_tokens, in_dim, out_dim);

    if (!ds4_gpu_set_model_map(weights, weight_count * sizeof(*weights))) {
        fprintf(stderr, "q8-matmul: model map setup failed\n");
        goto cleanup;
    }
    registered = 1;
    input_tensor = ds4_gpu_tensor_alloc(input_count * sizeof(*input));
    output_tensor = ds4_gpu_tensor_alloc(output_count * sizeof(*actual));
    if (!input_tensor || !output_tensor ||
        !ds4_gpu_tensor_write(input_tensor, 0, input,
                              input_count * sizeof(*input)) ||
        !ds4_gpu_matmul_q8_0_tensor(output_tensor, weights,
                                    weight_count * sizeof(*weights), 0u,
                                    in_dim, out_dim, input_tensor, n_tokens)) {
        fprintf(stderr, "q8-matmul: CUDA setup or wrapper failed\n");
        goto cleanup;
    }
    if (cudaDeviceSynchronize() != cudaSuccess ||
        !ds4_gpu_tensor_read(output_tensor, 0, actual,
                             output_count * sizeof(*actual))) {
        fprintf(stderr, "q8-matmul: CUDA completion or read failed\n");
        goto cleanup;
    }

    double square_error = 0.0;
    float max_abs = 0.0f;
    uint64_t worst = 0u;
    int finite = 1;
    for (uint64_t i = 0; i < output_count; i++) {
        if (!isfinite(actual[i]) || !isfinite(reference[i])) {
            finite = 0;
            worst = i;
            continue;
        }
        const float error = fabsf(actual[i] - reference[i]);
        if (error > max_abs) {
            max_abs = error;
            worst = i;
        }
        square_error += (double)error * error;
    }
    const double rms = finite ? sqrt(square_error / (double)output_count) : INFINITY;
    if (!finite || max_abs > 2.0e-3f || rms > 5.0e-4) {
        fprintf(stderr,
                "q8-matmul/model-prefill: max_abs=%g rms=%g token=%llu row=%llu actual=%g reference=%g\n",
                (double)max_abs, rms,
                (unsigned long long)(worst / out_dim),
                (unsigned long long)(worst % out_dim),
                (double)actual[worst], (double)reference[worst]);
        goto cleanup;
    }
    rc = 0;

cleanup:
    ds4_gpu_tensor_free(output_tensor);
    ds4_gpu_tensor_free(input_tensor);
    free(actual);
    free(reference);
    free(scales);
    free(quantized);
    free(input);
    if (registered) {
        *retained_model = (unsigned char *)weights;
    } else {
        free(weights);
    }
    return rc;
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
    const float theta_scale = powf(freq_base, -2.0f / (float)n_rot);
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
                head[i] = rms * head[i] * weights[i];
            }
            const uint32_t half_rot = n_rot / 2u;
            for (uint32_t i = 0; i < half_rot; i++) {
                const uint32_t rel_i0 = i * 2u;
                const float theta_extrap = (float)(pos0 + token) *
                    powf(theta_scale, rel_i0 / 2.0f);
                const float theta_interp = freq_scale * theta_extrap;
                float theta = theta_interp;
                float mscale = attn_factor;
                if (ext_factor != 0.0f) {
                    const float mix = yarn_ramp(corr_dims[0], corr_dims[1],
                                                 (int)rel_i0) * ext_factor;
                    theta = theta_interp * (1.0f - mix) + theta_extrap * mix;
                    mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
                }
                const float c = cosf(theta) * mscale;
                const float s = sinf(theta) * mscale;
                const float x0 = head[i];
                const float x1 = head[i + half_rot];
                head[i] = x0 * c - x1 * s;
                head[i + half_rot] = x0 * s + x1 * c;
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
    uint32_t n_heads;
    uint32_t head_dim;
    float max_abs_limit;
    float rms_limit;
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
        const uint64_t values_per_token =
            (uint64_t)spans[span].n_heads * spans[span].head_dim;
        if (spans[span].n_heads == 0u || spans[span].head_dim == 0u ||
            spans[span].count == 0u ||
            spans[span].count % values_per_token != 0u) {
            if (report) {
                fprintf(stderr, "%s/%s: invalid parity shape\n", case_name,
                        spans[span].name);
            }
            return 0;
        }
        for (uint64_t base = 0; base < spans[span].count;
             base += spans[span].head_dim) {
            double square_error = 0.0;
            float max_abs = 0.0f;
            uint32_t max_dim = 0u;
            const uint64_t flat_head = base / spans[span].head_dim;
            const uint64_t token = flat_head / spans[span].n_heads;
            const uint64_t head = flat_head % spans[span].n_heads;
            for (uint32_t dim = 0; dim < spans[span].head_dim; dim++) {
                const uint64_t i = base + dim;
                if (!isfinite(spans[span].actual[i]) ||
                    !isfinite(spans[span].reference[i])) {
                    if (report) {
                        fprintf(stderr,
                                "%s/%s: non-finite actual/reference token=%llu head=%llu dim=%u actual=%g reference=%g\n",
                                case_name, spans[span].name,
                                (unsigned long long)token,
                                (unsigned long long)head, dim,
                                (double)spans[span].actual[i],
                                (double)spans[span].reference[i]);
                    }
                    return 0;
                }
                const float error = fabsf(spans[span].actual[i] -
                                          spans[span].reference[i]);
                if (error > max_abs) {
                    max_abs = error;
                    max_dim = dim;
                }
                square_error += (double)error * error;
            }
            const double rms_error =
                sqrt(square_error / (double)spans[span].head_dim);
            if (max_abs > spans[span].max_abs_limit ||
                rms_error > spans[span].rms_limit) {
                if (report) {
                    const uint64_t max_index = base + max_dim;
                    fprintf(stderr,
                            "%s/%s: parity token=%llu head=%llu dim=%u actual=%g reference=%g max_abs=%g limit=%g local_rms=%g limit=%g\n",
                            case_name, spans[span].name,
                            (unsigned long long)token,
                            (unsigned long long)head, max_dim,
                            (double)spans[span].actual[max_index],
                            (double)spans[span].reference[max_index],
                            (double)max_abs,
                            (double)spans[span].max_abs_limit,
                            rms_error, (double)spans[span].rms_limit);
                }
                return 0;
            }
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
        { "q", q_actual, q_reference, 6, 1, 6, 2.0e-4f, 5.0e-5f },
        { "k", k_actual, k_reference, 1, 1, 1, 2.0e-4f, 5.0e-5f },
    };
    if (laguna_parity_spans_within_limits("qk-metric-dilution", spans,
                                          sizeof(spans) / sizeof(spans[0]), 0)) {
        fprintf(stderr,
                "qk-metric-dilution: accepted a K-only RMS error above the limit\n");
        return 1;
    }
    return 0;
}

static int run_grouped_metric_contract_cases(void) {
    float actual[2u * 128u] = {0};
    float reference[2u * 128u] = {0};
    int rc = 0;
    for (uint32_t i = 0; i < 128u; i++) actual[i] = 6.0e-5f;
    const laguna_parity_span grouped = {
        "grouped-rms", actual, reference, 2u * 128u,
        2u, 128u, 2.0e-4f, 5.0e-5f,
    };
    if (laguna_parity_spans_within_limits(
            "metric-contract", &grouped, 1u, 0)) {
        fprintf(stderr,
                "metric-contract: whole-tensor RMS hid a bad head\n");
        rc = 1;
    }
    memset(actual, 0, sizeof(actual));
    reference[0] = NAN;
    const laguna_parity_span nonfinite = {
        "reference-finite", actual, reference, 2u * 128u,
        2u, 128u, 2.0e-4f, 5.0e-5f,
    };
    if (laguna_parity_spans_within_limits(
            "metric-contract", &nonfinite, 1u, 0)) {
        fprintf(stderr,
                "metric-contract: accepted a non-finite host reference\n");
        rc = 1;
    }
    return rc;
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
        /* Host libm cannot reproduce CUDA --use_fast_math powf/cosf at long
         * YaRN positions.  The frozen Poolside case below is byte-exact; keep
         * these synthetic cases as wider shape and boundary checks. */
        const float max_abs_limit =
            c->ext_factor != 0.0f && c->pos0 != 0u ? 2.5e-3f : 2.0e-4f;
        const float rms_limit =
            c->ext_factor != 0.0f && c->pos0 != 0u ? 2.5e-4f : 5.0e-5f;
        const laguna_parity_span span = {
            "x", actual, reference, count, c->n_head, head_dim,
            max_abs_limit, rms_limit,
        };
        if (!laguna_parity_spans_within_limits(
                c->name, &span, 1u, 1)) {
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
        { "q", q_actual, q_reference, q_count, n_q_head, head_dim,
          c->ext_factor != 0.0f && c->pos0 != 0u ? 2.5e-3f : 2.0e-4f,
          c->ext_factor != 0.0f && c->pos0 != 0u ? 2.5e-4f : 5.0e-5f },
        { "k", k_actual, k_reference, k_count, n_k_head, head_dim,
          c->ext_factor != 0.0f && c->pos0 != 0u ? 2.5e-3f : 2.0e-4f,
          c->ext_factor != 0.0f && c->pos0 != 0u ? 2.5e-4f : 5.0e-5f },
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

#define LAGUNA_DECODE_VEC_FIXTURE_DIR \
    "tests/test-vectors/laguna-attention-decode-vec"

static int laguna_read_decode_vec_frozen(
        const char *name, float *values, uint64_t count) {
    char path[512];
    const int path_length = snprintf(
        path, sizeof(path), "%s/%s", LAGUNA_DECODE_VEC_FIXTURE_DIR, name);
    const uint64_t expected_bytes = count * sizeof(*values);
    if (path_length < 0 || (size_t)path_length >= sizeof(path) ||
        sizeof(float) != 4u) {
        fprintf(stderr, "decode-attention-frozen: invalid fixture contract\n");
        return 0;
    }
    const uint16_t endian_probe = 1u;
    if (*(const uint8_t *)&endian_probe != 1u) {
        fprintf(stderr, "decode-attention-frozen: little-endian host is required\n");
        return 0;
    }
    FILE *stream = fopen(path, "rb");
    if (!stream || fseek(stream, 0, SEEK_END) != 0) {
        fprintf(stderr, "decode-attention-frozen: cannot open %s\n", path);
        if (stream) fclose(stream);
        return 0;
    }
    const long file_bytes = ftell(stream);
    if (file_bytes < 0 || (uint64_t)file_bytes != expected_bytes) {
        fprintf(stderr, "decode-attention-frozen: %s bytes=%ld expected=%llu\n",
                path, file_bytes, (unsigned long long)expected_bytes);
        fclose(stream);
        return 0;
    }
    rewind(stream);
    const size_t read_count = fread(values, sizeof(*values), (size_t)count, stream);
    const int ok = read_count == count && !ferror(stream);
    fclose(stream);
    if (!ok) {
        fprintf(stderr, "decode-attention-frozen: cannot read %s\n", path);
    }
    return ok;
}

static int run_decode_attention_case(
        const laguna_decode_attention_case *c, const char *frozen_name,
        float frozen_max_abs, float frozen_rms) {
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

    const float q_input_scale = 1.0f / 23.0f;
    for (uint64_t i = 0; i < q_count; i++) {
        q_host[i] = (float)((int)((i * 17u) % 41u) - 20) * q_input_scale;
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
    if (frozen_name &&
        !laguna_read_decode_vec_frozen(frozen_name, reference, q_count)) {
        goto cleanup;
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
    const laguna_parity_span span = {
        "heads", actual, reference, q_count, c->n_head, head_dim,
        frozen_name ? frozen_max_abs : 1.0e-3f,
        frozen_name ? frozen_rms : 2.0e-4f,
    };
    if (!laguna_parity_spans_within_limits(
            c->family, &span, 1u, 1)) {
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
            if (run_decode_attention_case(&c, NULL, 0.0f, 0.0f) != 0) rc = 1;
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
            if (run_decode_attention_case(&c, NULL, 0.0f, 0.0f) != 0) rc = 1;
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
        if (run_decode_attention_case(
                &mutation_cases[i], NULL, 0.0f, 0.0f) != 0) rc = 1;
    }
    const laguna_decode_attention_case frozen_vec = {
        "gqa6-poolside-vec-token513", 48u, 8u, 768u, 512u, 0u, 513u,
        100.0f, false,
    };
    if (run_decode_attention_case(
            &frozen_vec, "gqa6-token513-gate100.f32", 5.0e-7f,
            1.0e-7f) != 0) rc = 1;
    return rc;
}

static float prefill_fixture_value(uint32_t logical_pos, uint64_t element,
                                   uint32_t multiplier, uint32_t modulus,
                                   int32_t bias) {
    return (float)((int32_t)((logical_pos * multiplier + element * 17u) % modulus) + bias) /
        (float)(modulus - 1u);
}

typedef struct {
    const char *name;
    uint32_t pos0;
    uint32_t n_tokens;
    uint32_t cache_cap;
    uint32_t n_head;
    uint32_t n_head_kv;
} laguna_prefill_case;

typedef struct {
    float *heads;
    uint16_t *key_cache;
    uint16_t *value_cache;
} laguna_prefill_result;

static int laguna_prefill_result_init(
        laguna_prefill_result *result, uint64_t q_count,
        uint64_t cache_count) {
    result->heads = (float *)malloc((size_t)q_count * sizeof(*result->heads));
    result->key_cache =
        (uint16_t *)malloc((size_t)cache_count * sizeof(*result->key_cache));
    result->value_cache =
        (uint16_t *)malloc((size_t)cache_count * sizeof(*result->value_cache));
    return result->heads && result->key_cache && result->value_cache;
}

static void laguna_prefill_result_free(laguna_prefill_result *result) {
    free(result->value_cache);
    free(result->key_cache);
    free(result->heads);
    memset(result, 0, sizeof(*result));
}

static int laguna_run_prefill_once(
        const laguna_prefill_case *c, const char *arm,
        const float *q_host, const float *k_host, const float *v_host,
        const float *gate_host, const uint16_t *key_initial,
        const uint16_t *value_initial, laguna_prefill_result *result) {
    const uint32_t head_dim = 128u;
    const uint64_t q_count =
        (uint64_t)c->n_tokens * c->n_head * head_dim;
    const uint64_t kv_count =
        (uint64_t)c->n_tokens * c->n_head_kv * head_dim;
    const uint64_t cache_count =
        (uint64_t)c->cache_cap * c->n_head_kv * head_dim;
    const uint64_t gate_count = (uint64_t)c->n_tokens * c->n_head;
    const float scale = 1.0f / sqrtf((float)head_dim);
    ds4_gpu_tensor *heads = NULL, *key_cache = NULL, *value_cache = NULL;
    ds4_gpu_tensor *staged_key = NULL, *staged_value = NULL;
    ds4_gpu_tensor *q = NULL, *k = NULL, *v = NULL, *gate = NULL;
    int rc = 1;

    heads = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    key_cache = ds4_gpu_tensor_alloc(cache_count * sizeof(uint16_t));
    value_cache = ds4_gpu_tensor_alloc(cache_count * sizeof(uint16_t));
    staged_key = ds4_gpu_tensor_alloc(kv_count * sizeof(uint16_t));
    staged_value = ds4_gpu_tensor_alloc(kv_count * sizeof(uint16_t));
    q = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    k = ds4_gpu_tensor_alloc(kv_count * sizeof(float));
    v = ds4_gpu_tensor_alloc(kv_count * sizeof(float));
    gate = ds4_gpu_tensor_alloc(gate_count * sizeof(float));
    for (uint64_t i = 0; i < q_count; i++) result->heads[i] = NAN;
    if (!heads || !key_cache || !value_cache || !staged_key ||
        !staged_value || !q || !k || !v || !gate ||
        !ds4_gpu_tensor_write(heads, 0, result->heads,
                              q_count * sizeof(float)) ||
        !ds4_gpu_tensor_write(key_cache, 0, key_initial,
                              cache_count * sizeof(uint16_t)) ||
        !ds4_gpu_tensor_write(value_cache, 0, value_initial,
                              cache_count * sizeof(uint16_t)) ||
        !ds4_gpu_tensor_write(q, 0, q_host, q_count * sizeof(float)) ||
        !ds4_gpu_tensor_write(k, 0, k_host, kv_count * sizeof(float)) ||
        !ds4_gpu_tensor_write(v, 0, v_host, kv_count * sizeof(float)) ||
        !ds4_gpu_tensor_write(gate, 0, gate_host,
                              gate_count * sizeof(float))) {
        fprintf(stderr, "prefill-attention/%s/%s: tensor setup failed\n",
                c->name, arm);
        goto cleanup;
    }
    const int wrapper_ok = ds4_gpu_laguna_attention_prefill_tensor(
            heads, key_cache, value_cache, staged_key, staged_value,
            q, k, v, gate, c->pos0, c->n_tokens, c->cache_cap,
            c->n_head, c->n_head_kv, head_dim, scale);
    const cudaError_t sync = cudaDeviceSynchronize();
    if (!wrapper_ok || sync != cudaSuccess) {
        fprintf(stderr,
                "prefill-attention/%s/%s: wrapper=%d sync=%s\n",
                c->name, arm, wrapper_ok, cudaGetErrorString(sync));
        goto cleanup;
    }
    if (!ds4_gpu_tensor_read(heads, 0, result->heads,
                             q_count * sizeof(float)) ||
        !ds4_gpu_tensor_read(key_cache, 0, result->key_cache,
                             cache_count * sizeof(uint16_t)) ||
        !ds4_gpu_tensor_read(value_cache, 0, result->value_cache,
                             cache_count * sizeof(uint16_t))) {
        fprintf(stderr, "prefill-attention/%s/%s: output read failed\n",
                c->name, arm);
        goto cleanup;
    }
    rc = 0;
cleanup:
    ds4_gpu_tensor_free(gate);
    ds4_gpu_tensor_free(v);
    ds4_gpu_tensor_free(k);
    ds4_gpu_tensor_free(q);
    ds4_gpu_tensor_free(staged_value);
    ds4_gpu_tensor_free(staged_key);
    ds4_gpu_tensor_free(value_cache);
    ds4_gpu_tensor_free(key_cache);
    ds4_gpu_tensor_free(heads);
    return rc;
}

static int laguna_run_decode_sequence(
        const laguna_prefill_case *c, const float *q_host,
        const float *k_host, const float *v_host, const float *gate_host,
        const uint16_t *key_initial, const uint16_t *value_initial,
        laguna_prefill_result *result) {
    const uint32_t head_dim = 128u;
    const uint64_t q_row_count = (uint64_t)c->n_head * head_dim;
    const uint64_t kv_row_count = (uint64_t)c->n_head_kv * head_dim;
    const uint64_t q_count = (uint64_t)c->n_tokens * q_row_count;
    const uint64_t kv_count = (uint64_t)c->n_tokens * kv_row_count;
    const uint64_t cache_count = (uint64_t)c->cache_cap * kv_row_count;
    const uint64_t gate_count = (uint64_t)c->n_tokens * c->n_head;
    const float scale = 1.0f / sqrtf((float)head_dim);
    ds4_gpu_tensor *heads = NULL, *key_cache = NULL, *value_cache = NULL;
    ds4_gpu_tensor *q = NULL, *k = NULL, *v = NULL, *gate = NULL;
    ds4_gpu_tensor **head_rows = NULL, **q_rows = NULL, **k_rows = NULL;
    ds4_gpu_tensor **v_rows = NULL, **gate_rows = NULL;
    int rc = 1;

    head_rows = (ds4_gpu_tensor **)calloc(c->n_tokens, sizeof(*head_rows));
    q_rows = (ds4_gpu_tensor **)calloc(c->n_tokens, sizeof(*q_rows));
    k_rows = (ds4_gpu_tensor **)calloc(c->n_tokens, sizeof(*k_rows));
    v_rows = (ds4_gpu_tensor **)calloc(c->n_tokens, sizeof(*v_rows));
    gate_rows = (ds4_gpu_tensor **)calloc(c->n_tokens, sizeof(*gate_rows));
    heads = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    key_cache = ds4_gpu_tensor_alloc(cache_count * sizeof(uint16_t));
    value_cache = ds4_gpu_tensor_alloc(cache_count * sizeof(uint16_t));
    q = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    k = ds4_gpu_tensor_alloc(kv_count * sizeof(float));
    v = ds4_gpu_tensor_alloc(kv_count * sizeof(float));
    gate = ds4_gpu_tensor_alloc(gate_count * sizeof(float));
    for (uint64_t i = 0; i < q_count; i++) result->heads[i] = NAN;
    if (!head_rows || !q_rows || !k_rows || !v_rows || !gate_rows ||
        !heads || !key_cache || !value_cache || !q || !k || !v || !gate ||
        !ds4_gpu_tensor_write(heads, 0, result->heads,
                              q_count * sizeof(float)) ||
        !ds4_gpu_tensor_write(key_cache, 0, key_initial,
                              cache_count * sizeof(uint16_t)) ||
        !ds4_gpu_tensor_write(value_cache, 0, value_initial,
                              cache_count * sizeof(uint16_t)) ||
        !ds4_gpu_tensor_write(q, 0, q_host, q_count * sizeof(float)) ||
        !ds4_gpu_tensor_write(k, 0, k_host, kv_count * sizeof(float)) ||
        !ds4_gpu_tensor_write(v, 0, v_host, kv_count * sizeof(float)) ||
        !ds4_gpu_tensor_write(gate, 0, gate_host,
                              gate_count * sizeof(float))) {
        fprintf(stderr, "prefill-attention/%s/sequential: setup failed\n",
                c->name);
        goto cleanup;
    }
    for (uint32_t t = 0; t < c->n_tokens; t++) {
        head_rows[t] = ds4_gpu_tensor_view(
                heads, (uint64_t)t * q_row_count * sizeof(float),
                q_row_count * sizeof(float));
        q_rows[t] = ds4_gpu_tensor_view(
                q, (uint64_t)t * q_row_count * sizeof(float),
                q_row_count * sizeof(float));
        k_rows[t] = ds4_gpu_tensor_view(
                k, (uint64_t)t * kv_row_count * sizeof(float),
                kv_row_count * sizeof(float));
        v_rows[t] = ds4_gpu_tensor_view(
                v, (uint64_t)t * kv_row_count * sizeof(float),
                kv_row_count * sizeof(float));
        gate_rows[t] = ds4_gpu_tensor_view(
                gate, (uint64_t)t * c->n_head * sizeof(float),
                (uint64_t)c->n_head * sizeof(float));
        if (!head_rows[t] || !q_rows[t] || !k_rows[t] || !v_rows[t] ||
            !gate_rows[t]) {
            fprintf(stderr,
                    "prefill-attention/%s/sequential: row view failed\n",
                    c->name);
            goto cleanup;
        }
    }
    int wrappers_ok = 1;
    for (uint32_t t = 0; t < c->n_tokens; t++) {
        const uint32_t pos = c->pos0 + t;
        const uint32_t key_start =
            pos + 1u > c->cache_cap ? pos + 1u - c->cache_cap : 0u;
        const uint32_t key_count = pos - key_start + 1u;
        if (!ds4_gpu_laguna_store_attention_tensor(
                head_rows[t], key_cache, value_cache, q_rows[t], k_rows[t],
                v_rows[t], gate_rows[t], pos, c->cache_cap, key_start,
                key_count, c->n_head, c->n_head_kv, head_dim, scale)) {
            wrappers_ok = 0;
            break;
        }
    }
    const cudaError_t sync = cudaDeviceSynchronize();
    if (!wrappers_ok || sync != cudaSuccess) {
        fprintf(stderr,
                "prefill-attention/%s/sequential: wrapper=%d sync=%s\n",
                c->name, wrappers_ok, cudaGetErrorString(sync));
        goto cleanup;
    }
    if (!ds4_gpu_tensor_read(heads, 0, result->heads,
                             q_count * sizeof(float)) ||
        !ds4_gpu_tensor_read(key_cache, 0, result->key_cache,
                             cache_count * sizeof(uint16_t)) ||
        !ds4_gpu_tensor_read(value_cache, 0, result->value_cache,
                             cache_count * sizeof(uint16_t))) {
        fprintf(stderr,
                "prefill-attention/%s/sequential: output read failed\n",
                c->name);
        goto cleanup;
    }
    rc = 0;
cleanup:
    for (uint32_t t = 0; t < c->n_tokens; t++) {
        if (gate_rows) ds4_gpu_tensor_free(gate_rows[t]);
        if (v_rows) ds4_gpu_tensor_free(v_rows[t]);
        if (k_rows) ds4_gpu_tensor_free(k_rows[t]);
        if (q_rows) ds4_gpu_tensor_free(q_rows[t]);
        if (head_rows) ds4_gpu_tensor_free(head_rows[t]);
    }
    free(gate_rows);
    free(v_rows);
    free(k_rows);
    free(q_rows);
    free(head_rows);
    ds4_gpu_tensor_free(gate);
    ds4_gpu_tensor_free(v);
    ds4_gpu_tensor_free(k);
    ds4_gpu_tensor_free(q);
    ds4_gpu_tensor_free(value_cache);
    ds4_gpu_tensor_free(key_cache);
    ds4_gpu_tensor_free(heads);
    return rc;
}

static int run_prefill_contract_case(const laguna_prefill_case *c) {
    const uint32_t head_dim = 128u;
    const uint64_t q_row_count = (uint64_t)c->n_head * head_dim;
    const uint64_t kv_width = (uint64_t)c->n_head_kv * head_dim;
    const uint64_t q_count = (uint64_t)c->n_tokens * q_row_count;
    const uint64_t kv_count = (uint64_t)c->n_tokens * kv_width;
    const uint64_t cache_count = (uint64_t)c->cache_cap * kv_width;
    const uint64_t gate_count = (uint64_t)c->n_tokens * c->n_head;
    const float scale = 1.0f / sqrtf((float)head_dim);
    float *q_host = (float *)calloc((size_t)q_count, sizeof(*q_host));
    float *k_host = (float *)calloc((size_t)kv_count, sizeof(*k_host));
    float *v_host = (float *)calloc((size_t)kv_count, sizeof(*v_host));
    float *gate_host =
        (float *)calloc((size_t)gate_count, sizeof(*gate_host));
    float *reference =
        (float *)calloc((size_t)q_count, sizeof(*reference));
    float *scores =
        (float *)calloc((size_t)c->cache_cap, sizeof(*scores));
    float *mutated_k =
        (float *)calloc((size_t)kv_count, sizeof(*mutated_k));
    float *mutated_v =
        (float *)calloc((size_t)kv_count, sizeof(*mutated_v));
    uint16_t *staged_key =
        (uint16_t *)calloc((size_t)kv_count, sizeof(*staged_key));
    uint16_t *staged_value =
        (uint16_t *)calloc((size_t)kv_count, sizeof(*staged_value));
    uint16_t *key_initial =
        (uint16_t *)calloc((size_t)cache_count, sizeof(*key_initial));
    uint16_t *value_initial =
        (uint16_t *)calloc((size_t)cache_count, sizeof(*value_initial));
    uint16_t *key_expected =
        (uint16_t *)calloc((size_t)cache_count, sizeof(*key_expected));
    uint16_t *value_expected =
        (uint16_t *)calloc((size_t)cache_count, sizeof(*value_expected));
    laguna_prefill_result prefill = {0}, sequential = {0}, mutation = {0};
    int rc = 1;

    if (!q_host || !k_host || !v_host || !gate_host || !reference ||
        !scores || !mutated_k || !mutated_v || !staged_key ||
        !staged_value || !key_initial || !value_initial || !key_expected ||
        !value_expected ||
        !laguna_prefill_result_init(&prefill, q_count, cache_count) ||
        !laguna_prefill_result_init(&sequential, q_count, cache_count) ||
        !laguna_prefill_result_init(&mutation, q_count, cache_count)) {
        fprintf(stderr, "prefill-attention/%s: host allocation failed\n",
                c->name);
        goto cleanup;
    }

    const uint32_t prior_start = c->pos0 > c->cache_cap
        ? c->pos0 - c->cache_cap
        : 0u;
    for (uint32_t logical = prior_start; logical < c->pos0; logical++) {
        const uint64_t row = (uint64_t)(logical % c->cache_cap) * kv_width;
        for (uint64_t i = 0; i < kv_width; i++) {
            key_initial[row + i] = reference_f32_to_f16(
                    prefill_fixture_value(logical, i, 13u, 61u, -30));
            value_initial[row + i] = reference_f32_to_f16(
                    prefill_fixture_value(logical, i, 19u, 67u, -33));
        }
    }
    static const float mixed_gates[] = {
        -20.0f, -2.0f, 0.0f, 2.0f, 20.0f,
    };
    for (uint32_t t = 0; t < c->n_tokens; t++) {
        const uint32_t logical = c->pos0 + t;
        for (uint64_t i = 0; i < kv_width; i++) {
            const uint64_t src = (uint64_t)t * kv_width + i;
            k_host[src] =
                prefill_fixture_value(logical, i, 13u, 61u, -30);
            v_host[src] =
                prefill_fixture_value(logical, i, 19u, 67u, -33);
            staged_key[src] = reference_f32_to_f16(k_host[src]);
            staged_value[src] = reference_f32_to_f16(v_host[src]);
        }
        for (uint32_t h = 0; h < c->n_head; h++) {
            gate_host[(uint64_t)t * c->n_head + h] =
                mixed_gates[(3u * t + h) % 5u];
            for (uint32_t d = 0; d < head_dim; d++) {
                q_host[((uint64_t)t * c->n_head + h) * head_dim + d] =
                    prefill_fixture_value(logical + h, d, 7u, 59u, -29);
            }
        }
    }

    for (uint32_t t = 0; t < c->n_tokens; t++) {
        const uint32_t pos = c->pos0 + t;
        const uint32_t key_start =
            pos + 1u > c->cache_cap ? pos + 1u - c->cache_cap : 0u;
        const uint32_t key_count = pos - key_start + 1u;
        for (uint32_t h = 0; h < c->n_head; h++) {
            const uint32_t kv_head = h / (c->n_head / c->n_head_kv);
            float max_score = -INFINITY;
            float sum = 0.0f;
            for (uint32_t r = 0; r < key_count; r++) {
                const uint32_t logical = key_start + r;
                const uint16_t *key_row = logical < c->pos0
                    ? key_initial +
                        (uint64_t)(logical % c->cache_cap) * kv_width
                    : staged_key +
                        (uint64_t)(logical - c->pos0) * kv_width;
                float dot = 0.0f;
                for (uint32_t d = 0; d < head_dim; d++) {
                    dot += q_host[
                            ((uint64_t)t * c->n_head + h) * head_dim + d] *
                        reference_f16_to_f32(
                                key_row[(uint64_t)kv_head * head_dim + d]);
                }
                scores[r] = scale * dot;
                if (scores[r] > max_score) max_score = scores[r];
            }
            for (uint32_t r = 0; r < key_count; r++) {
                sum += expf(scores[r] - max_score);
            }
            const float gate_scale = reference_softplus(
                    gate_host[(uint64_t)t * c->n_head + h]);
            for (uint32_t d = 0; d < head_dim; d++) {
                float weighted = 0.0f;
                for (uint32_t r = 0; r < key_count; r++) {
                    const uint32_t logical = key_start + r;
                    const uint16_t *value_row = logical < c->pos0
                        ? value_initial +
                            (uint64_t)(logical % c->cache_cap) * kv_width
                        : staged_value +
                            (uint64_t)(logical - c->pos0) * kv_width;
                    weighted += expf(scores[r] - max_score) *
                        reference_f16_to_f32(
                                value_row[(uint64_t)kv_head * head_dim + d]);
                }
                reference[
                        ((uint64_t)t * c->n_head + h) * head_dim + d] =
                    weighted / sum * gate_scale;
            }
        }
    }

    memcpy(key_expected, key_initial,
           (size_t)cache_count * sizeof(*key_expected));
    memcpy(value_expected, value_initial,
           (size_t)cache_count * sizeof(*value_expected));
    const uint32_t commit_start = c->n_tokens > c->cache_cap
        ? c->n_tokens - c->cache_cap
        : 0u;
    for (uint32_t t = commit_start; t < c->n_tokens; t++) {
        const uint64_t dst =
            (uint64_t)((c->pos0 + t) % c->cache_cap) * kv_width;
        memcpy(key_expected + dst, staged_key + (uint64_t)t * kv_width,
               (size_t)kv_width * sizeof(*key_expected));
        memcpy(value_expected + dst,
               staged_value + (uint64_t)t * kv_width,
               (size_t)kv_width * sizeof(*value_expected));
    }

    if (laguna_run_prefill_once(
            c, "prefill", q_host, k_host, v_host, gate_host,
            key_initial, value_initial, &prefill) != 0 ||
        laguna_run_decode_sequence(
            c, q_host, k_host, v_host, gate_host,
            key_initial, value_initial, &sequential) != 0) {
        goto cleanup;
    }
    const laguna_parity_span spans[] = {
        { "prefill-host", prefill.heads, reference, q_count,
          c->n_head, head_dim, 1.0e-3f, 2.0e-4f },
        { "decode-host", sequential.heads, reference, q_count,
          c->n_head, head_dim, 1.0e-3f, 2.0e-4f },
        { "prefill-decode", prefill.heads, sequential.heads, q_count,
          c->n_head, head_dim, 1.0e-3f, 2.0e-4f },
    };
    if (!laguna_parity_spans_within_limits(
            c->name, spans, sizeof(spans) / sizeof(spans[0]), 1)) {
        goto cleanup;
    }
    if (memcmp(prefill.key_cache, key_expected,
               (size_t)cache_count * sizeof(*key_expected)) != 0 ||
        memcmp(prefill.value_cache, value_expected,
               (size_t)cache_count * sizeof(*value_expected)) != 0 ||
        memcmp(sequential.key_cache, key_expected,
               (size_t)cache_count * sizeof(*key_expected)) != 0 ||
        memcmp(sequential.value_cache, value_expected,
               (size_t)cache_count * sizeof(*value_expected)) != 0 ||
        memcmp(prefill.key_cache, sequential.key_cache,
               (size_t)cache_count * sizeof(*key_expected)) != 0 ||
        memcmp(prefill.value_cache, sequential.value_cache,
               (size_t)cache_count * sizeof(*value_expected)) != 0) {
        fprintf(stderr,
                "prefill-attention/%s: final cache differs from host/decode\n",
                c->name);
        goto cleanup;
    }

    for (uint32_t j = 0; j + 1u < c->n_tokens; j++) {
        memcpy(mutated_k, k_host, (size_t)kv_count * sizeof(*mutated_k));
        memcpy(mutated_v, v_host, (size_t)kv_count * sizeof(*mutated_v));
        const uint64_t first_changed = (uint64_t)(j + 1u) * kv_width;
        for (uint64_t i = first_changed; i < kv_count; i++) {
            mutated_k[i] += 0.75f + 0.001f * (float)(i % 17u);
            mutated_v[i] -= 0.50f + 0.001f * (float)(i % 13u);
        }
        char arm[64];
        snprintf(arm, sizeof(arm), "causal-after-%u", j);
        if (laguna_run_prefill_once(
                c, arm, q_host, mutated_k, mutated_v, gate_host,
                key_initial, value_initial, &mutation) != 0) {
            goto cleanup;
        }
        const uint64_t prefix_count = (uint64_t)(j + 1u) * q_row_count;
        if (memcmp(prefill.heads, mutation.heads,
                   (size_t)prefix_count * sizeof(float)) != 0) {
            fprintf(stderr,
                    "prefill-attention/%s/%s: later KV changed output prefix\n",
                    c->name, arm);
            goto cleanup;
        }
        if (memcmp(prefill.heads + prefix_count,
                   mutation.heads + prefix_count,
                   (size_t)(q_count - prefix_count) * sizeof(float)) == 0) {
            fprintf(stderr,
                    "prefill-attention/%s/%s: mutation did not change suffix\n",
                    c->name, arm);
            goto cleanup;
        }
    }
    rc = 0;
cleanup:
    laguna_prefill_result_free(&mutation);
    laguna_prefill_result_free(&sequential);
    laguna_prefill_result_free(&prefill);
    free(value_expected);
    free(key_expected);
    free(value_initial);
    free(key_initial);
    free(staged_value);
    free(staged_key);
    free(mutated_v);
    free(mutated_k);
    free(scores);
    free(reference);
    free(gate_host);
    free(v_host);
    free(k_host);
    free(q_host);
    return rc;
}

#define LAGUNA_ATTENTION_AUTO_FIXTURE_DIR \
    "tests/test-vectors/laguna-attention-auto"

static float *laguna_read_frozen_f32(const char *name, uint64_t count) {
    char path[512];
    const int path_length = snprintf(
        path, sizeof(path), "%s/%s", LAGUNA_ATTENTION_AUTO_FIXTURE_DIR, name);
    const uint64_t expected_bytes = count * sizeof(float);
    if (path_length < 0 || (size_t)path_length >= sizeof(path)) {
        fprintf(stderr, "prefill-attention-frozen: fixture path is too long\n");
        return NULL;
    }
    if (sizeof(float) != 4u) {
        fprintf(stderr, "prefill-attention-frozen: float32 host is required\n");
        return NULL;
    }
    const uint16_t endian_probe = 1u;
    if (*(const uint8_t *)&endian_probe != 1u) {
        fprintf(stderr,
                "prefill-attention-frozen: little-endian host is required\n");
        return NULL;
    }

    FILE *stream = fopen(path, "rb");
    if (!stream) {
        fprintf(stderr, "prefill-attention-frozen: cannot open %s\n", path);
        return NULL;
    }
    if (fseek(stream, 0, SEEK_END) != 0) {
        fprintf(stderr, "prefill-attention-frozen: cannot seek %s\n", path);
        fclose(stream);
        return NULL;
    }
    const long file_bytes = ftell(stream);
    if (file_bytes < 0 || (uint64_t)file_bytes != expected_bytes) {
        fprintf(stderr,
                "prefill-attention-frozen: %s bytes=%ld expected=%llu\n",
                path, file_bytes, (unsigned long long)expected_bytes);
        fclose(stream);
        return NULL;
    }
    rewind(stream);

    float *values = (float *)malloc((size_t)expected_bytes);
    if (!values || fread(values, sizeof(*values), (size_t)count, stream) != count ||
        ferror(stream)) {
        fprintf(stderr, "prefill-attention-frozen: cannot read %s\n", path);
        free(values);
        fclose(stream);
        return NULL;
    }
    fclose(stream);
    return values;
}

#define LAGUNA_ATTENTION_AUTO_LONG_FIXTURE_DIR \
    "tests/test-vectors/laguna-attention-auto-long"

static float *laguna_read_long_frozen_f32(
        const char *name, uint64_t count) {
    char path[512];
    const int path_length = snprintf(
        path, sizeof(path), "%s/%s",
        LAGUNA_ATTENTION_AUTO_LONG_FIXTURE_DIR, name);
    const uint64_t expected_bytes = count * sizeof(float);
    if (path_length < 0 || (size_t)path_length >= sizeof(path) ||
        sizeof(float) != 4u) {
        fprintf(stderr,
                "prefill-attention-long-frozen: invalid fixture contract\n");
        return NULL;
    }
    const uint16_t endian_probe = 1u;
    if (*(const uint8_t *)&endian_probe != 1u) {
        fprintf(stderr,
                "prefill-attention-long-frozen: little-endian host is required\n");
        return NULL;
    }
    FILE *stream = fopen(path, "rb");
    if (!stream || fseek(stream, 0, SEEK_END) != 0) {
        fprintf(stderr,
                "prefill-attention-long-frozen: cannot open %s\n", path);
        if (stream) fclose(stream);
        return NULL;
    }
    const long file_bytes = ftell(stream);
    if (file_bytes < 0 || (uint64_t)file_bytes != expected_bytes) {
        fprintf(stderr,
                "prefill-attention-long-frozen: %s bytes=%ld expected=%llu\n",
                path, file_bytes, (unsigned long long)expected_bytes);
        fclose(stream);
        return NULL;
    }
    rewind(stream);
    float *values = (float *)malloc((size_t)expected_bytes);
    if (!values ||
        fread(values, sizeof(*values), (size_t)count, stream) != count ||
        ferror(stream)) {
        fprintf(stderr,
                "prefill-attention-long-frozen: cannot read %s\n", path);
        free(values);
        fclose(stream);
        return NULL;
    }
    fclose(stream);
    return values;
}

static float *laguna_frozen_qk_model_map;

static int laguna_frozen_qk_exact(
        const char *arm, const float *q_actual, const float *q_reference,
        uint64_t q_count, uint32_t n_q_head, const float *k_actual,
        const float *k_reference, uint64_t k_count, uint32_t n_k_head,
        uint32_t head_dim) {
    const int q_exact =
        memcmp(q_actual, q_reference, (size_t)q_count * sizeof(float)) == 0;
    const int k_exact =
        memcmp(k_actual, k_reference, (size_t)k_count * sizeof(float)) == 0;
    if (q_exact && k_exact) return 1;

    const laguna_parity_span diagnostics[] = {
        { "q-global", q_actual, q_reference, q_count,
          1u, (uint32_t)q_count, 0.0f, 0.0f },
        { "q-per-head", q_actual, q_reference, q_count,
          n_q_head, head_dim, 0.0f, 0.0f },
        { "k-global", k_actual, k_reference, k_count,
          1u, (uint32_t)k_count, 0.0f, 0.0f },
        { "k-per-head", k_actual, k_reference, k_count,
          n_k_head, head_dim, 0.0f, 0.0f },
    };
    for (size_t i = 0; i < sizeof(diagnostics) / sizeof(diagnostics[0]); i++) {
        (void)laguna_parity_spans_within_limits(
            arm, &diagnostics[i], 1u, 1);
    }
    return 0;
}

static int run_qk_norm_rope_frozen_t21_case(void) {
    const uint32_t n_tokens = 1u;
    const uint32_t n_q_head = 48u;
    const uint32_t n_k_head = 8u;
    const uint32_t head_dim = 128u;
    const uint32_t n_rot = 64u;
    const uint64_t q_count = (uint64_t)n_q_head * head_dim;
    const uint64_t k_count = (uint64_t)n_k_head * head_dim;
    const uint64_t q_full_count = 22u * q_count;
    const uint64_t k_full_count = 22u * k_count;
    const uint64_t weight_bytes = (uint64_t)head_dim * sizeof(float);
    float *q_input =
        laguna_read_frozen_f32("layer-00-q-proj-t21.f32", q_count);
    float *k_input =
        laguna_read_frozen_f32("layer-00-k-proj-t21.f32", k_count);
    float *q_weight =
        laguna_read_frozen_f32("layer-00-q-norm-weight.f32", head_dim);
    float *k_weight =
        laguna_read_frozen_f32("layer-00-k-norm-weight.f32", head_dim);
    float *q_reference_full =
        laguna_read_frozen_f32("layer-00-q-rope.f32", q_full_count);
    float *k_reference_full =
        laguna_read_frozen_f32("layer-00-k-rope.f32", k_full_count);
    float *q_actual = (float *)malloc((size_t)q_count * sizeof(*q_actual));
    float *k_actual = (float *)malloc((size_t)k_count * sizeof(*k_actual));
    float *model_map = (float *)malloc((size_t)(2u * weight_bytes));
    ds4_gpu_tensor *q = NULL;
    ds4_gpu_tensor *k = NULL;
    int rc = 1;

    if (!q_input || !k_input || !q_weight || !k_weight ||
        !q_reference_full || !k_reference_full || !q_actual || !k_actual ||
        !model_map || laguna_frozen_qk_model_map) {
        fprintf(stderr, "poolside-auto-qk-t21: fixture setup failed\n");
        goto cleanup;
    }
    memcpy(model_map, q_weight, (size_t)weight_bytes);
    memcpy((char *)model_map + weight_bytes, k_weight, (size_t)weight_bytes);
    if (!ds4_gpu_set_model_map(model_map, 2u * weight_bytes)) {
        fprintf(stderr, "poolside-auto-qk-t21: model-map setup failed\n");
        goto cleanup;
    }
    laguna_frozen_qk_model_map = model_map;
    model_map = NULL;

    q = ds4_gpu_tensor_alloc(q_count * sizeof(*q_input));
    k = ds4_gpu_tensor_alloc(k_count * sizeof(*k_input));
    if (!q || !k ||
        !ds4_gpu_tensor_write(q, 0, q_input, q_count * sizeof(*q_input)) ||
        !ds4_gpu_tensor_write(k, 0, k_input, k_count * sizeof(*k_input))) {
        fprintf(stderr, "poolside-auto-qk-t21: tensor setup failed\n");
        goto cleanup;
    }
    const int wrapper_ok = ds4_gpu_laguna_qk_head_rms_norm_rope_tensor(
        q, k, laguna_frozen_qk_model_map, 2u * weight_bytes,
        0u, weight_bytes, n_tokens, n_q_head, n_k_head, head_dim, n_rot,
        21u, 8192u, 500000.0f, 1.0f / 32.0f, 1.0f, 1.0f,
        32.0f, 1.0f, 1.0e-6f);
    const cudaError_t sync = cudaDeviceSynchronize();
    if (!wrapper_ok || sync != cudaSuccess) {
        fprintf(stderr,
                "poolside-auto-qk-t21: wrapper=%d sync=%s\n",
                wrapper_ok, cudaGetErrorString(sync));
        goto cleanup;
    }
    if (!ds4_gpu_tensor_read(q, 0, q_actual, q_count * sizeof(*q_actual)) ||
        !ds4_gpu_tensor_read(k, 0, k_actual, k_count * sizeof(*k_actual))) {
        fprintf(stderr, "poolside-auto-qk-t21: output read failed\n");
        goto cleanup;
    }

    const float *q_reference = q_reference_full + 21u * q_count;
    const float *k_reference = k_reference_full + 21u * k_count;
    const int pair_exact = laguna_frozen_qk_exact(
        "poolside-auto-qk-t21/pair", q_actual, q_reference,
        q_count, n_q_head, k_actual, k_reference, k_count, n_k_head,
        head_dim);

    if (!ds4_gpu_tensor_write(q, 0, q_input, q_count * sizeof(*q_input)) ||
        !ds4_gpu_tensor_write(k, 0, k_input, k_count * sizeof(*k_input))) {
        fprintf(stderr, "poolside-auto-qk-t21: tensor reset failed\n");
        goto cleanup;
    }
    const int q_wrapper_ok = ds4_gpu_laguna_head_rms_norm_rope_tensor(
        q, laguna_frozen_qk_model_map, 2u * weight_bytes, 0u,
        n_tokens, n_q_head, head_dim, n_rot, 21u, 8192u, 500000.0f,
        1.0f / 32.0f, 1.0f, 1.0f, 32.0f, 1.0f, 1.0e-6f);
    const int k_wrapper_ok = ds4_gpu_laguna_head_rms_norm_rope_tensor(
        k, laguna_frozen_qk_model_map, 2u * weight_bytes, weight_bytes,
        n_tokens, n_k_head, head_dim, n_rot, 21u, 8192u, 500000.0f,
        1.0f / 32.0f, 1.0f, 1.0f, 32.0f, 1.0f, 1.0e-6f);
    const cudaError_t single_sync = cudaDeviceSynchronize();
    if (!q_wrapper_ok || !k_wrapper_ok || single_sync != cudaSuccess) {
        fprintf(stderr,
                "poolside-auto-qk-t21: single q_wrapper=%d k_wrapper=%d sync=%s\n",
                q_wrapper_ok, k_wrapper_ok, cudaGetErrorString(single_sync));
        goto cleanup;
    }
    if (!ds4_gpu_tensor_read(q, 0, q_actual, q_count * sizeof(*q_actual)) ||
        !ds4_gpu_tensor_read(k, 0, k_actual, k_count * sizeof(*k_actual))) {
        fprintf(stderr, "poolside-auto-qk-t21: single output read failed\n");
        goto cleanup;
    }
    const int single_exact = laguna_frozen_qk_exact(
        "poolside-auto-qk-t21/single", q_actual, q_reference,
        q_count, n_q_head, k_actual, k_reference, k_count, n_k_head,
        head_dim);
    if (pair_exact && single_exact) rc = 0;

cleanup:
    ds4_gpu_tensor_free(k);
    ds4_gpu_tensor_free(q);
    free(model_map);
    free(k_actual);
    free(q_actual);
    free(k_reference_full);
    free(q_reference_full);
    free(k_weight);
    free(q_weight);
    free(k_input);
    free(q_input);
    return rc;
}

static int laguna_frozen_cache_matches(
        const laguna_prefill_case *c, const laguna_prefill_result *result,
        const float *k_host, const float *v_host) {
    const uint64_t kv_width = (uint64_t)c->n_head_kv * 128u;
    const uint64_t written = (uint64_t)c->n_tokens * kv_width;
    const uint64_t cache_count = (uint64_t)c->cache_cap * kv_width;
    for (uint64_t i = 0; i < cache_count; i++) {
        const uint16_t expected_key = i < written
            ? reference_f32_to_f16(k_host[i])
            : 0u;
        const uint16_t expected_value = i < written
            ? reference_f32_to_f16(v_host[i])
            : 0u;
        if (result->key_cache[i] != expected_key ||
            result->value_cache[i] != expected_value) {
            fprintf(stderr,
                    "prefill-attention-frozen: cache mismatch index=%llu key=0x%04x expected=0x%04x value=0x%04x expected=0x%04x\n",
                    (unsigned long long)i, result->key_cache[i], expected_key,
                    result->value_cache[i], expected_value);
            return 0;
        }
    }
    return 1;
}

static int run_prefill_attention_frozen_case(void) {
    static const laguna_prefill_case c = {
        "poolside-auto-t22", 0u, 22u, 256u, 48u, 8u,
    };
    const uint32_t head_dim = 128u;
    const uint64_t q_count =
        (uint64_t)c.n_tokens * c.n_head * head_dim;
    const uint64_t kv_count =
        (uint64_t)c.n_tokens * c.n_head_kv * head_dim;
    const uint64_t gate_count = (uint64_t)c.n_tokens * c.n_head;
    const uint64_t cache_count =
        (uint64_t)c.cache_cap * c.n_head_kv * head_dim;
    const uint64_t sentinel_offset =
        ((uint64_t)20u * c.n_head + 43u) * head_dim;
    float *q_host = laguna_read_frozen_f32("layer-00-q-rope.f32", q_count);
    float *k_host = laguna_read_frozen_f32("layer-00-k-rope.f32", kv_count);
    float *v_host = laguna_read_frozen_f32("layer-00-v-proj.f32", kv_count);
    float *gate_host =
        laguna_read_frozen_f32("layer-00-gate-proj.f32", gate_count);
    float *reference =
        laguna_read_frozen_f32("layer-00-attn-gated.f32", q_count);
    uint16_t *key_initial =
        (uint16_t *)calloc((size_t)cache_count, sizeof(*key_initial));
    uint16_t *value_initial =
        (uint16_t *)calloc((size_t)cache_count, sizeof(*value_initial));
    laguna_prefill_result actual = {0};
    int rc = 1;

    if (!q_host || !k_host || !v_host || !gate_host || !reference ||
        !key_initial || !value_initial ||
        !laguna_prefill_result_init(&actual, q_count, cache_count)) {
        fprintf(stderr, "prefill-attention-frozen: fixture setup failed\n");
        goto cleanup;
    }
    if (laguna_run_prefill_once(
            &c, "poolside-auto", q_host, k_host, v_host, gate_host,
            key_initial, value_initial, &actual) != 0) {
        goto cleanup;
    }
    if (!laguna_frozen_cache_matches(&c, &actual, k_host, v_host)) {
        goto cleanup;
    }

    static const float max_abs_limit = 0.0f;
    static const float rms_limit = 0.0f;
    const laguna_parity_span spans[] = {
        { "global", actual.heads, reference, q_count,
          1u, (uint32_t)q_count, max_abs_limit, rms_limit },
        { "per-head", actual.heads, reference, q_count,
          c.n_head, head_dim, max_abs_limit, rms_limit },
        { "token20-head43", actual.heads + sentinel_offset,
          reference + sentinel_offset, head_dim,
          1u, head_dim, max_abs_limit, rms_limit },
    };
    int parity_ok = 1;
    for (size_t i = 0; i < sizeof(spans) / sizeof(spans[0]); i++) {
        if (!laguna_parity_spans_within_limits(
                c.name, &spans[i], 1u, 1)) {
            parity_ok = 0;
        }
    }
    if (!parity_ok) goto cleanup;

    rc = 0;
cleanup:
    laguna_prefill_result_free(&actual);
    free(value_initial);
    free(key_initial);
    free(reference);
    free(gate_host);
    free(v_host);
    free(k_host);
    free(q_host);
    return rc;
}

static int run_prefill_attention_frozen_gqa9_case(void) {
    static const laguna_prefill_case c = {
        "poolside-auto-layer1-t22-gqa9", 0u, 22u, 256u, 72u, 8u,
    };
    const uint32_t head_dim = 128u;
    const uint64_t q_count =
        (uint64_t)c.n_tokens * c.n_head * head_dim;
    const uint64_t kv_count =
        (uint64_t)c.n_tokens * c.n_head_kv * head_dim;
    const uint64_t gate_count = (uint64_t)c.n_tokens * c.n_head;
    const uint64_t cache_count =
        (uint64_t)c.cache_cap * c.n_head_kv * head_dim;
    const uint64_t sentinel_offset =
        ((uint64_t)21u * c.n_head + 71u) * head_dim;
    float *q_host = laguna_read_frozen_f32("layer-01-q-rope.f32", q_count);
    float *k_host = laguna_read_frozen_f32("layer-01-k-rope.f32", kv_count);
    float *v_host = laguna_read_frozen_f32("layer-01-v-proj.f32", kv_count);
    float *gate_host =
        laguna_read_frozen_f32("layer-01-gate-proj.f32", gate_count);
    float *reference =
        laguna_read_frozen_f32("layer-01-attn-gated.f32", q_count);
    uint16_t *key_initial =
        (uint16_t *)calloc((size_t)cache_count, sizeof(*key_initial));
    uint16_t *value_initial =
        (uint16_t *)calloc((size_t)cache_count, sizeof(*value_initial));
    laguna_prefill_result actual = {0};
    int rc = 1;

    if (!q_host || !k_host || !v_host || !gate_host || !reference ||
        !key_initial || !value_initial ||
        !laguna_prefill_result_init(&actual, q_count, cache_count)) {
        fprintf(stderr, "prefill-attention-frozen-gqa9: fixture setup failed\n");
        goto cleanup;
    }
    if (laguna_run_prefill_once(
            &c, "poolside-auto-layer1-gqa9", q_host, k_host, v_host,
            gate_host, key_initial, value_initial, &actual) != 0) {
        goto cleanup;
    }
    if (!laguna_frozen_cache_matches(&c, &actual, k_host, v_host)) {
        goto cleanup;
    }

    static const float max_abs_limit = 0.0f;
    static const float rms_limit = 0.0f;
    const laguna_parity_span spans[] = {
        { "global-gqa9", actual.heads, reference, q_count,
          1u, (uint32_t)q_count, max_abs_limit, rms_limit },
        { "per-head-gqa9", actual.heads, reference, q_count,
          c.n_head, head_dim, max_abs_limit, rms_limit },
        { "token21-head71", actual.heads + sentinel_offset,
          reference + sentinel_offset, head_dim,
          1u, head_dim, max_abs_limit, rms_limit },
    };
    int parity_ok = 1;
    for (size_t i = 0; i < sizeof(spans) / sizeof(spans[0]); i++) {
        if (!laguna_parity_spans_within_limits(c.name, &spans[i], 1u, 1)) {
            parity_ok = 0;
        }
    }
    if (!parity_ok) goto cleanup;

    rc = 0;
cleanup:
    laguna_prefill_result_free(&actual);
    free(value_initial);
    free(key_initial);
    free(reference);
    free(gate_host);
    free(q_host);
    free(v_host);
    free(k_host);
    return rc;
}

typedef struct {
    const char *name;
    uint32_t n_tokens;
    uint32_t n_head;
    uint32_t first_q_head;
    uint32_t first_kv_head;
    const char *q_file;
    const char *k_file;
    const char *v_file;
    const char *gate_file;
    const char *expected_file;
} laguna_long_attention_frozen_case;

static int run_prefill_attention_long_frozen_case(
        const laguna_long_attention_frozen_case *fixture) {
    static const uint32_t n_head_kv = 8u;
    static const uint32_t selected_heads = 2u;
    static const uint32_t head_dim = 128u;
    const uint32_t n_tokens = fixture->n_tokens;
    const laguna_prefill_case c = {
        fixture->name, 0u, n_tokens, 512u, fixture->n_head, n_head_kv,
    };
    const uint64_t q_count =
        (uint64_t)n_tokens * fixture->n_head * head_dim;
    const uint64_t kv_count =
        (uint64_t)n_tokens * n_head_kv * head_dim;
    const uint64_t gate_count = (uint64_t)n_tokens * fixture->n_head;
    const uint64_t cache_count =
        (uint64_t)c.cache_cap * n_head_kv * head_dim;
    const uint64_t selected_q_count =
        (uint64_t)n_tokens * selected_heads * head_dim;
    const uint64_t selected_gate_count =
        (uint64_t)n_tokens * selected_heads;
    const uint64_t selected_kv_count =
        (uint64_t)n_tokens * selected_heads * head_dim;
    float *selected_q = laguna_read_long_frozen_f32(
        fixture->q_file, selected_q_count);
    float *selected_k = laguna_read_long_frozen_f32(
        fixture->k_file, selected_kv_count);
    float *selected_v = laguna_read_long_frozen_f32(
        fixture->v_file, selected_kv_count);
    float *selected_gate = laguna_read_long_frozen_f32(
        fixture->gate_file, selected_gate_count);
    float *reference = laguna_read_long_frozen_f32(
        fixture->expected_file, selected_q_count);
    float *q_host = (float *)calloc((size_t)q_count, sizeof(*q_host));
    float *k_host = (float *)calloc((size_t)kv_count, sizeof(*k_host));
    float *v_host = (float *)calloc((size_t)kv_count, sizeof(*v_host));
    float *gate_host =
        (float *)calloc((size_t)gate_count, sizeof(*gate_host));
    uint16_t *key_initial =
        (uint16_t *)calloc((size_t)cache_count, sizeof(*key_initial));
    uint16_t *value_initial =
        (uint16_t *)calloc((size_t)cache_count, sizeof(*value_initial));
    laguna_prefill_result actual = {0};
    int rc = 1;

    if (!selected_q || !selected_k || !selected_v || !selected_gate ||
        !reference || !q_host || !k_host || !v_host || !gate_host ||
        !key_initial || !value_initial ||
        !laguna_prefill_result_init(&actual, q_count, cache_count)) {
        fprintf(stderr,
                "prefill-attention-long-frozen/%s: fixture setup failed\n",
                fixture->name);
        goto cleanup;
    }
    for (uint32_t token = 0; token < n_tokens; token++) {
        const uint64_t q_source =
            (uint64_t)token * selected_heads * head_dim;
        const uint64_t q_destination =
            ((uint64_t)token * fixture->n_head + fixture->first_q_head) *
            head_dim;
        const uint64_t gate_source = (uint64_t)token * selected_heads;
        const uint64_t gate_destination =
            (uint64_t)token * fixture->n_head + fixture->first_q_head;
        const uint64_t kv_destination =
            ((uint64_t)token * n_head_kv + fixture->first_kv_head) * head_dim;
        memcpy(q_host + q_destination, selected_q + q_source,
               selected_heads * head_dim * sizeof(*selected_q));
        memcpy(gate_host + gate_destination, selected_gate + gate_source,
               selected_heads * sizeof(*selected_gate));
        memcpy(k_host + kv_destination, selected_k + q_source,
               selected_heads * head_dim * sizeof(*selected_k));
        memcpy(v_host + kv_destination, selected_v + q_source,
               selected_heads * head_dim * sizeof(*selected_v));
    }
    if (laguna_run_prefill_once(
            &c, "poolside-auto-long", q_host, k_host, v_host, gate_host,
            key_initial, value_initial, &actual) != 0 ||
        !laguna_frozen_cache_matches(&c, &actual, k_host, v_host)) {
        goto cleanup;
    }
    for (uint32_t token = 0; token < n_tokens; token++) {
        const uint64_t actual_offset =
            ((uint64_t)token * fixture->n_head + fixture->first_q_head) *
            head_dim;
        const uint64_t reference_offset =
            (uint64_t)token * selected_heads * head_dim;
        const laguna_parity_span span = {
            "causal-row-gqa-boundary", actual.heads + actual_offset,
            reference + reference_offset, selected_heads * head_dim,
            selected_heads, head_dim, 0.0f, 0.0f,
        };
        if (memcmp(actual.heads + actual_offset,
                   reference + reference_offset,
                   selected_heads * head_dim * sizeof(*reference)) != 0) {
            char case_name[160];
            snprintf(case_name, sizeof(case_name), "%s/token-%u",
                     fixture->name, token);
            (void)laguna_parity_spans_within_limits(
                case_name, &span, 1u, 1);
            goto cleanup;
        }
    }
    rc = 0;

cleanup:
    laguna_prefill_result_free(&actual);
    free(value_initial);
    free(key_initial);
    free(gate_host);
    free(v_host);
    free(k_host);
    free(q_host);
    free(reference);
    free(selected_gate);
    free(selected_v);
    free(selected_k);
    free(selected_q);
    return rc;
}

static int run_prefill_attention_long_frozen_cases(void) {
    static const laguna_long_attention_frozen_case cases[] = {
        {
            "poolside-auto-layer0-t512-gqa6", 512u, 48u, 41u, 6u,
            "layer-00-q-t0-t511-h41-h42.f32",
            "layer-00-k-t0-t511-kv6-kv7.f32",
            "layer-00-v-t0-t511-kv6-kv7.f32",
            "layer-00-gate-t0-t511-h41-h42.f32",
            "layer-00-attn-gated-t0-t511-h41-h42.f32",
        },
        {
            "poolside-auto-layer1-t64-gqa9", 64u, 72u, 62u, 6u,
            "layer-01-q-t0-t63-h62-h63.f32",
            "layer-01-k-t0-t63-kv6-kv7.f32",
            "layer-01-v-t0-t63-kv6-kv7.f32",
            "layer-01-gate-t0-t63-h62-h63.f32",
            "layer-01-attn-gated-t0-t63-h62-h63.f32",
        },
    };
    int rc = 0;
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        if (run_prefill_attention_long_frozen_case(&cases[i]) != 0) rc = 1;
    }
    return rc;
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
    uint64_t fixture_bytes = 0;
    unsigned char *fixture = NULL;
    int rc = 1;
    for (size_t i = 0; i < 9; i++) {
        if (sizes[i] > fixture_bytes) fixture_bytes = sizes[i];
    }
    fixture = malloc((size_t)fixture_bytes);
    if (!fixture) goto cleanup;
    memset(fixture, 0xa5, (size_t)fixture_bytes);
    for (size_t i = 0; i < 9; i++) {
        t[i] = ds4_gpu_tensor_alloc(sizes[i]);
        snapshots[i] = (laguna_tensor_snapshot){ names[i], t[i], sizes[i], NULL };
        if (!t[i] ||
            !ds4_gpu_tensor_write(t[i], 0, fixture, sizes[i]) ||
            !laguna_capture_tensor_snapshot(&snapshots[i])) goto cleanup;
    }
    const float scale = 1.0f / sqrtf((float)head_dim);
#define PREFILL_CALL(a, pos, ntok, cap, nh, nkh, dim, s) \
    ds4_gpu_laguna_attention_prefill_tensor((a)[0], (a)[1], (a)[2], (a)[3], (a)[4], \
        (a)[5], (a)[6], (a)[7], (a)[8], (pos), (ntok), (cap), (nh), (nkh), (dim), (s))
    const struct { const char *name; uint32_t pos, ntok, cap, nh, nkh, dim; float scale; } bad[] = {
        { "zero-tokens", 0u, 0u, cache_cap, n_head, n_head_kv, head_dim, scale },
        { "position-overflow", UINT32_MAX, 1u, cache_cap, n_head, n_head_kv, head_dim, scale },
        { "dimension-product-overflow", 0u, 0x7fff0001u, 0x7fff0001u,
          0x80010001u, 0x80010001u, head_dim, scale },
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
    free(fixture);
    return rc;
}

static int run_prefill_attention_cases(void) {
    int rc = run_prefill_rejection_cases();
    static const laguna_prefill_case cases[] = {
        { "global-one", 0u, 1u, 16u, 48u, 8u },
        { "global-three", 0u, 3u, 16u, 48u, 8u },
        { "fast-shape-wrap-guard", 0u, 22u, 16u, 48u, 8u },
        { "resumed-global", 3u, 2u, 16u, 48u, 8u },
        { "swa-512-crossing", 509u, 4u, 512u, 72u, 8u },
        { "multi-wrap-4", 0u, 9u, 4u, 48u, 8u },
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        if (run_prefill_contract_case(&cases[i]) != 0) rc = 1;
    }
    return rc;
}

#define LAGUNA_ROUTER_AUTO_FIXTURE_DIR \
    "tests/test-vectors/laguna-router-auto"

static void *laguna_read_router_frozen(
        const char *name, uint64_t count, size_t element_bytes) {
    char path[512];
    const int path_length = snprintf(
        path, sizeof(path), "%s/%s", LAGUNA_ROUTER_AUTO_FIXTURE_DIR, name);
    if (path_length < 0 || (size_t)path_length >= sizeof(path)) {
        fprintf(stderr, "router-frozen: fixture path is too long\n");
        return NULL;
    }
    if (element_bytes != 4u || count > SIZE_MAX / element_bytes) {
        fprintf(stderr, "router-frozen: invalid fixture element contract\n");
        return NULL;
    }
    const uint16_t endian_probe = 1u;
    if (*(const uint8_t *)&endian_probe != 1u) {
        fprintf(stderr, "router-frozen: little-endian host is required\n");
        return NULL;
    }
    const size_t expected_bytes = (size_t)count * element_bytes;
    FILE *stream = fopen(path, "rb");
    if (!stream) {
        fprintf(stderr, "router-frozen: cannot open %s\n", path);
        return NULL;
    }
    if (fseek(stream, 0, SEEK_END) != 0) {
        fprintf(stderr, "router-frozen: cannot seek %s\n", path);
        fclose(stream);
        return NULL;
    }
    const long file_bytes = ftell(stream);
    if (file_bytes < 0 || (uint64_t)file_bytes != expected_bytes) {
        fprintf(stderr,
                "router-frozen: %s bytes=%ld expected=%llu\n",
                path, file_bytes, (unsigned long long)expected_bytes);
        fclose(stream);
        return NULL;
    }
    rewind(stream);

    void *values = malloc(expected_bytes);
    if (!values || fread(values, element_bytes, (size_t)count, stream) != count ||
        ferror(stream)) {
        fprintf(stderr, "router-frozen: cannot read %s\n", path);
        free(values);
        fclose(stream);
        return NULL;
    }
    fclose(stream);
    return values;
}

static float *laguna_frozen_router_model_map;

static int run_router_frozen_case(void) {
    const uint32_t n_tokens = 22u;
    const uint32_t n_expert = 256u;
    const uint32_t n_expert_used = 10u;
    const uint64_t logits_count = (uint64_t)n_tokens * n_expert;
    const uint64_t selected_count = (uint64_t)n_tokens * n_expert_used;
    const uint64_t bias_bytes = (uint64_t)n_expert * sizeof(float);
    float *logits_reference = (float *)laguna_read_router_frozen(
        "layer-01-router-logits.f32", logits_count, sizeof(float));
    float *bias = (float *)laguna_read_router_frozen(
        "layer-01-router-bias.f32", n_expert, sizeof(float));
    int32_t *selected_reference = (int32_t *)laguna_read_router_frozen(
        "layer-01-router-selected.i32", selected_count, sizeof(int32_t));
    float *weights_reference = (float *)laguna_read_router_frozen(
        "layer-01-router-weights.f32", selected_count, sizeof(float));
    int32_t *selected_actual = (int32_t *)malloc(
        (size_t)selected_count * sizeof(*selected_actual));
    float *weights_actual = (float *)malloc(
        (size_t)selected_count * sizeof(*weights_actual));
    ds4_gpu_tensor *logits = NULL;
    ds4_gpu_tensor *probs = NULL;
    ds4_gpu_tensor *selected = NULL;
    ds4_gpu_tensor *weights = NULL;
    int rc = 1;

    if (!logits_reference || !bias || !selected_reference ||
        !weights_reference || !selected_actual || !weights_actual ||
        laguna_frozen_router_model_map) {
        fprintf(stderr, "router-frozen: fixture setup failed\n");
        goto cleanup;
    }
    if (!ds4_gpu_set_model_map(bias, bias_bytes)) {
        fprintf(stderr, "router-frozen: bias model-map setup failed\n");
        goto cleanup;
    }
    laguna_frozen_router_model_map = bias;
    bias = NULL;

    logits = ds4_gpu_tensor_alloc(logits_count * sizeof(float));
    probs = ds4_gpu_tensor_alloc(logits_count * sizeof(float));
    selected = ds4_gpu_tensor_alloc(selected_count * sizeof(int32_t));
    weights = ds4_gpu_tensor_alloc(selected_count * sizeof(float));
    if (!logits || !probs || !selected || !weights ||
        !ds4_gpu_tensor_write(logits, 0, logits_reference,
                              logits_count * sizeof(float))) {
        fprintf(stderr, "router-frozen: tensor setup failed\n");
        goto cleanup;
    }

    const int wrapper_ok = ds4_gpu_glm_router_select_batch_tensor(
        selected, weights, probs,
        laguna_frozen_router_model_map, bias_bytes,
        0u, logits, 256u, 10u, 2.5f, 22u);
    const cudaError_t sync = cudaDeviceSynchronize();
    if (!wrapper_ok || sync != cudaSuccess) {
        fprintf(stderr, "router-frozen: wrapper=%d sync=%s\n",
                wrapper_ok, cudaGetErrorString(sync));
        goto cleanup;
    }
    if (!ds4_gpu_tensor_read(selected, 0, selected_actual,
                             selected_count * sizeof(*selected_actual)) ||
        !ds4_gpu_tensor_read(weights, 0, weights_actual,
                             selected_count * sizeof(*weights_actual))) {
        fprintf(stderr, "router-frozen: output read failed\n");
        goto cleanup;
    }

    const int selected_exact = memcmp(
        selected_actual, selected_reference,
        (size_t)selected_count * sizeof(*selected_actual)) == 0;
    const int weights_exact = memcmp(
        weights_actual, weights_reference,
        (size_t)selected_count * sizeof(*weights_actual)) == 0;
    if (!selected_exact) {
        for (uint64_t i = 0; i < selected_count; i++) {
            if (selected_actual[i] != selected_reference[i]) {
                fprintf(stderr,
                        "router-frozen: selected mismatch token=%llu slot=%llu got=%d expected=%d\n",
                        (unsigned long long)(i / n_expert_used),
                        (unsigned long long)(i % n_expert_used),
                        selected_actual[i], selected_reference[i]);
                break;
            }
        }
    }
    if (!weights_exact) {
        for (uint64_t i = 0; i < selected_count; i++) {
            uint32_t actual_bits = 0u;
            uint32_t reference_bits = 0u;
            memcpy(&actual_bits, &weights_actual[i], sizeof(actual_bits));
            memcpy(&reference_bits, &weights_reference[i],
                   sizeof(reference_bits));
            if (actual_bits != reference_bits) {
                fprintf(stderr,
                        "router-frozen: weight mismatch token=%llu slot=%llu got=%a (0x%08x) expected=%a (0x%08x)\n",
                        (unsigned long long)(i / n_expert_used),
                        (unsigned long long)(i % n_expert_used),
                        weights_actual[i], actual_bits,
                        weights_reference[i], reference_bits);
                break;
            }
        }
    }
    if (selected_exact && weights_exact) rc = 0;

cleanup:
    ds4_gpu_tensor_free(weights);
    ds4_gpu_tensor_free(selected);
    ds4_gpu_tensor_free(probs);
    ds4_gpu_tensor_free(logits);
    free(weights_actual);
    free(selected_actual);
    free(weights_reference);
    free(selected_reference);
    free(bias);
    free(logits_reference);
    return rc;
}

#define LAGUNA_Q4_MMQ_AUTO_FIXTURE_DIR \
    "tests/test-vectors/laguna-q4-mmq-auto"

static void *laguna_read_q4_mmq_frozen(
        const char *name, size_t expected_bytes) {
    char path[512];
    const int path_length = snprintf(
        path, sizeof(path), "%s/%s", LAGUNA_Q4_MMQ_AUTO_FIXTURE_DIR, name);
    if (path_length < 0 || (size_t)path_length >= sizeof(path) ||
        expected_bytes == 0u) {
        fprintf(stderr, "q4-mmq-frozen: invalid fixture path or size\n");
        return NULL;
    }
    const uint16_t endian_probe = 1u;
    if (*(const uint8_t *)&endian_probe != 1u || sizeof(float) != 4u) {
        fprintf(stderr,
                "q4-mmq-frozen: little-endian float32 host is required\n");
        return NULL;
    }
    FILE *stream = fopen(path, "rb");
    if (!stream) {
        fprintf(stderr, "q4-mmq-frozen: cannot open %s\n", path);
        return NULL;
    }
    if (fseek(stream, 0, SEEK_END) != 0) {
        fprintf(stderr, "q4-mmq-frozen: cannot seek %s\n", path);
        fclose(stream);
        return NULL;
    }
    const long file_bytes = ftell(stream);
    if (file_bytes < 0 || (size_t)file_bytes != expected_bytes) {
        fprintf(stderr,
                "q4-mmq-frozen: %s bytes=%ld expected=%zu\n",
                path, file_bytes, expected_bytes);
        fclose(stream);
        return NULL;
    }
    rewind(stream);
    void *data = malloc(expected_bytes);
    if (!data || fread(data, 1u, expected_bytes, stream) != expected_bytes ||
        ferror(stream)) {
        fprintf(stderr, "q4-mmq-frozen: cannot read %s\n", path);
        free(data);
        fclose(stream);
        return NULL;
    }
    fclose(stream);
    return data;
}

static int run_q4_mmq_frozen_case(void) {
    enum {
        input_elements = 3072,
        q4_row_bytes = 12 * 144,
        result_count = 3,
    };
    float *input = (float *)laguna_read_q4_mmq_frozen(
        "input-token-00.f32", input_elements * sizeof(float));
    void *gate_row = laguna_read_q4_mmq_frozen(
        "expert-246-row-000-gate.q4k", q4_row_bytes);
    void *up_row = laguna_read_q4_mmq_frozen(
        "expert-246-row-000-up.q4k", q4_row_bytes);
    float *gate_expected = (float *)laguna_read_q4_mmq_frozen(
        "gate.f32", sizeof(float));
    float *up_expected = (float *)laguna_read_q4_mmq_frozen(
        "up.f32", sizeof(float));
    float *swiglu_expected = (float *)laguna_read_q4_mmq_frozen(
        "swiglu.f32", sizeof(float));
    ds4_gpu_tensor *input_t = NULL;
    ds4_gpu_tensor *gate_t = NULL;
    ds4_gpu_tensor *up_t = NULL;
    ds4_gpu_tensor *out_t = NULL;
    float actual[result_count] = {0.0f, 0.0f, 0.0f};
    int rc = 1;
    if (!input || !gate_row || !up_row || !gate_expected || !up_expected ||
        !swiglu_expected) {
        goto cleanup;
    }
    const float expected[result_count] = {
        gate_expected[0], up_expected[0], swiglu_expected[0]
    };
    input_t = ds4_gpu_tensor_alloc(input_elements * sizeof(float));
    gate_t = ds4_gpu_tensor_alloc(q4_row_bytes);
    up_t = ds4_gpu_tensor_alloc(q4_row_bytes);
    out_t = ds4_gpu_tensor_alloc(result_count * sizeof(float));
    if (!input_t || !gate_t || !up_t || !out_t ||
        !ds4_gpu_tensor_write(
            input_t, 0, input, input_elements * sizeof(float)) ||
        !ds4_gpu_tensor_write(gate_t, 0, gate_row, q4_row_bytes) ||
        !ds4_gpu_tensor_write(up_t, 0, up_row, q4_row_bytes) ||
        !ds4_gpu_test_glm_poolside_q4_mmq_gate_up_tensor(
            out_t, gate_t, up_t, input_t, input_elements) ||
        cudaDeviceSynchronize() != cudaSuccess ||
        !ds4_gpu_tensor_read(
            out_t, 0, actual, result_count * sizeof(float))) {
        fprintf(stderr, "q4-mmq-frozen: CUDA execution failed\n");
        goto cleanup;
    }
    static const char *const names[result_count] = {
        "gate", "up", "swiglu"
    };
    for (size_t i = 0; i < result_count; i++) {
        uint32_t actual_bits = 0u;
        uint32_t expected_bits = 0u;
        memcpy(&actual_bits, &actual[i], sizeof(actual_bits));
        memcpy(&expected_bits, &expected[i], sizeof(expected_bits));
        if (actual_bits != expected_bits) {
            fprintf(stderr,
                    "q4-mmq-frozen: %s got=%a (0x%08x) expected=%a (0x%08x)\n",
                    names[i], actual[i], actual_bits,
                    expected[i], expected_bits);
            goto cleanup;
        }
    }
    rc = 0;
cleanup:
    ds4_gpu_tensor_free(out_t);
    ds4_gpu_tensor_free(up_t);
    ds4_gpu_tensor_free(gate_t);
    ds4_gpu_tensor_free(input_t);
    free(swiglu_expected);
    free(up_expected);
    free(gate_expected);
    free(up_row);
    free(gate_row);
    free(input);
    return rc;
}

#define LAGUNA_Q4_L2_AUTO_FIXTURE_DIR \
    "tests/test-vectors/laguna-q4-l2-auto"

static void *laguna_read_q4_l2_frozen(
        const char *name, size_t expected_bytes) {
    char path[512];
    const int path_length = snprintf(
        path, sizeof(path), "%s/%s", LAGUNA_Q4_L2_AUTO_FIXTURE_DIR, name);
    if (path_length < 0 || (size_t)path_length >= sizeof(path) ||
        expected_bytes == 0u) {
        fprintf(stderr, "q4-l2-frozen: invalid fixture path or size\n");
        return NULL;
    }
    const uint16_t endian_probe = 1u;
    if (*(const uint8_t *)&endian_probe != 1u || sizeof(float) != 4u) {
        fprintf(stderr,
                "q4-l2-frozen: little-endian float32 host is required\n");
        return NULL;
    }
    FILE *stream = fopen(path, "rb");
    if (!stream) {
        fprintf(stderr, "q4-l2-frozen: cannot open %s\n", path);
        return NULL;
    }
    if (fseek(stream, 0, SEEK_END) != 0) {
        fprintf(stderr, "q4-l2-frozen: cannot seek %s\n", path);
        fclose(stream);
        return NULL;
    }
    const long file_bytes = ftell(stream);
    if (file_bytes < 0 || (size_t)file_bytes != expected_bytes) {
        fprintf(stderr,
                "q4-l2-frozen: %s bytes=%ld expected=%zu\n",
                path, file_bytes, expected_bytes);
        fclose(stream);
        return NULL;
    }
    rewind(stream);
    void *data = malloc(expected_bytes);
    if (!data || fread(data, 1u, expected_bytes, stream) != expected_bytes ||
        ferror(stream)) {
        fprintf(stderr, "q4-l2-frozen: cannot read %s\n", path);
        free(data);
        fclose(stream);
        return NULL;
    }
    fclose(stream);
    return data;
}

static int run_q4_l2_frozen_topology_case(
        const char *label,
        const char *mid_name,
        const char *column_l2_name,
        const char *down_input_name,
        uint64_t fixture_pair_count,
        uint64_t topology_pair_count) {
    enum { expert_mid_dim = 1024 };
    const uint64_t mid_values = fixture_pair_count * expert_mid_dim;
    const size_t mid_bytes = (size_t)mid_values * sizeof(float);
    const size_t column_l2_bytes =
        (size_t)fixture_pair_count * sizeof(float);
    float *mid_input = (float *)laguna_read_q4_l2_frozen(
        mid_name, mid_bytes);
    float *column_l2_expected = (float *)laguna_read_q4_l2_frozen(
        column_l2_name, column_l2_bytes);
    float *down_input_expected = (float *)laguna_read_q4_l2_frozen(
        down_input_name, mid_bytes);
    float *down_input_actual = (float *)malloc(mid_bytes);
    float *column_l2_actual = (float *)malloc(column_l2_bytes);
    ds4_gpu_tensor *mid = NULL;
    ds4_gpu_tensor *column_l2 = NULL;
    int rc = 1;
    if (!mid_input || !column_l2_expected || !down_input_expected ||
        !down_input_actual || !column_l2_actual) {
        goto cleanup;
    }
    mid = ds4_gpu_tensor_alloc(mid_bytes);
    column_l2 = ds4_gpu_tensor_alloc(column_l2_bytes);
    if (!mid || !column_l2 ||
        !ds4_gpu_tensor_write(mid, 0, mid_input, mid_bytes) ||
        !ds4_gpu_test_glm_poolside_q4_l2_tensor(
            mid, column_l2, expert_mid_dim,
            fixture_pair_count, topology_pair_count) ||
        cudaDeviceSynchronize() != cudaSuccess ||
        !ds4_gpu_tensor_read(
            column_l2, 0, column_l2_actual, column_l2_bytes) ||
        !ds4_gpu_tensor_read(mid, 0, down_input_actual, mid_bytes)) {
        fprintf(stderr, "q4-l2-frozen/%s: CUDA execution failed\n", label);
        goto cleanup;
    }
    uint32_t actual_bits = 0u;
    uint32_t expected_bits = 0u;
    for (uint64_t pair = 0; pair < fixture_pair_count; pair++) {
        memcpy(&actual_bits, &column_l2_actual[pair], sizeof(actual_bits));
        memcpy(&expected_bits, &column_l2_expected[pair], sizeof(expected_bits));
        if (actual_bits != expected_bits) {
            fprintf(stderr,
                    "q4-l2-frozen/%s: column-l2[%llu] got=%a (0x%08x) "
                    "expected=%a (0x%08x)\n",
                    label, (unsigned long long)pair,
                    column_l2_actual[pair], actual_bits,
                    column_l2_expected[pair], expected_bits);
            goto cleanup;
        }
    }
    for (uint64_t i = 0; i < mid_values; i++) {
        memcpy(&actual_bits, &down_input_actual[i], sizeof(actual_bits));
        memcpy(&expected_bits, &down_input_expected[i], sizeof(expected_bits));
        if (actual_bits != expected_bits) {
            fprintf(stderr,
                    "q4-l2-frozen/%s: down-input[%zu] got=%a (0x%08x) "
                    "expected=%a (0x%08x)\n",
                    label, (size_t)i, down_input_actual[i], actual_bits,
                    down_input_expected[i], expected_bits);
            goto cleanup;
        }
    }
    rc = 0;
cleanup:
    ds4_gpu_tensor_free(column_l2);
    ds4_gpu_tensor_free(mid);
    free(column_l2_actual);
    free(down_input_actual);
    free(down_input_expected);
    free(column_l2_expected);
    free(mid_input);
    return rc;
}

static int run_q4_l2_frozen_case(void) {
    if (run_q4_l2_frozen_topology_case(
            "prefill-128",
            "mid-pair0.f32",
            "expected-col-l2-pair0.f32",
            "expected-down-input-pair0.f32",
            1u,
            220u) != 0) {
        return 1;
    }
    return run_q4_l2_frozen_topology_case(
        "decode-512",
        "mid-decode.f32",
        "expected-col-l2-decode.f32",
        "expected-down-input-decode.f32",
        10u,
        10u);
}

#define LAGUNA_MOE_RESIDUAL_AUTO_FIXTURE_DIR \
    "tests/test-vectors/laguna-moe-residual-auto"

static void *laguna_read_moe_residual_frozen(
        const char *name, size_t expected_bytes) {
    char path[512];
    const int path_length = snprintf(
        path, sizeof(path), "%s/%s",
        LAGUNA_MOE_RESIDUAL_AUTO_FIXTURE_DIR, name);
    if (path_length < 0 || (size_t)path_length >= sizeof(path) ||
        expected_bytes == 0u) {
        fprintf(stderr,
                "moe-residual-frozen: invalid fixture path or size\n");
        return NULL;
    }
    const uint16_t endian_probe = 1u;
    if (*(const uint8_t *)&endian_probe != 1u || sizeof(float) != 4u) {
        fprintf(stderr,
                "moe-residual-frozen: little-endian float32 host is required\n");
        return NULL;
    }
    FILE *stream = fopen(path, "rb");
    if (!stream) {
        fprintf(stderr, "moe-residual-frozen: cannot open %s\n", path);
        return NULL;
    }
    if (fseek(stream, 0, SEEK_END) != 0) {
        fprintf(stderr, "moe-residual-frozen: cannot seek %s\n", path);
        fclose(stream);
        return NULL;
    }
    const long file_bytes = ftell(stream);
    if (file_bytes < 0 || (size_t)file_bytes != expected_bytes) {
        fprintf(stderr,
                "moe-residual-frozen: %s bytes=%ld expected=%zu\n",
                path, file_bytes, expected_bytes);
        fclose(stream);
        return NULL;
    }
    rewind(stream);
    void *data = malloc(expected_bytes);
    if (!data || fread(data, 1u, expected_bytes, stream) != expected_bytes ||
        ferror(stream)) {
        fprintf(stderr, "moe-residual-frozen: cannot read %s\n", path);
        free(data);
        fclose(stream);
        return NULL;
    }
    fclose(stream);
    return data;
}

static int run_moe_residual_frozen_case(void) {
    enum { width = 3072 };
    const size_t bytes = width * sizeof(float);
    float *residual_input = (float *)laguna_read_moe_residual_frozen(
        "residual-token0.f32", bytes);
    float *moe_input = (float *)laguna_read_moe_residual_frozen(
        "moe-token0.f32", bytes);
    float *shared_input = (float *)laguna_read_moe_residual_frozen(
        "shared-token0.f32", bytes);
    float *expected = (float *)laguna_read_moe_residual_frozen(
        "expected-token0.f32", bytes);
    float *actual = (float *)malloc(bytes);
    ds4_gpu_tensor *residual = NULL;
    ds4_gpu_tensor *moe = NULL;
    ds4_gpu_tensor *shared = NULL;
    ds4_gpu_tensor *out = NULL;
    int rc = 1;
    if (!residual_input || !moe_input || !shared_input || !expected ||
        !actual) {
        goto cleanup;
    }
    residual = ds4_gpu_tensor_alloc(bytes);
    moe = ds4_gpu_tensor_alloc(bytes);
    shared = ds4_gpu_tensor_alloc(bytes);
    out = ds4_gpu_tensor_alloc(bytes);
    if (!residual || !moe || !shared || !out ||
        !ds4_gpu_tensor_write(residual, 0, residual_input, bytes) ||
        !ds4_gpu_tensor_write(moe, 0, moe_input, bytes) ||
        !ds4_gpu_tensor_write(shared, 0, shared_input, bytes) ||
        !ds4_gpu_laguna_moe_residual_tensor(
            out, residual, moe, shared, width) ||
        cudaDeviceSynchronize() != cudaSuccess ||
        !ds4_gpu_tensor_read(out, 0, actual, bytes)) {
        fprintf(stderr, "moe-residual-frozen: CUDA execution failed\n");
        goto cleanup;
    }
    for (size_t i = 0; i < width; i++) {
        uint32_t actual_bits = 0u;
        uint32_t expected_bits = 0u;
        memcpy(&actual_bits, &actual[i], sizeof(actual_bits));
        memcpy(&expected_bits, &expected[i], sizeof(expected_bits));
        if (actual_bits != expected_bits) {
            fprintf(stderr,
                    "moe-residual-frozen: output[%zu] got=%a (0x%08x) "
                    "expected=%a (0x%08x)\n",
                    i, actual[i], actual_bits, expected[i], expected_bits);
            goto cleanup;
        }
    }
    rc = 0;
cleanup:
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(shared);
    ds4_gpu_tensor_free(moe);
    ds4_gpu_tensor_free(residual);
    free(actual);
    free(expected);
    free(shared_input);
    free(moe_input);
    free(residual_input);
    return rc;
}

/* Independent Poolside Q4_K/Q8_1 routed-MoE semantic oracle.  Both
 * quantization boundaries belong here: a float-only reference would test a
 * different kernel.  Small batches use Poolside MMVQ quantization/minimum
 * semantics; the 22-token prompt uses MMQ.  Kernel tile association remains
 * covered by the pinned full-model oracle.  Poolside applies router weights
 * after the down projection. */
#define LAGUNA_QK_K 256u
#define LAGUNA_QK8_1 32u
#define LAGUNA_MOE_DIM 256u
#define LAGUNA_MOE_EXPERTS 16u
#define LAGUNA_MOE_USED 10u
#define LAGUNA_MOE_GUARD 8u

typedef struct { uint16_t d, dmin; uint8_t scales[12]; uint8_t qs[LAGUNA_QK_K / 2u]; } laguna_q4k_block;
typedef struct { uint16_t d, s; int8_t qs[LAGUNA_QK8_1]; } laguna_q8_1_block;
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

static void laguna_quantize_q8_1(
        laguna_q8_1_block *out,
        const float *in,
        int mmq) {
    for (uint32_t block = 0; block < LAGUNA_QK_K / LAGUNA_QK8_1; block++) {
        const float *row = in + block * LAGUNA_QK8_1;
        float amax = 0.0f;
        float sum = 0.0f;
        for (uint32_t i = 0; i < LAGUNA_QK8_1; i++) {
            amax = fmaxf(amax, fabsf(row[i]));
            sum += row[i];
        }
        const float inverse_scale = amax == 0.0f ? 0.0f :
            (mmq ? 127.0f / amax : 1.0f / (amax / 127.0f));
        const float scale = inverse_scale == 0.0f ? 0.0f :
            (mmq ? 1.0f / inverse_scale : amax / 127.0f);
        for (uint32_t i = 0; i < LAGUNA_QK8_1; i++) {
            out[block].qs[i] = (int8_t)roundf(row[i] * inverse_scale);
        }
        out[block].d = reference_f32_to_f16(scale);
        out[block].s = reference_f32_to_f16(sum);
    }
}

static float laguna_q4k_q8_1_dot(
        const laguna_q4k_block *weights,
        const laguna_q8_1_block *input,
        int mmq) {
    const float d = reference_f16_to_f32(weights->d);
    const float dmin = reference_f16_to_f32(weights->dmin);
    if (mmq) {
        float sum = 0.0f;
        for (uint32_t group = 0; group < LAGUNA_QK_K / 32u; group++) {
            uint8_t scale, minimum;
            laguna_q4k_scale_min(
                group, weights->scales, &scale, &minimum);
            int integer_sum = 0;
            const uint32_t byte_offset = (group >> 1u) * 32u;
            const uint32_t shift = (group & 1u) * 4u;
            for (uint32_t i = 0; i < 32u; i++) {
                integer_sum +=
                    (int)((weights->qs[byte_offset + i] >> shift) & 0x0fu) *
                    (int)input[group].qs[i];
            }
            const float scaled_d = reference_f16_to_f32(
                reference_f32_to_f16(d * (float)scale));
            const float scaled_min = reference_f16_to_f32(
                reference_f32_to_f16(-dmin * (float)minimum));
            sum += scaled_d * reference_f16_to_f32(input[group].d) *
                (float)integer_sum;
            sum += scaled_min * reference_f16_to_f32(input[group].s);
        }
        return sum;
    }
    float scaled_sum = 0.0f;
    float minimum_sum = 0.0f;
    for (uint32_t group = 0; group < LAGUNA_QK_K / 32u; group++) {
        uint8_t scale, minimum;
        laguna_q4k_scale_min(group, weights->scales, &scale, &minimum);
        int integer_sum = 0;
        const uint32_t byte_offset = (group >> 1u) * 32u, shift = (group & 1u) * 4u;
        for (uint32_t i = 0; i < 32u; i++)
            integer_sum += (int)((weights->qs[byte_offset + i] >> shift) & 0x0fu) *
                (int)input[group].qs[i];
        scaled_sum += reference_f16_to_f32(input[group].d) *
            (float)((int)scale * integer_sum);
        int quantized_sum = 0;
        for (uint32_t i = 0; i < LAGUNA_QK8_1; i++) {
            quantized_sum += input[group].qs[i];
        }
        minimum_sum += reference_f16_to_f32(input[group].d) *
            (float)(quantized_sum * (int)minimum);
    }
    return d * scaled_sum - dmin * minimum_sum;
}

static void laguna_encode_q4k(laguna_q4k_block *out, uint32_t seed, float scale) {
    memset(out, 0, sizeof(*out));
    out->d = reference_f32_to_f16(scale);
    out->dmin = reference_f32_to_f16(
        scale * (seed % 13u == 0u ? 3.5f : 0.15f));
    for (uint32_t group = 0; group < 4u; group++) {
        out->scales[group] = 1u;
        out->scales[group + 4u] = 1u;
    }
    for (uint32_t group = 4u; group < 8u; group++) {
        out->scales[group + 4u] = 0x11u;
    }
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

static float laguna_silu(float value) {
    return value >= 0.0f ? value / (1.0f + expf(-value)) :
        value * expf(value) / (1.0f + expf(value));
}

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
    const int mmq = n_tokens > 8u;
    for (uint32_t token = 0; token < n_tokens; token++) {
        laguna_q8_1_block xq[LAGUNA_QK_K / LAGUNA_QK8_1];
        laguna_q8_1_block midq[LAGUNA_QK_K / LAGUNA_QK8_1];
        float mid[LAGUNA_MOE_DIM];
        laguna_quantize_q8_1(xq,
                             x + (uint64_t)token * LAGUNA_MOE_DIM,
                             mmq);
        memset(out + (uint64_t)token * LAGUNA_MOE_DIM, 0, LAGUNA_MOE_DIM * sizeof(*out));
        for (uint32_t slot = 0; slot < LAGUNA_MOE_USED; slot++) {
            const uint32_t expert = (uint32_t)selected[(uint64_t)token * LAGUNA_MOE_USED + slot];
            const float weight = router[(uint64_t)token * LAGUNA_MOE_USED + slot];
            const laguna_q4k_block *gate = routed_gate + (uint64_t)expert * LAGUNA_MOE_DIM;
            const laguna_q4k_block *up = routed_up + (uint64_t)expert * LAGUNA_MOE_DIM;
            const laguna_q4k_block *down = routed_down + (uint64_t)expert * LAGUNA_MOE_DIM;
            for (uint32_t row = 0; row < LAGUNA_MOE_DIM; row++) {
                mid[row] = laguna_silu(
                    laguna_q4k_q8_1_dot(gate + row, xq, mmq)) *
                    laguna_q4k_q8_1_dot(up + row, xq, mmq);
            }
            float square_sum = 0.0f;
            for (uint32_t row = 0; row < LAGUNA_MOE_DIM; row++) {
                square_sum += mid[row] * mid[row];
            }
            float column_l2 = sqrtf(square_sum);
            if (column_l2 < 1.0e-8f) column_l2 = 1.0e-8f;
            if (column_l2 > 1.0e30f) column_l2 = 1.0e30f;
            for (uint32_t row = 0; row < LAGUNA_MOE_DIM; row++) {
                mid[row] = (mid[row] * 32768.0f) / column_l2;
            }
            laguna_quantize_q8_1(midq, mid, mmq);
            for (uint32_t row = 0; row < LAGUNA_MOE_DIM; row++) {
                float expert_value =
                    laguna_q4k_q8_1_dot(down + row, midq, mmq);
                expert_value *= column_l2;
                expert_value *= 1.0f / 32768.0f;
                expert_value *= weight;
                out[(uint64_t)token * LAGUNA_MOE_DIM + row] += expert_value;
            }
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
    float mid_fixture[LAGUNA_MOE_USED * LAGUNA_MOE_DIM];
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
    memset(mid_fixture, 0xa5, sizeof(mid_fixture));

    if (!ds4_gpu_set_model_map(model, model_size)) goto cleanup;
    mid = ds4_gpu_tensor_alloc((uint64_t)LAGUNA_MOE_USED * LAGUNA_MOE_DIM * sizeof(float));
    selected_t = ds4_gpu_tensor_alloc(sizeof(selected));
    router_t = ds4_gpu_tensor_alloc(sizeof(router));
    x = ds4_gpu_tensor_alloc(sizeof(input));
    if (!mid || !selected_t || !router_t || !x ||
        !ds4_gpu_tensor_write(mid, 0, mid_fixture, sizeof(mid_fixture)) ||
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
    float prompt_input[22u * LAGUNA_MOE_DIM];
    float reuse_input[LAGUNA_MOE_DIM];
    float one_reference[LAGUNA_MOE_DIM], growth_reference[3u * LAGUNA_MOE_DIM];
    float prompt_reference[22u * LAGUNA_MOE_DIM];
    float reuse_reference[LAGUNA_MOE_DIM];
    int32_t one_selected[LAGUNA_MOE_USED], growth_selected[3u * LAGUNA_MOE_USED];
    int32_t prompt_selected[22u * LAGUNA_MOE_USED];
    int32_t reuse_selected[LAGUNA_MOE_USED];
    float one_router[LAGUNA_MOE_USED], growth_router[3u * LAGUNA_MOE_USED];
    float prompt_router[22u * LAGUNA_MOE_USED];
    float reuse_router[LAGUNA_MOE_USED];
    ds4_gpu_tensor *mid = NULL, *selected_t = NULL, *router_t = NULL, *x = NULL;
    laguna_guarded_output one_output = {0}, growth_output = {0};
    laguna_guarded_output prompt_output = {0};
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
    laguna_fill_routed_case(prompt_input, prompt_selected, prompt_router, 22u, 13u);
    laguna_fill_routed_case(reuse_input, reuse_selected, reuse_router, 1u, 19u);
    laguna_reference_routed_moe(one_reference, model, &routed,
                                one_selected, one_router, one_input, 1u);
    laguna_reference_routed_moe(growth_reference, model, &routed,
                                growth_selected, growth_router, growth_input, 3u);
    laguna_reference_routed_moe(prompt_reference, model, &routed,
                                prompt_selected, prompt_router, prompt_input, 22u);
    laguna_reference_routed_moe(reuse_reference, model, &routed,
                                reuse_selected, reuse_router, reuse_input, 1u);
    if (!laguna_reference_signal("one", one_reference, LAGUNA_MOE_DIM) ||
        !laguna_reference_signal("growth", growth_reference,
                                 3u * LAGUNA_MOE_DIM) ||
        !laguna_reference_signal("prompt", prompt_reference,
                                 22u * LAGUNA_MOE_DIM) ||
        !laguna_reference_signal("reuse", reuse_reference, LAGUNA_MOE_DIM)) {
        goto cleanup;
    }
    if (!ds4_gpu_set_model_map(model, model_size)) goto cleanup;
    mid = ds4_gpu_tensor_alloc(22u * LAGUNA_MOE_USED * LAGUNA_MOE_DIM * sizeof(float));
    selected_t = ds4_gpu_tensor_alloc(22u * LAGUNA_MOE_USED * sizeof(int32_t));
    router_t = ds4_gpu_tensor_alloc(22u * LAGUNA_MOE_USED * sizeof(float));
    x = ds4_gpu_tensor_alloc(22u * LAGUNA_MOE_DIM * sizeof(float));
    if (!mid || !selected_t || !router_t || !x ||
        !laguna_guarded_output_init(&one_output, LAGUNA_MOE_DIM) ||
        !laguna_guarded_output_init(&growth_output, 3u * LAGUNA_MOE_DIM) ||
        !laguna_guarded_output_init(&prompt_output, 22u * LAGUNA_MOE_DIM)) {
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

    if (!ds4_gpu_tensor_write(selected_t, 0, prompt_selected, sizeof(prompt_selected)) ||
        !ds4_gpu_tensor_write(router_t, 0, prompt_router, sizeof(prompt_router)) ||
        !ds4_gpu_tensor_write(x, 0, prompt_input, sizeof(prompt_input)) ||
        !laguna_guarded_output_prepare(&prompt_output) ||
        !laguna_glm_routed_moe_q4_call(prompt_output.view, mid, model, model_size,
                                       projection_bytes, expert_bytes, row_bytes,
                                       selected_t, router_t, x, 22u, true)) {
        fprintf(stderr, "routed-moe/prompt: public GLM bridge returned failure\n");
        goto cleanup;
    }
    if (cudaDeviceSynchronize() != cudaSuccess ||
        !laguna_guarded_output_check("prompt", &prompt_output, prompt_reference, 22u)) goto cleanup;

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
    laguna_guarded_output_free(&prompt_output);
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
    fprintf(stderr, "usage: %s --case norm-rope|decode-attention|prefill-attention|prefill-attention-frozen|prefill-attention-long-frozen|router-frozen|q4-mmq-frozen|q4-l2-frozen|moe-residual-frozen|routed-moe|poolside-q8|all\n", program);
}

int main(int argc, char **argv) {
    if (argc != 3 || strcmp(argv[1], "--case") != 0 ||
        (strcmp(argv[2], "norm-rope") != 0 &&
         strcmp(argv[2], "decode-attention") != 0 &&
         strcmp(argv[2], "prefill-attention") != 0 &&
         strcmp(argv[2], "prefill-attention-frozen") != 0 &&
         strcmp(argv[2], "prefill-attention-long-frozen") != 0 &&
         strcmp(argv[2], "router-frozen") != 0 &&
         strcmp(argv[2], "q4-mmq-frozen") != 0 &&
         strcmp(argv[2], "q4-l2-frozen") != 0 &&
         strcmp(argv[2], "moe-residual-frozen") != 0 &&
         strcmp(argv[2], "routed-moe") != 0 &&
         strcmp(argv[2], "poolside-q8") != 0 &&
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
    const bool run_prefill_frozen =
        strcmp(argv[2], "prefill-attention-frozen") == 0 ||
        strcmp(argv[2], "all") == 0;
    const bool run_prefill_long_frozen =
        strcmp(argv[2], "prefill-attention-long-frozen") == 0 ||
        strcmp(argv[2], "all") == 0;
    const bool run_router_frozen = strcmp(argv[2], "router-frozen") == 0 ||
        strcmp(argv[2], "all") == 0;
    const bool run_q4_mmq_frozen =
        strcmp(argv[2], "q4-mmq-frozen") == 0 ||
        strcmp(argv[2], "all") == 0;
    const bool run_q4_l2_frozen =
        strcmp(argv[2], "q4-l2-frozen") == 0 ||
        strcmp(argv[2], "all") == 0;
    const bool run_moe_residual_frozen =
        strcmp(argv[2], "moe-residual-frozen") == 0 ||
        strcmp(argv[2], "all") == 0;
    const bool run_routed_moe = strcmp(argv[2], "routed-moe") == 0 ||
        strcmp(argv[2], "all") == 0;
    const bool run_poolside_q8 = strcmp(argv[2], "poolside-q8") == 0 ||
        strcmp(argv[2], "all") == 0;
    if ((run_decode || run_prefill || run_prefill_frozen ||
         run_prefill_long_frozen || run_routed_moe || run_poolside_q8) &&
        run_f32_to_f16_reference_cases() != 0) return 1;
    if (!ds4_gpu_init()) {
        fprintf(stderr, "norm-rope: ds4_gpu_init failed\n");
        return 1;
    }
    float *weights = NULL;
    unsigned char *poolside_q8_model = NULL;
    unsigned char *generic_q8_model = NULL;
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
        { "qk-global-decode-one", 1, 48, 64, 8193, 8192, 500000.0f,
          1.0f / 32.0f, 1.0f, 1.0f, 32.0f, 1.0f, 0u },
        { "qk-swa-decode-one", 1, 72, 128, 513, 262144, 10000.0f,
          1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0u },
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
        if (run_grouped_metric_contract_cases() != 0) rc = 1;
        if (run_qk_norm_rope_frozen_t21_case() != 0) rc = 1;
    }
    if (run_decode && run_decode_attention_cases() != 0) {
        rc = 1;
    }
    if (run_prefill && run_prefill_attention_cases() != 0) {
        rc = 1;
    }
    if (run_prefill_frozen) {
        if (run_prefill_attention_frozen_case() != 0) rc = 1;
        if (run_prefill_attention_frozen_gqa9_case() != 0) rc = 1;
    }
    if (run_prefill_long_frozen &&
        run_prefill_attention_long_frozen_cases() != 0) {
        rc = 1;
    }
    if (run_routed_moe && run_routed_moe_cases() != 0) {
        rc = 1;
    }
    if (run_poolside_q8) {
        if (run_poolside_q8_projection_case(&poolside_q8_model) != 0) rc = 1;
        if (run_q8_matmul_prefill_case(&generic_q8_model) != 0) rc = 1;
    }
    if (run_router_frozen && run_router_frozen_case() != 0) {
        rc = 1;
    }
    if (run_q4_mmq_frozen && run_q4_mmq_frozen_case() != 0) {
        rc = 1;
    }
    if (run_q4_l2_frozen && run_q4_l2_frozen_case() != 0) {
        rc = 1;
    }
    if (run_moe_residual_frozen &&
        run_moe_residual_frozen_case() != 0) {
        rc = 1;
    }
    /* The model-map registration pins weights until GPU cleanup unregisters it. */
    ds4_gpu_cleanup();
    free(laguna_frozen_qk_model_map);
    free(laguna_frozen_router_model_map);
    free(generic_q8_model);
    free(poolside_q8_model);
    free(weights);
    return rc;
}
