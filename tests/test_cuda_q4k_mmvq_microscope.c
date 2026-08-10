#include "ds4_gpu.h"

#include <cuda_runtime.h>

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FIXTURE_DIR "tests/test-vectors/q4k-mmvq-microscope-auto"
#define DS4_SERIAL_EXPECTED_BITS UINT32_C(0xbdcdf5ef)

enum {
    INPUT_ELEMENTS = 3072,
    QK_K = 256,
    QK8_1 = 32,
    Q4_BLOCK_BYTES = 144,
    Q4_BLOCKS = INPUT_ELEMENTS / QK_K,
    Q8_BLOCKS = INPUT_ELEMENTS / QK8_1,
};

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[QK_K / 2];
} q4_k_block;

typedef struct {
    uint16_t d;
    uint16_t s;
    int8_t qs[QK8_1];
} q8_1_block;

_Static_assert(sizeof(q4_k_block) == Q4_BLOCK_BYTES,
               "unexpected Q4_K block layout");
_Static_assert(sizeof(q8_1_block) == 36,
               "unexpected Q8_1 block layout");

static void *read_fixture(const char *name, size_t expected_bytes) {
    char path[512];
    const int length = snprintf(path, sizeof(path), "%s/%s", FIXTURE_DIR, name);
    if (length < 0 || (size_t)length >= sizeof(path)) return NULL;
    FILE *stream = fopen(path, "rb");
    if (!stream) return NULL;
    if (fseek(stream, 0, SEEK_END) != 0) {
        fclose(stream);
        return NULL;
    }
    const long bytes = ftell(stream);
    if (bytes < 0 || (size_t)bytes != expected_bytes) {
        fclose(stream);
        return NULL;
    }
    rewind(stream);
    void *data = malloc(expected_bytes);
    if (!data || fread(data, 1, expected_bytes, stream) != expected_bytes ||
        ferror(stream)) {
        free(data);
        fclose(stream);
        return NULL;
    }
    fclose(stream);
    return data;
}

static double f16_to_f64(uint16_t bits) {
    const int sign = (bits >> 15) ? -1 : 1;
    const unsigned exponent = (bits >> 10) & 0x1fu;
    const unsigned fraction = bits & 0x03ffu;
    if (exponent == 0u) {
        return fraction == 0u ? copysign(0.0, (double)sign) :
            sign * ldexp((double)fraction, -24);
    }
    if (exponent == 0x1fu) {
        return fraction == 0u ? sign * INFINITY : NAN;
    }
    return sign * ldexp(1.0 + (double)fraction / 1024.0,
                        (int)exponent - 15);
}

static void q4_k_scale_min(
        uint32_t group,
        const uint8_t scales[12],
        uint8_t *scale,
        uint8_t *minimum) {
    if (group < 4u) {
        *scale = scales[group] & 63u;
        *minimum = scales[group + 4u] & 63u;
    } else {
        *scale = (scales[group + 4u] & 0x0fu) |
                 ((scales[group - 4u] >> 6u) << 4u);
        *minimum = (scales[group + 4u] >> 4u) |
                   ((scales[group] >> 6u) << 4u);
    }
}

static double q4_k_q8_1_reference(
        const q4_k_block row[Q4_BLOCKS],
        const q8_1_block input[Q8_BLOCKS]) {
    double result = 0.0;
    for (uint32_t block = 0; block < Q4_BLOCKS; block++) {
        const double d = f16_to_f64(row[block].d);
        const double dmin = f16_to_f64(row[block].dmin);
        for (uint32_t group = 0; group < 8u; group++) {
            uint8_t scale = 0u;
            uint8_t minimum = 0u;
            q4_k_scale_min(
                group, row[block].scales, &scale, &minimum);
            const q8_1_block *activation = input + block * 8u + group;
            const double d8 = f16_to_f64(activation->d);
            const uint32_t byte_offset = (group >> 1u) * 32u;
            const uint32_t shift = (group & 1u) * 4u;
            for (uint32_t index = 0; index < 32u; index++) {
                const uint8_t q4 =
                    (row[block].qs[byte_offset + index] >> shift) & 0x0fu;
                const double weight = d * (double)scale * (double)q4 -
                                      dmin * (double)minimum;
                result += weight * d8 * (double)activation->qs[index];
            }
        }
    }
    return result;
}

static uint32_t f32_bits(float value) {
    uint32_t bits = 0u;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

int main(void) {
    float *activation = read_fixture("input.f32", INPUT_ELEMENTS * sizeof(float));
    q4_k_block *weight_row = read_fixture(
        "weight-row.q4k", Q4_BLOCKS * sizeof(q4_k_block));
    float *poolside_expected = read_fixture(
        "poolside-output.f32", sizeof(float));
    q8_1_block *quantized_expected = read_fixture(
        "input.q8_1", Q8_BLOCKS * sizeof(q8_1_block));
    q8_1_block *quantized = malloc(Q8_BLOCKS * sizeof(q8_1_block));
    ds4_gpu_tensor *activation_t = NULL;
    ds4_gpu_tensor *weight_t = NULL;
    ds4_gpu_tensor *quantized_t = NULL;
    ds4_gpu_tensor *values_t = NULL;
    float values[130];
    for (size_t i = 0; i < 130u; i++) values[i] = NAN;
    int rc = 1;

    if (!activation || !weight_row || !poolside_expected ||
        !quantized_expected || !quantized) {
        fprintf(stderr, "q4k-mmvq-microscope: fixture load failed\n");
        goto cleanup;
    }
    if (!ds4_gpu_init()) {
        fprintf(stderr, "q4k-mmvq-microscope: CUDA init failed\n");
        goto cleanup;
    }
    activation_t = ds4_gpu_tensor_alloc(INPUT_ELEMENTS * sizeof(float));
    weight_t = ds4_gpu_tensor_alloc(Q4_BLOCKS * sizeof(q4_k_block));
    quantized_t = ds4_gpu_tensor_alloc(Q8_BLOCKS * sizeof(q8_1_block));
    values_t = ds4_gpu_tensor_alloc(sizeof(values));
    if (!activation_t || !weight_t || !quantized_t || !values_t ||
        !ds4_gpu_tensor_write(
            activation_t, 0, activation, INPUT_ELEMENTS * sizeof(float)) ||
        !ds4_gpu_tensor_write(
            weight_t, 0, weight_row, Q4_BLOCKS * sizeof(q4_k_block)) ||
        !ds4_gpu_test_q4_k_mmvq_microscope_tensor(
            values_t, quantized_t, weight_t, activation_t, INPUT_ELEMENTS) ||
        cudaDeviceSynchronize() != cudaSuccess ||
        !ds4_gpu_tensor_read(values_t, 0, values, sizeof(values)) ||
        !ds4_gpu_tensor_read(
            quantized_t, 0, quantized,
            Q8_BLOCKS * sizeof(q8_1_block))) {
        fprintf(stderr, "q4k-mmvq-microscope: CUDA execution failed\n");
        goto cleanup;
    }

    const double reference = q4_k_q8_1_reference(weight_row, quantized);
    if (memcmp(quantized, quantized_expected,
               Q8_BLOCKS * sizeof(q8_1_block)) != 0) {
        fprintf(stderr,
                "q4k-mmvq-microscope: activation quantization changed\n");
        goto cleanup;
    }
    printf("serial=%a bits=0x%08x abs_error=%.17g\n",
           values[0], f32_bits(values[0]), fabs((double)values[0] - reference));
    printf("poolside=%a bits=0x%08x abs_error=%.17g\n",
           values[1], f32_bits(values[1]), fabs((double)values[1] - reference));
    printf("reference=%.17g\n", reference);

    if (f32_bits(values[1]) != f32_bits(poolside_expected[0])) {
        size_t worst_lane = 0u;
        for (size_t lane = 1u; lane < 128u; lane++) {
            if (fabsf(values[2u + lane]) >
                fabsf(values[2u + worst_lane])) {
                worst_lane = lane;
            }
        }
        fprintf(stderr,
                "q4k-mmvq-microscope: Poolside got=%a bits=0x%08x "
                "expected=%a bits=0x%08x worst_tid=%zu partial=%a\n",
                values[1], f32_bits(values[1]),
                poolside_expected[0], f32_bits(poolside_expected[0]),
                worst_lane, values[2u + worst_lane]);
        goto cleanup;
    }
    if (f32_bits(values[0]) != DS4_SERIAL_EXPECTED_BITS) {
        fprintf(stderr,
                "q4k-mmvq-microscope: serial got=%a bits=0x%08x "
                "expected_bits=0x%08x\n",
                values[0], f32_bits(values[0]),
                DS4_SERIAL_EXPECTED_BITS);
        goto cleanup;
    }
    if (f32_bits(values[0]) == f32_bits(values[1])) {
        fprintf(stderr,
                "q4k-mmvq-microscope: reduction orders unexpectedly agree\n");
        goto cleanup;
    }
    if (!isfinite(reference) ||
        fabs((double)values[0] - reference) > 1.0e-5 ||
        fabs((double)values[1] - reference) > 1.0e-5) {
        fprintf(stderr,
                "q4k-mmvq-microscope: result is not close to FP64 reference\n");
        goto cleanup;
    }
    rc = 0;

cleanup:
    ds4_gpu_tensor_free(values_t);
    ds4_gpu_tensor_free(quantized_t);
    ds4_gpu_tensor_free(weight_t);
    ds4_gpu_tensor_free(activation_t);
    ds4_gpu_cleanup();
    free(quantized);
    free(quantized_expected);
    free(poolside_expected);
    free(weight_row);
    free(activation);
    return rc;
}
