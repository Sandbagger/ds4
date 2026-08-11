#include "ds4_gpu.h"

#include <cuda_runtime.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FIXTURE_DIR "tests/test-vectors/f32-mmvf-microscope-auto"
#define INPUT_FILE "../q4k-mmvq-microscope-auto/input.f32"
#define DS4_SERIAL_EXPECTED_BITS UINT32_C(0xbea377b8)
#define POOLSIDE_EXPECTED_BITS UINT32_C(0xbea377ba)

enum {
    INPUT_ELEMENTS = 3072,
    ROW_BYTES = INPUT_ELEMENTS * sizeof(float),
};

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

static uint32_t f32_bits(float value) {
    uint32_t bits = 0u;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static double f32_dot_reference(
        const float *weight,
        const float *activation) {
    double result = 0.0;
    for (uint32_t i = 0; i < INPUT_ELEMENTS; i++) {
        result += (double)weight[i] * (double)activation[i];
    }
    return result;
}

int main(void) {
    float *activation = read_fixture(INPUT_FILE, ROW_BYTES);
    float *weight = read_fixture("weight-row.f32", ROW_BYTES);
    ds4_gpu_tensor *activation_t = NULL;
    ds4_gpu_tensor *weight_t = NULL;
    ds4_gpu_tensor *values_t = NULL;
    float values[2] = {NAN, NAN};
    int rc = 1;

    if (!activation || !weight) {
        fprintf(stderr, "f32-mmvf-microscope: fixture load failed\n");
        goto cleanup;
    }
    if (!ds4_gpu_init()) {
        fprintf(stderr, "f32-mmvf-microscope: CUDA init failed\n");
        goto cleanup;
    }
    activation_t = ds4_gpu_tensor_alloc(ROW_BYTES);
    weight_t = ds4_gpu_tensor_alloc(ROW_BYTES);
    values_t = ds4_gpu_tensor_alloc(sizeof(values));
    if (!activation_t || !weight_t || !values_t ||
        !ds4_gpu_tensor_write(activation_t, 0, activation, ROW_BYTES) ||
        !ds4_gpu_tensor_write(weight_t, 0, weight, ROW_BYTES) ||
        !ds4_gpu_test_f32_mmvf_microscope_tensor(
            values_t, weight_t, activation_t, INPUT_ELEMENTS) ||
        cudaDeviceSynchronize() != cudaSuccess ||
        !ds4_gpu_tensor_read(values_t, 0, values, sizeof(values))) {
        fprintf(stderr, "f32-mmvf-microscope: CUDA execution failed\n");
        goto cleanup;
    }

    const double reference = f32_dot_reference(weight, activation);
    printf("serial=%a bits=0x%08x abs_error=%.17g\n",
           values[0], f32_bits(values[0]),
           fabs((double)values[0] - reference));
    printf("poolside=%a bits=0x%08x abs_error=%.17g\n",
           values[1], f32_bits(values[1]),
           fabs((double)values[1] - reference));
    printf("reference=%.17g\n", reference);

    if (f32_bits(values[0]) != DS4_SERIAL_EXPECTED_BITS) {
        fprintf(stderr,
                "f32-mmvf-microscope: serial got=%a bits=0x%08x "
                "expected_bits=0x%08x\n",
                values[0], f32_bits(values[0]),
                DS4_SERIAL_EXPECTED_BITS);
        goto cleanup;
    }
    if (f32_bits(values[1]) != POOLSIDE_EXPECTED_BITS) {
        fprintf(stderr,
                "f32-mmvf-microscope: Poolside got=%a bits=0x%08x "
                "expected_bits=0x%08x\n",
                values[1], f32_bits(values[1]),
                POOLSIDE_EXPECTED_BITS);
        goto cleanup;
    }
    if (f32_bits(values[0]) == f32_bits(values[1])) {
        fprintf(stderr,
                "f32-mmvf-microscope: reduction orders unexpectedly agree\n");
        goto cleanup;
    }
    if (!isfinite(reference) ||
        fabs((double)values[0] - reference) > 1.0e-5 ||
        fabs((double)values[1] - reference) > 1.0e-5) {
        fprintf(stderr,
                "f32-mmvf-microscope: result is not close to FP64 reference\n");
        goto cleanup;
    }
    rc = 0;

cleanup:
    ds4_gpu_tensor_free(values_t);
    ds4_gpu_tensor_free(weight_t);
    ds4_gpu_tensor_free(activation_t);
    ds4_gpu_cleanup();
    free(weight);
    free(activation);
    return rc;
}
