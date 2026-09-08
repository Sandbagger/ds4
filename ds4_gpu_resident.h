#ifndef DS4_GPU_RESIDENT_H
#define DS4_GPU_RESIDENT_H

#include <stdint.h>
#include "ds4_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef DS4_GPU_TENSOR_DEFINED
#define DS4_GPU_TENSOR_DEFINED
typedef struct ds4_gpu_tensor ds4_gpu_tensor;
#endif

/* CUDA resident observation only, not admission or snapshot readiness.
 * The descriptor and device storage each have an observed physical owner.
 * Keep this caller-owned handle and the attached tracker alive through cleanup;
 * do not independently free its descriptor, storage, or tracker records. */
typedef struct {
    ds4_gpu_tensor *tensor;
    uint64_t descriptor_record_id;
    uint64_t device_record_id;
} ds4_gpu_laguna_resident_tensor_owner;

/* Return1 on success,0 on refusal. Allocation requires a zero handle, nonzero
 * bytes, the identical attached tracker and its one current CUDA device.
 * Only KV_STATE/GRAPH_SCRATCH are admitted tensor callsites. Failed allocation
 * keeps a fresh handle zero; a nonempty handle is never overwritten.
 * Generic tensor use before attachment requires a fresh process instead.
 * Caller is quiescent and serializes direct tracker access with native work. */
int ds4_gpu_laguna_resident_tensor_alloc(
    ds4_runtime_tracker *tracker, uint32_t callsite_id, uint64_t bytes,
    ds4_gpu_laguna_resident_tensor_owner *out);

/* Query/synchronize/free failure preserves BOTH owners and the complete handle.
 * Retry remains legal while the tracker is sticky-unsafe. Only complete physical
 * cleanup clears the handle; a null/zero handle is a successful no-op. Tracker
 * identity mismatch refuses without mutating either tracker. No legacy fallback. */
int ds4_gpu_laguna_resident_tensor_free(
    ds4_runtime_tracker *tracker, ds4_gpu_laguna_resident_tensor_owner *owner);

#ifdef __cplusplus
}
#endif
#endif
