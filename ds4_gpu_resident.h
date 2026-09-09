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

/* Read-only attachment identity, including sticky-unsafe cleanup state.
 * No readiness/admission promise. Foreign/null pointers are never dereferenced.
 * Caller remains quiescent and retains the borrowed tracker through cleanup. */
int ds4_gpu_laguna_resident_observer_attached(const ds4_runtime_tracker *tracker);

/* Raw resident HOST payload, not a tensor descriptor or CUDA owner. No GPU
 * initialization/query is needed. Observation is not admission or a snapshot.
 * Keep this handle, tracker, records and callsites outside its payload and alive
 * through cleanup; serialize direct tracker mutations with native operations. */
typedef struct {
    void *base;
    uint64_t allocation_record_id;
} ds4_gpu_laguna_resident_host_owner;

/* Return1 on complete success,0 on refusal. Allocate requires a zero handle,
 * positive checked count/item_bytes, and the identical safe attached tracker.
 * Only LEDGER_ARRAYS and OTHER_HOST_ENGINE..OTHER_HOST_SERIALIZER are admitted.
 * Physical allocation/free events precede publication/retirement respectively.
 * A foreign/null tracker cannot mutate either tracker or a nonempty handle. */
int ds4_gpu_laguna_resident_host_calloc(
    ds4_runtime_tracker *tracker, uint32_t callsite_id,
    uint64_t count, uint64_t item_bytes,
    ds4_gpu_laguna_resident_host_owner *out);

/* Null/zero handles are successful no-ops even unattached. Nonempty free requires
 * attached identity and an authenticated live raw-host owner without relations.
 * Refusal retains the handle; legitimate cleanup works while sticky-unsafe and
 * clears it only after physical free and retirement. No generic fallback. */
int ds4_gpu_laguna_resident_host_free(
    ds4_runtime_tracker *tracker, ds4_gpu_laguna_resident_host_owner *owner);

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
