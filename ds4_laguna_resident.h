#ifndef DS4_LAGUNA_RESIDENT_H
#define DS4_LAGUNA_RESIDENT_H

#include "ds4_laguna_stream.h"
#include "ds4_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DS4_LAGUNA_RESIDENT_CALLSITE_COUNT 14u
/* Existing IDs are retained for shared graph/host ownership hooks. */
#define DS4_LAGUNA_RESIDENT_CALLSITE_OTHER_MANAGED 26u

typedef struct {
    uint32_t context_tokens;
    uint32_t prefill_rows;
    uint32_t session_count;
    uint64_t host_capacity_bytes;
    uint64_t cuda_capacity_bytes;
    uint64_t source_page_size;
} ds4_laguna_resident_plan_spec;

typedef struct {
    uint32_t context_tokens;
    uint32_t prefill_rows;
    uint32_t session_count;
    uint64_t category_bounds[DS4_RUNTIME_OWNED_CATEGORY_COUNT];
    uint64_t report_bounds[DS4_RUNTIME_REPORT_COUNT];
    uint64_t owned_total_bound_bytes;
    uint64_t qualification_total_bound_bytes;
    ds4_runtime_callsite callsites[DS4_LAGUNA_RESIDENT_CALLSITE_COUNT];
    size_t callsite_count;
} ds4_laguna_resident_plan;

/* Construct conservative reservation bounds, not an allocation inventory or
 * a runtime snapshot. The caller must supply observed host/CUDA capacities
 * and the native source page size. Separate device/managed/cache/pinned bounds
 * can overlap the same hardware capacity: their sum is not predicted usage,
 * a reservation, a fit guarantee, or qualification admission.
 *
 * Resident is exactly 32K/4K/one session and has no compact expert cache.
 * The native caller must supply a builder-produced ledger bound to its retained
 * model. This helper checks pinned scalar geometry, not backing arrays, range
 * contents, file identity, or model provenance. Synthetic geometry is enough
 * for bounds-only tests, never for accepting a native snapshot.
 *
 * Actual allocator events must populate a tracker initialized from this plan.
 * Resident pinned owners map to ID 11, other-device owners to ID 22, and managed
 * owners to ID 26; compact pool/workspace IDs are not interchangeable here.
 * Model mapping/registration are report-only; source residency is measured
 * separately. Failure leaves a nonnull output fully zeroed. */
bool ds4_laguna_resident_plan_make(
    ds4_laguna_resident_plan *out,
    const ds4_laguna_ledger *ledger,
    const ds4_laguna_resident_plan_spec *spec,
    char *error,
    size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
