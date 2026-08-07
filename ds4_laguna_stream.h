#ifndef DS4_LAGUNA_STREAM_H
#define DS4_LAGUNA_STREAM_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "ds4_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DS4_LAGUNA_MAX_DIMS 8u
#define DS4_LAGUNA_ALLOCATION_CALLSITE_COUNT 25u
#define DS4_LAGUNA_TENSOR_RANGE_HAS_ROUTED_IDENTITY 1

typedef enum {
    DS4_LAGUNA_TENSOR_UNCLASSIFIED = 0,
    DS4_LAGUNA_TENSOR_STATIC = 1,
    DS4_LAGUNA_TENSOR_ROUTED_EXPERT = 2,
} ds4_laguna_tensor_class;

typedef enum {
    DS4_LAGUNA_ROUTED_PROJECTION_NONE = 0,
    DS4_LAGUNA_ROUTED_PROJECTION_GATE = 1,
    DS4_LAGUNA_ROUTED_PROJECTION_UP = 2,
    DS4_LAGUNA_ROUTED_PROJECTION_DOWN = 3,
} ds4_laguna_routed_projection;

typedef enum {
    DS4_LAGUNA_SOURCE_HEADER = 1,
    DS4_LAGUNA_SOURCE_METADATA = 2,
    DS4_LAGUNA_SOURCE_TENSOR_DIRECTORY = 3,
    DS4_LAGUNA_SOURCE_ALIGNMENT_PADDING = 4,
    DS4_LAGUNA_SOURCE_TENSOR_PADDING = 5,
} ds4_laguna_source_range_kind;

typedef struct {
    uint64_t file_size;
    uint64_t header_end;
    uint64_t metadata_end;
    uint64_t tensor_directory_end;
    uint64_t tensor_data_start;
    uint64_t gguf_alignment;
    uint64_t device_alignment;
    uint32_t first_routed_layer;
    uint32_t layer_count;
    uint32_t expert_count;
} ds4_laguna_ledger_spec;

typedef struct {
    uint64_t stable_index;
    const char *name;
    size_t name_len;
    uint64_t source_offset;
    uint64_t source_bytes;
    uint32_t ndim;
    uint64_t dim[DS4_LAGUNA_MAX_DIMS];
    uint32_t gguf_type;
    uint32_t block_elems;
    uint32_t block_bytes;
    ds4_laguna_tensor_class tensor_class;
    uint32_t routed_layer;
    ds4_laguna_routed_projection routed_projection;
} ds4_laguna_tensor_desc;

typedef struct {
    uint64_t stable_index;
    ds4_laguna_tensor_class tensor_class;
    uint32_t routed_layer;
    ds4_laguna_routed_projection routed_projection;
    uint64_t source_offset;
    uint64_t source_bytes;
} ds4_laguna_tensor_range;

typedef struct {
    ds4_laguna_source_range_kind kind;
    uint64_t source_offset;
    uint64_t source_bytes;
} ds4_laguna_source_range;

typedef struct {
    uint64_t parent_stable_index;
    uint64_t source_offset;
    uint64_t source_bytes;
    uint64_t device_offset;
} ds4_laguna_expert_view;

typedef struct {
    uint32_t layer;
    uint32_t expert;
    ds4_laguna_expert_view gate;
    ds4_laguna_expert_view up;
    ds4_laguna_expert_view down;
    uint64_t used_bytes;
} ds4_laguna_expert_entry;

typedef struct {
    ds4_laguna_tensor_range *tensor_ranges;
    size_t tensor_range_count;
    ds4_laguna_source_range *source_ranges;
    size_t source_range_count;
    ds4_laguna_expert_entry *expert_entries;

    uint64_t file_size;
    uint64_t tensor_count;
    uint64_t static_parent_count;
    uint64_t routed_parent_count;
    uint64_t expert_entry_count;
    uint64_t routed_source_bytes;
    uint64_t static_source_bytes;
    uint64_t static_aligned_device_bytes;
    uint64_t non_tensor_source_bytes;
    uint64_t routed_projection_expert_bytes;
    uint64_t slot_stride_bytes;
} ds4_laguna_ledger;

typedef enum {
    DS4_LAGUNA_CALLSITE_STATIC_SLAB = 1,
    DS4_LAGUNA_CALLSITE_EXPERT_CACHE_PAYLOAD = 2,
    DS4_LAGUNA_CALLSITE_LEDGER_ARRAYS = 3,
    DS4_LAGUNA_CALLSITE_ROUTE_HOTNESS = 4,
    DS4_LAGUNA_CALLSITE_HOST_ENTRY_TO_SLOT = 5,
    DS4_LAGUNA_CALLSITE_DEVICE_ENTRY_TO_SLOT = 6,
    DS4_LAGUNA_CALLSITE_STATIC_OFFSETS = 7,
    DS4_LAGUNA_CALLSITE_SLOT_STATE = 8,
    DS4_LAGUNA_CALLSITE_KV_STATE = 9,
    DS4_LAGUNA_CALLSITE_GRAPH_SCRATCH = 10,
    DS4_LAGUNA_CALLSITE_PINNED_STAGING_0 = 11,
    DS4_LAGUNA_CALLSITE_PINNED_STAGING_1 = 12,
    DS4_LAGUNA_CALLSITE_PINNED_STAGING_2 = 13,
    DS4_LAGUNA_CALLSITE_PINNED_STAGING_3 = 14,
    DS4_LAGUNA_CALLSITE_OTHER_HOST_ENGINE = 15,
    DS4_LAGUNA_CALLSITE_OTHER_HOST_MODEL = 16,
    DS4_LAGUNA_CALLSITE_OTHER_HOST_BOOTSTRAP = 17,
    DS4_LAGUNA_CALLSITE_OTHER_HOST_VOCAB = 18,
    DS4_LAGUNA_CALLSITE_OTHER_HOST_SESSION = 19,
    DS4_LAGUNA_CALLSITE_OTHER_HOST_TRACKER = 20,
    DS4_LAGUNA_CALLSITE_OTHER_HOST_SERIALIZER = 21,
    DS4_LAGUNA_CALLSITE_OTHER_CUDA_KERNEL_TMP = 22,
    DS4_LAGUNA_CALLSITE_OTHER_CUDA_ROUTED_WORKSPACE = 23,
    DS4_LAGUNA_CALLSITE_OTHER_CUDA_DESCRIPTOR_UPLOAD = 24,
    DS4_LAGUNA_CALLSITE_OTHER_CUDA_TRANSIENT = 25,
} ds4_laguna_allocation_callsite_id;

typedef struct {
    uint64_t configured_cache_bytes;
    uint32_t context_tokens;
    uint32_t prefill_rows;
    uint32_t session_count;
} ds4_laguna_allocation_plan_spec;

typedef struct {
    const char *profile_id;
    uint32_t context_tokens;
    uint32_t prefill_rows;
    uint32_t session_count;
    uint64_t configured_cache_bytes;
    uint64_t effective_cache_limit_bytes;
    uint64_t slot_stride_bytes;
    uint64_t cache_payload_bytes;
    uint64_t cache_tail_uncharged_bytes;
    uint32_t slot_count;
    uint32_t staging_buffer_count;
    uint64_t staging_buffer_bytes;

    uint64_t owned_category_bounds[DS4_RUNTIME_OWNED_CATEGORY_COUNT];
    uint64_t report_bounds[DS4_RUNTIME_REPORT_COUNT];
    uint64_t owned_non_cache_bound_bytes;
    uint64_t owned_total_bound_bytes;
    uint64_t qualification_non_cache_bound_bytes;
    uint64_t planned_qualification_bytes;
    uint64_t qualification_total_bound_bytes;

    ds4_runtime_callsite callsites[DS4_LAGUNA_ALLOCATION_CALLSITE_COUNT];
    size_t callsite_count;
} ds4_laguna_allocation_plan;

typedef struct {
    uint64_t offset;
    uint64_t bytes;
} ds4_laguna_page_range;

#define DS4_LAGUNA_CACHE_SLOT_NONE UINT32_MAX

typedef enum {
    DS4_LAGUNA_CACHE_OK = 0,
    DS4_LAGUNA_CACHE_RECOVERABLE = 1,
    DS4_LAGUNA_CACHE_UNSAFE = 2,
} ds4_laguna_cache_status;

typedef enum {
    DS4_LAGUNA_CACHE_ACQUIRE_NONE = 0,
    DS4_LAGUNA_CACHE_ACQUIRE_HIT_RESERVED = 1,
    DS4_LAGUNA_CACHE_ACQUIRE_LOAD_OWNER = 2,
    DS4_LAGUNA_CACHE_ACQUIRE_BUSY_LOADING = 3,
    DS4_LAGUNA_CACHE_ACQUIRE_BUSY_IN_USE = 4,
    DS4_LAGUNA_CACHE_ACQUIRE_PRESSURE = 5,
} ds4_laguna_cache_acquire_outcome;

typedef enum {
    DS4_LAGUNA_CACHE_SLOT_EMPTY = 0,
    DS4_LAGUNA_CACHE_SLOT_LOADING = 1,
    DS4_LAGUNA_CACHE_SLOT_READY = 2,
    DS4_LAGUNA_CACHE_SLOT_IN_USE = 3,
} ds4_laguna_cache_slot_state;

typedef struct {
    uint32_t layer_id;
    uint32_t expert_id;
} ds4_laguna_expert_key;

/* This layout is part of the allocation-plan contract: slot metadata is
 * charged at exactly 32 bytes per cache slot. */
typedef struct {
    uint64_t generation;
    uint64_t last_used;
    uint32_t layer;
    uint32_t expert;
    uint32_t refs;
    uint32_t state;
} ds4_laguna_cache_slot;

typedef char ds4_laguna_cache_slot_must_be_32_bytes[
    sizeof(ds4_laguna_cache_slot) == 32u ? 1 : -1];

typedef struct {
    uint32_t slot_index;
    uint64_t generation;
    size_t entry_index;
    ds4_laguna_expert_key key;
} ds4_laguna_cache_handle;

typedef struct {
    uint32_t first_key;
    uint32_t key_count;
} ds4_laguna_expert_group;

/* Pure borrowed state. The caller owns every referenced array for the engine
 * lifetime; this policy never allocates, frees, performs I/O, or submits work. */
typedef struct {
    const ds4_laguna_expert_entry *entries;
    size_t entry_count;
    ds4_laguna_cache_slot *slots;
    size_t slot_count;
    uint64_t *route_hotness;
    uint32_t *entry_to_slot;
    size_t max_selected_per_token;
    uint64_t sequence;
} ds4_laguna_cache_policy;

bool ds4_laguna_ledger_build(ds4_laguna_ledger *out,
                             const ds4_laguna_ledger_spec *spec,
                             const ds4_laguna_tensor_desc *tensors,
                             size_t n_tensors,
                             char *err,
                             size_t errlen);

/* `out` must be zero-initialized or previously released with
 * ds4_laguna_ledger_free(). A successful ledger owns its three arrays. */

void ds4_laguna_ledger_free(ds4_laguna_ledger *ledger);

bool ds4_laguna_allocation_plan_make(
    ds4_laguna_allocation_plan *out,
    const ds4_laguna_ledger *ledger,
    const ds4_laguna_allocation_plan_spec *spec,
    char *err,
    size_t errlen);

bool ds4_laguna_full_page_union(
    const ds4_laguna_page_range *input,
    size_t input_count,
    uint64_t page_size,
    ds4_laguna_page_range *output,
    size_t output_capacity,
    size_t *output_count,
    uint64_t *output_bytes);

ds4_laguna_cache_status ds4_laguna_cache_policy_init(
    ds4_laguna_cache_policy *policy,
    const ds4_laguna_expert_entry *entries,
    size_t entry_count,
    ds4_laguna_cache_slot *slots,
    size_t slot_count,
    uint64_t *route_hotness,
    uint32_t *entry_to_slot,
    size_t max_selected_per_token);

ds4_laguna_cache_status ds4_laguna_cache_policy_note_route(
    ds4_laguna_cache_policy *policy,
    ds4_laguna_expert_key key);

ds4_laguna_cache_status ds4_laguna_cache_policy_note_routes(
    ds4_laguna_cache_policy *policy,
    const ds4_laguna_expert_key *keys,
    size_t key_count);

/* HIT_RESERVED and LOAD_OWNER are the only outcomes that return a valid,
 * generation-and-key-bound handle. HIT_RESERVED enters IN_USE with one ref;
 * publishing LOAD_OWNER does the same. BUSY and PRESSURE return an invalid
 * handle and require the caller to retry rather than mutate another owner's
 * slot. */
ds4_laguna_cache_status ds4_laguna_cache_policy_acquire(
    ds4_laguna_cache_policy *policy,
    ds4_laguna_expert_key key,
    ds4_laguna_cache_handle *handle,
    ds4_laguna_cache_acquire_outcome *outcome);

ds4_laguna_cache_status ds4_laguna_cache_policy_publish(
    ds4_laguna_cache_policy *policy,
    ds4_laguna_cache_handle handle);

ds4_laguna_cache_status ds4_laguna_cache_policy_pin(
    ds4_laguna_cache_policy *policy,
    ds4_laguna_cache_handle handle);

ds4_laguna_cache_status ds4_laguna_cache_policy_unpin(
    ds4_laguna_cache_policy *policy,
    ds4_laguna_cache_handle handle);

ds4_laguna_cache_status ds4_laguna_cache_policy_fail(
    ds4_laguna_cache_policy *policy,
    ds4_laguna_cache_handle handle);

ds4_laguna_cache_status ds4_laguna_cache_policy_cancel(
    ds4_laguna_cache_policy *policy,
    ds4_laguna_cache_handle handle);

/* Drain is a non-destructive readiness check: while any slot is LOADING or
 * IN_USE it returns RECOVERABLE without changing policy state. Once ready,
 * it clears only reusable READY slots. */
ds4_laguna_cache_status ds4_laguna_cache_policy_drain(
    ds4_laguna_cache_policy *policy);

/* Full O(entry_count + slot_count log entry_count) invariant audit for
 * startup, tests, and fault boundaries. Hot transitions intentionally perform
 * only touched-key/slot/map checks. */
ds4_laguna_cache_status ds4_laguna_cache_policy_audit(
    const ds4_laguna_cache_policy *policy);

/* `selected` is flattened token-major input for one routed layer. The helper
 * emits unique keys in stable first-occurrence order and slot-sized group
 * descriptors into caller-owned buffers. `grouped_keys == selected` supports
 * exact in-place stable compaction; any partial overlap, or overlap involving
 * `groups`, is unsafe and rejected before either array is modified. Successful
 * output capacity is zero-filled for byte-identical repeated output. The
 * disjoint path uses allocation-free O(selected_count * unique) stable
 * deduplication. Exact in-place compaction adds an O(selected_count^2)
 * preflight so capacity failure can preserve the input; caller-owned scratch
 * remains deferred until measured usage warrants extending the contract. */
ds4_laguna_cache_status ds4_laguna_cache_policy_group(
    const ds4_laguna_cache_policy *policy,
    const ds4_laguna_expert_key *selected,
    size_t token_count,
    size_t selected_per_token,
    ds4_laguna_expert_key *grouped_keys,
    size_t grouped_key_capacity,
    ds4_laguna_expert_group *groups,
    size_t group_capacity,
    size_t *grouped_key_count,
    size_t *group_count);

#ifdef __cplusplus
}
#endif

#endif
