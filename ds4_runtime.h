#ifndef DS4_RUNTIME_H
#define DS4_RUNTIME_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    DS4_RUNTIME_CATEGORY_STATIC_WEIGHTS = 0,
    DS4_RUNTIME_CATEGORY_EXPERT_CACHE_PAYLOAD = 1,
    DS4_RUNTIME_CATEGORY_CACHE_METADATA_ADDRESS_TABLES = 2,
    DS4_RUNTIME_CATEGORY_KV_STATE = 3,
    DS4_RUNTIME_CATEGORY_GRAPH_SCRATCH = 4,
    DS4_RUNTIME_CATEGORY_PINNED_STAGING = 5,
    DS4_RUNTIME_CATEGORY_OTHER_HOST = 6,
    DS4_RUNTIME_CATEGORY_OTHER_CUDA = 7,
    DS4_RUNTIME_OWNED_CATEGORY_COUNT = 8,
} ds4_runtime_category;

typedef enum {
    DS4_RUNTIME_REPORT_MODEL_MAPPED_VIRTUAL = 0,
    DS4_RUNTIME_REPORT_MODEL_MAPPING_REGISTERED = 1,
    DS4_RUNTIME_REPORT_MODEL_SOURCE_RESIDENT = 2,
    DS4_RUNTIME_REPORT_HOST_LIBRARY_UNATTRIBUTED = 3,
    DS4_RUNTIME_REPORT_CUDA_LIBRARY_UNATTRIBUTED = 4,
    DS4_RUNTIME_REPORT_COUNT = 5,
} ds4_runtime_report;

typedef enum {
    DS4_RUNTIME_DOMAIN_HOST = 0,
    DS4_RUNTIME_DOMAIN_CUDA_DEVICE = 1,
    DS4_RUNTIME_DOMAIN_CUDA_MANAGED = 2,
    DS4_RUNTIME_DOMAIN_COUNT = 3,
} ds4_runtime_physical_domain;

typedef enum {
    DS4_RUNTIME_RELATION_OWNED_ALLOCATION = 0,
    DS4_RUNTIME_RELATION_MODEL_MAPPING = 1,
    DS4_RUNTIME_RELATION_REGISTRATION = 2,
    DS4_RUNTIME_RELATION_MANAGED_HOST_VISIBLE = 3,
} ds4_runtime_relation;

typedef enum {
    DS4_RUNTIME_STATUS_OK = 0,
    DS4_RUNTIME_STATUS_UNSAFE = 1,
} ds4_runtime_status;

typedef enum {
    DS4_RUNTIME_VIOLATION_NONE = 0,
    DS4_RUNTIME_VIOLATION_INVALID_CONFIG,
    DS4_RUNTIME_VIOLATION_OVERFLOW,
    DS4_RUNTIME_VIOLATION_UNKNOWN_CALLSITE,
    DS4_RUNTIME_VIOLATION_DUPLICATE_CALLSITE,
    DS4_RUNTIME_VIOLATION_UNCLASSIFIED_CALLSITE,
    DS4_RUNTIME_VIOLATION_CAPACITY,
    DS4_RUNTIME_VIOLATION_DUPLICATE_ID,
    DS4_RUNTIME_VIOLATION_ADDRESS_OVERFLOW,
    DS4_RUNTIME_VIOLATION_UNDERCHARGE,
    DS4_RUNTIME_VIOLATION_OVERLAP,
    DS4_RUNTIME_VIOLATION_CALLSITE_BOUND,
    DS4_RUNTIME_VIOLATION_CATEGORY_BOUND,
    DS4_RUNTIME_VIOLATION_OWNED_TOTAL_BOUND,
    DS4_RUNTIME_VIOLATION_REPORT_BOUND,
    DS4_RUNTIME_VIOLATION_QUALIFICATION_TOTAL_BOUND,
    DS4_RUNTIME_VIOLATION_RELATION,
    DS4_RUNTIME_VIOLATION_NOT_LIVE,
    DS4_RUNTIME_VIOLATION_LIVE_RELATION,
    DS4_RUNTIME_VIOLATION_EXTERNAL_ATTRIBUTION,
} ds4_runtime_violation;

typedef struct {
    uint32_t id;
    const char *name;
    ds4_runtime_category category;
    ds4_runtime_physical_domain domain;
    uint64_t bound_bytes;
} ds4_runtime_callsite;

typedef struct {
    uint64_t id;
    uint64_t base;
    uint64_t requested_bytes;
    uint64_t charged_bytes;
    ds4_runtime_category category;
    ds4_runtime_physical_domain domain;
    uint32_t callsite_id;
    ds4_runtime_relation relation;
    uint64_t owner_id;
    bool live;
} ds4_runtime_allocation_record;

enum {
    DS4_RUNTIME_DEVICE_UUID_CAPACITY = 96,
    DS4_RUNTIME_NVML_LIBRARY_VERSION_CAPACITY = 96,
    DS4_RUNTIME_BUILD_IDENTITY_BYTES = 32,
};

typedef enum {
    DS4_RUNTIME_EXTERNAL_FAILURE_NONE = 0,
    DS4_RUNTIME_EXTERNAL_FAILURE_INVALID_INPUT,
    DS4_RUNTIME_EXTERNAL_FAILURE_SMAPS_PARSE,
    DS4_RUNTIME_EXTERNAL_FAILURE_SMAPS_OVERFLOW,
    DS4_RUNTIME_EXTERNAL_FAILURE_MODEL_MAPPING_MISMATCH,
    DS4_RUNTIME_EXTERNAL_FAILURE_TRACKED_RANGE_OVERLAP,
    DS4_RUNTIME_EXTERNAL_FAILURE_TRACKED_VMA_MISSING,
    DS4_RUNTIME_EXTERNAL_FAILURE_DUPLICATE_ATTRIBUTION,
    DS4_RUNTIME_EXTERNAL_FAILURE_NVML_API_MISMATCH,
    DS4_RUNTIME_EXTERNAL_FAILURE_NVML_LIBRARY_VERSION_MISMATCH,
    DS4_RUNTIME_EXTERNAL_FAILURE_DEVICE_UUID_MISMATCH,
    DS4_RUNTIME_EXTERNAL_FAILURE_PROCESS_ID_MISMATCH,
    DS4_RUNTIME_EXTERNAL_FAILURE_BUILD_IDENTITY_MISMATCH,
    DS4_RUNTIME_EXTERNAL_FAILURE_NVML_DUPLICATE_PID,
    DS4_RUNTIME_EXTERNAL_FAILURE_NVML_PROCESS_MISSING,
    DS4_RUNTIME_EXTERNAL_FAILURE_NVML_USAGE_UNKNOWN,
    DS4_RUNTIME_EXTERNAL_FAILURE_UNRELATED_PROCESS_CHANGED,
    DS4_RUNTIME_EXTERNAL_FAILURE_CUDA_NEGATIVE_GAP,
    DS4_RUNTIME_EXTERNAL_FAILURE_HOST_UNATTRIBUTED_BOUND,
    DS4_RUNTIME_EXTERNAL_FAILURE_CUDA_UNATTRIBUTED_BOUND,
} ds4_runtime_external_failure;

typedef struct {
    uint32_t pid;
    uint64_t used_bytes;
    bool used_bytes_known;
} ds4_runtime_nvml_process_sample;

typedef struct {
    uint32_t api_version;
    const char *library_version;
    const char *device_uuid;
    const ds4_runtime_nvml_process_sample *processes;
    size_t process_count;
} ds4_runtime_nvml_inventory;

typedef struct {
    const char *smaps_text;
    size_t smaps_text_bytes;
    uint32_t model_device_major;
    uint32_t model_device_minor;
    uint64_t model_inode;
    uint64_t model_map_base;
    uint64_t model_map_bytes;
    uint64_t model_file_offset;
    const ds4_runtime_allocation_record *attribution_records;
    size_t attribution_record_count;

    uint32_t expected_nvml_api_version;
    const char *expected_nvml_library_version;
    const char *expected_device_uuid;
    uint32_t own_pid;
    const uint8_t *expected_build_identity;
    const uint8_t *observed_build_identity;
    size_t build_identity_bytes;
    uint32_t baseline_nvml_api_version;
    const char *baseline_nvml_library_version;
    const char *baseline_device_uuid;
    uint32_t baseline_process_id;
    bool baseline_nvml_process_present;
    bool baseline_nvml_process_bytes_known;
    uint64_t baseline_nvml_process_bytes;
    uint64_t baseline_tracked_cuda_physical_bytes;
    const ds4_runtime_nvml_inventory *pre_child_inventory;
    const ds4_runtime_nvml_inventory *checkpoint_before_inventory;
    const ds4_runtime_nvml_inventory *inside_ds4_inventory;
    const ds4_runtime_nvml_inventory *checkpoint_after_inventory;

    bool cuda_mem_info_known;
    uint64_t cuda_mem_free_bytes;
    uint64_t cuda_mem_total_bytes;
    uint64_t model_source_page_size;
    uint64_t model_source_resident_bytes;
    uint64_t model_source_mapped_page_bytes;
} ds4_runtime_external_checkpoint_input;

typedef struct {
    ds4_runtime_external_failure failure;
    bool attributed_valid;
    uint64_t attributed_generation;
    uint64_t checkpoint_sequence;

    uint32_t smaps_model_device_major;
    uint32_t smaps_model_device_minor;
    uint64_t smaps_model_inode;
    uint64_t smaps_vma_count;
    uint64_t smaps_total_pss_bytes;
    uint64_t smaps_model_vma_count;
    uint64_t smaps_model_pss_bytes;
    uint64_t smaps_tracked_vma_count;
    uint64_t smaps_tracked_pss_bytes;
    uint64_t host_library_unattributed_bytes;

    uint32_t nvml_api_version;
    char nvml_library_version[
        DS4_RUNTIME_NVML_LIBRARY_VERSION_CAPACITY];
    char device_uuid[DS4_RUNTIME_DEVICE_UUID_CAPACITY];
    uint32_t process_id;
    bool nvml_process_baseline_present;
    uint64_t nvml_process_baseline_bytes;
    uint64_t tracked_cuda_physical_baseline_bytes;
    uint64_t nvml_process_bytes;
    uint64_t tracked_cuda_physical_bytes;
    uint64_t cuda_library_unattributed_bytes;
    uint64_t cuda_mem_free_bytes;
    uint64_t cuda_mem_total_bytes;
    bool unrelated_process_inventory_stable;
} ds4_runtime_external_sample;

typedef struct {
    const ds4_runtime_callsite *callsites;
    size_t callsite_count;
    ds4_runtime_allocation_record *records;
    size_t record_capacity;
    uint64_t category_bounds[DS4_RUNTIME_OWNED_CATEGORY_COUNT];
    uint64_t report_bounds[DS4_RUNTIME_REPORT_COUNT];
    uint64_t owned_total_bound_bytes;
    uint64_t qualification_total_bound_bytes;
} ds4_runtime_tracker_config;

typedef struct {
    const ds4_runtime_callsite *callsites;
    size_t callsite_count;
    ds4_runtime_allocation_record *records;
    size_t record_capacity;
    size_t record_count;
    uint64_t issued_sequence_high_water[256];

    uint64_t category_bounds[DS4_RUNTIME_OWNED_CATEGORY_COUNT];
    uint64_t category_current[DS4_RUNTIME_OWNED_CATEGORY_COUNT];
    uint64_t category_peak[DS4_RUNTIME_OWNED_CATEGORY_COUNT];
    uint64_t report_bounds[DS4_RUNTIME_REPORT_COUNT];
    uint64_t report_current[DS4_RUNTIME_REPORT_COUNT];
    uint64_t report_peak[DS4_RUNTIME_REPORT_COUNT];

    uint64_t owned_total_bound_bytes;
    uint64_t owned_total_current;
    uint64_t owned_total_peak;
    uint64_t qualification_total_bound_bytes;
    uint64_t qualification_total_current;
    uint64_t qualification_total_peak;
    uint64_t event_sequence;
    ds4_runtime_external_sample external_sample;
    ds4_runtime_violation violation;
} ds4_runtime_tracker;

typedef struct {
    uint64_t category_bounds[DS4_RUNTIME_OWNED_CATEGORY_COUNT];
    uint64_t category_current[DS4_RUNTIME_OWNED_CATEGORY_COUNT];
    uint64_t category_peak[DS4_RUNTIME_OWNED_CATEGORY_COUNT];
    uint64_t report_bounds[DS4_RUNTIME_REPORT_COUNT];
    uint64_t report_current[DS4_RUNTIME_REPORT_COUNT];
    uint64_t report_peak[DS4_RUNTIME_REPORT_COUNT];
    uint64_t owned_total_bound_bytes;
    uint64_t owned_total_current;
    uint64_t owned_total_peak;
    uint64_t qualification_total_bound_bytes;
    uint64_t qualification_total_current;
    uint64_t qualification_total_peak;
    uint64_t event_sequence;
    ds4_runtime_external_sample external_sample;
    ds4_runtime_violation violation;
    size_t active_record_count;
} ds4_runtime_snapshot;

typedef struct {
    uint64_t allocation_id;
    uint32_t callsite_id;
    uint64_t base;
    uint64_t requested_bytes;
    uint64_t charged_bytes;
} ds4_runtime_owned_descriptor;

/* Compute rows * per_row_bytes + fixed_bytes without overflowing. Failure
 * leaves bytes_out unchanged. */
bool ds4_runtime_checked_affine_bytes(
    uint64_t rows,
    uint64_t per_row_bytes,
    uint64_t fixed_bytes,
    uint64_t *bytes_out);

ds4_runtime_status ds4_runtime_tracker_init(
    ds4_runtime_tracker *tracker,
    const ds4_runtime_tracker_config *config);

ds4_runtime_status ds4_runtime_tracker_allocate(
    ds4_runtime_tracker *tracker,
    uint64_t allocation_id,
    uint32_t callsite_id,
    uint64_t base,
    uint64_t requested_bytes,
    uint64_t charged_bytes);

/* Allocate with the next sequence in producer_namespace. The returned ID is
 * monotonically increasing within that namespace, including across record
 * release and tombstone reuse. On failure allocation_id_out is unchanged
 * unless the underlying allocation was committed before a bound violation
 * was detected, in which case it receives the live record's ID. */
ds4_runtime_status ds4_runtime_tracker_allocate_next(
    ds4_runtime_tracker *tracker,
    uint8_t producer_namespace,
    uint32_t callsite_id,
    uint64_t base,
    uint64_t requested_bytes,
    uint64_t charged_bytes,
    uint64_t *allocation_id_out);

ds4_runtime_status ds4_runtime_tracker_replay_owned(
    ds4_runtime_tracker *tracker,
    const ds4_runtime_owned_descriptor *owners,
    size_t owner_count,
    bool *owner_live);

ds4_runtime_status ds4_runtime_tracker_release(
    ds4_runtime_tracker *tracker,
    uint64_t allocation_id);

ds4_runtime_status ds4_runtime_tracker_map_model(
    ds4_runtime_tracker *tracker,
    uint64_t mapping_id,
    uint64_t base,
    uint64_t bytes);

ds4_runtime_status ds4_runtime_tracker_unmap_model(
    ds4_runtime_tracker *tracker,
    uint64_t mapping_id);

ds4_runtime_status ds4_runtime_tracker_register(
    ds4_runtime_tracker *tracker,
    uint64_t registration_id,
    uint64_t base,
    uint64_t bytes,
    uint64_t owner_id);

ds4_runtime_status ds4_runtime_tracker_managed_host_relation(
    ds4_runtime_tracker *tracker,
    uint64_t relation_id,
    uint64_t base,
    uint64_t bytes,
    uint64_t owner_id);

ds4_runtime_status ds4_runtime_tracker_unregister(
    ds4_runtime_tracker *tracker,
    uint64_t relation_id);

ds4_runtime_status ds4_runtime_tracker_checkpoint_external(
    ds4_runtime_tracker *tracker,
    uint64_t model_source_resident_bytes,
    uint64_t host_library_unattributed_bytes,
    uint64_t cuda_library_unattributed_bytes);

/* Reconcile one synchronized qualification checkpoint from recorded smaps,
 * process-scoped NVML inventories, cudaMemGetInfo, and the live allocation
 * attribution table.  The operation is transactional: a failed sample
 * latches external attribution unsafe without committing report totals. */
ds4_runtime_status ds4_runtime_tracker_checkpoint_attributed(
    ds4_runtime_tracker *tracker,
    const ds4_runtime_external_checkpoint_input *input,
    ds4_runtime_external_sample *sample_out);

/* Update only the model-source observation while retaining the most recent
 * host- and CUDA-library unattributed observations in the same checkpoint. */
ds4_runtime_status ds4_runtime_tracker_checkpoint_model_source(
    ds4_runtime_tracker *tracker,
    uint64_t model_source_resident_bytes);

bool ds4_runtime_tracker_snapshot_copy(
    const ds4_runtime_tracker *tracker,
    ds4_runtime_snapshot *snapshot,
    ds4_runtime_allocation_record *active_records,
    size_t active_record_capacity);

bool ds4_runtime_reduction_qualified(
    uint64_t resident_qualification_total_peak,
    uint64_t streamed_qualification_total_peak,
    uint64_t *reduction_bytes_out);

#ifdef __cplusplus
}
#endif

#endif
