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
