#include "ds4_runtime.h"

#include <limits.h>
#include <string.h>

static bool add_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (UINT64_MAX - a < b) {
        if (out) *out = UINT64_MAX;
        return false;
    }
    if (out) *out = a + b;
    return true;
}

static bool range_end(uint64_t base, uint64_t bytes, uint64_t *end) {
    return bytes != 0 && add_u64(base, bytes, end);
}

static void latch(ds4_runtime_tracker *tracker,
                  ds4_runtime_violation violation) {
    if (tracker && tracker->violation == DS4_RUNTIME_VIOLATION_NONE) {
        tracker->violation = violation;
    }
}

static ds4_runtime_status status(const ds4_runtime_tracker *tracker) {
    return tracker && tracker->violation == DS4_RUNTIME_VIOLATION_NONE
        ? DS4_RUNTIME_STATUS_OK
        : DS4_RUNTIME_STATUS_UNSAFE;
}

static const ds4_runtime_callsite *find_callsite(
        const ds4_runtime_tracker *tracker, uint32_t id) {
    if (!tracker) return NULL;
    for (size_t i = 0; i < tracker->callsite_count; i++) {
        if (tracker->callsites[i].id == id) return &tracker->callsites[i];
    }
    return NULL;
}

static ds4_runtime_allocation_record *find_record(
        ds4_runtime_tracker *tracker, uint64_t id) {
    if (!tracker) return NULL;
    for (size_t i = 0; i < tracker->record_count; i++) {
        if (tracker->records[i].id == id) return &tracker->records[i];
    }
    return NULL;
}

static bool records_overlap(const ds4_runtime_allocation_record *a,
                            uint64_t base, uint64_t bytes) {
    uint64_t a_end = 0;
    uint64_t b_end = 0;
    return range_end(a->base, a->charged_bytes, &a_end) &&
           range_end(base, bytes, &b_end) &&
           a->base < b_end && base < a_end;
}

static bool relation_records_overlap(
        const ds4_runtime_allocation_record *a,
        uint64_t base,
        uint64_t bytes) {
    uint64_t a_end = 0;
    uint64_t b_end = 0;
    return range_end(a->base, a->requested_bytes, &a_end) &&
           range_end(base, bytes, &b_end) &&
           a->base < b_end && base < a_end;
}

static bool record_contains(const ds4_runtime_allocation_record *owner,
                            uint64_t base, uint64_t bytes) {
    uint64_t owner_end = 0;
    uint64_t range_limit = 0;
    return range_end(owner->base, owner->requested_bytes, &owner_end) &&
           range_end(base, bytes, &range_limit) &&
           owner->base <= base && range_limit <= owner_end;
}

static ds4_runtime_allocation_record *append_record(
        ds4_runtime_tracker *tracker, uint64_t id) {
    if (find_record(tracker, id)) {
        latch(tracker, DS4_RUNTIME_VIOLATION_DUPLICATE_ID);
        return NULL;
    }
    if (tracker->record_count >= tracker->record_capacity) {
        latch(tracker, DS4_RUNTIME_VIOLATION_CAPACITY);
        return NULL;
    }
    ds4_runtime_allocation_record *record =
        &tracker->records[tracker->record_count++];
    memset(record, 0, sizeof(*record));
    record->id = id;
    record->category = (ds4_runtime_category)DS4_RUNTIME_OWNED_CATEGORY_COUNT;
    record->live = true;
    return record;
}

static void recompute(ds4_runtime_tracker *tracker) {
    uint64_t categories[DS4_RUNTIME_OWNED_CATEGORY_COUNT] = {0};
    uint64_t reports[DS4_RUNTIME_REPORT_COUNT] = {0};
    reports[DS4_RUNTIME_REPORT_MODEL_SOURCE_RESIDENT] =
        tracker->report_current[DS4_RUNTIME_REPORT_MODEL_SOURCE_RESIDENT];
    reports[DS4_RUNTIME_REPORT_HOST_LIBRARY_UNATTRIBUTED] =
        tracker->report_current[DS4_RUNTIME_REPORT_HOST_LIBRARY_UNATTRIBUTED];
    reports[DS4_RUNTIME_REPORT_CUDA_LIBRARY_UNATTRIBUTED] =
        tracker->report_current[DS4_RUNTIME_REPORT_CUDA_LIBRARY_UNATTRIBUTED];

    for (size_t i = 0; i < tracker->record_count; i++) {
        const ds4_runtime_allocation_record *record = &tracker->records[i];
        if (!record->live) continue;
        if (record->relation == DS4_RUNTIME_RELATION_OWNED_ALLOCATION) {
            if (record->category < 0 ||
                record->category >= DS4_RUNTIME_OWNED_CATEGORY_COUNT) {
                latch(tracker, DS4_RUNTIME_VIOLATION_UNCLASSIFIED_CALLSITE);
                continue;
            }
            uint64_t sum = 0;
            if (!add_u64(categories[record->category],
                         record->charged_bytes, &sum)) {
                latch(tracker, DS4_RUNTIME_VIOLATION_OVERFLOW);
            }
            categories[record->category] = sum;
        } else if (record->relation == DS4_RUNTIME_RELATION_MODEL_MAPPING) {
            uint64_t sum = 0;
            if (!add_u64(reports[DS4_RUNTIME_REPORT_MODEL_MAPPED_VIRTUAL],
                         record->requested_bytes, &sum)) {
                latch(tracker, DS4_RUNTIME_VIOLATION_OVERFLOW);
            }
            reports[DS4_RUNTIME_REPORT_MODEL_MAPPED_VIRTUAL] = sum;
        } else if (record->relation == DS4_RUNTIME_RELATION_REGISTRATION) {
            uint64_t sum = 0;
            if (!add_u64(
                    reports[DS4_RUNTIME_REPORT_MODEL_MAPPING_REGISTERED],
                    record->requested_bytes, &sum)) {
                latch(tracker, DS4_RUNTIME_VIOLATION_OVERFLOW);
            }
            reports[DS4_RUNTIME_REPORT_MODEL_MAPPING_REGISTERED] = sum;
        }
    }

    for (size_t i = 0; i < tracker->callsite_count; i++) {
        uint64_t callsite_current = 0;
        for (size_t j = 0; j < tracker->record_count; j++) {
            const ds4_runtime_allocation_record *record = &tracker->records[j];
            if (!record->live ||
                record->relation != DS4_RUNTIME_RELATION_OWNED_ALLOCATION ||
                record->callsite_id != tracker->callsites[i].id) {
                continue;
            }
            uint64_t sum = 0;
            if (!add_u64(callsite_current, record->charged_bytes, &sum)) {
                latch(tracker, DS4_RUNTIME_VIOLATION_OVERFLOW);
            }
            callsite_current = sum;
        }
        if (callsite_current > tracker->callsites[i].bound_bytes) {
            latch(tracker, DS4_RUNTIME_VIOLATION_CALLSITE_BOUND);
        }
    }

    uint64_t owned_total = 0;
    for (size_t i = 0; i < DS4_RUNTIME_OWNED_CATEGORY_COUNT; i++) {
        tracker->category_current[i] = categories[i];
        if (categories[i] > tracker->category_peak[i]) {
            tracker->category_peak[i] = categories[i];
        }
        if (categories[i] > tracker->category_bounds[i]) {
            latch(tracker, DS4_RUNTIME_VIOLATION_CATEGORY_BOUND);
        }
        uint64_t sum = 0;
        if (!add_u64(owned_total, categories[i], &sum)) {
            latch(tracker, DS4_RUNTIME_VIOLATION_OVERFLOW);
        }
        owned_total = sum;
    }
    tracker->owned_total_current = owned_total;
    if (owned_total > tracker->owned_total_peak) {
        tracker->owned_total_peak = owned_total;
    }
    if (owned_total > tracker->owned_total_bound_bytes) {
        latch(tracker, DS4_RUNTIME_VIOLATION_OWNED_TOTAL_BOUND);
    }

    for (size_t i = 0; i < DS4_RUNTIME_REPORT_COUNT; i++) {
        tracker->report_current[i] = reports[i];
        if (reports[i] > tracker->report_peak[i]) {
            tracker->report_peak[i] = reports[i];
        }
        if (reports[i] > tracker->report_bounds[i]) {
            latch(tracker, DS4_RUNTIME_VIOLATION_REPORT_BOUND);
        }
    }

    uint64_t qualification_total = owned_total;
    const ds4_runtime_report physical_reports[] = {
        DS4_RUNTIME_REPORT_MODEL_SOURCE_RESIDENT,
        DS4_RUNTIME_REPORT_HOST_LIBRARY_UNATTRIBUTED,
        DS4_RUNTIME_REPORT_CUDA_LIBRARY_UNATTRIBUTED,
    };
    for (size_t i = 0; i < sizeof(physical_reports) /
                                sizeof(physical_reports[0]); i++) {
        uint64_t sum = 0;
        if (!add_u64(qualification_total,
                     reports[physical_reports[i]], &sum)) {
            latch(tracker, DS4_RUNTIME_VIOLATION_OVERFLOW);
        }
        qualification_total = sum;
    }
    tracker->qualification_total_current = qualification_total;
    if (qualification_total > tracker->qualification_total_peak) {
        tracker->qualification_total_peak = qualification_total;
    }
    if (qualification_total > tracker->qualification_total_bound_bytes) {
        latch(tracker, DS4_RUNTIME_VIOLATION_QUALIFICATION_TOTAL_BOUND);
    }
    tracker->event_sequence++;
}

ds4_runtime_status ds4_runtime_tracker_init(
        ds4_runtime_tracker *tracker,
        const ds4_runtime_tracker_config *config) {
    if (!tracker) return DS4_RUNTIME_STATUS_UNSAFE;
    memset(tracker, 0, sizeof(*tracker));
    if (!config || !config->callsites || config->callsite_count == 0 ||
        !config->records || config->record_capacity == 0) {
        latch(tracker, DS4_RUNTIME_VIOLATION_INVALID_CONFIG);
        return status(tracker);
    }
    if (config->record_capacity >
        SIZE_MAX / sizeof(config->records[0])) {
        latch(tracker, DS4_RUNTIME_VIOLATION_OVERFLOW);
        return status(tracker);
    }
    tracker->callsites = config->callsites;
    tracker->callsite_count = config->callsite_count;
    tracker->records = config->records;
    tracker->record_capacity = config->record_capacity;
    memcpy(tracker->category_bounds, config->category_bounds,
           sizeof(tracker->category_bounds));
    memcpy(tracker->report_bounds, config->report_bounds,
           sizeof(tracker->report_bounds));
    tracker->owned_total_bound_bytes = config->owned_total_bound_bytes;
    tracker->qualification_total_bound_bytes =
        config->qualification_total_bound_bytes;
    memset(tracker->records, 0,
           tracker->record_capacity * sizeof(tracker->records[0]));

    uint64_t callsite_bounds[DS4_RUNTIME_OWNED_CATEGORY_COUNT] = {0};
    for (size_t i = 0; i < tracker->callsite_count; i++) {
        const ds4_runtime_callsite *site = &tracker->callsites[i];
        if (site->id == 0 || !site->name || site->name[0] == '\0' ||
            site->domain < 0 || site->domain >= DS4_RUNTIME_DOMAIN_COUNT) {
            latch(tracker, DS4_RUNTIME_VIOLATION_INVALID_CONFIG);
            return status(tracker);
        }
        if (site->category < 0 ||
            site->category >= DS4_RUNTIME_OWNED_CATEGORY_COUNT) {
            latch(tracker, DS4_RUNTIME_VIOLATION_UNCLASSIFIED_CALLSITE);
            return status(tracker);
        }
        if (site->domain == DS4_RUNTIME_DOMAIN_CUDA_MANAGED &&
            site->category != DS4_RUNTIME_CATEGORY_OTHER_CUDA) {
            latch(tracker, DS4_RUNTIME_VIOLATION_UNCLASSIFIED_CALLSITE);
            return status(tracker);
        }
        for (size_t j = 0; j < i; j++) {
            if (tracker->callsites[j].id == site->id ||
                strcmp(tracker->callsites[j].name, site->name) == 0) {
                latch(tracker, DS4_RUNTIME_VIOLATION_DUPLICATE_CALLSITE);
                return status(tracker);
            }
        }
        uint64_t sum = 0;
        if (!add_u64(callsite_bounds[site->category],
                     site->bound_bytes, &sum)) {
            latch(tracker, DS4_RUNTIME_VIOLATION_OVERFLOW);
            return status(tracker);
        }
        callsite_bounds[site->category] = sum;
    }
    for (size_t i = 0; i < DS4_RUNTIME_OWNED_CATEGORY_COUNT; i++) {
        if (callsite_bounds[i] != tracker->category_bounds[i]) {
            latch(tracker, DS4_RUNTIME_VIOLATION_INVALID_CONFIG);
            return status(tracker);
        }
    }

    uint64_t owned_bound = 0;
    for (size_t i = 0; i < DS4_RUNTIME_OWNED_CATEGORY_COUNT; i++) {
        uint64_t sum = 0;
        if (!add_u64(owned_bound, tracker->category_bounds[i], &sum)) {
            latch(tracker, DS4_RUNTIME_VIOLATION_OVERFLOW);
            return status(tracker);
        }
        owned_bound = sum;
    }
    if (owned_bound != tracker->owned_total_bound_bytes) {
        latch(tracker, DS4_RUNTIME_VIOLATION_INVALID_CONFIG);
        return status(tracker);
    }

    uint64_t qualification_bound = owned_bound;
    const ds4_runtime_report external[] = {
        DS4_RUNTIME_REPORT_MODEL_SOURCE_RESIDENT,
        DS4_RUNTIME_REPORT_HOST_LIBRARY_UNATTRIBUTED,
        DS4_RUNTIME_REPORT_CUDA_LIBRARY_UNATTRIBUTED,
    };
    for (size_t i = 0; i < sizeof(external) / sizeof(external[0]); i++) {
        uint64_t sum = 0;
        if (!add_u64(qualification_bound,
                     tracker->report_bounds[external[i]], &sum)) {
            latch(tracker, DS4_RUNTIME_VIOLATION_OVERFLOW);
            return status(tracker);
        }
        qualification_bound = sum;
    }
    if (qualification_bound > tracker->qualification_total_bound_bytes) {
        latch(tracker, DS4_RUNTIME_VIOLATION_QUALIFICATION_TOTAL_BOUND);
        return status(tracker);
    }
    recompute(tracker);
    return status(tracker);
}

ds4_runtime_status ds4_runtime_tracker_allocate(
        ds4_runtime_tracker *tracker,
        uint64_t allocation_id,
        uint32_t callsite_id,
        uint64_t base,
        uint64_t requested_bytes,
        uint64_t charged_bytes) {
    if (!tracker || status(tracker) != DS4_RUNTIME_STATUS_OK) {
        return DS4_RUNTIME_STATUS_UNSAFE;
    }
    const ds4_runtime_callsite *site = find_callsite(tracker, callsite_id);
    if (!site) {
        latch(tracker, DS4_RUNTIME_VIOLATION_UNKNOWN_CALLSITE);
        return status(tracker);
    }
    uint64_t requested_end = 0;
    uint64_t charged_end = 0;
    if (base == 0 ||
        !range_end(base, requested_bytes, &requested_end) ||
        !range_end(base, charged_bytes, &charged_end)) {
        latch(tracker, DS4_RUNTIME_VIOLATION_ADDRESS_OVERFLOW);
        return status(tracker);
    }
    (void)requested_end;
    (void)charged_end;
    if (charged_bytes < requested_bytes) {
        latch(tracker, DS4_RUNTIME_VIOLATION_UNDERCHARGE);
        return status(tracker);
    }
    if (site->domain == DS4_RUNTIME_DOMAIN_CUDA_MANAGED &&
        (site->category != DS4_RUNTIME_CATEGORY_OTHER_CUDA ||
         charged_bytes != requested_bytes)) {
        latch(tracker, DS4_RUNTIME_VIOLATION_RELATION);
        return status(tracker);
    }
    if (find_record(tracker, allocation_id)) {
        latch(tracker, DS4_RUNTIME_VIOLATION_DUPLICATE_ID);
        return status(tracker);
    }
    for (size_t i = 0; i < tracker->record_count; i++) {
        const ds4_runtime_allocation_record *record = &tracker->records[i];
        if (record->live &&
            record->relation == DS4_RUNTIME_RELATION_OWNED_ALLOCATION &&
            record->domain == site->domain &&
            records_overlap(record, base, charged_bytes)) {
            latch(tracker, DS4_RUNTIME_VIOLATION_OVERLAP);
            return status(tracker);
        }
    }
    ds4_runtime_allocation_record *record =
        append_record(tracker, allocation_id);
    if (!record) return status(tracker);
    record->base = base;
    record->requested_bytes = requested_bytes;
    record->charged_bytes = charged_bytes;
    record->category = site->category;
    record->domain = site->domain;
    record->callsite_id = site->id;
    record->relation = DS4_RUNTIME_RELATION_OWNED_ALLOCATION;
    recompute(tracker);
    return status(tracker);
}

static bool has_live_relation(const ds4_runtime_tracker *tracker,
                              uint64_t owner_id) {
    for (size_t i = 0; i < tracker->record_count; i++) {
        const ds4_runtime_allocation_record *record = &tracker->records[i];
        if (record->live && record->owner_id == owner_id &&
            (record->relation == DS4_RUNTIME_RELATION_REGISTRATION ||
             record->relation ==
                 DS4_RUNTIME_RELATION_MANAGED_HOST_VISIBLE)) {
            return true;
        }
    }
    return false;
}

ds4_runtime_status ds4_runtime_tracker_release(
        ds4_runtime_tracker *tracker, uint64_t allocation_id) {
    if (!tracker) return DS4_RUNTIME_STATUS_UNSAFE;
    ds4_runtime_allocation_record *record =
        find_record(tracker, allocation_id);
    if (!record || !record->live ||
        record->relation != DS4_RUNTIME_RELATION_OWNED_ALLOCATION) {
        latch(tracker, DS4_RUNTIME_VIOLATION_NOT_LIVE);
        return status(tracker);
    }
    if (has_live_relation(tracker, allocation_id)) {
        latch(tracker, DS4_RUNTIME_VIOLATION_LIVE_RELATION);
        return status(tracker);
    }
    record->live = false;
    recompute(tracker);
    return status(tracker);
}

ds4_runtime_status ds4_runtime_tracker_map_model(
        ds4_runtime_tracker *tracker, uint64_t mapping_id,
        uint64_t base, uint64_t bytes) {
    if (!tracker || status(tracker) != DS4_RUNTIME_STATUS_OK) {
        return DS4_RUNTIME_STATUS_UNSAFE;
    }
    uint64_t end = 0;
    if (base == 0 || !range_end(base, bytes, &end)) {
        latch(tracker, DS4_RUNTIME_VIOLATION_ADDRESS_OVERFLOW);
        return status(tracker);
    }
    (void)end;
    for (size_t i = 0; i < tracker->record_count; i++) {
        if (tracker->records[i].live &&
            tracker->records[i].relation ==
                DS4_RUNTIME_RELATION_MODEL_MAPPING) {
            latch(tracker, DS4_RUNTIME_VIOLATION_RELATION);
            return status(tracker);
        }
    }
    ds4_runtime_allocation_record *record = append_record(tracker, mapping_id);
    if (!record) return status(tracker);
    record->base = base;
    record->requested_bytes = bytes;
    record->domain = DS4_RUNTIME_DOMAIN_HOST;
    record->relation = DS4_RUNTIME_RELATION_MODEL_MAPPING;
    recompute(tracker);
    return status(tracker);
}

ds4_runtime_status ds4_runtime_tracker_unmap_model(
        ds4_runtime_tracker *tracker, uint64_t mapping_id) {
    if (!tracker) return DS4_RUNTIME_STATUS_UNSAFE;
    ds4_runtime_allocation_record *record = find_record(tracker, mapping_id);
    if (!record || !record->live ||
        record->relation != DS4_RUNTIME_RELATION_MODEL_MAPPING) {
        latch(tracker, DS4_RUNTIME_VIOLATION_NOT_LIVE);
        return status(tracker);
    }
    if (has_live_relation(tracker, mapping_id)) {
        latch(tracker, DS4_RUNTIME_VIOLATION_LIVE_RELATION);
        return status(tracker);
    }
    record->live = false;
    recompute(tracker);
    return status(tracker);
}

static bool registration_owner_is_exact(
        const ds4_runtime_tracker *tracker,
        uint64_t owner_id, uint64_t base, uint64_t bytes) {
    size_t eligible_count = 0;
    bool selected_owner_eligible = false;
    for (size_t i = 0; i < tracker->record_count; i++) {
        const ds4_runtime_allocation_record *candidate = &tracker->records[i];
        if (!candidate->live) continue;
        bool eligible = false;
        if (candidate->relation == DS4_RUNTIME_RELATION_OWNED_ALLOCATION &&
            candidate->domain == DS4_RUNTIME_DOMAIN_HOST) {
            eligible = record_contains(candidate, base, bytes);
        } else if (candidate->relation ==
                       DS4_RUNTIME_RELATION_MODEL_MAPPING) {
            eligible = candidate->base == base &&
                       candidate->requested_bytes == bytes;
        }
        if (eligible) {
            eligible_count++;
            if (candidate->id == owner_id) selected_owner_eligible = true;
        }
    }
    return eligible_count == 1 && selected_owner_eligible;
}

ds4_runtime_status ds4_runtime_tracker_register(
        ds4_runtime_tracker *tracker, uint64_t registration_id,
        uint64_t base, uint64_t bytes, uint64_t owner_id) {
    if (!tracker || status(tracker) != DS4_RUNTIME_STATUS_OK) {
        return DS4_RUNTIME_STATUS_UNSAFE;
    }
    uint64_t end = 0;
    if (base == 0 || !range_end(base, bytes, &end)) {
        latch(tracker, DS4_RUNTIME_VIOLATION_ADDRESS_OVERFLOW);
        return status(tracker);
    }
    (void)end;
    if (!registration_owner_is_exact(tracker, owner_id, base, bytes)) {
        latch(tracker, DS4_RUNTIME_VIOLATION_RELATION);
        return status(tracker);
    }
    for (size_t i = 0; i < tracker->record_count; i++) {
        const ds4_runtime_allocation_record *record = &tracker->records[i];
        if (record->live &&
            record->relation == DS4_RUNTIME_RELATION_REGISTRATION &&
            relation_records_overlap(record, base, bytes)) {
            latch(tracker, DS4_RUNTIME_VIOLATION_OVERLAP);
            return status(tracker);
        }
    }
    ds4_runtime_allocation_record *record =
        append_record(tracker, registration_id);
    if (!record) return status(tracker);
    record->base = base;
    record->requested_bytes = bytes;
    record->domain = DS4_RUNTIME_DOMAIN_HOST;
    record->relation = DS4_RUNTIME_RELATION_REGISTRATION;
    record->owner_id = owner_id;
    recompute(tracker);
    return status(tracker);
}

ds4_runtime_status ds4_runtime_tracker_managed_host_relation(
        ds4_runtime_tracker *tracker, uint64_t relation_id,
        uint64_t base, uint64_t bytes, uint64_t owner_id) {
    if (!tracker || status(tracker) != DS4_RUNTIME_STATUS_OK) {
        return DS4_RUNTIME_STATUS_UNSAFE;
    }
    ds4_runtime_allocation_record *owner = find_record(tracker, owner_id);
    if (!owner || !owner->live ||
        owner->relation != DS4_RUNTIME_RELATION_OWNED_ALLOCATION ||
        owner->domain != DS4_RUNTIME_DOMAIN_CUDA_MANAGED ||
        owner->category != DS4_RUNTIME_CATEGORY_OTHER_CUDA ||
        owner->base != base || owner->requested_bytes != bytes ||
        owner->charged_bytes != bytes) {
        latch(tracker, DS4_RUNTIME_VIOLATION_RELATION);
        return status(tracker);
    }
    if (has_live_relation(tracker, owner_id)) {
        latch(tracker, DS4_RUNTIME_VIOLATION_RELATION);
        return status(tracker);
    }
    ds4_runtime_allocation_record *record = append_record(tracker, relation_id);
    if (!record) return status(tracker);
    record->base = base;
    record->requested_bytes = bytes;
    record->domain = DS4_RUNTIME_DOMAIN_HOST;
    record->relation = DS4_RUNTIME_RELATION_MANAGED_HOST_VISIBLE;
    record->owner_id = owner_id;
    recompute(tracker);
    return status(tracker);
}

ds4_runtime_status ds4_runtime_tracker_unregister(
        ds4_runtime_tracker *tracker, uint64_t relation_id) {
    if (!tracker) return DS4_RUNTIME_STATUS_UNSAFE;
    ds4_runtime_allocation_record *record = find_record(tracker, relation_id);
    if (!record || !record->live ||
        (record->relation != DS4_RUNTIME_RELATION_REGISTRATION &&
         record->relation !=
             DS4_RUNTIME_RELATION_MANAGED_HOST_VISIBLE)) {
        latch(tracker, DS4_RUNTIME_VIOLATION_NOT_LIVE);
        return status(tracker);
    }
    record->live = false;
    recompute(tracker);
    return status(tracker);
}

ds4_runtime_status ds4_runtime_tracker_checkpoint_external(
        ds4_runtime_tracker *tracker,
        uint64_t model_source_resident_bytes,
        uint64_t host_library_unattributed_bytes,
        uint64_t cuda_library_unattributed_bytes) {
    if (!tracker || status(tracker) != DS4_RUNTIME_STATUS_OK) {
        return DS4_RUNTIME_STATUS_UNSAFE;
    }
    tracker->report_current[DS4_RUNTIME_REPORT_MODEL_SOURCE_RESIDENT] =
        model_source_resident_bytes;
    tracker->report_current[DS4_RUNTIME_REPORT_HOST_LIBRARY_UNATTRIBUTED] =
        host_library_unattributed_bytes;
    tracker->report_current[DS4_RUNTIME_REPORT_CUDA_LIBRARY_UNATTRIBUTED] =
        cuda_library_unattributed_bytes;
    recompute(tracker);
    return status(tracker);
}

bool ds4_runtime_tracker_snapshot_copy(
        const ds4_runtime_tracker *tracker,
        ds4_runtime_snapshot *snapshot,
        ds4_runtime_allocation_record *active_records,
        size_t active_record_capacity) {
    if (!tracker || !snapshot ||
        (active_record_capacity != 0 && !active_records)) {
        return false;
    }
    memset(snapshot, 0, sizeof(*snapshot));
    memcpy(snapshot->category_bounds, tracker->category_bounds,
           sizeof(snapshot->category_bounds));
    memcpy(snapshot->category_current, tracker->category_current,
           sizeof(snapshot->category_current));
    memcpy(snapshot->category_peak, tracker->category_peak,
           sizeof(snapshot->category_peak));
    memcpy(snapshot->report_bounds, tracker->report_bounds,
           sizeof(snapshot->report_bounds));
    memcpy(snapshot->report_current, tracker->report_current,
           sizeof(snapshot->report_current));
    memcpy(snapshot->report_peak, tracker->report_peak,
           sizeof(snapshot->report_peak));
    snapshot->owned_total_bound_bytes = tracker->owned_total_bound_bytes;
    snapshot->owned_total_current = tracker->owned_total_current;
    snapshot->owned_total_peak = tracker->owned_total_peak;
    snapshot->qualification_total_bound_bytes =
        tracker->qualification_total_bound_bytes;
    snapshot->qualification_total_current =
        tracker->qualification_total_current;
    snapshot->qualification_total_peak = tracker->qualification_total_peak;
    snapshot->event_sequence = tracker->event_sequence;
    snapshot->violation = tracker->violation;
    for (size_t i = 0; i < tracker->record_count; i++) {
        if (tracker->records[i].live) snapshot->active_record_count++;
    }
    if (snapshot->active_record_count > active_record_capacity) return false;
    size_t copied = 0;
    for (size_t i = 0; i < tracker->record_count; i++) {
        if (tracker->records[i].live) {
            active_records[copied++] = tracker->records[i];
        }
    }
    return true;
}

bool ds4_runtime_reduction_qualified(
        uint64_t resident_qualification_total_peak,
        uint64_t streamed_qualification_total_peak,
        uint64_t *reduction_bytes_out) {
    if (reduction_bytes_out) *reduction_bytes_out = 0;
    if (resident_qualification_total_peak == 0 ||
        streamed_qualification_total_peak >
            resident_qualification_total_peak) {
        return false;
    }
    const uint64_t reduction = resident_qualification_total_peak -
                               streamed_qualification_total_peak;
    if (reduction_bytes_out) *reduction_bytes_out = reduction;
    const uint64_t gib = UINT64_C(1024) * 1024u * 1024u;
    if (reduction < 32u * gib) return false;

    const uint64_t quotient = resident_qualification_total_peak / 20u;
    const uint64_t remainder = resident_qualification_total_peak % 20u;
    const uint64_t ratio_floor = quotient * 9u +
        (remainder * 9u + 19u) / 20u;
    return reduction >= ratio_floor;
}
