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

bool ds4_runtime_checked_affine_bytes(
        uint64_t rows,
        uint64_t per_row_bytes,
        uint64_t fixed_bytes,
        uint64_t *bytes_out) {
    if (!bytes_out || (rows != 0 && per_row_bytes > UINT64_MAX / rows)) {
        return false;
    }
    const uint64_t row_bytes = rows * per_row_bytes;
    if (fixed_bytes > UINT64_MAX - row_bytes) return false;
    *bytes_out = row_bytes + fixed_bytes;
    return true;
}

static bool range_end(uint64_t base, uint64_t bytes, uint64_t *end) {
    return bytes != 0 && add_u64(base, bytes, end);
}

static bool record_contains(
    const ds4_runtime_allocation_record *owner,
    uint64_t base,
    uint64_t bytes);

typedef struct {
    uint64_t start;
    uint64_t end;
    uint32_t device_major;
    uint32_t device_minor;
    uint64_t inode;
    uint64_t file_offset;
    uint64_t pss_bytes;
    bool pss_seen;
} ds4_runtime_smaps_vma;

static bool external_bounded_string(
        const char *text, size_t capacity, size_t *length_out) {
    if (!text || capacity == 0) return false;
    for (size_t i = 0; i < capacity; i++) {
        if (text[i] == '\0') {
            if (length_out) *length_out = i;
            return i != 0;
        }
    }
    return false;
}

static bool external_uuid_equal(const char *a, const char *b) {
    size_t a_length = 0;
    size_t b_length = 0;
    return external_bounded_string(
               a, DS4_RUNTIME_DEVICE_UUID_CAPACITY, &a_length) &&
           external_bounded_string(
               b, DS4_RUNTIME_DEVICE_UUID_CAPACITY, &b_length) &&
           a_length == b_length && memcmp(a, b, a_length) == 0;
}

static bool external_version_equal(const char *a, const char *b) {
    size_t a_length = 0;
    size_t b_length = 0;
    return external_bounded_string(
               a, DS4_RUNTIME_NVML_LIBRARY_VERSION_CAPACITY, &a_length) &&
           external_bounded_string(
               b, DS4_RUNTIME_NVML_LIBRARY_VERSION_CAPACITY, &b_length) &&
           a_length == b_length && memcmp(a, b, a_length) == 0;
}

static int external_digit_value(char byte, unsigned radix) {
    int value = -1;
    if (byte >= '0' && byte <= '9') {
        value = byte - '0';
    } else if (byte >= 'a' && byte <= 'f') {
        value = byte - 'a' + 10;
    } else if (byte >= 'A' && byte <= 'F') {
        value = byte - 'A' + 10;
    }
    return value >= 0 && (unsigned)value < radix ? value : -1;
}

static bool external_parse_u64(
        const char **cursor,
        const char *end,
        unsigned radix,
        uint64_t *value_out,
        bool *overflow_out) {
    if (overflow_out) *overflow_out = false;
    if (!cursor || !*cursor || !value_out ||
        (radix != 10u && radix != 16u)) {
        return false;
    }
    const char *position = *cursor;
    const int first = position < end
        ? external_digit_value(*position, radix)
        : -1;
    if (first < 0) return false;

    uint64_t value = 0;
    while (position < end) {
        const int digit = external_digit_value(*position, radix);
        if (digit < 0) break;
        if (value > (UINT64_MAX - (uint64_t)digit) / radix) {
            if (overflow_out) *overflow_out = true;
            return false;
        }
        value = value * radix + (uint64_t)digit;
        position++;
    }
    *cursor = position;
    *value_out = value;
    return true;
}

static bool external_hspace(char byte) {
    return byte == ' ' || byte == '\t';
}

static bool external_consume_hspace(
        const char **cursor, const char *end) {
    if (!cursor || !*cursor || *cursor >= end ||
        !external_hspace(**cursor)) {
        return false;
    }
    do {
        (*cursor)++;
    } while (*cursor < end && external_hspace(**cursor));
    return true;
}

static bool external_smaps_header_candidate(
        const char *line, const char *line_end) {
    const char *position = line;
    if (position >= line_end ||
        external_digit_value(*position, 16u) < 0) {
        return false;
    }
    while (position < line_end &&
           external_digit_value(*position, 16u) >= 0) {
        position++;
    }
    return position < line_end && *position == '-';
}

static ds4_runtime_external_failure external_parse_smaps_header(
        const char *line,
        const char *line_end,
        ds4_runtime_smaps_vma *vma) {
    if (!line || !line_end || !vma || line >= line_end) {
        return DS4_RUNTIME_EXTERNAL_FAILURE_SMAPS_PARSE;
    }
    const char *position = line;
    bool overflow = false;
    uint64_t device_major = 0;
    uint64_t device_minor = 0;
    memset(vma, 0, sizeof(*vma));

    if (!external_parse_u64(
            &position, line_end, 16u, &vma->start, &overflow)) {
        return overflow ? DS4_RUNTIME_EXTERNAL_FAILURE_SMAPS_OVERFLOW
                        : DS4_RUNTIME_EXTERNAL_FAILURE_SMAPS_PARSE;
    }
    if (position >= line_end || *position++ != '-' ||
        !external_parse_u64(
            &position, line_end, 16u, &vma->end, &overflow)) {
        return overflow ? DS4_RUNTIME_EXTERNAL_FAILURE_SMAPS_OVERFLOW
                        : DS4_RUNTIME_EXTERNAL_FAILURE_SMAPS_PARSE;
    }
    if (vma->end <= vma->start ||
        !external_consume_hspace(&position, line_end)) {
        return DS4_RUNTIME_EXTERNAL_FAILURE_SMAPS_PARSE;
    }

    const char *permissions = position;
    while (position < line_end && !external_hspace(*position)) position++;
    if (position - permissions != 4 ||
        (permissions[0] != 'r' && permissions[0] != '-') ||
        (permissions[1] != 'w' && permissions[1] != '-') ||
        (permissions[2] != 'x' && permissions[2] != '-') ||
        (permissions[3] != 'p' && permissions[3] != 's') ||
        !external_consume_hspace(&position, line_end) ||
        !external_parse_u64(
            &position, line_end, 16u, &vma->file_offset, &overflow)) {
        return overflow ? DS4_RUNTIME_EXTERNAL_FAILURE_SMAPS_OVERFLOW
                        : DS4_RUNTIME_EXTERNAL_FAILURE_SMAPS_PARSE;
    }
    if (!external_consume_hspace(&position, line_end) ||
        !external_parse_u64(
            &position, line_end, 16u, &device_major, &overflow)) {
        return overflow ? DS4_RUNTIME_EXTERNAL_FAILURE_SMAPS_OVERFLOW
                        : DS4_RUNTIME_EXTERNAL_FAILURE_SMAPS_PARSE;
    }
    if (position >= line_end || *position++ != ':' ||
        !external_parse_u64(
            &position, line_end, 16u, &device_minor, &overflow)) {
        return overflow ? DS4_RUNTIME_EXTERNAL_FAILURE_SMAPS_OVERFLOW
                        : DS4_RUNTIME_EXTERNAL_FAILURE_SMAPS_PARSE;
    }
    if (device_major > UINT32_MAX || device_minor > UINT32_MAX) {
        return DS4_RUNTIME_EXTERNAL_FAILURE_SMAPS_OVERFLOW;
    }
    if (!external_consume_hspace(&position, line_end) ||
        !external_parse_u64(
            &position, line_end, 10u, &vma->inode, &overflow)) {
        return overflow ? DS4_RUNTIME_EXTERNAL_FAILURE_SMAPS_OVERFLOW
                        : DS4_RUNTIME_EXTERNAL_FAILURE_SMAPS_PARSE;
    }
    if (position < line_end && !external_hspace(*position)) {
        return DS4_RUNTIME_EXTERNAL_FAILURE_SMAPS_PARSE;
    }
    vma->device_major = (uint32_t)device_major;
    vma->device_minor = (uint32_t)device_minor;
    return DS4_RUNTIME_EXTERNAL_FAILURE_NONE;
}

static bool external_smaps_pss_line(
        const char *line, const char *line_end, const char **value_start) {
    while (line < line_end && external_hspace(*line)) line++;
    if ((size_t)(line_end - line) < 4u ||
        memcmp(line, "Pss:", 4u) != 0) {
        return false;
    }
    if (value_start) *value_start = line + 4u;
    return true;
}

static ds4_runtime_external_failure external_parse_smaps_pss(
        const char *value,
        const char *line_end,
        uint64_t *bytes_out) {
    bool overflow = false;
    uint64_t kibibytes = 0;
    if (!external_consume_hspace(&value, line_end) ||
        !external_parse_u64(
            &value, line_end, 10u, &kibibytes, &overflow)) {
        return overflow ? DS4_RUNTIME_EXTERNAL_FAILURE_SMAPS_OVERFLOW
                        : DS4_RUNTIME_EXTERNAL_FAILURE_SMAPS_PARSE;
    }
    if (!external_consume_hspace(&value, line_end) ||
        (size_t)(line_end - value) < 2u ||
        value[0] != 'k' || value[1] != 'B') {
        return DS4_RUNTIME_EXTERNAL_FAILURE_SMAPS_PARSE;
    }
    value += 2;
    while (value < line_end && external_hspace(*value)) value++;
    if (value != line_end) {
        return DS4_RUNTIME_EXTERNAL_FAILURE_SMAPS_PARSE;
    }
    if (kibibytes > UINT64_MAX / UINT64_C(1024)) {
        return DS4_RUNTIME_EXTERNAL_FAILURE_SMAPS_OVERFLOW;
    }
    *bytes_out = kibibytes * UINT64_C(1024);
    return DS4_RUNTIME_EXTERNAL_FAILURE_NONE;
}

static const ds4_runtime_allocation_record *external_find_record(
        const ds4_runtime_external_checkpoint_input *input, uint64_t id) {
    if (!input) return NULL;
    for (size_t i = 0; i < input->attribution_record_count; i++) {
        const ds4_runtime_allocation_record *record =
            &input->attribution_records[i];
        if (record->live && record->id == id) return record;
    }
    return NULL;
}

static bool external_managed_owner_has_relation(
        const ds4_runtime_external_checkpoint_input *input,
        uint64_t owner_id) {
    for (size_t i = 0; i < input->attribution_record_count; i++) {
        const ds4_runtime_allocation_record *record =
            &input->attribution_records[i];
        if (record->live &&
            record->relation == DS4_RUNTIME_RELATION_MANAGED_HOST_VISIBLE &&
            record->owner_id == owner_id) {
            return true;
        }
    }
    return false;
}

static bool external_host_candidate(
        const ds4_runtime_external_checkpoint_input *input,
        size_t record_index,
        uint64_t *start_out,
        uint64_t *end_out) {
    const ds4_runtime_allocation_record *record =
        &input->attribution_records[record_index];
    bool candidate = false;
    uint64_t bytes = 0;
    if (!record->live) return false;
    if (record->relation == DS4_RUNTIME_RELATION_OWNED_ALLOCATION) {
        if (record->domain == DS4_RUNTIME_DOMAIN_HOST) {
            candidate = true;
            bytes = record->charged_bytes;
        } else if (record->domain == DS4_RUNTIME_DOMAIN_CUDA_MANAGED &&
                   !external_managed_owner_has_relation(input, record->id)) {
            candidate = true;
            bytes = record->requested_bytes;
        }
    } else if (record->relation ==
                   DS4_RUNTIME_RELATION_MANAGED_HOST_VISIBLE) {
        candidate = true;
        bytes = record->requested_bytes;
    }
    if (!candidate) return false;
    uint64_t end = 0;
    if (!range_end(record->base, bytes, &end)) return false;
    if (start_out) *start_out = record->base;
    if (end_out) *end_out = end;
    return true;
}

static bool external_record_equal(
        const ds4_runtime_allocation_record *a,
        const ds4_runtime_allocation_record *b) {
    return a && b &&
        a->id == b->id && a->base == b->base &&
        a->requested_bytes == b->requested_bytes &&
        a->charged_bytes == b->charged_bytes &&
        a->category == b->category && a->domain == b->domain &&
        a->callsite_id == b->callsite_id &&
        a->relation == b->relation && a->owner_id == b->owner_id &&
        a->live == b->live;
}

static ds4_runtime_external_failure external_validate_attribution_table(
        const ds4_runtime_tracker *tracker,
        const ds4_runtime_external_checkpoint_input *input) {
    if (!tracker || !input || !input->attribution_records ||
        input->attribution_record_count == 0) {
        return DS4_RUNTIME_EXTERNAL_FAILURE_INVALID_INPUT;
    }
    for (size_t i = 0; i < input->attribution_record_count; i++) {
        const ds4_runtime_allocation_record *candidate =
            &input->attribution_records[i];
        if (!candidate->live) continue;
        size_t matches = 0;
        for (size_t j = 0; j < tracker->record_count; j++) {
            if (tracker->records[j].live &&
                external_record_equal(candidate, &tracker->records[j])) {
                matches++;
            }
        }
        if (matches != 1u) {
            return DS4_RUNTIME_EXTERNAL_FAILURE_DUPLICATE_ATTRIBUTION;
        }
    }
    for (size_t i = 0; i < tracker->record_count; i++) {
        const ds4_runtime_allocation_record *record = &tracker->records[i];
        if (!record->live) continue;
        size_t matches = 0;
        for (size_t j = 0; j < input->attribution_record_count; j++) {
            if (input->attribution_records[j].live &&
                external_record_equal(
                    record, &input->attribution_records[j])) {
                matches++;
            }
        }
        if (matches != 1u) {
            return DS4_RUNTIME_EXTERNAL_FAILURE_DUPLICATE_ATTRIBUTION;
        }
    }
    return DS4_RUNTIME_EXTERNAL_FAILURE_NONE;
}

static ds4_runtime_external_failure external_validate_records(
        const ds4_runtime_external_checkpoint_input *input,
        uint64_t *tracked_cuda_physical_bytes_out) {
    if (!input->attribution_records || input->attribution_record_count == 0 ||
        !tracked_cuda_physical_bytes_out) {
        return DS4_RUNTIME_EXTERNAL_FAILURE_INVALID_INPUT;
    }
    uint64_t tracked_cuda = 0;
    size_t model_mapping_count = 0;
    for (size_t i = 0; i < input->attribution_record_count; i++) {
        const ds4_runtime_allocation_record *record =
            &input->attribution_records[i];
        if (!record->live) continue;
        uint64_t ignored_end = 0;
        if (record->id == 0 || record->requested_bytes == 0 ||
            !range_end(record->base, record->requested_bytes, &ignored_end) ||
            record->relation < DS4_RUNTIME_RELATION_OWNED_ALLOCATION ||
            record->relation > DS4_RUNTIME_RELATION_MANAGED_HOST_VISIBLE ||
            record->domain < DS4_RUNTIME_DOMAIN_HOST ||
            record->domain >= DS4_RUNTIME_DOMAIN_COUNT) {
            return DS4_RUNTIME_EXTERNAL_FAILURE_INVALID_INPUT;
        }
        for (size_t j = 0; j < i; j++) {
            if (input->attribution_records[j].live &&
                input->attribution_records[j].id == record->id) {
                return DS4_RUNTIME_EXTERNAL_FAILURE_DUPLICATE_ATTRIBUTION;
            }
        }
        if (record->relation == DS4_RUNTIME_RELATION_OWNED_ALLOCATION) {
            if (record->charged_bytes == 0) {
                return DS4_RUNTIME_EXTERNAL_FAILURE_INVALID_INPUT;
            }
            if (record->domain == DS4_RUNTIME_DOMAIN_HOST &&
                !range_end(record->base,
                           record->charged_bytes, &ignored_end)) {
                return DS4_RUNTIME_EXTERNAL_FAILURE_INVALID_INPUT;
            }
            if (record->domain == DS4_RUNTIME_DOMAIN_CUDA_DEVICE ||
                record->domain == DS4_RUNTIME_DOMAIN_CUDA_MANAGED) {
                uint64_t sum = 0;
                if (!add_u64(tracked_cuda, record->charged_bytes, &sum)) {
                    return DS4_RUNTIME_EXTERNAL_FAILURE_INVALID_INPUT;
                }
                tracked_cuda = sum;
            }
        } else if (record->relation ==
                       DS4_RUNTIME_RELATION_MODEL_MAPPING) {
            model_mapping_count++;
            if (model_mapping_count != 1u ||
                record->base != input->model_map_base ||
                record->requested_bytes != input->model_map_bytes ||
                record->charged_bytes != 0 ||
                record->domain != DS4_RUNTIME_DOMAIN_HOST ||
                record->category !=
                    (ds4_runtime_category)DS4_RUNTIME_OWNED_CATEGORY_COUNT ||
                record->callsite_id != 0 || record->owner_id != 0) {
                return
                    DS4_RUNTIME_EXTERNAL_FAILURE_MODEL_MAPPING_MISMATCH;
            }
        } else if (record->relation ==
                       DS4_RUNTIME_RELATION_MANAGED_HOST_VISIBLE ||
                   record->relation == DS4_RUNTIME_RELATION_REGISTRATION) {
            const ds4_runtime_allocation_record *owner =
                external_find_record(input, record->owner_id);
            if (!owner) {
                return DS4_RUNTIME_EXTERNAL_FAILURE_INVALID_INPUT;
            }
            if (record->relation ==
                    DS4_RUNTIME_RELATION_MANAGED_HOST_VISIBLE) {
                if (owner->relation !=
                        DS4_RUNTIME_RELATION_OWNED_ALLOCATION ||
                    owner->domain != DS4_RUNTIME_DOMAIN_CUDA_MANAGED ||
                    !record_contains(
                        owner, record->base, record->requested_bytes)) {
                    return DS4_RUNTIME_EXTERNAL_FAILURE_INVALID_INPUT;
                }
            } else {
                const bool host_owner =
                    owner->relation ==
                        DS4_RUNTIME_RELATION_OWNED_ALLOCATION &&
                    owner->domain == DS4_RUNTIME_DOMAIN_HOST &&
                    record_contains(
                        owner, record->base, record->requested_bytes);
                const bool model_owner =
                    owner->relation == DS4_RUNTIME_RELATION_MODEL_MAPPING &&
                    owner->base == record->base &&
                    owner->requested_bytes == record->requested_bytes;
                if (!host_owner && !model_owner) {
                    return DS4_RUNTIME_EXTERNAL_FAILURE_INVALID_INPUT;
                }
            }
        }
    }

    if (model_mapping_count != 1u) {
        return DS4_RUNTIME_EXTERNAL_FAILURE_MODEL_MAPPING_MISMATCH;
    }

    for (size_t i = 0; i < input->attribution_record_count; i++) {
        uint64_t i_start = 0;
        uint64_t i_end = 0;
        if (!external_host_candidate(input, i, &i_start, &i_end)) continue;
        for (size_t j = i + 1; j < input->attribution_record_count; j++) {
            uint64_t j_start = 0;
            uint64_t j_end = 0;
            if (!external_host_candidate(input, j, &j_start, &j_end) ||
                i_start >= j_end || j_start >= i_end) {
                continue;
            }
            if (i_start == j_start && i_end == j_end) {
                return DS4_RUNTIME_EXTERNAL_FAILURE_DUPLICATE_ATTRIBUTION;
            }
            return DS4_RUNTIME_EXTERNAL_FAILURE_TRACKED_RANGE_OVERLAP;
        }
    }
    *tracked_cuda_physical_bytes_out = tracked_cuda;
    return DS4_RUNTIME_EXTERNAL_FAILURE_NONE;
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

static ds4_runtime_external_failure external_account_smaps_vma(
        const ds4_runtime_external_checkpoint_input *input,
        const ds4_runtime_smaps_vma *vma,
        ds4_runtime_external_sample *sample) {
    if (!vma->pss_seen) {
        return DS4_RUNTIME_EXTERNAL_FAILURE_SMAPS_PARSE;
    }
    const uint64_t span = vma->end - vma->start;
    if (vma->pss_bytes > span) {
        return DS4_RUNTIME_EXTERNAL_FAILURE_SMAPS_PARSE;
    }
    if (sample->smaps_vma_count == UINT64_MAX) {
        return DS4_RUNTIME_EXTERNAL_FAILURE_SMAPS_OVERFLOW;
    }
    sample->smaps_vma_count++;
    uint64_t sum = 0;
    if (!add_u64(
            sample->smaps_total_pss_bytes, vma->pss_bytes, &sum)) {
        return DS4_RUNTIME_EXTERNAL_FAILURE_SMAPS_OVERFLOW;
    }
    sample->smaps_total_pss_bytes = sum;

    const bool model_vma =
        vma->device_major == input->model_device_major &&
        vma->device_minor == input->model_device_minor &&
        vma->inode == input->model_inode;
    if (model_vma) {
        for (size_t i = 0; i < input->attribution_record_count; i++) {
            uint64_t range_start = 0;
            uint64_t range_limit = 0;
            if (external_host_candidate(
                    input, i, &range_start, &range_limit) &&
                range_start < vma->end && vma->start < range_limit) {
                return DS4_RUNTIME_EXTERNAL_FAILURE_DUPLICATE_ATTRIBUTION;
            }
        }
        if (sample->smaps_model_vma_count == UINT64_MAX ||
            !add_u64(sample->smaps_model_pss_bytes,
                     vma->pss_bytes, &sum)) {
            return DS4_RUNTIME_EXTERNAL_FAILURE_SMAPS_OVERFLOW;
        }
        sample->smaps_model_vma_count++;
        sample->smaps_model_pss_bytes = sum;
        return DS4_RUNTIME_EXTERNAL_FAILURE_NONE;
    }

    uint64_t tracked_overlap = 0;
    for (size_t i = 0; i < input->attribution_record_count; i++) {
        uint64_t range_start = 0;
        uint64_t range_limit = 0;
        if (!external_host_candidate(
                input, i, &range_start, &range_limit)) {
            continue;
        }
        const uint64_t overlap_start =
            range_start > vma->start ? range_start : vma->start;
        const uint64_t overlap_end =
            range_limit < vma->end ? range_limit : vma->end;
        if (overlap_start >= overlap_end) continue;
        if (!add_u64(tracked_overlap,
                     overlap_end - overlap_start, &sum)) {
            return DS4_RUNTIME_EXTERNAL_FAILURE_SMAPS_OVERFLOW;
        }
        tracked_overlap = sum;
    }

    uint64_t tracked_pss = 0;
    if (tracked_overlap != 0) {
        if (sample->smaps_tracked_vma_count == UINT64_MAX) {
            return DS4_RUNTIME_EXTERNAL_FAILURE_SMAPS_OVERFLOW;
        }
        sample->smaps_tracked_vma_count++;
        tracked_pss = tracked_overlap < vma->pss_bytes
            ? tracked_overlap
            : vma->pss_bytes;
        if (!add_u64(sample->smaps_tracked_pss_bytes,
                     tracked_pss, &sum)) {
            return DS4_RUNTIME_EXTERNAL_FAILURE_SMAPS_OVERFLOW;
        }
        sample->smaps_tracked_pss_bytes = sum;
    }
    if (!add_u64(sample->host_library_unattributed_bytes,
                 vma->pss_bytes - tracked_pss, &sum)) {
        return DS4_RUNTIME_EXTERNAL_FAILURE_SMAPS_OVERFLOW;
    }
    sample->host_library_unattributed_bytes = sum;
    return DS4_RUNTIME_EXTERNAL_FAILURE_NONE;
}

static bool external_model_inode_vma(
        const ds4_runtime_external_checkpoint_input *input,
        const ds4_runtime_smaps_vma *vma) {
    return input && vma &&
        vma->device_major == input->model_device_major &&
        vma->device_minor == input->model_device_minor &&
        vma->inode == input->model_inode;
}

static ds4_runtime_external_failure external_validate_model_vma(
        const ds4_runtime_external_checkpoint_input *input,
        const ds4_runtime_smaps_vma *vma,
        uint64_t model_map_end,
        uint64_t *next_address,
        uint64_t *next_file_offset,
        bool *model_seen) {
    if (!external_model_inode_vma(input, vma)) {
        return DS4_RUNTIME_EXTERNAL_FAILURE_NONE;
    }
    if (!next_address || !next_file_offset || !model_seen ||
        vma->start != *next_address || vma->end > model_map_end ||
        vma->file_offset != *next_file_offset) {
        return DS4_RUNTIME_EXTERNAL_FAILURE_MODEL_MAPPING_MISMATCH;
    }
    const uint64_t span = vma->end - vma->start;
    uint64_t next_offset = 0;
    if (!add_u64(*next_file_offset, span, &next_offset)) {
        return DS4_RUNTIME_EXTERNAL_FAILURE_MODEL_MAPPING_MISMATCH;
    }
    *next_address = vma->end;
    *next_file_offset = next_offset;
    *model_seen = true;
    return DS4_RUNTIME_EXTERNAL_FAILURE_NONE;
}

static ds4_runtime_external_failure external_parse_smaps(
        const ds4_runtime_external_checkpoint_input *input,
        ds4_runtime_external_sample *sample) {
    if (!input->smaps_text || input->smaps_text_bytes == 0) {
        return DS4_RUNTIME_EXTERNAL_FAILURE_INVALID_INPUT;
    }
    const char *cursor = input->smaps_text;
    const char *const text_end = cursor + input->smaps_text_bytes;
    ds4_runtime_smaps_vma current;
    memset(&current, 0, sizeof(current));
    bool have_vma = false;
    uint64_t previous_end = 0;
    uint64_t model_map_end = 0;
    uint64_t model_file_end = 0;
    if (!range_end(input->model_map_base,
                   input->model_source_mapped_page_bytes,
                   &model_map_end) ||
        !add_u64(input->model_file_offset,
                 input->model_source_mapped_page_bytes,
                 &model_file_end)) {
        return DS4_RUNTIME_EXTERNAL_FAILURE_MODEL_MAPPING_MISMATCH;
    }
    uint64_t next_model_address = input->model_map_base;
    uint64_t next_model_file_offset = input->model_file_offset;
    bool model_seen = false;

    while (cursor < text_end) {
        const char *line_end = cursor;
        while (line_end < text_end && *line_end != '\n') line_end++;
        if (memchr(cursor, '\0', (size_t)(line_end - cursor)) != NULL) {
            return DS4_RUNTIME_EXTERNAL_FAILURE_SMAPS_PARSE;
        }
        if (external_smaps_header_candidate(cursor, line_end)) {
            if (have_vma) {
                ds4_runtime_external_failure failure =
                    external_validate_model_vma(
                        input, &current, model_map_end,
                        &next_model_address, &next_model_file_offset,
                        &model_seen);
                if (failure != DS4_RUNTIME_EXTERNAL_FAILURE_NONE) {
                    return failure;
                }
                failure =
                    external_account_smaps_vma(input, &current, sample);
                if (failure != DS4_RUNTIME_EXTERNAL_FAILURE_NONE) {
                    return failure;
                }
            }
            ds4_runtime_external_failure failure =
                external_parse_smaps_header(cursor, line_end, &current);
            if (failure != DS4_RUNTIME_EXTERNAL_FAILURE_NONE) return failure;
            if (have_vma && current.start < previous_end) {
                return DS4_RUNTIME_EXTERNAL_FAILURE_SMAPS_PARSE;
            }
            previous_end = current.end;
            have_vma = true;
        } else {
            const char *pss_value = NULL;
            if (external_smaps_pss_line(cursor, line_end, &pss_value)) {
                if (!have_vma || current.pss_seen) {
                    return DS4_RUNTIME_EXTERNAL_FAILURE_SMAPS_PARSE;
                }
                ds4_runtime_external_failure failure =
                    external_parse_smaps_pss(
                        pss_value, line_end, &current.pss_bytes);
                if (failure != DS4_RUNTIME_EXTERNAL_FAILURE_NONE) {
                    return failure;
                }
                current.pss_seen = true;
            } else {
                const char *nonspace = cursor;
                while (nonspace < line_end &&
                       external_hspace(*nonspace)) {
                    nonspace++;
                }
                if (!have_vma && nonspace != line_end) {
                    return DS4_RUNTIME_EXTERNAL_FAILURE_SMAPS_PARSE;
                }
            }
        }
        cursor = line_end < text_end ? line_end + 1 : text_end;
    }
    if (!have_vma) {
        return DS4_RUNTIME_EXTERNAL_FAILURE_SMAPS_PARSE;
    }
    ds4_runtime_external_failure failure =
        external_validate_model_vma(
            input, &current, model_map_end,
            &next_model_address, &next_model_file_offset, &model_seen);
    if (failure != DS4_RUNTIME_EXTERNAL_FAILURE_NONE) return failure;
    failure = external_account_smaps_vma(input, &current, sample);
    if (failure != DS4_RUNTIME_EXTERNAL_FAILURE_NONE) return failure;
    if (!model_seen || sample->smaps_model_vma_count == 0 ||
        next_model_address != model_map_end ||
        next_model_file_offset != model_file_end) {
        return DS4_RUNTIME_EXTERNAL_FAILURE_MODEL_MAPPING_MISMATCH;
    }
    return DS4_RUNTIME_EXTERNAL_FAILURE_NONE;
}

static ds4_runtime_external_failure external_smaps_range_coverage(
        const ds4_runtime_external_checkpoint_input *input,
        uint64_t range_start,
        uint64_t range_limit,
        uint64_t *coverage_out) {
    const char *cursor = input->smaps_text;
    const char *const text_end = cursor + input->smaps_text_bytes;
    uint64_t coverage = 0;
    while (cursor < text_end) {
        const char *line_end = cursor;
        while (line_end < text_end && *line_end != '\n') line_end++;
        if (external_smaps_header_candidate(cursor, line_end)) {
            ds4_runtime_smaps_vma vma;
            ds4_runtime_external_failure failure =
                external_parse_smaps_header(cursor, line_end, &vma);
            if (failure != DS4_RUNTIME_EXTERNAL_FAILURE_NONE) return failure;
            const uint64_t overlap_start =
                range_start > vma.start ? range_start : vma.start;
            const uint64_t overlap_end =
                range_limit < vma.end ? range_limit : vma.end;
            if (overlap_start < overlap_end) {
                uint64_t sum = 0;
                if (!add_u64(coverage,
                             overlap_end - overlap_start, &sum)) {
                    return DS4_RUNTIME_EXTERNAL_FAILURE_SMAPS_OVERFLOW;
                }
                coverage = sum;
            }
        }
        cursor = line_end < text_end ? line_end + 1 : text_end;
    }
    *coverage_out = coverage;
    return DS4_RUNTIME_EXTERNAL_FAILURE_NONE;
}

static ds4_runtime_external_failure external_require_tracked_vmas(
        const ds4_runtime_external_checkpoint_input *input) {
    for (size_t i = 0; i < input->attribution_record_count; i++) {
        uint64_t range_start = 0;
        uint64_t range_limit = 0;
        if (!external_host_candidate(
                input, i, &range_start, &range_limit)) {
            continue;
        }
        uint64_t coverage = 0;
        ds4_runtime_external_failure failure =
            external_smaps_range_coverage(
                input, range_start, range_limit, &coverage);
        if (failure != DS4_RUNTIME_EXTERNAL_FAILURE_NONE) return failure;
        if (coverage != range_limit - range_start) {
            return DS4_RUNTIME_EXTERNAL_FAILURE_TRACKED_VMA_MISSING;
        }
    }
    return DS4_RUNTIME_EXTERNAL_FAILURE_NONE;
}

static ds4_runtime_external_failure external_validate_inventory(
        const ds4_runtime_nvml_inventory *inventory,
        uint32_t expected_api_version,
        const char *expected_library_version,
        const char *expected_device_uuid) {
    if (!inventory) return DS4_RUNTIME_EXTERNAL_FAILURE_INVALID_INPUT;
    if (inventory->api_version != expected_api_version) {
        return DS4_RUNTIME_EXTERNAL_FAILURE_NVML_API_MISMATCH;
    }
    if (!external_version_equal(
            inventory->library_version, expected_library_version)) {
        return
            DS4_RUNTIME_EXTERNAL_FAILURE_NVML_LIBRARY_VERSION_MISMATCH;
    }
    if (!external_uuid_equal(
            inventory->device_uuid, expected_device_uuid)) {
        return DS4_RUNTIME_EXTERNAL_FAILURE_DEVICE_UUID_MISMATCH;
    }
    if (inventory->process_count != 0 && !inventory->processes) {
        return DS4_RUNTIME_EXTERNAL_FAILURE_INVALID_INPUT;
    }
    for (size_t i = 0; i < inventory->process_count; i++) {
        const ds4_runtime_nvml_process_sample *process =
            &inventory->processes[i];
        if (process->pid == 0) {
            return DS4_RUNTIME_EXTERNAL_FAILURE_INVALID_INPUT;
        }
        if (!process->used_bytes_known) {
            return DS4_RUNTIME_EXTERNAL_FAILURE_NVML_USAGE_UNKNOWN;
        }
        for (size_t j = 0; j < i; j++) {
            if (inventory->processes[j].pid == process->pid) {
                return DS4_RUNTIME_EXTERNAL_FAILURE_NVML_DUPLICATE_PID;
            }
        }
    }
    return DS4_RUNTIME_EXTERNAL_FAILURE_NONE;
}

static const ds4_runtime_nvml_process_sample *external_find_process(
        const ds4_runtime_nvml_inventory *inventory, uint32_t pid) {
    if (!inventory) return NULL;
    for (size_t i = 0; i < inventory->process_count; i++) {
        if (inventory->processes[i].pid == pid) {
            return &inventory->processes[i];
        }
    }
    return NULL;
}

static bool external_peer_inventory_equal(
        const ds4_runtime_nvml_inventory *a,
        const ds4_runtime_nvml_inventory *b,
        uint32_t own_pid) {
    for (size_t i = 0; i < a->process_count; i++) {
        const ds4_runtime_nvml_process_sample *process = &a->processes[i];
        if (process->pid == own_pid) continue;
        const ds4_runtime_nvml_process_sample *other =
            external_find_process(b, process->pid);
        if (!other || other->pid == own_pid ||
            other->used_bytes != process->used_bytes) {
            return false;
        }
    }
    for (size_t i = 0; i < b->process_count; i++) {
        const ds4_runtime_nvml_process_sample *process = &b->processes[i];
        if (process->pid == own_pid) continue;
        if (!external_find_process(a, process->pid)) return false;
    }
    return true;
}

static ds4_runtime_status external_checkpoint_fail(
        ds4_runtime_tracker *tracker,
        ds4_runtime_external_sample *sample_out,
        ds4_runtime_external_failure failure) {
    ds4_runtime_external_sample sample;
    memset(&sample, 0, sizeof(sample));
    sample.failure = failure;
    if (sample_out &&
        (!tracker || sample_out != &tracker->external_sample)) {
        *sample_out = sample;
    }
    latch(tracker, DS4_RUNTIME_VIOLATION_EXTERNAL_ATTRIBUTION);
    return DS4_RUNTIME_STATUS_UNSAFE;
}

static ds4_runtime_allocation_record *append_record(
        ds4_runtime_tracker *tracker, uint64_t id) {
    const uint8_t producer_namespace = (uint8_t)(id >> 56);
    const uint64_t sequence = id & UINT64_C(0x00ffffffffffffff);
    if (sequence == 0 ||
        sequence <=
            tracker->issued_sequence_high_water[producer_namespace]) {
        latch(tracker, DS4_RUNTIME_VIOLATION_DUPLICATE_ID);
        return NULL;
    }

    ds4_runtime_allocation_record *record = NULL;
    for (size_t i = 0; i < tracker->record_count; i++) {
        if (!tracker->records[i].live) {
            record = &tracker->records[i];
            break;
        }
    }
    if (!record) {
        if (tracker->record_count >= tracker->record_capacity) {
            latch(tracker, DS4_RUNTIME_VIOLATION_CAPACITY);
            return NULL;
        }
        record = &tracker->records[tracker->record_count++];
    }

    tracker->issued_sequence_high_water[producer_namespace] = sequence;
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

ds4_runtime_status ds4_runtime_tracker_allocate_next(
        ds4_runtime_tracker *tracker,
        uint8_t producer_namespace,
        uint32_t callsite_id,
        uint64_t base,
        uint64_t requested_bytes,
        uint64_t charged_bytes,
        uint64_t *allocation_id_out) {
    if (!tracker) return DS4_RUNTIME_STATUS_UNSAFE;
    if (!allocation_id_out) {
        latch(tracker, DS4_RUNTIME_VIOLATION_INVALID_CONFIG);
        return status(tracker);
    }
    if (status(tracker) != DS4_RUNTIME_STATUS_OK) {
        return DS4_RUNTIME_STATUS_UNSAFE;
    }

    const uint64_t sequence_mask = UINT64_C(0x00ffffffffffffff);
    const uint64_t previous_sequence =
        tracker->issued_sequence_high_water[producer_namespace];
    if (previous_sequence >= sequence_mask) {
        latch(tracker, DS4_RUNTIME_VIOLATION_OVERFLOW);
        return status(tracker);
    }
    const uint64_t allocation_id =
        ((uint64_t)producer_namespace << 56) | (previous_sequence + 1u);
    const bool allocation_id_already_known =
        find_record(tracker, allocation_id) != NULL;
    const ds4_runtime_status result = ds4_runtime_tracker_allocate(
        tracker, allocation_id, callsite_id, base,
        requested_bytes, charged_bytes);
    const ds4_runtime_allocation_record *record =
        find_record(tracker, allocation_id);
    if (result == DS4_RUNTIME_STATUS_OK ||
        (!allocation_id_already_known && record && record->live &&
         record->relation == DS4_RUNTIME_RELATION_OWNED_ALLOCATION)) {
        *allocation_id_out = allocation_id;
    }
    return result;
}

ds4_runtime_status ds4_runtime_tracker_replay_owned(
        ds4_runtime_tracker *tracker,
        const ds4_runtime_owned_descriptor *owners,
        size_t owner_count,
        bool *owner_live) {
    if (owner_count != 0 && !owner_live) return DS4_RUNTIME_STATUS_UNSAFE;
    for (size_t i = 0; i < owner_count; i++) owner_live[i] = false;
    if (!tracker || (owner_count != 0 && !owners)) {
        return DS4_RUNTIME_STATUS_UNSAFE;
    }

    for (size_t i = 0; i < owner_count; i++) {
        const bool allocation_id_already_known =
            find_record(tracker, owners[i].allocation_id) != NULL;
        const ds4_runtime_status result = ds4_runtime_tracker_allocate(
            tracker,
            owners[i].allocation_id,
            owners[i].callsite_id,
            owners[i].base,
            owners[i].requested_bytes,
            owners[i].charged_bytes);
        const ds4_runtime_allocation_record *record =
            find_record(tracker, owners[i].allocation_id);
        if (!allocation_id_already_known && record && record->live &&
            record->relation == DS4_RUNTIME_RELATION_OWNED_ALLOCATION) {
            owner_live[i] = true;
        }
        if (result != DS4_RUNTIME_STATUS_OK) return result;
    }
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
    if (tracker->external_sample.attributed_generation == UINT64_MAX) {
        latch(tracker, DS4_RUNTIME_VIOLATION_OVERFLOW);
        return DS4_RUNTIME_STATUS_UNSAFE;
    }
    const uint64_t next_attribution_generation =
        tracker->external_sample.attributed_generation + 1u;
    memset(&tracker->external_sample, 0,
           sizeof(tracker->external_sample));
    tracker->external_sample.attributed_generation =
        next_attribution_generation;
    tracker->report_current[DS4_RUNTIME_REPORT_MODEL_SOURCE_RESIDENT] =
        model_source_resident_bytes;
    tracker->report_current[DS4_RUNTIME_REPORT_HOST_LIBRARY_UNATTRIBUTED] =
        host_library_unattributed_bytes;
    tracker->report_current[DS4_RUNTIME_REPORT_CUDA_LIBRARY_UNATTRIBUTED] =
        cuda_library_unattributed_bytes;
    recompute(tracker);
    return status(tracker);
}

ds4_runtime_status ds4_runtime_tracker_checkpoint_attributed(
        ds4_runtime_tracker *tracker,
        const ds4_runtime_external_checkpoint_input *input,
        ds4_runtime_external_sample *sample_out) {
    if (!tracker || !input || !sample_out ||
        status(tracker) != DS4_RUNTIME_STATUS_OK) {
        return external_checkpoint_fail(
            tracker, sample_out,
            DS4_RUNTIME_EXTERNAL_FAILURE_INVALID_INPUT);
    }

    ds4_runtime_external_sample sample;
    memset(&sample, 0, sizeof(sample));
    sample.smaps_model_device_major = input->model_device_major;
    sample.smaps_model_device_minor = input->model_device_minor;
    sample.smaps_model_inode = input->model_inode;

    size_t uuid_length = 0;
    size_t library_version_length = 0;
    if (input->expected_nvml_api_version != 2u ||
        input->baseline_nvml_api_version !=
            input->expected_nvml_api_version) {
        return external_checkpoint_fail(
            tracker, sample_out,
            DS4_RUNTIME_EXTERNAL_FAILURE_NVML_API_MISMATCH);
    }
    if (!external_bounded_string(
            input->expected_nvml_library_version,
            DS4_RUNTIME_NVML_LIBRARY_VERSION_CAPACITY,
            &library_version_length) ||
        !external_version_equal(
            input->expected_nvml_library_version,
            input->baseline_nvml_library_version)) {
        return external_checkpoint_fail(
            tracker, sample_out,
            DS4_RUNTIME_EXTERNAL_FAILURE_NVML_LIBRARY_VERSION_MISMATCH);
    }
    if (!external_bounded_string(
            input->expected_device_uuid,
            DS4_RUNTIME_DEVICE_UUID_CAPACITY,
            &uuid_length) ||
        !external_bounded_string(
            input->baseline_device_uuid,
            DS4_RUNTIME_DEVICE_UUID_CAPACITY,
            NULL) ||
        !external_uuid_equal(
            input->expected_device_uuid, input->baseline_device_uuid)) {
        return external_checkpoint_fail(
            tracker, sample_out,
            DS4_RUNTIME_EXTERNAL_FAILURE_DEVICE_UUID_MISMATCH);
    }
    if (input->own_pid == 0 ||
        input->baseline_process_id != input->own_pid) {
        return external_checkpoint_fail(
            tracker, sample_out,
            DS4_RUNTIME_EXTERNAL_FAILURE_PROCESS_ID_MISMATCH);
    }
    if (!input->expected_build_identity ||
        !input->observed_build_identity ||
        input->build_identity_bytes !=
            DS4_RUNTIME_BUILD_IDENTITY_BYTES) {
        return external_checkpoint_fail(
            tracker, sample_out,
            DS4_RUNTIME_EXTERNAL_FAILURE_INVALID_INPUT);
    }
    if (memcmp(input->expected_build_identity,
               input->observed_build_identity,
               input->build_identity_bytes) != 0) {
        return external_checkpoint_fail(
            tracker, sample_out,
            DS4_RUNTIME_EXTERNAL_FAILURE_BUILD_IDENTITY_MISMATCH);
    }
    if (!input->baseline_nvml_process_present &&
        (input->baseline_nvml_process_bytes_known ||
         input->baseline_nvml_process_bytes != 0 ||
         input->baseline_tracked_cuda_physical_bytes != 0)) {
        return external_checkpoint_fail(
            tracker, sample_out,
            DS4_RUNTIME_EXTERNAL_FAILURE_INVALID_INPUT);
    }
    if (input->baseline_nvml_process_present &&
        !input->baseline_nvml_process_bytes_known) {
        return external_checkpoint_fail(
            tracker, sample_out,
            DS4_RUNTIME_EXTERNAL_FAILURE_NVML_USAGE_UNKNOWN);
    }
    uint64_t rounded_model_map_bytes = 0;
    if (input->model_source_page_size == 0 ||
        (input->model_source_page_size &
         (input->model_source_page_size - 1u)) != 0 ||
        input->model_map_base == 0 || input->model_map_bytes == 0 ||
        input->model_map_base % input->model_source_page_size != 0 ||
        input->model_file_offset % input->model_source_page_size != 0 ||
        input->model_map_bytes >
            UINT64_MAX - (input->model_source_page_size - 1u)) {
        return external_checkpoint_fail(
            tracker, sample_out,
            DS4_RUNTIME_EXTERNAL_FAILURE_INVALID_INPUT);
    }
    rounded_model_map_bytes =
        (input->model_map_bytes + input->model_source_page_size - 1u) &
        ~(input->model_source_page_size - 1u);
    uint64_t model_map_end = 0;
    uint64_t model_file_end = 0;
    if (rounded_model_map_bytes !=
            input->model_source_mapped_page_bytes ||
        !range_end(input->model_map_base,
                   rounded_model_map_bytes, &model_map_end) ||
        !add_u64(input->model_file_offset,
                 rounded_model_map_bytes, &model_file_end)) {
        return external_checkpoint_fail(
            tracker, sample_out,
            DS4_RUNTIME_EXTERNAL_FAILURE_INVALID_INPUT);
    }
    (void)model_map_end;
    (void)model_file_end;
    if (!input->cuda_mem_info_known || input->cuda_mem_total_bytes == 0 ||
        input->cuda_mem_free_bytes > input->cuda_mem_total_bytes ||
        input->model_inode == 0 ||
        input->model_source_mapped_page_bytes == 0 ||
        input->model_source_mapped_page_bytes %
            input->model_source_page_size != 0 ||
        input->model_source_resident_bytes %
            input->model_source_page_size != 0 ||
        input->model_source_resident_bytes >
            input->model_source_mapped_page_bytes ||
        input->model_source_resident_bytes >
            tracker->report_bounds[
                DS4_RUNTIME_REPORT_MODEL_SOURCE_RESIDENT] ||
        tracker->event_sequence == UINT64_MAX ||
        tracker->external_sample.attributed_generation == UINT64_MAX) {
        return external_checkpoint_fail(
            tracker, sample_out,
            DS4_RUNTIME_EXTERNAL_FAILURE_INVALID_INPUT);
    }

    const ds4_runtime_nvml_inventory *inventories[] = {
        input->pre_child_inventory,
        input->checkpoint_before_inventory,
        input->inside_ds4_inventory,
        input->checkpoint_after_inventory,
    };
    for (size_t i = 0; i < sizeof(inventories) / sizeof(inventories[0]); i++) {
        const ds4_runtime_external_failure failure =
            external_validate_inventory(
                inventories[i], input->expected_nvml_api_version,
                input->expected_nvml_library_version,
                input->expected_device_uuid);
        if (failure != DS4_RUNTIME_EXTERNAL_FAILURE_NONE) {
            return external_checkpoint_fail(tracker, sample_out, failure);
        }
    }

    const ds4_runtime_nvml_process_sample *inside_process =
        external_find_process(input->inside_ds4_inventory, input->own_pid);
    if (!inside_process) {
        return external_checkpoint_fail(
            tracker, sample_out,
            DS4_RUNTIME_EXTERNAL_FAILURE_NVML_PROCESS_MISSING);
    }
    bool peer_inventory_stable =
        external_find_process(
            input->pre_child_inventory, input->own_pid) == NULL;
    for (size_t i = 0;
         peer_inventory_stable &&
             i < sizeof(inventories) / sizeof(inventories[0]);
         i++) {
        for (size_t j = i + 1u;
             peer_inventory_stable &&
                 j < sizeof(inventories) / sizeof(inventories[0]);
             j++) {
            peer_inventory_stable = external_peer_inventory_equal(
                inventories[i], inventories[j], input->own_pid);
        }
    }
    if (!peer_inventory_stable) {
        return external_checkpoint_fail(
            tracker, sample_out,
            DS4_RUNTIME_EXTERNAL_FAILURE_UNRELATED_PROCESS_CHANGED);
    }

    uint64_t tracked_cuda_physical_bytes = 0;
    ds4_runtime_external_failure failure =
        external_validate_attribution_table(tracker, input);
    if (failure != DS4_RUNTIME_EXTERNAL_FAILURE_NONE) {
        return external_checkpoint_fail(tracker, sample_out, failure);
    }
    failure =
        external_validate_records(input, &tracked_cuda_physical_bytes);
    if (failure != DS4_RUNTIME_EXTERNAL_FAILURE_NONE) {
        return external_checkpoint_fail(tracker, sample_out, failure);
    }

    failure = external_parse_smaps(input, &sample);
    if (failure != DS4_RUNTIME_EXTERNAL_FAILURE_NONE) {
        return external_checkpoint_fail(tracker, sample_out, failure);
    }
    if (sample.smaps_model_pss_bytes >
        input->model_source_resident_bytes) {
        return external_checkpoint_fail(
            tracker, sample_out,
            DS4_RUNTIME_EXTERNAL_FAILURE_MODEL_MAPPING_MISMATCH);
    }
    failure = external_require_tracked_vmas(input);
    if (failure != DS4_RUNTIME_EXTERNAL_FAILURE_NONE) {
        return external_checkpoint_fail(tracker, sample_out, failure);
    }

    const uint64_t unattributed_limit = UINT64_C(512) * 1024u * 1024u;
    if (sample.host_library_unattributed_bytes > unattributed_limit ||
        sample.host_library_unattributed_bytes >
            tracker->report_bounds[
                DS4_RUNTIME_REPORT_HOST_LIBRARY_UNATTRIBUTED]) {
        return external_checkpoint_fail(
            tracker, sample_out,
            DS4_RUNTIME_EXTERNAL_FAILURE_HOST_UNATTRIBUTED_BOUND);
    }

    const uint64_t nvml_process_bytes = inside_process->used_bytes;
    if (input->baseline_nvml_process_bytes <
            input->baseline_tracked_cuda_physical_bytes ||
        nvml_process_bytes < tracked_cuda_physical_bytes) {
        return external_checkpoint_fail(
            tracker, sample_out,
            DS4_RUNTIME_EXTERNAL_FAILURE_CUDA_NEGATIVE_GAP);
    }
    const uint64_t baseline_unattributed =
        input->baseline_nvml_process_bytes -
        input->baseline_tracked_cuda_physical_bytes;
    const uint64_t cuda_unattributed =
        nvml_process_bytes - tracked_cuda_physical_bytes;
    if (baseline_unattributed > unattributed_limit ||
        cuda_unattributed > unattributed_limit ||
        cuda_unattributed >
            tracker->report_bounds[
                DS4_RUNTIME_REPORT_CUDA_LIBRARY_UNATTRIBUTED]) {
        return external_checkpoint_fail(
            tracker, sample_out,
            DS4_RUNTIME_EXTERNAL_FAILURE_CUDA_UNATTRIBUTED_BOUND);
    }

    sample.failure = DS4_RUNTIME_EXTERNAL_FAILURE_NONE;
    sample.attributed_valid = true;
    sample.attributed_generation =
        tracker->external_sample.attributed_generation + 1u;
    sample.checkpoint_sequence = tracker->event_sequence + 1u;
    sample.nvml_api_version = input->expected_nvml_api_version;
    memcpy(sample.nvml_library_version,
           input->expected_nvml_library_version,
           library_version_length + 1u);
    memcpy(sample.device_uuid, input->expected_device_uuid, uuid_length + 1u);
    sample.process_id = input->own_pid;
    sample.nvml_process_baseline_present =
        input->baseline_nvml_process_present;
    sample.nvml_process_baseline_bytes =
        input->baseline_nvml_process_bytes;
    sample.tracked_cuda_physical_baseline_bytes =
        input->baseline_tracked_cuda_physical_bytes;
    sample.nvml_process_bytes = nvml_process_bytes;
    sample.tracked_cuda_physical_bytes = tracked_cuda_physical_bytes;
    sample.cuda_library_unattributed_bytes = cuda_unattributed;
    sample.cuda_mem_free_bytes = input->cuda_mem_free_bytes;
    sample.cuda_mem_total_bytes = input->cuda_mem_total_bytes;
    sample.unrelated_process_inventory_stable = true;

    ds4_runtime_tracker staged = *tracker;
    staged.report_current[DS4_RUNTIME_REPORT_MODEL_SOURCE_RESIDENT] =
        input->model_source_resident_bytes;
    staged.report_current[DS4_RUNTIME_REPORT_HOST_LIBRARY_UNATTRIBUTED] =
        sample.host_library_unattributed_bytes;
    staged.report_current[DS4_RUNTIME_REPORT_CUDA_LIBRARY_UNATTRIBUTED] =
        cuda_unattributed;
    staged.external_sample = sample;
    recompute(&staged);
    if (status(&staged) != DS4_RUNTIME_STATUS_OK) {
        return external_checkpoint_fail(
            tracker, sample_out,
            DS4_RUNTIME_EXTERNAL_FAILURE_INVALID_INPUT);
    }
    *tracker = staged;
    *sample_out = sample;
    return DS4_RUNTIME_STATUS_OK;
}

ds4_runtime_status ds4_runtime_tracker_checkpoint_model_source(
        ds4_runtime_tracker *tracker,
        uint64_t model_source_resident_bytes) {
    if (!tracker || status(tracker) != DS4_RUNTIME_STATUS_OK) {
        return DS4_RUNTIME_STATUS_UNSAFE;
    }
    return ds4_runtime_tracker_checkpoint_external(
        tracker,
        model_source_resident_bytes,
        tracker->report_current[DS4_RUNTIME_REPORT_HOST_LIBRARY_UNATTRIBUTED],
        tracker->report_current[DS4_RUNTIME_REPORT_CUDA_LIBRARY_UNATTRIBUTED]);
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
    snapshot->external_sample = tracker->external_sample;
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
