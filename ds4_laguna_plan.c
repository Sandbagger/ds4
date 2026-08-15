#include "ds4_laguna_plan.h"

#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DS4_LAGUNA_MODEL_REPOSITORY "poolside/Laguna-S-2.1-GGUF"
#define DS4_LAGUNA_MODEL_REVISION "e2ccc0579fc18e6ea2362fa25fccbcd470f0e332"
#define DS4_LAGUNA_MODEL_FILENAME "laguna-s-2.1-Q4_K_M.gguf"
#define DS4_LAGUNA_MODEL_SHA256 \
    "a34c74e46688122bef83122f4133031bababbefcf57436dde97048c91e2cc6ff"
#define DS4_LAGUNA_JSON_BUFFER_LIMIT (UINT64_C(64) * 1024u * 1024u)

typedef struct {
    char *bytes;
    size_t size;
    size_t capacity;
} json_buffer;

typedef struct {
    const char *name;
    ds4_runtime_category category;
    ds4_runtime_physical_domain domain;
} callsite_contract;

static const char *const category_names[DS4_RUNTIME_OWNED_CATEGORY_COUNT] = {
    "STATIC_WEIGHTS",
    "EXPERT_CACHE_PAYLOAD",
    "CACHE_METADATA_ADDRESS_TABLES",
    "KV_STATE",
    "GRAPH_SCRATCH",
    "PINNED_STAGING",
    "OTHER_HOST",
    "OTHER_CUDA",
};

static const char *const report_names[DS4_RUNTIME_REPORT_COUNT] = {
    "MODEL_MAPPED_VIRTUAL",
    "MODEL_MAPPING_REGISTERED",
    "MODEL_SOURCE_RESIDENT",
    "HOST_LIBRARY_UNATTRIBUTED",
    "CUDA_LIBRARY_UNATTRIBUTED",
};

static const char *const domain_names[DS4_RUNTIME_DOMAIN_COUNT] = {
    "HOST",
    "CUDA_DEVICE",
    "CUDA_MANAGED",
};

static const callsite_contract callsite_contracts[
        DS4_LAGUNA_ALLOCATION_CALLSITE_COUNT] = {
    { "laguna.static_slab", DS4_RUNTIME_CATEGORY_STATIC_WEIGHTS,
      DS4_RUNTIME_DOMAIN_CUDA_DEVICE },
    { "laguna.expert_cache_payload",
      DS4_RUNTIME_CATEGORY_EXPERT_CACHE_PAYLOAD,
      DS4_RUNTIME_DOMAIN_CUDA_DEVICE },
    { "laguna.ledger_arrays",
      DS4_RUNTIME_CATEGORY_CACHE_METADATA_ADDRESS_TABLES,
      DS4_RUNTIME_DOMAIN_HOST },
    { "laguna.route_hotness",
      DS4_RUNTIME_CATEGORY_CACHE_METADATA_ADDRESS_TABLES,
      DS4_RUNTIME_DOMAIN_HOST },
    { "laguna.host_entry_to_slot",
      DS4_RUNTIME_CATEGORY_CACHE_METADATA_ADDRESS_TABLES,
      DS4_RUNTIME_DOMAIN_HOST },
    { "laguna.device_entry_to_slot",
      DS4_RUNTIME_CATEGORY_CACHE_METADATA_ADDRESS_TABLES,
      DS4_RUNTIME_DOMAIN_CUDA_DEVICE },
    { "laguna.static_offsets",
      DS4_RUNTIME_CATEGORY_CACHE_METADATA_ADDRESS_TABLES,
      DS4_RUNTIME_DOMAIN_HOST },
    { "laguna.slot_state",
      DS4_RUNTIME_CATEGORY_CACHE_METADATA_ADDRESS_TABLES,
      DS4_RUNTIME_DOMAIN_HOST },
    { "laguna.kv_state", DS4_RUNTIME_CATEGORY_KV_STATE,
      DS4_RUNTIME_DOMAIN_CUDA_DEVICE },
    { "laguna.graph_scratch", DS4_RUNTIME_CATEGORY_GRAPH_SCRATCH,
      DS4_RUNTIME_DOMAIN_CUDA_DEVICE },
    { "laguna.pinned_staging.0", DS4_RUNTIME_CATEGORY_PINNED_STAGING,
      DS4_RUNTIME_DOMAIN_HOST },
    { "laguna.pinned_staging.1", DS4_RUNTIME_CATEGORY_PINNED_STAGING,
      DS4_RUNTIME_DOMAIN_HOST },
    { "laguna.pinned_staging.2", DS4_RUNTIME_CATEGORY_PINNED_STAGING,
      DS4_RUNTIME_DOMAIN_HOST },
    { "laguna.pinned_staging.3", DS4_RUNTIME_CATEGORY_PINNED_STAGING,
      DS4_RUNTIME_DOMAIN_HOST },
    { "laguna.other_host.engine", DS4_RUNTIME_CATEGORY_OTHER_HOST,
      DS4_RUNTIME_DOMAIN_HOST },
    { "laguna.other_host.model", DS4_RUNTIME_CATEGORY_OTHER_HOST,
      DS4_RUNTIME_DOMAIN_HOST },
    { "laguna.other_host.bootstrap", DS4_RUNTIME_CATEGORY_OTHER_HOST,
      DS4_RUNTIME_DOMAIN_HOST },
    { "laguna.other_host.vocab", DS4_RUNTIME_CATEGORY_OTHER_HOST,
      DS4_RUNTIME_DOMAIN_HOST },
    { "laguna.other_host.session", DS4_RUNTIME_CATEGORY_OTHER_HOST,
      DS4_RUNTIME_DOMAIN_HOST },
    { "laguna.other_host.tracker", DS4_RUNTIME_CATEGORY_OTHER_HOST,
      DS4_RUNTIME_DOMAIN_HOST },
    { "laguna.other_host.serializer", DS4_RUNTIME_CATEGORY_OTHER_HOST,
      DS4_RUNTIME_DOMAIN_HOST },
    { "laguna.other_cuda.kernel_tmp", DS4_RUNTIME_CATEGORY_OTHER_CUDA,
      DS4_RUNTIME_DOMAIN_CUDA_DEVICE },
    { "laguna.other_cuda.routed_workspace", DS4_RUNTIME_CATEGORY_OTHER_CUDA,
      DS4_RUNTIME_DOMAIN_CUDA_DEVICE },
    { "laguna.other_cuda.descriptor_upload", DS4_RUNTIME_CATEGORY_OTHER_CUDA,
      DS4_RUNTIME_DOMAIN_CUDA_DEVICE },
    { "laguna.other_cuda.transient", DS4_RUNTIME_CATEGORY_OTHER_CUDA,
      DS4_RUNTIME_DOMAIN_CUDA_DEVICE },
};

static void clear_error(char *error, size_t error_size) {
    if (error != NULL && error_size != 0) error[0] = '\0';
}

static bool set_error(char *error, size_t error_size, const char *format, ...) {
    if (error != NULL && error_size != 0) {
        va_list arguments;
        va_start(arguments, format);
        (void)vsnprintf(error, error_size, format, arguments);
        va_end(arguments);
    }
    return false;
}

static bool add_u64(uint64_t lhs, uint64_t rhs, uint64_t *out) {
    if (lhs > UINT64_MAX - rhs) return false;
    *out = lhs + rhs;
    return true;
}

static bool mul_u64(uint64_t lhs, uint64_t rhs, uint64_t *out) {
    if (lhs != 0 && rhs > UINT64_MAX / lhs) return false;
    *out = lhs * rhs;
    return true;
}

static bool is_power_of_two(uint64_t value) {
    return value != 0 && (value & (value - 1u)) == 0;
}

static bool align_u64(uint64_t value, uint64_t alignment, uint64_t *out) {
    if (!is_power_of_two(alignment)) return false;
    const uint64_t mask = alignment - 1u;
    if (value > UINT64_MAX - mask) return false;
    *out = (value + mask) & ~mask;
    return true;
}

static bool range_end(uint64_t offset, uint64_t bytes, uint64_t *end) {
    return bytes != 0 && add_u64(offset, bytes, end);
}

static bool ranges_overlap(uint64_t a_offset,
                           uint64_t a_end,
                           uint64_t b_offset,
                           uint64_t b_end) {
    return a_offset < b_end && b_offset < a_end;
}

static bool buffer_reserve(json_buffer *buffer,
                           size_t additional,
                           char *error,
                           size_t error_size) {
    if (additional > SIZE_MAX - buffer->size - 1u) {
        return set_error(error, error_size, "qualification JSON size overflow");
    }
    const size_t needed = buffer->size + additional + 1u;
    if ((uint64_t)needed > DS4_LAGUNA_JSON_BUFFER_LIMIT) {
        return set_error(error, error_size,
                         "qualification JSON exceeds serializer envelope");
    }
    if (needed <= buffer->capacity) return true;
    size_t capacity = buffer->capacity == 0 ? 4096u : buffer->capacity;
    while (capacity < needed) {
        if (capacity > SIZE_MAX / 2u) {
            capacity = needed;
            break;
        }
        capacity *= 2u;
    }
    char *bytes = realloc(buffer->bytes, capacity);
    if (bytes == NULL) {
        return set_error(error, error_size,
                         "qualification JSON allocation failed");
    }
    buffer->bytes = bytes;
    buffer->capacity = capacity;
    return true;
}

static bool buffer_append(json_buffer *buffer,
                          const void *bytes,
                          size_t size,
                          char *error,
                          size_t error_size) {
    if (bytes == NULL && size != 0) {
        return set_error(error, error_size,
                         "qualification JSON append source is null");
    }
    if (!buffer_reserve(buffer, size, error, error_size)) return false;
    if (buffer->bytes == NULL) {
        return set_error(error, error_size,
                         "qualification JSON buffer is unavailable");
    }
    if (size != 0) memcpy(buffer->bytes + buffer->size, bytes, size);
    buffer->size += size;
    buffer->bytes[buffer->size] = '\0';
    return true;
}

static bool append_literal(json_buffer *buffer,
                           const char *literal,
                           char *error,
                           size_t error_size) {
    return buffer_append(buffer, literal, strlen(literal), error, error_size);
}

static bool append_json_string(json_buffer *buffer,
                               const char *string,
                               char *error,
                               size_t error_size) {
    static const char hex[] = "0123456789abcdef";
    if (string == NULL) {
        return set_error(error, error_size,
                         "qualification JSON string is null");
    }
    if (!append_literal(buffer, "\"", error, error_size)) return false;
    for (const unsigned char *cursor = (const unsigned char *)string;
         *cursor != '\0'; cursor++) {
        const unsigned char value = *cursor;
        if (value == '"' || value == '\\') {
            const char escaped[2] = { '\\', (char)value };
            if (!buffer_append(buffer, escaped, sizeof(escaped),
                               error, error_size)) return false;
        } else if (value < 0x20u) {
            const char escaped[6] = {
                '\\', 'u', '0', '0', hex[value >> 4u], hex[value & 0x0fu],
            };
            if (!buffer_append(buffer, escaped, sizeof(escaped),
                               error, error_size)) return false;
        } else {
            if (!buffer_append(buffer, cursor, 1u, error, error_size)) {
                return false;
            }
        }
    }
    return append_literal(buffer, "\"", error, error_size);
}

static bool append_u64_string(json_buffer *buffer,
                              uint64_t value,
                              char *error,
                              size_t error_size) {
    char decimal[32];
    const int length = snprintf(decimal, sizeof(decimal), "\"%" PRIu64 "\"",
                                value);
    if (length < 0 || (size_t)length >= sizeof(decimal)) {
        return set_error(error, error_size,
                         "qualification uint64 formatting failed");
    }
    return buffer_append(buffer, decimal, (size_t)length, error, error_size);
}

static bool append_u32(json_buffer *buffer,
                       uint32_t value,
                       char *error,
                       size_t error_size) {
    char decimal[16];
    const int length = snprintf(decimal, sizeof(decimal), "%" PRIu32, value);
    if (length < 0 || (size_t)length >= sizeof(decimal)) {
        return set_error(error, error_size,
                         "qualification uint32 formatting failed");
    }
    return buffer_append(buffer, decimal, (size_t)length, error, error_size);
}

static const char *tensor_class_name(ds4_laguna_tensor_class tensor_class) {
    if (tensor_class == DS4_LAGUNA_TENSOR_STATIC) return "STATIC";
    if (tensor_class == DS4_LAGUNA_TENSOR_ROUTED_EXPERT) {
        return "ROUTED_EXPERT";
    }
    return NULL;
}

static const char *routed_projection_name(
        ds4_laguna_routed_projection projection) {
    switch (projection) {
        case DS4_LAGUNA_ROUTED_PROJECTION_GATE: return "GATE";
        case DS4_LAGUNA_ROUTED_PROJECTION_UP: return "UP";
        case DS4_LAGUNA_ROUTED_PROJECTION_DOWN: return "DOWN";
        default: return NULL;
    }
}

static const char *source_kind_name(ds4_laguna_source_range_kind kind) {
    switch (kind) {
        case DS4_LAGUNA_SOURCE_HEADER: return "HEADER";
        case DS4_LAGUNA_SOURCE_METADATA: return "METADATA";
        case DS4_LAGUNA_SOURCE_TENSOR_DIRECTORY: return "TENSOR_DIRECTORY";
        case DS4_LAGUNA_SOURCE_ALIGNMENT_PADDING: return "ALIGNMENT_PADDING";
        case DS4_LAGUNA_SOURCE_TENSOR_PADDING: return "TENSOR_PADDING";
    }
    return NULL;
}

void ds4_laguna_page_plan_free(ds4_laguna_page_plan *plan) {
    if (plan == NULL) return;
    free(plan->ranges);
    memset(plan, 0, sizeof(*plan));
}

bool ds4_laguna_page_plan_make(ds4_laguna_page_plan *out,
                               const ds4_laguna_ledger *ledger,
                               uint64_t page_size,
                               char *error,
                               size_t error_size) {
    clear_error(error, error_size);
    if (out == NULL || ledger == NULL) {
        return set_error(error, error_size, "page plan argument is null");
    }
    if (out->ranges != NULL) {
        return set_error(error, error_size,
                         "page plan output must be empty before build");
    }
    memset(out, 0, sizeof(*out));
    if (!is_power_of_two(page_size)) {
        return set_error(error, error_size,
                         "page plan page size must be a power of two");
    }
    if (ledger->file_size == 0 || ledger->tensor_range_count == 0 ||
        ledger->tensor_ranges == NULL) {
        return set_error(error, error_size,
                         "page plan ledger tensor ranges are empty");
    }
    if (ledger->tensor_range_count >
            SIZE_MAX / sizeof(ds4_laguna_page_range)) {
        return set_error(error, error_size,
                         "page plan range allocation overflow");
    }

    const size_t count = ledger->tensor_range_count;
    ds4_laguna_page_range *input = calloc(count, sizeof(*input));
    ds4_laguna_page_range *ranges = calloc(count, sizeof(*ranges));
    if (input == NULL || ranges == NULL) {
        free(ranges);
        free(input);
        return set_error(error, error_size,
                         "page plan range allocation failed");
    }
    bool valid = true;
    for (size_t i = 0; i < count; i++) {
        const ds4_laguna_tensor_range *tensor = &ledger->tensor_ranges[i];
        uint64_t end = 0;
        if (!range_end(tensor->source_offset, tensor->source_bytes, &end) ||
            end > ledger->file_size) {
            valid = set_error(error, error_size,
                              "page plan tensor range is outside the model");
            break;
        }
        input[i].offset = tensor->source_offset;
        input[i].bytes = tensor->source_bytes;
    }

    size_t range_count = 0;
    uint64_t eligible = 0;
    uint64_t mapped = 0;
    if (valid &&
        !ds4_laguna_full_page_union(input, count, page_size,
                                    ranges, count, &range_count, &eligible)) {
        valid = set_error(error, error_size,
                          "page plan full-page range union failed");
    }
    if (valid && !align_u64(ledger->file_size, page_size, &mapped)) {
        valid = set_error(error, error_size,
                          "page plan mapped size overflow");
    }
    if (valid && eligible > mapped) {
        valid = set_error(error, error_size,
                          "page plan eligible bytes exceed mapped bytes");
    }
    free(input);
    if (!valid) {
        free(ranges);
        return false;
    }

    if (range_count == 0) {
        free(ranges);
        ranges = NULL;
    } else {
        ds4_laguna_page_range *smaller =
            realloc(ranges, range_count * sizeof(*ranges));
        if (smaller != NULL) ranges = smaller;
    }
    out->ranges = ranges;
    out->range_count = range_count;
    out->page_size = page_size;
    out->mapped_page_bytes = mapped;
    out->eligible_unique_bytes = eligible;
    out->unavoidable_bytes = mapped - eligible;
    return true;
}

static const ds4_laguna_tensor_range *find_tensor(
        const ds4_laguna_ledger *ledger,
        uint64_t stable_index) {
    for (size_t i = 0; i < ledger->tensor_range_count; i++) {
        if (ledger->tensor_ranges[i].stable_index == stable_index) {
            return &ledger->tensor_ranges[i];
        }
    }
    return NULL;
}

static bool validate_expert_view(const ds4_laguna_ledger *ledger,
                                 const ds4_laguna_expert_entry *entry,
                                 const ds4_laguna_expert_view *view,
                                 ds4_laguna_routed_projection projection,
                                 char *error,
                                 size_t error_size) {
    const ds4_laguna_tensor_range *parent =
        find_tensor(ledger, view->parent_stable_index);
    uint64_t view_end = 0;
    uint64_t parent_end = 0;
    uint64_t device_end = 0;
    uint64_t expert_delta = 0;
    uint64_t expected_offset = 0;
    if (parent == NULL ||
        parent->tensor_class != DS4_LAGUNA_TENSOR_ROUTED_EXPERT ||
        parent->routed_layer != entry->layer ||
        parent->routed_projection != projection) {
        return set_error(error, error_size,
                         "ledger expert view has the wrong routed parent");
    }
    if (!mul_u64(entry->expert,
                 ledger->routed_projection_expert_bytes,
                 &expert_delta) ||
        !add_u64(parent->source_offset, expert_delta, &expected_offset) ||
        !range_end(parent->source_offset, parent->source_bytes, &parent_end) ||
        !range_end(view->source_offset, view->source_bytes, &view_end) ||
        view->source_offset != expected_offset ||
        view_end > parent_end ||
        view->source_bytes != ledger->routed_projection_expert_bytes) {
        return set_error(error, error_size,
                         "ledger expert view does not match its exact subrange");
    }
    if (!range_end(view->device_offset, view->source_bytes, &device_end) ||
        device_end > entry->used_bytes) {
        return set_error(error, error_size,
                         "ledger expert device view exceeds its slot");
    }
    return true;
}

static bool validate_ledger(const ds4_laguna_ledger *ledger,
                            char *error,
                            size_t error_size) {
    if (ledger == NULL) {
        return set_error(error, error_size, "qualification ledger is null");
    }
    if (ledger->file_size == 0 || ledger->tensor_range_count == 0 ||
        ledger->source_range_count < 3u ||
        ledger->expert_entry_count == 0 ||
        ledger->tensor_ranges == NULL ||
        ledger->source_ranges == NULL ||
        ledger->expert_entries == NULL) {
        return set_error(error, error_size,
                         "qualification ledger arrays are null or empty");
    }
    if (ledger->tensor_count != ledger->tensor_range_count ||
        ledger->tensor_range_count > 814u ||
        ledger->expert_entry_count > 12032u ||
        ledger->expert_entry_count > SIZE_MAX ||
        ledger->tensor_range_count > (SIZE_MAX - 5u) / 2u ||
        ledger->source_range_count >
            ledger->tensor_range_count * 2u + 5u) {
        return set_error(error, error_size,
                         "qualification ledger counts are inconsistent");
    }
    uint64_t parent_count = 0;
    if (!add_u64(ledger->static_parent_count,
                 ledger->routed_parent_count,
                 &parent_count) ||
        parent_count != ledger->tensor_count) {
        return set_error(error, error_size,
                         "qualification ledger parent counts do not reconcile");
    }

    uint64_t static_bytes = 0;
    uint64_t routed_bytes = 0;
    uint64_t static_count = 0;
    uint64_t routed_count = 0;
    for (size_t i = 0; i < ledger->tensor_range_count; i++) {
        const ds4_laguna_tensor_range *tensor = &ledger->tensor_ranges[i];
        uint64_t end = 0;
        if (i != 0 &&
            tensor->stable_index <= ledger->tensor_ranges[i - 1u].stable_index) {
            return set_error(error, error_size,
                             "qualification tensor order is not stable");
        }
        const bool routed_identity_valid =
            (tensor->tensor_class == DS4_LAGUNA_TENSOR_STATIC &&
             tensor->routed_layer == UINT32_MAX &&
             tensor->routed_projection ==
                 DS4_LAGUNA_ROUTED_PROJECTION_NONE) ||
            (tensor->tensor_class == DS4_LAGUNA_TENSOR_ROUTED_EXPERT &&
             tensor->routed_layer != UINT32_MAX &&
             routed_projection_name(tensor->routed_projection) != NULL);
        if (tensor_class_name(tensor->tensor_class) == NULL ||
            !routed_identity_valid ||
            !range_end(tensor->source_offset, tensor->source_bytes, &end) ||
            end > ledger->file_size) {
            return set_error(error, error_size,
                             "qualification tensor range is invalid");
        }
        if (tensor->tensor_class == DS4_LAGUNA_TENSOR_STATIC) {
            if (!add_u64(static_bytes, tensor->source_bytes, &static_bytes) ||
                !add_u64(static_count, 1u, &static_count)) {
                return set_error(error, error_size,
                                 "qualification static tensor total overflow");
            }
        } else {
            if (!add_u64(routed_bytes, tensor->source_bytes, &routed_bytes) ||
                !add_u64(routed_count, 1u, &routed_count)) {
                return set_error(error, error_size,
                                 "qualification routed tensor total overflow");
            }
        }
        for (size_t j = 0; j < i; j++) {
            uint64_t other_end = 0;
            (void)range_end(ledger->tensor_ranges[j].source_offset,
                            ledger->tensor_ranges[j].source_bytes,
                            &other_end);
            if (ranges_overlap(tensor->source_offset, end,
                               ledger->tensor_ranges[j].source_offset,
                               other_end)) {
                return set_error(error, error_size,
                                 "qualification tensor ranges overlap");
            }
        }
    }
    if (static_count != ledger->static_parent_count ||
        routed_count != ledger->routed_parent_count ||
        static_bytes != ledger->static_source_bytes ||
        routed_bytes != ledger->routed_source_bytes ||
        ledger->static_aligned_device_bytes < static_bytes) {
        return set_error(error, error_size,
                         "qualification tensor scalar totals do not reconcile");
    }

    uint64_t source_bytes = 0;
    uint64_t previous_source_end = 0;
    ds4_laguna_source_range_kind previous_kind = DS4_LAGUNA_SOURCE_HEADER;
    for (size_t i = 0; i < ledger->source_range_count; i++) {
        const ds4_laguna_source_range *source = &ledger->source_ranges[i];
        uint64_t end = 0;
        if (source_kind_name(source->kind) == NULL ||
            (i < 3u && source->kind !=
                (ds4_laguna_source_range_kind)(i + 1u)) ||
            (i >= 3u && source->kind < previous_kind) ||
            (i == 0 && source->source_offset != 0) ||
            !range_end(source->source_offset, source->source_bytes, &end) ||
            end > ledger->file_size ||
            (i != 0 && source->source_offset < previous_source_end) ||
            !add_u64(source_bytes, source->source_bytes, &source_bytes)) {
            return set_error(error, error_size,
                             "qualification source range is invalid or unordered");
        }
        for (size_t j = 0; j < ledger->tensor_range_count; j++) {
            uint64_t tensor_end = 0;
            (void)range_end(ledger->tensor_ranges[j].source_offset,
                            ledger->tensor_ranges[j].source_bytes,
                            &tensor_end);
            if (ranges_overlap(source->source_offset, end,
                               ledger->tensor_ranges[j].source_offset,
                               tensor_end)) {
                return set_error(error, error_size,
                                 "qualification source and tensor ranges overlap");
            }
        }
        previous_source_end = end;
        previous_kind = source->kind;
    }
    uint64_t tensor_bytes = 0;
    uint64_t reconciled = 0;
    if (source_bytes != ledger->non_tensor_source_bytes ||
        !add_u64(static_bytes, routed_bytes, &tensor_bytes) ||
        !add_u64(tensor_bytes, source_bytes, &reconciled) ||
        reconciled != ledger->file_size) {
        return set_error(error, error_size,
                         "qualification full-file ledger does not reconcile");
    }

    if (ledger->routed_projection_expert_bytes == 0 ||
        ledger->slot_stride_bytes == 0) {
        return set_error(error, error_size,
                         "qualification expert geometry is empty");
    }
    uint64_t maximum_used = 0;
    for (size_t i = 0; i < (size_t)ledger->expert_entry_count; i++) {
        const ds4_laguna_expert_entry *entry = &ledger->expert_entries[i];
        if (i != 0) {
            const ds4_laguna_expert_entry *previous =
                &ledger->expert_entries[i - 1u];
            if (entry->layer < previous->layer ||
                (entry->layer == previous->layer &&
                 entry->expert <= previous->expert)) {
                return set_error(error, error_size,
                                 "qualification expert order is not stable");
            }
        }
        if (entry->used_bytes == 0 ||
            entry->used_bytes > ledger->slot_stride_bytes ||
            entry->gate.parent_stable_index == entry->up.parent_stable_index ||
            entry->gate.parent_stable_index == entry->down.parent_stable_index ||
            entry->up.parent_stable_index == entry->down.parent_stable_index ||
            !validate_expert_view(ledger, entry, &entry->gate,
                                  DS4_LAGUNA_ROUTED_PROJECTION_GATE,
                                  error, error_size) ||
            !validate_expert_view(ledger, entry, &entry->up,
                                  DS4_LAGUNA_ROUTED_PROJECTION_UP,
                                  error, error_size) ||
            !validate_expert_view(ledger, entry, &entry->down,
                                  DS4_LAGUNA_ROUTED_PROJECTION_DOWN,
                                  error, error_size)) {
            if (error != NULL && error_size != 0 && error[0] == '\0') {
                (void)set_error(error, error_size,
                                "qualification expert entry is invalid");
            }
            return false;
        }
        uint64_t gate_end = 0;
        uint64_t up_end = 0;
        (void)range_end(entry->gate.device_offset,
                        entry->gate.source_bytes, &gate_end);
        (void)range_end(entry->up.device_offset,
                        entry->up.source_bytes, &up_end);
        if (entry->gate.device_offset != 0 ||
            entry->up.device_offset < gate_end ||
            entry->down.device_offset < up_end) {
            return set_error(error, error_size,
                             "qualification expert device views overlap");
        }
        if (entry->used_bytes > maximum_used) maximum_used = entry->used_bytes;
    }
    if (maximum_used != ledger->slot_stride_bytes) {
        return set_error(error, error_size,
                         "qualification expert slot stride does not reconcile");
    }
    return true;
}

static const char *profile_for_cache(uint64_t cache_bytes,
                                     uint32_t session_count) {
    const uint64_t gib = UINT64_C(1024) * 1024u * 1024u;
    if (session_count == 1u) {
        if (cache_bytes == 8u * gib) return "cache-8gib";
        if (cache_bytes == 12u * gib) return "cache-12gib";
        if (cache_bytes == 16u * gib) return "cache-16gib";
    } else if (session_count == 2u) {
        if (cache_bytes == 8u * gib) return "cache-8gib-sessions-2";
        if (cache_bytes == 12u * gib) return "cache-12gib-sessions-2";
        if (cache_bytes == 16u * gib) return "cache-16gib-sessions-2";
    }
    return NULL;
}

static bool validate_allocation(const ds4_laguna_allocation_plan *plan,
                                const ds4_laguna_ledger *ledger,
                                char *error,
                                size_t error_size) {
    if (plan == NULL) {
        return set_error(error, error_size,
                         "qualification allocation plan is null");
    }
    const char *expected_profile =
        profile_for_cache(plan->configured_cache_bytes,
                          plan->session_count);
    if (expected_profile == NULL || plan->profile_id == NULL ||
        strcmp(plan->profile_id, expected_profile) != 0 ||
        plan->context_tokens != 32768u ||
        plan->prefill_rows != 4096u ||
        (plan->session_count != 1u && plan->session_count != 2u)) {
        return set_error(error, error_size,
                         "qualification allocation profile is invalid");
    }
    uint64_t expected_payload = 0;
    uint64_t expected_staging = 0;
    if (plan->effective_cache_limit_bytes != plan->configured_cache_bytes ||
        plan->slot_stride_bytes == 0 ||
        plan->slot_stride_bytes != ledger->slot_stride_bytes ||
        plan->slot_count == 0 ||
        (uint64_t)plan->slot_count !=
            plan->configured_cache_bytes / plan->slot_stride_bytes ||
        !mul_u64(plan->slot_count, plan->slot_stride_bytes,
                 &expected_payload) ||
        expected_payload != plan->cache_payload_bytes ||
        plan->cache_tail_uncharged_bytes !=
            plan->configured_cache_bytes - expected_payload ||
        plan->staging_buffer_count != 4u ||
        plan->staging_buffer_bytes != plan->slot_stride_bytes ||
        !mul_u64(plan->staging_buffer_count,
                 plan->staging_buffer_bytes, &expected_staging)) {
        return set_error(error, error_size,
                         "qualification allocation cache geometry is invalid");
    }
    if (plan->owned_category_bounds[DS4_RUNTIME_CATEGORY_STATIC_WEIGHTS] !=
            ledger->static_aligned_device_bytes ||
        plan->owned_category_bounds[
            DS4_RUNTIME_CATEGORY_EXPERT_CACHE_PAYLOAD] != expected_payload ||
        plan->owned_category_bounds[DS4_RUNTIME_CATEGORY_PINNED_STAGING] !=
            expected_staging ||
        plan->report_bounds[DS4_RUNTIME_REPORT_MODEL_MAPPED_VIRTUAL] !=
            ledger->file_size ||
        plan->report_bounds[
            DS4_RUNTIME_REPORT_MODEL_MAPPING_REGISTERED] != 0) {
        return set_error(error, error_size,
                         "qualification allocation ledger bounds do not match");
    }

    uint64_t category_sums[DS4_RUNTIME_OWNED_CATEGORY_COUNT] = {0};
    if (plan->callsite_count != DS4_LAGUNA_ALLOCATION_CALLSITE_COUNT) {
        return set_error(error, error_size,
                         "qualification allocation callsite count is invalid");
    }
    for (size_t i = 0; i < plan->callsite_count; i++) {
        const ds4_runtime_callsite *site = &plan->callsites[i];
        const callsite_contract *contract = &callsite_contracts[i];
        if (site->id != i + 1u || site->name == NULL ||
            strcmp(site->name, contract->name) != 0 ||
            site->category != contract->category ||
            site->domain != contract->domain ||
            site->category >= DS4_RUNTIME_OWNED_CATEGORY_COUNT ||
            site->domain >= DS4_RUNTIME_DOMAIN_COUNT ||
            !add_u64(category_sums[site->category], site->bound_bytes,
                     &category_sums[site->category])) {
            return set_error(error, error_size,
                             "qualification allocation callsite is invalid");
        }
    }

    uint64_t owned_total = 0;
    uint64_t owned_non_cache = 0;
    for (size_t i = 0; i < DS4_RUNTIME_OWNED_CATEGORY_COUNT; i++) {
        if (category_sums[i] != plan->owned_category_bounds[i] ||
            !add_u64(owned_total, plan->owned_category_bounds[i],
                     &owned_total)) {
            return set_error(error, error_size,
                             "qualification allocation categories do not reconcile");
        }
        if (i != DS4_RUNTIME_CATEGORY_EXPERT_CACHE_PAYLOAD &&
            !add_u64(owned_non_cache, plan->owned_category_bounds[i],
                     &owned_non_cache)) {
            return set_error(error, error_size,
                             "qualification non-cache allocation overflow");
        }
    }
    uint64_t external = 0;
    uint64_t qualification_non_cache = 0;
    uint64_t planned = 0;
    uint64_t total_bound = 0;
    const uint64_t gib = UINT64_C(1024) * 1024u * 1024u;
    if (!add_u64(plan->report_bounds[
                     DS4_RUNTIME_REPORT_MODEL_SOURCE_RESIDENT],
                 plan->report_bounds[
                     DS4_RUNTIME_REPORT_HOST_LIBRARY_UNATTRIBUTED],
                 &external) ||
        !add_u64(external,
                 plan->report_bounds[
                     DS4_RUNTIME_REPORT_CUDA_LIBRARY_UNATTRIBUTED],
                 &external) ||
        !add_u64(owned_non_cache, external, &qualification_non_cache) ||
        !add_u64(owned_total, external, &planned) ||
        !add_u64(plan->configured_cache_bytes,
                 (plan->session_count == 1u ? 16u : 20u) * gib,
                 &total_bound) ||
        plan->owned_non_cache_bound_bytes != owned_non_cache ||
        plan->owned_total_bound_bytes != owned_total ||
        plan->qualification_non_cache_bound_bytes != qualification_non_cache ||
        plan->planned_qualification_bytes != planned ||
        plan->qualification_total_bound_bytes != total_bound ||
        qualification_non_cache >
            (plan->session_count == 1u ? 16u : 20u) * gib ||
        planned > total_bound) {
        return set_error(error, error_size,
                         "qualification allocation totals do not reconcile");
    }
    return true;
}

static bool validate_page_plan(const ds4_laguna_page_plan *page_plan,
                               const ds4_laguna_ledger *ledger,
                               char *error,
                               size_t error_size) {
    if (page_plan == NULL ||
        (page_plan->range_count != 0 && page_plan->ranges == NULL)) {
        return set_error(error, error_size,
                         "qualification page plan is null or incomplete");
    }
    ds4_laguna_page_plan expected;
    memset(&expected, 0, sizeof(expected));
    if (!ds4_laguna_page_plan_make(&expected, ledger, page_plan->page_size,
                                    error, error_size)) {
        return false;
    }
    bool matches =
        expected.range_count == page_plan->range_count &&
        expected.page_size == page_plan->page_size &&
        expected.mapped_page_bytes == page_plan->mapped_page_bytes &&
        expected.eligible_unique_bytes == page_plan->eligible_unique_bytes &&
        expected.unavoidable_bytes == page_plan->unavoidable_bytes;
    for (size_t i = 0; matches && i < expected.range_count; i++) {
        matches = expected.ranges[i].offset == page_plan->ranges[i].offset &&
                  expected.ranges[i].bytes == page_plan->ranges[i].bytes;
    }
    ds4_laguna_page_plan_free(&expected);
    if (!matches) {
        return set_error(error, error_size,
                         "qualification page plan is not the canonical union");
    }
    return true;
}

static bool validate_input(
        const ds4_laguna_qualification_plan_input *input,
        char *error,
        size_t error_size) {
    if (input == NULL) {
        return set_error(error, error_size,
                         "qualification plan input is null");
    }
    if (!validate_ledger(input->ledger, error, error_size) ||
        !validate_allocation(input->allocation, input->ledger,
                             error, error_size) ||
        !validate_page_plan(input->page_cache, input->ledger,
                            error, error_size)) {
        return false;
    }
    if (input->model_identity.device == 0 ||
        input->model_identity.inode == 0 ||
        input->model_identity.size_bytes == 0 ||
        input->model_identity.mtime_ns == 0 ||
        input->model_identity.size_bytes != input->ledger->file_size) {
        return set_error(error, error_size,
                         "qualification opened-model identity is invalid");
    }
    if (input->model_sha256 == NULL) {
        return set_error(error, error_size,
                         "qualification observed model digest is missing");
    }
    for (size_t i = 0; i < DS4_PLAN_IO_SHA256_HEX_LENGTH; i++) {
        const char c = input->model_sha256[i];
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) {
            return set_error(error, error_size,
                             "qualification observed model digest is malformed");
        }
    }
    if (input->model_sha256[DS4_PLAN_IO_SHA256_HEX_LENGTH] != '\0') {
        return set_error(error, error_size,
                         "qualification observed model digest is malformed");
    }
    if (strcmp(input->model_sha256, DS4_LAGUNA_MODEL_SHA256) != 0) {
        return set_error(error, error_size,
                         "qualification observed model digest is not pinned");
    }
    return true;
}

static bool serialize_expert_view(json_buffer *buffer,
                                  const ds4_laguna_expert_view *view,
                                  char *error,
                                  size_t error_size) {
    return append_literal(buffer, "{\"device_offset\":", error, error_size) &&
           append_u64_string(buffer, view->device_offset, error, error_size) &&
           append_literal(buffer, ",\"parent_stable_index\":",
                          error, error_size) &&
           append_u64_string(buffer, view->parent_stable_index,
                             error, error_size) &&
           append_literal(buffer, ",\"source_bytes\":", error, error_size) &&
           append_u64_string(buffer, view->source_bytes, error, error_size) &&
           append_literal(buffer, ",\"source_offset\":", error, error_size) &&
           append_u64_string(buffer, view->source_offset, error, error_size) &&
           append_literal(buffer, "}", error, error_size);
}

static bool serialize_ledger(const ds4_laguna_ledger *ledger,
                             json_buffer *buffer,
                             char *error,
                             size_t error_size) {
    if (!append_literal(buffer, "{\"expert_entries\":[", error, error_size)) {
        return false;
    }
    for (size_t i = 0; i < (size_t)ledger->expert_entry_count; i++) {
        const ds4_laguna_expert_entry *entry = &ledger->expert_entries[i];
        if ((i != 0 && !append_literal(buffer, ",", error, error_size)) ||
            !append_literal(buffer, "{\"down\":", error, error_size) ||
            !serialize_expert_view(buffer, &entry->down, error, error_size) ||
            !append_literal(buffer, ",\"expert\":", error, error_size) ||
            !append_u32(buffer, entry->expert, error, error_size) ||
            !append_literal(buffer, ",\"gate\":", error, error_size) ||
            !serialize_expert_view(buffer, &entry->gate, error, error_size) ||
            !append_literal(buffer, ",\"layer\":", error, error_size) ||
            !append_u32(buffer, entry->layer, error, error_size) ||
            !append_literal(buffer, ",\"up\":", error, error_size) ||
            !serialize_expert_view(buffer, &entry->up, error, error_size) ||
            !append_literal(buffer, ",\"used_bytes\":", error, error_size) ||
            !append_u64_string(buffer, entry->used_bytes, error, error_size) ||
            !append_literal(buffer, "}", error, error_size)) {
            return false;
        }
    }
    if (!append_literal(buffer, "],\"expert_entry_count\":",
                        error, error_size) ||
        !append_u64_string(buffer, ledger->expert_entry_count,
                           error, error_size) ||
        !append_literal(buffer, ",\"file_size\":", error, error_size) ||
        !append_u64_string(buffer, ledger->file_size, error, error_size) ||
        !append_literal(buffer, ",\"non_tensor_source_bytes\":",
                        error, error_size) ||
        !append_u64_string(buffer, ledger->non_tensor_source_bytes,
                           error, error_size) ||
        !append_literal(buffer, ",\"routed_parent_count\":",
                        error, error_size) ||
        !append_u64_string(buffer, ledger->routed_parent_count,
                           error, error_size) ||
        !append_literal(buffer, ",\"routed_projection_expert_bytes\":",
                        error, error_size) ||
        !append_u64_string(buffer, ledger->routed_projection_expert_bytes,
                           error, error_size) ||
        !append_literal(buffer, ",\"routed_source_bytes\":",
                        error, error_size) ||
        !append_u64_string(buffer, ledger->routed_source_bytes,
                           error, error_size) ||
        !append_literal(buffer, ",\"slot_stride_bytes\":",
                        error, error_size) ||
        !append_u64_string(buffer, ledger->slot_stride_bytes,
                           error, error_size) ||
        !append_literal(buffer, ",\"source_range_count\":",
                        error, error_size) ||
        !append_u64_string(buffer, (uint64_t)ledger->source_range_count,
                           error, error_size) ||
        !append_literal(buffer, ",\"source_ranges\":[", error, error_size)) {
        return false;
    }
    for (size_t i = 0; i < ledger->source_range_count; i++) {
        const ds4_laguna_source_range *range = &ledger->source_ranges[i];
        if ((i != 0 && !append_literal(buffer, ",", error, error_size)) ||
            !append_literal(buffer, "{\"kind\":", error, error_size) ||
            !append_json_string(buffer, source_kind_name(range->kind),
                                error, error_size) ||
            !append_literal(buffer, ",\"source_bytes\":", error, error_size) ||
            !append_u64_string(buffer, range->source_bytes,
                               error, error_size) ||
            !append_literal(buffer, ",\"source_offset\":", error, error_size) ||
            !append_u64_string(buffer, range->source_offset,
                               error, error_size) ||
            !append_literal(buffer, "}", error, error_size)) {
            return false;
        }
    }
    if (!append_literal(buffer, "],\"static_aligned_device_bytes\":",
                        error, error_size) ||
        !append_u64_string(buffer, ledger->static_aligned_device_bytes,
                           error, error_size) ||
        !append_literal(buffer, ",\"static_parent_count\":",
                        error, error_size) ||
        !append_u64_string(buffer, ledger->static_parent_count,
                           error, error_size) ||
        !append_literal(buffer, ",\"static_source_bytes\":",
                        error, error_size) ||
        !append_u64_string(buffer, ledger->static_source_bytes,
                           error, error_size) ||
        !append_literal(buffer, ",\"tensor_count\":", error, error_size) ||
        !append_u64_string(buffer, ledger->tensor_count, error, error_size) ||
        !append_literal(buffer, ",\"tensor_range_count\":",
                        error, error_size) ||
        !append_u64_string(buffer, (uint64_t)ledger->tensor_range_count,
                           error, error_size) ||
        !append_literal(buffer, ",\"tensor_ranges\":[", error, error_size)) {
        return false;
    }
    for (size_t i = 0; i < ledger->tensor_range_count; i++) {
        const ds4_laguna_tensor_range *range = &ledger->tensor_ranges[i];
        if ((i != 0 && !append_literal(buffer, ",", error, error_size)) ||
            !append_literal(buffer, "{\"class\":", error, error_size) ||
            !append_json_string(buffer, tensor_class_name(range->tensor_class),
                                error, error_size)) {
            return false;
        }
        if (range->tensor_class == DS4_LAGUNA_TENSOR_ROUTED_EXPERT &&
            (!append_literal(buffer, ",\"routed_layer\":",
                             error, error_size) ||
             !append_u32(buffer, range->routed_layer, error, error_size) ||
             !append_literal(buffer, ",\"routed_projection\":",
                             error, error_size) ||
             !append_json_string(
                 buffer, routed_projection_name(range->routed_projection),
                 error, error_size))) {
            return false;
        }
        if (!append_literal(buffer, ",\"source_bytes\":", error, error_size) ||
            !append_u64_string(buffer, range->source_bytes,
                               error, error_size) ||
            !append_literal(buffer, ",\"source_offset\":", error, error_size) ||
            !append_u64_string(buffer, range->source_offset,
                               error, error_size) ||
            !append_literal(buffer, ",\"stable_index\":", error, error_size) ||
            !append_u64_string(buffer, range->stable_index,
                               error, error_size) ||
            !append_literal(buffer, "}", error, error_size)) {
            return false;
        }
    }
    return append_literal(buffer, "]}", error, error_size);
}

static bool serialize_cache(const ds4_laguna_allocation_plan *plan,
                            json_buffer *buffer,
                            char *error,
                            size_t error_size) {
    return append_literal(buffer, "{\"cache_payload_bytes\":",
                          error, error_size) &&
           append_u64_string(buffer, plan->cache_payload_bytes,
                             error, error_size) &&
           append_literal(buffer, ",\"cache_tail_uncharged_bytes\":",
                          error, error_size) &&
           append_u64_string(buffer, plan->cache_tail_uncharged_bytes,
                             error, error_size) &&
           append_literal(buffer, ",\"configured_cache_bytes\":",
                          error, error_size) &&
           append_u64_string(buffer, plan->configured_cache_bytes,
                             error, error_size) &&
           append_literal(buffer, ",\"effective_cache_limit_bytes\":",
                          error, error_size) &&
           append_u64_string(buffer, plan->effective_cache_limit_bytes,
                             error, error_size) &&
           append_literal(buffer, ",\"slot_count\":", error, error_size) &&
           append_u32(buffer, plan->slot_count, error, error_size) &&
           append_literal(buffer, ",\"slot_stride_bytes\":",
                          error, error_size) &&
           append_u64_string(buffer, plan->slot_stride_bytes,
                             error, error_size) &&
           append_literal(buffer, ",\"staging_buffer_bytes\":",
                          error, error_size) &&
           append_u64_string(buffer, plan->staging_buffer_bytes,
                             error, error_size) &&
           append_literal(buffer, ",\"staging_buffer_count\":",
                          error, error_size) &&
           append_u32(buffer, plan->staging_buffer_count, error, error_size) &&
           append_literal(buffer, "}", error, error_size);
}

static bool serialize_allocation(const ds4_laguna_allocation_plan *plan,
                                 json_buffer *buffer,
                                 char *error,
                                 size_t error_size) {
    if (!append_literal(buffer, "{\"cache\":", error, error_size) ||
        !serialize_cache(plan, buffer, error, error_size) ||
        !append_literal(buffer, ",\"callsites\":[", error, error_size)) {
        return false;
    }
    for (size_t i = 0; i < plan->callsite_count; i++) {
        const ds4_runtime_callsite *site = &plan->callsites[i];
        if ((i != 0 && !append_literal(buffer, ",", error, error_size)) ||
            !append_literal(buffer, "{\"bound_bytes\":", error, error_size) ||
            !append_u64_string(buffer, site->bound_bytes, error, error_size) ||
            !append_literal(buffer, ",\"category\":", error, error_size) ||
            !append_json_string(buffer, category_names[site->category],
                                error, error_size) ||
            !append_literal(buffer, ",\"domain\":", error, error_size) ||
            !append_json_string(buffer, domain_names[site->domain],
                                error, error_size) ||
            !append_literal(buffer, ",\"id\":", error, error_size) ||
            !append_u32(buffer, site->id, error, error_size) ||
            !append_literal(buffer, ",\"name\":", error, error_size) ||
            !append_json_string(buffer, site->name, error, error_size) ||
            !append_literal(buffer, "}", error, error_size)) {
            return false;
        }
    }
    if (!append_literal(buffer, "],\"category_bounds\":[",
                        error, error_size)) {
        return false;
    }
    for (size_t i = 0; i < DS4_RUNTIME_OWNED_CATEGORY_COUNT; i++) {
        if ((i != 0 && !append_literal(buffer, ",", error, error_size)) ||
            !append_literal(buffer, "{\"bound_bytes\":", error, error_size) ||
            !append_u64_string(buffer, plan->owned_category_bounds[i],
                               error, error_size) ||
            !append_literal(buffer, ",\"category\":", error, error_size) ||
            !append_json_string(buffer, category_names[i],
                                error, error_size) ||
            !append_literal(buffer, "}", error, error_size)) {
            return false;
        }
    }
    if (!append_literal(buffer,
            "],\"configuration\":{\"backend\":\"cuda\",\"context_tokens\":",
            error, error_size) ||
        !append_u32(buffer, plan->context_tokens, error, error_size) ||
        !append_literal(buffer, ",\"prefill_rows\":", error, error_size) ||
        !append_u32(buffer, plan->prefill_rows, error, error_size) ||
        !append_literal(buffer, ",\"session_count\":", error, error_size) ||
        !append_u32(buffer, plan->session_count, error, error_size) ||
        !append_literal(buffer, "},\"non_owned_bounds\":[",
                        error, error_size)) {
        return false;
    }
    for (size_t i = 0; i < DS4_RUNTIME_REPORT_COUNT; i++) {
        if ((i != 0 && !append_literal(buffer, ",", error, error_size)) ||
            !append_literal(buffer, "{\"bound_bytes\":", error, error_size) ||
            !append_u64_string(buffer, plan->report_bounds[i],
                               error, error_size) ||
            !append_literal(buffer, ",\"report\":", error, error_size) ||
            !append_json_string(buffer, report_names[i], error, error_size) ||
            !append_literal(buffer, "}", error, error_size)) {
            return false;
        }
    }
    return append_literal(buffer, "],\"owned_non_cache_bound_bytes\":",
                          error, error_size) &&
           append_u64_string(buffer, plan->owned_non_cache_bound_bytes,
                             error, error_size) &&
           append_literal(buffer, ",\"owned_total_bound_bytes\":",
                          error, error_size) &&
           append_u64_string(buffer, plan->owned_total_bound_bytes,
                             error, error_size) &&
           append_literal(buffer, ",\"planned_qualification_bytes\":",
                          error, error_size) &&
           append_u64_string(buffer, plan->planned_qualification_bytes,
                             error, error_size) &&
           append_literal(buffer, ",\"profile_id\":", error, error_size) &&
           append_json_string(buffer, plan->profile_id, error, error_size) &&
           append_literal(buffer,
                          ",\"qualification_non_cache_bound_bytes\":",
                          error, error_size) &&
           append_u64_string(buffer,
                             plan->qualification_non_cache_bound_bytes,
                             error, error_size) &&
           append_literal(buffer, ",\"qualification_total_bound_bytes\":",
                          error, error_size) &&
           append_u64_string(buffer, plan->qualification_total_bound_bytes,
                             error, error_size) &&
           append_literal(buffer, "}", error, error_size);
}

static bool serialize_model(const ds4_laguna_file_identity *identity,
                            const char *model_sha256,
                            json_buffer *buffer,
                            char *error,
                            size_t error_size) {
    return append_literal(buffer, "{\"device\":", error, error_size) &&
           append_u64_string(buffer, identity->device, error, error_size) &&
           append_literal(buffer, ",\"filename\":", error, error_size) &&
           append_json_string(buffer, DS4_LAGUNA_MODEL_FILENAME,
                              error, error_size) &&
           append_literal(buffer, ",\"inode\":", error, error_size) &&
           append_u64_string(buffer, identity->inode, error, error_size) &&
           append_literal(buffer, ",\"mtime_ns\":", error, error_size) &&
           append_u64_string(buffer, identity->mtime_ns, error, error_size) &&
           append_literal(buffer, ",\"repository\":", error, error_size) &&
           append_json_string(buffer, DS4_LAGUNA_MODEL_REPOSITORY,
                              error, error_size) &&
           append_literal(buffer, ",\"revision\":", error, error_size) &&
           append_json_string(buffer, DS4_LAGUNA_MODEL_REVISION,
                              error, error_size) &&
           append_literal(buffer, ",\"sha256\":",
                          error, error_size) &&
           append_json_string(buffer, model_sha256, error, error_size) &&
           append_literal(buffer, ",\"size_bytes\":", error, error_size) &&
           append_u64_string(buffer, identity->size_bytes, error, error_size) &&
           append_literal(buffer, "}", error, error_size);
}

static bool serialize_page_cache(const ds4_laguna_page_plan *plan,
                                 json_buffer *buffer,
                                 char *error,
                                 size_t error_size) {
    if (!append_literal(buffer, "{\"eligible_unique_bytes\":",
                        error, error_size) ||
        !append_u64_string(buffer, plan->eligible_unique_bytes,
                           error, error_size) ||
        !append_literal(buffer, ",\"mapped_page_bytes\":",
                        error, error_size) ||
        !append_u64_string(buffer, plan->mapped_page_bytes,
                           error, error_size) ||
        !append_literal(buffer, ",\"page_size\":", error, error_size) ||
        !append_u64_string(buffer, plan->page_size, error, error_size) ||
        !append_literal(buffer, ",\"ranges\":[", error, error_size)) {
        return false;
    }
    for (size_t i = 0; i < plan->range_count; i++) {
        if ((i != 0 && !append_literal(buffer, ",", error, error_size)) ||
            !append_literal(buffer, "{\"bytes\":", error, error_size) ||
            !append_u64_string(buffer, plan->ranges[i].bytes,
                               error, error_size) ||
            !append_literal(buffer, ",\"offset\":", error, error_size) ||
            !append_u64_string(buffer, plan->ranges[i].offset,
                               error, error_size) ||
            !append_literal(buffer, "}", error, error_size)) {
            return false;
        }
    }
    return append_literal(buffer, "],\"unavoidable_bytes\":",
                          error, error_size) &&
           append_u64_string(buffer, plan->unavoidable_bytes,
                             error, error_size) &&
           append_literal(buffer, "}", error, error_size);
}

bool ds4_laguna_qualification_plan_serialize(
        const ds4_laguna_qualification_plan_input *input,
        char **bytes_out,
        size_t *size_out,
        char ledger_sha256[DS4_PLAN_IO_SHA256_HEX_SIZE],
        char *error,
        size_t error_size) {
    clear_error(error, error_size);
    if (bytes_out != NULL) *bytes_out = NULL;
    if (size_out != NULL) *size_out = 0;
    if (ledger_sha256 != NULL) ledger_sha256[0] = '\0';
    if (bytes_out == NULL || size_out == NULL || ledger_sha256 == NULL) {
        return set_error(error, error_size,
                         "qualification serializer output is null");
    }
    if (!validate_input(input, error, error_size)) return false;

    json_buffer ledger_buffer = {0};
    json_buffer document = {0};
    char digest[DS4_PLAN_IO_SHA256_HEX_SIZE];
    bool success = false;
    if (!serialize_ledger(input->ledger, &ledger_buffer, error, error_size) ||
        !ds4_plan_io_sha256(ledger_buffer.bytes, ledger_buffer.size,
                            digest, error, error_size) ||
        !append_literal(&document, "{\"allocation\":", error, error_size) ||
        !serialize_allocation(input->allocation, &document,
                              error, error_size) ||
        !append_literal(&document, ",\"ledger\":", error, error_size) ||
        !buffer_append(&document, ledger_buffer.bytes, ledger_buffer.size,
                       error, error_size) ||
        !append_literal(&document, ",\"ledger_sha256\":",
                        error, error_size) ||
        !append_json_string(&document, digest, error, error_size) ||
        !append_literal(&document, ",\"model\":", error, error_size) ||
        !serialize_model(&input->model_identity, input->model_sha256, &document,
                         error, error_size) ||
        !append_literal(&document, ",\"page_cache\":", error, error_size) ||
        !serialize_page_cache(input->page_cache, &document,
                              error, error_size) ||
        !append_literal(&document, ",\"schema\":", error, error_size) ||
        !append_json_string(&document, DS4_LAGUNA_QUALIFICATION_PLAN_SCHEMA,
                            error, error_size) ||
        !append_literal(&document, "}", error, error_size)) {
        goto cleanup;
    }

    *bytes_out = document.bytes;
    *size_out = document.size;
    memcpy(ledger_sha256, digest, sizeof(digest));
    document.bytes = NULL;
    success = true;

cleanup:
    free(document.bytes);
    free(ledger_buffer.bytes);
    return success;
}

void ds4_laguna_qualification_plan_bytes_free(char *bytes) {
    free(bytes);
}

bool ds4_laguna_qualification_plan_publish(
        const char *path,
        const ds4_laguna_qualification_plan_input *input,
        char plan_sha256[DS4_PLAN_IO_SHA256_HEX_SIZE],
        char ledger_sha256[DS4_PLAN_IO_SHA256_HEX_SIZE],
        char *error,
        size_t error_size) {
    clear_error(error, error_size);
    if (plan_sha256 != NULL) plan_sha256[0] = '\0';
    if (ledger_sha256 != NULL) ledger_sha256[0] = '\0';
    if (plan_sha256 == NULL || ledger_sha256 == NULL) {
        return set_error(error, error_size,
                         "qualification publisher digest output is null");
    }
    char *bytes = NULL;
    size_t size = 0;
    char ledger_digest[DS4_PLAN_IO_SHA256_HEX_SIZE];
    if (!ds4_laguna_qualification_plan_serialize(
            input, &bytes, &size, ledger_digest, error, error_size)) {
        return false;
    }
    const bool published = ds4_plan_io_publish(
        path, bytes, size, plan_sha256, error, error_size);
    free(bytes);
    if (!published) {
        plan_sha256[0] = '\0';
        ledger_sha256[0] = '\0';
        return false;
    }
    memcpy(ledger_sha256, ledger_digest, sizeof(ledger_digest));
    return true;
}
