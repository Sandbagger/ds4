#include "ds4_laguna_stream.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const ds4_laguna_tensor_desc *tensor;
} tensor_ref;

static bool set_error(char *err, size_t errlen, const char *fmt, ...) {
    if (err && errlen != 0) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(err, errlen, fmt, ap);
        va_end(ap);
    }
    return false;
}

static bool add_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (a > UINT64_MAX - b) return false;
    *out = a + b;
    return true;
}

static bool mul_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (a != 0 && b > UINT64_MAX / a) return false;
    *out = a * b;
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

static int compare_tensor_ref_source(const void *lhs, const void *rhs) {
    const tensor_ref *a = lhs;
    const tensor_ref *b = rhs;
    if (a->tensor->source_offset < b->tensor->source_offset) return -1;
    if (a->tensor->source_offset > b->tensor->source_offset) return 1;
    if (a->tensor->stable_index < b->tensor->stable_index) return -1;
    if (a->tensor->stable_index > b->tensor->stable_index) return 1;
    return 0;
}

static bool append_source_range(ds4_laguna_ledger *ledger,
                                size_t source_capacity,
                                ds4_laguna_source_range_kind kind,
                                uint64_t offset,
                                uint64_t bytes,
                                char *err,
                                size_t errlen) {
    if (bytes == 0) return true;
    if (ledger->source_range_count >= source_capacity) {
        return set_error(err, errlen, "source range allocation overflow");
    }
    ds4_laguna_source_range *range =
        &ledger->source_ranges[ledger->source_range_count++];
    range->kind = kind;
    range->source_offset = offset;
    range->source_bytes = bytes;
    if (!add_u64(ledger->non_tensor_source_bytes, bytes,
                 &ledger->non_tensor_source_bytes)) {
        return set_error(err, errlen, "non-tensor source byte overflow");
    }
    return true;
}

static bool tensor_names_equal(const ds4_laguna_tensor_desc *a,
                               const ds4_laguna_tensor_desc *b) {
    return a->name_len == b->name_len &&
           memcmp(a->name, b->name, a->name_len) == 0;
}

static bool tensor_shape_bytes(const ds4_laguna_tensor_desc *tensor,
                               uint64_t *elements,
                               uint64_t *bytes,
                               char *err,
                               size_t errlen) {
    if (tensor->ndim == 0 || tensor->ndim > DS4_LAGUNA_MAX_DIMS) {
        return set_error(err, errlen,
                         "tensor %llu has invalid dimension count %u",
                         (unsigned long long)tensor->stable_index,
                         tensor->ndim);
    }
    if (tensor->block_elems == 0 || tensor->block_bytes == 0) {
        return set_error(err, errlen,
                         "tensor %llu has invalid block geometry",
                         (unsigned long long)tensor->stable_index);
    }

    uint64_t count = 1;
    for (uint32_t d = 0; d < tensor->ndim; d++) {
        if (tensor->dim[d] == 0) {
            return set_error(err, errlen,
                             "tensor %llu has a zero dimension",
                             (unsigned long long)tensor->stable_index);
        }
        if (!mul_u64(count, tensor->dim[d], &count)) {
            return set_error(err, errlen,
                             "tensor %llu dimension multiplication overflow",
                             (unsigned long long)tensor->stable_index);
        }
    }

    uint64_t blocks = count / tensor->block_elems;
    if (count % tensor->block_elems != 0) {
        if (!add_u64(blocks, 1u, &blocks)) {
            return set_error(err, errlen,
                             "tensor %llu block count overflow",
                             (unsigned long long)tensor->stable_index);
        }
    }
    uint64_t packed = 0;
    if (!mul_u64(blocks, tensor->block_bytes, &packed)) {
        return set_error(err, errlen,
                         "tensor %llu packed size overflow",
                         (unsigned long long)tensor->stable_index);
    }
    *elements = count;
    *bytes = packed;
    return true;
}

void ds4_laguna_ledger_free(ds4_laguna_ledger *ledger) {
    if (!ledger) return;
    free(ledger->tensor_ranges);
    free(ledger->source_ranges);
    free(ledger->expert_entries);
    memset(ledger, 0, sizeof(*ledger));
}

bool ds4_laguna_ledger_build(ds4_laguna_ledger *out,
                             const ds4_laguna_ledger_spec *spec,
                             const ds4_laguna_tensor_desc *tensors,
                             size_t n_tensors,
                             char *err,
                             size_t errlen) {
    if (err && errlen != 0) err[0] = '\0';
    if (!out) return set_error(err, errlen, "ledger output is null");
    if (out->tensor_ranges || out->source_ranges || out->expert_entries) {
        return set_error(err, errlen,
                         "ledger output must be empty before build");
    }
    memset(out, 0, sizeof(*out));
    if (!spec) return set_error(err, errlen, "ledger spec is null");
    if (!tensors || n_tensors == 0) {
        return set_error(err, errlen, "ledger tensor table is null or empty");
    }
    if (!is_power_of_two(spec->gguf_alignment)) {
        return set_error(err, errlen,
                         "GGUF alignment must be a nonzero power of two");
    }
    if (spec->device_alignment != 256u) {
        return set_error(err, errlen,
                         "device alignment must be exactly 256 bytes");
    }
    if (spec->header_end == 0) {
        return set_error(err, errlen, "GGUF header range is empty");
    }
    if (spec->header_end > spec->metadata_end ||
        spec->metadata_end > spec->tensor_directory_end ||
        spec->tensor_directory_end > spec->tensor_data_start ||
        spec->tensor_data_start > spec->file_size) {
        return set_error(err, errlen, "malformed GGUF structural boundaries");
    }
    if (spec->tensor_directory_end == spec->metadata_end) {
        return set_error(err, errlen, "GGUF tensor directory range is empty");
    }
    uint64_t expected_tensor_data_start = 0;
    if (!align_u64(spec->tensor_directory_end, spec->gguf_alignment,
                   &expected_tensor_data_start) ||
        spec->tensor_data_start != expected_tensor_data_start) {
        return set_error(err, errlen,
                         "tensor data start violates exact alignment");
    }
    if (spec->first_routed_layer >= spec->layer_count ||
        spec->expert_count == 0) {
        return set_error(err, errlen, "invalid routed layer or expert bounds");
    }
    if (n_tensors > SIZE_MAX / sizeof(out->tensor_ranges[0]) ||
        n_tensors > SIZE_MAX / sizeof(tensor_ref)) {
        return set_error(err, errlen, "tensor table allocation overflow");
    }

    const uint64_t routed_layer_count =
        (uint64_t)spec->layer_count - spec->first_routed_layer;
    uint64_t expert_entry_count = 0;
    if (!mul_u64(routed_layer_count, spec->expert_count,
                 &expert_entry_count) ||
        expert_entry_count > SIZE_MAX /
            sizeof(out->expert_entries[0])) {
        return set_error(err, errlen, "expert entry allocation overflow");
    }
    uint64_t role_count_u64 = 0;
    if (!mul_u64(routed_layer_count, 3u, &role_count_u64) ||
        role_count_u64 > SIZE_MAX / sizeof(ds4_laguna_tensor_desc *)) {
        return set_error(err, errlen, "routed role allocation overflow");
    }
    if (n_tensors > (SIZE_MAX - 5u) / 2u) {
        return set_error(err, errlen, "source range allocation overflow");
    }
    const size_t source_capacity = n_tensors * 2u + 5u;
    if (source_capacity > SIZE_MAX / sizeof(out->source_ranges[0])) {
        return set_error(err, errlen, "source range allocation overflow");
    }

    out->tensor_ranges = calloc(n_tensors, sizeof(out->tensor_ranges[0]));
    out->source_ranges = calloc(source_capacity,
                                sizeof(out->source_ranges[0]));
    out->expert_entries = calloc((size_t)expert_entry_count,
                                 sizeof(out->expert_entries[0]));
    tensor_ref *source_order = calloc(n_tensors, sizeof(source_order[0]));
    const ds4_laguna_tensor_desc **roles =
        calloc((size_t)role_count_u64, sizeof(roles[0]));
    if (!out->tensor_ranges || !out->source_ranges ||
        !out->expert_entries || !source_order || !roles) {
        free(source_order);
        free(roles);
        ds4_laguna_ledger_free(out);
        return set_error(err, errlen, "ledger allocation failed");
    }

    out->file_size = spec->file_size;
    out->tensor_count = n_tensors;
    out->tensor_range_count = n_tensors;
    out->expert_entry_count = expert_entry_count;

    bool ok = true;
    uint64_t common_expert_bytes = 0;
    for (size_t i = 0; ok && i < n_tensors; i++) {
        const ds4_laguna_tensor_desc *tensor = &tensors[i];
        if (!tensor->name || tensor->name_len == 0) {
            ok = set_error(err, errlen,
                           "tensor %llu has an invalid name",
                           (unsigned long long)tensor->stable_index);
            break;
        }
        for (size_t j = 0; j < i; j++) {
            if (tensor->stable_index == tensors[j].stable_index) {
                ok = set_error(err, errlen,
                               "duplicate tensor stable index %llu",
                               (unsigned long long)tensor->stable_index);
                break;
            }
            if (tensor_names_equal(tensor, &tensors[j])) {
                ok = set_error(err, errlen,
                               "duplicate tensor name at stable index %llu",
                               (unsigned long long)tensor->stable_index);
                break;
            }
        }
        if (!ok) break;
        if (tensor->tensor_class == DS4_LAGUNA_TENSOR_UNCLASSIFIED) {
            ok = set_error(err, errlen,
                           "tensor %llu is unclassified",
                           (unsigned long long)tensor->stable_index);
            break;
        }
        if (tensor->tensor_class != DS4_LAGUNA_TENSOR_STATIC &&
            tensor->tensor_class != DS4_LAGUNA_TENSOR_ROUTED_EXPERT) {
            ok = set_error(err, errlen,
                           "tensor %llu has an invalid class",
                           (unsigned long long)tensor->stable_index);
            break;
        }
        if (tensor->source_offset < spec->tensor_data_start) {
            ok = set_error(err, errlen,
                           "tensor %llu starts before tensor data",
                           (unsigned long long)tensor->stable_index);
            break;
        }
        if ((tensor->source_offset - spec->tensor_data_start) %
                spec->gguf_alignment != 0) {
            ok = set_error(err, errlen,
                           "tensor %llu source offset violates GGUF alignment",
                           (unsigned long long)tensor->stable_index);
            break;
        }
        if (tensor->source_offset > spec->file_size ||
            tensor->source_bytes > spec->file_size - tensor->source_offset) {
            ok = set_error(err, errlen,
                           "tensor %llu points outside the file",
                           (unsigned long long)tensor->stable_index);
            break;
        }

        uint64_t elements = 0;
        uint64_t expected_bytes = 0;
        if (!tensor_shape_bytes(tensor, &elements, &expected_bytes,
                                err, errlen)) {
            ok = false;
            break;
        }
        (void)elements;

        uint64_t expert_bytes = 0;
        if (tensor->tensor_class == DS4_LAGUNA_TENSOR_STATIC) {
            if (tensor->routed_layer != UINT32_MAX ||
                tensor->routed_projection !=
                    DS4_LAGUNA_ROUTED_PROJECTION_NONE) {
                ok = set_error(err, errlen,
                               "static tensor %llu claims a routed role",
                               (unsigned long long)tensor->stable_index);
                break;
            }
            if (tensor->source_bytes != expected_bytes) {
                ok = set_error(err, errlen,
                               "static tensor %llu source size mismatch",
                               (unsigned long long)tensor->stable_index);
                break;
            }
            uint64_t aligned_bytes = 0;
            if (!align_u64(tensor->source_bytes,
                           spec->device_alignment,
                           &aligned_bytes) ||
                !add_u64(out->static_aligned_device_bytes,
                         aligned_bytes,
                         &out->static_aligned_device_bytes)) {
                ok = set_error(err, errlen,
                               "static tensor device alignment overflow");
                break;
            }
            if (!add_u64(out->static_source_bytes,
                         tensor->source_bytes,
                         &out->static_source_bytes)) {
                ok = set_error(err, errlen, "static source byte overflow");
                break;
            }
            out->static_parent_count++;
        } else {
            if (tensor->ndim != 3) {
                ok = set_error(err, errlen,
                               "routed tensor %llu must have three dimensions",
                               (unsigned long long)tensor->stable_index);
                break;
            }
            if (tensor->dim[0] % tensor->block_elems != 0) {
                ok = set_error(err, errlen,
                               "routed tensor %llu dim0 is not block-exact",
                               (unsigned long long)tensor->stable_index);
                break;
            }
            if (tensor->dim[2] != spec->expert_count) {
                ok = set_error(err, errlen,
                               "routed tensor %llu dim2 does not match expert count",
                               (unsigned long long)tensor->stable_index);
                break;
            }
            uint64_t row_bytes = 0;
            uint64_t parent_bytes = 0;
            if (!mul_u64(tensor->dim[0] / tensor->block_elems,
                         tensor->block_bytes,
                         &row_bytes) ||
                !mul_u64(tensor->dim[1], row_bytes, &expert_bytes) ||
                !mul_u64(tensor->dim[2], expert_bytes, &parent_bytes)) {
                ok = set_error(err, errlen,
                               "routed tensor %llu size overflow",
                               (unsigned long long)tensor->stable_index);
                break;
            }
            if (tensor->source_bytes != parent_bytes ||
                tensor->source_bytes != expected_bytes) {
                ok = set_error(err, errlen,
                               "routed tensor %llu source size mismatch",
                               (unsigned long long)tensor->stable_index);
                break;
            }
            if (tensor->routed_layer < spec->first_routed_layer ||
                tensor->routed_layer >= spec->layer_count) {
                ok = set_error(err, errlen,
                               "routed tensor %llu has an out-of-range layer",
                               (unsigned long long)tensor->stable_index);
                break;
            }
            if (tensor->routed_projection <
                    DS4_LAGUNA_ROUTED_PROJECTION_GATE ||
                tensor->routed_projection >
                    DS4_LAGUNA_ROUTED_PROJECTION_DOWN) {
                ok = set_error(err, errlen,
                               "routed tensor %llu has an invalid projection",
                               (unsigned long long)tensor->stable_index);
                break;
            }
            const size_t role =
                ((size_t)(tensor->routed_layer - spec->first_routed_layer) *
                 3u) +
                ((size_t)tensor->routed_projection - 1u);
            if (roles[role]) {
                ok = set_error(err, errlen,
                               "duplicate routed projection for layer %u",
                               tensor->routed_layer);
                break;
            }
            roles[role] = tensor;
            if (common_expert_bytes == 0) {
                common_expert_bytes = expert_bytes;
            } else if (expert_bytes != common_expert_bytes) {
                ok = set_error(err, errlen,
                               "inconsistent routed projection size");
                break;
            }
            if (!add_u64(out->routed_source_bytes,
                         tensor->source_bytes,
                         &out->routed_source_bytes)) {
                ok = set_error(err, errlen, "routed source byte overflow");
                break;
            }
            out->routed_parent_count++;
        }

        out->tensor_ranges[i].stable_index = tensor->stable_index;
        out->tensor_ranges[i].tensor_class = tensor->tensor_class;
        out->tensor_ranges[i].source_offset = tensor->source_offset;
        out->tensor_ranges[i].source_bytes = tensor->source_bytes;
        source_order[i].tensor = tensor;
    }

    for (size_t role = 0; ok && role < (size_t)role_count_u64; role++) {
        if (!roles[role]) {
            ok = set_error(err, errlen,
                           "missing routed projection for layer %u",
                           spec->first_routed_layer + (uint32_t)(role / 3u));
        }
    }

    if (ok) {
        out->routed_projection_expert_bytes = common_expert_bytes;
        size_t entry_index = 0;
        for (uint32_t layer = spec->first_routed_layer;
             ok && layer < spec->layer_count;
             layer++) {
            const size_t role_base =
                (size_t)(layer - spec->first_routed_layer) * 3u;
            const ds4_laguna_tensor_desc *gate = roles[role_base];
            const ds4_laguna_tensor_desc *up = roles[role_base + 1u];
            const ds4_laguna_tensor_desc *down = roles[role_base + 2u];
            for (uint32_t expert = 0;
                 ok && expert < spec->expert_count;
                 expert++, entry_index++) {
                ds4_laguna_expert_entry *entry =
                    &out->expert_entries[entry_index];
                entry->layer = layer;
                entry->expert = expert;
                const ds4_laguna_tensor_desc *parents[3] = {
                    gate, up, down,
                };
                ds4_laguna_expert_view *views[3] = {
                    &entry->gate, &entry->up, &entry->down,
                };
                uint64_t device_cursor = 0;
                for (size_t projection = 0; projection < 3; projection++) {
                    uint64_t expert_delta = 0;
                    if (!mul_u64(expert, common_expert_bytes,
                                 &expert_delta) ||
                        !add_u64(parents[projection]->source_offset,
                                 expert_delta,
                                 &views[projection]->source_offset)) {
                        ok = set_error(err, errlen,
                                       "expert source view overflow");
                        break;
                    }
                    views[projection]->parent_stable_index =
                        parents[projection]->stable_index;
                    views[projection]->source_bytes = common_expert_bytes;
                    views[projection]->device_offset = device_cursor;
                    if (!add_u64(device_cursor, common_expert_bytes,
                                 &device_cursor) ||
                        !align_u64(device_cursor, spec->device_alignment,
                                   &device_cursor)) {
                        ok = set_error(err, errlen,
                                       "expert slot alignment overflow");
                        break;
                    }
                }
                entry->used_bytes = device_cursor;
                if (entry->used_bytes > out->slot_stride_bytes) {
                    out->slot_stride_bytes = entry->used_bytes;
                }
            }
        }
    }

    if (ok) {
        qsort(source_order, n_tensors, sizeof(source_order[0]),
              compare_tensor_ref_source);
        ok = append_source_range(out, source_capacity,
                                 DS4_LAGUNA_SOURCE_HEADER,
                                 0, spec->header_end,
                                 err, errlen) &&
             append_source_range(out, source_capacity,
                                 DS4_LAGUNA_SOURCE_METADATA,
                                 spec->header_end,
                                 spec->metadata_end - spec->header_end,
                                 err, errlen) &&
             append_source_range(out, source_capacity,
                                 DS4_LAGUNA_SOURCE_TENSOR_DIRECTORY,
                                 spec->metadata_end,
                                 spec->tensor_directory_end - spec->metadata_end,
                                 err, errlen) &&
             append_source_range(out, source_capacity,
                                 DS4_LAGUNA_SOURCE_ALIGNMENT_PADDING,
                                 spec->tensor_directory_end,
                                 spec->tensor_data_start -
                                     spec->tensor_directory_end,
                                 err, errlen);
        uint64_t cursor = spec->tensor_data_start;
        for (size_t i = 0; ok && i < n_tensors; i++) {
            const ds4_laguna_tensor_desc *tensor = source_order[i].tensor;
            if (tensor->source_offset < cursor) {
                ok = set_error(err, errlen,
                               "tensor parent source ranges overlap");
                break;
            }
            ok = append_source_range(out, source_capacity,
                                     DS4_LAGUNA_SOURCE_TENSOR_PADDING,
                                     cursor,
                                     tensor->source_offset - cursor,
                                     err, errlen);
            if (!ok || !add_u64(tensor->source_offset,
                                tensor->source_bytes, &cursor)) {
                if (ok) ok = set_error(err, errlen,
                                       "tensor parent end overflow");
                break;
            }
        }
        if (ok) {
            ok = append_source_range(out, source_capacity,
                                     DS4_LAGUNA_SOURCE_TENSOR_PADDING,
                                     cursor, spec->file_size - cursor,
                                     err, errlen);
        }
    }

    if (ok) {
        uint64_t tensor_source_bytes = 0;
        uint64_t reconciled = 0;
        ok = add_u64(out->static_source_bytes,
                     out->routed_source_bytes,
                     &tensor_source_bytes) &&
             add_u64(tensor_source_bytes,
                     out->non_tensor_source_bytes,
                     &reconciled);
        if (!ok || reconciled != spec->file_size) {
            ok = set_error(err, errlen,
                           "full-file source range reconciliation failed");
        }
    }

    free(source_order);
    free(roles);
    if (!ok) ds4_laguna_ledger_free(out);
    return ok;
}
