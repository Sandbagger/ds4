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

static void allocation_callsite(ds4_laguna_allocation_plan *plan,
                                size_t index,
                                uint32_t id,
                                const char *name,
                                ds4_runtime_category category,
                                ds4_runtime_physical_domain domain,
                                uint64_t bound_bytes) {
    ds4_runtime_callsite *site = &plan->callsites[index];
    site->id = id;
    site->name = name;
    site->category = category;
    site->domain = domain;
    site->bound_bytes = bound_bytes;
}

bool ds4_laguna_allocation_plan_make(
        ds4_laguna_allocation_plan *out,
        const ds4_laguna_ledger *ledger,
        const ds4_laguna_allocation_plan_spec *spec,
        char *err,
        size_t errlen) {
    if (err && errlen != 0) err[0] = '\0';
    if (!out || !ledger || !spec) {
        return set_error(err, errlen, "allocation plan argument is null");
    }
    memset(out, 0, sizeof(*out));

    const uint64_t gib = UINT64_C(1024) * 1024u * 1024u;
    const uint64_t mib = UINT64_C(1024) * 1024u;
    const uint64_t expected_file_size = UINT64_C(68248759648);
    const uint64_t expected_static_bytes = UINT64_C(4374164480);
    const uint64_t expected_slot_stride = UINT64_C(5308416);
    const uint64_t expected_expert_entries = UINT64_C(12032);
    if (spec->context_tokens != 32768u || spec->prefill_rows != 4096u ||
        spec->session_count != 1u) {
        return set_error(err, errlen,
                         "allocation plan requires the 32K/4K/one-session profile");
    }
    if (spec->configured_cache_bytes != 8u * gib &&
        spec->configured_cache_bytes != 12u * gib &&
        spec->configured_cache_bytes != 16u * gib) {
        return set_error(err, errlen,
                         "allocation plan cache must be exactly 8, 12, or 16 GiB");
    }
    if (ledger->slot_stride_bytes == 0 ||
        spec->configured_cache_bytes / ledger->slot_stride_bytes == 0) {
        return set_error(err, errlen,
                         "allocation plan cache contains no complete slot");
    }
    if (ledger->file_size != expected_file_size ||
        ledger->tensor_count != UINT64_C(814) ||
        ledger->static_parent_count != UINT64_C(673) ||
        ledger->routed_parent_count != UINT64_C(141) ||
        ledger->static_aligned_device_bytes != expected_static_bytes ||
        ledger->expert_entry_count != expected_expert_entries ||
        ledger->slot_stride_bytes != expected_slot_stride) {
        return set_error(err, errlen,
                         "allocation plan requires the exact Laguna ledger");
    }

    const uint64_t slots = spec->configured_cache_bytes /
                           ledger->slot_stride_bytes;
    if (slots == 0 || slots > UINT32_MAX) {
        return set_error(err, errlen,
                         "allocation plan cache contains no valid slot count");
    }
    uint64_t cache_payload = 0;
    if (!mul_u64(slots, ledger->slot_stride_bytes, &cache_payload) ||
        cache_payload > spec->configured_cache_bytes) {
        return set_error(err, errlen, "allocation plan cache payload overflow");
    }

    const uint64_t ledger_arrays = UINT64_C(1412824);
    const uint64_t route_hotness = UINT64_C(96256);
    const uint64_t host_entry_to_slot = UINT64_C(48128);
    const uint64_t device_entry_to_slot = UINT64_C(48128);
    const uint64_t static_offsets = UINT64_C(6512);
    uint64_t slot_state = 0;
    uint64_t metadata = 0;
    if (!mul_u64(slots, UINT64_C(32), &slot_state) ||
        !add_u64(ledger_arrays, route_hotness, &metadata) ||
        !add_u64(metadata, host_entry_to_slot, &metadata) ||
        !add_u64(metadata, device_entry_to_slot, &metadata) ||
        !add_u64(metadata, static_offsets, &metadata) ||
        !add_u64(metadata, slot_state, &metadata)) {
        return set_error(err, errlen,
                         "allocation plan metadata arithmetic overflow");
    }

    uint64_t kv_tokens = 0;
    uint64_t kv_bytes = 0;
    if (!mul_u64(UINT64_C(12), spec->context_tokens, &kv_tokens) ||
        !add_u64(kv_tokens, UINT64_C(36) * 512u, &kv_tokens) ||
        !mul_u64(kv_tokens, UINT64_C(4096), &kv_bytes)) {
        return set_error(err, errlen, "allocation plan KV arithmetic overflow");
    }
    uint64_t graph_bytes = 0;
    if (!mul_u64(spec->prefill_rows, UINT64_C(375156), &graph_bytes) ||
        !add_u64(graph_bytes, UINT64_C(413704), &graph_bytes)) {
        return set_error(err, errlen,
                         "allocation plan graph arithmetic overflow");
    }
    uint64_t staging_bytes = 0;
    if (!mul_u64(UINT64_C(4), ledger->slot_stride_bytes, &staging_bytes)) {
        return set_error(err, errlen,
                         "allocation plan staging arithmetic overflow");
    }

    out->configured_cache_bytes = spec->configured_cache_bytes;
    out->effective_cache_limit_bytes = spec->configured_cache_bytes;
    out->slot_stride_bytes = ledger->slot_stride_bytes;
    out->cache_payload_bytes = cache_payload;
    out->cache_tail_uncharged_bytes =
        spec->configured_cache_bytes - cache_payload;
    out->slot_count = (uint32_t)slots;
    out->staging_buffer_count = 4u;
    out->staging_buffer_bytes = ledger->slot_stride_bytes;
    out->owned_category_bounds[DS4_RUNTIME_CATEGORY_STATIC_WEIGHTS] =
        ledger->static_aligned_device_bytes;
    out->owned_category_bounds[
        DS4_RUNTIME_CATEGORY_EXPERT_CACHE_PAYLOAD] = cache_payload;
    out->owned_category_bounds[
        DS4_RUNTIME_CATEGORY_CACHE_METADATA_ADDRESS_TABLES] = metadata;
    out->owned_category_bounds[DS4_RUNTIME_CATEGORY_KV_STATE] = kv_bytes;
    out->owned_category_bounds[DS4_RUNTIME_CATEGORY_GRAPH_SCRATCH] =
        graph_bytes;
    out->owned_category_bounds[DS4_RUNTIME_CATEGORY_PINNED_STAGING] =
        staging_bytes;
    out->owned_category_bounds[DS4_RUNTIME_CATEGORY_OTHER_HOST] = gib;
    out->owned_category_bounds[DS4_RUNTIME_CATEGORY_OTHER_CUDA] = 2u * gib;

    out->report_bounds[DS4_RUNTIME_REPORT_MODEL_MAPPED_VIRTUAL] =
        ledger->file_size;
    out->report_bounds[
        DS4_RUNTIME_REPORT_MODEL_MAPPING_REGISTERED] = 0;
    out->report_bounds[DS4_RUNTIME_REPORT_MODEL_SOURCE_RESIDENT] = 2u * gib;
    out->report_bounds[
        DS4_RUNTIME_REPORT_HOST_LIBRARY_UNATTRIBUTED] = gib / 2u;
    out->report_bounds[
        DS4_RUNTIME_REPORT_CUDA_LIBRARY_UNATTRIBUTED] = gib / 2u;

    uint64_t owned_total = 0;
    uint64_t owned_non_cache = 0;
    for (size_t i = 0; i < DS4_RUNTIME_OWNED_CATEGORY_COUNT; i++) {
        if (!add_u64(owned_total, out->owned_category_bounds[i],
                     &owned_total)) {
            return set_error(err, errlen,
                             "allocation plan owned total overflow");
        }
        if (i != DS4_RUNTIME_CATEGORY_EXPERT_CACHE_PAYLOAD &&
            !add_u64(owned_non_cache, out->owned_category_bounds[i],
                     &owned_non_cache)) {
            return set_error(err, errlen,
                             "allocation plan non-cache total overflow");
        }
    }
    uint64_t external = 0;
    if (!add_u64(out->report_bounds[
                     DS4_RUNTIME_REPORT_MODEL_SOURCE_RESIDENT],
                 out->report_bounds[
                     DS4_RUNTIME_REPORT_HOST_LIBRARY_UNATTRIBUTED],
                 &external) ||
        !add_u64(external,
                 out->report_bounds[
                     DS4_RUNTIME_REPORT_CUDA_LIBRARY_UNATTRIBUTED],
                 &external) ||
        !add_u64(owned_non_cache, external,
                 &out->qualification_non_cache_bound_bytes) ||
        !add_u64(owned_total, external,
                 &out->planned_qualification_bytes) ||
        !add_u64(spec->configured_cache_bytes, 16u * gib,
                 &out->qualification_total_bound_bytes)) {
        return set_error(err, errlen,
                         "allocation plan qualification arithmetic overflow");
    }
    out->owned_non_cache_bound_bytes = owned_non_cache;
    out->owned_total_bound_bytes = owned_total;
    if (out->qualification_non_cache_bound_bytes > 16u * gib ||
        out->planned_qualification_bytes >
            out->qualification_total_bound_bytes) {
        return set_error(err, errlen,
                         "allocation plan exceeds qualification total bound");
    }

    allocation_callsite(out, 0, DS4_LAGUNA_CALLSITE_STATIC_SLAB,
        "laguna.static_slab", DS4_RUNTIME_CATEGORY_STATIC_WEIGHTS,
        DS4_RUNTIME_DOMAIN_CUDA_DEVICE, expected_static_bytes);
    allocation_callsite(out, 1, DS4_LAGUNA_CALLSITE_EXPERT_CACHE_PAYLOAD,
        "laguna.expert_cache_payload",
        DS4_RUNTIME_CATEGORY_EXPERT_CACHE_PAYLOAD,
        DS4_RUNTIME_DOMAIN_CUDA_DEVICE, cache_payload);
    allocation_callsite(out, 2, DS4_LAGUNA_CALLSITE_LEDGER_ARRAYS,
        "laguna.ledger_arrays",
        DS4_RUNTIME_CATEGORY_CACHE_METADATA_ADDRESS_TABLES,
        DS4_RUNTIME_DOMAIN_HOST, ledger_arrays);
    allocation_callsite(out, 3, DS4_LAGUNA_CALLSITE_ROUTE_HOTNESS,
        "laguna.route_hotness",
        DS4_RUNTIME_CATEGORY_CACHE_METADATA_ADDRESS_TABLES,
        DS4_RUNTIME_DOMAIN_HOST, route_hotness);
    allocation_callsite(out, 4, DS4_LAGUNA_CALLSITE_HOST_ENTRY_TO_SLOT,
        "laguna.host_entry_to_slot",
        DS4_RUNTIME_CATEGORY_CACHE_METADATA_ADDRESS_TABLES,
        DS4_RUNTIME_DOMAIN_HOST, host_entry_to_slot);
    allocation_callsite(out, 5, DS4_LAGUNA_CALLSITE_DEVICE_ENTRY_TO_SLOT,
        "laguna.device_entry_to_slot",
        DS4_RUNTIME_CATEGORY_CACHE_METADATA_ADDRESS_TABLES,
        DS4_RUNTIME_DOMAIN_CUDA_DEVICE, device_entry_to_slot);
    allocation_callsite(out, 6, DS4_LAGUNA_CALLSITE_STATIC_OFFSETS,
        "laguna.static_offsets",
        DS4_RUNTIME_CATEGORY_CACHE_METADATA_ADDRESS_TABLES,
        DS4_RUNTIME_DOMAIN_HOST, static_offsets);
    allocation_callsite(out, 7, DS4_LAGUNA_CALLSITE_SLOT_STATE,
        "laguna.slot_state",
        DS4_RUNTIME_CATEGORY_CACHE_METADATA_ADDRESS_TABLES,
        DS4_RUNTIME_DOMAIN_HOST, slot_state);
    allocation_callsite(out, 8, DS4_LAGUNA_CALLSITE_KV_STATE,
        "laguna.kv_state", DS4_RUNTIME_CATEGORY_KV_STATE,
        DS4_RUNTIME_DOMAIN_CUDA_DEVICE, kv_bytes);
    allocation_callsite(out, 9, DS4_LAGUNA_CALLSITE_GRAPH_SCRATCH,
        "laguna.graph_scratch", DS4_RUNTIME_CATEGORY_GRAPH_SCRATCH,
        DS4_RUNTIME_DOMAIN_CUDA_DEVICE, graph_bytes);
    allocation_callsite(out, 10, DS4_LAGUNA_CALLSITE_PINNED_STAGING_0,
        "laguna.pinned_staging.0", DS4_RUNTIME_CATEGORY_PINNED_STAGING,
        DS4_RUNTIME_DOMAIN_HOST, ledger->slot_stride_bytes);
    allocation_callsite(out, 11, DS4_LAGUNA_CALLSITE_PINNED_STAGING_1,
        "laguna.pinned_staging.1", DS4_RUNTIME_CATEGORY_PINNED_STAGING,
        DS4_RUNTIME_DOMAIN_HOST, ledger->slot_stride_bytes);
    allocation_callsite(out, 12, DS4_LAGUNA_CALLSITE_PINNED_STAGING_2,
        "laguna.pinned_staging.2", DS4_RUNTIME_CATEGORY_PINNED_STAGING,
        DS4_RUNTIME_DOMAIN_HOST, ledger->slot_stride_bytes);
    allocation_callsite(out, 13, DS4_LAGUNA_CALLSITE_PINNED_STAGING_3,
        "laguna.pinned_staging.3", DS4_RUNTIME_CATEGORY_PINNED_STAGING,
        DS4_RUNTIME_DOMAIN_HOST, ledger->slot_stride_bytes);
    allocation_callsite(out, 14, DS4_LAGUNA_CALLSITE_OTHER_HOST_ENGINE,
        "laguna.other_host.engine", DS4_RUNTIME_CATEGORY_OTHER_HOST,
        DS4_RUNTIME_DOMAIN_HOST, 64u * mib);
    allocation_callsite(out, 15, DS4_LAGUNA_CALLSITE_OTHER_HOST_MODEL,
        "laguna.other_host.model", DS4_RUNTIME_CATEGORY_OTHER_HOST,
        DS4_RUNTIME_DOMAIN_HOST, 192u * mib);
    allocation_callsite(out, 16, DS4_LAGUNA_CALLSITE_OTHER_HOST_BOOTSTRAP,
        "laguna.other_host.bootstrap", DS4_RUNTIME_CATEGORY_OTHER_HOST,
        DS4_RUNTIME_DOMAIN_HOST, 128u * mib);
    allocation_callsite(out, 17, DS4_LAGUNA_CALLSITE_OTHER_HOST_VOCAB,
        "laguna.other_host.vocab", DS4_RUNTIME_CATEGORY_OTHER_HOST,
        DS4_RUNTIME_DOMAIN_HOST, 256u * mib);
    allocation_callsite(out, 18, DS4_LAGUNA_CALLSITE_OTHER_HOST_SESSION,
        "laguna.other_host.session", DS4_RUNTIME_CATEGORY_OTHER_HOST,
        DS4_RUNTIME_DOMAIN_HOST, 128u * mib);
    allocation_callsite(out, 19, DS4_LAGUNA_CALLSITE_OTHER_HOST_TRACKER,
        "laguna.other_host.tracker", DS4_RUNTIME_CATEGORY_OTHER_HOST,
        DS4_RUNTIME_DOMAIN_HOST, 128u * mib);
    allocation_callsite(out, 20, DS4_LAGUNA_CALLSITE_OTHER_HOST_SERIALIZER,
        "laguna.other_host.serializer", DS4_RUNTIME_CATEGORY_OTHER_HOST,
        DS4_RUNTIME_DOMAIN_HOST, 128u * mib);
    allocation_callsite(out, 21,
        DS4_LAGUNA_CALLSITE_OTHER_CUDA_KERNEL_TMP,
        "laguna.other_cuda.kernel_tmp", DS4_RUNTIME_CATEGORY_OTHER_CUDA,
        DS4_RUNTIME_DOMAIN_CUDA_DEVICE, 512u * mib);
    allocation_callsite(out, 22,
        DS4_LAGUNA_CALLSITE_OTHER_CUDA_ROUTED_WORKSPACE,
        "laguna.other_cuda.routed_workspace",
        DS4_RUNTIME_CATEGORY_OTHER_CUDA,
        DS4_RUNTIME_DOMAIN_CUDA_DEVICE, 768u * mib);
    allocation_callsite(out, 23,
        DS4_LAGUNA_CALLSITE_OTHER_CUDA_DESCRIPTOR_UPLOAD,
        "laguna.other_cuda.descriptor_upload",
        DS4_RUNTIME_CATEGORY_OTHER_CUDA,
        DS4_RUNTIME_DOMAIN_CUDA_DEVICE, 256u * mib);
    allocation_callsite(out, 24,
        DS4_LAGUNA_CALLSITE_OTHER_CUDA_TRANSIENT,
        "laguna.other_cuda.transient", DS4_RUNTIME_CATEGORY_OTHER_CUDA,
        DS4_RUNTIME_DOMAIN_CUDA_DEVICE, 512u * mib);
    out->callsite_count = DS4_LAGUNA_ALLOCATION_CALLSITE_COUNT;
    return true;
}

static int compare_page_range(const void *lhs, const void *rhs) {
    const ds4_laguna_page_range *a = lhs;
    const ds4_laguna_page_range *b = rhs;
    if (a->offset < b->offset) return -1;
    if (a->offset > b->offset) return 1;
    if (a->bytes < b->bytes) return -1;
    if (a->bytes > b->bytes) return 1;
    return 0;
}

bool ds4_laguna_full_page_union(
        const ds4_laguna_page_range *input,
        size_t input_count,
        uint64_t page_size,
        ds4_laguna_page_range *output,
        size_t output_capacity,
        size_t *output_count,
        uint64_t *output_bytes) {
    if (output_count) *output_count = 0;
    if (output_bytes) *output_bytes = 0;
    if (!output_count || !output_bytes || !is_power_of_two(page_size) ||
        (input_count != 0 && !input) ||
        (output_capacity != 0 && !output)) {
        return false;
    }

    size_t rounded_count = 0;
    for (size_t i = 0; i < input_count; i++) {
        if (input[i].bytes == 0) continue;
        uint64_t raw_end = 0;
        if (!add_u64(input[i].offset, input[i].bytes, &raw_end)) {
            return false;
        }
        uint64_t start_pages = input[i].offset / page_size;
        if (input[i].offset % page_size != 0) {
            if (start_pages == UINT64_MAX) return false;
            start_pages++;
        }
        if (start_pages > UINT64_MAX / page_size) return false;
        const uint64_t safe_start = start_pages * page_size;
        const uint64_t safe_end = (raw_end / page_size) * page_size;
        if (safe_start >= safe_end) continue;
        if (rounded_count >= output_capacity) {
            *output_count = 0;
            *output_bytes = 0;
            return false;
        }
        output[rounded_count].offset = safe_start;
        output[rounded_count].bytes = safe_end - safe_start;
        rounded_count++;
    }
    if (rounded_count == 0) return true;

    qsort(output, rounded_count, sizeof(output[0]), compare_page_range);
    size_t merged_count = 1;
    for (size_t i = 1; i < rounded_count; i++) {
        ds4_laguna_page_range *previous = &output[merged_count - 1u];
        uint64_t previous_end = 0;
        uint64_t current_end = 0;
        if (!add_u64(previous->offset, previous->bytes, &previous_end) ||
            !add_u64(output[i].offset, output[i].bytes, &current_end)) {
            *output_count = 0;
            *output_bytes = 0;
            return false;
        }
        if (output[i].offset <= previous_end) {
            if (current_end > previous_end) {
                previous->bytes = current_end - previous->offset;
            }
        } else {
            output[merged_count++] = output[i];
        }
    }

    uint64_t total = 0;
    for (size_t i = 0; i < merged_count; i++) {
        if (!add_u64(total, output[i].bytes, &total)) {
            *output_count = 0;
            *output_bytes = 0;
            return false;
        }
    }
    *output_count = merged_count;
    *output_bytes = total;
    return true;
}
