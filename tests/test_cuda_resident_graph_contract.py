#!/usr/bin/env python3
'''Host contract for the root-owned Laguna resident graph allocation.

The graph TU is C and contains the marked graph implementation extracted from
``ds4.c``. The observer and resident tensor TUs are C++ and are imported from
the existing fixtures; their implementation text is never copied here. A
small fake CUDA driver provides synthetic device handles and real, tiny host
descriptors. ``ds4_runtime.c`` is compiled as its own C TU.

This file is deliberately a RED-first contract. It does not select resident
mode in production and it never runs the whole DS4 build.
'''
from __future__ import annotations

import re
import subprocess
import tempfile
import unittest
from pathlib import Path

try:
    from test_cuda_resident_ownership_contract import (
        FAKE_PREFIX,
        RUNTIME_HEADER,
        RUNTIME_SOURCE,
        ROOT,
        OBSERVER_BLOCK,
        extract_definition,
    )
    from test_cuda_resident_tensor_contract import (
        GENERIC_SOURCE,
        TENSOR_BLOCK,
        TENSOR_FAKE_PREFIX,
        TENSOR_SUPPORT,
        TMP_DECLS,
        extract_marked_block,
    )
except ModuleNotFoundError:
    from tests.test_cuda_resident_ownership_contract import (
        FAKE_PREFIX,
        RUNTIME_HEADER,
        RUNTIME_SOURCE,
        ROOT,
        OBSERVER_BLOCK,
        extract_definition,
    )
    from tests.test_cuda_resident_tensor_contract import (
        GENERIC_SOURCE,
        TENSOR_BLOCK,
        TENSOR_FAKE_PREFIX,
        TENSOR_SUPPORT,
        TMP_DECLS,
        extract_marked_block,
    )

DS4_SOURCE = (ROOT / "ds4.c").read_text(encoding="utf-8")
GPU_RESIDENT_HEADER = (ROOT / "ds4_gpu_resident.h").read_text(encoding="utf-8")
CUDA_SOURCE = (ROOT / "ds4_cuda.cu").read_text(encoding="utf-8")
MAKEFILE_SOURCE = (ROOT / "Makefile").read_text(encoding="utf-8")

GRAPH_START_MARKER = "/* Laguna graph ownership and allocation. */"
GRAPH_END_MARKER = "/* End Laguna graph ownership and allocation. */"
GRAPH_BLOCK = extract_marked_block(DS4_SOURCE, GRAPH_START_MARKER, GRAPH_END_MARKER)


def source_definition(source: str, signature: str) -> str | None:
    '''Use the existing brace-aware source extractor for a real definition.'''
    return extract_definition(source, signature)


GRAPH_SOURCE_SEAMS = {
    "graph block": GRAPH_BLOCK,
    "checked free": source_definition(DS4_SOURCE, "static bool laguna_graph_free("),
    "graph alloc": source_definition(DS4_SOURCE, "static bool laguna_graph_alloc("),
}

# The C graph fixture uses the actual graph block and only an independently
# labelled frozen canonical planner. It does not reproduce observer or tensor
# ownership code. These declarations are the narrow seam needed by the
# extracted block; all resident physical ownership is supplied by the C++ TU.
GRAPH_C_PREFIX = r'''
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ds4_runtime.h"
#include "ds4_gpu_resident.h"
#include "ds4_laguna_resident.h"

/* Public/platform headers observe the real target first. The normal generated
 * graph TU selects its CUDA branch only after that boundary; Apple/ROCm
 * refusal probes intentionally omit this selector. */
#if defined(DS4_GRAPH_FIXTURE_CUDA_BRANCH)
# if defined(__APPLE__)
#  undef __APPLE__
# endif
# if defined(DS4_ROCM_BUILD)
#  undef DS4_ROCM_BUILD
# endif
#elif defined(DS4_GRAPH_FIXTURE_APPLE_BRANCH)
# if !defined(__APPLE__)
#  define __APPLE__ 1
# endif
# if defined(DS4_ROCM_BUILD)
#  undef DS4_ROCM_BUILD
# endif
#elif defined(DS4_GRAPH_FIXTURE_ROCM_BRANCH)
# if defined(__APPLE__)
#  undef __APPLE__
# endif
# if !defined(DS4_ROCM_BUILD)
#  define DS4_ROCM_BUILD 1
# endif
#endif

/* Independent frozen canonical plan for this host fixture only. It is not
 * the production planner and is intentionally limited to 32K/4K Laguna S21.
 * The graph block remains the source of truth for every allocation expression.
 */
enum {
    DS4_MAX_LAYER = 79,
    DS4_N_LAYER = 48,
    DS4_N_EMBD = 3072,
    DS4_N_VOCAB = 100352,
    DS4_N_HEAD = 72,
    DS4_N_HEAD_KV = 8,
    DS4_N_HEAD_DIM = 128,
    DS4_N_SWA = 512,
    DS4_N_EXPERT = 256,
    DS4_N_EXPERT_USED = 10,
    DS4_N_FF_EXP = 1024,
    DS4_N_FF_DENSE = 12288,
    DS4_CONTEXT_LENGTH = 262144,
    DS4_MODEL_FAMILY_LAGUNA = 2,
};
#define DS4_MODEL_FAMILY DS4_MODEL_FAMILY_LAGUNA

typedef struct {
    uint64_t total_bytes;
    uint64_t raw_bytes;
    uint64_t compressed_bytes;
    uint64_t scratch_bytes;
    uint32_t prefill_cap;
    uint32_t raw_cap;
    uint32_t comp_cap;
} ds4_context_memory;

/* Validated Laguna S21: full attention at 0,4,...,44; SWA elsewhere. */
static bool ds4_laguna_layer_is_swa(uint32_t layer) {
    return (layer % 4u) != 0u;
}

static bool ds4_laguna_prefill_memory_plan(
        uint32_t context_tokens, uint32_t prefill_rows,
        ds4_context_memory *out) {
    if (!out || context_tokens != 32768u || prefill_rows != 4096u) return false;
    ds4_context_memory plan = {0};
    plan.prefill_cap = prefill_rows;
    plan.raw_cap = context_tokens;
    plan.comp_cap = 512u;
    const uint64_t kv_row_bytes =
        2u * (uint64_t)DS4_N_HEAD_KV * DS4_N_HEAD_DIM * sizeof(uint16_t);
    for (uint32_t layer = 0; layer < DS4_N_LAYER; ++layer) {
        const uint32_t cap = ds4_laguna_layer_is_swa(layer) ? 512u : context_tokens;
        plan.raw_bytes += (uint64_t)cap * kv_row_bytes;
    }
    plan.scratch_bytes =
        (uint64_t)prefill_rows * UINT64_C(375156) + UINT64_C(413704);
    plan.total_bytes = plan.raw_bytes + plan.scratch_bytes;
    *out = plan;
    return true;
}

/* Fixture forwarding seams. The C graph retains the actual control flow;
 * these names only count and safely bound fake descriptor access at the C/C++
 * seam. They are not native CUDA implementations. */
extern ds4_gpu_tensor *graph_fixture_legacy_tensor_alloc(uint64_t bytes);
extern void graph_fixture_legacy_tensor_free(ds4_gpu_tensor *tensor);
extern uint64_t graph_fixture_legacy_tensor_bytes(const ds4_gpu_tensor *tensor);
extern void *graph_fixture_legacy_tensor_contents(ds4_gpu_tensor *tensor);
extern int graph_fixture_fake_tensor_write(ds4_gpu_tensor *tensor,
                                            uint64_t offset,
                                            const void *data, uint64_t bytes);
extern int graph_fixture_resident_tensor_alloc(
    ds4_runtime_tracker *tracker, uint32_t callsite_id, uint64_t bytes,
    ds4_gpu_laguna_resident_tensor_owner *out);
extern int graph_fixture_resident_tensor_free(
    ds4_runtime_tracker *tracker, ds4_gpu_laguna_resident_tensor_owner *owner);

#define ds4_gpu_tensor_alloc graph_fixture_legacy_tensor_alloc
#define ds4_gpu_tensor_free graph_fixture_legacy_tensor_free
#define ds4_gpu_tensor_bytes graph_fixture_legacy_tensor_bytes
#define ds4_gpu_tensor_contents graph_fixture_legacy_tensor_contents
#define ds4_gpu_tensor_write graph_fixture_fake_tensor_write
#define ds4_gpu_laguna_resident_tensor_alloc graph_fixture_resident_tensor_alloc
#define ds4_gpu_laguna_resident_tensor_free graph_fixture_resident_tensor_free
'''

GRAPH_C_SUFFIX = r'''

/* Limit C-to-C++ forwarding instrumentation to the extracted graph block. */
#undef ds4_gpu_tensor_alloc
#undef ds4_gpu_tensor_free
#undef ds4_gpu_tensor_bytes
#undef ds4_gpu_tensor_contents
#undef ds4_gpu_tensor_write
#undef ds4_gpu_laguna_resident_tensor_alloc
#undef ds4_gpu_laguna_resident_tensor_free

typedef struct {
    uint32_t runtime_owner_mode;
    uint32_t resident_owner_count;
    uint32_t runtime_record_count;
    uint32_t alias_count;
    uint32_t alias_nonnull;
    uint32_t owner_tensor_nonnull;
    uint32_t owner_slot_nonnull;
    uint32_t owner_slots_in_region;
    uint32_t owner_slots_foreign;
    uint32_t owner_exact_bindings;
    uint32_t owner_live_entries;
    uint32_t record_namespace_52;
    uint32_t record_namespace_4f;
    uint32_t live_namespace_52;
    uint32_t live_namespace_4f;
    uint64_t scratch_bytes;
    uint64_t kv_bytes;
    uint64_t first_record_id;
    uint64_t last_record_id;
} graph_fixture_snapshot;

static size_t graph_fixture_aliases(
        ds4_laguna_gpu_graph *g,
        ds4_gpu_tensor **aliases[DS4_LAGUNA_GRAPH_OWNER_CAPACITY]) {
    size_t count = 0;
#define DS4_GRAPH_ALIAS(name) aliases[count++] = &g->name
    DS4_GRAPH_ALIAS(tokens);
    DS4_GRAPH_ALIAS(cur);
    DS4_GRAPH_ALIAS(next);
    DS4_GRAPH_ALIAS(attn_norm);
    DS4_GRAPH_ALIAS(q);
    DS4_GRAPH_ALIAS(k);
    DS4_GRAPH_ALIAS(v);
    DS4_GRAPH_ALIAS(gate);
    DS4_GRAPH_ALIAS(heads);
    DS4_GRAPH_ALIAS(attn_out);
    DS4_GRAPH_ALIAS(after_attn);
    DS4_GRAPH_ALIAS(ffn_norm);
    DS4_GRAPH_ALIAS(ffn_gate);
    DS4_GRAPH_ALIAS(ffn_up);
    DS4_GRAPH_ALIAS(ffn_mid);
    DS4_GRAPH_ALIAS(ffn_out);
    DS4_GRAPH_ALIAS(shared_out);
    DS4_GRAPH_ALIAS(routed_mid);
    DS4_GRAPH_ALIAS(router_logits);
    DS4_GRAPH_ALIAS(router_probs);
    DS4_GRAPH_ALIAS(router_selected);
    DS4_GRAPH_ALIAS(router_weights);
    DS4_GRAPH_ALIAS(shared_selected);
    DS4_GRAPH_ALIAS(shared_weight);
    DS4_GRAPH_ALIAS(staged_key);
    DS4_GRAPH_ALIAS(staged_value);
    DS4_GRAPH_ALIAS(output_norm);
    DS4_GRAPH_ALIAS(logits);
#undef DS4_GRAPH_ALIAS
    for (uint32_t layer = 0; layer < DS4_N_LAYER; ++layer) {
        aliases[count++] = &g->key_cache[layer];
        aliases[count++] = &g->value_cache[layer];
    }
    return count;
}

int graph_fixture_plan(uint32_t context_tokens, uint32_t prefill_rows,
                       uint64_t *scratch_out, uint64_t *kv_out) {
    ds4_context_memory plan;
    if (!ds4_laguna_prefill_memory_plan(context_tokens, prefill_rows, &plan)) {
        return 0;
    }
    if (scratch_out) *scratch_out = plan.scratch_bytes;
    if (kv_out) *kv_out = plan.raw_bytes;
    return 1;
}

static int graph_fixture_delete_errors;

size_t graph_fixture_storage_size(void) { return sizeof(ds4_laguna_gpu_graph); }
void *graph_fixture_new_zero(void) {
    return calloc(1u, sizeof(ds4_laguna_gpu_graph));
}
void *graph_fixture_new_nonzero(void) {
    void *storage = malloc(sizeof(ds4_laguna_gpu_graph));
    if (storage) memset(storage, 0xa5, sizeof(ds4_laguna_gpu_graph));
    return storage;
}
int graph_fixture_is_zero(const void *storage) {
    if (!storage) return 1;
    const unsigned char *bytes = (const unsigned char *)storage;
    for (size_t i = 0; i < sizeof(ds4_laguna_gpu_graph); ++i) {
        if (bytes[i] != 0u) return 0;
    }
    return 1;
}
int graph_fixture_is_pattern(const void *storage, unsigned char value) {
    if (!storage) return 0;
    const unsigned char *bytes = (const unsigned char *)storage;
    for (size_t i = 0; i < sizeof(ds4_laguna_gpu_graph); ++i) {
        if (bytes[i] != value) return 0;
    }
    return 1;
}

/* Normal graph storage is caller-owned. Refuse to discard a live container;
 * each scenario must first complete checked graph cleanup. */
int graph_fixture_delete(void *storage) {
    if (!storage) return 1;
    if (!graph_fixture_is_zero(storage)) {
        ++graph_fixture_delete_errors;
        return 0;
    }
    free(storage);
    return 1;
}
int graph_fixture_delete_error_count(void) {
    return graph_fixture_delete_errors;
}

/* These discard helpers are only for known-unowned diagnostic buffers. */
int graph_fixture_discard_nonzero_pattern(void *storage) {
    if (!storage || !graph_fixture_is_pattern(storage, 0xa5)) return 0;
    memset(storage, 0, sizeof(ds4_laguna_gpu_graph));
    free(storage);
    return 1;
}

int graph_fixture_alloc(void *storage, uint32_t context_tokens,
                        uint32_t prefill_rows, int owner_mode,
                        ds4_runtime_tracker *tracker) {
    if (!storage) return 0;
    return laguna_graph_alloc(
        (ds4_laguna_gpu_graph *)storage, context_tokens, prefill_rows,
        tracker, (ds4_laguna_graph_owner_mode)owner_mode) ? 1 : 0;
}
int graph_fixture_owner_mode_legacy(void) {
    return (int)DS4_LAGUNA_GRAPH_OWNER_LEGACY;
}
int graph_fixture_owner_mode_resident(void) {
    return (int)DS4_LAGUNA_GRAPH_OWNER_RESIDENT;
}
int graph_fixture_free(void *storage) {
    if (!storage) return laguna_graph_free(NULL) ? 1 : 0;
    return laguna_graph_free((ds4_laguna_gpu_graph *)storage) ? 1 : 0;
}
int graph_fixture_replace_tracker(void *storage, ds4_runtime_tracker *tracker) {
    if (!storage) return 0;
    ((ds4_laguna_gpu_graph *)storage)->runtime_tracker = tracker;
    return 1;
}
void *graph_fixture_clone(const void *storage) {
    if (!storage) return NULL;
    void *copy = malloc(sizeof(ds4_laguna_gpu_graph));
    if (copy) memcpy(copy, storage, sizeof(ds4_laguna_gpu_graph));
    return copy;
}
int graph_fixture_discard_unowned_copy(void *copy) {
    if (!copy) return 1;
    /* The caller proved this is a byte-copied, non-owning diagnostic buffer. */
    memset(copy, 0, sizeof(ds4_laguna_gpu_graph));
    free(copy);
    return 1;
}
int graph_fixture_bind_foreign_slot(void *storage) {
    static ds4_gpu_tensor *foreign_slot;
    if (!storage) return 0;
    ds4_laguna_gpu_graph *g = (ds4_laguna_gpu_graph *)storage;
    if (g->resident_owner_count == 0u) return 0;
    g->resident_owners[0].slot = &foreign_slot;
    return 1;
}
int graph_fixture_restore_first_slot(void *storage) {
    if (!storage) return 0;
    ds4_laguna_gpu_graph *g = (ds4_laguna_gpu_graph *)storage;
    if (g->resident_owner_count == 0u) return 0;
    g->resident_owners[0].slot = &g->tokens;
    return 1;
}

int graph_fixture_snapshot_read(
        const void *storage, graph_fixture_snapshot *out,
        const ds4_runtime_tracker *tracker) {
    if (!out) return 0;
    memset(out, 0, sizeof(*out));
    if (!storage) return 1;
    const ds4_laguna_gpu_graph *const_graph =
        (const ds4_laguna_gpu_graph *)storage;
    ds4_laguna_gpu_graph *g = (ds4_laguna_gpu_graph *)const_graph;
    out->runtime_owner_mode = (uint32_t)g->runtime_owner_mode;
    out->resident_owner_count = g->resident_owner_count;
    out->runtime_record_count = g->runtime_record_count;
    out->scratch_bytes = g->scratch_bytes;
    out->kv_bytes = g->kv_bytes;

    ds4_gpu_tensor **aliases[DS4_LAGUNA_GRAPH_OWNER_CAPACITY];
    out->alias_count = (uint32_t)graph_fixture_aliases(g, aliases);
    for (uint32_t i = 0; i < out->alias_count; ++i) {
        if (*aliases[i]) out->alias_nonnull++;
    }

    const uintptr_t begin = (uintptr_t)&g->tokens;
    const uintptr_t end = (uintptr_t)&g->value_cache[DS4_MAX_LAYER - 1u] +
        sizeof(g->value_cache[0]);
    for (uint32_t i = 0; i < g->resident_owner_count &&
                         i < DS4_LAGUNA_GRAPH_OWNER_CAPACITY; ++i) {
        const ds4_laguna_graph_resident_owner *entry = &g->resident_owners[i];
        if (entry->owner.tensor) out->owner_tensor_nonnull++;
        if (entry->slot) {
            out->owner_slot_nonnull++;
            const uintptr_t slot = (uintptr_t)entry->slot;
            const bool aligned = (slot % sizeof(ds4_gpu_tensor *)) == 0u;
            if (aligned && slot >= begin && slot < end) {
                out->owner_slots_in_region++;
                if (*entry->slot == entry->owner.tensor) {
                    out->owner_exact_bindings++;
                }
            } else {
                out->owner_slots_foreign++;
            }
        }
        if (entry->owner.tensor) out->owner_live_entries++;
    }
    if (tracker) {
        if (tracker->record_count != 0u) {
            out->first_record_id = tracker->records[0].id;
            out->last_record_id = tracker->records[tracker->record_count - 1u].id;
        }
        for (size_t i = 0; i < tracker->record_count; ++i) {
            const uint8_t namespace_id = (uint8_t)(tracker->records[i].id >> 56);
            if (namespace_id == 0x52u) {
                out->record_namespace_52++;
                if (tracker->records[i].live) out->live_namespace_52++;
            }
            if (namespace_id == 0x4fu) {
                out->record_namespace_4f++;
                if (tracker->records[i].live) out->live_namespace_4f++;
            }
        }
    }
    return 1;
}

uint64_t graph_fixture_owner_id(const void *storage, uint32_t index,
                                int device_record) {
    if (!storage || index >= DS4_LAGUNA_GRAPH_OWNER_CAPACITY) return 0;
    const ds4_laguna_gpu_graph *g = (const ds4_laguna_gpu_graph *)storage;
    if (index >= g->resident_owner_count) return 0;
    return device_record ? g->resident_owners[index].owner.device_record_id
                         : g->resident_owners[index].owner.descriptor_record_id;
}

static const ds4_runtime_allocation_record *graph_fixture_record(
        const ds4_runtime_tracker *tracker, uint64_t id) {
    if (!tracker || id == 0u) return NULL;
    for (size_t i = 0; i < tracker->record_count; ++i) {
        if (tracker->records[i].id == id) return &tracker->records[i];
    }
    return NULL;
}

int graph_fixture_register_host_relation(
        ds4_runtime_tracker *tracker, const void *storage,
        uint64_t relation_id) {
    const uint64_t owner_id = graph_fixture_owner_id(storage, 0u, 0);
    const ds4_runtime_allocation_record *owner =
        graph_fixture_record(tracker, owner_id);
    if (!owner) return 0;
    return ds4_runtime_tracker_register(
        tracker, relation_id, owner->base, owner->requested_bytes, owner_id) ==
        DS4_RUNTIME_STATUS_OK ? 1 : 0;
}
int graph_fixture_unregister_relation(ds4_runtime_tracker *tracker,
                                      uint64_t relation_id) {
    if (!tracker) return 0;
    (void)ds4_runtime_tracker_unregister(tracker, relation_id);
    const ds4_runtime_allocation_record *record =
        graph_fixture_record(tracker, relation_id);
    return record && !record->live ? 1 : 0;
}

'''


# Deterministic nth-call failures are injected into the already shared fake
# driver. No CUDA or device address is ever dereferenced by these hooks.
GRAPH_FAIL_DECLS = r'''
static int graph_fail_device_malloc_at;
static int graph_partial_driver_failure_at;
static int graph_fail_device_free_at;
static int graph_fail_device_free_through;
static int graph_fail_write_at;
static int graph_write_calls;
static int graph_resident_alloc_calls;
static int graph_resident_free_calls;
static int graph_generic_alloc_calls;
static int graph_generic_free_calls;
static int graph_generic_nonnull_free_calls;
static int graph_generic_bytes_calls;
static int graph_generic_contents_calls;
static bool graph_capture_descriptor_alloc;
static void *graph_captured_descriptor_alloc;
static unsigned long long graph_descriptor_live_peak;
'''

# Make only the existing fake-driver seams deterministic for graph scenarios.
# The partial path still uses fake_partial_driver_failure's nonnull-on-error
# behavior; this extra selector limits that behavior to one requested attempt.
assert TENSOR_FAKE_PREFIX.count(
    "if (fake_fail_device_malloc) return kFakeCudaError;"
) == 1
_GRAPH_PARTIAL_DEVICE_RESULT = (
    "fake_live[*out] = {bytes, false};\n"
    "    return fake_partial_driver_failure ? kFakeCudaError : cudaSuccess;"
)
assert TENSOR_FAKE_PREFIX.count(_GRAPH_PARTIAL_DEVICE_RESULT) == 1
assert TENSOR_FAKE_PREFIX.count(
    "if (fake_fail_device_free || ptr == fake_fail_device_free_ptr) return kFakeCudaError;"
) == 1
GRAPH_CPP_FAKE = TENSOR_FAKE_PREFIX.replace(
    "if (fake_fail_device_malloc) return kFakeCudaError;",
    "if (fake_fail_device_malloc || (graph_fail_device_malloc_at > 0 && "
    "fake_device_malloc_calls == graph_fail_device_malloc_at)) "
    "return kFakeCudaError;",
).replace(
    _GRAPH_PARTIAL_DEVICE_RESULT,
    "fake_live[*out] = {bytes, false};\n"
    "    return (fake_partial_driver_failure || "
    "(graph_partial_driver_failure_at > 0 && "
    "fake_device_malloc_calls == graph_partial_driver_failure_at)) "
    "? kFakeCudaError : cudaSuccess;",
).replace(
    "if (fake_fail_device_free || ptr == fake_fail_device_free_ptr) return kFakeCudaError;",
    "if (fake_fail_device_free || ptr == fake_fail_device_free_ptr || "
    "(graph_fail_device_free_at > 0 && "
    "fake_device_free_calls == graph_fail_device_free_at) || "
    "(graph_fail_device_free_through > 0 && "
    "fake_device_free_calls <= graph_fail_device_free_through)) "
    "return kFakeCudaError;",
)

# Capture only the fresh descriptor returned by the shared fake_calloc seam.
# This is forwarding instrumentation, not a replacement observer/tensor body.
_GRAPH_TENSOR_SUPPORT_CALLOC_LINE = "if (ptr) fake_descriptor_live.insert(ptr);"
assert TENSOR_SUPPORT.count(_GRAPH_TENSOR_SUPPORT_CALLOC_LINE) == 1
GRAPH_TENSOR_SUPPORT = TENSOR_SUPPORT.replace(
    _GRAPH_TENSOR_SUPPORT_CALLOC_LINE,
    "if (ptr) {\n"
    "        fake_descriptor_live.insert(ptr);\n"
    "        if ((unsigned long long)fake_descriptor_live.size() > "
    "graph_descriptor_live_peak)\n"
    "            graph_descriptor_live_peak = "
    "(unsigned long long)fake_descriptor_live.size();\n"
    "        if (graph_capture_descriptor_alloc) {\n"
    "            if (graph_captured_descriptor_alloc) ++fake_api_errors;\n"
    "            graph_captured_descriptor_alloc = ptr;\n"
    "            fake_expected_descriptor = ptr;\n"
    "            fake_expected_device_id = 0u;\n"
    "        }\n"
    "    }",
)

GRAPH_CPP_WRAPPERS = r'''
/* C graph forwarding instrumentation. The actual resident APIs below remain
 * the imported observer/tensor implementation; these wrappers only arm the
 * existing fake-driver order sensors around one forwarded call. */
static void graph_fixture_sensor_disarm(void) {
    fake_check_descriptor_order = false;
    fake_expected_descriptor = nullptr;
    fake_expected_device_id = 0u;
    graph_capture_descriptor_alloc = false;
}

static int graph_fixture_descriptor_is_live(const ds4_gpu_tensor *tensor) {
    return tensor && fake_descriptor_live.count(
        const_cast<ds4_gpu_tensor *>(tensor)) == 1u;
}

extern "C" int graph_fixture_resident_tensor_alloc(
        ds4_runtime_tracker *tracker, uint32_t callsite_id, uint64_t bytes,
        ds4_gpu_laguna_resident_tensor_owner *out) {
    ++graph_resident_alloc_calls;
    graph_capture_descriptor_alloc = true;
    graph_captured_descriptor_alloc = nullptr;
    fake_check_descriptor_order = true;
    fake_expected_descriptor = nullptr;
    fake_expected_device_id = 0u;
    const int result = ds4_gpu_laguna_resident_tensor_alloc(
        tracker, callsite_id, bytes, out);
    graph_capture_descriptor_alloc = false;
    if (result) {
        if (!out || !out->tensor ||
            out->tensor != graph_captured_descriptor_alloc) ++fake_api_errors;
    } else if (out && (out->tensor || out->descriptor_record_id ||
                       out->device_record_id)) {
        ++fake_api_errors;
    }
    graph_fixture_sensor_disarm();
    return result;
}

extern "C" int graph_fixture_resident_tensor_free(
        ds4_runtime_tracker *tracker,
        ds4_gpu_laguna_resident_tensor_owner *owner) {
    ++graph_resident_free_calls;
    /* Read only the caller-owned handle fields. Never dereference tensor. */
    fake_expected_descriptor = owner ? owner->tensor : nullptr;
    fake_expected_device_id = owner ? owner->device_record_id : 0u;
    fake_check_descriptor_order = fake_expected_descriptor != nullptr;
    const int result = ds4_gpu_laguna_resident_tensor_free(tracker, owner);
    graph_fixture_sensor_disarm();
    return result;
}

/* Labeled fake I/O/accessor boundaries for unchanged LEGACY graph calls.
 * They validate descriptor membership before any descriptor field is read.
 * They are intentionally not CUDA-I/O evidence. */
extern "C" ds4_gpu_tensor *graph_fixture_legacy_tensor_alloc(uint64_t bytes) {
    ++graph_generic_alloc_calls;
    return ds4_gpu_tensor_alloc(bytes);
}
extern "C" void graph_fixture_legacy_tensor_free(ds4_gpu_tensor *tensor) {
    ++graph_generic_free_calls;
    if (!tensor) return;
    if (!graph_fixture_descriptor_is_live(tensor)) {
        ++fake_api_errors;
        return;
    }
    ++graph_generic_nonnull_free_calls;
    ds4_gpu_tensor_free(tensor);
}
extern "C" uint64_t graph_fixture_legacy_tensor_bytes(
        const ds4_gpu_tensor *tensor) {
    ++graph_generic_bytes_calls;
    if (!tensor) return 0u;
    if (!graph_fixture_descriptor_is_live(tensor)) {
        ++fake_api_errors;
        return 0u;
    }
    return tensor->bytes;
}
extern "C" void *graph_fixture_legacy_tensor_contents(ds4_gpu_tensor *tensor) {
    ++graph_generic_contents_calls;
    if (!tensor) return nullptr;
    if (!graph_fixture_descriptor_is_live(tensor)) {
        ++fake_api_errors;
        return nullptr;
    }
    return tensor->ptr;
}
'''

GRAPH_CPP_DECLS = r'''

extern "C" {
typedef struct graph_fixture_snapshot graph_fixture_snapshot;
size_t graph_fixture_storage_size(void);
void *graph_fixture_new_zero(void);
void *graph_fixture_new_nonzero(void);
int graph_fixture_delete(void *storage);
int graph_fixture_delete_error_count(void);
int graph_fixture_is_zero(const void *storage);
int graph_fixture_is_pattern(const void *storage, unsigned char value);
int graph_fixture_discard_nonzero_pattern(void *storage);
int graph_fixture_alloc(void *, uint32_t, uint32_t, int, ds4_runtime_tracker *);
int graph_fixture_owner_mode_legacy(void);
int graph_fixture_owner_mode_resident(void);
int graph_fixture_free(void *storage);
int graph_fixture_replace_tracker(void *, ds4_runtime_tracker *);
void *graph_fixture_clone(const void *storage);
int graph_fixture_discard_unowned_copy(void *copy);
int graph_fixture_bind_foreign_slot(void *storage);
int graph_fixture_restore_first_slot(void *storage);
int graph_fixture_snapshot_read(const void *, graph_fixture_snapshot *,
                                const ds4_runtime_tracker *);
uint64_t graph_fixture_owner_id(const void *, uint32_t, int);
int graph_fixture_plan(uint32_t, uint32_t, uint64_t *, uint64_t *);
int graph_fixture_register_host_relation(ds4_runtime_tracker *, const void *, uint64_t);
int graph_fixture_unregister_relation(ds4_runtime_tracker *, uint64_t);
}

struct graph_fixture_snapshot {
    uint32_t runtime_owner_mode;
    uint32_t resident_owner_count;
    uint32_t runtime_record_count;
    uint32_t alias_count;
    uint32_t alias_nonnull;
    uint32_t owner_tensor_nonnull;
    uint32_t owner_slot_nonnull;
    uint32_t owner_slots_in_region;
    uint32_t owner_slots_foreign;
    uint32_t owner_exact_bindings;
    uint32_t owner_live_entries;
    uint32_t record_namespace_52;
    uint32_t record_namespace_4f;
    uint32_t live_namespace_52;
    uint32_t live_namespace_4f;
    uint64_t scratch_bytes;
    uint64_t kv_bytes;
    uint64_t first_record_id;
    uint64_t last_record_id;
};

/* Fake write boundary for the graph C forwarding seam; it never dereferences
 * synthetic device storage and is counted separately from CUDA evidence. */
extern "C" int graph_fixture_fake_tensor_write(
        ds4_gpu_tensor *tensor, uint64_t offset, const void *data,
        uint64_t bytes) {
    (void)tensor;
    (void)offset;
    (void)data;
    (void)bytes;
    ++graph_write_calls;
    return graph_fail_write_at > 0 && graph_write_calls == graph_fail_write_at
        ? 0 : 1;
}
'''
GRAPH_CPP_SUFFIX = r'''
static ds4_runtime_tracker graph_tracker;
static ds4_runtime_tracker graph_foreign_tracker;
static ds4_runtime_allocation_record graph_records[512];
static ds4_runtime_allocation_record graph_foreign_records[512];
static ds4_runtime_callsite graph_callsites[6];
static ds4_runtime_tracker_config graph_tracker_config;
static uint8_t graph_tombstone_bytes[8];

static void graph_add_callsite(size_t *count, uint32_t id, const char *name,
                               ds4_runtime_category category,
                               ds4_runtime_physical_domain domain,
                               uint64_t bound) {
    graph_callsites[*count] = {id, name, category, domain, bound};
    ++*count;
}

static int graph_init_tracker(ds4_runtime_tracker *tracker,
                              ds4_runtime_allocation_record *records,
                              size_t capacity) {
    uint64_t scratch = 0;
    uint64_t kv = 0;
    const uint64_t host_bound = 124u * (uint64_t)sizeof(ds4_gpu_tensor);
    if (!tracker || !records || !graph_fixture_plan(32768u, 4096u, &scratch, &kv) ||
        capacity > 512u) return 0;
    std::memset(tracker, 0, sizeof(*tracker));
    std::memset(records, 0, sizeof(graph_records));
    std::memset(graph_callsites, 0, sizeof(graph_callsites));
    std::memset(&graph_tracker_config, 0, sizeof(graph_tracker_config));
    size_t count = 0;
    graph_add_callsite(&count, DS4_LAGUNA_CALLSITE_STATIC_SLAB,
                       "graph.static", DS4_RUNTIME_CATEGORY_STATIC_WEIGHTS,
                       DS4_RUNTIME_DOMAIN_CUDA_DEVICE, 64u);
    graph_add_callsite(&count, DS4_LAGUNA_CALLSITE_OTHER_CUDA_KERNEL_TMP,
                       "graph.other_cuda", DS4_RUNTIME_CATEGORY_OTHER_CUDA,
                       DS4_RUNTIME_DOMAIN_CUDA_DEVICE, 64u);
    graph_add_callsite(&count, DS4_LAGUNA_CALLSITE_PINNED_STAGING_0,
                       "graph.pinned", DS4_RUNTIME_CATEGORY_PINNED_STAGING,
                       DS4_RUNTIME_DOMAIN_HOST, 64u);
    graph_add_callsite(&count, DS4_LAGUNA_CALLSITE_KV_STATE,
                       "graph.kv", DS4_RUNTIME_CATEGORY_KV_STATE,
                       DS4_RUNTIME_DOMAIN_CUDA_DEVICE, kv);
    graph_add_callsite(&count, DS4_LAGUNA_CALLSITE_GRAPH_SCRATCH,
                       "graph.scratch", DS4_RUNTIME_CATEGORY_GRAPH_SCRATCH,
                       DS4_RUNTIME_DOMAIN_CUDA_DEVICE, scratch);
    graph_add_callsite(&count, DS4_LAGUNA_CALLSITE_OTHER_HOST_ENGINE + 4u,
                       "graph.other_host.session",
                       DS4_RUNTIME_CATEGORY_OTHER_HOST,
                       DS4_RUNTIME_DOMAIN_HOST, host_bound);
    graph_tracker_config.callsites = graph_callsites;
    graph_tracker_config.callsite_count = count;
    graph_tracker_config.records = records;
    graph_tracker_config.record_capacity = capacity;
    graph_tracker_config.category_bounds[DS4_RUNTIME_CATEGORY_STATIC_WEIGHTS] = 64u;
    graph_tracker_config.category_bounds[DS4_RUNTIME_CATEGORY_OTHER_CUDA] = 64u;
    graph_tracker_config.category_bounds[DS4_RUNTIME_CATEGORY_PINNED_STAGING] = 64u;
    graph_tracker_config.category_bounds[DS4_RUNTIME_CATEGORY_KV_STATE] = kv;
    graph_tracker_config.category_bounds[DS4_RUNTIME_CATEGORY_GRAPH_SCRATCH] = scratch;
    graph_tracker_config.category_bounds[DS4_RUNTIME_CATEGORY_OTHER_HOST] = host_bound;
    graph_tracker_config.report_bounds[
        DS4_RUNTIME_REPORT_MODEL_MAPPING_REGISTERED] = sizeof(ds4_gpu_tensor);
    uint64_t owned_bound = 0u;
    for (size_t i = 0; i < DS4_RUNTIME_OWNED_CATEGORY_COUNT; ++i)
        owned_bound += graph_tracker_config.category_bounds[i];
    graph_tracker_config.owned_total_bound_bytes = owned_bound;
    graph_tracker_config.qualification_total_bound_bytes = owned_bound;
    return ds4_runtime_tracker_init(tracker, &graph_tracker_config) ==
        DS4_RUNTIME_STATUS_OK;
}

static void graph_reset(size_t capacity = 248u) {
    fake_reset();
    fake_fail_device_query = false;
    fake_fail_device_set = false;
    fake_fail_device_sync = false;
    fake_fail_managed_malloc = false;
    fake_fail_calloc = false;
    fake_check_descriptor_order = false;
    fake_expected_descriptor = nullptr;
    fake_expected_device_id = 0u;
    fake_calloc_calls = 0;
    fake_descriptor_free_calls = 0;
    fake_descriptor_order_errors = 0;
    fake_descriptor_live.clear();
    fake_nonheap_descriptor.clear();
    std::memset(g_gpu, 0, sizeof(g_gpu));
    fake_current_device = 7;
    g_n_gpus = 1;
    g_gpu[0].device_id = fake_current_device;
    graph_fail_device_malloc_at = 0;
    graph_partial_driver_failure_at = 0;
    graph_fail_device_free_at = 0;
    graph_fail_device_free_through = 0;
    graph_fail_write_at = 0;
    graph_write_calls = 0;
    graph_resident_alloc_calls = 0;
    graph_resident_free_calls = 0;
    graph_generic_alloc_calls = 0;
    graph_generic_free_calls = 0;
    graph_generic_nonnull_free_calls = 0;
    graph_generic_bytes_calls = 0;
    graph_generic_contents_calls = 0;
    graph_capture_descriptor_alloc = false;
    graph_captured_descriptor_alloc = nullptr;
    graph_descriptor_live_peak = 0u;
    if (!graph_init_tracker(&graph_tracker, graph_records, capacity)) std::abort();
    fake_observed_tracker = &graph_tracker;
}

static int graph_begin(void) {
    fake_observed_tracker = &graph_tracker;
    return ds4_gpu_laguna_resident_observer_begin(&graph_tracker);
}

static void graph_delete_checked(void *storage) {
    if (storage && !graph_fixture_delete(storage)) std::abort();
}

static int graph_tracker_same(const ds4_runtime_tracker *left,
                              const ds4_runtime_tracker *right) {
    return left && right && std::memcmp(left, right, sizeof(*left)) == 0;
}

static int graph_live_records(const ds4_runtime_tracker *tracker) {
    int result = 0;
    if (!tracker) return 0;
    for (size_t i = 0; i < tracker->record_count; ++i) {
        if (tracker->records[i].live) ++result;
    }
    return result;
}

static int graph_live_device_count(void) {
    int result = 0;
    for (const auto &entry : fake_live) {
        if (!entry.second.host) ++result;
    }
    return result;
}

static int graph_live_descriptor_count(void) {
    return (int)fake_descriptor_live.size();
}

static int graph_unrecorded_device_live(const ds4_runtime_tracker *tracker) {
    int result = 0;
    for (const auto &entry : fake_live) {
        if (entry.second.host) continue;
        bool recorded = false;
        if (tracker) {
            for (size_t i = 0; i < tracker->record_count; ++i) {
                const ds4_runtime_allocation_record &record = tracker->records[i];
                if (record.live && (record.id >> 56u) == 0x52u &&
                    record.domain == DS4_RUNTIME_DOMAIN_CUDA_DEVICE &&
                    record.base == (uint64_t)(uintptr_t)entry.first) {
                    recorded = true;
                    break;
                }
            }
        }
        if (!recorded) ++result;
    }
    return result;
}

static void graph_emit_u64(const char *label, const char *key, uint64_t number) {
    std::printf("%s_%s=%llu\n", label, key, (unsigned long long)number);
}

static void graph_emit_int(const char *label, const char *key, int number) {
    std::printf("%s_%s=%d\n", label, key, number);
}

static void graph_emit(const char *label, const void *storage,
                       const ds4_runtime_tracker *tracker) {
    graph_fixture_snapshot snapshot = {};
    (void)graph_fixture_snapshot_read(storage, &snapshot, tracker);
    graph_emit_u64(label, "mode", snapshot.runtime_owner_mode);
    graph_emit_u64(label, "owner_count", snapshot.resident_owner_count);
    graph_emit_u64(label, "runtime_record_count", snapshot.runtime_record_count);
    graph_emit_u64(label, "alias_count", snapshot.alias_count);
    graph_emit_u64(label, "alias_nonnull", snapshot.alias_nonnull);
    graph_emit_u64(label, "owner_tensor_nonnull", snapshot.owner_tensor_nonnull);
    graph_emit_u64(label, "owner_slot_nonnull", snapshot.owner_slot_nonnull);
    graph_emit_u64(label, "owner_slots_in_region", snapshot.owner_slots_in_region);
    graph_emit_u64(label, "owner_slots_foreign", snapshot.owner_slots_foreign);
    graph_emit_u64(label, "owner_exact_bindings", snapshot.owner_exact_bindings);
    graph_emit_u64(label, "owner_live_entries", snapshot.owner_live_entries);
    graph_emit_u64(label, "record_namespace_52", snapshot.record_namespace_52);
    graph_emit_u64(label, "record_namespace_4f", snapshot.record_namespace_4f);
    graph_emit_u64(label, "live_namespace_52", snapshot.live_namespace_52);
    graph_emit_u64(label, "live_namespace_4f", snapshot.live_namespace_4f);
    graph_emit_u64(label, "record_count", tracker ? tracker->record_count : 0u);
    graph_emit_int(label, "active_records", graph_live_records(tracker));
    graph_emit_u64(label, "scratch_bytes", snapshot.scratch_bytes);
    graph_emit_u64(label, "kv_bytes", snapshot.kv_bytes);
    graph_emit_u64(label, "first_record_id", snapshot.first_record_id);
    graph_emit_u64(label, "last_record_id", snapshot.last_record_id);
    graph_emit_int(label, "device_live", graph_live_device_count());
    graph_emit_int(label, "host_descriptors_live", graph_live_descriptor_count());
    graph_emit_int(label, "unrecorded_device_live",
                   graph_unrecorded_device_live(tracker));
    graph_emit_int(label, "device_malloc_calls", fake_device_malloc_calls);
    graph_emit_int(label, "device_free_calls", fake_device_free_calls);
    graph_emit_int(label, "descriptor_calloc_calls", fake_calloc_calls);
    graph_emit_u64(label, "descriptor_live_peak", graph_descriptor_live_peak);
    graph_emit_int(label, "descriptor_free_calls", fake_descriptor_free_calls);
    graph_emit_int(label, "write_calls", graph_write_calls);
    graph_emit_int(label, "resident_alloc_calls", graph_resident_alloc_calls);
    graph_emit_int(label, "resident_free_calls", graph_resident_free_calls);
    graph_emit_int(label, "generic_alloc_calls", graph_generic_alloc_calls);
    graph_emit_int(label, "generic_free_calls", graph_generic_free_calls);
    graph_emit_int(label, "generic_nonnull_free_calls",
                   graph_generic_nonnull_free_calls);
    graph_emit_int(label, "generic_bytes_calls", graph_generic_bytes_calls);
    graph_emit_int(label, "generic_contents_calls", graph_generic_contents_calls);
    graph_emit_int(label, "delete_errors", graph_fixture_delete_error_count());
    graph_emit_u64(label, "descriptor_size", sizeof(ds4_gpu_tensor));
    graph_emit_u64(label, "graph_scratch_current", tracker ?
        tracker->category_current[DS4_RUNTIME_CATEGORY_GRAPH_SCRATCH] : 0u);
    graph_emit_u64(label, "graph_scratch_peak", tracker ?
        tracker->category_peak[DS4_RUNTIME_CATEGORY_GRAPH_SCRATCH] : 0u);
    graph_emit_u64(label, "kv_current", tracker ?
        tracker->category_current[DS4_RUNTIME_CATEGORY_KV_STATE] : 0u);
    graph_emit_u64(label, "kv_peak", tracker ?
        tracker->category_peak[DS4_RUNTIME_CATEGORY_KV_STATE] : 0u);
    graph_emit_u64(label, "other_host_current", tracker ?
        tracker->category_current[DS4_RUNTIME_CATEGORY_OTHER_HOST] : 0u);
    graph_emit_u64(label, "other_host_peak", tracker ?
        tracker->category_peak[DS4_RUNTIME_CATEGORY_OTHER_HOST] : 0u);
    graph_emit_u64(label, "owned_current", tracker ? tracker->owned_total_current : 0u);
    graph_emit_u64(label, "owned_peak", tracker ? tracker->owned_total_peak : 0u);
    graph_emit_u64(label, "qualification_current", tracker ?
        tracker->qualification_total_current : 0u);
    graph_emit_u64(label, "qualification_peak", tracker ?
        tracker->qualification_total_peak : 0u);
    graph_emit_int(label, "violation", tracker ? (int)tracker->violation : 0);
    graph_emit_int(label, "api_errors", fake_api_errors);
    graph_emit_int(label, "descriptor_order_errors", fake_descriptor_order_errors);
    std::fflush(stdout);
}

static int graph_scenario_native_success(void) {
    graph_reset();
    const int began = graph_begin();
    void *storage = graph_fixture_new_zero();
    const int allocated = graph_fixture_alloc(
        storage, 32768u, 4096u, graph_fixture_owner_mode_resident(), &graph_tracker);
    std::printf("began=%d\nallocated=%d\n", began, allocated);
    graph_emit("after", storage, &graph_tracker);
    const int freed = graph_fixture_free(storage);
    std::printf("freed=%d\ngraph_zero=%d\n", freed,
                graph_fixture_is_zero(storage));
    graph_emit("final", storage, &graph_tracker);
    const int ended = ds4_gpu_laguna_resident_observer_end(&graph_tracker);
    std::printf("ended=%d\n", ended);
    graph_delete_checked(storage);
    return 0;
}

static int graph_scenario_native_cycles(void) {
    graph_reset();
    const int began = graph_begin();
    std::printf("began=%d\n", began);
    for (int cycle = 0; cycle < 4; ++cycle) {
        void *storage = graph_fixture_new_zero();
        const int allocated = graph_fixture_alloc(
            storage, 32768u, 4096u, graph_fixture_owner_mode_resident(),
            &graph_tracker);
        const uint64_t owner_id = graph_fixture_owner_id(storage, 0u, 1);
        char after_label[32];
        (void)std::snprintf(after_label, sizeof(after_label), "cycle_%d_after", cycle);
        std::printf("cycle_%d_alloc=%d\ncycle_%d_owner_id=%llu\n"
                    "cycle_%d_record_count=%llu\n",
                    cycle, allocated, cycle, (unsigned long long)owner_id,
                    cycle, (unsigned long long)graph_tracker.record_count);
        graph_emit(after_label, storage, &graph_tracker);
        const int freed = graph_fixture_free(storage);
        char final_label[32];
        (void)std::snprintf(final_label, sizeof(final_label), "cycle_%d_final", cycle);
        std::printf("cycle_%d_freed=%d\ncycle_%d_zero=%d\n",
                    cycle, freed, cycle, graph_fixture_is_zero(storage));
        graph_emit(final_label, storage, &graph_tracker);
        graph_delete_checked(storage);
    }
    graph_emit("cycles", nullptr, &graph_tracker);
    std::printf("ended=%d\n", ds4_gpu_laguna_resident_observer_end(&graph_tracker));
    return 0;
}

static int graph_scenario_native_alloc_failure(int fail_at) {
    graph_reset();
    const int began = graph_begin();
    graph_fail_device_malloc_at = fail_at;
    void *storage = graph_fixture_new_zero();
    const int allocated = graph_fixture_alloc(
        storage, 32768u, 4096u, graph_fixture_owner_mode_resident(), &graph_tracker);
    std::printf("began=%d\nallocated=%d\nfail_at=%d\nfailure_graph_zero=%d\n",
                began, allocated, fail_at, graph_fixture_is_zero(storage));
    graph_emit("failure", storage, &graph_tracker);
    std::printf("ended=%d\n", ds4_gpu_laguna_resident_observer_end(&graph_tracker));
    graph_delete_checked(storage);
    return 0;
}

static int graph_scenario_native_write_failure(int fail_at) {
    graph_reset();
    const int began = graph_begin();
    graph_fail_write_at = fail_at;
    void *storage = graph_fixture_new_zero();
    const int allocated = graph_fixture_alloc(
        storage, 32768u, 4096u, graph_fixture_owner_mode_resident(), &graph_tracker);
    std::printf("began=%d\nallocated=%d\nfail_at=%d\nfailure_graph_zero=%d\n",
                began, allocated, fail_at, graph_fixture_is_zero(storage));
    graph_emit("failure", storage, &graph_tracker);
    std::printf("ended=%d\n", ds4_gpu_laguna_resident_observer_end(&graph_tracker));
    graph_delete_checked(storage);
    return 0;
}

static int graph_scenario_live_end(void) {
    graph_reset(256u);
    const int began = graph_begin();
    void *storage = graph_fixture_new_zero();
    const int allocated = graph_fixture_alloc(
        storage, 32768u, 4096u, graph_fixture_owner_mode_resident(), &graph_tracker);
    const int blocked_end = ds4_gpu_laguna_resident_observer_end(&graph_tracker);
    const int freed = graph_fixture_free(storage);
    const int ended = ds4_gpu_laguna_resident_observer_end(&graph_tracker);
    std::printf("began=%d\nallocated=%d\nblocked_end=%d\nfreed=%d\nended=%d\n",
                began, allocated, blocked_end, freed, ended);
    graph_delete_checked(storage);
    return 0;
}

/* Ordinary graph rollback: the failing allocation has no raw device handle. */
static int graph_scenario_rollback_failure(void) {
    graph_reset(256u);
    const int began = graph_begin();
    graph_fail_device_malloc_at = 30;
    graph_fail_device_free_at = 1;
    void *storage = graph_fixture_new_zero();
    const int allocated = graph_fixture_alloc(
        storage, 32768u, 4096u, graph_fixture_owner_mode_resident(), &graph_tracker);
    std::printf("began=%d\nallocated=%d\nrollback_graph_zero_after_alloc=%d\n",
                began, allocated, graph_fixture_is_zero(storage));
    graph_emit("rollback", storage, &graph_tracker);
    graph_fail_device_free_at = 0;
    const int retry = graph_fixture_free(storage);
    std::printf("retry=%d\ngraph_zero_after_retry=%d\n", retry,
                graph_fixture_is_zero(storage));
    graph_emit("rollback_final", storage, &graph_tracker);
    std::printf("ended=%d\n", ds4_gpu_laguna_resident_observer_end(&graph_tracker));
    graph_delete_checked(storage);
    return 0;
}

/* The existing partial-driver fake returns error with a nonnull raw handle.
 * First free fails private rollback; second free fails the first graph owner. */
static int graph_scenario_private_rollback(void) {
    graph_reset(256u);
    const int began = graph_begin();
    graph_partial_driver_failure_at = 30;
    graph_fail_device_free_through = 2;
    void *storage = graph_fixture_new_zero();
    const int allocated = graph_fixture_alloc(
        storage, 32768u, 4096u, graph_fixture_owner_mode_resident(), &graph_tracker);
    std::printf("began=%d\nallocated=%d\nprivate_graph_zero_after_alloc=%d\n",
                began, allocated, graph_fixture_is_zero(storage));
    graph_emit("private_retained", storage, &graph_tracker);
    const int blocked_end = ds4_gpu_laguna_resident_observer_end(&graph_tracker);
    std::printf("blocked_end=%d\nprivate_device_free_after_blocked_end=%d\n",
                blocked_end, fake_device_free_calls);
    graph_partial_driver_failure_at = 0;
    graph_fail_device_free_through = 0;
    const int retry = graph_fixture_free(storage);
    std::printf("retry=%d\nprivate_graph_zero_after_retry=%d\n", retry,
                graph_fixture_is_zero(storage));
    graph_emit("private_graph_clean", storage, &graph_tracker);
    fake_fail_device_free = true;
    const int end_failed = ds4_gpu_laguna_resident_observer_end(&graph_tracker);
    std::printf("end_failed=%d\n", end_failed);
    graph_emit("private_end_failed", nullptr, &graph_tracker);
    fake_fail_device_free = false;
    const int ended = ds4_gpu_laguna_resident_observer_end(&graph_tracker);
    std::printf("ended=%d\n", ended);
    graph_emit("private_final", nullptr, &graph_tracker);
    graph_delete_checked(storage);
    return 0;
}

static int graph_scenario_cleanup_failure(const char *kind) {
    graph_reset(256u);
    const int began = graph_begin();
    void *storage = graph_fixture_new_zero();
    const int allocated = graph_fixture_alloc(
        storage, 32768u, 4096u, graph_fixture_owner_mode_resident(), &graph_tracker);
    if (!allocated) {
        if (!graph_fixture_is_zero(storage) && !graph_fixture_free(storage)) std::abort();
        graph_delete_checked(storage);
        return 3;
    }
    if (std::strcmp(kind, "query") == 0) fake_fail_device_query = true;
    if (std::strcmp(kind, "sync") == 0) fake_fail_device_sync = true;
    if (std::strcmp(kind, "free") == 0) fake_fail_device_free = true;
    if (std::strcmp(kind, "middle") == 0) graph_fail_device_free_at = 41;
    if (std::strcmp(kind, "late") == 0) graph_fail_device_free_at = 123;
    const int first = graph_fixture_free(storage);
    std::printf("began=%d\nallocated=%d\nfirst=%d\ngraph_zero_after_first=%d\n",
                began, allocated, first, graph_fixture_is_zero(storage));
    graph_emit("retained", storage, &graph_tracker);
    fake_fail_device_query = false;
    fake_fail_device_sync = false;
    fake_fail_device_free = false;
    graph_fail_device_free_at = 0;
    const int retry = graph_fixture_free(storage);
    std::printf("retry=%d\ngraph_zero_after_retry=%d\n", retry,
                graph_fixture_is_zero(storage));
    graph_emit("final", storage, &graph_tracker);
    std::printf("ended=%d\n", ds4_gpu_laguna_resident_observer_end(&graph_tracker));
    graph_delete_checked(storage);
    return 0;
}

static int graph_scenario_copy_or_foreign(bool foreign_slot) {
    graph_reset(256u);
    const int began = graph_begin();
    void *storage = graph_fixture_new_zero();
    const int allocated = graph_fixture_alloc(
        storage, 32768u, 4096u, graph_fixture_owner_mode_resident(), &graph_tracker);
    if (!allocated) {
        graph_delete_checked(storage);
        return 3;
    }
    void *copy = nullptr;
    int refused = 0;
    int copy_discarded = 1;
    if (foreign_slot) {
        (void)graph_fixture_bind_foreign_slot(storage);
        refused = graph_fixture_free(storage);
    } else {
        copy = graph_fixture_clone(storage);
        if (!copy) {
            if (!graph_fixture_free(storage)) std::abort();
            (void)ds4_gpu_laguna_resident_observer_end(&graph_tracker);
            graph_delete_checked(storage);
            return 3;
        }
        refused = graph_fixture_free(copy);
    }
    std::printf("began=%d\nallocated=%d\nrefused=%d\n", began, allocated, refused);
    graph_emit("retained", storage, &graph_tracker);
    if (copy) copy_discarded = graph_fixture_discard_unowned_copy(copy);
    if (foreign_slot) (void)graph_fixture_restore_first_slot(storage);
    const int freed = graph_fixture_free(storage);
    std::printf("freed=%d\ngraph_zero=%d\ncopy_discarded=%d\n", freed,
                graph_fixture_is_zero(storage), copy_discarded);
    std::printf("ended=%d\n", ds4_gpu_laguna_resident_observer_end(&graph_tracker));
    graph_delete_checked(storage);
    return 0;
}

static int graph_scenario_foreign_tracker_free(void) {
    graph_reset(256u);
    const int began = graph_begin();
    if (!graph_init_tracker(&graph_foreign_tracker, graph_foreign_records, 256u)) {
        (void)ds4_gpu_laguna_resident_observer_end(&graph_tracker);
        return 3;
    }
    void *storage = graph_fixture_new_zero();
    const int allocated = graph_fixture_alloc(
        storage, 32768u, 4096u, graph_fixture_owner_mode_resident(), &graph_tracker);
    (void)graph_fixture_replace_tracker(storage, &graph_foreign_tracker);
    const ds4_runtime_tracker primary_before = graph_tracker;
    const ds4_runtime_tracker foreign_before = graph_foreign_tracker;
    const int foreign = graph_fixture_free(storage);
    const int primary_same = graph_tracker_same(&graph_tracker, &primary_before);
    const int foreign_same = graph_tracker_same(&graph_foreign_tracker, &foreign_before);
    (void)graph_fixture_replace_tracker(storage, &graph_tracker);
    std::printf("began=%d\nallocated=%d\nforeign_free=%d\n"
                "foreign_primary_same=%d\nforeign_tracker_same=%d\n", began,
                allocated, foreign, primary_same, foreign_same);
    graph_emit("retained", storage, &graph_tracker);
    const int freed = graph_fixture_free(storage);
    std::printf("freed=%d\ngraph_zero=%d\n", freed,
                graph_fixture_is_zero(storage));
    std::printf("ended=%d\n", ds4_gpu_laguna_resident_observer_end(&graph_tracker));
    graph_delete_checked(storage);
    return 0;
}

static int graph_scenario_foreign_tracker_alloc(void) {
    graph_reset(256u);
    const int began = graph_begin();
    if (!graph_init_tracker(&graph_foreign_tracker, graph_foreign_records, 256u)) {
        (void)ds4_gpu_laguna_resident_observer_end(&graph_tracker);
        return 3;
    }
    void *storage = graph_fixture_new_zero();
    const ds4_runtime_tracker primary_before = graph_tracker;
    const ds4_runtime_tracker foreign_before = graph_foreign_tracker;
    const int result = graph_fixture_alloc(
        storage, 32768u, 4096u, graph_fixture_owner_mode_resident(),
        &graph_foreign_tracker);
    const int primary_same = graph_tracker_same(&graph_tracker, &primary_before);
    const int foreign_same = graph_tracker_same(&graph_foreign_tracker, &foreign_before);
    std::printf("began=%d\nresult=%d\ngraph_zero=%d\n"
                "foreign_primary_same=%d\nforeign_tracker_same=%d\n"
                "device_malloc_calls=%d\n", began, result,
                graph_fixture_is_zero(storage), primary_same, foreign_same,
                fake_device_malloc_calls);
    graph_emit("foreign_alloc", storage, &graph_tracker);
    std::printf("ended=%d\n", ds4_gpu_laguna_resident_observer_end(&graph_tracker));
    graph_delete_checked(storage);
    return 0;
}

static int graph_scenario_nonzero(void) {
    graph_reset(256u);
    const int began = graph_begin();
    void *storage = graph_fixture_new_nonzero();
    const int result = graph_fixture_alloc(
        storage, 32768u, 4096u, graph_fixture_owner_mode_resident(), &graph_tracker);
    std::printf("began=%d\nresult=%d\npattern=%d\ndevice_malloc_calls=%d\n",
                began, result, graph_fixture_is_pattern(storage, 0xa5),
                fake_device_malloc_calls);
    /* Nonzero storage is deliberately not interpreted as a graph. */
    graph_emit("nonzero", nullptr, &graph_tracker);
    std::printf("ended=%d\n", ds4_gpu_laguna_resident_observer_end(&graph_tracker));
    if (!graph_fixture_discard_nonzero_pattern(storage)) std::abort();
    return 0;
}

static int graph_scenario_zero_refusal(const char *kind) {
    graph_reset(256u);
    const bool absent = std::strcmp(kind, "absent-observer") == 0;
    const int began = absent ? 0 : graph_begin();
    uint32_t context = 32768u;
    uint32_t rows = 4096u;
    int mode = graph_fixture_owner_mode_resident();
    ds4_runtime_tracker *tracker = &graph_tracker;
    if (std::strcmp(kind, "null-tracker") == 0) tracker = nullptr;
    if (std::strcmp(kind, "invalid-geometry") == 0) context = 0u;
    if (std::strcmp(kind, "invalid-mode") == 0) mode = 99;
    void *storage = graph_fixture_new_zero();
    const ds4_runtime_tracker before = graph_tracker;
    const int result = graph_fixture_alloc(storage, context, rows, mode, tracker);
    const int tracker_same = graph_tracker_same(&graph_tracker, &before);
    std::printf("began=%d\nresult=%d\ngraph_zero=%d\ntracker_same=%d\n"
                "device_malloc_calls=%d\n", began, result,
                graph_fixture_is_zero(storage), tracker_same, fake_device_malloc_calls);
    graph_emit("refusal", storage, &graph_tracker);
    std::printf("ended=%d\n", absent ? 0 :
                ds4_gpu_laguna_resident_observer_end(&graph_tracker));
    graph_delete_checked(storage);
    return 0;
}

static void graph_corrupt_callsite(const char *kind) {
    if (std::strcmp(kind, "missing-host") == 0) graph_callsites[5].id = UINT32_MAX;
    if (std::strcmp(kind, "wrong-host") == 0) {
        graph_callsites[5].category = DS4_RUNTIME_CATEGORY_OTHER_CUDA;
        graph_callsites[5].domain = DS4_RUNTIME_DOMAIN_CUDA_DEVICE;
    }
    if (std::strcmp(kind, "missing-kv") == 0) graph_callsites[3].id = UINT32_MAX;
    if (std::strcmp(kind, "wrong-kv") == 0) {
        graph_callsites[3].category = DS4_RUNTIME_CATEGORY_OTHER_HOST;
        graph_callsites[3].domain = DS4_RUNTIME_DOMAIN_HOST;
    }
    if (std::strcmp(kind, "missing-graph") == 0) graph_callsites[4].id = UINT32_MAX;
    if (std::strcmp(kind, "wrong-graph") == 0) {
        graph_callsites[4].category = DS4_RUNTIME_CATEGORY_OTHER_HOST;
        graph_callsites[4].domain = DS4_RUNTIME_DOMAIN_HOST;
    }
}

static int graph_scenario_preflight(const char *kind) {
    const bool tombstones = std::strcmp(kind, "tombstones") == 0;
    const bool capacity = std::strcmp(kind, "capacity") == 0;
    graph_reset(capacity ? 247u : (tombstones ? 248u : 256u));
    const int began = graph_begin();
    const ds4_runtime_tracker clean_tracker = graph_tracker;
    ds4_runtime_callsite clean_callsites[6];
    std::memcpy(clean_callsites, graph_callsites, sizeof(clean_callsites));
    if (std::strcmp(kind, "id-budget") == 0) {
        graph_tracker.issued_sequence_high_water[0x52u] =
            UINT64_C(0x00ffffffffffffff) - 247u;
    }
    if (std::strcmp(kind, "record-count") == 0) {
        /* Physical fixture storage stays 512 records; only metadata is corrupt. */
        graph_tracker.record_count = 257u;
    }
    graph_corrupt_callsite(kind);
    if (tombstones) {
        for (uint64_t i = 0; i < 4u; ++i) {
            uint64_t id = 0;
            if (ds4_runtime_tracker_allocate_next(
                    &graph_tracker, 0x52u, DS4_LAGUNA_CALLSITE_KV_STATE,
                    (uint64_t)(uintptr_t)&graph_tombstone_bytes[i], 1u, 1u,
                    &id) != DS4_RUNTIME_STATUS_OK ||
                ds4_runtime_tracker_release(&graph_tracker, id) !=
                    DS4_RUNTIME_STATUS_OK) {
                (void)ds4_gpu_laguna_resident_observer_end(&graph_tracker);
                return 3;
            }
        }
    }
    const ds4_runtime_tracker before = graph_tracker;
    void *storage = graph_fixture_new_zero();
    const int result = graph_fixture_alloc(
        storage, 32768u, 4096u, graph_fixture_owner_mode_resident(), &graph_tracker);
    const int tracker_same = graph_tracker_same(&graph_tracker, &before);
    int restored = 1;
    if (!tombstones) {
        /* Restore deliberate config/record corruption before normal end. */
        graph_tracker = clean_tracker;
        std::memcpy(graph_callsites, clean_callsites, sizeof(graph_callsites));
        restored = graph_tracker_same(&graph_tracker, &clean_tracker);
    }
    std::printf("began=%d\nresult=%d\npreflight_graph_zero=%d\n"
                "preflight_tracker_same=%d\npreflight_restored=%d\n"
                "preflight_device_malloc_calls=%d\n", began, result,
                graph_fixture_is_zero(storage), tracker_same, restored,
                fake_device_malloc_calls);
    graph_emit("preflight", storage, &graph_tracker);
    if (result) {
        const int freed = graph_fixture_free(storage);
        std::printf("freed=%d\n", freed);
    }
    std::printf("ended=%d\n", ds4_gpu_laguna_resident_observer_end(&graph_tracker));
    graph_delete_checked(storage);
    return 0;
}

static int graph_scenario_relation(void) {
    graph_reset(256u);
    const int began = graph_begin();
    void *storage = graph_fixture_new_zero();
    const int allocated = graph_fixture_alloc(
        storage, 32768u, 4096u, graph_fixture_owner_mode_resident(), &graph_tracker);
    const uint64_t relation_id = UINT64_C(0x5300000000000001);
    const int related = graph_fixture_register_host_relation(
        &graph_tracker, storage, relation_id);
    const int blocked = graph_fixture_free(storage);
    std::printf("began=%d\nallocated=%d\nrelated=%d\nblocked=%d\n",
                began, allocated, related, blocked);
    graph_emit("retained", storage, &graph_tracker);
    const int unregistered = graph_fixture_unregister_relation(
        &graph_tracker, relation_id);
    const int retry = graph_fixture_free(storage);
    std::printf("unregistered=%d\nretry=%d\ngraph_zero=%d\n", unregistered,
                retry, graph_fixture_is_zero(storage));
    graph_emit("final", storage, &graph_tracker);
    std::printf("ended=%d\n", ds4_gpu_laguna_resident_observer_end(&graph_tracker));
    graph_delete_checked(storage);
    return 0;
}

static int graph_scenario_legacy(bool with_tracker) {
    graph_reset(with_tracker ? 256u : 248u);
    void *storage = graph_fixture_new_zero();
    ds4_runtime_tracker *tracker = with_tracker ? &graph_tracker : nullptr;
    fake_observed_tracker = tracker;
    const int allocated = graph_fixture_alloc(
        storage, 32768u, 4096u, graph_fixture_owner_mode_legacy(), tracker);
    std::printf("allocated=%d\n", allocated);
    graph_emit("legacy", storage, tracker);
    const int freed = graph_fixture_free(storage);
    std::printf("freed=%d\ngraph_zero=%d\n", freed,
                graph_fixture_is_zero(storage));
    graph_emit("legacy_final", storage, tracker);
    graph_delete_checked(storage);
    return 0;
}

static int graph_scenario_legacy_retry_relation(void) {
    graph_reset(256u);
    void *storage = graph_fixture_new_zero();
    ds4_runtime_tracker *tracker = &graph_tracker;
    fake_observed_tracker = tracker;
    const int allocated = graph_fixture_alloc(
        storage, 32768u, 4096u, graph_fixture_owner_mode_legacy(), tracker);
    const int precondition =
        allocated == 1 && tracker->record_count == 124u &&
        (graph_records[0].id >> 56u) == 0x4fu && graph_records[0].live &&
        graph_records[0].relation == DS4_RUNTIME_RELATION_OWNED_ALLOCATION;
    std::printf("allocated=%d\nprecondition=%d\nexpected_not_live=%d\n",
                allocated, precondition, DS4_RUNTIME_VIOLATION_NOT_LIVE);
    if (!precondition) {
        (void)graph_fixture_free(storage);
        graph_delete_checked(storage);
        return 0;
    }
    /* Isolate retirement refusal: reverse-order recompute sees the deliberate
     * registration-shaped record first. Admit its report-only byte count so
     * REPORT_BOUND cannot mask the later NOT_LIVE sticky-first violation. */
    tracker->report_bounds[DS4_RUNTIME_REPORT_MODEL_MAPPING_REGISTERED] =
        graph_records[0].requested_bytes;
    const ds4_runtime_relation original_relation = graph_records[0].relation;
    graph_records[0].relation = DS4_RUNTIME_RELATION_REGISTRATION;
    const int first = graph_fixture_free(storage);
    std::printf("first=%d\ngraph_zero_after_first=%d\n", first,
                graph_fixture_is_zero(storage));
    graph_emit("retained", storage, tracker);
    graph_records[0].relation = original_relation;
    const int retry = graph_fixture_free(storage);
    std::printf("retry=%d\ngraph_zero_after_retry=%d\n", retry,
                graph_fixture_is_zero(storage));
    graph_emit("final", storage, tracker);
    /* Legacy mode never attached a resident observer. End must refuse it. */
    const int unattached_end = ds4_gpu_laguna_resident_observer_end(tracker);
    std::printf("unattached_end=%d\n", unattached_end);
    graph_delete_checked(storage);
    return 0;
}

static int graph_scenario_attached_query(void) {
    graph_reset(256u);
    if (!graph_init_tracker(&graph_foreign_tracker, graph_foreign_records, 256u))
        return 3;
    const ds4_runtime_tracker primary_before = graph_tracker;
    const ds4_runtime_tracker foreign_before = graph_foreign_tracker;
    const int null_before = ds4_gpu_laguna_resident_observer_attached(nullptr);
    const int foreign_before_result =
        ds4_gpu_laguna_resident_observer_attached(&graph_foreign_tracker);
    const int null_foreign_same =
        graph_tracker_same(&graph_tracker, &primary_before) &&
        graph_tracker_same(&graph_foreign_tracker, &foreign_before);
    const int began = graph_begin();
    const int identical_safe =
        ds4_gpu_laguna_resident_observer_attached(&graph_tracker);
    (void)ds4_runtime_tracker_latch_failure(
        &graph_tracker, DS4_RUNTIME_VIOLATION_EXTERNAL_ATTRIBUTION);
    const int identical_unsafe =
        ds4_gpu_laguna_resident_observer_attached(&graph_tracker);
    const int foreign_unsafe =
        ds4_gpu_laguna_resident_observer_attached(&graph_foreign_tracker);
    const int ended = ds4_gpu_laguna_resident_observer_end(&graph_tracker);
    const int after_end = ds4_gpu_laguna_resident_observer_attached(&graph_tracker);
    std::printf("query_null_before=%d\nquery_foreign_before=%d\n"
                "query_null_foreign_same=%d\nbegan=%d\n"
                "query_identical_safe=%d\nquery_identical_unsafe=%d\n"
                "query_foreign_unsafe=%d\nquery_violation=%d\nended=%d\n"
                "query_after_end=%d\n", null_before, foreign_before_result,
                null_foreign_same, began, identical_safe, identical_unsafe,
                foreign_unsafe, (int)graph_tracker.violation, ended, after_end);
    return 0;
}

int main(int argc, char **argv) {
    alarm(15);
    struct rlimit limit = {0, 0};
    (void)setrlimit(RLIMIT_CORE, &limit);
    if (argc != 2) return 2;
    const std::string name(argv[1]);
    if (name == "native-success") return graph_scenario_native_success();
    if (name == "native-cycles") return graph_scenario_native_cycles();
    if (name == "native-fail-1") return graph_scenario_native_alloc_failure(1);
    if (name == "native-fail-15") return graph_scenario_native_alloc_failure(15);
    if (name == "native-fail-30") return graph_scenario_native_alloc_failure(30);
    if (name == "native-fail-124") return graph_scenario_native_alloc_failure(124);
    if (name == "native-write-fail-1") return graph_scenario_native_write_failure(1);
    if (name == "native-write-fail-2") return graph_scenario_native_write_failure(2);
    if (name == "native-live-end") return graph_scenario_live_end();
    if (name == "native-rollback-failure") return graph_scenario_rollback_failure();
    if (name == "native-private-rollback") return graph_scenario_private_rollback();
    if (name == "native-cleanup-query") return graph_scenario_cleanup_failure("query");
    if (name == "native-cleanup-sync") return graph_scenario_cleanup_failure("sync");
    if (name == "native-cleanup-free") return graph_scenario_cleanup_failure("free");
    if (name == "native-cleanup-middle") return graph_scenario_cleanup_failure("middle");
    if (name == "native-cleanup-late") return graph_scenario_cleanup_failure("late");
    if (name == "native-copy") return graph_scenario_copy_or_foreign(false);
    if (name == "native-foreign-slot") return graph_scenario_copy_or_foreign(true);
    if (name == "native-foreign-tracker-free") return graph_scenario_foreign_tracker_free();
    if (name == "native-foreign-tracker-alloc") return graph_scenario_foreign_tracker_alloc();
    if (name == "native-nonzero") return graph_scenario_nonzero();
    if (name == "native-null-tracker") return graph_scenario_zero_refusal("null-tracker");
    if (name == "native-absent-observer") return graph_scenario_zero_refusal("absent-observer");
    if (name == "native-invalid-geometry") return graph_scenario_zero_refusal("invalid-geometry");
    if (name == "native-invalid-mode") return graph_scenario_zero_refusal("invalid-mode");
    if (name == "native-capacity-247") return graph_scenario_preflight("capacity");
    if (name == "native-id-budget") return graph_scenario_preflight("id-budget");
    if (name == "native-record-count") return graph_scenario_preflight("record-count");
    if (name == "native-missing-host") return graph_scenario_preflight("missing-host");
    if (name == "native-wrong-host") return graph_scenario_preflight("wrong-host");
    if (name == "native-missing-kv") return graph_scenario_preflight("missing-kv");
    if (name == "native-wrong-kv") return graph_scenario_preflight("wrong-kv");
    if (name == "native-missing-graph") return graph_scenario_preflight("missing-graph");
    if (name == "native-wrong-graph") return graph_scenario_preflight("wrong-graph");
    if (name == "native-tombstones") return graph_scenario_preflight("tombstones");
    if (name == "native-relation") return graph_scenario_relation();
    if (name == "native-attached-query") return graph_scenario_attached_query();
    if (name == "legacy-null") return graph_scenario_legacy(false);
    if (name == "legacy-compact") return graph_scenario_legacy(true);
    if (name == "legacy-retry-relation") return graph_scenario_legacy_retry_relation();
    return 2;
}

'''
# Assemble generated sources after the actual source blocks are extracted.
GRAPH_CPP_SOURCE = (
    GRAPH_FAIL_DECLS
    + "\n"
    + GRAPH_CPP_FAKE
    + "\n"
    + GRAPH_TENSOR_SUPPORT
    + "\n"
    + (TMP_DECLS or "")
    + "\n"
    + (OBSERVER_BLOCK or "")
    + "\n"
    + (TENSOR_BLOCK or "")
    + "\n"
    + GENERIC_SOURCE
    + "\n"
    + GRAPH_CPP_WRAPPERS
    + "\n"
    + GRAPH_CPP_DECLS
    + GRAPH_CPP_SUFFIX
)
assert GRAPH_CPP_SOURCE.endswith(GRAPH_CPP_SUFFIX)
GRAPH_C_SOURCE = GRAPH_C_PREFIX + "\n" + (GRAPH_BLOCK or "#error missing Laguna graph block") + "\n" + GRAPH_C_SUFFIX


# These probes deliberately link the graph C TU without any public resident
# API body. If a platform refusal reaches a legacy/native forwarding seam, the
# process aborts instead of silently exercising a substitute implementation.
GRAPH_REFUSAL_STUBS = r'''
#include <stdint.h>
#include <stdlib.h>
#include "ds4_runtime.h"
#include "ds4_gpu_resident.h"

static void graph_refusal_unexpected(void) { abort(); }

ds4_gpu_tensor *graph_fixture_legacy_tensor_alloc(uint64_t bytes) {
    (void)bytes;
    graph_refusal_unexpected();
    return NULL;
}
void graph_fixture_legacy_tensor_free(ds4_gpu_tensor *tensor) {
    (void)tensor;
    graph_refusal_unexpected();
}
uint64_t graph_fixture_legacy_tensor_bytes(const ds4_gpu_tensor *tensor) {
    (void)tensor;
    graph_refusal_unexpected();
    return 0u;
}
void *graph_fixture_legacy_tensor_contents(ds4_gpu_tensor *tensor) {
    (void)tensor;
    graph_refusal_unexpected();
    return NULL;
}
int graph_fixture_fake_tensor_write(ds4_gpu_tensor *tensor, uint64_t offset,
                                    const void *data, uint64_t bytes) {
    (void)tensor;
    (void)offset;
    (void)data;
    (void)bytes;
    graph_refusal_unexpected();
    return 0;
}
int graph_fixture_resident_tensor_alloc(
        ds4_runtime_tracker *tracker, uint32_t callsite_id, uint64_t bytes,
        ds4_gpu_laguna_resident_tensor_owner *out) {
    (void)tracker;
    (void)callsite_id;
    (void)bytes;
    (void)out;
    graph_refusal_unexpected();
    return 0;
}
int graph_fixture_resident_tensor_free(
        ds4_runtime_tracker *tracker,
        ds4_gpu_laguna_resident_tensor_owner *owner) {
    (void)tracker;
    (void)owner;
    graph_refusal_unexpected();
    return 0;
}
'''

GRAPH_REFUSAL_MAIN = r'''
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <unistd.h>
#include "ds4_runtime.h"

size_t graph_fixture_storage_size(void);
void *graph_fixture_new_zero(void);
int graph_fixture_delete(void *storage);
int graph_fixture_is_zero(const void *storage);
int graph_fixture_alloc(void *, uint32_t, uint32_t, int, ds4_runtime_tracker *);
int graph_fixture_owner_mode_resident(void);
int graph_fixture_free(void *storage);

int main(void) {
    alarm(15);
    struct rlimit limit = {0, 0};
    (void)setrlimit(RLIMIT_CORE, &limit);
    ds4_runtime_tracker tracker = {0};
    void *storage = graph_fixture_new_zero();
    if (!storage) return 3;
    const int result = graph_fixture_alloc(
        storage, 32768u, 4096u, graph_fixture_owner_mode_resident(), &tracker);
    const int zero = graph_fixture_is_zero(storage);
    const int freed = graph_fixture_free(storage);
    const int deleted = graph_fixture_delete(storage);
    printf("platform_result=%d\nplatform_zero=%d\nplatform_freed=%d\n"
           "platform_deleted=%d\n", result, zero, freed, deleted);
    return result == 0 && zero == 1 && freed == 1 && deleted == 1 ? 0 : 1;
}
'''

SAFE_ENV = {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"}


def parse_output(stdout: str) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in stdout.splitlines():
        key, separator, raw = line.partition("=")
        if separator and re.fullmatch(r"-?\d+", raw):
            values[key] = int(raw)
    return values


def value(values: dict[str, int], key: str) -> int:
    if key not in values:
        raise AssertionError(f"missing {key}: {values}")
    return values[key]


def logical_make_lines(source: str) -> list[str]:
    lines: list[str] = []
    pending = ""
    for raw in source.splitlines():
        line = raw.rstrip()
        if pending:
            pending += " " + line.lstrip()
        else:
            pending = line
        if pending.endswith("\\"):
            pending = pending[:-1].rstrip()
            continue
        lines.append(pending)
        pending = ""
    if pending:
        lines.append(pending)
    return lines


class ResidentGraphFixture(unittest.TestCase):
    '''Compile and execute only bounded generated fake-driver children.'''

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory(prefix="ds4-native-graph-contract-")
        cls.addClassCleanup(cls._tmp.cleanup)
        base = Path(cls._tmp.name)
        cls._env = dict(SAFE_ENV)
        for key, leaf in (
            ("HOME", "home"),
            ("TMPDIR", "tmp"),
            ("XDG_CONFIG_HOME", "config"),
            ("XDG_CACHE_HOME", "cache"),
            ("XDG_DATA_HOME", "data"),
        ):
            child = base / leaf
            child.mkdir()
            cls._env[key] = str(child)

        cls._cc_flags = [
            "-std=c11", "-O0", "-Wall", "-Wextra", "-Werror=format",
            "-I", str(ROOT),
        ]
        cls._cxx_flags = [
            "-std=c++17", "-O0", "-pthread", "-Wall", "-Wextra",
            "-Werror=format", "-I", str(ROOT),
        ]
        cls._graph_source = base / "graph_fixture.c"
        cls._cpp_source = base / "graph_observer_tensor_fixture.cc"
        cls._runtime_object = base / "ds4_runtime.o"
        cls._graph_object = base / "graph_fixture.o"
        cls._cpp_object = base / "graph_observer_tensor_fixture.o"
        cls._binary = base / "graph_fixture"
        cls._graph_source.write_text(GRAPH_C_SOURCE, encoding="utf-8")
        cls._cpp_source.write_text(GRAPH_CPP_SOURCE, encoding="utf-8")

        cls._runtime_compile = subprocess.run(
            ["cc", *cls._cc_flags, "-c", str(RUNTIME_SOURCE),
             "-o", str(cls._runtime_object)],
            cwd=base, env=cls._env, capture_output=True, text=True,
            timeout=15, check=False,
        )
        if cls._runtime_compile.returncode == 0:
            cls._graph_compile = subprocess.run(
                ["cc", *cls._cc_flags, "-DDS4_GRAPH_FIXTURE_CUDA_BRANCH=1",
                 "-c", str(cls._graph_source), "-o", str(cls._graph_object)],
                cwd=base, env=cls._env, capture_output=True, text=True,
                timeout=15, check=False,
            )
        else:
            cls._graph_compile = cls._runtime_compile
        if cls._graph_compile.returncode == 0:
            cls._cpp_compile = subprocess.run(
                ["c++", *cls._cxx_flags, "-c", str(cls._cpp_source),
                 "-o", str(cls._cpp_object)],
                cwd=base, env=cls._env, capture_output=True, text=True,
                timeout=15, check=False,
            )
        else:
            cls._cpp_compile = cls._graph_compile
        if cls._cpp_compile.returncode == 0:
            cls._link = subprocess.run(
                ["c++", *cls._cxx_flags, str(cls._graph_object),
                 str(cls._cpp_object), str(cls._runtime_object), "-o", str(cls._binary)],
                cwd=base, env=cls._env, capture_output=True, text=True,
                timeout=15, check=False,
            )
        else:
            cls._link = cls._cpp_compile

        # Apple/ROCm graph-only refusal probes. They use the real runtime.o but
        # never link the imported observer/tensor native API bodies.
        cls._refusal_stub_source = base / "graph_refusal_stubs.c"
        cls._refusal_main_source = base / "graph_refusal_main.c"
        cls._refusal_stub_object = base / "graph_refusal_stubs.o"
        cls._refusal_main_object = base / "graph_refusal_main.o"
        cls._refusal_stub_source.write_text(GRAPH_REFUSAL_STUBS, encoding="utf-8")
        cls._refusal_main_source.write_text(GRAPH_REFUSAL_MAIN, encoding="utf-8")
        if cls._runtime_compile.returncode == 0:
            cls._refusal_stub_compile = subprocess.run(
                ["cc", *cls._cc_flags, "-c", str(cls._refusal_stub_source),
                 "-o", str(cls._refusal_stub_object)],
                cwd=base, env=cls._env, capture_output=True, text=True,
                timeout=15, check=False,
            )
            cls._refusal_main_compile = subprocess.run(
                ["cc", *cls._cc_flags, "-c", str(cls._refusal_main_source),
                 "-o", str(cls._refusal_main_object)],
                cwd=base, env=cls._env, capture_output=True, text=True,
                timeout=15, check=False,
            )
        else:
            cls._refusal_stub_compile = cls._runtime_compile
            cls._refusal_main_compile = cls._runtime_compile
        cls._platform_probes: dict[str, tuple[object, object, object]] = {}
        for name, define in (
            ("apple", "-DDS4_GRAPH_FIXTURE_APPLE_BRANCH=1"),
            ("rocm", "-DDS4_GRAPH_FIXTURE_ROCM_BRANCH=1"),
        ):
            graph_object = base / ("graph_refusal_" + name + ".o")
            binary = base / ("graph_refusal_" + name)
            if (cls._runtime_compile.returncode == 0 and
                    cls._refusal_stub_compile.returncode == 0 and
                    cls._refusal_main_compile.returncode == 0):
                graph_compile = subprocess.run(
                    ["cc", *cls._cc_flags, define, "-c", str(cls._graph_source),
                     "-o", str(graph_object)],
                    cwd=base, env=cls._env, capture_output=True, text=True,
                    timeout=15, check=False,
                )
            else:
                graph_compile = cls._runtime_compile
            if graph_compile.returncode == 0:
                link = subprocess.run(
                    ["cc", *cls._cc_flags, str(graph_object),
                     str(cls._refusal_stub_object), str(cls._refusal_main_object),
                     str(cls._runtime_object), "-o", str(binary)],
                    cwd=base, env=cls._env, capture_output=True, text=True,
                    timeout=15, check=False,
                )
            else:
                link = graph_compile
            if link.returncode == 0:
                run = subprocess.run(
                    [str(binary)], cwd=base, env=cls._env, capture_output=True,
                    text=True, timeout=16, check=False,
                )
            else:
                run = link
            cls._platform_probes[name] = (graph_compile, link, run)

    def assert_build(self) -> None:
        failures = []
        for name, result in (
            ("runtime C", self._runtime_compile),
            ("graph C", self._graph_compile),
            ("observer/tensor C++", self._cpp_compile),
            ("link", self._link),
        ):
            if result.returncode != 0:
                failures.append(name + ":\n" + result.stderr)
        self.assertFalse(failures, "resident graph fixture RED:\n" + "\n".join(failures))

    def assert_platform_refusal_probes(self) -> None:
        failures = []
        for platform, results in self._platform_probes.items():
            for phase, result in zip(("graph C", "link", "run"), results):
                if result.returncode != 0:
                    failures.append(
                        platform + " " + phase + ":\n" + result.stderr
                    )
        self.assertFalse(
            failures,
            "platform resident-refusal probe RED:\n" + "\n".join(failures),
        )

    def run_case(self, name: str) -> dict[str, int]:
        self.assert_build()
        result = subprocess.run(
            [str(self._binary), name], cwd=self._binary.parent, env=self._env,
            capture_output=True, text=True, timeout=16, check=False,
        )
        self.assertEqual(
            result.returncode, 0,
            f"resident graph scenario {name} RED:\n"
            f"stdout={result.stdout}\nstderr={result.stderr}",
        )
        values = parse_output(result.stdout)
        for key, count in values.items():
            if key.endswith("_api_errors") or key.endswith("_descriptor_order_errors"):
                self.assertEqual(count, 0, f"fake driver protocol violation: {key}")
        return values

    def test_00_real_graph_seams_and_native_bridge(self) -> None:
        self.assertEqual(DS4_SOURCE.count(GRAPH_START_MARKER), 1)
        self.assertEqual(DS4_SOURCE.count(GRAPH_END_MARKER), 1)
        self.assertTrue(GRAPH_BLOCK, "actual graph block is not marked yet")
        self.assertTrue(GRAPH_CPP_SOURCE.endswith(GRAPH_CPP_SUFFIX))
        self.assertRegex(
            GPU_RESIDENT_HEADER,
            r"int\s+ds4_gpu_laguna_resident_observer_attached\s*\(\s*"
            r"const\s+ds4_runtime_tracker\s*\*tracker\s*\)\s*;",
        )
        self.assertIn("ds4_gpu_laguna_resident_observer_attached", OBSERVER_BLOCK or "")
        self.assertRegex(
            GRAPH_BLOCK or "",
            r"typedef\s+enum\s*\{[^}]*DS4_LAGUNA_GRAPH_OWNER_LEGACY\s*=\s*0"
            r"[^}]*DS4_LAGUNA_GRAPH_OWNER_RESIDENT\s*=\s*1",
        )
        for token in (
            "ds4_laguna_graph_resident_owner",
            "runtime_owner_mode",
            "resident_owners",
            "resident_owner_count",
            "ds4_gpu_laguna_resident_tensor_alloc",
            "ds4_gpu_laguna_resident_tensor_free",
            "DS4_LAGUNA_GRAPH_OWNER_RESIDENT",
        ):
            self.assertIn(token, GRAPH_BLOCK or "", token)
        self.assertNotIn("sizeof(ds4_gpu_tensor)", DS4_SOURCE)
        self.assertRegex(GRAPH_BLOCK or "", r"_Static_assert|static_assert")
        self.assertRegex(DS4_SOURCE, r"static\s+bool\s+laguna_graph_free\s*\(")
        self.assertNotRegex(
            GRAPH_BLOCK or "",
            r"\b(?:malloc|calloc|realloc)\s*\(",
            "graph metadata must remain caller-owned and heap-free",
        )

    def test_01_platform_refusal_probes_and_graph_preflight_scope(self) -> None:
        # The normal generated graph object selects CUDA only after headers.
        selector = GRAPH_C_PREFIX.index("DS4_GRAPH_FIXTURE_CUDA_BRANCH")
        self.assertGreater(selector, GRAPH_C_PREFIX.index('#include "ds4_laguna_resident.h"'))
        self.assertRegex(GRAPH_BLOCK or "", r"__APPLE__")
        self.assertRegex(GRAPH_BLOCK or "", r"DS4_ROCM_BUILD")
        # Graph ownership checks only graph's actual host/KV/scratch concerns.
        for token in (
            "DS4_RUNTIME_CATEGORY_OTHER_HOST",
            "DS4_RUNTIME_DOMAIN_HOST",
            "DS4_RUNTIME_CATEGORY_KV_STATE",
            "DS4_RUNTIME_CATEGORY_GRAPH_SCRATCH",
            "DS4_RUNTIME_DOMAIN_CUDA_DEVICE",
        ):
            self.assertIn(token, GRAPH_BLOCK or "", token)
        self.assertNotIn("ds4_gpu_laguna_resident_tensor_alloc(", GRAPH_REFUSAL_STUBS)
        self.assertNotIn("ds4_gpu_laguna_resident_tensor_free(", GRAPH_REFUSAL_STUBS)
        self.assert_platform_refusal_probes()

    def test_01_actual_graph_is_c_and_real_observer_tensor_are_cpp(self) -> None:
        self.assertTrue(GRAPH_BLOCK)
        self.assertNotIn("using namespace", GRAPH_C_SOURCE)
        self.assertIn('#include "ds4_runtime.h"', GRAPH_C_SOURCE)
        self.assertIn("ds4_runtime.c", str(RUNTIME_SOURCE))
        self.assertEqual(GRAPH_C_SOURCE.count("#error missing Laguna graph block"), 0)
        self.assertIn("graph_fixture_legacy_tensor_contents", GRAPH_CPP_SOURCE)
        self.assertIn("graph_fixture_resident_tensor_alloc", GRAPH_CPP_SOURCE)

    def test_02_build_controls_and_production_legacy_callers(self) -> None:
        target = "test-cuda-resident-graph-contract"
        self.assertRegex(
            MAKEFILE_SOURCE,
            r"(?m)^" + re.escape(target) + r":\s*\n\tpython3 tests/"
            r"test_cuda_resident_graph_contract\.py(?:\s+-v)?\s*$",
        )
        phony: set[str] = set()
        for line in logical_make_lines(MAKEFILE_SOURCE):
            if line.startswith(".PHONY:"):
                phony.update(line.split(":", 1)[1].split())
        self.assertIn(target, phony)
        rules = logical_make_lines(MAKEFILE_SOURCE)
        resident = next((line for line in rules if line.startswith("test-laguna-resident-path:")), "")
        default = next((line for line in rules if line.startswith("test:")), "")
        self.assertIn(target, resident.split())
        self.assertIn("test-laguna-resident-path", default.split())

        generator = source_definition(DS4_SOURCE, "static int generate_laguna_metal_argmax(")
        session_create = source_definition(DS4_SOURCE, "static int ds4_session_create_unchecked(")
        allocator = source_definition(DS4_SOURCE, "static bool laguna_graph_alloc(")
        session_free = source_definition(DS4_SOURCE, "int ds4_session_free_checked(")
        for body in (generator, session_create, allocator, session_free):
            self.assertTrue(body)
        for body in (generator, session_create):
            self.assertRegex(
                body or "",
                r"(?s)laguna_graph_alloc\s*\(.*?DS4_LAGUNA_GRAPH_OWNER_LEGACY",
            )
        self.assertNotIn(
            "DS4_LAGUNA_GRAPH_OWNER_RESIDENT",
            DS4_SOURCE.replace(GRAPH_BLOCK or "", ""),
        )
        # All three direct graph-free callers must observe checked failure.
        self.assertEqual(len(re.findall(r"\blaguna_graph_free\s*\(", DS4_SOURCE)), 4)
        self.assertRegex(
            allocator or "",
            r"(?s)if\s*\(\s*!\s*laguna_graph_free\s*\(\s*g\s*\)\s*\)\s*return\s+false\s*;",
        )
        self.assertRegex(
            generator or "",
            r"(?s)if\s*\(\s*!\s*laguna_graph_free\s*\(\s*&g\s*\)\s*\)\s*return\s+1\s*;",
        )
        session_body = session_free or ""
        graph_free_assignment = re.search(
            r"(?:const\s+)?bool\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*"
            r"laguna_graph_free\s*\(\s*&s->laguna_graph\s*\)",
            session_body,
        )
        self.assertIsNotNone(graph_free_assignment)
        assert graph_free_assignment is not None
        graph_free_name = graph_free_assignment.group("name")
        free_at = graph_free_assignment.start()
        unlock_at = session_body.index("ds4_engine_compact_tracker_unlock", free_at)
        failed_match = re.search(
            r"if\s*\(\s*!\s*" + re.escape(graph_free_name) + r"\s*\)",
            session_body[unlock_at:],
        )
        self.assertIsNotNone(failed_match)
        assert failed_match is not None
        failed_at = unlock_at + failed_match.start()
        common_free_at = session_body.index("token_vec_free", failed_at)
        self.assertLess(free_at, unlock_at)
        self.assertLess(unlock_at, failed_at)
        self.assertLess(failed_at, common_free_at)
        self.assertRegex(session_body[failed_at:common_free_at],
                         r"(?s)if\s*\(.*?\)\s*\{?.*?return\s+0\s*;")

    def test_03_fixture_builds_with_real_runtime(self) -> None:
        self.assert_build()

    def test_04_native_success_has_exact_aliases_records_currents_and_peaks(self) -> None:
        values = self.run_case("native-success")
        scratch = 4096 * 375156 + 413704
        kv = (12 * 32768 + 36 * 512) * 4096
        descriptor = value(values, "after_descriptor_size")
        host = 124 * descriptor
        total = scratch + kv + host
        self.assertEqual(value(values, "began"), 1)
        self.assertEqual(value(values, "allocated"), 1)
        self.assertEqual(value(values, "after_mode"), 1)
        self.assertEqual(value(values, "after_owner_count"), 124)
        self.assertEqual(value(values, "after_runtime_record_count"), 0)
        self.assertEqual(value(values, "after_alias_count"), 124)
        self.assertEqual(value(values, "after_alias_nonnull"), 124)
        self.assertEqual(value(values, "after_owner_tensor_nonnull"), 124)
        self.assertEqual(value(values, "after_owner_slot_nonnull"), 124)
        self.assertEqual(value(values, "after_owner_slots_in_region"), 124)
        self.assertEqual(value(values, "after_owner_slots_foreign"), 0)
        self.assertEqual(value(values, "after_owner_exact_bindings"), 124)
        self.assertEqual(value(values, "after_record_count"), 248)
        self.assertEqual(value(values, "after_active_records"), 248)
        self.assertEqual(value(values, "after_record_namespace_52"), 248)
        self.assertEqual(value(values, "after_record_namespace_4f"), 0)
        self.assertEqual(value(values, "after_live_namespace_52"), 248)
        self.assertEqual(value(values, "after_device_live"), 124)
        self.assertEqual(value(values, "after_host_descriptors_live"), 124)
        self.assertEqual(value(values, "after_unrecorded_device_live"), 0)
        self.assertEqual(value(values, "after_device_malloc_calls"), 124)
        self.assertEqual(value(values, "after_descriptor_calloc_calls"), 124)
        self.assertEqual(value(values, "after_descriptor_live_peak"), 124)
        self.assertEqual(value(values, "after_write_calls"), 2)
        self.assertEqual(value(values, "after_resident_alloc_calls"), 124)
        self.assertEqual(value(values, "after_resident_free_calls"), 0)
        for key in ("generic_alloc_calls", "generic_free_calls", "generic_bytes_calls", "generic_contents_calls"):
            self.assertEqual(value(values, "after_" + key), 0)
        for key, expected in (
            ("graph_scratch_current", scratch), ("graph_scratch_peak", scratch),
            ("kv_current", kv), ("kv_peak", kv),
            ("other_host_current", host), ("other_host_peak", host),
            ("owned_current", total), ("owned_peak", total),
            ("qualification_current", total), ("qualification_peak", total),
        ):
            self.assertEqual(value(values, "after_" + key), expected)
        self.assertEqual(value(values, "freed"), 1)
        self.assertEqual(value(values, "graph_zero"), 1)
        self.assertEqual(value(values, "final_active_records"), 0)
        self.assertEqual(value(values, "final_device_live"), 0)
        self.assertEqual(value(values, "final_host_descriptors_live"), 0)
        for key in ("graph_scratch_current", "kv_current", "other_host_current",
                    "owned_current", "qualification_current"):
            self.assertEqual(value(values, "final_" + key), 0)
        for key, expected in (
            ("graph_scratch_peak", scratch), ("kv_peak", kv),
            ("other_host_peak", host), ("owned_peak", total),
            ("qualification_peak", total),
        ):
            self.assertEqual(value(values, "final_" + key), expected)
        self.assertEqual(value(values, "final_delete_errors"), 0)
        self.assertEqual(value(values, "ended"), 1)

    def test_05_four_cycles_reuse_slots_advance_ids_and_retain_exact_peaks(self) -> None:
        values = self.run_case("native-cycles")
        scratch = 4096 * 375156 + 413704
        kv = (12 * 32768 + 36 * 512) * 4096
        descriptor = value(values, "cycle_0_after_descriptor_size")
        host = 124 * descriptor
        total = scratch + kv + host
        self.assertEqual(value(values, "began"), 1)
        ids = [value(values, f"cycle_{i}_owner_id") for i in range(4)]
        self.assertTrue(all(left < right for left, right in zip(ids, ids[1:])))
        for i in range(4):
            after = f"cycle_{i}_after"
            final = f"cycle_{i}_final"
            self.assertEqual(value(values, f"cycle_{i}_alloc"), 1)
            self.assertEqual(value(values, f"cycle_{i}_record_count"), 248)
            self.assertEqual(value(values, after + "_record_count"), 248)
            self.assertEqual(value(values, after + "_active_records"), 248)
            self.assertEqual(value(values, after + "_device_live"), 124)
            self.assertEqual(value(values, after + "_host_descriptors_live"), 124)
            self.assertEqual(value(values, after + "_owned_peak"), total)
            self.assertEqual(value(values, after + "_qualification_peak"), total)
            self.assertEqual(value(values, f"cycle_{i}_freed"), 1)
            self.assertEqual(value(values, f"cycle_{i}_zero"), 1)
            self.assertEqual(value(values, final + "_record_count"), 248)
            self.assertEqual(value(values, final + "_active_records"), 0)
            self.assertEqual(value(values, final + "_device_live"), 0)
            self.assertEqual(value(values, final + "_host_descriptors_live"), 0)
            for key in ("graph_scratch_current", "kv_current", "other_host_current",
                        "owned_current", "qualification_current"):
                self.assertEqual(value(values, final + "_" + key), 0)
            self.assertEqual(value(values, final + "_graph_scratch_peak"), scratch)
            self.assertEqual(value(values, final + "_kv_peak"), kv)
            self.assertEqual(value(values, final + "_other_host_peak"), host)
        self.assertEqual(value(values, "cycles_record_count"), 248)
        self.assertEqual(value(values, "cycles_active_records"), 0)
        self.assertEqual(value(values, "cycles_owned_peak"), total)
        self.assertEqual(value(values, "ended"), 1)

    def test_06_first_middle_late_allocation_failures_have_exact_prefixes(self) -> None:
        scratch_values = [
            4096 * 4,
            4096 * 3072 * 4, 4096 * 3072 * 4, 4096 * 3072 * 4,
            4096 * (72 * 128) * 4,
            4096 * (8 * 128) * 4, 4096 * (8 * 128) * 4,
            4096 * 72 * 4, 4096 * (72 * 128) * 4,
            4096 * 3072 * 4, 4096 * 3072 * 4, 4096 * 3072 * 4,
            4096 * 12288 * 4, 4096 * 12288 * 4, 4096 * 12288 * 4,
            4096 * 3072 * 4, 4096 * 3072 * 4,
            4096 * 10 * 1024 * 4,
            4096 * 256 * 4, 4096 * 256 * 4,
            4096 * 10 * 4, 4096 * 10 * 4,
            4, 4,
            4096 * (8 * 128) * 2, 4096 * (8 * 128) * 2,
            3072 * 4, 100352 * 4,
        ]
        self.assertEqual(sum(scratch_values), 4096 * 375156 + 413704)
        kv_tensors = []
        for layer in range(48):
            pair = (32768 if layer % 4 == 0 else 512) * 4096
            kv_tensors.extend((pair // 2, pair // 2))

        for fail_at in (1, 15, 30, 124):
            with self.subTest(fail_at=fail_at):
                values = self.run_case(f"native-fail-{fail_at}")
                successful = fail_at - 1
                prefix = scratch_values[:successful]
                scratch_prefix = sum(prefix)
                kv_prefix = 0
                if successful > len(scratch_values):
                    scratch_prefix = sum(scratch_values)
                    kv_prefix = sum(kv_tensors[:successful - len(scratch_values)])
                descriptor = value(values, "failure_descriptor_size")
                host_peak = fail_at * descriptor
                owned_peak = scratch_prefix + kv_prefix + host_peak
                self.assertEqual(value(values, "allocated"), 0)
                self.assertEqual(value(values, "failure_graph_zero"), 1)
                self.assertEqual(value(values, "failure_active_records"), 0)
                self.assertEqual(value(values, "failure_device_live"), 0)
                self.assertEqual(value(values, "failure_host_descriptors_live"), 0)
                self.assertEqual(value(values, "failure_unrecorded_device_live"), 0)
                self.assertNotEqual(value(values, "failure_violation"), 0)
                self.assertEqual(value(values, "failure_device_malloc_calls"), fail_at)
                self.assertEqual(value(values, "failure_device_free_calls"), successful)
                self.assertEqual(value(values, "failure_descriptor_calloc_calls"), fail_at)
                self.assertEqual(value(values, "failure_descriptor_live_peak"), fail_at)
                self.assertEqual(value(values, "failure_descriptor_free_calls"), fail_at)
                self.assertEqual(value(values, "failure_resident_alloc_calls"), fail_at)
                self.assertEqual(value(values, "failure_resident_free_calls"), successful)
                self.assertEqual(value(values, "failure_graph_scratch_current"), 0)
                self.assertEqual(value(values, "failure_kv_current"), 0)
                self.assertEqual(value(values, "failure_other_host_current"), 0)
                self.assertEqual(value(values, "failure_owned_current"), 0)
                self.assertEqual(value(values, "failure_graph_scratch_peak"), scratch_prefix)
                self.assertEqual(value(values, "failure_kv_peak"), kv_prefix)
                self.assertEqual(value(values, "failure_other_host_peak"), host_peak)
                self.assertEqual(value(values, "failure_owned_peak"), owned_peak)
                for key in ("generic_alloc_calls", "generic_free_calls", "generic_bytes_calls", "generic_contents_calls"):
                    self.assertEqual(value(values, "failure_" + key), 0)
                self.assertEqual(value(values, "ended"), 1)

    def test_07_shared_write_failures_stop_before_kv_with_exact_peaks(self) -> None:
        scratch = 4096 * 375156 + 413704
        for fail_at in (1, 2):
            with self.subTest(fail_at=fail_at):
                values = self.run_case(f"native-write-fail-{fail_at}")
                descriptor = value(values, "failure_descriptor_size")
                host = 28 * descriptor
                self.assertEqual(value(values, "allocated"), 0)
                self.assertEqual(value(values, "failure_graph_zero"), 1)
                self.assertEqual(value(values, "failure_write_calls"), fail_at)
                self.assertEqual(value(values, "failure_device_malloc_calls"), 28)
                self.assertEqual(value(values, "failure_descriptor_calloc_calls"), 28)
                self.assertEqual(value(values, "failure_descriptor_live_peak"), 28)
                self.assertEqual(value(values, "failure_device_free_calls"), 28)
                self.assertEqual(value(values, "failure_descriptor_free_calls"), 28)
                self.assertEqual(value(values, "failure_graph_scratch_peak"), scratch)
                self.assertEqual(value(values, "failure_kv_peak"), 0)
                self.assertEqual(value(values, "failure_other_host_peak"), host)
                self.assertEqual(value(values, "failure_owned_peak"), scratch + host)
                self.assertEqual(value(values, "failure_device_live"), 0)
                self.assertEqual(value(values, "failure_active_records"), 0)
                self.assertNotEqual(value(values, "failure_violation"), 0)
                self.assertEqual(value(values, "ended"), 1)

    def test_08_live_observer_end_refuses_then_checked_local_free_allows_end(self) -> None:
        values = self.run_case("native-live-end")
        self.assertEqual(value(values, "began"), 1)
        self.assertEqual(value(values, "allocated"), 1)
        self.assertEqual(value(values, "blocked_end"), 0)
        self.assertEqual(value(values, "freed"), 1)
        self.assertEqual(value(values, "ended"), 1)

    def test_09_cleanup_failures_retain_exact_prefix_then_retry_while_sticky(self) -> None:
        scratch = 4096 * 375156 + 413704
        kv = (12 * 32768 + 36 * 512) * 4096
        middle_kv = 10 * (32768 * 4096) + 32 * (512 * 4096)
        late_kv = 512 * 4096
        cases = {
            "native-cleanup-query": (124, scratch, kv, 0),
            "native-cleanup-sync": (124, scratch, kv, 0),
            "native-cleanup-free": (124, scratch, kv, 1),
            "native-cleanup-middle": (84, 0, middle_kv, 41),
            "native-cleanup-late": (2, 0, late_kv, 123),
        }
        for scenario, (live, scratch_current, kv_current, free_calls) in cases.items():
            with self.subTest(scenario=scenario):
                values = self.run_case(scenario)
                descriptor = value(values, "retained_descriptor_size")
                host_current = live * descriptor
                total_current = scratch_current + kv_current + host_current
                total_peak = scratch + kv + 124 * descriptor
                self.assertEqual(value(values, "allocated"), 1)
                self.assertEqual(value(values, "first"), 0)
                self.assertEqual(value(values, "retained_owner_count"), 124)
                self.assertEqual(value(values, "retained_owner_live_entries"), live)
                self.assertEqual(value(values, "retained_alias_nonnull"), live)
                self.assertEqual(value(values, "retained_owner_exact_bindings"), live)
                self.assertEqual(value(values, "retained_active_records"), 2 * live)
                self.assertEqual(value(values, "retained_device_live"), live)
                self.assertEqual(value(values, "retained_host_descriptors_live"), live)
                self.assertEqual(value(values, "retained_device_free_calls"), free_calls)
                self.assertEqual(value(values, "retained_descriptor_free_calls"), 124 - live)
                self.assertEqual(value(values, "retained_graph_scratch_current"), scratch_current)
                self.assertEqual(value(values, "retained_kv_current"), kv_current)
                self.assertEqual(value(values, "retained_other_host_current"), host_current)
                self.assertEqual(value(values, "retained_owned_current"), total_current)
                self.assertEqual(value(values, "retained_qualification_current"), total_current)
                self.assertEqual(value(values, "retained_graph_scratch_peak"), scratch)
                self.assertEqual(value(values, "retained_kv_peak"), kv)
                self.assertEqual(value(values, "retained_other_host_peak"), 124 * descriptor)
                self.assertEqual(value(values, "retained_owned_peak"), total_peak)
                self.assertNotEqual(value(values, "retained_violation"), 0)
                self.assertEqual(value(values, "retry"), 1)
                self.assertEqual(value(values, "graph_zero_after_retry"), 1)
                self.assertEqual(value(values, "final_active_records"), 0)
                self.assertEqual(value(values, "final_device_live"), 0)
                for key in ("graph_scratch_current", "kv_current", "other_host_current",
                            "owned_current", "qualification_current"):
                    self.assertEqual(value(values, "final_" + key), 0)
                self.assertEqual(value(values, "final_graph_scratch_peak"), scratch)
                self.assertEqual(value(values, "final_kv_peak"), kv)
                self.assertEqual(value(values, "final_owned_peak"), total_peak)
                self.assertEqual(value(values, "final_violation"), value(values, "retained_violation"))
                self.assertEqual(value(values, "ended"), 1)

    def test_10_ordinary_rollback_failure_preserves_graph_for_retry(self) -> None:
        values = self.run_case("native-rollback-failure")
        scratch = 4096 * 375156 + 413704
        first_kv_tensor = (32768 * 4096) // 2
        descriptor = value(values, "rollback_descriptor_size")
        self.assertEqual(value(values, "allocated"), 0)
        self.assertEqual(value(values, "rollback_graph_zero_after_alloc"), 0)
        self.assertEqual(value(values, "rollback_owner_count"), 29)
        self.assertEqual(value(values, "rollback_owner_live_entries"), 29)
        self.assertEqual(value(values, "rollback_active_records"), 58)
        self.assertEqual(value(values, "rollback_device_live"), 29)
        self.assertEqual(value(values, "rollback_host_descriptors_live"), 29)
        self.assertEqual(value(values, "rollback_unrecorded_device_live"), 0)
        self.assertEqual(value(values, "rollback_device_malloc_calls"), 30)
        self.assertEqual(value(values, "rollback_device_free_calls"), 1)
        self.assertEqual(value(values, "rollback_descriptor_calloc_calls"), 30)
        self.assertEqual(value(values, "rollback_descriptor_live_peak"), 30)
        self.assertEqual(value(values, "rollback_descriptor_free_calls"), 1)
        self.assertEqual(value(values, "rollback_graph_scratch_current"), scratch)
        self.assertEqual(value(values, "rollback_kv_current"), first_kv_tensor)
        self.assertEqual(value(values, "rollback_other_host_current"), 29 * descriptor)
        self.assertNotEqual(value(values, "rollback_violation"), 0)
        self.assertEqual(value(values, "retry"), 1)
        self.assertEqual(value(values, "graph_zero_after_retry"), 1)
        self.assertEqual(value(values, "rollback_final_active_records"), 0)
        self.assertEqual(value(values, "rollback_final_violation"), value(values, "rollback_violation"))
        self.assertEqual(value(values, "ended"), 1)

    def test_11_private_partial_rollback_is_not_a_graph_owner(self) -> None:
        values = self.run_case("native-private-rollback")
        scratch = 4096 * 375156 + 413704
        first_kv_tensor = (32768 * 4096) // 2
        descriptor = value(values, "private_retained_descriptor_size")
        self.assertEqual(value(values, "allocated"), 0)
        self.assertEqual(value(values, "private_graph_zero_after_alloc"), 0)
        self.assertEqual(value(values, "private_retained_owner_count"), 29)
        self.assertEqual(value(values, "private_retained_owner_live_entries"), 29)
        self.assertEqual(value(values, "private_retained_active_records"), 58)
        self.assertEqual(value(values, "private_retained_device_live"), 30)
        self.assertEqual(value(values, "private_retained_unrecorded_device_live"), 1)
        self.assertEqual(value(values, "private_retained_host_descriptors_live"), 29)
        self.assertEqual(value(values, "private_retained_device_malloc_calls"), 30)
        self.assertEqual(value(values, "private_retained_device_free_calls"), 2)
        self.assertEqual(value(values, "private_retained_descriptor_calloc_calls"), 30)
        self.assertEqual(value(values, "private_retained_descriptor_live_peak"), 30)
        self.assertEqual(value(values, "private_retained_graph_scratch_current"), scratch)
        self.assertEqual(value(values, "private_retained_kv_current"), first_kv_tensor)
        self.assertEqual(value(values, "private_retained_other_host_current"), 29 * descriptor)
        self.assertEqual(value(values, "blocked_end"), 0)
        self.assertEqual(value(values, "private_device_free_after_blocked_end"), 2)
        self.assertEqual(value(values, "retry"), 1)
        self.assertEqual(value(values, "private_graph_zero_after_retry"), 1)
        self.assertEqual(value(values, "private_graph_clean_active_records"), 0)
        self.assertEqual(value(values, "private_graph_clean_device_live"), 1)
        self.assertEqual(value(values, "private_graph_clean_unrecorded_device_live"), 1)
        self.assertEqual(value(values, "end_failed"), 0)
        self.assertEqual(value(values, "private_end_failed_device_live"), 1)
        self.assertEqual(value(values, "ended"), 1)
        self.assertEqual(value(values, "private_final_device_live"), 0)
        self.assertEqual(value(values, "private_final_unrecorded_device_live"), 0)
        self.assertNotEqual(value(values, "private_retained_violation"), 0)
        self.assertEqual(value(values, "private_graph_clean_violation"), value(values, "private_retained_violation"))
        self.assertEqual(value(values, "private_end_failed_violation"), value(values, "private_retained_violation"))
        self.assertEqual(value(values, "private_final_violation"), value(values, "private_retained_violation"))
        for key in ("graph_scratch_peak", "kv_peak", "other_host_peak", "owned_peak"):
            self.assertEqual(value(values, "private_graph_clean_" + key),
                             value(values, "private_retained_" + key))
            self.assertEqual(value(values, "private_final_" + key),
                             value(values, "private_retained_" + key))

    def test_12_copied_and_foreign_slots_refuse_before_dereference(self) -> None:
        for scenario in ("native-copy", "native-foreign-slot"):
            with self.subTest(scenario=scenario):
                values = self.run_case(scenario)
                self.assertEqual(value(values, "allocated"), 1)
                self.assertEqual(value(values, "refused"), 0)
                self.assertEqual(value(values, "retained_owner_count"), 124)
                self.assertEqual(value(values, "retained_owner_live_entries"), 124)
                self.assertEqual(value(values, "retained_device_live"), 124)
                self.assertEqual(value(values, "retained_device_free_calls"), 0)
                self.assertEqual(value(values, "copy_discarded"), 1)
                self.assertEqual(value(values, "freed"), 1)
                self.assertEqual(value(values, "graph_zero"), 1)
                self.assertEqual(value(values, "ended"), 1)

    def test_13_foreign_tracker_free_and_allocation_are_no_mutation(self) -> None:
        values = self.run_case("native-foreign-tracker-free")
        self.assertEqual(value(values, "allocated"), 1)
        self.assertEqual(value(values, "foreign_free"), 0)
        self.assertEqual(value(values, "foreign_primary_same"), 1)
        self.assertEqual(value(values, "foreign_tracker_same"), 1)
        self.assertEqual(value(values, "retained_owner_count"), 124)
        self.assertEqual(value(values, "retained_active_records"), 248)
        self.assertEqual(value(values, "retained_device_free_calls"), 0)
        self.assertEqual(value(values, "freed"), 1)
        self.assertEqual(value(values, "ended"), 1)

        values = self.run_case("native-foreign-tracker-alloc")
        self.assertEqual(value(values, "began"), 1)
        self.assertEqual(value(values, "result"), 0)
        self.assertEqual(value(values, "graph_zero"), 1)
        self.assertEqual(value(values, "foreign_primary_same"), 1)
        self.assertEqual(value(values, "foreign_tracker_same"), 1)
        self.assertEqual(value(values, "device_malloc_calls"), 0)
        self.assertEqual(value(values, "foreign_alloc_active_records"), 0)
        self.assertEqual(value(values, "ended"), 1)

    def test_14_independent_zero_refusals_and_nonzero_pattern_preservation(self) -> None:
        values = self.run_case("native-nonzero")
        self.assertEqual(value(values, "began"), 1)
        self.assertEqual(value(values, "result"), 0)
        self.assertEqual(value(values, "pattern"), 1)
        self.assertEqual(value(values, "device_malloc_calls"), 0)
        self.assertEqual(value(values, "nonzero_active_records"), 0)
        self.assertEqual(value(values, "ended"), 1)

        for scenario in (
            "native-null-tracker", "native-absent-observer",
            "native-invalid-geometry", "native-invalid-mode",
        ):
            with self.subTest(scenario=scenario):
                values = self.run_case(scenario)
                self.assertEqual(value(values, "result"), 0)
                self.assertEqual(value(values, "graph_zero"), 1)
                self.assertEqual(value(values, "tracker_same"), 1)
                self.assertEqual(value(values, "device_malloc_calls"), 0)
                self.assertEqual(value(values, "refusal_active_records"), 0)
                if scenario == "native-absent-observer":
                    self.assertEqual(value(values, "began"), 0)
                    self.assertEqual(value(values, "ended"), 0)
                else:
                    self.assertEqual(value(values, "began"), 1)
                    self.assertEqual(value(values, "ended"), 1)

    def test_15_capacity_tombstones_id_record_and_independent_site_preflight(self) -> None:
        for scenario in (
            "native-capacity-247", "native-record-count", "native-id-budget",
            "native-missing-host", "native-wrong-host",
            "native-missing-kv", "native-wrong-kv",
            "native-missing-graph", "native-wrong-graph",
        ):
            with self.subTest(scenario=scenario):
                values = self.run_case(scenario)
                self.assertEqual(value(values, "result"), 0)
                self.assertEqual(value(values, "preflight_graph_zero"), 1)
                self.assertEqual(value(values, "preflight_tracker_same"), 1)
                self.assertEqual(value(values, "preflight_restored"), 1)
                self.assertEqual(value(values, "preflight_device_malloc_calls"), 0)
                self.assertEqual(value(values, "preflight_active_records"), 0)
                self.assertEqual(value(values, "ended"), 1)

        values = self.run_case("native-tombstones")
        self.assertEqual(value(values, "result"), 1)
        self.assertEqual(value(values, "preflight_owner_count"), 124)
        self.assertEqual(value(values, "preflight_record_count"), 248)
        self.assertEqual(value(values, "preflight_active_records"), 248)
        self.assertEqual(value(values, "preflight_record_namespace_52"), 248)
        self.assertEqual(value(values, "preflight_first_record_id") >> 56, 0x52)
        self.assertEqual(value(values, "freed"), 1)
        self.assertEqual(value(values, "ended"), 1)

    def test_16_attached_identity_query_is_read_only_and_sticky_safe(self) -> None:
        values = self.run_case("native-attached-query")
        self.assertEqual(value(values, "query_null_before"), 0)
        self.assertEqual(value(values, "query_foreign_before"), 0)
        self.assertEqual(value(values, "query_null_foreign_same"), 1)
        self.assertEqual(value(values, "began"), 1)
        self.assertEqual(value(values, "query_identical_safe"), 1)
        self.assertEqual(value(values, "query_identical_unsafe"), 1)
        self.assertEqual(value(values, "query_foreign_unsafe"), 0)
        self.assertNotEqual(value(values, "query_violation"), 0)
        self.assertEqual(value(values, "ended"), 1)
        self.assertEqual(value(values, "query_after_end"), 0)

    def test_17_valid_host_relation_blocks_release_then_unregister_retry(self) -> None:
        values = self.run_case("native-relation")
        self.assertEqual(value(values, "related"), 1)
        self.assertEqual(value(values, "blocked"), 0)
        self.assertEqual(value(values, "retained_owner_count"), 124)
        self.assertEqual(value(values, "retained_active_records"), 249)
        self.assertNotEqual(value(values, "retained_violation"), 0)
        self.assertEqual(value(values, "unregistered"), 1)
        self.assertEqual(value(values, "retry"), 1)
        self.assertEqual(value(values, "graph_zero"), 1)
        self.assertEqual(value(values, "ended"), 1)

    def test_18_legacy_null_and_compact_controls_remain_distinct(self) -> None:
        raw = self.run_case("legacy-null")
        self.assertEqual(value(raw, "allocated"), 1)
        self.assertEqual(value(raw, "legacy_record_count"), 0)
        self.assertEqual(value(raw, "legacy_record_namespace_52"), 0)
        self.assertEqual(value(raw, "legacy_generic_alloc_calls"), 124)
        self.assertEqual(value(raw, "legacy_generic_bytes_calls"), 124)
        self.assertEqual(value(raw, "legacy_generic_contents_calls"), 0)
        self.assertEqual(value(raw, "legacy_write_calls"), 2)
        self.assertEqual(value(raw, "freed"), 1)
        self.assertEqual(value(raw, "legacy_final_generic_free_calls"), 28 + 2 * 79)
        self.assertEqual(value(raw, "legacy_final_resident_alloc_calls"), 0)
        self.assertEqual(value(raw, "legacy_final_resident_free_calls"), 0)
        self.assertEqual(value(raw, "graph_zero"), 1)

        compact = self.run_case("legacy-compact")
        self.assertEqual(value(compact, "allocated"), 1)
        self.assertEqual(value(compact, "legacy_record_count"), 124)
        self.assertEqual(value(compact, "legacy_record_namespace_4f"), 124)
        self.assertEqual(value(compact, "legacy_record_namespace_52"), 0)
        self.assertEqual(value(compact, "legacy_generic_alloc_calls"), 124)
        self.assertEqual(value(compact, "legacy_generic_bytes_calls"), 124)
        self.assertEqual(value(compact, "legacy_generic_contents_calls"), 124)
        self.assertEqual(value(compact, "legacy_write_calls"), 2)
        self.assertEqual(value(compact, "freed"), 1)
        self.assertEqual(value(compact, "legacy_final_generic_free_calls"), 28 + 2 * 79)
        self.assertEqual(value(compact, "graph_zero"), 1)

    def test_19_legacy_compact_relation_corruption_frees_once_then_retries(self) -> None:
        values = self.run_case("legacy-retry-relation")
        self.assertEqual(value(values, "allocated"), 1)
        self.assertEqual(value(values, "precondition"), 1)
        self.assertEqual(value(values, "first"), 0)
        self.assertEqual(value(values, "graph_zero_after_first"), 0)
        self.assertEqual(value(values, "retained_record_count"), 124)
        self.assertEqual(value(values, "retained_record_namespace_4f"), 124)
        self.assertEqual(value(values, "retained_record_namespace_52"), 0)
        self.assertEqual(value(values, "retained_active_records"), 1)
        self.assertEqual(value(values, "retained_live_namespace_4f"), 1)
        self.assertEqual(value(values, "retained_alias_nonnull"), 0)
        self.assertEqual(value(values, "retained_device_live"), 0)
        self.assertEqual(value(values, "retained_host_descriptors_live"), 0)
        self.assertEqual(value(values, "retained_owner_count"), 0)
        for key in ("generic_nonnull_free_calls", "device_free_calls",
                    "descriptor_free_calls"):
            self.assertEqual(value(values, "retained_" + key), 124)
        self.assertEqual(value(values, "retained_violation"),
                         value(values, "expected_not_live"))
        self.assertEqual(value(values, "retry"), 1)
        self.assertEqual(value(values, "graph_zero_after_retry"), 1)
        self.assertEqual(value(values, "final_active_records"), 0)
        self.assertEqual(value(values, "final_live_namespace_4f"), 0)
        self.assertEqual(value(values, "final_alias_nonnull"), 0)
        self.assertEqual(value(values, "final_device_live"), 0)
        self.assertEqual(value(values, "final_host_descriptors_live"), 0)
        for key in ("generic_nonnull_free_calls", "device_free_calls",
                    "descriptor_free_calls"):
            self.assertEqual(value(values, "final_" + key),
                             value(values, "retained_" + key))
        for key in ("graph_scratch_current", "kv_current", "other_host_current",
                    "owned_current", "qualification_current"):
            self.assertEqual(value(values, "final_" + key), 0)
        self.assertEqual(value(values, "final_violation"),
                         value(values, "retained_violation"))
        self.assertEqual(value(values, "unattached_end"), 0)

if __name__ == "__main__":
    unittest.main(verbosity=2)
