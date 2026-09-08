#!/usr/bin/env python3
'''Host-only contract for the root-owned native CUDA resident tensor.

The fixture extracts the real resident observer block, the real marked resident
native tensor block, and the existing generic tensor entry points from
``ds4_cuda.cu``.  It links those bodies against the real ``ds4_runtime.c`` and
a deterministic fake CUDA driver.  Device handles are synthetic integers and
are never dereferenced; only tiny host tensor descriptors are real allocations.
The existing ownership fixture contributes ``FAKE_PREFIX`` and the observer
extraction through a normal test-module import at test runtime.  This module
never copies an observer or runtime implementation.
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
        TMP_DECLS,
        extract_definition,
    )
except ModuleNotFoundError:
    from tests.test_cuda_resident_ownership_contract import (
        FAKE_PREFIX,
        RUNTIME_HEADER,
        RUNTIME_SOURCE,
        ROOT,
        OBSERVER_BLOCK,
        TMP_DECLS,
        extract_definition,
    )

CUDA_SOURCE = (ROOT / "ds4_cuda.cu").read_text(encoding="utf-8")
GPU_HEADER_SOURCE = (ROOT / "ds4_gpu.h").read_text(encoding="utf-8")
MGPU_HEADER_SOURCE = (ROOT / "ds4_gpu_mgpu.h").read_text(encoding="utf-8")
MAKEFILE_SOURCE = (ROOT / "Makefile").read_text(encoding="utf-8")

TENSOR_START_MARKER = "/* Resident native tensor ownership. */"
TENSOR_END_MARKER = "/* End resident native tensor ownership. */"


def extract_marked_block(source: str, start_marker: str, end_marker: str) -> str | None:
    '''Extract a whole-line marked source block without accepting prose hits.'''
    start = source.find(start_marker)
    if start < 0:
        return None
    line_start = source.rfind("\n", 0, start) + 1
    if source[line_start:start].strip():
        return None
    end = source.find(end_marker, start + len(start_marker))
    if end < 0:
        raise AssertionError("resident tensor start marker has no end marker")
    if source[source.rfind("\n", 0, end) + 1 : end].strip():
        return None
    end_line = source.find("\n", end)
    if end_line < 0:
        end_line = len(source)
    return source[line_start:end_line]


TENSOR_BLOCK = extract_marked_block(
    CUDA_SOURCE, TENSOR_START_MARKER, TENSOR_END_MARKER
)
GENERIC_SIGNATURES = {
    "alloc_on": 'extern "C" int ds4_gpu_tensor_alloc_on(',
    "free_in_place": 'extern "C" void ds4_gpu_tensor_free_in_place(',
    "alloc": 'extern "C" ds4_gpu_tensor *ds4_gpu_tensor_alloc(',
    "alloc_managed": 'extern "C" ds4_gpu_tensor *ds4_gpu_tensor_alloc_managed(',
    "alloc_ptr_on": 'extern "C" ds4_gpu_tensor *ds4_gpu_tensor_alloc_ptr_on(',
    "alloc_managed_on": 'extern "C" ds4_gpu_tensor *ds4_gpu_tensor_alloc_managed_on(',
    "view": 'extern "C" ds4_gpu_tensor *ds4_gpu_tensor_view(',
    "free": 'extern "C" void ds4_gpu_tensor_free(',
}
GENERIC_BODIES = {
    name: extract_definition(CUDA_SOURCE, signature)
    for name, signature in GENERIC_SIGNATURES.items()
}
GENERIC_SOURCE = "\n\n".join(
    body for body in GENERIC_BODIES.values() if body is not None
)
MISSING_SOURCE_SEAMS = [
    label
    for label, present in (
        ("resident tensor block", TENSOR_BLOCK),
        ("resident observer block", OBSERVER_BLOCK),
        ("tensor alloc_on", GENERIC_BODIES["alloc_on"]),
        ("tensor free_in_place", GENERIC_BODIES["free_in_place"]),
        ("tensor alloc", GENERIC_BODIES["alloc"]),
        ("tensor managed alloc", GENERIC_BODIES["alloc_managed"]),
        ("tensor tiered alloc", GENERIC_BODIES["alloc_ptr_on"]),
        ("tensor tiered managed alloc", GENERIC_BODIES["alloc_managed_on"]),
        ("tensor view", GENERIC_BODIES["view"]),
        ("tensor free", GENERIC_BODIES["free"]),
    )
    if not present
]

HEADER_PROBES = {
    "c-gpu-then-mgpu": r'''
#include "ds4_runtime.h"
#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"
#include "ds4_gpu_resident.h"
int resident_header_probe_c(ds4_runtime_tracker *tracker) {
    ds4_gpu_laguna_resident_tensor_owner owner = {0};
    return ds4_gpu_laguna_resident_tensor_alloc(tracker, 1u, 1u, &owner);
}
''',
    "c-mgpu-then-gpu": r'''
#include "ds4_runtime.h"
#include "ds4_gpu_mgpu.h"
#include "ds4_gpu.h"
#include "ds4_gpu_resident.h"
int resident_header_probe_c_reverse(ds4_runtime_tracker *tracker) {
    ds4_gpu_laguna_resident_tensor_owner owner = {0};
    return ds4_gpu_laguna_resident_tensor_free(tracker, &owner);
}
''',
    "cpp-gpu-then-mgpu": r'''
#include "ds4_runtime.h"
#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"
#include "ds4_gpu_resident.h"
int resident_header_probe_cpp(ds4_runtime_tracker *tracker) {
    ds4_gpu_laguna_resident_tensor_owner owner{};
    return ds4_gpu_laguna_resident_tensor_alloc(tracker, 1u, 1u, &owner);
}
''',
    "cpp-mgpu-then-gpu": r'''
#include "ds4_runtime.h"
#include "ds4_gpu_mgpu.h"
#include "ds4_gpu_resident.h"
#include "ds4_gpu.h"
int resident_header_probe_cpp_reverse(ds4_runtime_tracker *tracker) {
    ds4_gpu_laguna_resident_tensor_owner owner{};
    return ds4_gpu_laguna_resident_tensor_free(tracker, &owner);
}
''',
}



TENSOR_SUPPORT = r'''

#include "ds4_gpu_resident.h"

/* The resident ABI intentionally leaves the ordinary tensor layout alone. */
struct ds4_gpu_tensor {
    void *ptr;
    uint64_t bytes;
    int owner;
    int device_id;
};
struct ds4_gpu_ctx {
    int device_id;
    void *stream;
    void *cublas;
    int cublas_ready;
    void *scratch;
    size_t scratch_bytes;
    size_t budget_bytes;
    size_t used_bytes;
    void *boundary_event;
};
static ds4_gpu_ctx g_gpu[16];

static int fake_current_device = 7;
static int fake_device_query_calls;
static int fake_device_set_calls;
static int fake_device_sync_calls;
static int fake_managed_malloc_calls;
static bool fake_fail_device_query;
static bool fake_fail_device_set;
static bool fake_fail_device_sync;
static bool fake_fail_managed_malloc;
static bool fake_fail_calloc;
static bool fake_check_descriptor_order;
static void *fake_expected_descriptor;
static uint64_t fake_expected_device_id;
static int fake_calloc_calls;
static int fake_descriptor_free_calls;
static int fake_descriptor_order_errors;
static std::unordered_set<void *> fake_descriptor_live;
static std::unordered_set<void *> fake_nonheap_descriptor;

static void *fake_calloc(size_t count, size_t size) {
    ++fake_calloc_calls;
    if (fake_fail_calloc || (size != 0u && count > SIZE_MAX / size)) return nullptr;
    void *ptr = std::calloc(count, size);
    if (ptr) fake_descriptor_live.insert(ptr);
    return ptr;
}

static cudaError_t cudaFree(void *ptr) {
    if (fake_check_descriptor_order) {
        bool descriptor_live = false;
        if (fake_observed_tracker) {
            for (size_t i = 0; i < fake_observed_tracker->record_count; ++i) {
                const ds4_runtime_allocation_record &r = fake_observed_tracker->records[i];
                if (r.live && (r.id >> 56u) == 0x52u &&
                    r.base == (uint64_t)(uintptr_t)fake_expected_descriptor &&
                    r.domain == DS4_RUNTIME_DOMAIN_HOST) descriptor_live = true;
            }
        }
        if (!descriptor_live || !fake_descriptor_live.count(fake_expected_descriptor))
            ++fake_descriptor_order_errors;
    }
    return fake_raw_cudaFree(ptr);
}

static void fake_free(void *ptr) {
    if (!ptr) return;
    const auto descriptor = fake_descriptor_live.find(ptr);
    if (descriptor != fake_descriptor_live.end()) {
        ++fake_descriptor_free_calls;
        if (fake_check_descriptor_order) {
            bool live = false;
            if (fake_observed_tracker) {
                for (size_t i = 0; i < fake_observed_tracker->record_count; ++i) {
                    const ds4_runtime_allocation_record &record =
                        fake_observed_tracker->records[i];
                    if (record.id >> 56u == 0x52u &&
                        record.base == (uint64_t)(uintptr_t)ptr && record.live) {
                        live = true;
                        break;
                    }
                }
            }
            if (!live) ++fake_descriptor_order_errors;
            if (fake_observed_tracker) {
                for (size_t i = 0; i < fake_observed_tracker->record_count; ++i) {
                    const ds4_runtime_allocation_record &r = fake_observed_tracker->records[i];
                    if (r.id == fake_expected_device_id &&
                        (r.live || fake_live.count(reinterpret_cast<void *>((uintptr_t)r.base))))
                        ++fake_descriptor_order_errors;
                }
            }
        }
        fake_descriptor_live.erase(descriptor);
        std::free(ptr);
        return;
    }
    if (fake_nonheap_descriptor.count(ptr) != 0u) {
        ++fake_api_errors;
        return;
    }
    /* A raw CUDA handle must never reach C free.  Treat that as a protocol
     * error rather than passing a synthetic address to libc. */
    if (fake_live.count(ptr) != 0u) {
        ++fake_api_errors;
        return;
    }
    /* Unknown pointers are protocol errors, never passed to libc free. */
    ++fake_api_errors;
}

#define calloc(n, s) fake_calloc((n), (s))
#define free(p) fake_free((p))

static cudaError_t cudaGetDevice(int *out) {
    ++fake_device_query_calls;
    if (!out || fake_fail_device_query) {
        if (!out) ++fake_api_errors;
        if (out) *out = -1;
        return kFakeCudaError;
    }
    *out = fake_current_device;
    return cudaSuccess;
}
static cudaError_t cudaSetDevice(int device) {
    ++fake_device_set_calls;
    if (fake_fail_device_set || device < 0) {
        if (device < 0) ++fake_api_errors;
        return kFakeCudaError;
    }
    fake_current_device = device;
    return cudaSuccess;
}
static cudaError_t cudaDeviceSynchronize(void) {
    ++fake_device_sync_calls;
    return fake_fail_device_sync ? kFakeCudaError : cudaSuccess;
}
static cudaError_t cudaMallocManaged(void **out, size_t bytes) {
    ++fake_managed_malloc_calls;
    if (!out) {
        ++fake_api_errors;
        return kFakeCudaError;
    }
    *out = nullptr;
    if (fake_fail_managed_malloc) return kFakeCudaError;
    *out = fake_handle();
    fake_live[*out] = {bytes, false};
    return cudaSuccess;
}

/* Keep the extracted generic definitions source-compatible with ds4_cuda.cu. */
static int cuda_ok(cudaError_t result, const char *) {
    return result == cudaSuccess;
}
static inline int ds4_tensor_device_idx(const ds4_gpu_tensor *tensor) {
    if (!tensor) return 0;
    return tensor->device_id < 0 ? 0 : tensor->device_id;
}
static int cuda_current_tier(void) {
    int current = -1;
    if (g_n_gpus != 1 || cudaGetDevice(&current) != cudaSuccess) return -1;
    return current == g_gpu[0].device_id ? 0 : -1;
}
static int ds4_fake_set_current_device(int logical_tier) {
    if (logical_tier < 0 || logical_tier >= g_n_gpus) return 1;
    return cudaSetDevice(g_gpu[logical_tier].device_id) == cudaSuccess ? 0 : 1;
}

#define WITH_DEVICE(d)                                                      \
    for (int _wd_prev = -1, _wd_first = 1;                                  \
         _wd_first;                                                          \
         _wd_first = 0,                                                      \
         (_wd_prev >= 0 ? (void)cudaSetDevice(_wd_prev) : (void)0))         \
        if (cudaGetDevice(&_wd_prev) != cudaSuccess) { } else               \
        if (cudaSetDevice(d) != cudaSuccess) { } else

extern "C" int ds4_gpu_set_current_device(int logical_tier) {
    return ds4_fake_set_current_device(logical_tier);
}

/* Prototypes make the extracted real generic bodies independent of source
 * order while preserving their production signatures. */
extern "C" int ds4_gpu_tensor_alloc_on(ds4_gpu_tensor *, int, uint64_t);
extern "C" void ds4_gpu_tensor_free_in_place(ds4_gpu_tensor *);
extern "C" ds4_gpu_tensor *ds4_gpu_tensor_alloc(uint64_t);
extern "C" ds4_gpu_tensor *ds4_gpu_tensor_alloc_managed(uint64_t);
extern "C" ds4_gpu_tensor *ds4_gpu_tensor_alloc_ptr_on(int, uint64_t);
extern "C" ds4_gpu_tensor *ds4_gpu_tensor_alloc_managed_on(int, uint64_t);
extern "C" ds4_gpu_tensor *ds4_gpu_tensor_view(const ds4_gpu_tensor *, uint64_t, uint64_t);
extern "C" void ds4_gpu_tensor_free(ds4_gpu_tensor *);
extern "C" int ds4_gpu_tensor_device(const ds4_gpu_tensor *);

'''
TENSOR_SCENARIOS = r'''


static ds4_runtime_tracker tensor_tracker;
static ds4_runtime_tracker tensor_other_tracker;
static ds4_runtime_allocation_record tensor_records[32];
static ds4_runtime_allocation_record tensor_other_records[32];
static ds4_runtime_callsite tensor_callsites[8];
static ds4_runtime_tracker_config tensor_tracker_config;
static ds4_runtime_tracker_config tensor_other_config;

static void add_tensor_callsite(size_t *count, uint32_t id, const char *name,
                                ds4_runtime_category category,
                                ds4_runtime_physical_domain domain,
                                uint64_t bound) {
    tensor_callsites[*count] = {id, name, category, domain, bound};
    ++*count;
}

static int init_tensor_tracker(ds4_runtime_tracker *target,
                               ds4_runtime_tracker_config *config,
                               ds4_runtime_allocation_record *records,
                               size_t capacity,
                               uint64_t kv_bound,
                               uint64_t graph_bound,
                               uint64_t host_bound,
                               bool include_kv,
                               bool include_graph,
                               bool include_host,
                               bool wrong_kv,
                               bool wrong_graph) {
    std::memset(target, 0, sizeof(*target));
    std::memset(config, 0, sizeof(*config));
    std::memset(records, 0, 32u * sizeof(records[0]));
    std::memset(tensor_callsites, 0, sizeof(tensor_callsites));
    size_t count = 0;
    add_tensor_callsite(&count, DS4_LAGUNA_CALLSITE_STATIC_SLAB,
                        "tensor.static", DS4_RUNTIME_CATEGORY_STATIC_WEIGHTS,
                        DS4_RUNTIME_DOMAIN_CUDA_DEVICE, 64u);
    add_tensor_callsite(&count, DS4_LAGUNA_CALLSITE_OTHER_CUDA_KERNEL_TMP,
                        "tensor.other", DS4_RUNTIME_CATEGORY_OTHER_CUDA,
                        DS4_RUNTIME_DOMAIN_CUDA_DEVICE, 64u);
    add_tensor_callsite(&count, DS4_LAGUNA_CALLSITE_PINNED_STAGING_0,
                        "tensor.pinned", DS4_RUNTIME_CATEGORY_PINNED_STAGING,
                        DS4_RUNTIME_DOMAIN_HOST, 64u);
    if (include_kv) {
        add_tensor_callsite(&count, DS4_LAGUNA_CALLSITE_KV_STATE,
                            "tensor.kv", wrong_kv ? DS4_RUNTIME_CATEGORY_OTHER_HOST
                                                   : DS4_RUNTIME_CATEGORY_KV_STATE,
                            wrong_kv ? DS4_RUNTIME_DOMAIN_HOST
                                     : DS4_RUNTIME_DOMAIN_CUDA_DEVICE,
                            kv_bound);
    }
    if (include_graph) {
        add_tensor_callsite(&count, DS4_LAGUNA_CALLSITE_GRAPH_SCRATCH,
                            "tensor.graph", wrong_graph ? DS4_RUNTIME_CATEGORY_OTHER_HOST
                                                         : DS4_RUNTIME_CATEGORY_GRAPH_SCRATCH,
                            wrong_graph ? DS4_RUNTIME_DOMAIN_HOST
                                        : DS4_RUNTIME_DOMAIN_CUDA_DEVICE,
                            graph_bound);
    }
    /* The resident host inventory names this slot ENGINE + 4 (session). */
    if (include_host) {
        add_tensor_callsite(&count,
                            DS4_LAGUNA_CALLSITE_OTHER_HOST_ENGINE + 4u,
                            "tensor.other_host.session",
                            DS4_RUNTIME_CATEGORY_OTHER_HOST,
                            DS4_RUNTIME_DOMAIN_HOST, host_bound);
    }
    config->callsites = tensor_callsites;
    config->callsite_count = count;
    config->records = records;
    config->record_capacity = capacity;
    config->category_bounds[DS4_RUNTIME_CATEGORY_STATIC_WEIGHTS] = 64u;
    config->category_bounds[DS4_RUNTIME_CATEGORY_OTHER_CUDA] = 64u;
    config->category_bounds[DS4_RUNTIME_CATEGORY_PINNED_STAGING] = 64u;
    config->category_bounds[DS4_RUNTIME_CATEGORY_KV_STATE] =
        wrong_kv ? 0u : (include_kv ? kv_bound : 0u);
    config->category_bounds[DS4_RUNTIME_CATEGORY_GRAPH_SCRATCH] =
        wrong_graph ? 0u : (include_graph ? graph_bound : 0u);
    config->category_bounds[DS4_RUNTIME_CATEGORY_OTHER_HOST] =
        (include_host ? host_bound : 0u) + (wrong_kv ? kv_bound : 0u) +
        (wrong_graph ? graph_bound : 0u);
    config->report_bounds[DS4_RUNTIME_REPORT_MODEL_MAPPING_REGISTERED] =
        2u * sizeof(ds4_gpu_tensor);
    uint64_t owned_bound = 0u;
    for (size_t i = 0; i < DS4_RUNTIME_OWNED_CATEGORY_COUNT; ++i)
        owned_bound += config->category_bounds[i];
    config->owned_total_bound_bytes = owned_bound;
    config->qualification_total_bound_bytes = owned_bound;
    return ds4_runtime_tracker_init(target, config) == DS4_RUNTIME_STATUS_OK;
}

static void reset_tensor_case(size_t capacity = 8u,
                              uint64_t kv_bound = 64u,
                              uint64_t graph_bound = 64u,
                              uint64_t host_bound = 2u * sizeof(ds4_gpu_tensor),
                              bool include_kv = true,
                              bool include_graph = true,
                              bool include_host = true,
                              bool wrong_kv = false,
                              bool wrong_graph = false) {
    fake_reset();
    fake_current_device = 7;
    fake_device_query_calls = fake_device_set_calls = 0;
    fake_device_sync_calls = fake_managed_malloc_calls = 0;
    fake_fail_device_query = fake_fail_device_set = false;
    fake_fail_device_sync = fake_fail_managed_malloc = false;
    fake_fail_calloc = false;
    fake_check_descriptor_order = false;
    fake_calloc_calls = fake_descriptor_free_calls = 0;
    fake_descriptor_order_errors = 0;
    fake_descriptor_live.clear();
    fake_nonheap_descriptor.clear();
    std::memset(g_gpu, 0, sizeof(g_gpu));
    g_n_gpus = 1;
    g_gpu[0].device_id = 7;
    if (!init_tensor_tracker(&tensor_tracker, &tensor_tracker_config,
                             tensor_records, capacity, kv_bound, graph_bound,
                             host_bound, include_kv, include_graph, include_host,
                             wrong_kv, wrong_graph) ||
        !init_tensor_tracker(&tensor_other_tracker, &tensor_other_config,
                             tensor_other_records, capacity, kv_bound, graph_bound,
                             host_bound, include_kv, include_graph, include_host,
                             wrong_kv, wrong_graph)) {
        std::abort();
    }
    fake_observed_tracker = &tensor_tracker;
}

static int tensor_active_records(const ds4_runtime_tracker *target) {
    int count = 0;
    if (!target) return 0;
    for (size_t i = 0; i < target->record_count; ++i)
        if (target->records[i].live) ++count;
    return count;
}
static int tensor_live_physical(bool host) {
    int count = host ? (int)fake_descriptor_live.size() : 0;
    for (const auto &entry : fake_live)
        if (entry.second.host == host) ++count;
    return count;
}
static const ds4_runtime_allocation_record *tensor_record(
        const ds4_runtime_tracker *target, uint64_t id) {
    if (!target || id == 0u) return nullptr;
    for (size_t i = 0; i < target->record_count; ++i)
        if (target->records[i].id == id) return &target->records[i];
    return nullptr;
}
static void emit_tensor_owner(const char *label,
                              const ds4_gpu_laguna_resident_tensor_owner &owner,
                              bool inspect) {
    std::printf("%s_tensor=%llu\n%s_descriptor_id=%llu\n%s_device_id=%llu\n",
                label, (unsigned long long)(uintptr_t)owner.tensor,
                label, (unsigned long long)owner.descriptor_record_id,
                label, (unsigned long long)owner.device_record_id);
    if (inspect && owner.tensor &&
        fake_descriptor_live.count(owner.tensor) != 0u) {
        std::printf("%s_tensor_ptr=%llu\n%s_tensor_bytes=%llu\n"
                    "%s_tensor_owner=%d\n%s_tensor_device=%d\n",
                    label, (unsigned long long)(uintptr_t)owner.tensor->ptr,
                    label, (unsigned long long)owner.tensor->bytes,
                    label, owner.tensor->owner, label, owner.tensor->device_id);
    } else {
        std::printf("%s_tensor_ptr=0\n%s_tensor_bytes=0\n"
                    "%s_tensor_owner=0\n%s_tensor_device=-1\n",
                    label, label, label, label);
    }
}
static void emit_tensor_tracker(const char *label,
                                const ds4_runtime_tracker *target = &tensor_tracker) {
    std::printf("%s_violation=%d\n%s_record_count=%llu\n%s_active_records=%d\n"
                "%s_owned_current=%llu\n%s_owned_peak=%llu\n"
                "%s_qualification_current=%llu\n%s_qualification_peak=%llu\n"
                "%s_device_live=%d\n%s_host_live=%d\n"
                "%s_device_malloc_calls=%d\n%s_device_free_calls=%d\n"
                "%s_host_calloc_calls=%d\n%s_descriptor_free_calls=%d\n"
                "%s_device_query_calls=%d\n%s_device_sync_calls=%d\n"
                "%s_managed_malloc_calls=%d\n%s_api_errors=%d\n"
                "%s_descriptor_order_errors=%d\n",
                label, target ? (int)target->violation : -1,
                label, target ? (unsigned long long)target->record_count : 0u,
                label, tensor_active_records(target),
                label, target ? (unsigned long long)target->owned_total_current : 0u,
                label, target ? (unsigned long long)target->owned_total_peak : 0u,
                label, target ? (unsigned long long)target->qualification_total_current : 0u,
                label, target ? (unsigned long long)target->qualification_total_peak : 0u,
                label, tensor_live_physical(false),
                label, tensor_live_physical(true), label, fake_device_malloc_calls,
                label, fake_device_free_calls, label, fake_calloc_calls,
                label, fake_descriptor_free_calls, label, fake_device_query_calls,
                label, fake_device_sync_calls, label, fake_managed_malloc_calls,
                label, fake_api_errors, label, fake_descriptor_order_errors);
    for (int category = 0; category < DS4_RUNTIME_OWNED_CATEGORY_COUNT; ++category) {
        std::printf("%s_category_%d_current=%llu\n%s_category_%d_peak=%llu\n",
                    label, category,
                    target ? (unsigned long long)target->category_current[category] : 0u,
                    label, category,
                    target ? (unsigned long long)target->category_peak[category] : 0u);
    }
    if (!target) return;
    for (size_t i = 0; i < target->record_count; ++i) {
        const ds4_runtime_allocation_record &record = target->records[i];
        std::printf("%s_record_%llu_id=%llu\n%s_record_%llu_base=%llu\n"
                    "%s_record_%llu_requested=%llu\n%s_record_%llu_charged=%llu\n"
                    "%s_record_%llu_callsite=%u\n%s_record_%llu_category=%d\n"
                    "%s_record_%llu_domain=%d\n%s_record_%llu_relation=%d\n"
                    "%s_record_%llu_owner=%llu\n%s_record_%llu_live=%d\n",
                    label, (unsigned long long)i, (unsigned long long)record.id,
                    label, (unsigned long long)i, (unsigned long long)record.base,
                    label, (unsigned long long)i, (unsigned long long)record.requested_bytes,
                    label, (unsigned long long)i, (unsigned long long)record.charged_bytes,
                    label, (unsigned long long)i, record.callsite_id,
                    label, (unsigned long long)i, (int)record.category,
                    label, (unsigned long long)i, (int)record.domain,
                    label, (unsigned long long)i, (int)record.relation,
                    label, (unsigned long long)i, (unsigned long long)record.owner_id,
                    label, (unsigned long long)i, record.live ? 1 : 0);
    }
    std::fflush(stdout);
}
static int checked_tensor_free(
        ds4_gpu_laguna_resident_tensor_owner *owner) {
    fake_check_descriptor_order = true;
    fake_expected_descriptor = owner ? owner->tensor : nullptr;
    fake_expected_device_id = owner ? owner->device_record_id : 0u;
    const int result = ds4_gpu_laguna_resident_tensor_free(&tensor_tracker, owner);
    fake_check_descriptor_order = false;
    fake_expected_descriptor = nullptr;
    fake_expected_device_id = 0u;
    return result;
}
static void emit_handle_snapshot(const char *label,
                                 const ds4_gpu_laguna_resident_tensor_owner &owner) {
    emit_tensor_owner(label, owner, true);
}

static int scenario_tensor_success(void) {
    reset_tensor_case();
    ds4_gpu_laguna_resident_tensor_owner kv = {};
    ds4_gpu_laguna_resident_tensor_owner graph = {};
    const int began = ds4_gpu_laguna_resident_observer_begin(&tensor_tracker);
    const int kv_result = ds4_gpu_laguna_resident_tensor_alloc(
        &tensor_tracker, DS4_LAGUNA_CALLSITE_KV_STATE, 13u, &kv);
    const int graph_result = ds4_gpu_laguna_resident_tensor_alloc(
        &tensor_tracker, DS4_LAGUNA_CALLSITE_GRAPH_SCRATCH, 17u, &graph);
    std::printf("began=%d\nkv_result=%d\ngraph_result=%d\n"
                "descriptor_bytes=%llu\n",
                began, kv_result, graph_result,
                (unsigned long long)sizeof(ds4_gpu_tensor));
    emit_tensor_owner("kv_after", kv, true);
    emit_tensor_owner("graph_after", graph, true);
    emit_tensor_tracker("after");
    const int kv_free = checked_tensor_free(&kv);
    const int graph_free = checked_tensor_free(&graph);
    const int ended = ds4_gpu_laguna_resident_observer_end(&tensor_tracker);
    std::printf("kv_free=%d\ngraph_free=%d\nended=%d\n", kv_free, graph_free, ended);
    emit_tensor_owner("kv_final", kv, false);
    emit_tensor_owner("graph_final", graph, false);
    emit_tensor_tracker("final");
    return 0;
}

static int scenario_tensor_free_retry(const char *failure) {
    reset_tensor_case();
    ds4_gpu_laguna_resident_tensor_owner owner = {};
    const int began = ds4_gpu_laguna_resident_observer_begin(&tensor_tracker);
    const int allocated = ds4_gpu_laguna_resident_tensor_alloc(
        &tensor_tracker, DS4_LAGUNA_CALLSITE_KV_STATE, 13u, &owner);
    const uint64_t tensor_before = (uint64_t)(uintptr_t)owner.tensor;
    const uint64_t descriptor_before = owner.descriptor_record_id;
    const uint64_t device_before = owner.device_record_id;
    if (std::strcmp(failure, "device") == 0) fake_fail_device_free = true;
    if (std::strcmp(failure, "query") == 0) fake_fail_device_query = true;
    if (std::strcmp(failure, "sync") == 0) fake_fail_device_sync = true;
    const int first = checked_tensor_free(&owner);
    emit_tensor_owner("after_first", owner, true);
    const int after_first_device_live = tensor_live_physical(false);
    const int after_first_host_live = tensor_live_physical(true);
    const int after_first_active = tensor_active_records(&tensor_tracker);
    const int after_first_calloc_free = fake_descriptor_free_calls;
    fake_fail_device_free = false;
    fake_fail_device_query = false;
    fake_fail_device_sync = false;
    const int second = checked_tensor_free(&owner);
    const int ended = ds4_gpu_laguna_resident_observer_end(&tensor_tracker);
    std::printf("began=%d\nallocated=%d\nfirst=%d\nsecond=%d\nended=%d\n"
                "tensor_before=%llu\ndescriptor_before=%llu\n"
                "device_before=%llu\nafter_first_device_live=%d\n"
                "after_first_host_live=%d\nafter_first_active=%d\n"
                "after_first_descriptor_free_calls=%d\n",
                began, allocated, first, second, ended,
                (unsigned long long)tensor_before,
                (unsigned long long)descriptor_before,
                (unsigned long long)device_before,
                after_first_device_live, after_first_host_live,
                after_first_active, after_first_calloc_free);
    emit_tensor_owner("owner_final", owner, false);
    emit_tensor_tracker("final");
    return 0;
}

static int scenario_tensor_end_live(void) {
    reset_tensor_case();
    ds4_gpu_laguna_resident_tensor_owner owner = {};
    const int began = ds4_gpu_laguna_resident_observer_begin(&tensor_tracker);
    const int allocated = ds4_gpu_laguna_resident_tensor_alloc(
        &tensor_tracker, DS4_LAGUNA_CALLSITE_KV_STATE, 13u, &owner);
    const int while_live = ds4_gpu_laguna_resident_observer_end(&tensor_tracker);
    const int freed = checked_tensor_free(&owner);
    const int after_free = ds4_gpu_laguna_resident_observer_end(&tensor_tracker);
    std::printf("began=%d\nallocated=%d\nwhile_live=%d\nfreed=%d\n"
                "after_free=%d\n", began, allocated, while_live, freed, after_free);
    emit_tensor_tracker("final");
    return 0;
}

static int scenario_tensor_host_alloc_failure(void) {
    reset_tensor_case();
    ds4_gpu_laguna_resident_tensor_owner owner = {};
    const int began = ds4_gpu_laguna_resident_observer_begin(&tensor_tracker);
    fake_fail_calloc = true;
    const int result = ds4_gpu_laguna_resident_tensor_alloc(
        &tensor_tracker, DS4_LAGUNA_CALLSITE_KV_STATE, 13u, &owner);
    std::printf("began=%d\nresult=%d\n", began, result);
    emit_tensor_owner("owner", owner, false);
    emit_tensor_tracker("after");
    const int ended = ds4_gpu_laguna_resident_observer_end(&tensor_tracker);
    std::printf("ended=%d\n", ended);
    return 0;
}

static int scenario_tensor_cuda_failure(bool partial, bool rollback_fails) {
    reset_tensor_case();
    ds4_gpu_laguna_resident_tensor_owner owner = {};
    const int began = ds4_gpu_laguna_resident_observer_begin(&tensor_tracker);
    fake_fail_device_malloc = !partial;
    fake_partial_driver_failure = partial;
    fake_fail_device_free = rollback_fails;
    const int result = ds4_gpu_laguna_resident_tensor_alloc(
        &tensor_tracker, DS4_LAGUNA_CALLSITE_KV_STATE, 13u, &owner);
    const int ended_first = ds4_gpu_laguna_resident_observer_end(&tensor_tracker);
    std::printf("began=%d\nresult=%d\nended_first=%d\nrollback_fails=%d\n",
                began, result, ended_first, rollback_fails ? 1 : 0);
    emit_tensor_owner("owner", owner, false);
    emit_tensor_tracker("after");
    fake_fail_device_free = false;
    const int ended_second = rollback_fails
        ? ds4_gpu_laguna_resident_observer_end(&tensor_tracker) : ended_first;
    std::printf("ended_second=%d\n", ended_second);
    emit_tensor_tracker("final");
    return 0;
}

static int scenario_tensor_bound_failure(bool rollback_fails) {
    reset_tensor_case(8u, 1u, 64u, 2u * sizeof(ds4_gpu_tensor));
    ds4_gpu_laguna_resident_tensor_owner owner = {};
    const int began = ds4_gpu_laguna_resident_observer_begin(&tensor_tracker);
    fake_fail_device_free = rollback_fails;
    const int result = ds4_gpu_laguna_resident_tensor_alloc(
        &tensor_tracker, DS4_LAGUNA_CALLSITE_KV_STATE, 8u, &owner);
    const int ended_first = ds4_gpu_laguna_resident_observer_end(&tensor_tracker);
    std::printf("began=%d\nresult=%d\nended_first=%d\nrollback_fails=%d\n",
                began, result, ended_first, rollback_fails ? 1 : 0);
    emit_tensor_owner("owner", owner, false);
    emit_tensor_tracker("after");
    fake_fail_device_free = false;
    const int ended_second = rollback_fails
        ? ds4_gpu_laguna_resident_observer_end(&tensor_tracker) : ended_first;
    std::printf("ended_second=%d\n", ended_second);
    emit_tensor_tracker("final");
    return 0;
}

static int scenario_tensor_capacity1(void) {
    reset_tensor_case(1u);
    ds4_gpu_laguna_resident_tensor_owner owner = {};
    const int began = ds4_gpu_laguna_resident_observer_begin(&tensor_tracker);
    const int result = ds4_gpu_laguna_resident_tensor_alloc(
        &tensor_tracker, DS4_LAGUNA_CALLSITE_KV_STATE, 8u, &owner);
    std::printf("began=%d\nresult=%d\n", began, result);
    emit_tensor_owner("owner", owner, false);
    emit_tensor_tracker("after");
    return 0;
}

static int scenario_tensor_id_boundary(bool one_remaining) {
    reset_tensor_case();
    ds4_gpu_laguna_resident_tensor_owner owner = {};
    const int began = ds4_gpu_laguna_resident_observer_begin(&tensor_tracker);
    tensor_tracker.issued_sequence_high_water[0x52u] = one_remaining
        ? UINT64_C(0x00fffffffffffffe) : UINT64_C(0x00fffffffffffffd);
    const int result = ds4_gpu_laguna_resident_tensor_alloc(
        &tensor_tracker, DS4_LAGUNA_CALLSITE_KV_STATE, 8u, &owner);
    std::printf("began=%d\nresult=%d\none_remaining=%d\n", began, result,
                one_remaining ? 1 : 0);
    emit_tensor_owner("owner", owner, false);
    emit_tensor_tracker("after");
    if (!one_remaining && owner.tensor) {
        (void)checked_tensor_free(&owner);
        (void)ds4_gpu_laguna_resident_observer_end(&tensor_tracker);
    }
    return 0;
}

static int scenario_tensor_tombstone_reuse(void) {
    reset_tensor_case(2u);
    ds4_gpu_laguna_resident_tensor_owner first = {};
    ds4_gpu_laguna_resident_tensor_owner second = {};
    const int began = ds4_gpu_laguna_resident_observer_begin(&tensor_tracker);
    const int first_alloc = ds4_gpu_laguna_resident_tensor_alloc(
        &tensor_tracker, DS4_LAGUNA_CALLSITE_KV_STATE, 8u, &first);
    const uint64_t first_desc = first.descriptor_record_id;
    const uint64_t first_device = first.device_record_id;
    const int first_free = checked_tensor_free(&first);
    const uint64_t record_count_after_first = tensor_tracker.record_count;
    const int second_alloc = ds4_gpu_laguna_resident_tensor_alloc(
        &tensor_tracker, DS4_LAGUNA_CALLSITE_GRAPH_SCRATCH, 9u, &second);
    const uint64_t second_desc = second.descriptor_record_id;
    const uint64_t second_device = second.device_record_id;
    const int second_free = checked_tensor_free(&second);
    const int ended = ds4_gpu_laguna_resident_observer_end(&tensor_tracker);
    std::printf("began=%d\nfirst_alloc=%d\nfirst_free=%d\nsecond_alloc=%d\n"
                "second_free=%d\nended=%d\nrecord_count_after_first=%llu\n"
                "first_desc=%llu\nsecond_desc=%llu\nfirst_device=%llu\n"
                "second_device=%llu\n",
                began, first_alloc, first_free, second_alloc, second_free, ended,
                (unsigned long long)record_count_after_first,
                (unsigned long long)first_desc, (unsigned long long)second_desc,
                (unsigned long long)first_device, (unsigned long long)second_device);
    emit_tensor_tracker("final");
    return 0;
}

static int scenario_tensor_bad_site(const char *which) {
    const bool missing_kv = std::strcmp(which, "missing-kv") == 0;
    const bool missing_graph = std::strcmp(which, "missing-graph") == 0;
    const bool missing_host = std::strcmp(which, "missing-host") == 0;
    const bool wrong_kv = std::strcmp(which, "wrong-kv") == 0;
    const bool wrong_graph = std::strcmp(which, "wrong-graph") == 0;
    reset_tensor_case(8u, 64u, 64u, 2u * sizeof(ds4_gpu_tensor),
                      !missing_kv, !missing_graph, !missing_host,
                      wrong_kv, wrong_graph);
    ds4_gpu_laguna_resident_tensor_owner owner = {};
    const int began = ds4_gpu_laguna_resident_observer_begin(&tensor_tracker);
    const uint32_t callsite = missing_graph || wrong_graph
        ? DS4_LAGUNA_CALLSITE_GRAPH_SCRATCH : DS4_LAGUNA_CALLSITE_KV_STATE;
    const int result = ds4_gpu_laguna_resident_tensor_alloc(
        &tensor_tracker, callsite, 8u, &owner);
    std::printf("began=%d\nresult=%d\n", began, result);
    emit_tensor_owner("owner", owner, false);
    emit_tensor_tracker("after");
    return 0;
}

static int scenario_tensor_wrong_runtime_context(const char *which) {
    reset_tensor_case();
    ds4_gpu_laguna_resident_tensor_owner owner = {};
    const int began = ds4_gpu_laguna_resident_observer_begin(&tensor_tracker);
    if (std::strcmp(which, "wrong-tracker") == 0) {
        const int result = ds4_gpu_laguna_resident_tensor_alloc(
            &tensor_other_tracker, DS4_LAGUNA_CALLSITE_KV_STATE, 8u, &owner);
        std::printf("began=%d\nresult=%d\n", began, result);
        emit_tensor_tracker("attached");
        emit_tensor_tracker("other", &tensor_other_tracker);
        return 0;
    }
    if (std::strcmp(which, "query") == 0) fake_fail_device_query = true;
    if (std::strcmp(which, "device") == 0) fake_current_device = 99;
    if (std::strcmp(which, "no-gpu") == 0) g_n_gpus = 0;
    const int result = ds4_gpu_laguna_resident_tensor_alloc(
        &tensor_tracker, DS4_LAGUNA_CALLSITE_KV_STATE, 8u, &owner);
    std::printf("began=%d\nresult=%d\n", began, result);
    emit_tensor_owner("owner", owner, false);
    emit_tensor_tracker("after");
    return 0;
}

static int scenario_tensor_invalid_input(void) {
    reset_tensor_case();
    ds4_gpu_laguna_resident_tensor_owner owner = {};
    const int began = ds4_gpu_laguna_resident_observer_begin(&tensor_tracker);
    const int zero = ds4_gpu_laguna_resident_tensor_alloc(
        &tensor_tracker, DS4_LAGUNA_CALLSITE_KV_STATE, 0u, &owner);
    ds4_gpu_laguna_resident_tensor_owner sentinel = {};
    sentinel.tensor = reinterpret_cast<ds4_gpu_tensor *>(
        static_cast<uintptr_t>(0xdead0000u));
    sentinel.descriptor_record_id = UINT64_C(0x1111222233334444);
    sentinel.device_record_id = UINT64_C(0x5555666677778888);
    const int nonempty = ds4_gpu_laguna_resident_tensor_alloc(
        &tensor_tracker, DS4_LAGUNA_CALLSITE_KV_STATE, 8u, &sentinel);
    const int null_out = ds4_gpu_laguna_resident_tensor_alloc(
        &tensor_tracker, DS4_LAGUNA_CALLSITE_KV_STATE, 8u, nullptr);
    const int null_tracker = ds4_gpu_laguna_resident_tensor_alloc(
        nullptr, DS4_LAGUNA_CALLSITE_KV_STATE, 8u, &owner);
    std::printf("began=%d\nzero=%d\nnonempty=%d\nnull_out=%d\n"
                "null_tracker=%d\n", began, zero, nonempty, null_out, null_tracker);
    emit_tensor_owner("sentinel", sentinel, false);
    emit_tensor_owner("owner", owner, false);
    emit_tensor_tracker("after");
    return 0;
}

static int scenario_tensor_registration(bool gpu_registration) {
    reset_tensor_case(8u);
    ds4_gpu_laguna_resident_tensor_owner owner = {};
    const int began = ds4_gpu_laguna_resident_observer_begin(&tensor_tracker);
    const int allocated = ds4_gpu_laguna_resident_tensor_alloc(
        &tensor_tracker, DS4_LAGUNA_CALLSITE_KV_STATE, 13u, &owner);
    const ds4_runtime_allocation_record *descriptor = tensor_record(
        &tensor_tracker, owner.descriptor_record_id);
    const ds4_runtime_allocation_record *device = tensor_record(
        &tensor_tracker, owner.device_record_id);
    const uint64_t relation_base = gpu_registration
        ? (device ? device->base : 0u)
        : (uint64_t)(uintptr_t)owner.tensor;
    const uint64_t relation_bytes = gpu_registration
        ? (device ? device->requested_bytes : 0u)
        : (uint64_t)sizeof(ds4_gpu_tensor);
    const uint64_t relation_owner = gpu_registration
        ? owner.device_record_id : owner.descriptor_record_id;
    const ds4_runtime_status related = ds4_runtime_tracker_register(
        &tensor_tracker, UINT64_C(0x5300000000000001), relation_base,
        relation_bytes, relation_owner);
    const uint64_t owned_before = tensor_tracker.owned_total_current;
    const uint64_t qualification_before = tensor_tracker.qualification_total_current;
    const int record_count_before_free = tensor_tracker.record_count;
    const int failed = checked_tensor_free(&owner);
    std::printf("began=%d\nallocated=%d\nrelated=%d\n"
                "owned_before=%llu\nqualification_before=%llu\n"
                "record_count_before_free=%d\nfailed=%d\n",
                began, allocated, (int)related,
                (unsigned long long)owned_before,
                (unsigned long long)qualification_before,
                record_count_before_free, failed);
    emit_tensor_owner("owner_after_failed_free", owner, true);
    emit_tensor_tracker("after_failed_free");
    int unregistered = 0;
    if (!gpu_registration && related == DS4_RUNTIME_STATUS_OK) {
        unregistered = (int)ds4_runtime_tracker_unregister(
            &tensor_tracker, UINT64_C(0x5300000000000001));
    }
    const ds4_runtime_allocation_record *relation_after = tensor_record(
        &tensor_tracker, UINT64_C(0x5300000000000001));
    std::printf("relation_live_after_unregister=%d\n",
                relation_after && relation_after->live ? 1 : 0);
    const int freed = checked_tensor_free(&owner);
    const int ended = ds4_gpu_laguna_resident_observer_end(&tensor_tracker);
    std::printf("unregistered=%d\nfreed=%d\nended=%d\n", unregistered, freed, ended);
    emit_tensor_tracker("final");
    return 0;
}

static int scenario_tensor_handle_integrity(const char *which) {
    reset_tensor_case(8u);
    ds4_gpu_laguna_resident_tensor_owner first = {};
    ds4_gpu_laguna_resident_tensor_owner second = {};
    const int began = ds4_gpu_laguna_resident_observer_begin(&tensor_tracker);
    const int first_result = ds4_gpu_laguna_resident_tensor_alloc(
        &tensor_tracker, DS4_LAGUNA_CALLSITE_KV_STATE, 8u, &first);
    const int second_result = ds4_gpu_laguna_resident_tensor_alloc(
        &tensor_tracker, DS4_LAGUNA_CALLSITE_GRAPH_SCRATCH, 9u, &second);
    ds4_gpu_laguna_resident_tensor_owner bad = first;
    ds4_runtime_allocation_record *descriptor_record = const_cast<ds4_runtime_allocation_record *>(
        tensor_record(&tensor_tracker, first.descriptor_record_id));
    ds4_runtime_allocation_record *device_record = const_cast<ds4_runtime_allocation_record *>(
        tensor_record(&tensor_tracker, first.device_record_id));
    if (!descriptor_record || !device_record) return 3;
    const ds4_runtime_allocation_record descriptor_saved = *descriptor_record;
    const ds4_runtime_allocation_record device_saved = *device_record;
    if (std::strcmp(which, "stale") == 0) {
        const int first_free = checked_tensor_free(&first);
        const int bad_result = ds4_gpu_laguna_resident_tensor_free(
            &tensor_tracker, &bad);
        std::printf("began=%d\nfirst_result=%d\nsecond_result=%d\n"
                    "bad_result=%d\n", began, first_result, second_result,
                    bad_result);
        emit_tensor_owner("bad", bad, false);
        emit_tensor_tracker("after_bad");
        const int second_free = checked_tensor_free(&second);
        const int ended = ds4_gpu_laguna_resident_observer_end(&tensor_tracker);
        std::printf("first_free=%d\nsecond_free=%d\nended=%d\n",
                    first_free, second_free, ended);
        emit_tensor_tracker("final");
        return 0;
    }
    if (std::strcmp(which, "mismatched") == 0) {
        bad.device_record_id = second.device_record_id;
    } else if (std::strcmp(which, "partial-descriptor") == 0) {
        bad.descriptor_record_id = 0u;
    } else if (std::strcmp(which, "partial-device") == 0) {
        bad.device_record_id = 0u;
    } else if (std::strcmp(which, "partial-tensor") == 0) {
        bad.tensor = nullptr;
    } else if (std::strcmp(which, "wrong-record-descriptor") == 0) {
        descriptor_record->requested_bytes += 1u;
    } else if (std::strcmp(which, "wrong-record-device") == 0) {
        device_record->charged_bytes += 1u;
    } else if (std::strcmp(which, "wrong-record-site") == 0) {
        device_record->callsite_id = DS4_LAGUNA_CALLSITE_STATIC_SLAB;
    } else if (std::strcmp(which, "wrong-record-domain") == 0) {
        device_record->domain = DS4_RUNTIME_DOMAIN_HOST;
    } else if (std::strcmp(which, "wrong-bytes") == 0) {
        bad.tensor->bytes += 1u;
    } else if (std::strcmp(which, "wrong-owner") == 0) {
        bad.tensor->owner = 0;
    } else if (std::strcmp(which, "wrong-device") == 0) {
        bad.tensor->device_id = 1;
    } else if (std::strcmp(which, "wrong-ptr") == 0) {
        bad.tensor->ptr = reinterpret_cast<void *>(static_cast<uintptr_t>(0xbeef0000u));
    }
    const int bad_result = ds4_gpu_laguna_resident_tensor_free(
        &tensor_tracker, &bad);
    std::printf("began=%d\nfirst_result=%d\nsecond_result=%d\n"
                "bad_result=%d\n", began, first_result, second_result,
                bad_result);
    emit_tensor_owner("bad", bad, true);
    emit_tensor_tracker("after_bad");
    if (std::strcmp(which, "wrong-bytes") == 0) bad.tensor->bytes -= 1u;
    if (std::strcmp(which, "wrong-owner") == 0) bad.tensor->owner = 1;
    if (std::strcmp(which, "wrong-device") == 0) bad.tensor->device_id = 0;
    if (std::strcmp(which, "wrong-ptr") == 0) {
        const ds4_runtime_allocation_record *record = tensor_record(
            &tensor_tracker, first.device_record_id);
        bad.tensor->ptr = record ? reinterpret_cast<void *>(
            static_cast<uintptr_t>(record->base)) : nullptr;
    }
    *descriptor_record = descriptor_saved;
    *device_record = device_saved;
    const int first_free = checked_tensor_free(&first);
    const int second_free = checked_tensor_free(&second);
    const int ended = ds4_gpu_laguna_resident_observer_end(&tensor_tracker);
    std::printf("first_free=%d\nsecond_free=%d\nended=%d\n",
                first_free, second_free, ended);
    emit_tensor_tracker("final");
    return 0;
}

static int scenario_generic_active(const char *which) {
    reset_tensor_case();
    const int began = ds4_gpu_laguna_resident_observer_begin(&tensor_tracker);
    ds4_gpu_tensor stack_alloc = {
        reinterpret_cast<void *>(static_cast<uintptr_t>(0xcafe0000u)), 41u, 1, 0};
    ds4_gpu_tensor stack_in_place = stack_alloc;
    ds4_gpu_tensor stack_view = stack_alloc;
    fake_nonheap_descriptor.insert(&stack_alloc);
    fake_nonheap_descriptor.insert(&stack_in_place);
    ds4_gpu_tensor *heap = nullptr;
    ds4_gpu_tensor *managed = nullptr;
    ds4_gpu_tensor *tiered = nullptr;
    ds4_gpu_tensor *managed_tiered = nullptr;
    ds4_gpu_tensor *view = nullptr;
    int alloc_on_result = 1;
    if (std::strcmp(which, "alloc") == 0) heap = ds4_gpu_tensor_alloc(8u);
    if (std::strcmp(which, "managed") == 0) managed = ds4_gpu_tensor_alloc_managed(8u);
    if (std::strcmp(which, "tiered") == 0) tiered = ds4_gpu_tensor_alloc_ptr_on(0, 8u);
    if (std::strcmp(which, "managed-tiered") == 0)
        managed_tiered = ds4_gpu_tensor_alloc_managed_on(0, 8u);
    if (std::strcmp(which, "alloc-on") == 0)
        alloc_on_result = ds4_gpu_tensor_alloc_on(&stack_alloc, 0, 8u);
    if (std::strcmp(which, "view") == 0) view = ds4_gpu_tensor_view(&stack_view, 0u, 8u);
    if (std::strcmp(which, "free") == 0) ds4_gpu_tensor_free(&stack_alloc);
    if (std::strcmp(which, "in-place") == 0) ds4_gpu_tensor_free_in_place(&stack_in_place);
    std::printf("began=%d\nheap=%llu\nmanaged=%llu\ntiered=%llu\n"
                "managed_tiered=%llu\nview=%llu\nalloc_on_result=%d\n"
                "stack_alloc_ptr=%llu\nstack_alloc_bytes=%llu\n"
                "stack_alloc_owner=%d\nstack_alloc_device=%d\n"
                "stack_in_place_ptr=%llu\nstack_in_place_bytes=%llu\n"
                "stack_in_place_owner=%d\nstack_in_place_device=%d\n",
                began, (unsigned long long)(uintptr_t)heap,
                (unsigned long long)(uintptr_t)managed,
                (unsigned long long)(uintptr_t)tiered,
                (unsigned long long)(uintptr_t)managed_tiered,
                (unsigned long long)(uintptr_t)view, alloc_on_result,
                (unsigned long long)(uintptr_t)stack_alloc.ptr,
                (unsigned long long)stack_alloc.bytes, stack_alloc.owner,
                stack_alloc.device_id,
                (unsigned long long)(uintptr_t)stack_in_place.ptr,
                (unsigned long long)stack_in_place.bytes, stack_in_place.owner,
                stack_in_place.device_id);
    emit_tensor_tracker("after");
    const int ended = ds4_gpu_laguna_resident_observer_end(&tensor_tracker);
    std::printf("ended=%d\n", ended);
    return 0;
}

static int scenario_generic_null_frees(void) {
    reset_tensor_case();
    const int began = ds4_gpu_laguna_resident_observer_begin(&tensor_tracker);
    ds4_gpu_tensor_free(nullptr);
    ds4_gpu_tensor_free_in_place(nullptr);
    const int ended = ds4_gpu_laguna_resident_observer_end(&tensor_tracker);
    std::printf("began=%d\nended=%d\n", began, ended);
    emit_tensor_tracker("after");
    return 0;
}

static int scenario_generic_raw(void) {
    reset_tensor_case();
    ds4_gpu_tensor stack = {};
    const int stack_result = ds4_gpu_tensor_alloc_on(&stack, 0, 8u);
    const int stack_ptr_nonzero = stack.ptr != nullptr;
    ds4_gpu_tensor_free_in_place(&stack);
    ds4_gpu_tensor *heap = ds4_gpu_tensor_alloc(8u);
    ds4_gpu_tensor *managed = ds4_gpu_tensor_alloc_managed(8u);
    ds4_gpu_tensor *tiered = ds4_gpu_tensor_alloc_ptr_on(0, 8u);
    ds4_gpu_tensor *managed_tiered = ds4_gpu_tensor_alloc_managed_on(0, 8u);
    ds4_gpu_tensor view_base = {};
    const int view_base_result = ds4_gpu_tensor_alloc_on(&view_base, 0, 8u);
    ds4_gpu_tensor *view = ds4_gpu_tensor_view(&view_base, 0u, 8u);
    ds4_gpu_tensor_free(view);
    ds4_gpu_tensor_free_in_place(&view_base);
    ds4_gpu_tensor_free(heap);
    ds4_gpu_tensor_free(managed);
    ds4_gpu_tensor_free(tiered);
    ds4_gpu_tensor_free(managed_tiered);
    std::printf("stack_result=%d\nstack_ptr_nonzero=%d\nheap_nonnull=%d\n"
                "managed_nonnull=%d\ntiered_nonnull=%d\nmanaged_tiered_nonnull=%d\n"
                "view_base_result=%d\nview_nonnull=%d\n",
                stack_result, stack_ptr_nonzero, heap != nullptr,
                managed != nullptr, tiered != nullptr, managed_tiered != nullptr,
                view_base_result, view != nullptr);
    emit_tensor_tracker("after");
    return 0;
}

static int scenario_legacy_before_begin(void) {
    reset_tensor_case();
    ds4_gpu_tensor stack = {};
    const int allocated = ds4_gpu_tensor_alloc_on(&stack, 0, 8u);
    ds4_gpu_tensor_free_in_place(&stack);
    const int began = ds4_gpu_laguna_resident_observer_begin(&tensor_tracker);
    std::printf("allocated=%d\nbegan=%d\n", allocated, began);
    emit_tensor_tracker("after");
    return 0;
}

'''
TENSOR_MAIN = r'''


int main(int argc, char **argv) {
    alarm(15);
    struct rlimit limit = {0, 0};
    (void)setrlimit(RLIMIT_CORE, &limit);
    if (argc != 2) return 2;
    const std::string name(argv[1]);
    if (name == "success") return scenario_tensor_success();
    if (name == "free-device") return scenario_tensor_free_retry("device");
    if (name == "free-query") return scenario_tensor_free_retry("query");
    if (name == "free-sync") return scenario_tensor_free_retry("sync");
    if (name == "end-live") return scenario_tensor_end_live();
    if (name == "host-fail") return scenario_tensor_host_alloc_failure();
    if (name == "cuda-fail-null") return scenario_tensor_cuda_failure(false, false);
    if (name == "cuda-fail-safe") return scenario_tensor_cuda_failure(true, false);
    if (name == "cuda-fail-retained") return scenario_tensor_cuda_failure(true, true);
    if (name == "bound-safe") return scenario_tensor_bound_failure(false);
    if (name == "bound-retained") return scenario_tensor_bound_failure(true);
    if (name == "capacity1") return scenario_tensor_capacity1();
    if (name == "id-one") return scenario_tensor_id_boundary(true);
    if (name == "id-two") return scenario_tensor_id_boundary(false);
    if (name == "tombstone") return scenario_tensor_tombstone_reuse();
    if (name == "missing-kv" || name == "missing-graph" ||
        name == "missing-host" || name == "wrong-kv" ||
        name == "wrong-graph") return scenario_tensor_bad_site(name.c_str());
    if (name == "missing-observer") {
        reset_tensor_case();
        ds4_gpu_laguna_resident_tensor_owner owner = {};
        const int result = ds4_gpu_laguna_resident_tensor_alloc(
            &tensor_tracker, DS4_LAGUNA_CALLSITE_KV_STATE, 8u, &owner);
        std::printf("result=%d\n", result);
        emit_tensor_owner("owner", owner, false);
        emit_tensor_tracker("after");
        return 0;
    }
    if (name == "wrong-tracker" || name == "query" || name == "device" ||
        name == "no-gpu") return scenario_tensor_wrong_runtime_context(name.c_str());
    if (name == "invalid") return scenario_tensor_invalid_input();
    if (name == "host-registration") return scenario_tensor_registration(false);
    if (name == "gpu-registration") return scenario_tensor_registration(true);
    if (name == "stale" || name == "mismatched" ||
        name == "partial-descriptor" || name == "partial-device" ||
        name == "partial-tensor" || name == "wrong-bytes" ||
        name == "wrong-owner" || name == "wrong-device" ||
        name == "wrong-ptr" || name == "wrong-record-descriptor" ||
        name == "wrong-record-device" || name == "wrong-record-site" ||
        name == "wrong-record-domain") return scenario_tensor_handle_integrity(name.c_str());
    if (name.rfind("generic-active-", 0) == 0)
        return scenario_generic_active(name.c_str() + std::strlen("generic-active-"));
    if (name == "generic-null") return scenario_generic_null_frees();
    if (name == "generic-raw") return scenario_generic_raw();
    if (name == "legacy-before-begin") return scenario_legacy_before_begin();
    return 2;
}

'''

# The resident implementation is the only new native block.  Generic entry
# points come from the real CUDA source and remain separate guards in this
# fixture; no observer/runtime body is copied here.
assert FAKE_PREFIX.count("static cudaError_t cudaFree(void *ptr) {") == 1
TENSOR_FAKE_PREFIX = FAKE_PREFIX.replace(
    "static cudaError_t cudaFree(void *ptr) {",
    "static cudaError_t fake_raw_cudaFree(void *ptr) {",
)
FIXTURE_SOURCE = (
    TENSOR_FAKE_PREFIX
    + "\n"
    + TENSOR_SUPPORT
    + "\n"
    + TMP_DECLS
    + "\n"
    + (OBSERVER_BLOCK or "")
    + "\n"
    + (TENSOR_BLOCK or "")
    + "\n"
    + GENERIC_SOURCE
    + "\n"
    + TENSOR_SCENARIOS
    + "\n"
    + TENSOR_MAIN
)

ABI_PROBE = r'''
#include "ds4_runtime.h"
#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"
#include "ds4_gpu_resident.h"
int resident_tensor_abi_probe(ds4_runtime_tracker *tracker) {
    ds4_gpu_laguna_resident_tensor_owner owner = {0};
    return ds4_gpu_laguna_resident_tensor_alloc(tracker, 1u, 1u, &owner);
}
'''


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


def _make_logical_make_rules(source: str) -> list[tuple[str, str]]:
    logical: list[str] = []
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
        logical.append(pending)
        pending = ""
    if pending:
        logical.append(pending)
    rules: list[tuple[str, str]] = []
    for line in logical:
        if not line or line[0].isspace() or line.startswith("#"):
            continue
        target, separator, deps = line.partition(":")
        if not separator or target.startswith(("if ", "ifdef ", "ifndef ", "ifeq ", "ifneq ")):
            continue
        tokens = deps.split()
        if "ds4_gpu.h" in tokens or "ds4_gpu_mgpu.h" in tokens:
            rules.append((target.strip(), deps))
    return rules


class ResidentTensorFixture(unittest.TestCase):
    '''Compile and execute only generated fake-driver children.'''

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory(prefix="ds4-native-tensor-owner-")
        cls.addClassCleanup(cls._tmp.cleanup)
        base = Path(cls._tmp.name)
        cls._env = {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"}
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
        cls._source = base / "tensor_fixture.cc"
        cls._source.write_text(FIXTURE_SOURCE, encoding="utf-8")
        cls._runtime_object = base / "ds4_runtime.o"
        cls._binary = base / "tensor_fixture"
        cls._runtime_compile = subprocess.run(
            ["cc", "-std=c11", "-O0", "-I", str(ROOT), "-c", str(RUNTIME_SOURCE),
             "-o", str(cls._runtime_object)],
            cwd=base, env=cls._env, capture_output=True, text=True,
            timeout=15, check=False,
        )
        cls._header_compiles: list[tuple[str, subprocess.CompletedProcess[str]]] = []
        for name, body in HEADER_PROBES.items():
            extension = ".c" if name.startswith("c-") else ".cc"
            source = base / ("header-" + name + extension)
            object_path = base / ("header-" + name + ".o")
            source.write_text(body, encoding="utf-8")
            compiler = "cc" if name.startswith("c-") else "c++"
            standard = "-std=c99" if name.startswith("c-") else "-std=c++17"
            result = subprocess.run(
                [compiler, standard, "-O0", "-I", str(ROOT), "-c", str(source),
                 "-o", str(object_path)],
                cwd=base, env=cls._env, capture_output=True, text=True,
                timeout=15, check=False,
            )
            cls._header_compiles.append((name, result))
        cls._abi_source = base / "resident_tensor_abi.c"
        cls._abi_object = base / "resident_tensor_abi.o"
        cls._abi_source.write_text(ABI_PROBE, encoding="utf-8")
        cls._abi_compile = subprocess.run(
            ["cc", "-std=c99", "-O0", "-I", str(ROOT), "-c",
             str(cls._abi_source), "-o", str(cls._abi_object)],
            cwd=base, env=cls._env, capture_output=True, text=True,
            timeout=15, check=False,
        )
        if cls._runtime_compile.returncode == 0:
            cls._compile = subprocess.run(
                ["c++", "-std=c++17", "-O0", "-Werror=format", "-pthread", "-I", str(ROOT),
                 str(cls._source), str(cls._runtime_object), "-o", str(cls._binary)],
                cwd=base, env=cls._env, capture_output=True, text=True,
                timeout=15, check=False,
            )
        else:
            cls._compile = cls._runtime_compile

    def assert_compile_prerequisite(self) -> None:
        self.assertEqual(
            self._compile.returncode, 0,
            "resident tensor fixture compiler RED:\n" + self._compile.stderr,
        )

    def assert_header_prerequisites(self) -> None:
        failures = [
            name + ":\n" + result.stderr
            for name, result in self._header_compiles
            if result.returncode != 0
        ]
        self.assertFalse(failures, "resident header include-order RED:\n" + "\n".join(failures))
        self.assertEqual(
            self._abi_compile.returncode, 0,
            "resident C-safe ABI probe RED:\n" + self._abi_compile.stderr,
        )

    def run_case(self, name: str) -> dict[str, int]:
        self.assert_compile_prerequisite()
        result = subprocess.run(
            [str(self._binary), name], cwd=self._binary.parent, env=self._env,
            capture_output=True, text=True, timeout=16, check=False,
        )
        self.assertEqual(
            result.returncode, 0,
            f"resident tensor scenario {name} RED:\nstdout={result.stdout}\nstderr={result.stderr}",
        )
        values = parse_output(result.stdout)
        for key, count in values.items():
            if key.endswith("_api_errors") or key.endswith("_order_errors"):
                self.assertEqual(count, 0, f"fake driver protocol violation: {key}")
        return values

    def test_00_required_source_and_abi_seams_exist(self) -> None:
        self.assertFalse(
            MISSING_SOURCE_SEAMS,
            "resident tensor source RED; missing actual seams: "
            + ", ".join(MISSING_SOURCE_SEAMS),
        )
        self.assertTrue(
            (ROOT / "ds4_gpu_resident.h").is_file(),
            "resident tensor ABI RED: ds4_gpu_resident.h is absent",
        )
        resident_header = (ROOT / "ds4_gpu_resident.h").read_text(encoding="utf-8")
        self.assertRegex(
            resident_header,
            r"typedef struct\s*\{\s*ds4_gpu_tensor\s*\*tensor;"
            r"\s*uint64_t\s+descriptor_record_id;"
            r"\s*uint64_t\s+device_record_id;\s*\}\s*"
            r"ds4_gpu_laguna_resident_tensor_owner\s*;",
        )
        self.assertIn("ds4_runtime_tracker *tracker", resident_header)
        self.assertIn("ds4_gpu_laguna_resident_tensor_alloc", resident_header)
        self.assertIn("ds4_gpu_laguna_resident_tensor_free", resident_header)
        self.assertIn('#include "ds4_gpu_resident.h"', GPU_HEADER_SOURCE)
        self.assertIn('#include "ds4_gpu_resident.h"', MGPU_HEADER_SOURCE)
        self.assertEqual(CUDA_SOURCE.count(TENSOR_START_MARKER), 1)
        self.assertEqual(CUDA_SOURCE.count(TENSOR_END_MARKER), 1)
        self.assertIn("cuda_laguna_resident_legacy_tensor_permit", OBSERVER_BLOCK or "")
        layout = re.search(
            r"struct\s+ds4_gpu_tensor\s*\{(?P<body>.*?)\};", MGPU_HEADER_SOURCE,
            re.DOTALL,
        )
        self.assertIsNotNone(layout, "ordinary tensor layout missing")
        fields = re.sub(r"/\*.*?\*/", "", layout.group("body") if layout else "")
        self.assertEqual(
            re.sub(r"\s+", " ", fields).strip(),
            "void *ptr; uint64_t bytes; int owner; int device_id;",
        )

    def test_01_header_only_c99_cpp17_include_orders_compile(self) -> None:
        self.assert_header_prerequisites()

    def test_02_make_controls_cover_tensor_contract(self) -> None:
        target = "test-cuda-resident-tensor-contract"
        self.assertRegex(
            MAKEFILE_SOURCE,
            r"(?m)^" + re.escape(target)
            + r":\s*\n\tpython3 tests/test_cuda_resident_tensor_contract\.py -v\s*$",
        )
        phony: set[str] = set()
        for line in MAKEFILE_SOURCE.splitlines():
            if line.startswith(".PHONY:"):
                phony.update(line.split(":", 1)[1].split())
        self.assertIn(target, phony)
        resident = re.search(r"(?m)^test-laguna-resident-path:([^\n]*)", MAKEFILE_SOURCE)
        default = re.search(r"(?m)^test:([^\n]*)", MAKEFILE_SOURCE)
        self.assertIsNotNone(resident)
        self.assertIsNotNone(default)
        self.assertIn(target, resident.group(1).split() if resident else [])
        self.assertIn("test-laguna-resident-path", default.group(1).split() if default else [])
        rules = _make_logical_make_rules(MAKEFILE_SOURCE)
        self.assertGreaterEqual(len(rules), 37, "existing explicit GPU-header rules disappeared")
        missing = [target_name for target_name, deps in rules
                   if "ds4_gpu_resident.h" not in deps.split()]
        self.assertFalse(
            missing,
            "incremental-header dependency RED; missing ds4_gpu_resident.h from: "
            + ", ".join(missing),
        )

    def test_03_success_graph_and_kv_have_two_exact_namespace_records_each(self) -> None:
        values = self.run_case("success")
        self.assertEqual(value(values, "began"), 1)
        self.assertEqual(value(values, "kv_result"), 1)
        self.assertEqual(value(values, "graph_result"), 1)
        descriptor_bytes = value(values, "descriptor_bytes")
        self.assertEqual(value(values, "after_record_count"), 4)
        self.assertEqual(value(values, "after_active_records"), 4)
        self.assertEqual(value(values, "after_device_live"), 2)
        self.assertEqual(value(values, "after_host_live"), 2)
        self.assertEqual(value(values, "after_category_3_current"), 13)
        self.assertEqual(value(values, "after_category_4_current"), 17)
        self.assertEqual(value(values, "after_category_6_current"), 2 * descriptor_bytes)
        self.assertEqual(value(values, "after_owned_current"), 30 + 2 * descriptor_bytes)
        self.assertEqual(value(values, "after_qualification_current"), 30 + 2 * descriptor_bytes)
        self.assertEqual(value(values, "after_category_3_peak"), 13)
        self.assertEqual(value(values, "after_category_4_peak"), 17)
        self.assertEqual(value(values, "after_category_6_peak"), 2 * descriptor_bytes)
        ids = []
        for index in range(4):
            record_id = value(values, f"after_record_{index}_id")
            ids.append(record_id)
            self.assertEqual(record_id >> 56, 0x52)
            self.assertEqual(value(values, f"after_record_{index}_relation"), 0)
            self.assertEqual(value(values, f"after_record_{index}_owner"), 0)
            self.assertEqual(value(values, f"after_record_{index}_live"), 1)
        self.assertEqual(len(set(ids)), 4)
        expected_records = (
            (value(values, "kv_after_tensor"), descriptor_bytes, 6, 0),
            (value(values, "kv_after_tensor_ptr"), 13, 3, 1),
            (value(values, "graph_after_tensor"), descriptor_bytes, 6, 0),
            (value(values, "graph_after_tensor_ptr"), 17, 4, 1),
        )
        for index, (base, charged, category, domain) in enumerate(expected_records):
            self.assertEqual(value(values, f"after_record_{index}_base"), base)
            self.assertEqual(value(values, f"after_record_{index}_requested"), charged)
            self.assertEqual(value(values, f"after_record_{index}_charged"), charged)
            self.assertEqual(value(values, f"after_record_{index}_category"), category)
            self.assertEqual(value(values, f"after_record_{index}_domain"), domain)
        self.assertEqual(value(values, "kv_after_descriptor_id") < value(values, "kv_after_device_id"), True)
        self.assertEqual(value(values, "graph_after_descriptor_id") < value(values, "graph_after_device_id"), True)
        self.assertEqual(value(values, "kv_after_descriptor_id") + 1, value(values, "kv_after_device_id"))
        self.assertEqual(value(values, "graph_after_descriptor_id") + 1, value(values, "graph_after_device_id"))
        self.assertEqual(value(values, "kv_after_tensor_bytes"), 13)
        self.assertEqual(value(values, "graph_after_tensor_bytes"), 17)
        self.assertEqual(value(values, "kv_after_tensor_owner"), 1)
        self.assertEqual(value(values, "graph_after_tensor_owner"), 1)
        self.assertEqual(value(values, "kv_after_tensor_device"), 0)
        self.assertEqual(value(values, "graph_after_tensor_device"), 0)
        self.assertEqual(value(values, "kv_free"), 1)
        self.assertEqual(value(values, "graph_free"), 1)
        self.assertEqual(value(values, "ended"), 1)
        self.assertEqual(value(values, "final_active_records"), 0)
        self.assertEqual(value(values, "final_device_live"), 0)
        self.assertEqual(value(values, "final_host_live"), 0)
        self.assertEqual(value(values, "final_owned_current"), 0)
        self.assertEqual(value(values, "final_qualification_current"), 0)
        self.assertEqual(value(values, "final_category_3_peak"), 13)
        self.assertEqual(value(values, "final_category_4_peak"), 17)
        self.assertEqual(value(values, "final_category_6_peak"), 2 * descriptor_bytes)
        self.assertEqual(value(values, "kv_final_tensor"), 0)
        self.assertEqual(value(values, "kv_final_descriptor_id"), 0)
        self.assertEqual(value(values, "kv_final_device_id"), 0)

    def test_04_failed_device_query_sync_and_free_retain_both_owners_then_retry(self) -> None:
        for scenario in ("free-query", "free-sync", "free-device"):
            with self.subTest(scenario=scenario):
                values = self.run_case(scenario)
                self.assertEqual(value(values, "began"), 1)
                self.assertEqual(value(values, "allocated"), 1)
                self.assertEqual(value(values, "first"), 0)
                self.assertEqual(value(values, "second"), 1)
                self.assertEqual(value(values, "ended"), 1)
                self.assertEqual(value(values, "after_first_device_live"), 1)
                self.assertEqual(value(values, "after_first_host_live"), 1)
                self.assertEqual(value(values, "after_first_active"), 2)
                self.assertEqual(value(values, "after_first_descriptor_free_calls"), 0)
                self.assertEqual(value(values, "after_first_tensor"), value(values, "tensor_before"))
                self.assertEqual(value(values, "after_first_descriptor_id"), value(values, "descriptor_before"))
                self.assertEqual(value(values, "after_first_device_id"), value(values, "device_before"))
                self.assertEqual(value(values, "after_first_tensor_bytes"), 13)
                self.assertEqual(value(values, "after_first_tensor_owner"), 1)
                self.assertEqual(value(values, "after_first_tensor_device"), 0)
                self.assertNotEqual(value(values, "after_first_tensor_ptr"), 0)
                self.assertEqual(value(values, "final_active_records"), 0)
                self.assertEqual(value(values, "final_device_live"), 0)
                self.assertEqual(value(values, "final_host_live"), 0)
                self.assertNotEqual(value(values, "final_violation"), 0)
                self.assertEqual(value(values, "final_owned_current"), 0)
                self.assertGreater(value(values, "final_owned_peak"), 0)

    def test_05_end_refuses_live_owner(self) -> None:
        values = self.run_case("end-live")
        self.assertEqual(value(values, "began"), 1)
        self.assertEqual(value(values, "allocated"), 1)
        self.assertEqual(value(values, "while_live"), 0)
        self.assertEqual(value(values, "freed"), 1)
        self.assertEqual(value(values, "after_free"), 1)
        self.assertEqual(value(values, "final_active_records"), 0)
        self.assertEqual(value(values, "final_device_live"), 0)
        self.assertEqual(value(values, "final_host_live"), 0)

    def test_06_host_descriptor_failure_is_before_any_device_allocation(self) -> None:
        values = self.run_case("host-fail")
        self.assertEqual(value(values, "began"), 1)
        self.assertEqual(value(values, "result"), 0)
        self.assertEqual(value(values, "owner_tensor"), 0)
        self.assertEqual(value(values, "owner_descriptor_id"), 0)
        self.assertEqual(value(values, "owner_device_id"), 0)
        self.assertEqual(value(values, "after_device_malloc_calls"), 0)
        self.assertEqual(value(values, "after_host_calloc_calls"), 1)
        self.assertEqual(value(values, "after_descriptor_free_calls"), 0)
        self.assertEqual(value(values, "after_record_count"), 0)
        self.assertEqual(value(values, "after_active_records"), 0)
        self.assertEqual(value(values, "after_device_live"), 0)
        self.assertEqual(value(values, "after_host_live"), 0)
        self.assertEqual(value(values, "ended"), 1)

    def test_07_cuda_failure_null_and_nonnull_storage_rollbacks(self) -> None:
        null_values = self.run_case("cuda-fail-null")
        self.assertEqual(value(null_values, "result"), 0)
        self.assertEqual(value(null_values, "owner_tensor"), 0)
        self.assertEqual(value(null_values, "after_device_malloc_calls"), 1)
        self.assertEqual(value(null_values, "after_device_free_calls"), 0)
        self.assertEqual(value(null_values, "after_device_live"), 0)
        self.assertEqual(value(null_values, "after_host_live"), 0)
        self.assertEqual(value(null_values, "after_active_records"), 0)
        self.assertEqual(value(null_values, "after_descriptor_free_calls"), 1)
        self.assertEqual(value(null_values, "ended_first"), 1)
        safe_values = self.run_case("cuda-fail-safe")
        self.assertEqual(value(safe_values, "result"), 0)
        self.assertEqual(value(safe_values, "owner_tensor"), 0)
        self.assertEqual(value(safe_values, "after_device_malloc_calls"), 1)
        self.assertEqual(value(safe_values, "after_device_free_calls"), 1)
        self.assertEqual(value(safe_values, "after_device_live"), 0)
        self.assertEqual(value(safe_values, "after_host_live"), 0)
        self.assertEqual(value(safe_values, "after_active_records"), 0)
        self.assertEqual(value(safe_values, "after_descriptor_free_calls"), 1)
        self.assertEqual(value(safe_values, "ended_first"), 1)
        self.assertNotEqual(value(safe_values, "final_violation"), 0)
        retained = self.run_case("cuda-fail-retained")
        self.assertEqual(value(retained, "result"), 0)
        self.assertEqual(value(retained, "owner_tensor"), 0)
        self.assertEqual(value(retained, "after_device_live"), 1)
        self.assertEqual(value(retained, "ended_first"), 0)
        self.assertEqual(value(retained, "after_active_records"), 0)
        self.assertEqual(value(retained, "final_device_live"), 0)
        self.assertEqual(value(retained, "final_host_live"), 0)
        self.assertEqual(value(retained, "ended_second"), 1)

    def test_08_bound_failure_after_insertion_preserves_peaks_and_quarantines_on_rollback_failure(self) -> None:
        safe = self.run_case("bound-safe")
        self.assertEqual(value(safe, "result"), 0)
        self.assertEqual(value(safe, "ended_first"), 1)
        self.assertEqual(value(safe, "after_device_malloc_calls"), 1)
        self.assertEqual(value(safe, "after_device_free_calls"), 1)
        self.assertEqual(value(safe, "after_device_live"), 0)
        self.assertEqual(value(safe, "after_active_records"), 0)
        self.assertEqual(value(safe, "final_category_3_current"), 0)
        self.assertEqual(value(safe, "final_category_3_peak"), 8)
        self.assertGreater(value(safe, "final_category_6_peak"), 0)
        self.assertNotEqual(value(safe, "final_violation"), 0)
        retained = self.run_case("bound-retained")
        self.assertEqual(value(retained, "result"), 0)
        self.assertEqual(value(retained, "ended_first"), 0)
        self.assertEqual(value(retained, "after_device_live"), 1)
        self.assertGreater(value(retained, "after_active_records"), 0)
        self.assertEqual(value(retained, "ended_second"), 1)
        self.assertEqual(value(retained, "final_device_live"), 0)
        self.assertEqual(value(retained, "final_host_live"), 0)
        self.assertEqual(value(retained, "final_active_records"), 0)
        self.assertEqual(value(retained, "final_category_3_peak"), 8)

    def test_09_capacity_and_two_id_preflight_happen_before_physical_allocations(self) -> None:
        capacity = self.run_case("capacity1")
        self.assertEqual(value(capacity, "began"), 1)
        self.assertEqual(value(capacity, "result"), 0)
        self.assertEqual(value(capacity, "after_host_calloc_calls"), 0)
        self.assertEqual(value(capacity, "after_device_malloc_calls"), 0)
        self.assertEqual(value(capacity, "after_record_count"), 0)
        self.assertEqual(value(capacity, "after_active_records"), 0)
        self.assertNotEqual(value(capacity, "after_violation"), 0)
        one = self.run_case("id-one")
        self.assertEqual(value(one, "result"), 0)
        self.assertEqual(value(one, "after_host_calloc_calls"), 0)
        self.assertEqual(value(one, "after_device_malloc_calls"), 0)
        self.assertEqual(value(one, "after_record_count"), 0)
        self.assertNotEqual(value(one, "after_violation"), 0)
        two = self.run_case("id-two")
        self.assertEqual(value(two, "result"), 1)
        self.assertEqual(value(two, "after_host_calloc_calls"), 1)
        self.assertEqual(value(two, "after_device_malloc_calls"), 1)
        self.assertEqual(value(two, "after_active_records"), 2)
        self.assertEqual(value(two, "owner_descriptor_id"), (1 << 56) * 0x52 | 0x00fffffffffffffe)
        self.assertEqual(value(two, "owner_device_id"), (1 << 56) * 0x52 | 0x00ffffffffffffff)

    def test_10_reusable_tombstones_supply_two_slots_without_reusing_ids(self) -> None:
        values = self.run_case("tombstone")
        self.assertEqual(value(values, "began"), 1)
        self.assertEqual(value(values, "first_alloc"), 1)
        self.assertEqual(value(values, "first_free"), 1)
        self.assertEqual(value(values, "record_count_after_first"), 2)
        self.assertEqual(value(values, "second_alloc"), 1)
        self.assertEqual(value(values, "second_free"), 1)
        self.assertEqual(value(values, "ended"), 1)
        self.assertEqual(value(values, "second_desc"), value(values, "first_desc") + 2)
        self.assertEqual(value(values, "second_device"), value(values, "first_device") + 2)
        self.assertEqual(value(values, "final_record_count"), 2)
        self.assertEqual(value(values, "final_active_records"), 0)
        self.assertEqual(value(values, "final_device_live"), 0)

    def test_11_missing_and_wrong_sites_refuse_without_driver_effects(self) -> None:
        for scenario in ("missing-kv", "missing-graph", "missing-host", "wrong-kv", "wrong-graph"):
            with self.subTest(scenario=scenario):
                values = self.run_case(scenario)
                self.assertEqual(value(values, "began"), 1)
                self.assertEqual(value(values, "result"), 0)
                self.assertEqual(value(values, "owner_tensor"), 0)
                self.assertEqual(value(values, "owner_descriptor_id"), 0)
                self.assertEqual(value(values, "owner_device_id"), 0)
                self.assertEqual(value(values, "after_host_calloc_calls"), 0)
                self.assertEqual(value(values, "after_device_malloc_calls"), 0)
                self.assertEqual(value(values, "after_record_count"), 0)
                self.assertEqual(value(values, "after_active_records"), 0)
                self.assertEqual(value(values, "after_device_live"), 0)
                self.assertEqual(value(values, "after_host_live"), 0)
                self.assertNotEqual(value(values, "after_violation"), 0)

    def test_12_missing_or_mismatched_observer_and_active_device_refuse_preflight(self) -> None:
        missing = self.run_case("missing-observer")
        self.assertEqual(value(missing, "result"), 0)
        self.assertEqual(value(missing, "owner_tensor"), 0)
        self.assertEqual(value(missing, "after_host_calloc_calls"), 0)
        self.assertEqual(value(missing, "after_device_malloc_calls"), 0)
        self.assertEqual(value(missing, "after_record_count"), 0)
        self.assertEqual(value(missing, "after_active_records"), 0)
        wrong_tracker = self.run_case("wrong-tracker")
        self.assertEqual(value(wrong_tracker, "began"), 1)
        self.assertEqual(value(wrong_tracker, "result"), 0)
        self.assertEqual(value(wrong_tracker, "attached_active_records"), 0)
        self.assertEqual(value(wrong_tracker, "other_active_records"), 0)
        self.assertEqual(value(wrong_tracker, "attached_violation"), 0)
        for scenario in ("query", "device", "no-gpu"):
            with self.subTest(scenario=scenario):
                values = self.run_case(scenario)
                self.assertEqual(value(values, "began"), 1)
                self.assertEqual(value(values, "result"), 0)
                self.assertEqual(value(values, "owner_tensor"), 0)
                self.assertEqual(value(values, "after_host_calloc_calls"), 0)
                self.assertEqual(value(values, "after_device_malloc_calls"), 0)
                self.assertEqual(value(values, "after_record_count"), 0)
                self.assertEqual(value(values, "after_active_records"), 0)
                self.assertEqual(value(values, "after_device_live"), 0)

    def test_13_zero_bytes_nonempty_out_null_out_and_null_tracker_preserve_inputs(self) -> None:
        values = self.run_case("invalid")
        self.assertEqual(value(values, "began"), 1)
        self.assertEqual(value(values, "zero"), 0)
        self.assertEqual(value(values, "nonempty"), 0)
        self.assertEqual(value(values, "null_out"), 0)
        self.assertEqual(value(values, "null_tracker"), 0)
        self.assertEqual(value(values, "sentinel_tensor"), 0xdead0000)
        self.assertEqual(value(values, "sentinel_descriptor_id"), 0x1111222233334444)
        self.assertEqual(value(values, "sentinel_device_id"), 0x5555666677778888)
        self.assertEqual(value(values, "owner_tensor"), 0)
        self.assertEqual(value(values, "after_host_calloc_calls"), 0)
        self.assertEqual(value(values, "after_device_malloc_calls"), 0)
        self.assertEqual(value(values, "after_record_count"), 0)
        self.assertEqual(value(values, "after_active_records"), 0)

    def test_14_report_only_host_descriptor_registration_blocks_free_without_double_charge(self) -> None:
        values = self.run_case("host-registration")
        self.assertEqual(value(values, "began"), 1)
        self.assertEqual(value(values, "allocated"), 1)
        self.assertEqual(value(values, "related"), 0)
        self.assertEqual(value(values, "record_count_before_free"), 3)
        self.assertEqual(value(values, "owned_before"), value(values, "qualification_before"))
        self.assertEqual(value(values, "failed"), 0)
        self.assertEqual(value(values, "after_failed_free_device_live"), 1)
        self.assertEqual(value(values, "after_failed_free_host_live"), 1)
        self.assertEqual(value(values, "after_failed_free_device_free_calls"), 0)
        self.assertEqual(value(values, "after_failed_free_descriptor_free_calls"), 0)
        self.assertNotEqual(value(values, "unregistered"), 0)
        self.assertEqual(value(values, "relation_live_after_unregister"), 0)
        self.assertEqual(value(values, "freed"), 1)
        self.assertEqual(value(values, "ended"), 1)
        self.assertEqual(value(values, "final_active_records"), 0)
        self.assertEqual(value(values, "final_owned_current"), 0)
        self.assertEqual(value(values, "final_qualification_current"), 0)
        self.assertGreater(value(values, "final_owned_peak"), 0)
        self.assertGreater(value(values, "final_qualification_peak"), 0)

    def test_15_gpu_registration_is_not_a_valid_host_descriptor_relation(self) -> None:
        values = self.run_case("gpu-registration")
        self.assertEqual(value(values, "began"), 1)
        self.assertEqual(value(values, "allocated"), 1)
        self.assertNotEqual(value(values, "related"), 0)
        self.assertEqual(value(values, "record_count_before_free"), 2)
        self.assertEqual(value(values, "failed"), 1)
        self.assertEqual(value(values, "after_failed_free_device_free_calls"), 1)
        self.assertEqual(value(values, "after_failed_free_descriptor_free_calls"), 1)
        self.assertEqual(value(values, "freed"), 1)
        self.assertEqual(value(values, "ended"), 1)

    def test_16_stale_mismatched_partial_and_corrupt_handles_refuse_before_physical_effects(self) -> None:
        scenarios = (
            "stale", "mismatched", "partial-descriptor", "partial-device",
            "partial-tensor", "wrong-bytes", "wrong-owner", "wrong-device", "wrong-ptr",
            "wrong-record-descriptor", "wrong-record-device", "wrong-record-site",
            "wrong-record-domain",
        )
        for scenario in scenarios:
            with self.subTest(scenario=scenario):
                values = self.run_case(scenario)
                self.assertEqual(value(values, "began"), 1)
                self.assertEqual(value(values, "first_result"), 1)
                self.assertEqual(value(values, "second_result"), 1)
                self.assertEqual(value(values, "bad_result"), 0)
                if scenario == "stale":
                    self.assertEqual(value(values, "after_bad_device_free_calls"), 1)
                    self.assertEqual(value(values, "after_bad_descriptor_free_calls"), 1)
                else:
                    self.assertEqual(value(values, "after_bad_device_free_calls"), 0)
                    self.assertEqual(value(values, "after_bad_descriptor_free_calls"), 0)
                self.assertEqual(value(values, "after_bad_device_live"), 2 if scenario != "stale" else 1)
                self.assertGreater(value(values, "after_bad_active_records"), 0)
                self.assertEqual(value(values, "first_free"), 1)
                self.assertEqual(value(values, "second_free"), 1)
                self.assertEqual(value(values, "ended"), 1)
                self.assertEqual(value(values, "final_active_records"), 0)
                self.assertEqual(value(values, "final_device_live"), 0)
                self.assertEqual(value(values, "final_host_live"), 0)

    def test_17_active_observer_fences_all_generic_tensor_variants_before_mutation(self) -> None:
        for variant in ("alloc", "managed", "tiered", "managed-tiered", "alloc-on", "view", "free", "in-place"):
            with self.subTest(variant=variant):
                values = self.run_case("generic-active-" + variant)
                self.assertEqual(value(values, "began"), 1)
                self.assertEqual(value(values, "heap"), 0)
                self.assertEqual(value(values, "managed"), 0)
                self.assertEqual(value(values, "tiered"), 0)
                self.assertEqual(value(values, "managed_tiered"), 0)
                self.assertEqual(value(values, "view"), 0)
                self.assertNotEqual(value(values, "alloc_on_result"), 0)
                self.assertEqual(value(values, "stack_alloc_ptr"), 0xcafe0000)
                self.assertEqual(value(values, "stack_alloc_bytes"), 41)
                self.assertEqual(value(values, "stack_alloc_owner"), 1)
                self.assertEqual(value(values, "stack_alloc_device"), 0)
                self.assertEqual(value(values, "stack_in_place_ptr"), 0xcafe0000)
                self.assertEqual(value(values, "stack_in_place_bytes"), 41)
                self.assertEqual(value(values, "stack_in_place_owner"), 1)
                self.assertEqual(value(values, "stack_in_place_device"), 0)
                self.assertEqual(value(values, "after_host_calloc_calls"), 0)
                self.assertEqual(value(values, "after_managed_malloc_calls"), 0)
                self.assertEqual(value(values, "after_device_malloc_calls"), 0)
                self.assertEqual(value(values, "after_device_free_calls"), 0)
                self.assertEqual(value(values, "after_active_records"), 0)
                self.assertNotEqual(value(values, "after_violation"), 0)
                self.assertEqual(value(values, "ended"), 1)

    def test_18_null_generic_frees_remain_noops_while_observer_attached(self) -> None:
        values = self.run_case("generic-null")
        self.assertEqual(value(values, "began"), 1)
        self.assertEqual(value(values, "ended"), 1)
        self.assertEqual(value(values, "after_violation"), 0)
        self.assertEqual(value(values, "after_host_calloc_calls"), 0)
        self.assertEqual(value(values, "after_device_malloc_calls"), 0)
        self.assertEqual(value(values, "after_device_free_calls"), 0)
        self.assertEqual(value(values, "after_active_records"), 0)

    def test_19_no_observer_generic_paths_remain_raw(self) -> None:
        values = self.run_case("generic-raw")
        self.assertEqual(value(values, "stack_result"), 0)
        self.assertEqual(value(values, "stack_ptr_nonzero"), 1)
        self.assertEqual(value(values, "heap_nonnull"), 1)
        self.assertEqual(value(values, "managed_nonnull"), 1)
        self.assertEqual(value(values, "tiered_nonnull"), 1)
        self.assertEqual(value(values, "managed_tiered_nonnull"), 1)
        self.assertEqual(value(values, "view_base_result"), 0)
        self.assertEqual(value(values, "view_nonnull"), 1)
        self.assertGreater(value(values, "after_device_malloc_calls"), 0)
        self.assertGreater(value(values, "after_managed_malloc_calls"), 0)
        self.assertGreater(value(values, "after_device_free_calls"), 0)
        self.assertEqual(value(values, "after_device_live"), 0)
        self.assertEqual(value(values, "after_host_live"), 0)
        self.assertEqual(value(values, "after_descriptor_free_calls"), 5)
        self.assertEqual(value(values, "after_violation"), 0)
        self.assertEqual(value(values, "after_record_count"), 0)

    def test_20_legacy_generic_use_before_begin_is_one_way_fence(self) -> None:
        values = self.run_case("legacy-before-begin")
        self.assertEqual(value(values, "allocated"), 0)
        self.assertEqual(value(values, "began"), 0)
        self.assertEqual(value(values, "after_violation"), 0)
        self.assertEqual(value(values, "after_record_count"), 0)
        self.assertEqual(value(values, "after_active_records"), 0)

    def test_21_each_generic_tensor_entry_point_uses_the_shared_legacy_fence(self) -> None:
        self.assertIn("cuda_laguna_resident_legacy_tensor_permit", OBSERVER_BLOCK or "")
        for name, body in GENERIC_BODIES.items():
            with self.subTest(entry_point=name):
                self.assertIsNotNone(body, name + " source seam is absent")
                self.assertIn(
                    "cuda_laguna_resident_legacy_tensor_permit",
                    body or "",
                    name + " does not use the shared legacy tensor fence",
                )

if __name__ == "__main__":
    unittest.main(verbosity=2)
