#!/usr/bin/env python3
'''Host-only contract for root-owned native CUDA allocation ownership.

This fixture extracts the real resident observer block and the real scratch and
model-stage bodies from ``ds4_cuda.cu``.  It links those bodies against the real
``ds4_runtime.c`` and a deterministic fake CUDA driver.  The fake driver uses
synthetic pointer values, so no CUDA device, model, NVML, or network is used.
'''
from __future__ import annotations

import re
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CUDA_SOURCE = (ROOT / "ds4_cuda.cu").read_text(encoding="utf-8")
RUNTIME_SOURCE = ROOT / "ds4_runtime.c"
RUNTIME_HEADER = ROOT / "ds4_runtime.h"
SAFE_ENV = {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"}

START_MARKER = "/* Resident native allocation observer."
END_MARKER = "/* End resident native allocation observer. */"


def extract_definition(source: str, signature: str) -> str | None:
    '''Extract a real C/C++ body while ignoring braces in comments/literals.'''
    start = source.rfind(signature)
    if start < 0:
        return None
    i = start + len(signature)
    state = "code"
    while i < len(source):
        c = source[i]
        n = source[i + 1] if i + 1 < len(source) else ""
        if state == "code":
            if c == "/" and n == "/":
                state, i = "line", i + 2
                continue
            if c == "/" and n == "*":
                state, i = "block", i + 2
                continue
            if c == '"':
                state, i = "string", i + 1
                continue
            if c == "'":
                state, i = "char", i + 1
                continue
            if c == "{":
                break
            i += 1
            continue
        if state == "line":
            if c in "\r\n":
                state = "code"
            i += 1
            continue
        if state == "block":
            if c == "*" and n == "/":
                state, i = "code", i + 2
            else:
                i += 1
            continue
        if c == "\\":
            i += 2
        elif (state == "string" and c == '"') or (state == "char" and c == "'"):
            state, i = "code", i + 1
        else:
            i += 1
    else:
        raise AssertionError(f"no body brace after {signature}")

    depth = 0
    state = "code"
    while i < len(source):
        c = source[i]
        n = source[i + 1] if i + 1 < len(source) else ""
        if state == "code":
            if c == "/" and n == "/":
                state, i = "line", i + 2
                continue
            if c == "/" and n == "*":
                state, i = "block", i + 2
                continue
            if c == '"':
                state, i = "string", i + 1
                continue
            if c == "'":
                state, i = "char", i + 1
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return source[start : i + 1]
            i += 1
            continue
        if state == "line":
            if c in "\r\n":
                state = "code"
            i += 1
            continue
        if state == "block":
            if c == "*" and n == "/":
                state, i = "code", i + 2
            else:
                i += 1
            continue
        if c == "\\":
            i += 2
        elif (state == "string" and c == '"') or (state == "char" and c == "'"):
            state, i = "code", i + 1
        else:
            i += 1
    raise AssertionError(f"unterminated definition {signature}")


def extract_observer_block(source: str) -> str | None:
    '''Extract only the marked source block, not a prose/token assertion.'''
    start = source.find(START_MARKER)
    if start < 0:
        return None
    line_start = source.rfind("\n", 0, start) + 1
    if source[line_start:start].strip():
        return None
    end = source.find(END_MARKER, start + len(START_MARKER))
    if end < 0:
        raise AssertionError("resident observer start marker has no end marker")
    end_line = source.find("\n", end)
    if end_line < 0:
        end_line = len(source)
    if source[source.rfind("\n", 0, end) + 1 : end].strip():
        return None
    return source[line_start:end_line]


def source_statement(source: str, statement: str) -> str | None:
    match = re.search(r"(?m)^" + re.escape(statement) + r"\s*$", source)
    return match.group(0) if match else None


OBSERVER_BLOCK = extract_observer_block(CUDA_SOURCE)
TMP = extract_definition(CUDA_SOURCE, "static void *cuda_tmp_alloc(")
ALIGN = extract_definition(CUDA_SOURCE, "static void *cuda_align_ptr(")
STAGE_RELEASE = extract_definition(
    CUDA_SOURCE, "static int cuda_stage_slots_release("
)
MODEL_POOL = extract_definition(
    CUDA_SOURCE, "static int cuda_model_stage_pool_alloc("
)
LEGACY_PERMIT_BODY = extract_definition(CUDA_SOURCE, "class cuda_laguna_compact_legacy_permit")
LEGACY_PERMIT = LEGACY_PERMIT_BODY + ";" if LEGACY_PERMIT_BODY else None
ARENA_LIMIT = extract_definition(CUDA_SOURCE, "static uint64_t cuda_model_cache_limit_bytes(")
ARENA_CHUNK = extract_definition(CUDA_SOURCE, "static uint64_t cuda_model_arena_chunk_bytes(")
ARENA_ALLOC = extract_definition(CUDA_SOURCE, "static char *cuda_model_arena_alloc(")
PROGRESS_RESET = extract_definition(CUDA_SOURCE, "static void cuda_model_load_progress_reset(")
RANGE_RELEASE = (extract_definition(CUDA_SOURCE, "static int cuda_model_range_release_all(")
                 or extract_definition(CUDA_SOURCE, "static void cuda_model_range_release_all("))
TMP_DECLS = "\n".join(
    x
    for x in (
        source_statement(CUDA_SOURCE, "static void *g_cuda_tmp;"),
        source_statement(CUDA_SOURCE, "static uint64_t g_cuda_tmp_bytes;"),
    )
    if x is not None
)
MISSING_SOURCE_SEAMS = [
    name
    for name, value in (
        ("resident observer block", OBSERVER_BLOCK),
        ("cuda_laguna_resident_malloc", OBSERVER_BLOCK and "cuda_laguna_resident_malloc" in OBSERVER_BLOCK),
        ("cuda_laguna_resident_malloc_host", OBSERVER_BLOCK and "cuda_laguna_resident_malloc_host" in OBSERVER_BLOCK),
        ("cuda_laguna_resident_free", OBSERVER_BLOCK and "cuda_laguna_resident_free" in OBSERVER_BLOCK),
        ("cuda_laguna_resident_free_host", OBSERVER_BLOCK and "cuda_laguna_resident_free_host" in OBSERVER_BLOCK),
        ("observer begin", OBSERVER_BLOCK and "ds4_gpu_laguna_resident_observer_begin" in OBSERVER_BLOCK),
        ("observer end", OBSERVER_BLOCK and "ds4_gpu_laguna_resident_observer_end" in OBSERVER_BLOCK),
        ("cuda_tmp_alloc body", TMP),
        ("cuda_tmp_alloc globals", TMP_DECLS),
        ("cuda_model_stage_pool_alloc body", MODEL_POOL),
        ("cuda_stage_slots_release body", STAGE_RELEASE),
        ("arena allocator", ARENA_ALLOC),
        ("arena chunk", ARENA_CHUNK),
        ("arena limit", ARENA_LIMIT),
        ("range release", RANGE_RELEASE),
        ("progress reset", PROGRESS_RESET),
        ("compact permit", LEGACY_PERMIT),
    )
    if not value
]


# The block is intentionally the only native ownership implementation text in
# this fixture.  Everything below is driver/state shape or scenario code.
FAKE_PREFIX = r'''
#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "ds4_laguna_resident.h"
#include <sys/resource.h>
#include <unistd.h>

#include "''' + str(RUNTIME_HEADER) + r'''"

using cudaError_t = int;
using cudaStream_t = void *;
using cudaEvent_t = void *;
static constexpr cudaError_t cudaSuccess = 0;
static constexpr cudaError_t cudaErrorInvalidValue = 1;
static constexpr cudaError_t cudaErrorMemoryAllocation = 2;
static constexpr cudaError_t cudaErrorInvalidConfiguration = 3;
static constexpr cudaError_t cudaErrorInvalidDevicePointer = 4;
static constexpr cudaError_t cudaErrorNotReady = 5;
static constexpr cudaError_t cudaErrorNotSupported = 6;
static constexpr cudaError_t cudaErrorUnknown = 999;
static constexpr cudaError_t kFakeCudaError = 17;
static constexpr unsigned cudaStreamNonBlocking = 1u;
static constexpr unsigned cudaEventDisableTiming = 2u;

struct fake_physical_allocation {
    size_t requested;
    bool host;
};
static std::unordered_map<void *, fake_physical_allocation> fake_live;
static std::unordered_set<void *> fake_live_streams;
static std::unordered_set<void *> fake_live_events;
static uintptr_t fake_next_handle = static_cast<uintptr_t>(0x10000000u);
static int fake_device_malloc_calls;
static int fake_device_free_calls;
static int fake_host_malloc_calls;
static int fake_host_free_calls;
static int fake_stream_create_calls;
static int fake_stream_destroy_calls;
static int fake_event_create_calls;
static int fake_event_destroy_calls;
static bool fake_fail_device_malloc;
static bool fake_fail_host_malloc;
static bool fake_fail_device_free;
static bool fake_fail_host_free;
static bool fake_fail_stream_create;
static bool fake_fail_stream_destroy;
static bool fake_fail_event_create;
static bool fake_fail_event_destroy;
static int fake_api_errors;
static bool fake_invalid_device_address;
static bool fake_partial_driver_failure;
static void *fake_fail_device_free_ptr;
static ds4_runtime_tracker *fake_observed_tracker;

static void fake_assert_not_retired_before_free(void *ptr) {
    if (!fake_observed_tracker) return;
    for (size_t i = 0; i < fake_observed_tracker->record_count; i++) {
        const ds4_runtime_allocation_record *r = &fake_observed_tracker->records[i];
        if ((r->id >> 56) == 0x52u && r->base == (uint64_t)(uintptr_t)ptr && r->live)
            return;
    }
    for (size_t i = 0; i < fake_observed_tracker->record_count; i++) {
        const ds4_runtime_allocation_record *r = &fake_observed_tracker->records[i];
        if ((r->id >> 56) == 0x52u && r->base == (uint64_t)(uintptr_t)ptr) {
            ++fake_api_errors;
            return;
        }
    }
}

static void *fake_handle(void) {
    /* Fake arena reservations reach GiB sizes but allocate no host memory. */
    fake_next_handle += static_cast<uintptr_t>(UINT64_C(1) << 36);
    return reinterpret_cast<void *>(fake_next_handle);
}
static void fake_reset(void) {
    fake_live.clear();
    fake_live_streams.clear();
    fake_live_events.clear();
    fake_next_handle = static_cast<uintptr_t>(0x10000000u);
    fake_device_malloc_calls = fake_device_free_calls = 0;
    fake_host_malloc_calls = fake_host_free_calls = 0;
    fake_stream_create_calls = fake_stream_destroy_calls = 0;
    fake_event_create_calls = fake_event_destroy_calls = 0;
    fake_fail_device_malloc = fake_fail_host_malloc = false;
    fake_fail_device_free = fake_fail_host_free = false;
    fake_fail_stream_create = fake_fail_stream_destroy = false;
    fake_fail_event_create = fake_fail_event_destroy = false;
    fake_api_errors = 0;
    fake_invalid_device_address = false;
    fake_partial_driver_failure = false;
    fake_fail_device_free_ptr = nullptr;
    fake_observed_tracker = nullptr;
}
static cudaError_t cudaMalloc(void **out, size_t bytes) {
    ++fake_device_malloc_calls;
    if (!out) { ++fake_api_errors; return kFakeCudaError; }
    *out = nullptr;
    if (fake_fail_device_malloc) return kFakeCudaError;
    *out = fake_invalid_device_address
        ? reinterpret_cast<void *>(UINTPTR_MAX - 3u) : fake_handle();
    fake_live[*out] = {bytes, false};
    return fake_partial_driver_failure ? kFakeCudaError : cudaSuccess;
}
static cudaError_t cudaFree(void *ptr) {
    ++fake_device_free_calls;
    fake_assert_not_retired_before_free(ptr);
    const auto found = fake_live.find(ptr);
    if (found == fake_live.end() || found->second.host) {
        ++fake_api_errors;
        return kFakeCudaError;
    }
    if (fake_fail_device_free || ptr == fake_fail_device_free_ptr) return kFakeCudaError;
    fake_live.erase(found);
    return cudaSuccess;
}
static cudaError_t cudaMallocHost(void **out, size_t bytes) {
    ++fake_host_malloc_calls;
    if (!out) { ++fake_api_errors; return kFakeCudaError; }
    *out = nullptr;
    if (fake_fail_host_malloc) return kFakeCudaError;
    /* Deliberate fake raw/view displacement exercises reservation charging. */
    *out = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(fake_handle()) + 3u);
    fake_live[*out] = {bytes, true};
    return fake_partial_driver_failure ? kFakeCudaError : cudaSuccess;
}
/* Registration is not exercised by the arena-only scenarios. */
static cudaError_t cudaHostUnregister(void *) { ++fake_api_errors; return kFakeCudaError; }
static cudaError_t cudaFreeHost(void *ptr) {
    ++fake_host_free_calls;
    fake_assert_not_retired_before_free(ptr);
    const auto found = fake_live.find(ptr);
    if (found == fake_live.end() || !found->second.host) {
        ++fake_api_errors;
        return kFakeCudaError;
    }
    if (fake_fail_host_free) return kFakeCudaError;
    fake_live.erase(found);
    return cudaSuccess;
}
static cudaError_t cudaStreamCreateWithFlags(cudaStream_t *out, unsigned flags) {
    ++fake_stream_create_calls;
    if (!out || flags != cudaStreamNonBlocking || fake_fail_stream_create) {
        ++fake_api_errors;
        if (out) *out = nullptr;
        return kFakeCudaError;
    }
    *out = fake_handle();
    fake_live_streams.insert(*out);
    return cudaSuccess;
}
static cudaError_t cudaStreamDestroy(cudaStream_t stream) {
    ++fake_stream_destroy_calls;
    if (!stream || fake_live_streams.count(stream) != 1u) {
        ++fake_api_errors;
        return kFakeCudaError;
    }
    if (fake_fail_stream_destroy) return kFakeCudaError;
    fake_live_streams.erase(stream);
    return cudaSuccess;
}
static cudaError_t cudaEventCreateWithFlags(cudaEvent_t *out, unsigned flags) {
    ++fake_event_create_calls;
    if (!out || flags != cudaEventDisableTiming || fake_fail_event_create) {
        ++fake_api_errors;
        if (out) *out = nullptr;
        return kFakeCudaError;
    }
    *out = fake_handle();
    fake_live_events.insert(*out);
    return cudaSuccess;
}
static cudaError_t cudaEventDestroy(cudaEvent_t event) {
    ++fake_event_destroy_calls;
    if (!event || fake_live_events.count(event) != 1u) {
        ++fake_api_errors;
        return kFakeCudaError;
    }
    if (fake_fail_event_destroy) return kFakeCudaError;
    fake_live_events.erase(event);
    return cudaSuccess;
}
static const char *cudaGetErrorString(cudaError_t) { return "fake-cuda-error"; }
static cudaError_t cudaGetLastError(void) { return cudaSuccess; }

/* Existing-global data shape used by observer begin's legacy-owner fence. */
struct cuda_model_range {
    const void *host_base;
    uint64_t offset;
    uint64_t bytes;
    char *device_ptr;
    void *registered_base;
    char *registered_device_base;
    uint64_t registered_bytes;
    int host_registered;
    int arena_allocated;
};
struct cuda_model_arena {
    char *device_ptr;
    uint64_t bytes;
    uint64_t used;
};
struct cuda_q8_f16_range { const void *host_base; void *device_ptr; int device_id; };
struct cuda_q8_f32_range { const void *host_base; void *device_ptr; int device_id; };
static std::vector<cuda_model_range> g_model_ranges;
static std::vector<cuda_model_arena> g_model_arenas;
static std::vector<cuda_q8_f16_range> g_q8_f16_ranges;
static std::vector<cuda_q8_f32_range> g_q8_f32_ranges;
static uint64_t g_model_range_bytes;
static int g_model_range_release_failed;
static std::unordered_map<uint64_t, size_t> g_model_range_by_offset;
static uint64_t g_model_load_progress_next;
static double g_model_load_progress_last;
static int g_model_load_progress_started;
static int g_model_load_progress_tty;
static int g_model_cache_full;
static const void *g_model_host_base;
static const char *g_model_device_base;
static uint64_t g_model_registered_size;
static int g_model_registered;
static int g_model_device_owned;
static int g_model_direct_fd = -1;
static uint64_t g_model_file_size;
static int g_n_gpus;
static int g_model_range_mapping_supported = 1;

static void *g_model_stage_raw[4];
static void *g_model_stage[4];
static cudaEvent_t g_model_stage_event[4];
static uint64_t g_model_stage_reserved_bytes[4];
static uint64_t g_model_stage_bytes;
static void *g_stream_selected_stage_raw[4];
static void *g_stream_selected_stage[4];
static cudaEvent_t g_stream_selected_stage_event[4];
static uint64_t g_stream_selected_stage_reserved_bytes[4];
static uint64_t g_stream_selected_stage_bytes;
static cudaStream_t g_model_upload_stream;
static cudaStream_t g_stream_selected_upload_stream;
static uint64_t g_model_direct_align = 1;

enum {
    DS4_LAGUNA_COMPACT_IDLE = 0,
    DS4_LAGUNA_COMPACT_CREATING = 1,
    DS4_LAGUNA_COMPACT_ACTIVE = 2,
};
struct ds4_gpu_laguna_compact { ds4_runtime_tracker *tracker; uint64_t rejection_count; };
static ds4_gpu_laguna_compact g_laguna_compact_storage;
static std::atomic<int> g_laguna_compact_state{DS4_LAGUNA_COMPACT_IDLE};
static std::recursive_mutex g_laguna_compact_mutex;
'''

FAKE_SUFFIX = r'''

static void reset_legacy(void) {
    g_cuda_tmp = nullptr;
    g_cuda_tmp_bytes = 0;
    std::memset(g_model_stage_raw, 0, sizeof(g_model_stage_raw));
    std::memset(g_model_stage, 0, sizeof(g_model_stage));
    std::memset(g_model_stage_event, 0, sizeof(g_model_stage_event));
    std::memset(g_model_stage_reserved_bytes, 0, sizeof(g_model_stage_reserved_bytes));
    g_model_stage_bytes = 0;
    std::memset(g_stream_selected_stage_raw, 0, sizeof(g_stream_selected_stage_raw));
    std::memset(g_stream_selected_stage, 0, sizeof(g_stream_selected_stage));
    std::memset(g_stream_selected_stage_event, 0, sizeof(g_stream_selected_stage_event));
    std::memset(g_stream_selected_stage_reserved_bytes, 0, sizeof(g_stream_selected_stage_reserved_bytes));
    g_stream_selected_stage_bytes = 0;
    g_model_upload_stream = nullptr;
    g_stream_selected_upload_stream = nullptr;
    g_model_direct_align = 1;
    g_model_ranges.clear();
    g_model_arenas.clear();
    g_q8_f16_ranges.clear();
    g_q8_f32_ranges.clear();
    g_model_range_bytes = 0;
    g_model_range_release_failed = 0;
    g_model_range_by_offset.clear();
    g_model_load_progress_next = 0;
    g_model_load_progress_last = 0;
    g_model_load_progress_started = 0;
    g_model_load_progress_tty = 0;
    g_model_cache_full = 0;
    g_model_host_base = nullptr;
    g_model_device_base = nullptr;
    g_model_registered_size = 0;
    g_model_registered = 0;
    g_model_device_owned = 0;
    g_model_direct_fd = -1;
    g_model_file_size = 0;
    g_n_gpus = 0;
    g_model_range_mapping_supported = 1;
    g_laguna_compact_storage.tracker = nullptr;
    g_laguna_compact_state.store(DS4_LAGUNA_COMPACT_IDLE, std::memory_order_relaxed);
}

static ds4_runtime_tracker tracker;
static ds4_runtime_allocation_record records[16];
static ds4_runtime_callsite callsites[3];
static ds4_runtime_tracker_config tracker_config;

static int init_tracker(size_t capacity = 16u,
                        uint64_t static_bound = 64u,
                        uint64_t pinned_bound = 128u,
                        uint64_t other_bound = 128u,
                        uint64_t registration_bound = 0u) {
    std::memset(&tracker, 0, sizeof(tracker));
    std::memset(records, 0, sizeof(records));
    callsites[0] = {1u, "laguna.static_slab", DS4_RUNTIME_CATEGORY_STATIC_WEIGHTS,
                    DS4_RUNTIME_DOMAIN_CUDA_DEVICE, static_bound};
    callsites[1] = {11u, "laguna.pinned_staging.0", DS4_RUNTIME_CATEGORY_PINNED_STAGING,
                    DS4_RUNTIME_DOMAIN_HOST, pinned_bound};
    callsites[2] = {22u, "laguna.other_cuda.kernel_tmp", DS4_RUNTIME_CATEGORY_OTHER_CUDA,
                    DS4_RUNTIME_DOMAIN_CUDA_DEVICE, other_bound};
    std::memset(&tracker_config, 0, sizeof(tracker_config));
    tracker_config.callsites = callsites;
    tracker_config.callsite_count = 3u;
    tracker_config.records = records;
    tracker_config.record_capacity = capacity;
    tracker_config.category_bounds[DS4_RUNTIME_CATEGORY_STATIC_WEIGHTS] = static_bound;
    tracker_config.category_bounds[DS4_RUNTIME_CATEGORY_PINNED_STAGING] = pinned_bound;
    tracker_config.category_bounds[DS4_RUNTIME_CATEGORY_OTHER_CUDA] = other_bound;
    tracker_config.report_bounds[DS4_RUNTIME_REPORT_MODEL_MAPPING_REGISTERED] = registration_bound;
    tracker_config.owned_total_bound_bytes = static_bound + pinned_bound + other_bound;
    tracker_config.qualification_total_bound_bytes = tracker_config.owned_total_bound_bytes;
    return ds4_runtime_tracker_init(&tracker, &tracker_config) == DS4_RUNTIME_STATUS_OK;
}
static void reset_case(size_t capacity = 16u,
                       uint64_t static_bound = 64u,
                       uint64_t pinned_bound = 128u,
                       uint64_t other_bound = 128u,
                       uint64_t registration_bound = 0u) {
    fake_reset();
    reset_legacy();
    if (!init_tracker(capacity, static_bound, pinned_bound, other_bound, registration_bound))
        std::abort();
    fake_observed_tracker = &tracker;
}
static int active_records(void) {
    int count = 0;
    for (size_t i = 0; i < tracker.record_count; ++i)
        if (tracker.records[i].live) ++count;
    return count;
}
static int live_physical(bool host) {
    int count = 0;
    for (const auto &entry : fake_live)
        if (entry.second.host == host) ++count;
    return count;
}
static void emit_tracker(const char *label) {
    std::printf("%s_violation=%d\n", label, (int)tracker.violation);
    std::printf("%s_attributed_valid=%d\n", label,
                tracker.external_sample.attributed_valid ? 1 : 0);
    std::printf("%s_owned_current=%llu\n%s_owned_peak=%llu\n", label,
                (unsigned long long)tracker.owned_total_current, label,
                (unsigned long long)tracker.owned_total_peak);
    std::printf("%s_qualification_current=%llu\n%s_qualification_peak=%llu\n", label,
                (unsigned long long)tracker.qualification_total_current, label,
                (unsigned long long)tracker.qualification_total_peak);
    std::printf("%s_registered_current=%llu\n%s_registered_peak=%llu\n", label,
                (unsigned long long)tracker.report_current[DS4_RUNTIME_REPORT_MODEL_MAPPING_REGISTERED],
                label, (unsigned long long)tracker.report_peak[DS4_RUNTIME_REPORT_MODEL_MAPPING_REGISTERED]);
    std::printf("%s_record_count=%llu\n", label,
                (unsigned long long)tracker.record_count);
    std::printf("%s_active_records=%d\n", label, active_records());
    std::printf("%s_static_current=%llu\n", label,
                (unsigned long long)tracker.category_current[DS4_RUNTIME_CATEGORY_STATIC_WEIGHTS]);
    std::printf("%s_static_peak=%llu\n", label,
                (unsigned long long)tracker.category_peak[DS4_RUNTIME_CATEGORY_STATIC_WEIGHTS]);
    std::printf("%s_pinned_current=%llu\n", label,
                (unsigned long long)tracker.category_current[DS4_RUNTIME_CATEGORY_PINNED_STAGING]);
    std::printf("%s_pinned_peak=%llu\n", label,
                (unsigned long long)tracker.category_peak[DS4_RUNTIME_CATEGORY_PINNED_STAGING]);
    std::printf("%s_other_current=%llu\n", label,
                (unsigned long long)tracker.category_current[DS4_RUNTIME_CATEGORY_OTHER_CUDA]);
    std::printf("%s_other_peak=%llu\n", label,
                (unsigned long long)tracker.category_peak[DS4_RUNTIME_CATEGORY_OTHER_CUDA]);
    std::printf("%s_device_live=%d\n", label, live_physical(false));
    std::printf("%s_host_live=%d\n", label, live_physical(true));
    std::printf("%s_device_malloc_calls=%d\n", label, fake_device_malloc_calls);
    std::printf("%s_device_free_calls=%d\n", label, fake_device_free_calls);
    std::printf("%s_host_malloc_calls=%d\n", label, fake_host_malloc_calls);
    std::printf("%s_host_free_calls=%d\n", label, fake_host_free_calls);
    std::printf("%s_stream_create_calls=%d\n", label, fake_stream_create_calls);
    std::printf("%s_stream_destroy_calls=%d\n", label, fake_stream_destroy_calls);
    std::printf("%s_event_create_calls=%d\n", label, fake_event_create_calls);
    std::printf("%s_event_destroy_calls=%d\n", label, fake_event_destroy_calls);
    std::printf("%s_api_errors=%d\n", label, fake_api_errors);
    std::printf("%s_tmp_ptr=%llu\n", label,
                (unsigned long long)(uintptr_t)g_cuda_tmp);
    std::printf("%s_tmp_bytes=%llu\n", label,
                (unsigned long long)g_cuda_tmp_bytes);
    for (size_t i = 0; i < tracker.record_count; ++i) {
        const ds4_runtime_allocation_record &record = tracker.records[i];
        std::printf("%s_record_%llu_id=%llu\n", label, (unsigned long long)i,
                    (unsigned long long)record.id);
        std::printf("%s_record_%llu_base=%llu\n", label, (unsigned long long)i,
                    (unsigned long long)record.base);
        std::printf("%s_record_%llu_requested=%llu\n", label, (unsigned long long)i,
                    (unsigned long long)record.requested_bytes);
        std::printf("%s_record_%llu_charged=%llu\n", label, (unsigned long long)i,
                    (unsigned long long)record.charged_bytes);
        std::printf("%s_record_%llu_callsite=%u\n", label, (unsigned long long)i,
                    record.callsite_id);
        std::printf("%s_record_%llu_category=%d\n", label, (unsigned long long)i,
                    (int)record.category);
        std::printf("%s_record_%llu_domain=%d\n", label, (unsigned long long)i,
                    (int)record.domain);
        std::printf("%s_record_%llu_relation=%d\n", label, (unsigned long long)i,
                    (int)record.relation);
        std::printf("%s_record_%llu_live=%d\n", label, (unsigned long long)i,
                    record.live ? 1 : 0);
    }
    std::fflush(stdout);
}
static void emit_stage(const char *label) {
    for (size_t i = 0; i < 4; ++i) {
        std::printf("%s_raw_%llu=%llu\n", label, (unsigned long long)i,
                    (unsigned long long)(uintptr_t)g_model_stage_raw[i]);
        std::printf("%s_stage_%llu=%llu\n", label, (unsigned long long)i,
                    (unsigned long long)(uintptr_t)g_model_stage[i]);
        std::printf("%s_reserved_%llu=%llu\n", label, (unsigned long long)i,
                    (unsigned long long)g_model_stage_reserved_bytes[i]);
    }
    std::printf("%s_stage_bytes=%llu\n", label,
                (unsigned long long)g_model_stage_bytes);
    emit_tracker(label);
}

static void emit_arenas(const char *label) {
    std::printf("%s_arena_count=%llu\n%s_range_count=%llu\n%s_release_failed=%d\n",
        label, (unsigned long long)g_model_arenas.size(),
        label, (unsigned long long)g_model_ranges.size(), label, g_model_range_release_failed);
    for (size_t i = 0; i < g_model_arenas.size(); ++i) {
        std::printf("%s_arena_%llu_ptr=%llu\n", label, (unsigned long long)i,
                    (unsigned long long)(uintptr_t)g_model_arenas[i].device_ptr);
        std::printf("%s_arena_%llu_bytes=%llu\n", label, (unsigned long long)i,
                    (unsigned long long)g_model_arenas[i].bytes);
    }
    for (size_t i = 0; i < g_model_ranges.size(); ++i)
        std::printf("%s_range_%llu_ptr=%llu\n", label, (unsigned long long)i,
                    (unsigned long long)(uintptr_t)g_model_ranges[i].device_ptr);
    emit_tracker(label);
}
static int scenario_arena_observe(void) {
    const uint64_t chunk = UINT64_C(1792) * 1048576u;
    reset_case(16u, 4u * chunk);
    const int began = ds4_gpu_laguna_resident_observer_begin(&tracker);
    char *first = cuda_model_arena_alloc(13u, "small");
    char *second = cuda_model_arena_alloc(21u, "reuse");
    char *third = cuda_model_arena_alloc(chunk, "new-chunk");
    std::printf("began=%d\nfirst=%llu\nsecond=%llu\nthird=%llu\nchunk=%llu\n", began,
        (unsigned long long)(uintptr_t)first, (unsigned long long)(uintptr_t)second,
        (unsigned long long)(uintptr_t)third, (unsigned long long)chunk);
    emit_arenas("ready");
    (void)cuda_model_range_release_all();
    const int ended = ds4_gpu_laguna_resident_observer_end(&tracker);
    std::printf("ended=%d\n", ended);
    emit_arenas("final");
    return 0;
}
static int scenario_arena_release_failure(void) {
    reset_case();
    const int began = ds4_gpu_laguna_resident_observer_begin(&tracker);
    void *ptrs[3] = {};
    const size_t lengths[3] = {17u, 21u, 23u};
    for (size_t i = 0; i < 3u; ++i) {
        if (cuda_laguna_resident_malloc(&ptrs[i], lengths[i], 1u) != cudaSuccess)
            std::abort();
        g_model_arenas.push_back({(char *)ptrs[i], lengths[i], lengths[i]});
        g_model_ranges.push_back({nullptr, i * 5u, 5u, (char *)ptrs[i],
                                  nullptr, nullptr, 0u, 0, 1});
        g_model_range_by_offset[i * 5u] = i;
    }
    g_model_range_bytes = 15u;
    fake_fail_device_free_ptr = ptrs[1];
    (void)cuda_model_range_release_all();
    const int blocked = ds4_gpu_laguna_resident_observer_end(&tracker);
    std::printf("began=%d\nblocked=%d\nsecond_ptr=%llu\nthird_ptr=%llu\n", began, blocked,
        (unsigned long long)(uintptr_t)ptrs[1], (unsigned long long)(uintptr_t)ptrs[2]);
    emit_arenas("retained");
    fake_fail_device_free_ptr = nullptr;
    (void)cuda_model_range_release_all();
    const int ended = ds4_gpu_laguna_resident_observer_end(&tracker);
    std::printf("ended=%d\n", ended);
    emit_arenas("final");
    return 0;
}

static int scenario_host_free_retry(void) {
    reset_case();
    const int began = ds4_gpu_laguna_resident_observer_begin(&tracker);
    void *ptr = nullptr;
    if (cuda_laguna_resident_malloc_host(&ptr, 19u) != cudaSuccess) std::abort();
    fake_fail_host_free = true;
    const cudaError_t failed = cuda_laguna_resident_free_host(ptr);
    const int blocked = ds4_gpu_laguna_resident_observer_end(&tracker);
    std::printf("began=%d\nfailed=%d\nblocked=%d\n", began, (int)failed, blocked);
    emit_tracker("retained");
    fake_fail_host_free = false;
    const cudaError_t freed = cuda_laguna_resident_free_host(ptr);
    const int ended = ds4_gpu_laguna_resident_observer_end(&tracker);
    std::printf("freed=%d\nended=%d\n", (int)freed, ended);
    emit_tracker("final");
    return 0;
}
static int scenario_wrong_free(bool pinned) {
    reset_case();
    const int began = ds4_gpu_laguna_resident_observer_begin(&tracker);
    void *ptr = nullptr;
    const cudaError_t allocated = pinned ? cuda_laguna_resident_malloc_host(&ptr, 19u)
        : cuda_laguna_resident_malloc(&ptr, 19u, 22u);
    if (allocated != cudaSuccess) std::abort();
    const cudaError_t wrong = pinned ? cuda_laguna_resident_free(ptr)
        : cuda_laguna_resident_free_host(ptr);
    std::printf("began=%d\nwrong=%d\n", began, (int)wrong);
    emit_tracker("retained");
    const cudaError_t freed = pinned ? cuda_laguna_resident_free_host(ptr)
        : cuda_laguna_resident_free(ptr);
    const cudaError_t duplicate = pinned ? cuda_laguna_resident_free_host(ptr)
        : cuda_laguna_resident_free(ptr);
    const int ended = ds4_gpu_laguna_resident_observer_end(&tracker);
    std::printf("freed=%d\nduplicate=%d\nended=%d\n", (int)freed, (int)duplicate, ended);
    emit_tracker("final");
    return 0;
}
static int scenario_live_relation(void) {
    reset_case(16u, 64u, 128u, 128u, 32u);
    const int began = ds4_gpu_laguna_resident_observer_begin(&tracker);
    void *ptr = nullptr;
    if (cuda_laguna_resident_malloc_host(&ptr, 19u) != cudaSuccess) std::abort();
    const uint64_t owner_id = tracker.records[0].id;
    const uint64_t relation_id = UINT64_C(0x5300000000000001);
    /* This is reference relation metadata, not a claim of native registration. */
    const ds4_runtime_status related = ds4_runtime_tracker_register(
        &tracker, relation_id, (uint64_t)(uintptr_t)ptr, 19u, owner_id);
    std::printf("began=%d\nrelated=%d\n", began, (int)related);
    emit_tracker("related");
    const cudaError_t failed = cuda_laguna_resident_free_host(ptr);
    std::printf("failed=%d\n", (int)failed);
    emit_tracker("retained");
    (void)ds4_runtime_tracker_unregister(&tracker, relation_id);
    const cudaError_t freed = cuda_laguna_resident_free_host(ptr);
    const int ended = ds4_gpu_laguna_resident_observer_end(&tracker);
    std::printf("freed=%d\nended=%d\n", (int)freed, ended);
    emit_tracker("final");
    return 0;
}

static int scenario_alloc_observe(void) {
    reset_case();
    const int began = ds4_gpu_laguna_resident_observer_begin(&tracker);
    void *device_static = nullptr;
    void *device_other = nullptr;
    void *host = nullptr;
    const cudaError_t a = cuda_laguna_resident_malloc(&device_static, 17u, 1u);
    const cudaError_t b = cuda_laguna_resident_malloc(&device_other, 23u, 22u);
    const cudaError_t c = cuda_laguna_resident_malloc_host(&host, 19u);
    std::printf("began=%d\nalloc_static=%d\nalloc_other=%d\nalloc_host=%d\n",
                began, (int)a, (int)b, (int)c);
    emit_tracker("after");
    std::printf("static_ptr=%llu\nother_ptr=%llu\nhost_ptr=%llu\n",
                (unsigned long long)(uintptr_t)device_static,
                (unsigned long long)(uintptr_t)device_other,
                (unsigned long long)(uintptr_t)host);
    return 0;
}
static int scenario_free_retry(void) {
    reset_case();
    const int began = ds4_gpu_laguna_resident_observer_begin(&tracker);
    void *ptr = nullptr;
    const cudaError_t allocated = cuda_laguna_resident_malloc(&ptr, 17u, 1u);
    fake_fail_device_free = true;
    const cudaError_t first = cuda_laguna_resident_free(ptr);
    fake_fail_device_free = false;
    const cudaError_t second = cuda_laguna_resident_free(ptr);
    const int ended = ds4_gpu_laguna_resident_observer_end(&tracker);
    std::printf("began=%d\nallocated=%d\nfirst=%d\nsecond=%d\nended=%d\nptr=%llu\n",
                began, (int)allocated, (int)first, (int)second, ended,
                (unsigned long long)(uintptr_t)ptr);
    emit_tracker("after");
    return 0;
}
static int scenario_capacity(void) {
    reset_case(1u);
    const int began = ds4_gpu_laguna_resident_observer_begin(&tracker);
    void *first_ptr = nullptr;
    void *second_ptr = nullptr;
    const cudaError_t first = cuda_laguna_resident_malloc(&first_ptr, 8u, 1u);
    const int calls_before = fake_device_malloc_calls;
    const cudaError_t second = cuda_laguna_resident_malloc(&second_ptr, 8u, 1u);
    std::printf("began=%d\nfirst=%d\nsecond=%d\ncalls_before=%d\n",
                began, (int)first, (int)second, calls_before);
    emit_tracker("after");
    return 0;
}
static int scenario_unknown(void) {
    reset_case();
    const int began = ds4_gpu_laguna_resident_observer_begin(&tracker);
    void *ptr = nullptr;
    const cudaError_t result = cuda_laguna_resident_malloc(&ptr, 8u, 99u);
    std::printf("began=%d\nresult=%d\nptr=%llu\n", began, (int)result,
                (unsigned long long)(uintptr_t)ptr);
    emit_tracker("after");
    return 0;
}
static int scenario_alloc_failure(void) {
    reset_case();
    const int began = ds4_gpu_laguna_resident_observer_begin(&tracker);
    fake_fail_device_malloc = true;
    void *ptr = nullptr;
    const cudaError_t result = cuda_laguna_resident_malloc(&ptr, 17u, 1u);
    std::printf("began=%d\nresult=%d\nptr=%llu\n", began, (int)result,
                (unsigned long long)(uintptr_t)ptr);
    emit_tracker("after");
    return 0;
}
static int scenario_bound_rollback(bool free_fails) {
    reset_case(8u, 16u);
    const int began = ds4_gpu_laguna_resident_observer_begin(&tracker);
    void *first_ptr = nullptr;
    void *second_ptr = nullptr;
    const cudaError_t first = cuda_laguna_resident_malloc(&first_ptr, 16u, 1u);
    fake_fail_device_free = free_fails;
    const int malloc_before = fake_device_malloc_calls;
    const cudaError_t second = cuda_laguna_resident_malloc(&second_ptr, 1u, 1u);
    const int malloc_after = fake_device_malloc_calls;
    fake_fail_device_free = false;
    const cudaError_t third = cuda_laguna_resident_malloc(&second_ptr, 1u, 1u);
    const int ended = ds4_gpu_laguna_resident_observer_end(&tracker);
    std::printf("began=%d\nfirst=%d\nsecond=%d\nthird=%d\nended=%d\n"
                "malloc_before=%d\nmalloc_after=%d\nfirst_ptr=%llu\nsecond_ptr=%llu\n",
                began, (int)first, (int)second, (int)third, ended,
                malloc_before, malloc_after,
                (unsigned long long)(uintptr_t)first_ptr,
                (unsigned long long)(uintptr_t)second_ptr);
    emit_tracker("after");
    return 0;
}
static int scenario_partial_allocation(bool pinned, bool rollback_fails) {
    reset_case();
    const int began = ds4_gpu_laguna_resident_observer_begin(&tracker);
    tracker.external_sample.attributed_valid = true;
    fake_partial_driver_failure = true;
    fake_fail_device_free = rollback_fails;
    fake_fail_host_free = rollback_fails;
    void *ptr = nullptr;
    const cudaError_t result = pinned
        ? cuda_laguna_resident_malloc_host(&ptr, 19u)
        : cuda_laguna_resident_malloc(&ptr, 17u, 1u);
    std::printf("began=%d\nresult=%d\nptr=%llu\n", began, (int)result,
                (unsigned long long)(uintptr_t)ptr);
    emit_tracker("after");
    const int ended = ds4_gpu_laguna_resident_observer_end(&tracker);
    std::printf("ended=%d\n", ended);
    if (rollback_fails) {
        fake_fail_device_free = false;
        fake_fail_host_free = false;
        const int recovered = ds4_gpu_laguna_resident_observer_end(&tracker);
        std::printf("recovered=%d\n", recovered);
    }
    emit_tracker("final");
    return 0;
}

static int scenario_id_exhaustion(void) {
    reset_case();
    const int began = ds4_gpu_laguna_resident_observer_begin(&tracker);
    tracker.issued_sequence_high_water[0x52u] = UINT64_C(0x00ffffffffffffff);
    void *ptr = nullptr;
    const cudaError_t result = cuda_laguna_resident_malloc(&ptr, 17u, 1u);
    std::printf("began=%d\nresult=%d\nptr=%llu\n", began, (int)result,
                (unsigned long long)(uintptr_t)ptr);
    emit_tracker("after");
    return 0;
}

static int scenario_unrecorded_rollback(void) {
    reset_case();
    const int began = ds4_gpu_laguna_resident_observer_begin(&tracker);
    tracker.external_sample.attributed_valid = true;
    tracker.external_sample.attributed_generation = 7;
    fake_invalid_device_address = true;
    fake_fail_device_free = true;
    void *ptr = nullptr;
    const cudaError_t allocated = cuda_laguna_resident_malloc(&ptr, 8u, 1u);
    const int blocked_end = ds4_gpu_laguna_resident_observer_end(&tracker);
    std::printf("began=%d\nallocated=%d\nblocked_end=%d\nptr=%llu\n",
                began, (int)allocated, blocked_end, (unsigned long long)(uintptr_t)ptr);
    emit_tracker("retained");
    fake_fail_device_free = false;
    const int recovered_end = ds4_gpu_laguna_resident_observer_end(&tracker);
    std::printf("recovered_end=%d\n", recovered_end);
    emit_tracker("recovered");
    return 0;
}
static int scenario_scratch_growth(bool free_fails) {
    reset_case(8u, 64u, 128u, 64u);
    const int began = ds4_gpu_laguna_resident_observer_begin(&tracker);
    void *first = cuda_tmp_alloc(13u, "scratch");
    const int malloc_after_first = fake_device_malloc_calls;
    void *reuse = cuda_tmp_alloc(7u, "scratch");
    const int malloc_after_reuse = fake_device_malloc_calls;
    fake_fail_device_free = free_fails;
    void *grown = cuda_tmp_alloc(21u, "scratch");
    const int malloc_after_growth = fake_device_malloc_calls;
    fake_fail_device_free = false;
    std::printf("began=%d\nfirst=%llu\nreuse=%llu\ngrown=%llu\n"
                "malloc_after_first=%d\nmalloc_after_reuse=%d\n"
                "malloc_after_growth=%d\n",
                began,
                (unsigned long long)(uintptr_t)first,
                (unsigned long long)(uintptr_t)reuse,
                (unsigned long long)(uintptr_t)grown,
                malloc_after_first, malloc_after_reuse, malloc_after_growth);
    emit_tracker("after");
    if (!free_fails && grown) (void)cuda_laguna_resident_free(grown);
    else if (free_fails && first) {
        (void)cuda_laguna_resident_free(first);
    }
    std::printf("ended=%d\n", ds4_gpu_laguna_resident_observer_end(&tracker));
    emit_tracker("final");
    return 0;
}
static int scenario_scratch_alloc_failure(void) {
    reset_case(8u, 64u, 128u, 64u);
    const int began = ds4_gpu_laguna_resident_observer_begin(&tracker);
    fake_fail_device_malloc = true;
    void *ptr = cuda_tmp_alloc(13u, "scratch");
    std::printf("began=%d\nptr=%llu\n", began,
                (unsigned long long)(uintptr_t)ptr);
    emit_tracker("after");
    return 0;
}
static int scenario_stage_pool(void) {
    reset_case(16u, 64u, 80u, 128u);
    const int began = ds4_gpu_laguna_resident_observer_begin(&tracker);
    g_model_direct_align = 8u;
    const int allocated = cuda_model_stage_pool_alloc(20u);
    std::printf("began=%d\nallocated=%d\n", began, allocated);
    emit_stage("ready");
    const int released = cuda_stage_slots_release(
        g_model_stage_raw, g_model_stage, g_model_stage_event,
        g_model_stage_reserved_bytes);
    g_model_stage_bytes = 0;
    if (g_model_upload_stream) {
        (void)cudaStreamDestroy(g_model_upload_stream);
        g_model_upload_stream = nullptr;
    }
    std::printf("released=%d\n", released);
    emit_stage("released");
    std::printf("ended=%d\n", ds4_gpu_laguna_resident_observer_end(&tracker));
    return 0;
}
static int scenario_stage_pool_resize_failure(void) {
    reset_case(16u, 64u, 80u, 128u);
    const int began = ds4_gpu_laguna_resident_observer_begin(&tracker);
    g_model_direct_align = 8u;
    const int first = cuda_model_stage_pool_alloc(20u);
    const int malloc_before = fake_host_malloc_calls;
    fake_fail_host_free = true;
    const int second = cuda_model_stage_pool_alloc(21u);
    fake_fail_host_free = false;
    std::printf("began=%d\nfirst=%d\nsecond=%d\n"
                "malloc_before=%d\nmalloc_after=%d\n",
                began, first, second, malloc_before, fake_host_malloc_calls);
    emit_stage("after");
    return 0;
}
static int scenario_legacy_fence(void) {
    reset_case();
    g_cuda_tmp = reinterpret_cast<void *>(static_cast<uintptr_t>(0x77770000u));
    g_cuda_tmp_bytes = 8u;
    const int began = ds4_gpu_laguna_resident_observer_begin(&tracker);
    std::printf("began=%d\n", began);
    emit_tracker("after");
    return 0;
}
static int scenario_compact_fence(void) {
    reset_case();
    g_laguna_compact_state.store(DS4_LAGUNA_COMPACT_ACTIVE, std::memory_order_relaxed);
    const int began = ds4_gpu_laguna_resident_observer_begin(&tracker);
    std::printf("began=%d\n", began);
    emit_tracker("after");
    return 0;
}
static int scenario_owner_end_fence(void) {
    reset_case();
    const int began = ds4_gpu_laguna_resident_observer_begin(&tracker);
    void *ptr = nullptr;
    const cudaError_t allocated = cuda_laguna_resident_malloc(&ptr, 9u, 1u);
    const int while_live = ds4_gpu_laguna_resident_observer_end(&tracker);
    const cudaError_t freed = cuda_laguna_resident_free(ptr);
    const int after_free = ds4_gpu_laguna_resident_observer_end(&tracker);
    std::printf("began=%d\nallocated=%d\nwhile_live=%d\nfreed=%d\nafter_free=%d\n",
                began, (int)allocated, while_live, (int)freed, after_free);
    emit_tracker("after");
    return 0;
}
int main(int argc, char **argv) {
    alarm(15);
    struct rlimit limit = {0, 0};
    (void)setrlimit(RLIMIT_CORE, &limit);
    if (argc != 2) return 2;
    const std::string name(argv[1]);
    if (name == "alloc-observe") return scenario_alloc_observe();
    if (name == "free-retry") return scenario_free_retry();
    if (name == "capacity") return scenario_capacity();
    if (name == "unknown") return scenario_unknown();
    if (name == "alloc-failure") return scenario_alloc_failure();
    if (name == "bound-rollback") return scenario_bound_rollback(false);
    if (name == "bound-fallback") return scenario_bound_rollback(true);
    if (name == "unrecorded-rollback") return scenario_unrecorded_rollback();
    if (name == "partial-device-safe") return scenario_partial_allocation(false, false);
    if (name == "partial-device-retained") return scenario_partial_allocation(false, true);
    if (name == "partial-host-safe") return scenario_partial_allocation(true, false);
    if (name == "partial-host-retained") return scenario_partial_allocation(true, true);
    if (name == "id-exhaustion") return scenario_id_exhaustion();
    if (name == "arena-observe") return scenario_arena_observe();
    if (name == "arena-release-failure") return scenario_arena_release_failure();
    if (name == "host-free-retry") return scenario_host_free_retry();
    if (name == "wrong-free-host") return scenario_wrong_free(true);
    if (name == "wrong-free-device") return scenario_wrong_free(false);
    if (name == "live-relation") return scenario_live_relation();
    if (name == "scratch-growth") return scenario_scratch_growth(false);
    if (name == "scratch-free-failure") return scenario_scratch_growth(true);
    if (name == "scratch-alloc-failure") return scenario_scratch_alloc_failure();
    if (name == "stage-pool") return scenario_stage_pool();
    if (name == "stage-pool-resize-failure") return scenario_stage_pool_resize_failure();
    if (name == "legacy-fence") return scenario_legacy_fence();
    if (name == "compact-fence") return scenario_compact_fence();
    if (name == "owner-end-fence") return scenario_owner_end_fence();
    return 2;
}
'''

# Definitions are ordered after declarations and observer block.  Missing
# bodies intentionally remain absent so source/link RED is explicit.
ACTUAL_BODIES = "\n".join(
    x for x in (LEGACY_PERMIT, ALIGN, STAGE_RELEASE, MODEL_POOL, TMP,
                ARENA_LIMIT, ARENA_CHUNK, ARENA_ALLOC, PROGRESS_RESET, RANGE_RELEASE)
    if x is not None
)
FIXTURE_SOURCE = FAKE_PREFIX + "\n" + (TMP_DECLS or "") + r'''
static void *cuda_tmp_alloc(uint64_t, const char *);
static void *cuda_align_ptr(void *, uint64_t);
static int cuda_stage_slots_release(void **, void **, cudaEvent_t *, uint64_t *);
static int cuda_model_stage_pool_alloc(uint64_t);
''' + "\n" + (OBSERVER_BLOCK or "") + "\n" + ACTUAL_BODIES + "\n" + FAKE_SUFFIX


def parse_output(stdout: str) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator and re.fullmatch(r"-?\d+", value):
            values[key] = int(value)
    return values


def value(values: dict[str, int], key: str) -> int:
    if key not in values:
        raise AssertionError(f"missing {key}: {values}")
    return values[key]


class ResidentFixture(unittest.TestCase):
    '''Compile and execute only generated fake-driver children.'''

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory(prefix="ds4-native-owner-contract-")
        cls.addClassCleanup(cls._tmp.cleanup)
        base = Path(cls._tmp.name)
        cls._env = dict(SAFE_ENV)
        for key, leaf in (("HOME", "home"), ("TMPDIR", "tmp"),
                          ("XDG_CONFIG_HOME", "config"),
                          ("XDG_CACHE_HOME", "cache"),
                          ("XDG_DATA_HOME", "data")):
            child = base / leaf
            child.mkdir()
            cls._env[key] = str(child)
        cls._source = base / "fixture.cc"
        cls._runtime_object = base / "ds4_runtime.o"
        cls._binary = base / "fixture"
        cls._source.write_text(FIXTURE_SOURCE, encoding="utf-8")
        cls._runtime_compile = subprocess.run(
            ["cc", "-std=c11", "-O0", "-I", str(ROOT), "-c", str(RUNTIME_SOURCE),
             "-o", str(cls._runtime_object)],
            cwd=base, env=cls._env, capture_output=True, text=True,
            timeout=15, check=False,
        )
        if cls._runtime_compile.returncode == 0:
            cls._compile = subprocess.run(
                ["c++", "-std=c++17", "-O0", "-pthread", "-I", str(ROOT),
                 str(cls._source), str(cls._runtime_object), "-o", str(cls._binary)],
                cwd=base, env=cls._env, capture_output=True, text=True,
                timeout=15, check=False,
            )
        else:
            cls._compile = cls._runtime_compile

    def run_case(self, name: str) -> dict[str, int]:
        self.assertEqual(
            self._compile.returncode, 0,
            "resident native fixture compiler RED:\n" + self._compile.stderr,
        )
        result = subprocess.run(
            [str(self._binary), name], cwd=self._binary.parent, env=self._env,
            capture_output=True, text=True, timeout=16, check=False,
        )
        self.assertEqual(
            result.returncode, 0,
            f"resident native scenario {name} RED:\nstdout={result.stdout}\nstderr={result.stderr}",
        )
        values = parse_output(result.stdout)
        for key, count in values.items():
            if key.endswith("_api_errors"):
                self.assertEqual(count, 0, f"fake driver protocol or free ordering violation: {key}")
        return values

    def assert_compile_prerequisite(self) -> None:
        self.assertEqual(
            self._compile.returncode, 0,
            "resident native compiler prerequisite RED:\n" + self._compile.stderr,
        )

    def test_00_required_real_source_seams_exist(self) -> None:
        self.assertFalse(
            MISSING_SOURCE_SEAMS,
            "resident native source RED; missing actual seams: "
            + ", ".join(MISSING_SOURCE_SEAMS),
        )

    def test_01_required_abi_driver_compiles(self) -> None:
        self.assert_compile_prerequisite()

    def test_02_successful_allocations_emit_namespace_and_exact_records(self) -> None:
        values = self.run_case("alloc-observe")
        self.assertEqual(value(values, "began"), 1)
        self.assertEqual(value(values, "alloc_static"), 0)
        self.assertEqual(value(values, "alloc_other"), 0)
        self.assertEqual(value(values, "alloc_host"), 0)
        self.assertEqual(value(values, "after_record_count"), 3)
        self.assertEqual(value(values, "after_active_records"), 3)
        self.assertEqual(value(values, "after_static_current"), 17)
        self.assertEqual(value(values, "after_other_current"), 23)
        self.assertEqual(value(values, "after_pinned_current"), 19)
        self.assertEqual(value(values, "after_static_peak"), 17)
        self.assertEqual(value(values, "after_other_peak"), 23)
        self.assertEqual(value(values, "after_pinned_peak"), 19)
        self.assertEqual(value(values, "after_device_live"), 2)
        self.assertEqual(value(values, "after_host_live"), 1)
        self.assertEqual(value(values, "after_device_malloc_calls"), 2)
        self.assertEqual(value(values, "after_host_malloc_calls"), 1)
        for index, callsite, category, domain, requested in (
            (0, 1, 0, 1, 17), (1, 22, 7, 1, 23), (2, 11, 5, 0, 19)
        ):
            self.assertEqual(value(values, f"after_record_{index}_id") >> 56, 0x52)
            pointer_key = "static_ptr" if index == 0 else "other_ptr" if index == 1 else "host_ptr"
            self.assertEqual(value(values, f"after_record_{index}_base"), value(values, pointer_key))
            self.assertEqual(value(values, f"after_record_{index}_requested"), requested)
            self.assertEqual(value(values, f"after_record_{index}_charged"), requested)
            self.assertEqual(value(values, f"after_record_{index}_callsite"), callsite)
            self.assertEqual(value(values, f"after_record_{index}_category"), category)
            self.assertEqual(value(values, f"after_record_{index}_domain"), domain)
            self.assertEqual(value(values, f"after_record_{index}_live"), 1)
        self.assertEqual(value(values, "after_violation"), 0)

    def test_03_free_failure_retains_identity_and_retry_retires_owner(self) -> None:
        values = self.run_case("free-retry")
        self.assertEqual(value(values, "began"), 1)
        self.assertEqual(value(values, "allocated"), 0)
        self.assertNotEqual(value(values, "first"), 0)
        self.assertEqual(value(values, "after_record_count"), 1)
        self.assertEqual(value(values, "after_active_records"), 0)
        self.assertEqual(value(values, "after_device_live"), 0)
        self.assertEqual(value(values, "after_device_free_calls"), 2)
        self.assertNotEqual(value(values, "after_violation"), 0)
        self.assertEqual(value(values, "after_record_0_live"), 0)
        # The owner is retired; sticky-failure detach policy is root-owned.

    def test_04_capacity_and_unknown_site_are_preflighted_without_driver_calls(self) -> None:
        capacity = self.run_case("capacity")
        self.assertEqual(value(capacity, "began"), 1)
        self.assertEqual(value(capacity, "first"), 0)
        self.assertNotEqual(value(capacity, "second"), 0)
        self.assertEqual(value(capacity, "calls_before"), 1)
        self.assertEqual(value(capacity, "after_device_malloc_calls"), 1)
        self.assertEqual(value(capacity, "after_active_records"), 1)
        self.assertEqual(value(capacity, "after_violation"), 6)  # CAPACITY

        unknown = self.run_case("unknown")
        self.assertEqual(value(unknown, "began"), 1)
        self.assertNotEqual(value(unknown, "result"), 0)
        self.assertEqual(value(unknown, "after_device_malloc_calls"), 0)
        self.assertEqual(value(unknown, "after_record_count"), 0)
        self.assertEqual(value(unknown, "after_active_records"), 0)
        self.assertEqual(value(unknown, "after_violation"), 3)  # UNKNOWN_CALLSITE

    def test_05_driver_allocation_failure_has_no_tracker_observation(self) -> None:
        values = self.run_case("alloc-failure")
        self.assertEqual(value(values, "began"), 1)
        self.assertNotEqual(value(values, "result"), 0)
        self.assertEqual(value(values, "ptr"), 0)
        self.assertEqual(value(values, "after_device_malloc_calls"), 1)
        self.assertEqual(value(values, "after_record_count"), 0)
        self.assertEqual(value(values, "after_active_records"), 0)
        self.assertEqual(value(values, "after_static_current"), 0)
        self.assertEqual(value(values, "after_violation"), 0)

    def test_06_bounds_failure_after_insertion_is_sticky_and_safe_rollback_retires(self) -> None:
        values = self.run_case("bound-rollback")
        self.assertEqual(value(values, "began"), 1)
        self.assertEqual(value(values, "first"), 0)
        self.assertNotEqual(value(values, "second"), 0)
        self.assertNotEqual(value(values, "third"), 0)
        self.assertEqual(value(values, "malloc_before"), 1)
        self.assertEqual(value(values, "malloc_after"), 2)
        self.assertEqual(value(values, "after_device_malloc_calls"), 2)
        self.assertEqual(value(values, "after_device_free_calls"), 1)
        self.assertEqual(value(values, "after_device_live"), 1)
        self.assertEqual(value(values, "after_active_records"), 1)
        self.assertEqual(value(values, "after_static_current"), 16)
        self.assertEqual(value(values, "after_static_peak"), 17)
        self.assertNotEqual(value(values, "after_violation"), 0)
        # Safe physical rollback leaves no owner; the sticky violation itself
        # is asserted above, while end-detach policy remains root-owned.

    def test_07_failed_rollback_retains_fallback_physical_owner(self) -> None:
        values = self.run_case("bound-fallback")
        self.assertEqual(value(values, "began"), 1)
        self.assertNotEqual(value(values, "second"), 0)
        self.assertNotEqual(value(values, "third"), 0)
        self.assertEqual(value(values, "after_device_malloc_calls"), 2)
        self.assertEqual(value(values, "after_device_free_calls"), 1)
        self.assertEqual(value(values, "after_device_live"), 2)
        self.assertEqual(value(values, "after_active_records"), 2)
        self.assertNotEqual(value(values, "after_violation"), 0)
        self.assertEqual(value(values, "ended"), 0)

    def test_08_actual_scratch_growth_reuse_and_peak(self) -> None:
        values = self.run_case("scratch-growth")
        self.assertEqual(value(values, "began"), 1)
        self.assertNotEqual(value(values, "first"), 0)
        self.assertEqual(value(values, "first"), value(values, "reuse"))
        self.assertNotEqual(value(values, "grown"), 0)
        self.assertNotEqual(value(values, "grown"), value(values, "first"))
        self.assertEqual(value(values, "malloc_after_first"), 1)
        self.assertEqual(value(values, "malloc_after_reuse"), 1)
        self.assertEqual(value(values, "malloc_after_growth"), 2)
        self.assertEqual(value(values, "after_device_free_calls"), 1)
        self.assertEqual(value(values, "after_record_count"), 1)  # Reused tombstone.
        self.assertEqual(value(values, "after_record_0_id") & ((1 << 56) - 1), 2)
        self.assertEqual(value(values, "after_active_records"), 1)
        self.assertEqual(value(values, "after_other_current"), 21)
        self.assertEqual(value(values, "after_other_peak"), 21)
        self.assertEqual(value(values, "after_tmp_bytes"), 21)
        self.assertEqual(value(values, "final_active_records"), 0)
        self.assertEqual(value(values, "final_other_current"), 0)
        self.assertEqual(value(values, "final_other_peak"), 21)
        self.assertEqual(value(values, "ended"), 1)

    def test_09_scratch_free_failure_retains_old_owner_and_does_not_grow(self) -> None:
        values = self.run_case("scratch-free-failure")
        self.assertEqual(value(values, "began"), 1)
        self.assertNotEqual(value(values, "first"), 0)
        self.assertEqual(value(values, "grown"), 0)  # Old slab is too small to return.
        self.assertEqual(value(values, "after_tmp_ptr"), value(values, "first"))
        self.assertEqual(value(values, "malloc_after_first"), 1)
        self.assertEqual(value(values, "malloc_after_reuse"), 1)
        self.assertEqual(value(values, "malloc_after_growth"), 1)
        self.assertEqual(value(values, "after_device_free_calls"), 1)
        self.assertEqual(value(values, "after_device_live"), 1)
        self.assertEqual(value(values, "after_active_records"), 1)
        self.assertEqual(value(values, "after_other_current"), 13)
        self.assertEqual(value(values, "after_tmp_bytes"), 13)
        self.assertNotEqual(value(values, "after_violation"), 0)
        # Retry ownership is asserted by the final snapshot; detach policy
        # after a sticky failure is intentionally left to root implementation.

    def test_10_scratch_driver_failure_has_no_new_observation(self) -> None:
        values = self.run_case("scratch-alloc-failure")
        self.assertEqual(value(values, "began"), 1)
        self.assertEqual(value(values, "ptr"), 0)
        self.assertEqual(value(values, "after_device_malloc_calls"), 1)
        self.assertEqual(value(values, "after_record_count"), 0)
        self.assertEqual(value(values, "after_active_records"), 0)
        self.assertEqual(value(values, "after_other_current"), 0)
        self.assertEqual(value(values, "after_violation"), 0)

    def test_11_actual_model_stage_pool_charges_raw_aligned_reservations(self) -> None:
        values = self.run_case("stage-pool")
        self.assertEqual(value(values, "began"), 1)
        self.assertEqual(value(values, "allocated"), 1)
        self.assertEqual(value(values, "ready_record_count"), 4)
        self.assertEqual(value(values, "ready_active_records"), 4)
        self.assertEqual(value(values, "ready_host_malloc_calls"), 4)
        self.assertEqual(value(values, "ready_pinned_current"), 80)
        self.assertEqual(value(values, "ready_pinned_peak"), 80)
        for index in range(4):
            self.assertEqual(value(values, f"ready_record_{index}_callsite"), 11)
            self.assertEqual(value(values, f"ready_record_{index}_category"), 5)
            self.assertEqual(value(values, f"ready_record_{index}_domain"), 0)
            self.assertEqual(value(values, f"ready_record_{index}_requested"), 20)
            self.assertEqual(value(values, f"ready_record_{index}_charged"), 20)
            self.assertEqual(value(values, f"ready_record_{index}_live"), 1)
            raw = value(values, f"ready_raw_{index}")
            aligned = value(values, f"ready_stage_{index}")
            self.assertEqual(aligned % 8, 0)
            self.assertEqual(aligned - raw, 5)
            self.assertEqual(value(values, f"ready_reserved_{index}"), 20)
        self.assertEqual(value(values, "released"), 1)
        self.assertEqual(value(values, "released_active_records"), 0)
        self.assertEqual(value(values, "released_pinned_current"), 0)
        self.assertEqual(value(values, "released_host_free_calls"), 4)
        self.assertEqual(value(values, "released_violation"), 0)
        self.assertEqual(value(values, "ended"), 1)

    def test_12_actual_model_stage_growth_free_failure_does_not_allocate_new_pool(self) -> None:
        values = self.run_case("stage-pool-resize-failure")
        self.assertEqual(value(values, "began"), 1)
        self.assertEqual(value(values, "first"), 1)
        self.assertEqual(value(values, "second"), 0)
        self.assertEqual(value(values, "malloc_before"), 4)
        self.assertEqual(value(values, "malloc_after"), 4)
        self.assertEqual(value(values, "after_host_malloc_calls"), 4)
        self.assertNotEqual(value(values, "after_host_free_calls"), 0)
        self.assertNotEqual(value(values, "after_violation"), 0)
        self.assertGreater(value(values, "after_active_records"), 0)

    def test_13_begin_end_are_mutually_exclusive_and_fenced_by_legacy_owners(self) -> None:
        legacy = self.run_case("legacy-fence")
        self.assertEqual(value(legacy, "began"), 0)
        self.assertEqual(value(legacy, "after_device_malloc_calls"), 0)
        self.assertEqual(value(legacy, "after_host_malloc_calls"), 0)
        self.assertEqual(value(legacy, "after_record_count"), 0)

        compact = self.run_case("compact-fence")
        self.assertEqual(value(compact, "began"), 0)
        self.assertEqual(value(compact, "after_device_malloc_calls"), 0)
        self.assertEqual(value(compact, "after_record_count"), 0)

        owners = self.run_case("owner-end-fence")
        self.assertEqual(value(owners, "began"), 1)
        self.assertEqual(value(owners, "allocated"), 0)
        self.assertEqual(value(owners, "while_live"), 0)
        self.assertEqual(value(owners, "freed"), 0)
        self.assertEqual(value(owners, "after_free"), 1)
        self.assertEqual(value(owners, "after_active_records"), 0)


    def test_14_unrecorded_rollback_retains_native_owner_until_physical_retry(self) -> None:
        values = self.run_case("unrecorded-rollback")
        self.assertEqual(value(values, "began"), 1)
        self.assertNotEqual(value(values, "allocated"), 0)
        self.assertEqual(value(values, "ptr"), 0)
        self.assertEqual(value(values, "blocked_end"), 0)
        self.assertEqual(value(values, "retained_device_live"), 1)
        self.assertEqual(value(values, "retained_record_count"), 0)
        self.assertEqual(value(values, "retained_attributed_valid"), 0)
        self.assertNotEqual(value(values, "retained_violation"), 0)
        self.assertEqual(value(values, "recovered_end"), 1)
        self.assertEqual(value(values, "recovered_device_live"), 0)
        self.assertEqual(value(values, "recovered_active_records"), 0)
        self.assertEqual(value(values, "recovered_violation"), value(values, "retained_violation"))

    def test_15_error_with_nonnull_storage_is_not_lost_or_reported_safe(self) -> None:
        for domain in ("device", "host"):
            for disposition in ("safe", "retained"):
                with self.subTest(domain=domain, disposition=disposition):
                    values = self.run_case(f"partial-{domain}-{disposition}")
                    self.assertEqual(value(values, "began"), 1)
                    self.assertNotEqual(value(values, "result"), 0)
                    self.assertEqual(value(values, "ptr"), 0)
                    self.assertEqual(value(values, "after_record_count"), 0)
                    self.assertNotEqual(value(values, "after_violation"), 0)
                    self.assertEqual(value(values, "after_attributed_valid"), 0)
                    self.assertEqual(value(values, f"after_{domain}_malloc_calls"), 1)
                    self.assertEqual(value(values, f"after_{domain}_free_calls"), 1)
                    retained = disposition == "retained"
                    self.assertEqual(value(values, f"after_{domain}_live"), int(retained))
                    self.assertEqual(value(values, "ended"), int(not retained))
                    if retained:
                        self.assertEqual(value(values, "recovered"), 1)
                    self.assertEqual(value(values, f"final_{domain}_live"), 0)
                    self.assertEqual(value(values, "final_violation"), value(values, "after_violation"))

    def test_16_exhausted_producer_id_refuses_before_driver_allocation(self) -> None:
        values = self.run_case("id-exhaustion")
        self.assertEqual(value(values, "began"), 1)
        self.assertNotEqual(value(values, "result"), 0)
        self.assertEqual(value(values, "ptr"), 0)
        self.assertEqual(value(values, "after_device_malloc_calls"), 0)
        self.assertEqual(value(values, "after_record_count"), 0)
        self.assertNotEqual(value(values, "after_violation"), 0)

    def test_17_actual_arena_charges_reservation_not_reused_logical_spans(self) -> None:
        values = self.run_case("arena-observe")
        chunk = 1792 * 1048576
        self.assertEqual(value(values, "began"), 1)
        self.assertEqual(value(values, "chunk"), chunk)
        self.assertNotEqual(value(values, "first"), 0)
        self.assertEqual(value(values, "second") - value(values, "first"), 256)
        self.assertNotEqual(value(values, "third"), 0)
        self.assertEqual(value(values, "ready_arena_count"), 2)
        self.assertEqual(value(values, "ready_device_malloc_calls"), 2)
        self.assertEqual(value(values, "ready_record_count"), 2)
        self.assertEqual(value(values, "ready_static_current"), 2 * chunk)
        self.assertEqual(value(values, "ready_static_peak"), 2 * chunk)
        for i in range(2):
            self.assertEqual(value(values, f"ready_record_{i}_charged"), chunk)
            self.assertEqual(value(values, f"ready_record_{i}_requested"), chunk)
            self.assertEqual(value(values, f"ready_record_{i}_callsite"), 1)
        self.assertEqual(value(values, "ended"), 1)
        self.assertEqual(value(values, "final_device_live"), 0)
        self.assertEqual(value(values, "final_device_free_calls"), 2)
        self.assertEqual(value(values, "final_arena_count"), 0)
        self.assertEqual(value(values, "final_static_current"), 0)
        self.assertEqual(value(values, "final_static_peak"), 2 * chunk)

    def test_18_partial_arena_teardown_retains_failed_unvisited_and_sticky_peak(self) -> None:
        values = self.run_case("arena-release-failure")
        self.assertEqual(value(values, "began"), 1)
        self.assertEqual(value(values, "blocked"), 0)
        self.assertEqual(value(values, "retained_device_free_calls"), 2)
        self.assertEqual(value(values, "retained_device_live"), 2)
        self.assertEqual(value(values, "retained_static_current"), 44)
        self.assertEqual(value(values, "retained_static_peak"), 61)
        self.assertEqual(value(values, "retained_release_failed"), 1)
        self.assertEqual(value(values, "retained_arena_0_ptr"), 0)
        self.assertEqual(value(values, "retained_range_0_ptr"), 0)
        for i, label in ((1, "second_ptr"), (2, "third_ptr")):
            self.assertEqual(value(values, f"retained_arena_{i}_ptr"), value(values, label))
            self.assertEqual(value(values, f"retained_range_{i}_ptr"), value(values, label))
        self.assertNotEqual(value(values, "retained_violation"), 0)
        self.assertEqual(value(values, "ended"), 1)
        self.assertEqual(value(values, "final_device_free_calls"), 4)
        self.assertEqual(value(values, "final_device_live"), 0)
        self.assertEqual(value(values, "final_static_current"), 0)
        self.assertEqual(value(values, "final_static_peak"), 61)
        self.assertEqual(value(values, "final_release_failed"), 0)
        self.assertEqual(value(values, "final_arena_count"), 0)
        self.assertEqual(value(values, "final_range_count"), 0)
        self.assertEqual(value(values, "final_violation"), value(values, "retained_violation"))

    def test_19_range_release_callers_stop_before_rebinding_or_more_cleanup(self) -> None:
        cases = (
            ("extern \"C\" int ds4_gpu_cleanup_checked(void)", False),
            ("extern \"C\" int ds4_gpu_set_model_map(", True),
            ("extern \"C\" int ds4_gpu_register_model_map_no_copy(", True),
        )
        for signature, model_rebind in cases:
            with self.subTest(caller=signature):
                body = extract_definition(CUDA_SOURCE, signature)
                self.assertIsNotNone(body)
                code = body or ""
                if model_rebind:
                    self.assertRegex(
                        code,
                        r"if\s*\(\s*!cuda_model_range_release_all\(\)\s*\)\s*return\s*0\s*;",
                    )
                    self.assertIn("g_model_range_release_failed", code)
                    self.assertLess(
                        code.index("g_model_range_release_failed"),
                        code.index("g_model_host_base == model_map"),
                    )
                    continue

                self.assertRegex(
                    code,
                    r"if\s*\(\s*!cuda_model_range_release_all\(\)\s*\)\s*goto\s+cleanup_refused\s*;",
                )
                label = code.index("cleanup_refused:")
                latch = code.index("cuda_laguna_resident_note_failure();", label)
                returned = code.index("return 0;", latch)
                self.assertLess(label, latch)
                self.assertLess(latch, returned)

    def test_20_owned_host_contract_is_in_resident_and_default_aggregates(self) -> None:
        makefile = (ROOT / "Makefile").read_text(encoding="utf-8").replace("\\\n", " ")
        target = "test-cuda-resident-ownership-contract"
        self.assertRegex(makefile, r"(?m)^" + target + r":\s*\n\tpython3 tests/test_cuda_resident_ownership_contract\.py(?: -v)?\s*$")
        phony = set()
        for line in makefile.splitlines():
            if line.startswith(".PHONY:"):
                phony.update(line.split(":", 1)[1].split())
        self.assertIn(target, phony)
        resident = re.search(r"(?m)^test-laguna-resident-path:([^\n]*)", makefile)
        default = re.search(r"(?m)^test:([^\n]*)", makefile)
        self.assertIsNotNone(resident)
        self.assertIsNotNone(default)
        self.assertIn(target, resident.group(1).split() if resident else [])
        self.assertIn("test-laguna-resident-path", default.group(1).split() if default else [])

    def test_21_pinned_free_failure_retains_charge_and_retry_preserves_peak(self) -> None:
        values = self.run_case("host-free-retry")
        self.assertEqual(value(values, "began"), 1)
        self.assertNotEqual(value(values, "failed"), 0)
        self.assertEqual(value(values, "blocked"), 0)
        self.assertEqual(value(values, "retained_host_live"), 1)
        self.assertEqual(value(values, "retained_owned_current"), 19)
        self.assertEqual(value(values, "retained_record_0_live"), 1)
        self.assertEqual(value(values, "freed"), 0)
        self.assertEqual(value(values, "ended"), 1)
        self.assertEqual(value(values, "final_host_free_calls"), 2)
        self.assertEqual(value(values, "final_host_live"), 0)
        self.assertEqual(value(values, "final_owned_current"), 0)
        self.assertEqual(value(values, "final_owned_peak"), 19)
        self.assertEqual(value(values, "final_qualification_peak"), 19)
        self.assertNotEqual(value(values, "retained_violation"), 0)
        self.assertEqual(value(values, "final_violation"), value(values, "retained_violation"))

    def test_22_wrong_domain_and_double_free_refuse_without_physical_side_effects(self) -> None:
        for domain in ("host", "device"):
            with self.subTest(domain=domain):
                values = self.run_case(f"wrong-free-{domain}")
                self.assertEqual(value(values, "began"), 1)
                self.assertNotEqual(value(values, "wrong"), 0)
                self.assertEqual(value(values, "retained_host_free_calls"), 0)
                self.assertEqual(value(values, "retained_device_free_calls"), 0)
                self.assertEqual(value(values, "retained_owned_current"), 19)
                self.assertEqual(value(values, "retained_record_0_live"), 1)
                self.assertEqual(value(values, "freed"), 0)
                self.assertNotEqual(value(values, "duplicate"), 0)
                self.assertEqual(value(values, f"final_{domain}_free_calls"), 1)
                self.assertEqual(value(values, "final_owned_current"), 0)
                self.assertEqual(value(values, "final_owned_peak"), 19)
                self.assertEqual(value(values, "ended"), 1)

    def test_23_live_relation_refuses_before_free_without_double_charging_owner(self) -> None:
        values = self.run_case("live-relation")
        self.assertEqual(value(values, "began"), 1)
        self.assertEqual(value(values, "related"), 0)  # Independent positive control.
        self.assertEqual(value(values, "related_violation"), 0)
        self.assertEqual(value(values, "related_active_records"), 2)
        self.assertEqual(value(values, "related_owned_current"), 19)
        self.assertEqual(value(values, "related_qualification_current"), 19)
        self.assertEqual(value(values, "related_registered_current"), 19)
        self.assertEqual(value(values, "related_record_0_relation"), 0)
        self.assertEqual(value(values, "related_record_1_relation"), 2)
        self.assertNotEqual(value(values, "failed"), 0)
        self.assertEqual(value(values, "retained_host_free_calls"), 0)
        self.assertEqual(value(values, "retained_host_live"), 1)
        self.assertEqual(value(values, "retained_active_records"), 2)
        self.assertEqual(value(values, "freed"), 0)
        self.assertEqual(value(values, "ended"), 1)
        self.assertEqual(value(values, "final_active_records"), 0)
        self.assertEqual(value(values, "final_owned_current"), 0)
        self.assertEqual(value(values, "final_owned_peak"), 19)
        self.assertEqual(value(values, "final_registered_current"), 0)
        self.assertEqual(value(values, "final_registered_peak"), 19)
        self.assertEqual(value(values, "final_qualification_peak"), 19)
        self.assertEqual(value(values, "final_violation"), value(values, "retained_violation"))

if __name__ == "__main__":
    unittest.main(verbosity=2)
