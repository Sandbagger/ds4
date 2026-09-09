#!/usr/bin/env python3
'''Host-only contract for root-owned native host payload ownership.

The fixture extracts the real marked resident observer and host ownership
bodies from ``ds4_cuda.cu``.  It links those bodies against the real
``ds4_runtime.c`` as a separate C translation unit and a C-safe header caller.
The only physical host allocations are tiny calloc buffers.  CUDA boundaries
are deterministic sentinels and are never needed by host work.
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
RESIDENT_HEADER = ROOT / "ds4_gpu_resident.h"
MAKEFILE_SOURCE = (ROOT / "Makefile").read_text(encoding="utf-8")
SAFE_ENV = {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"}
OBSERVER_START_MARKER = "/* Resident native allocation observer."
OBSERVER_END_MARKER = "/* End resident native allocation observer. */"
HOST_START_MARKER = "/* Resident native host ownership. */"
HOST_END_MARKER = "/* End resident native host ownership. */"


def extract_marked_block(source: str, start_marker: str, end_marker: str) -> str | None:
    '''Extract a whole-line source block without accepting prose hits.'''
    start = source.find(start_marker)
    if start < 0:
        return None
    line_start = source.rfind("\n", 0, start) + 1
    if source[line_start:start].strip():
        return None
    end = source.find(end_marker, start + len(start_marker))
    if end < 0 or source[source.rfind("\n", 0, end) + 1 : end].strip():
        return None
    end_line = source.find("\n", end)
    return source[line_start:end_line if end_line >= 0 else len(source)]


OBSERVER_BLOCK = extract_marked_block(CUDA_SOURCE, OBSERVER_START_MARKER, OBSERVER_END_MARKER)
HOST_BLOCK = extract_marked_block(CUDA_SOURCE, HOST_START_MARKER, HOST_END_MARKER)
HEADER_SOURCE = RESIDENT_HEADER.read_text(encoding="utf-8")
HEADER_SEAMS = {
    "host owner struct": re.search(
        r"typedef struct\s*\{\s*void\s*\*base;\s*uint64_t\s+allocation_record_id;\s*\}\s*ds4_gpu_laguna_resident_host_owner\s*;",
        HEADER_SOURCE),
    "host calloc declaration": re.search(r"ds4_gpu_laguna_resident_host_calloc\s*\(", HEADER_SOURCE),
    "host free declaration": re.search(r"ds4_gpu_laguna_resident_host_free\s*\(", HEADER_SOURCE),
}
MISSING_SOURCE_SEAMS = [label for label, present in (
    ("resident observer block", OBSERVER_BLOCK),
    ("resident host block", HOST_BLOCK),
    ("resident host calloc body", HOST_BLOCK and "ds4_gpu_laguna_resident_host_calloc" in HOST_BLOCK),
    ("resident host free body", HOST_BLOCK and "ds4_gpu_laguna_resident_host_free" in HOST_BLOCK),
    *HEADER_SEAMS.items()) if not present]

ABI_PROBE = r'''
#include "ds4_runtime.h"
#include "ds4_gpu_resident.h"
int resident_host_header_probe_c(ds4_runtime_tracker *tracker) {
    ds4_gpu_laguna_resident_host_owner owner = {0};
    if (!ds4_gpu_laguna_resident_host_calloc(tracker, 15u, 1u, 1u, &owner)) return 0;
    return ds4_gpu_laguna_resident_host_free(tracker, &owner);
}
'''

FAKE_PREFIX = r'''

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <sys/resource.h>
#include <unistd.h>
#include "ds4_laguna_resident.h"
#include "ds4_gpu_resident.h"
#include "ds4_runtime.h"
extern "C" int resident_host_header_probe_c(ds4_runtime_tracker *tracker);
using cudaError_t = int; using cudaStream_t = void *; using cudaEvent_t = void *;
static constexpr cudaError_t cudaSuccess = 0, cudaErrorInvalidValue = 1, kFakeCudaError = 17;
struct fake_cuda_allocation { size_t bytes; bool host; };
static std::unordered_map<void *, fake_cuda_allocation> fake_cuda_live;
static std::unordered_map<void *, size_t> fake_real_live;
static uintptr_t fake_next_handle = static_cast<uintptr_t>(0x10000000u);
static int fake_device_malloc_calls, fake_device_free_calls, fake_host_malloc_calls, fake_host_free_calls;
static int fake_device_query_calls, fake_device_sync_calls, fake_cuda_last_error_calls, fake_api_errors, fake_unknown_free_calls;
static int fake_calloc_calls, fake_free_calls, fake_fixture_protocol_errors, fake_free_order_violations; static bool fake_fail_calloc;
static size_t fake_real_current, fake_real_peak, fake_last_calloc_bytes;
static uint64_t fake_last_calloc_count, fake_last_calloc_size, fake_last_free_owned_current;
static bool fake_last_calloc_zeroed, fake_last_free_live_record; static uint8_t fake_expected_owner_namespace;
static ds4_runtime_tracker *fake_observed_tracker;
static void *g_cuda_tmp; static uint64_t g_cuda_tmp_bytes; static int g_n_gpus;
static std::vector<int> g_model_ranges, g_model_arenas, g_q8_f16_ranges, g_q8_f32_ranges;
static uint64_t g_model_range_bytes; static int g_model_range_release_failed;
static void *g_model_host_base, *g_model_device_base; static int g_model_registered, g_model_device_owned;
static uint64_t g_model_registered_size, g_model_stage_bytes, g_stream_selected_stage_bytes;
static void *g_model_stage_raw[4], *g_model_stage[4], *g_stream_selected_stage_raw[4], *g_stream_selected_stage[4];
static cudaEvent_t g_model_stage_event[4], g_stream_selected_stage_event[4];
static uint64_t g_model_stage_reserved_bytes[4], g_stream_selected_stage_reserved_bytes[4];
static cudaStream_t g_model_upload_stream, g_stream_selected_upload_stream;
enum { DS4_LAGUNA_COMPACT_IDLE = 0, DS4_LAGUNA_COMPACT_CREATING = 1, DS4_LAGUNA_COMPACT_ACTIVE = 2 };
static std::atomic<int> g_laguna_compact_state{DS4_LAGUNA_COMPACT_IDLE}; static std::recursive_mutex g_laguna_compact_mutex;
static void *fake_handle(void) { fake_next_handle += static_cast<uintptr_t>(UINT64_C(1) << 36); return reinterpret_cast<void *>(fake_next_handle); }
static bool fake_live_owner_record(void *p, uint8_t ns) {
    if (!fake_observed_tracker) return false;
    for (size_t i = 0; i < fake_observed_tracker->record_count; ++i) { const auto &r = fake_observed_tracker->records[i];
        if (r.live && (r.id >> 56u) == ns && r.base == (uint64_t)(uintptr_t)p) return true; }
    return false;
}
static void fake_reset(void) {
    fake_cuda_live.clear(); fake_real_live.clear(); fake_next_handle = static_cast<uintptr_t>(0x10000000u);
    fake_device_malloc_calls = fake_device_free_calls = fake_host_malloc_calls = fake_host_free_calls = 0;
    fake_device_query_calls = fake_device_sync_calls = fake_cuda_last_error_calls = fake_api_errors = fake_unknown_free_calls = 0;
    fake_calloc_calls = fake_free_calls = fake_fixture_protocol_errors = fake_free_order_violations = 0; fake_fail_calloc = false; fake_real_current = fake_real_peak = 0;
    fake_last_calloc_count = fake_last_calloc_size = fake_last_calloc_bytes = 0; fake_last_calloc_zeroed = false;
    fake_last_free_live_record = false; fake_last_free_owned_current = 0; fake_expected_owner_namespace = 0x48u; fake_observed_tracker = nullptr;
}
static cudaError_t cudaMalloc(void **out, size_t bytes) { ++fake_device_malloc_calls; if (!out) { ++fake_api_errors; return kFakeCudaError; } *out = fake_handle(); fake_cuda_live[*out] = {bytes, false}; return cudaSuccess; }
static cudaError_t cudaFree(void *p) { ++fake_device_free_calls; auto it = fake_cuda_live.find(p); if (it == fake_cuda_live.end() || it->second.host) { ++fake_api_errors; return kFakeCudaError; } fake_cuda_live.erase(it); return cudaSuccess; }
static cudaError_t cudaMallocHost(void **out, size_t bytes) { ++fake_host_malloc_calls; if (!out) { ++fake_api_errors; return kFakeCudaError; } *out = fake_handle(); fake_cuda_live[*out] = {bytes, true}; return cudaSuccess; }
static cudaError_t cudaFreeHost(void *p) { ++fake_host_free_calls; auto it = fake_cuda_live.find(p); if (it == fake_cuda_live.end() || !it->second.host) { ++fake_api_errors; return kFakeCudaError; } fake_cuda_live.erase(it); return cudaSuccess; }
static cudaError_t cudaGetDevice(int *out) { ++fake_device_query_calls; if (!out) { ++fake_api_errors; return kFakeCudaError; } *out = 0; return cudaSuccess; }
static cudaError_t cudaDeviceSynchronize(void) { ++fake_device_sync_calls; return cudaSuccess; }
static cudaError_t cudaGetLastError(void) { ++fake_cuda_last_error_calls; return cudaSuccess; }
static void *fake_forward_calloc(size_t n, size_t s) {
    ++fake_calloc_calls; fake_last_calloc_count = n; fake_last_calloc_size = s;
    if (s && n > SIZE_MAX / s) { ++fake_fixture_protocol_errors; return nullptr; }
    fake_last_calloc_bytes = n * s;
    if (fake_last_calloc_bytes > 4096u) { ++fake_fixture_protocol_errors; return nullptr; }
    if (fake_fail_calloc) return nullptr;
    void *p = std::calloc(n, s); if (!p) return nullptr; fake_last_calloc_zeroed = true;
    for (size_t i = 0; i < fake_last_calloc_bytes; ++i) if (((const unsigned char *)p)[i]) fake_last_calloc_zeroed = false;
    fake_real_live[p] = fake_last_calloc_bytes; fake_real_current += fake_last_calloc_bytes;
    if (fake_real_current > fake_real_peak) fake_real_peak = fake_real_current; return p;
}
static void fake_forward_free(void *p) {
    ++fake_free_calls; if (!p) return; auto it = fake_real_live.find(p);
    if (it == fake_real_live.end()) { ++fake_unknown_free_calls; ++fake_fixture_protocol_errors; ++fake_free_order_violations; return; }
    fake_last_free_live_record = fake_live_owner_record(p, fake_expected_owner_namespace);
    if (fake_observed_tracker) fake_last_free_owned_current = fake_observed_tracker->owned_total_current;
    /* Every physical payload free must see both its live record and the full
     * current total. These scenarios have only sensor-owned HOST payloads. */
    if (!fake_last_free_live_record || fake_last_free_owned_current != fake_real_current)
        ++fake_free_order_violations;
    fake_real_current -= it->second; std::free(p); fake_real_live.erase(it);
}
#define calloc(n, s) fake_forward_calloc((n), (s))
#define free(p) fake_forward_free((p))

'''

SCENARIOS = r'''

static ds4_runtime_tracker tracker, foreign_tracker;
static ds4_runtime_allocation_record records[64], foreign_records[64];
static ds4_runtime_callsite callsites[16];
static ds4_runtime_tracker_config tracker_config, foreign_config;
static const ds4_runtime_callsite initial_sites[] = {
    {1, "static", DS4_RUNTIME_CATEGORY_STATIC_WEIGHTS, DS4_RUNTIME_DOMAIN_CUDA_DEVICE, 64},
    {DS4_LAGUNA_RESIDENT_CALLSITE_OTHER_MANAGED, "managed", DS4_RUNTIME_CATEGORY_OTHER_CUDA, DS4_RUNTIME_DOMAIN_CUDA_MANAGED, 64},
    {3, "ledger", DS4_RUNTIME_CATEGORY_CACHE_METADATA_ADDRESS_TABLES, DS4_RUNTIME_DOMAIN_HOST, 64},
    {11, "pinned", DS4_RUNTIME_CATEGORY_PINNED_STAGING, DS4_RUNTIME_DOMAIN_HOST, 64},
    {22, "cuda", DS4_RUNTIME_CATEGORY_OTHER_CUDA, DS4_RUNTIME_DOMAIN_CUDA_DEVICE, 64},
    {15, "engine", DS4_RUNTIME_CATEGORY_OTHER_HOST, DS4_RUNTIME_DOMAIN_HOST, 64},
    {16, "model", DS4_RUNTIME_CATEGORY_OTHER_HOST, DS4_RUNTIME_DOMAIN_HOST, 64},
    {17, "bootstrap", DS4_RUNTIME_CATEGORY_OTHER_HOST, DS4_RUNTIME_DOMAIN_HOST, 64},
    {18, "vocab", DS4_RUNTIME_CATEGORY_OTHER_HOST, DS4_RUNTIME_DOMAIN_HOST, 64},
    {19, "session", DS4_RUNTIME_CATEGORY_OTHER_HOST, DS4_RUNTIME_DOMAIN_HOST, 64},
    {20, "tracker", DS4_RUNTIME_CATEGORY_OTHER_HOST, DS4_RUNTIME_DOMAIN_HOST, 64},
    {21, "serializer", DS4_RUNTIME_CATEGORY_OTHER_HOST, DS4_RUNTIME_DOMAIN_HOST, 64},
};
static int init_tracker(ds4_runtime_tracker *t, ds4_runtime_tracker_config *cfg,
                        ds4_runtime_allocation_record *rs, size_t cap,
                        uint64_t host_bound) {
    std::memset(t, 0, sizeof(*t)); std::memset(cfg, 0, sizeof(*cfg));
    std::memset(rs, 0, 64u * sizeof(rs[0])); std::memcpy(callsites, initial_sites, sizeof(initial_sites));
    cfg->callsites = callsites; cfg->callsite_count = sizeof(initial_sites) / sizeof(initial_sites[0]);
    cfg->records = rs; cfg->record_capacity = cap;
    for (size_t i = 0; i < cfg->callsite_count; ++i) {
        ds4_runtime_callsite *s = &callsites[i];
        if (s->id == DS4_LAGUNA_CALLSITE_LEDGER_ARRAYS ||
            (s->id >= DS4_LAGUNA_CALLSITE_OTHER_HOST_ENGINE && s->id <= DS4_LAGUNA_CALLSITE_OTHER_HOST_SERIALIZER)) s->bound_bytes = host_bound;
        cfg->category_bounds[s->category] += s->bound_bytes;
        cfg->owned_total_bound_bytes += s->bound_bytes;
    }
    cfg->report_bounds[DS4_RUNTIME_REPORT_MODEL_MAPPING_REGISTERED] = 64;
    cfg->qualification_total_bound_bytes = cfg->owned_total_bound_bytes;
    return ds4_runtime_tracker_init(t, cfg) == DS4_RUNTIME_STATUS_OK;
}
static void reset_legacy(void) {
    g_laguna_resident_tracker.store(nullptr, std::memory_order_relaxed);
    std::memset(&g_laguna_resident_retained, 0, sizeof(g_laguna_resident_retained)); g_laguna_resident_legacy_tensor_seen = false;
    g_cuda_tmp = nullptr; g_cuda_tmp_bytes = 0; g_n_gpus = 0; g_model_host_base = g_model_device_base = nullptr;
    g_model_registered = g_model_device_owned = 0; g_model_registered_size = g_model_range_bytes = 0; g_model_range_release_failed = 0;
    g_model_ranges.clear(); g_model_arenas.clear(); g_q8_f16_ranges.clear(); g_q8_f32_ranges.clear();
    g_model_stage_bytes = g_stream_selected_stage_bytes = 0; g_model_upload_stream = g_stream_selected_upload_stream = nullptr;
    std::memset(g_model_stage_raw, 0, sizeof(g_model_stage_raw)); std::memset(g_model_stage, 0, sizeof(g_model_stage));
    std::memset(g_model_stage_event, 0, sizeof(g_model_stage_event)); std::memset(g_model_stage_reserved_bytes, 0, sizeof(g_model_stage_reserved_bytes));
    std::memset(g_stream_selected_stage_raw, 0, sizeof(g_stream_selected_stage_raw)); std::memset(g_stream_selected_stage, 0, sizeof(g_stream_selected_stage));
    std::memset(g_stream_selected_stage_event, 0, sizeof(g_stream_selected_stage_event)); std::memset(g_stream_selected_stage_reserved_bytes, 0, sizeof(g_stream_selected_stage_reserved_bytes));
    g_laguna_compact_state.store(DS4_LAGUNA_COMPACT_IDLE, std::memory_order_relaxed);
}
static void reset_case(size_t cap = 32, uint64_t host_bound = 64) {
    fake_reset(); reset_legacy(); if (!init_tracker(&tracker, &tracker_config, records, cap, host_bound)) std::abort(); fake_observed_tracker = &tracker;
}
static ds4_runtime_callsite *site(uint32_t id) { for (size_t i = 0; i < tracker_config.callsite_count; ++i) if (callsites[i].id == id) return &callsites[i]; return nullptr; }
static ds4_runtime_allocation_record *live_record(uint64_t id) { for (size_t i = 0; i < tracker.record_count; ++i) if (tracker.records[i].live && tracker.records[i].id == id) return &tracker.records[i]; return nullptr; }
static int active_records(const ds4_runtime_tracker *t = &tracker) { int n = 0; for (size_t i = 0; i < t->record_count; ++i) n += t->records[i].live ? 1 : 0; return n; }
static int owner_nonzero(const ds4_gpu_laguna_resident_host_owner &o) { return o.base != nullptr && o.allocation_record_id != 0; }
static int owner_zero(const ds4_gpu_laguna_resident_host_owner &o) { return o.base == nullptr && o.allocation_record_id == 0; }
static int owner_equal(const ds4_gpu_laguna_resident_host_owner &a, const ds4_gpu_laguna_resident_host_owner &b) { return a.base == b.base && a.allocation_record_id == b.allocation_record_id; }
static int tracker_storage_equal(const ds4_runtime_tracker &a, const ds4_runtime_tracker &b, const ds4_runtime_allocation_record *ar, const ds4_runtime_allocation_record *br) {
    return std::memcmp(&a, &b, sizeof(a)) == 0 && std::memcmp(ar, br, 64u * sizeof(ar[0])) == 0;
}
static void emit_common(const char *p) {
    std::printf("%s_violation=%d\n%s_record_count=%llu\n%s_active_records=%d\n", p, (int)tracker.violation, p, (unsigned long long)tracker.record_count, p, active_records());
    std::printf("%s_host_current=%llu\n%s_host_peak=%llu\n%s_cache_current=%llu\n%s_cache_peak=%llu\n", p, (unsigned long long)tracker.category_current[DS4_RUNTIME_CATEGORY_OTHER_HOST], p, (unsigned long long)tracker.category_peak[DS4_RUNTIME_CATEGORY_OTHER_HOST], p, (unsigned long long)tracker.category_current[DS4_RUNTIME_CATEGORY_CACHE_METADATA_ADDRESS_TABLES], p, (unsigned long long)tracker.category_peak[DS4_RUNTIME_CATEGORY_CACHE_METADATA_ADDRESS_TABLES]);
    std::printf("%s_owned_current=%llu\n%s_owned_peak=%llu\n%s_real_current=%llu\n%s_real_peak=%llu\n%s_real_live=%d\n", p, (unsigned long long)tracker.owned_total_current, p, (unsigned long long)tracker.owned_total_peak, p, (unsigned long long)fake_real_current, p, (unsigned long long)fake_real_peak, p, (int)fake_real_live.size());
    std::printf("%s_calloc_calls=%d\n%s_free_calls=%d\n%s_last_calloc_count=%llu\n%s_last_calloc_size=%llu\n%s_last_calloc_bytes=%llu\n%s_last_calloc_zeroed=%d\n", p, fake_calloc_calls, p, fake_free_calls, p, (unsigned long long)fake_last_calloc_count, p, (unsigned long long)fake_last_calloc_size, p, (unsigned long long)fake_last_calloc_bytes, p, fake_last_calloc_zeroed ? 1 : 0);
    std::printf("%s_last_free_live_record=%d\n%s_last_free_owned_current=%llu\n%s_unknown_free_calls=%d\n%s_free_order_violations=%d\n%s_fixture_protocol_errors=%d\n%s_expected_owner_namespace=%u\n%s_api_errors=%d\n%s_cuda_calls=%d\n", p, fake_last_free_live_record ? 1 : 0, p, (unsigned long long)fake_last_free_owned_current, p, fake_unknown_free_calls, p, fake_free_order_violations, p, fake_fixture_protocol_errors, p, (unsigned)fake_expected_owner_namespace, p, fake_api_errors, p, fake_device_malloc_calls + fake_device_free_calls + fake_host_malloc_calls + fake_host_free_calls + fake_device_query_calls + fake_device_sync_calls + fake_cuda_last_error_calls);
}

static int scenario_success(void) {
    reset_case(); ds4_gpu_laguna_resident_host_owner o = {};
    int began = ds4_gpu_laguna_resident_observer_begin(&tracker); int allocated = ds4_gpu_laguna_resident_host_calloc(&tracker, 3, 3, 5, &o);
    ds4_runtime_allocation_record *r = live_record(o.allocation_record_id);
    std::printf("began=%d\nallocated=%d\nowner_nonzero=%d\nnamespace48=%d\nrecord_requested=%llu\nrecord_charged=%llu\nrecord_site=%u\nrecord_category=%d\nrecord_domain=%d\nrecord_relation=%d\n", began, allocated, owner_nonzero(o), (int)((o.allocation_record_id >> 56u) == 0x48u), r ? (unsigned long long)r->requested_bytes : 0ull, r ? (unsigned long long)r->charged_bytes : 0ull, r ? r->callsite_id : 0u, r ? (int)r->category : -1, r ? (int)r->domain : -1, r ? (int)r->relation : -1);
    int freed = ds4_gpu_laguna_resident_host_free(&tracker, &o); std::printf("freed=%d\nowner_zero=%d\n", freed, owner_zero(o)); emit_common("after"); std::printf("ended=%d\n", ds4_gpu_laguna_resident_observer_end(&tracker)); return 0;
}
static int scenario_all_sites(void) {
    reset_case(); const uint32_t ids[] = {3,15,16,17,18,19,20,21}; ds4_gpu_laguna_resident_host_owner os[8] = {};
    int began = ds4_gpu_laguna_resident_observer_begin(&tracker), accepted = 0, ns = 1, all_zero = 1, exact = 1;
    for (size_t i = 0; i < 8; ++i) {
        int ok = ds4_gpu_laguna_resident_host_calloc(&tracker, ids[i], 1, 1, &os[i]);
        accepted += ok == 1; ns &= ok == 1 && (os[i].allocation_record_id >> 56u) == 0x48u;
        const auto *r = live_record(os[i].allocation_record_id);
        exact &= r && r->callsite_id == ids[i] && r->domain == DS4_RUNTIME_DOMAIN_HOST &&
            r->category == (ids[i] == 3 ? DS4_RUNTIME_CATEGORY_CACHE_METADATA_ADDRESS_TABLES : DS4_RUNTIME_CATEGORY_OTHER_HOST) &&
            r->relation == DS4_RUNTIME_RELATION_OWNED_ALLOCATION && r->owner_id == 0 &&
            r->base == (uint64_t)(uintptr_t)os[i].base && r->requested_bytes == 1 && r->charged_bytes == 1;
    }
    std::printf("began=%d\naccepted=%d\nnamespace48=%d\nper_record_exact=%d\nactive_after_alloc=%d\nrecord_count_after_alloc=%llu\ncache_current_after_alloc=%llu\nother_current_after_alloc=%llu\n", began, accepted, ns, exact, active_records(), (unsigned long long)tracker.record_count, (unsigned long long)tracker.category_current[2], (unsigned long long)tracker.category_current[6]);
    int freed = 1; for (size_t i = 0; i < 8; ++i) { freed &= ds4_gpu_laguna_resident_host_free(&tracker, &os[i]); all_zero &= owner_zero(os[i]); }
    std::printf("freed=%d\nall_owners_zero=%d\n", freed, all_zero); emit_common("after"); std::printf("ended=%d\n", ds4_gpu_laguna_resident_observer_end(&tracker)); return 0;
}
static int scenario_c_abi(void) { reset_case(); int began = ds4_gpu_laguna_resident_observer_begin(&tracker), probe = resident_host_header_probe_c(&tracker); std::printf("began=%d\nprobe=%d\n", began, probe); emit_common("after"); std::printf("ended=%d\n", ds4_gpu_laguna_resident_observer_end(&tracker)); return 0; }
static int scenario_unattached(void) {
    reset_case(); if (!init_tracker(&foreign_tracker, &foreign_config, foreign_records, 32, 64)) std::abort();
    ds4_runtime_tracker tb, fb; ds4_runtime_allocation_record rb[64], frb[64]; std::memcpy(&tb, &tracker, sizeof(tb)); std::memcpy(&fb, &foreign_tracker, sizeof(fb)); std::memcpy(rb, records, sizeof(records)); std::memcpy(frb, foreign_records, sizeof(foreign_records));
    ds4_gpu_laguna_resident_host_owner empty = {}; int ua = ds4_gpu_laguna_resident_host_calloc(&tracker, 15, 1, 1, &empty), ef = ds4_gpu_laguna_resident_host_free(&tracker, &empty);
    int unchanged_unattached = tracker_storage_equal(tracker, tb, records, rb) && tracker_storage_equal(foreign_tracker, fb, foreign_records, frb), calls_before_begin = fake_calloc_calls;
    int began = ds4_gpu_laguna_resident_observer_begin(&tracker); std::memcpy(&tb, &tracker, sizeof(tb)); std::memcpy(&fb, &foreign_tracker, sizeof(fb)); std::memcpy(rb, records, sizeof(records)); std::memcpy(frb, foreign_records, sizeof(foreign_records));
    ds4_gpu_laguna_resident_host_owner fo = {reinterpret_cast<void *>(static_cast<uintptr_t>(0x1111)), UINT64_C(0x4800000000000011)}, no = {reinterpret_cast<void *>(static_cast<uintptr_t>(0x2222)), UINT64_C(0x4800000000000022)}; auto fs = fo, ns = no;
    int fa = ds4_gpu_laguna_resident_host_calloc(&foreign_tracker, 15, 1, 1, &fo), na = ds4_gpu_laguna_resident_host_calloc(nullptr, 15, 1, 1, &no);
    int unchanged_alloc = tracker_storage_equal(tracker, tb, records, rb) && tracker_storage_equal(foreign_tracker, fb, foreign_records, frb), calls_before_valid = fake_calloc_calls;
    ds4_gpu_laguna_resident_host_owner o = {}; int allocated = ds4_gpu_laguna_resident_host_calloc(&tracker, 15, 2, 2, &o); std::memcpy(&tb, &tracker, sizeof(tb)); std::memcpy(&fb, &foreign_tracker, sizeof(fb)); std::memcpy(rb, records, sizeof(records)); std::memcpy(frb, foreign_records, sizeof(foreign_records)); auto saved = o;
    int ff = ds4_gpu_laguna_resident_host_free(&foreign_tracker, &o), of = owner_equal(o, saved), nf = ds4_gpu_laguna_resident_host_free(nullptr, &o), on = owner_equal(o, saved);
    int unchanged_free = tracker_storage_equal(tracker, tb, records, rb) && tracker_storage_equal(foreign_tracker, fb, foreign_records, frb), vf = ds4_gpu_laguna_resident_host_free(&tracker, &o);
    std::printf("unattached_alloc=%d\nempty_free=%d\nunattached_owner_zero=%d\nunattached_unchanged=%d\ncalls_before_begin=%d\nbegan=%d\nforeign_alloc=%d\nnull_alloc=%d\nforeign_alloc_owner_unchanged=%d\nnull_alloc_owner_unchanged=%d\nalloc_controls_unchanged=%d\ncalls_before_valid=%d\nallocated=%d\nforeign_free=%d\nnull_free=%d\nowner_after_foreign=%d\nowner_after_null=%d\nfree_controls_unchanged=%d\nvalid_free=%d\n", ua, ef, owner_zero(empty), unchanged_unattached, calls_before_begin, began, fa, na, owner_equal(fo, fs), owner_equal(no, ns), unchanged_alloc, calls_before_valid, allocated, ff, nf, of, on, unchanged_free, vf); emit_common("after"); return 0;
}
static int scenario_nonempty(void) { reset_case(); int began = ds4_gpu_laguna_resident_observer_begin(&tracker); ds4_gpu_laguna_resident_host_owner o = {reinterpret_cast<void *>(static_cast<uintptr_t>(0x1234)), UINT64_C(0x4800000000000077)}, saved = o; int r = ds4_gpu_laguna_resident_host_calloc(&tracker, 15, 1, 1, &o); std::printf("began=%d\nresult=%d\nowner_unchanged=%d\n", began, r, owner_equal(o, saved)); emit_common("after"); return 0; }
static int scenario_bad_site(const char *m) {
    reset_case(); std::string n(m); int began = ds4_gpu_laguna_resident_observer_begin(&tracker), changed = 0; uint32_t id = 15;
    if (n == "bad-gpu") id = 1; else if (n == "bad-managed") id = DS4_LAGUNA_RESIDENT_CALLSITE_OTHER_MANAGED; else if (n == "bad-pinned") id = 11; else if (n == "unknown") id = 99;
    else if (n == "wrong-class") { site(3)->category = DS4_RUNTIME_CATEGORY_OTHER_HOST; id = 3; changed = 1; }
    else if (n == "wrong-domain") { site(3)->domain = DS4_RUNTIME_DOMAIN_CUDA_DEVICE; id = 3; changed = 1; }
    ds4_gpu_laguna_resident_host_owner o = {}; int r = ds4_gpu_laguna_resident_host_calloc(&tracker, id, 1, 1, &o);
    std::printf("began=%d\nresult=%d\nowner_zero=%d\nchanged_config_refusal=%d\n", began, r, owner_zero(o), changed); emit_common("after"); std::printf("ended=%d\n", ds4_gpu_laguna_resident_observer_end(&tracker)); return 0;
}
static int scenario_preflight(const char *m) {
    reset_case(); int began = ds4_gpu_laguna_resident_observer_begin(&tracker); ds4_gpu_laguna_resident_host_owner o = {}; uint64_t n = 1, s = 1; int branch = 0; std::string name(m);
    if (name == "zero-count") n = 0; else if (name == "zero-item") s = 0; else if (name == "multiply") n = UINT64_MAX, s = 2;
    else if (name == "size-conversion") { if (sizeof(size_t) < sizeof(uint64_t)) n = 1, s = (uint64_t)SIZE_MAX + 1u, branch = 1; else n = UINT64_MAX, s = 2, branch = 2; }
    int r = ds4_gpu_laguna_resident_host_calloc(&tracker, 15, n, s, &o); std::printf("began=%d\nresult=%d\nowner_zero=%d\nsize_preflight_branch=%d\nsize_t_bytes=%llu\n", began, r, owner_zero(o), branch, (unsigned long long)sizeof(size_t)); emit_common("after"); std::printf("ended=%d\n", ds4_gpu_laguna_resident_observer_end(&tracker)); return 0;
}
static int scenario_calloc_failure(void) { reset_case(); int began = ds4_gpu_laguna_resident_observer_begin(&tracker); fake_fail_calloc = true; ds4_gpu_laguna_resident_host_owner o = {}; int r = ds4_gpu_laguna_resident_host_calloc(&tracker, 15, 2, 3, &o); std::printf("began=%d\nresult=%d\nowner_zero=%d\n", began, r, owner_zero(o)); emit_common("after"); std::printf("ended=%d\n", ds4_gpu_laguna_resident_observer_end(&tracker)); return 0; }
static int scenario_capacity(void) { reset_case(1); int began = ds4_gpu_laguna_resident_observer_begin(&tracker); ds4_gpu_laguna_resident_host_owner a = {}, b = {}; int ar = ds4_gpu_laguna_resident_host_calloc(&tracker, 15, 1, 1, &a), br = ds4_gpu_laguna_resident_host_calloc(&tracker, 15, 1, 1, &b); std::printf("began=%d\nfirst_result=%d\nsecond_result=%d\nsecond_zero=%d\nreal_live_after_second=%d\n", began, ar, br, owner_zero(b), (int)fake_real_live.size()); int f = ds4_gpu_laguna_resident_host_free(&tracker, &a); std::printf("freed=%d\n", f); emit_common("after"); std::printf("ended=%d\n", ds4_gpu_laguna_resident_observer_end(&tracker)); return 0; }
static int scenario_record_count_preflight(void) {
    reset_case(); int began = ds4_gpu_laguna_resident_observer_begin(&tracker); ds4_gpu_laguna_resident_host_owner o = {}; tracker.record_count = 33;
    int r = ds4_gpu_laguna_resident_host_calloc(&tracker, 15, 1, 1, &o), calls = fake_calloc_calls; tracker.record_count = 0;
    std::printf("began=%d\nmalformed_count=33\nresult=%d\nowner_zero=%d\nsensor_calls_before_restore=%d\nrecord_count_restored=%d\n", began, r, owner_zero(o), calls, tracker.record_count == 0); emit_common("after"); return 0;
}
static int scenario_id_exhaustion(void) { reset_case(); int began = ds4_gpu_laguna_resident_observer_begin(&tracker); tracker.issued_sequence_high_water[0x48] = UINT64_C(0x00ffffffffffffff); ds4_gpu_laguna_resident_host_owner o = {}; int r = ds4_gpu_laguna_resident_host_calloc(&tracker, 15, 1, 1, &o); std::printf("began=%d\nresult=%d\nowner_zero=%d\n", began, r, owner_zero(o)); emit_common("after"); return 0; }
static int scenario_tombstone(void) { reset_case(1); int began = ds4_gpu_laguna_resident_observer_begin(&tracker); ds4_gpu_laguna_resident_host_owner a = {}, b = {}; int ar = ds4_gpu_laguna_resident_host_calloc(&tracker, 15, 1, 1, &a); uint64_t ai = a.allocation_record_id; int af = ds4_gpu_laguna_resident_host_free(&tracker, &a); int br = ds4_gpu_laguna_resident_host_calloc(&tracker, 15, 1, 1, &b); uint64_t bi = b.allocation_record_id; int bf = ds4_gpu_laguna_resident_host_free(&tracker, &b); std::printf("began=%d\nfirst_result=%d\nfirst_free=%d\nsecond_result=%d\nsecond_free=%d\nid_distinct=%d\nid_monotonic=%d\nnamespace48=%d\n", began, ar, af, br, bf, ai != bi, bi > ai, (int)((ai >> 56u) == 0x48u && (bi >> 56u) == 0x48u)); emit_common("after"); return 0; }
static int scenario_bound(void) { reset_case(32, 1); int began = ds4_gpu_laguna_resident_observer_begin(&tracker); ds4_gpu_laguna_resident_host_owner o = {}; int r = ds4_gpu_laguna_resident_host_calloc(&tracker, 15, 2, 1, &o); std::printf("began=%d\nresult=%d\nowner_zero=%d\n", began, r, owner_zero(o)); emit_common("after"); std::printf("ended=%d\n", ds4_gpu_laguna_resident_observer_end(&tracker)); return 0; }
static int scenario_relation(void) {
    reset_case(); int began = ds4_gpu_laguna_resident_observer_begin(&tracker); ds4_gpu_laguna_resident_host_owner o = {}; int a = ds4_gpu_laguna_resident_host_calloc(&tracker, 15, 2, 2, &o); uint64_t rid = UINT64_C(0x4900000000000001);
    int reg = (int)ds4_runtime_tracker_register(&tracker, rid, (uint64_t)(uintptr_t)o.base, 4, o.allocation_record_id); int blocked = ds4_gpu_laguna_resident_host_free(&tracker, &o), calls = fake_free_calls; auto first = tracker.violation; auto *relation = live_record(rid);
    int unreg = (int)ds4_runtime_tracker_unregister(&tracker, rid), retired = relation && !relation->live, sticky = tracker.violation == first, retry = ds4_gpu_laguna_resident_host_free(&tracker, &o);
    std::printf("began=%d\nallocated=%d\nregistered=%d\nblocked=%d\nfree_calls_blocked=%d\nfirst_violation=%d\nunregister_status_unsafe=%d\nrelation_retired=%d\nsticky_retained=%d\nretried=%d\n", began, a, reg == DS4_RUNTIME_STATUS_OK, blocked, calls, (int)first, unreg == DS4_RUNTIME_STATUS_UNSAFE, retired, sticky, retry); emit_common("after"); std::printf("ended=%d\n", ds4_gpu_laguna_resident_observer_end(&tracker)); return 0;
}
static int scenario_unsafe_cleanup(void) { reset_case(); int began = ds4_gpu_laguna_resident_observer_begin(&tracker); ds4_gpu_laguna_resident_host_owner o = {}, bad = {}; int a = ds4_gpu_laguna_resident_host_calloc(&tracker, 15, 2, 2, &o); uint64_t peak = tracker.owned_total_peak; int vr = ds4_gpu_laguna_resident_host_calloc(&tracker, 15, 0, 4, &bad); auto violation = tracker.violation; int f = ds4_gpu_laguna_resident_host_free(&tracker, &o); std::printf("began=%d\nallocated=%d\nviolation_result=%d\nrejected_zero=%d\nfirst_violation=%d\nfreed=%d\npeak_unchanged=%d\n", began, a, vr, owner_zero(bad), (int)violation, f, tracker.owned_total_peak == peak); emit_common("after"); std::printf("ended=%d\n", ds4_gpu_laguna_resident_observer_end(&tracker)); return 0; }
static int scenario_bad_handle(const char *m) {
    reset_case(); int began = ds4_gpu_laguna_resident_observer_begin(&tracker); ds4_gpu_laguna_resident_host_owner o = {}; int a = ds4_gpu_laguna_resident_host_calloc(&tracker, 15, 2, 2, &o); auto saved = o; auto *r = live_record(o.allocation_record_id); auto rs = r ? *r : ds4_runtime_allocation_record{}; std::string n(m); int first = 0, retry = 0;
    if (n == "stale") { first = ds4_gpu_laguna_resident_host_free(&tracker, &o); o = saved; } else { if (n == "bad-id") o.allocation_record_id ^= 1; if (n == "bad-namespace") o.allocation_record_id = (saved.allocation_record_id & UINT64_C(0x00ffffffffffffff)) | UINT64_C(0x5200000000000000); if (n == "bad-base") o.base = reinterpret_cast<void *>((uintptr_t)saved.base + 1); if (n == "bad-site" && r) r->callsite_id = 1; if (n == "bad-domain" && r) r->domain = DS4_RUNTIME_DOMAIN_CUDA_DEVICE; if (n == "bad-category" && r) r->category = DS4_RUNTIME_CATEGORY_OTHER_CUDA; if (n == "bad-record" && r) r->relation = DS4_RUNTIME_RELATION_REGISTRATION; if (n == "bad-size" && r) r->charged_bytes = r->requested_bytes - 1; }
    int before = fake_free_calls; int bad = ds4_gpu_laguna_resident_host_free(&tracker, &o), after = fake_free_calls; int live = r && r->live; if (n != "stale") { if (r) *r = rs; o = saved; retry = ds4_gpu_laguna_resident_host_free(&tracker, &o); }
    std::printf("began=%d\nallocated=%d\nfirst_free=%d\nbad=%d\ncalls_before_bad=%d\ncalls_after_bad=%d\nrecord_live_after_bad=%d\nretry=%d\nowner_zero=%d\n", began, a, first, bad, before, after, live, retry, owner_zero(o)); emit_common("after"); std::printf("ended=%d\n", ds4_gpu_laguna_resident_observer_end(&tracker)); return 0;
}
static int scenario_other_producer(void) {
    reset_case(); int began = ds4_gpu_laguna_resident_observer_begin(&tracker); void *p = fake_forward_calloc(1, 4); uint64_t id = 0;
    int allocated = p && ds4_runtime_tracker_allocate_next(&tracker, 0x52u, 19, (uint64_t)(uintptr_t)p, 4, 4, &id) == DS4_RUNTIME_STATUS_OK; auto *r = live_record(id); ds4_gpu_laguna_resident_host_owner o = {p, id}; int before = fake_free_calls;
    int bad = ds4_gpu_laguna_resident_host_free(&tracker, &o), retained = owner_nonzero(o) && r && r->live, no_side = fake_free_calls == before && fake_real_live.size() == 1;
    fake_expected_owner_namespace = 0x52u; fake_forward_free(p); int sensor_free = fake_real_live.empty() && fake_last_free_live_record && fake_free_order_violations == 0; fake_expected_owner_namespace = 0x48u;
    (void)ds4_runtime_tracker_release(&tracker, id); int retired = r && !r->live;
    std::printf("began=%d\nother_producer_alloc=%d\nother_producer_record=%d\nother_namespace52=%d\nbad=%d\nowner_retained=%d\nrecord_retained=%d\nno_physical_side_effect=%d\ncleanup_sensor_free=%d\ncleanup_record_retired=%d\n", began, p != nullptr, allocated && r, (int)((id >> 56u) == 0x52u), bad, retained, retained, no_side, sensor_free, retired); emit_common("after"); std::printf("ended=%d\n", ds4_gpu_laguna_resident_observer_end(&tracker)); return 0;
}
static int scenario_end_live(void) { reset_case(); int began = ds4_gpu_laguna_resident_observer_begin(&tracker); ds4_gpu_laguna_resident_host_owner o = {}; int a = ds4_gpu_laguna_resident_host_calloc(&tracker, 15, 1, 4, &o), wl = ds4_gpu_laguna_resident_observer_end(&tracker), f = ds4_gpu_laguna_resident_host_free(&tracker, &o), af = ds4_gpu_laguna_resident_observer_end(&tracker); std::printf("began=%d\nallocated=%d\nwhile_live=%d\nfreed=%d\nafter_free=%d\n", began, a, wl, f, af); emit_common("after"); return 0; }

'''

MAIN = r'''

static int finish_scenario(int result) {
    if (result != 0) return result;
    int physical_owners_clear = fake_real_live.empty() && fake_cuda_live.empty(), records_clear = active_records() == 0;
    int attached_before = ds4_gpu_laguna_resident_observer_attached(&tracker), ended = 1;
    if (attached_before) ended = ds4_gpu_laguna_resident_observer_end(&tracker);
    int detached = !ds4_gpu_laguna_resident_observer_attached(&tracker);
    std::printf("cleanup_physical_owners_clear=%d\ncleanup_records_clear=%d\ncleanup_attached_before=%d\ncleanup_end_if_attached=%d\ncleanup_detached=%d\n", physical_owners_clear, records_clear, attached_before, ended, detached);
    return physical_owners_clear && records_clear && ended && detached ? 0 : 3;
}

int main(int argc, char **argv) {
    alarm(15); struct rlimit limit = {0, 0}; (void)setrlimit(RLIMIT_CORE, &limit); if (argc != 2) return 2;
    std::string n(argv[1]); int result = 2;
    if (n == "success") result = scenario_success(); else if (n == "all-sites") result = scenario_all_sites(); else if (n == "c-abi") result = scenario_c_abi();
    else if (n == "unattached") result = scenario_unattached(); else if (n == "nonempty") result = scenario_nonempty();
    else if (n == "bad-gpu" || n == "bad-managed" || n == "bad-pinned" || n == "unknown" || n == "wrong-class" || n == "wrong-domain") result = scenario_bad_site(n.c_str());
    else if (n == "zero-count" || n == "zero-item" || n == "multiply" || n == "size-conversion") result = scenario_preflight(n.c_str());
    else if (n == "calloc-failure") result = scenario_calloc_failure(); else if (n == "capacity") result = scenario_capacity(); else if (n == "record-count-preflight") result = scenario_record_count_preflight(); else if (n == "id-exhaustion") result = scenario_id_exhaustion(); else if (n == "tombstone") result = scenario_tombstone();
    else if (n == "bound") result = scenario_bound(); else if (n == "relation") result = scenario_relation(); else if (n == "unsafe-cleanup") result = scenario_unsafe_cleanup(); else if (n == "other-producer") result = scenario_other_producer(); else if (n == "end-live") result = scenario_end_live();
    else if (n == "bad-id" || n == "bad-namespace" || n == "bad-base" || n == "bad-site" || n == "bad-domain" || n == "bad-category" || n == "bad-record" || n == "bad-size" || n == "stale") result = scenario_bad_handle(n.c_str());
    return finish_scenario(result);
}

'''
FIXTURE_SOURCE = FAKE_PREFIX + "\n" + (OBSERVER_BLOCK or "") + "\n" + (HOST_BLOCK or "") + "\n" + SCENARIOS + "\n" + MAIN


def parse_output(stdout: str) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in stdout.splitlines():
        key, sep, raw = line.partition("=")
        if sep and re.fullmatch(r"-?\d+", raw): values[key] = int(raw)
    return values


def value(values: dict[str, int], key: str) -> int:
    if key not in values: raise AssertionError(f"missing {key}: {values}")
    return values[key]


def logical_make_lines(source: str) -> list[str]:
    lines, pending = [], ""
    for raw in source.splitlines():
        line = raw.rstrip(); pending = (pending + " " + line.lstrip()) if pending else line
        if pending.endswith("\\"): pending = pending[:-1].rstrip()
        else: lines.append(pending); pending = ""
    if pending: lines.append(pending)
    return lines


class ResidentHostFixture(unittest.TestCase):
    '''Compile and execute only generated fake-driver children.'''
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory(prefix="ds4-native-host-owner-contract-"); cls.addClassCleanup(cls._tmp.cleanup)
        base = Path(cls._tmp.name); cls._env = dict(SAFE_ENV)
        for key, leaf in (("HOME", "home"), ("TMPDIR", "tmp"), ("XDG_CONFIG_HOME", "config"), ("XDG_CACHE_HOME", "cache"), ("XDG_DATA_HOME", "data")):
            child = base / leaf; child.mkdir(); cls._env[key] = str(child)
        cls._source = base / "host_fixture.cc"; cls._source.write_text(FIXTURE_SOURCE, encoding="utf-8")
        cls._runtime_object = base / "ds4_runtime.o"; cls._abi_source = base / "host_abi_probe.c"; cls._abi_object = base / "host_abi_probe.o"; cls._binary = base / "host_fixture"
        cls._runtime_compile = cls._abi_compile = cls._compile = None
        if MISSING_SOURCE_SEAMS: return
        cls._runtime_compile = subprocess.run(["cc", "-std=c11", "-O0", "-Wall", "-Wextra", "-Werror=format", "-I", str(ROOT), "-c", str(RUNTIME_SOURCE), "-o", str(cls._runtime_object)], cwd=base, env=cls._env, capture_output=True, text=True, timeout=15, check=False)
        cls._abi_source.write_text(ABI_PROBE, encoding="utf-8")
        cls._abi_compile = subprocess.run(["cc", "-std=c11", "-O0", "-Wall", "-Wextra", "-Werror=format", "-I", str(ROOT), "-c", str(cls._abi_source), "-o", str(cls._abi_object)], cwd=base, env=cls._env, capture_output=True, text=True, timeout=15, check=False)
        if cls._runtime_compile.returncode == 0 and cls._abi_compile.returncode == 0:
            cls._compile = subprocess.run(["c++", "-std=c++17", "-O0", "-Wall", "-Wextra", "-Werror=format", "-pthread", "-I", str(ROOT), str(cls._source), str(cls._runtime_object), str(cls._abi_object), "-o", str(cls._binary)], cwd=base, env=cls._env, capture_output=True, text=True, timeout=15, check=False)

    def assert_compile_prerequisite(self) -> None:
        self.assertFalse(MISSING_SOURCE_SEAMS, "resident host source/header RED; missing seams: " + ", ".join(MISSING_SOURCE_SEAMS))
        failures = []
        for label, result in (("runtime C", self._runtime_compile), ("C ABI probe", self._abi_compile), ("host fixture", self._compile)):
            if result is None: failures.append(label + ": did not run")
            elif result.returncode != 0: failures.append(label + ":\n" + result.stderr)
        self.assertFalse(failures, "resident host compile/link RED:\n" + "\n".join(failures))

    def run_case(self, name: str) -> dict[str, int]:
        self.assert_compile_prerequisite()
        result = subprocess.run([str(self._binary), name], cwd=self._binary.parent, env=self._env, capture_output=True, text=True, timeout=16, check=False)
        self.assertEqual(result.returncode, 0, f"resident host scenario {name} RED:\nstdout={result.stdout}\nstderr={result.stderr}")
        values = parse_output(result.stdout)
        for key, count in values.items():
            if key.endswith("_api_errors"): self.assertEqual(count, 0, f"fake driver protocol error: {key}")
        self.assert_fields(values, {"after_free_order_violations": 0, "after_fixture_protocol_errors": 0, "after_expected_owner_namespace": 0x48, "cleanup_physical_owners_clear": 1, "cleanup_records_clear": 1, "cleanup_end_if_attached": 1, "cleanup_detached": 1})
        return values

    def assert_no_cuda(self, values: dict[str, int]) -> None: self.assertEqual(value(values, "after_cuda_calls"), 0)
    def assert_fields(self, values: dict[str, int], expected: dict[str, int]) -> None:
        for key, wanted in expected.items(): self.assertEqual(value(values, key), wanted, key)

    def test_00_source_and_header_seams(self) -> None:
        self.assertFalse(MISSING_SOURCE_SEAMS, "resident host feature RED; missing actual seams: " + ", ".join(MISSING_SOURCE_SEAMS))
        for marker in (OBSERVER_START_MARKER, OBSERVER_END_MARKER, HOST_START_MARKER, HOST_END_MARKER): self.assertEqual(CUDA_SOURCE.count(marker), 1)

    def test_01_c_header_caller_compiles_and_links(self) -> None:
        v = self.run_case("c-abi"); self.assert_fields(v, {"began": 1, "probe": 1, "after_calloc_calls": 1, "after_free_calls": 1, "after_real_current": 0, "ended": 1}); self.assert_no_cuda(v)

    def test_02_make_target_and_aggregate_wiring(self) -> None:
        target = "test-cuda-resident-host-contract"
        self.assertRegex(MAKEFILE_SOURCE, r"(?m)^" + re.escape(target) + r":\s*\n\tpython3 tests/test_cuda_resident_host_contract\.py -v\s*$")
        phony = {item for line in MAKEFILE_SOURCE.splitlines() if line.startswith(".PHONY:") for item in line.split(":", 1)[1].split()}; self.assertIn(target, phony)
        logical = logical_make_lines(MAKEFILE_SOURCE); resident = next(line for line in logical if line.startswith("test-laguna-resident-path:")); default = next(line for line in logical if line.startswith("test:"))
        self.assertIn(target, resident.split(":", 1)[1].split()); self.assertIn("test-laguna-resident-path", default.split(":", 1)[1].split())

    def test_03_success_real_bytes_records_and_zeroing(self) -> None:
        v = self.run_case("success"); self.assert_fields(v, {"began": 1, "allocated": 1, "owner_nonzero": 1, "namespace48": 1, "record_requested": 15, "record_charged": 15, "record_site": 3, "record_category": 2, "record_domain": 0, "record_relation": 0, "freed": 1, "owner_zero": 1, "after_host_current": 0, "after_cache_peak": 15, "after_owned_peak": 15, "after_real_peak": 15, "after_last_calloc_count": 3, "after_last_calloc_size": 5, "after_last_calloc_bytes": 15, "after_last_calloc_zeroed": 1, "after_last_free_live_record": 1, "after_last_free_owned_current": 15, "after_real_live": 0, "ended": 1}); self.assert_no_cuda(v)

    def test_04_all_eight_host_sites_exactly_classified(self) -> None:
        v = self.run_case("all-sites"); self.assert_fields(v, {"began": 1, "accepted": 8, "namespace48": 1, "active_after_alloc": 8, "record_count_after_alloc": 8, "per_record_exact": 1, "cache_current_after_alloc": 1, "other_current_after_alloc": 7, "freed": 1, "all_owners_zero": 1, "after_real_live": 0, "ended": 1}); self.assert_no_cuda(v)

    def test_05_identity_and_nonempty_output_no_mutation(self) -> None:
        v = self.run_case("unattached"); self.assert_fields(v, {"unattached_alloc": 0, "empty_free": 1, "unattached_owner_zero": 1, "unattached_unchanged": 1, "calls_before_begin": 0, "began": 1, "foreign_alloc": 0, "null_alloc": 0, "foreign_alloc_owner_unchanged": 1, "null_alloc_owner_unchanged": 1, "alloc_controls_unchanged": 1, "calls_before_valid": 0, "allocated": 1, "foreign_free": 0, "null_free": 0, "owner_after_foreign": 1, "owner_after_null": 1, "free_controls_unchanged": 1, "valid_free": 1, "after_calloc_calls": 1, "after_free_calls": 1}); self.assert_no_cuda(v)
        v = self.run_case("nonempty"); self.assert_fields(v, {"began": 1, "result": 0, "owner_unchanged": 1, "after_calloc_calls": 0, "after_free_calls": 0}); self.assert_no_cuda(v)

    def test_06_rejected_site_classes_do_not_reach_calloc(self) -> None:
        for name in ("bad-gpu", "bad-managed", "bad-pinned", "unknown", "wrong-class", "wrong-domain"):
            with self.subTest(name=name):
                v = self.run_case(name); self.assert_fields(v, {"result": 0, "owner_zero": 1, "changed_config_refusal": int(name in ("wrong-class", "wrong-domain")), "after_calloc_calls": 0, "after_real_live": 0, "ended": 1}); self.assertNotEqual(value(v, "after_violation"), 0); self.assert_no_cuda(v)

    def test_07_size_preflights_are_physical_noops(self) -> None:
        for name in ("zero-count", "zero-item", "multiply", "size-conversion"):
            with self.subTest(name=name):
                v = self.run_case(name); self.assert_fields(v, {"result": 0, "owner_zero": 1, "after_calloc_calls": 0, "after_record_count": 0, "after_real_live": 0, "ended": 1});
                if name == "size-conversion": self.assertEqual(value(v, "size_preflight_branch"), 1 if value(v, "size_t_bytes") < 8 else 2)
                self.assertNotEqual(value(v, "after_violation"), 0); self.assert_no_cuda(v)

    def test_08_calloc_failure_has_no_record_peak_or_cuda_effect(self) -> None:
        v = self.run_case("calloc-failure"); self.assert_fields(v, {"result": 0, "owner_zero": 1, "after_calloc_calls": 1, "after_free_calls": 0, "after_record_count": 0, "after_host_peak": 0, "after_owned_peak": 0, "after_real_live": 0, "ended": 1}); self.assertNotEqual(value(v, "after_violation"), 0); self.assert_no_cuda(v)

    def test_09_capacity_id_tombstone_and_count_preflights(self) -> None:
        v = self.run_case("capacity"); self.assert_fields(v, {"began": 1, "first_result": 1, "second_result": 0, "second_zero": 1, "real_live_after_second": 1, "after_calloc_calls": 1, "freed": 1, "after_host_peak": 1, "ended": 1}); self.assert_no_cuda(v)
        v = self.run_case("record-count-preflight"); self.assert_fields(v, {"began": 1, "malformed_count": 33, "result": 0, "owner_zero": 1, "sensor_calls_before_restore": 0, "record_count_restored": 1, "after_record_count": 0}); self.assertNotEqual(value(v, "after_violation"), 0); self.assert_no_cuda(v)
        v = self.run_case("id-exhaustion"); self.assert_fields(v, {"result": 0, "owner_zero": 1, "after_calloc_calls": 0}); self.assertNotEqual(value(v, "after_violation"), 0); self.assert_no_cuda(v)
        v = self.run_case("tombstone"); self.assert_fields(v, {"first_result": 1, "first_free": 1, "second_result": 1, "second_free": 1, "id_distinct": 1, "id_monotonic": 1, "namespace48": 1, "after_calloc_calls": 2, "after_real_live": 0}); self.assert_no_cuda(v)

    def test_10_overbound_transient_peak_then_checked_rollback(self) -> None:
        v = self.run_case("bound"); self.assert_fields(v, {"result": 0, "owner_zero": 1, "after_calloc_calls": 1, "after_free_calls": 1, "after_last_free_live_record": 1, "after_last_free_owned_current": 2, "after_host_current": 0, "after_host_peak": 2, "after_owned_peak": 2, "after_real_current": 0, "after_real_peak": 2, "after_active_records": 0, "ended": 1}); self.assertNotEqual(value(v, "after_violation"), 0); self.assert_no_cuda(v)

    def test_11_relation_blocks_then_unregister_allows_sticky_cleanup(self) -> None:
        v = self.run_case("relation"); self.assert_fields(v, {"began": 1, "allocated": 1, "registered": 1, "blocked": 0, "free_calls_blocked": 0, "unregister_status_unsafe": 1, "relation_retired": 1, "sticky_retained": 1, "retried": 1, "after_last_free_live_record": 1, "after_last_free_owned_current": 4, "after_real_live": 0, "ended": 1}); self.assertNotEqual(value(v, "first_violation"), 0); self.assert_no_cuda(v)

    def test_12_wrong_handles_and_other_producer_namespace_refuse_before_free(self) -> None:
        for name in ("bad-id", "bad-namespace", "bad-base", "bad-site", "bad-domain", "bad-category", "bad-record", "bad-size", "stale"):
            with self.subTest(name=name):
                v = self.run_case(name); self.assertEqual(value(v, "allocated"), 1); self.assertEqual(value(v, "bad"), 0); expected = {"after_unknown_free_calls": 0, "after_real_live": 0, "ended": 1}
                if name == "stale": expected.update(first_free=1, calls_before_bad=1, calls_after_bad=1)
                else: expected.update(first_free=0, calls_before_bad=0, calls_after_bad=0, retry=1)
                self.assert_fields(v, expected); self.assertNotEqual(value(v, "after_violation"), 0); self.assert_no_cuda(v)
        v = self.run_case("other-producer"); self.assert_fields(v, {"began": 1, "other_producer_alloc": 1, "other_producer_record": 1, "other_namespace52": 1, "bad": 0, "owner_retained": 1, "record_retained": 1, "no_physical_side_effect": 1, "cleanup_sensor_free": 1, "cleanup_record_retired": 1, "after_free_calls": 1, "after_last_free_live_record": 1, "after_real_live": 0, "ended": 1}); self.assert_no_cuda(v)

    def test_13_live_end_and_unsafe_cleanup_keep_sticky_state(self) -> None:
        v = self.run_case("end-live"); self.assert_fields(v, {"began": 1, "allocated": 1, "while_live": 0, "freed": 1, "after_free": 1, "after_last_free_live_record": 1}); self.assert_no_cuda(v)
        v = self.run_case("unsafe-cleanup"); self.assert_fields(v, {"allocated": 1, "violation_result": 0, "rejected_zero": 1, "peak_unchanged": 1, "freed": 1, "after_real_live": 0, "ended": 1}); self.assertNotEqual(value(v, "first_violation"), 0); self.assert_no_cuda(v)


if __name__ == "__main__": unittest.main()
