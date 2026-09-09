\
#!/usr/bin/env python3
"""Host-only contract for CUDA staging-pool ownership and teardown.

The fixture compiles bodies extracted from ``ds4_cuda.cu``.  It never copies a
pool algorithm.  ``CurrentSourceBehaviorContractTest`` always compiles the
currently available pool bodies, so old release failures are executed even
before the new ownership API exists.  ``NewOwnershipApiContractTest`` always
compiles a driver that calls the required new API; missing definitions are a
separate source and compiler RED, never a skipped behavioral test.

The fake CUDA layer owns only small host buffers.  It records requested bytes,
all live hosts/events/streams, related host/event order, and operation traces.
This is host behavior only; it makes no CUDA, Linux, or asynchronous-quiescence
claim.
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import TypeAlias

ROOT = Path(__file__).resolve().parents[1]
CUDA_SOURCE = (ROOT / "ds4_cuda.cu").read_text(encoding="utf-8")
SAFE_ENV = {
    "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
    "LANG": "C",
    "LC_ALL": "C",
}


# Keep extraction lexical.  Source contracts must not be satisfied by prose.
def extract_definition(source: str, signature: str) -> str | None:
    """Extract one C/C++ definition while ignoring lexical braces."""
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


def code_only(text: str) -> str:
    """Blank comments and literals without changing source positions."""
    result: list[str] = []
    i = 0
    state = "code"
    while i < len(text):
        c = text[i]
        n = text[i + 1] if i + 1 < len(text) else ""
        if state == "code":
            if c == "/" and n == "/":
                result.extend("  ")
                state, i = "line", i + 2
                continue
            if c == "/" and n == "*":
                result.extend("  ")
                state, i = "block", i + 2
                continue
            if c == '"':
                result.append(" ")
                state, i = "string", i + 1
                continue
            if c == "'":
                result.append(" ")
                state, i = "char", i + 1
                continue
            result.append(c)
            i += 1
            continue
        if state == "line":
            result.append("\n" if c in "\r\n" else " ")
            if c in "\r\n":
                state = "code"
            i += 1
            continue
        if state == "block":
            result.append("\n" if c in "\r\n" else " ")
            if c == "*" and n == "/":
                result.append(" ")
                state, i = "code", i + 2
            else:
                i += 1
            continue
        if c == "\\":
            result.extend("  ")
            i += 2
            continue
        result.append(" ")
        if (state == "string" and c == '"') or (state == "char" and c == "'"):
            state = "code"
        i += 1
    return "".join(result)


ALIGN = extract_definition(CUDA_SOURCE, "static void *cuda_align_ptr(")
MODEL_POOL = extract_definition(CUDA_SOURCE, "static int cuda_model_stage_pool_alloc(")
SELECTED_POOL = extract_definition(
    CUDA_SOURCE, "static int cuda_stream_selected_stage_pool_alloc("
)
SLOTS_RELEASE = extract_definition(CUDA_SOURCE, "static int cuda_stage_slots_release(")
MODEL_RELEASE = extract_definition(CUDA_SOURCE, "static int cuda_model_stage_release(")
SELECTED_RELEASE = extract_definition(
    CUDA_SOURCE, "static int cuda_stream_selected_stage_release("
)
SELECTED_RELEASE_ANY = SELECTED_RELEASE or extract_definition(
    CUDA_SOURCE, "static void cuda_stream_selected_stage_release("
)

REQUIRED_NEW_BODIES = {
    "cuda_stage_slots_release": SLOTS_RELEASE,
    "cuda_model_stage_release": MODEL_RELEASE,
    "cuda_stream_selected_stage_release": SELECTED_RELEASE,
}
MISSING_NEW_BODIES = [
    name for name, definition in REQUIRED_NEW_BODIES.items() if definition is None
]


# The fake keeps all physical allocations tiny.  Its registry is independent
# from the production-pointer arrays used by the extracted source bodies.
FAKE_PREFIX = r"""
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <signal.h>
#include <string>
#include <sys/resource.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#ifndef SIZE_MAX
#define SIZE_MAX UINT64_MAX
#endif

using cudaError_t = int;
using cudaStream_t = void *;
using cudaEvent_t = void *;
static constexpr cudaError_t cudaSuccess = 0;
static constexpr cudaError_t kFakeFailure = 17;
static constexpr unsigned cudaStreamNonBlocking = 1u;
static constexpr unsigned cudaEventDisableTiming = 2u;
static constexpr size_t kPhysicalBytes = 256;

struct host_owner {
    int id;
    uint64_t requested;
    int paired_event_id;
};
struct event_owner {
    int id;
    int paired_host_id;
};
struct stream_owner {
    int id;
};

static std::unordered_map<void *, host_owner> live_hosts;
static std::unordered_map<void *, event_owner> live_events;
static std::unordered_map<void *, stream_owner> live_streams;
static std::vector<std::string> operation_trace;
static int next_host_id = 1;
static int next_event_id = 1;
static int next_stream_id = 1;
static int pending_host_id;
static bool next_event_unpaired;
static int malloc_calls;
static int event_create_calls;
static int free_calls;
static int event_destroy_calls;
static int stream_create_calls;
static int stream_destroy_calls;
static int api_errors;
static int oversize_attempts;
static int fail_malloc_at;
static int fail_event_at;
static int fail_free_at;
static int fail_event_destroy_at;
static int fail_stream_create_at;
static int fail_stream_destroy_at;
static void *fail_free_ptr;
static void *fail_event_destroy_handle;
static void *fail_stream_destroy_handle;
static int release_failure_seq;
static int malloc_after_release_failure;
static int order_violations;

static std::string number(int value) { return std::to_string(value); }
static std::string pointer_number(const void *value) {
    return std::to_string((unsigned long long)(uintptr_t)value);
}
static void trace(const std::string &value) { operation_trace.push_back(value); }
static void note_release_failure(void) {
    if (release_failure_seq == 0) release_failure_seq = (int)operation_trace.size();
}
static bool host_live(const void *value) {
    return value && live_hosts.find(const_cast<void *>(value)) != live_hosts.end();
}
static bool event_live(const void *value) {
    return value && live_events.find(const_cast<void *>(value)) != live_events.end();
}
static bool stream_live(const void *value) {
    return value && live_streams.find(const_cast<void *>(value)) != live_streams.end();
}
static int observed_host_id(const void *value) {
    const auto found = live_hosts.find(const_cast<void *>(value));
    return found == live_hosts.end() ? 0 : found->second.id;
}
static uint64_t observed_host_requested(const void *value) {
    const auto found = live_hosts.find(const_cast<void *>(value));
    return found == live_hosts.end() ? 0 : found->second.requested;
}
static int observed_event_id(const void *value) {
    const auto found = live_events.find(const_cast<void *>(value));
    return found == live_events.end() ? 0 : found->second.id;
}
static int observed_event_paired_host(const void *value) {
    const auto found = live_events.find(const_cast<void *>(value));
    return found == live_events.end() ? 0 : found->second.paired_host_id;
}
static bool paired_event_is_live(int host_id) {
    for (const auto &entry : live_events) {
        if (entry.second.paired_host_id == host_id) return true;
    }
    return false;
}
static void fake_next_event_without_host(void) {
    pending_host_id = 0;
    next_event_unpaired = true;
}

static cudaError_t cudaMallocHost(void **out, size_t bytes) {
    ++malloc_calls;
    if (release_failure_seq != 0) malloc_after_release_failure = 1;
    if (!out) {
        ++api_errors;
        trace("M?");
        return kFakeFailure;
    }
    *out = nullptr;
    if (fail_malloc_at == malloc_calls) {
        trace("M!");
        return kFakeFailure;
    }
    if (bytes == 0 || (uint64_t)bytes > 1024u) {
        ++oversize_attempts;
        trace("MO");
        return kFakeFailure;
    }
    void *raw = nullptr;
    if (posix_memalign(&raw, 64, kPhysicalBytes) != 0 || !raw) {
        trace("MA");
        return kFakeFailure;
    }
    const int id = next_host_id++;
    live_hosts[raw] = {id, (uint64_t)bytes, 0};
    pending_host_id = id;
    trace("M" + number(id));
    *out = raw;
    return cudaSuccess;
}

static cudaError_t cudaFreeHost(void *raw) {
    ++free_calls;
    auto found = live_hosts.find(raw);
    if (found == live_hosts.end()) {
        ++api_errors;
        trace("F?");
        return kFakeFailure;
    }
    const int id = found->second.id;
    if (fail_free_at == free_calls || raw == fail_free_ptr) {
        trace("F!" + number(id));
        note_release_failure();
        return kFakeFailure;
    }
    if (paired_event_is_live(id)) {
        ++order_violations;
        trace("FO" + number(id));
    } else {
        trace("F" + number(id));
    }
    std::free(raw);
    live_hosts.erase(found);
    return cudaSuccess;
}

static cudaError_t cudaEventCreateWithFlags(cudaEvent_t *out, unsigned flags) {
    ++event_create_calls;
    if (!out || flags != cudaEventDisableTiming) {
        ++api_errors;
        trace("E?");
        return kFakeFailure;
    }
    *out = nullptr;
    if (fail_event_at == event_create_calls) {
        trace("E!");
        return kFakeFailure;
    }
    const int id = next_event_id++;
    const int host_id = next_event_unpaired ? 0 : pending_host_id;
    next_event_unpaired = false;
    pending_host_id = 0;
    void *event = reinterpret_cast<void *>(
        (uintptr_t)0x100000u + (uintptr_t)id * 0x100u);
    live_events[event] = {id, host_id};
    if (host_id != 0) {
        for (auto &entry : live_hosts) {
            if (entry.second.id == host_id) entry.second.paired_event_id = id;
        }
    }
    trace("E" + number(id) + "H" + number(host_id));
    *out = event;
    return cudaSuccess;
}

static cudaError_t cudaEventDestroy(cudaEvent_t event) {
    ++event_destroy_calls;
    auto found = live_events.find(event);
    if (found == live_events.end()) {
        ++api_errors;
        trace("D?");
        note_release_failure();
        return kFakeFailure;
    }
    const int id = found->second.id;
    if (fail_event_destroy_at == event_destroy_calls ||
        event == fail_event_destroy_handle) {
        trace("D!" + number(id));
        note_release_failure();
        return kFakeFailure;
    }
    const int host_id = found->second.paired_host_id;
    if (host_id != 0) {
        for (auto &entry : live_hosts) {
            if (entry.second.id == host_id) entry.second.paired_event_id = 0;
        }
    }
    trace("D" + number(id));
    live_events.erase(found);
    return cudaSuccess;
}

static cudaError_t cudaStreamCreateWithFlags(cudaStream_t *out, unsigned flags) {
    ++stream_create_calls;
    if (!out || flags != cudaStreamNonBlocking) {
        ++api_errors;
        trace("S?");
        return kFakeFailure;
    }
    *out = nullptr;
    if (fail_stream_create_at == stream_create_calls) {
        trace("S!");
        return kFakeFailure;
    }
    const int id = next_stream_id++;
    void *stream = reinterpret_cast<void *>(
        (uintptr_t)0x200000u + (uintptr_t)id * 0x100u);
    live_streams[stream] = {id};
    trace("S" + number(id));
    *out = stream;
    return cudaSuccess;
}

static cudaError_t cudaStreamDestroy(cudaStream_t stream) {
    ++stream_destroy_calls;
    auto found = live_streams.find(stream);
    if (found == live_streams.end()) {
        ++api_errors;
        trace("X?");
        note_release_failure();
        return kFakeFailure;
    }
    const int id = found->second.id;
    if (fail_stream_destroy_at == stream_destroy_calls ||
        stream == fail_stream_destroy_handle) {
        trace("X!" + number(id));
        note_release_failure();
        return kFakeFailure;
    }
    trace("X" + number(id));
    live_streams.erase(found);
    return cudaSuccess;
}

static const char *cudaGetErrorString(cudaError_t) { return "fake-cuda-error"; }
static cudaError_t cudaGetLastError(void) { return cudaSuccess; }

static void configure_process(void) {
    alarm(15);
    struct rlimit limit = {0, 0};
    (void)setrlimit(RLIMIT_CORE, &limit);
}
"""

# These are declaration shims only.  The separate source tests below require
# the reservation declarations in ds4_cuda.cu; no ownership algorithm is faked.
DECLARATION_SHIMS = r"""
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

/* Legacy extracted fixtures have no attached observer.  Keep the actual
 * production wrapper names visible in extracted bodies while routing their
 * inactive-observer calls to the existing fake raw CUDA layer. */
#define cuda_laguna_resident_malloc_host cudaMallocHost
#define cuda_laguna_resident_free_host cudaFreeHost
static int cuda_laguna_resident_observer_safe(void) { return 1; }
"""

# Only actual extracted definitions are included.  In particular, a missing
# helper is not replaced with a fixture helper: the ownership driver below
# calls it and therefore produces a compiler RED.
ACTUAL_BODY_TEXT = "\n".join(
    definition
    for definition in (
        ALIGN,
        SLOTS_RELEASE,
        MODEL_RELEASE,
        SELECTED_RELEASE_ANY,
        MODEL_POOL,
        SELECTED_POOL,
    )
    if definition is not None
)

FIXTURE_CONFIG = r"""
#ifndef DS4_CONTRACT_SIMULATED_SIZE_MAX
#define DS4_CONTRACT_SIMULATED_SIZE_MAX SIZE_MAX
#endif
#ifndef DS4_CONTRACT_NARROW_REQUEST
#define DS4_CONTRACT_NARROW_REQUEST 0u
#endif
"""
NARROW_SIZE_CONFIG = r"""
/* This is a host-preprocessor simulation, not a native 32-bit claim. */
#undef SIZE_MAX
#define DS4_CONTRACT_SIMULATED_SIZE_MAX 127u
#define SIZE_MAX DS4_CONTRACT_SIMULATED_SIZE_MAX
#define DS4_CONTRACT_NARROW_REQUEST (DS4_CONTRACT_SIMULATED_SIZE_MAX + 1u)
"""

FIXTURE_SUFFIX = r"""
struct pool_view {
    const char *name;
    void **raw;
    void **stage;
    cudaEvent_t *events;
    uint64_t *reserved;
    uint64_t ready;
    cudaStream_t stream;
};

static std::string join_records(std::vector<std::string> records) {
    std::sort(records.begin(), records.end());
    if (records.empty()) return "-";
    std::string result;
    for (size_t i = 0; i < records.size(); ++i) {
        if (i != 0) result += ",";
        result += records[i];
    }
    return result;
}

static std::string all_host_records(void) {
    std::vector<std::string> records;
    for (const auto &entry : live_hosts) {
        records.push_back(number(entry.second.id) + ":" +
                          std::to_string((unsigned long long)entry.second.requested));
    }
    return join_records(records);
}
static std::string all_event_records(void) {
    std::vector<std::string> records;
    for (const auto &entry : live_events) {
        records.push_back(number(entry.second.id) + ":" +
                          number(entry.second.paired_host_id));
    }
    return join_records(records);
}
static std::string all_stream_records(void) {
    std::vector<std::string> records;
    for (const auto &entry : live_streams) records.push_back(number(entry.second.id));
    return join_records(records);
}

static std::string owned_host_records(const pool_view *views, size_t count) {
    std::vector<std::string> records;
    for (size_t v = 0; v < count; ++v) {
        for (size_t i = 0; i < 4; ++i) {
            void *raw = views[v].raw[i];
            if (!raw) continue;
            const auto found = live_hosts.find(raw);
            if (found == live_hosts.end()) {
                records.push_back("dead@" + pointer_number(raw));
            } else {
                records.push_back(number(found->second.id) + ":" +
                                  std::to_string((unsigned long long)found->second.requested));
            }
        }
    }
    return join_records(records);
}
static std::string owned_event_records(const pool_view *views, size_t count) {
    std::vector<std::string> records;
    for (size_t v = 0; v < count; ++v) {
        for (size_t i = 0; i < 4; ++i) {
            cudaEvent_t event = views[v].events[i];
            if (!event) continue;
            const auto found = live_events.find(event);
            if (found == live_events.end()) {
                records.push_back("dead@" + pointer_number(event));
            } else {
                records.push_back(number(found->second.id) + ":" +
                                  number(found->second.paired_host_id));
            }
        }
    }
    return join_records(records);
}
static std::string owned_stream_records(const pool_view *views, size_t count) {
    std::vector<std::string> records;
    for (size_t v = 0; v < count; ++v) {
        cudaStream_t stream = views[v].stream;
        if (!stream) continue;
        const auto found = live_streams.find(stream);
        records.push_back(found == live_streams.end()
            ? "dead@" + pointer_number(stream)
            : number(found->second.id));
    }
    return join_records(records);
}
static std::string trace_records(void) {
    if (operation_trace.empty()) return "-";
    std::string result;
    for (size_t i = 0; i < operation_trace.size(); ++i) {
        if (i != 0) result += ",";
        result += operation_trace[i];
    }
    return result;
}

static void emit_pool(const char *label, const pool_view &view) {
    std::printf("%s_%s_ready=%llu\n", label, view.name,
                (unsigned long long)view.ready);
    std::printf("%s_%s_stream=%s\n", label, view.name,
                pointer_number(view.stream).c_str());
    std::printf("%s_%s_stream_live=%d\n", label, view.name,
                stream_live(view.stream));
    for (int i = 0; i < 4; ++i) {
        std::printf("%s_%s_raw_%d=%s\n", label, view.name, i,
                    pointer_number(view.raw[i]).c_str());
        std::printf("%s_%s_raw_live_%d=%d\n", label, view.name, i,
                    host_live(view.raw[i]));
        std::printf("%s_%s_raw_owner_id_%d=%d\n", label, view.name, i,
                    observed_host_id(view.raw[i]));
        std::printf("%s_%s_raw_requested_%d=%llu\n", label, view.name, i,
                    (unsigned long long)observed_host_requested(view.raw[i]));
        std::printf("%s_%s_stage_%d=%s\n", label, view.name, i,
                    pointer_number(view.stage[i]).c_str());
        std::printf("%s_%s_event_%d=%s\n", label, view.name, i,
                    pointer_number(view.events[i]).c_str());
        std::printf("%s_%s_event_live_%d=%d\n", label, view.name, i,
                    event_live(view.events[i]));
        std::printf("%s_%s_event_owner_id_%d=%d\n", label, view.name, i,
                    observed_event_id(view.events[i]));
        std::printf("%s_%s_event_paired_host_%d=%d\n", label, view.name, i,
                    observed_event_paired_host(view.events[i]));
        std::printf("%s_%s_reserved_%d=%llu\n", label, view.name, i,
                    (unsigned long long)view.reserved[i]);
    }
}

static void emit_snapshot(const char *label,
                          void **local_raw = nullptr,
                          void **local_stage = nullptr,
                          cudaEvent_t *local_events = nullptr,
                          uint64_t *local_reserved = nullptr) {
    pool_view views[3] = {
        {"model", g_model_stage_raw, g_model_stage, g_model_stage_event,
         g_model_stage_reserved_bytes, g_model_stage_bytes,
         g_model_upload_stream},
        {"selected", g_stream_selected_stage_raw, g_stream_selected_stage,
         g_stream_selected_stage_event, g_stream_selected_stage_reserved_bytes,
         g_stream_selected_stage_bytes, g_stream_selected_upload_stream},
        {"helper", local_raw, local_stage, local_events, local_reserved, 0, nullptr},
    };
    size_t count = local_raw ? 3 : 2;
    for (size_t i = 0; i < count; ++i) emit_pool(label, views[i]);
    std::printf("%s_ledger_hosts=%s\n", label, all_host_records().c_str());
    std::printf("%s_owners_hosts=%s\n", label,
                owned_host_records(views, count).c_str());
    std::printf("%s_ledger_events=%s\n", label, all_event_records().c_str());
    std::printf("%s_owners_events=%s\n", label,
                owned_event_records(views, count).c_str());
    std::printf("%s_ledger_streams=%s\n", label, all_stream_records().c_str());
    std::printf("%s_owners_streams=%s\n", label,
                owned_stream_records(views, count).c_str());
    std::printf("%s_malloc_calls=%d\n", label, malloc_calls);
    std::printf("%s_event_create_calls=%d\n", label, event_create_calls);
    std::printf("%s_free_calls=%d\n", label, free_calls);
    std::printf("%s_event_destroy_calls=%d\n", label, event_destroy_calls);
    std::printf("%s_stream_create_calls=%d\n", label, stream_create_calls);
    std::printf("%s_stream_destroy_calls=%d\n", label, stream_destroy_calls);
    std::printf("%s_api_errors=%d\n", label, api_errors);
    std::printf("%s_oversize_attempts=%d\n", label, oversize_attempts);
    std::printf("%s_release_failure_seq=%d\n", label, release_failure_seq);
    std::printf("%s_malloc_after_release_failure=%d\n", label,
                malloc_after_release_failure);
    std::printf("%s_order_violations=%d\n", label, order_violations);
    std::printf("%s_order_ok=%d\n", label, order_violations == 0 ? 1 : 0);
    std::printf("%s_trace=%s\n", label, trace_records().c_str());
    std::fflush(stdout);
}

static int model_control(void) {
    const int control = cuda_model_stage_pool_alloc(64);
    std::printf("control=%d\n", control);
    emit_snapshot("control");
    return control;
}
static int selected_control(void) {
    const int control = cuda_stream_selected_stage_pool_alloc(64);
    std::printf("control=%d\n", control);
    emit_snapshot("control");
    return control;
}

static int scenario_zero_from_empty(bool selected) {
    std::printf("control=1\n");
    emit_snapshot("control");
    const int result = selected
        ? cuda_stream_selected_stage_pool_alloc(0)
        : cuda_model_stage_pool_alloc(0);
    std::printf("target_result=%d\n", result);
    emit_snapshot("target");
    return 0;
}

static int scenario_zero_from_ready(bool selected) {
    const int control = selected ? selected_control() : model_control();
    if (!control) return 0;
    const int result = selected
        ? cuda_stream_selected_stage_pool_alloc(0)
        : cuda_model_stage_pool_alloc(0);
    std::printf("target_result=%d\n", result);
    emit_snapshot("target");
    return 0;
}

static int scenario_oracle(bool reversed) {
    void *raw = nullptr;
    cudaEvent_t event = nullptr;
    const int allocated = cudaMallocHost(&raw, 11) == cudaSuccess;
    const int created = allocated &&
        cudaEventCreateWithFlags(&event, cudaEventDisableTiming) == cudaSuccess;
    if (created && reversed) {
        (void)cudaFreeHost(raw);
        (void)cudaEventDestroy(event);
    } else if (created) {
        (void)cudaEventDestroy(event);
        (void)cudaFreeHost(raw);
    }
    std::printf("oracle_valid_control=%d\n",
                !reversed && created && order_violations == 0);
    std::printf("oracle_reversed_control=%d\n",
                reversed && created && order_violations == 1);
    emit_snapshot("oracle");
    return 0;
}

static int scenario_growth(bool selected) {
    int control = selected ? selected_control() : model_control();
    if (!control) return 0;
    const cudaStream_t old_stream = selected
        ? g_stream_selected_upload_stream : g_model_upload_stream;
    const int result = selected
        ? cuda_stream_selected_stage_pool_alloc(96)
        : cuda_model_stage_pool_alloc(96);
    const cudaStream_t new_stream = selected
        ? g_stream_selected_upload_stream : g_model_upload_stream;
    std::printf("target_result=%d\n", result);
    std::printf("target_stream_reused=%d\n", old_stream == new_stream);
    emit_snapshot("target");
    return 0;
}

static int scenario_reuse(bool selected) {
    const int first = selected
        ? cuda_stream_selected_stage_pool_alloc(96)
        : cuda_model_stage_pool_alloc(96);
    std::printf("control=%d\n", first);
    emit_snapshot("control");
    const int second = first && (selected
        ? cuda_stream_selected_stage_pool_alloc(32)
        : cuda_model_stage_pool_alloc(32));
    const int third = second && (selected
        ? cuda_stream_selected_stage_pool_alloc(96)
        : cuda_model_stage_pool_alloc(96));
    std::printf("target_result=%d\n", third);
    emit_snapshot("target");
    return 0;
}

static int scenario_initial_failure(bool selected, bool event_failure) {
    if (event_failure) fail_event_at = 3;
    else fail_malloc_at = 3;
    const int result = selected
        ? cuda_stream_selected_stage_pool_alloc(64)
        : cuda_model_stage_pool_alloc(64);
    std::printf("target_result=%d\n", result);
    emit_snapshot("target");
    return 0;
}

static int scenario_selected_rollback(const char *kind) {
    if (std::strcmp(kind, "clean") == 0) {
        fail_event_at = 2;
    } else if (std::strcmp(kind, "free") == 0) {
        fail_malloc_at = 2;
        fail_free_at = 1;
    } else if (std::strcmp(kind, "event") == 0) {
        fail_event_at = 2;
        fail_event_destroy_at = 1;
    } else if (std::strcmp(kind, "stream") == 0) {
        fail_malloc_at = 1;
        fail_stream_destroy_at = 1;
    } else {
        return 2;
    }
    const int result = cuda_stream_selected_stage_pool_alloc(64);
    std::printf("target_result=%d\n", result);
    emit_snapshot("target");
    return 0;
}

static int scenario_resize_failure(bool selected, const char *kind) {
    const int control = selected ? selected_control() : model_control();
    if (!control) return 0;
    if (std::strcmp(kind, "free") == 0) {
        fail_free_at = free_calls + 1;
    } else if (std::strcmp(kind, "event") == 0) {
        fail_event_destroy_at = event_destroy_calls + 1;
    } else if (std::strcmp(kind, "stream") == 0 && selected) {
        fail_stream_destroy_at = stream_destroy_calls + 1;
    } else {
        return 2;
    }
    const int result = selected
        ? cuda_stream_selected_stage_pool_alloc(96)
        : cuda_model_stage_pool_alloc(96);
    std::printf("target_result=%d\n", result);
    emit_snapshot("target");
    return 0;
}

static int scenario_retry_model(void) {
    const int control = model_control();
    if (!control) return 0;
    fail_free_at = free_calls + 1;
    const int first = cuda_model_stage_pool_alloc(96);
    std::printf("first=%d\n", first);
    emit_snapshot("after_first");
    fail_free_at = 0;
    const int second = cuda_model_stage_pool_alloc(96);
    std::printf("second=%d\n", second);
    emit_snapshot("target");
    return 0;
}

static int scenario_retry_selected(void) {
    fail_malloc_at = 2;
    fail_free_at = 1;
    const int first = cuda_stream_selected_stage_pool_alloc(64);
    std::printf("first=%d\n", first);
    emit_snapshot("after_first");
    fail_malloc_at = 0;
    fail_free_at = 0;
    const int second = cuda_stream_selected_stage_pool_alloc(96);
    std::printf("second=%d\n", second);
    emit_snapshot("target");
    return 0;
}

#if DS4_CONTRACT_NEW_API
static int scenario_full_release(bool selected) {
    const int control = selected ? selected_control() : model_control();
    if (!control) return 0;
    const int result = selected
        ? cuda_stream_selected_stage_release()
        : cuda_model_stage_release();
    std::printf("target_result=%d\n", result);
    emit_snapshot("target");
    return 0;
}

static int scenario_release_stream_failure(bool selected) {
    const int control = selected ? selected_control() : model_control();
    if (!control) return 0;
    if (selected) {
        fail_stream_destroy_handle = g_stream_selected_upload_stream;
    } else {
        fail_stream_destroy_handle = g_model_upload_stream;
    }
    const int first = selected
        ? cuda_stream_selected_stage_release()
        : cuda_model_stage_release();
    std::printf("first=%d\n", first);
    emit_snapshot("after_first");
    fail_stream_destroy_handle = nullptr;
    const int second = selected
        ? cuda_stream_selected_stage_release()
        : cuda_model_stage_release();
    std::printf("second=%d\n", second);
    emit_snapshot("target");
    return 0;
}

static int seed_local(void **raw, void **stage, cudaEvent_t *events,
                      uint64_t *reserved) {
    std::memset(raw, 0, 4 * sizeof(*raw));
    std::memset(stage, 0, 4 * sizeof(*stage));
    std::memset(events, 0, 4 * sizeof(*events));
    std::memset(reserved, 0, 4 * sizeof(*reserved));
    if (cudaMallocHost(&raw[0], 11) != cudaSuccess) return 0;
    stage[0] = raw[0];
    reserved[0] = 11;
    if (cudaEventCreateWithFlags(&events[0], cudaEventDisableTiming) != cudaSuccess) return 0;
    if (cudaMallocHost(&raw[1], 22) != cudaSuccess) return 0;
    stage[1] = raw[1];
    reserved[1] = 22;
    fake_next_event_without_host();
    return cudaEventCreateWithFlags(&events[2], cudaEventDisableTiming) == cudaSuccess;
}

static int scenario_helper(const char *kind) {
    void *raw[4];
    void *stage[4];
    cudaEvent_t events[4];
    uint64_t reserved[4];
    const int control = seed_local(raw, stage, events, reserved);
    std::printf("control=%d\n", control);
    emit_snapshot("control", raw, stage, events, reserved);
    if (!control) return 0;
    if (std::strcmp(kind, "error") == 0) fail_free_ptr = raw[1];
    const int result = cuda_stage_slots_release(raw, stage, events, reserved);
    std::printf("target_result=%d\n", result);
    emit_snapshot("target", raw, stage, events, reserved);
    return 0;
}

static int scenario_unknown_handle(void) {
    void *raw[4] = {reinterpret_cast<void *>(uintptr_t(0x12345u)), nullptr, nullptr, nullptr};
    void *stage[4] = {raw[0], nullptr, nullptr, nullptr};
    cudaEvent_t events[4] = {reinterpret_cast<void *>(uintptr_t(0x54321u)), nullptr, nullptr, nullptr};
    uint64_t reserved[4] = {7, 0, 0, 0};
    const int result = cuda_stage_slots_release(raw, stage, events, reserved);
    std::printf("target_result=%d\n", result);
    emit_snapshot("target", raw, stage, events, reserved);
    return 0;
}

static int scenario_both_pools(void) {
    const int model = cuda_model_stage_pool_alloc(64);
    const int selected = cuda_stream_selected_stage_pool_alloc(96);
    const int control = model && selected;
    std::printf("control=%d\n", control);
    emit_snapshot("control");
    if (!control) return 0;
    const int model_release = cuda_model_stage_release();
    std::printf("after_model_result=%d\n", model_release);
    emit_snapshot("after_model");
    const int selected_release = cuda_stream_selected_stage_release();
    std::printf("target_result=%d\n", selected_release);
    emit_snapshot("target");
    return 0;
}

static int scenario_max_malloc_failure(bool selected) {
    fail_malloc_at = 1;
    const int result = selected
        ? cuda_stream_selected_stage_pool_alloc(UINT64_MAX)
        : cuda_model_stage_pool_alloc(UINT64_MAX);
    std::printf("target_result=%d\n", result);
    std::printf("target_size_max_is_uint64=%d\n", SIZE_MAX == UINT64_MAX ? 1 : 0);
    emit_snapshot("target");
    return 0;
}

static int scenario_simulated_narrow(bool prepare = true) {
    const int model_control = prepare ? cuda_model_stage_pool_alloc(64) : 1;
    const int selected_control = prepare ? cuda_stream_selected_stage_pool_alloc(64) : 1;
    const int control = model_control && selected_control;
    std::printf("control=%d\n", control);
    emit_snapshot("control");
    if (!control) return 0;
    const uint64_t request = (uint64_t)DS4_CONTRACT_NARROW_REQUEST;
    const int model = cuda_model_stage_pool_alloc(request);
    const int selected = cuda_stream_selected_stage_pool_alloc(request);
    std::printf("target_model_result=%d\n", model);
    std::printf("target_selected_result=%d\n", selected);
    std::printf("target_simulated_size_max=%llu\n",
                (unsigned long long)DS4_CONTRACT_SIMULATED_SIZE_MAX);
    emit_snapshot("target");
    return 0;
}
#endif

static int scenario(const char *name) {
    if (std::strcmp(name, "oracle-valid") == 0) return scenario_oracle(false);
    if (std::strcmp(name, "oracle-reversed") == 0) return scenario_oracle(true);
    if (std::strcmp(name, "model-control") == 0) return model_control() ? 0 : 1;
    if (std::strcmp(name, "selected-control") == 0) return selected_control() ? 0 : 1;
    if (std::strcmp(name, "model-empty-zero") == 0) return scenario_zero_from_empty(false);
    if (std::strcmp(name, "selected-empty-zero") == 0) return scenario_zero_from_empty(true);
    if (std::strcmp(name, "model-zero") == 0) return scenario_zero_from_ready(false);
    if (std::strcmp(name, "selected-zero") == 0) return scenario_zero_from_ready(true);
    if (std::strcmp(name, "model-growth") == 0) return scenario_growth(false);
    if (std::strcmp(name, "selected-growth") == 0) return scenario_growth(true);
    if (std::strcmp(name, "model-reuse") == 0) return scenario_reuse(false);
    if (std::strcmp(name, "selected-reuse") == 0) return scenario_reuse(true);
    if (std::strcmp(name, "model-malloc-failure") == 0)
        return scenario_initial_failure(false, false);
    if (std::strcmp(name, "model-event-failure") == 0)
        return scenario_initial_failure(false, true);
    if (std::strcmp(name, "selected-rollback-clean") == 0)
        return scenario_selected_rollback("clean");
    if (std::strcmp(name, "selected-rollback-free-failure") == 0)
        return scenario_selected_rollback("free");
    if (std::strcmp(name, "selected-rollback-event-failure") == 0)
        return scenario_selected_rollback("event");
    if (std::strcmp(name, "selected-rollback-stream-failure") == 0)
        return scenario_selected_rollback("stream");
    if (std::strcmp(name, "model-resize-free-failure") == 0)
        return scenario_resize_failure(false, "free");
    if (std::strcmp(name, "model-resize-event-failure") == 0)
        return scenario_resize_failure(false, "event");
    if (std::strcmp(name, "selected-resize-free-failure") == 0)
        return scenario_resize_failure(true, "free");
    if (std::strcmp(name, "selected-resize-event-failure") == 0)
        return scenario_resize_failure(true, "event");
    if (std::strcmp(name, "selected-resize-stream-failure") == 0)
        return scenario_resize_failure(true, "stream");
    if (std::strcmp(name, "model-retry") == 0) return scenario_retry_model();
    if (std::strcmp(name, "selected-retry") == 0) return scenario_retry_selected();
#if DS4_CONTRACT_NEW_API
    if (std::strcmp(name, "model-release") == 0) return scenario_full_release(false);
    if (std::strcmp(name, "selected-release") == 0) return scenario_full_release(true);
    if (std::strcmp(name, "model-release-stream-failure") == 0)
        return scenario_release_stream_failure(false);
    if (std::strcmp(name, "selected-release-stream-failure") == 0)
        return scenario_release_stream_failure(true);
    if (std::strcmp(name, "helper-mixed-success") == 0) return scenario_helper("success");
    if (std::strcmp(name, "helper-mixed-error") == 0) return scenario_helper("error");
    if (std::strcmp(name, "unknown-resource") == 0) return scenario_unknown_handle();
    if (std::strcmp(name, "both-independent") == 0) return scenario_both_pools();
    if (std::strcmp(name, "model-max-malloc-failure") == 0)
        return scenario_max_malloc_failure(false);
    if (std::strcmp(name, "selected-max-malloc-failure") == 0)
        return scenario_max_malloc_failure(true);
    if (std::strcmp(name, "simulated-narrow-size-max-empty") == 0)
        return scenario_simulated_narrow(false);
    if (std::strcmp(name, "simulated-narrow-size-max") == 0)
        return scenario_simulated_narrow();
#endif
    return 2;
}

int main(int argc, char **argv) {
    configure_process();
    if (argc != 2) return 2;
    return scenario(argv[1]);
}
"""

CURRENT_SOURCE = (
    FAKE_PREFIX
    + DECLARATION_SHIMS
    + ACTUAL_BODY_TEXT
    + FIXTURE_CONFIG
    + "\n#define DS4_CONTRACT_NEW_API 0\n"
    + FIXTURE_SUFFIX
)
OWNERSHIP_SOURCE = (
    FAKE_PREFIX
    + DECLARATION_SHIMS
    + ACTUAL_BODY_TEXT
    + FIXTURE_CONFIG
    + "\n#define DS4_CONTRACT_NEW_API 1\n"
    + FIXTURE_SUFFIX
)
OWNERSHIP_NARROW_SOURCE = (
    FAKE_PREFIX
    + NARROW_SIZE_CONFIG
    + DECLARATION_SHIMS
    + ACTUAL_BODY_TEXT
    + FIXTURE_CONFIG
    + "\n#define DS4_CONTRACT_NEW_API 1\n"
    + FIXTURE_SUFFIX
)


Scalar: TypeAlias = int | str


def parse_output(stdout: str) -> dict[str, Scalar]:
    values: dict[str, Scalar] = {}
    for line in stdout.splitlines():
        key, separator, value = line.partition("=")
        if not separator:
            continue
        if re.fullmatch(r"-?\d+", value):
            values[key] = int(value)
        else:
            values[key] = value
    return values


def integer(values: dict[str, Scalar], key: str) -> int:
    value = values.get(key)
    if not isinstance(value, int):
        raise AssertionError(f"missing integer {key}: {values}")
    return value


def records(values: dict[str, Scalar], key: str) -> list[str]:
    value = values.get(key)
    if isinstance(value, int):
        return [str(value)]
    if not isinstance(value, str):
        raise AssertionError(f"missing record {key}: {values}")
    return [] if value == "-" else value.split(",")


def assert_regular_snapshot(
    test: unittest.TestCase, values: dict[str, Scalar], label: str
) -> None:
    """Every normal snapshot must be ordered and free of fake API misuse."""
    test.assertEqual(integer(values, f"{label}_order_ok"), 1, f"{label}: release order")
    test.assertEqual(integer(values, f"{label}_api_errors"), 0, f"{label}: fake API errors")


def assert_full_ledger(
    test: unittest.TestCase, values: dict[str, Scalar], label: str
) -> None:
    """Compare independent whole-ledger records to all fixture owner pools."""
    assert_regular_snapshot(test, values, label)
    for kind in ("hosts", "events", "streams"):
        with test.subTest(snapshot=label, resource=kind):
            test.assertEqual(
                records(values, f"{label}_owners_{kind}"),
                records(values, f"{label}_ledger_{kind}"),
                f"{label}: owner union must equal the complete live {kind} ledger",
            )


def assert_pool_shape(
    test: unittest.TestCase,
    values: dict[str, Scalar],
    label: str,
    pool: str,
    *,
    ready: int | None = None,
    raw_live: tuple[int, int, int, int] | None = None,
    event_live: tuple[int, int, int, int] | None = None,
    stream_live: int | None = None,
    reserved: tuple[int, int, int, int] | None = None,
    require_reservation_for_live_raw: bool = False,
    allow_event_only_slots: bool = False,
) -> None:
    """Check every handle plus independent owner IDs and requested bytes."""
    assert_regular_snapshot(test, values, label)
    if ready is not None:
        test.assertEqual(integer(values, f"{label}_{pool}_ready"), ready)
    if stream_live is not None:
        test.assertEqual(integer(values, f"{label}_{pool}_stream_live"), stream_live)
    for i in range(4):
        raw = integer(values, f"{label}_{pool}_raw_{i}")
        stage = integer(values, f"{label}_{pool}_stage_{i}")
        event = integer(values, f"{label}_{pool}_event_{i}")
        observed_raw_live = integer(values, f"{label}_{pool}_raw_live_{i}")
        observed_raw_id = integer(values, f"{label}_{pool}_raw_owner_id_{i}")
        observed_requested = integer(values, f"{label}_{pool}_raw_requested_{i}")
        observed_event_live = integer(values, f"{label}_{pool}_event_live_{i}")
        observed_event_id = integer(values, f"{label}_{pool}_event_owner_id_{i}")
        observed_paired_host = integer(values, f"{label}_{pool}_event_paired_host_{i}")
        byte_record = integer(values, f"{label}_{pool}_reserved_{i}")
        if raw_live is not None:
            test.assertEqual(observed_raw_live, raw_live[i], f"{label} {pool} raw {i}")
        if event_live is not None:
            test.assertEqual(observed_event_live, event_live[i], f"{label} {pool} event {i}")
        if observed_raw_live:
            test.assertNotEqual(raw, 0, f"{label} {pool} raw {i} live pointer")
            test.assertEqual(stage, raw, f"{label} {pool} aligned stage {i}")
            test.assertGreater(observed_raw_id, 0, f"{label} {pool} raw owner {i}")
            test.assertGreater(observed_requested, 0, f"{label} {pool} requested bytes {i}")
            if require_reservation_for_live_raw:
                test.assertEqual(
                    byte_record,
                    observed_requested,
                    f"{label} {pool} reservation must equal observed request {i}",
                )
        else:
            test.assertEqual(raw, 0, f"{label} {pool} cleared raw {i}")
            test.assertEqual(stage, 0, f"{label} {pool} cleared stage {i}")
            test.assertEqual(observed_raw_id, 0, f"{label} {pool} cleared owner {i}")
            test.assertEqual(observed_requested, 0, f"{label} {pool} cleared requested bytes {i}")
            if reserved is not None or require_reservation_for_live_raw:
                test.assertEqual(byte_record, 0, f"{label} {pool} cleared bytes {i}")
        if observed_event_live:
            test.assertNotEqual(event, 0, f"{label} {pool} live event {i}")
            test.assertGreater(observed_event_id, 0, f"{label} {pool} event owner {i}")
            if observed_raw_live:
                test.assertEqual(
                    observed_paired_host,
                    observed_raw_id,
                    f"{label} {pool} event must pair to same-slot raw {i}",
                )
            elif allow_event_only_slots:
                test.assertEqual(
                    observed_paired_host,
                    0,
                    f"{label} {pool} event-only slot must be unpaired {i}",
                )
            else:
                test.fail(f"{label} {pool} unexpected event-only slot {i}")
        else:
            test.assertEqual(event, 0, f"{label} {pool} cleared event {i}")
            test.assertEqual(observed_event_id, 0, f"{label} {pool} cleared event owner {i}")
            test.assertEqual(observed_paired_host, 0, f"{label} {pool} cleared event pair {i}")
        if reserved is not None:
            test.assertEqual(byte_record, reserved[i], f"{label} {pool} bytes {i}")


def assert_clean_control(
    test: unittest.TestCase,
    values: dict[str, Scalar],
    pool: str,
    *,
    reservation_required: bool,
) -> None:
    test.assertEqual(integer(values, "control"), 1)
    assert_full_ledger(test, values, "control")
    assert_pool_shape(
        test,
        values,
        "control",
        pool,
        ready=64,
        raw_live=(1, 1, 1, 1),
        event_live=(1, 1, 1, 1),
        stream_live=1,
        reserved=(64, 64, 64, 64) if reservation_required else None,
        require_reservation_for_live_raw=reservation_required,
    )


def snapshot_value_keys(label: str) -> list[str]:
    keys: list[str] = []
    for pool in ("model", "selected"):
        keys.extend((
            f"{label}_{pool}_ready",
            f"{label}_{pool}_stream",
            f"{label}_{pool}_stream_live",
        ))
        for i in range(4):
            keys.extend((
                f"{label}_{pool}_raw_{i}",
                f"{label}_{pool}_raw_live_{i}",
                f"{label}_{pool}_raw_owner_id_{i}",
                f"{label}_{pool}_raw_requested_{i}",
                f"{label}_{pool}_stage_{i}",
                f"{label}_{pool}_event_{i}",
                f"{label}_{pool}_event_live_{i}",
                f"{label}_{pool}_event_owner_id_{i}",
                f"{label}_{pool}_event_paired_host_{i}",
                f"{label}_{pool}_reserved_{i}",
            ))
    for resource in ("hosts", "events", "streams"):
        keys.extend((f"{label}_ledger_{resource}", f"{label}_owners_{resource}"))
    for field in (
        "malloc_calls", "event_create_calls", "free_calls", "event_destroy_calls",
        "stream_create_calls", "stream_destroy_calls", "api_errors", "oversize_attempts",
        "release_failure_seq", "malloc_after_release_failure", "order_violations",
        "order_ok", "trace",
    ):
        keys.append(f"{label}_{field}")
    return keys


def assert_snapshot_unchanged(
    test: unittest.TestCase,
    values: dict[str, Scalar],
    before: str = "control",
    after: str = "target",
) -> None:
    """A rejected request cannot mutate any pool, ledger, counter, or trace."""
    assert_full_ledger(test, values, before)
    assert_full_ledger(test, values, after)
    before_keys = snapshot_value_keys(before)
    after_keys = snapshot_value_keys(after)
    test.assertEqual(len(before_keys), len(after_keys))
    for before_key, after_key in zip(before_keys, after_keys, strict=True):
        test.assertIn(before_key, values)
        test.assertIn(after_key, values)
        test.assertEqual(
            values[after_key],
            values[before_key],
            f"rejection changed {after_key.removeprefix(after + '_')}",
        )


def assert_empty_snapshot(
    test: unittest.TestCase, values: dict[str, Scalar], label: str
) -> None:
    assert_full_ledger(test, values, label)
    for pool in ("model", "selected"):
        assert_pool_shape(
            test, values, label, pool, ready=0,
            raw_live=(0, 0, 0, 0), event_live=(0, 0, 0, 0), stream_live=0,
            reserved=(0, 0, 0, 0), require_reservation_for_live_raw=True,
        )
    for field in ("malloc_calls", "free_calls", "event_create_calls",
                  "event_destroy_calls", "stream_create_calls",
                  "stream_destroy_calls", "oversize_attempts"):
        test.assertEqual(integer(values, f"{label}_{field}"), 0)
    test.assertEqual(records(values, f"{label}_trace"), [])


def assert_rejected_from_ready(
    test: unittest.TestCase,
    values: dict[str, Scalar],
    pool: str,
    *,
    reservation_required: bool,
) -> None:
    assert_clean_control(test, values, pool, reservation_required=reservation_required)
    other = "selected" if pool == "model" else "model"
    # Keep the no-effect oracle observable even while ff2 still returns success.
    with test.subTest(pool=pool, rejection="result"):
        test.assertEqual(integer(values, "target_result"), 0)
    with test.subTest(pool=pool, rejection="complete-unchanged-snapshot"):
        assert_snapshot_unchanged(test, values)
    with test.subTest(pool=pool, rejection="active-pool-shape"):
        assert_pool_shape(
            test,
            values,
            "target",
            pool,
            ready=64,
            raw_live=(1, 1, 1, 1),
            event_live=(1, 1, 1, 1),
            stream_live=1,
            reserved=(64, 64, 64, 64) if reservation_required else None,
            require_reservation_for_live_raw=reservation_required,
        )
    with test.subTest(pool=pool, rejection="other-pool-shape"):
        assert_pool_shape(
            test,
            values,
            "target",
            other,
            ready=0,
            raw_live=(0, 0, 0, 0),
            event_live=(0, 0, 0, 0),
            stream_live=0,
            reserved=(0, 0, 0, 0),
            require_reservation_for_live_raw=reservation_required,
        )


def assert_no_alloc_after_release_failure(
    test: unittest.TestCase, values: dict[str, Scalar], label: str = "target"
) -> None:
    """At a failure boundary, the first failed release call is terminal."""
    trace = records(values, f"{label}_trace")
    failures = [
        index
        for index, operation in enumerate(trace)
        if operation.startswith(("D!", "F!", "X!"))
    ]
    test.assertTrue(failures, f"{label}: fake trace did not record the armed release failure")
    if failures:
        failure_index = failures[0]
        test.assertEqual(
            integer(values, f"{label}_release_failure_seq"),
            failure_index + 1,
            f"{label}: first release failure sequence",
        )
        test.assertEqual(
            len(trace),
            failure_index + 1,
            f"{label}: resource operation followed release failure: {trace}",
        )
    test.assertEqual(integer(values, f"{label}_malloc_after_release_failure"), 0)
    assert_regular_snapshot(test, values, label)


class FixtureCompiler:
    """Small reusable compile/run harness.  Temporary cleanup is registered first."""

    def build(self, prefix: str, source: str) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix=prefix)
        self.addClassCleanup(self._tmp.cleanup)  # type: ignore[attr-defined]
        base = Path(self._tmp.name)
        self._source = base / "fixture.cc"
        self._binary = base / "fixture"
        self._source.write_text(source, encoding="utf-8")
        self._compile = subprocess.run(
            ["c++", "-std=c++17", "-O0", str(self._source), "-o", str(self._binary)],
            cwd=ROOT,
            env=SAFE_ENV,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )

    def run_case(self, test: unittest.TestCase, name: str) -> dict[str, Scalar]:
        test.assertEqual(
            self._compile.returncode,
            0,
            "fixture compiler RED:\n" + self._compile.stderr,
        )
        result = subprocess.run(
            [str(self._binary), name],
            cwd=ROOT,
            env=SAFE_ENV,
            capture_output=True,
            text=True,
            timeout=16,
            check=False,
        )
        test.assertEqual(
            result.returncode,
            0,
            f"fixture scenario {name} RED:\nstdout={result.stdout}\nstderr={result.stderr}",
        )
        return parse_output(result.stdout)


class CurrentSourceBehaviorContractTest(FixtureCompiler, unittest.TestCase):
    """Execute current pool bodies even when the future ownership API is absent."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.build(cls, "ds4-stage-current-contract-", CURRENT_SOURCE)

    def _case(self, name: str) -> dict[str, Scalar]:
        return self.run_case(self, name)

    def _control(self, pool: str) -> dict[str, Scalar]:
        values = self._case(f"{pool}-control")
        assert_clean_control(self, values, pool, reservation_required=False)
        return values

    def _assert_resize_rejection(
        self,
        name: str,
        pool: str,
        *,
        raw_live: tuple[int, int, int, int],
        event_live: tuple[int, int, int, int],
        stream_live: int,
    ) -> None:
        # A separate fresh process proves the control before the fault case.
        self._control(pool)
        values = self._case(name)
        assert_clean_control(self, values, pool, reservation_required=False)
        with self.subTest(case=name, field="result"):
            self.assertEqual(integer(values, "target_result"), 0)
        with self.subTest(case=name, field="ready"):
            assert_pool_shape(
                self,
                values,
                "target",
                pool,
                ready=0,
                raw_live=raw_live,
                event_live=event_live,
                stream_live=stream_live,
            )
        with self.subTest(case=name, field="ledger"):
            assert_full_ledger(self, values, "target")
        with self.subTest(case=name, field="release-order-and-no-allocation"):
            assert_no_alloc_after_release_failure(self, values)
        with self.subTest(case=name, field="api"):
            self.assertEqual(integer(values, "target_api_errors"), 0)

    def test_00_actual_current_bodies_compile(self) -> None:
        self.assertEqual(self._compile.returncode, 0, self._compile.stderr)

    def test_01_fake_order_oracle_accepts_valid_and_rejects_reversed_order(self) -> None:
        valid = self._case("oracle-valid")
        self.assertEqual(integer(valid, "oracle_valid_control"), 1)
        self.assertEqual(integer(valid, "oracle_reversed_control"), 0)
        self.assertEqual(integer(valid, "oracle_order_ok"), 1)
        self.assertEqual(integer(valid, "oracle_api_errors"), 0)
        reversed_case = self._case("oracle-reversed")
        self.assertEqual(integer(reversed_case, "oracle_valid_control"), 0)
        self.assertEqual(integer(reversed_case, "oracle_reversed_control"), 1)
        self.assertEqual(integer(reversed_case, "oracle_order_ok"), 0)
        self.assertEqual(integer(reversed_case, "oracle_order_violations"), 1)
        self.assertEqual(integer(reversed_case, "oracle_api_errors"), 0)

    def test_02_zero_size_is_rejected_from_a_valid_pool_without_effect(self) -> None:
        for pool in ("model", "selected"):
            with self.subTest(pool=pool):
                values = self._case(f"{pool}-zero")
                assert_rejected_from_ready(
                    self, values, pool, reservation_required=False
                )
            with self.subTest(pool=pool, state="empty"):
                values = self._case(f"{pool}-empty-zero")
                assert_empty_snapshot(self, values, "control")
                with self.subTest(field="result"):
                    self.assertEqual(integer(values, "target_result"), 0)
                assert_snapshot_unchanged(self, values)
                assert_empty_snapshot(self, values, "target")

    def test_03_model_free_failure_blocks_new_allocation(self) -> None:
        self._assert_resize_rejection(
            "model-resize-free-failure",
            "model",
            raw_live=(1, 1, 1, 1),
            event_live=(0, 1, 1, 1),
            stream_live=1,
        )

    def test_04_model_event_failure_blocks_new_allocation(self) -> None:
        self._assert_resize_rejection(
            "model-resize-event-failure",
            "model",
            raw_live=(1, 1, 1, 1),
            event_live=(1, 1, 1, 1),
            stream_live=1,
        )

    def test_05_selected_free_failure_blocks_new_allocation(self) -> None:
        self._assert_resize_rejection(
            "selected-resize-free-failure",
            "selected",
            raw_live=(1, 1, 1, 1),
            event_live=(0, 1, 1, 1),
            stream_live=1,
        )

    def test_06_selected_event_failure_blocks_new_allocation(self) -> None:
        self._assert_resize_rejection(
            "selected-resize-event-failure",
            "selected",
            raw_live=(1, 1, 1, 1),
            event_live=(1, 1, 1, 1),
            stream_live=1,
        )

    def test_07_selected_stream_failure_blocks_new_allocation(self) -> None:
        self._assert_resize_rejection(
            "selected-resize-stream-failure",
            "selected",
            raw_live=(0, 0, 0, 0),
            event_live=(0, 0, 0, 0),
            stream_live=1,
        )

    def test_08_selected_failed_rollback_retains_all_unresolved_owners(self) -> None:
        expected = {
            "selected-rollback-free-failure": ((1, 0, 0, 0), (0, 0, 0, 0)),
            "selected-rollback-event-failure": ((1, 1, 0, 0), (1, 0, 0, 0)),
            "selected-rollback-stream-failure": ((0, 0, 0, 0), (0, 0, 0, 0)),
        }
        for name, (raw_live, event_live) in expected.items():
            with self.subTest(case=name):
                self._control("selected")
                values = self._case(name)
                with self.subTest(case=name, field="result"):
                    self.assertEqual(integer(values, "target_result"), 0)
                with self.subTest(case=name, field="retained-handles"):
                    assert_pool_shape(
                        self,
                        values,
                        "target",
                        "selected",
                        ready=0,
                        raw_live=raw_live,
                        event_live=event_live,
                        stream_live=1,
                    )
                with self.subTest(case=name, field="complete-ledger"):
                    assert_full_ledger(self, values, "target")
                with self.subTest(case=name, field="terminal-release-failure"):
                    assert_no_alloc_after_release_failure(self, values)
                with self.subTest(case=name, field="api"):
                    self.assertEqual(integer(values, "target_api_errors"), 0)


class NewOwnershipApiContractTest(FixtureCompiler, unittest.TestCase):
    """Required ownership API source, compiler, and runtime contract."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.build(cls, "ds4-stage-ownership-contract-", OWNERSHIP_SOURCE)
        cls._narrow_tmp = tempfile.TemporaryDirectory(prefix="ds4-stage-narrow-contract-")
        cls.addClassCleanup(cls._narrow_tmp.cleanup)
        base = Path(cls._narrow_tmp.name)
        cls._narrow_source = base / "fixture.cc"
        cls._narrow_binary = base / "fixture"
        cls._narrow_source.write_text(OWNERSHIP_NARROW_SOURCE, encoding="utf-8")
        cls._narrow_compile = subprocess.run(
            ["c++", "-std=c++17", "-O0", str(cls._narrow_source), "-o", str(cls._narrow_binary)],
            cwd=ROOT,
            env=SAFE_ENV,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )

    def _case(self, name: str) -> dict[str, Scalar]:
        return self.run_case(self, name)

    def _narrow_case(self, name: str) -> dict[str, Scalar]:
        self.assertEqual(
            self._narrow_compile.returncode,
            0,
            "simulated-narrow SIZE_MAX fixture compiler RED:\n" + self._narrow_compile.stderr,
        )
        result = subprocess.run(
            [str(self._narrow_binary), name],
            cwd=ROOT,
            env=SAFE_ENV,
            capture_output=True,
            text=True,
            timeout=16,
            check=False,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"simulated-narrow scenario {name} RED:\nstdout={result.stdout}\nstderr={result.stderr}",
        )
        return parse_output(result.stdout)

    def _control(self, pool: str) -> dict[str, Scalar]:
        values = self._case(f"{pool}-control")
        assert_clean_control(self, values, pool, reservation_required=True)
        return values

    def _assert_resize_rejection(
        self,
        name: str,
        pool: str,
        *,
        raw_live: tuple[int, int, int, int],
        event_live: tuple[int, int, int, int],
        stream_live: int,
        reserved: tuple[int, int, int, int],
    ) -> None:
        self._control(pool)
        values = self._case(name)
        assert_clean_control(self, values, pool, reservation_required=True)
        self.assertEqual(integer(values, "target_result"), 0)
        assert_pool_shape(
            self,
            values,
            "target",
            pool,
            ready=0,
            raw_live=raw_live,
            event_live=event_live,
            stream_live=stream_live,
            reserved=reserved,
            require_reservation_for_live_raw=True,
        )
        assert_full_ledger(self, values, "target")
        assert_no_alloc_after_release_failure(self, values)
        self.assertEqual(integer(values, "target_api_errors"), 0)

    def test_00_required_new_source_definitions_exist(self) -> None:
        self.assertFalse(
            MISSING_NEW_BODIES,
            "new ownership source RED; missing actual definitions: "
            + ", ".join(MISSING_NEW_BODIES),
        )

    def test_01_required_new_api_driver_compiles(self) -> None:
        self.assertEqual(
            self._compile.returncode,
            0,
            "new ownership compiler RED (available extracted bodies plus calls to all required APIs):\n"
            + self._compile.stderr,
        )

    def test_02_runtime_contract_after_explicit_compiler_prerequisite(self) -> None:
        # This is deliberately a failure, not a skip, until actual APIs land.
        self.assertEqual(
            self._compile.returncode,
            0,
            "new ownership runtime prerequisite is compiler RED:\n" + self._compile.stderr,
        )

        for pool, expected_reused, expected_stream_creates, expected_stream_destroys in (
            ("model", 1, 1, 0),
            ("selected", 0, 2, 1),
        ):
            with self.subTest(case=f"{pool}-growth"):
                values = self._case(f"{pool}-growth")
                assert_clean_control(self, values, pool, reservation_required=True)
                self.assertEqual(integer(values, "target_result"), 1)
                self.assertEqual(integer(values, "target_stream_reused"), expected_reused)
                self.assertEqual(integer(values, "target_stream_create_calls"), expected_stream_creates)
                self.assertEqual(integer(values, "target_stream_destroy_calls"), expected_stream_destroys)
                assert_pool_shape(
                    self, values, "target", pool, ready=96,
                    raw_live=(1, 1, 1, 1), event_live=(1, 1, 1, 1), stream_live=1,
                    reserved=(96, 96, 96, 96), require_reservation_for_live_raw=True,
                )
                assert_full_ledger(self, values, "target")

        for pool in ("model", "selected"):
            with self.subTest(case=f"{pool}-reuse"):
                values = self._case(f"{pool}-reuse")
                self.assertEqual(integer(values, "control"), 1)
                assert_pool_shape(
                    self, values, "control", pool, ready=96,
                    raw_live=(1, 1, 1, 1), event_live=(1, 1, 1, 1), stream_live=1,
                    reserved=(96, 96, 96, 96), require_reservation_for_live_raw=True,
                )
                assert_full_ledger(self, values, "control")
                self.assertEqual(integer(values, "target_result"), 1)
                self.assertEqual(integer(values, "target_malloc_calls"), 4)
                self.assertEqual(integer(values, "target_event_create_calls"), 4)
                assert_pool_shape(
                    self, values, "target", pool, ready=96,
                    raw_live=(1, 1, 1, 1), event_live=(1, 1, 1, 1), stream_live=1,
                    reserved=(96, 96, 96, 96), require_reservation_for_live_raw=True,
                )
                assert_full_ledger(self, values, "target")

        for name, hosts, events, raw_live, event_live in (
            ("model-malloc-failure", 2, 2, (1, 1, 0, 0), (1, 1, 0, 0)),
            ("model-event-failure", 3, 2, (1, 1, 1, 0), (1, 1, 0, 0)),
        ):
            with self.subTest(case=name):
                self._control("model")
                values = self._case(name)
                self.assertEqual(integer(values, "target_result"), 0)
                self.assertEqual(len(records(values, "target_ledger_hosts")), hosts)
                self.assertEqual(len(records(values, "target_ledger_events")), events)
                assert_pool_shape(
                    self, values, "target", "model", ready=0,
                    raw_live=raw_live, event_live=event_live, stream_live=1,
                    require_reservation_for_live_raw=True,
                )
                assert_full_ledger(self, values, "target")

        with self.subTest(case="selected-clean-rollback"):
            self._control("selected")
            values = self._case("selected-rollback-clean")
            self.assertEqual(integer(values, "target_result"), 0)
            assert_pool_shape(
                self, values, "target", "selected", ready=0,
                raw_live=(0, 0, 0, 0), event_live=(0, 0, 0, 0), stream_live=0,
                reserved=(0, 0, 0, 0), require_reservation_for_live_raw=True,
            )
            assert_full_ledger(self, values, "target")

        rollback_cases = (
            ("selected-rollback-free-failure", (1, 0, 0, 0), (0, 0, 0, 0), (64, 0, 0, 0)),
            ("selected-rollback-event-failure", (1, 1, 0, 0), (1, 0, 0, 0), (64, 64, 0, 0)),
            ("selected-rollback-stream-failure", (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)),
        )
        for name, raw_live, event_live, reserved in rollback_cases:
            with self.subTest(case=name):
                self._control("selected")
                values = self._case(name)
                self.assertEqual(integer(values, "target_result"), 0)
                assert_pool_shape(
                    self, values, "target", "selected", ready=0,
                    raw_live=raw_live, event_live=event_live, stream_live=1,
                    reserved=reserved, require_reservation_for_live_raw=True,
                )
                assert_full_ledger(self, values, "target")
                assert_no_alloc_after_release_failure(self, values)
                self.assertEqual(integer(values, "target_api_errors"), 0)

        for name, pool, raw_live, event_live, stream_live, reserved in (
            ("model-resize-free-failure", "model", (1, 1, 1, 1), (0, 1, 1, 1), 1, (64, 64, 64, 64)),
            ("model-resize-event-failure", "model", (1, 1, 1, 1), (1, 1, 1, 1), 1, (64, 64, 64, 64)),
            ("selected-resize-free-failure", "selected", (1, 1, 1, 1), (0, 1, 1, 1), 1, (64, 64, 64, 64)),
            ("selected-resize-event-failure", "selected", (1, 1, 1, 1), (1, 1, 1, 1), 1, (64, 64, 64, 64)),
            ("selected-resize-stream-failure", "selected", (0, 0, 0, 0), (0, 0, 0, 0), 1, (0, 0, 0, 0)),
        ):
            with self.subTest(case=name):
                self._assert_resize_rejection(
                    name, pool, raw_live=raw_live, event_live=event_live,
                    stream_live=stream_live, reserved=reserved,
                )

        with self.subTest(case="model-retry"):
            values = self._case("model-retry")
            assert_clean_control(self, values, "model", reservation_required=True)
            self.assertEqual(integer(values, "first"), 0)
            assert_pool_shape(
                self, values, "after_first", "model", ready=0,
                raw_live=(1, 1, 1, 1), event_live=(0, 1, 1, 1), stream_live=1,
                reserved=(64, 64, 64, 64), require_reservation_for_live_raw=True,
            )
            assert_full_ledger(self, values, "after_first")
            assert_no_alloc_after_release_failure(self, values, "after_first")
            self.assertEqual(integer(values, "second"), 1)
            self.assertEqual(integer(values, "target_malloc_calls"), 8)
            self.assertEqual(integer(values, "target_free_calls"), 5)
            assert_pool_shape(
                self, values, "target", "model", ready=96,
                raw_live=(1, 1, 1, 1), event_live=(1, 1, 1, 1), stream_live=1,
                reserved=(96, 96, 96, 96), require_reservation_for_live_raw=True,
            )
            assert_full_ledger(self, values, "target")

        with self.subTest(case="selected-retry"):
            self._control("selected")
            values = self._case("selected-retry")
            self.assertEqual(integer(values, "first"), 0)
            assert_pool_shape(
                self, values, "after_first", "selected", ready=0,
                raw_live=(1, 0, 0, 0), event_live=(0, 0, 0, 0), stream_live=1,
                reserved=(64, 0, 0, 0), require_reservation_for_live_raw=True,
            )
            assert_full_ledger(self, values, "after_first")
            assert_no_alloc_after_release_failure(self, values, "after_first")
            self.assertEqual(integer(values, "second"), 1)
            self.assertEqual(integer(values, "target_malloc_calls"), 6)
            self.assertEqual(integer(values, "target_free_calls"), 2)
            assert_pool_shape(
                self, values, "target", "selected", ready=96,
                raw_live=(1, 1, 1, 1), event_live=(1, 1, 1, 1), stream_live=1,
                reserved=(96, 96, 96, 96), require_reservation_for_live_raw=True,
            )
            assert_full_ledger(self, values, "target")

        for name, pool in (("model-release", "model"), ("selected-release", "selected")):
            with self.subTest(case=name):
                values = self._case(name)
                assert_clean_control(self, values, pool, reservation_required=True)
                self.assertEqual(integer(values, "target_result"), 1)
                assert_pool_shape(
                    self, values, "target", pool, ready=0,
                    raw_live=(0, 0, 0, 0), event_live=(0, 0, 0, 0), stream_live=0,
                    reserved=(0, 0, 0, 0), require_reservation_for_live_raw=True,
                )
                assert_full_ledger(self, values, "target")

        for name, pool in (
            ("model-release-stream-failure", "model"),
            ("selected-release-stream-failure", "selected"),
        ):
            with self.subTest(case=name):
                values = self._case(name)
                assert_clean_control(self, values, pool, reservation_required=True)
                self.assertEqual(integer(values, "first"), 0)
                assert_pool_shape(
                    self, values, "after_first", pool, ready=0,
                    raw_live=(0, 0, 0, 0), event_live=(0, 0, 0, 0), stream_live=1,
                    reserved=(0, 0, 0, 0), require_reservation_for_live_raw=True,
                )
                assert_full_ledger(self, values, "after_first")
                assert_no_alloc_after_release_failure(self, values, "after_first")
                self.assertEqual(integer(values, "second"), 1)
                assert_pool_shape(
                    self, values, "target", pool, ready=0,
                    raw_live=(0, 0, 0, 0), event_live=(0, 0, 0, 0), stream_live=0,
                    reserved=(0, 0, 0, 0), require_reservation_for_live_raw=True,
                )
                assert_full_ledger(self, values, "target")

        with self.subTest(case="helper-mixed-success"):
            values = self._case("helper-mixed-success")
            self.assertEqual(integer(values, "control"), 1)
            assert_pool_shape(
                self, values, "control", "helper", ready=0,
                raw_live=(1, 1, 0, 0), event_live=(1, 0, 1, 0), stream_live=0,
                reserved=(11, 22, 0, 0), require_reservation_for_live_raw=True,
                allow_event_only_slots=True,
            )
            assert_full_ledger(self, values, "control")
            self.assertEqual(integer(values, "target_result"), 1)
            assert_pool_shape(
                self, values, "target", "helper", ready=0,
                raw_live=(0, 0, 0, 0), event_live=(0, 0, 0, 0), stream_live=0,
                reserved=(0, 0, 0, 0), require_reservation_for_live_raw=True,
                allow_event_only_slots=True,
            )
            assert_full_ledger(self, values, "target")

        with self.subTest(case="helper-mixed-error"):
            values = self._case("helper-mixed-error")
            self.assertEqual(integer(values, "control"), 1)
            assert_pool_shape(
                self, values, "control", "helper", ready=0,
                raw_live=(1, 1, 0, 0), event_live=(1, 0, 1, 0), stream_live=0,
                reserved=(11, 22, 0, 0), require_reservation_for_live_raw=True,
                allow_event_only_slots=True,
            )
            assert_full_ledger(self, values, "control")
            self.assertEqual(integer(values, "target_result"), 0)
            assert_pool_shape(
                self, values, "target", "helper", ready=0,
                raw_live=(0, 1, 0, 0), event_live=(0, 0, 1, 0), stream_live=0,
                reserved=(0, 22, 0, 0), require_reservation_for_live_raw=True,
                allow_event_only_slots=True,
            )
            assert_full_ledger(self, values, "target")
            assert_no_alloc_after_release_failure(self, values)

        with self.subTest(case="unknown-resource"):
            values = self._case("unknown-resource")
            self.assertEqual(integer(values, "target_result"), 0)
            self.assertEqual(integer(values, "target_api_errors"), 1)
            self.assertEqual(integer(values, "target_event_destroy_calls"), 1)
            self.assertEqual(integer(values, "target_free_calls"), 0)
            self.assertEqual(integer(values, "target_helper_raw_0"), 0x12345)
            self.assertEqual(integer(values, "target_helper_stage_0"), 0x12345)
            self.assertEqual(integer(values, "target_helper_event_0"), 0x54321)
            self.assertEqual(integer(values, "target_helper_reserved_0"), 7)
            self.assertEqual(records(values, "target_ledger_hosts"), [])
            self.assertEqual(records(values, "target_ledger_events"), [])
            self.assertTrue(records(values, "target_owners_hosts")[0].startswith("dead@"))
            self.assertTrue(records(values, "target_owners_events")[0].startswith("dead@"))

        with self.subTest(case="both-independent"):
            values = self._case("both-independent")
            self.assertEqual(integer(values, "control"), 1)
            assert_pool_shape(
                self, values, "control", "model", ready=64,
                raw_live=(1, 1, 1, 1), event_live=(1, 1, 1, 1), stream_live=1,
                reserved=(64, 64, 64, 64), require_reservation_for_live_raw=True,
            )
            assert_pool_shape(
                self, values, "control", "selected", ready=96,
                raw_live=(1, 1, 1, 1), event_live=(1, 1, 1, 1), stream_live=1,
                reserved=(96, 96, 96, 96), require_reservation_for_live_raw=True,
            )
            assert_full_ledger(self, values, "control")
            self.assertEqual(integer(values, "after_model_result"), 1)
            assert_pool_shape(
                self, values, "after_model", "model", ready=0,
                raw_live=(0, 0, 0, 0), event_live=(0, 0, 0, 0), stream_live=0,
                reserved=(0, 0, 0, 0), require_reservation_for_live_raw=True,
            )
            assert_pool_shape(
                self, values, "after_model", "selected", ready=96,
                raw_live=(1, 1, 1, 1), event_live=(1, 1, 1, 1), stream_live=1,
                reserved=(96, 96, 96, 96), require_reservation_for_live_raw=True,
            )
            assert_full_ledger(self, values, "after_model")
            self.assertEqual(integer(values, "target_result"), 1)
            for pool in ("model", "selected"):
                assert_pool_shape(
                    self, values, "target", pool, ready=0,
                    raw_live=(0, 0, 0, 0), event_live=(0, 0, 0, 0), stream_live=0,
                    reserved=(0, 0, 0, 0), require_reservation_for_live_raw=True,
                )
            assert_full_ledger(self, values, "target")

        for pool in ("model", "selected"):
            with self.subTest(case=f"{pool}-zero"):
                values = self._case(f"{pool}-zero")
                assert_rejected_from_ready(
                    self, values, pool, reservation_required=True
                )

        for name, pool in (
            ("model-max-malloc-failure", "model"),
            ("selected-max-malloc-failure", "selected"),
        ):
            with self.subTest(case=name):
                values = self._case(name)
                self.assertEqual(integer(values, "target_result"), 0)
                representable = integer(values, "target_size_max_is_uint64")
                expected_calls = 1 if representable else 0
                self.assertEqual(integer(values, "target_malloc_calls"), expected_calls)
                self.assertEqual(integer(values, "target_oversize_attempts"), 0)
                active_stream = 1 if representable and pool == "model" else 0
                assert_pool_shape(
                    self, values, "target", pool, ready=0,
                    raw_live=(0, 0, 0, 0), event_live=(0, 0, 0, 0),
                    stream_live=active_stream, reserved=(0, 0, 0, 0),
                    require_reservation_for_live_raw=True,
                )
                other = "selected" if pool == "model" else "model"
                assert_pool_shape(
                    self, values, "target", other, ready=0,
                    raw_live=(0, 0, 0, 0), event_live=(0, 0, 0, 0),
                    stream_live=0, reserved=(0, 0, 0, 0),
                    require_reservation_for_live_raw=True,
                )
                assert_full_ledger(self, values, "target")

        with self.subTest(case="simulated-narrow-size-max-empty"):
            values = self._narrow_case("simulated-narrow-size-max-empty")
            self.assertEqual(integer(values, "target_simulated_size_max"), 127)
            assert_empty_snapshot(self, values, "control")
            for pool in ("model", "selected"):
                with self.subTest(pool=pool, field="result"):
                    self.assertEqual(integer(values, f"target_{pool}_result"), 0)
            assert_snapshot_unchanged(self, values)
            assert_empty_snapshot(self, values, "target")

        with self.subTest(case="simulated-narrow-size-max"):
            values = self._narrow_case("simulated-narrow-size-max")
            self.assertEqual(integer(values, "target_simulated_size_max"), 127)
            self.assertEqual(integer(values, "target_model_result"), 0)
            self.assertEqual(integer(values, "target_selected_result"), 0)
            self.assertEqual(integer(values, "control"), 1)
            assert_snapshot_unchanged(self, values)
            for pool in ("model", "selected"):
                assert_pool_shape(
                    self, values, "target", pool, ready=64,
                    raw_live=(1, 1, 1, 1), event_live=(1, 1, 1, 1), stream_live=1,
                    reserved=(64, 64, 64, 64), require_reservation_for_live_raw=True,
                )


class FixtureAssertionMutationControlTest(unittest.TestCase):
    """Small oracle-only mutations prove the fixture assertions are live."""

    @staticmethod
    def _snapshot(label: str) -> dict[str, Scalar]:
        values: dict[str, Scalar] = {
            f"{label}_ledger_hosts": "1:64",
            f"{label}_owners_hosts": "1:64",
            f"{label}_ledger_events": "1:1",
            f"{label}_owners_events": "1:1",
            f"{label}_ledger_streams": "1",
            f"{label}_owners_streams": "1",
            f"{label}_malloc_calls": 1,
            f"{label}_event_create_calls": 1,
            f"{label}_free_calls": 0,
            f"{label}_event_destroy_calls": 0,
            f"{label}_stream_create_calls": 1,
            f"{label}_stream_destroy_calls": 0,
            f"{label}_api_errors": 0,
            f"{label}_oversize_attempts": 0,
            f"{label}_release_failure_seq": 0,
            f"{label}_malloc_after_release_failure": 0,
            f"{label}_order_violations": 0,
            f"{label}_order_ok": 1,
            f"{label}_trace": "S1,M1,E1H1",
        }
        for pool in ("model", "selected"):
            model = pool == "model"
            values[f"{label}_{pool}_ready"] = 0  # Valid partial construction, not a ready pool.
            values[f"{label}_{pool}_stream"] = 200 if model else 0
            values[f"{label}_{pool}_stream_live"] = 1 if model else 0
            for i in range(4):
                live = model and i == 0
                values[f"{label}_{pool}_raw_{i}"] = 100 if live else 0
                values[f"{label}_{pool}_raw_live_{i}"] = int(live)
                values[f"{label}_{pool}_raw_owner_id_{i}"] = 1 if live else 0
                values[f"{label}_{pool}_raw_requested_{i}"] = 64 if live else 0
                values[f"{label}_{pool}_stage_{i}"] = 100 if live else 0
                values[f"{label}_{pool}_event_{i}"] = 300 if live else 0
                values[f"{label}_{pool}_event_live_{i}"] = int(live)
                values[f"{label}_{pool}_event_owner_id_{i}"] = 1 if live else 0
                values[f"{label}_{pool}_event_paired_host_{i}"] = 1 if live else 0
                values[f"{label}_{pool}_reserved_{i}"] = 64 if live else 0
        return values

    def _probe(self) -> unittest.TestCase:
        # It has no active unittest outcome, so subTest cannot swallow an error.
        return unittest.TestCase()

    def test_00_assertion_mutations_are_rejected(self) -> None:
        with self.subTest(check="wrong-order-bit"):
            values = self._snapshot("target")
            assert_full_ledger(self._probe(), values, "target")
            values["target_order_ok"] = 0
            with self.assertRaises(AssertionError):
                assert_full_ledger(self._probe(), values, "target")

        for operation in ("E2H1", "S2", "D2", "F2", "X2"):
            with self.subTest(check=f"post-release-{operation[0]}"):
                values = self._snapshot("target")
                values["target_release_failure_seq"] = 2
                values["target_trace"] = "S1,F!1"
                # Oracle-only trace input; no production helper is simulated.
                assert_no_alloc_after_release_failure(self._probe(), values)
                values["target_trace"] += "," + operation
                with self.assertRaises(AssertionError):
                    assert_no_alloc_after_release_failure(self._probe(), values)

        with self.subTest(check="wrong-event-pair"):
            values = self._snapshot("target")
            assert_pool_shape(
                self._probe(), values, "target", "model", ready=0,
                raw_live=(1, 0, 0, 0), event_live=(1, 0, 0, 0), stream_live=1,
                reserved=(64, 0, 0, 0), require_reservation_for_live_raw=True,
            )
            values["target_model_event_paired_host_0"] = 2
            with self.assertRaises(AssertionError):
                assert_pool_shape(
                    self._probe(), values, "target", "model", ready=0,
                    raw_live=(1, 0, 0, 0), event_live=(1, 0, 0, 0), stream_live=1,
                    reserved=(64, 0, 0, 0), require_reservation_for_live_raw=True,
                )

        with self.subTest(check="wrong-requested-reservation"):
            values = self._snapshot("target")
            assert_pool_shape(
                self._probe(), values, "target", "model", ready=0,
                raw_live=(1, 0, 0, 0), event_live=(1, 0, 0, 0), stream_live=1,
                reserved=(64, 0, 0, 0), require_reservation_for_live_raw=True,
            )
            values["target_model_reserved_0"] = 63
            with self.assertRaises(AssertionError):
                assert_pool_shape(
                    self._probe(), values, "target", "model", ready=0,
                    raw_live=(1, 0, 0, 0), event_live=(1, 0, 0, 0), stream_live=1,
                    reserved=(63, 0, 0, 0), require_reservation_for_live_raw=True,
                )

        for corrupt in ("target_model_ready", "target_model_reserved_0"):
            with self.subTest(check=f"no-io-{corrupt}"):
                values = self._snapshot("control")
                values.update({key.replace("control_", "target_", 1): value
                               for key, value in self._snapshot("control").items()})
                assert_snapshot_unchanged(self._probe(), values)
                values[corrupt] = integer(values, corrupt) + 1
                with self.assertRaises(AssertionError):
                    assert_snapshot_unchanged(self._probe(), values)


class SourceWiringContractTest(unittest.TestCase):
    """Source-only checks make shims unable to hide missing production wiring."""

    def _body(self, body: str | None, message: str) -> str | None:
        self.assertIsNotNone(body, message)
        return code_only(body) if body is not None else None

    def test_00_reservation_arrays_are_actual_private_source_declarations(self) -> None:
        source = code_only(CUDA_SOURCE)
        for name in (
            "g_model_stage_reserved_bytes",
            "g_stream_selected_stage_reserved_bytes",
        ):
            with self.subTest(name=name):
                self.assertEqual(
                    len(re.findall(r"static\s+uint64_t\s+" + name + r"\s*\[4\]\s*;", source)),
                    1,
                    f"missing actual source declaration: {name}",
                )

    def test_01_shared_release_is_four_slot_checked_and_has_no_allocator(self) -> None:
        body = self._body(SLOTS_RELEASE, "missing actual cuda_stage_slots_release definition")
        if body is None:
            return
        self.assertRegex(body, r"for\s*\(\s*size_t\s+i\s*=\s*0\s*;\s*i\s*<\s*4")
        event_destroy = body.find("cudaEventDestroy")
        physical_free = body.find("cuda_laguna_resident_free_host")
        self.assertGreaterEqual(event_destroy, 0)
        self.assertGreaterEqual(physical_free, 0)
        self.assertLess(event_destroy, physical_free)
        self.assertNotRegex(body, r"\b(?:cudaMallocHost|cuda_laguna_resident_malloc_host|cudaStreamCreate|malloc|calloc|realloc|new)\b")
        self.assertRegex(body, r"(?:return\s+0|return\s+false)")

    def test_02_pools_preflight_zero_and_size_max_before_reuse_or_io(self) -> None:
        for body, ready, label, allocator in (
            (MODEL_POOL, "g_model_stage_bytes", "model",
             "cuda_laguna_resident_malloc_host"),
            (SELECTED_POOL, "g_stream_selected_stage_bytes", "selected",
             "cudaMallocHost"),
        ):
            with self.subTest(pool=label):
                code = self._body(body, f"missing actual {label} pool definition")
                if code is None:
                    continue
                zero = re.search(r"bytes\s*==\s*0", code)
                size = re.search(r"bytes\s*>\s*(?:\(\s*uint64_t\s*\)\s*)?SIZE_MAX", code)
                reuse = code.find(ready + " >= bytes")
                malloc = code.find(allocator)
                self.assertIsNotNone(zero)
                self.assertIsNotNone(size)
                self.assertGreaterEqual(reuse, 0)
                self.assertGreaterEqual(malloc, 0)
                if zero is not None:
                    self.assertLess(zero.start(), reuse)
                    self.assertLess(zero.start(), malloc)
                if size is not None:
                    self.assertLess(size.start(), reuse)
                    self.assertLess(size.start(), malloc)

    def test_03_successful_nonnull_malloc_publishes_bytes_before_event_create(self) -> None:
        pools = (
            (MODEL_POOL, "g_model_stage_raw", "g_model_stage_reserved_bytes",
             "g_model_stage_event", "cuda_laguna_resident_malloc_host"),
            (SELECTED_POOL, "g_stream_selected_stage_raw",
             "g_stream_selected_stage_reserved_bytes",
             "g_stream_selected_stage_event", "cudaMallocHost"),
        )
        for body, raw, reserved, event, allocator in pools:
            with self.subTest(raw=raw):
                code = self._body(body, f"missing actual pool for {raw}")
                if code is None:
                    continue
                malloc = code.find(allocator)
                publish = code.find(reserved + "[i] = bytes")
                create = code.find("cudaEventCreateWithFlags")
                self.assertGreaterEqual(malloc, 0)
                self.assertGreater(publish, malloc)
                self.assertGreater(create, publish)
                self.assertIn(raw + "[i]", code)
                self.assertIn(event + "[i]", code)
                self.assertRegex(code[malloc:publish], re.escape(raw) + r"\s*\[\s*i\s*\]")

    def test_04_model_resize_uses_shared_slots_without_destroying_reusable_stream(self) -> None:
        code = self._body(MODEL_POOL, "missing actual model pool definition")
        if code is None:
            return
        self.assertIn("cuda_stage_slots_release", code)
        self.assertNotIn("cudaStreamDestroy(g_model_upload_stream)", code)

    def test_05_selected_resize_checks_release_before_stream_or_pinned_allocation(self) -> None:
        code = self._body(SELECTED_POOL, "missing actual selected pool definition")
        if code is None:
            return
        release = code.find("cuda_stream_selected_stage_release")
        stream = code.find("cudaStreamCreateWithFlags")
        malloc = code.find("cudaMallocHost")
        self.assertGreaterEqual(release, 0)
        self.assertGreater(stream, release)
        self.assertGreater(malloc, stream)
        self.assertRegex(
            code[release:stream],
            r"(?:if\s*\(\s*!|return\s+0|return\s+false)",
        )

    def test_06_release_wrappers_are_checked_and_cleanup_uses_model_wrapper(self) -> None:
        model = self._body(MODEL_RELEASE, "missing actual model release wrapper")
        selected = self._body(SELECTED_RELEASE, "missing checked selected release wrapper")
        if model is None or selected is None:
            return
        self.assertIn("cuda_stage_slots_release", model)
        self.assertIn("cudaStreamDestroy", model)
        self.assertIn("cuda_stage_slots_release", selected)
        self.assertIn("cudaStreamDestroy", selected)
        cleanup = extract_definition(
            CUDA_SOURCE, 'extern "C" int ds4_gpu_cleanup_checked(void)'
        )
        cleanup_code = self._body(
            cleanup, "missing actual ds4_gpu_cleanup_checked definition"
        )
        if cleanup_code is None:
            return
        call = cleanup_code.find("cuda_model_stage_release(")
        guard = cleanup_code.find("compact_cleanup_required")
        self.assertEqual(cleanup_code.count("cuda_model_stage_release("), 1)
        self.assertGreater(call, guard)
        self.assertNotRegex(
            cleanup_code, r"g_model_stage_(?:raw|event)\s*\[[^\]]+\]\s*=(?!=)"
        )
        self.assertNotRegex(
            cleanup_code,
            r"\b(?:cudaFreeHost|cudaEventDestroy)\s*\([^;]*g_model_stage_(?:raw|event)\s*\[",
        )

        legacy = extract_definition(
            CUDA_SOURCE, 'extern "C" void ds4_gpu_cleanup(void)'
        )
        legacy_code = self._body(legacy, "missing actual ds4_gpu_cleanup definition")
        if legacy_code is None:
            return
        self.assertRegex(
            legacy_code,
            r"\(\s*void\s*\)\s*ds4_gpu_cleanup_checked\s*\(\s*\)\s*;",
        )
        self.assertNotIn("cuda_model_stage_release(", legacy_code)
        self.assertEqual(
            len(re.findall(r"\bds4_gpu_cleanup_checked\s*\(\s*\)", legacy_code)),
            1,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
