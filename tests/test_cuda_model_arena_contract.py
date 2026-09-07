#!/usr/bin/env python3
"""Host-only TDD contract for the model-weight arena geometry.

The C++ fixtures compile the two production arena bodies extracted from
``ds4_cuda.cu``.  CUDA reservations are host simulations: a reported arena
may be 1792 MiB while its owned backing buffer is only 4096 bytes.  No test
uses a fabricated pointer or dereferences outside an owned backing buffer.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CUDA_SOURCE = (ROOT / "ds4_cuda.cu").read_text(encoding="utf-8")
SAFE_ENV = {
    "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
    "LANG": "C",
    "LC_ALL": "C",
}


def extract_definition(source: str, signature: str) -> str | None:
    """Extract one C/C++ definition while ignoring braces in literals/comments."""
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


# This helper is an existing production dependency of the allocator.  It is
# extracted too, rather than replaced with a test algorithm or a stub.
CACHE_LIMIT = extract_definition(CUDA_SOURCE, "static uint64_t cuda_model_cache_limit_bytes(")
CHUNK = extract_definition(CUDA_SOURCE, "static uint64_t cuda_model_arena_chunk_bytes(")
ALLOC = extract_definition(CUDA_SOURCE, "static char *cuda_model_arena_alloc(")

FIXTURE_PREFIX = r"""
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <vector>
#include <csignal>
#include <sys/resource.h>
#include <unistd.h>
#ifndef SIZE_MAX
#define SIZE_MAX UINT64_MAX
#endif

enum cudaError_t {
    cudaSuccess = 0,
    cudaErrorMemoryAllocation = 2,
};

struct cuda_model_arena {
    char *device_ptr;
    uint64_t bytes;
    uint64_t used;
};

struct fake_owner {
    uint64_t id;
    unsigned char *backing;
    uint64_t requested_bytes;
    uint64_t reserved_bytes;
    int source; /* 0 = seeded owner, 1 = fake cudaMalloc owner. */
};

struct fake_attempt {
    uint64_t request;
    int result; /* 0 = error, 1 = success-with-null, 2 = owner. */
    uint64_t owner_id;
};

static std::vector<cuda_model_arena> g_model_arenas;
static std::vector<uint64_t> g_arena_owner_ids;
static uint64_t g_model_range_bytes;
static int g_model_cache_full;

/* 0: virtual success with a tiny owned buffer; 1: ordinary failure;
 * 2: success reported with a null output pointer. */
static int g_fake_cuda_mode;
static size_t g_cuda_malloc_calls;
static uint64_t g_last_cuda_request;
static const uint64_t kBackingBytes = 4096;
static std::vector<fake_owner> g_fake_owners;
static std::vector<fake_attempt> g_fake_attempts;
static uint64_t g_next_owner_id = 1;
static char *g_last_control;

static const char *cudaGetErrorString(cudaError_t error) {
    return error == cudaSuccess ? "fake success" : "fake allocation failure";
}
static cudaError_t cudaGetLastError(void) { return cudaSuccess; }

static char *register_owner(uint64_t requested_bytes,
                            uint64_t reserved_bytes,
                            int source,
                            uint64_t *owner_id) {
    unsigned char *storage = new (std::nothrow) unsigned char[kBackingBytes];
    if (!storage) {
        if (owner_id) *owner_id = 0;
        return nullptr;
    }
    const uint64_t id = g_next_owner_id++;
    g_fake_owners.push_back({id, storage, requested_bytes, reserved_bytes, source});
    if (owner_id) *owner_id = id;
    return reinterpret_cast<char *>(storage);
}

static cudaError_t cudaMalloc(void **out, size_t request) {
    ++g_cuda_malloc_calls;
    g_last_cuda_request = static_cast<uint64_t>(request);
    if (out) *out = nullptr;
    if (g_fake_cuda_mode == 1) {
        g_fake_attempts.push_back({static_cast<uint64_t>(request), 0, 0});
        return cudaErrorMemoryAllocation;
    }
    if (g_fake_cuda_mode == 2) {
        g_fake_attempts.push_back({static_cast<uint64_t>(request), 1, 0});
        return cudaSuccess;
    }
    uint64_t owner_id = 0;
    char *storage = register_owner(static_cast<uint64_t>(request),
                                   static_cast<uint64_t>(request), 1, &owner_id);
    if (!storage) {
        g_fake_attempts.push_back({static_cast<uint64_t>(request), 0, 0});
        return cudaErrorMemoryAllocation;
    }
    g_fake_attempts.push_back({static_cast<uint64_t>(request), 2, owner_id});
    if (out) *out = storage;
    return cudaSuccess;
}

/* The extracted body only needs the permit's public interface.  The fixture
 * deliberately keeps the permit allowed and does not model CUDA lifecycle. */
class cuda_laguna_compact_legacy_permit {
public:
    cuda_laguna_compact_legacy_permit() = default;
    bool allowed(void) const { return true; }
};

static char *new_seed_backing(uint64_t virtual_bytes, uint64_t *owner_id) {
    return register_owner(virtual_bytes, virtual_bytes, 0, owner_id);
}

static void seed_arena(uint64_t bytes, uint64_t used) {
    uint64_t owner_id = 0;
    char *base = new_seed_backing(bytes, &owner_id);
    g_model_arenas.push_back({base, bytes, used});
    g_arena_owner_ids.push_back(owner_id);
}

static uint64_t owner_for_pointer(const char *value) {
    if (!value) return 0;
    const uintptr_t address = reinterpret_cast<uintptr_t>(value);
    for (const fake_owner &owner : g_fake_owners) {
        const uintptr_t base = reinterpret_cast<uintptr_t>(owner.backing);
        if (address >= base && address - base < kBackingBytes)
            return owner.id;
    }
    return 0;
}

static long long pointer_offset(const char *value) {
    if (!value) return -1;
    const uintptr_t address = reinterpret_cast<uintptr_t>(value);
    for (const fake_owner &owner : g_fake_owners) {
        const uintptr_t base = reinterpret_cast<uintptr_t>(owner.backing);
        if (address >= base && address - base < kBackingBytes)
            return static_cast<long long>(address - base);
    }
    return -1;
}

static void reconcile_arena_owners(void) {
    while (g_arena_owner_ids.size() < g_model_arenas.size()) {
        const size_t i = g_arena_owner_ids.size();
        g_arena_owner_ids.push_back(owner_for_pointer(g_model_arenas[i].device_ptr));
    }
}

static void emit_snapshot(const char *prefix, const char *result) {
    reconcile_arena_owners();
    std::printf("%sresult=%d\n", prefix, result != nullptr);
    std::printf("%sresult_offset=%lld\n", prefix, pointer_offset(result));
    std::printf("%sresult_owner=%llu\n", prefix,
                (unsigned long long)owner_for_pointer(result));
    std::printf("%scuda_calls=%zu\n", prefix, g_cuda_malloc_calls);
    std::printf("%slast_request=%llu\n", prefix,
                (unsigned long long)g_last_cuda_request);
    std::printf("%scache_full=%d\n", prefix, g_model_cache_full);
    std::printf("%srange_bytes=%llu\n", prefix,
                (unsigned long long)g_model_range_bytes);
    std::printf("%sarena_count=%zu\n", prefix, g_model_arenas.size());
    for (size_t i = 0; i < g_model_arenas.size(); ++i) {
        const cuda_model_arena &a = g_model_arenas[i];
        std::printf("%sarena_%zu_base=%d\n", prefix, i, a.device_ptr != nullptr);
        std::printf("%sarena_%zu_ptr=%llu\n", prefix, i,
                    (unsigned long long)reinterpret_cast<uintptr_t>(a.device_ptr));
        std::printf("%sarena_%zu_owner=%llu\n", prefix, i,
                    (unsigned long long)owner_for_pointer(a.device_ptr));
        std::printf("%sarena_%zu_offset=%lld\n", prefix, i,
                    pointer_offset(a.device_ptr));
        std::printf("%sarena_%zu_cached_owner=%llu\n", prefix, i,
                    (unsigned long long)g_arena_owner_ids[i]);
        std::printf("%sarena_%zu_bytes=%llu\n", prefix, i,
                    (unsigned long long)a.bytes);
        std::printf("%sarena_%zu_used=%llu\n", prefix, i,
                    (unsigned long long)a.used);
    }
    std::printf("%sowner_count=%zu\n", prefix, g_fake_owners.size());
    for (size_t i = 0; i < g_fake_owners.size(); ++i) {
        const fake_owner &owner = g_fake_owners[i];
        std::printf("%sowner_%zu_id=%llu\n", prefix, i,
                    (unsigned long long)owner.id);
        std::printf("%sowner_%zu_backing=%llu\n", prefix, i,
                    (unsigned long long)reinterpret_cast<uintptr_t>(owner.backing));
        std::printf("%sowner_%zu_requested=%llu\n", prefix, i,
                    (unsigned long long)owner.requested_bytes);
        std::printf("%sowner_%zu_reserved=%llu\n", prefix, i,
                    (unsigned long long)owner.reserved_bytes);
        std::printf("%sowner_%zu_source=%d\n", prefix, i, owner.source);
    }
    std::printf("%sattempt_count=%zu\n", prefix, g_fake_attempts.size());
    for (size_t i = 0; i < g_fake_attempts.size(); ++i) {
        const fake_attempt &attempt = g_fake_attempts[i];
        std::printf("%sattempt_%zu_request=%llu\n", prefix, i,
                    (unsigned long long)attempt.request);
        std::printf("%sattempt_%zu_result=%d\n", prefix, i, attempt.result);
        std::printf("%sattempt_%zu_owner=%llu\n", prefix, i,
                    (unsigned long long)attempt.owner_id);
    }
}

static void emit_state(const char *result) { emit_snapshot("", result); }

static char *cuda_model_arena_alloc(uint64_t bytes, const char *what);

static int valid_empty_metadata(uint64_t expected_extent, const char *what) {
    seed_arena(expected_extent, 0);
    reconcile_arena_owners();
    char *control = g_model_arenas[0].device_ptr;
    const uint64_t owner_id = g_arena_owner_ids[0];
    const bool owner_matches = owner_id != 0 && owner_for_pointer(control) == owner_id;
    const bool reservation_matches = g_fake_owners[0].reserved_bytes == expected_extent &&
                                     g_fake_owners[0].requested_bytes == expected_extent;
    const bool valid = control != nullptr && pointer_offset(control) == 0 &&
                       g_model_arenas[0].used == 0 &&
                       g_model_arenas[0].bytes == expected_extent &&
                       owner_matches && reservation_matches;
    g_last_control = control;
    std::printf("control_ok=%d\n", valid ? 1 : 0);
    return valid ? 1 : 0;
}

static int valid_control(uint64_t expected_extent, const char *what) {
    seed_arena(expected_extent, 0);
    char *control = cuda_model_arena_alloc(1, what);
    reconcile_arena_owners();
    const uint64_t owner_id = g_arena_owner_ids[0];
    const bool owner_matches = owner_id != 0 && owner_for_pointer(control) == owner_id;
    const bool reservation_matches = g_fake_owners[0].reserved_bytes == expected_extent &&
                                     g_fake_owners[0].requested_bytes == expected_extent;
    const bool valid = control != nullptr && pointer_offset(control) == 0 &&
                       g_model_arenas[0].used == 256 &&
                       g_model_arenas[0].bytes == expected_extent &&
                       owner_matches && reservation_matches;
    g_last_control = control;
    std::printf("control_ok=%d\n", valid ? 1 : 0);
    return valid ? 1 : 0;
}

static void configure_process(void) {
    alarm(15);
    struct rlimit limit = {0, 0};
    (void)setrlimit(RLIMIT_CORE, &limit);
}

static void cleanup_backings(void) {
    for (const fake_owner &owner : g_fake_owners) delete[] owner.backing;
    g_fake_owners.clear();
}
"""

FIXTURE_DRIVER = r"""
static int run_case(const char *name) {
    const uint64_t MiB = 1048576ull;
    const uint64_t chunk_align = 256ull * MiB;

    if (std::strcmp(name, "helper-zero") == 0) {
        std::printf("result=%llu\n", (unsigned long long)cuda_model_arena_chunk_bytes(0));
        return 0;
    }
    if (std::strcmp(name, "helper-default") == 0) {
        std::printf("result=%llu\n", (unsigned long long)cuda_model_arena_chunk_bytes(1));
        return 0;
    }
    if (std::strcmp(name, "helper-min") == 0) {
        std::printf("result=%llu\n", (unsigned long long)cuda_model_arena_chunk_bytes(1));
        return 0;
    }
    if (std::strcmp(name, "helper-max") == 0) {
        std::printf("result=%llu\n", (unsigned long long)cuda_model_arena_chunk_bytes(1));
        return 0;
    }
    if (std::strcmp(name, "helper-round") == 0) {
        std::printf("result=%llu\n", (unsigned long long)cuda_model_arena_chunk_bytes(chunk_align + 1));
        return 0;
    }
    if (std::strcmp(name, "helper-round-exact") == 0) {
        std::printf("result=%llu\n", (unsigned long long)cuda_model_arena_chunk_bytes(chunk_align));
        return 0;
    }
    if (std::strcmp(name, "helper-highest") == 0) {
        const uint64_t highest = UINT64_MAX - (UINT64_MAX % chunk_align);
        std::printf("result=%llu\n", (unsigned long long)cuda_model_arena_chunk_bytes(highest));
        return 0;
    }
    if (std::strcmp(name, "helper-next") == 0) {
        const uint64_t highest = UINT64_MAX - (UINT64_MAX % chunk_align);
        std::printf("result=%llu\n", (unsigned long long)cuda_model_arena_chunk_bytes(highest + 1));
        return 0;
    }
    if (std::strcmp(name, "narrow-helper-above") == 0) {
        std::printf("result=%llu\n", (unsigned long long)cuda_model_arena_chunk_bytes(UINT64_C(4294967296)));
        return 0;
    }
    if (std::strcmp(name, "narrow-helper-oversized-env") == 0) {
        std::printf("result=%llu\n", (unsigned long long)cuda_model_arena_chunk_bytes(1));
        return 0;
    }
    if (std::strcmp(name, "narrow-helper-boundary") == 0) {
        const uint64_t highest = UINT32_MAX - (UINT32_MAX % chunk_align);
        std::printf("result=%llu\n", (unsigned long long)cuda_model_arena_chunk_bytes(highest));
        return 0;
    }
    if (std::strcmp(name, "narrow-helper-next") == 0) {
        const uint64_t highest = UINT32_MAX - (UINT32_MAX % chunk_align);
        std::printf("result=%llu\n", (unsigned long long)cuda_model_arena_chunk_bytes(highest + 1));
        return 0;
    }

    if (std::strcmp(name, "small-reuse") == 0) {
        char *first = cuda_model_arena_alloc(1, "first");
        char *second = cuda_model_arena_alloc(257, "second");
        emit_state(second);
        std::printf("first_offset=%lld\n", pointer_offset(first));
        std::printf("second_offset=%lld\n", pointer_offset(second));
        return 0;
    }
    if (std::strcmp(name, "no-fit-transition") == 0) {
        (void)valid_control(512, "transition-control");
        emit_snapshot("before_", g_last_control);
        g_fake_cuda_mode = 0;
        char *result = cuda_model_arena_alloc(257, "transition");
        emit_state(result);
        return 0;
    }
    if (std::strcmp(name, "budget-exact") == 0) {
        g_model_range_bytes = (1ull << 30) - 256;
        char *result = cuda_model_arena_alloc(1, "budget-exact");
        emit_state(result);
        return 0;
    }
    if (std::strcmp(name, "budget-over") == 0) {
        g_model_range_bytes = (1ull << 30);
        char *result = cuda_model_arena_alloc(1, "budget-over");
        emit_state(result);
        return 0;
    }
    if (std::strcmp(name, "cache-full-guard") == 0) {
        g_model_range_bytes = 12345;
        g_model_cache_full = 1;
        char *result = cuda_model_arena_alloc(1, "cache-full");
        emit_state(result);
        return 0;
    }
    if (std::strcmp(name, "zero-request") == 0) {
        g_model_range_bytes = 67890;
        g_fake_cuda_mode = 1;
        char *result = cuda_model_arena_alloc(0, "zero-request");
        emit_state(result);
        return 0;
    }

    if (std::strcmp(name, "request-overflow-empty") == 0) {
        g_model_range_bytes = 12345;
        const int control_ok = cuda_model_arena_chunk_bytes(1) != 0 ? 1 : 0;
        std::printf("control_ok=%d\n", control_ok);
        emit_snapshot("before_", nullptr);
        g_fake_cuda_mode = 1;
        char *result = cuda_model_arena_alloc(UINT64_MAX, "request-overflow");
        emit_state(result);
        return 0;
    }
    if (std::strcmp(name, "request-chunk-rounding-overflow") == 0) {
        // The 256-byte claim fits uint64, but rounding the slab to 256 MiB does not.
        // Zero logical bytes keeps the unrelated budget guard from rejecting first.
        g_model_range_bytes = 0;
        std::printf("control_ok=%d\n",
                    cuda_model_arena_chunk_bytes(1) == 1792ull * MiB);
        emit_snapshot("before_", nullptr);
        g_fake_cuda_mode = 1;
        char *result = cuda_model_arena_alloc(UINT64_MAX - 255u, "chunk-overflow");
        emit_state(result);
        return 0;
    }
    if (std::strcmp(name, "request-overflow-existing") == 0) {
        g_model_range_bytes = 23456;
        (void)valid_control(4096, "request-overflow-control");
        emit_snapshot("before_", g_last_control);
        g_fake_cuda_mode = 1;
        char *result = cuda_model_arena_alloc(UINT64_MAX, "request-overflow-existing");
        emit_state(result);
        return 0;
    }
    if (std::strcmp(name, "narrow-request-overflow") == 0) {
        g_model_range_bytes = 23456;
        const int control_ok = cuda_model_arena_chunk_bytes(1) != 0 ? 1 : 0;
        std::printf("control_ok=%d\n", control_ok);
        emit_snapshot("before_", nullptr);
        g_fake_cuda_mode = 1;
        char *result = cuda_model_arena_alloc(UINT32_MAX, "narrow-request-overflow");
        emit_state(result);
        return 0;
    }
    if (std::strcmp(name, "narrow-request-above") == 0) {
        g_model_range_bytes = 34567;
        const int control_ok = cuda_model_arena_chunk_bytes(1) != 0 ? 1 : 0;
        std::printf("control_ok=%d\n", control_ok);
        emit_snapshot("before_", nullptr);
        g_fake_cuda_mode = 1;
        char *result = cuda_model_arena_alloc(UINT64_C(4294967296), "narrow-request-above");
        emit_state(result);
        return 0;
    }
    if (std::strcmp(name, "narrow-env-oversized-alloc") == 0) {
        g_model_range_bytes = 45678;
        (void)setenv("DS4_CUDA_WEIGHT_ARENA_CHUNK_MB", "256", 1);
        const uint64_t control_result = cuda_model_arena_chunk_bytes(1);
        const int control_ok = control_result == chunk_align ? 1 : 0;
        std::printf("control_result=%llu\n", (unsigned long long)control_result);
        std::printf("control_ok=%d\n", control_ok);
        (void)setenv("DS4_CUDA_WEIGHT_ARENA_CHUNK_MB", "8192", 1);
        emit_snapshot("before_", nullptr);
        g_fake_cuda_mode = 1;
        char *result = cuda_model_arena_alloc(1, "narrow-oversized-chunk");
        emit_state(result);
        return 0;
    }
    if (std::strcmp(name, "narrow-existing-extent") == 0) {
        g_model_range_bytes = 34567;
        (void)valid_control(4096, "narrow-existing-control");
        g_model_arenas[0].bytes = UINT64_C(4294967296);
        emit_snapshot("before_", g_last_control);
        g_fake_cuda_mode = 1;
        char *result = cuda_model_arena_alloc(1, "narrow-existing-extent");
        emit_state(result);
        return 0;
    }

    if (std::strcmp(name, "narrow-used-rounding") == 0) {
        // Numeric SIZE_MAX boundary only; normal control accesses offset0.
        g_model_range_bytes = 34567;
        (void)valid_control(UINT32_MAX, "narrow-used-control");
        g_model_arenas[0].used = UINT32_MAX;
        emit_snapshot("before_", g_last_control);
        g_fake_cuda_mode = 1;
        char *result = cuda_model_arena_alloc(1, "narrow-used-rounding");
        emit_state(result);
        return 0;
    }
    if (std::strcmp(name, "used-rounding-overflow") == 0) {
        g_model_range_bytes = 45678;
        (void)valid_control(UINT64_MAX, "used-rounding-control");
        g_model_arenas[0].used = UINT64_MAX;
        emit_snapshot("before_", g_last_control);
        g_fake_cuda_mode = 1;
        char *result = cuda_model_arena_alloc(1, "used-overflow");
        emit_state(result);
        return 0;
    }
    if (std::strcmp(name, "used-over-extent") == 0) {
        g_model_range_bytes = 45678;
        (void)valid_control(4096, "used-extent-control");
        g_model_arenas[0].used = 4097;
        emit_snapshot("before_", g_last_control);
        g_fake_cuda_mode = 1;
        char *result = cuda_model_arena_alloc(1, "used-over-extent");
        emit_state(result);
        return 0;
    }
    if (std::strcmp(name, "null-base") == 0) {
        g_model_range_bytes = 45678;
        (void)valid_control(4096, "null-base-control");
        g_model_arenas[0].device_ptr = nullptr;
        emit_snapshot("before_", g_last_control);
        g_fake_cuda_mode = 1;
        char *result = cuda_model_arena_alloc(1, "null-base");
        emit_state(result);
        return 0;
    }
    if (std::strcmp(name, "zero-extent") == 0) {
        g_model_range_bytes = 45678;
        (void)valid_empty_metadata(4096, "zero-extent-control");
        g_model_arenas[0].bytes = 0;
        emit_snapshot("before_", g_last_control);
        g_fake_cuda_mode = 1;
        char *result = cuda_model_arena_alloc(1, "zero-extent");
        emit_state(result);
        return 0;
    }

    if (std::strcmp(name, "ordinary-failure") == 0) {
        g_model_range_bytes = 56789;
        (void)valid_control(256, "ordinary-failure-control");
        emit_snapshot("before_", g_last_control);
        g_fake_cuda_mode = 1;
        char *result = cuda_model_arena_alloc(256, "ordinary-failure");
        emit_state(result);
        return 0;
    }
    if (std::strcmp(name, "success-with-null") == 0) {
        g_model_range_bytes = 56789;
        (void)valid_control(256, "success-null-control");
        emit_snapshot("before_", g_last_control);
        g_fake_cuda_mode = 2;
        char *result = cuda_model_arena_alloc(256, "success-with-null");
        emit_state(result);
        return 0;
    }
    return 2;
}

int main(int argc, char **argv) {
    configure_process();
    if (argc != 2) return 2;
    const int status = run_case(argv[1]);
    cleanup_backings();
    return status;
}
"""


def fixture_source(narrow: bool) -> str:
    # This macro is deliberately placed after all host headers.  It is a
    # numeric SIZE_MAX simulation, not a native 32-bit build claim.
    narrow_macro = """\n#undef SIZE_MAX\n#define SIZE_MAX UINT32_MAX\n""" if narrow else ""
    if CACHE_LIMIT is None or CHUNK is None or ALLOC is None:
        raise AssertionError("arena source extraction RED: required definition missing")
    return FIXTURE_PREFIX + narrow_macro + CACHE_LIMIT + "\n" + CHUNK + "\n" + ALLOC + "\n" + FIXTURE_DRIVER


def parse_output(stdout: str) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in stdout.splitlines():
        key, sep, value = line.partition("=")
        if not sep:
            continue
        try:
            values[key] = int(value)
        except ValueError:
            raise AssertionError(f"non-numeric fixture output: {line!r}")
    return values


def state_snapshot(values: dict[str, int], prefix: str = "") -> dict[str, int]:
    """Return the complete emitted state, including owners and attempts."""
    def key(name: str) -> str:
        return prefix + name

    snapshot: dict[str, int] = {}
    scalar_fields = (
        "result", "result_offset", "result_owner", "cuda_calls", "last_request",
        "cache_full", "range_bytes", "arena_count",
    )
    for field in scalar_fields:
        snapshot[field] = values[key(field)]
    for i in range(snapshot["arena_count"]):
        for field in ("base", "ptr", "owner", "offset", "cached_owner", "bytes", "used"):
            snapshot[f"arena_{i}_{field}"] = values[key(f"arena_{i}_{field}")]
    snapshot["owner_count"] = values[key("owner_count")]
    for i in range(snapshot["owner_count"]):
        for field in ("id", "backing", "requested", "reserved", "source"):
            snapshot[f"owner_{i}_{field}"] = values[key(f"owner_{i}_{field}")]
    snapshot["attempt_count"] = values[key("attempt_count")]
    for i in range(snapshot["attempt_count"]):
        for field in ("request", "result", "owner"):
            snapshot[f"attempt_{i}_{field}"] = values[key(f"attempt_{i}_{field}")]
    return snapshot


def without_result(snapshot: dict[str, int]) -> dict[str, int]:
    return {
        key: value for key, value in snapshot.items()
        if key not in {"result", "result_offset", "result_owner"}
    }


def persistent_owner_state(snapshot: dict[str, int]) -> dict[str, int]:
    return {
        key: value for key, value in snapshot.items()
        if key not in {
            "result", "result_offset", "result_owner", "cuda_calls",
            "last_request", "cache_full", "attempt_count",
        } and not key.startswith("attempt_")
    }


def assert_snapshot_matches(actual: dict[str, int], expected: dict[str, int]) -> None:
    """Small deterministic oracle used by both fixture assertions and its control test."""
    if actual != expected:
        raise AssertionError(f"snapshot mismatch: actual={actual!r} expected={expected!r}")


def owner_records(values: dict[str, int], prefix: str = "") -> dict[int, dict[str, int]]:
    snapshot = state_snapshot(values, prefix)
    owners: dict[int, dict[str, int]] = {}
    for i in range(snapshot["owner_count"]):
        owner_id = snapshot[f"owner_{i}_id"]
        owners[owner_id] = {
            field: snapshot[f"owner_{i}_{field}"]
            for field in ("backing", "requested", "reserved", "source")
        }
    return owners


def assert_arena_reservations(testcase: unittest.TestCase,
                              values: dict[str, int],
                              prefix: str = "") -> None:
    snapshot = state_snapshot(values, prefix)
    owners = owner_records(values, prefix)
    for i in range(snapshot["arena_count"]):
        owner_id = snapshot[f"arena_{i}_owner"]
        if owner_id == 0:
            testcase.assertEqual(snapshot[f"arena_{i}_base"], 0)
            testcase.assertEqual(snapshot[f"arena_{i}_ptr"], 0)
            testcase.assertEqual(snapshot[f"arena_{i}_offset"], -1)
            continue
        testcase.assertIn(owner_id, owners)
        owner = owners[owner_id]
        testcase.assertGreater(owner["backing"], 0)
        testcase.assertEqual(snapshot[f"arena_{i}_ptr"], owner["backing"])
        testcase.assertEqual(snapshot[f"arena_{i}_offset"], 0)
        testcase.assertEqual(snapshot[f"arena_{i}_cached_owner"], owner_id)
        testcase.assertEqual(owner["requested"], snapshot[f"arena_{i}_bytes"])
        testcase.assertEqual(owner["reserved"], snapshot[f"arena_{i}_bytes"])


def assert_rejection_snapshot(testcase: unittest.TestCase,
                              values: dict[str, int]) -> None:
    testcase.assertEqual(values["control_ok"], 1)
    before = state_snapshot(values, "before_")
    after = state_snapshot(values)
    testcase.assertEqual(before["result"], 1)
    testcase.assertGreaterEqual(before["result_offset"], 0)
    testcase.assertLess(before["result_offset"], 4096)
    testcase.assertNotEqual(before["result_owner"], 0)
    # Geometry rejection must not mutate any arena, owner, logical-byte,
    # cache, request, or allocation-attempt field.
    assert_snapshot_matches(without_result(after), without_result(before))
    testcase.assertEqual(after["result"], 0)
    testcase.assertEqual(after["result_offset"], -1)
    testcase.assertEqual(after["result_owner"], 0)
    owners = owner_records(values)
    for i in range(after["arena_count"]):
        owner_id = after[f"arena_{i}_owner"]
        if owner_id:
            testcase.assertIn(owner_id, owners)


def assert_empty_rejection_snapshot(testcase: unittest.TestCase,
                                    values: dict[str, int]) -> None:
    testcase.assertEqual(values["control_ok"], 1)
    before = state_snapshot(values, "before_")
    after = state_snapshot(values)
    testcase.assertEqual(before["result"], 0)
    testcase.assertEqual(before["arena_count"], 0)
    testcase.assertEqual(before["owner_count"], 0)
    assert_snapshot_matches(without_result(after), without_result(before))
    testcase.assertEqual(after["result"], 0)
    testcase.assertEqual(after["result_offset"], -1)
    testcase.assertEqual(after["result_owner"], 0)
    assert_arena_reservations(testcase, values)


class ArenaGeometryContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Register cleanup before either source is compiled.  Each behavior
        # assertion invokes a separate bounded subprocess below.
        cls._tmp = tempfile.TemporaryDirectory(prefix="ds4-model-arena-contract-")
        cls.addClassCleanup(cls._tmp.cleanup)
        base = Path(cls._tmp.name)
        cls._fixtures: dict[str, tuple[Path, subprocess.CompletedProcess[str]]] = {}
        for variant, narrow in (("native", False), ("narrow-size", True)):
            source = base / f"{variant}.cc"
            binary = base / variant
            source.write_text(fixture_source(narrow), encoding="utf-8")
            compile_result = subprocess.run(
                ["c++", "-std=c++17", "-O0", str(source), "-o", str(binary)],
                cwd=ROOT,
                env=SAFE_ENV,
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            cls._fixtures[variant] = (binary, compile_result)

    def _case(self, variant: str, name: str,
              extra_env: dict[str, str] | None = None) -> dict[str, int]:
        binary, compile_result = self._fixtures[variant]
        self.assertEqual(
            compile_result.returncode,
            0,
            f"{variant} fixture compile RED:\n{compile_result.stderr}",
        )
        env = dict(SAFE_ENV)
        if extra_env:
            env.update(extra_env)
        result = subprocess.run(
            [str(binary), name],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=16,
            check=False,
        )
        self.assertGreaterEqual(
            result.returncode,
            0,
            f"{variant}/{name} died by signal {-result.returncode}; stdout={result.stdout!r}",
        )
        self.assertEqual(result.returncode, 0, f"{variant}/{name}: {result.stderr}")
        values = parse_output(result.stdout)
        self.assertIn("result", values, f"{variant}/{name} missing result: {result.stdout!r}")
        return values

    def test_00_actual_bodies_extract_and_compile_in_both_host_fixtures(self) -> None:
        self.assertIsNotNone(CHUNK, "chunk body-source RED: definition is absent")
        self.assertIsNotNone(ALLOC, "allocator body-source RED: definition is absent")
        self.assertIsNotNone(CACHE_LIMIT, "budget dependency extraction RED: definition is absent")
        for variant, (_, compile_result) in self._fixtures.items():
            with self.subTest(variant=variant):
                self.assertEqual(compile_result.returncode, 0, compile_result.stderr)

    def test_01_helper_defaults_clamps_rounding_and_representable_controls(self) -> None:
        MiB = 1048576
        align = 256 * MiB
        highest = (1 << 64) - 1
        highest -= highest % align
        cases = (
            ("helper-zero", {}, 0),
            ("helper-default", {}, 1792 * MiB),
            ("helper-min", {"DS4_CUDA_WEIGHT_ARENA_CHUNK_MB": "1"}, 256 * MiB),
            ("helper-max", {"DS4_CUDA_WEIGHT_ARENA_CHUNK_MB": "999999"}, 8192 * MiB),
            ("helper-round", {"DS4_CUDA_WEIGHT_ARENA_CHUNK_MB": "256"}, 512 * MiB),
            ("helper-round-exact", {"DS4_CUDA_WEIGHT_ARENA_CHUNK_MB": "256"}, align),
            ("helper-highest", {"DS4_CUDA_WEIGHT_ARENA_CHUNK_MB": "8192"}, highest),
            # This is a valid boundary test, not a claim that the old body is RED.
            ("helper-next", {"DS4_CUDA_WEIGHT_ARENA_CHUNK_MB": "8192"}, 0),
        )
        for name, env, expected in cases:
            with self.subTest(case=name):
                self.assertEqual(self._case("native", name, env)["result"], expected)

    def test_02_allocator_controls_are_independent_and_owner_exact(self) -> None:
        MiB = 1048576
        with self.subTest(case="small-reuse"):
            values = self._case("native", "small-reuse")
            snapshot = state_snapshot(values)
            self.assertEqual(values["result"], 1)
            self.assertEqual(values["first_offset"], 0)
            self.assertEqual(values["second_offset"], 256)
            self.assertEqual(values["result_owner"], values["arena_0_owner"])
            self.assertEqual(values["cuda_calls"], 1)
            self.assertEqual(values["arena_count"], 1)
            self.assertEqual(values["arena_0_bytes"], 1792 * MiB)
            self.assertEqual(values["arena_0_used"], 768)
            self.assertEqual(values["last_request"], 1792 * MiB)
            self.assertEqual(values["cache_full"], 0)
            self.assertEqual(values["attempt_count"], 1)
            self.assertEqual(values["attempt_0_result"], 2)
            self.assertEqual(values["attempt_0_owner"], values["arena_0_owner"])
            assert_arena_reservations(self, values)
            self.assertEqual(snapshot["owner_count"], 1)

        with self.subTest(case="no-fit-transition"):
            values = self._case("native", "no-fit-transition")
            self.assertEqual(values["control_ok"], 1)
            self.assertEqual(values["result"], 1)
            self.assertEqual(values["result_offset"], 0)
            self.assertNotEqual(values["result_owner"], values["arena_0_owner"])
            self.assertEqual(values["cuda_calls"], 1)
            self.assertEqual(values["arena_count"], 2)
            self.assertEqual(values["arena_0_bytes"], 512)
            self.assertEqual(values["arena_0_used"], 256)
            self.assertEqual(values["arena_1_used"], 512)
            self.assertEqual(values["arena_1_owner"], values["result_owner"])
            self.assertEqual(values["attempt_0_result"], 2)
            self.assertEqual(values["attempt_0_owner"], values["arena_1_owner"])
            assert_arena_reservations(self, values)

        with self.subTest(case="budget-exact"):
            limit = 1 << 30
            values = self._case(
                "native", "budget-exact", {"DS4_CUDA_WEIGHT_CACHE_LIMIT_GB": "1"}
            )
            self.assertEqual(values["result"], 1)
            self.assertEqual(values["cuda_calls"], 1)
            self.assertEqual(values["range_bytes"], limit - 256)
            self.assertEqual(values["arena_0_used"], 256)
            self.assertEqual(values["last_request"], 1792 * MiB)
            self.assertEqual(values["attempt_0_request"], 1792 * MiB)
            assert_arena_reservations(self, values)

        with self.subTest(case="budget-over"):
            values = self._case(
                "native", "budget-over", {"DS4_CUDA_WEIGHT_CACHE_LIMIT_GB": "1"}
            )
            self.assertEqual(values["result"], 0)
            self.assertEqual(values["cuda_calls"], 0)
            self.assertEqual(values["arena_count"], 0)
            self.assertEqual(values["owner_count"], 0)
            self.assertEqual(values["attempt_count"], 0)
            self.assertEqual(values["cache_full"], 0)

        with self.subTest(case="cache-full-guard"):
            values = self._case("native", "cache-full-guard")
            self.assertEqual(values["result"], 0)
            self.assertEqual(values["cuda_calls"], 0)
            self.assertEqual(values["arena_count"], 0)
            self.assertEqual(values["owner_count"], 0)
            self.assertEqual(values["range_bytes"], 12345)
            self.assertEqual(values["cache_full"], 1)

        with self.subTest(case="zero-request"):
            values = self._case("native", "zero-request")
            self.assertEqual(values["result"], 0)
            self.assertEqual(values["cuda_calls"], 0)
            self.assertEqual(values["arena_count"], 0)
            self.assertEqual(values["owner_count"], 0)
            self.assertEqual(values["range_bytes"], 67890)
            self.assertEqual(values["cache_full"], 0)

    def test_03_request_alignment_overflow_cases_are_independent(self) -> None:
        for name in ("request-overflow-empty", "request-chunk-rounding-overflow", "request-overflow-existing"):
            with self.subTest(case=name):
                values = self._case("native", name)
                if name in {"request-overflow-empty", "request-chunk-rounding-overflow"}:
                    assert_empty_rejection_snapshot(self, values)
                else:
                    assert_rejection_snapshot(self, values)
                self.assertEqual(values["cuda_calls"], values["before_cuda_calls"])
                self.assertEqual(values["attempt_count"], values["before_attempt_count"])

    def test_04_invalid_existing_geometry_has_isolated_valid_baseline(self) -> None:
        for name in (
            "used-rounding-overflow", "used-over-extent", "null-base", "zero-extent",
        ):
            with self.subTest(case=name):
                values = self._case("native", name)
                assert_rejection_snapshot(self, values)
                self.assertEqual(values["cuda_calls"], values["before_cuda_calls"])
                self.assertEqual(values["attempt_count"], values["before_attempt_count"])

    def test_05_narrow_size_max_simulation_is_independent(self) -> None:
        MiB = 1048576
        align = 256 * MiB
        narrow_max = (1 << 32) - 1
        highest = narrow_max - narrow_max % align
        helper_cases = (
            ("narrow-helper-above", {}, 0),
            (
                "narrow-helper-oversized-env",
                {"DS4_CUDA_WEIGHT_ARENA_CHUNK_MB": "8192"},
                0,
            ),
            ("narrow-helper-boundary", {}, highest),
            ("narrow-helper-next", {}, 0),
        )
        for name, env, expected in helper_cases:
            with self.subTest(case=name):
                self.assertEqual(self._case("narrow-size", name, env)["result"], expected)

        rejection_cases = (
            ("narrow-request-overflow", {}),
            ("narrow-request-above", {}),
            (
                "narrow-env-oversized-alloc",
                {"DS4_CUDA_WEIGHT_ARENA_CHUNK_MB": "8192"},
            ),
            ("narrow-existing-extent", {}),
            ("narrow-used-rounding", {}),
        )
        for name, env in rejection_cases:
            with self.subTest(case=name):
                values = self._case("narrow-size", name, env)
                if name == "narrow-env-oversized-alloc":
                    self.assertEqual(values["control_result"], 256 * MiB)
                if name in {"narrow-request-overflow", "narrow-request-above", "narrow-env-oversized-alloc"}:
                    assert_empty_rejection_snapshot(self, values)
                else:
                    assert_rejection_snapshot(self, values)
                self.assertEqual(values["cuda_calls"], values["before_cuda_calls"])
                self.assertEqual(values["attempt_count"], values["before_attempt_count"])

    def test_06_allocation_failure_modes_preserve_existing_owners(self) -> None:
        for name in ("ordinary-failure", "success-with-null"):
            with self.subTest(case=name):
                values = self._case("native", name)
                self.assertEqual(values["control_ok"], 1)
                before = state_snapshot(values, "before_")
                after = state_snapshot(values)
                self.assertEqual(after["result"], 0)
                self.assertEqual(after["range_bytes"], before["range_bytes"])
                assert_snapshot_matches(
                    persistent_owner_state(after), persistent_owner_state(before)
                )
                self.assertEqual(after["arena_count"], before["arena_count"])
                self.assertEqual(after["owner_count"], before["owner_count"])
                self.assertEqual(after["cache_full"], 1)
                self.assertEqual(after["cuda_calls"], before["cuda_calls"] + 1)
                self.assertEqual(after["attempt_count"], before["attempt_count"] + 1)
                attempt = after["attempt_count"] - 1
                self.assertEqual(after[f"attempt_{attempt}_request"], 1792 * 1048576)
                if name == "ordinary-failure":
                    self.assertEqual(after[f"attempt_{attempt}_result"], 0)
                else:
                    self.assertEqual(after[f"attempt_{attempt}_result"], 1)
                self.assertEqual(after[f"attempt_{attempt}_owner"], 0)
                assert_arena_reservations(self, values)


class SnapshotOracleControlTest(unittest.TestCase):
    """Standalone control: the snapshot oracle must detect ledger corruption."""

    @classmethod
    def setUpClass(cls) -> None:
        # Keep this class independent of ArenaGeometryContractTest's cleanup
        # lifecycle while still deriving its state from an actual compiled
        # production-body fixture.
        cls._tmp = tempfile.TemporaryDirectory(prefix="ds4-model-arena-oracle-")
        cls.addClassCleanup(cls._tmp.cleanup)
        base = Path(cls._tmp.name)
        cls._source = base / "native.cc"
        cls._binary = base / "native"
        cls._source.write_text(fixture_source(False), encoding="utf-8")
        cls._compile = subprocess.run(
            ["c++", "-std=c++17", "-O0", str(cls._source), "-o", str(cls._binary)],
            cwd=ROOT,
            env=SAFE_ENV,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )

    def _compiled_control(self) -> dict[str, int]:
        self.assertEqual(self._compile.returncode, 0, self._compile.stderr)
        result = subprocess.run(
            [str(self._binary), "small-reuse"],
            cwd=ROOT,
            env=SAFE_ENV,
            capture_output=True,
            text=True,
            timeout=16,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return parse_output(result.stdout)

    def test_accepted_snapshot_then_owner_reservation_and_logical_corruption_raise(self) -> None:
        # Derive a complete accepted state from a successful compiled
        # production-body control, then exercise the same projection/assertion
        # helpers used by the contract tests.  This class is standalone (no
        # active subTest).
        values = self._compiled_control()
        accepted = state_snapshot(values)
        assert_snapshot_matches(accepted, dict(accepted))
        assert_arena_reservations(self, values)

        synthetic_rejection = dict(values)
        for key, value in accepted.items():
            synthetic_rejection[f"before_{key}"] = value
        synthetic_rejection["control_ok"] = 1
        synthetic_rejection["result"] = 0
        synthetic_rejection["result_offset"] = -1
        synthetic_rejection["result_owner"] = 0
        assert_rejection_snapshot(self, synthetic_rejection)

        corrupted_base = dict(values)
        corrupted_base["arena_0_ptr"] += 256
        with self.assertRaises(AssertionError):
            assert_arena_reservations(self, corrupted_base)

        corrupted_identity = dict(values)
        corrupted_identity["arena_0_owner"] += 999
        with self.assertRaises(AssertionError):
            assert_arena_reservations(self, corrupted_identity)

        corrupted_reservation = dict(values)
        corrupted_reservation["owner_0_reserved"] += 256
        with self.assertRaises(AssertionError):
            assert_arena_reservations(self, corrupted_reservation)

        corrupted_logical_bytes = dict(synthetic_rejection)
        corrupted_logical_bytes["range_bytes"] += 256
        with self.assertRaises(AssertionError):
            assert_rejection_snapshot(self, corrupted_logical_bytes)


if __name__ == "__main__":
    unittest.main(verbosity=2)
