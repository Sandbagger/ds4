#!/usr/bin/env python3
"""Static-first host fixture for the checked CUDA cleanup contract.

The fixture extracts the authenticated baseline cleanup, mapped helpers, and
resident-release dependencies at runtime.  Its generated C++ links those real
bodies to explicit fake CUDA/global state only; handles are synthetic and never
dereferenced.  No production algorithm or checked result is copied here.
"""
from __future__ import annotations

import hashlib
import os
import re
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CUDA_SOURCE_PATH = ROOT / "ds4_cuda.cu"
try:
    CUDA_SOURCE = CUDA_SOURCE_PATH.read_text(encoding="utf-8")
except OSError:
    raise SystemExit(125)
SAFE_ENV = {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"}

# Frozen input identity is recorded for review; runtime source selection still
# reads only this repository file relative to __file__.
BASELINE_COMMIT = "3711f489a5aef83ea9b747467bd173ef05cc6adc"
BASELINE_TREE = "cf770eae24cbc08b7a8c830d1663339df1c59f5c"
SCOPE_SHA256 = "5292e916e84ea46952aea2f9c5e98ae0467d9371432feba6357857d4060e6fa4"
FD_DECISION_SHA256 = "f6623630bae7ced3d4f9d75c076563fe1f520372a8a7e3efc62fc25b5dd2d025"
REPAIR_SCOPE_SHA256 = "bd27ea7801368d305d03aeba41041e54a17912d2ae9b60ea86f7bec3815e628c"
REPAIR_CONTRACT_SHA256 = "799a93d4fd8cd66a4cb9ea84d440ad876355fdae8bcc2fbf84557902a8144c18"
FROZEN_FIXTURE_SHA256 = "f36747dface38b97e8602818df48ff6287daa32ce9a0278b682e9b22db5c6052"
FROZEN_REPORT_SHA256 = "cd63b8d3aa3c1bb73b3a824e1ded3c22a5e129b78042fc6cfb8c94d7e0523b27"


def extract_definition(source: str, signature: str) -> str | None:
    """Extract one real definition while ignoring braces in comments/literals."""
    start = source.rfind(signature)
    if start < 0:
        return None
    i, state = start + len(signature), "code"
    while i < len(source):
        c, n = source[i], source[i + 1] if i + 1 < len(source) else ""
        if state == "code":
            if c == "/" and n == "/": state, i = "line", i + 2; continue
            if c == "/" and n == "*": state, i = "block", i + 2; continue
            if c == '"': state, i = "string", i + 1; continue
            if c == "'": state, i = "char", i + 1; continue
            if c == ";": return None  # declaration, not a definition
            if c == "{": break
            i += 1; continue
        if state == "line":
            if c in "\r\n": state = "code"
            i += 1; continue
        if state == "block":
            if c == "*" and n == "/": state, i = "code", i + 2
            else: i += 1
            continue
        if c == "\\": i += 2
        elif (state == "string" and c == '"') or (state == "char" and c == "'"): state, i = "code", i + 1
        else: i += 1
    else: raise AssertionError(f"no body brace after {signature}")
    depth, state = 0, "code"
    while i < len(source):
        c, n = source[i], source[i + 1] if i + 1 < len(source) else ""
        if state == "code":
            if c == "/" and n == "/": state, i = "line", i + 2; continue
            if c == "/" and n == "*": state, i = "block", i + 2; continue
            if c == '"': state, i = "string", i + 1; continue
            if c == "'": state, i = "char", i + 1; continue
            if c == "{": depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0: return source[start:i + 1]
            i += 1; continue
        if state == "line":
            if c in "\r\n": state = "code"
            i += 1; continue
        if state == "block":
            if c == "*" and n == "/": state, i = "code", i + 2
            else: i += 1
            continue
        if c == "\\": i += 2
        elif (state == "string" and c == '"') or (state == "char" and c == "'"): state, i = "code", i + 1
        else: i += 1
    raise AssertionError(f"unterminated definition {signature}")


HELPER_SIGNATURES = {
    "score_graph_destroy": ("static int attention_decode_score_split_graph_destroy_one(", "int attention_decode_score_split_graph_destroy_one(", "static void attention_decode_score_split_graph_destroy_one(", "void attention_decode_score_split_graph_destroy_one("),
    "moe_graph_destroy": ("static int routed_moe_decode_graph_destroy_one(", "int routed_moe_decode_graph_destroy_one(", "static void routed_moe_decode_graph_destroy_one(", "void routed_moe_decode_graph_destroy_one("),
    "selected_cache_release": ("static int cuda_stream_selected_cache_release(", "int cuda_stream_selected_cache_release(", "static void cuda_stream_selected_cache_release(", "void cuda_stream_selected_cache_release("),
    "stage_slots_release": ("static int cuda_stage_slots_release(", "int cuda_stage_slots_release(", "static void cuda_stage_slots_release(", "void cuda_stage_slots_release("),
    "model_stage_release": ("static int cuda_model_stage_release(", "int cuda_model_stage_release(", "static void cuda_model_stage_release(", "void cuda_model_stage_release("),
    "selected_stage_release": ("static int cuda_stream_selected_stage_release(", "int cuda_stream_selected_stage_release(", "static void cuda_stream_selected_stage_release(", "void cuda_stream_selected_stage_release("),
    "model_range_release": ("static int cuda_model_range_release_all(", "int cuda_model_range_release_all(", "static void cuda_model_range_release_all(", "void cuda_model_range_release_all("),
    "q8_f16_release": ("static int cuda_q8_f16_cache_release_all(", "int cuda_q8_f16_cache_release_all(", "static void cuda_q8_f16_cache_release_all(", "void cuda_q8_f16_cache_release_all("),
}
OPTIONAL_SIGNATURES = {
    "q8_f32_release": ("static int cuda_q8_f32_cache_release_all(", "int cuda_q8_f32_cache_release_all(", "static void cuda_q8_f32_cache_release_all(", "void cuda_q8_f32_cache_release_all("),
}
DEPENDENCY_SIGNATURES = {
    "resident_fail": ("static cudaError_t cuda_laguna_resident_fail(",),
    "resident_site": ("static const ds4_runtime_callsite *cuda_laguna_resident_site(",),
    "resident_site_matches": ("static int cuda_laguna_resident_site_matches(",),
    "resident_record": ("static ds4_runtime_allocation_record *cuda_laguna_resident_record(",),
    "resident_has_relation": ("static int cuda_laguna_resident_has_relation(",),
    "resident_note_failure": ("static void cuda_laguna_resident_note_failure(",),
    "resident_release": ("static cudaError_t cuda_laguna_resident_release(",),
    "resident_free": ("static cudaError_t cuda_laguna_resident_free(",),
    "resident_free_host": ("static cudaError_t cuda_laguna_resident_free_host(",),
    "load_progress_reset": ("static void cuda_model_load_progress_reset(",),
    "set_current_device": ('extern "C" int ds4_gpu_set_current_device(', "int ds4_gpu_set_current_device("),
    "cuda_ok": ("static int cuda_ok(",),
}

def select_definition(source: str, signatures: tuple[str, ...]) -> str | None:
    for signature in signatures:
        body = extract_definition(source, signature)
        if body is not None:
            return body
    return None


def extract_braced_declaration(source: str, marker: str) -> str | None:
    """Extract one source-owned class/typedef declaration with nested braces."""
    start = source.rfind(marker)
    if start < 0:
        return None
    i = source.find("{", start, start + len(marker))
    if i < 0:
        raise AssertionError(f"no declaration brace after {marker}")
    depth, state = 0, "code"
    while i < len(source):
        c, n = source[i], source[i + 1] if i + 1 < len(source) else ""
        if state == "code":
            if c == "/" and n == "/": state, i = "line", i + 2; continue
            if c == "/" and n == "*": state, i = "block", i + 2; continue
            if c == '"': state, i = "string", i + 1; continue
            if c == "'": state, i = "char", i + 1; continue
            if c == "{": depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end = source.find(";", i + 1)
                    if end < 0:
                        raise AssertionError(f"no declaration terminator after {marker}")
                    return source[start:end + 1]
            i += 1; continue
        if state == "line":
            if c in "\r\n": state = "code"
            i += 1; continue
        if state == "block":
            if c == "*" and n == "/": state, i = "code", i + 2
            else: i += 1
            continue
        if c == "\\": i += 2
        elif (state == "string" and c == '"') or (state == "char" and c == "'"): state, i = "code", i + 1
        else: i += 1
    raise AssertionError(f"unterminated declaration {marker}")


DIRECT_HELPER_SIGNATURES = {
    "selected_cache_invalidate": ("static void cuda_stream_selected_cache_invalidate(",),
    "selected_ranges_valid": ("static int cuda_stream_selected_ranges_valid(",),
    "selected_ensure_bytes": ("static int cuda_stream_selected_ensure_bytes(",),
    "selected_ensure_i32": ("static int cuda_stream_selected_ensure_i32(",),
    "model_copy": ("static int cuda_model_copy_to_device_streamed(",),
    "model_copy_chunk_bytes": ("static uint64_t cuda_model_copy_chunk_bytes(",),
    "discard_source_pages": ("static void cuda_model_discard_source_pages(",),
    "drop_file_pages": ("static void cuda_model_drop_file_pages(",),
    "round_down": ("static uint64_t cuda_round_down(",),
    "stage_usable_bytes": ("static uint64_t cuda_stage_usable_bytes(",),
    "pread_full": ("static int cuda_pread_full(",),
    "model_stage_read": ("static int cuda_model_stage_read(",),
    "align_ptr": ("static void *cuda_align_ptr(",),
    "selected_stage_pool_alloc": ("static int cuda_stream_selected_stage_pool_alloc(",),
}
CALLER_SIGNATURES = {
    "score_graph_launch": ("static int attention_decode_score_split_graph_launch(",),
    "moe_graph_launch": ("static int routed_moe_decode_q4_graph_launch(",),
    "selected_cache_begin_load": ("static int cuda_stream_selected_cache_begin_load(",),
}

try:
    HELPERS = {k: select_definition(CUDA_SOURCE, v) for k, v in HELPER_SIGNATURES.items()}
    OPTIONAL_HELPERS = {k: select_definition(CUDA_SOURCE, v) for k, v in OPTIONAL_SIGNATURES.items()}
    DEPENDENCIES = {k: select_definition(CUDA_SOURCE, v) for k, v in DEPENDENCY_SIGNATURES.items()}
    DIRECT_HELPERS = {k: select_definition(CUDA_SOURCE, v) for k, v in DIRECT_HELPER_SIGNATURES.items()}
    CALLERS = {k: select_definition(CUDA_SOURCE, v) for k, v in CALLER_SIGNATURES.items()}
    STREAM_EXPERT_TABLE_TYPE = extract_braced_declaration(
        CUDA_SOURCE, "typedef struct ds4_gpu_stream_expert_table {")
    COMPACT_PERMIT_CLASS = extract_braced_declaration(
        CUDA_SOURCE, "class cuda_laguna_compact_legacy_permit {")
    CLEANUP_BODY = extract_definition(CUDA_SOURCE, 'extern "C" void ds4_gpu_cleanup(')
    CHECKED_BODY = select_definition(CUDA_SOURCE, ('extern "C" int ds4_gpu_cleanup_checked(', 'int ds4_gpu_cleanup_checked('))
    _all_required = {
        **HELPERS, **DEPENDENCIES, **DIRECT_HELPERS, **CALLERS,
        "stream_expert_table_type": STREAM_EXPERT_TABLE_TYPE,
        "compact_permit_class": COMPACT_PERMIT_CLASS,
        "cleanup_void": CLEANUP_BODY,
    }
    MISSING_SEAMS = [k for k, v in _all_required.items() if not v]
except (AssertionError, ValueError) as exc:
    sys.stderr.write(f"source extraction refused: {exc}\n")
    raise SystemExit(125)


FAKE_PREFIX = r'''
/* Private host-only fakes.  No production implementation or CUDA ABI is asserted. */
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cerrno>
#include <cstdlib>
#include <fcntl.h>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>
#include "ds4_gpu_mgpu.h"
#include "ds4_laguna_stream.h"

using cudaError_t = int;
using cudaStream_t = void *;
using cudaEvent_t = void *;
using cudaGraph_t = void *;
using cudaGraphExec_t = void *;
using cudaGraphNode_t = void *;
using cublasHandle_t = void *;
using cublasStatus_t = int;
using __half = unsigned short;
static const cudaError_t cudaSuccess = 0;
static const cudaError_t cudaErrorInvalidValue = 1;
static const cudaError_t cudaErrorInvalidDevice = 2;
static const cudaError_t cudaErrorUnknown = 3;
static const cublasStatus_t CUBLAS_STATUS_SUCCESS = 0;
static const int cudaMemcpyHostToDevice = 1;
static const unsigned int cudaStreamNonBlocking = 1u;
static const unsigned int cudaEventDisableTiming = 2u;

/* Host-only layouts for the actual graph callers.  The launch bodies are
 * source-selected; these symbols only record control flow and never execute
 * a kernel or inspect numerical buffers. */
struct dim3 {
    unsigned int x, y, z;
    dim3(unsigned int x_ = 1u, unsigned int y_ = 1u, unsigned int z_ = 1u)
        : x(x_), y(y_), z(z_) {}
};
struct cudaKernelNodeParams {
    void *func;
    dim3 gridDim, blockDim;
    size_t sharedMemBytes;
    void **kernelParams;
    void **extra;
};
struct cuda_block_q8_K { unsigned char opaque[1]; };
typedef struct {
    uint32_t n_rot, pos0, n_ctx_orig;
    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
} cuda_attention_inv_rope_params;
#ifndef CUDA_QK_K
#define CUDA_QK_K 256u
#endif
static void attention_decode_score_split_scores_kernel(void) {}
static void attention_decode_score_split_finalize_kernel(void) {}
static void rope_tail_kernel(void) {}
static void q8_K_quantize_kernel(void) {}
static void moe_gate_up_mid_decode_q4K_warp32_kernel(void) {}
static void moe_down_q4K_sum6_qwarp32_kernel(void) {}
static void moe_down_q4K_sum3_qwarp32_kernel(void) {}

/* Public layout is supplied by ds4_gpu_mgpu.h; the following explicitly
 * labelled FAKE private layouts are copied only to link selected bodies. */
struct cuda_score_split_graph_cache {
    cudaGraph_t graph; cudaGraphExec_t exec; cudaGraphNode_t score_node;
    cudaGraphNode_t final_node; cudaGraphNode_t rope_node;
    uint32_t n_head; uint32_t head_dim; uint32_t S; uint32_t final_threads;
    uint32_t n_rot; int fuses_inv_rope; int valid;
};
struct cuda_moe_decode_graph_cache {
    cudaGraph_t graph; cudaGraphExec_t exec; cudaGraphNode_t xq_node;
    cudaGraphNode_t gate_node; cudaGraphNode_t midq_node; cudaGraphNode_t down_node;
    uint32_t n_expert; uint32_t expert_in_dim; uint32_t expert_mid_dim;
    uint32_t out_dim; int valid;
};
struct cuda_stream_selected_cache {
    int valid; int logical_tier; const void *model_map; uint32_t layer;
    uint32_t n_total_expert; uint32_t slot_count; uint32_t compact_count;
    uint64_t gate_offset; uint64_t up_offset; uint64_t down_offset;
    uint64_t gate_expert_bytes; uint64_t down_expert_bytes;
    char *gate_ptr; char *up_ptr; char *down_ptr;
    uint64_t gate_capacity; uint64_t up_capacity; uint64_t down_capacity;
    int32_t *slot_selected_ptr; uint64_t slot_selected_capacity;
    ds4_gpu_tensor slot_selected_tensor;
};
struct cuda_device_cache { void *base; size_t bytes; int present; };
struct cache_range_entry { uint64_t source_offset; uint64_t bytes; int device_id; void *device_ptr; };
struct cuda_model_range {
    const void *host_base; uint64_t offset; uint64_t bytes; char *device_ptr;
    void *registered_base; char *registered_device_base; uint64_t registered_bytes;
    int host_registered; int arena_allocated;
};
struct cuda_model_arena { char *device_ptr; uint64_t bytes; uint64_t used; };
struct cuda_q8_f16_range {
    const void *host_base; uint64_t offset; uint64_t weight_bytes;
    uint64_t in_dim; uint64_t out_dim; __half *device_ptr; int device_id;
};
struct cuda_q8_f32_range {
    const void *host_base; uint64_t offset; uint64_t weight_bytes;
    uint64_t in_dim; uint64_t out_dim; float *device_ptr; int device_id;
};
struct ds4_gpu_laguna_compact { int marker; uint64_t rejection_count; };
enum ds4_gpu_laguna_destroy_status {
    DS4_GPU_LAGUNA_DESTROY_OK = 0,
    DS4_GPU_LAGUNA_DESTROY_RECOVERABLE = 1,
    DS4_GPU_LAGUNA_DESTROY_UNSAFE = 2,
};

static std::atomic<uint64_t> g_laguna_compact_generic_cleanup_attempts{0};
static std::atomic<int> g_laguna_compact_state{0};
static ds4_gpu_laguna_compact g_laguna_compact_storage{};
static ds4_gpu_laguna_destroy_status cuda_laguna_compact_destroy_checked(
    ds4_gpu_laguna_compact *);
static const int DS4_LAGUNA_COMPACT_IDLE = 0;
static int g_compact_fail_once;
static int g_compact_destroy_attempts, g_compact_destroy_refused, g_compact_destroy_completed;
static std::mutex g_laguna_resident_mutex;
static std::recursive_mutex g_laguna_compact_mutex;
static std::atomic<ds4_runtime_tracker *> g_laguna_resident_tracker{NULL};
static bool g_laguna_resident_legacy_tensor_seen;
static const uint8_t cuda_laguna_resident_namespace = 0x52u;
static const uint8_t cuda_laguna_resident_host_namespace = 0x48u;
typedef struct { void *base; uint64_t record_id; int pinned; }
    cuda_laguna_resident_retained_owner;
static cuda_laguna_resident_retained_owner g_laguna_resident_retained{};

/* Private globals consumed by the actual helper bodies. */
static const void *g_model_host_base;
static const char *g_model_device_base;
static uint64_t g_model_registered_size;
static int g_model_registered, g_model_device_owned;
static int g_model_range_mapping_supported = 1, g_model_hmm_direct;
static int g_model_fd = -1, g_model_direct_fd = -1;
static const void *g_model_fd_host_base;
static uint64_t g_model_direct_align = 1, g_model_file_size;
static const void *g_support_host_base;
static uint64_t g_support_host_size, g_support_offset_bias;
static int g_model_cache_full;
static cudaStream_t g_model_prefetch_stream, g_model_upload_stream;
static int g_cublas_ready;
static int g_current_logical_tier = -1;
static int g_ssd_streaming_mode;
static int g_cuda_no_setdevice_cache;
static int g_q8_f16_disabled_after_oom, g_q8_f16_budget_notice_printed;
static int g_model_range_release_failed;
static uint64_t g_model_range_bytes, g_q8_f16_bytes, g_q8_f32_bytes;
static int g_q8_cache_suppressed;
static uint64_t g_model_load_progress_next;
static double g_model_load_progress_last;
static int g_model_load_progress_started, g_model_load_progress_tty;
static void *g_cuda_tmp;
static uint64_t g_cuda_tmp_bytes;
static void *g_model_stage_raw[4], *g_model_stage[4];
static cudaEvent_t g_model_stage_event[4];
static uint64_t g_model_stage_reserved_bytes[4], g_model_stage_bytes;
static void *g_stream_selected_stage_raw[4], *g_stream_selected_stage[4];
static cudaEvent_t g_stream_selected_stage_event[4];
static uint64_t g_stream_selected_stage_reserved_bytes[4], g_stream_selected_stage_bytes;
static cuda_stream_selected_cache g_stream_selected_cache;
static void *g_xdev_bounce[DS4_MAX_GPUS][DS4_MAX_GPUS];
static size_t g_xdev_bounce_bytes[DS4_MAX_GPUS][DS4_MAX_GPUS];
static cudaStream_t g_stream_selected_upload_stream;
static cuda_device_cache g_dev_cache[DS4_MAX_GPUS];
static std::vector<cache_range_entry> g_cache_ranges;
static std::vector<cuda_model_range> g_model_ranges;
static std::vector<cuda_model_arena> g_model_arenas;
static std::unordered_map<uint64_t, size_t> g_model_range_by_offset;
static std::vector<cuda_q8_f16_range> g_q8_f16_ranges;
static std::unordered_map<uint64_t, size_t> g_q8_f16_by_offset;
static std::vector<cuda_q8_f32_range> g_q8_f32_ranges;
static std::unordered_map<uint64_t, size_t> g_q8_f32_by_offset;
static cuda_score_split_graph_cache g_score_split_graph[DS4_MAX_GPUS];
static cuda_moe_decode_graph_cache g_moe_decode_graph[DS4_MAX_GPUS];

ds4_gpu_ctx g_gpu[DS4_MAX_GPUS];
int g_n_gpus;
int g_gpu_peer_ok[DS4_MAX_GPUS][DS4_MAX_GPUS];

/* All fake API calls are declared before extracted production definitions. */
static cudaError_t cudaDeviceSynchronize(void);
static cudaError_t cudaSetDevice(int);
static cudaError_t cudaGetDevice(int *);
static cudaError_t cudaEventDestroy(cudaEvent_t);
static cudaError_t cudaStreamDestroy(cudaStream_t);
static cublasStatus_t cublasDestroy(cublasHandle_t);
static cudaError_t cudaFree(void *);
static cudaError_t cudaFreeHost(void *);
static cudaError_t cudaHostUnregister(void *);
static cudaError_t cudaGraphExecDestroy(cudaGraphExec_t);
static cudaError_t cudaGraphDestroy(cudaGraph_t);
static cudaError_t cudaMalloc(void **, size_t);
static cudaError_t cudaMallocHost(void **, size_t);
static cudaError_t cudaMemcpy(void *, const void *, size_t, int);
static cudaError_t cudaMemcpyAsync(void *, const void *, size_t, int, cudaStream_t);
static cudaError_t cudaEventSynchronize(cudaEvent_t);
static cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t);
static cudaError_t cudaStreamSynchronize(cudaStream_t);
static cudaError_t cudaStreamCreateWithFlags(cudaStream_t *, unsigned int);
static cudaError_t cudaEventCreateWithFlags(cudaEvent_t *, unsigned int);
static cudaError_t cudaGraphCreate(cudaGraph_t *, unsigned int);
static cudaError_t cudaGraphAddKernelNode(cudaGraphNode_t *, cudaGraph_t,
                                          const cudaGraphNode_t *, size_t,
                                          const cudaKernelNodeParams *);
static cudaError_t cudaGraphInstantiate(cudaGraphExec_t *, cudaGraph_t,
                                        void *, void *, size_t);
static cudaError_t cudaGraphExecKernelNodeSetParams(
    cudaGraphExec_t, cudaGraphNode_t, const cudaKernelNodeParams *);
static cudaError_t cudaGraphLaunch(cudaGraphExec_t, cudaStream_t);
static cudaError_t cudaGetLastError(void);
static const char *cudaGetErrorString(cudaError_t);
static int fake_close(int);
/* Source-selected read/advice bodies are retained, but their host effects are
 * fail-closed boundary sentinels. Any unexpected entry is infrastructure125. */
static ssize_t fake_pread(int, void *, size_t, off_t);
static int fake_posix_madvise(void *, size_t, int);
static int fake_posix_fadvise(int, off_t, off_t, int);
/* Forward declaration only; the checked definition is always source-selected. */
extern "C" int ds4_gpu_cleanup_checked(void);
#define close fake_close
#define pread fake_pread
#define posix_madvise fake_posix_madvise
#define posix_fadvise fake_posix_fadvise

'''
FAKE_SUFFIX = r'''


/* The registry is a fake ledger, not CUDA-driver behavior. Handles are never
 * dereferenced. Every effect is recorded while the owner is live, then the
 * ledger consumes it; failures leave the owner live. */
enum FakeKind { K_DEV, K_HOST, K_EVENT, K_STREAM, K_GRAPH, K_EXEC, K_BLAS, K_REG, K_COUNT };
static const char *kind_name(int k) {
    static const char *n[K_COUNT] = {"dev","host","event","stream","graph","exec","blas","reg"};
    return (k >= 0 && k < K_COUNT) ? n[k] : "bad";
}
struct FakeOwner { int kind; int id; int expected_device; int strict; bool live; bool effect; };
static std::map<uintptr_t, FakeOwner> owners;
static std::set<uintptr_t> consumed;
static std::vector<void *> known_ptrs;
static std::map<int, FakeOwner> fd_owners;
static std::set<int> consumed_fds;
static std::vector<int> known_fds;
static uint64_t attempted[K_COUNT], refused[K_COUNT], completed[K_COUNT];
static int set_attempted, set_refused, set_completed, sync_attempted, sync_refused, sync_completed;
static int close_attempted, close_refused, close_completed, close_effects;
static int current_device, sticky_cuda_error, cuda_last_error, last_error_reads;
static int last_checked_result = -2;
static int fake_errno_last, unknown_refusals, fault_hits, fault_id;
static int trace_index, trace_overflow, first_refusal_index = -1;
static int driver_protocol_errors, effect_order_errors, rescue_effects;
static int rescue_protocol_errors, rescue_tracker_errors, tracker_release_errors, setup_error;
static int fail_kind = -1, fail_id, fail_set_device = -1, fail_get_device;
static int fail_sync, fail_close_errno;
/* Bounded ordinal controls are tied to the checked source's clean seed_full
 * order: represented-device sync3/device7, cache set/sync4/5, staging
 * restore7, global restore11, and final restore16 (ds4_cuda.cu:8243-8253,
 * 8299-8300, 8316-8317, 8356-8357).  A device mismatch is an infrastructure
 * sensor failure. */
static int fail_set_ordinal = -1, fail_set_device_target = -1;
static int fail_sync_ordinal = -1, fail_sync_device_target = -1;
static int fired_set_ordinal = -1, fired_sync_ordinal = -1;
static int fired_sync_device = -1, target_mismatches;
static int armed_late_ordinal = -1, armed_late_device = -1;
static int cuda_malloc_attempts, cuda_malloc_host_attempts, allow_cuda_malloc;
static int graph_create_calls, graph_add_calls, graph_instantiate_calls;
static int graph_update_calls, graph_launch_calls;
static int memcpy_calls, event_sync_calls, event_record_calls, stream_sync_calls;
static int next_malloc_id, next_host_alloc_id, next_graph_id, next_exec_id;
static int next_event_id, next_stream_id, next_node_id;
static int last_created_graph_id = -1, last_created_exec_id = -1;
static std::vector<std::string> trace_log;
static char fake_model_map[256], fake_model_map_2[256];
static int tracker_latches;

static void trace(const std::string &s) {
    if (trace_index >= 512) { trace_overflow = 1; return; }
    trace_log.push_back(s); trace_index++;
}
static void fixture_boundary_violation(const char *name) {
    setup_error = 1;
    trace(std::string("I:unsupported-boundary:") + name);
}
static void refuse_trace(const std::string &s) {
    if (first_refusal_index < 0) first_refusal_index = trace_index;
    trace("R:" + s);
}
static void *fake_handle(int kind, int id) {
    return reinterpret_cast<void *>(static_cast<uintptr_t>(0x100000u + kind * 0x10000u + id * 0x10u + 1u));
}
static void register_ptr_at(int kind, int id, void *p, int expected = -1, int strict = 0) {
    if (!owners.emplace(reinterpret_cast<uintptr_t>(p), FakeOwner{kind,id,expected,strict,true,false}).second)
        driver_protocol_errors++;
    known_ptrs.push_back(p);
}
static void register_ptr(int kind, int id, int expected = -1, int strict = 0) {
    register_ptr_at(kind, id, fake_handle(kind, id), expected, strict);
}
static void register_fd(int fd, int id) {
    std::map<int,FakeOwner>::iterator old=fd_owners.find(fd);
    if(old!=fd_owners.end() && old->second.live) { driver_protocol_errors++; return; }
    fd_owners[fd]=FakeOwner{K_REG,id,-1,0,true,false}; known_fds.push_back(fd);
}
static bool take_fault(int kind, int id) {
    if (fail_kind == kind && (fail_id == id || fail_id < 0)) {
        fault_hits++; fault_id = id; fail_kind = -1; fail_id = 0; return true;
    }
    return false;
}
static void record_cuda_error(cudaError_t err) {
    cuda_last_error = err;
    if (sticky_cuda_error == cudaSuccess) sticky_cuda_error = err;
}
static void record_fixture_refusal(const std::string &label) { refuse_trace(label); }
static int production_represents(const void *p) {
    if(!p) return 1;
    for(int i=0;i<DS4_MAX_GPUS;i++) {
        if(g_gpu[i].boundary_event==p || g_gpu[i].stream==p || g_gpu[i].cublas==p || g_gpu[i].scratch==p) return 1;
        if(g_score_split_graph[i].graph==p || g_score_split_graph[i].exec==p || g_moe_decode_graph[i].graph==p || g_moe_decode_graph[i].exec==p) return 1;
        if(g_dev_cache[i].present && g_dev_cache[i].base==p) return 1;
        for(int j=0;j<DS4_MAX_GPUS;j++) if(g_xdev_bounce[i][j]==p) return 1;
    }
    if(g_stream_selected_cache.gate_ptr==p || g_stream_selected_cache.up_ptr==p || g_stream_selected_cache.down_ptr==p || g_stream_selected_cache.slot_selected_ptr==p || g_cuda_tmp==p || (g_model_device_owned && g_model_device_base==p) || g_model_prefetch_stream==p || g_model_upload_stream==p || g_stream_selected_upload_stream==p) return 1;
    for(int i=0;i<4;i++) if(g_model_stage_raw[i]==p || g_model_stage_event[i]==p || g_stream_selected_stage_raw[i]==p || g_stream_selected_stage_event[i]==p) return 1;
    for(size_t i=0;i<g_model_ranges.size();i++) if(g_model_ranges[i].device_ptr==p || (g_model_ranges[i].host_registered && g_model_ranges[i].registered_base==p)) return 1;
    for(size_t i=0;i<g_model_arenas.size();i++) if(g_model_arenas[i].device_ptr==p) return 1;
    for(size_t i=0;i<g_q8_f16_ranges.size();i++) if(g_q8_f16_ranges[i].device_ptr==p) return 1;
    for(size_t i=0;i<g_q8_f32_ranges.size();i++) if(g_q8_f32_ranges[i].device_ptr==p) return 1;
    if(g_model_registered && g_model_host_base==p) return 1;
    return 0;
}
static bool physical_retire_ptr(FakeOwner &o, const void *ptr, bool rescue) {
    if(!rescue && !production_represents(ptr)) effect_order_errors++;
    trace((rescue ? "RE:" : "E:") + std::to_string(o.id));
    o.effect = true;
    if(!o.live || !o.effect) { if(rescue) rescue_protocol_errors++; else effect_order_errors++; return false; }
    o.live = false; consumed.insert((uintptr_t)ptr);
    if(rescue) { rescue_effects++; trace("RC:" + std::to_string(o.id)); }
    else trace("C:" + std::to_string(o.id));
    return true;
}
static cudaError_t consume_ptr(int kind, const void *ptr) {
    if (!ptr) return cudaSuccess;
    uintptr_t key = reinterpret_cast<uintptr_t>(ptr);
    std::map<uintptr_t, FakeOwner>::iterator it = owners.find(key);
    if (it == owners.end() || consumed.count(key) || !it->second.live || it->second.kind != kind) {
        refused[kind]++; unknown_refusals++; refuse_trace("unknown:" + std::to_string(kind));
        record_cuda_error(cudaErrorInvalidValue); return cudaErrorInvalidValue;
    }
    FakeOwner &o = it->second;
    attempted[kind]++;
    trace("A:" + std::string(kind_name(kind)) + ":" + std::to_string(o.id));
    if (o.strict && current_device != o.expected_device) {
        refused[kind]++; refuse_trace("affinity:" + std::to_string(o.id));
        record_cuda_error(cudaErrorInvalidDevice); return cudaErrorInvalidDevice;
    }
    if (take_fault(kind, o.id)) {
        refused[kind]++; refuse_trace("fault:" + std::to_string(o.id));
        record_cuda_error(cudaErrorUnknown); return cudaErrorUnknown;
    }
    /* The common primitive performs the fake physical effect before retirement. */
    if(!physical_retire_ptr(o,ptr,false)) return cudaErrorUnknown;
    completed[kind]++;
    return cudaSuccess;
}
static int any_live_owner(void) {
    int n = 0; for (std::map<uintptr_t,FakeOwner>::const_iterator i=owners.begin(); i!=owners.end(); ++i)
        if (i->second.live) n++;
    for(std::map<int,FakeOwner>::const_iterator i=fd_owners.begin();i!=fd_owners.end();++i) if(i->second.live) n++;
    return n;
}
static int live_kind(int kind) {
    int n = 0; for (std::map<uintptr_t,FakeOwner>::const_iterator i=owners.begin(); i!=owners.end(); ++i)
        if (i->second.live && i->second.kind == kind) n++;
    return n;
}
static cudaError_t cudaDeviceSynchronize(void) {
    sync_attempted++; trace("A:sync");
    const bool ordinal_match = fail_sync_ordinal > 0 &&
        sync_attempted == fail_sync_ordinal;
    if (ordinal_match && fail_sync_device_target >= 0 &&
        current_device != fail_sync_device_target) {
        target_mismatches++;
        trace("I:sync-target:" + std::to_string(sync_attempted) + ":" +
              std::to_string(current_device) + ":" + std::to_string(fail_sync_device_target));
        fail_sync_ordinal = -1;
        fail_sync_device_target = -1;
        fixture_boundary_violation("sync-target");
        return cudaErrorUnknown;
    }
    if (fail_sync || ordinal_match) {
        fail_sync = 0;
        if (ordinal_match) {
            fired_sync_ordinal = sync_attempted;
            fired_sync_device = current_device;
            fail_sync_ordinal = -1;
            fail_sync_device_target = -1;
        }
        fault_hits++; fault_id = -100; sync_refused++; refuse_trace("sync");
        record_cuda_error(cudaErrorUnknown); return cudaErrorUnknown;
    }
    sync_completed++; trace("E:sync"); return cudaSuccess;
}
static cudaError_t cudaSetDevice(int device) {
    set_attempted++; trace("A:set:" + std::to_string(device));
    const bool ordinal_match = fail_set_ordinal > 0 &&
        set_attempted == fail_set_ordinal;
    if (ordinal_match && device != fail_set_device_target) {
        target_mismatches++;
        trace("I:set-target:" + std::to_string(set_attempted) + ":" +
              std::to_string(device) + ":" + std::to_string(fail_set_device_target));
        fail_set_ordinal = -1;
        fail_set_device_target = -1;
        fixture_boundary_violation("set-target");
        return cudaErrorUnknown;
    }
    if (fail_set_device == device || fail_set_device == -2 ||
        (ordinal_match && device == fail_set_device_target)) {
        fail_set_device = -1;
        if (ordinal_match) {
            fired_set_ordinal = set_attempted;
            fail_set_ordinal = -1;
        }
        fault_hits++; fault_id = device; set_refused++;
        refuse_trace("set:" + std::to_string(device));
        record_cuda_error(cudaErrorInvalidDevice); return cudaErrorInvalidDevice;
    }
    current_device = device; set_completed++; trace("E:set:" + std::to_string(device)); return cudaSuccess;
}
static cudaError_t cudaGetDevice(int *out) {
    trace("A:get");
    if (fail_get_device) { fail_get_device = 0; fault_hits++; fault_id = -101; refuse_trace("get"); record_cuda_error(cudaErrorUnknown); return cudaErrorUnknown; }
    if (!out) return cudaErrorInvalidValue; *out = current_device; trace("E:get"); return cudaSuccess;
}
static cudaError_t cudaEventDestroy(cudaEvent_t p) { return consume_ptr(K_EVENT,p); }
static cudaError_t cudaStreamDestroy(cudaStream_t p) { return consume_ptr(K_STREAM,p); }
static cublasStatus_t cublasDestroy(cublasHandle_t p) { return consume_ptr(K_BLAS,p) == cudaSuccess ? CUBLAS_STATUS_SUCCESS : 1; }
static cudaError_t cudaFree(void *p) { return consume_ptr(K_DEV,p); }
static cudaError_t cudaFreeHost(void *p) { return consume_ptr(K_HOST,p); }
static cudaError_t cudaHostUnregister(void *p) { return consume_ptr(K_REG,p); }
static cudaError_t cudaGraphExecDestroy(cudaGraphExec_t p) { return consume_ptr(K_EXEC,p); }
static cudaError_t cudaGraphDestroy(cudaGraph_t p) { return consume_ptr(K_GRAPH,p); }
static const size_t FAKE_DEVICE_ALLOCATION_MAX_BYTES = 4096u;
static cudaError_t cudaMalloc(void **out, size_t bytes) {
    cuda_malloc_attempts++;
    if (!allow_cuda_malloc) {
        fixture_boundary_violation("cudaMalloc");
        return cudaErrorUnknown;
    }
    if (!out || bytes == 0 || bytes > FAKE_DEVICE_ALLOCATION_MAX_BYTES) {
        fixture_boundary_violation("cudaMalloc-size");
        return cudaErrorUnknown;
    }
    const int id = next_malloc_id++;
    *out = fake_handle(K_DEV, id);
    register_ptr(K_DEV, id, current_device, 1);
    return cudaSuccess;
}
static cudaError_t cudaMallocHost(void **out, size_t bytes) {
    (void)out; (void)bytes; cuda_malloc_host_attempts++;
    fixture_boundary_violation("cudaMallocHost");
    return cudaErrorUnknown;
}
static cudaError_t cudaMemcpy(void *dst, const void *src, size_t bytes, int kind) {
    (void)dst; (void)src; (void)bytes; (void)kind; memcpy_calls++;
    fixture_boundary_violation("cudaMemcpy");
    return cudaErrorUnknown;
}
static cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t bytes,
                                   int kind, cudaStream_t stream) {
    (void)dst; (void)src; (void)bytes; (void)kind; (void)stream;
    memcpy_calls++; fixture_boundary_violation("cudaMemcpyAsync");
    return cudaErrorUnknown;
}
static cudaError_t cudaEventSynchronize(cudaEvent_t event) {
    (void)event; event_sync_calls++; fixture_boundary_violation("cudaEventSynchronize");
    return cudaErrorUnknown;
}
static cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    (void)event; (void)stream; event_record_calls++; fixture_boundary_violation("cudaEventRecord");
    return cudaErrorUnknown;
}
static cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    (void)stream; stream_sync_calls++; fixture_boundary_violation("cudaStreamSynchronize");
    return cudaErrorUnknown;
}
static cudaError_t cudaStreamCreateWithFlags(cudaStream_t *out, unsigned int flags) {
    (void)out; (void)flags; fixture_boundary_violation("cudaStreamCreateWithFlags");
    return cudaErrorUnknown;
}
static cudaError_t cudaEventCreateWithFlags(cudaEvent_t *out, unsigned int flags) {
    (void)out; (void)flags; fixture_boundary_violation("cudaEventCreateWithFlags");
    return cudaErrorUnknown;
}
static cudaError_t cudaGraphCreate(cudaGraph_t *out, unsigned int flags) {
    (void)flags;
    if (!out) return cudaErrorInvalidValue;
    const int id = next_graph_id++;
    *out = fake_handle(K_GRAPH, id);
    register_ptr(K_GRAPH, id, current_device, 1);
    last_created_graph_id = id; graph_create_calls++; return cudaSuccess;
}
static cudaError_t cudaGraphAddKernelNode(cudaGraphNode_t *out, cudaGraph_t graph,
                                          const cudaGraphNode_t *deps, size_t n,
                                          const cudaKernelNodeParams *params) {
    (void)graph; (void)deps; (void)n; (void)params;
    if (!out) return cudaErrorInvalidValue;
    *out = reinterpret_cast<cudaGraphNode_t>(
        static_cast<uintptr_t>(0x70000000u + next_node_id++));
    graph_add_calls++; return cudaSuccess;
}
static cudaError_t cudaGraphInstantiate(cudaGraphExec_t *out, cudaGraph_t graph,
                                        void *log, void *err, size_t bytes) {
    (void)log; (void)err; (void)bytes;
    if (!out || !graph) return cudaErrorInvalidValue;
    const int id = next_exec_id++;
    *out = fake_handle(K_EXEC, id);
    register_ptr(K_EXEC, id, current_device, 1);
    last_created_exec_id = id; graph_instantiate_calls++; return cudaSuccess;
}
static cudaError_t cudaGraphExecKernelNodeSetParams(
        cudaGraphExec_t exec, cudaGraphNode_t node,
        const cudaKernelNodeParams *params) {
    (void)exec; (void)node; (void)params; graph_update_calls++; return cudaSuccess;
}
static cudaError_t cudaGraphLaunch(cudaGraphExec_t exec, cudaStream_t stream) {
    (void)exec; (void)stream; graph_launch_calls++; return cudaSuccess;
}
static cudaError_t cudaGetLastError(void) { int e=cuda_last_error; cuda_last_error=cudaSuccess; last_error_reads++; return e; }
static const char *cudaGetErrorString(cudaError_t) { return "fake-cuda-status"; }
static bool physical_retire_fd(FakeOwner &o, int fd, bool rescue, bool complete) {
    if(!rescue && g_model_direct_fd!=fd) effect_order_errors++;
    trace((rescue ? "RE:fd:" : "E:fd:") + std::to_string(fd));
    o.effect=true;
    if(!o.live || !o.effect) { if(rescue) rescue_protocol_errors++; else effect_order_errors++; return false; }
    o.live=false; consumed_fds.insert(fd); close_effects++;
    if(rescue) { rescue_effects++; trace("RC:fd:" + std::to_string(fd)); }
    else if(complete) { close_completed++; trace("C:fd:" + std::to_string(fd)); }
    return true;
}
static int fake_close(int fd) {
    close_attempted++; trace("A:close:" + std::to_string(fd));
    std::map<int,FakeOwner>::iterator it=fd_owners.find(fd);
    if (it == fd_owners.end() || !it->second.live) {
        close_refused++; unknown_refusals++; refuse_trace("close-unknown"); fake_errno_last=EBADF; errno=EBADF; return -1;
    }
    FakeOwner &o=it->second;
    if (fail_close_errno) {
        int e=fail_close_errno; fail_close_errno=0; fault_hits++; fault_id = -200;
        if(!physical_retire_fd(o,fd,false,false)) return -1;
        close_refused++; fake_errno_last=e; errno=e; refuse_trace("close:" + std::to_string(e)); return -1;
    }
    if(!physical_retire_fd(o,fd,false,true)) return -1;
    return 0;
}
static ssize_t fake_pread(int fd, void *buf, size_t bytes, off_t offset) {
    (void)fd; (void)buf; (void)bytes; (void)offset;
    fixture_boundary_violation("pread");
    errno = EIO;
    return -1;
}
static int fake_posix_madvise(void *addr, size_t bytes, int advice) {
    (void)addr; (void)bytes; (void)advice;
    fixture_boundary_violation("posix_madvise");
    return EIO;
}
static int fake_posix_fadvise(int fd, off_t offset, off_t bytes, int advice) {
    (void)fd; (void)offset; (void)bytes; (void)advice;
    fixture_boundary_violation("posix_fadvise");
    return EIO;
}
static ds4_gpu_laguna_destroy_status cuda_laguna_compact_destroy_checked(ds4_gpu_laguna_compact *) {
    g_compact_destroy_attempts++;
    if (g_compact_fail_once) {
        g_compact_fail_once=0; fault_hits++; fault_id = -300;
        g_compact_destroy_refused++; refuse_trace("compact");
        return DS4_GPU_LAGUNA_DESTROY_RECOVERABLE;
    }
    g_compact_destroy_completed++; g_laguna_compact_state.store(0, std::memory_order_release); return DS4_GPU_LAGUNA_DESTROY_OK;
}

/* The actual resident release body calls these fakes.  They preserve sticky
 * first violation and retire records only after a fake physical success. */
extern "C" ds4_runtime_status ds4_runtime_tracker_latch_failure(
        ds4_runtime_tracker *tracker, ds4_runtime_violation violation) {
    tracker_latches++;
    if (!tracker) return DS4_RUNTIME_STATUS_UNSAFE;
    if (tracker->violation == DS4_RUNTIME_VIOLATION_NONE) tracker->violation=violation;
    return DS4_RUNTIME_STATUS_UNSAFE;
}
extern "C" ds4_runtime_status ds4_runtime_tracker_release(
        ds4_runtime_tracker *tracker, uint64_t id) {
    if (!tracker) return DS4_RUNTIME_STATUS_UNSAFE;
    for (size_t i=0;i<tracker->record_count;i++) if (tracker->records[i].live && tracker->records[i].id==id) {
        uintptr_t key=(uintptr_t)tracker->records[i].base;
        std::map<uintptr_t,FakeOwner>::const_iterator owner=owners.find(key);
        if (owner==owners.end() || owner->second.live || !owner->second.effect || !consumed.count(key)) {
            tracker_release_errors++; return DS4_RUNTIME_STATUS_UNSAFE;
        }
        tracker->records[i].live=false; return DS4_RUNTIME_STATUS_OK;
    }
    tracker_release_errors++; return DS4_RUNTIME_STATUS_UNSAFE;
}

static ds4_runtime_callsite tracker_sites[2];
static ds4_runtime_allocation_record tracker_records[8];
static ds4_runtime_tracker tracker;
static void attach_tracker(void *p, int pinned, uint64_t id) {
    bool busy=false; for(size_t i=0;i<tracker.record_count;i++) if(tracker.records[i].live) busy=true;
    if(busy || g_laguna_resident_tracker.load() || tracker.violation!=DS4_RUNTIME_VIOLATION_NONE || sticky_cuda_error!=cudaSuccess) { setup_error=1; return; }
    std::memset(&tracker,0,sizeof(tracker)); std::memset(tracker_sites,0,sizeof(tracker_sites));
    std::memset(tracker_records,0,sizeof(tracker_records));
    tracker_sites[0] = ds4_runtime_callsite{DS4_LAGUNA_CALLSITE_PINNED_STAGING_0,"stage",DS4_RUNTIME_CATEGORY_PINNED_STAGING,DS4_RUNTIME_DOMAIN_HOST,UINT64_MAX};
    tracker_sites[1] = ds4_runtime_callsite{DS4_LAGUNA_CALLSITE_OTHER_CUDA_KERNEL_TMP,"tmp",DS4_RUNTIME_CATEGORY_OTHER_CUDA,DS4_RUNTIME_DOMAIN_CUDA_DEVICE,UINT64_MAX};
    tracker.callsites=tracker_sites; tracker.callsite_count=2; tracker.records=tracker_records; tracker.record_capacity=8;
    tracker.records[0] = ds4_runtime_allocation_record{id,(uint64_t)(uintptr_t)p,8,8,DS4_RUNTIME_CATEGORY_PINNED_STAGING,DS4_RUNTIME_DOMAIN_HOST,DS4_LAGUNA_CALLSITE_PINNED_STAGING_0,DS4_RUNTIME_RELATION_OWNED_ALLOCATION,0,true};
    tracker.record_count=1; g_laguna_resident_tracker.store(&tracker,std::memory_order_release);
    (void)pinned;
}
static int tracker_live(void) { int n=0; for(size_t i=0;i<tracker.record_count;i++) if(tracker.records[i].live)n++; return n; }

static void reset_state(void) {
    bool tracker_busy=false; for(size_t i=0;i<tracker.record_count;i++) if(tracker.records[i].live) tracker_busy=true;
    if(!owners.empty() || !fd_owners.empty() || sticky_cuda_error!=cudaSuccess || cuda_last_error!=cudaSuccess || tracker_busy || g_laguna_resident_tracker.load() || g_laguna_compact_state.load()!=DS4_LAGUNA_COMPACT_IDLE) { setup_error=1; return; }
    owners.clear(); consumed.clear(); known_ptrs.clear(); fd_owners.clear(); consumed_fds.clear(); known_fds.clear();
    trace_log.clear(); std::memset(attempted,0,sizeof(attempted)); std::memset(refused,0,sizeof(refused)); std::memset(completed,0,sizeof(completed));
    std::memset(g_gpu,0,sizeof(g_gpu)); std::memset(g_dev_cache,0,sizeof(g_dev_cache));
    std::memset(g_score_split_graph,0,sizeof(g_score_split_graph)); std::memset(g_moe_decode_graph,0,sizeof(g_moe_decode_graph));
    std::memset(g_xdev_bounce,0,sizeof(g_xdev_bounce)); std::memset(g_xdev_bounce_bytes,0,sizeof(g_xdev_bounce_bytes));
    std::memset(&g_stream_selected_cache,0,sizeof(g_stream_selected_cache)); g_stream_selected_cache.logical_tier=-1;
    std::memset(g_model_stage_raw,0,sizeof(g_model_stage_raw)); std::memset(g_model_stage,0,sizeof(g_model_stage)); std::memset(g_model_stage_event,0,sizeof(g_model_stage_event)); std::memset(g_model_stage_reserved_bytes,0,sizeof(g_model_stage_reserved_bytes));
    std::memset(g_stream_selected_stage_raw,0,sizeof(g_stream_selected_stage_raw)); std::memset(g_stream_selected_stage,0,sizeof(g_stream_selected_stage)); std::memset(g_stream_selected_stage_event,0,sizeof(g_stream_selected_stage_event)); std::memset(g_stream_selected_stage_reserved_bytes,0,sizeof(g_stream_selected_stage_reserved_bytes));
    g_model_ranges.clear(); g_model_arenas.clear(); g_model_range_by_offset.clear(); g_q8_f16_ranges.clear(); g_q8_f16_by_offset.clear(); g_q8_f32_ranges.clear(); g_q8_f32_by_offset.clear(); g_cache_ranges.clear();
    g_n_gpus=0; g_current_logical_tier=-1; g_ssd_streaming_mode=0; g_stream_selected_upload_stream=NULL; current_device=3; sticky_cuda_error=0; cuda_last_error=0; last_error_reads=0; last_checked_result=-2; set_attempted=set_refused=set_completed=sync_attempted=sync_refused=sync_completed=0; close_attempted=close_refused=close_completed=close_effects=0;
    graph_create_calls=graph_add_calls=graph_instantiate_calls=graph_update_calls=graph_launch_calls=0; memcpy_calls=event_sync_calls=event_record_calls=stream_sync_calls=0; cuda_malloc_attempts=cuda_malloc_host_attempts=allow_cuda_malloc=0; next_malloc_id=17000; next_host_alloc_id=18000; next_graph_id=3000; next_exec_id=4000; next_event_id=5000; next_stream_id=6000; next_node_id=7000; last_created_graph_id=last_created_exec_id=-1;
    trace_index=trace_overflow=driver_protocol_errors=effect_order_errors=rescue_effects=0; rescue_protocol_errors=rescue_tracker_errors=tracker_release_errors=0; setup_error=0; unknown_refusals=fault_hits=fault_id=fake_errno_last=0; first_refusal_index=-1; fail_kind=-1; fail_id=0; fail_set_device=-1; fail_get_device=0; fail_sync=0; fail_set_ordinal=-1; fail_sync_ordinal=-1; fail_set_device_target=-1; fail_sync_device_target=-1; fired_set_ordinal=fired_sync_ordinal=-1; fired_sync_device=-1; armed_late_ordinal=armed_late_device=-1; target_mismatches=0; fail_close_errno=0; g_compact_fail_once=0; g_compact_destroy_attempts=g_compact_destroy_refused=g_compact_destroy_completed=0; tracker_latches=0;
    g_laguna_compact_generic_cleanup_attempts.store(0); g_laguna_compact_state.store(0); g_laguna_compact_storage.rejection_count=0; g_laguna_resident_tracker.store(NULL); std::memset(&g_laguna_resident_retained,0,sizeof(g_laguna_resident_retained)); g_laguna_resident_legacy_tensor_seen=false;
    g_model_host_base=NULL; g_model_device_base=NULL; g_model_registered_size=0; g_model_registered=0; g_model_device_owned=0; g_model_range_mapping_supported=1; g_model_hmm_direct=0; g_model_fd=-1; g_model_direct_fd=-1; g_model_fd_host_base=NULL; g_model_direct_align=1; g_model_file_size=0; g_support_host_base=NULL; g_support_host_size=0; g_support_offset_bias=0; g_model_cache_full=0; g_model_prefetch_stream=NULL; g_model_upload_stream=NULL; g_cublas_ready=0; g_q8_f16_disabled_after_oom=0; g_q8_f16_budget_notice_printed=0; g_model_range_release_failed=0; g_model_range_bytes=0; g_q8_f16_bytes=0; g_q8_f32_bytes=0; g_q8_cache_suppressed=0; g_cuda_tmp=NULL; g_cuda_tmp_bytes=0; g_model_stage_bytes=0; g_stream_selected_stage_bytes=0; g_model_load_progress_next=0; g_model_load_progress_last=0; g_model_load_progress_started=0; g_model_load_progress_tty=0;
}
static void seed_full(void) {
    g_n_gpus=2; g_gpu[0].device_id=3; g_gpu[1].device_id=7; current_device=3;
    for(int i=0;i<2;i++) {
        int p=g_gpu[i].device_id;
        g_gpu[i].boundary_event=fake_handle(K_EVENT,300+i); register_ptr(K_EVENT,300+i,p,1);
        g_gpu[i].stream=fake_handle(K_STREAM,400+i); register_ptr(K_STREAM,400+i,p,1);
        g_gpu[i].cublas=fake_handle(K_BLAS,500+i); register_ptr(K_BLAS,500+i,p,1); g_gpu[i].cublas_ready=1;
        g_gpu[i].scratch=fake_handle(K_DEV,600+i); register_ptr(K_DEV,600+i,p,1); g_gpu[i].scratch_bytes=32;
        g_score_split_graph[i].exec=fake_handle(K_EXEC,100+i); register_ptr(K_EXEC,100+i,p,1); g_score_split_graph[i].graph=fake_handle(K_GRAPH,200+i); register_ptr(K_GRAPH,200+i,p,1); g_score_split_graph[i].valid=1;
        g_moe_decode_graph[i].exec=fake_handle(K_EXEC,110+i); register_ptr(K_EXEC,110+i,p,1); g_moe_decode_graph[i].graph=fake_handle(K_GRAPH,210+i); register_ptr(K_GRAPH,210+i,p,1); g_moe_decode_graph[i].valid=1;
    }
    for(int i=0;i<2;i++) { g_xdev_bounce[i][i^1]=fake_handle(K_HOST,650+i); register_ptr(K_HOST,650+i,-1,0); g_xdev_bounce_bytes[i][i^1]=16; }
    g_stream_selected_cache.valid=1; g_stream_selected_cache.logical_tier=1; g_stream_selected_cache.gate_ptr=(char*)fake_handle(K_DEV,700); register_ptr(K_DEV,700,7,1); g_stream_selected_cache.up_ptr=(char*)fake_handle(K_DEV,701); register_ptr(K_DEV,701,7,1); g_stream_selected_cache.down_ptr=(char*)fake_handle(K_DEV,702); register_ptr(K_DEV,702,7,1); g_stream_selected_cache.slot_selected_ptr=(int32_t*)fake_handle(K_DEV,703); register_ptr(K_DEV,703,7,1); g_stream_selected_cache.slot_selected_tensor.ptr=g_stream_selected_cache.slot_selected_ptr; g_stream_selected_cache.slot_selected_tensor.device_id=7;
    for(int d=0;d<2;d++) { int physical=g_gpu[d].device_id; g_dev_cache[physical].present=1; g_dev_cache[physical].bytes=24; g_dev_cache[physical].base=fake_handle(K_DEV,750+d); register_ptr(K_DEV,750+d,physical,1); g_cache_ranges.push_back(cache_range_entry{0,24,physical,g_dev_cache[physical].base}); }
    for(int i=0;i<2;i++) { cuda_model_range r{}; r.host_base=fake_model_map; r.offset=100+i*16; r.bytes=16; r.device_ptr=(char*)fake_handle(K_DEV,800+i); register_ptr(K_DEV,800+i,-1,0); g_model_ranges.push_back(r); g_model_range_by_offset[r.offset]=i; g_model_range_bytes+=16; }
    cuda_model_range rr{}; rr.host_base=fake_model_map_2; rr.offset=200; rr.bytes=16; rr.host_registered=1; rr.registered_base=fake_model_map_2; rr.registered_device_base=(char*)fake_handle(K_DEV,820); rr.registered_bytes=16; register_ptr_at(K_REG,820,(void*)fake_model_map_2,-1,0); g_model_ranges.push_back(rr); g_model_range_by_offset[rr.offset]=2; g_model_range_bytes+=16;
    cuda_model_arena a{}; a.device_ptr=(char*)fake_handle(K_DEV,830); a.bytes=64; a.used=32; register_ptr(K_DEV,830,-1,0); g_model_arenas.push_back(a); cuda_model_range av{}; av.host_base=fake_model_map; av.offset=300; av.bytes=8; av.device_ptr=(char *)(uintptr_t)((uintptr_t)a.device_ptr+8); av.arena_allocated=1; g_model_ranges.push_back(av); g_model_range_bytes+=8;
    cuda_q8_f16_range f16{}; f16.host_base=fake_model_map; f16.offset=400; f16.weight_bytes=8; f16.device_ptr=(__half*)fake_handle(K_DEV,900); f16.device_id=3; register_ptr(K_DEV,900,3,0); g_q8_f16_ranges.push_back(f16); g_q8_f16_by_offset[400]=0; g_q8_f16_bytes=8;
    cuda_q8_f32_range f32{}; f32.host_base=fake_model_map; f32.offset=500; f32.weight_bytes=8; f32.device_ptr=(float*)fake_handle(K_DEV,910); f32.device_id=7; register_ptr(K_DEV,910,7,0); g_q8_f32_ranges.push_back(f32); g_q8_f32_by_offset[500]=0; g_q8_f32_bytes=8;
    g_cuda_tmp=fake_handle(K_DEV,950); register_ptr(K_DEV,950,-1,0); g_cuda_tmp_bytes=12;
    for(int i=0;i<4;i++) { g_model_stage_raw[i]=fake_handle(K_HOST,1000+i); register_ptr(K_HOST,1000+i,-1,0); g_model_stage[i]=(void*)((uintptr_t)g_model_stage_raw[i]+5); g_model_stage_event[i]=fake_handle(K_EVENT,1050+i); register_ptr(K_EVENT,1050+i,-1,0); g_model_stage_reserved_bytes[i]=8; g_model_stage_bytes+=8; g_stream_selected_stage_raw[i]=fake_handle(K_HOST,1100+i); register_ptr(K_HOST,1100+i,-1,0); g_stream_selected_stage[i]=(void*)((uintptr_t)g_stream_selected_stage_raw[i]+5); g_stream_selected_stage_event[i]=fake_handle(K_EVENT,1150+i); register_ptr(K_EVENT,1150+i,-1,0); g_stream_selected_stage_reserved_bytes[i]=8; g_stream_selected_stage_bytes+=8; }
    g_model_upload_stream=fake_handle(K_STREAM,1200); register_ptr(K_STREAM,1200,-1,0); g_stream_selected_upload_stream=fake_handle(K_STREAM,1210); register_ptr(K_STREAM,1210,-1,0);
    g_model_device_owned=1; g_model_device_base=(const char*)fake_handle(K_DEV,1250); register_ptr(K_DEV,1250,-1,0); g_model_registered=1; g_model_host_base=fake_model_map; g_model_registered_size=256; register_ptr_at(K_REG,1260,(void*)fake_model_map,-1,0);
    register_fd(41,41); g_model_direct_fd=41; g_model_prefetch_stream=fake_handle(K_STREAM,1300); register_ptr(K_STREAM,1300,-1,0);
}
static void seed_empty(void) { g_model_direct_fd=-1; g_stream_selected_cache.logical_tier=-1; }
static void arm_named(const char *name) {
    if (!std::strcmp(name,"sync")) fail_sync=1;
    else if (!std::strcmp(name,"graph-exec")) { fail_kind=K_EXEC; fail_id=100; }
    else if (!std::strcmp(name,"graph")) { fail_kind=K_GRAPH; fail_id=200; }
    else if (!std::strcmp(name,"moe-exec")) { fail_kind=K_EXEC; fail_id=110; }
    else if (!std::strcmp(name,"moe-graph")) { fail_kind=K_GRAPH; fail_id=210; }
    else if (!std::strcmp(name,"event")) { fail_kind=K_EVENT; fail_id=300; }
    else if (!std::strcmp(name,"event-second")) { fail_kind=K_EVENT; fail_id=301; }
    else if (!std::strcmp(name,"stream")) { fail_kind=K_STREAM; fail_id=400; }
    else if (!std::strcmp(name,"stream-second")) { fail_kind=K_STREAM; fail_id=401; }
    else if (!std::strcmp(name,"blas")) { fail_kind=K_BLAS; fail_id=500; }
    else if (!std::strcmp(name,"blas-second")) { fail_kind=K_BLAS; fail_id=501; }
    else if (!std::strcmp(name,"scratch")) { fail_kind=K_DEV; fail_id=600; }
    else if (!std::strcmp(name,"scratch-second")) { fail_kind=K_DEV; fail_id=601; }
    else if (!std::strcmp(name,"bounce")) { fail_kind=K_HOST; fail_id=650; }
    else if (!std::strcmp(name,"selected-cache")) { fail_kind=K_DEV; fail_id=700; }
    else if (!std::strcmp(name,"selected-stage-event")) { fail_kind=K_EVENT; fail_id=1150; }
    else if (!std::strcmp(name,"model-stage-event")) { fail_kind=K_EVENT; fail_id=1050; }
    else if (!std::strcmp(name,"selected-stage-host")) { fail_kind=K_HOST; fail_id=1100; }
    else if (!std::strcmp(name,"model-stage")) { fail_kind=K_HOST; fail_id=1000; }
    else if (!std::strcmp(name,"model-upload")) { fail_kind=K_STREAM; fail_id=1200; }
    else if (!std::strcmp(name,"selected-stage-upload")) { fail_kind=K_STREAM; fail_id=1210; }
    else if (!std::strcmp(name,"range")) { fail_kind=K_DEV; fail_id=801; }
    else if (!std::strcmp(name,"arena")) { fail_kind=K_DEV; fail_id=830; }
    else if (!std::strcmp(name,"q8")) { fail_kind=K_DEV; fail_id=900; }
    else if (!std::strcmp(name,"q8-f32")) { fail_kind=K_DEV; fail_id=910; }
    else if (!std::strcmp(name,"tmp")) { fail_kind=K_DEV; fail_id=950; }
    else if (!std::strcmp(name,"model-device")) { fail_kind=K_DEV; fail_id=1250; }
    else if (!std::strcmp(name,"registration")) { fail_kind=K_REG; fail_id=1260; }
    else if (!std::strcmp(name,"prefetch")) { fail_kind=K_STREAM; fail_id=1300; }
    else if (!std::strcmp(name,"select-device")) fail_set_device=7;
    else if (!std::strcmp(name,"get-device")) fail_get_device=1;
    else if (!std::strcmp(name,"compact")) g_compact_fail_once=1;
}
static int arm_late(const char *name) {
    /* These exact suffixed names and ordinals are source-bound to seed_full's
     * clean checked path. An unadmitted name is infrastructure125. */
    if (!name) {
        trace("I:late-arm-unknown:<null>");
        setup_error=1;
        return 0;
    }
    if (!std::strcmp(name,"cache-selection-failure")) {
        armed_late_ordinal=4; armed_late_device=7;
        fail_set_ordinal=armed_late_ordinal; fail_set_device_target=armed_late_device; /* 8249-8253 */
    } else if (!std::strcmp(name,"represented-device-sync-failure")) {
        armed_late_ordinal=3; armed_late_device=7;
        fail_sync_ordinal=armed_late_ordinal; fail_sync_device_target=armed_late_device; /* 8243-8247 */
    } else if (!std::strcmp(name,"cache-sync-failure")) {
        armed_late_ordinal=5; armed_late_device=7;
        fail_sync_ordinal=armed_late_ordinal; fail_sync_device_target=armed_late_device; /* 8249-8253 */
    } else if (!std::strcmp(name,"staging-restore-failure")) {
        armed_late_ordinal=7; armed_late_device=3;
        fail_set_ordinal=armed_late_ordinal; fail_set_device_target=armed_late_device; /* 8299-8300 */
    } else if (!std::strcmp(name,"global-restore-failure")) {
        armed_late_ordinal=11; armed_late_device=3;
        fail_set_ordinal=armed_late_ordinal; fail_set_device_target=armed_late_device; /* 8316-8317 */
    } else if (!std::strcmp(name,"final-restore-failure")) {
        armed_late_ordinal=16; armed_late_device=3;
        fail_set_ordinal=armed_late_ordinal; fail_set_device_target=armed_late_device; /* 8356-8357 */
    } else {
        trace(std::string("I:late-arm-unknown:") + (name ? name : "<null>"));
        setup_error=1;
        return 0;
    }
    return 1;
}
static void add_unknown_selected(void) { g_n_gpus=1; g_gpu[0].device_id=3; g_stream_selected_cache.valid=1; g_stream_selected_cache.logical_tier=0; g_stream_selected_cache.gate_ptr=(char*)fake_handle(K_DEV,1999); }
static void add_tracker_stage(void) { reset_state(); void *p=fake_handle(K_HOST,1400); register_ptr(K_HOST,1400,-1,0); g_model_stage_raw[0]=p; g_model_stage[0]=(void*)((uintptr_t)p+5); g_model_stage_reserved_bytes[0]=8; g_model_stage_bytes=8; attach_tracker(p,1,UINT64_C(0x5200000000000001)); }
static void reuse_fd(void) { register_fd(41,99); }
static bool rescue_ptr(int kind, const void *ptr) {
    if(!ptr) return true;
    std::map<uintptr_t,FakeOwner>::iterator o=owners.find((uintptr_t)ptr);
    if(o==owners.end() || !o->second.live || o->second.kind!=kind) { rescue_protocol_errors++; return false; }
    /* Same effect primitive/order as production, with a distinct rescue label. */
    trace("RA:" + std::string(kind_name(kind)) + ":" + std::to_string(o->second.id));
    return physical_retire_ptr(o->second,ptr,true);
}
static bool rescue_fd(int fd) {
    std::map<int,FakeOwner>::iterator o=fd_owners.find(fd);
    if(o==fd_owners.end() || !o->second.live) { rescue_protocol_errors++; return false; }
    trace("RA:fd:" + std::to_string(fd));
    return physical_retire_fd(o->second,fd,true,true);
}
static void rescue_all_known_live(void) {
    /* Bounded to explicit witnesses.  The physical effect is traced before
     * retirement; injection budgets and sticky history are untouched. */
    if(g_laguna_compact_state.load()!=DS4_LAGUNA_COMPACT_IDLE) {
        /* This case consumed its synthetic refusal before the saved snapshot.
         * Retry the known compact witness; never reset its live state directly. */
        if(g_compact_fail_once) { rescue_protocol_errors++; return; }
        trace("RA:compact");
        if(cuda_laguna_compact_destroy_checked(&g_laguna_compact_storage)!=DS4_GPU_LAGUNA_DESTROY_OK ||
           g_laguna_compact_state.load()!=DS4_LAGUNA_COMPACT_IDLE) {
            rescue_protocol_errors++; return;
        }
        trace("RC:compact");
    }
    std::set<uintptr_t> seen_ptrs;
    for(size_t i=0;i<known_ptrs.size();i++) {
        void *p=known_ptrs[i]; if(!seen_ptrs.insert((uintptr_t)p).second) continue;
        std::map<uintptr_t,FakeOwner>::iterator o=owners.find((uintptr_t)p);
        if(o!=owners.end() && o->second.live) (void)rescue_ptr(o->second.kind,p);
    }
    std::set<int> seen_fds;
    for(size_t i=0;i<known_fds.size();i++) {
        if(!seen_fds.insert(known_fds[i]).second) continue;
        std::map<int,FakeOwner>::iterator o=fd_owners.find(known_fds[i]);
        if(o!=fd_owners.end() && o->second.live) (void)rescue_fd(known_fds[i]);
    }
    if(g_laguna_resident_tracker.load()) for(size_t i=0;i<tracker.record_count;i++) if(tracker.records[i].live) {
        uintptr_t key=(uintptr_t)tracker.records[i].base;
        std::map<uintptr_t,FakeOwner>::const_iterator o=owners.find(key);
        if(o==owners.end() || o->second.live || !o->second.effect || !consumed.count(key)) rescue_tracker_errors++;
        else (void)ds4_runtime_tracker_release(&tracker,tracker.records[i].id);
    }
}
static void add_ref(std::set<uintptr_t> &r, const void *p) { if(p) r.insert((uintptr_t)p); }
static void collect_owner_refs(std::set<uintptr_t> &r) {
    for(int i=0;i<DS4_MAX_GPUS;i++) {
        add_ref(r,g_gpu[i].boundary_event); add_ref(r,g_gpu[i].stream); add_ref(r,g_gpu[i].cublas); add_ref(r,g_gpu[i].scratch);
        add_ref(r,g_score_split_graph[i].graph); add_ref(r,g_score_split_graph[i].exec); add_ref(r,g_moe_decode_graph[i].graph); add_ref(r,g_moe_decode_graph[i].exec);
        for(int j=0;j<DS4_MAX_GPUS;j++) add_ref(r,g_xdev_bounce[i][j]);
        if(g_dev_cache[i].present) add_ref(r,g_dev_cache[i].base);
    }
    add_ref(r,g_stream_selected_cache.gate_ptr); add_ref(r,g_stream_selected_cache.up_ptr); add_ref(r,g_stream_selected_cache.down_ptr); add_ref(r,g_stream_selected_cache.slot_selected_ptr);
    for(size_t i=0;i<g_model_ranges.size();i++) { add_ref(r,g_model_ranges[i].device_ptr); if(g_model_ranges[i].host_registered) add_ref(r,g_model_ranges[i].registered_base); }
    for(size_t i=0;i<g_model_arenas.size();i++) add_ref(r,g_model_arenas[i].device_ptr);
    for(size_t i=0;i<g_q8_f16_ranges.size();i++) add_ref(r,g_q8_f16_ranges[i].device_ptr);
    for(size_t i=0;i<g_q8_f32_ranges.size();i++) add_ref(r,g_q8_f32_ranges[i].device_ptr);
    for(int i=0;i<4;i++) { add_ref(r,g_model_stage_raw[i]); add_ref(r,g_model_stage_event[i]); add_ref(r,g_stream_selected_stage_raw[i]); add_ref(r,g_stream_selected_stage_event[i]); }
    add_ref(r,g_model_upload_stream); add_ref(r,g_stream_selected_upload_stream); add_ref(r,g_cuda_tmp); if(g_model_device_owned) add_ref(r,g_model_device_base); if(g_model_registered) add_ref(r,g_model_host_base); add_ref(r,g_model_prefetch_stream);
}
static int state_refs(void) { std::set<uintptr_t> r; collect_owner_refs(r); return (int)r.size(); }
static int stale_refs(void) {
    std::set<uintptr_t> r; collect_owner_refs(r); int n=0;
    for(std::set<uintptr_t>::const_iterator i=r.begin();i!=r.end();++i) { std::map<uintptr_t,FakeOwner>::const_iterator o=owners.find(*i); if(o!=owners.end()&&!o->second.live&&consumed.count(*i)) n++; }
    return n;
}
static int unowned_live(void) {
    std::set<uintptr_t> r; collect_owner_refs(r); int n=0;
    for(std::map<uintptr_t,FakeOwner>::const_iterator i=owners.begin();i!=owners.end();++i) if(i->second.live&&!r.count(i->first)) n++;
    return n;
}

static int remaining_budget(void) {
    return (fail_kind>=0)+(fail_set_device!=-1)+(fail_get_device!=0)+fail_sync+
        (fail_set_ordinal>0)+(fail_sync_ordinal>0)+(fail_close_errno!=0)+g_compact_fail_once;
}
static int owner_live_for(int kind, int id) {
    for(std::map<uintptr_t,FakeOwner>::const_iterator i=owners.begin();i!=owners.end();++i)
        if(i->second.kind==kind && i->second.id==id) return i->second.live?1:0;
    return 0;
}
static int owner_live_id(int id) {
    for(std::map<uintptr_t,FakeOwner>::const_iterator i=owners.begin();i!=owners.end();++i)
        if(i->second.id==id) return i->second.live?1:0;
    return 0;
}
static int owner_id_for_ptr(const void *p) {
    if (!p) return -1;
    for(std::map<uintptr_t,FakeOwner>::const_iterator i=owners.begin();i!=owners.end();++i)
        if(i->first==(uintptr_t)p) return i->second.id;
    return -1;
}
static int fd_live_for(int fd) {
    std::map<int,FakeOwner>::const_iterator i=fd_owners.find(fd);
    return i!=fd_owners.end() && i->second.live ? 1 : 0;
}
static int total_counter(const uint64_t *a) { int n=0; for(int k=0;k<K_COUNT;k++) n+=(int)a[k]; return n; }
static void emit(void) {
    std::printf("generic_attempts=%llu\n",(unsigned long long)g_laguna_compact_generic_cleanup_attempts.load());
    std::printf("g_n_gpus=%d\ncurrent_device=%d\nchecked_available=%d\nfd_owner_live=%d\nfd_state_live=%d\n",g_n_gpus,current_device,CHECKED_AVAILABLE,fd_live_for(41),g_model_direct_fd>=0?fd_live_for(g_model_direct_fd):0);
    std::printf("gpu0_device=%d\ngpu1_device=%d\ncompact_pending=%d\n",g_gpu[0].device_id,g_gpu[1].device_id,g_laguna_compact_state.load()!=DS4_LAGUNA_COMPACT_IDLE);
    std::printf("compact_attempted=%d\ncompact_refused=%d\ncompact_completed=%d\n",g_compact_destroy_attempts,g_compact_destroy_refused,g_compact_destroy_completed);
    std::printf("physical_live=%d\nstate_refs=%d\nunowned_live=%d\nstale_refs=%d\n",any_live_owner(),state_refs(),unowned_live(),stale_refs());
    std::printf("trace_n=%d\ntrace_overflow=%d\nfirst_refusal=%d\n",trace_index,trace_overflow,first_refusal_index);
    std::printf("driver_protocol_errors=%d\nunknown_refusals=%d\neffect_order_errors=%d\n",driver_protocol_errors,unknown_refusals,effect_order_errors);
    std::printf("fault_hits=%d\nfault_id=%d\nsticky_cuda_error=%d\ncuda_last_error=%d\nfake_errno=%d\nlast_error_reads=%d\n",fault_hits,fault_id,sticky_cuda_error,cuda_last_error,fake_errno_last,last_error_reads);
    std::printf("fired_set_ordinal=%d\nfired_sync_ordinal=%d\nfired_sync_device=%d\narmed_late_ordinal=%d\narmed_late_device=%d\ntarget_mismatches=%d\n",fired_set_ordinal,fired_sync_ordinal,fired_sync_device,armed_late_ordinal,armed_late_device,target_mismatches);
    std::printf("cuda_malloc_attempts=%d\ncuda_malloc_host_attempts=%d\ngraph_create_calls=%d\ngraph_add_calls=%d\ngraph_instantiate_calls=%d\ngraph_update_calls=%d\ngraph_launch_calls=%d\n",cuda_malloc_attempts,cuda_malloc_host_attempts,graph_create_calls,graph_add_calls,graph_instantiate_calls,graph_update_calls,graph_launch_calls);
    std::printf("memcpy_calls=%d\nevent_sync_calls=%d\nevent_record_calls=%d\nstream_sync_calls=%d\nlast_created_graph_id=%d\nlast_created_exec_id=%d\n",memcpy_calls,event_sync_calls,event_record_calls,stream_sync_calls,last_created_graph_id,last_created_exec_id);
    std::printf("tracker_violation=%d\ntracker_live=%d\ntracker_latches=%d\ntracker_release_errors=%d\n",g_laguna_resident_tracker.load()?tracker.violation:0,g_laguna_resident_tracker.load()?tracker_live():0,tracker_latches,tracker_release_errors);
    std::printf("rescue_protocol_errors=%d\nrescue_tracker_errors=%d\nsetup_error=%d\n",rescue_protocol_errors,rescue_tracker_errors,setup_error);
    for(int k=0;k<K_COUNT;k++) std::printf("attempt_%s=%llu\nrefused_%s=%llu\ncompleted_%s=%llu\n",kind_name(k),(unsigned long long)attempted[k],kind_name(k),(unsigned long long)refused[k],kind_name(k),(unsigned long long)completed[k]);
    std::printf("attempt_total=%d\nrefused_total=%d\ncompleted_total=%d\n",total_counter(attempted),total_counter(refused),total_counter(completed));
    for(int id : {100,110,200,210,300,301,400,401,500,501,600,601,650,700,750,751,800,801,830,900,910,1000,1050,1100,1150,1200,1210,1250,1260,1300}) std::printf("physical_%d=%d\n",id,owner_live_id(id));
    std::printf("set_attempted=%d\nset_refused=%d\nset_completed=%d\nsync_attempted=%d\nsync_refused=%d\nsync_completed=%d\nclose_attempted=%d\nclose_refused=%d\nclose_completed=%d\nclose_effects=%d\nclose_slot=%d\nmodel_stage_bytes=%llu\nselected_stage_bytes=%llu\nrange_count=%llu\nrange_bytes=%llu\nq8_f16_count=%llu\nq8_f32_count=%llu\nprefetch_present=%d\nremaining_budget=%d\n",set_attempted,set_refused,set_completed,sync_attempted,sync_refused,sync_completed,close_attempted,close_refused,close_completed,close_effects,g_model_direct_fd,(unsigned long long)g_model_stage_bytes,(unsigned long long)g_stream_selected_stage_bytes,(unsigned long long)g_model_ranges.size(),(unsigned long long)g_model_range_bytes,(unsigned long long)g_q8_f16_ranges.size(),(unsigned long long)g_q8_f32_ranges.size(),g_model_prefetch_stream!=NULL,remaining_budget());
}
static void emit_after_rescue(int before_live, int sticky_before, int budget_before, int fd_reused) {
    int reuse_before=fd_reused ? fd_live_for(41) : 0;
    std::printf("production_snapshot_recorded=1\nexpectations_recorded=1\nrescue_before_live=%d\nreuse_live_before_rescue=%d\n",before_live,reuse_before);
    rescue_all_known_live();
    std::printf("rescue_after_live=%d\nrescue_effects=%d\nrescue_sticky_same=%d\nrescue_budget_same=%d\nrescue_complete=%d\nreuse_live_after_rescue=%d\n",any_live_owner(),rescue_effects,(sticky_cuda_error==sticky_before),remaining_budget()==budget_before,(any_live_owner()==0 && g_laguna_compact_state.load()==DS4_LAGUNA_COMPACT_IDLE && (!g_laguna_resident_tracker.load() || tracker_live()==0)),fd_reused?fd_live_for(41):0);
}
static int call_cleanup(const char *label) {
#if CHECKED_AVAILABLE
    last_checked_result=ds4_gpu_cleanup_checked(); std::printf("%s=%d\n",label,last_checked_result); return last_checked_result;
#else
    last_checked_result=-2; ds4_gpu_cleanup(); return 0;
#endif
}
static void call_legacy_cleanup(void) { ds4_gpu_cleanup(); std::printf("legacy_called=1\n"); }
static int finish_case(int fd_reused=0) {
    emit();
    int before=any_live_owner(), sticky=sticky_cuda_error, budget=remaining_budget();
    emit_after_rescue(before,sticky,budget,fd_reused);
    if(setup_error || trace_overflow || effect_order_errors || driver_protocol_errors || rescue_protocol_errors || rescue_tracker_errors || tracker_release_errors || any_live_owner()!=0 || g_laguna_compact_state.load()!=DS4_LAGUNA_COMPACT_IDLE || (g_laguna_resident_tracker.load() && tracker_live()!=0)) return 125;
    return 0;
}
static void seed_single_gpu(void) {
    g_n_gpus=1; g_gpu[0].device_id=3; current_device=3; g_current_logical_tier=-1;
}
static int call_score_graph(void) {
    static float heads[1], sinks[1], scores[1], q[1], raw_kv[1], comp_kv[1], comp_mask[1];
    return attention_decode_score_split_graph_launch(
        0, heads, sinks, scores, q, raw_kv, comp_kv, comp_mask,
        0u, 0u, 1u, 1u, 0u, 0u, 0u, 0u, 1u, 1u, 256u, 1u, NULL);
}
static int call_moe_graph(void) {
    static float out[1], gate_out[1], up_out[1], mid_out[1], weights[3], x[1];
    static char gate_w[1], up_w[1], down_w[1];
    static cuda_block_q8_K xq[1], midq[3];
    static int32_t selected[3];
    return routed_moe_decode_q4_graph_launch(
        0, out, gate_out, up_out, mid_out, gate_w, up_w, down_w,
        xq, midq, selected, weights, 16u, 16u, 16u, 16u,
        256u, 256u, 32u, 3u, 0u, 0.0f, x);
}
static int graph_reuse_case(bool moe) {
    reset_state(); seed_single_gpu();
    const int first = moe ? call_moe_graph() : call_score_graph();
    const int first_graph = last_created_graph_id;
    const int first_exec = last_created_exec_id;
    const int first_shape_live = (owner_live_id(first_graph) && owner_live_id(first_exec));
    if (first != 1 || !first_shape_live || graph_create_calls != 1 ||
        graph_instantiate_calls != 1 || graph_launch_calls != 1 ||
        graph_update_calls != 0 || sticky_cuda_error != cudaSuccess) {
        fixture_boundary_violation("graph-healthy-control");
        return finish_case();
    }
    fail_kind = K_EXEC; fail_id = first_exec;
    const int refused = moe
        ? routed_moe_decode_graph_destroy_one(0)
        : attention_decode_score_split_graph_destroy_one(0);
    const int first_marker = moe ? g_moe_decode_graph[0].valid : g_score_split_graph[0].valid;
    const int first_graph_live = owner_live_id(first_graph);
    const int first_exec_live = owner_live_id(first_exec);
    const int first_sticky = sticky_cuda_error;
    const int first_refusal = first_refusal_index;
    const int create_after_refusal = graph_create_calls;
    const int update_after_refusal = graph_update_calls;
    const int launch_after_refusal = graph_launch_calls;
    fail_kind = K_EXEC; fail_id = first_exec;
    const int blocked = moe ? call_moe_graph() : call_score_graph();
    const int blocked_marker = moe ? g_moe_decode_graph[0].valid : g_score_split_graph[0].valid;
    const int blocked_graph_live = owner_live_id(first_graph);
    const int blocked_exec_live = owner_live_id(first_exec);
    const int create_after_block = graph_create_calls;
    const int update_after_block = graph_update_calls;
    const int launch_after_block = graph_launch_calls;
    /* Consume the completed exec prefix, then refuse the graph suffix. */
    fail_kind = K_GRAPH; fail_id = first_graph;
    const int graph_refused = moe
        ? routed_moe_decode_graph_destroy_one(0)
        : attention_decode_score_split_graph_destroy_one(0);
    const int graph_refusal_marker = moe ? g_moe_decode_graph[0].valid : g_score_split_graph[0].valid;
    const int exec_live_after_graph_refusal = owner_live_id(first_exec);
    const int graph_live_after_graph_refusal = owner_live_id(first_graph);
    const int create_after_graph_refusal = graph_create_calls;
    const int update_after_graph_refusal = graph_update_calls;
    const int launch_after_graph_refusal = graph_launch_calls;
    const int retry = moe ? call_moe_graph() : call_score_graph();
    const int retry_graph = last_created_graph_id;
    const int retry_exec = last_created_exec_id;
    const int first_error_same = sticky_cuda_error == first_sticky;
    const int first_index_same = first_refusal_index == first_refusal;
    std::printf("graph_retry_old_graph_live=%d\ngraph_retry_old_exec_live=%d\ngraph_retry_unowned=%d\n",
                owner_live_id(first_graph), owner_live_id(first_exec), unowned_live());
    const int no_effect_while_retained =
        create_after_block == create_after_refusal &&
        update_after_block == update_after_refusal &&
        launch_after_block == launch_after_refusal &&
        create_after_graph_refusal == create_after_refusal &&
        update_after_graph_refusal == update_after_refusal &&
        launch_after_graph_refusal == launch_after_refusal;
    std::printf("graph_moe=%d\ngraph_first=%d\ngraph_refused=%d\ngraph_blocked=%d\ngraph_refused_after_exec=%d\ngraph_retry=%d\n",moe,first,refused,blocked,graph_refused,retry);
    std::printf("graph_first_id=%d\ngraph_first_exec=%d\ngraph_first_shape_live=%d\ngraph_first_marker=%d\ngraph_first_graph_live=%d\ngraph_first_exec_live=%d\n",first_graph,first_exec,first_shape_live,first_marker,first_graph_live,first_exec_live);
    std::printf("graph_blocked_marker=%d\ngraph_blocked_graph_live=%d\ngraph_blocked_exec_live=%d\ngraph_retained_after_block=%d\n",blocked_marker,blocked_graph_live,blocked_exec_live,blocked_exec_live);
    std::printf("graph_graph_refusal_marker=%d\ngraph_exec_live_after_graph_refusal=%d\ngraph_graph_live_after_graph_refusal=%d\n",graph_refusal_marker,exec_live_after_graph_refusal,graph_live_after_graph_refusal);
    std::printf("graph_create_after_refusal=%d\ngraph_update_after_refusal=%d\ngraph_launch_after_refusal=%d\n",create_after_refusal,update_after_refusal,launch_after_refusal);
    std::printf("graph_create_after_block=%d\ngraph_update_after_block=%d\ngraph_launch_after_block=%d\n",create_after_block,update_after_block,launch_after_block);
    std::printf("graph_create_after_graph_refusal=%d\ngraph_update_after_graph_refusal=%d\ngraph_launch_after_graph_refusal=%d\ngraph_no_effect_while_retained=%d\n",create_after_graph_refusal,update_after_graph_refusal,launch_after_graph_refusal,no_effect_while_retained);
    std::printf("graph_retry_id=%d\ngraph_retry_exec=%d\ngraph_retry_new_graph=%d\ngraph_retry_new_exec=%d\n",retry_graph,retry_exec,retry_graph!=first_graph,retry_exec!=first_exec);
    std::printf("graph_first_refusal=%d\ngraph_first_sticky=%d\ngraph_sticky=%d\ngraph_first_refusal_after_retry=%d\ngraph_sticky_same=%d\ngraph_first_refusal_same=%d\n",first_refusal,first_sticky,sticky_cuda_error,first_refusal_index,first_error_same,first_index_same);
    return finish_case();
}
static int selected_resize_case(bool refuse) {
    reset_state(); seed_single_gpu();
    char *old = (char *)fake_handle(K_DEV, 1700);
    register_ptr(K_DEV, 1700, 3, 1);
    g_stream_selected_cache.logical_tier=0;
    g_stream_selected_cache.gate_ptr=old;
    g_stream_selected_cache.gate_capacity=8;
    if (refuse) { fail_kind=K_DEV; fail_id=1700; }
    allow_cuda_malloc=1;
    const int result = cuda_stream_selected_ensure_bytes(
        &g_stream_selected_cache.gate_ptr,
        &g_stream_selected_cache.gate_capacity, 16u, "resize control");
    allow_cuda_malloc=0;
    const int resize_field_id = owner_id_for_ptr(g_stream_selected_cache.gate_ptr);
    if (!refuse && (result != 1 || owner_live_id(1700) != 0 ||
                   resize_field_id != 17000 || owner_live_id(resize_field_id) != 1 ||
                   g_stream_selected_cache.gate_capacity != 16u ||
                   cuda_malloc_attempts != 1 || unowned_live() != 0))
        fixture_boundary_violation("resize-healthy-control");
    std::printf("resize_refuse=%d\nresize_result=%d\nresize_old_live=%d\nresize_old_represented=%d\nresize_field_id=%d\nresize_field_known=%d\nresize_field_live=%d\nresize_capacity=%llu\nresize_malloc_attempts=%d\nresize_unowned_before_rescue=%d\n",refuse,result,owner_live_id(1700),production_represents(old),resize_field_id,resize_field_id>=0,owner_live_id(resize_field_id),(unsigned long long)g_stream_selected_cache.gate_capacity,cuda_malloc_attempts,unowned_live());
    return finish_case();
}
static int selected_begin_reuse_case(void) {
    reset_state(); seed_single_gpu(); g_ssd_streaming_mode=0;
    char *old = (char *)fake_handle(K_DEV, 1800);
    register_ptr(K_DEV, 1800, 3, 1);
    g_stream_selected_cache.valid=1; g_stream_selected_cache.logical_tier=0;
    g_stream_selected_cache.gate_ptr=old; g_stream_selected_cache.gate_capacity=8;
    static int32_t selected_ids[1]={0};
    ds4_gpu_stream_expert_table table{};
    fail_kind=K_DEV; fail_id=1800;
    const int release_refused = cuda_stream_selected_cache_release();
    const int initial_release_sticky = sticky_cuda_error;
    const int initial_release_index = first_refusal_index;
    const int after_release_valid = g_stream_selected_cache.valid;
    const int after_release_ptr_id = owner_id_for_ptr(g_stream_selected_cache.gate_ptr);
    const int after_release_capacity = (int)g_stream_selected_cache.gate_capacity;
    const int after_release_old_live = owner_live_id(1800);
    const int after_release_malloc_attempts = cuda_malloc_attempts;
    cuda_stream_selected_cache_invalidate();
    const int after_invalidate_valid = g_stream_selected_cache.valid;
    const int after_invalidate_ptr_id = owner_id_for_ptr(g_stream_selected_cache.gate_ptr);
    const int after_invalidate_old_live = owner_live_id(1800);
    fail_kind=K_DEV; fail_id=1800;
    const int first = cuda_stream_selected_cache_begin_load(&table, selected_ids, 1u);
    const int first_valid = g_stream_selected_cache.valid;
    const int first_ptr_id = owner_id_for_ptr(g_stream_selected_cache.gate_ptr);
    const int first_capacity = (int)g_stream_selected_cache.gate_capacity;
    const int first_old_live = owner_live_id(1800);
    const int first_malloc_attempts = cuda_malloc_attempts;
    const int first_unowned = unowned_live();
    const int first_failure = initial_release_index;
    const int first_sticky = initial_release_sticky;
    const int retry = cuda_stream_selected_cache_begin_load(&table, selected_ids, 1u);
    std::printf("begin_release_refused=%d\nbegin_after_release_valid=%d\nbegin_after_release_ptr_id=%d\nbegin_after_release_capacity=%d\nbegin_after_release_old_live=%d\nbegin_after_release_malloc_attempts=%d\nbegin_invalidated_valid=%d\nbegin_invalidated_ptr_id=%d\nbegin_invalidated_old_live=%d\n",release_refused,after_release_valid,after_release_ptr_id,after_release_capacity,after_release_old_live,after_release_malloc_attempts,after_invalidate_valid,after_invalidate_ptr_id,after_invalidate_old_live);
    std::printf("begin_first=%d\nbegin_first_valid=%d\nbegin_first_ptr_id=%d\nbegin_first_capacity=%d\nbegin_first_old_live=%d\nbegin_first_malloc_attempts=%d\nbegin_first_unowned=%d\n",first,first_valid,first_ptr_id,first_capacity,first_old_live,first_malloc_attempts,first_unowned);
    std::printf("begin_retry=%d\nbegin_retry_valid=%d\nbegin_retry_ptr_id=%d\nbegin_retry_capacity=%llu\nbegin_retry_old_live=%d\nbegin_retry_sticky_same=%d\nbegin_retry_first_refusal_same=%d\n",retry,g_stream_selected_cache.valid,owner_id_for_ptr(g_stream_selected_cache.gate_ptr),(unsigned long long)g_stream_selected_cache.gate_capacity,owner_live_id(1800),sticky_cuda_error==first_sticky,first_refusal_index==first_failure);
    return finish_case();
}
static int late_device_case(const char *name) {
    reset_state(); seed_full();
    const int admitted = arm_late(name);
    if (!admitted) {
        std::printf("late_arm_admitted=0\n");
        return finish_case();
    }
    const int first = call_cleanup("checked_result");
    const int first_fault = fault_hits, first_id = fault_id;
    const int first_set = fired_set_ordinal, first_sync = fired_sync_ordinal;
    const int first_sync_device = fired_sync_device;
    const int first_mismatch = target_mismatches, first_n = g_n_gpus;
    if (first_fault != 1 || first_mismatch != 0 ||
        (first_set != armed_late_ordinal && first_sync != armed_late_ordinal))
        fixture_boundary_violation("late-target-unproved");
    const int first_gpu0 = g_gpu[0].device_id, first_gpu1 = g_gpu[1].device_id;
    const int first_current_device = current_device;
    const int first_physical = any_live_owner(), first_refs = state_refs();
    const int first_unowned = unowned_live();
    const int p300 = owner_live_id(300), p301 = owner_live_id(301);
    const int p400 = owner_live_id(400), p401 = owner_live_id(401);
    const int p500 = owner_live_id(500), p501 = owner_live_id(501);
    const int p600 = owner_live_id(600), p601 = owner_live_id(601);
    const int p650 = owner_live_id(650), p651 = owner_live_id(651);
    const int p700 = owner_live_id(700), p750 = owner_live_id(750), p751 = owner_live_id(751);
    const int p800 = owner_live_id(800), p801 = owner_live_id(801), p820 = owner_live_id(820);
    const int p830 = owner_live_id(830), p900 = owner_live_id(900), p910 = owner_live_id(910);
    const int p950 = owner_live_id(950);
    const int p1000 = owner_live_id(1000), p1001 = owner_live_id(1001);
    const int p1050 = owner_live_id(1050), p1100 = owner_live_id(1100);
    const int p1150 = owner_live_id(1150), p1200 = owner_live_id(1200), p1210 = owner_live_id(1210);
    const int p1250 = owner_live_id(1250), p1260 = owner_live_id(1260), p1300 = owner_live_id(1300);
    const int first_refusal = first_refusal_index, first_sticky = sticky_cuda_error;
    const int first_affinity = first_gpu0 == 3 && first_gpu1 == 7 && first_n == 2;
    const int first_source_base_retained = g_model_host_base == fake_model_map;
    const int first_source_registered_size = (int)g_model_registered_size;
    const int first_source_registered = g_model_registered;
    const int first_source_direct_align = g_model_direct_align;
    const int first_source_file_size = (int)g_model_file_size;
    const int first_source_fd = g_model_fd, first_direct_fd = g_model_direct_fd;
    const int retry = call_cleanup("retry_checked_result");
    std::printf("late_arm_admitted=1\nlate_armed_ordinal=%d\nlate_armed_target_device=%d\nlate_first=%d\nlate_first_fault_hits=%d\nlate_first_fault_id=%d\nlate_first_set_ordinal=%d\nlate_first_sync_ordinal=%d\nlate_first_sync_device=%d\nlate_first_target_mismatches=%d\nlate_first_g_n_gpus=%d\nlate_first_gpu0_device=%d\nlate_first_gpu1_device=%d\nlate_first_current_device=%d\nlate_first_affinity=%d\nlate_first_physical_live=%d\nlate_first_state_refs=%d\nlate_first_unowned=%d\n",armed_late_ordinal,armed_late_device,first,first_fault,first_id,first_set,first_sync,first_sync_device,first_mismatch,first_n,first_gpu0,first_gpu1,first_current_device,first_affinity,first_physical,first_refs,first_unowned);
    std::printf("late_first_physical_300=%d\nlate_first_physical_301=%d\nlate_first_physical_400=%d\nlate_first_physical_401=%d\nlate_first_physical_500=%d\nlate_first_physical_501=%d\nlate_first_physical_600=%d\nlate_first_physical_601=%d\nlate_first_physical_650=%d\nlate_first_physical_651=%d\nlate_first_physical_700=%d\nlate_first_physical_750=%d\nlate_first_physical_751=%d\nlate_first_physical_800=%d\nlate_first_physical_801=%d\nlate_first_physical_820=%d\nlate_first_physical_830=%d\nlate_first_physical_900=%d\nlate_first_physical_910=%d\nlate_first_physical_950=%d\nlate_first_physical_1000=%d\nlate_first_physical_1001=%d\nlate_first_physical_1050=%d\nlate_first_physical_1100=%d\nlate_first_physical_1150=%d\nlate_first_physical_1200=%d\nlate_first_physical_1210=%d\nlate_first_physical_1250=%d\nlate_first_physical_1260=%d\nlate_first_physical_1300=%d\n",p300,p301,p400,p401,p500,p501,p600,p601,p650,p651,p700,p750,p751,p800,p801,p820,p830,p900,p910,p950,p1000,p1001,p1050,p1100,p1150,p1200,p1210,p1250,p1260,p1300);
    std::printf("late_first_source_base_retained=%d\nlate_first_source_registered_size=%d\nlate_first_source_registered=%d\nlate_first_source_direct_align=%d\nlate_first_source_file_size=%d\nlate_first_source_fd=%d\nlate_first_direct_fd=%d\nlate_retry=%d\nlate_retry_sticky_same=%d\nlate_retry_first_refusal_same=%d\n",first_source_base_retained,first_source_registered_size,first_source_registered,first_source_direct_align,first_source_file_size,first_source_fd,first_direct_fd,retry,sticky_cuda_error==first_sticky,first_refusal_index==first_refusal);
    return finish_case();
}
static int dispatch_case(const char *name) {
    reset_state();
    if (!std::strcmp(name,"score-graph-reuse")) return graph_reuse_case(false);
    if (!std::strcmp(name,"moe-graph-reuse")) return graph_reuse_case(true);
    if (!std::strcmp(name,"resize-refusal")) return selected_resize_case(true);
    if (!std::strcmp(name,"resize-success")) return selected_resize_case(false);
    if (!std::strcmp(name,"selected-begin-reuse")) return selected_begin_reuse_case();
    if (!std::strcmp(name,"cache-selection-failure") ||
        !std::strcmp(name,"represented-device-sync-failure") ||
        !std::strcmp(name,"cache-sync-failure") ||
        !std::strcmp(name,"staging-restore-failure") ||
        !std::strcmp(name,"global-restore-failure") ||
        !std::strcmp(name,"final-restore-failure")) {
#if CHECKED_AVAILABLE
        return late_device_case(name);
#else
        return 2;
#endif
    }
    if (!std::strcmp(name,"empty")) {
        seed_empty(); call_cleanup("checked_result");
#if CHECKED_AVAILABLE
        call_legacy_cleanup();
#endif
        return finish_case();
    }
    if (!std::strcmp(name,"unknown")) {
        seed_empty(); add_unknown_selected(); call_cleanup("checked_result"); return finish_case();
    }
    if (!std::strcmp(name,"compact-failure")) {
        seed_full(); g_laguna_compact_state.store(1); g_compact_fail_once=1; call_cleanup("checked_result"); return finish_case();
    }
    if (!std::strcmp(name,"tracker-stage") || !std::strcmp(name,"tracker-stage-failure")) {
        add_tracker_stage();
        if(!std::strcmp(name,"tracker-stage-failure")) { fail_kind=K_HOST; fail_id=1400; }
        call_cleanup("checked_result");
        std::printf("tracker_first_live=%d\ntracker_first_violation=%d\ntracker_first_physical=%d\n",tracker_live(),tracker.violation,owner_live_for(K_HOST,1400));
        if(!std::strcmp(name,"tracker-stage-failure")) {
            call_cleanup("retry_checked_result");
            std::printf("tracker_retry_live=%d\ntracker_retry_physical=%d\n",tracker_live(),owner_live_for(K_HOST,1400));
        }
        return finish_case();
    }
    if (!std::strcmp(name,"fd-eio") || !std::strcmp(name,"fd-eintr")) {
        seed_full(); fail_close_errno=!std::strcmp(name,"fd-eio")?EIO:EINTR;
        call_cleanup("checked_result");
        std::printf("first_close_attempted=%d\nfirst_close_effects=%d\nfirst_close_slot=%d\nfirst_close_errno=%d\nfirst_physical_1300=%d\nfirst_prefetch_present=%d\nfirst_checked_result=%d\n",close_attempted,close_effects,g_model_direct_fd,fake_errno_last,owner_live_for(K_STREAM,1300),g_model_prefetch_stream!=NULL,last_checked_result);
        reuse_fd();
        call_cleanup("retry_checked_result");
        std::printf("retry_close_attempted=%d\nretry_close_slot=%d\nretry_checked=%d\n",close_attempted,g_model_direct_fd,last_checked_result);
        return finish_case(1);
    }
    if (!std::strcmp(name,"prefix-retry")) {
        reset_state(); current_device=3;
        cuda_model_range r0{}; r0.host_base=fake_model_map; r0.offset=1; r0.bytes=1; r0.device_ptr=(char*)fake_handle(K_DEV,1600); register_ptr(K_DEV,1600,-1,0); g_model_ranges.push_back(r0);
        cuda_model_range r1{}; r1.host_base=fake_model_map; r1.offset=2; r1.bytes=1; r1.device_ptr=(char*)fake_handle(K_DEV,1601); register_ptr(K_DEV,1601,-1,0); g_model_ranges.push_back(r1); g_model_range_bytes=2; fail_kind=K_DEV; fail_id=1601;
        call_cleanup("checked_result"); std::printf("prefix_first_completed_live=%d\nprefix_first_failed_live=%d\nprefix_first_checked=%d\n",owner_live_for(K_DEV,1600),owner_live_for(K_DEV,1601),last_checked_result);
        call_cleanup("retry_checked_result"); std::printf("prefix_retry_live=%d\n",owner_live_for(K_DEV,1601)); return finish_case();
    }
    if (!std::strcmp(name,"full")) { seed_full(); }
    else {
        seed_full();
        if (!std::strcmp(name,"sync-failure")) arm_named("sync");
        else if (!std::strcmp(name,"graph-exec-failure")) arm_named("graph-exec");
        else if (!std::strcmp(name,"graph-failure")) arm_named("graph");
        else if (!std::strcmp(name,"moe-exec-failure")) arm_named("moe-exec");
        else if (!std::strcmp(name,"moe-graph-failure")) arm_named("moe-graph");
        else if (!std::strcmp(name,"event-failure")) arm_named("event");
        else if (!std::strcmp(name,"event-second-failure")) arm_named("event-second");
        else if (!std::strcmp(name,"stream-failure")) arm_named("stream");
        else if (!std::strcmp(name,"stream-second-failure")) arm_named("stream-second");
        else if (!std::strcmp(name,"blas-failure")) arm_named("blas");
        else if (!std::strcmp(name,"blas-second-failure")) arm_named("blas-second");
        else if (!std::strcmp(name,"scratch-failure")) arm_named("scratch");
        else if (!std::strcmp(name,"scratch-second-failure")) arm_named("scratch-second");
        else if (!std::strcmp(name,"bounce-failure")) arm_named("bounce");
        else if (!std::strcmp(name,"selected-cache-failure")) arm_named("selected-cache");
        else if (!std::strcmp(name,"selected-stage-event-failure")) arm_named("selected-stage-event");
        else if (!std::strcmp(name,"model-stage-event-failure")) arm_named("model-stage-event");
        else if (!std::strcmp(name,"selected-stage-host-failure")) arm_named("selected-stage-host");
        else if (!std::strcmp(name,"model-stage-failure")) arm_named("model-stage");
        else if (!std::strcmp(name,"model-upload-failure")) arm_named("model-upload");
        else if (!std::strcmp(name,"selected-stage-upload-failure")) arm_named("selected-stage-upload");
        else if (!std::strcmp(name,"range-failure")) arm_named("range");
        else if (!std::strcmp(name,"arena-failure")) arm_named("arena");
        else if (!std::strcmp(name,"q8-f16-failure")) arm_named("q8");
        else if (!std::strcmp(name,"q8-f32-failure")) arm_named("q8-f32");
        else if (!std::strcmp(name,"tmp-failure")) arm_named("tmp");
        else if (!std::strcmp(name,"model-device-failure")) arm_named("model-device");
        else if (!std::strcmp(name,"registration-failure")) arm_named("registration");
        else if (!std::strcmp(name,"prefetch-failure")) arm_named("prefetch");
        else if (!std::strcmp(name,"device-selection-failure")) arm_named("select-device");
        else if (!std::strcmp(name,"device-query-failure")) arm_named("get-device");
        else return 2;
    }
    call_cleanup("checked_result"); return finish_case();
}
int main(int argc, char **argv) {
    if (argc != 2) return 2;
    return dispatch_case(argv[1]);
}

'''


def generated_source(checked: bool) -> str:
    """Assemble only source-selected definitions; never fabricate checked code."""
    if checked and CHECKED_BODY is None:
        raise AssertionError("checked source requested before the actual API exists")
    deps = "\n\n".join(DEPENDENCIES[k] for k in (
        "cuda_ok", "resident_fail", "resident_site", "resident_site_matches",
        "resident_record", "resident_has_relation", "resident_note_failure",
        "resident_release", "resident_free", "resident_free_host",
        "load_progress_reset", "set_current_device") if DEPENDENCIES[k])
    helper_values = list(HELPERS.values()) + [v for v in OPTIONAL_HELPERS.values() if v]
    helpers = "\n\n".join(v for v in helper_values if v)
    # Keep the caller bodies source-owned and intact.  These are the smallest
    # direct dependency closure needed by selected-cache begin-load; CUDA,
    # kernel, and file-I/O effects remain explicit host fakes below.
    direct_values = [
        COMPACT_PERMIT_CLASS, STREAM_EXPERT_TABLE_TYPE,
        DIRECT_HELPERS["model_copy_chunk_bytes"],
        DIRECT_HELPERS["discard_source_pages"],
        DIRECT_HELPERS["drop_file_pages"],
        DIRECT_HELPERS["round_down"],
        DIRECT_HELPERS["stage_usable_bytes"],
        DIRECT_HELPERS["pread_full"],
        DIRECT_HELPERS["model_stage_read"],
        DIRECT_HELPERS["align_ptr"],
        DIRECT_HELPERS["selected_stage_pool_alloc"],
        DIRECT_HELPERS["model_copy"],
        DIRECT_HELPERS["selected_cache_invalidate"],
        DIRECT_HELPERS["selected_ranges_valid"],
        DIRECT_HELPERS["selected_ensure_bytes"],
        DIRECT_HELPERS["selected_ensure_i32"],
    ]
    direct = "\n\n".join(v for v in direct_values if v)
    callers = "\n\n".join(v for v in CALLERS.values() if v)
    entries = [v for v in (CLEANUP_BODY, CHECKED_BODY) if v]
    if not entries:
        raise AssertionError("no actual cleanup entry")
    return (FAKE_PREFIX.replace("CHECKED_AVAILABLE", "1" if checked else "0")
            + "\n" + deps + "\n" + helpers + "\n" + direct + "\n"
            + callers + "\n\n" + "\n\n".join(entries) + "\n"
            + FAKE_SUFFIX.replace("CHECKED_AVAILABLE", "1" if checked else "0"))


def parse_output(stdout: str) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in stdout.splitlines():
        key, sep, value = line.partition("=")
        if sep and re.fullmatch(r"-?\d+", value): values[key] = int(value)
    return values


def v(values: dict[str, int], key: str) -> int:
    if key not in values:
        sys.stderr.write(f"fixture sensor missing {key}: {values}\n")
        raise SystemExit(125)
    return values[key]


_FIXTURE_INTERRUPTED = 0

def run_bounded_child(command: list[str], *, cwd: Path,
                      env: dict[str, str], timeout: int) -> subprocess.CompletedProcess[str]:
    """Own a compiler/native group and forbid new children after interruption."""
    if _FIXTURE_INTERRUPTED:
        sys.stderr.write(f"fixture interrupted by signal {_FIXTURE_INTERRUPTED}; no new child\n")
        raise SystemExit(125)
    process = None
    previous = {}
    def text(value: object) -> str:
        return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value or "")
    def interrupted(signum: int, _frame: object) -> None:
        global _FIXTURE_INTERRUPTED
        _FIXTURE_INTERRUPTED = signum
        raise SystemExit(128 + signum)
    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous[signum] = signal.signal(signum, interrupted)
        process = subprocess.Popen(command, cwd=cwd, env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, start_new_session=True)
        stdout, stderr = process.communicate(timeout=timeout)
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    except BaseException as exc:
        stdout, stderr = text(getattr(exc, "stdout", None)), text(getattr(exc, "stderr", None))
        if process is not None:
            try:
                if process.returncode is None:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                try:
                    stopped_out, stopped_err = process.communicate(timeout=1)
                except subprocess.TimeoutExpired:
                    if process.returncode is None:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    stopped_out, stopped_err = process.communicate(timeout=1)
                stdout, stderr = text(stopped_out) or stdout, text(stopped_err) or stderr
            except BaseException as cleanup_exc:
                sys.stderr.write(f"fixture child cleanup unproved: {cleanup_exc!r}\n")
        sys.stderr.write(f"fixture child refused: {exc!r}\n" + stdout + stderr)
        raise SystemExit(125) from exc
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


class FixtureBase(unittest.TestCase):
    checked = False

    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls._tmp = tempfile.TemporaryDirectory(prefix="ds4-cuda-cleanup-contract-")
            cls.addClassCleanup(cls._tmp.cleanup)
            base = Path(cls._tmp.name)
            cls._env = dict(SAFE_ENV)
            for key, leaf in (("HOME", "home"), ("TMPDIR", "tmp"),
                              ("XDG_CONFIG_HOME", "config"), ("XDG_CACHE_HOME", "cache"),
                              ("XDG_DATA_HOME", "data")):
                child = base / leaf
                child.mkdir(mode=0o700)
                cls._env[key] = str(child)
            cls._env["PYTHONDONTWRITEBYTECODE"] = "1"
            cls._env["PYTHONNOUSERSITE"] = "1"
            cls._source = base / "fixture.cc"
            cls._binary = base / "fixture"
            cls._source.write_text(generated_source(cls.checked), encoding="utf-8")
            cls._compile = run_bounded_child(
                ["c++", "-std=c++17", "-O0", "-pthread", "-I", str(ROOT),
                 str(cls._source), "-o", str(cls._binary)],
                cwd=base, env=cls._env, timeout=15)
        except (AssertionError, OSError, subprocess.TimeoutExpired) as exc:
            sys.stderr.write(f"fixture setup refused: {exc}\n")
            raise SystemExit(125)

    def require_fixture(self) -> None:
        if self._compile.returncode != 0:
            if self._compile.stderr:
                sys.stderr.write(self._compile.stderr)
            raise SystemExit(125)

    def run_case(self, case: str) -> dict[str, int]:
        self.require_fixture()
        try:
            result = run_bounded_child([str(self._binary), case], cwd=self._binary.parent,
                                       env=self._env, timeout=10)
        except (OSError, subprocess.TimeoutExpired) as exc:
            sys.stderr.write(f"fixture child {case} refused: {exc}\n")
            raise SystemExit(125)
        if result.returncode != 0:
            sys.stderr.write(f"fixture infrastructure refusal: {case}; exit={result.returncode}\n"
                             + result.stdout + result.stderr)
            raise SystemExit(125)
        values = parse_output(result.stdout)
        self.assertEqual(v(values, "trace_overflow"), 0)
        self.assertEqual(v(values, "effect_order_errors"), 0)
        self.assertEqual(v(values, "driver_protocol_errors"), 0)
        self.assertEqual(v(values, "rescue_protocol_errors"), 0)
        self.assertEqual(v(values, "rescue_tracker_errors"), 0)
        self.assertEqual(v(values, "tracker_release_errors"), 0)
        self.assertEqual(v(values, "rescue_complete"), 1)
        self.assertEqual(v(values, "rescue_sticky_same"), 1)
        self.assertEqual(v(values, "rescue_budget_same"), 1)
        return values


class SourceContract(unittest.TestCase):
    def test_actual_body_selection_and_separate_checked_status(self) -> None:
        if CLEANUP_BODY is None or MISSING_SEAMS:
            raise SystemExit(125)
        # Baseline absence is a reportable source fact, not a fabricated checked path.
        self.assertNotIn("int ds4_gpu_cleanup_checked(void) {", FAKE_PREFIX + FAKE_SUFFIX)
        if CHECKED_BODY is not None:
            self.assertIn(CHECKED_BODY, generated_source(True))
        if OPTIONAL_HELPERS["q8_f32_release"] is not None:
            self.assertIn(OPTIONAL_HELPERS["q8_f32_release"], generated_source(False))
        self.assertIn("extern \"C\" void ds4_gpu_cleanup", CLEANUP_BODY)
        self.assertIsNotNone(DEPENDENCIES["cuda_ok"])
        self.assertIn(DEPENDENCIES["cuda_ok"], generated_source(False))
        for seam in ("g_xdev_bounce", "g_xdev_bounce_bytes", "g_stream_selected_upload_stream", "cuda_laguna_compact_destroy_checked"):
            self.assertIn(seam, FAKE_PREFIX)
        self.assertIn("physical_retire_ptr", FAKE_SUFFIX)
        self.assertIn("physical_retire_fd", FAKE_SUFFIX)

    def test_frozen_baseline_identity_is_not_silently_changed(self) -> None:
        self.assertEqual(BASELINE_COMMIT, "3711f489a5aef83ea9b747467bd173ef05cc6adc")
        self.assertEqual(BASELINE_TREE, "cf770eae24cbc08b7a8c830d1663339df1c59f5c")
        self.assertEqual(SCOPE_SHA256, "5292e916e84ea46952aea2f9c5e98ae0467d9371432feba6357857d4060e6fa4")
        self.assertEqual(FD_DECISION_SHA256, "f6623630bae7ced3d4f9d75c076563fe1f520372a8a7e3efc62fc25b5dd2d025")
        self.assertEqual(REPAIR_SCOPE_SHA256, "bd27ea7801368d305d03aeba41041e54a17912d2ae9b60ea86f7bec3815e628c")
        self.assertEqual(REPAIR_CONTRACT_SHA256, "799a93d4fd8cd66a4cb9ea84d440ad876355fdae8bcc2fbf84557902a8144c18")
        self.assertEqual(FROZEN_FIXTURE_SHA256, "f36747dface38b97e8602818df48ff6287daa32ce9a0278b682e9b22db5c6052")
        self.assertEqual(FROZEN_REPORT_SHA256, "cd63b8d3aa3c1bb73b3a824e1ded3c22a5e129b78042fc6cfb8c94d7e0523b27")

    def test_reuse_callers_and_dependencies_are_actual_source_bodies(self) -> None:
        if MISSING_SEAMS:
            raise SystemExit(125)
        selected = (
            DIRECT_HELPERS["selected_ensure_bytes"],
            CALLERS["score_graph_launch"], CALLERS["moe_graph_launch"],
            CALLERS["selected_cache_begin_load"],
            DIRECT_HELPERS["model_copy"],
        )
        for body in selected:
            self.assertIsNotNone(body)
            self.assertIn(body, generated_source(True))
        self.assertIn(STREAM_EXPERT_TABLE_TYPE, generated_source(True))
        self.assertIn(COMPACT_PERMIT_CLASS, generated_source(True))


class BaselineVoidCleanup(FixtureBase):
    """Runs actual void cleanup; contract violations are genuine baseline RED."""

    def test_empty_and_legacy_projection_control(self) -> None:
        values = self.run_case("empty")
        self.assertEqual(v(values, "generic_attempts"), 1)
        self.assertEqual(v(values, "physical_live"), 0)
        self.assertEqual(v(values, "unowned_live"), 0)
        self.assertEqual(v(values, "first_refusal"), -1)

    def test_unknown_handle_is_refused_without_driver_protocol_failure(self) -> None:
        values = self.run_case("unknown")
        self.assertEqual(v(values, "unknown_refusals"), 1)
        self.assertEqual(v(values, "driver_protocol_errors"), 0)
        self.assertGreaterEqual(v(values, "first_refusal"), 0)

    def test_full_teardown_has_no_lost_owner_or_unrelated_refusal(self) -> None:
        values = self.run_case("full")
        self.assertEqual(v(values, "generic_attempts"), 1)
        self.assertEqual(v(values, "unowned_live"), 0)
        self.assertEqual(v(values, "physical_live"), 0)
        self.assertEqual(v(values, "refused_total"), 0)
        self.assertEqual(v(values, "g_n_gpus"), 0)

    def test_named_fault_reaches_the_requested_live_owner_and_is_retained(self) -> None:
        targets = {
            "graph-exec-failure": 100, "graph-failure": 200,
            "moe-exec-failure": 110, "moe-graph-failure": 210,
            "event-failure": 300, "event-second-failure": 301,
            "stream-failure": 400, "stream-second-failure": 401,
            "blas-failure": 500, "blas-second-failure": 501,
            "scratch-failure": 600, "scratch-second-failure": 601,
            "bounce-failure": 650, "selected-cache-failure": 700,
            "q8-f16-failure": 900, "q8-f32-failure": 910,
            "model-device-failure": 1250, "registration-failure": 1260,
            "prefetch-failure": 1300, "device-selection-failure": 7,
        }
        for case, target in targets.items():
            with self.subTest(case=case):
                values = self.run_case(case)
                self.assertEqual(v(values, "fault_hits"), 1)
                self.assertEqual(v(values, "fault_id"), target)
                self.assertGreaterEqual(v(values, "first_refusal"), 0)
                self.assertEqual(v(values, "unowned_live"), 0)

    def test_sync_and_query_are_non_destructive_gates(self) -> None:
        with self.subTest(case="sync-failure"):
            sync = self.run_case("sync-failure")
            self.assertEqual(v(sync, "fault_hits"), 1)
            self.assertEqual(v(sync, "fault_id"), -100)
            self.assertEqual(v(sync, "sync_refused"), 1)
            self.assertEqual(v(sync, "attempt_total"), 0)
        with self.subTest(case="device-query-failure"):
            query = self.run_case("device-query-failure")
            self.assertEqual(v(query, "fault_hits"), 1)
            self.assertEqual(v(query, "fault_id"), -101)
            self.assertEqual(v(query, "physical_750"), 1)
            self.assertEqual(v(query, "physical_751"), 1)

    def test_checked_helpers_stop_suffix_and_keep_completed_prefix(self) -> None:
        for case in ("selected-stage-event-failure", "selected-stage-host-failure",
                     "model-stage-event-failure", "model-stage-failure", "model-upload-failure",
                     "selected-stage-upload-failure"):
            with self.subTest(case=case):
                values = self.run_case(case)
                self.assertEqual(v(values, "fault_hits"), 1)
                self.assertEqual(v(values, "unowned_live"), 0)
                self.assertEqual(v(values, "physical_1250"), 1)
        ranges = self.run_case("range-failure")
        self.assertEqual(v(ranges, "physical_800"), 0)
        self.assertEqual(v(ranges, "physical_801"), 1)
        self.assertEqual(v(ranges, "physical_900"), 1)
        self.assertEqual(v(ranges, "physical_1250"), 1)
        arena = self.run_case("arena-failure")
        self.assertEqual(v(arena, "physical_830"), 1)
        self.assertEqual(v(arena, "physical_900"), 1)
        tmp = self.run_case("tmp-failure")
        self.assertEqual(v(tmp, "physical_1250"), 1)

    def test_compact_refusal_is_recorded_without_fake_success(self) -> None:
        values = self.run_case("compact-failure")
        self.assertEqual(v(values, "fault_hits"), 1)
        self.assertEqual(v(values, "fault_id"), -300)
        self.assertEqual(v(values, "compact_refused"), 1)
        self.assertEqual(v(values, "compact_pending"), 1)
        self.assertEqual(v(values, "unowned_live"), 0)

    def test_completed_prefix_is_not_replayed_on_retry(self) -> None:
        values = self.run_case("prefix-retry")
        self.assertEqual(v(values, "prefix_first_completed_live"), 0)
        self.assertEqual(v(values, "prefix_first_failed_live"), 1)
        self.assertEqual(v(values, "prefix_retry_live"), 0)
        self.assertEqual(v(values, "attempt_dev"), 3)
        self.assertEqual(v(values, "completed_dev"), 2)

    def test_attached_resident_release_keeps_sticky_failure_and_retries(self) -> None:
        values = self.run_case("tracker-stage-failure")
        self.assertNotEqual(v(values, "tracker_violation"), 0)
        self.assertEqual(v(values, "tracker_first_live"), 1)
        self.assertEqual(v(values, "tracker_retry_live"), 0)
        self.assertEqual(v(values, "tracker_live"), 0)
        self.assertEqual(v(values, "generic_attempts"), 2)
        self.assertEqual(v(values, "cuda_last_error"), 0)

    def test_linux_close_records_consumption_and_reuse_before_rescue(self) -> None:
        for case in ("fd-eio", "fd-eintr"):
            with self.subTest(case=case):
                values = self.run_case(case)
                self.assertEqual(v(values, "first_close_attempted"), 1)
                self.assertEqual(v(values, "first_close_effects"), 1)
                self.assertEqual(v(values, "first_close_slot"), -1)
                self.assertEqual(v(values, "first_prefetch_present"), 1)
                self.assertEqual(v(values, "first_close_errno"), 5 if case == "fd-eio" else 4)
                self.assertEqual(v(values, "retry_close_attempted"), 1)
                self.assertEqual(v(values, "reuse_live_before_rescue"), 1)
                self.assertEqual(v(values, "close_slot"), -1)
                self.assertEqual(v(values, "close_attempted"), 1)


@unittest.skipUnless(CHECKED_BODY is not None, "checked API missing at authenticated baseline; source status is reported separately")
class CheckedCleanup(FixtureBase):
    checked = True

    def test_empty_checks_both_entries_and_one_attempt_per_call(self) -> None:
        values = self.run_case("empty")
        self.assertEqual(v(values, "checked_result"), 1)
        self.assertEqual(v(values, "legacy_called"), 1)
        self.assertEqual(v(values, "generic_attempts"), 2)
        self.assertEqual(v(values, "physical_live"), 0)

    def test_named_refusal_returns_zero_and_retains_failed_unvisited_state(self) -> None:
        targets = {
            "sync-failure": -100, "device-query-failure": -101,
            "graph-exec-failure": 100, "graph-failure": 200,
            "moe-exec-failure": 110, "moe-graph-failure": 210,
            "event-failure": 300, "event-second-failure": 301,
            "stream-failure": 400, "blas-failure": 500,
            "scratch-failure": 600, "bounce-failure": 650,
            "selected-cache-failure": 700, "selected-stage-event-failure": 1150,
            "model-stage-event-failure": 1050, "selected-stage-host-failure": 1100,
            "model-stage-failure": 1000, "model-upload-failure": 1200,
            "selected-stage-upload-failure": 1210, "range-failure": 801,
            "arena-failure": 830, "q8-f16-failure": 900, "q8-f32-failure": 910,
            "tmp-failure": 950, "model-device-failure": 1250,
            "registration-failure": 1260, "prefetch-failure": 1300,
            "device-selection-failure": 7, "compact-failure": -300,
        }
        for case, target in targets.items():
            with self.subTest(case=case):
                values = self.run_case(case)
                self.assertEqual(v(values, "checked_result"), 0)
                self.assertEqual(v(values, "fault_hits"), 1)
                self.assertEqual(v(values, "fault_id"), target)
                self.assertEqual(v(values, "unowned_live"), 0)
                self.assertGreaterEqual(v(values, "first_refusal"), 0)
                self.assertEqual(v(values, "g_n_gpus"), 2)

    def test_checked_unknown_handle_returns_zero(self) -> None:
        values = self.run_case("unknown")
        self.assertEqual(v(values, "checked_result"), 0)
        self.assertEqual(v(values, "unknown_refusals"), 1)
        self.assertEqual(v(values, "unowned_live"), 0)

    def test_success_releases_all_resources_and_preserves_affinity(self) -> None:
        values = self.run_case("full")
        self.assertEqual(v(values, "checked_result"), 1)
        self.assertEqual(v(values, "physical_live"), 0)
        self.assertEqual(v(values, "unowned_live"), 0)
        self.assertEqual(v(values, "refused_total"), 0)
        self.assertEqual(v(values, "g_n_gpus"), 0)

    def test_prefix_retry_returns_zero_then_success_without_erasing_sticky_failure(self) -> None:
        values = self.run_case("prefix-retry")
        self.assertEqual(v(values, "checked_result"), 0)
        self.assertEqual(v(values, "retry_checked_result"), 1)
        self.assertEqual(v(values, "prefix_first_completed_live"), 0)
        self.assertEqual(v(values, "prefix_first_failed_live"), 1)
        self.assertEqual(v(values, "prefix_retry_live"), 0)
        self.assertNotEqual(v(values, "sticky_cuda_error"), 0)
        self.assertEqual(v(values, "attempt_dev"), 3)
        self.assertEqual(v(values, "completed_dev"), 2)

    def test_attached_resident_refusal_and_retry_keep_tracker_unsafe(self) -> None:
        values = self.run_case("tracker-stage-failure")
        self.assertEqual(v(values, "checked_result"), 0)
        self.assertEqual(v(values, "retry_checked_result"), 1)
        self.assertEqual(v(values, "tracker_first_live"), 1)
        self.assertEqual(v(values, "tracker_retry_live"), 0)
        self.assertNotEqual(v(values, "tracker_violation"), 0)
        self.assertEqual(v(values, "rescue_sticky_same"), 1)

    def test_linux_close_error_is_consumed_once_then_retry_avoids_reuse(self) -> None:
        for case in ("fd-eio", "fd-eintr"):
            with self.subTest(case=case):
                values = self.run_case(case)
                self.assertEqual(v(values, "first_checked_result"), 0)
                self.assertEqual(v(values, "retry_checked_result"), 1)
                self.assertEqual(v(values, "first_close_attempted"), 1)
                self.assertEqual(v(values, "first_close_effects"), 1)
                self.assertEqual(v(values, "first_close_slot"), -1)
                expected_errno = 5 if case == "fd-eio" else 4
                self.assertEqual(v(values, "first_close_errno"), expected_errno)
                self.assertEqual(v(values, "fake_errno"), expected_errno)
                self.assertEqual(v(values, "first_prefetch_present"), 1)
                self.assertEqual(v(values, "retry_close_attempted"), 1)
                self.assertEqual(v(values, "reuse_live_before_rescue"), 1)
                self.assertEqual(v(values, "physical_1300"), 0)


@unittest.skipUnless(CHECKED_BODY is not None, "checked API missing at authenticated baseline; source status is reported separately")
class ReuseFollowup(FixtureBase):
    checked = True

    def test_selected_cache_resize_refusal_is_a_behavioral_red(self) -> None:
        values = self.run_case("resize-refusal")
        self.assertEqual(v(values, "resize_result"), 0)
        self.assertEqual(v(values, "resize_old_live"), 1)
        self.assertEqual(v(values, "resize_old_represented"), 1)
        self.assertEqual(v(values, "resize_field_id"), 1700)
        self.assertEqual(v(values, "resize_field_known"), 1)
        self.assertEqual(v(values, "resize_field_live"), 1)
        self.assertEqual(v(values, "resize_capacity"), 8)
        self.assertEqual(v(values, "resize_malloc_attempts"), 0)
        self.assertEqual(v(values, "resize_unowned_before_rescue"), 0)

    def test_selected_cache_healthy_resize_control(self) -> None:
        values = self.run_case("resize-success")
        self.assertEqual(v(values, "resize_result"), 1)
        self.assertEqual(v(values, "resize_old_live"), 0)
        self.assertEqual(v(values, "resize_field_id"), 17000)
        self.assertEqual(v(values, "resize_field_known"), 1)
        self.assertEqual(v(values, "resize_field_live"), 1)
        self.assertEqual(v(values, "resize_capacity"), 16)
        self.assertEqual(v(values, "resize_malloc_attempts"), 1)
        self.assertEqual(v(values, "resize_unowned_before_rescue"), 0)

    def test_score_and_moe_graph_callers_block_retained_destroy_reuse(self) -> None:
        for case in ("score-graph-reuse", "moe-graph-reuse"):
            with self.subTest(case=case):
                values = self.run_case(case)
                self.assertEqual(v(values, "graph_first"), 1)
                self.assertEqual(v(values, "graph_first_shape_live"), 1)
                self.assertEqual(v(values, "graph_refused"), 0)
                self.assertEqual(v(values, "graph_first_marker"), -1)
                self.assertEqual(v(values, "graph_first_graph_live"), 1)
                self.assertEqual(v(values, "graph_first_exec_live"), 1)
                self.assertEqual(v(values, "graph_blocked"), -1)
                self.assertEqual(v(values, "graph_blocked_marker"), -1)
                self.assertEqual(v(values, "graph_blocked_graph_live"), 1)
                self.assertEqual(v(values, "graph_blocked_exec_live"), 1)
                self.assertEqual(v(values, "graph_retained_after_block"), 1)
                self.assertEqual(v(values, "graph_create_after_refusal"), 1)
                self.assertEqual(v(values, "graph_create_after_block"), 1)
                self.assertEqual(v(values, "graph_update_after_block"), 0)
                self.assertEqual(v(values, "graph_launch_after_block"), 1)
                self.assertEqual(v(values, "graph_refused_after_exec"), 0)
                self.assertEqual(v(values, "graph_graph_refusal_marker"), -1)
                self.assertEqual(v(values, "graph_exec_live_after_graph_refusal"), 0)
                self.assertEqual(v(values, "graph_graph_live_after_graph_refusal"), 1)
                self.assertEqual(v(values, "graph_no_effect_while_retained"), 1)
                self.assertEqual(v(values, "graph_create_after_graph_refusal"), 1)
                self.assertEqual(v(values, "graph_update_after_graph_refusal"), 0)
                self.assertEqual(v(values, "graph_launch_after_graph_refusal"), 1)
                self.assertEqual(v(values, "graph_retry"), 1)
                self.assertEqual(v(values, "graph_retry_new_graph"), 1)
                self.assertEqual(v(values, "graph_retry_new_exec"), 1)
                self.assertEqual(v(values, "graph_retry_old_graph_live"), 0)
                self.assertEqual(v(values, "graph_retry_old_exec_live"), 0)
                self.assertEqual(v(values, "graph_retry_unowned"), 0)
                self.assertEqual(v(values, "graph_create_calls"), 2)
                self.assertEqual(v(values, "graph_instantiate_calls"), 2)
                self.assertEqual(v(values, "graph_update_calls"), 0)
                self.assertEqual(v(values, "graph_launch_calls"), 2)
                self.assertNotEqual(v(values, "graph_first_sticky"), 0)
                self.assertEqual(v(values, "graph_sticky_same"), 1)
                self.assertEqual(v(values, "graph_first_refusal_same"), 1)
                self.assertGreaterEqual(v(values, "graph_first_refusal"), 0)

    def test_selected_cache_begin_load_honors_retained_negative_marker(self) -> None:
        values = self.run_case("selected-begin-reuse")
        self.assertEqual(v(values, "begin_release_refused"), 0)
        self.assertEqual(v(values, "begin_after_release_valid"), -1)
        self.assertEqual(v(values, "begin_after_release_ptr_id"), 1800)
        self.assertEqual(v(values, "begin_after_release_capacity"), 8)
        self.assertEqual(v(values, "begin_after_release_old_live"), 1)
        self.assertEqual(v(values, "begin_after_release_malloc_attempts"), 0)
        self.assertEqual(v(values, "begin_invalidated_valid"), -1)
        self.assertEqual(v(values, "begin_invalidated_ptr_id"), 1800)
        self.assertEqual(v(values, "begin_invalidated_old_live"), 1)
        self.assertEqual(v(values, "begin_first"), 0)
        self.assertEqual(v(values, "begin_first_valid"), -1)
        self.assertEqual(v(values, "begin_first_ptr_id"), 1800)
        self.assertEqual(v(values, "begin_first_capacity"), 8)
        self.assertEqual(v(values, "begin_first_old_live"), 1)
        self.assertEqual(v(values, "begin_first_malloc_attempts"), 0)
        self.assertEqual(v(values, "begin_first_unowned"), 0)
        self.assertEqual(v(values, "begin_retry"), 1)
        self.assertEqual(v(values, "begin_retry_valid"), 0)
        self.assertEqual(v(values, "begin_retry_ptr_id"), -1)
        self.assertEqual(v(values, "begin_retry_capacity"), 0)
        self.assertEqual(v(values, "begin_retry_old_live"), 0)
        self.assertEqual(v(values, "begin_retry_sticky_same"), 1)
        self.assertEqual(v(values, "begin_retry_first_refusal_same"), 1)

    def test_late_device_gates_attribute_refusal_and_preserve_prefix(self) -> None:
        expected = {
            "cache-selection-failure": (7, 4, -1, 1, 1, 1, 1, 1),
            "represented-device-sync-failure": (-100, -1, 3, 1, 1, 1, 1, 1),
            "cache-sync-failure": (-100, -1, 5, 1, 1, 1, 1, 1),
            "staging-restore-failure": (3, 7, -1, 1, 1, 1, 1, 1),
            "global-restore-failure": (3, 11, -1, 0, 0, 1, 1, 1),
            "final-restore-failure": (3, 16, -1, 0, 0, 0, 0, 0),
        }
        for case, (fault_id, set_ordinal, sync_ordinal,
                   p700, p750, p800, p1000, p1300) in expected.items():
            with self.subTest(case=case):
                values = self.run_case(case)
                self.assertEqual(v(values, "late_arm_admitted"), 1)
                self.assertEqual(v(values, "late_armed_ordinal"), set_ordinal if set_ordinal > 0 else sync_ordinal)
                self.assertEqual(v(values, "late_armed_target_device"), 7 if case in ("cache-selection-failure", "represented-device-sync-failure", "cache-sync-failure") else 3)
                self.assertEqual(v(values, "late_first"), 0)
                self.assertEqual(v(values, "late_first_fault_hits"), 1)
                self.assertEqual(v(values, "late_first_fault_id"), fault_id)
                self.assertEqual(v(values, "late_first_set_ordinal"), set_ordinal)
                self.assertEqual(v(values, "late_first_sync_ordinal"), sync_ordinal)
                self.assertEqual(v(values, "late_first_sync_device"), 7 if sync_ordinal > 0 else -1)
                self.assertEqual(v(values, "late_first_target_mismatches"), 0)
                self.assertEqual(v(values, "late_first_g_n_gpus"), 2)
                self.assertEqual(v(values, "late_first_gpu0_device"), 3)
                self.assertEqual(v(values, "late_first_gpu1_device"), 7)
                self.assertEqual(v(values, "late_first_affinity"), 1)
                self.assertEqual(v(values, "late_first_unowned"), 0)
                self.assertEqual(v(values, "late_first_physical_700"), p700)
                self.assertEqual(v(values, "late_first_physical_750"), p750)
                self.assertEqual(v(values, "late_first_physical_800"), p800)
                self.assertEqual(v(values, "late_first_physical_1000"), p1000)
                self.assertEqual(v(values, "late_first_physical_1300"), p1300)
                self.assertEqual(v(values, "late_first_source_base_retained"), 1)
                self.assertEqual(v(values, "late_first_source_registered_size"), 256)
                self.assertEqual(v(values, "late_first_source_registered"), 0 if case == "final-restore-failure" else 1)
                self.assertEqual(v(values, "late_first_direct_fd"), -1 if case == "final-restore-failure" else 41)
                self.assertEqual(v(values, "late_first_source_direct_align"), 1)
                self.assertEqual(v(values, "late_first_source_file_size"), 0)
                self.assertEqual(v(values, "late_first_source_fd"), -1)
                self.assertEqual(v(values, "late_retry"), 1)
                self.assertEqual(v(values, "late_retry_sticky_same"), 1)
                self.assertEqual(v(values, "late_retry_first_refusal_same"), 1)


if __name__ == "__main__":
    program = unittest.main(verbosity=2, exit=False)
    # unittest catches SystemExit raised inside a method. Preserve the fixture
    # infrastructure status instead of presenting its errors as product RED.
    result = program.result
    sys.exit(125 if result.errors else (0 if result.wasSuccessful() else 1))
