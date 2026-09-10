#!/usr/bin/env python3
"""Failing-test-first host fixture for the root-selected checked release.

Runtime paths are portable: the checkout root derives from this fixture location.
The frozen/new preflights, not this runtime test, hold absolute scope provenance.
The fixture extracts the actual marked C block and never supplies a release
algorithm. Its generated driver provides only CONTROL-FLOW FAKES for private
shapes and effects. It performs no real session, engine, graph, runtime, CUDA,
model, distributed, or GPU work. Physical ownership is tiny libc allocation
only (individual <= 4096 bytes and aggregate <= 65536 bytes), with known-live
sensors that retire only after the corresponding libc free. This is a contract
gate, not evidence of native graph/GPU correctness or external attribution.
"""
from __future__ import annotations

import re
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATHS = {
    "AGENT.md": ROOT / "AGENT.md",
    "ds4.c": ROOT / "ds4.c",
    "ds4.h": ROOT / "ds4.h",
    "ds4_runtime.h": ROOT / "ds4_runtime.h",
    "ds4_ssd.h": ROOT / "ds4_ssd.h",
    "Makefile": ROOT / "Makefile",
    "tests/test_cuda_build_contract.py": ROOT / "tests" / "test_cuda_build_contract.py",
    "tests/test_cuda_resident_host_contract.py": ROOT / "tests" / "test_cuda_resident_host_contract.py",
}
DS4_SOURCE_PATH = ROOT / "ds4.c"
DS4_HEADER_PATH = ROOT / "ds4.h"
MAKEFILE_PATH = ROOT / "Makefile"
SAFE_ENV = {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"}

SESSION_START_MARKER = "/* Session checked release. */"
SESSION_END_MARKER = "/* End session checked release. */"
DS4_SOURCE = DS4_SOURCE_PATH.read_text(encoding="utf-8")
DS4_HEADER = DS4_HEADER_PATH.read_text(encoding="utf-8")
MAKEFILE_SOURCE = MAKEFILE_PATH.read_text(encoding="utf-8")


def extract_marked_block(source: str, start_marker: str, end_marker: str) -> str | None:
    """Extract whole source lines; prose or indented marker hits are invalid."""
    start = source.find(start_marker)
    if start < 0:
        return None
    line_start = source.rfind("\n", 0, start) + 1
    if source[line_start:start].strip():
        return None
    end = source.find(end_marker, start + len(start_marker))
    if end < 0 or source[source.rfind("\n", 0, end) + 1:end].strip():
        return None
    end_line = source.find("\n", end)
    return source[line_start:end_line if end_line >= 0 else len(source)]


SESSION_BLOCK = extract_marked_block(
    DS4_SOURCE, SESSION_START_MARKER, SESSION_END_MARKER
)
HEADER_DECLARATION = re.compile(
    r"(?m)^\s*int\s+ds4_session_free_checked\s*"
    r"\(\s*ds4_session\s*\*\s*\*\s*owner\s*\)\s*;"
)
MISSING_SOURCE_SEAMS: list[str] = []
if DS4_SOURCE.count(SESSION_START_MARKER) != 1:
    MISSING_SOURCE_SEAMS.append("checked-release start marker")
if DS4_SOURCE.count(SESSION_END_MARKER) != 1:
    MISSING_SOURCE_SEAMS.append("checked-release end marker")
if SESSION_BLOCK is None:
    MISSING_SOURCE_SEAMS.append("whole-line checked-release block")
if not HEADER_DECLARATION.search(DS4_HEADER):
    MISSING_SOURCE_SEAMS.append("ds4_session_free_checked header declaration")
if SESSION_BLOCK is not None:
    if not re.search(
        r"(?m)^\s*static\s+int\s+ds4_session_release_exact_cache_reservation\s*\(",
        SESSION_BLOCK,
    ):
        MISSING_SOURCE_SEAMS.append("exact-cache reservation helper body")
    if not re.search(
        r"(?m)^\s*int\s+ds4_session_free_checked\s*\(", SESSION_BLOCK
    ):
        MISSING_SOURCE_SEAMS.append("checked release entry body")
    if not re.search(r"(?m)^\s*void\s+ds4_session_free\s*\(", SESSION_BLOCK):
        MISSING_SOURCE_SEAMS.append("legacy release wrapper body")

# The actual extracted block is the only checked-release algorithm in this TU.
# Empty block text is intentional: missing source seams fail feature RED before C work.
FIXTURE_SOURCE = (
    '/* Generated checked-release driver: all non-libc effects below are CONTROL-FLOW FAKES. */\n#include <errno.h>\n#include <pthread.h>\n#include <signal.h>\n#include <stdbool.h>\n#include <stddef.h>\n#include <stdint.h>\n#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n#include <sys/resource.h>\n#include <unistd.h>\n#include "ds4.h"\n#include "ds4_runtime.h"\n#include "ds4_ssd.h"\n#include "checked_release_caller.h"\n\nenum {\n    FAKE_FAMILY_CPU = 0,\n    FAKE_FAMILY_GLM = 1,\n    FAKE_FAMILY_LAGUNA = 2,\n    FAKE_FAMILY_GENERIC = 3,\n    FAKE_POINTER_CAP = 128,\n    FAKE_EVENT_CAP = 512,\n};\n\ntypedef struct ds4_dist_session ds4_dist_session;\ntypedef struct {\n    void *payload;\n    int active;\n} fake_graph;\ntypedef struct {\n    void *payload;\n    int active;\n} fake_glm_graph;\ntypedef struct {\n    void *payload;\n    int active;\n} fake_laguna_graph;\ntypedef struct {\n    void *payload;\n} fake_cache;\ntypedef struct {\n    void *payload;\n} fake_scratch;\ntypedef struct {\n    int *v;\n    int len;\n    int cap;\n} fake_token_vec;\nstruct ds4_dist_session {\n    void *payload;\n};\nstruct ds4_tp {\n    int failed;\n};\nstruct ds4_engine {\n    ds4_backend backend;\n    int fake_family;\n    struct {\n        struct ds4_tp *ctx;\n        int active;\n        int rank;\n    } tp;\n    pthread_mutex_t runtime_snapshot_mutex;\n    bool runtime_snapshot_mutex_initialized;\n    bool laguna_compact_runtime;\n    pthread_mutex_t exact_cache_session_mutex;\n    bool exact_cache_session_mutex_initialized;\n    uint32_t exact_cache_active_sessions;\n    uint32_t session_container_count;\n    int tracker_cookie;\n#ifdef DS4_TEST_HOOKS\n    bool test_session_create_no_alloc;\n    bool test_direct_graph_no_alloc;\n#endif\n};\nstruct ds4_session {\n    struct ds4_engine *engine;\n    ds4_dist_session *distributed;\n    uint64_t tp_session_id;\n    fake_graph graph;\n    fake_glm_graph glm_graph;\n    bool glm_graph_ready;\n    fake_laguna_graph laguna_graph;\n    bool laguna_graph_ready;\n    fake_cache cpu_cache;\n    fake_scratch cpu_scratch;\n    fake_token_vec checkpoint;\n    fake_token_vec greedy_splitkv_segment;\n    float *logits;\n    float *sample_probs;\n    float *glm_mtp_hc;\n    float *glm_mtp_logits0;\n    float *mtp_logits;\n    float *spec_row_logits;\n    float *dspark_markov_bias;\n    float *dspark_conf_features;\n    bool exact_cache_session_reserved;\n#ifdef DS4_TEST_HOOKS\n    bool test_no_alloc;\n#endif\n};\n\nstatic ds4_session **g_owner_slot;\nstatic ds4_session *g_current_session;\nstatic ds4_engine *g_mutex_engine;\nstatic int g_events[FAKE_EVENT_CAP];\nstatic size_t g_event_len;\nstatic int g_lock_refuse;\nstatic int g_unlock_refuse;\nstatic int g_graph_refuse;\nstatic int g_exact_lock_refuse;\nstatic int g_exact_unlock_refuse;\nstatic int g_exact_lock_calls;\nstatic int g_exact_unlock_calls;\nstatic int g_exact_lock_snapshot_valid;\nstatic int g_exact_lock_snapshot_reserved;\nstatic uint32_t g_exact_lock_snapshot_active;\nstatic int g_exact_unlock_entry_checks;\nstatic int g_exact_unlock_entry_violations;\nstatic int g_compact_lock_calls;\nstatic int g_compact_unlock_calls;\nstatic int g_graph_calls;\nstatic int g_graph_physical_frees;\nstatic int g_cpu_free_calls;\nstatic int g_scratch_free_calls;\nstatic int g_dist_free_calls;\nstatic int g_tp_destroy_calls;\nstatic int g_old_release_calls;\nstatic int g_container_free_calls;\nstatic uint64_t g_container_tp_id_at_free;\nstatic uintptr_t g_container_dist_at_free;\nstatic int g_container_reserved_at_free;\nstatic uint32_t g_container_active_at_free;\nstatic int g_container_common_empty_at_free;\nstatic int g_container_graph_empty_at_free;\nstatic int g_container_cpu_empty_at_free;\nstatic int g_container_laguna_ready_at_free;\nstatic int g_container_glm_ready_at_free;\nstatic int g_unknown_free_calls;\nstatic int g_double_free_calls;\nstatic int g_unknown_mutex_calls;\nstatic int g_free_order_violations;\nstatic int g_fixture_protocol_errors;\nstatic int g_sensor_sum_violations;\nstatic int g_allocation_attempts;\nstatic int g_release_boundary_calls;\nstatic int g_release_allocation_attempts;\nstatic int g_alloc_calls;\nstatic int g_malloc_calls;\nstatic int g_calloc_calls;\nstatic int g_physical_free_calls;\nstatic size_t g_alloc_bytes;\nstatic size_t g_real_current;\nstatic size_t g_real_peak;\nstatic size_t g_max_alloc;\nstatic size_t g_live_count;\nstatic int g_pointer_kind[FAKE_POINTER_CAP];\nstatic void *g_pointer[FAKE_POINTER_CAP];\nstatic size_t g_pointer_bytes[FAKE_POINTER_CAP];\nstatic int g_pointer_live[FAKE_POINTER_CAP];\nstatic int g_pointer_count;\n\nstatic void fixture_protocol_error(void) {\n    g_fixture_protocol_errors++;\n}\nstatic void fake_event(int event) {\n    if (g_event_len < FAKE_EVENT_CAP) g_events[g_event_len++] = event;\n    else fixture_protocol_error();\n}\nstatic int fake_event_is(size_t i, int event) {\n    return i < g_event_len && g_events[i] == event;\n}\nstatic int fixture_find_pointer(const void *p) {\n    for (int i = 0; i < g_pointer_count; i++) {\n        if (g_pointer[i] == p) return i;\n    }\n    return -1;\n}\nstatic int fixture_pointer_is_live(const void *p) {\n    int i = fixture_find_pointer(p);\n    return p != NULL && i >= 0 && g_pointer_live[i] != 0;\n}\nstatic size_t fixture_known_live_bytes(void) {\n    size_t total = 0;\n    for (int i = 0; i < g_pointer_count; i++) {\n        if (g_pointer_live[i]) total += g_pointer_bytes[i];\n    }\n    return total;\n}\nstatic size_t fixture_known_live_count(void) {\n    size_t total = 0;\n    for (int i = 0; i < g_pointer_count; i++) {\n        if (g_pointer_live[i]) total++;\n    }\n    return total;\n}\nstatic int fixture_current_matches_known_live(void) {\n    return g_real_current == fixture_known_live_bytes() &&\n        g_live_count == fixture_known_live_count();\n}\nstatic void fixture_check_current_boundary(void) {\n    if (!fixture_current_matches_known_live()) {\n        g_sensor_sum_violations++;\n        fixture_protocol_error();\n    }\n}\nstatic int fixture_session_is_live(const ds4_session *s) {\n    return fixture_pointer_is_live(s);\n}\nstatic int fixture_engine_is_live(const ds4_engine *e) {\n    return fixture_pointer_is_live(e);\n}\nstatic int fixture_current_session_is_live(void) {\n    return fixture_session_is_live(g_current_session);\n}\nstatic int fake_common_fields_empty(const ds4_session *s) {\n    if (!fixture_session_is_live(s)) {\n        fixture_protocol_error();\n        return 0;\n    }\n    return !s->checkpoint.v && s->checkpoint.len == 0 &&\n        s->checkpoint.cap == 0 && !s->greedy_splitkv_segment.v &&\n        s->greedy_splitkv_segment.len == 0 &&\n        s->greedy_splitkv_segment.cap == 0 && !s->logits &&\n        !s->sample_probs && !s->glm_mtp_hc && !s->glm_mtp_logits0 &&\n        !s->mtp_logits && !s->spec_row_logits &&\n        !s->dspark_markov_bias && !s->dspark_conf_features;\n}\nstatic int fixture_current_has_cache(const fake_cache *cache) {\n    return fixture_current_session_is_live() &&\n        cache == &g_current_session->cpu_cache;\n}\nstatic int fixture_current_has_scratch(const fake_scratch *scratch) {\n    return fixture_current_session_is_live() &&\n        scratch == &g_current_session->cpu_scratch;\n}\nstatic int fixture_current_has_laguna_graph(const fake_laguna_graph *graph) {\n    return fixture_current_session_is_live() &&\n        graph == &g_current_session->laguna_graph;\n}\nstatic int fixture_current_has_glm_graph(const fake_glm_graph *graph) {\n    return fixture_current_session_is_live() &&\n        graph == &g_current_session->glm_graph;\n}\nstatic int fixture_current_has_graph(const fake_graph *graph) {\n    return fixture_current_session_is_live() &&\n        graph == &g_current_session->graph;\n}\nstatic int fixture_current_has_tokens(const fake_token_vec *tokens) {\n    return fixture_current_session_is_live() &&\n        (tokens == &g_current_session->checkpoint ||\n         tokens == &g_current_session->greedy_splitkv_segment);\n}\n\nstatic void fixture_free(void *p);\n\nstatic void *fixture_malloc(size_t bytes) {\n    g_allocation_attempts++;\n    if (bytes > 4096u || g_alloc_bytes > 65536u - bytes) {\n        fixture_protocol_error();\n        return NULL;\n    }\n    if (g_pointer_count >= FAKE_POINTER_CAP) {\n        fixture_protocol_error();\n        return NULL;\n    }\n    void *p = malloc(bytes);\n    if (!p) return NULL;\n    g_pointer[g_pointer_count] = p;\n    g_pointer_bytes[g_pointer_count] = bytes;\n    g_pointer_kind[g_pointer_count] = 1;\n    g_pointer_live[g_pointer_count] = 1;\n    g_pointer_count++;\n    g_live_count++;\n    g_alloc_calls++;\n    g_malloc_calls++;\n    g_alloc_bytes += bytes;\n    g_real_current += bytes;\n    if (g_real_current > g_real_peak) g_real_peak = g_real_current;\n    if (bytes > g_max_alloc) g_max_alloc = bytes;\n    fixture_check_current_boundary();\n    return p;\n}\nstatic void *fixture_calloc(size_t count, size_t bytes) {\n    g_allocation_attempts++;\n    if (bytes != 0 && count > SIZE_MAX / bytes) {\n        fixture_protocol_error();\n        return NULL;\n    }\n    size_t total = count * bytes;\n    if (total > 4096u || g_alloc_bytes > 65536u - total) {\n        fixture_protocol_error();\n        return NULL;\n    }\n    if (g_pointer_count >= FAKE_POINTER_CAP) {\n        fixture_protocol_error();\n        return NULL;\n    }\n    void *p = calloc(count, bytes);\n    if (!p) return NULL;\n    g_pointer[g_pointer_count] = p;\n    g_pointer_bytes[g_pointer_count] = total;\n    g_pointer_kind[g_pointer_count] = 2;\n    g_pointer_live[g_pointer_count] = 1;\n    g_pointer_count++;\n    g_live_count++;\n    g_alloc_calls++;\n    g_calloc_calls++;\n    g_alloc_bytes += total;\n    g_real_current += total;\n    if (g_real_current > g_real_peak) g_real_peak = g_real_current;\n    if (total > g_max_alloc) g_max_alloc = total;\n    const unsigned char *raw = (const unsigned char *)p;\n    for (size_t i = 0; i < total; i++) {\n        if (raw[i] != 0) fixture_protocol_error();\n    }\n    fixture_check_current_boundary();\n    return p;\n}\nstatic int fixture_free_known_payload(void *p) {\n    if (!p) return 1;\n    if (!fixture_pointer_is_live(p)) {\n        fixture_protocol_error();\n        return 0;\n    }\n    fixture_free(p);\n    return 1;\n}\nstatic void fixture_free(void *p) {\n    if (!p) return;\n    int i = fixture_find_pointer(p);\n    if (i < 0) {\n        g_unknown_free_calls++;\n        fixture_protocol_error();\n        return;\n    }\n    if (!g_pointer_live[i]) {\n        g_double_free_calls++;\n        fixture_protocol_error();\n        return;\n    }\n    fixture_check_current_boundary();\n    if (p == (void *)g_current_session) {\n        if (!g_owner_slot || *g_owner_slot != g_current_session) {\n            g_free_order_violations++;\n        } else {\n            g_container_free_calls++;\n            g_container_tp_id_at_free = g_current_session->tp_session_id;\n            g_container_dist_at_free = (uintptr_t)g_current_session->distributed;\n            g_container_reserved_at_free =\n                g_current_session->exact_cache_session_reserved;\n            if (!fixture_engine_is_live(g_current_session->engine)) {\n                fixture_protocol_error();\n                g_container_active_at_free = UINT32_MAX;\n            } else {\n                g_container_active_at_free =\n                    g_current_session->engine->exact_cache_active_sessions;\n            }\n            g_container_common_empty_at_free =\n                fake_common_fields_empty(g_current_session);\n            g_container_cpu_empty_at_free =\n                !g_current_session->cpu_cache.payload &&\n                !g_current_session->cpu_scratch.payload;\n            g_container_graph_empty_at_free =\n                (!g_current_session->laguna_graph.payload &&\n                 !g_current_session->laguna_graph.active) &&\n                (!g_current_session->glm_graph.payload &&\n                 !g_current_session->glm_graph.active) &&\n                (!g_current_session->graph.payload &&\n                 !g_current_session->graph.active);\n            g_container_laguna_ready_at_free =\n                g_current_session->laguna_graph_ready;\n            g_container_glm_ready_at_free =\n                g_current_session->glm_graph_ready;\n            fake_event(\'C\');\n        }\n    } else {\n        fake_event(\'f\');\n    }\n    /* libc free is physical cleanup; retire sensor state only afterwards. */\n    free(p);\n    g_physical_free_calls++;\n    g_pointer_live[i] = 0;\n    if (g_live_count == 0 || g_real_current < g_pointer_bytes[i]) {\n        fixture_protocol_error();\n    } else {\n        g_live_count--;\n        g_real_current -= g_pointer_bytes[i];\n    }\n    fixture_check_current_boundary();\n}\n\n/* These wrappers are deliberately defined before the macros used by the extracted body. */\nstatic int fixture_mutex_lock(pthread_mutex_t *mutex) {\n    if (!fixture_engine_is_live(g_mutex_engine) ||\n        mutex != &g_mutex_engine->exact_cache_session_mutex) {\n        g_unknown_mutex_calls++;\n        fixture_protocol_error();\n        return EINVAL;\n    }\n    if (!fixture_current_session_is_live()) {\n        fixture_protocol_error();\n        return EINVAL;\n    }\n    g_exact_lock_calls++;\n    fake_event(\'R\');\n    if (!fake_common_fields_empty(g_current_session))\n        g_free_order_violations++;\n    if (g_exact_lock_refuse) return EBUSY;\n    if (g_exact_lock_snapshot_valid) {\n        g_exact_unlock_entry_violations++;\n        fixture_protocol_error();\n        return EBUSY;\n    }\n    g_exact_lock_snapshot_valid = 1;\n    g_exact_lock_snapshot_reserved =\n        g_current_session->exact_cache_session_reserved;\n    g_exact_lock_snapshot_active = g_mutex_engine->exact_cache_active_sessions;\n    return 0;\n}\nstatic int fixture_mutex_unlock(pthread_mutex_t *mutex) {\n    if (!fixture_engine_is_live(g_mutex_engine) ||\n        mutex != &g_mutex_engine->exact_cache_session_mutex) {\n        g_unknown_mutex_calls++;\n        fixture_protocol_error();\n        return EINVAL;\n    }\n    if (!fixture_current_session_is_live()) {\n        fixture_protocol_error();\n        return EINVAL;\n    }\n    g_exact_unlock_calls++;\n    fake_event(\'u\');\n    /* Generated-control-flow temporal oracle; not a claim about pthread state. */\n    g_exact_unlock_entry_checks++;\n    if (!g_exact_lock_snapshot_valid) {\n        g_exact_unlock_entry_violations++;\n        fixture_protocol_error();\n    } else if (!g_exact_lock_snapshot_reserved) {\n        g_exact_unlock_entry_violations++;\n        fixture_protocol_error();\n    } else if (g_exact_lock_snapshot_active > 0u) {\n        if (g_current_session->exact_cache_session_reserved ||\n            g_mutex_engine->exact_cache_active_sessions !=\n                g_exact_lock_snapshot_active - 1u) {\n            g_exact_unlock_entry_violations++;\n            fixture_protocol_error();\n        }\n    } else if (g_current_session->exact_cache_session_reserved !=\n                   g_exact_lock_snapshot_reserved ||\n               g_mutex_engine->exact_cache_active_sessions !=\n                   g_exact_lock_snapshot_active) {\n        g_exact_unlock_entry_violations++;\n        fixture_protocol_error();\n    }\n    g_exact_lock_snapshot_valid = 0;\n    return g_exact_unlock_refuse ? EBUSY : 0;\n}\n#define free(p) fixture_free((p))\n#define calloc(n, s) fixture_calloc((n), (s))\n#define malloc(n) fixture_malloc((n))\n#define pthread_mutex_lock(m) fixture_mutex_lock((m))\n#define pthread_mutex_unlock(m) fixture_mutex_unlock((m))\n\nstatic bool ds4_session_tp_leader(const ds4_session *s) {\n    if (!fixture_session_is_live(s) || !fixture_engine_is_live(s->engine)) {\n        fixture_protocol_error();\n        return false;\n    }\n    return s->engine->tp.active && s->engine->tp.rank == 0;\n}\nstatic bool ds4_tp_failed(struct ds4_tp *tp) {\n    if (!tp) return false;\n    if (!fixture_pointer_is_live(tp)) {\n        fixture_protocol_error();\n        return true;\n    }\n    return tp->failed;\n}\nstatic bool ds4_session_is_cpu(const ds4_session *s) {\n    if (!fixture_session_is_live(s) || !fixture_engine_is_live(s->engine)) {\n        fixture_protocol_error();\n        return false;\n    }\n    return s->engine->fake_family == FAKE_FAMILY_CPU;\n}\nstatic bool ds4_session_is_glm(const ds4_session *s) {\n    if (!fixture_session_is_live(s) || !fixture_engine_is_live(s->engine)) {\n        fixture_protocol_error();\n        return false;\n    }\n    return s->engine->fake_family == FAKE_FAMILY_GLM;\n}\nstatic bool ds4_session_is_laguna(const ds4_session *s) {\n    if (!fixture_session_is_live(s) || !fixture_engine_is_live(s->engine)) {\n        fixture_protocol_error();\n        return false;\n    }\n    return s->engine->fake_family == FAKE_FAMILY_LAGUNA;\n}\nstatic int ds4_tp_send_session_destroy(struct ds4_tp *tp, uint64_t id) {\n    (void)id;\n    if (!fixture_pointer_is_live(tp)) {\n        fixture_protocol_error();\n        return 0;\n    }\n    g_tp_destroy_calls++;\n    fake_event(\'T\');\n    return 1;\n}\nstatic int ds4_tp_wait_command_ack(struct ds4_tp *tp, uint64_t id,\n                                   const char *what, char *err, size_t errcap) {\n    (void)id; (void)what; (void)err; (void)errcap;\n    if (!fixture_pointer_is_live(tp)) {\n        fixture_protocol_error();\n        return 0;\n    }\n    return 1;\n}\nstatic void ds4_session_print_dspark_stats(ds4_session *s) {\n    if (!fixture_session_is_live(s)) fixture_protocol_error();\n}\nstatic void ds4_dist_session_free(ds4_dist_session *d) {\n    if (!d) return;\n    if (!fixture_pointer_is_live(d)) {\n        fixture_protocol_error();\n        return;\n    }\n    if (d->payload && !fixture_pointer_is_live(d->payload)) {\n        fixture_protocol_error();\n        return;\n    }\n    g_dist_free_calls++;\n    fake_event(\'D\');\n    void *payload = d->payload;\n    if (!fixture_free_known_payload(payload)) return;\n    d->payload = NULL;\n    fixture_free(d);\n}\nstatic void kv_cache_free(fake_cache *cache) {\n    if (!fixture_current_has_cache(cache)) {\n        fixture_protocol_error();\n        return;\n    }\n    if (cache->payload && !fixture_pointer_is_live(cache->payload)) {\n        fixture_protocol_error();\n        return;\n    }\n    g_cpu_free_calls++;\n    void *payload = cache->payload;\n    if (!fixture_free_known_payload(payload)) return;\n    cache->payload = NULL;\n}\nstatic void cpu_decode_scratch_free(fake_scratch *scratch) {\n    if (!fixture_current_has_scratch(scratch)) {\n        fixture_protocol_error();\n        return;\n    }\n    if (scratch->payload && !fixture_pointer_is_live(scratch->payload)) {\n        fixture_protocol_error();\n        return;\n    }\n    g_scratch_free_calls++;\n    void *payload = scratch->payload;\n    if (!fixture_free_known_payload(payload)) return;\n    scratch->payload = NULL;\n}\nstatic bool laguna_graph_free(fake_laguna_graph *graph) {\n    g_graph_calls++;\n    fake_event(\'G\');\n    if (!fixture_current_has_laguna_graph(graph)) {\n        fixture_protocol_error();\n        return false;\n    }\n    if (g_graph_refuse) return false;\n    if (!graph->payload) {\n        graph->active = 0;\n        return true;\n    }\n    if (!fixture_pointer_is_live(graph->payload)) {\n        fixture_protocol_error();\n        return false;\n    }\n    void *payload = graph->payload;\n    fixture_free(payload);\n    graph->payload = NULL;\n    graph->active = 0;\n    g_graph_physical_frees++;\n    return true;\n}\nstatic void glm_graph_free(fake_glm_graph *graph) {\n    if (!fixture_current_has_glm_graph(graph) ||\n        (graph->payload && !fixture_pointer_is_live(graph->payload))) {\n        fixture_protocol_error();\n        return;\n    }\n    void *payload = graph->payload;\n    if (!fixture_free_known_payload(payload)) return;\n    graph->payload = NULL;\n    graph->active = 0;\n}\nstatic void metal_graph_free(fake_graph *graph) {\n    if (!fixture_current_has_graph(graph) ||\n        (graph->payload && !fixture_pointer_is_live(graph->payload))) {\n        fixture_protocol_error();\n        return;\n    }\n    void *payload = graph->payload;\n    if (!fixture_free_known_payload(payload)) return;\n    graph->payload = NULL;\n    graph->active = 0;\n}\nstatic void token_vec_free(fake_token_vec *tokens) {\n    if (!fixture_current_has_tokens(tokens) ||\n        (tokens->v && !fixture_pointer_is_live(tokens->v))) {\n        fixture_protocol_error();\n        return;\n    }\n    int *payload = tokens->v;\n    if (!fixture_free_known_payload(payload)) return;\n    tokens->v = NULL;\n    tokens->len = 0;\n    tokens->cap = 0;\n}\nstatic int ds4_engine_compact_tracker_lock(ds4_engine *e) {\n    if (!fixture_engine_is_live(e)) {\n        fixture_protocol_error();\n        return 0;\n    }\n    g_compact_lock_calls++;\n    fake_event(\'K\');\n    if (!e->laguna_compact_runtime) return 1;\n    return g_lock_refuse ? 0 : 1;\n}\nstatic int ds4_engine_compact_tracker_unlock(ds4_engine *e) {\n    if (!fixture_engine_is_live(e)) {\n        fixture_protocol_error();\n        return 0;\n    }\n    g_compact_unlock_calls++;\n    fake_event(\'U\');\n    if (!e->laguna_compact_runtime) return 1;\n    return g_unlock_refuse ? 0 : 1;\n}\nstatic void ds4_engine_release_exact_cache_session(ds4_engine *e) {\n    if (!fixture_engine_is_live(e)) fixture_protocol_error();\n    g_old_release_calls++;\n}\n\nstatic int fixture_release_begin(void) {\n    fixture_check_current_boundary();\n    g_release_boundary_calls++;\n    return g_allocation_attempts;\n}\nstatic int fixture_release_end(int before) {\n    if (g_allocation_attempts < before) {\n        fixture_protocol_error();\n        return 0;\n    }\n    g_release_allocation_attempts += g_allocation_attempts - before;\n    fixture_check_current_boundary();\n    return g_allocation_attempts == before;\n}\nstatic int fixture_call_checked(ds4_session **owner) {\n    int before = fixture_release_begin();\n    int result = ds4_session_free_checked(owner);\n    (void)fixture_release_end(before);\n    return result;\n}\n/* The actual marked block below supplies this private definition. */\nstatic int ds4_session_release_exact_cache_reservation(ds4_session *s);\n\nstatic int fixture_call_exact_helper(ds4_session *s) {\n    int before = fixture_release_begin();\n    int result = ds4_session_release_exact_cache_reservation(s);\n    (void)fixture_release_end(before);\n    return result;\n}\nstatic int fixture_call_legacy(ds4_session *s) {\n    int before = fixture_release_begin();\n    ds4_session_free(s);\n    return fixture_release_end(before);\n}\nstatic int fixture_call_abi_null_owner(void) {\n    int before = fixture_release_begin();\n    int result = checked_release_caller_null_owner();\n    (void)fixture_release_end(before);\n    return result;\n}\nstatic int fixture_call_abi_empty_owner(void) {\n    int before = fixture_release_begin();\n    int result = checked_release_caller_empty_owner();\n    (void)fixture_release_end(before);\n    return result;\n}\n'
    + "\n"
    + (SESSION_BLOCK or "")
    + "\n"
    + '\n\nstatic int reset_fixture(void) {\n    if (g_live_count != 0 || g_real_current != 0 ||\n        !fixture_current_matches_known_live() ||\n        g_fixture_protocol_errors != 0 || g_unknown_free_calls != 0 ||\n        g_double_free_calls != 0 || g_unknown_mutex_calls != 0 ||\n        g_exact_lock_snapshot_valid != 0 ||\n        g_exact_unlock_entry_violations != 0 ||\n        g_free_order_violations != 0 || g_sensor_sum_violations != 0) {\n        fixture_protocol_error();\n        return 0;\n    }\n    memset(g_events, 0, sizeof(g_events));\n    g_event_len = 0;\n    g_owner_slot = NULL;\n    g_current_session = NULL;\n    g_mutex_engine = NULL;\n    g_lock_refuse = 0;\n    g_unlock_refuse = 0;\n    g_graph_refuse = 0;\n    g_exact_lock_refuse = 0;\n    g_exact_unlock_refuse = 0;\n    g_exact_lock_calls = 0;\n    g_exact_unlock_calls = 0;\n    g_exact_lock_snapshot_valid = 0;\n    g_exact_lock_snapshot_reserved = 0;\n    g_exact_lock_snapshot_active = 0;\n    g_exact_unlock_entry_checks = 0;\n    g_exact_unlock_entry_violations = 0;\n    g_compact_lock_calls = 0;\n    g_compact_unlock_calls = 0;\n    g_graph_calls = 0;\n    g_graph_physical_frees = 0;\n    g_cpu_free_calls = 0;\n    g_scratch_free_calls = 0;\n    g_dist_free_calls = 0;\n    g_tp_destroy_calls = 0;\n    g_old_release_calls = 0;\n    g_container_free_calls = 0;\n    g_container_tp_id_at_free = UINT64_MAX;\n    g_container_dist_at_free = UINTPTR_MAX;\n    g_container_reserved_at_free = -1;\n    g_container_active_at_free = UINT32_MAX;\n    g_container_common_empty_at_free = 0;\n    g_container_graph_empty_at_free = 0;\n    g_container_cpu_empty_at_free = 0;\n    g_container_laguna_ready_at_free = -1;\n    g_container_glm_ready_at_free = -1;\n    g_unknown_free_calls = 0;\n    g_double_free_calls = 0;\n    g_unknown_mutex_calls = 0;\n    g_free_order_violations = 0;\n    g_fixture_protocol_errors = 0;\n    g_sensor_sum_violations = 0;\n    g_allocation_attempts = 0;\n    g_release_boundary_calls = 0;\n    g_release_allocation_attempts = 0;\n    g_alloc_calls = 0;\n    g_malloc_calls = 0;\n    g_calloc_calls = 0;\n    g_physical_free_calls = 0;\n    g_alloc_bytes = 0;\n    g_real_current = 0;\n    g_real_peak = 0;\n    g_max_alloc = 0;\n    g_live_count = 0;\n    memset(g_pointer, 0, sizeof(g_pointer));\n    memset(g_pointer_bytes, 0, sizeof(g_pointer_bytes));\n    memset(g_pointer_kind, 0, sizeof(g_pointer_kind));\n    memset(g_pointer_live, 0, sizeof(g_pointer_live));\n    g_pointer_count = 0;\n    return 1;\n}\nstatic ds4_engine *make_engine(int family, int exact_ready) {\n    ds4_engine *e = (ds4_engine *)fixture_calloc(1, sizeof(*e));\n    if (!e) return NULL;\n    e->backend = family == FAKE_FAMILY_CPU ? DS4_BACKEND_CPU :\n        (family == FAKE_FAMILY_GENERIC ? DS4_BACKEND_METAL : DS4_BACKEND_CUDA);\n    e->fake_family = family;\n    e->laguna_compact_runtime = family == FAKE_FAMILY_LAGUNA;\n    e->exact_cache_session_mutex_initialized = exact_ready != 0;\n    e->exact_cache_active_sessions = exact_ready ? 1u : 0u;\n    e->tracker_cookie = 0x5a17;\n    g_mutex_engine = e;\n    return e;\n}\nstatic int fill_common(ds4_session *s) {\n    if (!fixture_session_is_live(s)) {\n        fixture_protocol_error();\n        return 0;\n    }\n    s->checkpoint.v = (int *)fixture_calloc(3u, sizeof(int));\n    s->checkpoint.len = 3;\n    s->checkpoint.cap = 3;\n    s->greedy_splitkv_segment.v = (int *)fixture_calloc(2u, sizeof(int));\n    s->greedy_splitkv_segment.len = 2;\n    s->greedy_splitkv_segment.cap = 2;\n    s->logits = (float *)fixture_malloc(8u);\n    s->sample_probs = (float *)fixture_malloc(8u);\n    s->glm_mtp_hc = (float *)fixture_malloc(8u);\n    s->glm_mtp_logits0 = (float *)fixture_malloc(8u);\n    s->mtp_logits = (float *)fixture_malloc(8u);\n    s->spec_row_logits = (float *)fixture_malloc(8u);\n    s->dspark_markov_bias = (float *)fixture_malloc(8u);\n    s->dspark_conf_features = (float *)fixture_malloc(8u);\n    return s->checkpoint.v && s->greedy_splitkv_segment.v && s->logits &&\n        s->sample_probs && s->glm_mtp_hc && s->glm_mtp_logits0 &&\n        s->mtp_logits && s->spec_row_logits && s->dspark_markov_bias &&\n        s->dspark_conf_features;\n}\nstatic ds4_session *make_session(ds4_engine *e, int with_common, int with_dist) {\n    if (!fixture_engine_is_live(e)) {\n        fixture_protocol_error();\n        return NULL;\n    }\n    ds4_session *s = (ds4_session *)fixture_calloc(1, sizeof(*s));\n    if (!s) return NULL;\n    s->engine = e;\n    e->session_container_count++; /* Fake construction owns one container pin. */\n    int complete = !with_common || fill_common(s);\n    if (e->fake_family == FAKE_FAMILY_CPU) {\n        s->cpu_cache.payload = fixture_malloc(8u);\n        s->cpu_scratch.payload = fixture_malloc(8u);\n        complete = complete && s->cpu_cache.payload && s->cpu_scratch.payload;\n    } else if (e->fake_family == FAKE_FAMILY_LAGUNA) {\n        s->laguna_graph.payload = fixture_malloc(8u);\n        s->laguna_graph.active = s->laguna_graph.payload != NULL;\n        s->laguna_graph_ready = s->laguna_graph.payload != NULL;\n        complete = complete && s->laguna_graph.payload;\n    } else if (e->fake_family == FAKE_FAMILY_GLM) {\n        s->glm_graph.payload = fixture_malloc(8u);\n        s->glm_graph.active = s->glm_graph.payload != NULL;\n        s->glm_graph_ready = s->glm_graph.payload != NULL;\n        complete = complete && s->glm_graph.payload;\n    } else {\n        s->graph.payload = fixture_malloc(8u);\n        s->graph.active = s->graph.payload != NULL;\n        complete = complete && s->graph.payload;\n    }\n    if (with_dist) {\n        s->distributed = (ds4_dist_session *)fixture_calloc(1, sizeof(*s->distributed));\n        if (s->distributed) s->distributed->payload = fixture_malloc(8u);\n        complete = complete && s->distributed && s->distributed->payload;\n    }\n    return complete ? s : NULL;\n}\nstatic ds4_session *make_no_alloc_session(ds4_engine *e, int reserved) {\n    if (!fixture_engine_is_live(e)) {\n        fixture_protocol_error();\n        return NULL;\n    }\n    ds4_session *s = (ds4_session *)fixture_calloc(1, sizeof(*s));\n    if (!s) return NULL;\n    s->engine = e;\n    e->session_container_count++; /* Fake construction owns one container pin. */\n    s->test_no_alloc = true;\n    s->exact_cache_session_reserved = reserved != 0;\n    return s;\n}\nstatic int publish(ds4_session **owner, ds4_session *s) {\n    if (!owner || !fixture_session_is_live(s)) {\n        fixture_protocol_error();\n        return 0;\n    }\n    *owner = s;\n    g_owner_slot = owner;\n    g_current_session = s;\n    return 1;\n}\nstatic int session_graph_empty(const ds4_session *s, int family) {\n    if (!fixture_session_is_live(s)) {\n        fixture_protocol_error();\n        return 0;\n    }\n    if (family == FAKE_FAMILY_LAGUNA)\n        return !s->laguna_graph.payload && s->laguna_graph.active == 0;\n    if (family == FAKE_FAMILY_GLM)\n        return !s->glm_graph.payload && s->glm_graph.active == 0;\n    if (family == FAKE_FAMILY_GENERIC)\n        return !s->graph.payload && s->graph.active == 0;\n    return 1;\n}\nstatic int session_cpu_empty(const ds4_session *s) {\n    if (!fixture_session_is_live(s)) {\n        fixture_protocol_error();\n        return 0;\n    }\n    return !s->cpu_cache.payload && !s->cpu_scratch.payload;\n}\nstatic int fixture_safe_retry_owner(ds4_session **owner, ds4_session *expected) {\n    return owner && (*owner == NULL ||\n        (*owner == expected && fixture_session_is_live(expected)));\n}\nstatic int fixture_cleanup_engine(ds4_engine *e) {\n    if (!fixture_engine_is_live(e)) {\n        fixture_protocol_error();\n        return 0;\n    }\n    if (e->tp.ctx) {\n        if (!fixture_pointer_is_live(e->tp.ctx)) {\n            fixture_protocol_error();\n            return 0;\n        }\n        struct ds4_tp *tp = e->tp.ctx;\n        fixture_free(tp);\n        e->tp.ctx = NULL;\n    }\n    fixture_free(e);\n    return !fixture_engine_is_live(e);\n}\nstatic int standard_sensor_clean(void) {\n    return g_live_count == 0 && g_real_current == 0 &&\n        fixture_known_live_bytes() == 0 && fixture_known_live_count() == 0 &&\n        fixture_current_matches_known_live() &&\n        g_unknown_free_calls == 0 && g_double_free_calls == 0 &&\n        g_unknown_mutex_calls == 0 && g_free_order_violations == 0 &&\n        g_exact_lock_snapshot_valid == 0 &&\n        g_exact_unlock_entry_violations == 0 &&\n        g_fixture_protocol_errors == 0 && g_sensor_sum_violations == 0 &&\n        g_release_allocation_attempts == 0 &&\n        g_max_alloc <= 4096u && g_alloc_bytes <= 65536u;\n}\nstatic int run_success_family(int family) {\n    if (!reset_fixture()) return 0;\n    ds4_engine *e = make_engine(family, 0);\n    ds4_session *owner = NULL;\n    ds4_session *s = make_session(e, 1, 0);\n    if (!fixture_engine_is_live(e) || !fixture_session_is_live(s)) return 0;\n    if (family == FAKE_FAMILY_LAGUNA) {\n        e->tp.active = 1;\n        e->tp.rank = 0;\n        e->tp.ctx = (struct ds4_tp *)fixture_calloc(1, sizeof(struct ds4_tp));\n        if (!fixture_pointer_is_live(e->tp.ctx)) return 0;\n        s->tp_session_id = 9;\n    }\n    if (!publish(&owner, s)) return 0;\n    int attempts_before = g_allocation_attempts;\n    int first = fixture_call_checked(&owner);\n    int second = fixture_safe_retry_owner(&owner, s)\n        ? fixture_call_checked(&owner) : 0;\n    int consumed = owner == NULL && !fixture_session_is_live(s);\n    int engine_live = fixture_engine_is_live(e);\n    int state_ok = consumed && g_container_free_calls == 1 &&\n        g_container_tp_id_at_free == 0u &&\n        g_container_common_empty_at_free &&\n        (family == FAKE_FAMILY_CPU ? g_container_cpu_empty_at_free :\n            g_container_graph_empty_at_free) &&\n        (family == FAKE_FAMILY_LAGUNA ? !g_container_laguna_ready_at_free : 1) &&\n        (family == FAKE_FAMILY_GLM ? !g_container_glm_ready_at_free : 1);\n    int tp_calls_ok = family == FAKE_FAMILY_LAGUNA\n        ? g_tp_destroy_calls == 1 : g_tp_destroy_calls == 0;\n    int no_alloc = g_allocation_attempts == attempts_before &&\n        g_release_allocation_attempts == 0;\n    int old_wrapper_unused = g_old_release_calls == 0;\n    int engine_cleaned = consumed && engine_live && fixture_cleanup_engine(e);\n    printf("first=%d\\nsecond=%d\\nstate_ok=%d\\ntp_calls_ok=%d\\nno_alloc=%d\\nold_wrapper_unused=%d\\nengine_cleaned=%d\\n",\n           first, second, state_ok, tp_calls_ok, no_alloc, old_wrapper_unused,\n           engine_cleaned);\n    return first == 1 && second == 1 && state_ok && tp_calls_ok && no_alloc &&\n        old_wrapper_unused && engine_cleaned && standard_sensor_clean();\n}\nstatic int scenario_nulls(void) {\n    if (!reset_fixture()) return 0;\n    int before_attempts = g_allocation_attempts;\n    int before_free = g_container_free_calls;\n    int null_owner = fixture_call_abi_null_owner();\n    int empty_owner = fixture_call_abi_empty_owner();\n    int no_alloc = g_allocation_attempts == before_attempts &&\n        g_release_allocation_attempts == 0;\n    int no_free = g_container_free_calls == before_free;\n    printf("null_owner=%d\\nempty_owner=%d\\nnulls_no_alloc=%d\\nnulls_no_free=%d\\n",\n           null_owner, empty_owner, no_alloc, no_free);\n    return null_owner == 1 && empty_owner == 1 && no_alloc && no_free &&\n        standard_sensor_clean();\n}\nstatic int scenario_success(const char *name) {\n    if (!strcmp(name, "success-cpu")) return run_success_family(FAKE_FAMILY_CPU);\n    if (!strcmp(name, "success-laguna")) return run_success_family(FAKE_FAMILY_LAGUNA);\n    if (!strcmp(name, "success-glm")) return run_success_family(FAKE_FAMILY_GLM);\n    if (!strcmp(name, "success-generic")) return run_success_family(FAKE_FAMILY_GENERIC);\n    return 0;\n}\nstatic int scenario_lock_refusal(void) {\n    if (!reset_fixture()) return 0;\n    ds4_engine *e = make_engine(FAKE_FAMILY_LAGUNA, 0);\n    ds4_session *owner = NULL;\n    ds4_session *s = make_session(e, 1, 0);\n    if (!fixture_engine_is_live(e) || !fixture_session_is_live(s) ||\n        !publish(&owner, s)) return 0;\n    g_lock_refuse = 1;\n    int first = fixture_call_checked(&owner);\n    int s_live = fixture_session_is_live(s);\n    int e_live = fixture_engine_is_live(e);\n    int retained = first == 0 && owner == s && s_live && e_live &&\n        s->laguna_graph.payload && !fake_common_fields_empty(s) &&\n        g_graph_calls == 0 && g_compact_unlock_calls == 0 &&\n        e->tracker_cookie == 0x5a17;\n    g_lock_refuse = 0;\n    int second = retained && fixture_safe_retry_owner(&owner, s)\n        ? fixture_call_checked(&owner) : 0;\n    int consumed = owner == NULL && !fixture_session_is_live(s);\n    int retried = consumed && fixture_engine_is_live(e) && g_graph_calls == 1 &&\n        g_graph_physical_frees == 1 && g_compact_lock_calls == 2 &&\n        g_compact_unlock_calls == 1 && e->tracker_cookie == 0x5a17;\n    int engine_cleaned = consumed && fixture_engine_is_live(e) &&\n        fixture_cleanup_engine(e);\n    printf("first=%d\\nretained=%d\\nsecond=%d\\nretried=%d\\ngraph_calls=%d\\nengine_cleaned=%d\\n",\n           first, retained, second, retried, g_graph_calls, engine_cleaned);\n    return first == 0 && retained && second == 1 && retried && engine_cleaned &&\n        standard_sensor_clean();\n}\nstatic int scenario_graph_refusal(void) {\n    if (!reset_fixture()) return 0;\n    ds4_engine *e = make_engine(FAKE_FAMILY_LAGUNA, 0);\n    ds4_session *owner = NULL;\n    ds4_session *s = make_session(e, 1, 0);\n    if (!fixture_engine_is_live(e) || !fixture_session_is_live(s) ||\n        !publish(&owner, s)) return 0;\n    g_graph_refuse = 1;\n    int first = fixture_call_checked(&owner);\n    int s_live = fixture_session_is_live(s);\n    int e_live = fixture_engine_is_live(e);\n    int failed_graph = first == 0 && owner == s && s_live && e_live &&\n        s->laguna_graph.payload && !fake_common_fields_empty(s) &&\n        g_graph_calls == 1 && g_compact_lock_calls == 1 &&\n        g_compact_unlock_calls == 1 && fake_event_is(0, \'K\') &&\n        fake_event_is(1, \'G\') && fake_event_is(2, \'U\');\n    g_graph_refuse = 0;\n    int second = failed_graph && fixture_safe_retry_owner(&owner, s)\n        ? fixture_call_checked(&owner) : 0;\n    int consumed = owner == NULL && !fixture_session_is_live(s);\n    int retried = consumed && fixture_engine_is_live(e) && g_graph_calls == 2 &&\n        g_graph_physical_frees == 1 && g_compact_unlock_calls == 2 &&\n        e->tracker_cookie == 0x5a17;\n    int engine_cleaned = consumed && fixture_engine_is_live(e) &&\n        fixture_cleanup_engine(e);\n    printf("first=%d\\nfailed_graph=%d\\nsecond=%d\\nretried=%d\\ngraph_calls=%d\\nengine_cleaned=%d\\n",\n           first, failed_graph, second, retried, g_graph_calls, engine_cleaned);\n    return first == 0 && failed_graph && second == 1 && retried &&\n        engine_cleaned && standard_sensor_clean();\n}\nstatic int scenario_unlock_refusal(void) {\n    if (!reset_fixture()) return 0;\n    ds4_engine *e = make_engine(FAKE_FAMILY_LAGUNA, 0);\n    ds4_session *owner = NULL;\n    ds4_session *s = make_session(e, 1, 0);\n    if (!fixture_engine_is_live(e) || !fixture_session_is_live(s) ||\n        !publish(&owner, s)) return 0;\n    g_unlock_refuse = 1;\n    int first = fixture_call_checked(&owner);\n    int s_live = fixture_session_is_live(s);\n    int e_live = fixture_engine_is_live(e);\n    int retained_after_unlock = first == 0 && owner == s && s_live && e_live &&\n        !s->laguna_graph.payload && g_graph_physical_frees == 1 &&\n        g_graph_calls == 1 && g_compact_lock_calls == 1 &&\n        g_compact_unlock_calls == 1 && !fake_common_fields_empty(s) &&\n        e->tracker_cookie == 0x5a17;\n    g_unlock_refuse = 0;\n    int second = retained_after_unlock && fixture_safe_retry_owner(&owner, s)\n        ? fixture_call_checked(&owner) : 0;\n    int consumed = owner == NULL && !fixture_session_is_live(s);\n    int retried = consumed && fixture_engine_is_live(e) && g_graph_calls == 2 &&\n        g_graph_physical_frees == 1 && g_compact_unlock_calls == 2 &&\n        g_container_free_calls == 1 && e->tracker_cookie == 0x5a17;\n    int engine_cleaned = consumed && fixture_engine_is_live(e) &&\n        fixture_cleanup_engine(e);\n    printf("first=%d\\nretained_after_unlock=%d\\nsecond=%d\\nretried=%d\\ngraph_calls=%d\\nengine_cleaned=%d\\n",\n           first, retained_after_unlock, second, retried, g_graph_calls,\n           engine_cleaned);\n    return first == 0 && retained_after_unlock && second == 1 && retried &&\n        engine_cleaned && standard_sensor_clean();\n}\nstatic int scenario_distributed_prefix(void) {\n    if (!reset_fixture()) return 0;\n    ds4_engine *e = make_engine(FAKE_FAMILY_LAGUNA, 0);\n    ds4_session *owner = NULL;\n    ds4_session *s = make_session(e, 1, 1);\n    if (!fixture_engine_is_live(e) || !fixture_session_is_live(s) ||\n        !publish(&owner, s)) return 0;\n    g_lock_refuse = 1;\n    int first = fixture_call_checked(&owner);\n    int s_live = fixture_session_is_live(s);\n    int e_live = fixture_engine_is_live(e);\n    int prefix = first == 0 && owner == s && s_live && e_live &&\n        s->distributed == NULL && g_dist_free_calls == 1 &&\n        g_graph_calls == 0 && s->laguna_graph.payload &&\n        e->tracker_cookie == 0x5a17;\n    g_lock_refuse = 0;\n    int second = prefix && fixture_safe_retry_owner(&owner, s)\n        ? fixture_call_checked(&owner) : 0;\n    int consumed = owner == NULL && !fixture_session_is_live(s);\n    int retried = consumed && fixture_engine_is_live(e) &&\n        g_dist_free_calls == 1 && g_graph_physical_frees == 1 &&\n        g_container_dist_at_free == 0 && e->tracker_cookie == 0x5a17;\n    int engine_cleaned = consumed && fixture_engine_is_live(e) &&\n        fixture_cleanup_engine(e);\n    printf("first=%d\\nprefix=%d\\nsecond=%d\\nretried=%d\\ndist_free_calls=%d\\nengine_cleaned=%d\\n",\n           first, prefix, second, retried, g_dist_free_calls, engine_cleaned);\n    return first == 0 && prefix && second == 1 && retried && engine_cleaned &&\n        standard_sensor_clean();\n}\nstatic int exact_case(const char *name) {\n    if (!reset_fixture()) return 0;\n    ds4_engine *e = make_engine(FAKE_FAMILY_GENERIC, 1);\n    ds4_session *owner = NULL;\n    ds4_session *s = make_session(e, 1, 0);\n    if (!fixture_engine_is_live(e) || !fixture_session_is_live(s)) return 0;\n    int no_reservation = !strcmp(name, "exact-no-reservation");\n    s->exact_cache_session_reserved = !no_reservation;\n    if (!strcmp(name, "exact-missing-mutex")) {\n        e->exact_cache_session_mutex_initialized = false;\n    } else if (!strcmp(name, "exact-lock-refusal")) {\n        g_exact_lock_refuse = 1;\n    } else if (!strcmp(name, "exact-zero-count")) {\n        e->exact_cache_active_sessions = 0;\n    } else if (!strcmp(name, "exact-unlock-refusal")) {\n        g_exact_unlock_refuse = 1;\n    } else if (no_reservation) {\n        e->exact_cache_active_sessions = 0;\n    } else {\n        return 0;\n    }\n    if (!publish(&owner, s)) return 0;\n    int first = fixture_call_checked(&owner);\n    int s_live = fixture_session_is_live(s);\n    int e_live = fixture_engine_is_live(e);\n    int common_done = no_reservation\n        ? owner == NULL && g_container_common_empty_at_free &&\n            g_container_graph_empty_at_free\n        : s_live && fake_common_fields_empty(s) &&\n            session_graph_empty(s, FAKE_FAMILY_GENERIC);\n    int retained = !no_reservation && first == 0 && owner == s && s_live &&\n        e_live && common_done && g_container_free_calls == 0 &&\n        g_cpu_free_calls == 0 && g_scratch_free_calls == 0;\n    int no_reservation_consumed = no_reservation && owner == NULL && !s_live &&\n        e_live && g_container_free_calls == 1 &&\n        g_container_reserved_at_free == 0 && g_container_active_at_free == 0 &&\n        g_container_common_empty_at_free && g_container_graph_empty_at_free;\n    int flag_before_retry = s_live ? s->exact_cache_session_reserved :\n        no_reservation_consumed ? g_container_reserved_at_free : -1;\n    uint32_t count_before_retry = e_live ? e->exact_cache_active_sessions : UINT32_MAX;\n    if (retained && (!strcmp(name, "exact-missing-mutex") ||\n        !strcmp(name, "exact-lock-refusal"))) {\n        e->exact_cache_session_mutex_initialized = true;\n        g_exact_lock_refuse = 0;\n    } else if (retained && !strcmp(name, "exact-zero-count")) {\n        e->exact_cache_active_sessions = 1;\n    } else if (retained && !strcmp(name, "exact-unlock-refusal")) {\n        g_exact_unlock_refuse = 0;\n    }\n    int second = (no_reservation || retained) && fixture_safe_retry_owner(&owner, s)\n        ? fixture_call_checked(&owner) : 0;\n    int consumed = owner == NULL && !fixture_session_is_live(s);\n    int current_before_engine_cleanup = consumed && fixture_engine_is_live(e) &&\n        g_live_count == 1 && g_real_current == fixture_known_live_bytes() &&\n        g_real_current == sizeof(*e) && fixture_current_matches_known_live();\n    int retry_done = consumed && g_container_free_calls == 1 &&\n        current_before_engine_cleanup && e->exact_cache_active_sessions == 0 &&\n        e->tracker_cookie == 0x5a17;\n    int common_once = g_cpu_free_calls == 0 && g_scratch_free_calls == 0;\n    int expected_unlock_entry_checks = no_reservation ? 0 :\n        !strcmp(name, "exact-zero-count") ? 2 : 1;\n    int exact_state = 0;\n    if (!strcmp(name, "exact-missing-mutex")) {\n        exact_state = flag_before_retry == 1 && count_before_retry == 1u &&\n            g_exact_lock_calls == 1 && g_exact_unlock_calls == 1;\n    } else if (!strcmp(name, "exact-lock-refusal")) {\n        exact_state = flag_before_retry == 1 && count_before_retry == 1u &&\n            g_exact_lock_calls == 2 && g_exact_unlock_calls == 1;\n    } else if (!strcmp(name, "exact-zero-count")) {\n        exact_state = flag_before_retry == 1 && count_before_retry == 0u &&\n            g_exact_lock_calls == 2 && g_exact_unlock_calls == 2;\n    } else if (!strcmp(name, "exact-unlock-refusal")) {\n        exact_state = flag_before_retry == 0 && count_before_retry == 0u &&\n            g_exact_lock_calls == 1 && g_exact_unlock_calls == 1;\n    } else if (no_reservation) {\n        exact_state = no_reservation_consumed && flag_before_retry == 0 &&\n            count_before_retry == 0u && g_exact_lock_calls == 0 &&\n            g_exact_unlock_calls == 0;\n    }\n    exact_state = exact_state &&\n        g_exact_unlock_entry_checks == expected_unlock_entry_checks &&\n        g_exact_unlock_entry_violations == 0;\n    int engine_cleaned = consumed && fixture_engine_is_live(e) &&\n        fixture_cleanup_engine(e);\n    printf("first=%d\\nsecond=%d\\ncommon_done=%d\\nretained=%d\\nno_reservation_consumed=%d\\nflag_before_retry=%d\\ncount_before_retry=%u\\nretry_done=%d\\nexact_state=%d\\ncommon_once=%d\\ncurrent_before_engine_cleanup=%d\\nengine_cleaned=%d\\n",\n           first, second, common_done, retained, no_reservation_consumed,\n           flag_before_retry, count_before_retry, retry_done, exact_state,\n           common_once, current_before_engine_cleanup, engine_cleaned);\n    return first == (no_reservation ? 1 : 0) && retained == !no_reservation &&\n        second == 1 && retry_done && exact_state && common_once &&\n        engine_cleaned && standard_sensor_clean();\n}\nstatic int scenario_exact_missing_engine(void) {\n    if (!reset_fixture()) return 0;\n    ds4_engine *e = make_engine(FAKE_FAMILY_GENERIC, 1);\n    ds4_session *owner = NULL;\n    ds4_session *s = make_session(e, 1, 0);\n    if (!fixture_engine_is_live(e) || !fixture_session_is_live(s)) return 0;\n    s->exact_cache_session_reserved = true;\n    if (!publish(&owner, s)) return 0;\n    /* Helper boundary only: a null engine is not a valid real-session predicate. */\n    s->engine = NULL;\n    int first = fixture_call_exact_helper(s);\n    int s_live = fixture_session_is_live(s);\n    int e_live = fixture_engine_is_live(e);\n    int retained = first == 0 && owner == s && s_live && e_live &&\n        s->engine == NULL && s->exact_cache_session_reserved &&\n        e->exact_cache_active_sessions == 1u && g_exact_lock_calls == 0 &&\n        g_exact_unlock_calls == 0;\n    if (retained) s->engine = e; /* Restore the borrowed engine before real release. */\n    int second = retained && fixture_safe_retry_owner(&owner, s)\n        ? fixture_call_checked(&owner) : 0;\n    int consumed = owner == NULL && !fixture_session_is_live(s);\n    int current_before_engine_cleanup = consumed && fixture_engine_is_live(e) &&\n        g_live_count == 1 && g_real_current == fixture_known_live_bytes() &&\n        g_real_current == sizeof(*e) && fixture_current_matches_known_live();\n    int retry_done = consumed && g_container_free_calls == 1 &&\n        current_before_engine_cleanup && e->exact_cache_active_sessions == 0 &&\n        g_exact_unlock_entry_checks == 1 &&\n        g_exact_unlock_entry_violations == 0;\n    int engine_cleaned = consumed && fixture_engine_is_live(e) &&\n        fixture_cleanup_engine(e);\n    printf("first=%d\\nretained=%d\\nsecond=%d\\nretry_done=%d\\ncurrent_before_engine_cleanup=%d\\nengine_cleaned=%d\\n",\n           first, retained, second, retry_done, current_before_engine_cleanup,\n           engine_cleaned);\n    return first == 0 && retained && second == 1 && retry_done &&\n        engine_cleaned && standard_sensor_clean();\n}\nstatic int scenario_no_alloc(void) {\n    if (!reset_fixture()) return 0;\n    ds4_engine *e = make_engine(FAKE_FAMILY_GENERIC, 1);\n    ds4_session *owner = NULL;\n    ds4_session *s = make_no_alloc_session(e, 1);\n    if (!fixture_engine_is_live(e) || !fixture_session_is_live(s) ||\n        !publish(&owner, s)) return 0;\n    int before = g_allocation_attempts;\n    int first = fixture_call_checked(&owner);\n    int consumed = owner == NULL && !fixture_session_is_live(s);\n    int state = consumed && fixture_engine_is_live(e) &&\n        g_allocation_attempts == before && g_release_allocation_attempts == 0 &&\n        g_container_reserved_at_free == 0 && g_container_active_at_free == 0 &&\n        g_exact_lock_calls == 1 && g_exact_unlock_calls == 1 &&\n        g_exact_unlock_entry_checks == 1 &&\n        g_exact_unlock_entry_violations == 0 && g_old_release_calls == 0;\n    int engine_cleaned = consumed && fixture_engine_is_live(e) &&\n        fixture_cleanup_engine(e);\n    printf("first=%d\\nstate=%d\\nengine_cleaned=%d\\n", first, state,\n           engine_cleaned);\n    return first == 1 && state && engine_cleaned && standard_sensor_clean();\n}\nstatic int scenario_legacy_retry(void) {\n    if (!reset_fixture()) return 0;\n    ds4_engine *e = make_engine(FAKE_FAMILY_GENERIC, 1);\n    ds4_session *owner = NULL;\n    ds4_session *s = make_no_alloc_session(e, 1);\n    if (!fixture_engine_is_live(e) || !fixture_session_is_live(s) ||\n        !publish(&owner, s)) return 0;\n    g_exact_lock_refuse = 1;\n    int before = g_allocation_attempts;\n    int legacy_no_alloc = fixture_call_legacy(owner);\n    int s_live = fixture_session_is_live(s);\n    int e_live = fixture_engine_is_live(e);\n    int legacy_retained = s_live && e_live && owner == s &&\n        s->exact_cache_session_reserved && e->exact_cache_active_sessions == 1 &&\n        g_old_release_calls == 0 && legacy_no_alloc;\n    g_exact_lock_refuse = 0;\n    int checked = legacy_retained && fixture_safe_retry_owner(&owner, s)\n        ? fixture_call_checked(&owner) : 0;\n    int consumed = owner == NULL && !fixture_session_is_live(s);\n    int retry_done = consumed && fixture_engine_is_live(e) &&\n        g_allocation_attempts == before && g_release_allocation_attempts == 0 &&\n        g_old_release_calls == 0 && g_exact_lock_calls == 2 &&\n        g_exact_unlock_calls == 1 && g_exact_unlock_entry_checks == 1 &&\n        g_exact_unlock_entry_violations == 0;\n    int engine_cleaned = consumed && fixture_engine_is_live(e) &&\n        fixture_cleanup_engine(e);\n    printf("checked=%d\\nlegacy_retained=%d\\nlegacy_no_alloc=%d\\nretry_done=%d\\nengine_cleaned=%d\\n",\n           checked, legacy_retained, legacy_no_alloc, retry_done,\n           engine_cleaned);\n    return checked == 1 && legacy_retained && legacy_no_alloc && retry_done &&\n        engine_cleaned && standard_sensor_clean();\n}\n'
    + "\n"
    + '\n\nstatic void print_sensor_state(void) {\n    printf("allocation_attempts=%d\\n", g_allocation_attempts);\n    printf("release_boundary_calls=%d\\n", g_release_boundary_calls);\n    printf("release_allocation_attempts=%d\\n", g_release_allocation_attempts);\n    printf("alloc_calls=%d\\n", g_alloc_calls);\n    printf("malloc_calls=%d\\n", g_malloc_calls);\n    printf("calloc_calls=%d\\n", g_calloc_calls);\n    printf("alloc_bytes=%llu\\n", (unsigned long long)g_alloc_bytes);\n    printf("real_current=%llu\\n", (unsigned long long)g_real_current);\n    printf("known_live_bytes=%llu\\n", (unsigned long long)fixture_known_live_bytes());\n    printf("sensor_current_matches=%d\\n", fixture_current_matches_known_live());\n    printf("real_peak=%llu\\n", (unsigned long long)g_real_peak);\n    printf("max_alloc=%llu\\n", (unsigned long long)g_max_alloc);\n    printf("remaining_live=%llu\\n", (unsigned long long)g_live_count);\n    printf("physical_free_calls=%d\\n", g_physical_free_calls);\n    printf("unknown_free_calls=%d\\n", g_unknown_free_calls);\n    printf("double_free_calls=%d\\n", g_double_free_calls);\n    printf("unknown_mutex_calls=%d\\n", g_unknown_mutex_calls);\n    printf("free_order_violations=%d\\n", g_free_order_violations);\n    printf("fixture_protocol_errors=%d\\n", g_fixture_protocol_errors);\n    printf("sensor_sum_violations=%d\\n", g_sensor_sum_violations);\n    printf("container_free_calls=%d\\n", g_container_free_calls);\n    printf("container_published_at_free=%d\\n", g_container_free_calls == 1);\n    printf("container_common_empty_at_free=%d\\n", g_container_common_empty_at_free);\n    printf("container_reserved_at_free=%d\\n", g_container_reserved_at_free);\n    printf("container_active_at_free=%u\\n", g_container_active_at_free);\n    printf("exact_lock_calls=%d\\n", g_exact_lock_calls);\n    printf("exact_unlock_calls=%d\\n", g_exact_unlock_calls);\n    printf("exact_unlock_entry_checks=%d\\n", g_exact_unlock_entry_checks);\n    printf("exact_unlock_entry_violations=%d\\n", g_exact_unlock_entry_violations);\n    printf("compact_lock_calls=%d\\n", g_compact_lock_calls);\n    printf("compact_unlock_calls=%d\\n", g_compact_unlock_calls);\n    printf("graph_calls=%d\\n", g_graph_calls);\n    printf("graph_physical_frees=%d\\n", g_graph_physical_frees);\n    printf("cpu_free_calls=%d\\n", g_cpu_free_calls);\n    printf("scratch_free_calls=%d\\n", g_scratch_free_calls);\n    printf("dist_free_calls=%d\\n", g_dist_free_calls);\n    printf("tp_destroy_calls=%d\\n", g_tp_destroy_calls);\n    printf("old_release_calls=%d\\n", g_old_release_calls);\n}\nstatic int finish_case(int scenario_ok) {\n    print_sensor_state();\n    return scenario_ok == 1 && standard_sensor_clean() ? 0 : 3;\n}\nint main(int argc, char **argv) {\n    alarm(15);\n    struct rlimit core_limit = {0, 0};\n    (void)setrlimit(RLIMIT_CORE, &core_limit);\n    if (argc != 2) return 2;\n    int scenario_ok = 0;\n    if (!strcmp(argv[1], "nulls")) scenario_ok = scenario_nulls();\n    else if (!strncmp(argv[1], "success-", 8)) scenario_ok = scenario_success(argv[1]);\n    else if (!strcmp(argv[1], "lock-refusal")) scenario_ok = scenario_lock_refusal();\n    else if (!strcmp(argv[1], "graph-refusal")) scenario_ok = scenario_graph_refusal();\n    else if (!strcmp(argv[1], "unlock-refusal")) scenario_ok = scenario_unlock_refusal();\n    else if (!strcmp(argv[1], "distributed-prefix")) scenario_ok = scenario_distributed_prefix();\n    else if (!strncmp(argv[1], "exact-", 6)) {\n        scenario_ok = !strcmp(argv[1], "exact-missing-engine")\n            ? scenario_exact_missing_engine() : exact_case(argv[1]);\n    } else if (!strcmp(argv[1], "no-alloc")) scenario_ok = scenario_no_alloc();\n    else if (!strcmp(argv[1], "legacy-retry")) scenario_ok = scenario_legacy_retry();\n    else return 2;\n    return finish_case(scenario_ok);\n}\n'
)

ABI_HEADER = '#ifndef DS4_CHECKED_RELEASE_CALLER_H\n#define DS4_CHECKED_RELEASE_CALLER_H\n#include "ds4.h"\n#ifdef __cplusplus\nextern "C" {\n#endif\nint checked_release_caller_null_owner(void);\nint checked_release_caller_empty_owner(void);\n#ifdef __cplusplus\n}\n#endif\n#endif\n'
ABI_CALLER = '#include "checked_release_caller.h"\nint checked_release_caller_null_owner(void) {\n    return ds4_session_free_checked(NULL) == 0;\n}\nint checked_release_caller_empty_owner(void) {\n    ds4_session *owner = NULL;\n    return ds4_session_free_checked(&owner) == 1 && owner == NULL;\n}\n'


def parse_output(stdout: str) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in stdout.splitlines():
        key, sep, raw = line.partition("=")
        if sep and re.fullmatch(r"-?\d+", raw):
            values[key] = int(raw)
    return values


def logical_make_lines(source: str) -> list[str]:
    lines: list[str] = []
    pending = ""
    for raw in source.splitlines():
        line = raw.rstrip()
        pending = pending + " " + line.lstrip() if pending else line
        if pending.endswith("\\"):
            pending = pending[:-1].rstrip()
        else:
            lines.append(pending)
            pending = ""
    if pending:
        lines.append(pending)
    return lines


class SessionCheckedReleaseFixture(unittest.TestCase):
    """Compile and execute only generated, fake-driver children."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory(
            prefix="ds4-session-checked-release-contract-"
        )
        cls.addClassCleanup(cls._tmp.cleanup)
        base = Path(cls._tmp.name)
        cls._env = dict(SAFE_ENV)
        for key, leaf in (
            ("HOME", "home"),
            ("TMPDIR", "tmp"),
            ("XDG_CONFIG_HOME", "xdg-config"),
            ("XDG_CACHE_HOME", "xdg-cache"),
            ("XDG_DATA_HOME", "xdg-data"),
        ):
            child = base / leaf
            child.mkdir()
            cls._env[key] = str(child)
        cls._source = base / "checked_release_fixture.c"
        cls._abi_header = base / "checked_release_caller.h"
        cls._abi_source = base / "checked_release_caller.c"
        cls._source_object = base / "checked_release_fixture.o"
        cls._abi_object = base / "checked_release_caller.o"
        cls._binary = base / "checked_release_fixture"
        cls._source.write_text(FIXTURE_SOURCE, encoding="utf-8")
        cls._abi_header.write_text(ABI_HEADER, encoding="utf-8")
        cls._abi_source.write_text(ABI_CALLER, encoding="utf-8")
        cls._abi_compile = cls._fixture_compile = cls._link = None
        if MISSING_SOURCE_SEAMS:
            return
        common = [
            "cc",
            "-std=c11",
            "-O0",
            "-Wall",
            "-Wextra",
            "-Werror=format",
            "-pthread",
            "-DDS4_TEST_HOOKS",
            "-I",
            str(ROOT),
            "-I",
            str(base),
        ]
        cls._abi_compile = subprocess.run(
            common + ["-c", str(cls._abi_source), "-o", str(cls._abi_object)],
            cwd=base,
            env=cls._env,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        cls._fixture_compile = subprocess.run(
            common + ["-c", str(cls._source), "-o", str(cls._source_object)],
            cwd=base,
            env=cls._env,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if cls._abi_compile.returncode == 0 and cls._fixture_compile.returncode == 0:
            cls._link = subprocess.run(
                common
                + [
                    str(cls._source_object),
                    str(cls._abi_object),
                    "-o",
                    str(cls._binary),
                ],
                cwd=base,
                env=cls._env,
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )

    def assert_compile_prerequisite(self) -> None:
        if MISSING_SOURCE_SEAMS:
            self.fail(
                "checked-release feature RED; missing actual source/header seams: "
                + ", ".join(MISSING_SOURCE_SEAMS)
            )
        failures = []
        for label, result in (
            ("C ABI caller", self._abi_compile),
            ("checked-release fixture", self._fixture_compile),
            ("checked-release link", self._link),
        ):
            if result is None:
                failures.append(label + ": did not run")
            elif result.returncode != 0:
                failures.append(label + ":\n" + result.stderr)
        self.assertFalse(
            failures,
            "checked-release fixture infrastructure failure (compile/link; not feature RED):\n"
            + "\n".join(failures),
        )

    def run_case(self, name: str) -> dict[str, int]:
        self.assert_compile_prerequisite()
        result = subprocess.run(
            [str(self._binary), name],
            cwd=self._binary.parent,
            env=self._env,
            capture_output=True,
            text=True,
            timeout=16,
            check=False,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"checked-release behavioral failure in executed scenario {name}:\n"
            f"stdout={result.stdout}\nstderr={result.stderr}",
        )
        values = parse_output(result.stdout)
        self.assertEqual(values.get("release_allocation_attempts"), 0)
        self.assertGreater(values.get("release_boundary_calls", 0), 0)
        self.assertEqual(values.get("unknown_free_calls"), 0)
        self.assertEqual(values.get("double_free_calls"), 0)
        self.assertEqual(values.get("unknown_mutex_calls"), 0)
        self.assertEqual(values.get("exact_unlock_entry_violations"), 0)
        self.assertEqual(values.get("free_order_violations"), 0)
        self.assertEqual(values.get("fixture_protocol_errors"), 0)
        self.assertEqual(values.get("sensor_sum_violations"), 0)
        self.assertEqual(values.get("sensor_current_matches"), 1)
        self.assertEqual(values.get("remaining_live"), 0)
        self.assertEqual(values.get("known_live_bytes"), 0)
        self.assertEqual(values.get("real_current"), 0)
        self.assertLessEqual(values.get("max_alloc", 65537), 4096)
        self.assertLessEqual(values.get("alloc_bytes", 65537), 65536)
        return values

    @staticmethod
    def assert_fields(values: dict[str, int], expected: dict[str, int]) -> None:
        for key, wanted in expected.items():
            if key not in values:
                raise AssertionError(f"missing {key}: {values}")
            if values[key] != wanted:
                raise AssertionError(f"{key}: wanted {wanted}, got {values[key]}")

    def test_00_runtime_paths_and_source_header_seams_are_real(self) -> None:
        self.assertEqual(ROOT, Path(__file__).resolve().parents[1])
        self.assertEqual(DS4_SOURCE_PATH, ROOT / "ds4.c")
        self.assertEqual(DS4_HEADER_PATH, ROOT / "ds4.h")
        self.assertEqual(MAKEFILE_PATH, ROOT / "Makefile")
        self.assertFalse(
            MISSING_SOURCE_SEAMS,
            "checked-release feature RED; missing actual seams: "
            + ", ".join(MISSING_SOURCE_SEAMS),
        )
        self.assertEqual(DS4_SOURCE.count(SESSION_START_MARKER), 1)
        self.assertEqual(DS4_SOURCE.count(SESSION_END_MARKER), 1)
        self.assertIsNotNone(SESSION_BLOCK)
        assert SESSION_BLOCK is not None
        self.assertIn("ds4_session_release_exact_cache_reservation", SESSION_BLOCK)
        self.assertIn("ds4_session_free_checked", SESSION_BLOCK)
        self.assertIn("ds4_session_free", SESSION_BLOCK)

    def test_01_c_header_link_and_null_owner_noops(self) -> None:
        values = self.run_case("nulls")
        self.assert_fields(
            values,
            {
                "null_owner": 1,
                "empty_owner": 1,
                "nulls_no_alloc": 1,
                "nulls_no_free": 1,
            },
        )

    def test_02_successful_backend_branches_consume_once(self) -> None:
        for name in ("success-cpu", "success-laguna", "success-glm", "success-generic"):
            with self.subTest(name=name):
                values = self.run_case(name)
                self.assert_fields(
                    values,
                    {
                        "first": 1,
                        "second": 1,
                        "state_ok": 1,
                        "tp_calls_ok": 1,
                        "no_alloc": 1,
                        "old_wrapper_unused": 1,
                        "engine_cleaned": 1,
                        "container_free_calls": 1,
                        "container_published_at_free": 1,
                    },
                )

    def test_03_laguna_lock_graph_and_unlock_refusals_retry(self) -> None:
        expected = {
            "lock-refusal": "retained",
            "graph-refusal": "failed_graph",
            "unlock-refusal": "retained_after_unlock",
        }
        for name, state_key in expected.items():
            with self.subTest(name=name):
                values = self.run_case(name)
                self.assert_fields(
                    values,
                    {
                        "first": 0,
                        state_key: 1,
                        "second": 1,
                        "retried": 1,
                        "engine_cleaned": 1,
                    },
                )

    def test_04_distributed_prefix_clears_before_laguna_retry(self) -> None:
        values = self.run_case("distributed-prefix")
        self.assert_fields(
            values,
            {
                "first": 0,
                "prefix": 1,
                "second": 1,
                "retried": 1,
                "dist_free_calls": 1,
                "engine_cleaned": 1,
            },
        )

    def test_05_exact_reservation_controls_retain_and_retry(self) -> None:
        for name in (
            "exact-missing-mutex",
            "exact-lock-refusal",
            "exact-zero-count",
            "exact-unlock-refusal",
            "exact-no-reservation",
        ):
            with self.subTest(name=name):
                values = self.run_case(name)
                expected = {
                    "second": 1,
                    "retry_done": 1,
                    "exact_state": 1,
                    "common_once": 1,
                    "current_before_engine_cleanup": 1,
                    "engine_cleaned": 1,
                }
                if name == "exact-no-reservation":
                    expected.update(
                        first=1,
                        retained=0,
                        common_done=1,
                        no_reservation_consumed=1,
                    )
                else:
                    expected.update(first=0, retained=1, common_done=1)
                self.assert_fields(values, expected)

    def test_06_missing_engine_helper_boundary_restores_then_retries(self) -> None:
        values = self.run_case("exact-missing-engine")
        self.assert_fields(
            values,
            {
                "first": 0,
                "retained": 1,
                "second": 1,
                "retry_done": 1,
                "current_before_engine_cleanup": 1,
                "engine_cleaned": 1,
            },
        )

    def test_07_no_alloc_hook_and_legacy_retry_delegate(self) -> None:
        no_alloc = self.run_case("no-alloc")
        self.assert_fields(
            no_alloc,
            {"first": 1, "state": 1, "engine_cleaned": 1, "container_free_calls": 1},
        )
        legacy = self.run_case("legacy-retry")
        self.assert_fields(
            legacy,
            {
                "checked": 1,
                "legacy_retained": 1,
                "legacy_no_alloc": 1,
                "retry_done": 1,
                "engine_cleaned": 1,
            },
        )

    def test_08_sensor_limits_and_typed_coverage_are_explicit(self) -> None:
        self.assertIn("CONTROL-FLOW FAKES", FIXTURE_SOURCE)
        self.assertIn("4096u", FIXTURE_SOURCE)
        self.assertIn("65536u", FIXTURE_SOURCE)
        self.assertIn("alarm(15)", FIXTURE_SOURCE)
        self.assertIn("RLIMIT_CORE", FIXTURE_SOURCE)
        self.assertIn("fixture_pointer_is_live", FIXTURE_SOURCE)
        self.assertIn("g_release_allocation_attempts", FIXTURE_SOURCE)
        self.assertIn("g_unknown_mutex_calls", FIXTURE_SOURCE)
        self.assertIn("g_exact_unlock_entry_violations", FIXTURE_SOURCE)
        self.assertIn("Generated-control-flow temporal oracle", FIXTURE_SOURCE)
        self.assertIn(
            "static int ds4_session_release_exact_cache_reservation(ds4_session *s);",
            FIXTURE_SOURCE,
        )
        self.assertIn("no_reservation_consumed", FIXTURE_SOURCE)
        self.assertIn("libc free is physical cleanup", FIXTURE_SOURCE)
        self.assertIn("exact-missing-engine", FIXTURE_SOURCE)
        self.assertNotIn('#include "ds4.c"', FIXTURE_SOURCE)
        self.assertNotIn("cudaMalloc", FIXTURE_SOURCE)

    def test_09_make_target_is_in_all_required_aggregates(self) -> None:
        target = "test-session-checked-release"
        self.assertRegex(
            MAKEFILE_SOURCE,
            r"(?m)^" + re.escape(target) + r":\s*\n"
            r"\tpython3 tests/test_session_checked_release\.py -v\s*$",
        )
        phony = {
            item
            for line in MAKEFILE_SOURCE.splitlines()
            if line.startswith(".PHONY:")
            for item in line.split(":", 1)[1].split()
        }
        self.assertIn(target, phony)
        logical = logical_make_lines(MAKEFILE_SOURCE)
        resident = next(line for line in logical if line.startswith("test-laguna-resident-path:"))
        default = next(line for line in logical if line.startswith("test:"))
        self.assertIn(target, resident.split(":", 1)[1].split())
        self.assertIn("test-laguna-resident-path", default.split(":", 1)[1].split())
        self.assertIn(target, default.split(":", 1)[1].split())


if __name__ == "__main__":
    unittest.main()
