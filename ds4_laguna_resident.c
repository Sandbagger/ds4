#include "ds4_laguna_resident.h"

#include <stdio.h>
#include <string.h>

static bool resident_error(char *error, size_t size, const char *message) {
    if (error && size != 0u) snprintf(error, size, "%s", message);
    return false;
}

static bool resident_add(uint64_t a, uint64_t b, uint64_t *out) {
    if (a > UINT64_MAX - b) return false;
    *out = a + b;
    return true;
}

static bool resident_mul(uint64_t a, uint64_t b, uint64_t *out) {
    if (a != 0u && b > UINT64_MAX / a) return false;
    *out = a * b;
    return true;
}

static bool resident_callsite(
        ds4_laguna_resident_plan *plan, uint32_t id, const char *name,
        ds4_runtime_category category, ds4_runtime_physical_domain domain,
        uint64_t bound) {
    if (plan->callsite_count >= DS4_LAGUNA_RESIDENT_CALLSITE_COUNT ||
        !resident_add(plan->category_bounds[category], bound,
                      &plan->category_bounds[category])) return false;
    plan->callsites[plan->callsite_count++] = (ds4_runtime_callsite){
        .id = id, .name = name, .category = category, .domain = domain,
        .bound_bytes = bound,
    };
    return true;
}

bool ds4_laguna_resident_plan_make(
        ds4_laguna_resident_plan *out,
        const ds4_laguna_ledger *ledger,
        const ds4_laguna_resident_plan_spec *spec,
        char *error, size_t error_size) {
    if (error && error_size != 0u) error[0] = '\0';
    if (out) memset(out, 0, sizeof(*out));
    if (!out || !ledger || !spec) {
        return resident_error(error, error_size, "resident plan argument is null");
    }
    if (spec->context_tokens != 32768u || spec->prefill_rows != 4096u ||
        spec->session_count != 1u) {
        return resident_error(error, error_size,
                              "resident plan requires 32K/4K/one session");
    }
    const uint64_t page_size = spec->source_page_size;
    if (spec->host_capacity_bytes == 0u || spec->cuda_capacity_bytes == 0u ||
        page_size < 512u || page_size > 65536u ||
        (page_size & (page_size - 1u)) != 0u) {
        return resident_error(error, error_size,
                              "resident capacities or source page size are invalid");
    }
    /* The same pinned ledger geometry as the streamed qualification, without
     * reinterpreting an 8/12/16-GiB cache plan as a resident observation. */
    if (ledger->file_size != UINT64_C(68248759648) ||
        ledger->tensor_count != UINT64_C(814) ||
        ledger->static_parent_count != UINT64_C(673) ||
        ledger->routed_parent_count != UINT64_C(141) ||
        ledger->static_aligned_device_bytes != UINT64_C(4374164480) ||
        ledger->expert_entry_count != UINT64_C(12032) ||
        ledger->slot_stride_bytes != UINT64_C(5308416)) {
        return resident_error(error, error_size,
                              "resident plan requires the exact Laguna ledger");
    }

    ds4_laguna_resident_plan plan = {0};
    plan.context_tokens = spec->context_tokens;
    plan.prefill_rows = spec->prefill_rows;
    plan.session_count = spec->session_count;
    uint64_t source_capacity = 0u;
    uint64_t tensor_bytes = 0u, source_bytes = 0u, expert_bytes = 0u;
    uint64_t ledger_bytes = 0u, kv_tokens = 0u, kv_bytes = 0u;
    uint64_t graph_bytes = 0u, rounded_source = 0u;
    if (!resident_mul(ledger->tensor_count, 2u, &source_capacity) ||
        !resident_add(source_capacity, 5u, &source_capacity) ||
        !resident_mul(ledger->tensor_count, sizeof(ledger->tensor_ranges[0]),
                      &tensor_bytes) ||
        !resident_mul(source_capacity, sizeof(ledger->source_ranges[0]),
                      &source_bytes) ||
        !resident_mul(ledger->expert_entry_count, sizeof(ledger->expert_entries[0]),
                      &expert_bytes) ||
        !resident_add(tensor_bytes, source_bytes, &ledger_bytes) ||
        !resident_add(ledger_bytes, expert_bytes, &ledger_bytes) ||
        !resident_mul(12u, spec->context_tokens, &kv_tokens) ||
        !resident_add(kv_tokens, UINT64_C(36) * 512u, &kv_tokens) ||
        !resident_mul(kv_tokens, 4096u, &kv_bytes) ||
        !ds4_runtime_checked_affine_bytes(
            spec->prefill_rows, UINT64_C(375156), UINT64_C(413704), &graph_bytes) ||
        !resident_add(ledger->file_size, page_size - 1u, &rounded_source)) {
        return resident_error(error, error_size, "resident geometry overflow");
    }
    rounded_source &= ~(page_size - 1u);

    /* Capacities bound possible legacy owners, including full/range/arena and
     * optional Q8 caches. They must never be copied into current/peak bytes.
     * Pinned owners charge raw reservations, not aligned usable stage spans. */
#define RESIDENT_SITE(id, name, category, domain, bytes) do { \
        if (!resident_callsite(&plan, id, name, category, domain, bytes)) { \
            return resident_error(error, error_size, "resident callsite overflow"); \
        } \
    } while (0)
    RESIDENT_SITE(DS4_LAGUNA_CALLSITE_STATIC_SLAB,
        "laguna.resident.static_weights", DS4_RUNTIME_CATEGORY_STATIC_WEIGHTS,
        DS4_RUNTIME_DOMAIN_CUDA_DEVICE, spec->cuda_capacity_bytes);
    RESIDENT_SITE(DS4_LAGUNA_CALLSITE_LEDGER_ARRAYS,
        "laguna.ledger_arrays", DS4_RUNTIME_CATEGORY_CACHE_METADATA_ADDRESS_TABLES,
        DS4_RUNTIME_DOMAIN_HOST, ledger_bytes);
    RESIDENT_SITE(DS4_LAGUNA_CALLSITE_KV_STATE,
        "laguna.kv_state", DS4_RUNTIME_CATEGORY_KV_STATE,
        DS4_RUNTIME_DOMAIN_CUDA_DEVICE, kv_bytes);
    RESIDENT_SITE(DS4_LAGUNA_CALLSITE_GRAPH_SCRATCH,
        "laguna.graph_scratch", DS4_RUNTIME_CATEGORY_GRAPH_SCRATCH,
        DS4_RUNTIME_DOMAIN_CUDA_DEVICE, graph_bytes);
    RESIDENT_SITE(DS4_LAGUNA_CALLSITE_PINNED_STAGING_0,
        "laguna.resident.pinned_staging", DS4_RUNTIME_CATEGORY_PINNED_STAGING,
        DS4_RUNTIME_DOMAIN_HOST, spec->host_capacity_bytes);

    /* Preserve the existing one-session host inventory envelopes and IDs. */
    const uint64_t mib = UINT64_C(1024) * 1024u;
    const uint64_t host_mib[] = {64u, 192u, 128u, 256u, 128u, 128u, 128u};
    const char *const host_names[] = {
        "laguna.other_host.engine", "laguna.other_host.model",
        "laguna.other_host.bootstrap", "laguna.other_host.vocab",
        "laguna.other_host.session", "laguna.other_host.tracker",
        "laguna.other_host.serializer",
    };
    for (size_t i = 0u; i < sizeof(host_mib) / sizeof(host_mib[0]); i++) {
        RESIDENT_SITE(DS4_LAGUNA_CALLSITE_OTHER_HOST_ENGINE + (uint32_t)i,
            host_names[i], DS4_RUNTIME_CATEGORY_OTHER_HOST,
            DS4_RUNTIME_DOMAIN_HOST, host_mib[i] * mib);
    }
    RESIDENT_SITE(DS4_LAGUNA_CALLSITE_OTHER_CUDA_KERNEL_TMP,
        "laguna.resident.other_cuda", DS4_RUNTIME_CATEGORY_OTHER_CUDA,
        DS4_RUNTIME_DOMAIN_CUDA_DEVICE, spec->cuda_capacity_bytes);
    RESIDENT_SITE(DS4_LAGUNA_RESIDENT_CALLSITE_OTHER_MANAGED,
        "laguna.resident.other_managed", DS4_RUNTIME_CATEGORY_OTHER_CUDA,
        DS4_RUNTIME_DOMAIN_CUDA_MANAGED, spec->cuda_capacity_bytes);
#undef RESIDENT_SITE

    plan.report_bounds[DS4_RUNTIME_REPORT_MODEL_MAPPED_VIRTUAL] = ledger->file_size;
    plan.report_bounds[DS4_RUNTIME_REPORT_MODEL_MAPPING_REGISTERED] = rounded_source;
    plan.report_bounds[DS4_RUNTIME_REPORT_MODEL_SOURCE_RESIDENT] = rounded_source;
    plan.report_bounds[DS4_RUNTIME_REPORT_HOST_LIBRARY_UNATTRIBUTED] =
        spec->host_capacity_bytes;
    plan.report_bounds[DS4_RUNTIME_REPORT_CUDA_LIBRARY_UNATTRIBUTED] =
        spec->cuda_capacity_bytes;
    for (size_t i = 0u; i < DS4_RUNTIME_OWNED_CATEGORY_COUNT; i++) {
        if (!resident_add(plan.owned_total_bound_bytes, plan.category_bounds[i],
                          &plan.owned_total_bound_bytes)) {
            return resident_error(error, error_size, "resident owned total overflow");
        }
    }
    plan.qualification_total_bound_bytes = plan.owned_total_bound_bytes;
    const ds4_runtime_report external[] = {
        DS4_RUNTIME_REPORT_MODEL_SOURCE_RESIDENT,
        DS4_RUNTIME_REPORT_HOST_LIBRARY_UNATTRIBUTED,
        DS4_RUNTIME_REPORT_CUDA_LIBRARY_UNATTRIBUTED,
    };
    for (size_t i = 0u; i < sizeof(external) / sizeof(external[0]); i++) {
        if (!resident_add(plan.qualification_total_bound_bytes,
                          plan.report_bounds[external[i]],
                          &plan.qualification_total_bound_bytes)) {
            return resident_error(error, error_size,
                                  "resident qualification total overflow");
        }
    }
    *out = plan;
    return true;
}
