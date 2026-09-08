#define _POSIX_C_SOURCE 200809L

/* Standalone reservation and ownership accounting contract tests for the resident
 * Laguna plan.  These tests use only synthetic ledger geometry and fake
 * addresses; they never open a model or require a GPU. */

#include "ds4_laguna_resident.h"
#include "ds4_laguna_stream.h"
#include "ds4_runtime.h"

#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int g_failed;
static int g_total;

#define CHECK(condition, message) do {                                         \
    g_total++;                                                                 \
    if (!(condition)) {                                                         \
        fprintf(stderr, "FAIL: %s (line %d)\n", (message), __LINE__);        \
        g_failed++;                                                            \
    }                                                                          \
} while (0)

enum {
    RESIDENT_CALLSITE_COUNT = 14,
    RESIDENT_CALLSITE_MANAGED = 26,
};

static const uint64_t GIB = UINT64_C(1024) * 1024u * 1024u;
static const uint64_t MIB = UINT64_C(1024) * 1024u;

static bool all_zero(const void *value, size_t size) {
    const unsigned char *bytes = value;
    for (size_t i = 0; i < size; i++) {
        if (bytes[i] != 0) return false;
    }
    return true;
}

static ds4_laguna_ledger canonical_ledger(void) {
    ds4_laguna_ledger ledger;
    memset(&ledger, 0, sizeof(ledger));
    ledger.file_size = UINT64_C(68248759648);
    ledger.tensor_count = UINT64_C(814);
    ledger.static_parent_count = UINT64_C(673);
    ledger.routed_parent_count = UINT64_C(141);
    ledger.static_aligned_device_bytes = UINT64_C(4374164480);
    ledger.expert_entry_count = UINT64_C(12032);
    ledger.slot_stride_bytes = UINT64_C(5308416);
    return ledger;
}

static ds4_laguna_resident_plan_spec canonical_spec(void) {
    ds4_laguna_resident_plan_spec spec;
    memset(&spec, 0, sizeof(spec));
    spec.context_tokens = 32768u;
    spec.prefill_rows = 4096u;
    spec.session_count = 1u;
    /* Deliberately not a streamed cache profile: capacities are independent
     * physical envelopes used by the resident plan. */
    spec.host_capacity_bytes = 3u * GIB;
    spec.cuda_capacity_bytes = 5u * GIB;
    spec.source_page_size = 4096u;
    return spec;
}

static uint64_t page_round(uint64_t bytes, uint64_t page_size) {
    return ((bytes + page_size - 1u) / page_size) * page_size;
}

/* This is the ledger-array component used by the existing streamed plan.
 * Resident mode has no compact cache owners, so it does not add the streamed
 * route/slot metadata envelopes to callsite 3. */
static uint64_t expected_ledger_reservation(const ds4_laguna_ledger *ledger) {
    const uint64_t source_range_count = ledger->tensor_count * 2u + 5u;
    return ledger->tensor_count * sizeof(ds4_laguna_tensor_range) +
           source_range_count * sizeof(ds4_laguna_source_range) +
           ledger->expert_entry_count * sizeof(ds4_laguna_expert_entry);
}

static void expect_rejected(const ds4_laguna_ledger *ledger,
                            const ds4_laguna_resident_plan_spec *spec,
                            const char *message) {
    ds4_laguna_resident_plan plan;
    char error[256];
    memset(&plan, 0xa5, sizeof(plan));
    memset(error, 0, sizeof(error));
    const bool ok = ds4_laguna_resident_plan_make(
        &plan, ledger, spec, error, sizeof(error));
    CHECK(!ok, message);
    CHECK(all_zero(&plan, sizeof(plan)),
          "resident plan failure clears the complete output");
}

static void check_plan_contract(const ds4_laguna_ledger *ledger,
                                const ds4_laguna_resident_plan_spec *spec,
                                const ds4_laguna_resident_plan *plan) {
    const uint64_t kv_bytes =
        (UINT64_C(12) * spec->context_tokens + UINT64_C(36) * 512u) * 4096u;
    const uint64_t graph_bytes =
        UINT64_C(4096) * UINT64_C(375156) + UINT64_C(413704);
    const uint64_t metadata_bytes = expected_ledger_reservation(ledger);
    const uint64_t source_resident =
        page_round(ledger->file_size, spec->source_page_size);
    const uint64_t expected_category_bounds[DS4_RUNTIME_OWNED_CATEGORY_COUNT] = {
        [DS4_RUNTIME_CATEGORY_STATIC_WEIGHTS] = spec->cuda_capacity_bytes,
        [DS4_RUNTIME_CATEGORY_EXPERT_CACHE_PAYLOAD] = 0,
        [DS4_RUNTIME_CATEGORY_CACHE_METADATA_ADDRESS_TABLES] = metadata_bytes,
        [DS4_RUNTIME_CATEGORY_KV_STATE] = kv_bytes,
        [DS4_RUNTIME_CATEGORY_GRAPH_SCRATCH] = graph_bytes,
        [DS4_RUNTIME_CATEGORY_PINNED_STAGING] = spec->host_capacity_bytes,
        [DS4_RUNTIME_CATEGORY_OTHER_HOST] = GIB,
        [DS4_RUNTIME_CATEGORY_OTHER_CUDA] = 2u * spec->cuda_capacity_bytes,
    };
    const uint64_t expected_report_bounds[DS4_RUNTIME_REPORT_COUNT] = {
        [DS4_RUNTIME_REPORT_MODEL_MAPPED_VIRTUAL] = ledger->file_size,
        [DS4_RUNTIME_REPORT_MODEL_MAPPING_REGISTERED] = source_resident,
        [DS4_RUNTIME_REPORT_MODEL_SOURCE_RESIDENT] = source_resident,
        [DS4_RUNTIME_REPORT_HOST_LIBRARY_UNATTRIBUTED] =
            spec->host_capacity_bytes,
        [DS4_RUNTIME_REPORT_CUDA_LIBRARY_UNATTRIBUTED] =
            spec->cuda_capacity_bytes,
    };
    uint64_t expected_owned_total = 0;
    uint64_t expected_qualification_total = 0;
    for (size_t i = 0; i < DS4_RUNTIME_OWNED_CATEGORY_COUNT; i++) {
        expected_owned_total += expected_category_bounds[i];
        CHECK(plan->category_bounds[i] == expected_category_bounds[i],
              "resident category bound follows the explicit physical contract");
    }
    for (size_t i = 0; i < DS4_RUNTIME_REPORT_COUNT; i++) {
        expected_qualification_total += expected_report_bounds[i];
        CHECK(plan->report_bounds[i] == expected_report_bounds[i],
              "resident report bound follows the page-rounded source contract");
    }
    /* Mapping and registration are report-only.  Qualification adds only the
     * source-resident, host-unattributed, and CUDA-unattributed reports. */
    expected_qualification_total = expected_owned_total +
        source_resident + spec->host_capacity_bytes + spec->cuda_capacity_bytes;

    CHECK(plan->context_tokens == 32768u && plan->prefill_rows == 4096u &&
              plan->session_count == 1u,
          "resident plan retains the canonical one-session geometry");
    CHECK(plan->owned_total_bound_bytes == expected_owned_total,
          "resident owned total is the checked sum of category bounds");
    CHECK(plan->qualification_total_bound_bytes == expected_qualification_total,
          "resident qualification total excludes mapping and registration twice");

    const uint64_t host_split[7] = {
        64u * MIB, 192u * MIB, 128u * MIB, 256u * MIB,
        128u * MIB, 128u * MIB, 128u * MIB,
    };
    const struct {
        uint32_t id;
        ds4_runtime_category category;
        ds4_runtime_physical_domain domain;
        uint64_t bound;
    } expected_callsites[RESIDENT_CALLSITE_COUNT] = {
        { 1u, DS4_RUNTIME_CATEGORY_STATIC_WEIGHTS,
          DS4_RUNTIME_DOMAIN_CUDA_DEVICE, spec->cuda_capacity_bytes },
        { 3u, DS4_RUNTIME_CATEGORY_CACHE_METADATA_ADDRESS_TABLES,
          DS4_RUNTIME_DOMAIN_HOST, metadata_bytes },
        { 9u, DS4_RUNTIME_CATEGORY_KV_STATE,
          DS4_RUNTIME_DOMAIN_CUDA_DEVICE, kv_bytes },
        { 10u, DS4_RUNTIME_CATEGORY_GRAPH_SCRATCH,
          DS4_RUNTIME_DOMAIN_CUDA_DEVICE, graph_bytes },
        { 11u, DS4_RUNTIME_CATEGORY_PINNED_STAGING,
          DS4_RUNTIME_DOMAIN_HOST, spec->host_capacity_bytes },
        { 15u, DS4_RUNTIME_CATEGORY_OTHER_HOST,
          DS4_RUNTIME_DOMAIN_HOST, host_split[0] },
        { 16u, DS4_RUNTIME_CATEGORY_OTHER_HOST,
          DS4_RUNTIME_DOMAIN_HOST, host_split[1] },
        { 17u, DS4_RUNTIME_CATEGORY_OTHER_HOST,
          DS4_RUNTIME_DOMAIN_HOST, host_split[2] },
        { 18u, DS4_RUNTIME_CATEGORY_OTHER_HOST,
          DS4_RUNTIME_DOMAIN_HOST, host_split[3] },
        { 19u, DS4_RUNTIME_CATEGORY_OTHER_HOST,
          DS4_RUNTIME_DOMAIN_HOST, host_split[4] },
        { 20u, DS4_RUNTIME_CATEGORY_OTHER_HOST,
          DS4_RUNTIME_DOMAIN_HOST, host_split[5] },
        { 21u, DS4_RUNTIME_CATEGORY_OTHER_HOST,
          DS4_RUNTIME_DOMAIN_HOST, host_split[6] },
        { 22u, DS4_RUNTIME_CATEGORY_OTHER_CUDA,
          DS4_RUNTIME_DOMAIN_CUDA_DEVICE, spec->cuda_capacity_bytes },
        { RESIDENT_CALLSITE_MANAGED, DS4_RUNTIME_CATEGORY_OTHER_CUDA,
          DS4_RUNTIME_DOMAIN_CUDA_MANAGED, spec->cuda_capacity_bytes },
    };
    uint64_t category_from_callsites[DS4_RUNTIME_OWNED_CATEGORY_COUNT] = {0};
    CHECK(plan->callsite_count == RESIDENT_CALLSITE_COUNT,
          "resident plan exposes exactly fourteen allocation callsites");
    for (size_t i = 0; i < plan->callsite_count &&
                        i < RESIDENT_CALLSITE_COUNT; i++) {
        const ds4_runtime_callsite *site = &plan->callsites[i];
        CHECK(site->id == expected_callsites[i].id,
              "resident callsite preserves the stable streamed identifier");
        CHECK(site->name != NULL && site->name[0] != '\0',
              "resident callsite has a stable non-empty name");
        CHECK(site->category == expected_callsites[i].category &&
                  site->domain == expected_callsites[i].domain,
              "resident callsite category and physical domain are coherent");
        CHECK(site->bound_bytes == expected_callsites[i].bound,
              "resident callsite bound matches its category reservation");
        if (site->category >= 0 &&
            site->category < DS4_RUNTIME_OWNED_CATEGORY_COUNT) {
            category_from_callsites[site->category] += site->bound_bytes;
        }
        for (size_t j = 0; j < i; j++) {
            CHECK(site->id != plan->callsites[j].id,
                  "resident callsite identifiers are unique");
        }
    }
    for (size_t i = 0; i < DS4_RUNTIME_OWNED_CATEGORY_COUNT; i++) {
        CHECK(category_from_callsites[i] == plan->category_bounds[i],
              "resident callsites reconcile exactly to each category");
    }
}

static void test_valid_plan(void) {
    const ds4_laguna_ledger ledger = canonical_ledger();
    ds4_laguna_resident_plan_spec spec = canonical_spec();
    const uint64_t valid_pages[] = { 512u, 4096u, 65536u };

    for (size_t i = 0; i < sizeof(valid_pages) / sizeof(valid_pages[0]); i++) {
        spec.source_page_size = valid_pages[i];
        ds4_laguna_resident_plan plan;
        char error[256];
        memset(&plan, 0, sizeof(plan));
        memset(error, 0xa5, sizeof(error));
        const bool ok = ds4_laguna_resident_plan_make(
            &plan, &ledger, &spec, error, sizeof(error));
        CHECK(ok, "canonical resident plan accepts every inclusive page bound");
        if (ok) check_plan_contract(&ledger, &spec, &plan);
        CHECK(ok && error[0] == '\0',
              "successful resident plan leaves no diagnostic");
    }
}

static void test_rejections(void) {
    ds4_laguna_ledger ledger = canonical_ledger();
    ds4_laguna_resident_plan_spec spec = canonical_spec();

    ledger.file_size++;
    expect_rejected(&ledger, &spec, "wrong canonical file size is rejected");
    ledger = canonical_ledger();
    ledger.tensor_count--;
    expect_rejected(&ledger, &spec, "wrong canonical tensor count is rejected");
    ledger = canonical_ledger();
    ledger.static_parent_count--;
    expect_rejected(&ledger, &spec, "wrong canonical static parent count is rejected");
    ledger = canonical_ledger();
    ledger.routed_parent_count--;
    expect_rejected(&ledger, &spec, "wrong canonical routed parent count is rejected");
    ledger = canonical_ledger();
    ledger.static_aligned_device_bytes++;
    expect_rejected(&ledger, &spec, "wrong canonical static bytes are rejected");
    ledger = canonical_ledger();
    ledger.expert_entry_count--;
    expect_rejected(&ledger, &spec, "wrong canonical expert count is rejected");
    ledger = canonical_ledger();
    ledger.slot_stride_bytes++;
    expect_rejected(&ledger, &spec, "wrong canonical slot stride is rejected");

    /* Geometry/capacity failures must not pass via an already-invalid ledger. */
    ledger = canonical_ledger();
    spec = canonical_spec();
    spec.context_tokens = 32767u;
    expect_rejected(&ledger, &spec, "non-canonical context is rejected");
    spec = canonical_spec();
    spec.prefill_rows = 4095u;
    expect_rejected(&ledger, &spec, "non-canonical prefill rows are rejected");
    spec = canonical_spec();
    spec.session_count = 2u;
    expect_rejected(&ledger, &spec, "non-canonical session count is rejected");
    spec = canonical_spec();
    spec.host_capacity_bytes = 0;
    expect_rejected(&ledger, &spec, "zero host capacity is rejected");
    spec = canonical_spec();
    spec.cuda_capacity_bytes = 0;
    expect_rejected(&ledger, &spec, "zero CUDA capacity is rejected");

    spec = canonical_spec();
    spec.source_page_size = 0;
    expect_rejected(&ledger, &spec, "zero source page size is rejected");
    spec = canonical_spec();
    spec.source_page_size = 1000u;
    expect_rejected(&ledger, &spec, "non-power-of-two source page size is rejected");
    spec = canonical_spec();
    spec.source_page_size = 256u;
    expect_rejected(&ledger, &spec, "too-small source page size is rejected");
    spec = canonical_spec();
    spec.source_page_size = 131072u;
    expect_rejected(&ledger, &spec, "too-large source page size is rejected");

    /* The doubled other-CUDA envelope must be checked before it wraps. */
    spec = canonical_spec();
    spec.cuda_capacity_bytes = UINT64_MAX;
    expect_rejected(&ledger, &spec, "CUDA capacity arithmetic overflow is rejected");
    spec = canonical_spec();
    spec.host_capacity_bytes = UINT64_MAX;
    expect_rejected(&ledger, &spec, "host capacity arithmetic overflow is rejected");
}

static void test_null_arguments(void) {
    const ds4_laguna_ledger ledger = canonical_ledger();
    const ds4_laguna_resident_plan_spec spec = canonical_spec();
    ds4_laguna_resident_plan plan;
    char error[256];

    memset(error, 0, sizeof(error));
    CHECK(!ds4_laguna_resident_plan_make(
              NULL, &ledger, &spec, error, sizeof(error)),
          "null resident plan output fails cleanly");

    memset(&plan, 0xa5, sizeof(plan));
    CHECK(!ds4_laguna_resident_plan_make(
              &plan, NULL, &spec, error, sizeof(error)),
          "null ledger fails cleanly");
    CHECK(all_zero(&plan, sizeof(plan)), "null ledger clears plan output");

    memset(&plan, 0xa5, sizeof(plan));
    CHECK(!ds4_laguna_resident_plan_make(
              &plan, &ledger, NULL, error, sizeof(error)),
          "null resident spec fails cleanly");
    CHECK(all_zero(&plan, sizeof(plan)), "null spec clears plan output");
}

typedef struct {
    ds4_runtime_allocation_record records[32];
    ds4_runtime_tracker_config config;
    ds4_runtime_tracker tracker;
} tracker_fixture;

static bool tracker_fixture_init(tracker_fixture *fixture,
                                  const ds4_laguna_resident_plan *plan) {
    memset(fixture, 0, sizeof(*fixture));
    fixture->config.callsites = plan->callsites;
    fixture->config.callsite_count = plan->callsite_count;
    fixture->config.records = fixture->records;
    fixture->config.record_capacity =
        sizeof(fixture->records) / sizeof(fixture->records[0]);
    memcpy(fixture->config.category_bounds, plan->category_bounds,
           sizeof(fixture->config.category_bounds));
    memcpy(fixture->config.report_bounds, plan->report_bounds,
           sizeof(fixture->config.report_bounds));
    fixture->config.owned_total_bound_bytes = plan->owned_total_bound_bytes;
    fixture->config.qualification_total_bound_bytes =
        plan->qualification_total_bound_bytes;
    return ds4_runtime_tracker_init(&fixture->tracker, &fixture->config) ==
           DS4_RUNTIME_STATUS_OK;
}

static void test_tracker_observation_and_accounting(void) {
    const ds4_laguna_ledger ledger = canonical_ledger();
    const ds4_laguna_resident_plan_spec spec = canonical_spec();
    ds4_laguna_resident_plan plan;
    char error[256] = {0};
    const bool plan_ok = ds4_laguna_resident_plan_make(
        &plan, &ledger, &spec, error, sizeof(error));
    CHECK(plan_ok, "resident plan is usable as a real tracker configuration");
    if (!plan_ok) return;

    tracker_fixture fixture;
    const bool tracker_ok = tracker_fixture_init(&fixture, &plan);
    CHECK(tracker_ok, "resident plan initializes a real runtime tracker");
    if (!tracker_ok) return;

    for (size_t i = 0; i < DS4_RUNTIME_OWNED_CATEGORY_COUNT; i++) {
        CHECK(fixture.tracker.category_current[i] == 0 &&
                  fixture.tracker.category_peak[i] == 0,
              "tracker starts with no observed owned allocations");
    }
    for (size_t i = 0; i < DS4_RUNTIME_REPORT_COUNT; i++) {
        CHECK(fixture.tracker.report_current[i] == 0 &&
                  fixture.tracker.report_peak[i] == 0,
              "tracker starts with no observed report bytes");
    }
    CHECK(fixture.tracker.record_count == 0 &&
              fixture.tracker.owned_total_current == 0 &&
              fixture.tracker.owned_total_peak == 0 &&
              fixture.tracker.qualification_total_current == 0 &&
              fixture.tracker.qualification_total_peak == 0,
          "plan bounds do not pretend to observe allocations or snapshots");

    const uint64_t static_id = UINT64_C(0x1001);
    const uint64_t kv_id = UINT64_C(0x1002);
    const uint64_t graph_id = UINT64_C(0x1003);
    const uint64_t pinned_id = UINT64_C(0x1004);
    const uint64_t host_id = UINT64_C(0x1005);
    const uint64_t other_cuda_id = UINT64_C(0x1006);
    const uint64_t managed_id = UINT64_C(0x1007);
    const uint64_t static_bytes = 4096u;
    const uint64_t kv_bytes = 8192u;
    const uint64_t graph_bytes = 16384u;
    const uint64_t pinned_bytes = 2048u;
    const uint64_t host_bytes = 1024u;
    const uint64_t other_cuda_bytes = 32768u;
    const uint64_t managed_bytes = 65536u;
    const uint64_t owned_live = static_bytes + kv_bytes + graph_bytes +
        pinned_bytes + host_bytes + other_cuda_bytes + managed_bytes;

    CHECK(ds4_runtime_tracker_allocate(
              &fixture.tracker, static_id, 1u,
              UINT64_C(0x100000000), static_bytes, static_bytes) ==
              DS4_RUNTIME_STATUS_OK,
          "fake static device ownership is recorded");
    CHECK(ds4_runtime_tracker_allocate(
              &fixture.tracker, kv_id, 9u,
              UINT64_C(0x200000000), kv_bytes, kv_bytes) ==
              DS4_RUNTIME_STATUS_OK,
          "fake KV device ownership is recorded");
    CHECK(ds4_runtime_tracker_allocate(
              &fixture.tracker, graph_id, 10u,
              UINT64_C(0x300000000), graph_bytes, graph_bytes) ==
              DS4_RUNTIME_STATUS_OK,
          "fake graph device ownership is recorded");
    CHECK(ds4_runtime_tracker_allocate(
              &fixture.tracker, pinned_id, 11u,
              UINT64_C(0x100000), pinned_bytes, pinned_bytes) ==
              DS4_RUNTIME_STATUS_OK,
          "fake pinned host ownership is recorded");
    CHECK(ds4_runtime_tracker_allocate(
              &fixture.tracker, host_id, 15u,
              UINT64_C(0x200000), host_bytes, host_bytes) ==
              DS4_RUNTIME_STATUS_OK,
          "fake other-host ownership is recorded");
    CHECK(ds4_runtime_tracker_allocate(
              &fixture.tracker, other_cuda_id, 22u,
              UINT64_C(0x400000000), other_cuda_bytes, other_cuda_bytes) ==
              DS4_RUNTIME_STATUS_OK,
          "fake other-CUDA device ownership is recorded");
    const uint64_t managed_base = UINT64_C(0x500000000);
    CHECK(ds4_runtime_tracker_allocate(
              &fixture.tracker, managed_id, RESIDENT_CALLSITE_MANAGED,
              managed_base, managed_bytes, managed_bytes) ==
              DS4_RUNTIME_STATUS_OK,
          "fake managed ownership is recorded");

    CHECK(fixture.tracker.category_current[
              DS4_RUNTIME_CATEGORY_STATIC_WEIGHTS] == static_bytes &&
              fixture.tracker.category_current[
                  DS4_RUNTIME_CATEGORY_KV_STATE] == kv_bytes &&
              fixture.tracker.category_current[
                  DS4_RUNTIME_CATEGORY_GRAPH_SCRATCH] == graph_bytes &&
              fixture.tracker.category_current[
                  DS4_RUNTIME_CATEGORY_PINNED_STAGING] == pinned_bytes &&
              fixture.tracker.category_current[
                  DS4_RUNTIME_CATEGORY_OTHER_HOST] == host_bytes &&
              fixture.tracker.category_current[
                  DS4_RUNTIME_CATEGORY_OTHER_CUDA] ==
                  other_cuda_bytes + managed_bytes &&
              fixture.tracker.owned_total_current == owned_live &&
              fixture.tracker.owned_total_peak == owned_live,
          "all resident owned domains charge once to their declared categories");

    const uint64_t relation_id = UINT64_C(0xa007);
    CHECK(ds4_runtime_tracker_managed_host_relation(
              &fixture.tracker, relation_id, managed_base, managed_bytes,
              managed_id) == DS4_RUNTIME_STATUS_OK,
          "managed host-visible relation is accepted for the managed owner");
    CHECK(fixture.tracker.category_current[
              DS4_RUNTIME_CATEGORY_OTHER_CUDA] == other_cuda_bytes + managed_bytes &&
              fixture.tracker.owned_total_current == owned_live,
          "managed host-visible relation does not double-charge CUDA ownership");

    const uint64_t mapping_id = UINT64_C(0xb001);
    const uint64_t registration_id = UINT64_C(0xb002);
    const uint64_t mapping_base = UINT64_C(0x600000000);
    const uint64_t mapping_bytes = 65536u;
    CHECK(ds4_runtime_tracker_map_model(
              &fixture.tracker, mapping_id, mapping_base, mapping_bytes) ==
              DS4_RUNTIME_STATUS_OK,
          "fake model mapping is recorded as report-only state");
    CHECK(fixture.tracker.report_current[
              DS4_RUNTIME_REPORT_MODEL_MAPPED_VIRTUAL] == mapping_bytes &&
              fixture.tracker.owned_total_current == owned_live &&
              fixture.tracker.qualification_total_current == owned_live,
          "model mapping does not charge owned or qualification bytes");
    CHECK(ds4_runtime_tracker_register(
              &fixture.tracker, registration_id, mapping_base, mapping_bytes,
              mapping_id) == DS4_RUNTIME_STATUS_OK,
          "fake model registration is accepted inside the mapping");
    CHECK(fixture.tracker.report_current[
              DS4_RUNTIME_REPORT_MODEL_MAPPING_REGISTERED] == mapping_bytes &&
              fixture.tracker.owned_total_current == owned_live &&
              fixture.tracker.qualification_total_current == owned_live,
          "model registration does not double-charge mapped pages");

    const uint64_t source_resident =
        page_round(ledger.file_size, spec.source_page_size);
    CHECK(ds4_runtime_tracker_checkpoint_external(
              &fixture.tracker, source_resident, spec.host_capacity_bytes,
              spec.cuda_capacity_bytes) == DS4_RUNTIME_STATUS_OK,
          "source and unattributed observations commit as one checkpoint");
    const uint64_t qualification_live = owned_live + source_resident +
        spec.host_capacity_bytes + spec.cuda_capacity_bytes;
    CHECK(fixture.tracker.qualification_total_current == qualification_live &&
              fixture.tracker.qualification_total_peak == qualification_live,
          "qualification adds only live owned and external report bytes once");

    CHECK(ds4_runtime_tracker_unregister(&fixture.tracker, relation_id) ==
              DS4_RUNTIME_STATUS_OK &&
              ds4_runtime_tracker_release(&fixture.tracker, managed_id) ==
                  DS4_RUNTIME_STATUS_OK &&
              ds4_runtime_tracker_release(&fixture.tracker, other_cuda_id) ==
                  DS4_RUNTIME_STATUS_OK &&
              ds4_runtime_tracker_release(&fixture.tracker, host_id) ==
                  DS4_RUNTIME_STATUS_OK &&
              ds4_runtime_tracker_release(&fixture.tracker, pinned_id) ==
                  DS4_RUNTIME_STATUS_OK &&
              ds4_runtime_tracker_release(&fixture.tracker, graph_id) ==
                  DS4_RUNTIME_STATUS_OK &&
              ds4_runtime_tracker_release(&fixture.tracker, kv_id) ==
                  DS4_RUNTIME_STATUS_OK &&
              ds4_runtime_tracker_release(&fixture.tracker, static_id) ==
                  DS4_RUNTIME_STATUS_OK,
          "all fake owned allocations release cleanly after their relation");
    CHECK(ds4_runtime_tracker_unregister(&fixture.tracker, registration_id) ==
              DS4_RUNTIME_STATUS_OK &&
              ds4_runtime_tracker_unmap_model(&fixture.tracker, mapping_id) ==
                  DS4_RUNTIME_STATUS_OK,
          "model registration and mapping release in dependency order");

    CHECK(fixture.tracker.owned_total_current == 0 &&
              fixture.tracker.qualification_total_current ==
                  source_resident + spec.host_capacity_bytes +
                  spec.cuda_capacity_bytes,
          "release removes owned bytes but retains the latest external sample");
    for (size_t i = 0; i < DS4_RUNTIME_OWNED_CATEGORY_COUNT; i++) {
        CHECK(fixture.tracker.category_current[i] == 0,
              "released owned category has zero current bytes");
    }
    CHECK(fixture.tracker.owned_total_peak == owned_live &&
              fixture.tracker.category_peak[DS4_RUNTIME_CATEGORY_STATIC_WEIGHTS] ==
                  static_bytes &&
              fixture.tracker.category_peak[DS4_RUNTIME_CATEGORY_KV_STATE] ==
                  kv_bytes &&
              fixture.tracker.category_peak[DS4_RUNTIME_CATEGORY_GRAPH_SCRATCH] ==
                  graph_bytes &&
              fixture.tracker.category_peak[DS4_RUNTIME_CATEGORY_PINNED_STAGING] ==
                  pinned_bytes &&
              fixture.tracker.category_peak[DS4_RUNTIME_CATEGORY_OTHER_HOST] ==
                  host_bytes &&
              fixture.tracker.category_peak[DS4_RUNTIME_CATEGORY_OTHER_CUDA] ==
                  other_cuda_bytes + managed_bytes &&
              fixture.tracker.qualification_total_peak == qualification_live,
          "simultaneous owned and qualification peaks survive release");
    CHECK(fixture.tracker.report_current[
              DS4_RUNTIME_REPORT_MODEL_MAPPED_VIRTUAL] == 0 &&
              fixture.tracker.report_current[
                  DS4_RUNTIME_REPORT_MODEL_MAPPING_REGISTERED] == 0 &&
              fixture.tracker.report_current[
                  DS4_RUNTIME_REPORT_MODEL_SOURCE_RESIDENT] == source_resident &&
              fixture.tracker.report_current[
                  DS4_RUNTIME_REPORT_HOST_LIBRARY_UNATTRIBUTED] ==
                  spec.host_capacity_bytes &&
              fixture.tracker.report_current[
                  DS4_RUNTIME_REPORT_CUDA_LIBRARY_UNATTRIBUTED] ==
                  spec.cuda_capacity_bytes,
          "mapping reports clear while external reports retain their checkpoint");
}

int main(void) {
    test_valid_plan();
    test_rejections();
    test_null_arguments();
    test_tracker_observation_and_accounting();
    if (g_failed != 0) {
        fprintf(stderr, "%d/%d resident-plan checks failed\n", g_failed, g_total);
        return 1;
    }
    printf("resident-plan: %d checks, 0 failures\n", g_total);
    return 0;
}
