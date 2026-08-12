#define _POSIX_C_SOURCE 200809L

/* Pure-C external-attribution tests for compact Laguna qualification.
 *
 * Inputs are recorded syscall/CUDA/NVML observations.  Range parsing,
 * de-duplication, reconciliation, and failure classification stay real so
 * this binary never needs a CUDA context or a model artifact. */

#include "ds4_runtime.h"

#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_assertions;
static int g_failures;

#define CHECK(condition, message) do {                                        \
    g_assertions++;                                                           \
    if (!(condition)) {                                                       \
        fprintf(stderr, "FAIL: %s (line %d)\n", (message), __LINE__);       \
        g_failures++;                                                         \
    }                                                                         \
} while (0)

enum {
    OWN_PID = 4242,
    PEER_PID = 9001,
    NVML_API_VERSION = 2,
};

static const uint64_t kib = UINT64_C(1024);
static const uint64_t mib = UINT64_C(1024) * 1024u;
static const char device_uuid[] = "GPU-01234567-89ab-cdef-0123-456789abcdef";
static const char nvml_library_version[] = "580.95.05";

/* Eight VMAs deliberately exercise exact-inode matching and physical-domain
 * exclusions.  The model mapping has 64 KiB PSS but is charged separately by
 * mincore.  The tracked host, pinned, and managed VMAs contribute 28 KiB and
 * must be excluded exactly once.  The different-device inode alias, libcuda,
 * heap, and stack leave 42 KiB of unattributed host PSS. */
static const char valid_smaps[] =
    "10000000-20000000 r--s 00000000 08:11 777 /models/laguna.gguf\n"
    "Size:             262144 kB\n"
    "Pss:                  64 kB\n"
    "20000000-20002000 rw-p 00000000 00:00 0 [anon:host]\n"
    "Size:                  8 kB\n"
    "Pss:                   8 kB\n"
    "21000000-21001000 r--s 00000000 08:12 777 /models/same-inode-other-device\n"
    "Size:                  4 kB\n"
    "Pss:                   4 kB\n"
    "30000000-30004000 rw-s 00000000 00:01 99 /dev/nvidia-uvm\n"
    "Size:                 16 kB\n"
    "Pss:                  16 kB\n"
    "40000000-40400000 rw-s 00000000 00:01 100 /dev/nvidia-uvm\n"
    "Size:               4096 kB\n"
    "Pss:                   4 kB\n"
    "50000000-50008000 r-xp 00000000 08:01 12 /usr/lib/libcuda.so.1\n"
    "Size:                 32 kB\n"
    "Pss:                  24 kB\n"
    "60000000-60002000 rw-p 00000000 00:00 0 [heap]\n"
    "Size:                  8 kB\n"
    "Pss:                   8 kB\n"
    "70000000-70002000 rw-p 00000000 00:00 0 [stack]\n"
    "Size:                  8 kB\n"
    "Pss:                   6 kB\n";

typedef struct {
    ds4_runtime_callsite callsites[4];
    ds4_runtime_allocation_record records[12];
    ds4_runtime_tracker_config config;
    ds4_runtime_tracker tracker;
    uint8_t build_identity[DS4_RUNTIME_BUILD_IDENTITY_BYTES];
    ds4_runtime_nvml_process_sample pre_processes[2];
    ds4_runtime_nvml_process_sample before_processes[2];
    ds4_runtime_nvml_process_sample inside_processes[2];
    ds4_runtime_nvml_process_sample after_processes[2];
    ds4_runtime_nvml_inventory pre_inventory;
    ds4_runtime_nvml_inventory before_inventory;
    ds4_runtime_nvml_inventory inside_inventory;
    ds4_runtime_nvml_inventory after_inventory;
    ds4_runtime_external_checkpoint_input input;
} external_fixture;

static void fixture_prepare(external_fixture *fixture) {
    memset(fixture, 0, sizeof(*fixture));
    const ds4_runtime_callsite sites[] = {
        { 1u, "host", DS4_RUNTIME_CATEGORY_OTHER_HOST,
          DS4_RUNTIME_DOMAIN_HOST, 8u * kib },
        { 2u, "pinned", DS4_RUNTIME_CATEGORY_PINNED_STAGING,
          DS4_RUNTIME_DOMAIN_HOST, 16u * kib },
        { 3u, "managed", DS4_RUNTIME_CATEGORY_OTHER_CUDA,
          DS4_RUNTIME_DOMAIN_CUDA_MANAGED, 4u * mib },
        { 4u, "device", DS4_RUNTIME_CATEGORY_STATIC_WEIGHTS,
          DS4_RUNTIME_DOMAIN_CUDA_DEVICE, 64u * mib },
    };
    memcpy(fixture->callsites, sites, sizeof(sites));
    fixture->config.callsites = fixture->callsites;
    fixture->config.callsite_count =
        sizeof(fixture->callsites) / sizeof(fixture->callsites[0]);
    fixture->config.records = fixture->records;
    fixture->config.record_capacity =
        sizeof(fixture->records) / sizeof(fixture->records[0]);
    fixture->config.category_bounds[DS4_RUNTIME_CATEGORY_OTHER_HOST] = 8u * kib;
    fixture->config.category_bounds[DS4_RUNTIME_CATEGORY_PINNED_STAGING] =
        16u * kib;
    fixture->config.category_bounds[DS4_RUNTIME_CATEGORY_OTHER_CUDA] = 4u * mib;
    fixture->config.category_bounds[DS4_RUNTIME_CATEGORY_STATIC_WEIGHTS] =
        64u * mib;
    fixture->config.report_bounds[DS4_RUNTIME_REPORT_MODEL_SOURCE_RESIDENT] =
        2u * 1024u * mib;
    fixture->config.report_bounds[DS4_RUNTIME_REPORT_MODEL_MAPPED_VIRTUAL] =
        256u * mib;
    fixture->config.report_bounds[
        DS4_RUNTIME_REPORT_MODEL_MAPPING_REGISTERED] = 4u * kib;
    fixture->config.report_bounds[
        DS4_RUNTIME_REPORT_HOST_LIBRARY_UNATTRIBUTED] = 512u * mib;
    fixture->config.report_bounds[
        DS4_RUNTIME_REPORT_CUDA_LIBRARY_UNATTRIBUTED] = 512u * mib;
    fixture->config.owned_total_bound_bytes = 68u * mib + 24u * kib;
    fixture->config.qualification_total_bound_bytes =
        fixture->config.owned_total_bound_bytes + 3u * 1024u * mib;

    for (size_t i = 0; i < sizeof(fixture->build_identity); i++) {
        fixture->build_identity[i] = (uint8_t)(0xa0u + i);
    }

    fixture->pre_processes[0] = (ds4_runtime_nvml_process_sample){
        PEER_PID, 80u * mib, true,
    };
    fixture->pre_processes[1] = (ds4_runtime_nvml_process_sample){
        OWN_PID, 48u * mib, true,
    };
    fixture->before_processes[0] = (ds4_runtime_nvml_process_sample){
        OWN_PID, 7u * mib, true,
    };
    fixture->before_processes[1] = fixture->pre_processes[0];
    fixture->inside_processes[0] = (ds4_runtime_nvml_process_sample){
        OWN_PID, 116u * mib, true,
    };
    fixture->inside_processes[1] = fixture->pre_processes[0];
    fixture->after_processes[0] = fixture->pre_processes[0];
    fixture->after_processes[1] = (ds4_runtime_nvml_process_sample){
        OWN_PID, 511u * mib, true,
    };

    fixture->pre_inventory = (ds4_runtime_nvml_inventory){
        .api_version = NVML_API_VERSION,
        .library_version = nvml_library_version,
        .device_uuid = device_uuid,
        .processes = fixture->pre_processes,
        .process_count = 1u,
    };
    fixture->before_inventory = (ds4_runtime_nvml_inventory){
        .api_version = NVML_API_VERSION,
        .library_version = nvml_library_version,
        .device_uuid = device_uuid,
        .processes = fixture->before_processes,
        .process_count = 2u,
    };
    fixture->inside_inventory = (ds4_runtime_nvml_inventory){
        .api_version = NVML_API_VERSION,
        .library_version = nvml_library_version,
        .device_uuid = device_uuid,
        .processes = fixture->inside_processes,
        .process_count = 2u,
    };
    fixture->after_inventory = (ds4_runtime_nvml_inventory){
        .api_version = NVML_API_VERSION,
        .library_version = nvml_library_version,
        .device_uuid = device_uuid,
        .processes = fixture->after_processes,
        .process_count = 2u,
    };
    fixture->input = (ds4_runtime_external_checkpoint_input){
        .smaps_text = valid_smaps,
        .smaps_text_bytes = sizeof(valid_smaps) - 1u,
        .model_device_major = 0x08u,
        .model_device_minor = 0x11u,
        .model_inode = 777u,
        .model_map_base = UINT64_C(0x10000000),
        .model_map_bytes = 256u * mib,
        .model_file_offset = 0u,
        .attribution_records = fixture->records,
        .attribution_record_count =
            sizeof(fixture->records) / sizeof(fixture->records[0]),
        .expected_nvml_api_version = NVML_API_VERSION,
        .expected_nvml_library_version = nvml_library_version,
        .expected_device_uuid = device_uuid,
        .own_pid = OWN_PID,
        .expected_build_identity = fixture->build_identity,
        .observed_build_identity = fixture->build_identity,
        .build_identity_bytes = sizeof(fixture->build_identity),
        .baseline_nvml_api_version = NVML_API_VERSION,
        .baseline_nvml_library_version = nvml_library_version,
        .baseline_device_uuid = device_uuid,
        .baseline_process_id = OWN_PID,
        .baseline_nvml_process_present = true,
        .baseline_nvml_process_bytes_known = true,
        .baseline_nvml_process_bytes = 48u * mib,
        .baseline_tracked_cuda_physical_bytes = 0u,
        .pre_child_inventory = &fixture->pre_inventory,
        .checkpoint_before_inventory = &fixture->before_inventory,
        .inside_ds4_inventory = &fixture->inside_inventory,
        .checkpoint_after_inventory = &fixture->after_inventory,
        .cuda_mem_info_known = true,
        .cuda_mem_free_bytes = 60u * 1024u * mib,
        .cuda_mem_total_bytes = 128u * 1024u * mib,
        .model_source_page_size = 4096u,
        .model_source_resident_bytes = 256u * mib,
        .model_source_mapped_page_bytes = 256u * mib,
    };
}

static bool fixture_init_host_request(
        external_fixture *fixture, uint64_t host_requested_bytes) {
    fixture_prepare(fixture);
    if (ds4_runtime_tracker_init(&fixture->tracker, &fixture->config) !=
            DS4_RUNTIME_STATUS_OK) {
        return false;
    }
    /* Host and CUDA-visible VMAs are intentionally non-overlapping. */
    return ds4_runtime_tracker_allocate(
               &fixture->tracker, 1u, 1u, UINT64_C(0x20000000),
               host_requested_bytes, 8u * kib) == DS4_RUNTIME_STATUS_OK &&
           ds4_runtime_tracker_allocate(
               &fixture->tracker, 2u, 2u, UINT64_C(0x30000000),
               16u * kib, 16u * kib) == DS4_RUNTIME_STATUS_OK &&
           ds4_runtime_tracker_allocate(
               &fixture->tracker, 3u, 3u, UINT64_C(0x40000000),
               4u * mib, 4u * mib) == DS4_RUNTIME_STATUS_OK &&
           ds4_runtime_tracker_allocate(
               &fixture->tracker, 4u, 4u, UINT64_C(0x80000000),
               64u * mib, 64u * mib) == DS4_RUNTIME_STATUS_OK &&
           ds4_runtime_tracker_managed_host_relation(
               &fixture->tracker, 5u, UINT64_C(0x40000000),
               4u * mib, 3u) == DS4_RUNTIME_STATUS_OK &&
           ds4_runtime_tracker_map_model(
               &fixture->tracker, 6u, UINT64_C(0x10000000),
               256u * mib) == DS4_RUNTIME_STATUS_OK;
}

static bool fixture_init(external_fixture *fixture) {
    return fixture_init_host_request(fixture, 8u * kib);
}

static ds4_runtime_external_sample run_checkpoint(external_fixture *fixture) {
    ds4_runtime_external_sample sample;
    memset(&sample, 0, sizeof(sample));
    const ds4_runtime_status result = ds4_runtime_tracker_checkpoint_attributed(
        &fixture->tracker, &fixture->input, &sample);
    if (result != DS4_RUNTIME_STATUS_OK) {
        fprintf(stderr,
                "checkpoint failed unexpectedly: failure=%d violation=%d\n",
                (int)sample.failure, (int)fixture->tracker.violation);
    }
    CHECK(result == DS4_RUNTIME_STATUS_OK,
          "valid external attribution checkpoint reconciles");
    return sample;
}

static void test_valid_external_attribution(void) {
    external_fixture fixture;
    CHECK(fixture_init(&fixture), "external attribution fixture initializes");
    if (fixture.tracker.violation != DS4_RUNTIME_VIOLATION_NONE) return;

    const ds4_runtime_external_sample sample = run_checkpoint(&fixture);
    CHECK(sample.failure == DS4_RUNTIME_EXTERNAL_FAILURE_NONE &&
              sample.attributed_valid &&
              sample.attributed_generation != 0u &&
              sample.smaps_vma_count == 8u &&
              sample.smaps_total_pss_bytes == 134u * kib &&
              sample.smaps_model_vma_count == 1u &&
              sample.smaps_model_pss_bytes == 64u * kib &&
              sample.smaps_tracked_vma_count == 3u &&
              sample.smaps_tracked_pss_bytes == 28u * kib &&
              sample.host_library_unattributed_bytes == 42u * kib,
          "smaps attribution excludes model and every tracked VMA exactly once");
    CHECK(sample.smaps_model_device_major == 0x08u &&
              sample.smaps_model_device_minor == 0x11u &&
              sample.smaps_model_inode == 777u,
          "smaps sample binds the exact opened model device and inode");
    CHECK(sample.nvml_api_version == NVML_API_VERSION &&
              strcmp(sample.nvml_library_version,
                     nvml_library_version) == 0 &&
              strcmp(sample.device_uuid, device_uuid) == 0 &&
              sample.process_id == OWN_PID &&
              sample.nvml_process_baseline_present &&
              sample.nvml_process_baseline_bytes == 48u * mib &&
              sample.tracked_cuda_physical_baseline_bytes == 0u &&
              sample.nvml_process_bytes == 116u * mib &&
              sample.nvml_process_bytes !=
                  fixture.before_processes[0].used_bytes &&
              sample.nvml_process_bytes !=
                  fixture.after_processes[1].used_bytes &&
              sample.tracked_cuda_physical_bytes == 68u * mib &&
              sample.cuda_library_unattributed_bytes == 48u * mib,
          "inside-DS4 NVML usage alone subtracts current tracked CUDA physical bytes");
    CHECK(sample.cuda_library_unattributed_bytes ==
              fixture.input.baseline_nvml_process_bytes,
          "pre-existing CUDA context and library bytes remain charged");
    CHECK(sample.cuda_mem_free_bytes == 60u * 1024u * mib &&
              sample.cuda_mem_total_bytes == 128u * 1024u * mib &&
              sample.unrelated_process_inventory_stable,
          "device-wide memory remains only a cross-check and peer inventory is stable");
    CHECK(fixture.tracker.report_current[
              DS4_RUNTIME_REPORT_MODEL_SOURCE_RESIDENT] == 256u * mib &&
              fixture.tracker.report_current[
                  DS4_RUNTIME_REPORT_HOST_LIBRARY_UNATTRIBUTED] == 42u * kib &&
              fixture.tracker.report_current[
                  DS4_RUNTIME_REPORT_CUDA_LIBRARY_UNATTRIBUTED] == 48u * mib,
          "reconciled external categories commit to the runtime tracker");
    CHECK(fixture.tracker.qualification_total_current ==
              68u * mib + 24u * kib + 256u * mib + 48u * mib + 42u * kib &&
              fixture.tracker.qualification_total_peak ==
                  fixture.tracker.qualification_total_current,
          "qualification total and peak use one simultaneous de-duplicated sample");

    ds4_runtime_snapshot snapshot;
    ds4_runtime_allocation_record active[6];
    CHECK(ds4_runtime_tracker_snapshot_copy(
              &fixture.tracker, &snapshot, active,
              sizeof(active) / sizeof(active[0])) &&
              memcmp(&snapshot.external_sample, &sample, sizeof(sample)) == 0,
          "immutable runtime snapshot exposes the raw and reconciled external sample");
}

static void check_failure(
    external_fixture *fixture,
    ds4_runtime_external_failure expected,
    const char *message);

static void test_partial_vma_credit_and_managed_deduplication(void) {
    static const char partial_smaps[] =
        "10000000-20000000 r--s 00000000 08:11 777 /models/laguna.gguf\n"
        "Pss: 4 kB\n"
        "20000000-20004000 rw-p 00000000 00:00 0 [heap]\n"
        "Pss: 12 kB\n"
        "40000000-40400000 rw-s 00000000 00:01 100 /dev/nvidia-uvm\n"
        "Pss: 4 kB\n";

    external_fixture fixture;
    CHECK(fixture_init(&fixture), "partial VMA fixture initializes");
    fixture.input.smaps_text = partial_smaps;
    fixture.input.smaps_text_bytes = sizeof(partial_smaps) - 1u;
    fixture.input.model_source_resident_bytes = 4096u;
    fixture.records[0].base = UINT64_C(0x20001000);
    fixture.records[0].requested_bytes = 4u * kib;
    fixture.records[0].charged_bytes = 4u * kib;
    fixture.records[1].live = false;
    const ds4_runtime_external_sample sample = run_checkpoint(&fixture);
    CHECK(sample.smaps_total_pss_bytes == 20u * kib &&
              sample.smaps_model_pss_bytes == 4u * kib &&
              sample.smaps_tracked_pss_bytes == 8u * kib &&
              sample.host_library_unattributed_bytes == 8u * kib,
          "partial tracked VMA receives conservative min(PSS, overlap) credit");
    CHECK(sample.smaps_tracked_vma_count == 2u &&
              sample.tracked_cuda_physical_bytes == 68u * mib &&
              sample.cuda_library_unattributed_bytes == 48u * mib,
          "managed VMA credit is host-side only while its physical CUDA charge is subtracted once");
}

static void test_host_charge_and_registration_deduplication(void) {
    external_fixture fixture;
    CHECK(fixture_init_host_request(&fixture, 4u * kib),
          "aligned host-overcharge fixture initializes through the tracker");
    const ds4_runtime_external_sample charged = run_checkpoint(&fixture);
    CHECK(charged.smaps_tracked_pss_bytes == 28u * kib &&
              charged.host_library_unattributed_bytes == 42u * kib,
          "owned host VMA coverage uses charged rather than requested bytes");

    CHECK(fixture_init(&fixture),
          "real registration de-duplication fixture initializes");
    CHECK(ds4_runtime_tracker_register(
              &fixture.tracker, 7u, UINT64_C(0x20001000),
              4u * kib, 1u) == DS4_RUNTIME_STATUS_OK,
          "real tracker registration creates an alias relation");
    const ds4_runtime_external_sample registered = run_checkpoint(&fixture);
    CHECK(registered.smaps_tracked_vma_count == 3u &&
              registered.smaps_tracked_pss_bytes == 28u * kib &&
              registered.host_library_unattributed_bytes == 42u * kib,
          "registration alias bytes are validated but not charged twice");
}

static void test_model_source_tail_page_semantics(void) {
    external_fixture fixture;
    CHECK(fixture_init(&fixture), "source tail-page fixture initializes");
    /* A 3,424-byte tail still occupies one complete mapped/resident page.
     * smaps excludes the model VMA wholesale, so the mincore charge must not
     * clamp the last resident page back to file size. */
    char tail_smaps[sizeof(valid_smaps)];
    memcpy(tail_smaps, valid_smaps, sizeof(tail_smaps));
    char *tail_end = strstr(tail_smaps, "10000000-20000000");
    char *tail_pss = strstr(tail_smaps, "64 kB");
    CHECK(tail_end != NULL, "tail-page fixture locates the model VMA header");
    CHECK(tail_pss != NULL, "tail-page fixture locates the model PSS field");
    if (!tail_end || !tail_pss) return;
    memcpy(tail_end + 9u, "10001000", 8u);
    memcpy(tail_pss, "04", 2u);
    CHECK(ds4_runtime_tracker_unmap_model(&fixture.tracker, 6u) ==
              DS4_RUNTIME_STATUS_OK &&
              ds4_runtime_tracker_map_model(
                  &fixture.tracker, 7u, UINT64_C(0x10000000), 3424u) ==
                  DS4_RUNTIME_STATUS_OK,
          "tail-page fixture tracks the exact unaligned model mapping");
    fixture.input.smaps_text = tail_smaps;
    fixture.input.smaps_text_bytes = sizeof(tail_smaps) - 1u;
    fixture.input.model_map_bytes = 3424u;
    fixture.input.model_source_resident_bytes = 4096u;
    fixture.input.model_source_mapped_page_bytes = 4096u;
    const ds4_runtime_external_sample sample = run_checkpoint(&fixture);
    CHECK(sample.failure == DS4_RUNTIME_EXTERNAL_FAILURE_NONE &&
              fixture.tracker.report_current[
                  DS4_RUNTIME_REPORT_MODEL_SOURCE_RESIDENT] == 4096u,
          "model source residency counts a full mapped tail page");

    CHECK(fixture_init(&fixture), "invalid source page count fixture initializes");
    memcpy(tail_smaps, valid_smaps, sizeof(tail_smaps));
    tail_end = strstr(tail_smaps, "10000000-20000000");
    tail_pss = strstr(tail_smaps, "64 kB");
    CHECK(tail_end != NULL,
          "invalid tail-page fixture locates the model VMA header");
    CHECK(tail_pss != NULL,
          "invalid tail-page fixture locates the model PSS field");
    if (!tail_end || !tail_pss) return;
    memcpy(tail_end + 9u, "10001000", 8u);
    memcpy(tail_pss, "04", 2u);
    CHECK(ds4_runtime_tracker_unmap_model(&fixture.tracker, 6u) ==
              DS4_RUNTIME_STATUS_OK &&
              ds4_runtime_tracker_map_model(
                  &fixture.tracker, 7u, UINT64_C(0x10000000), 3424u) ==
                  DS4_RUNTIME_STATUS_OK,
          "invalid tail-page fixture tracks the unaligned model mapping");
    fixture.input.smaps_text = tail_smaps;
    fixture.input.smaps_text_bytes = sizeof(tail_smaps) - 1u;
    fixture.input.model_map_bytes = 3424u;
    fixture.input.model_source_resident_bytes = 4096u;
    fixture.input.model_source_mapped_page_bytes = 3424u;
    check_failure(&fixture, DS4_RUNTIME_EXTERNAL_FAILURE_INVALID_INPUT,
                  "model source residency cannot exceed mapped page bytes");
}

static void check_failure(
        external_fixture *fixture,
        ds4_runtime_external_failure expected,
        const char *message) {
    const uint64_t before_sequence = fixture->tracker.event_sequence;
    const uint64_t before_total = fixture->tracker.qualification_total_current;
    ds4_runtime_external_sample sample;
    memset(&sample, 0, sizeof(sample));
    CHECK(ds4_runtime_tracker_checkpoint_attributed(
              &fixture->tracker, &fixture->input, &sample) ==
              DS4_RUNTIME_STATUS_UNSAFE &&
              sample.failure == expected &&
              fixture->tracker.violation ==
                  DS4_RUNTIME_VIOLATION_EXTERNAL_ATTRIBUTION &&
              fixture->tracker.event_sequence == before_sequence &&
              fixture->tracker.qualification_total_current == before_total,
          message);
}

static void test_exact_model_mapping_binding(void) {
    external_fixture fixture;
    char wrong_offset[sizeof(valid_smaps)];
    CHECK(fixture_init(&fixture), "wrong model-offset fixture initializes");
    memcpy(wrong_offset, valid_smaps, sizeof(wrong_offset));
    char *model_header = strstr(
        wrong_offset,
        "10000000-20000000 r--s 00000000 08:11 777");
    CHECK(model_header != NULL,
          "wrong model-offset fixture locates the model VMA header");
    if (!model_header) return;
    memcpy(model_header + 23u, "00001000", 8u);
    fixture.input.smaps_text = wrong_offset;
    fixture.input.smaps_text_bytes = sizeof(wrong_offset) - 1u;
    check_failure(&fixture,
                  DS4_RUNTIME_EXTERNAL_FAILURE_MODEL_MAPPING_MISMATCH,
                  "model VMA file offset must match the opened mapping offset");

    CHECK(fixture_init(&fixture), "model mapping-gap fixture initializes");
    char mapping_gap[sizeof(valid_smaps)];
    memcpy(mapping_gap, valid_smaps, sizeof(mapping_gap));
    model_header = strstr(mapping_gap, "10000000-20000000");
    CHECK(model_header != NULL,
          "mapping-gap fixture locates the model VMA header");
    if (!model_header) return;
    memcpy(model_header + 9u, "18000000", 8u);
    fixture.input.smaps_text = mapping_gap;
    fixture.input.smaps_text_bytes = sizeof(mapping_gap) - 1u;
    check_failure(&fixture,
                  DS4_RUNTIME_EXTERNAL_FAILURE_MODEL_MAPPING_MISMATCH,
                  "same-inode model VMAs must cover the complete mapped address range");

    static const char alias_suffix[] =
        "80000000-80001000 r--s 00000000 08:11 777 /models/alias.gguf\n"
        "Pss: 4 kB\n";
    char unexpected_alias[sizeof(valid_smaps) + sizeof(alias_suffix)];
    CHECK(fixture_init(&fixture), "unexpected model-alias fixture initializes");
    memcpy(unexpected_alias, valid_smaps, sizeof(valid_smaps) - 1u);
    memcpy(unexpected_alias + sizeof(valid_smaps) - 1u,
           alias_suffix, sizeof(alias_suffix));
    fixture.input.smaps_text = unexpected_alias;
    fixture.input.smaps_text_bytes =
        sizeof(valid_smaps) - 1u + sizeof(alias_suffix) - 1u;
    check_failure(&fixture,
                  DS4_RUNTIME_EXTERNAL_FAILURE_MODEL_MAPPING_MISMATCH,
                  "unexpected same-device/inode aliases invalidate attribution");

    CHECK(fixture_init(&fixture), "missing model tracker record fixture initializes");
    CHECK(ds4_runtime_tracker_unmap_model(&fixture.tracker, 6u) ==
              DS4_RUNTIME_STATUS_OK,
          "missing model tracker fixture removes the real mapping record");
    check_failure(&fixture,
                  DS4_RUNTIME_EXTERNAL_FAILURE_MODEL_MAPPING_MISMATCH,
                  "smaps identity cannot substitute for an exact model mapping record");

    CHECK(fixture_init(&fixture), "model PSS bound fixture initializes");
    fixture.input.model_source_resident_bytes = 32u * kib;
    check_failure(&fixture,
                  DS4_RUNTIME_EXTERNAL_FAILURE_MODEL_MAPPING_MISMATCH,
                  "model smaps PSS cannot exceed exact mincore resident bytes");
}

static void test_failed_checkpoint_retains_last_good_sample(void) {
    external_fixture fixture;
    CHECK(fixture_init(&fixture), "sample-retention fixture initializes");
    const ds4_runtime_external_sample committed = run_checkpoint(&fixture);
    const uint64_t committed_event_sequence = fixture.tracker.event_sequence;
    const uint64_t committed_total =
        fixture.tracker.qualification_total_current;
    const uint64_t committed_peak =
        fixture.tracker.qualification_total_peak;
    uint64_t committed_report_current[DS4_RUNTIME_REPORT_COUNT];
    uint64_t committed_report_peak[DS4_RUNTIME_REPORT_COUNT];
    memcpy(committed_report_current, fixture.tracker.report_current,
           sizeof(committed_report_current));
    memcpy(committed_report_peak, fixture.tracker.report_peak,
           sizeof(committed_report_peak));

    char wrong_offset[sizeof(valid_smaps)];
    memcpy(wrong_offset, valid_smaps, sizeof(wrong_offset));
    char *model_header = strstr(
        wrong_offset,
        "10000000-20000000 r--s 00000000 08:11 777");
    CHECK(model_header != NULL,
          "sample-retention fixture locates the model VMA header");
    if (!model_header) return;
    memcpy(model_header + 23u, "00001000", 8u);
    fixture.input.smaps_text = wrong_offset;
    fixture.input.smaps_text_bytes = sizeof(wrong_offset) - 1u;

    ds4_runtime_external_sample failure;
    memset(&failure, 0, sizeof(failure));
    CHECK(ds4_runtime_tracker_checkpoint_attributed(
              &fixture.tracker, &fixture.input, &failure) ==
              DS4_RUNTIME_STATUS_UNSAFE &&
              failure.failure ==
                  DS4_RUNTIME_EXTERNAL_FAILURE_MODEL_MAPPING_MISMATCH,
          "failure output reports the rejected checkpoint cause");
    CHECK(memcmp(&fixture.tracker.external_sample,
                 &committed, sizeof(committed)) == 0 &&
              fixture.tracker.event_sequence == committed_event_sequence &&
              fixture.tracker.qualification_total_current == committed_total &&
              fixture.tracker.qualification_total_peak == committed_peak &&
              memcmp(fixture.tracker.report_current,
                     committed_report_current,
                     sizeof(committed_report_current)) == 0 &&
              memcmp(fixture.tracker.report_peak,
                     committed_report_peak,
                     sizeof(committed_report_peak)) == 0,
          "failed checkpoint retains the last committed sample, reports, and peaks");

    CHECK(fixture_init(&fixture), "aliased failure-output fixture initializes");
    const ds4_runtime_external_sample aliased_committed =
        run_checkpoint(&fixture);
    const uint64_t aliased_event_sequence = fixture.tracker.event_sequence;
    const uint64_t aliased_total =
        fixture.tracker.qualification_total_current;
    const uint64_t aliased_peak =
        fixture.tracker.qualification_total_peak;
    uint64_t aliased_report_current[DS4_RUNTIME_REPORT_COUNT];
    uint64_t aliased_report_peak[DS4_RUNTIME_REPORT_COUNT];
    memcpy(aliased_report_current, fixture.tracker.report_current,
           sizeof(aliased_report_current));
    memcpy(aliased_report_peak, fixture.tracker.report_peak,
           sizeof(aliased_report_peak));
    memcpy(wrong_offset, valid_smaps, sizeof(wrong_offset));
    model_header = strstr(
        wrong_offset,
        "10000000-20000000 r--s 00000000 08:11 777");
    CHECK(model_header != NULL,
          "aliased failure-output fixture locates the model VMA header");
    if (!model_header) return;
    memcpy(model_header + 23u, "00001000", 8u);
    fixture.input.smaps_text = wrong_offset;
    fixture.input.smaps_text_bytes = sizeof(wrong_offset) - 1u;
    CHECK(ds4_runtime_tracker_checkpoint_attributed(
              &fixture.tracker, &fixture.input,
              &fixture.tracker.external_sample) ==
              DS4_RUNTIME_STATUS_UNSAFE,
          "aliased failure output is rejected without overwriting tracker storage");
    CHECK(memcmp(&fixture.tracker.external_sample,
                 &aliased_committed, sizeof(aliased_committed)) == 0 &&
              fixture.tracker.event_sequence == aliased_event_sequence &&
              fixture.tracker.qualification_total_current == aliased_total &&
              fixture.tracker.qualification_total_peak == aliased_peak &&
              memcmp(fixture.tracker.report_current,
                     aliased_report_current,
                     sizeof(aliased_report_current)) == 0 &&
              memcmp(fixture.tracker.report_peak,
                     aliased_report_peak,
                     sizeof(aliased_report_peak)) == 0,
          "sample_out aliasing preserves the committed sample, reports, and peaks");
}

static void test_legacy_checkpoint_invalidates_attribution(void) {
    external_fixture fixture;
    CHECK(fixture_init(&fixture),
          "legacy checkpoint provenance fixture initializes");
    const ds4_runtime_external_sample attributed = run_checkpoint(&fixture);
    CHECK(attributed.attributed_valid &&
              attributed.attributed_generation != UINT64_MAX,
          "attributed checkpoint establishes qualified provenance");

    CHECK(ds4_runtime_tracker_checkpoint_external(
              &fixture.tracker,
              fixture.tracker.report_current[
                  DS4_RUNTIME_REPORT_MODEL_SOURCE_RESIDENT],
              fixture.tracker.report_current[
                  DS4_RUNTIME_REPORT_HOST_LIBRARY_UNATTRIBUTED],
              fixture.tracker.report_current[
                  DS4_RUNTIME_REPORT_CUDA_LIBRARY_UNATTRIBUTED]) ==
              DS4_RUNTIME_STATUS_OK,
          "legacy external setter remains available for non-qualified callers");
    CHECK(!fixture.tracker.external_sample.attributed_valid &&
              fixture.tracker.external_sample.attributed_generation ==
                  attributed.attributed_generation + 1u,
          "legacy external setter invalidates and advances attribution provenance");

    ds4_runtime_snapshot snapshot;
    ds4_runtime_allocation_record active[6];
    CHECK(ds4_runtime_tracker_snapshot_copy(
              &fixture.tracker, &snapshot, active,
              sizeof(active) / sizeof(active[0])) &&
              !snapshot.external_sample.attributed_valid &&
              snapshot.external_sample.attributed_generation ==
                  attributed.attributed_generation + 1u,
          "snapshot cannot present a legacy setter as qualified attribution");
}

static void test_smaps_fail_closed(void) {
    static const char malformed[] =
        "1000-2000 rw-p not-a-header\nPss: 4 kB\n";
    static const char overflow[] =
        "1000-2000 rw-p 00000000 00:00 0 [heap]\n"
        "Pss: 18446744073709551615 kB\n";
    static const char missing_pss[] =
        "1000-2000 rw-p 00000000 00:00 0 [heap]\n"
        "Size: 4 kB\n";
    static const char duplicate_pss[] =
        "1000-3000 rw-p 00000000 00:00 0 [heap]\n"
        "Pss: 4 kB\n"
        "Pss: 4 kB\n";
    static const char overlapping_vmas[] =
        "2000-4000 rw-p 00000000 00:00 0 [heap]\n"
        "Pss: 4 kB\n"
        "3000-5000 rw-p 00000000 00:00 0 [stack]\n"
        "Pss: 4 kB\n";
    static const char pss_exceeds_span[] =
        "1000-2000 rw-p 00000000 00:00 0 [heap]\n"
        "Pss: 8 kB\n";

    external_fixture fixture;
    CHECK(fixture_init(&fixture), "malformed smaps fixture initializes");
    fixture.input.smaps_text = malformed;
    fixture.input.smaps_text_bytes = sizeof(malformed) - 1u;
    check_failure(&fixture, DS4_RUNTIME_EXTERNAL_FAILURE_SMAPS_PARSE,
                  "malformed smaps fails before committing a checkpoint");

    CHECK(fixture_init(&fixture), "overflow smaps fixture initializes");
    fixture.input.smaps_text = overflow;
    fixture.input.smaps_text_bytes = sizeof(overflow) - 1u;
    check_failure(&fixture, DS4_RUNTIME_EXTERNAL_FAILURE_SMAPS_OVERFLOW,
                  "overflowing smaps kB conversion fails closed");

    CHECK(fixture_init(&fixture), "missing PSS fixture initializes");
    fixture.input.smaps_text = missing_pss;
    fixture.input.smaps_text_bytes = sizeof(missing_pss) - 1u;
    check_failure(&fixture, DS4_RUNTIME_EXTERNAL_FAILURE_SMAPS_PARSE,
                  "every smaps VMA requires exactly one Pss field");

    CHECK(fixture_init(&fixture), "duplicate PSS fixture initializes");
    fixture.input.smaps_text = duplicate_pss;
    fixture.input.smaps_text_bytes = sizeof(duplicate_pss) - 1u;
    check_failure(&fixture, DS4_RUNTIME_EXTERNAL_FAILURE_SMAPS_PARSE,
                  "duplicate Pss fields fail closed");

    CHECK(fixture_init(&fixture), "overlapping VMA fixture initializes");
    fixture.input.smaps_text = overlapping_vmas;
    fixture.input.smaps_text_bytes = sizeof(overlapping_vmas) - 1u;
    check_failure(&fixture, DS4_RUNTIME_EXTERNAL_FAILURE_SMAPS_PARSE,
                  "smaps VMAs must be ordered and non-overlapping");

    CHECK(fixture_init(&fixture), "impossible PSS fixture initializes");
    fixture.input.smaps_text = pss_exceeds_span;
    fixture.input.smaps_text_bytes = sizeof(pss_exceeds_span) - 1u;
    check_failure(&fixture, DS4_RUNTIME_EXTERNAL_FAILURE_SMAPS_PARSE,
                  "VMA Pss cannot exceed its virtual span");

    CHECK(fixture_init(&fixture), "missing tracked VMA fixture initializes");
    fixture.records[0].base = UINT64_C(0x21000000);
    check_failure(&fixture, DS4_RUNTIME_EXTERNAL_FAILURE_TRACKED_VMA_MISSING,
                  "every live host-visible tracked range must match a smaps VMA");

    CHECK(fixture_init(&fixture), "overlapping tracked range fixture initializes");
    fixture.records[1].base = UINT64_C(0x20001000);
    fixture.records[1].requested_bytes = 4u * kib;
    check_failure(&fixture,
                  DS4_RUNTIME_EXTERNAL_FAILURE_TRACKED_RANGE_OVERLAP,
                  "overlapping tracked physical ranges fail de-duplication");

    CHECK(fixture_init(&fixture), "duplicate attribution fixture initializes");
    ds4_runtime_allocation_record duplicate_records[12];
    memcpy(duplicate_records, fixture.records, sizeof(duplicate_records));
    duplicate_records[6] = duplicate_records[0];
    fixture.input.attribution_records = duplicate_records;
    check_failure(&fixture,
                  DS4_RUNTIME_EXTERNAL_FAILURE_DUPLICATE_ATTRIBUTION,
                  "a tracker record cannot appear twice in the attribution table");
}

static void test_nvml_fail_closed(void) {
    external_fixture fixture;
    CHECK(fixture_init(&fixture), "NVML mismatch fixture initializes");
    fixture.after_inventory.api_version++;
    check_failure(&fixture, DS4_RUNTIME_EXTERNAL_FAILURE_NVML_API_MISMATCH,
                  "one frozen NVML API version is required at every inventory");

    CHECK(fixture_init(&fixture), "NVML library drift fixture initializes");
    fixture.inside_inventory.library_version = "580.95.06";
    check_failure(
        &fixture,
        DS4_RUNTIME_EXTERNAL_FAILURE_NVML_LIBRARY_VERSION_MISMATCH,
        "all inventories must use the frozen NVML library version");

    CHECK(fixture_init(&fixture), "NVML baseline version fixture initializes");
    fixture.input.baseline_nvml_library_version = "580.95.06";
    check_failure(
        &fixture,
        DS4_RUNTIME_EXTERNAL_FAILURE_NVML_LIBRARY_VERSION_MISMATCH,
        "inside-DS4 baseline must use the frozen NVML library version");

    CHECK(fixture_init(&fixture), "NVML baseline API fixture initializes");
    fixture.input.baseline_nvml_api_version++;
    check_failure(&fixture, DS4_RUNTIME_EXTERNAL_FAILURE_NVML_API_MISMATCH,
                  "inside-DS4 baseline must use the frozen NVML API version");

    CHECK(fixture_init(&fixture), "UUID mismatch fixture initializes");
    fixture.before_inventory.device_uuid = "GPU-wrong";
    check_failure(&fixture, DS4_RUNTIME_EXTERNAL_FAILURE_DEVICE_UUID_MISMATCH,
                  "every inventory must bind the expected GPU UUID");

    CHECK(fixture_init(&fixture), "build mismatch fixture initializes");
    uint8_t wrong_build[DS4_RUNTIME_BUILD_IDENTITY_BYTES];
    memcpy(wrong_build, fixture.build_identity, sizeof(wrong_build));
    wrong_build[7] ^= 1u;
    fixture.input.observed_build_identity = wrong_build;
    check_failure(&fixture, DS4_RUNTIME_EXTERNAL_FAILURE_BUILD_IDENTITY_MISMATCH,
                  "external sample must bind the running DS4 build identity");

    CHECK(fixture_init(&fixture), "baseline PID mismatch fixture initializes");
    fixture.input.baseline_process_id++;
    check_failure(&fixture, DS4_RUNTIME_EXTERNAL_FAILURE_PROCESS_ID_MISMATCH,
                  "inside-DS4 baseline and checkpoint must bind the same PID");

    CHECK(fixture_init(&fixture), "missing own process fixture initializes");
    fixture.inside_processes[0] = fixture.inside_processes[1];
    fixture.inside_inventory.process_count = 1u;
    check_failure(&fixture, DS4_RUNTIME_EXTERNAL_FAILURE_NVML_PROCESS_MISSING,
                  "own PID must exist in the inside-DS4 checkpoint inventory");

    CHECK(fixture_init(&fixture),
          "own-PID-free bracket inventory fixture initializes");
    fixture.before_processes[0] = fixture.before_processes[1];
    fixture.before_inventory.process_count = 1u;
    fixture.after_inventory.process_count = 1u;
    const ds4_runtime_external_sample bracket_peers_only =
        run_checkpoint(&fixture);
    CHECK(bracket_peers_only.nvml_process_bytes == 116u * mib,
          "before/after inventories bracket peers while inside inventory owns the charge");

    CHECK(fixture_init(&fixture), "unknown usage fixture initializes");
    fixture.inside_processes[0].used_bytes_known = false;
    check_failure(&fixture, DS4_RUNTIME_EXTERNAL_FAILURE_NVML_USAGE_UNKNOWN,
                  "unknown own-process NVML bytes invalidate attribution");

    CHECK(fixture_init(&fixture), "duplicate PID fixture initializes");
    fixture.inside_processes[1] = fixture.inside_processes[0];
    check_failure(&fixture, DS4_RUNTIME_EXTERNAL_FAILURE_NVML_DUPLICATE_PID,
                  "duplicate NVML PID records fail closed");

    CHECK(fixture_init(&fixture), "negative CUDA gap fixture initializes");
    fixture.inside_processes[0].used_bytes = 67u * mib;
    check_failure(&fixture, DS4_RUNTIME_EXTERNAL_FAILURE_CUDA_NEGATIVE_GAP,
                  "NVML usage below tracked CUDA physical bytes fails reconciliation");

    CHECK(fixture_init(&fixture), "no-context baseline fixture initializes");
    fixture.input.baseline_nvml_process_present = false;
    fixture.input.baseline_nvml_process_bytes_known = false;
    fixture.input.baseline_nvml_process_bytes = 0u;
    fixture.input.baseline_tracked_cuda_physical_bytes = 0u;
    const ds4_runtime_external_sample no_context = run_checkpoint(&fixture);
    CHECK(!no_context.nvml_process_baseline_present &&
              no_context.nvml_process_baseline_bytes == 0u,
          "an absent pre-context own PID is represented as valid zero usage");

    CHECK(fixture_init(&fixture), "unknown baseline fixture initializes");
    fixture.input.baseline_nvml_process_present = true;
    fixture.input.baseline_nvml_process_bytes_known = false;
    check_failure(&fixture, DS4_RUNTIME_EXTERNAL_FAILURE_NVML_USAGE_UNKNOWN,
                  "present-but-unknown baseline usage differs from no context");
}

static void test_peer_inventory_and_ceilings(void) {
    external_fixture fixture;
    CHECK(fixture_init(&fixture), "inside peer-drift fixture initializes");
    fixture.inside_processes[1].used_bytes += mib;
    check_failure(&fixture,
                  DS4_RUNTIME_EXTERNAL_FAILURE_UNRELATED_PROCESS_CHANGED,
                  "peer changes inside DS4 invalidate frozen infrastructure");

    CHECK(fixture_init(&fixture), "peer growth fixture initializes");
    fixture.before_processes[1].used_bytes += mib;
    fixture.after_processes[0].used_bytes += mib;
    check_failure(&fixture,
                  DS4_RUNTIME_EXTERNAL_FAILURE_UNRELATED_PROCESS_CHANGED,
                  "peer growth from pre-child baseline invalidates the sample");

    CHECK(fixture_init(&fixture), "peer narrow-window fixture initializes");
    fixture.pre_processes[0].used_bytes -= mib;
    check_failure(&fixture,
                  DS4_RUNTIME_EXTERNAL_FAILURE_UNRELATED_PROCESS_CHANGED,
                  "peer stable only across before/after still fails frozen-baseline comparison");

    CHECK(fixture_init(&fixture), "peer disappearance fixture initializes");
    fixture.after_inventory.process_count = 1u;
    fixture.after_processes[0] = fixture.after_processes[1];
    check_failure(&fixture,
                  DS4_RUNTIME_EXTERNAL_FAILURE_UNRELATED_PROCESS_CHANGED,
                  "a peer disappearing after the frozen baseline invalidates infrastructure");

    CHECK(fixture_init(&fixture), "peer appearance fixture initializes");
    fixture.pre_inventory.process_count = 0u;
    check_failure(&fixture,
                  DS4_RUNTIME_EXTERNAL_FAILURE_UNRELATED_PROCESS_CHANGED,
                  "a peer appearing after the frozen baseline invalidates infrastructure");

    CHECK(fixture_init(&fixture), "CUDA ceiling fixture initializes");
    fixture.inside_processes[0].used_bytes =
        68u * mib + 512u * mib + 1u;
    check_failure(&fixture,
                  DS4_RUNTIME_EXTERNAL_FAILURE_CUDA_UNATTRIBUTED_BOUND,
                  "CUDA library unattributed bytes are capped at 512 MiB");

    CHECK(fixture_init(&fixture), "exact CUDA ceiling fixture initializes");
    fixture.inside_processes[0].used_bytes = 68u * mib + 512u * mib;
    ds4_runtime_external_sample exact_cuda = run_checkpoint(&fixture);
    CHECK(exact_cuda.cuda_library_unattributed_bytes == 512u * mib,
          "exactly 512 MiB CUDA unattributed remains admissible");

    CHECK(fixture_init(&fixture), "host ceiling fixture initializes");
    static const char huge_host_smaps[] =
        "10000000-20000000 r--s 00000000 08:11 777 /models/laguna.gguf\n"
        "Pss: 4 kB\n"
        "50000000-80001000 r-xp 00000000 08:01 12 /usr/lib/libcuda.so.1\n"
        "Pss: 524289 kB\n";
    fixture.input.smaps_text = huge_host_smaps;
    fixture.input.smaps_text_bytes = sizeof(huge_host_smaps) - 1u;
    /* This focused fixture has no host-visible tracked allocations. */
    fixture.records[0].live = false;
    fixture.records[1].live = false;
    fixture.records[2].domain = DS4_RUNTIME_DOMAIN_CUDA_DEVICE;
    fixture.records[4].live = false;
    check_failure(&fixture,
                  DS4_RUNTIME_EXTERNAL_FAILURE_HOST_UNATTRIBUTED_BOUND,
                  "host library unattributed PSS is capped at 512 MiB");

    CHECK(fixture_init(&fixture), "exact host ceiling fixture initializes");
    static const char exact_host_smaps[] =
        "10000000-20000000 r--s 00000000 08:11 777 /models/laguna.gguf\n"
        "Pss: 4 kB\n"
        "50000000-70000000 r-xp 00000000 08:01 12 /usr/lib/libcuda.so.1\n"
        "Pss: 524288 kB\n";
    fixture.input.smaps_text = exact_host_smaps;
    fixture.input.smaps_text_bytes = sizeof(exact_host_smaps) - 1u;
    fixture.records[0].live = false;
    fixture.records[1].live = false;
    fixture.records[2].domain = DS4_RUNTIME_DOMAIN_CUDA_DEVICE;
    fixture.records[4].live = false;
    ds4_runtime_external_sample exact_host = run_checkpoint(&fixture);
    CHECK(exact_host.host_library_unattributed_bytes == 512u * mib,
          "exactly 512 MiB host unattributed remains admissible");
}

static int run_external_attribution(void) {
    test_valid_external_attribution();
    test_partial_vma_credit_and_managed_deduplication();
    test_host_charge_and_registration_deduplication();
    test_model_source_tail_page_semantics();
    test_exact_model_mapping_binding();
    test_failed_checkpoint_retains_last_good_sample();
    test_legacy_checkpoint_invalidates_attribution();
    test_smaps_fail_closed();
    test_nvml_fail_closed();
    test_peer_inventory_and_ceilings();
    if (g_failures != 0) {
        fprintf(stderr,
                "external-attribution: %d/%d assertions failed\n",
                g_failures, g_assertions);
        return 1;
    }
    printf("external-attribution: %d assertions passed\n", g_assertions);
    return 0;
}

static void usage(const char *program) {
    fprintf(stderr, "Usage: %s --case external-attribution\n", program);
}

int main(int argc, char **argv) {
    if (argc != 3 || strcmp(argv[1], "--case") != 0) {
        usage(argv[0]);
        return 2;
    }
    if (strcmp(argv[2], "external-attribution") == 0) {
        return run_external_attribution();
    }
    usage(argv[0]);
    return 2;
}
