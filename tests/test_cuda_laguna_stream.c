#define _POSIX_C_SOURCE 200809L

/* CUDA startup contract for the compact Laguna attachment.
 *
 * The synthetic case is model-independent and safe to run beside an existing
 * model server.  The model-startup case opens the pinned Laguna artifact and
 * must only be run by the guarded qualification workflow.
 */

#include "ds4.h"
#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"
#include "ds4_laguna_stream.h"
#include "ds4_runtime.h"

#include <cuda_runtime.h>

#include <errno.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define ARRAY_LEN(a) (sizeof(a) / sizeof((a)[0]))

static int g_assertions;
static int g_failures;

#define CHECK(condition, message)                                           \
    do {                                                                    \
        g_assertions++;                                                     \
        if (!(condition)) {                                                 \
            g_failures++;                                                   \
            fprintf(stderr, "FAIL: %s (line %d)\n", (message), __LINE__); \
        }                                                                   \
    } while (0)

enum {
    FIXTURE_TENSOR_COUNT = 8,
    FIXTURE_MODEL_BYTES = 1152,
    TRACKER_RECORD_CAPACITY = 16,
};

static const char *const forbidden_cuda_env[] = {
    "DS4_CUDA_COPY_MODEL",
    "DS4_CUDA_COPY_MODEL_CHUNKED",
    "DS4_CUDA_WEIGHT_CACHE",
    "DS4_CUDA_WEIGHT_PRELOAD",
    "DS4_CUDA_DIRECT_MODEL",
    "DS4_CUDA_WEIGHT_CACHE_LIMIT_GB",
    "DS4_CUDA_WEIGHT_PRELOAD_SPAN_MB",
    "DS4_CUDA_WEIGHT_ARENA_CHUNK_MB",
    "DS4_CUDA_Q8_F32_PRELOAD",
    "DS4_CUDA_KEEP_MODEL_PAGES",
    "DS4_CUDA_MODEL_PREFETCH_SYNC",
    "DS4_CUDA_Q8_F16_ALL",
    "DS4_CUDA_Q8_F32_ALL",
    "DS4_CUDA_Q8_F32_LARGE",
    "DS4_CUDA_ATTN_Q_B_F32_CACHE",
    "DS4_CUDA_ATTENTION_OUTPUT_PRELOAD",
    "DS4_CUDA_MODEL_COPY_CHUNK_MB",
    "DS4_CUDA_Q8_F16_CACHE_MB",
    "DS4_CUDA_Q8_F16_CACHE_RESERVE_MB",
    "DS4_CUDA_STRICT_WEIGHT_CACHE",
};

typedef struct {
    ds4_laguna_ledger_spec spec;
    ds4_laguna_tensor_desc tensors[FIXTURE_TENSOR_COUNT];
    char names[FIXTURE_TENSOR_COUNT][40];
} ledger_fixture;

typedef struct {
    ds4_runtime_allocation_record records[TRACKER_RECORD_CAPACITY];
    ds4_runtime_tracker tracker;
    uint64_t ledger_ids[3];
} tracker_fixture;

typedef struct {
    char *values[ARRAY_LEN(forbidden_cuda_env)];
} saved_environment;

static void fixture_tensor(
        ledger_fixture *fixture,
        size_t index,
        const char *name,
        uint64_t source_offset,
        ds4_laguna_tensor_class tensor_class,
        uint32_t routed_layer,
        ds4_laguna_routed_projection projection) {
    ds4_laguna_tensor_desc *tensor = &fixture->tensors[index];
    memset(tensor, 0, sizeof(*tensor));
    snprintf(fixture->names[index], sizeof(fixture->names[index]), "%s", name);
    tensor->stable_index = (uint64_t)index + 100u;
    tensor->name = fixture->names[index];
    tensor->name_len = strlen(fixture->names[index]);
    tensor->source_offset = source_offset;
    tensor->gguf_type = 2;
    tensor->block_elems = 32;
    tensor->block_bytes = 18;
    tensor->tensor_class = tensor_class;
    tensor->routed_layer = routed_layer;
    tensor->routed_projection = projection;
    if (tensor_class == DS4_LAGUNA_TENSOR_STATIC) {
        tensor->ndim = 1;
        tensor->dim[0] = 33;
        tensor->source_bytes = 36;
    } else {
        tensor->ndim = 3;
        tensor->dim[0] = 32;
        tensor->dim[1] = 2;
        tensor->dim[2] = 2;
        tensor->source_bytes = 72;
    }
}

static void ledger_fixture_prepare(ledger_fixture *fixture) {
    memset(fixture, 0, sizeof(*fixture));
    fixture->spec.file_size = FIXTURE_MODEL_BYTES;
    fixture->spec.header_end = 16;
    fixture->spec.metadata_end = 32;
    fixture->spec.tensor_directory_end = 65;
    fixture->spec.tensor_data_start = 128;
    fixture->spec.gguf_alignment = 64;
    fixture->spec.device_alignment = 256;
    fixture->spec.first_routed_layer = 1;
    fixture->spec.layer_count = 3;
    fixture->spec.expert_count = 2;

    fixture_tensor(fixture, 0, "token_embd.weight", 128,
                   DS4_LAGUNA_TENSOR_STATIC, UINT32_MAX,
                   DS4_LAGUNA_ROUTED_PROJECTION_NONE);
    fixture_tensor(fixture, 1, "output_norm.weight", 192,
                   DS4_LAGUNA_TENSOR_STATIC, UINT32_MAX,
                   DS4_LAGUNA_ROUTED_PROJECTION_NONE);
    fixture_tensor(fixture, 2, "blk.1.ffn_gate_exps.weight", 320,
                   DS4_LAGUNA_TENSOR_ROUTED_EXPERT, 1,
                   DS4_LAGUNA_ROUTED_PROJECTION_GATE);
    fixture_tensor(fixture, 3, "blk.1.ffn_up_exps.weight", 448,
                   DS4_LAGUNA_TENSOR_ROUTED_EXPERT, 1,
                   DS4_LAGUNA_ROUTED_PROJECTION_UP);
    fixture_tensor(fixture, 4, "blk.1.ffn_down_exps.weight", 576,
                   DS4_LAGUNA_TENSOR_ROUTED_EXPERT, 1,
                   DS4_LAGUNA_ROUTED_PROJECTION_DOWN);
    fixture_tensor(fixture, 5, "blk.2.ffn_gate_exps.weight", 704,
                   DS4_LAGUNA_TENSOR_ROUTED_EXPERT, 2,
                   DS4_LAGUNA_ROUTED_PROJECTION_GATE);
    fixture_tensor(fixture, 6, "blk.2.ffn_up_exps.weight", 832,
                   DS4_LAGUNA_TENSOR_ROUTED_EXPERT, 2,
                   DS4_LAGUNA_ROUTED_PROJECTION_UP);
    fixture_tensor(fixture, 7, "blk.2.ffn_down_exps.weight", 960,
                   DS4_LAGUNA_TENSOR_ROUTED_EXPERT, 2,
                   DS4_LAGUNA_ROUTED_PROJECTION_DOWN);
}

static bool ledger_fixture_build(ds4_laguna_ledger *ledger) {
    ledger_fixture fixture;
    char error[256] = {0};
    ledger_fixture_prepare(&fixture);
    memset(ledger, 0, sizeof(*ledger));
    if (!ds4_laguna_ledger_build(
            ledger, &fixture.spec, fixture.tensors,
            ARRAY_LEN(fixture.tensors), error, sizeof(error))) {
        fprintf(stderr, "FAIL: synthetic ledger: %s\n", error);
        return false;
    }
    return true;
}

static void plan_prepare(
        ds4_laguna_allocation_plan *plan,
        const ds4_laguna_ledger *ledger) {
    memset(plan, 0, sizeof(*plan));
    plan->profile_id = "synthetic-startup";
    plan->context_tokens = 32768;
    plan->prefill_rows = 4096;
    plan->session_count = 1;
    plan->slot_stride_bytes = ledger->slot_stride_bytes;
    plan->owned_category_bounds[DS4_RUNTIME_CATEGORY_STATIC_WEIGHTS] =
        ledger->static_aligned_device_bytes;
    const uint64_t static_offset_bytes =
        ledger->tensor_range_count * sizeof(uint64_t);
    const uint64_t ledger_array_bytes =
        ledger->tensor_range_count * sizeof(ledger->tensor_ranges[0]) +
        (ledger->tensor_range_count * 2u + 5u) *
            sizeof(ledger->source_ranges[0]) +
        ledger->expert_entry_count * sizeof(ledger->expert_entries[0]);
    plan->owned_category_bounds[
        DS4_RUNTIME_CATEGORY_CACHE_METADATA_ADDRESS_TABLES] =
            static_offset_bytes + ledger_array_bytes;
    plan->report_bounds[DS4_RUNTIME_REPORT_MODEL_MAPPED_VIRTUAL] =
        ledger->file_size;
    plan->report_bounds[DS4_RUNTIME_REPORT_MODEL_MAPPING_REGISTERED] = 0;
    plan->owned_total_bound_bytes =
        ledger->static_aligned_device_bytes + static_offset_bytes +
        ledger_array_bytes;
    plan->qualification_total_bound_bytes = plan->owned_total_bound_bytes;
    plan->callsite_count = 4;
    plan->callsites[0] = (ds4_runtime_callsite){
        .id = DS4_LAGUNA_CALLSITE_STATIC_SLAB,
        .name = "laguna.static_slab",
        .category = DS4_RUNTIME_CATEGORY_STATIC_WEIGHTS,
        .domain = DS4_RUNTIME_DOMAIN_CUDA_DEVICE,
        .bound_bytes = ledger->static_aligned_device_bytes,
    };
    plan->callsites[1] = (ds4_runtime_callsite){
        .id = DS4_LAGUNA_CALLSITE_EXPERT_CACHE_PAYLOAD,
        .name = "laguna.expert_cache_payload",
        .category = DS4_RUNTIME_CATEGORY_EXPERT_CACHE_PAYLOAD,
        .domain = DS4_RUNTIME_DOMAIN_CUDA_DEVICE,
        .bound_bytes = 0,
    };
    plan->callsites[2] = (ds4_runtime_callsite){
        .id = DS4_LAGUNA_CALLSITE_STATIC_OFFSETS,
        .name = "laguna.static_offsets",
        .category = DS4_RUNTIME_CATEGORY_CACHE_METADATA_ADDRESS_TABLES,
        .domain = DS4_RUNTIME_DOMAIN_HOST,
        .bound_bytes = static_offset_bytes,
    };
    plan->callsites[3] = (ds4_runtime_callsite){
        .id = DS4_LAGUNA_CALLSITE_LEDGER_ARRAYS,
        .name = "laguna.ledger_arrays",
        .category = DS4_RUNTIME_CATEGORY_CACHE_METADATA_ADDRESS_TABLES,
        .domain = DS4_RUNTIME_DOMAIN_HOST,
        .bound_bytes = ledger_array_bytes,
    };
}

static bool tracker_fixture_init(
        tracker_fixture *fixture,
        const ds4_laguna_allocation_plan *plan,
        const ds4_laguna_ledger *ledger) {
    memset(fixture, 0, sizeof(*fixture));
    ds4_runtime_tracker_config config = {
        .callsites = plan->callsites,
        .callsite_count = plan->callsite_count,
        .records = fixture->records,
        .record_capacity = ARRAY_LEN(fixture->records),
        .owned_total_bound_bytes = plan->owned_total_bound_bytes,
        .qualification_total_bound_bytes =
            plan->qualification_total_bound_bytes,
    };
    memcpy(config.category_bounds, plan->owned_category_bounds,
           sizeof(config.category_bounds));
    memcpy(config.report_bounds, plan->report_bounds,
           sizeof(config.report_bounds));
    if (ds4_runtime_tracker_init(&fixture->tracker, &config) !=
        DS4_RUNTIME_STATUS_OK) {
        return false;
    }
    const uint64_t source_capacity = ledger->tensor_range_count * 2u + 5u;
    const uint64_t requested[3] = {
        ledger->tensor_range_count * sizeof(ledger->tensor_ranges[0]),
        source_capacity * sizeof(ledger->source_ranges[0]),
        ledger->expert_entry_count * sizeof(ledger->expert_entries[0]),
    };
    const void *base[3] = {
        ledger->tensor_ranges,
        ledger->source_ranges,
        ledger->expert_entries,
    };
    for (size_t i = 0; i < ARRAY_LEN(fixture->ledger_ids); i++) {
        fixture->ledger_ids[i] = UINT64_C(0x4c45444745520001) + i;
        if (!base[i] || requested[i] == 0 ||
            ds4_runtime_tracker_allocate(
                &fixture->tracker, fixture->ledger_ids[i],
                DS4_LAGUNA_CALLSITE_LEDGER_ARRAYS,
                (uint64_t)(uintptr_t)base[i], requested[i], requested[i]) !=
                DS4_RUNTIME_STATUS_OK) {
            return false;
        }
    }
    return true;
}

static bool tracker_fixture_release_ledger(tracker_fixture *fixture) {
    bool ok = true;
    for (size_t i = 0; i < ARRAY_LEN(fixture->ledger_ids); i++) {
        if (ds4_runtime_tracker_release(
                &fixture->tracker, fixture->ledger_ids[i]) !=
            DS4_RUNTIME_STATUS_OK) {
            ok = false;
        }
    }
    return ok;
}

static void save_and_clear_forbidden_environment(saved_environment *saved) {
    memset(saved, 0, sizeof(*saved));
    for (size_t i = 0; i < ARRAY_LEN(forbidden_cuda_env); i++) {
        const char *value = getenv(forbidden_cuda_env[i]);
        if (value) saved->values[i] = strdup(value);
        unsetenv(forbidden_cuda_env[i]);
    }
}

static void restore_forbidden_environment(saved_environment *saved) {
    for (size_t i = 0; i < ARRAY_LEN(forbidden_cuda_env); i++) {
        if (saved->values[i]) {
            setenv(forbidden_cuda_env[i], saved->values[i], 1);
            free(saved->values[i]);
        } else {
            unsetenv(forbidden_cuda_env[i]);
        }
    }
}

static bool tracker_has_only_ledger(const ds4_runtime_tracker *tracker) {
    if (!tracker || tracker->violation != DS4_RUNTIME_VIOLATION_NONE ||
        tracker->record_count != 3) {
        return false;
    }
    for (size_t i = 0; i < DS4_RUNTIME_OWNED_CATEGORY_COUNT; i++) {
        if (i == DS4_RUNTIME_CATEGORY_CACHE_METADATA_ADDRESS_TABLES) continue;
        if (tracker->category_current[i] != 0) return false;
    }
    for (size_t i = 0; i < DS4_RUNTIME_REPORT_COUNT; i++) {
        if (tracker->report_current[i] != 0) return false;
    }
    return tracker->owned_total_current ==
        tracker->category_current[
            DS4_RUNTIME_CATEGORY_CACHE_METADATA_ADDRESS_TABLES];
}

static ds4_laguna_ledger ledger_with_copied_ranges(
        const ds4_laguna_ledger *source) {
    ds4_laguna_ledger copy = *source;
    copy.tensor_ranges = malloc(
        source->tensor_range_count * sizeof(source->tensor_ranges[0]));
    if (copy.tensor_ranges) {
        memcpy(copy.tensor_ranges, source->tensor_ranges,
               source->tensor_range_count * sizeof(source->tensor_ranges[0]));
    }
    return copy;
}

static void expect_invalid_ledger_before_allocation(
        int model_fd,
        const void *model_map,
        uint64_t model_size,
        const ds4_laguna_ledger *ledger,
        const ds4_laguna_allocation_plan *plan,
        const char *message) {
    tracker_fixture runtime;
    CHECK(tracker_fixture_init(&runtime, plan, ledger),
          "invalid-ledger tracker initializes");
    const uint64_t attempts_before =
        ds4_gpu_test_laguna_compact_static_allocation_attempts();
    ds4_gpu_laguna_compact *context = NULL;
    CHECK(!ds4_gpu_laguna_compact_create(
              &context, model_fd, model_map, model_size,
              ledger, plan, &runtime.tracker) && context == NULL,
          message);
    CHECK(ds4_gpu_test_laguna_compact_static_allocation_attempts() ==
              attempts_before,
          "invalid ledger is rejected before a static CUDA allocation attempt");
    CHECK(tracker_has_only_ledger(&runtime.tracker),
          "invalid ledger leaves only its live ledger records");
    ds4_gpu_laguna_compact_destroy(context);
}

static int run_startup(void) {
    int result = 1;
    int fd = -1;
    unsigned char *model_map = MAP_FAILED;
    ds4_laguna_ledger ledger;
    ds4_laguna_allocation_plan plan;
    ds4_gpu_laguna_compact *context = NULL;
    saved_environment saved;
    char path[] = "/tmp/ds4-cuda-laguna-startup.XXXXXX";
    memset(&ledger, 0, sizeof(ledger));
    memset(&plan, 0, sizeof(plan));
    save_and_clear_forbidden_environment(&saved);

    int device_count = 0;
    CHECK(cudaGetDeviceCount(&device_count) == cudaSuccess && device_count >= 1,
          "startup test has one visible CUDA device");
    if (device_count < 1) goto cleanup;
    CHECK(ds4_gpu_init() != 0, "CUDA backend initializes");

    fd = mkstemp(path);
    CHECK(fd >= 0, "synthetic model file opens");
    if (fd < 0) goto cleanup_gpu;
    unsigned char model_bytes[FIXTURE_MODEL_BYTES];
    for (size_t i = 0; i < sizeof(model_bytes); i++) {
        model_bytes[i] = (unsigned char)((i * 37u + 11u) & 0xffu);
    }
    CHECK(write(fd, model_bytes, sizeof(model_bytes)) ==
              (ssize_t)sizeof(model_bytes),
          "synthetic model file has exact bytes");
    model_map = mmap(NULL, sizeof(model_bytes), PROT_READ, MAP_SHARED, fd, 0);
    CHECK(model_map != MAP_FAILED, "synthetic model mapping opens read-only");
    if (model_map == MAP_FAILED) goto cleanup_file;
    CHECK(ledger_fixture_build(&ledger), "synthetic Laguna ledger builds");
    if (!ledger.tensor_ranges) goto cleanup_map;
    plan_prepare(&plan, &ledger);
    CHECK(ledger.static_parent_count == 2 &&
              ledger.routed_parent_count == 6 &&
              ledger.static_source_bytes == 72 &&
              ledger.static_aligned_device_bytes == 512,
          "synthetic ledger has the intended static/routed partition");
    const uint64_t ledger_array_bytes =
        ledger.tensor_range_count * sizeof(ledger.tensor_ranges[0]) +
        (ledger.tensor_range_count * 2u + 5u) *
            sizeof(ledger.source_ranges[0]) +
        ledger.expert_entry_count * sizeof(ledger.expert_entries[0]);

    ds4_laguna_ledger bad = ledger_with_copied_ranges(&ledger);
    CHECK(bad.tensor_ranges != NULL, "overlap mutation has private ranges");
    if (bad.tensor_ranges) {
        bad.tensor_ranges[1].source_offset =
            bad.tensor_ranges[0].source_offset + 1u;
        expect_invalid_ledger_before_allocation(
            fd, model_map, sizeof(model_bytes), &bad, &plan,
            "overlapping parent ranges fail closed");
        free(bad.tensor_ranges);
    }

    bad = ledger_with_copied_ranges(&ledger);
    CHECK(bad.tensor_ranges != NULL, "truncation mutation has private ranges");
    if (bad.tensor_ranges) {
        bad.tensor_ranges[7].source_offset = sizeof(model_bytes) - 1u;
        expect_invalid_ledger_before_allocation(
            fd, model_map, sizeof(model_bytes), &bad, &plan,
            "truncated parent range fails closed");
        free(bad.tensor_ranges);
    }

    bad = ledger_with_copied_ranges(&ledger);
    CHECK(bad.tensor_ranges != NULL,
          "misclassification mutation has private ranges");
    if (bad.tensor_ranges) {
        bad.tensor_ranges[2].tensor_class = DS4_LAGUNA_TENSOR_STATIC;
        expect_invalid_ledger_before_allocation(
            fd, model_map, sizeof(model_bytes), &bad, &plan,
            "routed/static misclassification fails closed");
        free(bad.tensor_ranges);
    }

    ds4_laguna_allocation_plan bad_plan = plan;
    bad_plan.owned_category_bounds[DS4_RUNTIME_CATEGORY_STATIC_WEIGHTS]--;
    bad_plan.owned_total_bound_bytes--;
    bad_plan.qualification_total_bound_bytes--;
    bad_plan.callsites[0].bound_bytes--;
    expect_invalid_ledger_before_allocation(
        fd, model_map, sizeof(model_bytes), &ledger, &bad_plan,
        "truncated static allocation plan fails closed");

    for (size_t i = 0; i < ARRAY_LEN(forbidden_cuda_env); i++) {
        tracker_fixture runtime;
        CHECK(tracker_fixture_init(&runtime, &plan, &ledger),
              "forbidden-environment tracker initializes");
        const uint64_t attempts_before =
            ds4_gpu_test_laguna_compact_static_allocation_attempts();
        CHECK(setenv(forbidden_cuda_env[i], "1", 1) == 0,
              "forbidden CUDA environment can be injected");
        ds4_gpu_laguna_compact *rejected = NULL;
        const int created = ds4_gpu_laguna_compact_create(
            &rejected, fd, model_map, sizeof(model_bytes),
            &ledger, &plan, &runtime.tracker);
        if (created || rejected) {
            fprintf(stderr, "unexpected acceptance: %s\n",
                    forbidden_cuda_env[i]);
        }
        CHECK(!created && rejected == NULL,
              "unsafe positive CUDA model/cache option is rejected");
        CHECK(ds4_gpu_test_laguna_compact_static_allocation_attempts() ==
                  attempts_before,
              "unsafe CUDA option fails before static CUDA allocation");
        CHECK(tracker_has_only_ledger(&runtime.tracker),
              "unsafe CUDA option leaves only live ledger records");
        ds4_gpu_laguna_compact_destroy(rejected);
        unsetenv(forbidden_cuda_env[i]);
    }

    CHECK(setenv("DS4_CUDA_NO_MODEL_COPY", "1", 1) == 0 &&
              setenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE", "1", 1) == 0,
          "harmless disabling and verbose CUDA knobs can be injected");
    tracker_fixture runtime;
    CHECK(tracker_fixture_init(&runtime, &plan, &ledger),
          "startup runtime tracker initializes");
    const uint64_t attempts_before_create =
        ds4_gpu_test_laguna_compact_static_allocation_attempts();
    CHECK(ds4_gpu_laguna_compact_create(
              &context, fd, model_map, sizeof(model_bytes),
              &ledger, &plan, &runtime.tracker) && context != NULL,
          "compact context attaches the synthetic model");
    CHECK(ds4_gpu_test_laguna_compact_static_allocation_attempts() ==
              attempts_before_create + 1u,
          "startup performs exactly one static CUDA slab allocation");

    ds4_gpu_laguna_compact_test_snapshot compact;
    memset(&compact, 0, sizeof(compact));
    CHECK(ds4_gpu_test_laguna_compact_snapshot(context, &compact),
          "compact context exposes a test snapshot");
    CHECK(compact.model_fd == fd && compact.model_map == model_map &&
              compact.model_size == sizeof(model_bytes),
          "compact attachment preserves exact fd/base/size identity");
    CHECK(compact.static_range_count == 2 &&
              compact.static_source_copied_bytes == 72 &&
              compact.static_slab_bytes == 512 &&
              compact.static_offset_count == FIXTURE_TENSOR_COUNT &&
              compact.static_offset_bytes ==
                  FIXTURE_TENSOR_COUNT * sizeof(uint64_t),
          "startup copies exactly the ledger-approved static ranges");
    CHECK(compact.model_mapping_registered_bytes == 0 &&
              compact.whole_model_copied_bytes == 0 &&
              compact.routed_payload_bytes == 0,
          "startup registers/copies no whole map and no routed payload");
    CHECK(compact.opportunistic_range_allocated_bytes == 0 &&
              compact.legacy_model_range_count == 0 &&
              compact.legacy_model_arena_count == 0,
          "startup creates no opportunistic range allocation or arena");

    ds4_runtime_snapshot runtime_snapshot;
    ds4_runtime_allocation_record active[8];
    CHECK(ds4_runtime_tracker_snapshot_copy(
              &runtime.tracker, &runtime_snapshot,
              active, ARRAY_LEN(active)) &&
              runtime_snapshot.active_record_count == 6,
          "tracker preserves ledger records and adds compact attachment records");
    CHECK(runtime_snapshot.category_current[
              DS4_RUNTIME_CATEGORY_STATIC_WEIGHTS] == 512 &&
              runtime_snapshot.category_current[
              DS4_RUNTIME_CATEGORY_EXPERT_CACHE_PAYLOAD] == 0 &&
              runtime_snapshot.category_current[
              DS4_RUNTIME_CATEGORY_CACHE_METADATA_ADDRESS_TABLES] ==
                  FIXTURE_TENSOR_COUNT * sizeof(uint64_t) +
                  ledger_array_bytes &&
              runtime_snapshot.report_current[
              DS4_RUNTIME_REPORT_MODEL_MAPPED_VIRTUAL] ==
                  sizeof(model_bytes) &&
              runtime_snapshot.report_current[
              DS4_RUNTIME_REPORT_MODEL_MAPPING_REGISTERED] == 0,
          "mapped bytes are metadata-only and static bytes reconcile exactly");

    for (size_t i = 0; i < 2; i++) {
        const uint64_t offset = ledger.tensor_ranges[i].source_offset;
        const uint64_t bytes = ledger.tensor_ranges[i].source_bytes;
        void *device_ptr = NULL;
        CHECK(ds4_gpu_test_laguna_compact_lookup(
                  context, offset, bytes, 0, &device_ptr) &&
                  device_ptr != NULL,
              "strict lookup hits a complete static range");
        unsigned char copied[64] = {0};
        CHECK(bytes <= sizeof(copied) &&
                  cudaMemcpy(copied, device_ptr, (size_t)bytes,
                             cudaMemcpyDeviceToHost) == cudaSuccess &&
                  memcmp(copied, model_map + offset, (size_t)bytes) == 0,
              "strict static lookup contains exact source bytes");
        const void *resolved = ds4_gpu_test_laguna_compact_resolve_weight_ptr(
            model_map, offset, bytes, 0, "startup-static");
        CHECK(resolved == device_ptr,
              "single-GPU kernel resolver uses the strict static lookup");
    }
    void *crossing = NULL;
    CHECK(!ds4_gpu_test_laguna_compact_lookup(
              context,
              ledger.tensor_ranges[0].source_offset,
              ledger.tensor_ranges[0].source_bytes + 1u,
              0, &crossing),
          "strict lookup rejects a static-range overrun");
    CHECK(!ds4_gpu_test_laguna_compact_lookup(
              context,
              ledger.tensor_ranges[0].source_offset,
              0, 0, &crossing),
          "strict lookup rejects a zero-byte request");

    const uint64_t routed_offset = ledger.tensor_ranges[2].source_offset;
    const uint64_t routed_bytes = ledger.tensor_ranges[2].source_bytes;
    void *routed = NULL;
    CHECK(!ds4_gpu_test_laguna_compact_lookup(
              context,
              routed_offset, routed_bytes, 0, &routed),
          "strict lookup misses routed payload at startup");
    CHECK(setenv("DS4_CUDA_DIRECT_MODEL", "1", 1) == 0 &&
              setenv("DS4_CUDA_WEIGHT_CACHE", "1", 1) == 0,
          "late unsafe fallback options can be injected");
    CHECK(ds4_gpu_test_laguna_compact_resolve_weight_ptr(
              model_map, routed_offset, routed_bytes, 0,
              "startup-routed-miss") == NULL,
          "compact resolver returns NULL on a routed miss without fallback");
    unsetenv("DS4_CUDA_DIRECT_MODEL");
    unsetenv("DS4_CUDA_WEIGHT_CACHE");
    memset(&compact, 0, sizeof(compact));
    CHECK(ds4_gpu_test_laguna_compact_snapshot(context, &compact) &&
              compact.opportunistic_range_allocated_bytes == 0 &&
              compact.legacy_model_range_count == 0 &&
              compact.legacy_model_arena_count == 0 &&
              compact.routed_payload_bytes == 0,
          "strict routed miss allocates, registers, and copies nothing");

    tracker_fixture second_runtime;
    CHECK(tracker_fixture_init(&second_runtime, &plan, &ledger),
          "second-context tracker initializes");
    ds4_gpu_laguna_compact *second = NULL;
    const uint64_t attempts_before_second =
        ds4_gpu_test_laguna_compact_static_allocation_attempts();
    CHECK(!ds4_gpu_laguna_compact_create(
              &second, fd, model_map, sizeof(model_bytes),
              &ledger, &plan, &second_runtime.tracker) && second == NULL,
          "only one compact context may be active per process");
    CHECK(ds4_gpu_test_laguna_compact_static_allocation_attempts() ==
              attempts_before_second &&
              tracker_has_only_ledger(&second_runtime.tracker),
          "second-context refusal preserves only its ledger records");

    ds4_gpu_laguna_compact_destroy(context);
    context = NULL;
    CHECK(!ds4_gpu_test_laguna_compact_active_snapshot(&compact),
          "destroy clears the process-global compact attachment");
    CHECK(ds4_runtime_tracker_snapshot_copy(
              &runtime.tracker, &runtime_snapshot,
              active, ARRAY_LEN(active)) &&
              runtime_snapshot.active_record_count == 3 &&
              runtime_snapshot.owned_total_current ==
                  runtime_snapshot.category_current[
                      DS4_RUNTIME_CATEGORY_CACHE_METADATA_ADDRESS_TABLES] &&
              runtime_snapshot.category_current[
                  DS4_RUNTIME_CATEGORY_CACHE_METADATA_ADDRESS_TABLES] ==
                  ledger_array_bytes &&
              runtime_snapshot.report_current[
              DS4_RUNTIME_REPORT_MODEL_MAPPED_VIRTUAL] == 0 &&
              runtime_snapshot.report_current[
              DS4_RUNTIME_REPORT_MODEL_MAPPING_REGISTERED] == 0 &&
              runtime_snapshot.violation == DS4_RUNTIME_VIOLATION_NONE,
          "compact teardown preserves only the three live ledger records");
    void *after_destroy = NULL;
    CHECK(!ds4_gpu_test_laguna_compact_lookup(
              context,
              ledger.tensor_ranges[0].source_offset,
              ledger.tensor_ranges[0].source_bytes,
              0, &after_destroy),
          "teardown removes compact static lookup entries before freeing slab");
    ds4_laguna_ledger_free(&ledger);
    CHECK(tracker_fixture_release_ledger(&runtime),
          "ledger records release after their physical arrays are freed");
    CHECK(ds4_runtime_tracker_snapshot_copy(
              &runtime.tracker, &runtime_snapshot,
              active, ARRAY_LEN(active)) &&
              runtime_snapshot.active_record_count == 0 &&
              runtime_snapshot.owned_total_current == 0 &&
              runtime_snapshot.violation == DS4_RUNTIME_VIOLATION_NONE,
          "full startup teardown reconciles every runtime record");
    result = g_failures == 0 ? 0 : 1;

cleanup_map:
    ds4_gpu_laguna_compact_destroy(context);
    ds4_laguna_ledger_free(&ledger);
    if (model_map != MAP_FAILED) munmap(model_map, FIXTURE_MODEL_BYTES);
cleanup_file:
    if (fd >= 0) close(fd);
    unlink(path);
cleanup_gpu:
    unsetenv("DS4_CUDA_NO_MODEL_COPY");
    unsetenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE");
    ds4_gpu_cleanup();
cleanup:
    restore_forbidden_environment(&saved);
    return result;
}

static int run_model_startup(void) {
    const char *model = getenv("DS4_TEST_MODEL");
    if (!model || !model[0]) {
        fprintf(stderr, "FAIL: DS4_TEST_MODEL is not set\n");
        return 1;
    }
    saved_environment saved;
    save_and_clear_forbidden_environment(&saved);
    const ds4_engine_options options = {
        .model_path = model,
        .backend = DS4_BACKEND_CUDA,
        .context_size = 32768,
        .prefill_chunk = 4096,
        .session_slots = 1,
        .ssd_streaming = true,
        .ssd_streaming_cache_bytes = UINT64_C(8) * 1024u * 1024u * 1024u,
        .ssd_streaming_cache_bytes_set = true,
    };
    ds4_engine *engine = NULL;
    CHECK(ds4_engine_open(&engine, &options) == 0 && engine != NULL,
          "pinned Laguna compact engine opens");
    ds4_gpu_laguna_compact_test_snapshot compact;
    memset(&compact, 0, sizeof(compact));
    CHECK(engine && ds4_gpu_test_laguna_compact_active_snapshot(&compact),
          "pinned engine owns one active compact attachment");
    CHECK(compact.model_map != NULL && compact.model_size != 0 &&
              compact.model_mapping_registered_bytes == 0 &&
              compact.whole_model_copied_bytes == 0 &&
              compact.routed_payload_bytes == 0 &&
              compact.opportunistic_range_allocated_bytes == 0,
          "pinned startup remains static-only and never registers the model map");
    ds4_engine_close(engine);
    CHECK(!ds4_gpu_test_laguna_compact_active_snapshot(&compact),
          "pinned engine teardown destroys compact attachment");
    restore_forbidden_environment(&saved);
    return g_failures == 0 ? 0 : 1;
}

static void usage(const char *program) {
    fprintf(stderr, "Usage: %s --case startup|model-startup\n", program);
}

int main(int argc, char **argv) {
    if (argc != 3 || strcmp(argv[1], "--case") != 0) {
        usage(argv[0]);
        return 2;
    }
    int rc = 2;
    if (strcmp(argv[2], "startup") == 0) {
        rc = run_startup();
    } else if (strcmp(argv[2], "model-startup") == 0) {
        rc = run_model_startup();
    } else {
        usage(argv[0]);
        return 2;
    }
    if (rc == 0) {
        fprintf(stderr, "test_cuda_laguna_stream %s PASS (%d assertions)\n",
                argv[2], g_assertions);
    } else {
        fprintf(stderr,
                "test_cuda_laguna_stream %s FAIL (%d/%d assertions)\n",
                argv[2], g_failures, g_assertions);
    }
    return rc;
}
