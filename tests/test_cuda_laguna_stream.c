#define _GNU_SOURCE
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
#include "ds4_plan_io.h"
#include "ds4_runtime.h"

#include <cuda_runtime.h>

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <unistd.h>

#ifndef MAP_ANONYMOUS
#define MAP_ANONYMOUS MAP_ANON
#endif

#define ARRAY_LEN(a) (sizeof(a) / sizeof((a)[0]))

static int g_assertions;
static int g_failures;

/* Task 12 RED seam.  First use on an ACTIVE context resets page-advice
 * telemetry and the retained post-advice residency to zero.  Each injection
 * then applies to the next completed compact-cache transfer: page_size
 * controls inward full-page selection, the two samples replace pre/post
 * mincore results, and each nonzero errno makes exactly the next syscall of
 * its named kind fail.  Keeping the failure controls separate proves that
 * fadvise and madvise are both attempted and accounted independently. */
void ds4_gpu_test_laguna_compact_page_advice_inject(
    uint64_t page_size,
    uint64_t exact_pre_resident_bytes,
    uint64_t exact_post_resident_bytes,
    int fadvise_errno,
    int madvise_errno);

/* Task 14 RED seam.  The policy layer can already form deterministic groups,
 * but the live compact routed path still rejects a batch whose unique working
 * set exceeds the fixed slots.  Keep this weak so the rest of the pressure
 * lifecycle runs and reports its independent invariants before the one missing
 * production integration is made explicit. */

/* Compile-only contract.  Runnable lifecycle behavior is driven separately
 * after this typed seam lands. */
static void compile_typed_lifecycle_contract(
        ds4_gpu_laguna_compact **context,
        int model_fd,
        const void *model_map,
        uint64_t model_size,
        const ds4_laguna_file_identity *identity,
        const ds4_laguna_ledger *ledger,
        const ds4_laguna_allocation_plan *plan,
        ds4_runtime_tracker *tracker) {
    const int created = ds4_gpu_laguna_compact_create(
        context, model_fd, model_map, model_size, identity,
        ledger, plan, tracker);
    const ds4_gpu_laguna_destroy_status destroyed =
        ds4_gpu_laguna_compact_destroy(created ? *context : NULL);
    ds4_gpu_laguna_compact_test_snapshot snapshot;
    memset(&snapshot, 0, sizeof(snapshot));
    snapshot.lifecycle = DS4_GPU_LAGUNA_LIFECYCLE_RELEASING;
    snapshot.model_identity = *identity;
    snapshot.static_slab = NULL;
    snapshot.static_offsets = NULL;
    snapshot.model_fd_live = true;
    snapshot.static_slab_live = true;
    snapshot.static_offsets_live = true;
    snapshot.tracker_mapping_live = true;
    snapshot.tracker_static_live = true;
    snapshot.tracker_offsets_live = true;
    snapshot.sync_attempt_count = 1;
    snapshot.release_attempt_count = 1;
    snapshot.rejection_count = 1;
    ds4_gpu_test_laguna_compact_fail_sync_once();
    ds4_gpu_test_laguna_compact_fail_release_once();
    (void)ds4_gpu_test_generic_cleanup_attempts();
    ds4_test_laguna_compact_close_observation observation;
    memset(&observation, 0, sizeof(observation));
    observation.destroy_result = (int)destroyed;
    observation.engine_retained = true;
    observation.gpu_cleanup_before = 1;
    observation.gpu_cleanup_after = 1;
    (void)ds4_test_laguna_compact_close_observation_get(&observation);
}

#define CHECK(condition, message)                                           \
    do {                                                                    \
        g_assertions++;                                                     \
        if (!(condition)) {                                                 \
            g_failures++;                                                   \
            fprintf(stderr, "FAIL: %s (line %d)\n", (message), __LINE__); \
        }                                                                   \
    } while (0)

static bool inherited_model_fd(int *out, bool *set) {
    const char *value = getenv("DS4_TEST_MODEL_FD");
    *out = -1;
    *set = false;
    if (!value) {
        return true;
    }
    errno = 0;
    char *end = NULL;
    const long parsed = strtol(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' ||
        parsed < 0 || parsed > INT_MAX || fcntl((int)parsed, F_GETFD) == -1) {
        fprintf(stderr,
                "FAIL: DS4_TEST_MODEL_FD is not an open descriptor: %s\n",
                value);
        return false;
    }
    *out = (int)parsed;
    *set = true;
    return true;
}

static bool cold_prepare_model_fd(
        int fd,
        uint64_t page_size,
        uint64_t *resident_bytes_out) {
    if (fd < 0 || page_size == 0 || !resident_bytes_out) return false;
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0 ||
        posix_fadvise(fd, 0, st.st_size, POSIX_FADV_DONTNEED) != 0) {
        return false;
    }
    const uint64_t model_size = (uint64_t)st.st_size;
    void *mapping = mmap(NULL, (size_t)model_size, PROT_READ,
                         MAP_PRIVATE, fd, 0);
    if (mapping == MAP_FAILED) return false;
    unsigned char residency[65536];
    uint64_t resident = 0;
    uint64_t offset = 0;
    bool ok = true;
    while (offset < model_size) {
        const uint64_t remaining = model_size - offset;
        uint64_t pages = remaining / page_size;
        if (remaining % page_size != 0) pages++;
        if (pages > sizeof(residency)) pages = sizeof(residency);
        uint64_t span = pages * page_size;
        if (span > remaining) span = remaining;
        memset(residency, 0, (size_t)pages);
        if (mincore((char *)mapping + offset,
                    (size_t)span, residency) != 0) {
            ok = false;
            break;
        }
        for (uint64_t page = 0; page < pages; page++) {
            if ((residency[page] & 1u) == 0) continue;
            const uint64_t page_offset = page * page_size;
            const uint64_t page_bytes = page_size < remaining - page_offset ?
                page_size : remaining - page_offset;
            if (resident > UINT64_MAX - page_bytes) {
                ok = false;
                break;
            }
            resident += page_bytes;
        }
        if (!ok) break;
        offset += span;
    }
    if (munmap(mapping, (size_t)model_size) != 0) ok = false;
    if (ok) *resident_bytes_out = resident;
    return ok;
}

enum {
    FIXTURE_TENSOR_COUNT = 8,
    FIXTURE_MODEL_BYTES = 1152,
    TRACKER_RECORD_CAPACITY = 16,
    PINNED_OWNER_COUNT = 6,
    PINNED_LEDGER_OWNER_COUNT = 3,
    PINNED_ACTIVE_RECORD_COUNT = 22,
};

typedef enum {
    PINNED_MODEL_STARTUP_NORMAL,
    PINNED_MODEL_TEARDOWN_RECONCILE_UNSAFE,
    PINNED_MODEL_CLEANUP_RELEASE_UNSAFE,
} pinned_model_case;

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

typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    unsigned arrived;
    unsigned target;
    unsigned generation;
    bool cancelled;
} test_barrier;

static bool test_barrier_init(test_barrier *barrier, unsigned target) {
    memset(barrier, 0, sizeof(*barrier));
    barrier->target = target;
    if (target == 0 || pthread_mutex_init(&barrier->mutex, NULL) != 0) {
        return false;
    }
    if (pthread_cond_init(&barrier->condition, NULL) != 0) {
        pthread_mutex_destroy(&barrier->mutex);
        return false;
    }
    return true;
}

static bool test_barrier_wait(test_barrier *barrier) {
    if (pthread_mutex_lock(&barrier->mutex) != 0) return false;
    const unsigned generation = barrier->generation;
    barrier->arrived++;
    if (barrier->arrived == barrier->target) {
        barrier->arrived = 0;
        barrier->generation++;
        pthread_cond_broadcast(&barrier->condition);
    } else {
        while (generation == barrier->generation && !barrier->cancelled) {
            if (pthread_cond_wait(
                    &barrier->condition, &barrier->mutex) != 0) {
                pthread_mutex_unlock(&barrier->mutex);
                return false;
            }
        }
    }
    const bool passed = !barrier->cancelled;
    return pthread_mutex_unlock(&barrier->mutex) == 0 && passed;
}

static void test_barrier_cancel(test_barrier *barrier) {
    if (pthread_mutex_lock(&barrier->mutex) != 0) return;
    barrier->cancelled = true;
    barrier->generation++;
    pthread_cond_broadcast(&barrier->condition);
    pthread_mutex_unlock(&barrier->mutex);
}

static void test_barrier_destroy(test_barrier *barrier) {
    pthread_cond_destroy(&barrier->condition);
    pthread_mutex_destroy(&barrier->mutex);
}

typedef struct {
    test_barrier *barrier;
    int model_fd;
    const void *model_map;
    uint64_t model_size;
    const ds4_laguna_file_identity *model_identity;
    const ds4_laguna_ledger *ledger;
    const ds4_laguna_allocation_plan *plan;
    tracker_fixture *runtime;
    ds4_gpu_laguna_compact *context;
    int created;
} creator_race;

static void *creator_race_run(void *opaque) {
    creator_race *race = opaque;
    if (!test_barrier_wait(race->barrier)) return NULL;
    race->created = ds4_gpu_laguna_compact_create(
        &race->context, race->model_fd, race->model_map, race->model_size,
        race->model_identity,
        race->ledger, race->plan, &race->runtime->tracker);
    return NULL;
}

static void *direct_creator_run(void *opaque) {
    creator_race *creator = opaque;
    creator->created = ds4_gpu_laguna_compact_create(
        &creator->context,
        creator->model_fd,
        creator->model_map,
        creator->model_size,
        creator->model_identity,
        creator->ledger,
        creator->plan,
        &creator->runtime->tracker);
    return NULL;
}

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

    fixture_tensor(fixture, 0, "token_embd.weight", 192,
                   DS4_LAGUNA_TENSOR_STATIC, UINT32_MAX,
                   DS4_LAGUNA_ROUTED_PROJECTION_NONE);
    fixture_tensor(fixture, 1, "output_norm.weight", 128,
                   DS4_LAGUNA_TENSOR_STATIC, UINT32_MAX,
                   DS4_LAGUNA_ROUTED_PROJECTION_NONE);
    fixture_tensor(fixture, 2, "blk.1.ffn_gate_exps.weight", 832,
                   DS4_LAGUNA_TENSOR_ROUTED_EXPERT, 1,
                   DS4_LAGUNA_ROUTED_PROJECTION_GATE);
    fixture_tensor(fixture, 3, "blk.1.ffn_up_exps.weight", 320,
                   DS4_LAGUNA_TENSOR_ROUTED_EXPERT, 1,
                   DS4_LAGUNA_ROUTED_PROJECTION_UP);
    fixture_tensor(fixture, 4, "blk.1.ffn_down_exps.weight", 704,
                   DS4_LAGUNA_TENSOR_ROUTED_EXPERT, 1,
                   DS4_LAGUNA_ROUTED_PROJECTION_DOWN);
    fixture_tensor(fixture, 5, "blk.2.ffn_gate_exps.weight", 448,
                   DS4_LAGUNA_TENSOR_ROUTED_EXPERT, 2,
                   DS4_LAGUNA_ROUTED_PROJECTION_GATE);
    fixture_tensor(fixture, 6, "blk.2.ffn_up_exps.weight", 960,
                   DS4_LAGUNA_TENSOR_ROUTED_EXPERT, 2,
                   DS4_LAGUNA_ROUTED_PROJECTION_UP);
    fixture_tensor(fixture, 7, "blk.2.ffn_down_exps.weight", 576,
                   DS4_LAGUNA_TENSOR_ROUTED_EXPERT, 2,
                   DS4_LAGUNA_ROUTED_PROJECTION_DOWN);
}

static bool ledger_fixture_build_for_size(
        ds4_laguna_ledger *ledger,
        uint64_t model_size) {
    ledger_fixture fixture;
    char error[256] = {0};
    ledger_fixture_prepare(&fixture);
    fixture.spec.file_size = model_size;
    memset(ledger, 0, sizeof(*ledger));
    if (!ds4_laguna_ledger_build(
            ledger, &fixture.spec, fixture.tensors,
            ARRAY_LEN(fixture.tensors), error, sizeof(error))) {
        fprintf(stderr, "FAIL: synthetic ledger: %s\n", error);
        return false;
    }
    return true;
}

static bool ledger_fixture_build(ds4_laguna_ledger *ledger) {
    return ledger_fixture_build_for_size(ledger, FIXTURE_MODEL_BYTES);
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

static void plan_prepare_cache(
        ds4_laguna_allocation_plan *plan,
        const ds4_laguna_ledger *ledger) {
    plan_prepare(plan, ledger);
    const uint64_t payload = 2u * ledger->slot_stride_bytes;
    const uint64_t staging = 4u * ledger->slot_stride_bytes;
    const uint64_t route_hotness =
        ledger->expert_entry_count * sizeof(uint64_t);
    const uint64_t entry_to_slot =
        ledger->expert_entry_count * sizeof(uint32_t);
    const uint64_t device_entry_to_slot = entry_to_slot;
    const uint64_t slot_state = 2u * sizeof(ds4_laguna_cache_slot);
    const uint64_t page_advice_state =
        (ledger->static_parent_count + 3u * ledger->expert_entry_count) *
        3u * sizeof(ds4_laguna_page_range);

    plan->profile_id = "synthetic-cache-io";
    plan->configured_cache_bytes = payload;
    plan->effective_cache_limit_bytes = payload;
    plan->cache_payload_bytes = payload;
    plan->slot_count = 2u;
    plan->staging_buffer_count = 4u;
    plan->staging_buffer_bytes = ledger->slot_stride_bytes;
    plan->owned_category_bounds[
        DS4_RUNTIME_CATEGORY_EXPERT_CACHE_PAYLOAD] = payload;
    plan->owned_category_bounds[
        DS4_RUNTIME_CATEGORY_CACHE_METADATA_ADDRESS_TABLES] +=
            route_hotness + entry_to_slot + device_entry_to_slot +
            slot_state;
    plan->owned_category_bounds[
        DS4_RUNTIME_CATEGORY_PINNED_STAGING] = staging;
    plan->owned_category_bounds[DS4_RUNTIME_CATEGORY_OTHER_HOST] =
        page_advice_state;
    plan->owned_total_bound_bytes +=
        payload + route_hotness + entry_to_slot + device_entry_to_slot +
        slot_state + staging + page_advice_state;
    plan->owned_non_cache_bound_bytes =
        plan->owned_total_bound_bytes - payload;
    plan->qualification_non_cache_bound_bytes =
        plan->owned_non_cache_bound_bytes;
    plan->planned_qualification_bytes = plan->owned_total_bound_bytes;
    plan->qualification_total_bound_bytes = plan->owned_total_bound_bytes;
    plan->report_bounds[DS4_RUNTIME_REPORT_MODEL_SOURCE_RESIDENT] =
        ledger->file_size;
    plan->qualification_total_bound_bytes += ledger->file_size;

    plan->callsites[1].bound_bytes = payload;
    plan->callsites[4] = (ds4_runtime_callsite){
        .id = DS4_LAGUNA_CALLSITE_ROUTE_HOTNESS,
        .name = "laguna.route_hotness",
        .category = DS4_RUNTIME_CATEGORY_CACHE_METADATA_ADDRESS_TABLES,
        .domain = DS4_RUNTIME_DOMAIN_HOST,
        .bound_bytes = route_hotness,
    };
    plan->callsites[5] = (ds4_runtime_callsite){
        .id = DS4_LAGUNA_CALLSITE_HOST_ENTRY_TO_SLOT,
        .name = "laguna.host_entry_to_slot",
        .category = DS4_RUNTIME_CATEGORY_CACHE_METADATA_ADDRESS_TABLES,
        .domain = DS4_RUNTIME_DOMAIN_HOST,
        .bound_bytes = entry_to_slot,
    };
    plan->callsites[6] = (ds4_runtime_callsite){
        .id = DS4_LAGUNA_CALLSITE_DEVICE_ENTRY_TO_SLOT,
        .name = "laguna.device_entry_to_slot",
        .category = DS4_RUNTIME_CATEGORY_CACHE_METADATA_ADDRESS_TABLES,
        .domain = DS4_RUNTIME_DOMAIN_CUDA_DEVICE,
        .bound_bytes = device_entry_to_slot,
    };
    plan->callsites[7] = (ds4_runtime_callsite){
        .id = DS4_LAGUNA_CALLSITE_SLOT_STATE,
        .name = "laguna.slot_state",
        .category = DS4_RUNTIME_CATEGORY_CACHE_METADATA_ADDRESS_TABLES,
        .domain = DS4_RUNTIME_DOMAIN_HOST,
        .bound_bytes = slot_state,
    };
    for (size_t i = 0; i < 4; i++) {
        plan->callsites[8u + i] = (ds4_runtime_callsite){
            .id = (uint32_t)DS4_LAGUNA_CALLSITE_PINNED_STAGING_0 +
                  (uint32_t)i,
            .name = i == 0 ? "laguna.pinned_staging.0" :
                    i == 1 ? "laguna.pinned_staging.1" :
                    i == 2 ? "laguna.pinned_staging.2" :
                             "laguna.pinned_staging.3",
            .category = DS4_RUNTIME_CATEGORY_PINNED_STAGING,
            .domain = DS4_RUNTIME_DOMAIN_HOST,
            .bound_bytes = ledger->slot_stride_bytes,
        };
    }
    plan->callsites[12] = (ds4_runtime_callsite){
        .id = DS4_LAGUNA_CALLSITE_OTHER_HOST_TRACKER,
        .name = "laguna.page_advice_state",
        .category = DS4_RUNTIME_CATEGORY_OTHER_HOST,
        .domain = DS4_RUNTIME_DOMAIN_HOST,
        .bound_bytes = page_advice_state,
    };
    plan->callsite_count = 13u;
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
    ds4_runtime_snapshot snapshot;
    ds4_runtime_allocation_record active[3];
    if (!tracker || !ds4_runtime_tracker_snapshot_copy(
            tracker, &snapshot, active, ARRAY_LEN(active)) ||
        snapshot.violation != DS4_RUNTIME_VIOLATION_NONE ||
        snapshot.active_record_count != ARRAY_LEN(active)) {
        return false;
    }
    for (size_t i = 0; i < ARRAY_LEN(active); i++) {
        if (!active[i].live ||
            active[i].callsite_id != DS4_LAGUNA_CALLSITE_LEDGER_ARRAYS ||
            active[i].category !=
                DS4_RUNTIME_CATEGORY_CACHE_METADATA_ADDRESS_TABLES ||
            active[i].domain != DS4_RUNTIME_DOMAIN_HOST ||
            active[i].relation != DS4_RUNTIME_RELATION_OWNED_ALLOCATION ||
            active[i].requested_bytes == 0 ||
            active[i].charged_bytes != active[i].requested_bytes) {
            return false;
        }
    }
    for (size_t i = 0; i < DS4_RUNTIME_OWNED_CATEGORY_COUNT; i++) {
        if (i == DS4_RUNTIME_CATEGORY_CACHE_METADATA_ADDRESS_TABLES) continue;
        if (snapshot.category_current[i] != 0) return false;
    }
    for (size_t i = 0; i < DS4_RUNTIME_REPORT_COUNT; i++) {
        if (snapshot.report_current[i] != 0) return false;
    }
    const uint64_t ledger_bytes = snapshot.category_current[
        DS4_RUNTIME_CATEGORY_CACHE_METADATA_ADDRESS_TABLES];
    return ledger_bytes != 0 && snapshot.owned_total_current == ledger_bytes &&
        snapshot.qualification_total_current == ledger_bytes;
}

static bool tracker_fixture_fill_with_tombstones(
        tracker_fixture *fixture) {
    const size_t tombstone_count =
        ARRAY_LEN(fixture->records) - ARRAY_LEN(fixture->ledger_ids);
    for (size_t i = 0; i < tombstone_count; i++) {
        if (ds4_runtime_tracker_allocate(
                &fixture->tracker,
                UINT64_C(0x5400000000000001) + i,
                DS4_LAGUNA_CALLSITE_STATIC_SLAB,
                UINT64_C(0x7000000000000000) +
                    i * UINT64_C(0x100),
                1u, 1u) != DS4_RUNTIME_STATUS_OK) {
            return false;
        }
    }
    for (size_t i = 0; i < tombstone_count; i++) {
        if (ds4_runtime_tracker_release(
                &fixture->tracker,
                UINT64_C(0x5400000000000001) + i) !=
            DS4_RUNTIME_STATUS_OK) {
            return false;
        }
    }
    return fixture->tracker.record_count == ARRAY_LEN(fixture->records) &&
        tracker_has_only_ledger(&fixture->tracker);
}

static bool capture_file_identity(
        int fd,
        ds4_laguna_file_identity *identity) {
    char error[160] = {0};
    if (!identity) return false;
    memset(identity, 0, sizeof(*identity));
    if (!ds4_test_laguna_file_identity_capture(
            fd,
            &identity->device,
            &identity->inode,
            &identity->size_bytes,
            &identity->mtime_ns,
            error,
            sizeof(error))) {
        fprintf(stderr, "FAIL: capture model identity: %s\n", error);
        return false;
    }
    return true;
}

static bool identities_equal(
        const ds4_laguna_file_identity *a,
        const ds4_laguna_file_identity *b) {
    return a && b &&
        a->device == b->device &&
        a->inode == b->inode &&
        a->size_bytes == b->size_bytes &&
        a->mtime_ns == b->mtime_ns;
}

static bool write_pattern_file(int fd, uint64_t bytes, unsigned seed) {
    unsigned char buffer[4096];
    uint64_t offset = 0;
    while (offset < bytes) {
        const size_t amount =
            bytes - offset < sizeof(buffer) ?
                (size_t)(bytes - offset) : sizeof(buffer);
        for (size_t i = 0; i < amount; i++) {
            buffer[i] = (unsigned char)(
                ((offset + i) * UINT64_C(37) + seed) & UINT64_C(0xff));
        }
        const ssize_t written = pwrite(fd, buffer, amount, (off_t)offset);
        if (written != (ssize_t)amount) return false;
        offset += amount;
    }
    return fsync(fd) == 0;
}

static int create_pattern_file(
        char *path,
        uint64_t bytes,
        unsigned seed) {
    const int fd = mkstemp(path);
    if (fd < 0) return -1;
    if (!write_pattern_file(fd, bytes, seed)) {
        close(fd);
        unlink(path);
        return -1;
    }
    return fd;
}

static int open_fd_count(void) {
    DIR *directory = opendir("/proc/self/fd");
    if (!directory) return -1;
    int count = 0;
    struct dirent *entry;
    while ((entry = readdir(directory)) != NULL) {
        if (strcmp(entry->d_name, ".") != 0 &&
            strcmp(entry->d_name, "..") != 0) {
            count++;
        }
    }
    closedir(directory);
    return count;
}

static bool proc_maps_has_readable_split_cover(
        const void *base,
        size_t bytes,
        unsigned *entry_count) {
    if (entry_count) *entry_count = 0;
    const uintptr_t range_start = (uintptr_t)base;
    if (!base || bytes == 0 || bytes > UINTPTR_MAX - range_start) {
        return false;
    }
    const unsigned long long range_end =
        (unsigned long long)(range_start + bytes);
    unsigned long long cursor = (unsigned long long)range_start;
    unsigned readable_entries = 0;
    FILE *maps = fopen("/proc/self/maps", "r");
    if (!maps) return false;

    char line[512];
    while (cursor < range_end && fgets(line, sizeof(line), maps)) {
        unsigned long long map_start = 0;
        unsigned long long map_end = 0;
        char permissions[5] = {0};
        if (sscanf(line, "%llx-%llx %4s",
                   &map_start, &map_end, permissions) != 3) {
            continue;
        }
        if (map_end <= cursor) continue;
        if (map_start > cursor || permissions[0] != 'r') break;
        readable_entries++;
        cursor = map_end < range_end ? map_end : range_end;
    }
    fclose(maps);
    if (entry_count) *entry_count = readable_entries;
    return cursor == range_end && readable_entries >= 2;
}

static bool compact_snapshot_equal_except_sync_lifecycle(
        ds4_gpu_laguna_compact_test_snapshot a,
        ds4_gpu_laguna_compact_test_snapshot b) {
    a.lifecycle = DS4_GPU_LAGUNA_LIFECYCLE_IDLE;
    b.lifecycle = DS4_GPU_LAGUNA_LIFECYCLE_IDLE;
    a.sync_attempt_count = 0;
    b.sync_attempt_count = 0;
    return memcmp(&a, &b, sizeof(a)) == 0;
}

static bool compact_snapshot_equal_except_teardown_progress(
        ds4_gpu_laguna_compact_test_snapshot a,
        ds4_gpu_laguna_compact_test_snapshot b) {
    a.lifecycle = DS4_GPU_LAGUNA_LIFECYCLE_IDLE;
    b.lifecycle = DS4_GPU_LAGUNA_LIFECYCLE_IDLE;
    a.sync_attempt_count = 0;
    b.sync_attempt_count = 0;
    a.release_attempt_count = 0;
    b.release_attempt_count = 0;
    return memcmp(&a, &b, sizeof(a)) == 0;
}

typedef struct {
    uint64_t offsets[FIXTURE_TENSOR_COUNT];
    unsigned char payloads[2][64];
} compact_allocation_bytes;

static bool capture_compact_allocation_bytes(
        const ds4_gpu_laguna_compact_test_snapshot *snapshot,
        const ds4_laguna_ledger *ledger,
        compact_allocation_bytes *out) {
    if (!snapshot || !ledger || !out ||
        !snapshot->static_offsets || !snapshot->static_slab ||
        snapshot->static_offset_bytes != sizeof(out->offsets) ||
        ledger->tensor_range_count < 2) {
        return false;
    }
    memset(out, 0, sizeof(*out));
    memcpy(out->offsets, snapshot->static_offsets, sizeof(out->offsets));
    for (size_t i = 0; i < 2; i++) {
        const uint64_t bytes = ledger->tensor_ranges[i].source_bytes;
        if (bytes > sizeof(out->payloads[i]) ||
            cudaMemcpy(
                out->payloads[i],
                (const char *)snapshot->static_slab + out->offsets[i],
                (size_t)bytes,
                cudaMemcpyDeviceToHost) != cudaSuccess) {
            return false;
        }
    }
    return true;
}

static bool tracker_snapshots_equal(
        const ds4_runtime_snapshot *a,
        const ds4_runtime_allocation_record *a_records,
        const ds4_runtime_snapshot *b,
        const ds4_runtime_allocation_record *b_records) {
    return a && b && a_records && b_records &&
        memcmp(a, b, sizeof(*a)) == 0 &&
        a->active_record_count == b->active_record_count &&
        a->active_record_count <= TRACKER_RECORD_CAPACITY &&
        memcmp(a_records, b_records,
               a->active_record_count * sizeof(a_records[0])) == 0;
}

static bool capture_tracker_snapshot(
        const ds4_runtime_tracker *tracker,
        ds4_runtime_snapshot *snapshot,
        ds4_runtime_allocation_record *records) {
    if (!tracker || !snapshot || !records) return false;
    memset(snapshot, 0, sizeof(*snapshot));
    memset(records, 0,
           TRACKER_RECORD_CAPACITY * sizeof(records[0]));
    return ds4_runtime_tracker_snapshot_copy(
        tracker, snapshot, records, TRACKER_RECORD_CAPACITY);
}

static bool owned_record_matches(
        const ds4_runtime_allocation_record *record,
        uint64_t base,
        uint64_t bytes,
        uint32_t callsite_id,
        ds4_runtime_category category,
        ds4_runtime_physical_domain domain) {
    return record && record->live && record->base == base &&
        record->requested_bytes == bytes && record->charged_bytes == bytes &&
        record->callsite_id == callsite_id &&
        record->category == category && record->domain == domain &&
        record->relation == DS4_RUNTIME_RELATION_OWNED_ALLOCATION &&
        record->owner_id == 0;
}

static bool checked_add_u64_test(uint64_t a, uint64_t b, uint64_t *out) {
    if (!out || a > UINT64_MAX - b) return false;
    *out = a + b;
    return true;
}

static bool bytes_are_value(
        const void *bytes, size_t byte_count, unsigned char value) {
    if (!bytes) return false;
    const unsigned char *cursor = bytes;
    for (size_t i = 0; i < byte_count; i++) {
        if (cursor[i] != value) return false;
    }
    return true;
}

static bool pinned_live_owners_valid(
        const ds4_test_laguna_live_owner owners[PINNED_OWNER_COUNT],
        uint64_t *total_bytes) {
    if (!owners || !total_bytes) return false;
    size_t engine_count = 0;
    size_t model_count = 0;
    size_t vocab_count = 0;
    uint64_t total = 0;
    for (size_t i = 0; i < PINNED_OWNER_COUNT; i++) {
        if (owners[i].base == 0 || owners[i].bytes == 0 ||
            owners[i].base > UINT64_MAX - owners[i].bytes ||
            !checked_add_u64_test(total, owners[i].bytes, &total)) {
            return false;
        }
        switch (owners[i].callsite_id) {
        case DS4_LAGUNA_CALLSITE_OTHER_HOST_ENGINE:
            engine_count++;
            break;
        case DS4_LAGUNA_CALLSITE_OTHER_HOST_MODEL:
            model_count++;
            break;
        case DS4_LAGUNA_CALLSITE_OTHER_HOST_VOCAB:
            vocab_count++;
            break;
        default:
            return false;
        }
        for (size_t j = 0; j < i; j++) {
            const uint64_t i_end = owners[i].base + owners[i].bytes;
            const uint64_t j_end = owners[j].base + owners[j].bytes;
            if (owners[i].base < j_end && owners[j].base < i_end) {
                return false;
            }
        }
    }
    *total_bytes = total;
    return engine_count == 1 && model_count == 2 && vocab_count == 3;
}

static bool pinned_ledger_owners_valid(
        const ds4_test_laguna_live_owner
            owners[PINNED_LEDGER_OWNER_COUNT],
        uint64_t *total_bytes) {
    if (!owners || !total_bytes) return false;
    uint64_t total = 0;
    for (size_t i = 0; i < PINNED_LEDGER_OWNER_COUNT; i++) {
        if (owners[i].base == 0 || owners[i].bytes == 0 ||
            owners[i].callsite_id != DS4_LAGUNA_CALLSITE_LEDGER_ARRAYS ||
            owners[i].base > UINT64_MAX - owners[i].bytes ||
            !checked_add_u64_test(total, owners[i].bytes, &total)) {
            return false;
        }
        for (size_t j = 0; j < i; j++) {
            const uint64_t i_end = owners[i].base + owners[i].bytes;
            const uint64_t j_end = owners[j].base + owners[j].bytes;
            if (owners[i].base < j_end && owners[j].base < i_end) {
                return false;
            }
        }
    }
    *total_bytes = total;
    return true;
}

static bool pinned_ledger_records_match(
        const ds4_runtime_snapshot *runtime,
        const ds4_runtime_allocation_record *records,
        const ds4_test_laguna_live_owner
            owners[PINNED_LEDGER_OWNER_COUNT],
        uint64_t *ledger_bytes) {
    if (!runtime || !records || !owners || !ledger_bytes) return false;
    bool owner_seen[PINNED_LEDGER_OWNER_COUNT] = {false};
    size_t ledger_count = 0;
    uint64_t total = 0;
    for (size_t i = 0; i < runtime->active_record_count; i++) {
        const ds4_runtime_allocation_record *record = &records[i];
        if (record->callsite_id != DS4_LAGUNA_CALLSITE_LEDGER_ARRAYS) {
            continue;
        }
        size_t match = PINNED_LEDGER_OWNER_COUNT;
        for (size_t j = 0; j < PINNED_LEDGER_OWNER_COUNT; j++) {
            if (record->base == owners[j].base &&
                record->requested_bytes == owners[j].bytes) {
                if (match != PINNED_LEDGER_OWNER_COUNT) return false;
                match = j;
            }
        }
        if (match == PINNED_LEDGER_OWNER_COUNT || owner_seen[match] ||
            !owned_record_matches(
                record, owners[match].base, owners[match].bytes,
                DS4_LAGUNA_CALLSITE_LEDGER_ARRAYS,
                DS4_RUNTIME_CATEGORY_CACHE_METADATA_ADDRESS_TABLES,
                DS4_RUNTIME_DOMAIN_HOST) ||
            !checked_add_u64_test(
                total, owners[match].bytes, &total)) {
            return false;
        }
        owner_seen[match] = true;
        ledger_count++;
    }
    if (ledger_count != PINNED_LEDGER_OWNER_COUNT) return false;
    for (size_t i = 0; i < PINNED_LEDGER_OWNER_COUNT; i++) {
        if (!owner_seen[i]) return false;
    }
    *ledger_bytes = total;
    return true;
}

static bool pinned_inventory_records_match(
        const ds4_runtime_snapshot *runtime,
        const ds4_runtime_allocation_record *records,
        const ds4_test_laguna_live_owner owners[PINNED_OWNER_COUNT]) {
    if (!runtime || !records || !owners ||
        runtime->active_record_count != PINNED_ACTIVE_RECORD_COUNT) {
        return false;
    }
    bool owner_seen[PINNED_OWNER_COUNT] = {false};
    bool sequence_seen[PINNED_OWNER_COUNT] = {false};
    size_t inventory_count = 0;
    size_t engine_count = 0;
    size_t model_count = 0;
    size_t vocab_count = 0;
    const uint64_t sequence_mask = UINT64_C(0x00ffffffffffffff);

    for (size_t i = 0; i < runtime->active_record_count; i++) {
        const ds4_runtime_allocation_record *record = &records[i];
        if (record->callsite_id != DS4_LAGUNA_CALLSITE_OTHER_HOST_ENGINE &&
            record->callsite_id != DS4_LAGUNA_CALLSITE_OTHER_HOST_MODEL &&
            record->callsite_id != DS4_LAGUNA_CALLSITE_OTHER_HOST_VOCAB) {
            continue;
        }
        inventory_count++;
        if ((uint8_t)(record->id >> 56) != 0x4eu) return false;
        const uint64_t sequence = record->id & sequence_mask;
        if (sequence == 0 || sequence > PINNED_OWNER_COUNT ||
            sequence_seen[sequence - 1u]) {
            return false;
        }
        sequence_seen[sequence - 1u] = true;
        const size_t owner_index = (size_t)(sequence - 1u);
        const ds4_test_laguna_live_owner *owner = &owners[owner_index];
        if (owner_seen[owner_index] ||
            !owned_record_matches(
                record, owner->base, owner->bytes, owner->callsite_id,
                DS4_RUNTIME_CATEGORY_OTHER_HOST,
                DS4_RUNTIME_DOMAIN_HOST)) {
            return false;
        }
        owner_seen[owner_index] = true;
        if (record->callsite_id ==
                DS4_LAGUNA_CALLSITE_OTHER_HOST_ENGINE) {
            engine_count++;
        } else if (record->callsite_id ==
                       DS4_LAGUNA_CALLSITE_OTHER_HOST_MODEL) {
            model_count++;
        } else {
            vocab_count++;
        }
    }
    if (inventory_count != PINNED_OWNER_COUNT || engine_count != 1 ||
        model_count != 2 || vocab_count != 3) {
        return false;
    }
    for (size_t i = 0; i < PINNED_OWNER_COUNT; i++) {
        if (!owner_seen[i] || !sequence_seen[i]) return false;
    }
    return true;
}

static bool pinned_runtime_inventory_reconciles(
        const ds4_runtime_snapshot *runtime,
        const ds4_runtime_allocation_record *records,
        const ds4_test_laguna_live_owner owners[PINNED_OWNER_COUNT],
        const ds4_test_laguna_live_owner
            ledger_owners[PINNED_LEDGER_OWNER_COUNT],
        uint64_t owner_bytes,
        const ds4_gpu_laguna_compact_test_snapshot *compact) {
    if (!runtime || !records || !owners || !ledger_owners || !compact ||
        runtime->violation != DS4_RUNTIME_VIOLATION_NONE ||
        runtime->active_record_count != PINNED_ACTIVE_RECORD_COUNT) {
        return false;
    }
    uint64_t ledger_bytes = 0;
    if (!pinned_inventory_records_match(runtime, records, owners) ||
        !pinned_ledger_records_match(
            runtime, records, ledger_owners, &ledger_bytes)) {
        return false;
    }
    size_t ledger_count = 0;
    size_t static_count = 0;
    size_t offsets_count = 0;
    size_t mapping_count = 0;
    size_t inventory_count = 0;
    size_t cache_payload_count = 0;
    size_t route_hotness_count = 0;
    size_t host_entry_to_slot_count = 0;
    size_t device_entry_to_slot_count = 0;
    size_t slot_state_count = 0;
    size_t page_advice_state_count = 0;
    bool staging_seen[4] = {false, false, false, false};
    const uint64_t route_hotness_bytes = UINT64_C(12032) * sizeof(uint64_t);
    const uint64_t entry_to_slot_bytes = UINT64_C(12032) * sizeof(uint32_t);
    const uint64_t slot_state_bytes =
        compact->cache_slot_count * sizeof(ds4_laguna_cache_slot);
    uint64_t charged_bytes = 0;
    for (size_t i = 0; i < runtime->active_record_count; i++) {
        const ds4_runtime_allocation_record *record = &records[i];
        if (!record->live ||
            !checked_add_u64_test(
                charged_bytes, record->charged_bytes, &charged_bytes)) {
            return false;
        }
        if (record->callsite_id == DS4_LAGUNA_CALLSITE_LEDGER_ARRAYS) {
            ledger_count++;
        } else if (record->callsite_id ==
                       DS4_LAGUNA_CALLSITE_STATIC_SLAB) {
            if (!owned_record_matches(
                    record, (uint64_t)(uintptr_t)compact->static_slab,
                    compact->static_slab_bytes,
                    DS4_LAGUNA_CALLSITE_STATIC_SLAB,
                    DS4_RUNTIME_CATEGORY_STATIC_WEIGHTS,
                    DS4_RUNTIME_DOMAIN_CUDA_DEVICE)) {
                return false;
            }
            static_count++;
        } else if (record->callsite_id ==
                       DS4_LAGUNA_CALLSITE_STATIC_OFFSETS) {
            if (!owned_record_matches(
                    record, (uint64_t)(uintptr_t)compact->static_offsets,
                    compact->static_offset_bytes,
                    DS4_LAGUNA_CALLSITE_STATIC_OFFSETS,
                    DS4_RUNTIME_CATEGORY_CACHE_METADATA_ADDRESS_TABLES,
                    DS4_RUNTIME_DOMAIN_HOST)) {
                return false;
            }
            offsets_count++;
        } else if (record->callsite_id ==
                       DS4_LAGUNA_CALLSITE_EXPERT_CACHE_PAYLOAD) {
            if (!owned_record_matches(
                    record, (uint64_t)(uintptr_t)compact->cache_payload,
                    compact->cache_payload_bytes,
                    DS4_LAGUNA_CALLSITE_EXPERT_CACHE_PAYLOAD,
                    DS4_RUNTIME_CATEGORY_EXPERT_CACHE_PAYLOAD,
                    DS4_RUNTIME_DOMAIN_CUDA_DEVICE)) {
                return false;
            }
            cache_payload_count++;
        } else if (record->callsite_id ==
                       DS4_LAGUNA_CALLSITE_ROUTE_HOTNESS) {
            if (record->base == 0 || !owned_record_matches(
                    record, record->base, route_hotness_bytes,
                    DS4_LAGUNA_CALLSITE_ROUTE_HOTNESS,
                    DS4_RUNTIME_CATEGORY_CACHE_METADATA_ADDRESS_TABLES,
                    DS4_RUNTIME_DOMAIN_HOST)) {
                return false;
            }
            route_hotness_count++;
        } else if (record->callsite_id ==
                       DS4_LAGUNA_CALLSITE_HOST_ENTRY_TO_SLOT) {
            if (record->base == 0 || !owned_record_matches(
                    record, record->base, entry_to_slot_bytes,
                    DS4_LAGUNA_CALLSITE_HOST_ENTRY_TO_SLOT,
                    DS4_RUNTIME_CATEGORY_CACHE_METADATA_ADDRESS_TABLES,
                    DS4_RUNTIME_DOMAIN_HOST)) {
                return false;
            }
            host_entry_to_slot_count++;
        } else if (record->callsite_id ==
                       DS4_LAGUNA_CALLSITE_DEVICE_ENTRY_TO_SLOT) {
            if (!owned_record_matches(
                    record,
                    (uint64_t)(uintptr_t)compact->device_entry_to_slot,
                    entry_to_slot_bytes,
                    DS4_LAGUNA_CALLSITE_DEVICE_ENTRY_TO_SLOT,
                    DS4_RUNTIME_CATEGORY_CACHE_METADATA_ADDRESS_TABLES,
                    DS4_RUNTIME_DOMAIN_CUDA_DEVICE)) {
                return false;
            }
            device_entry_to_slot_count++;
        } else if (record->callsite_id ==
                       DS4_LAGUNA_CALLSITE_SLOT_STATE) {
            if (!owned_record_matches(
                    record, (uint64_t)(uintptr_t)compact->cache_slots,
                    slot_state_bytes,
                    DS4_LAGUNA_CALLSITE_SLOT_STATE,
                    DS4_RUNTIME_CATEGORY_CACHE_METADATA_ADDRESS_TABLES,
                    DS4_RUNTIME_DOMAIN_HOST)) {
                return false;
            }
            slot_state_count++;
        } else if (record->callsite_id ==
                       DS4_LAGUNA_CALLSITE_OTHER_HOST_TRACKER) {
            if (!compact->page_advice_state_live ||
                !owned_record_matches(
                    record,
                    (uint64_t)(uintptr_t)compact->page_advice_state,
                    compact->page_advice_state_bytes,
                    DS4_LAGUNA_CALLSITE_OTHER_HOST_TRACKER,
                    DS4_RUNTIME_CATEGORY_OTHER_HOST,
                    DS4_RUNTIME_DOMAIN_HOST)) {
                return false;
            }
            page_advice_state_count++;
        } else if (record->callsite_id >=
                       DS4_LAGUNA_CALLSITE_PINNED_STAGING_0 &&
                   record->callsite_id <=
                       DS4_LAGUNA_CALLSITE_PINNED_STAGING_3) {
            const size_t staging_index = record->callsite_id -
                DS4_LAGUNA_CALLSITE_PINNED_STAGING_0;
            if (staging_index >= ARRAY_LEN(staging_seen) ||
                staging_seen[staging_index] || record->base == 0 ||
                !owned_record_matches(
                    record, record->base,
                    compact->cache_slot_stride_bytes,
                    record->callsite_id,
                    DS4_RUNTIME_CATEGORY_PINNED_STAGING,
                    DS4_RUNTIME_DOMAIN_HOST)) {
                return false;
            }
            staging_seen[staging_index] = true;
        } else if (record->relation ==
                       DS4_RUNTIME_RELATION_MODEL_MAPPING) {
            if (record->base != (uint64_t)(uintptr_t)compact->model_map ||
                record->requested_bytes != compact->model_size ||
                record->charged_bytes != 0 || record->callsite_id != 0 ||
                record->category !=
                    (ds4_runtime_category)DS4_RUNTIME_OWNED_CATEGORY_COUNT ||
                record->domain != DS4_RUNTIME_DOMAIN_HOST ||
                record->owner_id != 0) {
                return false;
            }
            mapping_count++;
        } else if (record->callsite_id ==
                       DS4_LAGUNA_CALLSITE_OTHER_HOST_ENGINE ||
                   record->callsite_id ==
                       DS4_LAGUNA_CALLSITE_OTHER_HOST_MODEL ||
                   record->callsite_id ==
                       DS4_LAGUNA_CALLSITE_OTHER_HOST_VOCAB) {
            inventory_count++;
        } else {
            return false;
        }
        for (size_t j = 0; j < i; j++) {
            if (records[j].id == record->id) return false;
        }
    }
    if (ledger_count != 3 || static_count != 1 || offsets_count != 1 ||
        mapping_count != 1 || inventory_count != PINNED_OWNER_COUNT ||
        cache_payload_count != 1 || route_hotness_count != 1 ||
        host_entry_to_slot_count != 1 ||
        device_entry_to_slot_count != 1 || slot_state_count != 1 ||
        page_advice_state_count != 1) {
        return false;
    }
    for (size_t i = 0; i < ARRAY_LEN(staging_seen); i++) {
        if (!staging_seen[i]) return false;
    }
    uint64_t metadata_bytes = 0;
    uint64_t expected_owned = 0;
    if (!checked_add_u64_test(
            ledger_bytes, compact->static_offset_bytes, &metadata_bytes) ||
        !checked_add_u64_test(
            metadata_bytes, route_hotness_bytes, &metadata_bytes) ||
        !checked_add_u64_test(
            metadata_bytes, entry_to_slot_bytes, &metadata_bytes) ||
        !checked_add_u64_test(
            metadata_bytes, entry_to_slot_bytes, &metadata_bytes) ||
        !checked_add_u64_test(
            metadata_bytes, slot_state_bytes, &metadata_bytes) ||
        !checked_add_u64_test(
            compact->static_slab_bytes, metadata_bytes, &expected_owned) ||
        !checked_add_u64_test(
            expected_owned, compact->cache_payload_bytes,
            &expected_owned) ||
        !checked_add_u64_test(
            expected_owned, compact->pinned_staging_bytes,
            &expected_owned) ||
        !checked_add_u64_test(
            expected_owned, compact->page_advice_state_bytes,
            &expected_owned) ||
        !checked_add_u64_test(
            expected_owned, owner_bytes, &expected_owned) ||
        charged_bytes != expected_owned) {
        return false;
    }
    for (size_t i = 0; i < DS4_RUNTIME_OWNED_CATEGORY_COUNT; i++) {
        uint64_t expected = 0;
        if (i == DS4_RUNTIME_CATEGORY_STATIC_WEIGHTS) {
            expected = compact->static_slab_bytes;
        } else if (i ==
                   DS4_RUNTIME_CATEGORY_EXPERT_CACHE_PAYLOAD) {
            expected = compact->cache_payload_bytes;
        } else if (i ==
                   DS4_RUNTIME_CATEGORY_CACHE_METADATA_ADDRESS_TABLES) {
            expected = metadata_bytes;
        } else if (i == DS4_RUNTIME_CATEGORY_PINNED_STAGING) {
            expected = compact->pinned_staging_bytes;
        } else if (i == DS4_RUNTIME_CATEGORY_OTHER_HOST) {
            if (!checked_add_u64_test(
                    owner_bytes, compact->page_advice_state_bytes,
                    &expected)) {
                return false;
            }
        }
        if (runtime->category_current[i] != expected ||
            runtime->category_peak[i] != expected ||
            runtime->category_current[i] > runtime->category_bounds[i]) {
            return false;
        }
    }
    for (size_t i = 0; i < DS4_RUNTIME_REPORT_COUNT; i++) {
        if (i == DS4_RUNTIME_REPORT_MODEL_SOURCE_RESIDENT) {
            if (runtime->report_current[i] !=
                    compact->page_advice_post_source_resident_bytes ||
                runtime->report_peak[i] <
                    compact->page_advice_precharge_source_resident_bytes ||
                runtime->report_peak[i] < runtime->report_current[i] ||
                runtime->report_peak[i] > runtime->report_bounds[i]) {
                return false;
            }
            continue;
        }
        const uint64_t expected = i ==
                DS4_RUNTIME_REPORT_MODEL_MAPPED_VIRTUAL ?
            compact->model_size : 0;
        if (runtime->report_current[i] != expected ||
            runtime->report_peak[i] != expected ||
            runtime->report_current[i] > runtime->report_bounds[i]) {
            return false;
        }
    }
    const uint64_t source_current = runtime->report_current[
        DS4_RUNTIME_REPORT_MODEL_SOURCE_RESIDENT];
    const uint64_t source_peak = runtime->report_peak[
        DS4_RUNTIME_REPORT_MODEL_SOURCE_RESIDENT];
    uint64_t expected_qualification_current = 0;
    uint64_t expected_qualification_peak = 0;
    if (!checked_add_u64_test(
            expected_owned, source_current,
            &expected_qualification_current) ||
        !checked_add_u64_test(
            expected_owned, source_peak,
            &expected_qualification_peak)) {
        return false;
    }
    return runtime->owned_total_current == expected_owned &&
        runtime->owned_total_peak == expected_owned &&
        runtime->qualification_total_current ==
            expected_qualification_current &&
        runtime->qualification_total_peak == expected_qualification_peak &&
        runtime->owned_total_current <= runtime->owned_total_bound_bytes &&
        runtime->qualification_total_current <=
            runtime->qualification_total_bound_bytes;
}

static bool runtime_snapshot_is_clean(
        const ds4_runtime_snapshot *snapshot) {
    if (!snapshot ||
        snapshot->violation != DS4_RUNTIME_VIOLATION_NONE ||
        snapshot->active_record_count != 0 ||
        snapshot->owned_total_current != 0 ||
        snapshot->qualification_total_current != 0) {
        return false;
    }
    for (size_t i = 0; i < DS4_RUNTIME_OWNED_CATEGORY_COUNT; i++) {
        if (snapshot->category_current[i] != 0) return false;
    }
    for (size_t i = 0; i < DS4_RUNTIME_REPORT_COUNT; i++) {
        if (snapshot->report_current[i] != 0) return false;
    }
    return true;
}

static bool retained_tracker_matches_compact(
        const tracker_fixture *fixture,
        const ds4_laguna_ledger *ledger,
        const ds4_gpu_laguna_compact_test_snapshot *compact,
        const ds4_runtime_snapshot *runtime,
        const ds4_runtime_allocation_record *records) {
    if (!fixture || !ledger || !compact || !runtime || !records ||
        runtime->violation != DS4_RUNTIME_VIOLATION_NONE ||
        runtime->active_record_count != 6 ||
        compact->static_slab_bytes != 512 ||
        compact->static_offset_bytes !=
            FIXTURE_TENSOR_COUNT * sizeof(uint64_t) ||
        compact->model_size != FIXTURE_MODEL_BYTES) {
        return false;
    }

    const uint64_t source_capacity =
        ledger->tensor_range_count * 2u + 5u;
    const uint64_t ledger_bytes[3] = {
        ledger->tensor_range_count * sizeof(ledger->tensor_ranges[0]),
        source_capacity * sizeof(ledger->source_ranges[0]),
        ledger->expert_entry_count * sizeof(ledger->expert_entries[0]),
    };
    const uint64_t ledger_bases[3] = {
        (uint64_t)(uintptr_t)ledger->tensor_ranges,
        (uint64_t)(uintptr_t)ledger->source_ranges,
        (uint64_t)(uintptr_t)ledger->expert_entries,
    };
    bool ledger_seen[3] = {false, false, false};
    bool static_seen = false;
    bool offsets_seen = false;
    bool mapping_seen = false;
    uint64_t static_id = 0;
    uint64_t offsets_id = 0;
    uint64_t mapping_id = 0;

    for (size_t i = 0; i < runtime->active_record_count; i++) {
        const ds4_runtime_allocation_record *record = &records[i];
        bool matched = false;
        for (size_t j = 0; j < ARRAY_LEN(ledger_seen); j++) {
            if (record->id != fixture->ledger_ids[j]) continue;
            if (ledger_seen[j] ||
                !owned_record_matches(
                    record, ledger_bases[j], ledger_bytes[j],
                    DS4_LAGUNA_CALLSITE_LEDGER_ARRAYS,
                    DS4_RUNTIME_CATEGORY_CACHE_METADATA_ADDRESS_TABLES,
                    DS4_RUNTIME_DOMAIN_HOST)) {
                return false;
            }
            ledger_seen[j] = true;
            matched = true;
            break;
        }
        if (matched) continue;

        if (record->relation == DS4_RUNTIME_RELATION_MODEL_MAPPING) {
            if (mapping_seen || !record->live ||
                record->base != (uint64_t)(uintptr_t)compact->model_map ||
                record->requested_bytes != compact->model_size ||
                record->charged_bytes != 0 ||
                record->category !=
                    (ds4_runtime_category)DS4_RUNTIME_OWNED_CATEGORY_COUNT ||
                record->domain != DS4_RUNTIME_DOMAIN_HOST ||
                record->callsite_id != 0 || record->owner_id != 0) {
                return false;
            }
            mapping_seen = true;
            mapping_id = record->id;
        } else if (record->callsite_id ==
                   DS4_LAGUNA_CALLSITE_STATIC_SLAB) {
            if (static_seen ||
                !owned_record_matches(
                    record,
                    (uint64_t)(uintptr_t)compact->static_slab,
                    512,
                    DS4_LAGUNA_CALLSITE_STATIC_SLAB,
                    DS4_RUNTIME_CATEGORY_STATIC_WEIGHTS,
                    DS4_RUNTIME_DOMAIN_CUDA_DEVICE)) {
                return false;
            }
            static_seen = true;
            static_id = record->id;
        } else if (record->callsite_id ==
                   DS4_LAGUNA_CALLSITE_STATIC_OFFSETS) {
            if (offsets_seen ||
                !owned_record_matches(
                    record,
                    (uint64_t)(uintptr_t)compact->static_offsets,
                    FIXTURE_TENSOR_COUNT * sizeof(uint64_t),
                    DS4_LAGUNA_CALLSITE_STATIC_OFFSETS,
                    DS4_RUNTIME_CATEGORY_CACHE_METADATA_ADDRESS_TABLES,
                    DS4_RUNTIME_DOMAIN_HOST)) {
                return false;
            }
            offsets_seen = true;
            offsets_id = record->id;
        } else {
            return false;
        }
    }

    if (!ledger_seen[0] || !ledger_seen[1] || !ledger_seen[2] ||
        !static_seen || !offsets_seen || !mapping_seen) {
        return false;
    }
    const uint64_t sequence_mask = UINT64_C(0x00ffffffffffffff);
    if ((uint8_t)(offsets_id >> 56) != 0x4du ||
        (offsets_id & sequence_mask) == 0 ||
        (offsets_id & sequence_mask) > sequence_mask - 2u ||
        static_id != offsets_id + 1u || mapping_id != offsets_id + 2u) {
        return false;
    }
    const uint64_t ledger_total =
        ledger_bytes[0] + ledger_bytes[1] + ledger_bytes[2];
    const uint64_t metadata_total =
        ledger_total + FIXTURE_TENSOR_COUNT * sizeof(uint64_t);
    const uint64_t owned_total = metadata_total + 512u;
    for (size_t i = 0; i < DS4_RUNTIME_OWNED_CATEGORY_COUNT; i++) {
        const uint64_t expected =
            i == DS4_RUNTIME_CATEGORY_STATIC_WEIGHTS ? 512u :
            i == DS4_RUNTIME_CATEGORY_CACHE_METADATA_ADDRESS_TABLES ?
                metadata_total : 0;
        if (runtime->category_current[i] != expected) return false;
    }
    for (size_t i = 0; i < DS4_RUNTIME_REPORT_COUNT; i++) {
        const uint64_t expected =
            i == DS4_RUNTIME_REPORT_MODEL_MAPPED_VIRTUAL ?
                FIXTURE_MODEL_BYTES : 0;
        if (runtime->report_current[i] != expected) return false;
    }
    return runtime->owned_total_current == owned_total &&
        runtime->qualification_total_current == owned_total;
}

static void expect_only_compact_rejection(
        const ds4_gpu_laguna_compact *context,
        const ds4_gpu_laguna_compact_test_snapshot *baseline,
        uint64_t expected_rejection_count,
        const ds4_runtime_tracker *tracker,
        const ds4_runtime_snapshot *runtime_baseline,
        const ds4_runtime_allocation_record *records_baseline,
        const char *message) {
    ds4_gpu_laguna_compact_test_snapshot current;
    ds4_runtime_snapshot runtime_current;
    ds4_runtime_allocation_record records_current[TRACKER_RECORD_CAPACITY];
    memset(&current, 0, sizeof(current));
    const bool compact_captured =
        ds4_gpu_test_laguna_compact_snapshot(context, &current) != 0;
    CHECK(compact_captured &&
              current.rejection_count == expected_rejection_count,
          message);
    if (compact_captured) {
        ds4_gpu_laguna_compact_test_snapshot normalized = current;
        normalized.rejection_count = baseline->rejection_count;
        CHECK(memcmp(&normalized, baseline, sizeof(normalized)) == 0,
              "compact rejection mutates no other compact state or counter");
    }
    CHECK(capture_tracker_snapshot(
              tracker, &runtime_current, records_current) &&
              tracker_snapshots_equal(
                  runtime_baseline, records_baseline,
                  &runtime_current, records_current),
          "compact rejection leaves runtime tracker byte-identical");
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
        const ds4_laguna_file_identity *model_identity,
        const ds4_laguna_ledger *ledger,
        const ds4_laguna_allocation_plan *plan,
        const char *message) {
    tracker_fixture runtime;
    CHECK(tracker_fixture_init(&runtime, plan, ledger),
          "invalid-ledger tracker initializes");
    const uint64_t attempts_before =
        ds4_gpu_test_laguna_compact_static_allocation_attempts();
    const int fds_before = open_fd_count();
    ds4_gpu_laguna_compact *context = NULL;
    CHECK(!ds4_gpu_laguna_compact_create(
              &context, model_fd, model_map, model_size,
              model_identity, ledger, plan, &runtime.tracker) &&
              context == NULL,
          message);
    CHECK(ds4_gpu_test_laguna_compact_static_allocation_attempts() ==
              attempts_before,
          "invalid ledger is rejected before a static CUDA allocation attempt");
    CHECK(tracker_has_only_ledger(&runtime.tracker),
          "invalid ledger leaves only its live ledger records");
    ds4_gpu_laguna_compact_destroy(context);
    CHECK(fds_before >= 0 && open_fd_count() == fds_before,
          "invalid ledger closes its validation duplicate immediately");
}

static void expect_invalid_model_source_before_allocation(
        int model_fd,
        const void *model_map,
        uint64_t model_size,
        const ds4_laguna_file_identity *model_identity,
        const ds4_laguna_ledger *ledger,
        const ds4_laguna_allocation_plan *plan,
        const char *message) {
    tracker_fixture runtime;
    CHECK(tracker_fixture_init(&runtime, plan, ledger),
          "invalid-source tracker initializes");
    const uint64_t attempts_before =
        ds4_gpu_test_laguna_compact_static_allocation_attempts();
    const int fds_before = open_fd_count();
    ds4_gpu_laguna_compact *context = NULL;
    const int created = ds4_gpu_laguna_compact_create(
        &context, model_fd, model_map, model_size, model_identity,
        ledger, plan, &runtime.tracker);
    CHECK(!created && context == NULL, message);
    CHECK(ds4_gpu_test_laguna_compact_static_allocation_attempts() ==
              attempts_before,
          "invalid model source fails before a static CUDA allocation attempt");
    if (context) {
        CHECK(ds4_gpu_laguna_compact_destroy(context) ==
                  DS4_GPU_LAGUNA_DESTROY_OK,
              "unexpected invalid-source context is contained before continuing");
    }
    const int fds_after = open_fd_count();
    CHECK(fds_before >= 0 && fds_after == fds_before,
          "invalid model source closes its validation duplicate immediately");
    CHECK(tracker_has_only_ledger(&runtime.tracker),
          "invalid model source leaves only its live ledger records");
    CHECK(tracker_fixture_release_ledger(&runtime),
          "invalid-source ledger records release");
}

static void exercise_invalid_model_sources(void) {
    const long page_value = sysconf(_SC_PAGESIZE);
    CHECK(page_value > 0, "host page size is available for VMA tests");
    if (page_value <= 0) return;
    const uint64_t page = (uint64_t)page_value;
    const uint64_t model_size = page * 3u;
    char path_a[] = "/tmp/ds4-laguna-source-a.XXXXXX";
    char path_b[] = "/tmp/ds4-laguna-source-b.XXXXXX";
    char path_truncated[] = "/tmp/ds4-laguna-source-truncated.XXXXXX";
    int fd_a = create_pattern_file(path_a, model_size, 11u);
    int fd_b = create_pattern_file(path_b, model_size, 173u);
    int fd_truncated =
        create_pattern_file(path_truncated, model_size, 91u);
    CHECK(fd_a >= 0 && fd_b >= 0 && fd_truncated >= 0,
          "source-validation files open with exact multi-page sizes");
    if (fd_a < 0 || fd_b < 0 || fd_truncated < 0) goto cleanup_files;

    ds4_laguna_file_identity identity_a;
    ds4_laguna_file_identity identity_truncated;
    const bool identities_ready =
        capture_file_identity(fd_a, &identity_a) &&
        capture_file_identity(fd_truncated, &identity_truncated);
    CHECK(identities_ready,
          "source-validation identities are captured after fsync");
    if (!identities_ready) goto cleanup_files;
    ds4_laguna_ledger ledger;
    memset(&ledger, 0, sizeof(ledger));
    CHECK(ledger_fixture_build_for_size(&ledger, model_size),
          "multi-page source-validation ledger builds");
    if (!ledger.tensor_ranges) goto cleanup_files;
    ds4_laguna_allocation_plan plan;
    plan_prepare(&plan, &ledger);

    unsigned char *map_a = mmap(
        NULL, (size_t)model_size, PROT_READ, MAP_SHARED, fd_a, 0);
    unsigned char *map_b = mmap(
        NULL, (size_t)model_size, PROT_READ, MAP_SHARED, fd_b, 0);
    CHECK(map_a != MAP_FAILED && map_b != MAP_FAILED,
          "distinct source-validation file mappings open");
    if (map_a != MAP_FAILED && map_b != MAP_FAILED) {
        ds4_laguna_file_identity mtime_mismatch = identity_a;
        mtime_mismatch.mtime_ns ^= UINT64_C(1);
        expect_invalid_model_source_before_allocation(
            fd_a, map_a, model_size, &mtime_mismatch, &ledger, &plan,
            "mtime-only expected identity mismatch rejects the exact source");
        expect_invalid_model_source_before_allocation(
            fd_a, map_b, model_size, &identity_a, &ledger, &plan,
            "fd A with mmap B is rejected before CUDA allocation");
        expect_invalid_model_source_before_allocation(
            fd_b, map_b, model_size, &identity_a, &ledger, &plan,
            "same-size fd B cannot satisfy expected identity A");
    }

    unsigned char *anonymous = mmap(
        NULL, (size_t)model_size, PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    CHECK(anonymous != MAP_FAILED,
          "anonymous source-validation mapping opens");
    if (anonymous != MAP_FAILED) {
        const ssize_t copied = pread(fd_a, anonymous, (size_t)model_size, 0);
        CHECK(copied == (ssize_t)model_size,
              "anonymous mapping receives exact model bytes");
        if (copied == (ssize_t)model_size) {
            CHECK(mprotect(anonymous, (size_t)model_size, PROT_READ) == 0,
                  "anonymous source-validation mapping becomes read-only");
            expect_invalid_model_source_before_allocation(
                fd_a, anonymous, model_size, &identity_a, &ledger, &plan,
                "anonymous mapping cannot impersonate the opened model file");
        }
        munmap(anonymous, (size_t)model_size);
    }

    unsigned char *protected_map = mmap(
        NULL, (size_t)model_size, PROT_READ, MAP_SHARED, fd_a, 0);
    CHECK(protected_map != MAP_FAILED,
          "partial-PROT_NONE source mapping opens");
    if (protected_map != MAP_FAILED) {
        CHECK(mprotect(protected_map + page, (size_t)page, PROT_NONE) == 0,
              "middle source VMA can be made unreadable");
        expect_invalid_model_source_before_allocation(
            fd_a, protected_map, model_size, &identity_a, &ledger, &plan,
            "PROT_NONE segment rejects the complete source mapping");
        munmap(protected_map, (size_t)model_size);
    }

    unsigned char *mixed = mmap(
        NULL, (size_t)model_size, PROT_NONE,
        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    CHECK(mixed != MAP_FAILED, "mixed-inode address range reserves");
    if (mixed != MAP_FAILED) {
        void *first = mmap(mixed, (size_t)page, PROT_READ,
                           MAP_SHARED | MAP_FIXED, fd_a, 0);
        void *rest = mmap(mixed + page, (size_t)(model_size - page),
                          PROT_READ, MAP_SHARED | MAP_FIXED, fd_b,
                          (off_t)page);
        CHECK(first == mixed && rest == mixed + page,
              "adjacent mixed-inode VMAs occupy one requested range");
        if (first == mixed && rest == mixed + page) {
            expect_invalid_model_source_before_allocation(
                fd_a, mixed, model_size, &identity_a, &ledger, &plan,
                "adjacent VMAs from mixed inodes reject the model mapping");
        }
        munmap(mixed, (size_t)model_size);
    }

    unsigned char *split = mmap(
        NULL, (size_t)model_size, PROT_NONE,
        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    CHECK(split != MAP_FAILED,
          "same-inode split-VMA address range reserves");
    if (split != MAP_FAILED) {
        void *first = mmap(split, (size_t)page, PROT_READ,
                           MAP_SHARED | MAP_FIXED, fd_a, 0);
        void *rest = mmap(split + page, (size_t)(model_size - page),
                          PROT_READ, MAP_PRIVATE | MAP_FIXED, fd_a,
                          (off_t)page);
        const bool split_mapped =
            first == split && rest == split + page;
        CHECK(split_mapped,
              "same-inode exact-offset mappings occupy one contiguous range");
        unsigned readable_entries = 0;
        const bool split_covered = split_mapped &&
            proc_maps_has_readable_split_cover(
                split, (size_t)model_size, &readable_entries);
        CHECK(split_covered && readable_entries >= 2,
              "/proc/self/maps confirms two readable VMAs cover the exact range");
        if (split_covered) {
            tracker_fixture split_runtime;
            const bool tracker_ready =
                tracker_fixture_init(&split_runtime, &plan, &ledger);
            CHECK(tracker_ready,
                  "same-inode split-VMA tracker initializes");
            if (tracker_ready) {
                const uint64_t attempts_before =
                    ds4_gpu_test_laguna_compact_static_allocation_attempts();
                ds4_gpu_laguna_compact *split_context = NULL;
                const int created = ds4_gpu_laguna_compact_create(
                    &split_context, fd_a, split, model_size,
                    &identity_a, &ledger, &plan, &split_runtime.tracker);
                CHECK(created && split_context != NULL,
                      "same-inode contiguous split VMAs form a valid source");
                CHECK(ds4_gpu_test_laguna_compact_static_allocation_attempts() ==
                          attempts_before + 1u,
                      "split-VMA source performs exactly one slab allocation");
                ds4_gpu_laguna_compact_test_snapshot split_snapshot;
                memset(&split_snapshot, 0, sizeof(split_snapshot));
                CHECK(split_context &&
                          ds4_gpu_test_laguna_compact_snapshot(
                              split_context, &split_snapshot) &&
                          split_snapshot.model_map == split &&
                          split_snapshot.model_size == model_size &&
                          identities_equal(
                              &split_snapshot.model_identity, &identity_a),
                      "split-VMA context binds the supplied map and identity");
                CHECK(ds4_gpu_laguna_compact_destroy(split_context) ==
                          DS4_GPU_LAGUNA_DESTROY_OK,
                      "split-VMA context reports typed teardown success");
                CHECK(tracker_has_only_ledger(&split_runtime.tracker),
                      "split-VMA teardown restores ledger-only tracking");
                CHECK(tracker_fixture_release_ledger(&split_runtime),
                      "split-VMA ledger records release before unmap");
            }
        }
        munmap(split, (size_t)model_size);
    }

    unsigned char *gap = mmap(
        NULL, (size_t)model_size, PROT_READ, MAP_SHARED, fd_a, 0);
    CHECK(gap != MAP_FAILED, "gapped source mapping opens");
    if (gap != MAP_FAILED) {
        CHECK(munmap(gap + page, (size_t)page) == 0,
              "middle page can be removed from source mapping");
        expect_invalid_model_source_before_allocation(
            fd_a, gap, model_size, &identity_a, &ledger, &plan,
            "a VMA gap rejects the complete source mapping");
        munmap(gap, (size_t)page);
        munmap(gap + 2u * page, (size_t)page);
    }

    unsigned char *shifted = mmap(
        NULL, (size_t)model_size, PROT_READ, MAP_SHARED, fd_a,
        (off_t)page);
    CHECK(shifted != MAP_FAILED, "shifted-offset source mapping opens");
    if (shifted != MAP_FAILED) {
        expect_invalid_model_source_before_allocation(
            fd_a, shifted, model_size, &identity_a, &ledger, &plan,
            "shifted file offset rejects the model mapping");
        munmap(shifted, (size_t)model_size);
    }

    unsigned char *truncated_map = mmap(
        NULL, (size_t)model_size, PROT_READ, MAP_SHARED, fd_truncated, 0);
    CHECK(truncated_map != MAP_FAILED,
          "post-map truncation source mapping opens");
    if (truncated_map != MAP_FAILED) {
        CHECK(ftruncate(fd_truncated, (off_t)(model_size - page)) == 0,
              "source file truncates after its mapping and identity capture");
        expect_invalid_model_source_before_allocation(
            fd_truncated, truncated_map, model_size, &identity_truncated,
            &ledger, &plan,
            "post-map file truncation rejects stale descriptor identity");
        munmap(truncated_map, (size_t)model_size);
    }

    if (map_a != MAP_FAILED) munmap(map_a, (size_t)model_size);
    if (map_b != MAP_FAILED) munmap(map_b, (size_t)model_size);
    ds4_laguna_ledger_free(&ledger);

cleanup_files:
    if (fd_a >= 0) close(fd_a);
    if (fd_b >= 0) close(fd_b);
    if (fd_truncated >= 0) close(fd_truncated);
    unlink(path_a);
    unlink(path_b);
    unlink(path_truncated);
}

static int run_startup(void) {
    int result = 1;
    int fd = -1;
    int fixture_fd = -1;
    int other_fd = -1;
    int reused_fd = -1;
    unsigned char *model_map = MAP_FAILED;
    unsigned char *alias_map = MAP_FAILED;
    unsigned char *other_map = MAP_FAILED;
    ds4_laguna_ledger ledger;
    ds4_laguna_allocation_plan plan;
    ds4_laguna_file_identity model_identity;
    ds4_gpu_laguna_compact *context = NULL;
    saved_environment saved;
    char path[] = "/tmp/ds4-cuda-laguna-startup.XXXXXX";
    char other_path[] = "/tmp/ds4-cuda-laguna-other.XXXXXX";
    char reuse_path[] = "/tmp/ds4-cuda-laguna-reuse.XXXXXX";
    memset(&ledger, 0, sizeof(ledger));
    memset(&plan, 0, sizeof(plan));
    memset(&model_identity, 0, sizeof(model_identity));
    save_and_clear_forbidden_environment(&saved);

    int device_count = 0;
    CHECK(cudaGetDeviceCount(&device_count) == cudaSuccess && device_count >= 1,
          "startup test has one visible CUDA device");
    if (device_count < 1) goto cleanup;
    CHECK(ds4_gpu_init() != 0, "CUDA backend initializes");
    exercise_invalid_model_sources();

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
    CHECK(fsync(fd) == 0 && capture_file_identity(fd, &model_identity),
          "synthetic model identity is captured after its bytes are durable");
    fixture_fd = dup(fd);
    CHECK(fixture_fd >= 0, "stable fixture descriptor duplicates caller fd");
    model_map = mmap(NULL, sizeof(model_bytes), PROT_READ, MAP_SHARED, fd, 0);
    CHECK(model_map != MAP_FAILED, "synthetic model mapping opens read-only");
    if (model_map == MAP_FAILED) goto cleanup_file;
    alias_map = mmap(NULL, sizeof(model_bytes), PROT_READ, MAP_SHARED, fd, 0);
    CHECK(alias_map != MAP_FAILED,
          "same-inode alias mapping opens at a distinct address");
    other_fd = create_pattern_file(
        other_path, sizeof(model_bytes), 173u);
    CHECK(other_fd >= 0, "different-inode synthetic model opens");
    if (other_fd >= 0) {
        other_map = mmap(NULL, sizeof(model_bytes), PROT_READ,
                         MAP_SHARED, other_fd, 0);
    }
    CHECK(other_map != MAP_FAILED,
          "different-inode second model mapping opens");
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
            fd, model_map, sizeof(model_bytes), &model_identity,
            &bad, &plan,
            "overlapping parent ranges fail closed");
        free(bad.tensor_ranges);
    }

    bad = ledger_with_copied_ranges(&ledger);
    CHECK(bad.tensor_ranges != NULL, "truncation mutation has private ranges");
    if (bad.tensor_ranges) {
        bad.tensor_ranges[7].source_offset = sizeof(model_bytes) - 1u;
        expect_invalid_ledger_before_allocation(
            fd, model_map, sizeof(model_bytes), &model_identity,
            &bad, &plan,
            "truncated parent range fails closed");
        free(bad.tensor_ranges);
    }

    bad = ledger_with_copied_ranges(&ledger);
    CHECK(bad.tensor_ranges != NULL,
          "misclassification mutation has private ranges");
    if (bad.tensor_ranges) {
        bad.tensor_ranges[2].tensor_class = DS4_LAGUNA_TENSOR_STATIC;
        expect_invalid_ledger_before_allocation(
            fd, model_map, sizeof(model_bytes), &model_identity,
            &bad, &plan,
            "routed/static misclassification fails closed");
        free(bad.tensor_ranges);
    }

    ds4_laguna_allocation_plan bad_plan = plan;
    bad_plan.owned_category_bounds[DS4_RUNTIME_CATEGORY_STATIC_WEIGHTS]--;
    bad_plan.owned_total_bound_bytes--;
    bad_plan.qualification_total_bound_bytes--;
    bad_plan.callsites[0].bound_bytes--;
    expect_invalid_ledger_before_allocation(
        fd, model_map, sizeof(model_bytes), &model_identity,
        &ledger, &bad_plan,
        "truncated static allocation plan fails closed");

    const char *const present_values[] = {"", "0", "1"};
    for (size_t i = 0; i < ARRAY_LEN(forbidden_cuda_env); i++) {
        for (size_t j = 0; j < ARRAY_LEN(present_values); j++) {
            tracker_fixture runtime;
            CHECK(tracker_fixture_init(&runtime, &plan, &ledger),
                  "forbidden-environment tracker initializes");
            const uint64_t attempts_before =
                ds4_gpu_test_laguna_compact_static_allocation_attempts();
            CHECK(setenv(forbidden_cuda_env[i], present_values[j], 1) == 0,
                  "forbidden CUDA environment can be injected by presence");
            ds4_gpu_laguna_compact *rejected = NULL;
            const int created = ds4_gpu_laguna_compact_create(
                &rejected, fd, model_map, sizeof(model_bytes),
                &model_identity, &ledger, &plan, &runtime.tracker);
            if (created || rejected) {
                fprintf(stderr, "unexpected acceptance: %s=%s\n",
                        forbidden_cuda_env[i], present_values[j]);
            }
            CHECK(!created && rejected == NULL,
                  "unsafe present CUDA model/cache option is rejected");
            CHECK(ds4_gpu_test_laguna_compact_static_allocation_attempts() ==
                      attempts_before,
                  "unsafe CUDA option fails before static CUDA allocation");
            CHECK(tracker_has_only_ledger(&runtime.tracker),
                  "unsafe CUDA option leaves only live ledger records");
            ds4_gpu_laguna_compact_destroy(rejected);
            CHECK(tracker_fixture_release_ledger(&runtime),
                  "forbidden-environment ledger records release");
            unsetenv(forbidden_cuda_env[i]);
        }
    }

    tracker_fixture creating_runtime;
    ds4_runtime_snapshot creating_tracker_baseline;
    ds4_runtime_allocation_record
        creating_records_baseline[TRACKER_RECORD_CAPACITY];
    const bool creating_tracker_ready =
        tracker_fixture_init(&creating_runtime, &plan, &ledger) &&
        capture_tracker_snapshot(
            &creating_runtime.tracker,
            &creating_tracker_baseline,
            creating_records_baseline);
    CHECK(creating_tracker_ready,
          "CREATING legacy-rejection tracker baseline captures");
    creator_race creating_creator = {
        .model_fd = fixture_fd,
        .model_map = model_map,
        .model_size = sizeof(model_bytes),
        .model_identity = &model_identity,
        .ledger = &ledger,
        .plan = &plan,
        .runtime = &creating_runtime,
    };
    pthread_t creating_thread;
    bool creating_thread_started = false;
    if (creating_tracker_ready) {
        ds4_gpu_test_laguna_compact_pause_creating_once();
        creating_thread_started = pthread_create(
            &creating_thread, NULL,
            direct_creator_run, &creating_creator) == 0;
    }
    CHECK(creating_thread_started,
          "direct compact creator launches for a deterministic CREATING probe");
    if (creating_thread_started) {
        ds4_gpu_test_laguna_compact_wait_creating_paused();
        int creating_probe_result = -1;
        const bool copy_model_set =
            setenv("DS4_CUDA_COPY_MODEL", "1", 1) == 0;
        if (copy_model_set) {
            creating_probe_result = ds4_gpu_set_model_map(
                alias_map, sizeof(model_bytes));
        }
        unsetenv("DS4_CUDA_COPY_MODEL");
        CHECK(copy_model_set && creating_probe_result == 0,
              "CREATING rejects canonical legacy model placement");

        ds4_runtime_snapshot creating_tracker_paused;
        ds4_runtime_allocation_record
            creating_records_paused[TRACKER_RECORD_CAPACITY];
        CHECK(capture_tracker_snapshot(
                  &creating_runtime.tracker,
                  &creating_tracker_paused,
                  creating_records_paused) &&
                  tracker_snapshots_equal(
                      &creating_tracker_baseline,
                      creating_records_baseline,
                      &creating_tracker_paused,
                      creating_records_paused),
              "CREATING rejection leaves its tracker byte-identical while paused");

        ds4_gpu_test_laguna_compact_resume_creating();
        const int creating_join_result =
            pthread_join(creating_thread, NULL);
        CHECK(creating_join_result == 0,
              "paused direct compact creator resumes and joins");
        CHECK(creating_creator.created && creating_creator.context != NULL,
              "creator publishes after the CREATING rejection probe");

        ds4_gpu_laguna_compact_test_snapshot creating_active;
        memset(&creating_active, 0, sizeof(creating_active));
        CHECK(creating_creator.context &&
                  ds4_gpu_test_laguna_compact_snapshot(
                      creating_creator.context, &creating_active) &&
                  creating_active.lifecycle ==
                      DS4_GPU_LAGUNA_LIFECYCLE_ACTIVE &&
                  creating_active.rejection_count == 1u &&
                  creating_active.model_mapping_registered_bytes == 0 &&
                  creating_active.whole_model_copied_bytes == 0 &&
                  creating_active.routed_payload_bytes == 0 &&
                  creating_active.opportunistic_range_allocated_bytes == 0 &&
                  creating_active.legacy_model_range_count == 0 &&
                  creating_active.legacy_model_arena_count == 0,
              "CREATING publishes ACTIVE with one rejection and no legacy mutation");
        CHECK(ds4_gpu_laguna_compact_destroy(creating_creator.context) ==
                  DS4_GPU_LAGUNA_DESTROY_OK,
              "CREATING probe context reports typed teardown success");
        CHECK(tracker_has_only_ledger(&creating_runtime.tracker),
              "CREATING probe teardown restores ledger-only tracking");
    }
    if (creating_tracker_ready) {
        CHECK(tracker_fixture_release_ledger(&creating_runtime),
              "CREATING probe ledger records release");
    }
    if (!creating_thread_started) goto cleanup_map;

    ds4_gpu_set_ssd_streaming(true);

    CHECK(setenv("DS4_CUDA_NO_MODEL_COPY", "1", 1) == 0 &&
              setenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE", "1", 1) == 0,
          "harmless disabling and verbose CUDA knobs can be injected");

    tracker_fixture tombstone_runtime;
    const bool tombstone_runtime_ready =
        tracker_fixture_init(&tombstone_runtime, &plan, &ledger) &&
        tracker_fixture_fill_with_tombstones(&tombstone_runtime);
    CHECK(tombstone_runtime_ready,
          "full tracker retains exactly three live ledger records over tombstones");
    if (tombstone_runtime_ready) {
        ds4_gpu_laguna_compact *tombstone_context = NULL;
        const int tombstone_created = ds4_gpu_laguna_compact_create(
            &tombstone_context, fd, model_map, sizeof(model_bytes),
            &model_identity, &ledger, &plan,
            &tombstone_runtime.tracker);
        CHECK(tombstone_created && tombstone_context != NULL &&
                  tombstone_runtime.tracker.record_count ==
                      TRACKER_RECORD_CAPACITY,
              "compact create accepts the ledger baseline and reuses dead tracker slots");
        if (tombstone_context) {
            CHECK(ds4_gpu_laguna_compact_destroy(tombstone_context) ==
                      DS4_GPU_LAGUNA_DESTROY_OK,
                  "tombstone-backed compact context tears down");
        }
        CHECK(tracker_has_only_ledger(&tombstone_runtime.tracker),
              "tombstone-backed teardown restores the ledger-only baseline");
        CHECK(tracker_fixture_release_ledger(&tombstone_runtime),
              "tombstone-backed ledger records release");
    }

    tracker_fixture unwind_runtime;
    const bool unwind_tracker_ready =
        tracker_fixture_init(&unwind_runtime, &plan, &ledger);
    CHECK(unwind_tracker_ready,
          "late pre-publication unwind tracker initializes");
    if (unwind_tracker_ready) {
        const int unwind_fds_before = open_fd_count();
        const uint64_t unwind_attempts_before =
            ds4_gpu_test_laguna_compact_static_allocation_attempts();
        ds4_gpu_laguna_compact *unwind_context = NULL;
        ds4_gpu_test_laguna_compact_fail_before_publish_once();
        const int unwind_created = ds4_gpu_laguna_compact_create(
            &unwind_context, fd, model_map, sizeof(model_bytes),
            &model_identity, &ledger, &plan, &unwind_runtime.tracker);
        CHECK(!unwind_created && unwind_context == NULL,
              "late pre-publication failure returns no compact context");
        CHECK(ds4_gpu_test_laguna_compact_static_allocation_attempts() ==
                  unwind_attempts_before + 1u,
              "late pre-publication failure follows one slab allocation");
        CHECK(unwind_fds_before >= 0 &&
                  open_fd_count() == unwind_fds_before,
              "successful late unwind restores the descriptor baseline");
        CHECK(tracker_has_only_ledger(&unwind_runtime.tracker),
              "successful late unwind restores ledger-only tracking");
        ds4_gpu_laguna_compact_test_snapshot unwind_nonidle;
        memset(&unwind_nonidle, 0, sizeof(unwind_nonidle));
        CHECK(!ds4_gpu_test_laguna_compact_nonidle_snapshot(&unwind_nonidle),
              "successful late unwind restores singleton IDLE");
        if (unwind_context) {
            CHECK(ds4_gpu_laguna_compact_destroy(unwind_context) ==
                      DS4_GPU_LAGUNA_DESTROY_OK,
                  "unexpected late-unwind context is contained");
        }
        CHECK(tracker_fixture_release_ledger(&unwind_runtime),
              "late-unwind ledger records release");
    }

    tracker_fixture runtime;
    CHECK(tracker_fixture_init(&runtime, &plan, &ledger),
          "startup runtime tracker initializes");
    const uint64_t attempts_before_create =
        ds4_gpu_test_laguna_compact_static_allocation_attempts();
    CHECK(ds4_gpu_laguna_compact_create(
              &context, fd, model_map, sizeof(model_bytes),
              &model_identity, &ledger, &plan, &runtime.tracker) &&
              context != NULL,
          "singleton is reusable after late unwind and attaches the model");
    CHECK(ds4_gpu_test_laguna_compact_static_allocation_attempts() ==
              attempts_before_create + 1u,
          "startup performs exactly one static CUDA slab allocation");

    ds4_gpu_laguna_compact_test_snapshot compact;
    memset(&compact, 0, sizeof(compact));
    CHECK(ds4_gpu_test_laguna_compact_snapshot(context, &compact),
          "compact context exposes a test snapshot");
    CHECK(compact.lifecycle == DS4_GPU_LAGUNA_LIFECYCLE_ACTIVE &&
              compact.model_fd >= 0 && compact.model_fd != fd &&
              compact.model_map == model_map &&
              compact.model_size == sizeof(model_bytes) &&
              identities_equal(&compact.model_identity, &model_identity),
          "compact attachment owns a duplicate with exact map identity");
    CHECK(compact.model_fd >= 0 &&
              (fcntl(compact.model_fd, F_GETFD) & FD_CLOEXEC) != 0,
          "owned compact descriptor is close-on-exec");
    CHECK(compact.model_fd_live && compact.static_slab_live &&
              compact.static_offsets_live && compact.tracker_mapping_live &&
              compact.tracker_static_live && compact.tracker_offsets_live,
          "active compact snapshot reports all six owners live");
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

    const ds4_runtime_allocation_record *offset_record = NULL;
    for (size_t i = 0; i < runtime_snapshot.active_record_count; i++) {
        if (active[i].callsite_id == DS4_LAGUNA_CALLSITE_STATIC_OFFSETS) {
            offset_record = &active[i];
            break;
        }
    }
    CHECK(offset_record && offset_record->live &&
              offset_record->domain == DS4_RUNTIME_DOMAIN_HOST &&
              offset_record->requested_bytes ==
                  FIXTURE_TENSOR_COUNT * sizeof(uint64_t) &&
              offset_record->charged_bytes ==
                  FIXTURE_TENSOR_COUNT * sizeof(uint64_t) &&
              offset_record->base != 0,
          "static-offset record names the exact live host table");
    const uint64_t *static_offsets = offset_record ?
        (const uint64_t *)(uintptr_t)offset_record->base : NULL;
    CHECK(static_offsets && static_offsets[0] == 0 &&
              static_offsets[1] == 256 &&
              static_offsets[2] == UINT64_MAX &&
              static_offsets[3] == UINT64_MAX &&
              static_offsets[4] == UINT64_MAX &&
              static_offsets[5] == UINT64_MAX &&
              static_offsets[6] == UINT64_MAX &&
              static_offsets[7] == UINT64_MAX,
          "offset table follows ledger order and marks every routed tensor");

    ds4_gpu_tensor *q8_input = ds4_gpu_tensor_alloc(
        2u * 32u * sizeof(float));
    ds4_gpu_tensor *q8_output = ds4_gpu_tensor_alloc(
        2u * 2u * sizeof(float));
    ds4_gpu_tensor *selected_tensor =
        ds4_gpu_tensor_alloc(sizeof(int32_t));
    const int32_t selected_zero = 0;
    CHECK(q8_input != NULL && q8_output != NULL &&
              selected_tensor != NULL &&
              ds4_gpu_tensor_write(
                  selected_tensor, 0, &selected_zero,
                  sizeof(selected_zero)) != 0,
          "tiny Q8 and selected-cache wrapper probes allocate test tensors");

    ds4_gpu_laguna_compact_test_snapshot rejection_baseline;
    ds4_runtime_snapshot rejection_runtime_baseline;
    ds4_runtime_allocation_record
        rejection_records_baseline[TRACKER_RECORD_CAPACITY];
    memset(&rejection_baseline, 0, sizeof(rejection_baseline));
    CHECK(ds4_gpu_test_laguna_compact_snapshot(
              context, &rejection_baseline) &&
              capture_tracker_snapshot(
                  &runtime.tracker,
                  &rejection_runtime_baseline,
                  rejection_records_baseline),
          "legacy-exclusivity baseline captures compact and tracker state");
    uint64_t expected_rejections = rejection_baseline.rejection_count;

#define EXPECT_COMPACT_REJECTION(call_is_rejected, label)                  \
    do {                                                                    \
        CHECK((call_is_rejected), (label));                                 \
        expected_rejections++;                                              \
        expect_only_compact_rejection(                                      \
            context, &rejection_baseline, expected_rejections,              \
            &runtime.tracker, &rejection_runtime_baseline,                  \
            rejection_records_baseline,                                    \
            "legacy API increments exactly one compact rejection");       \
    } while (0)

    if (context && alias_map != MAP_FAILED && other_map != MAP_FAILED &&
        fixture_fd >= 0 && other_fd >= 0) {
        CHECK(setenv("DS4_CUDA_NO_FD_CACHE", "1", 1) == 0,
              "ACTIVE alias prewarm disables the large legacy fd arena");
        EXPECT_COMPACT_REJECTION(
            ds4_gpu_cache_model_range(
                alias_map, sizeof(model_bytes),
                ledger.tensor_ranges[2].source_offset,
                ledger.tensor_ranges[2].source_bytes,
                "laguna-alias-prewarm-f32") == 0,
            "compact ACTIVE rejects an alias range prewarm for Q8-F32");
        EXPECT_COMPACT_REJECTION(
            ds4_gpu_cache_model_range(
                alias_map, sizeof(model_bytes),
                ledger.tensor_ranges[3].source_offset,
                ledger.tensor_ranges[3].source_bytes,
                "laguna-alias-prewarm-f16") == 0,
            "compact ACTIVE rejects an alias range prewarm for Q8-F16");
        unsetenv("DS4_CUDA_NO_FD_CACHE");
        EXPECT_COMPACT_REJECTION(
            ds4_gpu_test_laguna_compact_resolve_weight_ptr(
                alias_map,
                ledger.tensor_ranges[2].source_offset,
                ledger.tensor_ranges[2].source_bytes,
                0, "laguna-warm-alias") == NULL,
            "compact lookup rejects a warm same-inode alias mapping");

        CHECK(setenv("DS4_CUDA_Q8_F32_ALL", "1", 1) == 0,
              "Q8-F32 internal wrapper probe is enabled late");
        EXPECT_COMPACT_REJECTION(
            q8_input && q8_output &&
                ds4_gpu_matmul_q8_0_tensor(
                    q8_output, alias_map, sizeof(model_bytes),
                    ledger.tensor_ranges[2].source_offset,
                    32, 2, q8_input, 2) == 0,
            "compact ACTIVE rejects Q8-F32 resolution from a warm alias");
        unsetenv("DS4_CUDA_Q8_F32_ALL");
        CHECK(setenv("DS4_CUDA_Q8_F16_ALL", "1", 1) == 0,
              "Q8-F16 internal wrapper probe is enabled late");
        EXPECT_COMPACT_REJECTION(
            q8_input && q8_output &&
                ds4_gpu_matmul_q8_0_tensor(
                    q8_output, alias_map, sizeof(model_bytes),
                    ledger.tensor_ranges[3].source_offset,
                    32, 2, q8_input, 2) == 0,
            "compact ACTIVE rejects Q8-F16 resolution from a warm alias");
        unsetenv("DS4_CUDA_Q8_F16_ALL");

        int setter_fds_before = open_fd_count();
        EXPECT_COMPACT_REJECTION(
            ds4_gpu_set_model_fd_for_map(fixture_fd, alias_map) == 0,
            "compact ACTIVE rejects model-fd-for-map placement");
        CHECK(setter_fds_before >= 0 &&
                  open_fd_count() == setter_fds_before,
              "rejected model-fd-for-map placement leaks no descriptor");
        EXPECT_COMPACT_REJECTION(
            ds4_gpu_cache_model_range(
                alias_map, sizeof(model_bytes),
                ledger.tensor_ranges[4].source_offset,
                ledger.tensor_ranges[4].source_bytes,
                "laguna-fd-arena") == 0,
            "compact ACTIVE rejects the internal fd-arena path");
        setter_fds_before = open_fd_count();
        EXPECT_COMPACT_REJECTION(
            ds4_gpu_set_model_fd(fixture_fd) == 0,
            "compact ACTIVE rejects model-fd placement");
        CHECK(setter_fds_before >= 0 &&
                  open_fd_count() == setter_fds_before,
              "rejected model-fd placement leaks no descriptor");

        int lookup_device = 77;
        void *lookup_pointer = (void *)(uintptr_t)UINT64_C(0x1234);
        const int lookup_result = ds4_gpu_lookup_cache(
            ledger.tensor_ranges[4].source_offset,
            ledger.tensor_ranges[4].source_bytes,
            &lookup_device, &lookup_pointer);
        EXPECT_COMPACT_REJECTION(
            lookup_result == 0 && lookup_device == 77 &&
                lookup_pointer == (void *)(uintptr_t)UINT64_C(0x1234),
            "compact ACTIVE rejects fd-arena lookup without touching outputs");
        EXPECT_COMPACT_REJECTION(
            ds4_gpu_lookup_cache_device(
                ledger.tensor_ranges[5].source_offset,
                ledger.tensor_ranges[5].source_bytes) == -1,
            "compact ACTIVE rejects fd-arena device lookup");

        EXPECT_COMPACT_REJECTION(
            ds4_gpu_set_model_map(other_map, sizeof(model_bytes)) == 0,
            "compact ACTIVE rejects a distinct model mapping");
        EXPECT_COMPACT_REJECTION(
            ds4_gpu_register_model_map_no_copy(
                other_map, sizeof(model_bytes)) == 0,
            "compact ACTIVE rejects no-copy model-map registration");
        EXPECT_COMPACT_REJECTION(
            ds4_gpu_set_model_map_range(
                other_map, sizeof(model_bytes), 0,
                sizeof(model_bytes), 72) == 0,
            "compact ACTIVE rejects range model-map placement");
        const uint64_t span_offsets[] = {0, 256};
        const uint64_t span_sizes[] = {64, 64};
        EXPECT_COMPACT_REJECTION(
            ds4_gpu_set_model_map_spans(
                alias_map, sizeof(model_bytes),
                span_offsets, span_sizes, ARRAY_LEN(span_offsets), 72) == 0,
            "compact ACTIVE rejects span model-map placement");
        EXPECT_COMPACT_REJECTION(
            ds4_gpu_register_support_map(
                other_map, sizeof(model_bytes), UINT64_C(0x100000000)) == 0,
            "compact ACTIVE rejects support-model mapping");

        const ds4_tensor_range cache_range = {
            .source_offset = ledger.tensor_ranges[0].source_offset,
            .bytes = ledger.tensor_ranges[0].source_bytes,
            .target_device = 0,
        };
        EXPECT_COMPACT_REJECTION(
            ds4_gpu_device_cache_tensors(0, &cache_range, 1) != 0,
            "compact ACTIVE rejects main selective-cache placement");
        EXPECT_COMPACT_REJECTION(
            ds4_gpu_device_cache_support_tensors(
                0, 0, &cache_range, 1, 0) != 0,
            "compact ACTIVE rejects support selective-cache placement");
        void *main_strict =
            (void *)(uintptr_t)UINT64_C(0x4444);
        EXPECT_COMPACT_REJECTION(
            !ds4_gpu_lookup_cache_strict(
                cache_range.source_offset, cache_range.bytes,
                0, &main_strict) &&
                main_strict == (void *)(uintptr_t)UINT64_C(0x4444),
            "compact ACTIVE rejects strict main-cache lookup");
        void *support_strict =
            (void *)(uintptr_t)UINT64_C(0x5555);
        EXPECT_COMPACT_REJECTION(
            !ds4_gpu_lookup_cache_strict(
                UINT64_C(0x100000000) + cache_range.source_offset,
                cache_range.bytes, 0, &support_strict) &&
                support_strict == (void *)(uintptr_t)UINT64_C(0x5555),
            "compact ACTIVE rejects strict support-cache lookup");

        EXPECT_COMPACT_REJECTION(
            ds4_gpu_cache_model_range(
                model_map, sizeof(model_bytes),
                ledger.tensor_ranges[4].source_offset,
                ledger.tensor_ranges[4].source_bytes,
                "laguna-generic-range") == 0,
            "compact ACTIVE rejects generic model-range caching");

        CHECK(setenv("DS4_CUDA_Q8_F16_ALL", "1", 1) == 0,
              "Q8-F16 cache wrapper probe is enabled late");
        EXPECT_COMPACT_REJECTION(
            ds4_gpu_cache_q8_f16_range(
                model_map, sizeof(model_bytes),
                ledger.tensor_ranges[2].source_offset,
                ledger.tensor_ranges[2].source_bytes,
                32, 2, "laguna-q8-f16") == 0,
            "compact ACTIVE rejects the Q8-F16 cache entrypoint");
        unsetenv("DS4_CUDA_Q8_F16_ALL");
        CHECK(setenv("DS4_CUDA_Q8_F32_PRELOAD", "1", 1) == 0 &&
                  setenv("DS4_CUDA_Q8_F32_ALL", "1", 1) == 0,
              "Q8-F32 cache wrapper probe is enabled late");
        EXPECT_COMPACT_REJECTION(
            ds4_gpu_cache_q8_f16_range(
                model_map, sizeof(model_bytes),
                ledger.tensor_ranges[3].source_offset,
                ledger.tensor_ranges[3].source_bytes,
                32, 2, "laguna-q8-f32") == 0,
            "compact ACTIVE rejects the Q8-F32 cache entrypoint");
        unsetenv("DS4_CUDA_Q8_F32_PRELOAD");
        unsetenv("DS4_CUDA_Q8_F32_ALL");

        EXPECT_COMPACT_REJECTION(
            ds4_gpu_preload_q4_expert_tables(
                model_map, sizeof(model_bytes),
                ledger.tensor_ranges[2].source_offset,
                ledger.tensor_ranges[3].source_offset,
                ledger.tensor_ranges[4].source_offset,
                36, 36, 2) == 0,
            "compact ACTIVE rejects Q4 expert-table preload");
        const ds4_gpu_stream_expert_table table = {
            .model_map = model_map,
            .model_size = sizeof(model_bytes),
            .layer = 1,
            .n_total_expert = 2,
            .gate_offset = ledger.tensor_ranges[2].source_offset,
            .up_offset = ledger.tensor_ranges[3].source_offset,
            .down_offset = ledger.tensor_ranges[4].source_offset,
            .gate_expert_bytes = 36,
            .down_expert_bytes = 36,
        };
        const int32_t selected[] = {0};
        EXPECT_COMPACT_REJECTION(
            ds4_gpu_stream_expert_cache_begin_selected_load(
                &table, selected, ARRAY_LEN(selected)) == 0,
            "compact ACTIVE rejects selected-expert begin");
        EXPECT_COMPACT_REJECTION(
            selected_tensor &&
                ds4_gpu_glm_stream_expert_cache_begin_selected_load_tensor(
                    &table, selected_tensor, ARRAY_LEN(selected)) == 0,
            "compact ACTIVE rejects GLM selected-tensor begin");
#if defined(DS4_ROCM_BUILD) || \
    (!defined(DS4_NO_GPU) && !defined(__APPLE__))
        EXPECT_COMPACT_REJECTION(
            ds4_gpu_stream_expert_cache_prepare_selected_batch(
                &table, selected, 1, ARRAY_LEN(selected)) == 0,
            "compact ACTIVE rejects selected-expert prepare");
#endif
        EXPECT_COMPACT_REJECTION(
            ds4_gpu_stream_expert_cache_seed_selected(
                &table, selected, ARRAY_LEN(selected)) == 0,
            "compact ACTIVE rejects selected-expert seed");
        const uint32_t priorities[] = {1};
        EXPECT_COMPACT_REJECTION(
            ds4_gpu_stream_expert_cache_seed_experts(
                &table, selected, priorities, ARRAY_LEN(selected)) == 0,
            "compact ACTIVE rejects prioritized selected-expert seed");

        EXPECT_COMPACT_REJECTION(
            ds4_gpu_test_laguna_compact_resolve_weight_ptr(
                other_map,
                ledger.tensor_ranges[3].source_offset,
                ledger.tensor_ranges[3].source_bytes,
                0, "laguna-distinct-map") == NULL,
            "compact lookup rejects a different-inode second mapping");
    }
#undef EXPECT_COMPACT_REJECTION

    CHECK(ds4_gpu_stream_expert_cache_current_count() == 0,
          "legacy selected-expert cache remains empty under compact ACTIVE");
    ds4_gpu_tensor_free(q8_input);
    ds4_gpu_tensor_free(q8_output);
    ds4_gpu_tensor_free(selected_tensor);

    const int caller_fd = fd;
    const int owned_fd = compact.model_fd;
    CHECK(close(fd) == 0, "caller closes its model descriptor after create");
    fd = -1;
    reused_fd = create_pattern_file(
        reuse_path, sizeof(model_bytes), 211u);
    if (reused_fd >= 0 && reused_fd != caller_fd) {
        const int duplicate = dup2(reused_fd, caller_fd);
        close(reused_fd);
        reused_fd = duplicate;
    }
    CHECK(reused_fd == caller_fd,
          "caller model descriptor number is deliberately reused");
    unsigned char owned_probe[32] = {0};
    ds4_laguna_file_identity owned_identity;
    memset(&owned_identity, 0, sizeof(owned_identity));
    CHECK(owned_fd >= 0 &&
              pread(owned_fd, owned_probe, sizeof(owned_probe), 0) ==
                  (ssize_t)sizeof(owned_probe) &&
              memcmp(owned_probe, model_bytes, sizeof(owned_probe)) == 0 &&
              capture_file_identity(owned_fd, &owned_identity) &&
              identities_equal(&owned_identity, &model_identity),
          "compact owned duplicate still reads identity A after caller-fd reuse");

    void *first_static_ptr = NULL;
    for (size_t i = 0; i < 2; i++) {
        const uint64_t offset = ledger.tensor_ranges[i].source_offset;
        const uint64_t bytes = ledger.tensor_ranges[i].source_bytes;
        void *device_ptr = NULL;
        CHECK(ds4_gpu_test_laguna_compact_lookup(
                  context, offset, bytes, 0, &device_ptr) &&
                  device_ptr != NULL,
              "strict lookup hits a complete static range");
        if (i == 0) first_static_ptr = device_ptr;
        if (i == 1) {
            CHECK((uintptr_t)device_ptr ==
                      (uintptr_t)first_static_ptr + 256u,
                  "static destinations advance by exact 256-byte alignment");
        }
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
        const void *subrange = ds4_gpu_test_laguna_compact_resolve_weight_ptr(
            model_map, offset + 5u, bytes - 5u, 0, "startup-subrange");
        CHECK((uintptr_t)subrange == (uintptr_t)device_ptr + 5u,
              "strict resolver preserves subrange pointer arithmetic");
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
              &second, fixture_fd, model_map, sizeof(model_bytes),
              &model_identity, &ledger, &plan,
              &second_runtime.tracker) && second == NULL,
          "only one compact context may be active per process");
    CHECK(ds4_gpu_test_laguna_compact_static_allocation_attempts() ==
              attempts_before_second &&
              tracker_has_only_ledger(&second_runtime.tracker),
          "second-context refusal preserves only its ledger records");

    ds4_gpu_laguna_compact_test_snapshot before_recoverable;
    ds4_gpu_laguna_compact_test_snapshot after_recoverable;
    ds4_runtime_snapshot runtime_before_recoverable;
    ds4_runtime_snapshot runtime_after_recoverable;
    ds4_runtime_allocation_record
        records_before_recoverable[TRACKER_RECORD_CAPACITY];
    ds4_runtime_allocation_record
        records_after_recoverable[TRACKER_RECORD_CAPACITY];
    memset(&before_recoverable, 0, sizeof(before_recoverable));
    memset(&after_recoverable, 0, sizeof(after_recoverable));
    CHECK(ds4_gpu_test_laguna_compact_snapshot(
              context, &before_recoverable) &&
              capture_tracker_snapshot(
                  &runtime.tracker,
                  &runtime_before_recoverable,
                  records_before_recoverable),
          "recoverable teardown captures byte-exact owners before sync");
    uint64_t offsets_before_recoverable[FIXTURE_TENSOR_COUNT] = {0};
    unsigned char payload_before_recoverable[2][64] = {{0}};
    bool allocation_data_ready =
        before_recoverable.static_offsets &&
        before_recoverable.static_slab &&
        before_recoverable.static_offset_bytes ==
            sizeof(offsets_before_recoverable);
    if (allocation_data_ready) {
        memcpy(offsets_before_recoverable,
               before_recoverable.static_offsets,
               sizeof(offsets_before_recoverable));
        for (size_t i = 0; i < 2 && allocation_data_ready; i++) {
            const uint64_t bytes = ledger.tensor_ranges[i].source_bytes;
            allocation_data_ready =
                bytes <= sizeof(payload_before_recoverable[i]) &&
                cudaMemcpy(
                    payload_before_recoverable[i],
                    (const char *)before_recoverable.static_slab +
                        offsets_before_recoverable[i],
                    (size_t)bytes,
                    cudaMemcpyDeviceToHost) == cudaSuccess;
        }
    }
    CHECK(allocation_data_ready,
          "recoverable teardown captures offset and static payload bytes");
    ds4_gpu_test_laguna_compact_fail_sync_once();
    const ds4_gpu_laguna_destroy_status recoverable =
        ds4_gpu_laguna_compact_destroy(context);
    CHECK(recoverable == DS4_GPU_LAGUNA_DESTROY_RECOVERABLE,
          "pre-commit synchronization failure is recoverable");
    CHECK(ds4_gpu_test_laguna_compact_snapshot(
              context, &after_recoverable) &&
              after_recoverable.lifecycle ==
                  DS4_GPU_LAGUNA_LIFECYCLE_DESTROYING &&
              after_recoverable.sync_attempt_count ==
                  before_recoverable.sync_attempt_count + 1u &&
              after_recoverable.release_attempt_count ==
                  before_recoverable.release_attempt_count &&
              compact_snapshot_equal_except_sync_lifecycle(
                  before_recoverable, after_recoverable),
          "recoverable sync failure changes only lifecycle and sync count");
    uint64_t offsets_after_recoverable[FIXTURE_TENSOR_COUNT] = {0};
    unsigned char payload_after_recoverable[2][64] = {{0}};
    bool allocation_data_preserved =
        allocation_data_ready && after_recoverable.static_offsets &&
        after_recoverable.static_slab;
    if (allocation_data_preserved) {
        memcpy(offsets_after_recoverable,
               after_recoverable.static_offsets,
               sizeof(offsets_after_recoverable));
        allocation_data_preserved = memcmp(
            offsets_before_recoverable,
            offsets_after_recoverable,
            sizeof(offsets_before_recoverable)) == 0;
        for (size_t i = 0; i < 2 && allocation_data_preserved; i++) {
            const uint64_t bytes = ledger.tensor_ranges[i].source_bytes;
            allocation_data_preserved =
                cudaMemcpy(
                    payload_after_recoverable[i],
                    (const char *)after_recoverable.static_slab +
                        offsets_after_recoverable[i],
                    (size_t)bytes,
                    cudaMemcpyDeviceToHost) == cudaSuccess &&
                memcmp(payload_before_recoverable[i],
                       payload_after_recoverable[i],
                       (size_t)bytes) == 0;
        }
    }
    CHECK(allocation_data_preserved,
          "recoverable sync failure preserves offset and static payload bytes");
    CHECK(capture_tracker_snapshot(
              &runtime.tracker,
              &runtime_after_recoverable,
              records_after_recoverable) &&
              tracker_snapshots_equal(
                  &runtime_before_recoverable,
                  records_before_recoverable,
                  &runtime_after_recoverable,
                  records_after_recoverable),
          "recoverable sync failure preserves tracker records byte-for-byte");

    int destroying_probe_result = -1;
    const bool destroying_copy_model_set =
        setenv("DS4_CUDA_COPY_MODEL", "1", 1) == 0;
    if (destroying_copy_model_set) {
        destroying_probe_result = ds4_gpu_set_model_map(
            alias_map, sizeof(model_bytes));
    }
    unsetenv("DS4_CUDA_COPY_MODEL");
    CHECK(destroying_copy_model_set && destroying_probe_result == 0,
          "DESTROYING rejects canonical legacy model placement");
    expect_only_compact_rejection(
        context,
        &after_recoverable,
        after_recoverable.rejection_count + 1u,
        &runtime.tracker,
        &runtime_after_recoverable,
        records_after_recoverable,
        "DESTROYING placement increments exactly one rejection");

    memset(owned_probe, 0, sizeof(owned_probe));
    CHECK(owned_fd >= 0 &&
              pread(owned_fd, owned_probe, sizeof(owned_probe), 0) ==
                  (ssize_t)sizeof(owned_probe) &&
              memcmp(owned_probe, model_bytes, sizeof(owned_probe)) == 0,
          "recoverable sync failure preserves readable descriptor ownership");

    ds4_gpu_laguna_compact *destroyed_context = context;
    CHECK(ds4_gpu_laguna_compact_destroy(context) ==
              DS4_GPU_LAGUNA_DESTROY_OK,
          "one retry commits and releases every compact owner");
    context = NULL;
    errno = 0;
    CHECK(fcntl(owned_fd, F_GETFD) == -1 && errno == EBADF,
          "successful compact destroy closes its owned descriptor last");
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
              destroyed_context,
              ledger.tensor_ranges[0].source_offset,
              ledger.tensor_ranges[0].source_bytes,
              0, &after_destroy),
          "teardown removes compact static lookup entries before freeing slab");

    const ds4_tensor_range post_destroy_range = {
        .source_offset = ledger.tensor_ranges[0].source_offset,
        .bytes = ledger.tensor_ranges[0].source_bytes,
        .target_device = 0,
    };
    CHECK(ds4_gpu_device_cache_tensors(
              0, &post_destroy_range, 1) == 3,
          "rejected main-map placement leaves no hidden registered base");
    CHECK(ds4_gpu_device_cache_support_tensors(
              0, 0, &post_destroy_range, 1, 0) == 3,
          "rejected support-map placement leaves no hidden registered base");
    void *post_strict = (void *)(uintptr_t)UINT64_C(0x9999);
    CHECK(!ds4_gpu_lookup_cache_strict(
              post_destroy_range.source_offset,
              post_destroy_range.bytes, 0, &post_strict) &&
              post_strict == (void *)(uintptr_t)UINT64_C(0x9999),
          "rejected main selective cache leaves no post-destroy entry");
    post_strict = (void *)(uintptr_t)UINT64_C(0xaaaa);
    CHECK(!ds4_gpu_lookup_cache_strict(
              UINT64_C(0x100000000) + post_destroy_range.source_offset,
              post_destroy_range.bytes, 0, &post_strict) &&
              post_strict == (void *)(uintptr_t)UINT64_C(0xaaaa),
          "rejected support selective cache leaves no post-destroy entry");
    CHECK(setenv("DS4_CUDA_WEIGHT_CACHE_LIMIT_GB", "1", 1) == 0,
          "post-destroy hidden-fd probe has a bounded no-allocation limit");
    int bounded_device = 99;
    void *bounded_pointer = (void *)(uintptr_t)UINT64_C(0xbbbb);
    CHECK(!ds4_gpu_lookup_cache(
              1, UINT64_C(1073741824) + 1u,
              &bounded_device, &bounded_pointer) &&
              bounded_device == 99 &&
              bounded_pointer == (void *)(uintptr_t)UINT64_C(0xbbbb),
          "rejected fd setters leave no bounded post-destroy fd fallback");
    unsetenv("DS4_CUDA_WEIGHT_CACHE_LIMIT_GB");

    tracker_fixture race_runtime[2];
    CHECK(tracker_fixture_init(&race_runtime[0], &plan, &ledger) &&
              tracker_fixture_init(&race_runtime[1], &plan, &ledger),
          "creator-race trackers initialize with ledger ownership");
    test_barrier barrier;
    pthread_t creators[2];
    creator_race race[2] = {
        {.barrier = &barrier, .model_fd = fixture_fd,
         .model_map = model_map, .model_size = sizeof(model_bytes),
         .model_identity = &model_identity,
         .ledger = &ledger, .plan = &plan,
         .runtime = &race_runtime[0]},
        {.barrier = &barrier, .model_fd = fixture_fd,
         .model_map = model_map, .model_size = sizeof(model_bytes),
         .model_identity = &model_identity,
         .ledger = &ledger, .plan = &plan,
         .runtime = &race_runtime[1]},
    };
    const bool barrier_ready = test_barrier_init(&barrier, 3);
    int creator_count = 0;
    if (barrier_ready &&
        pthread_create(&creators[0], NULL, creator_race_run, &race[0]) == 0) {
        creator_count++;
        if (pthread_create(
                &creators[1], NULL, creator_race_run, &race[1]) == 0) {
            creator_count++;
        }
    }
    CHECK(barrier_ready && creator_count == 2,
          "two concurrent compact creators reach one barrier");
    if (barrier_ready && creator_count == 2) {
        const uint64_t attempts_before_race =
            ds4_gpu_test_laguna_compact_static_allocation_attempts();
        const bool main_released = test_barrier_wait(&barrier);
        CHECK(main_released,
              "main thread releases the concurrent creator barrier");
        if (!main_released) test_barrier_cancel(&barrier);
        CHECK(pthread_join(creators[0], NULL) == 0 &&
                  pthread_join(creators[1], NULL) == 0,
              "concurrent compact creators finish");
        const int race_successes = race[0].created + race[1].created;
        CHECK(race_successes == 1 &&
                  ds4_gpu_test_laguna_compact_static_allocation_attempts() ==
                      attempts_before_race + 1u,
              "creator race publishes one context with one slab attempt");
        ds4_gpu_laguna_compact_destroy(
            race[0].created ? race[0].context : race[1].context);
    } else if (barrier_ready) {
        test_barrier_cancel(&barrier);
        for (int i = 0; i < creator_count; i++) {
            (void)pthread_join(creators[i], NULL);
        }
    }
    if (barrier_ready) test_barrier_destroy(&barrier);

    tracker_fixture reusable_runtime;
    ds4_gpu_laguna_compact *reusable = NULL;
    CHECK(tracker_fixture_init(&reusable_runtime, &plan, &ledger) &&
              ds4_gpu_laguna_compact_create(
                  &reusable, fixture_fd, model_map, sizeof(model_bytes),
                  &model_identity,
                  &ledger, &plan, &reusable_runtime.tracker) &&
              reusable != NULL,
          "singleton is reusable after raced attachment teardown");
    CHECK(ds4_gpu_laguna_compact_destroy(reusable) ==
              DS4_GPU_LAGUNA_DESTROY_OK,
          "reusable singleton reports typed teardown success");
    CHECK(tracker_has_only_ledger(&reusable_runtime.tracker),
          "typed teardown reconciles reusable compact ownership");
    ds4_gpu_cleanup();
    CHECK(tracker_has_only_ledger(&reusable_runtime.tracker),
          "global CUDA cleanup retries and reconciles compact ownership");
    reusable = NULL;

    ds4_laguna_ledger_free(&ledger);
    CHECK(tracker_fixture_release_ledger(&runtime) &&
              tracker_fixture_release_ledger(&second_runtime) &&
              tracker_fixture_release_ledger(&race_runtime[0]) &&
              tracker_fixture_release_ledger(&race_runtime[1]) &&
              tracker_fixture_release_ledger(&reusable_runtime),
          "all surviving ledger records release after physical arrays are freed");
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
    if (alias_map != MAP_FAILED) munmap(alias_map, FIXTURE_MODEL_BYTES);
    if (other_map != MAP_FAILED) munmap(other_map, FIXTURE_MODEL_BYTES);
    if (model_map != MAP_FAILED) munmap(model_map, FIXTURE_MODEL_BYTES);
cleanup_file:
    if (fd >= 0) close(fd);
    if (fixture_fd >= 0) close(fixture_fd);
    if (other_fd >= 0) close(other_fd);
    if (reused_fd >= 0) close(reused_fd);
    unlink(path);
    unlink(other_path);
    unlink(reuse_path);
cleanup_gpu:
    unsetenv("DS4_CUDA_NO_MODEL_COPY");
    unsetenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE");
    ds4_gpu_cleanup();
cleanup:
    restore_forbidden_environment(&saved);
    return result;
}

static int run_create_unwind_unsafe(void) {
    saved_environment saved;
    save_and_clear_forbidden_environment(&saved);
    char path[] = "/tmp/ds4-cuda-laguna-create-unsafe.XXXXXX";
    int fd = -1;
    unsigned char *model_map = MAP_FAILED;
    unsigned char *alias_map = MAP_FAILED;
    ds4_laguna_ledger ledger;
    ds4_laguna_allocation_plan plan;
    tracker_fixture runtime;
    ds4_laguna_file_identity identity;
    ds4_gpu_laguna_compact *context = NULL;
    memset(&ledger, 0, sizeof(ledger));
    memset(&plan, 0, sizeof(plan));
    memset(&runtime, 0, sizeof(runtime));
    memset(&identity, 0, sizeof(identity));

    int device_count = 0;
    CHECK(cudaGetDeviceCount(&device_count) == cudaSuccess && device_count >= 1,
          "unsafe create-unwind test has one visible CUDA device");
    const bool gpu_ready = device_count >= 1 && ds4_gpu_init() != 0;
    CHECK(gpu_ready, "unsafe create-unwind initializes CUDA");
    fd = create_pattern_file(path, FIXTURE_MODEL_BYTES, 11u);
    const bool source_ready =
        fd >= 0 && capture_file_identity(fd, &identity);
    CHECK(source_ready,
          "unsafe create-unwind opens an identity-bearing synthetic model");
    if (fd >= 0) {
        model_map = mmap(NULL, FIXTURE_MODEL_BYTES, PROT_READ,
                         MAP_SHARED, fd, 0);
        alias_map = mmap(NULL, FIXTURE_MODEL_BYTES, PROT_READ,
                         MAP_SHARED, fd, 0);
    }
    const bool maps_ready =
        model_map != MAP_FAILED && alias_map != MAP_FAILED;
    CHECK(maps_ready,
          "unsafe create-unwind opens exact and alias mappings");
    const bool ledger_ready = ledger_fixture_build(&ledger);
    CHECK(ledger_ready,
          "unsafe create-unwind synthetic ledger builds");
    if (ledger_ready) plan_prepare(&plan, &ledger);
    const bool tracker_ready = ledger_ready &&
        tracker_fixture_init(&runtime, &plan, &ledger);
    CHECK(tracker_ready,
          "unsafe create-unwind runtime tracker initializes");
    const bool fixture_ready =
        gpu_ready && source_ready && maps_ready && tracker_ready;

    const int fd_baseline = open_fd_count();
    const uint64_t attempts_before =
        ds4_gpu_test_laguna_compact_static_allocation_attempts();
    int created = 0;
    if (fixture_ready) {
        ds4_gpu_test_laguna_compact_fail_before_publish_once();
        ds4_gpu_test_laguna_compact_fail_release_once();
        created = ds4_gpu_laguna_compact_create(
            &context, fd, model_map, FIXTURE_MODEL_BYTES,
            &identity, &ledger, &plan, &runtime.tracker);
    }
    CHECK(fixture_ready && !created && context == NULL,
          "failed unsafe pre-publication unwind returns no context");
    CHECK(fixture_ready &&
              ds4_gpu_test_laguna_compact_static_allocation_attempts() ==
                  attempts_before + 1u,
          "unsafe pre-publication unwind follows one slab allocation");

    ds4_gpu_laguna_compact_test_snapshot retained;
    memset(&retained, 0, sizeof(retained));
    const bool retained_captured = fixture_ready &&
        ds4_gpu_test_laguna_compact_nonidle_snapshot(&retained);
    CHECK(retained_captured &&
              retained.lifecycle != DS4_GPU_LAGUNA_LIFECYCLE_IDLE &&
              retained.lifecycle != DS4_GPU_LAGUNA_LIFECYCLE_ACTIVE,
          "unsafe create unwind latches a fail-closed transitional state");
    CHECK(retained_captured &&
              retained.model_map == model_map &&
              retained.model_size == FIXTURE_MODEL_BYTES &&
              identities_equal(&retained.model_identity, &identity) &&
              retained.model_fd_live && retained.static_slab_live &&
              retained.static_offsets_live &&
              retained.tracker_mapping_live &&
              retained.tracker_static_live &&
              retained.tracker_offsets_live &&
              retained.static_slab_bytes == 512 &&
              retained.static_source_copied_bytes ==
                  ledger.static_source_bytes &&
              retained.static_range_count == ledger.static_parent_count &&
              retained.static_offset_count == FIXTURE_TENSOR_COUNT &&
              retained.static_offset_bytes ==
                  FIXTURE_TENSOR_COUNT * sizeof(uint64_t),
          "unsafe create unwind retains all six exact owners");

    const uint64_t expected_offsets[FIXTURE_TENSOR_COUNT] = {
        0, 256, UINT64_MAX, UINT64_MAX,
        UINT64_MAX, UINT64_MAX, UINT64_MAX, UINT64_MAX,
    };
    compact_allocation_bytes retained_allocation;
    bool retained_allocation_exact = retained_captured &&
        capture_compact_allocation_bytes(
            &retained, &ledger, &retained_allocation) &&
        memcmp(retained_allocation.offsets,
               expected_offsets, sizeof(expected_offsets)) == 0;
    for (size_t i = 0; i < 2 && retained_allocation_exact; i++) {
        const uint64_t bytes = ledger.tensor_ranges[i].source_bytes;
        retained_allocation_exact =
            bytes <= sizeof(retained_allocation.payloads[i]) &&
            memcmp(retained_allocation.payloads[i],
                   model_map + ledger.tensor_ranges[i].source_offset,
                   (size_t)bytes) == 0;
    }
    CHECK(retained_allocation_exact,
          "unsafe create unwind retains readable exact offsets and payload");

    unsigned char expected[32];
    unsigned char retained_probe[32] = {0};
    for (size_t i = 0; i < sizeof(expected); i++) {
        expected[i] = (unsigned char)((i * 37u + 11u) & 0xffu);
    }
    ds4_laguna_file_identity retained_identity;
    memset(&retained_identity, 0, sizeof(retained_identity));
    const int retained_fd_flags = retained_captured && retained.model_fd >= 0
        ? fcntl(retained.model_fd, F_GETFD) : -1;
    CHECK(retained_captured && retained.model_fd >= 0 &&
              retained.model_fd != fd && retained_fd_flags >= 0 &&
              (retained_fd_flags & FD_CLOEXEC) != 0 &&
              pread(retained.model_fd,
                    retained_probe, sizeof(retained_probe), 0) ==
                  (ssize_t)sizeof(retained_probe) &&
              memcmp(retained_probe, expected, sizeof(expected)) == 0 &&
              capture_file_identity(retained.model_fd, &retained_identity) &&
              identities_equal(&retained_identity, &identity),
          "unsafe create unwind retains a distinct CLOEXEC exact model fd");
    CHECK(fd_baseline >= 0 && open_fd_count() == fd_baseline + 1,
          "unsafe create unwind retains exactly one owned descriptor");

    ds4_runtime_snapshot retained_tracker;
    ds4_runtime_allocation_record
        retained_records[TRACKER_RECORD_CAPACITY];
    const bool retained_tracker_captured = tracker_ready &&
        capture_tracker_snapshot(
            &runtime.tracker, &retained_tracker, retained_records);
    CHECK(retained_tracker_captured &&
              retained_tracker_matches_compact(
                  &runtime, &ledger, &retained,
                  &retained_tracker, retained_records),
          "unsafe create unwind retains six exact reconciled tracker records");

    const uint64_t attempts_after_unwind =
        ds4_gpu_test_laguna_compact_static_allocation_attempts();
    const int operation_fds_before = open_fd_count();
    tracker_fixture second_runtime;
    const bool second_tracker_ready = ledger_ready &&
        tracker_fixture_init(&second_runtime, &plan, &ledger);
    CHECK(second_tracker_ready,
          "poisoned-singleton refusal tracker initializes");
    ds4_gpu_laguna_compact *second_context = NULL;
    const int second_created = fixture_ready && second_tracker_ready
        ? ds4_gpu_laguna_compact_create(
              &second_context, fd, model_map, FIXTURE_MODEL_BYTES,
              &identity, &ledger, &plan, &second_runtime.tracker)
        : 0;
    CHECK(fixture_ready && second_tracker_ready &&
              !second_created && second_context == NULL,
          "poisoned singleton rejects a second compact creator");
    CHECK(ds4_gpu_test_laguna_compact_static_allocation_attempts() ==
              attempts_after_unwind &&
              (!second_tracker_ready ||
               tracker_has_only_ledger(&second_runtime.tracker)),
          "second creator rejection performs no allocation or tracking mutation");

    ds4_gpu_laguna_compact_test_snapshot after_second_create;
    memset(&after_second_create, 0, sizeof(after_second_create));
    const bool after_second_captured =
        ds4_gpu_test_laguna_compact_nonidle_snapshot(&after_second_create);
    CHECK(retained_captured && after_second_captured &&
              memcmp(&retained, &after_second_create,
                     sizeof(retained)) == 0,
          "second creator rejection mutates no retained owner or counter");
    if (!retained_captured && second_context) {
        CHECK(ds4_gpu_laguna_compact_destroy(second_context) ==
                  DS4_GPU_LAGUNA_DESTROY_OK,
              "unexpected current-behavior second context is contained");
        second_context = NULL;
    }

    int placement_result = -1;
    const bool placement_copy_model_set = fixture_ready &&
        setenv("DS4_CUDA_COPY_MODEL", "1", 1) == 0;
    if (placement_copy_model_set) {
        placement_result = ds4_gpu_set_model_map(
            alias_map, FIXTURE_MODEL_BYTES);
    }
    unsetenv("DS4_CUDA_COPY_MODEL");
    CHECK(placement_copy_model_set && placement_result == 0,
          "poisoned singleton rejects canonical legacy model placement");
    ds4_gpu_laguna_compact_test_snapshot after_placement;
    memset(&after_placement, 0, sizeof(after_placement));
    const bool after_placement_captured =
        ds4_gpu_test_laguna_compact_nonidle_snapshot(&after_placement);
    CHECK(retained_captured && after_placement_captured &&
              after_placement.rejection_count ==
                  retained.rejection_count + 1u,
          "poisoned singleton records exactly one legacy rejection");
    if (retained_captured && after_placement_captured) {
        ds4_gpu_laguna_compact_test_snapshot normalized = after_placement;
        normalized.rejection_count = retained.rejection_count;
        CHECK(memcmp(&normalized, &retained, sizeof(normalized)) == 0,
              "poisoned legacy rejection mutates no owner, data, or release state");
    }

    ds4_runtime_snapshot final_tracker;
    ds4_runtime_allocation_record final_records[TRACKER_RECORD_CAPACITY];
    CHECK(retained_tracker_captured &&
              capture_tracker_snapshot(
                  &runtime.tracker, &final_tracker, final_records) &&
              tracker_snapshots_equal(
                  &retained_tracker, retained_records,
                  &final_tracker, final_records),
          "poisoned creator and placement rejections leave tracking byte-identical");
    CHECK(ds4_gpu_test_laguna_compact_static_allocation_attempts() ==
              attempts_after_unwind,
          "poisoned creator and placement perform no new static allocation");
    CHECK(operation_fds_before >= 0 &&
              open_fd_count() == operation_fds_before,
          "poisoned creator and placement leak no descriptors");

    /* Deliberately retain the poisoned singleton, mappings, descriptors,
     * ledger, and trackers until this isolated process exits. */
    unlink(path);
    restore_forbidden_environment(&saved);
    const int result = g_failures == 0 ? 0 : 1;
    if (result == 0) {
        fprintf(stderr,
                "test_cuda_laguna_stream create-unwind-unsafe "
                "PASS (%d assertions)\n",
                g_assertions);
    } else {
        fprintf(stderr,
                "test_cuda_laguna_stream create-unwind-unsafe "
                "FAIL (%d/%d assertions)\n",
                g_failures, g_assertions);
    }
    fflush(NULL);
    _Exit(result);
}

static int run_teardown_unsafe(void) {
    saved_environment saved;
    save_and_clear_forbidden_environment(&saved);
    char path[] = "/tmp/ds4-cuda-laguna-unsafe.XXXXXX";
    int fd = -1;
    unsigned char *model_map = MAP_FAILED;
    unsigned char *alias_map = MAP_FAILED;
    ds4_laguna_ledger ledger;
    ds4_laguna_allocation_plan plan;
    tracker_fixture runtime;
    ds4_laguna_file_identity identity;
    ds4_gpu_laguna_compact *context = NULL;
    memset(&ledger, 0, sizeof(ledger));
    memset(&plan, 0, sizeof(plan));
    memset(&runtime, 0, sizeof(runtime));
    memset(&identity, 0, sizeof(identity));

    int device_count = 0;
    CHECK(cudaGetDeviceCount(&device_count) == cudaSuccess && device_count >= 1,
          "unsafe teardown test has one visible CUDA device");
    CHECK(device_count >= 1 && ds4_gpu_init() != 0,
          "unsafe teardown initializes CUDA");
    fd = create_pattern_file(path, FIXTURE_MODEL_BYTES, 11u);
    CHECK(fd >= 0 && capture_file_identity(fd, &identity),
          "unsafe teardown opens an identity-bearing synthetic model");
    if (fd >= 0) {
        model_map = mmap(NULL, FIXTURE_MODEL_BYTES, PROT_READ,
                         MAP_SHARED, fd, 0);
        alias_map = mmap(NULL, FIXTURE_MODEL_BYTES, PROT_READ,
                         MAP_SHARED, fd, 0);
    }
    CHECK(model_map != MAP_FAILED && alias_map != MAP_FAILED,
          "unsafe teardown opens exact and alias mappings");
    CHECK(ledger_fixture_build(&ledger),
          "unsafe teardown synthetic ledger builds");
    if (ledger.tensor_ranges) plan_prepare(&plan, &ledger);
    CHECK(ledger.tensor_ranges &&
              tracker_fixture_init(&runtime, &plan, &ledger),
          "unsafe teardown runtime tracker initializes");
    CHECK(model_map != MAP_FAILED && ledger.tensor_ranges &&
              ds4_gpu_laguna_compact_create(
                  &context, fd, model_map, FIXTURE_MODEL_BYTES,
                  &identity, &ledger, &plan, &runtime.tracker) &&
              context != NULL,
          "unsafe teardown creates one active compact context");

    ds4_gpu_laguna_compact_test_snapshot before;
    ds4_gpu_laguna_compact_test_snapshot after;
    ds4_gpu_laguna_compact_test_snapshot repeated;
    ds4_runtime_snapshot runtime_before;
    ds4_runtime_snapshot runtime_after;
    ds4_runtime_allocation_record
        records_before[TRACKER_RECORD_CAPACITY];
    ds4_runtime_allocation_record
        records_after[TRACKER_RECORD_CAPACITY];
    memset(&before, 0, sizeof(before));
    memset(&after, 0, sizeof(after));
    memset(&repeated, 0, sizeof(repeated));
    CHECK(context &&
              ds4_gpu_test_laguna_compact_snapshot(context, &before) &&
              capture_tracker_snapshot(
                  &runtime.tracker, &runtime_before, records_before),
          "unsafe teardown captures all owners before commit");
    compact_allocation_bytes allocation_before;
    compact_allocation_bytes allocation_after;
    compact_allocation_bytes allocation_repeated;
    CHECK(capture_compact_allocation_bytes(
              &before, &ledger, &allocation_before),
          "unsafe teardown captures offset and static payload bytes");

    ds4_gpu_test_laguna_compact_fail_release_once();
    const ds4_gpu_laguna_destroy_status unsafe_result =
        ds4_gpu_laguna_compact_destroy(context);
    CHECK(unsafe_result == DS4_GPU_LAGUNA_DESTROY_UNSAFE,
          "first post-commit release failure returns UNSAFE");
    const bool after_captured = context &&
        ds4_gpu_test_laguna_compact_snapshot(context, &after);
    CHECK(after_captured &&
              after.lifecycle == DS4_GPU_LAGUNA_LIFECYCLE_RELEASING &&
              after.sync_attempt_count == before.sync_attempt_count + 1u &&
              after.release_attempt_count ==
                  before.release_attempt_count + 1u &&
              after.model_fd_live && after.static_slab_live &&
              after.static_offsets_live && after.tracker_mapping_live &&
              after.tracker_static_live && after.tracker_offsets_live &&
              compact_snapshot_equal_except_teardown_progress(before, after),
          "UNSAFE release latches RELEASING with every owner retained");
    CHECK(capture_compact_allocation_bytes(
              &after, &ledger, &allocation_after) &&
              memcmp(&allocation_before,
                     &allocation_after,
                     sizeof(allocation_before)) == 0,
          "UNSAFE release preserves offset and static payload bytes");
    CHECK(capture_tracker_snapshot(
              &runtime.tracker, &runtime_after, records_after) &&
              tracker_snapshots_equal(
                  &runtime_before, records_before,
                  &runtime_after, records_after),
          "UNSAFE release retains all six engine-owned tracker records");
    unsigned char unsafe_expected[32];
    unsigned char unsafe_probe[32] = {0};
    for (size_t i = 0; i < sizeof(unsafe_expected); i++) {
        unsafe_expected[i] = (unsigned char)((i * 37u + 11u) & 0xffu);
    }
    ds4_laguna_file_identity retained_identity;
    memset(&retained_identity, 0, sizeof(retained_identity));
    CHECK(after_captured && after.model_fd >= 0 &&
              pread(after.model_fd, unsafe_probe, sizeof(unsafe_probe), 0) ==
                  (ssize_t)sizeof(unsafe_probe) &&
              memcmp(unsafe_probe, unsafe_expected,
                     sizeof(unsafe_probe)) == 0 &&
              capture_file_identity(after.model_fd, &retained_identity) &&
              identities_equal(&retained_identity, &identity),
          "UNSAFE release retains a readable exact model duplicate");

    const ds4_gpu_laguna_destroy_status repeated_result =
        ds4_gpu_laguna_compact_destroy(context);
    const bool repeated_captured = context &&
        ds4_gpu_test_laguna_compact_snapshot(context, &repeated);
    CHECK(repeated_result == DS4_GPU_LAGUNA_DESTROY_UNSAFE &&
              repeated_captured &&
              memcmp(&after, &repeated, sizeof(after)) == 0,
          "UNSAFE destroy retry is rejected without repeating a release");
    CHECK(capture_compact_allocation_bytes(
              &repeated, &ledger, &allocation_repeated) &&
              memcmp(&allocation_before,
                     &allocation_repeated,
                     sizeof(allocation_before)) == 0,
          "rejected UNSAFE retry preserves offset and static payload bytes");
    memset(unsafe_probe, 0, sizeof(unsafe_probe));
    memset(&retained_identity, 0, sizeof(retained_identity));
    CHECK(repeated_captured && repeated.model_fd >= 0 &&
              pread(repeated.model_fd,
                    unsafe_probe, sizeof(unsafe_probe), 0) ==
                  (ssize_t)sizeof(unsafe_probe) &&
              memcmp(unsafe_probe, unsafe_expected,
                     sizeof(unsafe_probe)) == 0 &&
              capture_file_identity(
                  repeated.model_fd, &retained_identity) &&
              identities_equal(&retained_identity, &identity),
          "rejected UNSAFE retry still retains the readable duplicate");

    ds4_gpu_laguna_compact_test_snapshot unsafe_runtime_baseline = repeated;
    ds4_runtime_snapshot unsafe_tracker_baseline;
    ds4_runtime_allocation_record
        unsafe_records_baseline[TRACKER_RECORD_CAPACITY];
    CHECK(capture_tracker_snapshot(
              &runtime.tracker,
              &unsafe_tracker_baseline,
              unsafe_records_baseline),
          "UNSAFE legacy-rejection tracker baseline captures");
    CHECK(alias_map != MAP_FAILED &&
              ds4_gpu_set_model_map(alias_map, FIXTURE_MODEL_BYTES) == 0,
          "RELEASING rejects legacy model placement");
    expect_only_compact_rejection(
        context, &unsafe_runtime_baseline,
        unsafe_runtime_baseline.rejection_count + 1u,
        &runtime.tracker,
        &unsafe_tracker_baseline,
        unsafe_records_baseline,
        "RELEASING placement increments exactly one rejection");

    /* Deliberately retain the unsafe context, mappings, descriptors, ledger,
     * and tracker until this isolated process exits. */
    unlink(path);
    restore_forbidden_environment(&saved);
    return g_failures == 0 ? 0 : 1;
}

static int run_model_startup(pinned_model_case model_case) {
    const char *model = getenv("DS4_TEST_MODEL");
    if (!model || !model[0]) {
        fprintf(stderr, "FAIL: DS4_TEST_MODEL is not set\n");
        return 1;
    }
    int model_fd = -1;
    bool model_fd_set = false;
    if (!inherited_model_fd(&model_fd, &model_fd_set)) return 1;
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
        .qualification_model_fd = model_fd,
        .qualification_model_fd_set = model_fd_set,
    };
    ds4_engine *engine = NULL;
    CHECK(ds4_engine_open(&engine, &options) == 0 && engine != NULL,
          "pinned Laguna compact engine opens");
    ds4_gpu_laguna_compact_test_snapshot compact;
    memset(&compact, 0, sizeof(compact));
    CHECK(engine && ds4_gpu_test_laguna_compact_active_snapshot(&compact),
          "pinned engine owns one active compact attachment");
    const int owned_model_fd = compact.model_fd;
    CHECK(compact.model_map != NULL && compact.model_size != 0 &&
              compact.static_offset_count == 814 &&
              compact.static_offset_bytes == 6512 &&
              compact.static_range_count == 673 &&
              compact.static_slab_bytes == UINT64_C(4374164480) &&
              compact.page_advice_state_live &&
              compact.page_advice_state != NULL &&
              compact.page_advice_state_bytes == UINT64_C(1764912) &&
              compact.cache_payload_live && compact.cache_policy_live &&
              compact.device_entry_to_slot_live &&
              compact.cache_payload_bytes != 0 &&
              compact.cache_payload_bytes <=
                  UINT64_C(8) * 1024u * 1024u * 1024u &&
              compact.cache_slot_count != 0 &&
              compact.cache_slot_stride_bytes == UINT64_C(5308416) &&
              compact.pinned_staging_live_count == 4u &&
              compact.pinned_staging_bytes ==
                  4u * compact.cache_slot_stride_bytes &&
              compact.model_mapping_registered_bytes == 0 &&
              compact.whole_model_copied_bytes == 0 &&
              compact.routed_payload_bytes == 0 &&
              compact.opportunistic_range_allocated_bytes == 0,
          "pinned startup allocates fixed cache capacity without loading routed experts");
    ds4_test_laguna_compact_bypass_snapshot bypass;
    memset(&bypass, 0, sizeof(bypass));
    CHECK(ds4_test_laguna_compact_bypass_snapshot_get(&bypass) &&
              bypass.model_fd_entries == 0 &&
              bypass.model_map_entries == 0 &&
              bypass.model_range_entries == 0 &&
              bypass.model_span_entries == 0 &&
              bypass.model_cache_entries == 0 &&
              bypass.model_warm_entries == 0,
          "pinned compact startup bypasses every legacy model entrypoint");

    ds4_test_laguna_live_owner owners[PINNED_OWNER_COUNT];
    ds4_test_laguna_live_owner fresh_owners[PINNED_OWNER_COUNT];
    size_t owner_required = 0;
    size_t fresh_owner_required = 0;
    uint64_t owner_bytes = 0;
    memset(owners, 0, sizeof(owners));
    memset(fresh_owners, 0, sizeof(fresh_owners));
    const bool owners_captured = engine &&
        ds4_test_engine_laguna_live_owners(
            engine, owners, ARRAY_LEN(owners), &owner_required);
    CHECK(owners_captured && owner_required == PINNED_OWNER_COUNT &&
              pinned_live_owners_valid(owners, &owner_bytes),
          "pinned engine exposes six exact private allocation owners");
    CHECK(engine &&
              ds4_test_engine_laguna_live_owners(
                  engine, fresh_owners, ARRAY_LEN(fresh_owners),
                  &fresh_owner_required) &&
              fresh_owner_required == PINNED_OWNER_COUNT &&
              memcmp(owners, fresh_owners, sizeof(owners)) == 0,
          "pinned engine recomputes a stable fresh owner snapshot");

    ds4_test_laguna_live_owner
        ledger_owners[PINNED_LEDGER_OWNER_COUNT];
    size_t ledger_owner_required = 0;
    uint64_t ledger_owner_bytes = 0;
    memset(ledger_owners, 0, sizeof(ledger_owners));
    const bool ledger_owners_captured = engine &&
        ds4_test_engine_laguna_ledger_owners(
            engine, ledger_owners, ARRAY_LEN(ledger_owners),
            &ledger_owner_required);
    CHECK(ledger_owners_captured &&
              ledger_owner_required == PINNED_LEDGER_OWNER_COUNT &&
              pinned_ledger_owners_valid(
                  ledger_owners, &ledger_owner_bytes),
          "pinned engine exposes three independent ledger array owners");

    ds4_runtime_snapshot runtime;
    ds4_runtime_allocation_record
        records[PINNED_ACTIVE_RECORD_COUNT];
    size_t record_required = 0;
    memset(&runtime, 0, sizeof(runtime));
    memset(records, 0, sizeof(records));
    const bool runtime_captured = engine &&
        ds4_test_engine_laguna_runtime_snapshot(
            engine, &runtime, records, ARRAY_LEN(records),
            &record_required);

    ds4_runtime_snapshot rejected_runtime;
    size_t null_buffer_required = SIZE_MAX;
    memset(&rejected_runtime, 0xa5, sizeof(rejected_runtime));
    CHECK(engine && record_required != 0 &&
              !ds4_test_engine_laguna_runtime_snapshot(
                  engine, &rejected_runtime, NULL,
                  PINNED_ACTIVE_RECORD_COUNT, &null_buffer_required) &&
              null_buffer_required == record_required,
          "null record buffer still reports the exact live requirement");

    ds4_runtime_allocation_record
        undersized_records[PINNED_ACTIVE_RECORD_COUNT];
    size_t undersized_required = SIZE_MAX;
    memset(&rejected_runtime, 0xa5, sizeof(rejected_runtime));
    memset(undersized_records, 0xa5, sizeof(undersized_records));
    CHECK(engine && record_required != 0 &&
              !ds4_test_engine_laguna_runtime_snapshot(
                  engine, &rejected_runtime, undersized_records,
                  record_required - 1u, &undersized_required) &&
              undersized_required == record_required &&
              bytes_are_value(
                  undersized_records, sizeof(undersized_records), 0xa5),
          "undersized record buffer reports required without partial writes");

    CHECK(runtime_captured &&
              record_required == PINNED_ACTIVE_RECORD_COUNT &&
              runtime.active_record_count == PINNED_ACTIVE_RECORD_COUNT,
          "pinned runtime snapshot exposes twenty-two active records");
    uint64_t tracked_ledger_bytes = 0;
    CHECK(runtime_captured && ledger_owners_captured &&
              pinned_ledger_records_match(
                  &runtime, records, ledger_owners,
                  &tracked_ledger_bytes) &&
              tracked_ledger_bytes == ledger_owner_bytes,
          "pinned tracker contains the exact independent ledger multiset");
    CHECK(runtime_captured && owners_captured &&
              pinned_inventory_records_match(
                  &runtime, records, owners),
          "pinned tracker contains the exact 0x4e owner multiset");
    CHECK(runtime_captured && owners_captured &&
              ledger_owners_captured &&
              pinned_runtime_inventory_reconciles(
                  &runtime, records, owners, ledger_owners,
                  owner_bytes, &compact),
          "pinned runtime snapshot reconciles the production inventory");

    if (model_case == PINNED_MODEL_TEARDOWN_RECONCILE_UNSAFE) {
        CHECK(ds4_test_engine_laguna_inventory_live_flag_clear(
                  engine, PINNED_OWNER_COUNT - 1u),
              "test seam leaves one tracker owner live but clears its engine flag");
        const uint64_t cleanup_before =
            ds4_gpu_test_generic_cleanup_attempts();
        ds4_engine_close(engine);

        ds4_test_laguna_compact_close_observation observation;
        memset(&observation, 0, sizeof(observation));
        CHECK(ds4_test_laguna_compact_close_observation_get(&observation) &&
                  observation.first_destroy_result ==
                      DS4_GPU_LAGUNA_DESTROY_OK &&
                  observation.destroy_result ==
                      DS4_GPU_LAGUNA_DESTROY_OK &&
                  observation.destroy_attempt_count == 1u &&
                  observation.engine_retained &&
                  observation.gpu_cleanup_before == cleanup_before &&
                  observation.gpu_cleanup_after == cleanup_before &&
                  ds4_gpu_test_generic_cleanup_attempts() == cleanup_before,
              "successful compact destroy still retains an unreconciled engine");
        ds4_gpu_laguna_compact_test_snapshot nonidle;
        memset(&nonidle, 0, sizeof(nonidle));
        CHECK(!ds4_gpu_test_laguna_compact_nonidle_snapshot(&nonidle),
              "successful compact destroy leaves no non-IDLE residue");
        errno = 0;
        CHECK(owned_model_fd >= 0 &&
                  fcntl(owned_model_fd, F_GETFD) == -1 && errno == EBADF,
              "successful compact destroy closes its owned model descriptor");
        ds4_runtime_snapshot closed_runtime;
        memset(&closed_runtime, 0, sizeof(closed_runtime));
        CHECK(!ds4_test_laguna_last_close_snapshot(&closed_runtime),
              "unreconciled retained engine publishes no clean close snapshot");
        restore_forbidden_environment(&saved);
        return g_failures == 0 ? 0 : 1;
    }

    if (model_case == PINNED_MODEL_CLEANUP_RELEASE_UNSAFE) {
        CHECK(ds4_test_engine_laguna_inventory_release_reject_once(
                  engine, 3u),
              "test seam arms one rejected vocabulary-owner release");
        const uint64_t cleanup_before =
            ds4_gpu_test_generic_cleanup_attempts();
        ds4_engine_close(engine);

        ds4_test_laguna_compact_close_observation observation;
        memset(&observation, 0, sizeof(observation));
        CHECK(ds4_test_laguna_compact_close_observation_get(&observation) &&
                  observation.first_destroy_result ==
                      DS4_GPU_LAGUNA_DESTROY_OK &&
                  observation.destroy_result ==
                      DS4_GPU_LAGUNA_DESTROY_OK &&
                  observation.destroy_attempt_count == 1u &&
                  observation.engine_retained &&
                  observation.gpu_cleanup_before == cleanup_before &&
                  observation.gpu_cleanup_after == cleanup_before &&
                  ds4_gpu_test_generic_cleanup_attempts() == cleanup_before,
              "rejected vocabulary release retains its engine before cleanup");
        ds4_runtime_snapshot closed_runtime;
        memset(&closed_runtime, 0, sizeof(closed_runtime));
        CHECK(!ds4_test_laguna_last_close_snapshot(&closed_runtime),
              "rejected ordinary release publishes no clean close snapshot");
        restore_forbidden_environment(&saved);
        return g_failures == 0 ? 0 : 1;
    }

    const uint64_t cleanup_before =
        ds4_gpu_test_generic_cleanup_attempts();
    ds4_gpu_test_laguna_compact_fail_sync_once();
    ds4_engine_close(engine);
    ds4_test_laguna_compact_close_observation observation;
    memset(&observation, 0, sizeof(observation));
    CHECK(ds4_test_laguna_compact_close_observation_get(&observation) &&
              observation.first_destroy_result ==
                  DS4_GPU_LAGUNA_DESTROY_RECOVERABLE &&
              observation.destroy_result == DS4_GPU_LAGUNA_DESTROY_OK &&
              observation.destroy_attempt_count == 2u &&
              !observation.engine_retained &&
              observation.gpu_cleanup_before == cleanup_before &&
              observation.gpu_cleanup_after == cleanup_before + 1u &&
              ds4_gpu_test_generic_cleanup_attempts() == cleanup_before + 1u,
          "engine retries one RECOVERABLE close and cleans up after terminal OK");
    ds4_gpu_laguna_compact_test_snapshot nonidle;
    memset(&nonidle, 0, sizeof(nonidle));
    CHECK(!ds4_gpu_test_laguna_compact_nonidle_snapshot(&nonidle),
          "terminal OK leaves no non-IDLE compact residue");
    errno = 0;
    CHECK(owned_model_fd >= 0 &&
              fcntl(owned_model_fd, F_GETFD) == -1 && errno == EBADF,
          "terminal OK closes the compact-owned model descriptor");
    CHECK(!ds4_gpu_test_laguna_compact_active_snapshot(&compact),
          "pinned engine teardown destroys compact attachment");
    ds4_runtime_snapshot closed_runtime;
    memset(&closed_runtime, 0, sizeof(closed_runtime));
    CHECK(ds4_test_laguna_last_close_snapshot(&closed_runtime) &&
              runtime_snapshot_is_clean(&closed_runtime),
          "pinned engine close captures one clean runtime snapshot");
    restore_forbidden_environment(&saved);
    return g_failures == 0 ? 0 : 1;
}

static int run_model_teardown_unsafe(void) {
    const char *model = getenv("DS4_TEST_MODEL");
    if (!model || !model[0]) {
        fprintf(stderr, "FAIL: DS4_TEST_MODEL is not set\n");
        return 1;
    }
    int model_fd = -1;
    bool model_fd_set = false;
    if (!inherited_model_fd(&model_fd, &model_fd_set)) return 1;
    if (!model_fd_set) {
        fprintf(stderr,
                "FAIL: model-teardown-unsafe requires inherited "
                "DS4_TEST_MODEL_FD\n");
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
        .qualification_model_fd = model_fd,
        .qualification_model_fd_set = true,
    };
    ds4_engine *engine = NULL;
    CHECK(ds4_engine_open(&engine, &options) == 0 && engine != NULL,
          "pinned unsafe Laguna engine opens from retained fd 9");
    ds4_gpu_laguna_compact_test_snapshot active;
    memset(&active, 0, sizeof(active));
    const bool active_captured = engine &&
        ds4_gpu_test_laguna_compact_active_snapshot(&active);
    CHECK(active_captured &&
              active.lifecycle == DS4_GPU_LAGUNA_LIFECYCLE_ACTIVE,
          "pinned unsafe engine starts with one ACTIVE compact attachment");
    const uint64_t cleanup_before =
        ds4_gpu_test_generic_cleanup_attempts();
    ds4_gpu_test_laguna_compact_fail_release_once();
    ds4_engine_close(engine);

    ds4_test_laguna_compact_close_observation observation;
    memset(&observation, 0, sizeof(observation));
    CHECK(ds4_test_laguna_compact_close_observation_get(&observation) &&
              observation.first_destroy_result ==
                  DS4_GPU_LAGUNA_DESTROY_UNSAFE &&
              observation.destroy_result ==
                  DS4_GPU_LAGUNA_DESTROY_UNSAFE &&
              observation.destroy_attempt_count == 1u &&
              observation.engine_retained &&
              observation.gpu_cleanup_before == cleanup_before &&
              observation.gpu_cleanup_after == cleanup_before &&
              ds4_gpu_test_generic_cleanup_attempts() == cleanup_before,
          "engine UNSAFE close retains owners and bypasses generic cleanup");
    unsigned char retained_probe[32] = {0};
    ds4_laguna_file_identity retained_identity;
    memset(&retained_identity, 0, sizeof(retained_identity));
    CHECK(active_captured && active.model_fd >= 0 &&
              (fcntl(active.model_fd, F_GETFD) & FD_CLOEXEC) != 0 &&
              pread(active.model_fd,
                    retained_probe, sizeof(retained_probe), 0) ==
                  (ssize_t)sizeof(retained_probe) &&
              capture_file_identity(
                  active.model_fd, &retained_identity) &&
              identities_equal(
                  &retained_identity, &active.model_identity),
          "engine UNSAFE close retains its readable exact model duplicate");
    memset(&active, 0, sizeof(active));
    CHECK(!ds4_gpu_test_laguna_compact_active_snapshot(&active),
          "engine UNSAFE close leaves no falsely ACTIVE compact snapshot");
    const int placement_fds_before = open_fd_count();
    CHECK(ds4_gpu_set_model_fd(model_fd) == 0,
          "retained engine state rejects legacy placement after UNSAFE close");
    CHECK(placement_fds_before >= 0 &&
              open_fd_count() == placement_fds_before,
          "post-UNSAFE legacy rejection leaks no descriptor");

    /* The close observation proves the engine is intentionally retained;
     * direct teardown-unsafe proves the corresponding exact RELEASING state.
     * This isolated process owns both until exit. */
    restore_forbidden_environment(&saved);
    return g_failures == 0 ? 0 : 1;
}

static int run_model_create_unwind_unsafe(void) {
    const char *model = getenv("DS4_TEST_MODEL");
    if (!model || !model[0]) {
        fprintf(stderr, "FAIL: DS4_TEST_MODEL is not set\n");
        return 1;
    }
    int model_fd = -1;
    bool model_fd_set = false;
    if (!inherited_model_fd(&model_fd, &model_fd_set)) return 1;
    if (!model_fd_set) {
        fprintf(stderr,
                "FAIL: model-create-unwind-unsafe requires inherited "
                "DS4_TEST_MODEL_FD\n");
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
        .qualification_model_fd = model_fd,
        .qualification_model_fd_set = true,
    };
    const uint64_t cleanup_before =
        ds4_gpu_test_generic_cleanup_attempts();
    ds4_gpu_test_laguna_compact_fail_after_identity_once();
    ds4_gpu_test_laguna_compact_fail_release_once();
    ds4_engine *engine = NULL;
    const int open_result = ds4_engine_open(&engine, &options);
    CHECK(open_result != 0 && engine == NULL,
          "early unsafe compact unwind fails engine publication");

    ds4_test_laguna_compact_close_observation observation;
    memset(&observation, 0, sizeof(observation));
    CHECK(ds4_test_laguna_compact_close_observation_get(&observation) &&
              observation.first_destroy_result ==
                  DS4_GPU_LAGUNA_DESTROY_UNSAFE &&
              observation.destroy_result ==
                  DS4_GPU_LAGUNA_DESTROY_UNSAFE &&
              observation.destroy_attempt_count == 0u &&
              observation.engine_retained &&
              observation.gpu_cleanup_before == cleanup_before &&
              observation.gpu_cleanup_after == cleanup_before &&
              ds4_gpu_test_generic_cleanup_attempts() == cleanup_before,
          "early unsafe create unwind retains engine before generic cleanup");

    ds4_gpu_laguna_compact_test_snapshot retained;
    memset(&retained, 0, sizeof(retained));
    const bool retained_captured =
        ds4_gpu_test_laguna_compact_nonidle_snapshot(&retained);
    CHECK(retained_captured &&
              retained.lifecycle == DS4_GPU_LAGUNA_LIFECYCLE_RELEASING &&
              retained.model_fd_live && !retained.static_slab_live &&
              !retained.static_offsets_live &&
              !retained.tracker_mapping_live &&
              !retained.tracker_static_live &&
              !retained.tracker_offsets_live &&
              retained.release_attempt_count == 1u,
          "early unsafe unwind retains only its exact descriptor owner");
    const int retained_fd_flags = retained_captured && retained.model_fd >= 0
        ? fcntl(retained.model_fd, F_GETFD) : -1;
    errno = 0;
    CHECK(retained_fd_flags >= 0 &&
              (retained_fd_flags & FD_CLOEXEC) != 0,
          "early unsafe unwind retains a live CLOEXEC descriptor");

    /* The unpublished engine and compact descriptor intentionally remain
     * retained until this isolated process exits. */
    restore_forbidden_environment(&saved);
    return g_failures == 0 ? 0 : 1;
}

static int run_model_teardown_second_recoverable(void) {
    const char *model = getenv("DS4_TEST_MODEL");
    if (!model || !model[0]) {
        fprintf(stderr, "FAIL: DS4_TEST_MODEL is not set\n");
        return 1;
    }
    int model_fd = -1;
    bool model_fd_set = false;
    if (!inherited_model_fd(&model_fd, &model_fd_set)) return 1;
    if (!model_fd_set) {
        fprintf(stderr,
                "FAIL: model-teardown-second-recoverable requires inherited "
                "DS4_TEST_MODEL_FD\n");
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
        .qualification_model_fd = model_fd,
        .qualification_model_fd_set = true,
    };
    ds4_engine *engine = NULL;
    CHECK(ds4_engine_open(&engine, &options) == 0 && engine != NULL,
          "pinned second-RECOVERABLE Laguna engine opens from retained fd 9");
    ds4_gpu_laguna_compact_test_snapshot active;
    memset(&active, 0, sizeof(active));
    const bool active_captured = engine &&
        ds4_gpu_test_laguna_compact_active_snapshot(&active);
    CHECK(active_captured &&
              active.lifecycle == DS4_GPU_LAGUNA_LIFECYCLE_ACTIVE,
          "pinned second-RECOVERABLE engine starts ACTIVE");
    uint64_t offsets_before[814] = {0};
    unsigned char payload_before[32] = {0};
    uint64_t payload_offset = UINT64_MAX;
    bool allocation_before_ready = active_captured &&
        active.static_offsets && active.static_slab &&
        active.static_offset_bytes == sizeof(offsets_before);
    if (allocation_before_ready) {
        memcpy(offsets_before,
               active.static_offsets, sizeof(offsets_before));
        for (size_t i = 0; i < 814; i++) {
            if (offsets_before[i] != UINT64_MAX &&
                offsets_before[i] <= active.static_slab_bytes &&
                active.static_slab_bytes - offsets_before[i] >=
                    sizeof(payload_before)) {
                payload_offset = offsets_before[i];
                break;
            }
        }
        allocation_before_ready = payload_offset != UINT64_MAX &&
            cudaMemcpy(payload_before,
                       (const char *)active.static_slab + payload_offset,
                       sizeof(payload_before),
                       cudaMemcpyDeviceToHost) == cudaSuccess;
    }
    CHECK(allocation_before_ready,
          "second-RECOVERABLE engine captures exact live allocation bytes");
    const uint64_t cleanup_before =
        ds4_gpu_test_generic_cleanup_attempts();
    ds4_gpu_test_laguna_compact_fail_sync_once();
    ds4_gpu_test_laguna_compact_fail_sync_once();
    ds4_engine_close(engine);

    ds4_test_laguna_compact_close_observation observation;
    memset(&observation, 0, sizeof(observation));
    CHECK(ds4_test_laguna_compact_close_observation_get(&observation) &&
              observation.first_destroy_result ==
                  DS4_GPU_LAGUNA_DESTROY_RECOVERABLE &&
              observation.destroy_result ==
                  DS4_GPU_LAGUNA_DESTROY_RECOVERABLE &&
              observation.destroy_attempt_count == 2u &&
              observation.engine_retained &&
              observation.gpu_cleanup_before == cleanup_before &&
              observation.gpu_cleanup_after == cleanup_before &&
              ds4_gpu_test_generic_cleanup_attempts() == cleanup_before,
          "engine retains owners after a second RECOVERABLE close");

    ds4_gpu_laguna_compact_test_snapshot retained;
    memset(&retained, 0, sizeof(retained));
    const bool retained_captured =
        ds4_gpu_test_laguna_compact_nonidle_snapshot(&retained);
    CHECK(active_captured && retained_captured &&
              retained.lifecycle == DS4_GPU_LAGUNA_LIFECYCLE_DESTROYING &&
              retained.model_fd_live && retained.static_slab_live &&
              retained.static_offsets_live &&
              retained.tracker_mapping_live &&
              retained.tracker_static_live &&
              retained.tracker_offsets_live &&
              retained.sync_attempt_count == active.sync_attempt_count + 2u &&
              retained.release_attempt_count == active.release_attempt_count &&
              compact_snapshot_equal_except_sync_lifecycle(active, retained),
          "second RECOVERABLE close retains exact DESTROYING ownership");
    uint64_t offsets_after[814] = {0};
    unsigned char payload_after[32] = {0};
    bool allocation_after_ready = retained_captured &&
        retained.static_offsets && retained.static_slab &&
        retained.static_offset_bytes == sizeof(offsets_after) &&
        payload_offset != UINT64_MAX &&
        payload_offset <= retained.static_slab_bytes &&
        retained.static_slab_bytes - payload_offset >= sizeof(payload_after);
    if (allocation_after_ready) {
        memcpy(offsets_after,
               retained.static_offsets, sizeof(offsets_after));
        allocation_after_ready =
            cudaMemcpy(payload_after,
                       (const char *)retained.static_slab + payload_offset,
                       sizeof(payload_after),
                       cudaMemcpyDeviceToHost) == cudaSuccess;
    }
    CHECK(allocation_before_ready && allocation_after_ready &&
              memcmp(offsets_before,
                     offsets_after, sizeof(offsets_before)) == 0 &&
              memcmp(payload_before,
                     payload_after, sizeof(payload_before)) == 0,
          "second RECOVERABLE close preserves exact offset and payload bytes");
    unsigned char retained_probe[32] = {0};
    ds4_laguna_file_identity retained_identity;
    memset(&retained_identity, 0, sizeof(retained_identity));
    CHECK(retained_captured && retained.model_fd >= 0 &&
              (fcntl(retained.model_fd, F_GETFD) & FD_CLOEXEC) != 0 &&
              pread(retained.model_fd,
                    retained_probe, sizeof(retained_probe), 0) ==
                  (ssize_t)sizeof(retained_probe) &&
              capture_file_identity(
                  retained.model_fd, &retained_identity) &&
              identities_equal(
                  &retained_identity, &retained.model_identity),
          "second RECOVERABLE close retains its readable exact model duplicate");
    ds4_gpu_laguna_compact_test_snapshot active_after;
    memset(&active_after, 0, sizeof(active_after));
    CHECK(!ds4_gpu_test_laguna_compact_active_snapshot(&active_after),
          "second RECOVERABLE close leaves no falsely ACTIVE snapshot");
    const int placement_fds_before = open_fd_count();
    CHECK(ds4_gpu_set_model_fd(model_fd) == 0,
          "second RECOVERABLE retained state rejects legacy placement");
    CHECK(placement_fds_before >= 0 &&
              open_fd_count() == placement_fds_before,
          "post-second-RECOVERABLE rejection leaks no descriptor");

    /* The engine and exact compact owners intentionally remain retained until
     * this isolated process exits. */
    restore_forbidden_environment(&saved);
    return g_failures == 0 ? 0 : 1;
}

typedef struct {
    char path[64];
    int fd;
    unsigned char *model_map;
    ds4_laguna_ledger ledger;
    ds4_laguna_allocation_plan plan;
    tracker_fixture runtime;
    ds4_laguna_file_identity identity;
    ds4_gpu_laguna_compact *context;
    saved_environment saved;
    bool gpu_initialized;
    bool tracker_initialized;
} cache_cuda_fixture;

typedef enum {
    CACHE_LEDGER_VALID = 0,
    CACHE_LEDGER_WRONG_PARENT = 1,
    CACHE_LEDGER_WRONG_SUBRANGE = 2,
    CACHE_LEDGER_OVERLAPPING_DEVICE_VIEWS = 3,
    CACHE_LEDGER_INCONSISTENT_LAYER_EXPERT_COUNT = 4,
    CACHE_PLAN_UNDERSLOTTED = 5,
    CACHE_PLAN_TWO_SESSION_PRESSURE = 6,
} cache_boundary_mutation;

static void cache_ledger_make_layer_expert_counts_inconsistent(
        ds4_laguna_ledger *ledger) {
    const uint64_t expert_bytes = ledger->routed_projection_expert_bytes;
    const ds4_laguna_expert_entry layer_two_zero =
        ledger->expert_entries[2];
    const ds4_laguna_expert_entry layer_two_one =
        ledger->expert_entries[3];
    for (size_t i = 0; i < ledger->tensor_range_count; i++) {
        ds4_laguna_tensor_range *range = &ledger->tensor_ranges[i];
        if (range->tensor_class != DS4_LAGUNA_TENSOR_ROUTED_EXPERT) {
            continue;
        }
        range->source_bytes = range->routed_layer == 1u ?
            expert_bytes : 3u * expert_bytes;
    }
    ledger->expert_entries[1] = layer_two_zero;
    ledger->expert_entries[2] = layer_two_one;
    ledger->expert_entries[3] = layer_two_one;
    ledger->expert_entries[3].expert = 2u;
    ledger->expert_entries[3].gate.source_offset += expert_bytes;
    ledger->expert_entries[3].up.source_offset += expert_bytes;
    ledger->expert_entries[3].down.source_offset += expert_bytes;
}

static bool cache_cuda_fixture_open_mutated(
        cache_cuda_fixture *fixture,
        cache_boundary_mutation mutation) {
    memset(fixture, 0, sizeof(*fixture));
    fixture->fd = -1;
    fixture->model_map = MAP_FAILED;
    snprintf(fixture->path, sizeof(fixture->path),
             "/tmp/ds4-cuda-laguna-cache.XXXXXX");
    save_and_clear_forbidden_environment(&fixture->saved);

    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess ||
        device_count < 1 || !ds4_gpu_init()) {
        return false;
    }
    fixture->gpu_initialized = true;
    fixture->fd = create_pattern_file(
        fixture->path, FIXTURE_MODEL_BYTES, 11u);
    if (fixture->fd < 0 ||
        !capture_file_identity(fixture->fd, &fixture->identity)) {
        return false;
    }
    fixture->model_map = mmap(
        NULL, FIXTURE_MODEL_BYTES, PROT_READ, MAP_SHARED,
        fixture->fd, 0);
    if (fixture->model_map == MAP_FAILED ||
        !ledger_fixture_build(&fixture->ledger)) {
        return false;
    }
    plan_prepare_cache(&fixture->plan, &fixture->ledger);
    if (mutation == CACHE_LEDGER_WRONG_PARENT) {
        fixture->ledger.expert_entries[0].gate.parent_stable_index =
            fixture->ledger.expert_entries[0].up.parent_stable_index;
    } else if (mutation == CACHE_LEDGER_WRONG_SUBRANGE) {
        fixture->ledger.expert_entries[0].gate.source_offset++;
    } else if (mutation == CACHE_LEDGER_OVERLAPPING_DEVICE_VIEWS) {
        fixture->ledger.expert_entries[0].up.device_offset =
            fixture->ledger.expert_entries[0].gate.device_offset;
    } else if (mutation ==
                   CACHE_LEDGER_INCONSISTENT_LAYER_EXPERT_COUNT) {
        cache_ledger_make_layer_expert_counts_inconsistent(
            &fixture->ledger);
    } else if (mutation == CACHE_PLAN_UNDERSLOTTED) {
        fixture->plan.configured_cache_bytes +=
            fixture->ledger.slot_stride_bytes;
        fixture->plan.effective_cache_limit_bytes =
            fixture->plan.configured_cache_bytes;
        fixture->plan.cache_tail_uncharged_bytes =
            fixture->ledger.slot_stride_bytes;
    } else if (mutation == CACHE_PLAN_TWO_SESSION_PRESSURE) {
        fixture->plan.profile_id =
            "synthetic-4k-two-logical-actor-cache-pressure";
        fixture->plan.context_tokens = 4096u;
        fixture->plan.prefill_rows = 4096u;
        fixture->plan.session_count = 2u;
    }
    if (!tracker_fixture_init(
            &fixture->runtime, &fixture->plan, &fixture->ledger)) {
        return false;
    }
    fixture->tracker_initialized = true;
    return ds4_gpu_laguna_compact_create(
        &fixture->context,
        fixture->fd,
        fixture->model_map,
        FIXTURE_MODEL_BYTES,
        &fixture->identity,
        &fixture->ledger,
        &fixture->plan,
        &fixture->runtime.tracker) != 0 &&
        fixture->context != NULL;
}

static bool cache_cuda_fixture_open(cache_cuda_fixture *fixture) {
    return cache_cuda_fixture_open_mutated(fixture, CACHE_LEDGER_VALID);
}

static bool cache_cuda_fixture_open_session_pressure(
        cache_cuda_fixture *fixture) {
    return cache_cuda_fixture_open_mutated(
        fixture, CACHE_PLAN_TWO_SESSION_PRESSURE);
}

static void cache_cuda_fixture_close(cache_cuda_fixture *fixture) {
    if (fixture->context) {
        const ds4_gpu_laguna_destroy_status destroyed =
            ds4_gpu_laguna_compact_destroy(fixture->context);
        if (destroyed == DS4_GPU_LAGUNA_DESTROY_OK) {
            fixture->context = NULL;
        } else {
            CHECK(false,
                  "cache fixture cleanup retains a non-quiescent context");
            /* The compact context still owns or references every object below.
             * Leave the CUDA runtime, tracker, ledger, mapping, and descriptors
             * intact for process exit rather than turning the original failure
             * into cleanup-time use-after-free. */
            if (fixture->path[0]) {
                unlink(fixture->path);
                fixture->path[0] = '\0';
            }
            restore_forbidden_environment(&fixture->saved);
            return;
        }
    }
    if (fixture->gpu_initialized) ds4_gpu_cleanup();
    if (fixture->tracker_initialized) {
        (void)tracker_fixture_release_ledger(&fixture->runtime);
    }
    ds4_laguna_ledger_free(&fixture->ledger);
    if (fixture->model_map != MAP_FAILED) {
        munmap(fixture->model_map, FIXTURE_MODEL_BYTES);
    }
    if (fixture->fd >= 0) close(fixture->fd);
    if (fixture->path[0]) unlink(fixture->path);
    restore_forbidden_environment(&fixture->saved);
}

static ds4_laguna_expert_key cache_fixture_key(
        uint32_t layer,
        uint32_t expert) {
    const ds4_laguna_expert_key key = {
        .layer_id = layer,
        .expert_id = expert,
    };
    return key;
}

static bool cache_projection_matches_source(
        cache_cuda_fixture *fixture,
        ds4_laguna_cache_handle handle,
        ds4_laguna_routed_projection projection,
        const ds4_laguna_expert_view *view) {
    const void *device_ptr = NULL;
    uint64_t bytes = 0;
    unsigned char expected[128];
    unsigned char actual[128];
    if (!view || view->source_bytes > sizeof(expected) ||
        !ds4_gpu_laguna_compact_cache_view(
            fixture->context, handle, projection,
            &device_ptr, &bytes) ||
        !device_ptr || bytes != view->source_bytes ||
        pread(fixture->fd, expected, (size_t)bytes,
              (off_t)view->source_offset) != (ssize_t)bytes ||
        cudaMemcpy(actual, device_ptr, (size_t)bytes,
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
    }
    return memcmp(actual, expected, (size_t)bytes) == 0;
}

static bool cache_device_map_matches_slots(
        cache_cuda_fixture *fixture,
        const ds4_gpu_laguna_compact_test_snapshot *snapshot,
        size_t expected_resident) {
    if (!fixture || !snapshot || !snapshot->cache_slots ||
        !snapshot->device_entry_to_slot || snapshot->cache_slot_count == 0) {
        return false;
    }
    const size_t count = (size_t)fixture->ledger.expert_entry_count;
    uint32_t *mapping = calloc(count, sizeof(*mapping));
    bool *slot_mapped = calloc(
        (size_t)snapshot->cache_slot_count, sizeof(*slot_mapped));
    if (!mapping || !slot_mapped ||
        cudaMemcpy(mapping, snapshot->device_entry_to_slot,
                   count * sizeof(*mapping),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        free(mapping);
        free(slot_mapped);
        return false;
    }
    size_t resident = 0;
    bool matches = true;
    for (size_t entry_index = 0; entry_index < count; entry_index++) {
        const uint32_t slot_index = mapping[entry_index];
        if (slot_index == DS4_LAGUNA_CACHE_SLOT_NONE) continue;
        if (slot_index >= snapshot->cache_slot_count ||
            slot_mapped[slot_index]) {
            matches = false;
            break;
        }
        const ds4_laguna_cache_slot *slot =
            &snapshot->cache_slots[slot_index];
        const ds4_laguna_expert_entry *entry =
            &fixture->ledger.expert_entries[entry_index];
        if ((slot->state != DS4_LAGUNA_CACHE_SLOT_READY &&
             slot->state != DS4_LAGUNA_CACHE_SLOT_IN_USE) ||
            slot->layer != entry->layer || slot->expert != entry->expert) {
            matches = false;
            break;
        }
        slot_mapped[slot_index] = true;
        resident++;
    }
    if (matches) {
        for (size_t slot_index = 0;
             slot_index < (size_t)snapshot->cache_slot_count; slot_index++) {
            const ds4_laguna_cache_slot *slot =
                &snapshot->cache_slots[slot_index];
            const bool should_be_mapped =
                slot->state == DS4_LAGUNA_CACHE_SLOT_READY ||
                slot->state == DS4_LAGUNA_CACHE_SLOT_IN_USE;
            if (slot_mapped[slot_index] != should_be_mapped) {
                matches = false;
                break;
            }
        }
    }
    free(mapping);
    free(slot_mapped);
    return matches && resident == expected_resident;
}

static bool cache_snapshot_has_no_fallback(
        const ds4_gpu_laguna_compact_test_snapshot *snapshot) {
    return snapshot && snapshot->model_mapping_registered_bytes == 0 &&
        snapshot->whole_model_copied_bytes == 0 &&
        snapshot->opportunistic_range_allocated_bytes == 0 &&
        snapshot->legacy_model_range_count == 0 &&
        snapshot->legacy_model_arena_count == 0;
}

static bool cache_snapshot_keeps_fixed_allocations(
        const ds4_gpu_laguna_compact_test_snapshot *baseline,
        const ds4_gpu_laguna_compact_test_snapshot *snapshot) {
    return baseline && snapshot && snapshot->cache_payload_live &&
        snapshot->cache_policy_live &&
        snapshot->device_entry_to_slot_live &&
        snapshot->page_advice_state_live &&
        snapshot->cache_payload == baseline->cache_payload &&
        snapshot->cache_slots == baseline->cache_slots &&
        snapshot->device_entry_to_slot == baseline->device_entry_to_slot &&
        snapshot->page_advice_state == baseline->page_advice_state &&
        snapshot->page_advice_state_bytes ==
            baseline->page_advice_state_bytes &&
        snapshot->cache_payload_bytes == baseline->cache_payload_bytes &&
        snapshot->cache_slot_count == baseline->cache_slot_count &&
        snapshot->cache_slot_stride_bytes ==
            baseline->cache_slot_stride_bytes &&
        snapshot->pinned_staging_live_count ==
            baseline->pinned_staging_live_count &&
        snapshot->pinned_staging_bytes == baseline->pinned_staging_bytes &&
        snapshot->cache_payload_allocation_attempts ==
            baseline->cache_payload_allocation_attempts &&
        snapshot->pinned_staging_allocation_attempts ==
            baseline->pinned_staging_allocation_attempts;
}

static bool cache_slots_match_residents(
        const ds4_gpu_laguna_compact_test_snapshot *snapshot,
        const ds4_laguna_expert_key *expected,
        size_t expected_count) {
    bool seen[16] = {false};
    if (!snapshot || !snapshot->cache_slots ||
        expected_count > ARRAY_LEN(seen) ||
        (expected_count != 0 && !expected)) {
        return false;
    }
    size_t resident_count = 0;
    for (size_t slot_index = 0;
         slot_index < (size_t)snapshot->cache_slot_count; slot_index++) {
        const ds4_laguna_cache_slot *slot =
            &snapshot->cache_slots[slot_index];
        if (slot->state == DS4_LAGUNA_CACHE_SLOT_EMPTY) {
            if (slot->refs != 0 || slot->layer != UINT32_MAX ||
                slot->expert != UINT32_MAX) {
                return false;
            }
            continue;
        }
        if ((slot->state != DS4_LAGUNA_CACHE_SLOT_READY &&
             slot->state != DS4_LAGUNA_CACHE_SLOT_IN_USE) ||
            (slot->state == DS4_LAGUNA_CACHE_SLOT_READY && slot->refs != 0) ||
            (slot->state == DS4_LAGUNA_CACHE_SLOT_IN_USE && slot->refs == 0)) {
            return false;
        }
        size_t found = SIZE_MAX;
        for (size_t i = 0; i < expected_count; i++) {
            if (expected[i].layer_id == slot->layer &&
                expected[i].expert_id == slot->expert) {
                found = i;
                break;
            }
        }
        if (found == SIZE_MAX || seen[found]) return false;
        seen[found] = true;
        resident_count++;
    }
    if (resident_count != expected_count) return false;
    for (size_t i = 0; i < expected_count; i++) {
        if (!seen[i]) return false;
    }
    return true;
}

static bool cache_snapshot_matches_residents(
        cache_cuda_fixture *fixture,
        const ds4_gpu_laguna_compact_test_snapshot *snapshot,
        const ds4_laguna_expert_key *expected,
        size_t expected_count,
        uint64_t expected_source_bytes) {
    return snapshot && snapshot->routed_payload_bytes == expected_source_bytes &&
        cache_slots_match_residents(snapshot, expected, expected_count) &&
        cache_device_map_matches_slots(fixture, snapshot, expected_count);
}

static bool cache_find_only_loading(
        cache_cuda_fixture *fixture,
        const ds4_gpu_laguna_compact_test_snapshot *snapshot,
        ds4_laguna_expert_key expected,
        ds4_laguna_cache_handle *handle) {
    if (handle) {
        memset(handle, 0, sizeof(*handle));
        handle->slot_index = DS4_LAGUNA_CACHE_SLOT_NONE;
        handle->entry_index = SIZE_MAX;
    }
    if (!fixture || !snapshot || !snapshot->cache_slots || !handle) {
        return false;
    }
    size_t entry_index = SIZE_MAX;
    for (size_t i = 0;
         i < (size_t)fixture->ledger.expert_entry_count; i++) {
        const ds4_laguna_expert_entry *entry =
            &fixture->ledger.expert_entries[i];
        if (entry->layer == expected.layer_id &&
            entry->expert == expected.expert_id) {
            entry_index = i;
            break;
        }
    }
    if (entry_index == SIZE_MAX) return false;
    size_t loading_count = 0;
    for (size_t slot_index = 0;
         slot_index < (size_t)snapshot->cache_slot_count; slot_index++) {
        const ds4_laguna_cache_slot *slot =
            &snapshot->cache_slots[slot_index];
        if (slot->state == DS4_LAGUNA_CACHE_SLOT_EMPTY) {
            if (slot->refs != 0 || slot->layer != UINT32_MAX ||
                slot->expert != UINT32_MAX) {
                return false;
            }
            continue;
        }
        if (slot->state != DS4_LAGUNA_CACHE_SLOT_LOADING ||
            slot->refs != 0 || slot->generation == 0 ||
            slot->layer != expected.layer_id ||
            slot->expert != expected.expert_id || loading_count != 0) {
            return false;
        }
        loading_count++;
        handle->slot_index = (uint32_t)slot_index;
        handle->generation = slot->generation;
        handle->entry_index = entry_index;
        handle->key = expected;
    }
    return loading_count == 1u;
}

typedef struct {
    ds4_gpu_laguna_compact *context;
    ds4_laguna_cache_handle handle;
    ds4_laguna_cache_status status;
} cache_complete_probe;

static void *cache_complete_probe_run(void *opaque) {
    cache_complete_probe *load = opaque;
    load->status = ds4_gpu_laguna_compact_cache_complete(
        load->context, &load->handle);
    return NULL;
}

typedef struct {
    ds4_gpu_laguna_compact *context;
    ds4_laguna_expert_key key;
    ds4_laguna_cache_handle handle;
    ds4_laguna_cache_acquire_outcome outcome;
    ds4_laguna_cache_status status;
} cache_begin_probe;

static void *cache_begin_probe_run(void *opaque) {
    cache_begin_probe *begin = opaque;
    begin->status = ds4_gpu_laguna_compact_cache_begin(
        begin->context, begin->key, &begin->handle, &begin->outcome);
    return NULL;
}

static int run_cache_validation(void) {
    const cache_boundary_mutation mutations[] = {
        CACHE_LEDGER_WRONG_PARENT,
        CACHE_LEDGER_WRONG_SUBRANGE,
        CACHE_LEDGER_OVERLAPPING_DEVICE_VIEWS,
        CACHE_LEDGER_INCONSISTENT_LAYER_EXPERT_COUNT,
        CACHE_PLAN_UNDERSLOTTED,
    };
    const char *const messages[] = {
        "cache boundary rejects a view with the wrong routed parent",
        "cache boundary rejects a view outside its exact expert subrange",
        "cache boundary rejects overlapping projection device views",
        "cache boundary rejects inconsistent expert counts across layers",
        "cache boundary rejects slot counts below configured capacity",
    };
    for (size_t i = 0; i < ARRAY_LEN(mutations); i++) {
        cache_cuda_fixture fixture;
        const uint64_t attempts_before =
            ds4_gpu_test_laguna_compact_static_allocation_attempts();
        const bool opened = cache_cuda_fixture_open_mutated(
            &fixture, mutations[i]);
        CHECK(!opened && fixture.context == NULL, messages[i]);
        CHECK(ds4_gpu_test_laguna_compact_static_allocation_attempts() ==
                  attempts_before,
              "invalid cache expert geometry fails before CUDA allocation");
        cache_cuda_fixture_close(&fixture);
    }
    return g_failures == 0 ? 0 : 1;
}

static int run_cache_io(void) {
    cache_cuda_fixture fixture;
    CHECK(cache_cuda_fixture_open(&fixture),
          "cache I/O fixture creates a compact CUDA context");
    if (!fixture.context) {
        cache_cuda_fixture_close(&fixture);
        return 1;
    }

    ds4_gpu_laguna_compact_test_snapshot initial;
    memset(&initial, 0, sizeof(initial));
    CHECK(ds4_gpu_test_laguna_compact_snapshot(
              fixture.context, &initial),
          "cache I/O exposes its fixed-allocation snapshot");
    CHECK(ds4_gpu_test_laguna_compact_snapshot_size() == sizeof(initial),
          "public and CUDA snapshot layouts remain byte-identical");
    CHECK(initial.cache_payload_live && initial.cache_policy_live &&
              initial.device_entry_to_slot_live &&
              initial.page_advice_state_live &&
              initial.cache_payload != NULL &&
              initial.cache_slots != NULL &&
              initial.device_entry_to_slot != NULL &&
              initial.page_advice_state != NULL &&
              initial.page_advice_state_bytes ==
                  (fixture.ledger.static_parent_count +
                       3u * fixture.ledger.expert_entry_count) *
                      3u * sizeof(ds4_laguna_page_range) &&
              initial.cache_payload_bytes ==
                  fixture.plan.cache_payload_bytes &&
              initial.cache_payload_bytes <=
                  fixture.plan.configured_cache_bytes &&
              initial.cache_slot_count == fixture.plan.slot_count &&
              initial.cache_slot_stride_bytes ==
                  fixture.plan.slot_stride_bytes &&
              initial.routed_payload_bytes == 0,
          "cache payload is one fixed plan-bounded slot slab");
    uint32_t device_entry_to_slot[4] = {0, 0, 0, 0};
    CHECK(fixture.ledger.expert_entry_count ==
              ARRAY_LEN(device_entry_to_slot) &&
              cudaMemcpy(
                  device_entry_to_slot,
                  initial.device_entry_to_slot,
                  sizeof(device_entry_to_slot),
                  cudaMemcpyDeviceToHost) == cudaSuccess &&
              device_entry_to_slot[0] == DS4_LAGUNA_CACHE_SLOT_NONE &&
              device_entry_to_slot[1] == DS4_LAGUNA_CACHE_SLOT_NONE &&
              device_entry_to_slot[2] == DS4_LAGUNA_CACHE_SLOT_NONE &&
              device_entry_to_slot[3] == DS4_LAGUNA_CACHE_SLOT_NONE,
          "unpublished device reverse map starts entirely at NONE");
    CHECK(initial.pinned_staging_live_count == 4u &&
              initial.pinned_staging_bytes ==
                  4u * fixture.plan.staging_buffer_bytes &&
              initial.cache_payload_allocation_attempts == 1u &&
              initial.pinned_staging_allocation_attempts == 4u,
          "exactly four fixed staging buffers accompany one payload allocation");

    ds4_runtime_snapshot runtime_before;
    ds4_runtime_allocation_record active_before[TRACKER_RECORD_CAPACITY];
    CHECK(ds4_runtime_tracker_snapshot_copy(
              &fixture.runtime.tracker, &runtime_before,
              active_before, ARRAY_LEN(active_before)) &&
              runtime_before.violation == DS4_RUNTIME_VIOLATION_NONE &&
              runtime_before.active_record_count == 16u &&
              runtime_before.category_current[
                  DS4_RUNTIME_CATEGORY_EXPERT_CACHE_PAYLOAD] ==
                  fixture.plan.cache_payload_bytes &&
              runtime_before.category_current[
                  DS4_RUNTIME_CATEGORY_PINNED_STAGING] ==
                  4u * fixture.plan.staging_buffer_bytes &&
              runtime_before.category_current[
                  DS4_RUNTIME_CATEGORY_OTHER_HOST] ==
                  initial.page_advice_state_bytes,
          "fixed cache and staging allocations reconcile through the tracker");
    const ds4_runtime_allocation_record *page_advice_state_record = NULL;
    size_t page_advice_state_record_count = 0;
    for (size_t i = 0; i < runtime_before.active_record_count; i++) {
        if (active_before[i].callsite_id !=
                DS4_LAGUNA_CALLSITE_OTHER_HOST_TRACKER) {
            continue;
        }
        page_advice_state_record = &active_before[i];
        page_advice_state_record_count++;
    }
    CHECK(page_advice_state_record_count == 1u &&
              owned_record_matches(
                  page_advice_state_record,
                  (uint64_t)(uintptr_t)initial.page_advice_state,
                  initial.page_advice_state_bytes,
                  DS4_LAGUNA_CALLSITE_OTHER_HOST_TRACKER,
                  DS4_RUNTIME_CATEGORY_OTHER_HOST,
                  DS4_RUNTIME_DOMAIN_HOST),
          "page-advice state has one exact fixed host allocation record");

    ds4_laguna_cache_handle first = {0};
    ds4_laguna_cache_acquire_outcome outcome =
        DS4_LAGUNA_CACHE_ACQUIRE_NONE;
    const bool first_loaded =
        ds4_gpu_laguna_compact_cache_acquire(
            fixture.context, cache_fixture_key(1u, 0u),
            &first, &outcome) == DS4_LAGUNA_CACHE_OK &&
        outcome == DS4_LAGUNA_CACHE_ACQUIRE_LOAD_OWNER &&
        first.entry_index < fixture.ledger.expert_entry_count;
    CHECK(first_loaded,
          "first expert acquire performs one bounded cache load");
    if (!first_loaded) {
        cache_cuda_fixture_close(&fixture);
        return 1;
    }
    const ds4_laguna_expert_entry *entry =
        &fixture.ledger.expert_entries[first.entry_index];
    CHECK(cache_projection_matches_source(
              &fixture, first, DS4_LAGUNA_ROUTED_PROJECTION_GATE,
              &entry->gate) &&
              cache_projection_matches_source(
                  &fixture, first, DS4_LAGUNA_ROUTED_PROJECTION_UP,
                  &entry->up) &&
              cache_projection_matches_source(
                  &fixture, first, DS4_LAGUNA_ROUTED_PROJECTION_DOWN,
                  &entry->down),
          "published slot contains exact gate/up/down file subranges");

    ds4_gpu_laguna_compact_test_snapshot after_first;
    memset(&after_first, 0, sizeof(after_first));
    CHECK(ds4_gpu_test_laguna_compact_snapshot(
              fixture.context, &after_first) &&
              after_first.cache_acquire_misses == 1u &&
              after_first.cache_load_successes == 1u &&
              after_first.model_file_read_calls == 3u &&
              after_first.model_file_read_bytes ==
                  entry->gate.source_bytes + entry->up.source_bytes +
                  entry->down.source_bytes &&
              after_first.routed_payload_bytes ==
                  entry->gate.source_bytes + entry->up.source_bytes +
                  entry->down.source_bytes,
          "first miss accounts exactly three projection reads");
    CHECK(cache_device_map_matches_slots(&fixture, &after_first, 1u),
          "successful publication commits one coherent device reverse map");
    CHECK(ds4_gpu_laguna_compact_cache_unpin(
              fixture.context, first) == DS4_LAGUNA_CACHE_OK,
          "first loaded slot returns to reusable READY state");

    ds4_laguna_cache_handle hit = {0};
    outcome = DS4_LAGUNA_CACHE_ACQUIRE_NONE;
    CHECK(ds4_gpu_laguna_compact_cache_acquire(
              fixture.context, cache_fixture_key(1u, 0u),
              &hit, &outcome) == DS4_LAGUNA_CACHE_OK &&
              outcome == DS4_LAGUNA_CACHE_ACQUIRE_HIT_RESERVED,
          "second acquire of the same expert is a cache hit");
    ds4_gpu_laguna_compact_test_snapshot after_hit;
    memset(&after_hit, 0, sizeof(after_hit));
    CHECK(ds4_gpu_test_laguna_compact_snapshot(
              fixture.context, &after_hit) &&
              after_hit.cache_acquire_hits == 1u &&
              after_hit.model_file_read_calls ==
                  after_first.model_file_read_calls &&
              after_hit.model_file_read_bytes ==
                  after_first.model_file_read_bytes,
          "cache hit performs zero additional model-file I/O");
    CHECK(ds4_gpu_laguna_compact_cache_unpin(
              fixture.context, hit) == DS4_LAGUNA_CACHE_OK,
          "hit reservation unpins cleanly");

    const ds4_laguna_expert_key pressure_keys[] = {
        {1u, 1u}, {2u, 0u}, {2u, 1u}, {1u, 0u},
    };
    for (size_t i = 0; i < ARRAY_LEN(pressure_keys); i++) {
        ds4_laguna_cache_handle handle = {0};
        outcome = DS4_LAGUNA_CACHE_ACQUIRE_NONE;
        CHECK(ds4_gpu_laguna_compact_cache_acquire(
                  fixture.context, pressure_keys[i],
                  &handle, &outcome) == DS4_LAGUNA_CACHE_OK,
              "cache reuses fixed slots under eviction pressure");
        CHECK(ds4_gpu_laguna_compact_cache_unpin(
                  fixture.context, handle) == DS4_LAGUNA_CACHE_OK,
              "pressure load releases its slot reservation");
    }
    ds4_gpu_laguna_compact_test_snapshot after_pressure;
    ds4_runtime_snapshot runtime_after;
    ds4_runtime_allocation_record active_after[TRACKER_RECORD_CAPACITY];
    memset(&after_pressure, 0, sizeof(after_pressure));
    CHECK(ds4_gpu_test_laguna_compact_snapshot(
              fixture.context, &after_pressure) &&
              after_pressure.cache_payload_allocation_attempts == 1u &&
              after_pressure.pinned_staging_allocation_attempts == 4u &&
              after_pressure.cache_evictions == 3u &&
              after_pressure.cache_payload_bytes ==
                  initial.cache_payload_bytes &&
              after_pressure.pinned_staging_bytes ==
                  initial.pinned_staging_bytes &&
              after_pressure.routed_payload_bytes ==
                  2u * (entry->gate.source_bytes +
                        entry->up.source_bytes +
                        entry->down.source_bytes) &&
              after_pressure.routed_payload_bytes <=
                  after_pressure.cache_payload_bytes,
          "pressure cannot grow cache payload or staging allocations");
    CHECK(cache_device_map_matches_slots(&fixture, &after_pressure, 2u),
          "eviction pressure leaves only current residents in the device map");
    CHECK(ds4_runtime_tracker_snapshot_copy(
              &fixture.runtime.tracker, &runtime_after,
              active_after, ARRAY_LEN(active_after)) &&
              runtime_after.owned_total_current ==
                  runtime_before.owned_total_current &&
              runtime_after.category_peak[
                  DS4_RUNTIME_CATEGORY_EXPERT_CACHE_PAYLOAD] <=
                  fixture.plan.owned_category_bounds[
                      DS4_RUNTIME_CATEGORY_EXPERT_CACHE_PAYLOAD] &&
              runtime_after.category_peak[
                  DS4_RUNTIME_CATEGORY_PINNED_STAGING] <=
                  fixture.plan.owned_category_bounds[
                      DS4_RUNTIME_CATEGORY_PINNED_STAGING],
          "tracker remains fixed and below plan ceilings under pressure");

    const ds4_gpu_laguna_destroy_status cache_io_destroyed =
        ds4_gpu_laguna_compact_destroy(fixture.context);
    CHECK(cache_io_destroyed == DS4_GPU_LAGUNA_DESTROY_OK,
          "quiescent fixed cache tears down safely");
    if (cache_io_destroyed == DS4_GPU_LAGUNA_DESTROY_OK) {
        fixture.context = NULL;
    }
    CHECK(cache_io_destroyed == DS4_GPU_LAGUNA_DESTROY_OK &&
              tracker_has_only_ledger(&fixture.runtime.tracker),
          "cache teardown restores the ledger-only accounting baseline");

    const bool recreated = cache_io_destroyed == DS4_GPU_LAGUNA_DESTROY_OK &&
        ds4_gpu_laguna_compact_create(
            &fixture.context, fixture.fd, fixture.model_map,
            FIXTURE_MODEL_BYTES, &fixture.identity, &fixture.ledger,
            &fixture.plan, &fixture.runtime.tracker) != 0 &&
        fixture.context != NULL;
    CHECK(recreated,
          "cache context can recreate after a complete tracked teardown");
    if (recreated) {
        ds4_laguna_cache_handle recreated_owner = {0};
        outcome = DS4_LAGUNA_CACHE_ACQUIRE_NONE;
        CHECK(ds4_gpu_laguna_compact_cache_begin(
                  fixture.context, cache_fixture_key(1u, 0u),
                  &recreated_owner, &outcome) == DS4_LAGUNA_CACHE_OK &&
                  outcome == DS4_LAGUNA_CACHE_ACQUIRE_LOAD_OWNER &&
                  recreated_owner.slot_index == first.slot_index &&
                  recreated_owner.generation == first.generation &&
                  recreated_owner.entry_index == first.entry_index &&
                  recreated_owner.lifecycle_epoch != 0 &&
                  recreated_owner.lifecycle_epoch != first.lifecycle_epoch,
              "recreated lifecycle can reuse the exact slot generation without reusing its epoch");
        const void *stale_view = (const void *)(uintptr_t)1u;
        uint64_t stale_bytes = 1u;
        CHECK(ds4_gpu_laguna_compact_cache_cancel(
                  fixture.context, first) == DS4_LAGUNA_CACHE_RECOVERABLE &&
                  !ds4_gpu_laguna_compact_cache_view(
                      fixture.context, first,
                      DS4_LAGUNA_ROUTED_PROJECTION_GATE,
                      &stale_view, &stale_bytes) &&
                  stale_view == NULL && stale_bytes == 0 &&
                  ds4_gpu_laguna_compact_cache_unpin(
                      fixture.context, first) ==
                      DS4_LAGUNA_CACHE_RECOVERABLE,
              "stale lifecycle handles cannot cancel, view, or unpin a colliding owner");
        CHECK(ds4_gpu_laguna_compact_cache_complete(
                  fixture.context, &recreated_owner) ==
                      DS4_LAGUNA_CACHE_OK,
              "current lifecycle owner still publishes after stale-handle probes");
        const ds4_laguna_expert_entry *recreated_entry =
            &fixture.ledger.expert_entries[recreated_owner.entry_index];
        CHECK(cache_projection_matches_source(
                  &fixture, recreated_owner,
                  DS4_LAGUNA_ROUTED_PROJECTION_GATE,
                  &recreated_entry->gate) &&
                  ds4_gpu_laguna_compact_cache_unpin(
                      fixture.context, recreated_owner) ==
                      DS4_LAGUNA_CACHE_OK,
              "recreated lifecycle retains the current owner's exact payload and pin");
        cache_begin_probe queued_begin = {
            .context = fixture.context,
            .key = cache_fixture_key(2u, 0u),
            .outcome = DS4_LAGUNA_CACHE_ACQUIRE_NONE,
            .status = DS4_LAGUNA_CACHE_UNSAFE,
        };
        pthread_t begin_thread;
        ds4_gpu_test_laguna_compact_pause_cache_begin_once();
        const bool begin_started = pthread_create(
            &begin_thread, NULL, cache_begin_probe_run,
            &queued_begin) == 0;
        const bool begin_paused = begin_started &&
            ds4_gpu_test_laguna_compact_wait_cache_begin_paused();
        CHECK(begin_paused,
              "cache begin captures its lifecycle before waiting for the mutex");

        const ds4_gpu_laguna_destroy_status recreated_destroyed =
            ds4_gpu_laguna_compact_destroy(fixture.context);
        CHECK(recreated_destroyed == DS4_GPU_LAGUNA_DESTROY_OK,
              "queued begin does not prevent its original lifecycle teardown");
        if (recreated_destroyed == DS4_GPU_LAGUNA_DESTROY_OK) {
            fixture.context = NULL;
        }
        const bool third_created =
            recreated_destroyed == DS4_GPU_LAGUNA_DESTROY_OK &&
            ds4_gpu_laguna_compact_create(
                &fixture.context, fixture.fd, fixture.model_map,
                FIXTURE_MODEL_BYTES, &fixture.identity, &fixture.ledger,
                &fixture.plan, &fixture.runtime.tracker) != 0 &&
            fixture.context != NULL;
        CHECK(third_created,
              "a new compact lifecycle can win before queued begin resumes");
        ds4_gpu_laguna_compact_test_snapshot before_queued_resume;
        memset(&before_queued_resume, 0, sizeof(before_queued_resume));
        CHECK(third_created &&
                  ds4_gpu_test_laguna_compact_snapshot(
                      fixture.context, &before_queued_resume) &&
                  before_queued_resume.cache_acquire_misses == 0 &&
                  before_queued_resume.routed_payload_bytes == 0 &&
                  cache_device_map_matches_slots(
                      &fixture, &before_queued_resume, 0u),
              "replacement lifecycle starts with an untouched empty cache");

        if (begin_started) {
            ds4_gpu_test_laguna_compact_resume_cache_begin();
        }
        CHECK(begin_started && pthread_join(begin_thread, NULL) == 0 &&
                  queued_begin.status == DS4_LAGUNA_CACHE_RECOVERABLE &&
                  queued_begin.outcome == DS4_LAGUNA_CACHE_ACQUIRE_NONE &&
                  queued_begin.handle.slot_index ==
                      DS4_LAGUNA_CACHE_SLOT_NONE &&
                  queued_begin.handle.lifecycle_epoch == 0,
              "queued begin refuses to cross into a replacement lifecycle");
        ds4_gpu_laguna_compact_test_snapshot after_queued_resume;
        memset(&after_queued_resume, 0, sizeof(after_queued_resume));
        CHECK(third_created &&
                  ds4_gpu_test_laguna_compact_snapshot(
                      fixture.context, &after_queued_resume) &&
                  after_queued_resume.cache_acquire_misses == 0 &&
                  after_queued_resume.routed_payload_bytes == 0 &&
                  cache_device_map_matches_slots(
                      &fixture, &after_queued_resume, 0u),
              "stale queued begin cannot mutate replacement cache state");
        if (third_created) {
            const ds4_gpu_laguna_destroy_status third_destroyed =
                ds4_gpu_laguna_compact_destroy(fixture.context);
            CHECK(third_destroyed == DS4_GPU_LAGUNA_DESTROY_OK,
                  "replacement lifecycle tears down after ABA refusal");
            if (third_destroyed == DS4_GPU_LAGUNA_DESTROY_OK) {
                fixture.context = NULL;
            }
        }
    }
    CHECK(fixture.context == NULL &&
              tracker_has_only_ledger(&fixture.runtime.tracker),
          "recreated lifecycle returns to the same ledger-only baseline");
    const int result = g_failures == 0 ? 0 : 1;
    cache_cuda_fixture_close(&fixture);
    return result;
}

static bool cache_slots_are_quiescent_ready(
        const ds4_gpu_laguna_compact_test_snapshot *snapshot) {
    if (!snapshot || !snapshot->cache_slots ||
        snapshot->cache_slot_count == 0) {
        return false;
    }
    for (size_t i = 0; i < (size_t)snapshot->cache_slot_count; i++) {
        const ds4_laguna_cache_slot *slot = &snapshot->cache_slots[i];
        if (slot->state != DS4_LAGUNA_CACHE_SLOT_READY || slot->refs != 0) {
            return false;
        }
    }
    return true;
}

static bool session_pressure_grouping_contract(void) {
    ds4_laguna_expert_entry entries[4];
    ds4_laguna_cache_slot slots[2];
    uint64_t hotness[4];
    uint32_t entry_to_slot[4];
    ds4_laguna_cache_policy policy;
    memset(entries, 0, sizeof(entries));
    memset(slots, 0, sizeof(slots));
    memset(hotness, 0, sizeof(hotness));
    memset(entry_to_slot, 0xff, sizeof(entry_to_slot));
    memset(&policy, 0, sizeof(policy));
    for (uint32_t expert = 0; expert < ARRAY_LEN(entries); expert++) {
        entries[expert].layer = 1u;
        entries[expert].expert = expert;
    }
    if (ds4_laguna_cache_policy_init(
            &policy, entries, ARRAY_LEN(entries), slots, ARRAY_LEN(slots),
            hotness, entry_to_slot, 2u) != DS4_LAGUNA_CACHE_OK) {
        return false;
    }

    /* Two synthetic logical actors contribute one two-expert token each.
     * Each token fits, while the combined policy working set exceeds slots. */
    const ds4_laguna_expert_key selected[] = {
        {1u, 0u}, {1u, 1u},
        {1u, 2u}, {1u, 3u},
    };
    ds4_laguna_expert_key grouped[4];
    ds4_laguna_expert_group groups[2];
    size_t grouped_count = 0;
    size_t group_count = 0;
    memset(grouped, 0, sizeof(grouped));
    memset(groups, 0, sizeof(groups));
    return ds4_laguna_cache_policy_group(
               &policy, selected, 2u, 2u,
               grouped, ARRAY_LEN(grouped), groups, ARRAY_LEN(groups),
               &grouped_count, &group_count) == DS4_LAGUNA_CACHE_OK &&
        grouped_count == ARRAY_LEN(selected) && group_count == 2u &&
        groups[0].first_key == 0u && groups[0].key_count == 2u &&
        groups[1].first_key == 2u && groups[1].key_count == 2u &&
        memcmp(grouped, selected, sizeof(selected)) == 0;
}

static int run_session_pressure(void) {
    cache_cuda_fixture fixture;
    CHECK(cache_cuda_fixture_open_session_pressure(&fixture),
          "synthetic 4K/two-logical-actor pressure fixture creates one cache");
    if (!fixture.context) {
        cache_cuda_fixture_close(&fixture);
        return 1;
    }
    CHECK(fixture.plan.context_tokens == 4096u &&
              fixture.plan.prefill_rows == 4096u &&
              fixture.plan.session_count == 2u &&
              fixture.plan.slot_count == 2u,
          "pressure metadata declares 4K, two logical actors, and two slots");

    ds4_gpu_laguna_compact_test_snapshot initial;
    ds4_runtime_snapshot runtime_initial;
    ds4_runtime_allocation_record initial_records[TRACKER_RECORD_CAPACITY];
    memset(&initial, 0, sizeof(initial));
    memset(&runtime_initial, 0, sizeof(runtime_initial));
    memset(initial_records, 0, sizeof(initial_records));
    CHECK(ds4_gpu_test_laguna_compact_snapshot(fixture.context, &initial) &&
              ds4_runtime_tracker_snapshot_copy(
                  &fixture.runtime.tracker, &runtime_initial,
                  initial_records, ARRAY_LEN(initial_records)) &&
              runtime_initial.violation == DS4_RUNTIME_VIOLATION_NONE &&
              initial.cache_payload_allocation_attempts == 1u &&
              initial.pinned_staging_allocation_attempts == 4u &&
              initial.cache_slot_count == 2u &&
              cache_device_map_matches_slots(&fixture, &initial, 0u),
          "pressure starts with one fixed engine-lifetime cache and no residents");

    /* Actor A and actor B interleave cold misses and deliberately keep both
     * handles pinned.  A third miss must report pressure, never grow capacity. */
    ds4_laguna_cache_handle actor_a = {0};
    ds4_laguna_cache_handle actor_b = {0};
    ds4_laguna_cache_handle pressure = {0};
    ds4_laguna_cache_acquire_outcome outcome =
        DS4_LAGUNA_CACHE_ACQUIRE_NONE;
    CHECK(ds4_gpu_laguna_compact_cache_acquire(
              fixture.context, cache_fixture_key(1u, 0u),
              &actor_a, &outcome) == DS4_LAGUNA_CACHE_OK &&
              outcome == DS4_LAGUNA_CACHE_ACQUIRE_LOAD_OWNER,
          "logical actor A owns the first cold miss");
    outcome = DS4_LAGUNA_CACHE_ACQUIRE_NONE;
    CHECK(ds4_gpu_laguna_compact_cache_acquire(
              fixture.context, cache_fixture_key(1u, 1u),
              &actor_b, &outcome) == DS4_LAGUNA_CACHE_OK &&
              outcome == DS4_LAGUNA_CACHE_ACQUIRE_LOAD_OWNER,
          "logical actor B interleaves a second cold miss");
    outcome = DS4_LAGUNA_CACHE_ACQUIRE_NONE;
    CHECK(ds4_gpu_laguna_compact_cache_begin(
              fixture.context, cache_fixture_key(2u, 0u),
              &pressure, &outcome) == DS4_LAGUNA_CACHE_RECOVERABLE &&
              outcome == DS4_LAGUNA_CACHE_ACQUIRE_PRESSURE &&
              pressure.slot_index == DS4_LAGUNA_CACHE_SLOT_NONE,
          "held pins turn an interleaved third miss into typed pressure");

    ds4_gpu_laguna_compact_test_snapshot pinned;
    memset(&pinned, 0, sizeof(pinned));
    CHECK(ds4_gpu_test_laguna_compact_snapshot(fixture.context, &pinned) &&
              pinned.cache_slot_count == 2u &&
              pinned.cache_slots[0].state == DS4_LAGUNA_CACHE_SLOT_IN_USE &&
              pinned.cache_slots[0].refs == 1u &&
              pinned.cache_slots[1].state == DS4_LAGUNA_CACHE_SLOT_IN_USE &&
              pinned.cache_slots[1].refs == 1u &&
              pinned.cache_payload == initial.cache_payload &&
              pinned.cache_payload_allocation_attempts ==
                  initial.cache_payload_allocation_attempts &&
              pinned.pinned_staging_allocation_attempts ==
                  initial.pinned_staging_allocation_attempts,
          "pressure preserves both owners and cannot allocate an overflow slot");

    CHECK(ds4_gpu_laguna_compact_cache_unpin(
              fixture.context, actor_a) == DS4_LAGUNA_CACHE_OK,
          "logical actor A releases one current-group pin");
    ds4_laguna_cache_handle actor_b_pressure = {0};
    outcome = DS4_LAGUNA_CACHE_ACQUIRE_NONE;
    CHECK(ds4_gpu_laguna_compact_cache_acquire(
              fixture.context, cache_fixture_key(2u, 0u),
              &actor_b_pressure, &outcome) == DS4_LAGUNA_CACHE_OK &&
              outcome == DS4_LAGUNA_CACHE_ACQUIRE_LOAD_OWNER,
          "logical actor B reuses the released slot by deterministic eviction");
    CHECK(ds4_gpu_laguna_compact_cache_unpin(
              fixture.context, actor_b_pressure) == DS4_LAGUNA_CACHE_OK,
          "eviction owner releases its pin before the next actor runs");

    /* Cancel a RESERVED miss before I/O.  Completion of the old capability is
     * stale, and the emptied slot must be immediately reusable by actor A. */
    ds4_laguna_cache_handle cancelled = {0};
    outcome = DS4_LAGUNA_CACHE_ACQUIRE_NONE;
    CHECK(ds4_gpu_laguna_compact_cache_begin(
              fixture.context, cache_fixture_key(2u, 1u),
              &cancelled, &outcome) == DS4_LAGUNA_CACHE_OK &&
              outcome == DS4_LAGUNA_CACHE_ACQUIRE_LOAD_OWNER,
          "logical actor A reserves an eviction miss for cancellation");
    const ds4_laguna_cache_handle stale_cancelled = cancelled;
    CHECK(ds4_gpu_laguna_compact_cache_cancel(
              fixture.context, cancelled) == DS4_LAGUNA_CACHE_RECOVERABLE &&
              ds4_gpu_laguna_compact_cache_complete(
                  fixture.context, &cancelled) ==
                  DS4_LAGUNA_CACHE_RECOVERABLE &&
              memcmp(&cancelled, &stale_cancelled,
                     sizeof(cancelled)) == 0,
          "reserved cancellation restores the slot and leaves no published key");

    ds4_laguna_cache_handle actor_a_reuse = {0};
    outcome = DS4_LAGUNA_CACHE_ACQUIRE_NONE;
    CHECK(ds4_gpu_laguna_compact_cache_acquire(
              fixture.context, cache_fixture_key(1u, 0u),
              &actor_a_reuse, &outcome) == DS4_LAGUNA_CACHE_OK &&
              outcome == DS4_LAGUNA_CACHE_ACQUIRE_LOAD_OWNER &&
              ds4_gpu_laguna_compact_cache_unpin(
                  fixture.context, actor_a_reuse) == DS4_LAGUNA_CACHE_OK,
          "logical actor A immediately reuses cancelled capacity");
    CHECK(ds4_gpu_laguna_compact_cache_unpin(
              fixture.context, actor_b) == DS4_LAGUNA_CACHE_OK,
          "logical actor B releases its original engine-cache entry");

    ds4_laguna_cache_handle cross_session_hit = {0};
    outcome = DS4_LAGUNA_CACHE_ACQUIRE_NONE;
    CHECK(ds4_gpu_laguna_compact_cache_acquire(
              fixture.context, cache_fixture_key(1u, 1u),
              &cross_session_hit, &outcome) == DS4_LAGUNA_CACHE_OK &&
              outcome == DS4_LAGUNA_CACHE_ACQUIRE_HIT_RESERVED &&
              ds4_gpu_laguna_compact_cache_unpin(
                  fixture.context, cross_session_hit) == DS4_LAGUNA_CACHE_OK,
          "actor A can hit actor B's surviving engine-lifetime cache entry");

    CHECK(session_pressure_grouping_contract(),
          "separate two-actor batch policy deterministically forms two groups");

    ds4_gpu_laguna_compact_test_snapshot final;
    ds4_runtime_snapshot runtime_final;
    ds4_runtime_allocation_record final_records[TRACKER_RECORD_CAPACITY];
    memset(&final, 0, sizeof(final));
    memset(&runtime_final, 0, sizeof(runtime_final));
    memset(final_records, 0, sizeof(final_records));
    CHECK(ds4_gpu_test_laguna_compact_snapshot(fixture.context, &final) &&
              !final.cache_unsafe && cache_slots_are_quiescent_ready(&final) &&
              cache_device_map_matches_slots(&fixture, &final, 2u) &&
              final.cache_acquire_hits == 1u &&
              final.cache_acquire_misses == 5u &&
              final.cache_load_successes == 4u &&
              final.cache_load_failures == 1u &&
              final.cache_acquire_misses ==
                  final.cache_load_successes + final.cache_load_failures &&
              final.cache_evictions == 2u &&
              final.cache_cancellations == 1u,
          "interleaved pressure leaves exact counters and zero live pins");
    CHECK(final.cache_payload == initial.cache_payload &&
              final.cache_slots == initial.cache_slots &&
              final.device_entry_to_slot == initial.device_entry_to_slot &&
              final.cache_payload_bytes == initial.cache_payload_bytes &&
              final.pinned_staging_bytes == initial.pinned_staging_bytes &&
              final.cache_payload_allocation_attempts == 1u &&
              final.pinned_staging_allocation_attempts == 4u &&
              final.routed_payload_bytes <= final.cache_payload_bytes &&
              cache_snapshot_has_no_fallback(&final),
          "pressure changes residents and counters without changing fixed owners");
    CHECK(ds4_runtime_tracker_snapshot_copy(
              &fixture.runtime.tracker, &runtime_final,
              final_records, ARRAY_LEN(final_records)) &&
              runtime_final.violation == DS4_RUNTIME_VIOLATION_NONE &&
              runtime_final.active_record_count ==
                  runtime_initial.active_record_count &&
              runtime_final.owned_total_current ==
                  runtime_initial.owned_total_current &&
              memcmp(runtime_final.category_current,
                     runtime_initial.category_current,
                     sizeof(runtime_final.category_current)) == 0 &&
              memcmp(final_records, initial_records,
                     runtime_initial.active_record_count *
                         sizeof(initial_records[0])) == 0,
          "request pressure and cancellation restore exact tracker ownership");

    const ds4_gpu_laguna_destroy_status destroyed =
        ds4_gpu_laguna_compact_destroy(fixture.context);
    CHECK(destroyed == DS4_GPU_LAGUNA_DESTROY_OK,
          "quiescent two-actor pressure cache tears down cleanly");
    if (destroyed == DS4_GPU_LAGUNA_DESTROY_OK) fixture.context = NULL;
    CHECK(destroyed == DS4_GPU_LAGUNA_DESTROY_OK &&
              tracker_has_only_ledger(&fixture.runtime.tracker),
          "pressure teardown returns from engine owners to ledger baseline");

    const int result = g_failures == 0 ? 0 : 1;
    cache_cuda_fixture_close(&fixture);
    return result;
}

static bool page_advice_sequence_is_ordered(
        const ds4_gpu_laguna_compact_test_snapshot *snapshot) {
    return snapshot && snapshot->page_advice_upload_completed_sequence != 0 &&
        snapshot->page_advice_upload_completed_sequence <
            snapshot->page_advice_precharge_sequence &&
        snapshot->page_advice_precharge_sequence <
            snapshot->page_advice_attempt_sequence &&
        snapshot->page_advice_attempt_sequence <
            snapshot->page_advice_post_sample_sequence &&
        snapshot->page_advice_post_sample_sequence <
            snapshot->page_advice_complete_sequence;
}

static bool page_advice_mapping_touch_sequence_is_ordered(
        const ds4_gpu_laguna_compact_test_snapshot *snapshot) {
    return snapshot &&
        snapshot->page_advice_upload_completed_sequence != 0 &&
        snapshot->page_advice_upload_completed_sequence <
            snapshot->page_advice_precharge_sequence &&
        snapshot->page_advice_precharge_sequence <
            snapshot->page_advice_mapping_touch_sequence &&
        snapshot->page_advice_mapping_touch_sequence <
            snapshot->page_advice_attempt_sequence &&
        snapshot->page_advice_attempt_sequence <
            snapshot->page_advice_post_sample_sequence &&
        snapshot->page_advice_post_sample_sequence <
            snapshot->page_advice_complete_sequence;
}

static bool page_advice_sequence_advanced(
        const ds4_gpu_laguna_compact_test_snapshot *before,
        const ds4_gpu_laguna_compact_test_snapshot *after) {
    return before && after &&
        after->page_advice_upload_completed_sequence >
            before->page_advice_upload_completed_sequence &&
        after->page_advice_precharge_sequence >
            before->page_advice_precharge_sequence &&
        after->page_advice_attempt_sequence >
            before->page_advice_attempt_sequence &&
        after->page_advice_post_sample_sequence >
            before->page_advice_post_sample_sequence &&
        after->page_advice_complete_sequence >
            before->page_advice_complete_sequence;
}

static const ds4_laguna_page_advice_errno_bucket *
page_advice_snapshot_errno_bucket(
        const ds4_gpu_laguna_compact_test_snapshot *snapshot,
        int error_number) {
    if (!snapshot) return NULL;
    for (size_t i = 0; i < snapshot->page_advice_errno_bucket_count; i++) {
        if (snapshot->page_advice_errno_buckets[i].error_number ==
                error_number) {
            return &snapshot->page_advice_errno_buckets[i];
        }
    }
    return NULL;
}

static bool cache_snapshot_contains_key(
        const ds4_gpu_laguna_compact_test_snapshot *snapshot,
        ds4_laguna_expert_key key) {
    if (!snapshot || !snapshot->cache_slots) return false;
    for (size_t slot_index = 0;
         slot_index < (size_t)snapshot->cache_slot_count; slot_index++) {
        const ds4_laguna_cache_slot *slot =
            &snapshot->cache_slots[slot_index];
        if ((slot->state == DS4_LAGUNA_CACHE_SLOT_READY ||
             slot->state == DS4_LAGUNA_CACHE_SLOT_IN_USE) &&
            slot->layer == key.layer_id &&
            slot->expert == key.expert_id) {
            return true;
        }
    }
    return false;
}

static int run_page_advice(void) {
    cache_cuda_fixture fixture;
    CHECK(cache_cuda_fixture_open(&fixture),
          "page-advice fixture creates a compact CUDA context");
    if (!fixture.context) {
        cache_cuda_fixture_close(&fixture);
        return 1;
    }

    ds4_runtime_snapshot runtime_before;
    ds4_runtime_allocation_record active_before[TRACKER_RECORD_CAPACITY];
    memset(&runtime_before, 0, sizeof(runtime_before));
    CHECK(ds4_runtime_tracker_snapshot_copy(
              &fixture.runtime.tracker, &runtime_before,
              active_before, ARRAY_LEN(active_before)),
          "page-advice captures the pre-load allocation baseline");

    /* layer 1 / expert 0 has three 36-byte source views at offsets aligned
     * to 16 bytes.  Inward rounding therefore makes exactly 32 bytes from
     * each view eligible, with no overlaps: 96 unique bytes total. */
    ds4_gpu_test_laguna_compact_page_advice_inject(
        16u, 128u, 16u, 0, 0);
    ds4_laguna_cache_handle first = {0};
    ds4_laguna_cache_acquire_outcome outcome =
        DS4_LAGUNA_CACHE_ACQUIRE_NONE;
    CHECK(ds4_gpu_laguna_compact_cache_acquire(
              fixture.context, cache_fixture_key(1u, 0u),
              &first, &outcome) == DS4_LAGUNA_CACHE_OK &&
              outcome == DS4_LAGUNA_CACHE_ACQUIRE_LOAD_OWNER,
          "real cache miss completes before source-page disposal");

    ds4_gpu_laguna_compact_test_snapshot success;
    memset(&success, 0, sizeof(success));
    CHECK(ds4_gpu_test_laguna_compact_snapshot(
              fixture.context, &success),
          "successful page advice is visible in the compact snapshot");
    CHECK(success.page_advice_touched_eligible_unique_bytes == 96u &&
              success.page_advice_touched_eligible_unique_pages == 6u &&
              success.page_advice_attempted_calls == 6u &&
              success.page_advice_attempted_bytes == 192u &&
              success.page_advice_successful_calls == 6u &&
              success.page_advice_successful_bytes == 192u &&
              success.page_advice_failed_calls == 0u &&
              success.page_advice_failed_bytes == 0u &&
              success.page_advice_advised_unique_bytes == 96u &&
              success.page_advice_advised_unique_pages == 6u &&
              success.page_advice_errno_bucket_count == 0u,
          "successful load distinguishes unique coverage from two advice syscalls per range");
    CHECK(success.page_advice_fadvise_attempted_calls == 3u &&
              success.page_advice_fadvise_attempted_bytes == 96u &&
              success.page_advice_fadvise_successful_calls == 3u &&
              success.page_advice_fadvise_successful_bytes == 96u &&
              success.page_advice_fadvise_failed_calls == 0u &&
              success.page_advice_fadvise_failed_bytes == 0u &&
              success.page_advice_madvise_attempted_calls == 3u &&
              success.page_advice_madvise_attempted_bytes == 96u &&
              success.page_advice_madvise_successful_calls == 3u &&
              success.page_advice_madvise_successful_bytes == 96u &&
              success.page_advice_madvise_failed_calls == 0u &&
              success.page_advice_madvise_failed_bytes == 0u,
          "fadvise and madvise each receive every safe full-page range");
    CHECK(success.page_advice_precharge_source_resident_bytes == 128u &&
              success.page_advice_post_source_resident_bytes == 16u,
          "exact pre-advice sample wins the conservative source charge");
    CHECK(success.page_advice_mapping_touch_pages == 0u &&
              success.page_advice_mapping_touch_bytes == 0u &&
              success.page_advice_mapping_touch_sequence == 0u,
          "injected page advice performs no real model-mapping touches");
    CHECK(page_advice_sequence_is_ordered(&success) &&
              success.page_advice_complete_monotonic_ns != 0,
          "upload completion strictly precedes charge, advice, post-sample, and completion");

    ds4_runtime_snapshot runtime_after_success;
    ds4_runtime_allocation_record
        active_after_success[TRACKER_RECORD_CAPACITY];
    memset(&runtime_after_success, 0, sizeof(runtime_after_success));
    CHECK(ds4_runtime_tracker_snapshot_copy(
              &fixture.runtime.tracker, &runtime_after_success,
              active_after_success, ARRAY_LEN(active_after_success)) &&
              runtime_after_success.report_current[
                  DS4_RUNTIME_REPORT_MODEL_SOURCE_RESIDENT] == 16u &&
              runtime_after_success.report_peak[
                  DS4_RUNTIME_REPORT_MODEL_SOURCE_RESIDENT] == 128u &&
              runtime_after_success.qualification_total_peak >=
                  runtime_before.owned_total_current + 128u,
          "lower post-advice residency cannot erase the pre-advice qualification peak");
    CHECK(ds4_gpu_laguna_compact_cache_unpin(
              fixture.context, first) == DS4_LAGUNA_CACHE_OK,
          "successful page-advice load releases its cache reservation");

    /* The prior post sample (16) plus expert 1's 48 new eligible bytes wins
     * over the injected exact pre-sample (32), producing a charge of 64. */
    ds4_gpu_test_laguna_compact_page_advice_inject(
        16u, 32u, 8u, EINVAL, 0);
    ds4_laguna_cache_handle invalid = {0};
    outcome = DS4_LAGUNA_CACHE_ACQUIRE_NONE;
    CHECK(ds4_gpu_laguna_compact_cache_acquire(
              fixture.context, cache_fixture_key(1u, 1u),
              &invalid, &outcome) == DS4_LAGUNA_CACHE_OK &&
              outcome == DS4_LAGUNA_CACHE_ACQUIRE_LOAD_OWNER,
          "advice failure does not misreport a completed CUDA upload as a load failure");
    ds4_gpu_laguna_compact_test_snapshot after_einval;
    memset(&after_einval, 0, sizeof(after_einval));
    CHECK(ds4_gpu_test_laguna_compact_snapshot(
              fixture.context, &after_einval) &&
              after_einval.page_advice_touched_eligible_unique_bytes == 144u &&
              after_einval.page_advice_touched_eligible_unique_pages == 9u &&
              after_einval.page_advice_attempted_calls == 12u &&
              after_einval.page_advice_attempted_bytes == 288u &&
              after_einval.page_advice_failed_calls == 1u &&
              after_einval.page_advice_failed_bytes == 16u &&
              after_einval.page_advice_errno_einval_calls == 1u &&
              after_einval.page_advice_errno_einval_bytes == 16u &&
              after_einval.page_advice_successful_calls == 11u &&
              after_einval.page_advice_successful_bytes == 272u &&
              after_einval.page_advice_advised_unique_bytes == 144u &&
              after_einval.page_advice_advised_unique_pages == 9u,
          "EINVAL counts only the failed fadvise call while madvise still advises its range");
    CHECK(after_einval.page_advice_fadvise_attempted_calls == 6u &&
              after_einval.page_advice_fadvise_attempted_bytes == 144u &&
              after_einval.page_advice_fadvise_successful_calls == 5u &&
              after_einval.page_advice_fadvise_successful_bytes == 128u &&
              after_einval.page_advice_fadvise_failed_calls == 1u &&
              after_einval.page_advice_fadvise_failed_bytes == 16u &&
              after_einval.page_advice_madvise_attempted_calls == 6u &&
              after_einval.page_advice_madvise_attempted_bytes == 144u &&
              after_einval.page_advice_madvise_successful_calls == 6u &&
              after_einval.page_advice_madvise_successful_bytes == 144u &&
              after_einval.page_advice_madvise_failed_calls == 0u &&
              after_einval.page_advice_madvise_failed_bytes == 0u,
          "per-syscall totals identify fadvise as the EINVAL source");
    const ds4_laguna_page_advice_errno_bucket *einval =
        page_advice_snapshot_errno_bucket(&after_einval, EINVAL);
    CHECK(after_einval.page_advice_errno_bucket_count == 1u && einval &&
              einval->calls == 1u && einval->bytes == 16u,
          "snapshot errno buckets preserve EINVAL call and byte totals");
    CHECK(after_einval.page_advice_precharge_source_resident_bytes == 64u &&
              after_einval.page_advice_post_source_resident_bytes == 8u &&
              page_advice_sequence_is_ordered(&after_einval) &&
              page_advice_sequence_advanced(&success, &after_einval) &&
              after_einval.page_advice_complete_monotonic_ns >
                  success.page_advice_complete_monotonic_ns,
          "conservative prior-plus-touch charge and ordered completion survive advice failure");
    CHECK(ds4_gpu_laguna_compact_cache_unpin(
              fixture.context, invalid) == DS4_LAGUNA_CACHE_OK,
          "EINVAL page-advice load releases its cache reservation");

    ds4_gpu_test_laguna_compact_page_advice_inject(
        16u, 4u, 4u, 0, EIO);
    ds4_laguna_cache_handle io = {0};
    outcome = DS4_LAGUNA_CACHE_ACQUIRE_NONE;
    CHECK(ds4_gpu_laguna_compact_cache_acquire(
              fixture.context, cache_fixture_key(2u, 0u),
              &io, &outcome) == DS4_LAGUNA_CACHE_OK &&
              outcome == DS4_LAGUNA_CACHE_ACQUIRE_LOAD_OWNER,
          "second injected advice failure follows another real cache load");
    ds4_gpu_laguna_compact_test_snapshot after_eio;
    memset(&after_eio, 0, sizeof(after_eio));
    CHECK(ds4_gpu_test_laguna_compact_snapshot(
              fixture.context, &after_eio) &&
              after_eio.page_advice_touched_eligible_unique_bytes == 240u &&
              after_eio.page_advice_touched_eligible_unique_pages == 15u &&
              after_eio.page_advice_attempted_calls == 18u &&
              after_eio.page_advice_attempted_bytes == 480u &&
              after_eio.page_advice_failed_calls == 2u &&
              after_eio.page_advice_failed_bytes == 48u &&
              after_eio.page_advice_errno_einval_calls == 1u &&
              after_eio.page_advice_errno_einval_bytes == 16u &&
              after_eio.page_advice_errno_eio_calls == 1u &&
              after_eio.page_advice_errno_eio_bytes == 32u &&
              after_eio.page_advice_successful_calls == 16u &&
              after_eio.page_advice_successful_bytes == 432u &&
              after_eio.page_advice_advised_unique_bytes == 240u &&
              after_eio.page_advice_advised_unique_pages == 15u,
          "EIO has a distinct byte-exact failure bucket without losing unique coverage");
    CHECK(after_eio.page_advice_fadvise_attempted_calls == 9u &&
              after_eio.page_advice_fadvise_attempted_bytes == 240u &&
              after_eio.page_advice_fadvise_successful_calls == 8u &&
              after_eio.page_advice_fadvise_successful_bytes == 224u &&
              after_eio.page_advice_fadvise_failed_calls == 1u &&
              after_eio.page_advice_fadvise_failed_bytes == 16u &&
              after_eio.page_advice_madvise_attempted_calls == 9u &&
              after_eio.page_advice_madvise_attempted_bytes == 240u &&
              after_eio.page_advice_madvise_successful_calls == 8u &&
              after_eio.page_advice_madvise_successful_bytes == 208u &&
              after_eio.page_advice_madvise_failed_calls == 1u &&
              after_eio.page_advice_madvise_failed_bytes == 32u,
          "per-syscall totals identify madvise as the EIO source");
    const ds4_laguna_page_advice_errno_bucket *eio =
        page_advice_snapshot_errno_bucket(&after_eio, EIO);
    einval = page_advice_snapshot_errno_bucket(&after_eio, EINVAL);
    CHECK(after_eio.page_advice_errno_bucket_count == 2u && einval && eio &&
              einval->calls == 1u && einval->bytes == 16u &&
              eio->calls == 1u && eio->bytes == 32u,
          "generic errno buckets retain distinct failure sizes and causes");
    CHECK(after_eio.page_advice_attempted_calls ==
              after_eio.page_advice_successful_calls +
                  after_eio.page_advice_failed_calls &&
              after_eio.page_advice_attempted_bytes ==
              after_eio.page_advice_successful_bytes +
                  after_eio.page_advice_failed_bytes,
          "attempted page advice reconciles exactly into success and failure");
    CHECK(after_eio.page_advice_precharge_source_resident_bytes == 104u &&
              after_eio.page_advice_post_source_resident_bytes == 4u &&
              page_advice_sequence_is_ordered(&after_eio) &&
              page_advice_sequence_advanced(&after_einval, &after_eio) &&
              after_eio.page_advice_complete_monotonic_ns >
                  after_einval.page_advice_complete_monotonic_ns,
          "third transfer carries 8 post bytes plus 96 new bytes before advancing every stage");

    ds4_runtime_snapshot runtime_after_eio;
    ds4_runtime_allocation_record active_after_eio[TRACKER_RECORD_CAPACITY];
    memset(&runtime_after_eio, 0, sizeof(runtime_after_eio));
    CHECK(ds4_runtime_tracker_snapshot_copy(
              &fixture.runtime.tracker, &runtime_after_eio,
              active_after_eio, ARRAY_LEN(active_after_eio)) &&
              runtime_after_eio.report_current[
                  DS4_RUNTIME_REPORT_MODEL_SOURCE_RESIDENT] == 4u &&
              runtime_after_eio.report_peak[
                  DS4_RUNTIME_REPORT_MODEL_SOURCE_RESIDENT] == 128u,
          "third post sample lowers current residency without erasing the first peak");
    CHECK(ds4_gpu_laguna_compact_cache_unpin(
              fixture.context, io) == DS4_LAGUNA_CACHE_OK,
          "EIO page-advice load releases its cache reservation");

    /* Two slots hold three previously loaded experts, so exactly one of the
     * first two keys was evicted.  Reload that key: it must attempt both
     * disposal syscalls again because the pages were touched again, while
     * both process-lifetime unique unions remain unchanged. */
    const ds4_laguna_expert_key first_key = cache_fixture_key(1u, 0u);
    const ds4_laguna_expert_key invalid_key = cache_fixture_key(1u, 1u);
    const bool first_resident =
        cache_snapshot_contains_key(&after_eio, first_key);
    const bool invalid_resident =
        cache_snapshot_contains_key(&after_eio, invalid_key);
    CHECK(first_resident != invalid_resident,
          "two-slot pressure leaves exactly one earlier expert available for a dedup reload");
    const ds4_laguna_expert_key reload_key =
        first_resident ? invalid_key : first_key;
    const uint64_t reload_eligible_bytes = first_resident ? 48u : 96u;
    ds4_gpu_test_laguna_compact_page_advice_inject(
        16u, 2u, 2u, 0, 0);
    ds4_laguna_cache_handle reload = {0};
    outcome = DS4_LAGUNA_CACHE_ACQUIRE_NONE;
    CHECK(ds4_gpu_laguna_compact_cache_acquire(
              fixture.context, reload_key,
              &reload, &outcome) == DS4_LAGUNA_CACHE_OK &&
              outcome == DS4_LAGUNA_CACHE_ACQUIRE_LOAD_OWNER,
          "evicted expert reload performs another completed source transfer");
    ds4_gpu_laguna_compact_test_snapshot after_reload;
    memset(&after_reload, 0, sizeof(after_reload));
    CHECK(ds4_gpu_test_laguna_compact_snapshot(
              fixture.context, &after_reload) &&
              after_reload.page_advice_touched_eligible_unique_bytes == 240u &&
              after_reload.page_advice_touched_eligible_unique_pages == 15u &&
              after_reload.page_advice_advised_unique_bytes == 240u &&
              after_reload.page_advice_advised_unique_pages == 15u &&
              after_reload.page_advice_attempted_calls ==
                  after_eio.page_advice_attempted_calls + 6u &&
              after_reload.page_advice_attempted_bytes ==
                  after_eio.page_advice_attempted_bytes +
                      2u * reload_eligible_bytes &&
              after_reload.page_advice_successful_calls ==
                  after_eio.page_advice_successful_calls + 6u &&
              after_reload.page_advice_successful_bytes ==
                  after_eio.page_advice_successful_bytes +
                      2u * reload_eligible_bytes &&
              after_reload.page_advice_failed_calls ==
                  after_eio.page_advice_failed_calls &&
              after_reload.page_advice_failed_bytes ==
                  after_eio.page_advice_failed_bytes,
          "reload re-advises touched pages but lifetime coverage is byte-exactly deduplicated");
    CHECK(after_reload.page_advice_fadvise_attempted_calls ==
                  after_eio.page_advice_fadvise_attempted_calls + 3u &&
              after_reload.page_advice_fadvise_attempted_bytes ==
                  after_eio.page_advice_fadvise_attempted_bytes +
                      reload_eligible_bytes &&
              after_reload.page_advice_madvise_attempted_calls ==
                  after_eio.page_advice_madvise_attempted_calls + 3u &&
              after_reload.page_advice_madvise_attempted_bytes ==
                  after_eio.page_advice_madvise_attempted_bytes +
                      reload_eligible_bytes &&
              after_reload.page_advice_errno_bucket_count == 2u,
          "reload repeats three fadvise and three madvise calls without new failures");
    CHECK(after_reload.page_advice_precharge_source_resident_bytes ==
                  4u + reload_eligible_bytes &&
              after_reload.page_advice_post_source_resident_bytes == 2u &&
              page_advice_sequence_is_ordered(&after_reload) &&
              page_advice_sequence_advanced(&after_eio, &after_reload) &&
              after_reload.page_advice_complete_monotonic_ns >
                  after_eio.page_advice_complete_monotonic_ns,
          "reload charge uses prior post plus since-sample touches and advances all stages");
    CHECK(after_reload.page_advice_mapping_touch_pages == 0u &&
              after_reload.page_advice_mapping_touch_bytes == 0u &&
              after_reload.page_advice_mapping_touch_sequence == 0u,
          "all injected transfers leave real mapping-touch telemetry unchanged");
    CHECK(ds4_gpu_laguna_compact_cache_unpin(
              fixture.context, reload) == DS4_LAGUNA_CACHE_OK,
          "deduplicated reload releases its cache reservation");

    cache_cuda_fixture_close(&fixture);
    return g_failures == 0 ? 0 : 1;
}

enum {
    MODEL_PAGE_ADVICE_RECORD_CAPACITY = 256,
};

static bool model_page_advice_counters_reconcile(
        const ds4_gpu_laguna_compact_test_snapshot *snapshot,
        uint64_t page_size) {
    if (!snapshot || page_size == 0u ||
        snapshot->page_advice_touched_eligible_unique_bytes == 0u ||
        snapshot->page_advice_mapping_touch_pages == 0u ||
        snapshot->page_advice_mapping_touch_bytes == 0u ||
        snapshot->page_advice_attempted_calls == 0u ||
        snapshot->page_advice_attempted_bytes == 0u ||
        snapshot->page_advice_successful_calls == 0u ||
        snapshot->page_advice_failed_calls != 0u ||
        snapshot->page_advice_failed_bytes != 0u ||
        snapshot->page_advice_errno_einval_calls != 0u ||
        snapshot->page_advice_errno_eio_calls != 0u ||
        snapshot->page_advice_errno_bucket_count != 0u) {
        return false;
    }
    return snapshot->page_advice_attempted_calls ==
               snapshot->page_advice_successful_calls &&
        snapshot->page_advice_attempted_bytes ==
            snapshot->page_advice_successful_bytes &&
        snapshot->page_advice_attempted_calls ==
            snapshot->page_advice_fadvise_attempted_calls +
                snapshot->page_advice_madvise_attempted_calls &&
        snapshot->page_advice_attempted_bytes ==
            snapshot->page_advice_fadvise_attempted_bytes +
                snapshot->page_advice_madvise_attempted_bytes &&
        snapshot->page_advice_fadvise_attempted_calls ==
            snapshot->page_advice_fadvise_successful_calls &&
        snapshot->page_advice_fadvise_attempted_bytes ==
            snapshot->page_advice_fadvise_successful_bytes &&
        snapshot->page_advice_madvise_attempted_calls ==
            snapshot->page_advice_madvise_successful_calls &&
        snapshot->page_advice_madvise_attempted_bytes ==
            snapshot->page_advice_madvise_successful_bytes &&
        snapshot->page_advice_fadvise_failed_calls == 0u &&
        snapshot->page_advice_fadvise_failed_bytes == 0u &&
        snapshot->page_advice_madvise_failed_calls == 0u &&
        snapshot->page_advice_madvise_failed_bytes == 0u &&
        snapshot->page_advice_fadvise_attempted_bytes ==
            snapshot->page_advice_touched_eligible_unique_bytes &&
        snapshot->page_advice_madvise_attempted_bytes ==
            snapshot->page_advice_touched_eligible_unique_bytes &&
        snapshot->page_advice_advised_unique_bytes ==
            snapshot->page_advice_touched_eligible_unique_bytes &&
        snapshot->page_advice_touched_eligible_unique_pages * page_size ==
            snapshot->page_advice_touched_eligible_unique_bytes &&
        snapshot->page_advice_advised_unique_pages * page_size ==
            snapshot->page_advice_advised_unique_bytes &&
        snapshot->page_advice_attempted_bytes % page_size == 0u &&
        snapshot->page_advice_successful_bytes % page_size == 0u &&
        snapshot->page_advice_advised_unique_bytes % page_size == 0u &&
        snapshot->page_advice_attempted_bytes / page_size >=
            snapshot->page_advice_attempted_calls &&
        snapshot->page_advice_mapping_touch_pages <=
            UINT64_MAX / page_size &&
        snapshot->page_advice_mapping_touch_bytes ==
            snapshot->page_advice_mapping_touch_pages * page_size &&
        page_advice_mapping_touch_sequence_is_ordered(snapshot) &&
        snapshot->page_advice_complete_monotonic_ns != 0u;
}

/* Real-model complement to the injected arithmetic microscope above.  A
 * single token touches ten distinct experts in each routed layer, which is
 * below the fixed 8 GiB cache capacity and therefore keeps cumulative source
 * coverage free of reload/eviction ambiguity. */
static int run_model_page_advice(void) {
    const char *model = getenv("DS4_TEST_MODEL");
    if (!model || !model[0]) {
        fprintf(stderr, "FAIL: DS4_TEST_MODEL is not set\n");
        return 1;
    }
    int model_fd = -1;
    bool model_fd_set = false;
    if (!inherited_model_fd(&model_fd, &model_fd_set)) return 1;
    bool close_model_fd = false;
    if (!model_fd_set) {
        model_fd = open(model, O_RDONLY | O_CLOEXEC);
        model_fd_set = model_fd >= 0;
        close_model_fd = model_fd_set;
    }

    const long system_page_size = sysconf(_SC_PAGESIZE);
    CHECK(system_page_size > 0,
          "model page advice obtains the real system page size");
    if (system_page_size <= 0 || !model_fd_set) {
        if (close_model_fd) close(model_fd);
        return 1;
    }
    const uint64_t page_size = (uint64_t)system_page_size;
    uint64_t cold_resident_bytes = UINT64_MAX;
    const bool cold_prepared = cold_prepare_model_fd(
        model_fd, page_size, &cold_resident_bytes);
    CHECK(cold_prepared && cold_resident_bytes == 0u,
          "qualification cold-prepares and verifies only the exact model inode");
    if (!cold_prepared || cold_resident_bytes != 0u) {
        if (close_model_fd) close(model_fd);
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
        .qualification_model_fd = model_fd,
        .qualification_model_fd_set = model_fd_set,
    };
    ds4_engine *engine = NULL;
    CHECK(ds4_engine_open(&engine, &options) == 0 && engine != NULL,
          "real Laguna model opens for page-advice qualification");
    if (!engine) {
        restore_forbidden_environment(&saved);
        if (close_model_fd) close(model_fd);
        return 1;
    }

    ds4_gpu_laguna_compact_test_snapshot startup;
    memset(&startup, 0, sizeof(startup));
    const bool startup_captured =
        ds4_gpu_test_laguna_compact_active_snapshot(&startup);
    CHECK(startup_captured,
          "real model exposes its post-static-copy advice snapshot");

    ds4_laguna_file_identity owned_identity;
    unsigned char descriptor_bytes[64] = {0};
    unsigned char mapping_residency = 0;
    const bool exact_mapping = startup_captured &&
        startup.model_fd >= 0 && startup.model_map != NULL &&
        startup.model_size >= page_size &&
        capture_file_identity(startup.model_fd, &owned_identity) &&
        identities_equal(&owned_identity, &startup.model_identity) &&
        startup.model_size == owned_identity.size_bytes &&
        pread(startup.model_fd,
              descriptor_bytes, sizeof(descriptor_bytes), 0) ==
            (ssize_t)sizeof(descriptor_bytes) &&
        memcmp(startup.model_map,
               descriptor_bytes, sizeof(descriptor_bytes)) == 0 &&
        mincore((void *)startup.model_map,
                (size_t)page_size, &mapping_residency) == 0;
    CHECK(exact_mapping,
          "advice telemetry belongs to the exact opened descriptor and mmap");

    CHECK(startup_captured &&
              startup.static_source_copied_bytes != 0u &&
              model_page_advice_counters_reconcile(&startup, page_size),
          "real static H2D copies advise one exact inward system-page union");

    ds4_runtime_snapshot startup_runtime;
    ds4_runtime_allocation_record
        startup_records[MODEL_PAGE_ADVICE_RECORD_CAPACITY];
    size_t startup_required = 0;
    memset(&startup_runtime, 0, sizeof(startup_runtime));
    memset(startup_records, 0, sizeof(startup_records));
    const bool startup_runtime_captured =
        ds4_test_engine_laguna_runtime_snapshot(
            engine, &startup_runtime, startup_records,
            ARRAY_LEN(startup_records), &startup_required);
    CHECK(startup_runtime_captured && startup_required != 0u &&
              startup_runtime.report_current[
                  DS4_RUNTIME_REPORT_MODEL_SOURCE_RESIDENT] ==
                  startup.page_advice_post_source_resident_bytes &&
              startup_runtime.report_peak[
                  DS4_RUNTIME_REPORT_MODEL_SOURCE_RESIDENT] >=
                  startup.page_advice_precharge_source_resident_bytes &&
              startup_runtime.qualification_total_peak >=
                  startup_runtime.owned_total_current +
                      startup.page_advice_precharge_source_resident_bytes,
          "static advice preserves its simultaneous source precharge peak");

    ds4_tokens tokens = {0};
    /* Laguna's GGUF has no dedicated user-control token.  Vocabulary ID 0
     * is an ordinary, frozen in-range embedding row and keeps this case to
     * exactly one routed token without invoking chat-template tokenization. */
    const int token_id = 0;
    ds4_tokens_push(&tokens, token_id);
    CHECK(tokens.len == 1 && tokens.v[0] == token_id,
          "model page advice stages exactly one valid token");

    ds4_session *session = NULL;
    char sync_error[256] = {0};
    const bool session_created = tokens.len == 1 &&
        ds4_session_create(&session, engine, 32768) == 0 && session != NULL;
    CHECK(session_created,
          "model page advice creates one compact CUDA session");
    const bool session_synced = session_created &&
        ds4_session_sync(session, &tokens,
                         sync_error, sizeof(sync_error)) == 0;
    if (!session_synced && sync_error[0]) {
        fprintf(stderr, "FAIL: model page-advice sync: %s\n", sync_error);
    }
    CHECK(session_synced,
          "one-token sync performs real routed expert reads and H2D copies");
    CHECK(!session_synced || cudaDeviceSynchronize() == cudaSuccess,
          "real routed page-advice run leaves CUDA fully synchronized");

    ds4_gpu_laguna_compact_test_snapshot routed;
    memset(&routed, 0, sizeof(routed));
    const bool routed_captured = session_synced &&
        ds4_gpu_test_laguna_compact_active_snapshot(&routed);
    CHECK(routed_captured,
          "real routed execution exposes its completed advice snapshot");
    CHECK(routed_captured &&
              routed.routed_payload_bytes > startup.routed_payload_bytes &&
              routed.model_file_read_bytes > startup.model_file_read_bytes &&
              routed.cache_load_successes > startup.cache_load_successes &&
              routed.page_advice_touched_eligible_unique_bytes >
                  startup.page_advice_touched_eligible_unique_bytes &&
              routed.page_advice_attempted_calls >
                  startup.page_advice_attempted_calls &&
              routed.page_advice_attempted_bytes >
                  startup.page_advice_attempted_bytes &&
              routed.page_advice_mapping_touch_pages >
                  startup.page_advice_mapping_touch_pages &&
              routed.page_advice_mapping_touch_bytes >
                  startup.page_advice_mapping_touch_bytes &&
              routed.page_advice_advised_unique_bytes >
                  startup.page_advice_advised_unique_bytes,
          "one token adds routed reads, mapping touches, and advised system-page coverage");
    CHECK(routed_captured &&
              model_page_advice_counters_reconcile(&routed, page_size) &&
              routed.page_advice_attempted_bytes -
                      startup.page_advice_attempted_bytes ==
                  2u * (routed.page_advice_touched_eligible_unique_bytes -
                            startup.page_advice_touched_eligible_unique_bytes) &&
              routed.page_advice_successful_bytes -
                      startup.page_advice_successful_bytes ==
                  routed.page_advice_attempted_bytes -
                      startup.page_advice_attempted_bytes &&
              routed.page_advice_advised_unique_bytes -
                      startup.page_advice_advised_unique_bytes ==
                  routed.page_advice_touched_eligible_unique_bytes -
                      startup.page_advice_touched_eligible_unique_bytes,
          "real routed advice reconciles exact page-byte deltas without failures");
    CHECK(routed_captured &&
              routed.page_advice_upload_completed_sequence >
                  startup.page_advice_complete_sequence &&
              routed.page_advice_complete_sequence ==
                  startup.page_advice_complete_sequence + 30u &&
              routed.page_advice_complete_monotonic_ns >
                  startup.page_advice_complete_monotonic_ns &&
              page_advice_mapping_touch_sequence_is_ordered(&routed),
          "routed advice orders upload, precharge, mapping touch, advice, post-sample, and completion");

    ds4_runtime_snapshot live_runtime;
    ds4_runtime_allocation_record
        live_records[MODEL_PAGE_ADVICE_RECORD_CAPACITY];
    size_t live_required = 0;
    memset(&live_runtime, 0, sizeof(live_runtime));
    memset(live_records, 0, sizeof(live_records));
    const bool live_runtime_captured = routed_captured &&
        ds4_test_engine_laguna_runtime_snapshot(
            engine, &live_runtime, live_records,
            ARRAY_LEN(live_records), &live_required);
    CHECK(live_runtime_captured && live_required > startup_required &&
              live_runtime.report_current[
                  DS4_RUNTIME_REPORT_MODEL_SOURCE_RESIDENT] ==
                  routed.page_advice_post_source_resident_bytes &&
              live_runtime.report_peak[
                  DS4_RUNTIME_REPORT_MODEL_SOURCE_RESIDENT] >=
                  routed.page_advice_precharge_source_resident_bytes &&
              live_runtime.qualification_total_current >=
                  live_runtime.owned_total_current +
                      routed.page_advice_post_source_resident_bytes &&
              live_runtime.qualification_total_peak >=
                  live_runtime.owned_total_current +
                      routed.page_advice_precharge_source_resident_bytes,
          "routed source precharge contributes to the simultaneous qualification peak");

    ds4_session_free(session);
    ds4_tokens_free(&tokens);
    ds4_engine_close(engine);
    ds4_test_laguna_compact_close_observation close_observation;
    ds4_gpu_laguna_compact_test_snapshot nonidle;
    ds4_runtime_snapshot closed_runtime;
    memset(&close_observation, 0, sizeof(close_observation));
    memset(&nonidle, 0, sizeof(nonidle));
    memset(&closed_runtime, 0, sizeof(closed_runtime));
    CHECK(ds4_test_laguna_compact_close_observation_get(
              &close_observation) &&
              close_observation.first_destroy_result ==
                  DS4_GPU_LAGUNA_DESTROY_OK &&
              close_observation.destroy_result ==
                  DS4_GPU_LAGUNA_DESTROY_OK &&
              close_observation.destroy_attempt_count == 1u &&
              !close_observation.engine_retained &&
              !ds4_gpu_test_laguna_compact_nonidle_snapshot(&nonidle) &&
              ds4_test_laguna_last_close_snapshot(&closed_runtime) &&
              runtime_snapshot_is_clean(&closed_runtime),
          "real page-advice teardown releases every compact owner cleanly");
    if (close_model_fd) close(model_fd);
    restore_forbidden_environment(&saved);
    return g_failures == 0 ? 0 : 1;
}

static bool nvml_snapshot_string_valid(const char *value, size_t capacity) {
    return value && capacity != 0u && value[0] != '\0' &&
        memchr(value, '\0', capacity) != NULL;
}

static const ds4_runtime_nvml_process_sample *nvml_snapshot_process(
        const ds4_gpu_nvml_inventory_snapshot *snapshot,
        uint32_t pid) {
    if (!snapshot || pid == 0u ||
        snapshot->process_count > DS4_GPU_NVML_PROCESS_CAPACITY) {
        return NULL;
    }
    const ds4_runtime_nvml_process_sample *found = NULL;
    for (size_t i = 0; i < snapshot->process_count; i++) {
        if (snapshot->processes[i].pid != pid) continue;
        if (found) return NULL;
        found = &snapshot->processes[i];
    }
    return found;
}

static ds4_runtime_nvml_process_sample *nvml_snapshot_process_mutable(
        ds4_gpu_nvml_inventory_snapshot *snapshot,
        uint32_t pid) {
    return (ds4_runtime_nvml_process_sample *)
        nvml_snapshot_process(snapshot, pid);
}

static bool nvml_snapshot_contract(
        const ds4_gpu_nvml_inventory_snapshot *snapshot) {
    if (!snapshot || snapshot->api_version != 2u ||
        strcmp(snapshot->api_identity,
               "nvmlDeviceGetComputeRunningProcesses_v2") != 0 ||
        !nvml_snapshot_string_valid(
            snapshot->library_version,
            sizeof(snapshot->library_version)) ||
        !nvml_snapshot_string_valid(
            snapshot->device_uuid, sizeof(snapshot->device_uuid)) ||
        strncmp(snapshot->device_uuid, "GPU-", 4u) != 0 ||
        snapshot->process_count > DS4_GPU_NVML_PROCESS_CAPACITY) {
        return false;
    }
    for (size_t i = 0; i < snapshot->process_count; i++) {
        if (snapshot->processes[i].pid == 0u) return false;
        for (size_t j = i + 1u; j < snapshot->process_count; j++) {
            if (snapshot->processes[i].pid == snapshot->processes[j].pid) {
                return false;
            }
        }
    }
    return true;
}

static bool nvml_snapshot_binding_equal(
        const ds4_gpu_nvml_inventory_snapshot *a,
        const ds4_gpu_nvml_inventory_snapshot *b) {
    return nvml_snapshot_contract(a) && nvml_snapshot_contract(b) &&
        a->api_version == b->api_version &&
        strcmp(a->api_identity, b->api_identity) == 0 &&
        strcmp(a->library_version, b->library_version) == 0 &&
        strcmp(a->device_uuid, b->device_uuid) == 0;
}

static int run_nvml_fd_stability(void) {
    ds4_gpu_nvml_inventory_snapshot warmup;
    memset(&warmup, 0, sizeof(warmup));
    const bool warmup_captured =
        ds4_gpu_nvml_inventory_capture(&warmup) != 0;
    CHECK(warmup_captured && nvml_snapshot_contract(&warmup),
          "NVML descriptor stability warmup captures a valid inventory");
    if (!warmup_captured || !nvml_snapshot_contract(&warmup)) return 1;

    const int fd_baseline = open_fd_count();
    CHECK(fd_baseline > 0,
          "NVML descriptor stability records the post-warmup baseline");
    if (fd_baseline <= 0) return 1;

    bool captures_valid = true;
    bool descriptors_stable = true;
    for (unsigned iteration = 0; iteration < 4u; iteration++) {
        ds4_gpu_nvml_inventory_snapshot repeated;
        memset(&repeated, 0, sizeof(repeated));
        if (!ds4_gpu_nvml_inventory_capture(&repeated) ||
            !nvml_snapshot_binding_equal(&warmup, &repeated)) {
            captures_valid = false;
        }
        if (open_fd_count() != fd_baseline) descriptors_stable = false;
    }
    CHECK(captures_valid,
          "repeated NVML captures preserve the warmed inventory binding");
    CHECK(descriptors_stable,
          "repeated NVML captures preserve the post-warmup descriptor count");
    return g_failures == 0 ? 0 : 1;
}

static int hexadecimal_nibble(char byte) {
    if (byte >= '0' && byte <= '9') return byte - '0';
    if (byte >= 'a' && byte <= 'f') return byte - 'a' + 10;
    if (byte >= 'A' && byte <= 'F') return byte - 'A' + 10;
    return -1;
}

static bool capture_running_build_identity(
        uint8_t out[DS4_RUNTIME_BUILD_IDENTITY_BYTES]) {
    if (!out) return false;
    int fd = open("/proc/self/exe", O_RDONLY | O_CLOEXEC);
    if (fd < 0) return false;
    struct stat st;
    char digest[DS4_PLAN_IO_SHA256_HEX_SIZE] = {0};
    char error[192] = {0};
    bool ok = fstat(fd, &st) == 0 && st.st_size > 0 &&
        ds4_plan_io_sha256_fd(
            fd, (uint64_t)st.st_size, digest, error, sizeof(error));
    if (close(fd) != 0) ok = false;
    if (!ok) {
        if (error[0]) {
            fprintf(stderr, "FAIL: hash running executable: %s\n", error);
        }
        return false;
    }
    for (size_t i = 0; i < DS4_RUNTIME_BUILD_IDENTITY_BYTES; i++) {
        const int high = hexadecimal_nibble(digest[2u * i]);
        const int low = hexadecimal_nibble(digest[2u * i + 1u]);
        if (high < 0 || low < 0) return false;
        out[i] = (uint8_t)((unsigned)high * 16u + (unsigned)low);
    }
    return digest[2u * DS4_RUNTIME_BUILD_IDENTITY_BYTES] == '\0';
}

static bool nvml_snapshot_set_own_bytes(
        ds4_gpu_nvml_inventory_snapshot *snapshot,
        uint32_t own_pid,
        uint64_t used_bytes) {
    ds4_runtime_nvml_process_sample *process =
        nvml_snapshot_process_mutable(snapshot, own_pid);
    if (!process) return false;
    process->used_bytes = used_bytes;
    process->used_bytes_known = true;
    return true;
}

static bool nvml_snapshots_add_or_change_peer(
        const ds4_gpu_nvml_inventory_snapshot *frozen,
        ds4_gpu_nvml_inventory_snapshot *before,
        ds4_gpu_nvml_inventory_snapshot *inside,
        ds4_gpu_nvml_inventory_snapshot *after,
        uint32_t own_pid) {
    if (!frozen || !before || !inside || !after) return false;
    for (size_t i = 0; i < frozen->process_count; i++) {
        const ds4_runtime_nvml_process_sample *baseline =
            &frozen->processes[i];
        if (baseline->pid == own_pid || !baseline->used_bytes_known ||
            baseline->used_bytes > UINT64_MAX - 4096u) {
            continue;
        }
        const uint64_t changed = baseline->used_bytes + 4096u;
        ds4_runtime_nvml_process_sample *before_peer =
            nvml_snapshot_process_mutable(before, baseline->pid);
        ds4_runtime_nvml_process_sample *inside_peer =
            nvml_snapshot_process_mutable(inside, baseline->pid);
        ds4_runtime_nvml_process_sample *after_peer =
            nvml_snapshot_process_mutable(after, baseline->pid);
        if (!before_peer || !inside_peer || !after_peer) return false;
        before_peer->used_bytes = changed;
        inside_peer->used_bytes = changed;
        after_peer->used_bytes = changed;
        before_peer->used_bytes_known = true;
        inside_peer->used_bytes_known = true;
        after_peer->used_bytes_known = true;
        return true;
    }

    if (before->process_count >= DS4_GPU_NVML_PROCESS_CAPACITY ||
        inside->process_count >= DS4_GPU_NVML_PROCESS_CAPACITY ||
        after->process_count >= DS4_GPU_NVML_PROCESS_CAPACITY) {
        return false;
    }
    uint32_t peer_pid = UINT32_MAX;
    while (peer_pid != 0u &&
           (peer_pid == own_pid || nvml_snapshot_process(frozen, peer_pid) ||
            nvml_snapshot_process(before, peer_pid) ||
            nvml_snapshot_process(inside, peer_pid) ||
            nvml_snapshot_process(after, peer_pid))) {
        peer_pid--;
    }
    if (peer_pid == 0u) return false;
    const ds4_runtime_nvml_process_sample peer = {
        .pid = peer_pid,
        .used_bytes = 4096u,
        .used_bytes_known = true,
    };
    before->processes[before->process_count++] = peer;
    inside->processes[inside->process_count++] = peer;
    after->processes[after->process_count++] = peer;
    return true;
}

/* Live Task 13 integration.  Python owns descriptor-bound safe-union cold
 * preparation, so this case intentionally does not issue whole-file advice.
 * It consumes an inherited qualification fd when supplied, captures NVML
 * before the first CUDA call, and then proves the engine's synchronized
 * checkpoint binds real process/model evidence to the runtime tracker. */
static int run_external_attribution(void) {
    const char *model = getenv("DS4_TEST_MODEL");
    if (!model || !model[0]) {
        fprintf(stderr, "FAIL: DS4_TEST_MODEL is not set\n");
        return 1;
    }

    ds4_gpu_nvml_inventory_snapshot frozen_pre_child;
    memset(&frozen_pre_child, 0, sizeof(frozen_pre_child));
    /* This must remain the first operation capable of consulting the GPU. */
    const bool frozen_captured =
        ds4_gpu_nvml_inventory_capture(&frozen_pre_child) != 0;
    CHECK(frozen_captured && nvml_snapshot_contract(&frozen_pre_child),
          "pre-child inventory uses immutable process-scoped NVML v2");
    const uint32_t own_pid = (uint32_t)getpid();
    CHECK(frozen_captured && own_pid != 0u &&
              nvml_snapshot_process(&frozen_pre_child, own_pid) == NULL,
          "pre-child NVML capture creates no CUDA context for this process");
    if (!frozen_captured || !nvml_snapshot_contract(&frozen_pre_child) ||
        own_pid == 0u ||
        nvml_snapshot_process(&frozen_pre_child, own_pid) != NULL) {
        return 1;
    }

    uint8_t build_identity[DS4_RUNTIME_BUILD_IDENTITY_BYTES] = {0};
    CHECK(capture_running_build_identity(build_identity),
          "qualification harness hashes the running executable before CUDA init");

    int model_fd = -1;
    bool model_fd_set = false;
    if (!inherited_model_fd(&model_fd, &model_fd_set)) return 1;
    bool close_model_fd = false;
    if (!model_fd_set) {
        model_fd = open(model, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
        model_fd_set = model_fd >= 0;
        close_model_fd = model_fd_set;
    }
    ds4_laguna_file_identity opened_identity;
    memset(&opened_identity, 0, sizeof(opened_identity));
    CHECK(model_fd_set && capture_file_identity(model_fd, &opened_identity),
          "external attribution binds the already-opened model descriptor");
    if (!model_fd_set || opened_identity.size_bytes == 0u) {
        if (close_model_fd) close(model_fd);
        return 1;
    }

    const long system_page_size = sysconf(_SC_PAGESIZE);
    CHECK(system_page_size > 0 &&
              ((uint64_t)system_page_size &
               ((uint64_t)system_page_size - 1u)) == 0u,
          "external attribution obtains a power-of-two system page size");
    if (system_page_size <= 0) {
        if (close_model_fd) close(model_fd);
        return 1;
    }
    const uint64_t page_size = (uint64_t)system_page_size;
    CHECK(opened_identity.size_bytes % page_size != 0u,
          "pinned Laguna artifact has a partial final page for ceil accounting");

    saved_environment saved;
    save_and_clear_forbidden_environment(&saved);
    const ds4_engine_options options = {
        .model_path = model,
        .backend = DS4_BACKEND_CUDA,
        .context_size = 32768,
        .prefill_chunk = 4096,
        .session_slots = 1,
        .ssd_streaming = true,
        .ssd_streaming_cache_bytes =
            UINT64_C(8) * 1024u * 1024u * 1024u,
        .ssd_streaming_cache_bytes_set = true,
        .qualification_model_fd = model_fd,
        .qualification_model_fd_set = true,
    };
    ds4_engine *engine = NULL;
    CHECK(ds4_engine_open(&engine, &options) == 0 && engine != NULL,
          "real Laguna compact engine opens after pre-child inventory capture");
    if (!engine) goto cleanup;

    ds4_gpu_laguna_compact_test_snapshot compact;
    memset(&compact, 0, sizeof(compact));
    const bool compact_captured =
        ds4_gpu_test_laguna_compact_active_snapshot(&compact);
    CHECK(compact_captured && compact.model_map != NULL &&
              compact.model_size == opened_identity.size_bytes &&
              identities_equal(&compact.model_identity, &opened_identity),
          "external checkpoint starts from the exact compact fd/mapping identity");
    if (!compact_captured || !compact.model_map || compact.model_size == 0u) {
        goto cleanup_engine;
    }

    /* Demand full-page tail accounting: faulting the final file byte makes the
     * partial last page resident, but the physical charge is one whole page. */
    const volatile unsigned char tail_byte =
        ((const volatile unsigned char *)compact.model_map)[
            compact.model_size - 1u];
    (void)tail_byte;
    CHECK(cudaDeviceSynchronize() == cudaSuccess,
          "external checkpoint begins with all compact CUDA work synchronized");

    ds4_engine_laguna_external_checkpoint_observation real;
    memset(&real, 0, sizeof(real));
    const ds4_runtime_status real_status =
        ds4_engine_laguna_external_checkpoint(
            engine, &frozen_pre_child, build_identity, &real);
    CHECK(real_status == DS4_RUNTIME_STATUS_OK &&
              real.sample.failure == DS4_RUNTIME_EXTERNAL_FAILURE_NONE &&
              real.sample.attributed_valid &&
              real.sample.attributed_generation != 0u,
          "live synchronized external checkpoint reconciles successfully");

    const ds4_runtime_nvml_process_sample *inside_process =
        nvml_snapshot_process(&real.inside_ds4, own_pid);
    CHECK(nvml_snapshot_binding_equal(
              &frozen_pre_child, &real.checkpoint_before) &&
              nvml_snapshot_binding_equal(
                  &frozen_pre_child, &real.inside_ds4) &&
              nvml_snapshot_binding_equal(
                  &frozen_pre_child, &real.checkpoint_after),
          "all live inventories bind one NVML v2 symbol, library, and GPU UUID");
    CHECK(inside_process && inside_process->used_bytes_known &&
              real.sample.process_id == own_pid &&
              real.sample.nvml_process_bytes == inside_process->used_bytes &&
              real.sample.nvml_process_bytes >=
                  real.sample.tracked_cuda_physical_bytes &&
              real.sample.cuda_library_unattributed_bytes ==
                  real.sample.nvml_process_bytes -
                      real.sample.tracked_cuda_physical_bytes,
          "process-scoped inside-DS4 NVML bytes are the CUDA attribution source");
    CHECK(real.sample.tracked_cuda_physical_baseline_bytes == 0u &&
              real.sample.nvml_process_baseline_present &&
              real.sample.nvml_process_baseline_bytes != 0u,
          "pre-allocation baseline still charges an existing CUDA context/library");
    CHECK(strcmp(real.sample.nvml_library_version,
                 frozen_pre_child.library_version) == 0 &&
              strcmp(real.sample.device_uuid,
                     frozen_pre_child.device_uuid) == 0 &&
              real.sample.nvml_api_version == frozen_pre_child.api_version &&
              real.sample.unrelated_process_inventory_stable,
          "committed sample preserves raw NVML identity and peer stability");

    const uint64_t rounded_model_bytes =
        (opened_identity.size_bytes + page_size - 1u) & ~(page_size - 1u);
    CHECK(identities_equal(&real.model_identity, &opened_identity) &&
              real.model_map_base == (uint64_t)(uintptr_t)compact.model_map &&
              real.model_map_bytes == compact.model_size &&
              real.model_file_offset == 0u,
          "smaps/mincore evidence names the exact descriptor mapping at file offset zero");
    CHECK(real.sample.smaps_model_device_major ==
                  (uint32_t)major((dev_t)opened_identity.device) &&
              real.sample.smaps_model_device_minor ==
                  (uint32_t)minor((dev_t)opened_identity.device) &&
              real.sample.smaps_model_inode == opened_identity.inode &&
              real.sample.smaps_model_vma_count != 0u,
          "smaps model VMAs bind the opened descriptor device and inode");
    CHECK(real.model_source_page_size == page_size &&
              real.model_source_mapped_page_bytes == rounded_model_bytes &&
              real.model_source_resident_bytes >= page_size &&
              real.model_source_resident_bytes % page_size == 0u &&
              real.model_source_resident_bytes <= rounded_model_bytes,
          "mincore charges every resident bit as one full physical page including the tail");

    const uint64_t unattributed_limit =
        UINT64_C(512) * 1024u * 1024u;
    CHECK(real.sample.host_library_unattributed_bytes <=
                  unattributed_limit &&
              real.sample.cuda_library_unattributed_bytes <=
                  unattributed_limit,
          "live host and CUDA residuals each satisfy the exact 512 MiB ceiling");
    CHECK(memcmp(real.observed_build_identity, build_identity,
                 sizeof(build_identity)) == 0,
          "live checkpoint binds the executable digest observed by the harness");

    ds4_runtime_snapshot real_runtime;
    ds4_runtime_allocation_record
        real_records[MODEL_PAGE_ADVICE_RECORD_CAPACITY];
    size_t real_required = 0u;
    memset(&real_runtime, 0, sizeof(real_runtime));
    memset(real_records, 0, sizeof(real_records));
    CHECK(ds4_test_engine_laguna_runtime_snapshot(
              engine, &real_runtime, real_records,
              ARRAY_LEN(real_records), &real_required) &&
              real_required != 0u &&
              memcmp(&real_runtime.external_sample,
                     &real.sample, sizeof(real.sample)) == 0 &&
              real_runtime.report_current[
                  DS4_RUNTIME_REPORT_MODEL_SOURCE_RESIDENT] ==
                  real.model_source_resident_bytes,
          "engine runtime snapshot publishes the complete reconciled live sample");

    /* Force all three own-process values to differ.  Peers and every non-NVML
     * evidence source remain real, proving the inside capture is selected
     * rather than either harness bracket or a device-wide free-memory delta. */
    ds4_gpu_nvml_inventory_snapshot injected_before = real.inside_ds4;
    ds4_gpu_nvml_inventory_snapshot injected_inside = real.inside_ds4;
    ds4_gpu_nvml_inventory_snapshot injected_after = real.inside_ds4;
    uint64_t before_bytes = 0u;
    uint64_t inside_bytes = 0u;
    uint64_t after_bytes = 0u;
    const bool injected_values_ready = inside_process &&
        checked_add_u64_test(
            inside_process->used_bytes, page_size, &before_bytes) &&
        checked_add_u64_test(
            inside_process->used_bytes, 2u * page_size, &inside_bytes) &&
        checked_add_u64_test(
            inside_process->used_bytes, 3u * page_size, &after_bytes) &&
        nvml_snapshot_set_own_bytes(
            &injected_before, own_pid, before_bytes) &&
        nvml_snapshot_set_own_bytes(
            &injected_inside, own_pid, inside_bytes) &&
        nvml_snapshot_set_own_bytes(
            &injected_after, own_pid, after_bytes);
    CHECK(injected_values_ready &&
              ds4_test_engine_laguna_external_checkpoint_inject_nvml_once(
                  &injected_before, &injected_inside, &injected_after),
          "test seam freezes three distinct one-shot own-process NVML values");

    ds4_engine_laguna_external_checkpoint_observation selected;
    memset(&selected, 0, sizeof(selected));
    const ds4_runtime_status selected_status = injected_values_ready
        ? ds4_engine_laguna_external_checkpoint(
              engine, &frozen_pre_child, build_identity, &selected)
        : DS4_RUNTIME_STATUS_UNSAFE;
    CHECK(selected_status == DS4_RUNTIME_STATUS_OK &&
              selected.sample.attributed_valid &&
              selected.sample.nvml_process_bytes == inside_bytes &&
              selected.sample.nvml_process_bytes != before_bytes &&
              selected.sample.nvml_process_bytes != after_bytes &&
              selected.sample.attributed_generation ==
                  real.sample.attributed_generation + 1u,
          "only the exact inside-DS4 NVML value advances external attribution");

    ds4_runtime_snapshot committed_runtime;
    ds4_runtime_allocation_record
        committed_records[MODEL_PAGE_ADVICE_RECORD_CAPACITY];
    size_t committed_required = 0u;
    memset(&committed_runtime, 0, sizeof(committed_runtime));
    memset(committed_records, 0, sizeof(committed_records));
    CHECK(ds4_test_engine_laguna_runtime_snapshot(
              engine, &committed_runtime, committed_records,
              ARRAY_LEN(committed_records), &committed_required) &&
              memcmp(&committed_runtime.external_sample,
                     &selected.sample, sizeof(selected.sample)) == 0,
          "successful injected-source checkpoint commits one new generation");

    ds4_gpu_nvml_inventory_snapshot failed_before = selected.inside_ds4;
    ds4_gpu_nvml_inventory_snapshot failed_inside = selected.inside_ds4;
    ds4_gpu_nvml_inventory_snapshot failed_after = selected.inside_ds4;
    const bool peer_changed = nvml_snapshots_add_or_change_peer(
        &frozen_pre_child, &failed_before, &failed_inside, &failed_after,
        own_pid);
    CHECK(peer_changed &&
              ds4_test_engine_laguna_external_checkpoint_inject_nvml_once(
                  &failed_before, &failed_inside, &failed_after),
          "test seam injects a narrow-window-stable peer change from pre-child baseline");

    ds4_engine_laguna_external_checkpoint_observation rejected;
    memset(&rejected, 0, sizeof(rejected));
    const ds4_runtime_status rejected_status = peer_changed
        ? ds4_engine_laguna_external_checkpoint(
              engine, &frozen_pre_child, build_identity, &rejected)
        : DS4_RUNTIME_STATUS_OK;
    CHECK(rejected_status == DS4_RUNTIME_STATUS_UNSAFE &&
              !rejected.sample.attributed_valid &&
              rejected.sample.failure ==
                  DS4_RUNTIME_EXTERNAL_FAILURE_UNRELATED_PROCESS_CHANGED,
          "peer appearance or byte change since pre-child invalidates the checkpoint");

    ds4_runtime_snapshot failed_runtime;
    ds4_runtime_allocation_record
        failed_records[MODEL_PAGE_ADVICE_RECORD_CAPACITY];
    size_t failed_required = 0u;
    memset(&failed_runtime, 0, sizeof(failed_runtime));
    memset(failed_records, 0, sizeof(failed_records));
    CHECK(ds4_test_engine_laguna_runtime_snapshot(
              engine, &failed_runtime, failed_records,
              ARRAY_LEN(failed_records), &failed_required) &&
              failed_required == committed_required &&
              memcmp(&failed_runtime.external_sample,
                     &committed_runtime.external_sample,
                     sizeof(failed_runtime.external_sample)) == 0 &&
              memcmp(failed_runtime.report_current,
                     committed_runtime.report_current,
                     sizeof(failed_runtime.report_current)) == 0 &&
              failed_runtime.qualification_total_current ==
                  committed_runtime.qualification_total_current &&
              failed_runtime.qualification_total_peak ==
                  committed_runtime.qualification_total_peak &&
              failed_runtime.violation ==
                  DS4_RUNTIME_VIOLATION_EXTERNAL_ATTRIBUTION,
          "failed checkpoint preserves the last sample/totals and latches attribution unsafe");

cleanup_engine: {
    const uint64_t cleanup_before =
        ds4_gpu_test_generic_cleanup_attempts();
    ds4_engine_close(engine);
    ds4_test_laguna_compact_close_observation close_observation;
    ds4_gpu_laguna_compact_test_snapshot nonidle;
    ds4_runtime_snapshot closed_runtime;
    memset(&close_observation, 0, sizeof(close_observation));
    memset(&nonidle, 0, sizeof(nonidle));
    memset(&closed_runtime, 0, sizeof(closed_runtime));
    CHECK(ds4_test_laguna_compact_close_observation_get(
              &close_observation) &&
              close_observation.first_destroy_result ==
                  DS4_GPU_LAGUNA_DESTROY_OK &&
              close_observation.destroy_result ==
                  DS4_GPU_LAGUNA_DESTROY_OK &&
              close_observation.destroy_attempt_count == 1u &&
              !close_observation.engine_retained &&
              close_observation.gpu_cleanup_before == cleanup_before &&
              close_observation.gpu_cleanup_after == cleanup_before + 1u &&
              ds4_gpu_test_generic_cleanup_attempts() == cleanup_before + 1u &&
              !ds4_gpu_test_laguna_compact_nonidle_snapshot(&nonidle) &&
              !ds4_test_laguna_last_close_snapshot(&closed_runtime),
          "sticky attribution failure still releases every compact owner without relabelling history clean");
}
cleanup:
    restore_forbidden_environment(&saved);
    if (close_model_fd) close(model_fd);
    return g_failures == 0 ? 0 : 1;
}

static int run_cache_faults(void) {
    cache_cuda_fixture fixture;
    CHECK(cache_cuda_fixture_open(&fixture),
          "cache fault fixture creates a compact CUDA context");
    if (!fixture.context) {
        cache_cuda_fixture_close(&fixture);
        return 1;
    }

    ds4_gpu_laguna_compact_test_snapshot fixed_baseline;
    ds4_runtime_snapshot tracker_baseline;
    ds4_runtime_allocation_record
        tracker_records_baseline[TRACKER_RECORD_CAPACITY];
    memset(&fixed_baseline, 0, sizeof(fixed_baseline));
    const bool baseline_captured =
        ds4_gpu_test_laguna_compact_snapshot(
            fixture.context, &fixed_baseline) &&
        capture_tracker_snapshot(
            &fixture.runtime.tracker, &tracker_baseline,
            tracker_records_baseline);
    CHECK(baseline_captured &&
              fixed_baseline.cache_payload_allocation_attempts == 1u &&
              fixed_baseline.pinned_staging_allocation_attempts == 4u &&
              cache_snapshot_has_no_fallback(&fixed_baseline),
          "cache faults start from one fixed allocation set without fallback");
    if (!baseline_captured) {
        cache_cuda_fixture_close(&fixture);
        return 1;
    }

    ds4_laguna_cache_handle handle = {0};
    ds4_laguna_cache_acquire_outcome outcome =
        DS4_LAGUNA_CACHE_ACQUIRE_NONE;
    ds4_gpu_test_laguna_compact_cache_fault_once(
        DS4_GPU_LAGUNA_CACHE_FAULT_PREAD_EINTR);
    const bool eintr_loaded =
        ds4_gpu_laguna_compact_cache_acquire(
            fixture.context, cache_fixture_key(1u, 0u),
            &handle, &outcome) == DS4_LAGUNA_CACHE_OK &&
        outcome == DS4_LAGUNA_CACHE_ACQUIRE_LOAD_OWNER &&
        handle.entry_index < fixture.ledger.expert_entry_count;
    CHECK(eintr_loaded,
          "EINTR retries the same exact read and still publishes");
    if (!eintr_loaded) {
        cache_cuda_fixture_close(&fixture);
        return 1;
    }
    const ds4_laguna_expert_entry *eintr_entry =
        &fixture.ledger.expert_entries[handle.entry_index];
    const uint64_t resident_source_bytes =
        eintr_entry->gate.source_bytes + eintr_entry->up.source_bytes +
        eintr_entry->down.source_bytes;
    CHECK(cache_projection_matches_source(
              &fixture, handle, DS4_LAGUNA_ROUTED_PROJECTION_GATE,
              &eintr_entry->gate) &&
              cache_projection_matches_source(
                  &fixture, handle, DS4_LAGUNA_ROUTED_PROJECTION_UP,
                  &eintr_entry->up) &&
              cache_projection_matches_source(
                  &fixture, handle, DS4_LAGUNA_ROUTED_PROJECTION_DOWN,
                  &eintr_entry->down),
          "EINTR retry preserves exact gate/up/down source offsets");
    const ds4_laguna_expert_key resident_keys[] = {
        cache_fixture_key(1u, 0u),
    };
    ds4_gpu_laguna_compact_test_snapshot after_eintr;
    memset(&after_eintr, 0, sizeof(after_eintr));
    CHECK(ds4_gpu_test_laguna_compact_snapshot(
              fixture.context, &after_eintr) &&
              after_eintr.pread_eintr_retries == 1u &&
              after_eintr.model_file_read_calls == 4u &&
              after_eintr.model_file_read_bytes == resident_source_bytes &&
              after_eintr.cache_acquire_misses == 1u &&
              after_eintr.cache_load_successes == 1u &&
              cache_snapshot_keeps_fixed_allocations(
                  &fixed_baseline, &after_eintr) &&
              cache_snapshot_has_no_fallback(&after_eintr) &&
              cache_snapshot_matches_residents(
                  &fixture, &after_eintr, resident_keys,
                  ARRAY_LEN(resident_keys), resident_source_bytes),
          "EINTR adds one retry call, exact bytes, and one published resident");
    CHECK(ds4_gpu_laguna_compact_cache_unpin(
              fixture.context, handle) == DS4_LAGUNA_CACHE_OK,
          "EINTR-recovered load unpins cleanly");

    const ds4_gpu_laguna_cache_test_fault recoverable_faults[] = {
        DS4_GPU_LAGUNA_CACHE_FAULT_PREAD_EOF,
        DS4_GPU_LAGUNA_CACHE_FAULT_PREAD_SHORT,
        DS4_GPU_LAGUNA_CACHE_FAULT_PREAD_ERROR,
        DS4_GPU_LAGUNA_CACHE_FAULT_CUDA_COPY,
        DS4_GPU_LAGUNA_CACHE_FAULT_EVENT_RECORD,
        DS4_GPU_LAGUNA_CACHE_FAULT_CANCELLATION,
    };
    for (size_t i = 0; i < ARRAY_LEN(recoverable_faults); i++) {
        memset(&handle, 0, sizeof(handle));
        outcome = DS4_LAGUNA_CACHE_ACQUIRE_NONE;
        ds4_gpu_test_laguna_compact_cache_fault_once(
            recoverable_faults[i]);
        CHECK(ds4_gpu_laguna_compact_cache_acquire(
                  fixture.context, cache_fixture_key(1u, 1u),
                  &handle, &outcome) == DS4_LAGUNA_CACHE_RECOVERABLE &&
                  outcome == DS4_LAGUNA_CACHE_ACQUIRE_LOAD_OWNER &&
                  handle.slot_index == DS4_LAGUNA_CACHE_SLOT_NONE,
              "injected cache load fault fails closed and releases capacity");
        ds4_gpu_laguna_compact_test_snapshot after_fault;
        ds4_runtime_snapshot tracker_after_fault;
        ds4_runtime_allocation_record
            tracker_records_after_fault[TRACKER_RECORD_CAPACITY];
        memset(&after_fault, 0, sizeof(after_fault));
        const bool after_fault_captured =
            ds4_gpu_test_laguna_compact_snapshot(
                fixture.context, &after_fault) &&
            capture_tracker_snapshot(
                &fixture.runtime.tracker, &tracker_after_fault,
                tracker_records_after_fault);
        CHECK(after_fault_captured &&
                  after_fault.cache_load_failures == i + 1u &&
                  !after_fault.cache_unsafe &&
                  cache_snapshot_keeps_fixed_allocations(
                      &fixed_baseline, &after_fault) &&
                  cache_snapshot_has_no_fallback(&after_fault) &&
                  cache_snapshot_matches_residents(
                      &fixture, &after_fault, resident_keys,
                      ARRAY_LEN(resident_keys), resident_source_bytes) &&
                  tracker_snapshots_equal(
                      &tracker_baseline, tracker_records_baseline,
                      &tracker_after_fault, tracker_records_after_fault),
              "each recoverable fault immediately restores the exact fixed cache");
    }

    ds4_gpu_laguna_compact_test_snapshot faults;
    memset(&faults, 0, sizeof(faults));
    CHECK(ds4_gpu_test_laguna_compact_snapshot(
              fixture.context, &faults) &&
              faults.pread_eintr_retries == 1u &&
              faults.pread_eof_failures == 1u &&
              faults.pread_short_failures == 1u &&
              faults.pread_error_failures == 1u &&
              faults.cuda_copy_failures == 1u &&
              faults.event_record_failures == 1u &&
              faults.event_completion_failures == 0u &&
              faults.cache_cancellations == 1u &&
              faults.cache_load_failures ==
                  ARRAY_LEN(recoverable_faults) &&
              !faults.cache_unsafe,
          "each recoverable failure is typed without latching unsafe state");

    memset(&handle, 0, sizeof(handle));
    outcome = DS4_LAGUNA_CACHE_ACQUIRE_NONE;
    CHECK(ds4_gpu_laguna_compact_cache_acquire(
              fixture.context, cache_fixture_key(1u, 1u),
              &handle, &outcome) == DS4_LAGUNA_CACHE_OK &&
              outcome == DS4_LAGUNA_CACHE_ACQUIRE_LOAD_OWNER,
          "slot remains reusable after every recoverable fault");
    CHECK(ds4_gpu_laguna_compact_cache_unpin(
              fixture.context, handle) == DS4_LAGUNA_CACHE_OK,
          "post-fault successful load returns READY");

    memset(&handle, 0, sizeof(handle));
    outcome = DS4_LAGUNA_CACHE_ACQUIRE_NONE;
    CHECK(ds4_gpu_laguna_compact_cache_acquire(
              fixture.context, cache_fixture_key(1u, 0u),
              &handle, &outcome) == DS4_LAGUNA_CACHE_OK,
          "IN_USE teardown probe reserves a published slot");
    CHECK(ds4_gpu_laguna_compact_destroy(fixture.context) ==
              DS4_GPU_LAGUNA_DESTROY_RECOVERABLE,
          "teardown refuses while a cache slot is IN_USE");
    CHECK(ds4_gpu_laguna_compact_cache_unpin(
              fixture.context, handle) == DS4_LAGUNA_CACHE_OK,
          "IN_USE teardown probe can release its reservation");

    ds4_laguna_cache_handle loading_handle = {0};
    outcome = DS4_LAGUNA_CACHE_ACQUIRE_NONE;
    CHECK(ds4_gpu_laguna_compact_cache_begin(
              fixture.context, cache_fixture_key(2u, 0u),
              &loading_handle, &outcome) == DS4_LAGUNA_CACHE_OK &&
              outcome == DS4_LAGUNA_CACHE_ACQUIRE_LOAD_OWNER &&
              loading_handle.slot_index != DS4_LAGUNA_CACHE_SLOT_NONE &&
              loading_handle.lifecycle_epoch != 0,
          "split acquire exposes a lifecycle-bound LOADING owner before I/O");
    cache_complete_probe probe = {
        .context = fixture.context,
        .handle = loading_handle,
        .status = DS4_LAGUNA_CACHE_UNSAFE,
    };
    pthread_t load_thread;
    ds4_gpu_test_laguna_compact_pause_cache_load_once();
    const bool load_started =
        pthread_create(&load_thread, NULL, cache_complete_probe_run, &probe) == 0;
    CHECK(load_started &&
              ds4_gpu_test_laguna_compact_wait_cache_load_paused(NULL),
          "real cache load pauses after submitting CUDA work");
    ds4_gpu_laguna_compact_test_snapshot while_loading;
    memset(&while_loading, 0, sizeof(while_loading));
    CHECK(load_started &&
              ds4_gpu_test_laguna_compact_snapshot(
                  fixture.context, &while_loading) &&
              cache_device_map_matches_slots(
                  &fixture, &while_loading, 1u),
          "device reverse map advertises neither victim nor LOADING entry");
    ds4_laguna_cache_handle concurrent_hit = {0};
    outcome = DS4_LAGUNA_CACHE_ACQUIRE_NONE;
    CHECK(ds4_gpu_laguna_compact_cache_begin(
              fixture.context, cache_fixture_key(1u, 0u),
              &concurrent_hit, &outcome) == DS4_LAGUNA_CACHE_OK &&
              outcome == DS4_LAGUNA_CACHE_ACQUIRE_HIT_RESERVED &&
              ds4_gpu_laguna_compact_cache_unpin(
                  fixture.context, concurrent_hit) == DS4_LAGUNA_CACHE_OK,
          "a resident hit does not wait behind an unrelated cache load");
    ds4_laguna_cache_handle duplicate = {0};
    outcome = DS4_LAGUNA_CACHE_ACQUIRE_NONE;
    CHECK(ds4_gpu_laguna_compact_cache_begin(
              fixture.context, cache_fixture_key(2u, 0u),
              &duplicate, &outcome) == DS4_LAGUNA_CACHE_RECOVERABLE &&
              outcome == DS4_LAGUNA_CACHE_ACQUIRE_BUSY_LOADING &&
              duplicate.slot_index == DS4_LAGUNA_CACHE_SLOT_NONE,
          "a duplicate acquire reports LOADING without waiting for I/O");
    ds4_laguna_cache_handle concurrent_pressure = {0};
    outcome = DS4_LAGUNA_CACHE_ACQUIRE_NONE;
    CHECK(ds4_gpu_laguna_compact_cache_begin(
              fixture.context, cache_fixture_key(2u, 1u),
              &concurrent_pressure, &outcome) ==
                  DS4_LAGUNA_CACHE_RECOVERABLE &&
              outcome == DS4_LAGUNA_CACHE_ACQUIRE_PRESSURE &&
              concurrent_pressure.slot_index ==
                  DS4_LAGUNA_CACHE_SLOT_NONE,
          "an unrelated miss cannot reserve a second loader or evict in flight");
    CHECK(ds4_gpu_laguna_compact_destroy(fixture.context) ==
              DS4_GPU_LAGUNA_DESTROY_RECOVERABLE,
          "teardown observes a real in-flight LOADING owner without blocking");
    CHECK(ds4_gpu_laguna_compact_cache_cancel(
              fixture.context, loading_handle) ==
              DS4_LAGUNA_CACHE_RECOVERABLE,
          "cancellation can reach the real loader while its I/O is paused");
    if (load_started) {
        ds4_gpu_test_laguna_compact_resume_cache_load();
        CHECK(pthread_join(load_thread, NULL) == 0 &&
                  probe.status == DS4_LAGUNA_CACHE_RECOVERABLE &&
                  probe.handle.slot_index == DS4_LAGUNA_CACHE_SLOT_NONE,
              "cancelled loader drains submitted CUDA work and restores its slot");
    }
    ds4_gpu_laguna_compact_test_snapshot after_cancel;
    memset(&after_cancel, 0, sizeof(after_cancel));
    CHECK(ds4_gpu_test_laguna_compact_snapshot(
              fixture.context, &after_cancel) &&
              after_cancel.cache_cancellations == 2u &&
              after_cancel.cache_load_failures ==
                  ARRAY_LEN(recoverable_faults) + 1u &&
              after_cancel.cache_acquire_misses ==
                  after_cancel.cache_load_successes +
                  after_cancel.cache_load_failures &&
              cache_device_map_matches_slots(
                  &fixture, &after_cancel, 1u),
          "real cancellation is one typed failed miss with a coherent map");

    ds4_laguna_cache_handle after_drain = {0};
    outcome = DS4_LAGUNA_CACHE_ACQUIRE_NONE;
    const bool after_drain_loaded =
        ds4_gpu_laguna_compact_cache_acquire(
            fixture.context, cache_fixture_key(2u, 1u),
            &after_drain, &outcome) == DS4_LAGUNA_CACHE_OK &&
        outcome == DS4_LAGUNA_CACHE_ACQUIRE_LOAD_OWNER &&
        after_drain.entry_index < fixture.ledger.expert_entry_count;
    CHECK(after_drain_loaded,
          "a cancelled slot is immediately reusable after the upload drain");
    if (after_drain_loaded) {
        const ds4_laguna_expert_entry *after_drain_entry =
            &fixture.ledger.expert_entries[after_drain.entry_index];
        CHECK(cache_projection_matches_source(
                  &fixture, after_drain,
                  DS4_LAGUNA_ROUTED_PROJECTION_GATE,
                  &after_drain_entry->gate) &&
                  cache_projection_matches_source(
                      &fixture, after_drain,
                      DS4_LAGUNA_ROUTED_PROJECTION_UP,
                      &after_drain_entry->up) &&
                  cache_projection_matches_source(
                      &fixture, after_drain,
                      DS4_LAGUNA_ROUTED_PROJECTION_DOWN,
                      &after_drain_entry->down),
              "post-cancel reuse contains only the replacement expert bytes");
        CHECK(ds4_gpu_laguna_compact_cache_unpin(
                  fixture.context, after_drain) == DS4_LAGUNA_CACHE_OK,
              "post-cancel replacement releases its reservation");
    }

    ds4_laguna_cache_handle reserved_cancel = {0};
    outcome = DS4_LAGUNA_CACHE_ACQUIRE_NONE;
    CHECK(ds4_gpu_laguna_compact_cache_begin(
              fixture.context, cache_fixture_key(1u, 1u),
              &reserved_cancel, &outcome) == DS4_LAGUNA_CACHE_OK &&
              outcome == DS4_LAGUNA_CACHE_ACQUIRE_LOAD_OWNER,
          "request cancellation can own a reservation before I/O starts");
    const ds4_laguna_cache_handle cancelled_reservation = reserved_cancel;
    CHECK(ds4_gpu_laguna_compact_cache_cancel(
              fixture.context, reserved_cancel) ==
                  DS4_LAGUNA_CACHE_RECOVERABLE &&
              ds4_gpu_laguna_compact_cache_complete(
                  fixture.context, &reserved_cancel) ==
                  DS4_LAGUNA_CACHE_RECOVERABLE &&
              memcmp(&reserved_cancel, &cancelled_reservation,
                     sizeof(reserved_cancel)) == 0,
          "RESERVED cancellation restores capacity and makes completion stale");
    ds4_gpu_laguna_compact_test_snapshot after_reserved_cancel;
    memset(&after_reserved_cancel, 0, sizeof(after_reserved_cancel));
    CHECK(ds4_gpu_test_laguna_compact_snapshot(
              fixture.context, &after_reserved_cancel) &&
              after_reserved_cancel.cache_cancellations == 3u &&
              after_reserved_cancel.cache_load_failures ==
                  ARRAY_LEN(recoverable_faults) + 2u &&
              after_reserved_cancel.cache_acquire_misses ==
                  after_reserved_cancel.cache_load_successes +
                  after_reserved_cancel.cache_load_failures &&
              cache_device_map_matches_slots(
                  &fixture, &after_reserved_cancel, 1u),
          "RESERVED cancellation is one reconciled failed miss with no publication");
    const ds4_gpu_laguna_destroy_status cache_faults_destroyed =
        ds4_gpu_laguna_compact_destroy(fixture.context);
    CHECK(cache_faults_destroyed == DS4_GPU_LAGUNA_DESTROY_OK,
          "teardown succeeds after every cache owner is released");
    if (cache_faults_destroyed == DS4_GPU_LAGUNA_DESTROY_OK) {
        fixture.context = NULL;
    }
    CHECK(cache_faults_destroyed == DS4_GPU_LAGUNA_DESTROY_OK &&
              tracker_has_only_ledger(&fixture.runtime.tracker),
          "fault-case teardown restores ledger-only accounting");

    const int result = g_failures == 0 ? 0 : 1;
    cache_cuda_fixture_close(&fixture);
    return result;
}

static int run_cache_unsafe(void) {
    cache_cuda_fixture fixture;
    CHECK(cache_cuda_fixture_open(&fixture),
          "unsafe cache fixture creates a compact CUDA context");
    if (!fixture.context) {
        cache_cuda_fixture_close(&fixture);
        return 1;
    }

    ds4_gpu_laguna_compact_test_snapshot fixed_baseline;
    ds4_runtime_snapshot tracker_baseline;
    ds4_runtime_allocation_record
        tracker_records_baseline[TRACKER_RECORD_CAPACITY];
    memset(&fixed_baseline, 0, sizeof(fixed_baseline));
    const bool baseline_captured =
        ds4_gpu_test_laguna_compact_snapshot(
            fixture.context, &fixed_baseline) &&
        capture_tracker_snapshot(
            &fixture.runtime.tracker, &tracker_baseline,
            tracker_records_baseline);
    CHECK(baseline_captured &&
              tracker_baseline.violation == DS4_RUNTIME_VIOLATION_NONE &&
              tracker_baseline.active_record_count == 16u &&
              fixed_baseline.cache_payload_allocation_attempts == 1u &&
              fixed_baseline.pinned_staging_allocation_attempts == 4u &&
              fixed_baseline.pinned_staging_live_count == 4u &&
              cache_snapshot_keeps_fixed_allocations(
                  &fixed_baseline, &fixed_baseline) &&
              cache_snapshot_has_no_fallback(&fixed_baseline),
          "unsafe cache case captures every fixed owner before loading");
    if (!baseline_captured) {
        cache_cuda_fixture_close(&fixture);
        return 1;
    }

    const ds4_laguna_expert_key poisoned_key = cache_fixture_key(1u, 0u);
    ds4_laguna_cache_handle handle = {0};
    ds4_laguna_cache_acquire_outcome outcome =
        DS4_LAGUNA_CACHE_ACQUIRE_NONE;
    const bool owner_reserved =
        ds4_gpu_laguna_compact_cache_begin(
            fixture.context, poisoned_key, &handle, &outcome) ==
                DS4_LAGUNA_CACHE_OK &&
        outcome == DS4_LAGUNA_CACHE_ACQUIRE_LOAD_OWNER &&
        handle.slot_index != DS4_LAGUNA_CACHE_SLOT_NONE &&
        handle.entry_index < fixture.ledger.expert_entry_count &&
        handle.lifecycle_epoch != 0;
    CHECK(owner_reserved,
          "unsafe completion starts from a lifecycle-bound LOADING owner");
    if (!owner_reserved) {
        cache_cuda_fixture_close(&fixture);
        return 1;
    }
    const ds4_laguna_cache_handle poisoned_handle = handle;
    ds4_gpu_test_laguna_compact_cache_fault_once(
        DS4_GPU_LAGUNA_CACHE_FAULT_EVENT_COMPLETION);
    CHECK(ds4_gpu_laguna_compact_cache_complete(
              fixture.context, &handle) == DS4_LAGUNA_CACHE_UNSAFE &&
              handle.slot_index == DS4_LAGUNA_CACHE_SLOT_NONE &&
              handle.entry_index == SIZE_MAX && handle.lifecycle_epoch == 0,
          "event-completion uncertainty poisons the load permanently");
    ds4_gpu_laguna_compact_test_snapshot snapshot;
    ds4_runtime_snapshot tracker_after_fault;
    ds4_runtime_allocation_record
        tracker_records_after_fault[TRACKER_RECORD_CAPACITY];
    ds4_laguna_cache_handle observed_loading = {0};
    memset(&snapshot, 0, sizeof(snapshot));
    const bool fault_captured =
        ds4_gpu_test_laguna_compact_snapshot(
            fixture.context, &snapshot) &&
        capture_tracker_snapshot(
            &fixture.runtime.tracker, &tracker_after_fault,
            tracker_records_after_fault);
    CHECK(fault_captured &&
              snapshot.cache_unsafe &&
              snapshot.lifecycle == DS4_GPU_LAGUNA_LIFECYCLE_ACTIVE &&
              snapshot.model_fd_live && snapshot.static_slab_live &&
              snapshot.static_offsets_live && snapshot.cache_payload_live &&
              snapshot.cache_policy_live &&
              snapshot.device_entry_to_slot_live &&
              snapshot.tracker_mapping_live && snapshot.tracker_static_live &&
              snapshot.tracker_offsets_live &&
              snapshot.cache_acquire_misses == 1u &&
              snapshot.cache_load_failures == 1u &&
              snapshot.cache_load_successes == 0u &&
              snapshot.event_completion_failures == 1u &&
              snapshot.routed_payload_bytes == 0 &&
              cache_snapshot_keeps_fixed_allocations(
                  &fixed_baseline, &snapshot) &&
              cache_snapshot_has_no_fallback(&snapshot) &&
              cache_find_only_loading(
                  &fixture, &snapshot, poisoned_key, &observed_loading) &&
              observed_loading.slot_index == poisoned_handle.slot_index &&
              observed_loading.generation == poisoned_handle.generation &&
              observed_loading.entry_index == poisoned_handle.entry_index &&
              poisoned_handle.lifecycle_epoch != 0 &&
              cache_device_map_matches_slots(&fixture, &snapshot, 0u) &&
              tracker_snapshots_equal(
                  &tracker_baseline, tracker_records_baseline,
                  &tracker_after_fault, tracker_records_after_fault) &&
              ds4_gpu_laguna_compact_ownership_pending(
                  &fixture.runtime.tracker),
          "unsafe completion retains one unpublished LOADING owner and all accounting");

    ds4_laguna_cache_handle rejected = {0};
    outcome = DS4_LAGUNA_CACHE_ACQUIRE_NONE;
    const void *rejected_view = (const void *)(uintptr_t)1u;
    uint64_t rejected_bytes = 1u;
    CHECK(ds4_gpu_laguna_compact_cache_begin(
              fixture.context, cache_fixture_key(1u, 1u),
              &rejected, &outcome) == DS4_LAGUNA_CACHE_UNSAFE &&
              outcome == DS4_LAGUNA_CACHE_ACQUIRE_NONE &&
              rejected.slot_index == DS4_LAGUNA_CACHE_SLOT_NONE &&
              rejected.entry_index == SIZE_MAX && rejected.lifecycle_epoch == 0 &&
              !ds4_gpu_laguna_compact_cache_view(
                  fixture.context, poisoned_handle,
                  DS4_LAGUNA_ROUTED_PROJECTION_GATE,
                  &rejected_view, &rejected_bytes) &&
              rejected_view == NULL && rejected_bytes == 0 &&
              ds4_gpu_laguna_compact_cache_unpin(
                  fixture.context, poisoned_handle) ==
                  DS4_LAGUNA_CACHE_UNSAFE &&
              ds4_gpu_laguna_compact_cache_cancel(
                  fixture.context, poisoned_handle) ==
                  DS4_LAGUNA_CACHE_UNSAFE,
          "poisoned cache rejects begin, view, unpin, and cancellation APIs");

    ds4_gpu_laguna_compact_test_snapshot after_rejection;
    ds4_runtime_snapshot tracker_after_rejection;
    ds4_runtime_allocation_record
        tracker_records_after_rejection[TRACKER_RECORD_CAPACITY];
    ds4_laguna_cache_handle still_loading = {0};
    memset(&after_rejection, 0, sizeof(after_rejection));
    CHECK(ds4_gpu_test_laguna_compact_snapshot(
              fixture.context, &after_rejection) &&
              capture_tracker_snapshot(
                  &fixture.runtime.tracker, &tracker_after_rejection,
                  tracker_records_after_rejection) &&
              after_rejection.cache_unsafe &&
              after_rejection.cache_acquire_misses ==
                  snapshot.cache_acquire_misses &&
              after_rejection.cache_load_failures ==
                  snapshot.cache_load_failures &&
              after_rejection.event_completion_failures ==
                  snapshot.event_completion_failures &&
              after_rejection.routed_payload_bytes == 0 &&
              cache_snapshot_keeps_fixed_allocations(
                  &fixed_baseline, &after_rejection) &&
              cache_snapshot_has_no_fallback(&after_rejection) &&
              cache_find_only_loading(
                  &fixture, &after_rejection, poisoned_key,
                  &still_loading) &&
              still_loading.slot_index == poisoned_handle.slot_index &&
              still_loading.generation == poisoned_handle.generation &&
              cache_device_map_matches_slots(
                  &fixture, &after_rejection, 0u) &&
              tracker_snapshots_equal(
                  &tracker_baseline, tracker_records_baseline,
                  &tracker_after_rejection,
                  tracker_records_after_rejection),
          "rejected cache APIs cannot mutate the retained unsafe owner");
    CHECK(ds4_gpu_laguna_compact_destroy(fixture.context) ==
              DS4_GPU_LAGUNA_DESTROY_UNSAFE,
          "poisoned cache teardown is permanently UNSAFE");

    ds4_gpu_laguna_compact_test_snapshot retained;
    ds4_runtime_snapshot tracker_retained;
    ds4_runtime_allocation_record
        tracker_records_retained[TRACKER_RECORD_CAPACITY];
    ds4_laguna_cache_handle retained_loading = {0};
    memset(&retained, 0, sizeof(retained));
    CHECK(ds4_gpu_test_laguna_compact_snapshot(
              fixture.context, &retained) &&
              capture_tracker_snapshot(
                  &fixture.runtime.tracker, &tracker_retained,
                  tracker_records_retained) &&
              retained.lifecycle == DS4_GPU_LAGUNA_LIFECYCLE_ACTIVE &&
              retained.cache_unsafe && retained.model_fd_live &&
              retained.static_slab_live && retained.static_offsets_live &&
              retained.cache_payload_live && retained.cache_policy_live &&
              retained.device_entry_to_slot_live &&
              cache_snapshot_keeps_fixed_allocations(
                  &fixed_baseline, &retained) &&
              cache_snapshot_has_no_fallback(&retained) &&
              cache_find_only_loading(
                  &fixture, &retained, poisoned_key, &retained_loading) &&
              retained_loading.slot_index == poisoned_handle.slot_index &&
              retained_loading.generation == poisoned_handle.generation &&
              cache_device_map_matches_slots(&fixture, &retained, 0u) &&
              tracker_snapshots_equal(
                  &tracker_baseline, tracker_records_baseline,
                  &tracker_retained, tracker_records_retained) &&
              ds4_gpu_laguna_compact_ownership_pending(
                  &fixture.runtime.tracker),
          "unsafe destroy retains every compact owner and unpublished slot");

    /* This isolated process intentionally retains the poisoned CUDA context;
     * physical cleanup belongs to process exit after an uncertain completion. */
    if (fixture.path[0]) unlink(fixture.path);
    restore_forbidden_environment(&fixture.saved);
    return g_failures == 0 ? 0 : 1;
}

static int run_cache_unsafe_race(void) {
    cache_cuda_fixture fixture;
    CHECK(cache_cuda_fixture_open(&fixture),
          "unsafe race fixture creates a compact CUDA context");
    if (!fixture.context) {
        cache_cuda_fixture_close(&fixture);
        return 1;
    }

    const ds4_laguna_expert_key key = cache_fixture_key(1u, 0u);
    ds4_laguna_cache_handle owner = {0};
    ds4_laguna_cache_acquire_outcome outcome =
        DS4_LAGUNA_CACHE_ACQUIRE_NONE;
    const bool reserved = ds4_gpu_laguna_compact_cache_begin(
            fixture.context, key, &owner, &outcome) ==
                DS4_LAGUNA_CACHE_OK &&
        outcome == DS4_LAGUNA_CACHE_ACQUIRE_LOAD_OWNER;
    CHECK(reserved,
          "unsafe race owns one lifecycle-bound cache reservation");
    if (!reserved) {
        cache_cuda_fixture_close(&fixture);
        return 1;
    }

    cache_complete_probe probe = {
        .context = fixture.context,
        .handle = owner,
        .status = DS4_LAGUNA_CACHE_RECOVERABLE,
    };
    pthread_t worker;
    ds4_gpu_test_laguna_compact_pause_cache_load_once();
    const bool worker_started =
        pthread_create(&worker, NULL, cache_complete_probe_run, &probe) == 0;
    const bool submitted = worker_started &&
        ds4_gpu_test_laguna_compact_wait_cache_load_paused(NULL);
    CHECK(submitted,
          "unsafe race pauses after at least one H2D submission");
    if (!submitted) {
        if (worker_started) {
            ds4_gpu_test_laguna_compact_resume_cache_load();
            (void)pthread_join(worker, NULL);
        } else {
            (void)ds4_gpu_laguna_compact_cache_cancel(
                fixture.context, owner);
        }
        cache_cuda_fixture_close(&fixture);
        return 1;
    }

    CHECK(ds4_gpu_laguna_compact_cache_unpin(
              fixture.context, owner) == DS4_LAGUNA_CACHE_UNSAFE,
          "a concurrent invalid transition latches the cache unsafe");
    ds4_gpu_test_laguna_compact_resume_cache_load();
    CHECK(pthread_join(worker, NULL) == 0 &&
              probe.status == DS4_LAGUNA_CACHE_UNSAFE &&
              probe.handle.slot_index == DS4_LAGUNA_CACHE_SLOT_NONE,
          "loader revalidates unsafe state and refuses publication");

    ds4_gpu_laguna_compact_test_snapshot snapshot;
    ds4_laguna_cache_handle observed = {0};
    memset(&snapshot, 0, sizeof(snapshot));
    CHECK(ds4_gpu_test_laguna_compact_snapshot(
              fixture.context, &snapshot) &&
              snapshot.cache_unsafe &&
              snapshot.cache_acquire_misses == 1u &&
              snapshot.cache_load_failures == 1u &&
              snapshot.cache_load_successes == 0u &&
              snapshot.routed_payload_bytes == 0 &&
              cache_find_only_loading(
                  &fixture, &snapshot, key, &observed) &&
              cache_device_map_matches_slots(&fixture, &snapshot, 0u) &&
              cache_snapshot_has_no_fallback(&snapshot),
          "unsafe race retains one unpublished LOADING owner without fallback");
    CHECK(ds4_gpu_laguna_compact_destroy(fixture.context) ==
              DS4_GPU_LAGUNA_DESTROY_UNSAFE,
          "unsafe race teardown cannot release uncertain ownership");

    /* Isolated poison case: process exit owns physical CUDA cleanup. */
    if (fixture.path[0]) unlink(fixture.path);
    restore_forbidden_environment(&fixture.saved);
    return g_failures == 0 ? 0 : 1;
}

static int run_prefill_allocation(void) {
    static const uint32_t context_tokens = 32768u;
    static const uint32_t prefill_rows = 4096u;
    static const uint32_t swa_rows = 512u;
    static const uint64_t expected_scratch_bytes = UINT64_C(1537052680);
    static const uint64_t expected_kv_bytes = UINT64_C(1686110208);
    static const uint64_t expected_total_bytes = UINT64_C(3223162888);

    int device_count = 0;
    CHECK(cudaGetDeviceCount(&device_count) == cudaSuccess &&
              device_count >= 1,
          "prefill allocation test has a visible CUDA device");
    if (device_count < 1) return 1;
    CHECK(ds4_gpu_init() != 0,
          "prefill allocation test initializes the CUDA backend");
    if (g_failures != 0) {
        ds4_gpu_cleanup();
        return 1;
    }

    ds4_test_model_shape_state shape_before;
    ds4_test_model_shape_state shape_after;
    ds4_test_model_shape_state_get(&shape_before);

    ds4_test_laguna_prefill_allocation_snapshot invalid;
    ds4_test_laguna_prefill_allocation_snapshot invalid_before;
    memset(&invalid, 0xa5, sizeof(invalid));
    memcpy(&invalid_before, &invalid, sizeof(invalid));
    CHECK(!ds4_test_laguna_prefill_allocation(
              context_tokens, 0u, &invalid) &&
              memcmp(&invalid, &invalid_before, sizeof(invalid)) == 0,
          "zero compact prefill rows are rejected before output mutation");

    memset(&invalid, 0x5a, sizeof(invalid));
    memcpy(&invalid_before, &invalid, sizeof(invalid));
    CHECK(!ds4_test_laguna_prefill_allocation(
              context_tokens, context_tokens + 1u, &invalid) &&
              memcmp(&invalid, &invalid_before, sizeof(invalid)) == 0,
          "prefill rows beyond context are rejected before output mutation");

    ds4_test_laguna_prefill_allocation_snapshot snapshot;
    memset(&snapshot, 0, sizeof(snapshot));
    CHECK(ds4_test_laguna_prefill_allocation(
              context_tokens, prefill_rows, &snapshot),
          "model-free compact Laguna graph allocation succeeds on CUDA");
    ds4_test_model_shape_state_get(&shape_after);
    CHECK(memcmp(&shape_before, &shape_after, sizeof(shape_before)) == 0,
          "prefill allocation restores the process model-shape state");
    CHECK(snapshot.context_tokens == context_tokens &&
              snapshot.graph_prefill_cap == prefill_rows &&
              snapshot.session_prefill_cap == prefill_rows,
          "compact graph and synthetic session retain the planned 4K cap");
    CHECK(snapshot.scratch_tensor_bytes == expected_scratch_bytes &&
              snapshot.kv_tensor_bytes == expected_kv_bytes,
          "actual CUDA tensor sizes match the exact 4K scratch and 32K KV plan");

    uint32_t full_layer_count = 0;
    uint32_t swa_layer_count = 0;
    bool exact_layer_caps =
        snapshot.layer_count ==
            DS4_TEST_LAGUNA_PREFILL_ALLOCATION_LAYER_COUNT;
    for (uint32_t il = 0; il < snapshot.layer_count &&
                              il < ARRAY_LEN(snapshot.cache_caps); il++) {
        const uint32_t expected_cap =
            (il % 4u) == 0 ? context_tokens : swa_rows;
        exact_layer_caps = exact_layer_caps &&
            snapshot.cache_caps[il] == expected_cap;
        if (snapshot.cache_caps[il] == context_tokens) {
            full_layer_count++;
        } else if (snapshot.cache_caps[il] == swa_rows) {
            swa_layer_count++;
        }
    }
    CHECK(exact_layer_caps && full_layer_count == 12u &&
              swa_layer_count == 36u,
          "all 12 full-attention and 36 SWA layers retain exact KV caps");

    bool allocated_categories_exact =
        snapshot.allocated.violation == DS4_RUNTIME_VIOLATION_NONE &&
        snapshot.allocated.active_record_count == 124u;
    bool released_categories_exact =
        snapshot.released.violation == DS4_RUNTIME_VIOLATION_NONE &&
        snapshot.released.active_record_count == 0u;
    for (size_t i = 0; i < DS4_RUNTIME_OWNED_CATEGORY_COUNT; i++) {
        const uint64_t expected =
            i == DS4_RUNTIME_CATEGORY_GRAPH_SCRATCH ?
                expected_scratch_bytes :
            i == DS4_RUNTIME_CATEGORY_KV_STATE ? expected_kv_bytes : 0u;
        allocated_categories_exact = allocated_categories_exact &&
            snapshot.allocated.category_current[i] == expected &&
            snapshot.allocated.category_peak[i] == expected;
        released_categories_exact = released_categories_exact &&
            snapshot.released.category_current[i] == 0u &&
            snapshot.released.category_peak[i] == expected;
    }
    CHECK(allocated_categories_exact &&
              snapshot.allocated.owned_total_current ==
                  expected_total_bytes &&
              snapshot.allocated.owned_total_peak == expected_total_bytes &&
              snapshot.allocated.qualification_total_current ==
                  expected_total_bytes &&
              snapshot.allocated.qualification_total_peak ==
                  expected_total_bytes,
          "live tracker charges every graph tensor to exact scratch and KV categories");
    CHECK(released_categories_exact &&
              snapshot.released.owned_total_current == 0u &&
              snapshot.released.owned_total_peak == expected_total_bytes &&
              snapshot.released.qualification_total_current == 0u &&
              snapshot.released.qualification_total_peak ==
                  expected_total_bytes,
          "graph teardown clears current ownership while retaining exact peaks");

    ds4_gpu_cleanup();
    return g_failures == 0 ? 0 : 1;
}

static void usage(const char *program) {
    fprintf(stderr,
            "Usage: %s --case CASE [--case CASE ...]\n"
            "Cases: "
            "startup|nvml-fd-stability|"
            "create-unwind-unsafe|teardown-unsafe|"
            "model-startup|model-create-unwind-unsafe|"
            "model-teardown-unsafe|"
            "model-teardown-reconcile-unsafe|"
            "model-cleanup-release-unsafe|"
            "model-teardown-second-recoverable|cache-validation|cache-io|"
            "cache-faults|cache-unsafe|cache-unsafe-race|"
            "prefill-allocation|page-advice|model-page-advice|"
            "session-pressure|"
            "external-attribution\n",
            program);
}

static int run_session_pressure(void);

static int run_named_case(const char *name) {
    if (strcmp(name, "startup") == 0) {
        return run_startup();
    } else if (strcmp(name, "nvml-fd-stability") == 0) {
        return run_nvml_fd_stability();
    } else if (strcmp(name, "create-unwind-unsafe") == 0) {
        return run_create_unwind_unsafe();
    } else if (strcmp(name, "teardown-unsafe") == 0) {
        return run_teardown_unsafe();
    } else if (strcmp(name, "model-startup") == 0) {
        return run_model_startup(PINNED_MODEL_STARTUP_NORMAL);
    } else if (strcmp(name, "model-create-unwind-unsafe") == 0) {
        return run_model_create_unwind_unsafe();
    } else if (strcmp(name, "model-teardown-unsafe") == 0) {
        return run_model_teardown_unsafe();
    } else if (strcmp(name,
                      "model-teardown-reconcile-unsafe") == 0) {
        return run_model_startup(PINNED_MODEL_TEARDOWN_RECONCILE_UNSAFE);
    } else if (strcmp(name,
                      "model-cleanup-release-unsafe") == 0) {
        return run_model_startup(PINNED_MODEL_CLEANUP_RELEASE_UNSAFE);
    } else if (strcmp(name,
                      "model-teardown-second-recoverable") == 0) {
        return run_model_teardown_second_recoverable();
    } else if (strcmp(name, "cache-validation") == 0) {
        return run_cache_validation();
    } else if (strcmp(name, "cache-io") == 0) {
        return run_cache_io();
    } else if (strcmp(name, "cache-faults") == 0) {
        return run_cache_faults();
    } else if (strcmp(name, "cache-unsafe") == 0) {
        return run_cache_unsafe();
    } else if (strcmp(name, "cache-unsafe-race") == 0) {
        return run_cache_unsafe_race();
    } else if (strcmp(name, "prefill-allocation") == 0) {
        return run_prefill_allocation();
    } else if (strcmp(name, "page-advice") == 0) {
        return run_page_advice();
    } else if (strcmp(name, "model-page-advice") == 0) {
        return run_model_page_advice();
    } else if (strcmp(name, "session-pressure") == 0) {
        return run_session_pressure();
    } else if (strcmp(name, "external-attribution") == 0) {
        return run_external_attribution();
    }
    return -1;
}

int main(int argc, char **argv) {
    (void)compile_typed_lifecycle_contract;
    if (argc < 3 || (argc % 2) == 0) {
        usage(argv[0]);
        return 2;
    }
    int overall = 0;
    for (int argument = 1; argument < argc; argument += 2) {
        if (strcmp(argv[argument], "--case") != 0) {
            usage(argv[0]);
            return 2;
        }
        const char *name = argv[argument + 1];
        const int assertions_before = g_assertions;
        const int failures_before = g_failures;
        const int rc = run_named_case(name);
        if (rc < 0) {
            usage(argv[0]);
            return 2;
        }
        const int case_assertions = g_assertions - assertions_before;
        const int case_failures = g_failures - failures_before;
        if (rc == 0 && case_failures == 0) {
            fprintf(stderr,
                    "test_cuda_laguna_stream %s PASS (%d assertions)\n",
                    name, case_assertions);
        } else {
            fprintf(stderr,
                    "test_cuda_laguna_stream %s FAIL (%d/%d assertions)\n",
                    name, case_failures, case_assertions);
        }
        if (rc != 0 || case_failures != 0) overall = 1;
    }
    return overall;
}
