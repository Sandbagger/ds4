#define DS4_TEST_HOOKS 1

#include "ds4.h"
#include "ds4_gpu.h"
#include "ds4_laguna_stream.h"
#include "ds4_runtime.h"

#include <type_traits>

using laguna_compact_create_fn = int (*)(
    ds4_gpu_laguna_compact **,
    int,
    const void *,
    uint64_t,
    const ds4_laguna_file_identity *,
    const ds4_laguna_ledger *,
    const ds4_laguna_allocation_plan *,
    ds4_runtime_tracker *);
using laguna_compact_destroy_fn = ds4_gpu_laguna_destroy_status (*)(
    ds4_gpu_laguna_compact *);
using laguna_compact_failure_fn = void (*)();
using laguna_compact_snapshot_get_fn = int (*)(
    ds4_gpu_laguna_compact_test_snapshot *);
using gpu_cleanup_attempts_fn = uint64_t (*)();
using close_observation_get_fn = bool (*)(
    ds4_test_laguna_compact_close_observation *);
using engine_runtime_snapshot_fn = bool (*)(
    const ds4_engine *,
    ds4_runtime_snapshot *,
    ds4_runtime_allocation_record *,
    size_t,
    size_t *);
using last_close_snapshot_fn = bool (*)(ds4_runtime_snapshot *);
using engine_live_owners_fn = bool (*)(
    const ds4_engine *,
    ds4_test_laguna_live_owner *,
    size_t,
    size_t *);

static_assert(std::is_same<
              decltype(&ds4_gpu_laguna_compact_create),
              laguna_compact_create_fn>::value,
              "compact create C ABI drifted");
static_assert(std::is_same<
              decltype(&ds4_gpu_laguna_compact_destroy),
              laguna_compact_destroy_fn>::value,
              "compact destroy C ABI drifted");
static_assert(std::is_same<
              decltype(&ds4_gpu_test_laguna_compact_fail_sync_once),
              laguna_compact_failure_fn>::value,
              "compact sync failure hook C ABI drifted");
static_assert(std::is_same<
              decltype(&ds4_gpu_test_laguna_compact_fail_release_once),
              laguna_compact_failure_fn>::value,
              "compact release failure hook C ABI drifted");
static_assert(std::is_same<
              decltype(&ds4_gpu_test_laguna_compact_pause_creating_once),
              laguna_compact_failure_fn>::value,
              "compact creating pause hook C ABI drifted");
static_assert(std::is_same<
              decltype(&ds4_gpu_test_laguna_compact_wait_creating_paused),
              laguna_compact_failure_fn>::value,
              "compact creating wait hook C ABI drifted");
static_assert(std::is_same<
              decltype(&ds4_gpu_test_laguna_compact_resume_creating),
              laguna_compact_failure_fn>::value,
              "compact creating resume hook C ABI drifted");
static_assert(std::is_same<
              decltype(&ds4_gpu_test_laguna_compact_fail_before_publish_once),
              laguna_compact_failure_fn>::value,
              "compact pre-publish failure hook C ABI drifted");
static_assert(std::is_same<
              decltype(&ds4_gpu_test_laguna_compact_nonidle_snapshot),
              laguna_compact_snapshot_get_fn>::value,
              "compact non-idle snapshot hook C ABI drifted");
static_assert(std::is_same<
              decltype(&ds4_gpu_test_generic_cleanup_attempts),
              gpu_cleanup_attempts_fn>::value,
              "generic cleanup attempt hook C ABI drifted");
static_assert(std::is_same<
              decltype(&ds4_test_laguna_compact_close_observation_get),
              close_observation_get_fn>::value,
              "engine close observation C ABI drifted");
static_assert(std::is_same<
              decltype(&ds4_test_engine_laguna_runtime_snapshot),
              engine_runtime_snapshot_fn>::value,
              "engine runtime snapshot C ABI drifted");
static_assert(std::is_same<
              decltype(&ds4_test_laguna_last_close_snapshot),
              last_close_snapshot_fn>::value,
              "last close runtime snapshot C ABI drifted");
static_assert(std::is_same<
              decltype(&ds4_test_engine_laguna_live_owners),
              engine_live_owners_fn>::value,
              "engine live owners C ABI drifted");
static_assert(std::is_same<
              decltype(ds4_test_laguna_live_owner::base),
              uint64_t>::value &&
              std::is_same<
              decltype(ds4_test_laguna_live_owner::bytes),
              uint64_t>::value &&
              std::is_same<
              decltype(ds4_test_laguna_live_owner::callsite_id),
              uint32_t>::value,
              "engine live owner layout drifted");
static_assert(DS4_GPU_LAGUNA_DESTROY_OK == 0 &&
              DS4_GPU_LAGUNA_DESTROY_RECOVERABLE == 1 &&
              DS4_GPU_LAGUNA_DESTROY_UNSAFE == 2,
              "compact destroy result values drifted");
static_assert(DS4_GPU_LAGUNA_LIFECYCLE_IDLE == 0 &&
              DS4_GPU_LAGUNA_LIFECYCLE_CREATING == 1 &&
              DS4_GPU_LAGUNA_LIFECYCLE_ACTIVE == 2 &&
              DS4_GPU_LAGUNA_LIFECYCLE_DESTROYING == 3 &&
              DS4_GPU_LAGUNA_LIFECYCLE_RELEASING == 4,
              "compact lifecycle values drifted");
static_assert(std::is_same<
              decltype(ds4_gpu_laguna_compact_test_snapshot::lifecycle),
              ds4_gpu_laguna_lifecycle>::value,
              "compact snapshot lifecycle is not typed");
static_assert(std::is_same<
              decltype(ds4_gpu_laguna_compact_test_snapshot::model_identity),
              ds4_laguna_file_identity>::value,
              "compact snapshot model identity drifted");
static_assert(std::is_same<
              decltype(ds4_gpu_laguna_compact_test_snapshot::static_slab),
              void *>::value,
              "compact snapshot static slab owner drifted");
static_assert(std::is_same<
              decltype(ds4_gpu_laguna_compact_test_snapshot::static_offsets),
              uint64_t *>::value,
              "compact snapshot static offsets owner drifted");
static_assert(std::is_same<
              decltype(ds4_gpu_laguna_compact_test_snapshot::model_fd_live),
              bool>::value &&
              std::is_same<
              decltype(ds4_gpu_laguna_compact_test_snapshot::static_slab_live),
              bool>::value &&
              std::is_same<
              decltype(ds4_gpu_laguna_compact_test_snapshot::static_offsets_live),
              bool>::value &&
              std::is_same<
              decltype(ds4_gpu_laguna_compact_test_snapshot::tracker_mapping_live),
              bool>::value &&
              std::is_same<
              decltype(ds4_gpu_laguna_compact_test_snapshot::tracker_static_live),
              bool>::value &&
              std::is_same<
              decltype(ds4_gpu_laguna_compact_test_snapshot::tracker_offsets_live),
              bool>::value,
              "compact snapshot owner-live flags drifted");
static_assert(std::is_same<
              decltype(ds4_gpu_laguna_compact_test_snapshot::sync_attempt_count),
              uint64_t>::value &&
              std::is_same<
              decltype(ds4_gpu_laguna_compact_test_snapshot::release_attempt_count),
              uint64_t>::value &&
              std::is_same<
              decltype(ds4_gpu_laguna_compact_test_snapshot::rejection_count),
              uint64_t>::value,
              "compact snapshot lifecycle counters drifted");
static_assert(std::is_same<
              decltype(ds4_test_laguna_compact_close_observation::first_destroy_result),
              int>::value,
              "engine close observation first destroy result drifted");
static_assert(std::is_same<
              decltype(ds4_test_laguna_compact_close_observation::destroy_result),
              int>::value,
              "engine close observation destroy result drifted");
static_assert(std::is_same<
              decltype(ds4_test_laguna_compact_close_observation::destroy_attempt_count),
              uint64_t>::value,
              "engine close observation destroy attempt count drifted");
static_assert(std::is_same<
              decltype(ds4_test_laguna_compact_close_observation::engine_retained),
              bool>::value &&
              std::is_same<
              decltype(ds4_test_laguna_compact_close_observation::gpu_cleanup_before),
              uint64_t>::value &&
              std::is_same<
              decltype(ds4_test_laguna_compact_close_observation::gpu_cleanup_after),
              uint64_t>::value,
              "engine close observation layout drifted");

int main() {
    const uint64_t gib = 1024ull * 1024ull * 1024ull;
    uint64_t reduction = 0;
    if (!ds4_runtime_reduction_qualified(80ull * gib, 44ull * gib,
                                         &reduction) ||
        reduction != 36ull * gib) {
        return 1;
    }

    ds4_laguna_page_range output[1];
    size_t output_count = 1;
    uint64_t output_bytes = 1;
    if (!ds4_laguna_full_page_union(nullptr, 0, 4096u,
                                     output, 1,
                                     &output_count, &output_bytes) ||
        output_count != 0 || output_bytes != 0) {
        return 2;
    }
    return 0;
}
