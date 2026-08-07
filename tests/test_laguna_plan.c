#define _POSIX_C_SOURCE 200809L

#include "ds4_laguna_plan.h"

#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

static int g_failed;
static int g_total;

static const char pinned_model_sha256[] =
    "e163b2c98908809a71245d6bb68b2226994d9969cb2a438eccb72196a1c4147a";

#define CHECK(cond, msg) do {                                                  \
    g_total++;                                                                 \
    if (!(cond)) {                                                             \
        fprintf(stderr, "FAIL: %s (line %d)\n", (msg), __LINE__);             \
        g_failed++;                                                            \
    }                                                                          \
} while (0)

typedef struct {
    ds4_laguna_tensor_range tensors[8];
    ds4_laguna_source_range sources[5];
    ds4_laguna_expert_entry experts[4];
    ds4_laguna_tensor_desc expert_tensors[7];
    char expert_names[7][32];
    ds4_laguna_ledger ledger;
    ds4_laguna_allocation_plan allocation;
    ds4_laguna_page_plan pages;
    ds4_laguna_qualification_plan_input input;
} plan_fixture;

static const char *const callsite_names[] = {
    "laguna.static_slab",
    "laguna.expert_cache_payload",
    "laguna.ledger_arrays",
    "laguna.route_hotness",
    "laguna.host_entry_to_slot",
    "laguna.device_entry_to_slot",
    "laguna.static_offsets",
    "laguna.slot_state",
    "laguna.kv_state",
    "laguna.graph_scratch",
    "laguna.pinned_staging.0",
    "laguna.pinned_staging.1",
    "laguna.pinned_staging.2",
    "laguna.pinned_staging.3",
    "laguna.other_host.engine",
    "laguna.other_host.model",
    "laguna.other_host.bootstrap",
    "laguna.other_host.vocab",
    "laguna.other_host.session",
    "laguna.other_host.tracker",
    "laguna.other_host.serializer",
    "laguna.other_cuda.kernel_tmp",
    "laguna.other_cuda.routed_workspace",
    "laguna.other_cuda.descriptor_upload",
    "laguna.other_cuda.transient",
};

static const ds4_runtime_category callsite_categories[] = {
    DS4_RUNTIME_CATEGORY_STATIC_WEIGHTS,
    DS4_RUNTIME_CATEGORY_EXPERT_CACHE_PAYLOAD,
    DS4_RUNTIME_CATEGORY_CACHE_METADATA_ADDRESS_TABLES,
    DS4_RUNTIME_CATEGORY_CACHE_METADATA_ADDRESS_TABLES,
    DS4_RUNTIME_CATEGORY_CACHE_METADATA_ADDRESS_TABLES,
    DS4_RUNTIME_CATEGORY_CACHE_METADATA_ADDRESS_TABLES,
    DS4_RUNTIME_CATEGORY_CACHE_METADATA_ADDRESS_TABLES,
    DS4_RUNTIME_CATEGORY_CACHE_METADATA_ADDRESS_TABLES,
    DS4_RUNTIME_CATEGORY_KV_STATE,
    DS4_RUNTIME_CATEGORY_GRAPH_SCRATCH,
    DS4_RUNTIME_CATEGORY_PINNED_STAGING,
    DS4_RUNTIME_CATEGORY_PINNED_STAGING,
    DS4_RUNTIME_CATEGORY_PINNED_STAGING,
    DS4_RUNTIME_CATEGORY_PINNED_STAGING,
    DS4_RUNTIME_CATEGORY_OTHER_HOST,
    DS4_RUNTIME_CATEGORY_OTHER_HOST,
    DS4_RUNTIME_CATEGORY_OTHER_HOST,
    DS4_RUNTIME_CATEGORY_OTHER_HOST,
    DS4_RUNTIME_CATEGORY_OTHER_HOST,
    DS4_RUNTIME_CATEGORY_OTHER_HOST,
    DS4_RUNTIME_CATEGORY_OTHER_HOST,
    DS4_RUNTIME_CATEGORY_OTHER_CUDA,
    DS4_RUNTIME_CATEGORY_OTHER_CUDA,
    DS4_RUNTIME_CATEGORY_OTHER_CUDA,
    DS4_RUNTIME_CATEGORY_OTHER_CUDA,
};

static const ds4_runtime_physical_domain callsite_domains[] = {
    DS4_RUNTIME_DOMAIN_CUDA_DEVICE,
    DS4_RUNTIME_DOMAIN_CUDA_DEVICE,
    DS4_RUNTIME_DOMAIN_HOST,
    DS4_RUNTIME_DOMAIN_HOST,
    DS4_RUNTIME_DOMAIN_HOST,
    DS4_RUNTIME_DOMAIN_CUDA_DEVICE,
    DS4_RUNTIME_DOMAIN_HOST,
    DS4_RUNTIME_DOMAIN_HOST,
    DS4_RUNTIME_DOMAIN_CUDA_DEVICE,
    DS4_RUNTIME_DOMAIN_CUDA_DEVICE,
    DS4_RUNTIME_DOMAIN_HOST,
    DS4_RUNTIME_DOMAIN_HOST,
    DS4_RUNTIME_DOMAIN_HOST,
    DS4_RUNTIME_DOMAIN_HOST,
    DS4_RUNTIME_DOMAIN_HOST,
    DS4_RUNTIME_DOMAIN_HOST,
    DS4_RUNTIME_DOMAIN_HOST,
    DS4_RUNTIME_DOMAIN_HOST,
    DS4_RUNTIME_DOMAIN_HOST,
    DS4_RUNTIME_DOMAIN_HOST,
    DS4_RUNTIME_DOMAIN_HOST,
    DS4_RUNTIME_DOMAIN_CUDA_DEVICE,
    DS4_RUNTIME_DOMAIN_CUDA_DEVICE,
    DS4_RUNTIME_DOMAIN_CUDA_DEVICE,
    DS4_RUNTIME_DOMAIN_CUDA_DEVICE,
};

static uint64_t sum_u64(const uint64_t *values, size_t count) {
    uint64_t total = 0;
    for (size_t i = 0; i < count; i++) total += values[i];
    return total;
}

static void set_view(ds4_laguna_expert_view *view,
                     uint64_t parent,
                     uint64_t source_offset,
                     uint64_t device_offset) {
    view->parent_stable_index = parent;
    view->source_offset = source_offset;
    view->source_bytes = 4096u;
    view->device_offset = device_offset;
}

static void prepare_ledger(plan_fixture *fixture) {
    memset(fixture, 0, sizeof(*fixture));
    const uint64_t offsets[] = { 100u, 8193u, 16384u, 20480u, 24576u };
    const uint64_t sizes[] = { 8093u, 8191u, 4096u, 4096u, 4096u };
    for (size_t i = 0; i < 5u; i++) {
        fixture->tensors[i].stable_index = i;
        fixture->tensors[i].tensor_class = i < 2u ?
            DS4_LAGUNA_TENSOR_STATIC : DS4_LAGUNA_TENSOR_ROUTED_EXPERT;
        fixture->tensors[i].source_offset = offsets[i];
        fixture->tensors[i].source_bytes = sizes[i];
#ifdef DS4_LAGUNA_TENSOR_RANGE_HAS_ROUTED_IDENTITY
        fixture->tensors[i].routed_layer = i < 2u ? UINT32_MAX : 1u;
        fixture->tensors[i].routed_projection = i < 2u ?
            DS4_LAGUNA_ROUTED_PROJECTION_NONE :
            (ds4_laguna_routed_projection)(i - 1u);
#endif
    }
    fixture->sources[0] = (ds4_laguna_source_range){
        DS4_LAGUNA_SOURCE_HEADER, 0u, 20u,
    };
    fixture->sources[1] = (ds4_laguna_source_range){
        DS4_LAGUNA_SOURCE_METADATA, 20u, 30u,
    };
    fixture->sources[2] = (ds4_laguna_source_range){
        DS4_LAGUNA_SOURCE_TENSOR_DIRECTORY, 50u, 40u,
    };
    fixture->sources[3] = (ds4_laguna_source_range){
        DS4_LAGUNA_SOURCE_ALIGNMENT_PADDING, 90u, 10u,
    };
    fixture->sources[4] = (ds4_laguna_source_range){
        DS4_LAGUNA_SOURCE_TENSOR_PADDING, 28672u, 1u,
    };
    fixture->experts[0].layer = 1u;
    fixture->experts[0].expert = 0u;
    set_view(&fixture->experts[0].gate, 2u, 16384u, 0u);
    set_view(&fixture->experts[0].up, 3u, 20480u, 4096u);
    set_view(&fixture->experts[0].down, 4u, 24576u, 8192u);
    fixture->experts[0].used_bytes = 12288u;

    fixture->ledger.tensor_ranges = fixture->tensors;
    fixture->ledger.tensor_range_count = 5u;
    fixture->ledger.source_ranges = fixture->sources;
    fixture->ledger.source_range_count = 5u;
    fixture->ledger.expert_entries = fixture->experts;
    fixture->ledger.file_size = 28673u;
    fixture->ledger.tensor_count = 5u;
    fixture->ledger.static_parent_count = 2u;
    fixture->ledger.routed_parent_count = 3u;
    fixture->ledger.expert_entry_count = 1u;
    fixture->ledger.routed_source_bytes = 12288u;
    fixture->ledger.static_source_bytes = 16284u;
    fixture->ledger.static_aligned_device_bytes = 16384u;
    fixture->ledger.non_tensor_source_bytes = 101u;
    fixture->ledger.routed_projection_expert_bytes = 4096u;
    fixture->ledger.slot_stride_bytes = 12288u;
}

static void prepare_allocation(plan_fixture *fixture) {
    ds4_laguna_allocation_plan *plan = &fixture->allocation;
    const uint64_t gib = UINT64_C(1024) * 1024u * 1024u;
    plan->profile_id = "cache-8gib";
    plan->context_tokens = 32768u;
    plan->prefill_rows = 4096u;
    plan->session_count = 1u;
    plan->configured_cache_bytes = 8u * gib;
    plan->effective_cache_limit_bytes = 8u * gib;
    plan->slot_stride_bytes = fixture->ledger.slot_stride_bytes;
    plan->slot_count = (uint32_t)(plan->configured_cache_bytes /
                                  plan->slot_stride_bytes);
    plan->cache_payload_bytes =
        (uint64_t)plan->slot_count * plan->slot_stride_bytes;
    plan->cache_tail_uncharged_bytes =
        plan->configured_cache_bytes - plan->cache_payload_bytes;
    plan->staging_buffer_count = 4u;
    plan->staging_buffer_bytes = plan->slot_stride_bytes;
    plan->owned_category_bounds[DS4_RUNTIME_CATEGORY_STATIC_WEIGHTS] =
        fixture->ledger.static_aligned_device_bytes;
    plan->owned_category_bounds[DS4_RUNTIME_CATEGORY_EXPERT_CACHE_PAYLOAD] =
        plan->cache_payload_bytes;
    plan->owned_category_bounds[
        DS4_RUNTIME_CATEGORY_CACHE_METADATA_ADDRESS_TABLES] = 100u;
    plan->owned_category_bounds[DS4_RUNTIME_CATEGORY_KV_STATE] = 200u;
    plan->owned_category_bounds[DS4_RUNTIME_CATEGORY_GRAPH_SCRATCH] = 300u;
    plan->owned_category_bounds[DS4_RUNTIME_CATEGORY_PINNED_STAGING] =
        4u * plan->slot_stride_bytes;
    plan->owned_category_bounds[DS4_RUNTIME_CATEGORY_OTHER_HOST] = 400u;
    plan->owned_category_bounds[DS4_RUNTIME_CATEGORY_OTHER_CUDA] = 500u;
    plan->report_bounds[DS4_RUNTIME_REPORT_MODEL_MAPPED_VIRTUAL] =
        fixture->ledger.file_size;
    plan->report_bounds[DS4_RUNTIME_REPORT_MODEL_MAPPING_REGISTERED] = 0u;
    plan->report_bounds[DS4_RUNTIME_REPORT_MODEL_SOURCE_RESIDENT] = 1000u;
    plan->report_bounds[DS4_RUNTIME_REPORT_HOST_LIBRARY_UNATTRIBUTED] = 200u;
    plan->report_bounds[DS4_RUNTIME_REPORT_CUDA_LIBRARY_UNATTRIBUTED] = 300u;
    plan->owned_non_cache_bound_bytes =
        sum_u64(plan->owned_category_bounds, DS4_RUNTIME_OWNED_CATEGORY_COUNT) -
        plan->cache_payload_bytes;
    plan->owned_total_bound_bytes =
        sum_u64(plan->owned_category_bounds, DS4_RUNTIME_OWNED_CATEGORY_COUNT);
    plan->qualification_non_cache_bound_bytes =
        plan->owned_non_cache_bound_bytes + 1500u;
    plan->planned_qualification_bytes = plan->owned_total_bound_bytes + 1500u;
    plan->qualification_total_bound_bytes = 24u * gib;

    for (size_t i = 0; i < DS4_LAGUNA_ALLOCATION_CALLSITE_COUNT; i++) {
        ds4_runtime_callsite *site = &plan->callsites[i];
        site->id = (uint32_t)i + 1u;
        site->name = callsite_names[i];
        site->category = callsite_categories[i];
        site->domain = callsite_domains[i];
    }
    plan->callsites[0].bound_bytes = fixture->ledger.static_aligned_device_bytes;
    plan->callsites[1].bound_bytes = plan->cache_payload_bytes;
    plan->callsites[2].bound_bytes = 100u;
    plan->callsites[8].bound_bytes = 200u;
    plan->callsites[9].bound_bytes = 300u;
    for (size_t i = 10u; i < 14u; i++) {
        plan->callsites[i].bound_bytes = plan->slot_stride_bytes;
    }
    plan->callsites[14].bound_bytes = 400u;
    plan->callsites[21].bound_bytes = 500u;
    plan->callsite_count = DS4_LAGUNA_ALLOCATION_CALLSITE_COUNT;
}

static bool prepare_fixture(plan_fixture *fixture, char *error, size_t error_size) {
    prepare_ledger(fixture);
    prepare_allocation(fixture);
    if (!ds4_laguna_page_plan_make(&fixture->pages,
                                    &fixture->ledger,
                                    4096u,
                                    error,
                                    error_size)) {
        return false;
    }
    fixture->input.model_identity = (ds4_laguna_file_identity){
        42u, 99u, 28673u, 123456789u,
    };
    fixture->input.model_sha256 = pinned_model_sha256;
    fixture->input.ledger = &fixture->ledger;
    fixture->input.allocation = &fixture->allocation;
    fixture->input.page_cache = &fixture->pages;
    return true;
}

static bool prepare_expert_fixture(plan_fixture *fixture,
                                   char *error,
                                   size_t error_size) {
    memset(fixture, 0, sizeof(*fixture));
    const ds4_laguna_ledger_spec spec = {
        .file_size = 1024u,
        .header_end = 16u,
        .metadata_end = 32u,
        .tensor_directory_end = 65u,
        .tensor_data_start = 128u,
        .gguf_alignment = 64u,
        .device_alignment = 256u,
        .first_routed_layer = 1u,
        .layer_count = 3u,
        .expert_count = 2u,
    };
    const uint64_t offsets[] = {
        128u, 192u, 320u, 448u, 576u, 704u, 832u,
    };
    const ds4_laguna_routed_projection projections[] = {
        DS4_LAGUNA_ROUTED_PROJECTION_NONE,
        DS4_LAGUNA_ROUTED_PROJECTION_GATE,
        DS4_LAGUNA_ROUTED_PROJECTION_UP,
        DS4_LAGUNA_ROUTED_PROJECTION_DOWN,
        DS4_LAGUNA_ROUTED_PROJECTION_GATE,
        DS4_LAGUNA_ROUTED_PROJECTION_UP,
        DS4_LAGUNA_ROUTED_PROJECTION_DOWN,
    };
    for (size_t i = 0; i < 7u; i++) {
        ds4_laguna_tensor_desc *tensor = &fixture->expert_tensors[i];
        snprintf(fixture->expert_names[i], sizeof(fixture->expert_names[i]),
                 "tensor.%zu", i);
        tensor->stable_index = i + 10u;
        tensor->name = fixture->expert_names[i];
        tensor->name_len = strlen(tensor->name);
        tensor->source_offset = offsets[i];
        tensor->gguf_type = 2u;
        tensor->block_elems = 32u;
        tensor->block_bytes = 18u;
        tensor->tensor_class = i == 0u ? DS4_LAGUNA_TENSOR_STATIC :
            DS4_LAGUNA_TENSOR_ROUTED_EXPERT;
        tensor->routed_layer = i == 0u ? UINT32_MAX :
            (uint32_t)((i - 1u) / 3u) + 1u;
        tensor->routed_projection = projections[i];
        if (i == 0u) {
            tensor->ndim = 1u;
            tensor->dim[0] = 33u;
            tensor->source_bytes = 36u;
        } else {
            tensor->ndim = 3u;
            tensor->dim[0] = 32u;
            tensor->dim[1] = 2u;
            tensor->dim[2] = 2u;
            tensor->source_bytes = 72u;
        }
    }
    if (!ds4_laguna_ledger_build(&fixture->ledger, &spec,
                                 fixture->expert_tensors, 7u,
                                 error, error_size)) {
        return false;
    }
    prepare_allocation(fixture);
    if (!ds4_laguna_page_plan_make(&fixture->pages,
                                    &fixture->ledger,
                                    4096u,
                                    error,
                                    error_size)) {
        return false;
    }
    fixture->input.model_identity = (ds4_laguna_file_identity){
        42u, 99u, fixture->ledger.file_size, 123456789u,
    };
    fixture->input.model_sha256 = pinned_model_sha256;
    fixture->input.ledger = &fixture->ledger;
    fixture->input.allocation = &fixture->allocation;
    fixture->input.page_cache = &fixture->pages;
    return true;
}

static void test_page_plan(void) {
    plan_fixture fixture;
    char error[256];
    prepare_ledger(&fixture);
    memset(error, 0, sizeof(error));
    CHECK(ds4_laguna_page_plan_make(&fixture.pages,
                                    &fixture.ledger,
                                    4096u,
                                    error,
                                    sizeof(error)),
          "page plan builds from ledger tensor ranges");
    CHECK(error[0] == '\0', "successful page planning clears stale error");
    CHECK(fixture.pages.page_size == 4096u &&
              fixture.pages.mapped_page_bytes == 32768u &&
              fixture.pages.eligible_unique_bytes == 20480u &&
              fixture.pages.unavoidable_bytes == 12288u,
          "page plan charges shared-boundary and final-partial pages");
    CHECK(fixture.pages.range_count == 2u &&
              fixture.pages.ranges[0].offset == 4096u &&
              fixture.pages.ranges[0].bytes == 4096u &&
              fixture.pages.ranges[1].offset == 12288u &&
              fixture.pages.ranges[1].bytes == 16384u,
          "page plan unions only independently inward-rounded tensor pages");
    ds4_laguna_page_plan_free(&fixture.pages);
    CHECK(fixture.pages.ranges == NULL && fixture.pages.range_count == 0u,
          "page plan free clears ownership");

    prepare_ledger(&fixture);
    fixture.tensors[4].source_bytes = UINT64_MAX;
    CHECK(!ds4_laguna_page_plan_make(&fixture.pages,
                                     &fixture.ledger,
                                     4096u,
                                     error,
                                     sizeof(error)) &&
              strstr(error, "range") != NULL,
          "page plan rejects overflowing or out-of-file tensor ranges");
}

static const char expected_ledger_json[] =
    "{\"expert_entries\":[{\"down\":{\"device_offset\":\"8192\","
    "\"parent_stable_index\":\"4\",\"source_bytes\":\"4096\","
    "\"source_offset\":\"24576\"},\"expert\":0,\"gate\":{"
    "\"device_offset\":\"0\",\"parent_stable_index\":\"2\","
    "\"source_bytes\":\"4096\",\"source_offset\":\"16384\"},"
    "\"layer\":1,\"up\":{\"device_offset\":\"4096\","
    "\"parent_stable_index\":\"3\",\"source_bytes\":\"4096\","
    "\"source_offset\":\"20480\"},\"used_bytes\":\"12288\"}],"
    "\"expert_entry_count\":\"1\",\"file_size\":\"28673\","
    "\"non_tensor_source_bytes\":\"101\",\"routed_parent_count\":\"3\","
    "\"routed_projection_expert_bytes\":\"4096\","
    "\"routed_source_bytes\":\"12288\",\"slot_stride_bytes\":\"12288\","
    "\"source_range_count\":\"5\",\"source_ranges\":[{\"kind\":\"HEADER\","
    "\"source_bytes\":\"20\",\"source_offset\":\"0\"},{\"kind\":\"METADATA\","
    "\"source_bytes\":\"30\",\"source_offset\":\"20\"},{\"kind\":"
    "\"TENSOR_DIRECTORY\",\"source_bytes\":\"40\",\"source_offset\":\"50\"},{"
    "\"kind\":\"ALIGNMENT_PADDING\",\"source_bytes\":\"10\","
    "\"source_offset\":\"90\"},{"
    "\"kind\":\"TENSOR_PADDING\",\"source_bytes\":\"1\","
    "\"source_offset\":\"28672\"}],\"static_aligned_device_bytes\":"
    "\"16384\",\"static_parent_count\":\"2\","
    "\"static_source_bytes\":\"16284\",\"tensor_count\":\"5\","
    "\"tensor_range_count\":\"5\",\"tensor_ranges\":[{\"class\":"
    "\"STATIC\",\"source_bytes\":\"8093\",\"source_offset\":\"100\","
    "\"stable_index\":\"0\"},{\"class\":\"STATIC\",\"source_bytes\":"
    "\"8191\",\"source_offset\":\"8193\",\"stable_index\":\"1\"},{"
    "\"class\":\"ROUTED_EXPERT\","
#ifdef DS4_LAGUNA_TENSOR_RANGE_HAS_ROUTED_IDENTITY
    "\"routed_layer\":1,\"routed_projection\":\"GATE\","
#endif
    "\"source_bytes\":\"4096\",\"source_offset\":\"16384\","
    "\"stable_index\":\"2\"},{\"class\":\"ROUTED_EXPERT\","
#ifdef DS4_LAGUNA_TENSOR_RANGE_HAS_ROUTED_IDENTITY
    "\"routed_layer\":1,\"routed_projection\":\"UP\","
#endif
    "\"source_bytes\":\"4096\",\"source_offset\":\"20480\","
    "\"stable_index\":\"3\"},{\"class\":\"ROUTED_EXPERT\","
#ifdef DS4_LAGUNA_TENSOR_RANGE_HAS_ROUTED_IDENTITY
    "\"routed_layer\":1,\"routed_projection\":\"DOWN\","
#endif
    "\"source_bytes\":\"4096\",\"source_offset\":\"24576\","
    "\"stable_index\":\"4\"}]}";

static bool extract_ledger(const char *json, size_t json_size) {
    static const char prefix[] = "{\"allocation\":";
    const char *ledger = strstr(json, ",\"ledger\":");
    const char *digest = strstr(json, ",\"ledger_sha256\":");
    CHECK(json_size == strlen(json), "serialized bytes expose an exact size");
    CHECK(strncmp(json, prefix, sizeof(prefix) - 1u) == 0,
          "top-level allocation key is first");
    CHECK(ledger != NULL && digest != NULL && ledger < digest,
          "top-level ledger keys follow canonical order");
    if (!ledger || !digest) return false;
    ledger += sizeof(",\"ledger\":") - 1u;
    CHECK((size_t)(digest - ledger) == strlen(expected_ledger_json) &&
              memcmp(ledger, expected_ledger_json,
                     strlen(expected_ledger_json)) == 0,
          "included ledger object is exact canonical golden bytes");
    return true;
}

static void test_serialize(void) {
    plan_fixture fixture;
    char error[512] = "stale";
    CHECK(prepare_fixture(&fixture, error, sizeof(error)),
          "serializer fixture prepares");
    char *json = NULL;
    size_t json_size = 0;
    char ledger_digest[DS4_PLAN_IO_SHA256_HEX_SIZE];
    CHECK(ds4_laguna_qualification_plan_serialize(
              &fixture.input, &json, &json_size, ledger_digest,
              error, sizeof(error)),
          "valid qualification plan serializes");
    CHECK(error[0] == '\0', "successful serialization clears stale error");
    CHECK(json != NULL && extract_ledger(json, json_size),
          "serialized plan contains its full canonical ledger");
    CHECK(strcmp(ledger_digest,
#ifdef DS4_LAGUNA_TENSOR_RANGE_HAS_ROUTED_IDENTITY
                 "a2e857b0221fd6ca07ad57443e160396b"
                 "b76155b542ae7a9d88c63287de5d8a5"
#else
                 "dd6f102d4cd87b5d9fcbbaa5236797e3"
                 "b5474376e054f9cfe3eb7bf44b43ccfc"
#endif
                 ) == 0,
          "ledger digest hashes only the included canonical ledger object");
    CHECK(strstr(json, "\"file_size\":\"28673\"") != NULL &&
              strstr(json, "\"context_tokens\":32768") != NULL &&
              strstr(json, "\"context_tokens\":\"") == NULL,
          "uint64 fields are strings while bounded uint32 fields are integers");
    CHECK(strstr(json, "\"category\":\"STATIC_WEIGHTS\"") != NULL &&
              strstr(json, "\"domain\":\"CUDA_DEVICE\"") != NULL &&
              strstr(json, "\"report\":\"MODEL_SOURCE_RESIDENT\"") != NULL,
          "allocation enums use their fixed qualification names");
    CHECK(strstr(json, "\"repository\":\"poolside/Laguna-S-2.1-GGUF\"") != NULL &&
              strstr(json, "706fa69799926b6afde1af9e24ca2a4923f110a1") != NULL &&
              strstr(json,
                     "\"sha256\":\"e163b2c98908809a71245d6bb68b2226994d9969cb2a438eccb72196a1c4147a\"") != NULL &&
              strstr(json, "\"expected_sha256\"") == NULL &&
              strstr(json, "path") == NULL,
          "model identity publishes the observed pinned digest without a path");
    for (size_t i = 0; i < json_size; i++) {
        CHECK(json[i] != ' ' && json[i] != '\n' &&
                  json[i] != '\r' && json[i] != '\t',
              "canonical JSON contains no formatting whitespace");
    }
    const char *ledger_key = strstr(json, ",\"ledger\":");
    const char *ledger_sha_key = strstr(json, ",\"ledger_sha256\":");
    const char *model_key = strstr(json, ",\"model\":");
    const char *page_key = strstr(json, ",\"page_cache\":");
    const char *schema_key = strstr(json, ",\"schema\":");
    CHECK(ledger_key && ledger_sha_key && model_key && page_key && schema_key &&
              ledger_key < ledger_sha_key && ledger_sha_key < model_key &&
              model_key < page_key && page_key < schema_key,
          "top-level keys have the exact canonical order");

    char *again = NULL;
    size_t again_size = 0;
    char again_digest[DS4_PLAN_IO_SHA256_HEX_SIZE];
    CHECK(ds4_laguna_qualification_plan_serialize(
              &fixture.input, &again, &again_size, again_digest,
              error, sizeof(error)) &&
              again_size == json_size && memcmp(again, json, json_size) == 0 &&
              strcmp(again_digest, ledger_digest) == 0,
          "same plan serializes byte-for-byte deterministically");
    ds4_laguna_qualification_plan_bytes_free(again);

    fixture.experts[0].layer = 2u;
#ifdef DS4_LAGUNA_TENSOR_RANGE_HAS_ROUTED_IDENTITY
    for (size_t i = 2u; i < 5u; i++) fixture.tensors[i].routed_layer = 2u;
#endif
    again = NULL;
    again_size = 0;
    CHECK(ds4_laguna_qualification_plan_serialize(
              &fixture.input, &again, &again_size, again_digest,
              error, sizeof(error)) &&
              strcmp(again_digest, ledger_digest) != 0 &&
              (again_size != json_size || memcmp(again, json, json_size) != 0),
          "changing one valid ledger field changes both ledger and plan bytes");
    ds4_laguna_qualification_plan_bytes_free(again);
    fixture.experts[0].layer = 1u;
#ifdef DS4_LAGUNA_TENSOR_RANGE_HAS_ROUTED_IDENTITY
    for (size_t i = 2u; i < 5u; i++) fixture.tensors[i].routed_layer = 1u;
#endif

    ds4_laguna_qualification_plan_bytes_free(json);
    ds4_laguna_page_plan_free(&fixture.pages);
}

static unsigned char *read_file(const char *path, size_t *size_out) {
    FILE *file = fopen(path, "rb");
    if (!file) return NULL;
    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        return NULL;
    }
    const long length = ftell(file);
    if (length < 0 || fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return NULL;
    }
    unsigned char *bytes = malloc(length == 0 ? 1u : (size_t)length);
    if (!bytes) {
        fclose(file);
        return NULL;
    }
    if (length != 0 && fread(bytes, 1, (size_t)length, file) != (size_t)length) {
        free(bytes);
        fclose(file);
        return NULL;
    }
    if (fclose(file) != 0) {
        free(bytes);
        return NULL;
    }
    *size_out = (size_t)length;
    return bytes;
}

static bool make_temporary_directory(char *path_template) {
    const int descriptor = mkstemp(path_template);
    if (descriptor < 0) return false;
    if (close(descriptor) != 0) {
        (void)unlink(path_template);
        return false;
    }
    if (unlink(path_template) != 0) return false;
    return mkdir(path_template, 0700) == 0;
}

static void test_publish(void) {
    plan_fixture fixture;
    char error[512];
    CHECK(prepare_fixture(&fixture, error, sizeof(error)),
          "publication fixture prepares");
    char directory[] = "/tmp/ds4-laguna-plan.XXXXXX";
    CHECK(make_temporary_directory(directory),
          "publication test directory created");
    char path[512];
    char sidecar[520];
    CHECK(snprintf(path, sizeof(path), "%s/plan.json", directory) > 0 &&
              snprintf(sidecar, sizeof(sidecar), "%s.sha256", path) > 0,
          "publication paths fit");
    char plan_digest[DS4_PLAN_IO_SHA256_HEX_SIZE];
    char ledger_digest[DS4_PLAN_IO_SHA256_HEX_SIZE];
    CHECK(ds4_laguna_qualification_plan_publish(
              path, &fixture.input, plan_digest, ledger_digest,
              error, sizeof(error)),
          "qualification wrapper publishes plan and sidecar");
    size_t plan_size = 0;
    size_t sidecar_size = 0;
    unsigned char *plan_bytes = read_file(path, &plan_size);
    unsigned char *sidecar_bytes = read_file(sidecar, &sidecar_size);
    CHECK(plan_bytes != NULL && plan_size != 0,
          "published qualification plan contains exact bytes");
    CHECK(sidecar_bytes != NULL && sidecar_size == 65u &&
              memcmp(sidecar_bytes, plan_digest, 64u) == 0 &&
              sidecar_bytes[64] == '\n',
          "qualification publication delegates the exact digest sidecar");
    char recomputed[DS4_PLAN_IO_SHA256_HEX_SIZE];
    CHECK(ds4_plan_io_sha256(plan_bytes, plan_size, recomputed,
                             error, sizeof(error)) &&
              strcmp(recomputed, plan_digest) == 0,
          "published plan digest reproduces from exact file bytes");
    free(sidecar_bytes);
    free(plan_bytes);
    CHECK(unlink(sidecar) == 0 && unlink(path) == 0 && rmdir(directory) == 0,
          "publication test artifacts are removed");
    ds4_laguna_page_plan_free(&fixture.pages);
}

static void expect_serialize_rejected(plan_fixture *fixture,
                                      const char *message) {
    char *json = (char *)(uintptr_t)1u;
    size_t size = SIZE_MAX;
    char digest[DS4_PLAN_IO_SHA256_HEX_SIZE];
    memset(digest, 'x', sizeof(digest));
    char error[256];
    const bool ok = ds4_laguna_qualification_plan_serialize(
        &fixture->input, &json, &size, digest, error, sizeof(error));
    CHECK(!ok && json == NULL && size == 0u && digest[0] == '\0' &&
              error[0] != '\0',
          message);
}

static void expect_validation_rejected_without_mutation(
        plan_fixture *fixture,
        const char *message) {
    const plan_fixture fixture_before = *fixture;
    ds4_laguna_tensor_range tensors_before[8];
    ds4_laguna_source_range sources_before[19];
    ds4_laguna_expert_entry experts_before[4];
    ds4_laguna_page_range ranges_before[8];
    CHECK(fixture->ledger.tensor_range_count <= 8u &&
              fixture->ledger.source_range_count <= 19u &&
              fixture->ledger.expert_entry_count <= 4u &&
              fixture->pages.range_count <= 8u,
          "expert fixture snapshots fit");
    memcpy(tensors_before, fixture->ledger.tensor_ranges,
           fixture->ledger.tensor_range_count * sizeof(tensors_before[0]));
    memcpy(sources_before, fixture->ledger.source_ranges,
           fixture->ledger.source_range_count * sizeof(sources_before[0]));
    memcpy(experts_before, fixture->ledger.expert_entries,
           fixture->ledger.expert_entry_count * sizeof(experts_before[0]));
    memcpy(ranges_before, fixture->pages.ranges,
           fixture->pages.range_count * sizeof(ranges_before[0]));

    char *json = NULL;
    size_t size = 0;
    char digest[DS4_PLAN_IO_SHA256_HEX_SIZE];
    char error[256];
    const bool ok = ds4_laguna_qualification_plan_serialize(
        &fixture->input, &json, &size, digest, error, sizeof(error));
    CHECK(!ok && json == NULL && size == 0u && error[0] != '\0', message);
    CHECK(memcmp(&fixture_before, fixture, sizeof(fixture_before)) == 0 &&
              memcmp(tensors_before, fixture->ledger.tensor_ranges,
                     fixture->ledger.tensor_range_count *
                         sizeof(tensors_before[0])) == 0 &&
              memcmp(sources_before, fixture->ledger.source_ranges,
                     fixture->ledger.source_range_count *
                         sizeof(sources_before[0])) == 0 &&
              memcmp(experts_before, fixture->ledger.expert_entries,
                     fixture->ledger.expert_entry_count *
                         sizeof(experts_before[0])) == 0 &&
              memcmp(ranges_before, fixture->pages.ranges,
                     fixture->pages.range_count * sizeof(ranges_before[0])) == 0,
          "expert evidence rejection leaves the fixture byte-identical");
    ds4_laguna_qualification_plan_bytes_free(json);
}

static void test_expert_evidence_rejections(void) {
    plan_fixture fixture;
    char error[256];

    CHECK(prepare_expert_fixture(&fixture, error, sizeof(error)),
          "wrong-expert-offset fixture prepares");
    fixture.ledger.expert_entries[1].gate.source_offset =
        fixture.ledger.expert_entries[0].gate.source_offset;
    fixture.ledger.expert_entries[1].up.source_offset =
        fixture.ledger.expert_entries[0].up.source_offset;
    fixture.ledger.expert_entries[1].down.source_offset =
        fixture.ledger.expert_entries[0].down.source_offset;
    expect_validation_rejected_without_mutation(
        &fixture, "expert 1 cannot reuse expert 0 source offsets");
    ds4_laguna_page_plan_free(&fixture.pages);
    ds4_laguna_ledger_free(&fixture.ledger);

    CHECK(prepare_expert_fixture(&fixture, error, sizeof(error)),
          "wrong-projection-parent fixture prepares");
    ds4_laguna_expert_entry *entry = &fixture.ledger.expert_entries[1];
    const uint64_t gate_parent = entry->gate.parent_stable_index;
    const uint64_t gate_offset = entry->gate.source_offset;
    entry->gate.parent_stable_index = entry->up.parent_stable_index;
    entry->gate.source_offset = entry->up.source_offset;
    entry->up.parent_stable_index = gate_parent;
    entry->up.source_offset = gate_offset;
    expect_validation_rejected_without_mutation(
        &fixture, "gate and up views cannot exchange routed parents");
    ds4_laguna_page_plan_free(&fixture.pages);
    ds4_laguna_ledger_free(&fixture.ledger);

    CHECK(prepare_expert_fixture(&fixture, error, sizeof(error)),
          "cross-layer-parent fixture prepares");
    entry = &fixture.ledger.expert_entries[0];
    const ds4_laguna_expert_entry *other = &fixture.ledger.expert_entries[2];
    entry->gate.parent_stable_index = other->gate.parent_stable_index;
    entry->gate.source_offset = other->gate.source_offset;
    entry->up.parent_stable_index = other->up.parent_stable_index;
    entry->up.source_offset = other->up.source_offset;
    entry->down.parent_stable_index = other->down.parent_stable_index;
    entry->down.source_offset = other->down.source_offset;
    expect_validation_rejected_without_mutation(
        &fixture, "expert views cannot use another layer's parent triplet");
    ds4_laguna_page_plan_free(&fixture.pages);
    ds4_laguna_ledger_free(&fixture.ledger);
}

static void test_rejections(void) {
    plan_fixture fixture;
    char error[256];
    CHECK(prepare_fixture(&fixture, error, sizeof(error)),
          "rejection fixture prepares");

    fixture.tensors[1].stable_index = 0u;
    expect_serialize_rejected(&fixture, "duplicate or unordered tensor index rejected");
    fixture.tensors[1].stable_index = 1u;

    fixture.tensors[0].tensor_class = DS4_LAGUNA_TENSOR_UNCLASSIFIED;
    expect_serialize_rejected(&fixture, "unclassified tensor rejected");
    fixture.tensors[0].tensor_class = DS4_LAGUNA_TENSOR_STATIC;

    fixture.tensors[0].source_bytes = UINT64_MAX;
    expect_serialize_rejected(&fixture, "overflowing tensor range rejected");
    fixture.tensors[0].source_bytes = 8093u;

    fixture.sources[1].source_offset = 10u;
    expect_serialize_rejected(&fixture, "unordered source range rejected");
    fixture.sources[1].source_offset = 20u;

    fixture.sources[1].kind = DS4_LAGUNA_SOURCE_HEADER;
    expect_serialize_rejected(&fixture, "misclassified source range rejected");
    fixture.sources[1].kind = DS4_LAGUNA_SOURCE_METADATA;

    fixture.allocation.callsites[0].category =
        DS4_RUNTIME_CATEGORY_EXPERT_CACHE_PAYLOAD;
    expect_serialize_rejected(&fixture, "wrong callsite category rejected");
    fixture.allocation.callsites[0].category = DS4_RUNTIME_CATEGORY_STATIC_WEIGHTS;

    fixture.allocation.callsites[0].domain = DS4_RUNTIME_DOMAIN_HOST;
    expect_serialize_rejected(&fixture, "wrong callsite physical domain rejected");
    fixture.allocation.callsites[0].domain = DS4_RUNTIME_DOMAIN_CUDA_DEVICE;

    fixture.allocation.callsites[0].id = 2u;
    expect_serialize_rejected(&fixture, "reordered or duplicate callsite rejected");
    fixture.allocation.callsites[0].id = 1u;

    fixture.allocation.profile_id = "cache-10gib";
    expect_serialize_rejected(&fixture, "unknown allocation profile rejected");
    fixture.allocation.profile_id = "cache-8gib";

    fixture.input.model_identity.size_bytes++;
    expect_serialize_rejected(&fixture, "opened-file size mismatch rejected");
    fixture.input.model_identity.size_bytes--;

    fixture.input.model_identity.device = 0u;
    expect_serialize_rejected(&fixture, "placeholder model identity rejected");
    fixture.input.model_identity.device = 42u;

    fixture.input.model_sha256 = NULL;
    expect_serialize_rejected(&fixture, "missing observed model digest rejected");
    fixture.input.model_sha256 = "E163b2c98908809a71245d6bb68b2226994d9969cb2a438eccb72196a1c4147a";
    expect_serialize_rejected(&fixture, "malformed observed model digest rejected");
    fixture.input.model_sha256 =
        "0000000000000000000000000000000000000000000000000000000000000000";
    expect_serialize_rejected(&fixture, "non-pinned observed model digest rejected");
    fixture.input.model_sha256 = pinned_model_sha256;

    fixture.pages.ranges[0].offset = 0u;
    expect_serialize_rejected(&fixture, "noncanonical page union rejected");
    fixture.pages.ranges[0].offset = 4096u;

    fixture.ledger.tensor_ranges = NULL;
    expect_serialize_rejected(&fixture, "null nonempty ledger array rejected");
    fixture.ledger.tensor_ranges = fixture.tensors;

    ds4_laguna_page_plan_free(&fixture.pages);
}

int main(void) {
    test_page_plan();
    test_serialize();
    test_publish();
    test_rejections();
    test_expert_evidence_rejections();
    if (g_failed != 0) {
        fprintf(stderr, "%d/%d Laguna plan checks failed\n", g_failed, g_total);
        return 1;
    }
    printf("ok: %d Laguna plan checks\n", g_total);
    return 0;
}
