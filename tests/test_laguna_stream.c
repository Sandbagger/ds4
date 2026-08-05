/* Pure-C policy tests for the Laguna compact runtime.
 *
 * Keep option parsing here independent of CUDA and model files so every host
 * can enforce the same canonical configuration contract. */

#include "ds4.h"
#include "ds4_laguna_stream.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int g_failed;
static int g_total;

#define CHECK(cond, msg) do {                                                  \
    g_total++;                                                                 \
    if (!(cond)) {                                                             \
        fprintf(stderr, "FAIL: %s (line %d)\n", (msg), __LINE__);             \
        g_failed++;                                                            \
    }                                                                          \
} while (0)

static void check_parse_ok(const char *text, uint64_t expected) {
    uint64_t value = 0;
    const bool ok = ds4_parse_positive_u64_decimal(text, &value);
    if (!ok || value != expected) {
        fprintf(stderr,
                "parse mismatch for '%s': ok=%d value=%" PRIu64
                " expected=%" PRIu64 "\n",
                text,
                ok ? 1 : 0,
                value,
                expected);
    }
    CHECK(ok && value == expected, "canonical positive uint64 accepted exactly");
}

static void check_parse_rejected(const char *text) {
    uint64_t value = UINT64_MAX;
    const bool ok = ds4_parse_positive_u64_decimal(text, &value);
    if (ok || value != 0) {
        fprintf(stderr,
                "unexpected parse for %s: ok=%d value=%" PRIu64 "\n",
                text ? text : "<null>",
                ok ? 1 : 0,
                value);
    }
    CHECK(!ok && value == 0, "non-canonical uint64 rejected and output cleared");
}

static void test_options(void) {
    check_parse_ok("1", UINT64_C(1));
    check_parse_ok("8589934592", UINT64_C(8589934592));
    check_parse_ok("18446744073709551615", UINT64_MAX);

    static const char *const rejected[] = {
        "",
        "0",
        "+1",
        "-1",
        " 1",
        "1 ",
        "1\n",
        "1GB",
        "01",
        "00",
        "18446744073709551616",
        "1x",
    };
    CHECK(!ds4_parse_positive_u64_decimal("1", NULL),
          "null output pointer rejected");
    check_parse_rejected(NULL);
    for (size_t i = 0; i < sizeof(rejected) / sizeof(rejected[0]); i++) {
        check_parse_rejected(rejected[i]);
    }

    const uint64_t gib = UINT64_C(1024) * 1024u * 1024u;
    const uint64_t recommended = 24u * gib;
    const uint64_t graph_context = 4u * gib;
    const uint64_t graph_safe =
        ds4_test_graph_cache_safe_bytes(recommended, graph_context);
    CHECK(graph_safe == 17u * gib,
          "graph safety applies the exact 7/8-minus-context policy");
    char err[256];
    CHECK(18u * gib < recommended && 18u * gib > graph_safe,
          "graph-pressure case is below device recommendation and above safe bound");
    memset(err, 0, sizeof(err));
    CHECK(ds4_test_exact_cache_options_preflight(
              true, true, false, 18u * gib, true, graph_safe,
              err, sizeof(err)) == 2,
          "exact cache above graph-pressure safety limit is config error");
    CHECK(strstr(err, "graph working-set pressure budget") != NULL,
          "graph-pressure preflight explains the safe bound");

    memset(err, 0, sizeof(err));
    CHECK(ds4_test_exact_cache_options_preflight(
              true, true, false, 16u * gib, true, graph_safe,
              err, sizeof(err)) == 0,
          "exact cache below graph-pressure safety limit passes preflight");
    CHECK(err[0] == '\0', "successful cache preflight leaves no error");

    const uint64_t exhausted_safe =
        ds4_test_graph_cache_safe_bytes(15u * gib, 13u * gib);
    CHECK(exhausted_safe == 0,
          "strict graph safety does not invent a 1GiB residual budget");
    memset(err, 0, sizeof(err));
    CHECK(ds4_test_exact_cache_options_preflight(
              true, true, false, 1u, true, exhausted_safe,
              err, sizeof(err)) == 2,
          "known exhausted graph budget fails canonical exact cache closed");
    memset(err, 0, sizeof(err));
    CHECK(ds4_test_exact_cache_options_preflight(
              true, true, false, 1u, false, 0,
              err, sizeof(err)) == 0,
          "unavailable device bound remains distinct from exhausted bound");

    const uint64_t max_target =
        UINT64_MAX - UINT64_MAX / 8u - 1u;
    CHECK(ds4_test_graph_cache_safe_bytes(UINT64_MAX, 0) ==
              (max_target / gib) * gib,
          "7/8 working-set target remains exact at uint64 boundary");

    ds4_test_graph_family worst_family = DS4_TEST_GRAPH_FAMILY_FLASH;
    const uint64_t max_bound = ds4_test_graph_context_max_bound(
        32768u, 4096u, &worst_family);
    const uint64_t flash_bound = ds4_test_graph_context_bound(
        DS4_TEST_GRAPH_FAMILY_FLASH, 32768u, 4096u);
    const uint64_t pro_bound = ds4_test_graph_context_bound(
        DS4_TEST_GRAPH_FAMILY_PRO, 32768u, 4096u);
    const uint64_t glm_bound = ds4_test_graph_context_bound(
        DS4_TEST_GRAPH_FAMILY_GLM, 32768u, 4096u);
    const uint64_t laguna_bound = ds4_test_graph_context_bound(
        DS4_TEST_GRAPH_FAMILY_LAGUNA, 32768u, 4096u);
    CHECK(max_bound != 0 && flash_bound != 0 && pro_bound != 0 &&
          glm_bound != 0 && laguna_bound != 0,
          "model-independent graph context bounds are available at 32K");
    CHECK(worst_family == DS4_TEST_GRAPH_FAMILY_GLM &&
          max_bound == glm_bound,
          "preflight selects the worst supported family without model metadata");
    CHECK(glm_bound >= flash_bound &&
          glm_bound >= pro_bound &&
          glm_bound >= laguna_bound,
          "selected worst-family bound dominates every supported family");

    ds4_test_graph_family full_prefill_worst =
        DS4_TEST_GRAPH_FAMILY_FLASH;
    const uint64_t full_prefill_max = ds4_test_graph_context_max_bound(
        32768u, 32768u, &full_prefill_worst);
    const uint64_t full_prefill_pro = ds4_test_graph_context_bound(
        DS4_TEST_GRAPH_FAMILY_PRO, 32768u, 32768u);
    CHECK(full_prefill_worst == DS4_TEST_GRAPH_FAMILY_PRO &&
          full_prefill_max == full_prefill_pro,
          "32K full-prefill safety selects DeepSeek Pro as worst family");
    const uint64_t deepseek_full_prefill_floor =
        UINT64_C(32768) * 2u * 1024u * 1024u +
        UINT64_C(2) * (32768u / 4u + 2u) * 32768u * sizeof(float) +
        gib;
    CHECK(full_prefill_pro >= deepseek_full_prefill_floor,
          "DeepSeek Pro bound includes row, score/mask, and fixed prefill scratch");
    CHECK(ds4_test_graph_context_max_bound(
              UINT32_MAX, UINT32_MAX, NULL) == UINT64_MAX,
          "overflowing graph bounds saturate and fail cache admission closed");
    CHECK(ds4_test_graph_cache_safe_bytes(128u * gib, full_prefill_max) >=
              16u * gib,
          "DGX 128GB still admits the 16GiB profile at a full 32K prefill cap");
    CHECK(laguna_bound > 7u * gib,
          "Laguna bound includes its 16K-row prefill scratch allocation");

    const uint64_t current_laguna_allocation =
        UINT64_C(16384) * 375156u + 413704u +
        (UINT64_C(12) * 32768u + UINT64_C(36) * 512u) * 4096u;
    CHECK(laguna_bound >= current_laguna_allocation,
          "coarse Laguna bound dominates the current exact 16K-row allocator");
    CHECK(ds4_test_graph_context_bound(DS4_TEST_GRAPH_FAMILY_LAGUNA,
                                       32768u,
                                       16384u) == laguna_bound,
          "Laguna bound remains conservative across Task 11's 4K-row reduction");

    const ds4_test_graph_family families[] = {
        DS4_TEST_GRAPH_FAMILY_FLASH,
        DS4_TEST_GRAPH_FAMILY_PRO,
        DS4_TEST_GRAPH_FAMILY_GLM,
        DS4_TEST_GRAPH_FAMILY_LAGUNA,
    };
    const uint64_t family_bounds[] = {
        flash_bound,
        pro_bound,
        glm_bound,
        laguna_bound,
    };
    for (size_t i = 0; i < sizeof(families) / sizeof(families[0]); i++) {
        const uint64_t post_bind = ds4_test_graph_context_memory_bytes(
            families[i], 32768u, 4096u);
        CHECK(post_bind != 0 && family_bounds[i] >= post_bind,
              "conservative family bound dominates real post-bind estimator at 32K");
    }

    const uint64_t unaligned_flash = ds4_test_graph_context_bound(
        DS4_TEST_GRAPH_FAMILY_FLASH, 4200u, 4096u);
    const uint64_t unaligned_pro = ds4_test_graph_context_bound(
        DS4_TEST_GRAPH_FAMILY_PRO, 4200u, 4096u);
    CHECK(unaligned_flash != 0 && unaligned_pro != 0,
          "all-family bounds are available at non-256-aligned context");
    CHECK(unaligned_flash >=
              ds4_test_graph_context_memory_bytes(
                  DS4_TEST_GRAPH_FAMILY_FLASH, 4200u, 4096u) &&
          unaligned_pro >=
              ds4_test_graph_context_memory_bytes(
                  DS4_TEST_GRAPH_FAMILY_PRO, 4200u, 4096u),
          "DeepSeek bounds include graph raw-cap alignment at ctx 4200");

    const uint64_t tiny_flash = ds4_test_graph_context_bound(
        DS4_TEST_GRAPH_FAMILY_FLASH, 127u, 1u);
    const uint64_t tiny_pro = ds4_test_graph_context_bound(
        DS4_TEST_GRAPH_FAMILY_PRO, 127u, 1u);
    CHECK(tiny_flash != 0 && tiny_pro != 0,
          "all-family bounds are available below one raw-cap alignment block");
    CHECK(tiny_flash >=
              ds4_test_graph_context_memory_bytes(
                  DS4_TEST_GRAPH_FAMILY_FLASH, 127u, 1u) &&
          tiny_pro >=
              ds4_test_graph_context_memory_bytes(
                  DS4_TEST_GRAPH_FAMILY_PRO, 127u, 1u),
          "DeepSeek bounds include the minimum 256-row graph allocation");

    const uint64_t dgx_recommended = 128u * gib;
    const uint64_t dgx_safe =
        ds4_test_graph_cache_safe_bytes(dgx_recommended, max_bound);
    CHECK(dgx_safe >= 16u * gib,
          "DGX 128GB 32K bound admits the planned 16GiB profile");
    CHECK(8u * gib <= dgx_safe &&
          12u * gib <= dgx_safe &&
          16u * gib <= dgx_safe,
          "DGX bound admits all planned exact cache profiles");
    memset(err, 0, sizeof(err));
    CHECK(ds4_test_engine_exact_cache_preflight(
              true, dgx_recommended, 16u * gib, 32768u, 4096u,
              1u, false,
              err, sizeof(err)) == 0,
          "engine preflight admits the planned 16GiB DGX profile");

    memset(err, 0, sizeof(err));
    CHECK(ds4_test_exact_cache_cuda_topology_preflight(
              1u, 0, err, sizeof(err)) == 0,
          "exact graph cache accepts one explicitly budgeted CUDA device");

    memset(err, 0, sizeof(err));
    CHECK(ds4_test_exact_cache_cuda_topology_preflight(
              2u, 0, err, sizeof(err)) == 2 &&
          strstr(err, "one CUDA device") != NULL,
          "exact graph cache rejects aggregate multi-GPU budget pricing");

    memset(err, 0, sizeof(err));
    CHECK(ds4_test_exact_cache_cuda_topology_preflight(
              1u, 3, err, sizeof(err)) == 2 &&
          strstr(err, "CUDA device 0") != NULL,
          "exact graph cache rejects a budget for a device it will not initialize");

    memset(err, 0, sizeof(err));
    CHECK(ds4_test_engine_exact_cache_preflight(
              true, dgx_recommended, 16u * gib, 0u, 4096u,
              1u, false,
              err, sizeof(err)) == 2 &&
          strstr(err, "--ctx") != NULL,
          "exact graph cache requires a declared positive safety context");

    memset(err, 0, sizeof(err));
    CHECK(ds4_test_engine_exact_cache_preflight(
              true, dgx_recommended, 16u * gib, 32768u, 4096u,
              2u, false,
              err, sizeof(err)) == 2 &&
          strstr(err, "--session-slots") != NULL,
          "exact graph cache rejects unpriced multi-session serving");

    memset(err, 0, sizeof(err));
    CHECK(ds4_test_engine_exact_cache_preflight(
              true, dgx_recommended, 16u * gib, 32768u, 4096u,
              1u, true,
              err, sizeof(err)) == 2 &&
          strstr(err, "shared prefill workspace") != NULL,
          "exact graph cache rejects persistent shared prefill workspace");

    const uint64_t selected_budgets[] = { 40u * gib, 12u * gib };
    uint64_t selected_working_set = 0;
    CHECK(ds4_test_gpu_config_working_set_bytes(
              selected_budgets,
              sizeof(selected_budgets) / sizeof(selected_budgets[0]),
              &selected_working_set) &&
          selected_working_set == 52u * gib,
          "selected CUDA topology uses the sum of its explicit budgets");
    uint64_t default_working_set = 0;
    CHECK(ds4_test_default_single_tier_working_set_bytes(
              24u * gib, 8u, &default_working_set) &&
          default_working_set == 24u * gib,
          "no-config CUDA topology uses only device 0 when eight GPUs are visible");
    memset(err, 0, sizeof(err));
    CHECK(ds4_test_engine_exact_cache_preflight(
              true, default_working_set, 8u * gib, 32768u, 4096u,
              1u, false,
              err, sizeof(err)) == 2,
          "no-config engine rejects cache unsafe for its default single GPU");
    memset(err, 0, sizeof(err));
    CHECK(ds4_test_engine_exact_cache_preflight(
              false, 0, 8u * gib, 32768u, 4096u,
              1u, false,
              err, sizeof(err)) == 2 &&
          strstr(err, "working-set limit is unavailable") != NULL,
          "engine exact graph preflight fails closed when CUDA probing fails");

    memset(err, 0, sizeof(err));
    CHECK(dgx_safe + gib < dgx_recommended &&
          ds4_test_exact_cache_options_preflight(
              true, true, false, dgx_safe + gib, true, dgx_safe,
              err, sizeof(err)) == 2,
          "below-device value above conservative all-family bound is rejected");

    const uint64_t stride = 64u * 1024u * 1024u;
    uint64_t effective_cache_bytes = 0;
    uint32_t slot_count = 0;
    CHECK(ds4_test_exact_cache_plan_make(8u * gib, stride,
                                         &effective_cache_bytes,
                                         &slot_count),
          "exact cache plan accepts an integral expert-slot count");
    CHECK(effective_cache_bytes == 8u * gib,
          "exact cache plan effective bytes equal configured bytes");
    CHECK(slot_count == 128u,
          "exact cache plan derives the exact expert-slot count");

    CHECK(ds4_test_exact_cache_plan_make(8u * gib + 1u, stride,
                                         &effective_cache_bytes,
                                         &slot_count),
          "exact cache plan accepts a partial trailing slot");
    CHECK(effective_cache_bytes == 8u * gib + 1u && slot_count == 128u,
          "partial-slot plan floors slots without rewriting exact limit");
    CHECK(!ds4_test_exact_cache_plan_make(8u * gib, 0,
                                          &effective_cache_bytes,
                                          &slot_count),
          "exact cache plan rejects an unknown zero stride");
    CHECK(!ds4_test_exact_cache_plan_make(stride - 1u, stride,
                                          &effective_cache_bytes,
                                          &slot_count),
          "exact cache plan rejects a limit with zero complete slots");
    CHECK(!ds4_test_exact_cache_plan_make((uint64_t)UINT32_MAX + 1u,
                                         1u,
                                         &effective_cache_bytes,
                                         &slot_count),
          "exact cache plan rejects slot counts wider than uint32");

    uint64_t post_prefill = 0;
    CHECK(ds4_test_post_prefill_cache_budget(8u * gib,
                                            2u * gib,
                                            true,
                                            &post_prefill) &&
          post_prefill == 8u * gib,
          "exact cache ceiling never grows after prefill");
    CHECK(ds4_test_post_prefill_cache_budget(8u * gib,
                                            2u * gib,
                                            false,
                                            &post_prefill) &&
          post_prefill == 10u * gib,
          "legacy cache budget may reclaim prefill headroom");
    post_prefill = UINT64_MAX;
    CHECK(!ds4_test_post_prefill_cache_budget(UINT64_MAX,
                                             1u,
                                             false,
                                             &post_prefill) &&
          post_prefill == 0,
          "legacy post-prefill growth fails closed on overflow");
}

enum {
    FIXTURE_TENSOR_COUNT = 7,
};

typedef struct {
    ds4_laguna_ledger_spec spec;
    ds4_laguna_tensor_desc tensors[FIXTURE_TENSOR_COUNT];
    char names[FIXTURE_TENSOR_COUNT][24];
} ledger_fixture;

static void fixture_tensor(
        ledger_fixture *f,
        size_t index,
        const char *name,
        uint64_t source_offset,
        ds4_laguna_tensor_class tensor_class,
        uint32_t routed_layer,
        ds4_laguna_routed_projection projection) {
    ds4_laguna_tensor_desc *t = &f->tensors[index];
    memset(t, 0, sizeof(*t));
    snprintf(f->names[index], sizeof(f->names[index]), "%s", name);
    t->stable_index = (uint64_t)index + 10u;
    t->name = f->names[index];
    t->name_len = strlen(f->names[index]);
    t->source_offset = source_offset;
    t->gguf_type = 2;
    t->block_elems = 32;
    t->block_bytes = 18;
    t->tensor_class = tensor_class;
    t->routed_layer = routed_layer;
    t->routed_projection = projection;

    if (tensor_class == DS4_LAGUNA_TENSOR_STATIC) {
        t->ndim = 1;
        t->dim[0] = 33;
        t->source_bytes = 36;
    } else {
        t->ndim = 3;
        t->dim[0] = 32;
        t->dim[1] = 2;
        t->dim[2] = 2;
        t->source_bytes = 72;
    }
}

static void valid_ledger_fixture(ledger_fixture *f) {
    memset(f, 0, sizeof(*f));
    f->spec.file_size = 1024;
    f->spec.header_end = 16;
    f->spec.metadata_end = 32;
    f->spec.tensor_directory_end = 65;
    f->spec.tensor_data_start = 128;
    f->spec.gguf_alignment = 64;
    f->spec.device_alignment = 256;
    f->spec.first_routed_layer = 1;
    f->spec.layer_count = 3;
    f->spec.expert_count = 2;

    fixture_tensor(f, 0, "blk.0.ffn_shexp.weight", 128,
                   DS4_LAGUNA_TENSOR_STATIC, UINT32_MAX,
                   DS4_LAGUNA_ROUTED_PROJECTION_NONE);
    fixture_tensor(f, 1, "blk.1.ffn_gate_exps.weight", 192,
                   DS4_LAGUNA_TENSOR_ROUTED_EXPERT, 1,
                   DS4_LAGUNA_ROUTED_PROJECTION_GATE);
    fixture_tensor(f, 2, "blk.1.ffn_up_exps.weight", 320,
                   DS4_LAGUNA_TENSOR_ROUTED_EXPERT, 1,
                   DS4_LAGUNA_ROUTED_PROJECTION_UP);
    fixture_tensor(f, 3, "blk.1.ffn_down_exps.weight", 448,
                   DS4_LAGUNA_TENSOR_ROUTED_EXPERT, 1,
                   DS4_LAGUNA_ROUTED_PROJECTION_DOWN);
    fixture_tensor(f, 4, "blk.2.ffn_gate_exps.weight", 576,
                   DS4_LAGUNA_TENSOR_ROUTED_EXPERT, 2,
                   DS4_LAGUNA_ROUTED_PROJECTION_GATE);
    fixture_tensor(f, 5, "blk.2.ffn_up_exps.weight", 704,
                   DS4_LAGUNA_TENSOR_ROUTED_EXPERT, 2,
                   DS4_LAGUNA_ROUTED_PROJECTION_UP);
    fixture_tensor(f, 6, "blk.2.ffn_down_exps.weight", 832,
                   DS4_LAGUNA_TENSOR_ROUTED_EXPERT, 2,
                   DS4_LAGUNA_ROUTED_PROJECTION_DOWN);
}

static bool build_fixture(const ledger_fixture *f,
                          ds4_laguna_ledger *ledger,
                          char *err,
                          size_t errlen) {
    memset(ledger, 0, sizeof(*ledger));
    memset(err, 0, errlen);
    return ds4_laguna_ledger_build(ledger,
                                   &f->spec,
                                   f->tensors,
                                   FIXTURE_TENSOR_COUNT,
                                   err,
                                   errlen);
}

static void check_fixture_rejected(ledger_fixture *f, const char *needle,
                                   const char *message) {
    ds4_laguna_ledger ledger;
    char err[256];
    const bool ok = build_fixture(f, &ledger, err, sizeof(err));
    if (ok || (needle && !strstr(err, needle))) {
        fprintf(stderr, "unexpected ledger result: ok=%d err=%s\n",
                ok ? 1 : 0, err);
    }
    CHECK(!ok, message);
    CHECK(needle == NULL || strstr(err, needle) != NULL,
          "ledger rejection has a stable diagnostic");
    ds4_laguna_ledger_free(&ledger);
}

static const ds4_laguna_source_range *find_source_range(
        const ds4_laguna_ledger *ledger,
        ds4_laguna_source_range_kind kind) {
    for (size_t i = 0; i < ledger->source_range_count; i++) {
        if (ledger->source_ranges[i].kind == kind) {
            return &ledger->source_ranges[i];
        }
    }
    return NULL;
}

static void test_ledger_valid(void) {
    ledger_fixture f;
    valid_ledger_fixture(&f);
    ds4_laguna_ledger ledger;
    char err[256];
    CHECK(build_fixture(&f, &ledger, err, sizeof(err)),
          "valid synthetic Laguna tensor partition builds");
    CHECK(err[0] == '\0', "successful ledger build leaves no error");
    CHECK(ledger.file_size == 1024, "ledger preserves exact file size");
    CHECK(ledger.tensor_count == 7 &&
          ledger.static_parent_count == 1 &&
          ledger.routed_parent_count == 6,
          "ledger counts tensor parents by exact class");
    CHECK(ledger.expert_entry_count == 4,
          "expert entries count layer/expert pairs rather than projections");
    CHECK(ledger.routed_source_bytes == 432 &&
          ledger.static_source_bytes == 36 &&
          ledger.static_aligned_device_bytes == 256,
          "ledger reports exact routed, static, and device-aligned bytes");
    CHECK(ledger.non_tensor_source_bytes == 556 &&
          ledger.routed_source_bytes + ledger.static_source_bytes +
              ledger.non_tensor_source_bytes == ledger.file_size,
          "parent and non-tensor ranges reconcile the file exactly");
    CHECK(ledger.routed_projection_expert_bytes == 36,
          "ledger records the common routed projection/expert size");
    CHECK(ledger.slot_stride_bytes == 768,
          "slot stride includes gate/up/down device alignment");
    CHECK(ledger.tensor_range_count == FIXTURE_TENSOR_COUNT,
          "expert views do not become duplicate parent tensor ranges");

    const ds4_laguna_source_range *header =
        find_source_range(&ledger, DS4_LAGUNA_SOURCE_HEADER);
    const ds4_laguna_source_range *metadata =
        find_source_range(&ledger, DS4_LAGUNA_SOURCE_METADATA);
    const ds4_laguna_source_range *directory =
        find_source_range(&ledger, DS4_LAGUNA_SOURCE_TENSOR_DIRECTORY);
    const ds4_laguna_source_range *alignment =
        find_source_range(&ledger, DS4_LAGUNA_SOURCE_ALIGNMENT_PADDING);
    CHECK(header && header->source_offset == 0 && header->source_bytes == 16,
          "header source range is explicit");
    CHECK(metadata && metadata->source_offset == 16 &&
              metadata->source_bytes == 16,
          "metadata source range is explicit");
    CHECK(directory && directory->source_offset == 32 &&
              directory->source_bytes == 33,
          "tensor directory source range is explicit");
    CHECK(alignment && alignment->source_offset == 65 &&
              alignment->source_bytes == 63,
          "pre-data alignment padding is explicit");

    size_t tensor_padding_count = 0;
    uint64_t tensor_padding_bytes = 0;
    for (size_t i = 0; i < ledger.source_range_count; i++) {
        if (ledger.source_ranges[i].kind ==
            DS4_LAGUNA_SOURCE_TENSOR_PADDING) {
            tensor_padding_count++;
            tensor_padding_bytes += ledger.source_ranges[i].source_bytes;
        }
    }
    CHECK(tensor_padding_count == 7 && tensor_padding_bytes == 428,
          "tensor gaps and tail are synthesized as exact padding ranges");

    const ds4_laguna_expert_entry *entry = &ledger.expert_entries[0];
    CHECK(entry->layer == 1 && entry->expert == 0,
          "expert entries are deterministically ordered by layer and expert");
    CHECK(entry->gate.source_offset == 192 &&
          entry->gate.source_bytes == 36 &&
          entry->gate.device_offset == 0,
          "gate view points into its parent and starts the slot");
    CHECK(entry->up.source_offset == 320 &&
          entry->up.source_bytes == 36 &&
          entry->up.device_offset == 256,
          "up view follows aligned gate storage");
    CHECK(entry->down.source_offset == 448 &&
          entry->down.source_bytes == 36 &&
          entry->down.device_offset == 512 &&
          entry->used_bytes == 768,
          "down view and used bytes close the aligned slot");
    entry = &ledger.expert_entries[3];
    CHECK(entry->layer == 2 && entry->expert == 1 &&
          entry->gate.source_offset == 612 &&
          entry->up.source_offset == 740 &&
          entry->down.source_offset == 868,
          "last expert views use dim0-fastest parent arithmetic");

    CHECK(ledger.tensor_ranges[0].tensor_class == DS4_LAGUNA_TENSOR_STATIC,
          "quantized shared-expert tensor remains static");
    CHECK(!ds4_laguna_ledger_build(&ledger, &f.spec, f.tensors,
                                   FIXTURE_TENSOR_COUNT,
                                   err, sizeof(err)) &&
              strstr(err, "empty") != NULL,
          "building into an owned ledger is rejected without leaking it");
    ds4_laguna_ledger_free(&ledger);
    CHECK(ledger.tensor_ranges == NULL && ledger.expert_entries == NULL &&
          ledger.source_ranges == NULL,
          "ledger free clears owned arrays");
}

static void test_ledger_identity_and_class_rejections(void) {
    ledger_fixture f;
    valid_ledger_fixture(&f);
    f.tensors[0].tensor_class = DS4_LAGUNA_TENSOR_UNCLASSIFIED;
    check_fixture_rejected(&f, "unclassified", "unclassified tensor rejected");

    valid_ledger_fixture(&f);
    f.tensors[1].stable_index = f.tensors[0].stable_index;
    check_fixture_rejected(&f, "stable index", "duplicate stable index rejected");

    valid_ledger_fixture(&f);
    static const char duplicate_name_storage[] =
        "xxblk.0.ffn_shexp.weightyy";
    f.tensors[1].name = duplicate_name_storage + 2;
    f.tensors[1].name_len = strlen("blk.0.ffn_shexp.weight");
    check_fixture_rejected(&f, "name", "duplicate length-delimited name rejected");

    valid_ledger_fixture(&f);
    f.tensors[0].routed_projection = DS4_LAGUNA_ROUTED_PROJECTION_GATE;
    check_fixture_rejected(&f, "static", "static tensor cannot claim routed identity");

    valid_ledger_fixture(&f);
    f.tensors[0].name = NULL;
    check_fixture_rejected(&f, "name", "null length-delimited name rejected");

    valid_ledger_fixture(&f);
    f.tensors[0].name_len = 0;
    check_fixture_rejected(&f, "name", "empty length-delimited name rejected");
}

static void test_ledger_source_rejections(void) {
    ledger_fixture f;
    valid_ledger_fixture(&f);
    f.tensors[2].source_offset = 256;
    check_fixture_rejected(&f, "overlap", "overlapping parent ranges rejected");

    valid_ledger_fixture(&f);
    f.tensors[6].source_offset = 960;
    check_fixture_rejected(&f, "file", "truncated tensor range rejected");

    valid_ledger_fixture(&f);
    f.tensors[0].source_offset = f.spec.tensor_data_start - 64;
    check_fixture_rejected(&f, "tensor data", "tensor before data section rejected");

    valid_ledger_fixture(&f);
    f.tensors[0].source_bytes++;
    check_fixture_rejected(&f, "source size", "static source size mismatch rejected");

    valid_ledger_fixture(&f);
    f.tensors[1].source_bytes++;
    check_fixture_rejected(&f, "source size", "routed source size mismatch rejected");

    valid_ledger_fixture(&f);
    f.spec.gguf_alignment = 0;
    check_fixture_rejected(&f, "alignment", "zero GGUF alignment rejected");

    valid_ledger_fixture(&f);
    f.spec.gguf_alignment = 48;
    check_fixture_rejected(&f, "power of two", "non-power-of-two GGUF alignment rejected");

    valid_ledger_fixture(&f);
    f.spec.device_alignment = UINT64_MAX;
    check_fixture_rejected(&f, "alignment", "unsafe device alignment rejected");
}

static void test_ledger_shape_rejections(void) {
    ledger_fixture f;
    valid_ledger_fixture(&f);
    f.tensors[0].ndim = 0;
    check_fixture_rejected(&f, "dimension", "zero-rank static tensor rejected");

    valid_ledger_fixture(&f);
    f.tensors[0].ndim = DS4_LAGUNA_MAX_DIMS + 1u;
    check_fixture_rejected(&f, "dimension", "over-rank static tensor rejected");

    valid_ledger_fixture(&f);
    f.tensors[1].block_elems = 0;
    check_fixture_rejected(&f, "block", "zero block geometry rejected");

    valid_ledger_fixture(&f);
    f.tensors[1].dim[0] = 33;
    check_fixture_rejected(&f, "dim0", "routed dim0 must divide block geometry exactly");

    valid_ledger_fixture(&f);
    f.tensors[1].dim[1] = 0;
    check_fixture_rejected(&f, "zero dimension", "zero routed dimension rejected");

    valid_ledger_fixture(&f);
    f.tensors[1].dim[2] = 3;
    check_fixture_rejected(&f, "expert count", "routed dim2 must equal expert count");

    valid_ledger_fixture(&f);
    f.tensors[1].dim[1] = UINT64_MAX;
    check_fixture_rejected(&f, "overflow", "routed dimension multiplication overflow rejected");
}

static void test_ledger_role_rejections(void) {
    ledger_fixture f;
    valid_ledger_fixture(&f);
    f.tensors[1].routed_layer = 0;
    check_fixture_rejected(&f, "layer", "routed layer below first routed layer rejected");

    valid_ledger_fixture(&f);
    f.tensors[1].routed_layer = 3;
    check_fixture_rejected(&f, "layer", "routed layer at layer count rejected");

    valid_ledger_fixture(&f);
    f.tensors[1].routed_projection = (ds4_laguna_routed_projection)99;
    check_fixture_rejected(&f, "projection", "unknown routed projection rejected");

    valid_ledger_fixture(&f);
    f.tensors[1].routed_projection = DS4_LAGUNA_ROUTED_PROJECTION_UP;
    check_fixture_rejected(&f, "duplicate", "duplicate routed projection rejected");

    valid_ledger_fixture(&f);
    f.tensors[1].tensor_class = DS4_LAGUNA_TENSOR_STATIC;
    f.tensors[1].routed_layer = UINT32_MAX;
    f.tensors[1].routed_projection = DS4_LAGUNA_ROUTED_PROJECTION_NONE;
    check_fixture_rejected(&f, "missing", "missing routed projection rejected");

    valid_ledger_fixture(&f);
    f.tensors[1].dim[1] = 4;
    f.tensors[1].source_bytes = 144;
    check_fixture_rejected(&f, "projection size", "inconsistent projection expert size rejected");
}

static void test_ledger_boundary_rejections(void) {
    ledger_fixture f;
    valid_ledger_fixture(&f);
    f.spec.metadata_end = f.spec.header_end - 1;
    check_fixture_rejected(&f, "boundaries", "reversed structural boundaries rejected");

    valid_ledger_fixture(&f);
    f.spec.tensor_data_start = 96;
    check_fixture_rejected(&f, "alignment", "misaligned tensor data start rejected");

    valid_ledger_fixture(&f);
    f.spec.tensor_directory_end = f.spec.tensor_data_start + 1;
    check_fixture_rejected(&f, "boundaries", "directory crossing data start rejected");

    valid_ledger_fixture(&f);
    f.spec.tensor_data_start = 192;
    check_fixture_rejected(&f, "exact alignment", "noncanonical aligned data start rejected");

    valid_ledger_fixture(&f);
    f.spec.header_end = 0;
    check_fixture_rejected(&f, "header", "empty GGUF header rejected");

    valid_ledger_fixture(&f);
    f.spec.tensor_directory_end = f.spec.metadata_end;
    check_fixture_rejected(&f, "directory", "empty tensor directory rejected");

    valid_ledger_fixture(&f);
    f.spec.layer_count = UINT32_MAX;
    f.spec.expert_count = UINT32_MAX;
    check_fixture_rejected(&f, "allocation", "oversized expert table allocation rejected");
}

static void test_ledger(void) {
    test_ledger_valid();
    test_ledger_identity_and_class_rejections();
    test_ledger_source_rejections();
    test_ledger_shape_rejections();
    test_ledger_role_rejections();
    test_ledger_boundary_rejections();
}

static void usage(const char *argv0) {
    fprintf(stderr, "Usage: %s --case options|ledger\n", argv0);
}

int main(int argc, char **argv) {
    if (argc != 3 || strcmp(argv[1], "--case") != 0 ||
        (strcmp(argv[2], "options") != 0 &&
         strcmp(argv[2], "ledger") != 0)) {
        usage(argv[0]);
        return 2;
    }

    if (strcmp(argv[2], "options") == 0) test_options();
    else test_ledger();
    if (g_failed != 0) {
        fprintf(stderr,
                "test_laguna_stream: %d/%d assertion(s) failed\n",
                g_failed,
                g_total);
        return 1;
    }
    fprintf(stdout, "test_laguna_stream: PASS (%d assertions)\n", g_total);
    return 0;
}
