/* Pure-C policy tests for the Laguna compact runtime.
 *
 * Keep option parsing here independent of CUDA and model files so every host
 * can enforce the same canonical configuration contract. */

#include "ds4.h"
#include "ds4_laguna_stream.h"
#include "ds4_runtime.h"

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

static ds4_engine_options reference_qualification_options(uint64_t cache_bytes) {
    ds4_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = "/dev/null";
    opt.qualification_plan_path = "/tmp/ds4-plan.json";
    opt.qualification_plan_path_set = true;
    opt.backend = DS4_BACKEND_CUDA;
    opt.context_size = 32768;
    opt.prefill_chunk = 4096u;
    opt.session_slots = 1u;
    opt.ssd_streaming = true;
    opt.ssd_streaming_cache_bytes = cache_bytes;
    opt.ssd_streaming_cache_bytes_set = true;
    return opt;
}

static void test_qualification_option_preflight(void) {
    const uint64_t gib = UINT64_C(1024) * 1024u * 1024u;
    char err[256];
    ds4_engine_options opt = reference_qualification_options(8u * gib);
    memset(err, 0, sizeof(err));
    CHECK(ds4_test_qualification_plan_preflight(
              &opt, false, err, sizeof(err)) == 0 && err[0] == '\0',
          "reference qualification options pass without hardware pricing");

    opt.ssd_streaming_cache_bytes = 10u * gib;
    CHECK(ds4_test_qualification_plan_preflight(
              &opt, false, err, sizeof(err)) == 2 &&
              strstr(err, "8, 12, or 16 GiB") != NULL,
          "qualification accepts only the frozen cache profiles");
    opt = reference_qualification_options(8u * gib);
    opt.backend = DS4_BACKEND_CPU;
    CHECK(ds4_test_qualification_plan_preflight(
              &opt, false, err, sizeof(err)) == 2,
          "qualification requires the declared CUDA backend");
    opt = reference_qualification_options(8u * gib);
    opt.context_size = 32767;
    CHECK(ds4_test_qualification_plan_preflight(
              &opt, false, err, sizeof(err)) == 2,
          "qualification requires exactly 32K context");
    opt = reference_qualification_options(8u * gib);
    opt.prefill_chunk = 8192u;
    CHECK(ds4_test_qualification_plan_preflight(
              &opt, false, err, sizeof(err)) == 2,
          "qualification requires exactly 4K prefill");
    opt = reference_qualification_options(8u * gib);
    opt.session_slots = 2u;
    CHECK(ds4_test_qualification_plan_preflight(
              &opt, false, err, sizeof(err)) == 2,
          "qualification prices exactly one live session");
    opt = reference_qualification_options(8u * gib);
    opt.warm_weights = true;
    CHECK(ds4_test_qualification_plan_preflight(
              &opt, false, err, sizeof(err)) == 2 &&
              strstr(err, "warm") != NULL,
          "qualification forbids a whole-file warm scan");
    opt = reference_qualification_options(8u * gib);
    opt.simulate_used_memory_bytes = gib;
    CHECK(ds4_test_qualification_plan_preflight(
              &opt, false, err, sizeof(err)) == 2,
          "qualification forbids simulated external pressure");
    opt = reference_qualification_options(8u * gib);
    opt.ssd_streaming_cache_experts_set = true;
    CHECK(ds4_test_qualification_plan_preflight(
              &opt, false, err, sizeof(err)) == 2,
          "qualification rejects the legacy cache alias");
    opt = reference_qualification_options(8u * gib);
    opt.mtp_path = "/dev/null";
    CHECK(ds4_test_qualification_plan_preflight(
              &opt, false, err, sizeof(err)) == 2,
          "qualification excludes support-model allocations");
    opt = reference_qualification_options(8u * gib);
    CHECK(ds4_test_qualification_plan_preflight(
              &opt, true, err, sizeof(err)) == 2 &&
              strstr(err, "GPU layout") != NULL,
          "qualification rejects GPU-layout probing before hardware access");
}

static void test_options(void) {
    test_qualification_option_preflight();
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

static uint64_t sum_u64(const uint64_t *values, size_t count) {
    uint64_t sum = 0;
    for (size_t i = 0; i < count; i++) sum += values[i];
    return sum;
}

static void test_allocation_profiles(void) {
    const uint64_t gib = UINT64_C(1024) * 1024u * 1024u;
    const uint64_t cache_bytes[] = { 8u * gib, 12u * gib, 16u * gib };
    const uint32_t slot_counts[] = { 1618u, 2427u, 3236u };
    const uint64_t payload_bytes[] = {
        UINT64_C(8589017088),
        UINT64_C(12883525632),
        UINT64_C(17178034176),
    };
    const uint64_t tail_bytes[] = {
        UINT64_C(917504),
        UINT64_C(1376256),
        UINT64_C(1835008),
    };
    const uint64_t metadata_bytes[] = {
        UINT64_C(1663624),
        UINT64_C(1689512),
        UINT64_C(1715400),
    };
    const uint64_t qualification_non_cache[] = {
        UINT64_C(14062675600),
        UINT64_C(14062701488),
        UINT64_C(14062727376),
    };
    const uint64_t profile_bounds[] = {
        UINT64_C(25769803776),
        UINT64_C(30064771072),
        UINT64_C(34359738368),
    };
    ds4_laguna_ledger ledger;
    memset(&ledger, 0, sizeof(ledger));
    ledger.file_size = UINT64_C(68248759648);
    ledger.tensor_count = UINT64_C(814);
    ledger.static_parent_count = UINT64_C(673);
    ledger.routed_parent_count = UINT64_C(141);
    ledger.static_aligned_device_bytes = UINT64_C(4374164480);
    ledger.expert_entry_count = UINT64_C(12032);
    ledger.slot_stride_bytes = UINT64_C(5308416);

    for (size_t i = 0; i < sizeof(cache_bytes) / sizeof(cache_bytes[0]); i++) {
        ds4_laguna_allocation_plan_spec spec;
        memset(&spec, 0, sizeof(spec));
        spec.configured_cache_bytes = cache_bytes[i];
        spec.context_tokens = 32768u;
        spec.prefill_rows = 4096u;
        spec.session_count = 1u;
        ds4_laguna_allocation_plan plan;
        char err[256];
        memset(&plan, 0, sizeof(plan));
        memset(err, 0, sizeof(err));
        CHECK(ds4_laguna_allocation_plan_make(
                  &plan, &ledger, &spec, err, sizeof(err)),
              "reference allocation profile builds");
        CHECK(err[0] == '\0', "successful allocation plan leaves no error");
        CHECK(plan.configured_cache_bytes == cache_bytes[i] &&
                  plan.effective_cache_limit_bytes == cache_bytes[i],
              "effective cache limit preserves the configured byte ceiling");
        CHECK(plan.context_tokens == 32768u &&
                  plan.prefill_rows == 4096u &&
                  plan.session_count == 1u,
              "allocation plan retains the exact reference runtime shape");
        CHECK(plan.profile_id != NULL &&
                  ((i == 0 && strcmp(plan.profile_id, "cache-8gib") == 0) ||
                   (i == 1 && strcmp(plan.profile_id, "cache-12gib") == 0) ||
                   (i == 2 && strcmp(plan.profile_id, "cache-16gib") == 0)),
              "allocation plan assigns the frozen profile identity");
        CHECK(plan.slot_stride_bytes == UINT64_C(5308416) &&
                  plan.slot_count == slot_counts[i],
              "reference profile floors exact cache slots from ledger stride");
        CHECK(plan.cache_payload_bytes == payload_bytes[i] &&
                  plan.cache_tail_uncharged_bytes == tail_bytes[i] &&
                  plan.cache_payload_bytes +
                      plan.cache_tail_uncharged_bytes == cache_bytes[i],
              "physical cache payload charges slot padding and leaves only tail slack");
        CHECK(plan.owned_category_bounds[
                  DS4_RUNTIME_CATEGORY_STATIC_WEIGHTS] ==
                      UINT64_C(4374164480),
              "static category uses the exact aligned tensor ledger bytes");
        CHECK(plan.owned_category_bounds[
                  DS4_RUNTIME_CATEGORY_CACHE_METADATA_ADDRESS_TABLES] ==
                      metadata_bytes[i],
              "cache metadata freezes exact ledger, table, and slot-state arithmetic");
        CHECK(plan.owned_category_bounds[DS4_RUNTIME_CATEGORY_KV_STATE] ==
                  UINT64_C(1686110208),
              "32K one-session KV allocation is exact");
        CHECK(plan.owned_category_bounds[DS4_RUNTIME_CATEGORY_GRAPH_SCRATCH] ==
                  UINT64_C(1537052680),
              "4K Laguna graph and scratch allocation is exact");
        CHECK(plan.owned_category_bounds[DS4_RUNTIME_CATEGORY_PINNED_STAGING] ==
                  UINT64_C(21233664) &&
                  plan.staging_buffer_count == 4u &&
                  plan.staging_buffer_bytes == UINT64_C(5308416),
              "four fixed pinned staging buffers are charged exactly");
        CHECK(plan.owned_category_bounds[DS4_RUNTIME_CATEGORY_OTHER_HOST] ==
                  gib &&
                  plan.owned_category_bounds[DS4_RUNTIME_CATEGORY_OTHER_CUDA] ==
                      2u * gib,
              "provisional named other-host and other-CUDA envelopes are explicit");
        CHECK(plan.report_bounds[DS4_RUNTIME_REPORT_MODEL_MAPPED_VIRTUAL] ==
                  ledger.file_size &&
                  plan.report_bounds[DS4_RUNTIME_REPORT_MODEL_MAPPING_REGISTERED] ==
                      0 &&
                  plan.report_bounds[DS4_RUNTIME_REPORT_MODEL_SOURCE_RESIDENT] ==
                      2u * gib &&
                  plan.report_bounds[DS4_RUNTIME_REPORT_HOST_LIBRARY_UNATTRIBUTED] ==
                      gib / 2u &&
                  plan.report_bounds[DS4_RUNTIME_REPORT_CUDA_LIBRARY_UNATTRIBUTED] ==
                      gib / 2u,
              "report-only model and unattributed ceilings are distinct from owned bytes");
        CHECK(plan.qualification_non_cache_bound_bytes ==
                  qualification_non_cache[i] &&
                  plan.qualification_non_cache_bound_bytes <= 16u * gib,
              "owned non-cache plus all external charges stays inside 16GiB");
        CHECK(plan.qualification_total_bound_bytes == profile_bounds[i] &&
                  plan.planned_qualification_bytes <= profile_bounds[i],
              "reference profile fits its configured-cache-plus-16GiB ceiling");
        CHECK(sum_u64(plan.owned_category_bounds,
                      DS4_RUNTIME_OWNED_CATEGORY_COUNT) ==
                  plan.owned_total_bound_bytes,
              "owned category bounds reconcile byte-exactly");
        uint64_t callsite_sum[DS4_RUNTIME_OWNED_CATEGORY_COUNT] = {0};
        size_t other_host_callsites = 0;
        size_t other_cuda_callsites = 0;
        bool static_offsets_are_host = false;
        CHECK(plan.callsite_count == DS4_LAGUNA_ALLOCATION_CALLSITE_COUNT,
              "allocation plan exposes the complete stable callsite registry");
        for (size_t j = 0; j < plan.callsite_count; j++) {
            const ds4_runtime_callsite *site = &plan.callsites[j];
            CHECK(site->id != 0 && site->name && site->name[0] != '\0',
                  "every planned allocation callsite has stable identity");
            callsite_sum[site->category] += site->bound_bytes;
            if (site->category == DS4_RUNTIME_CATEGORY_OTHER_HOST) {
                other_host_callsites++;
            }
            if (site->category == DS4_RUNTIME_CATEGORY_OTHER_CUDA) {
                other_cuda_callsites++;
            }
            if (site->id == DS4_LAGUNA_CALLSITE_STATIC_OFFSETS &&
                site->domain == DS4_RUNTIME_DOMAIN_HOST) {
                static_offsets_are_host = true;
            }
        }
        CHECK(other_host_callsites == 7u && other_cuda_callsites == 4u,
              "other envelopes decompose into explicit stable allocation callsites");
        CHECK(static_offsets_are_host,
              "static offset metadata is attributed to its host physical domain");
        for (size_t j = 0; j < DS4_RUNTIME_OWNED_CATEGORY_COUNT; j++) {
            CHECK(callsite_sum[j] == plan.owned_category_bounds[j],
                  "callsite bounds reconcile exactly to their owned category");
        }
    }

    ds4_laguna_allocation_plan_spec invalid = {
        8u * gib, 32768u, 4096u, 1u,
    };
    ds4_laguna_allocation_plan plan;
    char err[256];
    ledger.slot_stride_bytes = 8u * gib + 1u;
    CHECK(!ds4_laguna_allocation_plan_make(
              &plan, &ledger, &invalid, err, sizeof(err)) &&
              strstr(err, "slot") != NULL,
          "cache smaller than one complete slot is rejected");
    ledger.slot_stride_bytes = UINT64_C(5308416);

    invalid.context_tokens = 32767u;
    CHECK(!ds4_laguna_allocation_plan_make(
              &plan, &ledger, &invalid, err, sizeof(err)),
          "non-reference context is rejected");
    invalid.context_tokens = 32768u;
    invalid.prefill_rows = 8192u;
    CHECK(!ds4_laguna_allocation_plan_make(
              &plan, &ledger, &invalid, err, sizeof(err)),
          "non-reference prefill shape is rejected");
    invalid.prefill_rows = 4096u;
    invalid.session_count = 2u;
    CHECK(!ds4_laguna_allocation_plan_make(
              &plan, &ledger, &invalid, err, sizeof(err)),
          "non-reference session count is rejected");
    invalid.session_count = 1u;
    invalid.configured_cache_bytes = 10u * gib;
    CHECK(!ds4_laguna_allocation_plan_make(
              &plan, &ledger, &invalid, err, sizeof(err)),
          "unqualified cache profile is rejected");
    invalid.configured_cache_bytes = 8u * gib;
    ledger.static_aligned_device_bytes = UINT64_MAX;
    CHECK(!ds4_laguna_allocation_plan_make(
              &plan, &ledger, &invalid, err, sizeof(err)) &&
              strstr(err, "ledger") != NULL,
          "non-reference or overflowing ledger input fails closed");
}

typedef struct {
    ds4_runtime_callsite callsites[9];
    ds4_runtime_allocation_record records[32];
    ds4_runtime_tracker_config config;
    ds4_runtime_tracker tracker;
} tracker_fixture;

static void tracker_fixture_prepare(tracker_fixture *f) {
    memset(f, 0, sizeof(*f));
    const ds4_runtime_callsite sites[] = {
        { 1u, "static", DS4_RUNTIME_CATEGORY_STATIC_WEIGHTS,
          DS4_RUNTIME_DOMAIN_CUDA_DEVICE, 100u },
        { 2u, "cache", DS4_RUNTIME_CATEGORY_EXPERT_CACHE_PAYLOAD,
          DS4_RUNTIME_DOMAIN_CUDA_DEVICE, 200u },
        { 3u, "metadata-host", DS4_RUNTIME_CATEGORY_CACHE_METADATA_ADDRESS_TABLES,
          DS4_RUNTIME_DOMAIN_HOST, 40u },
        { 4u, "metadata-device", DS4_RUNTIME_CATEGORY_CACHE_METADATA_ADDRESS_TABLES,
          DS4_RUNTIME_DOMAIN_CUDA_DEVICE, 60u },
        { 5u, "kv", DS4_RUNTIME_CATEGORY_KV_STATE,
          DS4_RUNTIME_DOMAIN_CUDA_DEVICE, 60u },
        { 6u, "graph", DS4_RUNTIME_CATEGORY_GRAPH_SCRATCH,
          DS4_RUNTIME_DOMAIN_CUDA_DEVICE, 70u },
        { 7u, "staging", DS4_RUNTIME_CATEGORY_PINNED_STAGING,
          DS4_RUNTIME_DOMAIN_HOST, 80u },
        { 8u, "other-host", DS4_RUNTIME_CATEGORY_OTHER_HOST,
          DS4_RUNTIME_DOMAIN_HOST, 50u },
        { 9u, "managed", DS4_RUNTIME_CATEGORY_OTHER_CUDA,
          DS4_RUNTIME_DOMAIN_CUDA_MANAGED, 90u },
    };
    memcpy(f->callsites, sites, sizeof(sites));
    f->config.callsites = f->callsites;
    f->config.callsite_count = sizeof(f->callsites) / sizeof(f->callsites[0]);
    f->config.records = f->records;
    f->config.record_capacity = sizeof(f->records) / sizeof(f->records[0]);
    const uint64_t category_bounds[] = {
        100u, 200u, 100u, 60u, 70u, 80u, 50u, 90u,
    };
    memcpy(f->config.category_bounds, category_bounds,
           sizeof(category_bounds));
    f->config.report_bounds[DS4_RUNTIME_REPORT_MODEL_MAPPED_VIRTUAL] = 1000u;
    f->config.report_bounds[DS4_RUNTIME_REPORT_MODEL_MAPPING_REGISTERED] = 1000u;
    f->config.report_bounds[DS4_RUNTIME_REPORT_MODEL_SOURCE_RESIDENT] = 100u;
    f->config.report_bounds[DS4_RUNTIME_REPORT_HOST_LIBRARY_UNATTRIBUTED] = 100u;
    f->config.report_bounds[DS4_RUNTIME_REPORT_CUDA_LIBRARY_UNATTRIBUTED] = 100u;
    f->config.owned_total_bound_bytes = 750u;
    f->config.qualification_total_bound_bytes = 1050u;
}

static bool tracker_fixture_init(tracker_fixture *f) {
    tracker_fixture_prepare(f);
    return ds4_runtime_tracker_init(&f->tracker, &f->config) ==
           DS4_RUNTIME_STATUS_OK;
}

static void test_tracker_reconciliation_and_peaks(void) {
    tracker_fixture f;
    CHECK(tracker_fixture_init(&f), "valid predeclared tracker registry initializes");
    CHECK(ds4_runtime_tracker_allocate(&f.tracker, 101u, 1u,
                                       UINT64_C(0x1000), 100u, 100u) ==
              DS4_RUNTIME_STATUS_OK,
          "tracked static allocation succeeds");
    CHECK(ds4_runtime_tracker_allocate(&f.tracker, 102u, 8u,
                                       UINT64_C(0x1000), 50u, 50u) ==
              DS4_RUNTIME_STATUS_OK,
          "equal numeric address in a distinct physical domain is allowed");
    CHECK(f.tracker.category_current[DS4_RUNTIME_CATEGORY_STATIC_WEIGHTS] ==
              100u &&
              f.tracker.category_current[DS4_RUNTIME_CATEGORY_OTHER_HOST] ==
                  50u &&
              f.tracker.owned_total_current == 150u &&
              f.tracker.owned_total_peak == 150u,
          "owned current and simultaneous peak reconcile after every event");
    CHECK(ds4_runtime_tracker_release(&f.tracker, 101u) ==
              DS4_RUNTIME_STATUS_OK &&
              ds4_runtime_tracker_allocate(&f.tracker, 103u, 2u,
                                            UINT64_C(0x2000), 200u, 200u) ==
                  DS4_RUNTIME_STATUS_OK,
          "free followed by a disjoint category allocation succeeds");
    CHECK(f.tracker.category_peak[DS4_RUNTIME_CATEGORY_STATIC_WEIGHTS] == 100u &&
              f.tracker.category_peak[
                  DS4_RUNTIME_CATEGORY_EXPERT_CACHE_PAYLOAD] == 200u &&
              f.tracker.owned_total_peak == 250u,
          "owned peak is simultaneous rather than sum of category peaks");

    CHECK(ds4_runtime_tracker_map_model(&f.tracker, 200u,
                                        UINT64_C(0x5000), 1000u) ==
              DS4_RUNTIME_STATUS_OK,
          "model mapping is recorded without a physical charge");
    CHECK(f.tracker.report_current[
              DS4_RUNTIME_REPORT_MODEL_MAPPED_VIRTUAL] == 1000u &&
              f.tracker.owned_total_current == 250u,
          "mapped virtual bytes are report-only");
    CHECK(ds4_runtime_tracker_report_set(
              &f.tracker, DS4_RUNTIME_REPORT_MODEL_SOURCE_RESIDENT, 80u) ==
              DS4_RUNTIME_STATUS_OK &&
              ds4_runtime_tracker_report_set(
                  &f.tracker, DS4_RUNTIME_REPORT_HOST_LIBRARY_UNATTRIBUTED,
                  30u) == DS4_RUNTIME_STATUS_OK &&
              ds4_runtime_tracker_report_set(
                  &f.tracker, DS4_RUNTIME_REPORT_CUDA_LIBRARY_UNATTRIBUTED,
                  20u) == DS4_RUNTIME_STATUS_OK,
          "external physical observations update through bounded reports");
    CHECK(f.tracker.qualification_total_current == 380u &&
              f.tracker.qualification_total_peak == 380u,
          "qualification total adds source and unattributed bytes exactly once");

    ds4_runtime_snapshot snapshot;
    ds4_runtime_allocation_record active_records[8];
    CHECK(ds4_runtime_tracker_snapshot_copy(
              &f.tracker, &snapshot, active_records,
              sizeof(active_records) / sizeof(active_records[0])) &&
              snapshot.active_record_count == 3u &&
              snapshot.owned_total_current == 250u &&
              snapshot.qualification_total_current == 380u,
          "snapshot copies simultaneous scalar state and active records");
    CHECK(active_records[0].id == 102u && active_records[0].live &&
              active_records[0].base == UINT64_C(0x1000) &&
              active_records[0].requested_bytes == 50u &&
              active_records[0].charged_bytes == 50u &&
              active_records[0].category == DS4_RUNTIME_CATEGORY_OTHER_HOST &&
              active_records[0].domain == DS4_RUNTIME_DOMAIN_HOST &&
              active_records[0].callsite_id == 8u &&
              active_records[0].relation ==
                  DS4_RUNTIME_RELATION_OWNED_ALLOCATION,
          "snapshot attribution records preserve every required field");
    active_records[0].live = false;
    CHECK(f.tracker.records[1].live,
          "snapshot records are copies rather than tracker-owned pointers");
    CHECK(!ds4_runtime_tracker_snapshot_copy(
              &f.tracker, &snapshot, active_records, 2u) &&
              snapshot.active_record_count == 3u,
          "snapshot reports required active capacity without reallocating");

    CHECK(ds4_runtime_tracker_register(&f.tracker, 201u,
                                        UINT64_C(0x5000), 1000u, 200u) ==
              DS4_RUNTIME_STATUS_OK,
          "exact model mapping registration is accepted as a zero-charge relation");
    CHECK(f.tracker.report_current[
              DS4_RUNTIME_REPORT_MODEL_MAPPING_REGISTERED] == 1000u &&
              f.tracker.owned_total_current == 250u &&
              f.tracker.qualification_total_current == 380u,
          "registration metadata never double-charges mapped pages");
    CHECK(ds4_runtime_tracker_unmap_model(&f.tracker, 200u) ==
              DS4_RUNTIME_STATUS_UNSAFE &&
              f.tracker.violation == DS4_RUNTIME_VIOLATION_LIVE_RELATION,
          "model mapping cannot be removed while registration is live");
    CHECK(ds4_runtime_tracker_unregister(&f.tracker, 201u) ==
              DS4_RUNTIME_STATUS_UNSAFE &&
              ds4_runtime_tracker_unmap_model(&f.tracker, 200u) ==
                  DS4_RUNTIME_STATUS_UNSAFE &&
              f.tracker.violation == DS4_RUNTIME_VIOLATION_LIVE_RELATION,
          "cleanup events do not clear a latched unsafe status");
}

static void test_tracker_relations(void) {
    tracker_fixture f;
    CHECK(tracker_fixture_init(&f), "relation tracker initializes");
    CHECK(ds4_runtime_tracker_allocate(&f.tracker, 1u, 8u,
                                       UINT64_C(0x10000), 50u, 50u) ==
              DS4_RUNTIME_STATUS_OK,
          "host owner allocation succeeds");
    CHECK(ds4_runtime_tracker_register(&f.tracker, 2u,
                                        UINT64_C(0x10010), 20u, 1u) ==
              DS4_RUNTIME_STATUS_OK,
          "registration contained in exactly one host allocation succeeds");
    CHECK(ds4_runtime_tracker_release(&f.tracker, 1u) ==
              DS4_RUNTIME_STATUS_UNSAFE,
          "host allocation cannot be freed with live registration");

    tracker_fixture_prepare(&f);
    CHECK(ds4_runtime_tracker_init(&f.tracker, &f.config) ==
              DS4_RUNTIME_STATUS_OK &&
              ds4_runtime_tracker_allocate(&f.tracker, 3u, 8u,
                                            UINT64_C(0x18000), 50u, 50u) ==
                  DS4_RUNTIME_STATUS_OK &&
              ds4_runtime_tracker_register(&f.tracker, 4u,
                                             UINT64_C(0x18008), 24u, 3u) ==
                  DS4_RUNTIME_STATUS_OK,
          "first contained host registration succeeds");
    CHECK(ds4_runtime_tracker_register(&f.tracker, 5u,
                                        UINT64_C(0x18010), 16u, 3u) ==
              DS4_RUNTIME_STATUS_UNSAFE &&
              f.tracker.violation == DS4_RUNTIME_VIOLATION_OVERLAP,
          "overlapping registration identity is rejected without a second charge");

    tracker_fixture_prepare(&f);
    CHECK(ds4_runtime_tracker_init(&f.tracker, &f.config) ==
              DS4_RUNTIME_STATUS_OK,
          "fresh tracker for managed relation initializes");
    CHECK(ds4_runtime_tracker_allocate(&f.tracker, 10u, 9u,
                                       UINT64_C(0x30000), 90u, 90u) ==
              DS4_RUNTIME_STATUS_OK &&
              ds4_runtime_tracker_managed_host_relation(
                  &f.tracker, 11u, UINT64_C(0x30000), 90u, 10u) ==
                  DS4_RUNTIME_STATUS_OK,
          "managed allocation gains an exact zero-charge host-visible relation");
    CHECK(f.tracker.category_current[DS4_RUNTIME_CATEGORY_OTHER_CUDA] == 90u &&
              f.tracker.owned_total_current == 90u,
          "managed requested bytes are charged once to other CUDA");
    CHECK(ds4_runtime_tracker_unregister(&f.tracker, 11u) ==
              DS4_RUNTIME_STATUS_OK &&
              ds4_runtime_tracker_release(&f.tracker, 10u) ==
                  DS4_RUNTIME_STATUS_OK,
          "managed host relation is removed before its owner");
    CHECK(ds4_runtime_tracker_unregister(&f.tracker, 11u) ==
              DS4_RUNTIME_STATUS_UNSAFE &&
              f.tracker.violation == DS4_RUNTIME_VIOLATION_NOT_LIVE,
          "double unregister latches unsafe");

    tracker_fixture_prepare(&f);
    CHECK(ds4_runtime_tracker_init(&f.tracker, &f.config) ==
              DS4_RUNTIME_STATUS_OK &&
              ds4_runtime_tracker_allocate(&f.tracker, 20u, 8u,
                                            UINT64_C(0x40000), 50u, 50u) ==
                  DS4_RUNTIME_STATUS_OK &&
              ds4_runtime_tracker_release(&f.tracker, 20u) ==
                  DS4_RUNTIME_STATUS_OK,
          "fresh owner can be allocated and freed exactly once");
    CHECK(ds4_runtime_tracker_release(&f.tracker, 20u) ==
              DS4_RUNTIME_STATUS_UNSAFE,
          "double free latches unsafe");
}

static void test_tracker_rejections(void) {
    tracker_fixture f;
    CHECK(tracker_fixture_init(&f), "rejection tracker initializes");
    CHECK(ds4_runtime_tracker_allocate(&f.tracker, 1u, 999u,
                                       UINT64_C(0x1000), 1u, 1u) ==
              DS4_RUNTIME_STATUS_UNSAFE &&
              f.tracker.violation == DS4_RUNTIME_VIOLATION_UNKNOWN_CALLSITE,
          "unknown allocation callsite latches unsafe");
    const ds4_runtime_violation first = f.tracker.violation;
    CHECK(ds4_runtime_tracker_report_set(
              &f.tracker, DS4_RUNTIME_REPORT_MODEL_SOURCE_RESIDENT, 1u) ==
              DS4_RUNTIME_STATUS_UNSAFE &&
              f.tracker.violation == first,
          "first violation remains permanently latched");

    tracker_fixture_prepare(&f);
    f.callsites[1].id = f.callsites[0].id;
    CHECK(ds4_runtime_tracker_init(&f.tracker, &f.config) ==
              DS4_RUNTIME_STATUS_UNSAFE &&
              f.tracker.violation == DS4_RUNTIME_VIOLATION_DUPLICATE_CALLSITE,
          "duplicate predeclared callsite is rejected");
    tracker_fixture_prepare(&f);
    f.callsites[0].category = (ds4_runtime_category)99;
    CHECK(ds4_runtime_tracker_init(&f.tracker, &f.config) ==
              DS4_RUNTIME_STATUS_UNSAFE &&
              f.tracker.violation == DS4_RUNTIME_VIOLATION_UNCLASSIFIED_CALLSITE,
          "unclassified callsite is rejected");

    tracker_fixture_prepare(&f);
    f.config.qualification_total_bound_bytes = 1049u;
    CHECK(ds4_runtime_tracker_init(&f.tracker, &f.config) ==
              DS4_RUNTIME_STATUS_UNSAFE &&
              f.tracker.violation ==
                  DS4_RUNTIME_VIOLATION_QUALIFICATION_TOTAL_BOUND,
          "plan whose owned and external maxima exceed total is rejected");
    tracker_fixture_prepare(&f);
    f.config.category_bounds[DS4_RUNTIME_CATEGORY_OTHER_CUDA] = UINT64_MAX;
    f.callsites[8].bound_bytes = UINT64_MAX;
    f.config.owned_total_bound_bytes = UINT64_MAX;
    f.config.qualification_total_bound_bytes = UINT64_MAX;
    CHECK(ds4_runtime_tracker_init(&f.tracker, &f.config) ==
              DS4_RUNTIME_STATUS_UNSAFE &&
              f.tracker.violation == DS4_RUNTIME_VIOLATION_OVERFLOW,
          "uint64 plan reconciliation overflow fails closed");
    tracker_fixture_prepare(&f);
    f.config.record_capacity =
        SIZE_MAX / sizeof(ds4_runtime_allocation_record) + 1u;
    CHECK(ds4_runtime_tracker_init(&f.tracker, &f.config) ==
              DS4_RUNTIME_STATUS_UNSAFE &&
              f.tracker.violation == DS4_RUNTIME_VIOLATION_OVERFLOW,
          "record storage size overflow is rejected before initialization");

    tracker_fixture_prepare(&f);
    CHECK(ds4_runtime_tracker_init(&f.tracker, &f.config) ==
              DS4_RUNTIME_STATUS_OK &&
              ds4_runtime_tracker_allocate(&f.tracker, 30u, 1u,
                                            UINT64_C(0x1000), 80u, 80u) ==
                  DS4_RUNTIME_STATUS_OK,
          "overlap fixture first allocation succeeds");
    CHECK(ds4_runtime_tracker_allocate(&f.tracker, 31u, 2u,
                                       UINT64_C(0x1040), 20u, 20u) ==
              DS4_RUNTIME_STATUS_UNSAFE &&
              f.tracker.violation == DS4_RUNTIME_VIOLATION_OVERLAP,
          "overlapping charged ranges in one physical domain are rejected");

    tracker_fixture_prepare(&f);
    CHECK(ds4_runtime_tracker_init(&f.tracker, &f.config) ==
              DS4_RUNTIME_STATUS_OK,
          "category-overrun tracker initializes");
    CHECK(ds4_runtime_tracker_allocate(&f.tracker, 40u, 1u,
                                       UINT64_C(0x2000), 101u, 101u) ==
              DS4_RUNTIME_STATUS_UNSAFE &&
              (f.tracker.violation == DS4_RUNTIME_VIOLATION_CALLSITE_BOUND ||
               f.tracker.violation == DS4_RUNTIME_VIOLATION_CATEGORY_BOUND),
          "category or callsite bound crossing latches unsafe");

    tracker_fixture_prepare(&f);
    CHECK(ds4_runtime_tracker_init(&f.tracker, &f.config) ==
              DS4_RUNTIME_STATUS_OK,
          "undercharge tracker initializes");
    CHECK(ds4_runtime_tracker_allocate(&f.tracker, 41u, 8u,
                                       UINT64_C(0x3000), 50u, 49u) ==
              DS4_RUNTIME_STATUS_UNSAFE &&
              f.tracker.violation == DS4_RUNTIME_VIOLATION_UNDERCHARGE,
          "charged bytes may never be smaller than requested bytes");
}

static void test_reduction_floor(void) {
    const uint64_t gib = UINT64_C(1024) * 1024u * 1024u;
    uint64_t reduction = 0;
    CHECK(ds4_runtime_reduction_qualified(80u * gib, 44u * gib,
                                          &reduction) &&
              reduction == 36u * gib,
          "exact 45-percent reduction above 32GiB passes");
    CHECK(!ds4_runtime_reduction_qualified(80u * gib, 44u * gib + 1u,
                                           &reduction),
          "one byte below the exact 45-percent ratio fails");
    CHECK(ds4_runtime_reduction_qualified(64u * gib, 32u * gib,
                                          &reduction) &&
              reduction == 32u * gib,
          "exact 32GiB reduction at a sufficient ratio passes");
    CHECK(!ds4_runtime_reduction_qualified(80u * gib, 48u * gib,
                                           &reduction),
          "32GiB reduction below 45 percent fails");
    CHECK(!ds4_runtime_reduction_qualified(31u * gib, 0, &reduction),
          "ratio cannot compensate for a sub-32GiB reduction");
    CHECK(!ds4_runtime_reduction_qualified(0, 0, &reduction) &&
              !ds4_runtime_reduction_qualified(1u, 2u, &reduction),
          "zero resident or inverted peaks fail safely");
    CHECK(ds4_runtime_reduction_qualified(UINT64_MAX, 0, &reduction) &&
              reduction == UINT64_MAX,
          "ratio comparison remains exact at uint64 boundary");
}

static void test_inward_page_union(void) {
    const ds4_laguna_page_range input[] = {
        { 1u, 8192u },
        { 4096u, 8192u },
        { 12288u, 4096u },
        { 0u, 2048u },
        { 2048u, 2048u },
    };
    ds4_laguna_page_range output[5];
    size_t output_count = 0;
    uint64_t output_bytes = 0;
    CHECK(ds4_laguna_full_page_union(input,
                                      sizeof(input) / sizeof(input[0]),
                                      4096u, output,
                                      sizeof(output) / sizeof(output[0]),
                                      &output_count, &output_bytes),
          "safe full-page union succeeds");
    CHECK(output_count == 1u && output[0].offset == 4096u &&
              output[0].bytes == 12288u && output_bytes == 12288u,
          "inward-rounded ranges sort, deduplicate, and merge adjacent pages");

    const ds4_laguna_page_range split_page[] = {
        { 0u, 2048u }, { 2048u, 2048u },
    };
    CHECK(ds4_laguna_full_page_union(split_page, 2u, 4096u,
                                      output, 5u, &output_count,
                                      &output_bytes) &&
              output_count == 0 && output_bytes == 0,
          "raw ranges are never merged before each is rounded inward");
    const ds4_laguna_page_range exact_page[] = {
        { 4096u, 4096u },
    };
    CHECK(ds4_laguna_full_page_union(exact_page, 1u, 4096u,
                                      output, 5u, &output_count,
                                      &output_bytes) &&
              output_count == 1u && output[0].offset == 4096u &&
              output[0].bytes == 4096u,
          "an exact full page remains eligible");
    const ds4_laguna_page_range overflow[] = {
        { UINT64_MAX - 1u, 4u },
    };
    CHECK(!ds4_laguna_full_page_union(overflow, 1u, 4096u,
                                       output, 5u, &output_count,
                                       &output_bytes) &&
              output_count == 0 && output_bytes == 0,
          "overflowing source interval fails closed and clears outputs");
    CHECK(!ds4_laguna_full_page_union(exact_page, 1u, 0,
                                       output, 5u, &output_count,
                                       &output_bytes),
          "zero page size is rejected");
    CHECK(!ds4_laguna_full_page_union(exact_page, 1u, 3000u,
                                       output, 5u, &output_count,
                                       &output_bytes),
          "non-power-of-two page size is rejected");
    CHECK(!ds4_laguna_full_page_union(input, 3u, 4096u,
                                       output, 1u, &output_count,
                                       &output_bytes),
          "insufficient fixed output capacity fails without allocation");
}

static void test_allocation(void) {
    test_allocation_profiles();
    test_tracker_reconciliation_and_peaks();
    test_tracker_relations();
    test_tracker_rejections();
    test_reduction_floor();
    test_inward_page_union();
}

static void usage(const char *argv0) {
    fprintf(stderr, "Usage: %s --case options|ledger|allocation\n", argv0);
}

int main(int argc, char **argv) {
    if (argc != 3 || strcmp(argv[1], "--case") != 0 ||
        (strcmp(argv[2], "options") != 0 &&
         strcmp(argv[2], "ledger") != 0 &&
         strcmp(argv[2], "allocation") != 0)) {
        usage(argv[0]);
        return 2;
    }

    if (strcmp(argv[2], "options") == 0) test_options();
    else if (strcmp(argv[2], "ledger") == 0) test_ledger();
    else test_allocation();
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
