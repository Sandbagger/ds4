/* Pure-C policy tests for the Laguna compact runtime.
 *
 * Keep option parsing here independent of CUDA and model files so every host
 * can enforce the same canonical configuration contract. */

#include "ds4.h"

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

static void usage(const char *argv0) {
    fprintf(stderr, "Usage: %s --case options\n", argv0);
}

int main(int argc, char **argv) {
    if (argc != 3 || strcmp(argv[1], "--case") != 0 ||
        strcmp(argv[2], "options") != 0) {
        usage(argv[0]);
        return 2;
    }

    test_options();
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
