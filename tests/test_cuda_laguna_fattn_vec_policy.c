#include "ds4_cuda_laguna_fattn_vec_policy.h"

#include <stdint.h>
#include <stdio.h>

static ds4_cuda_laguna_fattn_vec_decode_shape canonical_shape(void) {
    return (ds4_cuda_laguna_fattn_vec_decode_shape) {
        .rollback = 0,
        .compute_major = 12,
        .compute_minor = 1,
        .sm_count = 48,
        .poolside_partitions = 3u,
        .query_tokens = 1u,
        .pos = 512u,
        .cache_cap = 4096u,
        .key_start = 0u,
        .key_count = 513u,
        .padded_key_count = 768u,
        .n_head = 48u,
        .n_head_kv = 8u,
        .head_dim = 128u,
    };
}

static ds4_cuda_laguna_fattn_vec_decode_shape canonical_swa_shape(void) {
    return (ds4_cuda_laguna_fattn_vec_decode_shape) {
        .rollback = 0,
        .compute_major = 12,
        .compute_minor = 1,
        .sm_count = 48,
        .poolside_partitions = 4u,
        .query_tokens = 1u,
        .pos = 512u,
        .cache_cap = 512u,
        .key_start = 1u,
        .key_count = 512u,
        .padded_key_count = 768u,
        .n_head = 72u,
        .n_head_kv = 8u,
        .head_dim = 128u,
    };
}

static int expect_partitions(
        const char *name,
        ds4_cuda_laguna_fattn_vec_decode_shape shape,
        uint32_t expected) {
    const uint32_t actual =
        ds4_cuda_laguna_fattn_vec_decode_partitions(&shape);
    if (actual != expected) {
        fprintf(stderr, "%s: got %u partitions, expected %u\n",
                name, actual, expected);
        return 1;
    }
    return 0;
}

int main(void) {
    ds4_cuda_laguna_fattn_vec_decode_shape shape = canonical_shape();
    if (expect_partitions("canonical", shape, 3u)) return 1;

#define EXPECT_REJECTED(field, value) do { \
        shape = canonical_shape(); \
        shape.field = (value); \
        if (expect_partitions(#field, shape, 0u)) return 1; \
    } while (0)
    EXPECT_REJECTED(rollback, 1);
    EXPECT_REJECTED(compute_major, 11);
    EXPECT_REJECTED(compute_minor, 0);
    EXPECT_REJECTED(sm_count, 47);
    EXPECT_REJECTED(poolside_partitions, 1u);
    EXPECT_REJECTED(poolside_partitions, 2u);
    EXPECT_REJECTED(poolside_partitions, 4u);
    EXPECT_REJECTED(query_tokens, 2u);
    EXPECT_REJECTED(pos, 511u);
    EXPECT_REJECTED(cache_cap, 767u);
    EXPECT_REJECTED(key_start, 1u);
    EXPECT_REJECTED(key_count, 512u);
    EXPECT_REJECTED(padded_key_count, 512u);
    EXPECT_REJECTED(n_head, 72u);
    EXPECT_REJECTED(n_head_kv, 4u);
    EXPECT_REJECTED(head_dim, 64u);
#undef EXPECT_REJECTED

    shape = canonical_swa_shape();
    if (expect_partitions("canonical-swa", shape, 4u)) return 1;

#define EXPECT_SWA_REJECTED(field, value) do { \
        shape = canonical_swa_shape(); \
        shape.field = (value); \
        if (expect_partitions("swa-" #field, shape, 0u)) return 1; \
    } while (0)
    EXPECT_SWA_REJECTED(rollback, 1);
    EXPECT_SWA_REJECTED(compute_major, 11);
    EXPECT_SWA_REJECTED(compute_minor, 0);
    EXPECT_SWA_REJECTED(sm_count, 47);
    EXPECT_SWA_REJECTED(poolside_partitions, 3u);
    EXPECT_SWA_REJECTED(poolside_partitions, 5u);
    EXPECT_SWA_REJECTED(query_tokens, 2u);
    EXPECT_SWA_REJECTED(pos, 511u);
    EXPECT_SWA_REJECTED(pos, 513u);
    EXPECT_SWA_REJECTED(cache_cap, 511u);
    EXPECT_SWA_REJECTED(cache_cap, 513u);
    EXPECT_SWA_REJECTED(cache_cap, 768u);
    EXPECT_SWA_REJECTED(key_start, 0u);
    EXPECT_SWA_REJECTED(key_start, 2u);
    EXPECT_SWA_REJECTED(key_count, 511u);
    EXPECT_SWA_REJECTED(key_count, 513u);
    EXPECT_SWA_REJECTED(padded_key_count, 512u);
    EXPECT_SWA_REJECTED(padded_key_count, 1024u);
    EXPECT_SWA_REJECTED(n_head, 48u);
    EXPECT_SWA_REJECTED(n_head, 71u);
    EXPECT_SWA_REJECTED(n_head_kv, 4u);
    EXPECT_SWA_REJECTED(head_dim, 64u);
#undef EXPECT_SWA_REJECTED

    puts("test_cuda_laguna_fattn_vec_policy PASS");
    return 0;
}
