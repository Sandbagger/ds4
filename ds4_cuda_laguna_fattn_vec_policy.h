#ifndef DS4_CUDA_LAGUNA_FATTN_VEC_POLICY_H
#define DS4_CUDA_LAGUNA_FATTN_VEC_POLICY_H

#include <stdint.h>

/* Pure dispatch policy for the first Poolside-compatible Laguna decode
 * slice.  Keep the complete geometry here so a near-miss cannot silently
 * enter a kernel whose arithmetic is only qualified for token 513. */
typedef struct {
    int rollback;
    int compute_major;
    int compute_minor;
    int sm_count;
    uint32_t poolside_partitions;
    uint32_t query_tokens;
    uint32_t pos;
    uint32_t cache_cap;
    uint32_t key_start;
    uint32_t key_count;
    uint32_t padded_key_count;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t head_dim;
} ds4_cuda_laguna_fattn_vec_decode_shape;

static inline uint32_t ds4_cuda_laguna_fattn_vec_decode_partitions(
        const ds4_cuda_laguna_fattn_vec_decode_shape *shape) {
    if (!shape || shape->rollback ||
        shape->compute_major != 12 || shape->compute_minor != 1 ||
        shape->sm_count != 48 || shape->poolside_partitions != 3u ||
        shape->query_tokens != 1u || shape->pos != 512u ||
        shape->cache_cap < 768u || shape->key_start != 0u ||
        shape->key_count != 513u || shape->padded_key_count != 768u ||
        shape->n_head != 48u || shape->n_head_kv != 8u ||
        shape->head_dim != 128u) {
        return 0u;
    }
    return 3u;
}

#endif
