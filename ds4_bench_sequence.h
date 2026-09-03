#ifndef DS4_BENCH_SEQUENCE_H
#define DS4_BENCH_SEQUENCE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DS4_BENCH_SEQUENCE_SCHEMA "ds4.qualification-sequence/v1"
#define DS4_BENCH_SEQUENCE_LINE_COUNT 24u
#define DS4_BENCH_SEQUENCE_SHA256_HEX_LENGTH 64u
#define DS4_BENCH_SEQUENCE_SHA256_HEX_SIZE \
    (DS4_BENCH_SEQUENCE_SHA256_HEX_LENGTH + 1u)
#define DS4_BENCH_SEQUENCE_MAX_FILE_BYTES (16u * 1024u * 1024u)
#define DS4_BENCH_SEQUENCE_MAX_INPUT_BYTES (16u * 1024u * 1024u)

/* Owned result of one strict qualification-sequence parse.  All scalar
 * fields are valid only when ds4_bench_sequence_parse_file returns true.
 * input_bytes is allocated by the parser and belongs to this struct. */
typedef struct ds4_bench_sequence {
    char manifest_sha256[DS4_BENCH_SEQUENCE_SHA256_HEX_SIZE];
    char profile_id[32];
    uint64_t cache_bytes;
    uint32_t prompt_order_index;
    char prompt_id[32];
    uint32_t prompt_tokens;
    uint64_t input_size_bytes;
    size_t input_size;
    unsigned char *input_bytes;
    char input_sha256[DS4_BENCH_SEQUENCE_SHA256_HEX_SIZE];
} ds4_bench_sequence;

/* Initialize or release an owned result.  The caller must initialize a result
 * before its first parse (either with this function or with {0}); parsing
 * releases any prior result. */
void ds4_bench_sequence_init(ds4_bench_sequence *sequence);
void ds4_bench_sequence_free(ds4_bench_sequence *sequence);

/* Parse one immutable, exact 24-line sequence file.  The function never exits
 * the process.  On failure it returns false and writes a bounded diagnostic to
 * error when a buffer is supplied. */
bool ds4_bench_sequence_parse_file(const char *path,
                                   ds4_bench_sequence *sequence,
                                   char *error,
                                   size_t error_size);

#ifdef DS4_BENCH_SEQUENCE_TEST_HOOKS
typedef void (*ds4_bench_sequence_test_after_first_read_hook)(
    int fd,
    void *context);

/* Test-only deterministic mutation seam.  Production builds do not expose or
 * retain this hook. */
void ds4_bench_sequence_test_set_after_first_read_hook(
    ds4_bench_sequence_test_after_first_read_hook hook,
    void *context);
#endif

#ifdef __cplusplus
}
#endif

#endif
