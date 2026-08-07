#ifndef DS4_LAGUNA_PLAN_H
#define DS4_LAGUNA_PLAN_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "ds4_laguna_stream.h"
#include "ds4_plan_io.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DS4_LAGUNA_QUALIFICATION_PLAN_SCHEMA \
    "ds4.laguna.qualification-plan/v1"

typedef struct {
    uint64_t device;
    uint64_t inode;
    uint64_t size_bytes;
    uint64_t mtime_ns;
} ds4_laguna_file_identity;

typedef struct {
    ds4_laguna_page_range *ranges;
    size_t range_count;
    uint64_t page_size;
    uint64_t mapped_page_bytes;
    uint64_t eligible_unique_bytes;
    uint64_t unavoidable_bytes;
} ds4_laguna_page_plan;

typedef struct {
    ds4_laguna_file_identity model_identity;
    const char *model_sha256;
    const ds4_laguna_ledger *ledger;
    const ds4_laguna_allocation_plan *allocation;
    const ds4_laguna_page_plan *page_cache;
} ds4_laguna_qualification_plan_input;

/* Build the unique union of pages wholly contained in individual tensor
 * ranges.  Boundary pages shared by adjacent tensors are intentionally not
 * made eligible.  `out` must be zero-initialized or previously freed. */
bool ds4_laguna_page_plan_make(ds4_laguna_page_plan *out,
                               const ds4_laguna_ledger *ledger,
                               uint64_t page_size,
                               char *error,
                               size_t error_size);

void ds4_laguna_page_plan_free(ds4_laguna_page_plan *plan);

/* Serialize one closed, deterministic JSON qualification plan.  The returned
 * allocation is nul-terminated for convenience; `size_out` excludes that nul
 * and is the exact byte sequence to hash or publish.  `ledger_sha256` hashes
 * exactly the canonical ledger object embedded in those bytes. */
bool ds4_laguna_qualification_plan_serialize(
    const ds4_laguna_qualification_plan_input *input,
    char **bytes_out,
    size_t *size_out,
    char ledger_sha256[DS4_PLAN_IO_SHA256_HEX_SIZE],
    char *error,
    size_t error_size);

void ds4_laguna_qualification_plan_bytes_free(char *bytes);

/* Serialize and durably publish via ds4_plan_io_publish(). */
bool ds4_laguna_qualification_plan_publish(
    const char *path,
    const ds4_laguna_qualification_plan_input *input,
    char plan_sha256[DS4_PLAN_IO_SHA256_HEX_SIZE],
    char ledger_sha256[DS4_PLAN_IO_SHA256_HEX_SIZE],
    char *error,
    size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
