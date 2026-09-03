#ifndef DS4_BENCH_QUALIFICATION_H
#define DS4_BENCH_QUALIFICATION_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "ds4_bench_sequence.h"
#include "ds4_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    DS4_BENCH_QUALIFICATION_EVENT_REQUEST_ACCEPTED = 0,
    DS4_BENCH_QUALIFICATION_EVENT_FIRST_TOKEN = 1,
    DS4_BENCH_QUALIFICATION_EVENT_REQUEST_COMPLETE = 2,
} ds4_bench_qualification_event;

typedef struct {
    const ds4_bench_sequence *sequence;
    ds4_bench_qualification_event event;
    const char *request_id;
    uint32_t repetition_index;
    uint64_t monotonic_ns;
    uint64_t session_payload_bytes;
    const ds4_runtime_wire_snapshot *runtime_snapshot;
    const ds4_runtime_request_metrics *request_metrics;
} ds4_bench_qualification_record;

/* Validate and emit one closed qualification lifecycle object followed by LF.
 * The stream remains owned by the caller and is never closed. */
bool ds4_bench_qualification_emit_record(
    FILE *stream,
    const ds4_bench_qualification_record *record,
    char *error,
    size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
