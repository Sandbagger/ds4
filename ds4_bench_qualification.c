#include "ds4_bench_qualification.h"

#include <inttypes.h>
#include <stdarg.h>
#include <string.h>

#define DS4_BENCH_QUALIFICATION_JSON_CAPACITY \
    (DS4_RUNTIME_JSON_CAPACITY + DS4_RUNTIME_REQUEST_JSON_CAPACITY + 4096u)

#define DS4_BENCH_QUALIFICATION_CONTEXT_TOKENS UINT32_C(32768)
#define DS4_BENCH_QUALIFICATION_PREFILL_CHUNK_TOKENS UINT32_C(4096)
#define DS4_BENCH_QUALIFICATION_SESSION_SLOTS UINT32_C(1)

typedef struct {
    const char *id;
    uint64_t cache_bytes;
    const char *prompt_ids[4];
    uint32_t prompt_tokens[4];
} ds4_bench_qualification_profile;

static const ds4_bench_qualification_profile
    ds4_bench_qualification_profiles[] = {
        {
            "cache-8gib",
            UINT64_C(8589934592),
            {"native-512", "native-2048", "native-28672", "native-8192"},
            {UINT32_C(512), UINT32_C(2048), UINT32_C(28672), UINT32_C(8192)},
        },
        {
            "cache-12gib",
            UINT64_C(12884901888),
            {"native-2048", "native-8192", "native-512", "native-28672"},
            {UINT32_C(2048), UINT32_C(8192), UINT32_C(512), UINT32_C(28672)},
        },
        {
            "cache-16gib",
            UINT64_C(17179869184),
            {"native-8192", "native-28672", "native-2048", "native-512"},
            {UINT32_C(8192), UINT32_C(28672), UINT32_C(2048), UINT32_C(512)},
        },
    };

typedef struct {
    char *buffer;
    size_t capacity;
    size_t length;
    bool ok;
} ds4_bench_qualification_json_writer;

static void ds4_bench_qualification_clear_error(
        char *error, size_t error_size) {
    if (error && error_size != 0u) error[0] = '\0';
}

static bool ds4_bench_qualification_fail(
        char *error,
        size_t error_size,
        const char *format,
        ...) {
    if (error && error_size != 0u) {
        va_list arguments;
        va_start(arguments, format);
        (void)vsnprintf(error, error_size, format, arguments);
        va_end(arguments);
        error[error_size - 1u] = '\0';
    }
    return false;
}

static bool ds4_bench_qualification_bounded_equals(
        const char *value,
        size_t capacity,
        const char *expected) {
    if (!value || !expected) return false;
    const size_t expected_length = strlen(expected);
    if (expected_length >= capacity ||
        memcmp(value, expected, expected_length) != 0) {
        return false;
    }
    return value[expected_length] == '\0';
}

static bool ds4_bench_qualification_sha256_valid(
        const char value[DS4_BENCH_SEQUENCE_SHA256_HEX_SIZE]) {
    bool nonzero = false;
    if (!value) return false;
    for (size_t i = 0u; i < DS4_BENCH_SEQUENCE_SHA256_HEX_LENGTH; i++) {
        const unsigned char byte = (unsigned char)value[i];
        if (!((byte >= (unsigned char)'0' && byte <= (unsigned char)'9') ||
              (byte >= (unsigned char)'a' && byte <= (unsigned char)'f'))) {
            return false;
        }
        if (byte != (unsigned char)'0') nonzero = true;
    }
    return value[DS4_BENCH_SEQUENCE_SHA256_HEX_LENGTH] == '\0' && nonzero;
}

static bool ds4_bench_qualification_uuid_valid(const char *uuid) {
    bool nonzero = false;
    if (!uuid) return false;
    for (size_t i = 0u; i < 36u; i++) {
        const char byte = uuid[i];
        const bool hyphen = i == 8u || i == 13u || i == 18u || i == 23u;
        if (byte == '\0') return false;
        if (hyphen) {
            if (byte != '-') return false;
        } else {
            if (!((byte >= '0' && byte <= '9') ||
                  (byte >= 'a' && byte <= 'f'))) {
                return false;
            }
            if (byte != '0') nonzero = true;
        }
    }
    return uuid[36] == '\0' && nonzero;
}

static const ds4_bench_qualification_profile *
 ds4_bench_qualification_profile_for_sequence(
        const ds4_bench_sequence *sequence) {
    for (size_t i = 0u;
         i < sizeof(ds4_bench_qualification_profiles) /
                 sizeof(ds4_bench_qualification_profiles[0]);
         i++) {
        const ds4_bench_qualification_profile *profile =
            &ds4_bench_qualification_profiles[i];
        if (ds4_bench_qualification_bounded_equals(
                sequence->profile_id, sizeof(sequence->profile_id),
                profile->id)) {
            return profile;
        }
    }
    return NULL;
}

static const char *ds4_bench_qualification_event_name(
        ds4_bench_qualification_event event) {
    switch (event) {
    case DS4_BENCH_QUALIFICATION_EVENT_REQUEST_ACCEPTED:
        return "request_accepted";
    case DS4_BENCH_QUALIFICATION_EVENT_FIRST_TOKEN:
        return "first_token";
    case DS4_BENCH_QUALIFICATION_EVENT_REQUEST_COMPLETE:
        return "request_complete";
    }
    return NULL;
}

static const char *ds4_bench_qualification_terminal_status_name(
        ds4_runtime_request_terminal_status status) {
    switch (status) {
    case DS4_RUNTIME_REQUEST_COMPLETED:
        return "completed";
    case DS4_RUNTIME_REQUEST_CANCELLED:
        return "cancelled";
    case DS4_RUNTIME_REQUEST_REJECTED:
        return "rejected";
    case DS4_RUNTIME_REQUEST_RECOVERABLE_ERROR:
        return "recoverable_error";
    case DS4_RUNTIME_REQUEST_UNSAFE_ERROR:
        return "unsafe_error";
    case DS4_RUNTIME_REQUEST_TERMINAL_STATUS_COUNT:
        break;
    }
    return NULL;
}

static bool ds4_bench_qualification_validate_sequence(
        const ds4_bench_sequence *sequence,
        const ds4_bench_qualification_profile **profile_out,
        char *error,
        size_t error_size) {
    if (!sequence || !profile_out) {
        return ds4_bench_qualification_fail(
            error, error_size, "qualification record sequence is null");
    }
    if (!ds4_bench_qualification_sha256_valid(sequence->manifest_sha256) ||
        !ds4_bench_qualification_sha256_valid(sequence->sequence_sha256) ||
        !ds4_bench_qualification_sha256_valid(sequence->input_sha256)) {
        return ds4_bench_qualification_fail(
            error, error_size,
            "qualification sequence has a non-canonical SHA-256 digest");
    }
    if (sequence->input_size_bytes == 0u ||
        sequence->input_size_bytes > DS4_BENCH_SEQUENCE_MAX_INPUT_BYTES ||
        (uint64_t)sequence->input_size != sequence->input_size_bytes ||
        !sequence->input_bytes) {
        return ds4_bench_qualification_fail(
            error, error_size,
            "qualification sequence input is not a parsed bounded payload");
    }

    const ds4_bench_qualification_profile *profile =
        ds4_bench_qualification_profile_for_sequence(sequence);
    if (!profile) {
        return ds4_bench_qualification_fail(
            error, error_size,
            "qualification sequence has an unknown profile_id");
    }
    if (sequence->cache_bytes != profile->cache_bytes) {
        return ds4_bench_qualification_fail(
            error, error_size,
            "qualification sequence cache_bytes does not match profile");
    }
    if (sequence->prompt_order_index > 3u ||
        !ds4_bench_qualification_bounded_equals(
            sequence->prompt_id, sizeof(sequence->prompt_id),
            profile->prompt_ids[sequence->prompt_order_index]) ||
        sequence->prompt_tokens !=
            profile->prompt_tokens[sequence->prompt_order_index]) {
        return ds4_bench_qualification_fail(
            error, error_size,
            "qualification sequence prompt binding does not match profile order");
    }
    *profile_out = profile;
    return true;
}

static bool ds4_bench_qualification_validate_runtime(
        const ds4_runtime_wire_snapshot *snapshot,
        const ds4_bench_qualification_profile *profile,
        char *runtime_json,
        size_t runtime_json_capacity,
        size_t *runtime_json_length,
        char *error,
        size_t error_size) {
    if (!snapshot || !profile || !runtime_json ||
        !runtime_json_length) {
        return ds4_bench_qualification_fail(
            error, error_size, "qualification runtime snapshot is null");
    }
    if (!ds4_bench_qualification_uuid_valid(snapshot->instance_id)) {
        return ds4_bench_qualification_fail(
            error, error_size,
            "qualification runtime snapshot has an invalid instance_id");
    }
    if (snapshot->configured_context_tokens !=
            DS4_BENCH_QUALIFICATION_CONTEXT_TOKENS ||
        snapshot->configured_prefill_chunk_tokens !=
            DS4_BENCH_QUALIFICATION_PREFILL_CHUNK_TOKENS ||
        snapshot->configured_session_slots !=
            DS4_BENCH_QUALIFICATION_SESSION_SLOTS ||
        !snapshot->configured_ssd_streaming ||
        snapshot->configured_ssd_streaming_cache_bytes != profile->cache_bytes ||
        snapshot->effective_context_tokens !=
            DS4_BENCH_QUALIFICATION_CONTEXT_TOKENS ||
        snapshot->effective_prefill_chunk_tokens !=
            DS4_BENCH_QUALIFICATION_PREFILL_CHUNK_TOKENS ||
        snapshot->effective_session_slots !=
            DS4_BENCH_QUALIFICATION_SESSION_SLOTS ||
        snapshot->expert_cache_limit_bytes != profile->cache_bytes ||
        snapshot->allocations.category_bounds[
            DS4_RUNTIME_CATEGORY_EXPERT_CACHE_PAYLOAD] != profile->cache_bytes ||
        !snapshot->allocations.external_sample.unrelated_process_inventory_stable) {
        return ds4_bench_qualification_fail(
            error, error_size,
            "qualification runtime snapshot does not use the pinned configuration");
    }
    if (!ds4_runtime_wire_snapshot_json(
            snapshot, runtime_json, runtime_json_capacity,
            runtime_json_length) ||
        *runtime_json_length >= runtime_json_capacity ||
        runtime_json[*runtime_json_length] != '\0') {
        return ds4_bench_qualification_fail(
            error, error_size,
            "qualification runtime snapshot is not canonical JSON");
    }
    return true;
}

static bool ds4_bench_qualification_json_append_bytes(
        ds4_bench_qualification_json_writer *writer,
        const char *bytes,
        size_t length) {
    if (!writer || !writer->ok || !bytes ||
        writer->length >= writer->capacity ||
        length >= writer->capacity - writer->length) {
        if (writer) writer->ok = false;
        return false;
    }
    memcpy(writer->buffer + writer->length, bytes, length);
    writer->length += length;
    writer->buffer[writer->length] = '\0';
    return true;
}

static bool ds4_bench_qualification_json_append_literal(
        ds4_bench_qualification_json_writer *writer,
        const char *literal) {
    return literal && ds4_bench_qualification_json_append_bytes(
        writer, literal, strlen(literal));
}

static bool ds4_bench_qualification_json_appendf(
        ds4_bench_qualification_json_writer *writer,
        const char *format,
        ...) {
    if (!writer || !writer->ok || !format ||
        writer->length >= writer->capacity) {
        if (writer) writer->ok = false;
        return false;
    }
    va_list arguments;
    va_start(arguments, format);
    const int result = vsnprintf(
        writer->buffer + writer->length,
        writer->capacity - writer->length,
        format,
        arguments);
    va_end(arguments);
    if (result < 0 ||
        (size_t)result >= writer->capacity - writer->length) {
        writer->ok = false;
        return false;
    }
    writer->length += (size_t)result;
    return true;
}

bool ds4_bench_qualification_emit_record(
        FILE *stream,
        const ds4_bench_qualification_record *record,
        char *error,
        size_t error_size) {
    ds4_bench_qualification_clear_error(error, error_size);
    if (!stream) {
        return ds4_bench_qualification_fail(
            error, error_size, "qualification output stream is null");
    }
    if (!record) {
        return ds4_bench_qualification_fail(
            error, error_size, "qualification record is null");
    }
    if (record->repetition_index > 3u) {
        return ds4_bench_qualification_fail(
            error, error_size,
            "qualification repetition_index is outside 0..3");
    }
    const char *event = ds4_bench_qualification_event_name(record->event);
    if (!event) {
        return ds4_bench_qualification_fail(
            error, error_size, "qualification event is invalid");
    }
    if (record->monotonic_ns == 0u) {
        return ds4_bench_qualification_fail(
            error, error_size, "qualification event time is zero");
    }
    if (!ds4_bench_qualification_uuid_valid(record->request_id)) {
        return ds4_bench_qualification_fail(
            error, error_size,
            "qualification request_id is not a canonical nonzero UUID");
    }

    const ds4_bench_qualification_profile *profile = NULL;
    if (!ds4_bench_qualification_validate_sequence(
            record->sequence, &profile, error, error_size)) {
        return false;
    }

    char runtime_json[DS4_RUNTIME_JSON_CAPACITY];
    size_t runtime_json_length = 0u;
    if (!ds4_bench_qualification_validate_runtime(
            record->runtime_snapshot, profile,
            runtime_json, sizeof(runtime_json), &runtime_json_length,
            error, error_size)) {
        return false;
    }
    if (strcmp(record->request_id, record->runtime_snapshot->instance_id) == 0) {
        return ds4_bench_qualification_fail(
            error, error_size,
            "qualification request_id must differ from runtime instance_id");
    }

    char metrics_json[DS4_RUNTIME_REQUEST_JSON_CAPACITY];
    size_t metrics_json_length = 0u;
    const char *terminal_status = NULL;
    if (record->event == DS4_BENCH_QUALIFICATION_EVENT_REQUEST_COMPLETE) {
        const ds4_runtime_request_metrics *metrics = record->request_metrics;
        if (!metrics) {
            return ds4_bench_qualification_fail(
                error, error_size,
                "request_complete requires request metrics");
        }
        terminal_status = ds4_bench_qualification_terminal_status_name(
            metrics->terminal_status);
        if (!terminal_status ||
            !ds4_bench_qualification_uuid_valid(metrics->request_id) ||
            !ds4_bench_qualification_uuid_valid(metrics->instance_id) ||
            strcmp(metrics->request_id, record->request_id) != 0 ||
            strcmp(metrics->instance_id,
                   record->runtime_snapshot->instance_id) != 0 ||
            metrics->prompt_tokens != profile->prompt_tokens[
                record->sequence->prompt_order_index] ||
            metrics->snapshot_seq == UINT64_MAX ||
            metrics->snapshot_seq + 1u !=
                record->runtime_snapshot->snapshot_seq) {
            return ds4_bench_qualification_fail(
                error, error_size,
                "request metrics do not match the completed qualification record");
        }
        if (!ds4_runtime_request_metrics_json(
                metrics, metrics_json, sizeof(metrics_json),
                &metrics_json_length) ||
            metrics_json_length >= sizeof(metrics_json) ||
            metrics_json[metrics_json_length] != '\0') {
            return ds4_bench_qualification_fail(
                error, error_size,
                "request metrics are not canonical JSON");
        }
    } else if (record->request_metrics != NULL) {
        return ds4_bench_qualification_fail(
            error, error_size,
            "pre-terminal qualification events must not carry request metrics");
    }

    char output[DS4_BENCH_QUALIFICATION_JSON_CAPACITY];
    ds4_bench_qualification_json_writer writer = {
        .buffer = output,
        .capacity = sizeof(output),
        .length = 0u,
        .ok = true,
    };
    output[0] = '\0';
    if (!ds4_bench_qualification_json_appendf(
            &writer,
            "{\"schema\":\"ds4.bench.qualification/v1\","
            "\"manifest_sha256\":\"%s\","
            "\"sequence_sha256\":\"%s\","
            "\"profile_id\":\"%s\","
            "\"prompt_order_index\":%" PRIu32 ","
            "\"prompt_id\":\"%s\","
            "\"input_sha256\":\"%s\","
            "\"event\":\"%s\","
            "\"request_id\":\"%s\","
            "\"instance_id\":\"%s\","
            "\"snapshot_seq\":\"%" PRIu64 "\","
            "\"repetition_index\":%" PRIu32 ","
            "\"monotonic_ns\":\"%" PRIu64 "\","
            "\"mode\":\"streamed\","
            "\"session_payload_bytes\":\"%" PRIu64 "\","
            "\"kv_allocated_bytes\":\"%" PRIu64 "\","
            "\"configured_prefill_rows\":%" PRIu32 ","
            "\"allocated_prefill_rows\":%" PRIu32 ","
            "\"expert_cache_limit_bytes\":\"%" PRIu64 "\","
            "\"expert_cache_current_bytes\":\"%" PRIu64 "\","
            "\"expert_cache_peak_bytes\":\"%" PRIu64 "\","
            "\"qualification_total_current_bytes\":\"%" PRIu64 "\","
            "\"qualification_total_bound_bytes\":\"%" PRIu64 "\","
            "\"qualification_total_peak_bytes\":\"%" PRIu64 "\","
            "\"model_inode_resident_bytes\":\"%" PRIu64 "\","
            "\"external_attribution\":{\"model_source_resident\":\"%" PRIu64
            "\",\"host_library_unattributed\":\"%" PRIu64
            "\",\"cuda_library_unattributed\":\"%" PRIu64
            "\",\"unrelated_process_inventory_stable\":%s},"
            "\"runtime\":",
            record->sequence->manifest_sha256,
            record->sequence->sequence_sha256,
            record->sequence->profile_id,
            record->sequence->prompt_order_index,
            record->sequence->prompt_id,
            record->sequence->input_sha256,
            event,
            record->request_id,
            record->runtime_snapshot->instance_id,
            record->runtime_snapshot->snapshot_seq,
            record->repetition_index,
            record->monotonic_ns,
            record->session_payload_bytes,
            record->runtime_snapshot->allocations.category_current[
                DS4_RUNTIME_CATEGORY_KV_STATE],
            record->runtime_snapshot->configured_prefill_rows,
            record->runtime_snapshot->allocated_prefill_rows,
            record->runtime_snapshot->expert_cache_limit_bytes,
            record->runtime_snapshot->allocations.category_current[
                DS4_RUNTIME_CATEGORY_EXPERT_CACHE_PAYLOAD],
            record->runtime_snapshot->allocations.category_peak[
                DS4_RUNTIME_CATEGORY_EXPERT_CACHE_PAYLOAD],
            record->runtime_snapshot->allocations.qualification_total_current,
            record->runtime_snapshot->allocations.qualification_total_bound_bytes,
            record->runtime_snapshot->allocations.qualification_total_peak,
            record->runtime_snapshot->allocations.report_current[
                DS4_RUNTIME_REPORT_MODEL_SOURCE_RESIDENT],
            record->runtime_snapshot->allocations.report_current[
                DS4_RUNTIME_REPORT_MODEL_SOURCE_RESIDENT],
            record->runtime_snapshot->allocations.report_current[
                DS4_RUNTIME_REPORT_HOST_LIBRARY_UNATTRIBUTED],
            record->runtime_snapshot->allocations.report_current[
                DS4_RUNTIME_REPORT_CUDA_LIBRARY_UNATTRIBUTED],
            record->runtime_snapshot->allocations.external_sample
                .unrelated_process_inventory_stable ? "true" : "false") ||
        !ds4_bench_qualification_json_append_bytes(
            &writer, runtime_json, runtime_json_length)) {
        return ds4_bench_qualification_fail(
            error, error_size,
            "qualification record exceeds the JSON serialization limit");
    }
    if (record->event == DS4_BENCH_QUALIFICATION_EVENT_REQUEST_COMPLETE &&
        (!ds4_bench_qualification_json_append_literal(
             &writer, ",\"request_metrics\":") ||
         !ds4_bench_qualification_json_append_bytes(
             &writer, metrics_json, metrics_json_length) ||
         !ds4_bench_qualification_json_appendf(
             &writer, ",\"terminal_status\":\"%s\"", terminal_status))) {
        return ds4_bench_qualification_fail(
            error, error_size,
            "qualification record exceeds the JSON serialization limit");
    }
    if (!ds4_bench_qualification_json_append_literal(&writer, "}\n")) {
        return ds4_bench_qualification_fail(
            error, error_size,
            "qualification record exceeds the JSON serialization limit");
    }

    if (fwrite(output, 1u, writer.length, stream) != writer.length) {
        return ds4_bench_qualification_fail(
            error, error_size, "failed to write qualification record");
    }
    if (fflush(stream) != 0) {
        return ds4_bench_qualification_fail(
            error, error_size, "failed to flush qualification record");
    }
    return true;
}
