#define _POSIX_C_SOURCE 200809L
#include "../ds4_bench_qualification.h"
#include "../ds4_bench_sequence.h"
#include "../ds4_runtime.h"
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
static unsigned check_count;
static unsigned successful_emissions;
static int failures;
#define CHECK(condition, message) do {                                      \
    check_count++;                                                          \
    if (!(condition)) {                                                     \
        fprintf(stderr, "FAIL: %s (line %d)\n", (message), __LINE__);    \
        failures++;                                                         \
    }                                                                        \
} while (0)
static const uint32_t prompt_tokens[] = {512u, 2048u, 8192u, 28672u};
static const char *const prompt_ids[] = {"native-512", "native-2048", "native-8192", "native-28672"};
static const char *const request_ids[] = {
    "123e4567-e89b-12d3-a456-426614174001",
    "123e4567-e89b-12d3-a456-426614174002",
    "123e4567-e89b-12d3-a456-426614174003",
    "123e4567-e89b-12d3-a456-426614174004",
};
static bool read_line(int fd, char *line, size_t capacity) {
    if (fd < 0 || !line || capacity < 2u) return false;
    size_t length = 0u;
    while (length + 1u < capacity) {
        struct pollfd descriptor = {.fd = fd, .events = POLLIN | POLLHUP};
        int ready;
        do {
            ready = poll(&descriptor, 1u, 1000);
        } while (ready < 0 && errno == EINTR);
        if (ready <= 0) return false;
        unsigned char byte = 0u;
        const ssize_t count = read(fd, &byte, 1u);
        if (count < 0 && errno == EINTR) continue;
        if (count != 1) return false;
        line[length++] = (char)byte;
        if (byte == '\n') {
            line[length] = '\0';
            return true;
        }
    }
    return false;
}
static void fill_runtime(ds4_runtime_wire_snapshot *snapshot) {
    memset(snapshot, 0, sizeof(*snapshot));
    memcpy(snapshot->instance_id,
           "123e4567-e89b-12d3-a456-426614174000",
           sizeof(snapshot->instance_id));
    snapshot->snapshot_seq = 100u;
    snapshot->state = DS4_RUNTIME_WIRE_STATE_READY;
    memcpy(snapshot->build.revision,
           "1111111111111111111111111111111111111111",
           sizeof(snapshot->build.revision));
    memcpy(snapshot->build.backend, "cuda", sizeof("cuda"));
    memcpy(snapshot->build.features[0], "resident", sizeof("resident"));
    snapshot->build.feature_count = 1u;
    snapshot->executable = (ds4_runtime_file_identity){
        .device = 11u, .inode = 12u, .size_bytes = 13u, .mtime_ns = 14u,
    };
    snapshot->model = (ds4_runtime_file_identity){
        .device = 21u, .inode = 22u, .size_bytes = 23u, .mtime_ns = 24u,
    };
    memcpy(snapshot->model_id, "resident-s-2.1", sizeof("resident-s-2.1"));
    memcpy(snapshot->model_family, "resident", sizeof("resident"));
    snapshot->configured_context_tokens = 32768u;
    snapshot->configured_prefill_chunk_tokens = 4096u;
    snapshot->configured_session_slots = 1u;
    snapshot->configured_ssd_streaming = false;
    snapshot->configured_ssd_streaming_cache_bytes = 0u;
    snapshot->effective_context_tokens = 32768u;
    snapshot->effective_prefill_chunk_tokens = 4096u;
    snapshot->effective_session_slots = 1u;
    snapshot->expert_cache_limit_bytes = 0u;
    snapshot->configured_prefill_rows = 4096u;
    snapshot->allocated_prefill_rows = 4096u;
    static const uint64_t category_current[DS4_RUNTIME_OWNED_CATEGORY_COUNT] =
        {4096u, 0u, 0u, 0u, 0u, 0u, 0u, 1024u};
    static const uint64_t category_bounds[DS4_RUNTIME_OWNED_CATEGORY_COUNT] =
        {8192u, 0u, 0u, 0u, 0u, 0u, 0u, 2048u};
    static const uint64_t report_current[DS4_RUNTIME_REPORT_COUNT] =
        {8192u, 4096u, 5000u, 100u, 200u};
    static const uint64_t report_bounds[DS4_RUNTIME_REPORT_COUNT] =
        {16384u, 8192u, 8192u, 1024u, 1024u};
    memcpy(snapshot->allocations.category_current, category_current,
           sizeof(category_current));
    memcpy(snapshot->allocations.category_peak, category_current,
           sizeof(category_current));
    memcpy(snapshot->allocations.category_bounds, category_bounds,
           sizeof(category_bounds));
    memcpy(snapshot->allocations.report_current, report_current,
           sizeof(report_current));
    memcpy(snapshot->allocations.report_peak, report_current,
           sizeof(report_current));
    memcpy(snapshot->allocations.report_bounds, report_bounds,
           sizeof(report_bounds));
    snapshot->allocations.owned_total_current = 5120u;
    snapshot->allocations.owned_total_peak = 5120u;
    snapshot->allocations.owned_total_bound_bytes = 16384u;
    snapshot->allocations.qualification_total_current = 10420u;
    snapshot->allocations.qualification_total_peak = 10420u;
    snapshot->allocations.qualification_total_bound_bytes = 20000u;
    snapshot->allocations.external_sample.host_library_unattributed_bytes = 100u; snapshot->allocations.external_sample.cuda_library_unattributed_bytes = 200u;
    snapshot->allocations.external_sample.attributed_valid = true; snapshot->allocations.external_sample.checkpoint_sequence = 1u;
    snapshot->allocations.external_sample.unrelated_process_inventory_stable = true;
}
static void fill_metrics(ds4_runtime_request_metrics *metrics, const char *request_id,
                          uint64_t snapshot_seq, uint32_t prompt) {
    memset(metrics, 0, sizeof(*metrics));
    memcpy(metrics->request_id, request_id, sizeof(metrics->request_id));
    memcpy(metrics->instance_id,
           "123e4567-e89b-12d3-a456-426614174000",
           sizeof(metrics->instance_id));
    metrics->snapshot_seq = snapshot_seq;
    metrics->prompt_tokens = prompt;
    metrics->generated_tokens = 1u;
    metrics->ttft_present = true;
    metrics->ttft_ns = 100u;
    metrics->prefill_tokens_per_second = 512.0;
    metrics->visible_decode_tokens_per_second = 1.0;
    metrics->wall_time_ns = 200u;
    metrics->terminal_status = DS4_RUNTIME_REQUEST_COMPLETED;
}
static void fill_sequence(ds4_bench_resident_sequence *resident, uint32_t order_index) {
    static unsigned char input[] = "resident fixture";
    memset(resident, 0, sizeof(*resident));
    memcpy(resident->sequence.manifest_sha256,
           "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
           sizeof(resident->sequence.manifest_sha256));
    memcpy(resident->sequence.sequence_sha256,
           "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
           sizeof(resident->sequence.sequence_sha256));
    memcpy(resident->sequence.profile_id, "resident", sizeof("resident"));
    resident->sequence.cache_bytes = 0u;
    resident->sequence.prompt_order_index = order_index;
    snprintf(resident->sequence.prompt_id,
             sizeof(resident->sequence.prompt_id), "%s", prompt_ids[order_index]);
    resident->sequence.prompt_tokens = prompt_tokens[order_index];
    resident->sequence.input_size_bytes = sizeof(input) - 1u;
    resident->sequence.input_size = sizeof(input) - 1u;
    resident->sequence.input_bytes = input;
    memcpy(resident->sequence.input_sha256,
           "f470936709873547a3647d5da4d6c9e6d2f8ef049804dfa0cde4487a32d86a0f",
           sizeof(resident->sequence.input_sha256));
}
static void expect_resident_reject(const char *label,
                                    const ds4_bench_resident_qualification_record *record) {
    FILE *stream = tmpfile();
    CHECK(stream != NULL, "create caller-owned failure stream");
    if (!stream) return;
    char error[256] = {0};
    CHECK(!ds4_bench_resident_qualification_emit_record(
              stream, record, error, sizeof(error)), label);
    CHECK(error[0] != '\0', "resident rejection includes a diagnostic");
    CHECK(ftell(stream) == 0L, "validation failure writes no output bytes");
    CHECK(fputc('x', stream) == 'x' && fflush(stream) == 0,
          "validation failure leaves FILE caller-owned");
    fclose(stream);
}
static void expect_old_reject(const char *label,
                               const ds4_bench_qualification_record *record) {
    FILE *stream = tmpfile();
    CHECK(stream != NULL, "create old-emitter failure stream");
    if (!stream) return;
    char error[256] = {0};
    CHECK(!ds4_bench_qualification_emit_record(
              stream, record, error, sizeof(error)), label);
    CHECK(error[0] != '\0', "old-emitter rejection includes a diagnostic");
    CHECK(ftell(stream) == 0L, "old-emitter cross-mode writes no bytes");
    CHECK(fputc('x', stream) == 'x' && fflush(stream) == 0,
          "old-emitter failure leaves FILE caller-owned");
    fclose(stream);
}
static int finish_test(void) {
    alarm(0);
    printf("resident-bench-qualification-emitter: successful-emissions=%u checks=%u failures=%d\n",
           successful_emissions, check_count, failures);
    return failures == 0 ? 0 : 1;
}
static void check_success_line(const char *line,
                               const char *runtime_json,
                               const char *metrics_json,
                               uint32_t order_index,
                               uint32_t repetition,
                               ds4_bench_qualification_event event) {
    static const char *const event_names[] = {
        "request_accepted", "first_token", "request_complete"};
    CHECK(line != NULL && line[0] != '\0', "resident emitter returns a line");
    const size_t length = strlen(line);
    CHECK(length >= 3u && line[length - 1u] == '\n' &&
              line[length - 2u] == '}', "resident record is one closed LF line");
    CHECK(strstr(line, "NaN") == NULL && strstr(line, "Infinity") == NULL &&
              strstr(line, "nan") == NULL && strstr(line, "inf") == NULL,
          "resident record has no non-finite values");
    CHECK(strstr(line, "\"schema\":\"" DS4_BENCH_RESIDENT_QUALIFICATION_SCHEMA
                     "\"") != NULL,
          "resident schema is exact and distinct");
    CHECK(strstr(line, "\"mode\":\"resident\"") != NULL,
          "resident mode is exact");
    CHECK(strstr(line, "\"profile_id\":\"resident\"") != NULL,
          "resident profile is exact");
    char prompt[128];
    snprintf(prompt, sizeof(prompt),
             "\"prompt_order_index\":%u,\"prompt_id\":\"%s\",\"input_sha256\":",
             order_index, prompt_ids[order_index]);
    CHECK(strstr(line, prompt) != NULL, "canonical prompt order is copied");
    CHECK(strstr(line, "\"event\":\"") != NULL &&
              strstr(line, event_names[event]) != NULL,
          "resident milestone is emitted");
    char repetition_text[32];
    snprintf(repetition_text, sizeof(repetition_text), "\"repetition_index\":%u", repetition);
    CHECK(strstr(line, repetition_text) != NULL && strstr(line, runtime_json) != NULL,
          "resident record copies repetition and canonical runtime JSON");
    CHECK(strstr(line,
                 "\"expert_cache_limit_bytes\":\"0\","
                 "\"expert_cache_current_bytes\":\"0\","
                 "\"expert_cache_peak_bytes\":\"0\"") != NULL,
          "resident cache limit/current/peak are all zero");
    CHECK(strstr(line,
                 "\"external_attribution\":{\"model_source_resident\":\"5000\"") != NULL,
          "resident keeps raw physical attribution");
    CHECK(strstr(line, "\"static_weights\":{\"current_bytes\":\"4096\"") != NULL &&
              strstr(line, "\"model_mapping_registered\":{\"current_bytes\":\"4096\"") != NULL,
          "resident keeps static and registration footprint evidence");
    if (event == DS4_BENCH_QUALIFICATION_EVENT_REQUEST_COMPLETE) {
        CHECK(metrics_json != NULL && strstr(line, metrics_json) != NULL,
              "completion carries canonical request metrics");
        CHECK(strstr(line, "\"terminal_status\":\"completed\"") != NULL,
              "completion carries terminal status");
        const char *runtime_marker = strstr(line, "\"runtime\":");
        const char *metrics_marker = strstr(line, "\"request_metrics\":");
        CHECK(runtime_marker != NULL && metrics_marker != NULL &&
                  metrics_marker > runtime_marker,
              "completion metrics are appended after runtime");
    } else {
        CHECK(metrics_json == NULL && strstr(line, "\"request_metrics\"") == NULL,
              "pre-terminal records omit request metrics");
        CHECK(strstr(line, "\"terminal_status\"") == NULL,
              "pre-terminal records omit terminal status");
    }
    (void)repetition;
}
int main(void) {
    alarm(15);
    CHECK(strcmp(DS4_BENCH_RESIDENT_QUALIFICATION_SCHEMA,
                 "ds4.bench.resident-qualification/v1") == 0,
          "resident qualification schema macro is exact");
    ds4_runtime_wire_snapshot runtime;
    fill_runtime(&runtime);
    char runtime_json[DS4_RUNTIME_JSON_CAPACITY];
    size_t runtime_length = 0u;
    CHECK(ds4_runtime_wire_snapshot_json(
              &runtime, runtime_json, sizeof(runtime_json), &runtime_length),
          "resident fixture is valid generic runtime JSON");
    int pipe_fds[2] = {-1, -1};
    CHECK(pipe(pipe_fds) == 0, "create resident lifecycle pipe");
    if (failures != 0) {
        if (pipe_fds[0] >= 0) close(pipe_fds[0]);
        if (pipe_fds[1] >= 0) close(pipe_fds[1]);
        return finish_test();
    }
    const int reader_flags = fcntl(pipe_fds[0], F_GETFL, 0);
    CHECK(reader_flags >= 0 && fcntl(pipe_fds[0], F_SETFL,
                                     reader_flags | O_NONBLOCK) == 0,
          "make resident lifecycle reader nonblocking");
    const int writer_flags = fcntl(pipe_fds[1], F_GETFL, 0);
    CHECK(writer_flags >= 0 && fcntl(pipe_fds[1], F_SETFL,
                                     writer_flags | O_NONBLOCK) == 0,
          "make resident lifecycle writer nonblocking");
    if (failures != 0) {
        close(pipe_fds[0]);
        close(pipe_fds[1]);
        return finish_test();
    }
    FILE *writer = fdopen(pipe_fds[1], "wb");
    CHECK(writer != NULL, "open buffered resident lifecycle writer");
    if (!writer) {
        close(pipe_fds[0]);
        close(pipe_fds[1]);
        return finish_test();
    }
    char line[DS4_RUNTIME_JSON_CAPACITY * 2u];
    bool lifecycle_ok = true;
    for (uint32_t order = 0u; order < 4u && lifecycle_ok; order++) {
        ds4_bench_resident_sequence resident;
        fill_sequence(&resident, order);
        for (uint32_t repetition = 0u; repetition < 4u && lifecycle_ok;
             repetition++) {
            const char *request_id = request_ids[repetition];
            ds4_runtime_request_metrics metrics;
            fill_metrics(&metrics, request_id, runtime.snapshot_seq - 1u,
                         prompt_tokens[order]);
            for (ds4_bench_qualification_event event =
                     DS4_BENCH_QUALIFICATION_EVENT_REQUEST_ACCEPTED;
                 event <= DS4_BENCH_QUALIFICATION_EVENT_REQUEST_COMPLETE;
                 event++) {
                runtime.snapshot_seq = 100u + order * 12u + repetition * 3u +
                    (uint64_t)event + 1u;
                metrics.snapshot_seq = runtime.snapshot_seq - 1u;
                const bool runtime_valid = ds4_runtime_wire_snapshot_json(
                    &runtime, runtime_json, sizeof(runtime_json), &runtime_length);
                CHECK(runtime_valid, "each resident snapshot serializes canonically");
                if (!runtime_valid) {
                    lifecycle_ok = false;
                    break;
                }
                ds4_bench_resident_qualification_record record = {
                    .sequence = &resident,
                    .event = event,
                    .request_id = request_id,
                    .repetition_index = repetition,
                    .monotonic_ns = 1000000u + order * 1000u +
                        repetition * 10u + (uint64_t)event + 1u,
                    .session_payload_bytes = 9000u,
                    .runtime_snapshot = &runtime,
                    .request_metrics = event ==
                        DS4_BENCH_QUALIFICATION_EVENT_REQUEST_COMPLETE
                            ? &metrics : NULL,
                };
                const ds4_runtime_wire_snapshot runtime_before = runtime;
                const ds4_bench_resident_sequence sequence_before = resident;
                const ds4_bench_resident_qualification_record record_before = record;
                char metrics_json[DS4_RUNTIME_REQUEST_JSON_CAPACITY];
                size_t metrics_length = 0u;
                const char *metrics_text = NULL;
                if (record.request_metrics) {
                    const bool metrics_valid = ds4_runtime_request_metrics_json(
                        &metrics, metrics_json, sizeof(metrics_json), &metrics_length);
                    CHECK(metrics_valid, "completion metrics serialize canonically");
                    if (!metrics_valid) {
                        lifecycle_ok = false;
                        break;
                    }
                    metrics_text = metrics_json;
                }
                char error[256] = {0};
                const bool emitted = ds4_bench_resident_qualification_emit_record(
                    writer, &record, error, sizeof(error));
                CHECK(emitted, error[0] ? error : "resident emitter accepts fixture");
                if (!emitted) {
                    lifecycle_ok = false;
                    break;
                }
                successful_emissions++;
                const bool line_read = read_line(pipe_fds[0], line, sizeof(line));
                CHECK(line_read, "resident emitter flushes one record before close");
                if (!line_read) {
                    lifecycle_ok = false;
                    break;
                }
                check_success_line(line, runtime_json, metrics_text, order,
                                   repetition, event);
                CHECK(memcmp(&runtime, &runtime_before, sizeof(runtime)) == 0,
                      "borrowed runtime snapshot is unchanged");
                CHECK(memcmp(&resident, &sequence_before, sizeof(resident)) == 0,
                      "borrowed resident sequence is unchanged");
                CHECK(memcmp(resident.sequence.input_bytes, "resident fixture",
                             resident.sequence.input_size) == 0,
                      "borrowed input bytes are unchanged");
                CHECK(memcmp(&record, &record_before, sizeof(record)) == 0,
                      "borrowed resident record is unchanged");
                if (failures != 0) {
                    lifecycle_ok = false;
                    break;
                }
            }
        }
    }
    CHECK(fclose(writer) == 0, "close resident lifecycle writer");
    close(pipe_fds[0]);
    if (!lifecycle_ok) return finish_test();
    ds4_bench_resident_sequence resident;
    fill_sequence(&resident, 0u);
    ds4_runtime_request_metrics metrics;
    fill_metrics(&metrics, request_ids[0], 199u, prompt_tokens[0]);
    runtime.snapshot_seq = 200u;
    metrics.snapshot_seq = 199u;
    ds4_bench_resident_qualification_record record = {
        .sequence = &resident,
        .event = DS4_BENCH_QUALIFICATION_EVENT_REQUEST_COMPLETE,
        .request_id = request_ids[0], .repetition_index = 0u,
        .monotonic_ns = 2000000u, .session_payload_bytes = 9000u,
        .runtime_snapshot = &runtime, .request_metrics = &metrics,
    };
    char null_error[256] = {0};
    CHECK(!ds4_bench_resident_qualification_emit_record(
              NULL, &record, null_error, sizeof(null_error)),
          "null resident stream is rejected");
    expect_resident_reject("null resident record is rejected", NULL);
    ds4_bench_resident_qualification_record bad;
    ds4_bench_resident_sequence bad_sequence;
    ds4_runtime_wire_snapshot bad_runtime;
    ds4_runtime_request_metrics bad_metrics;
    bad = record;
    bad.request_id = "not-a-uuid";
    expect_resident_reject("bad request UUID is rejected", &bad);
    bad = record;
    bad_sequence = resident;
    bad_sequence.sequence.manifest_sha256[0] = 'A';
    bad.sequence = &bad_sequence;
    expect_resident_reject("bad manifest hash is rejected", &bad);
    bad = record;
    bad_sequence = resident;
    bad_sequence.sequence.sequence_sha256[0] = 'A';
    bad.sequence = &bad_sequence;
    expect_resident_reject("bad sequence hash is rejected", &bad);
    bad = record;
    bad_sequence = resident;
    bad_sequence.sequence.input_sha256[0] = 'A';
    bad.sequence = &bad_sequence;
    expect_resident_reject("bad input hash is rejected", &bad);
    bad = record;
    bad.sequence = NULL;
    expect_resident_reject("null sequence pointer is rejected", &bad);
    bad = record;
    bad.runtime_snapshot = NULL;
    expect_resident_reject("null runtime snapshot is rejected", &bad);
    bad = record;
    bad.event = DS4_BENCH_QUALIFICATION_EVENT_FIRST_TOKEN;
    bad.request_metrics = &metrics;
    expect_resident_reject("pre-terminal metrics are rejected", &bad);
    bad = record;
    bad.request_metrics = NULL;
    expect_resident_reject("completion without metrics is rejected", &bad);
    bad = record;
    bad_sequence = resident;
    bad_sequence.sequence.prompt_order_index = 1u;
    bad.sequence = &bad_sequence;
    expect_resident_reject("noncanonical prompt order is rejected", &bad);
    bad = record;
    bad_sequence = resident;
    bad_sequence.sequence.prompt_order_index = 4u;
    bad.sequence = &bad_sequence;
    expect_resident_reject("out-of-range prompt order is rejected", &bad);
    bad = record;
    bad_runtime = runtime;
    bad_runtime.configured_context_tokens = 16384u;
    bad_runtime.effective_context_tokens = 16384u;
    bad.runtime_snapshot = &bad_runtime;
    expect_resident_reject("unqualified 16K context geometry is rejected", &bad);
    bad = record;
    bad_runtime = runtime;
    bad_runtime.configured_prefill_rows = 16384u;
    bad.runtime_snapshot = &bad_runtime;
    expect_resident_reject("resident configured prefill rows are pinned", &bad);
    bad = record;
    bad_runtime = runtime;
    bad_runtime.allocated_prefill_rows = 16384u;
    bad.runtime_snapshot = &bad_runtime;
    expect_resident_reject("resident allocated prefill rows are pinned", &bad);
    bad = record;
    bad_runtime = runtime;
    bad_runtime.allocations.category_current[
        DS4_RUNTIME_CATEGORY_EXPERT_CACHE_PAYLOAD] = 1u;
    bad_runtime.allocations.category_peak[
        DS4_RUNTIME_CATEGORY_EXPERT_CACHE_PAYLOAD] = 1u;
    bad_runtime.allocations.owned_total_current++;
    bad_runtime.allocations.owned_total_peak++;
    bad_runtime.allocations.qualification_total_current++;
    bad_runtime.allocations.qualification_total_peak++;
    bad.runtime_snapshot = &bad_runtime;
    expect_resident_reject("nonzero expert cache current is rejected", &bad);
    bad = record;
    bad_runtime = runtime;
    bad_runtime.allocations.category_peak[
        DS4_RUNTIME_CATEGORY_EXPERT_CACHE_PAYLOAD] = 1u;
    bad_runtime.allocations.owned_total_peak++;
    bad_runtime.allocations.qualification_total_peak++;
    bad.runtime_snapshot = &bad_runtime;
    expect_resident_reject("nonzero expert cache peak is rejected", &bad);
    bad = record;
    bad_runtime = runtime;
    bad_runtime.configured_ssd_streaming = true;
    bad.runtime_snapshot = &bad_runtime;
    expect_resident_reject("resident SSD streaming is rejected", &bad);
    bad = record;
    bad_runtime = runtime;
    bad_runtime.configured_ssd_streaming_cache_bytes = 1u;
    bad.runtime_snapshot = &bad_runtime;
    expect_resident_reject("resident configured cache is zero", &bad);
    bad = record;
    bad_runtime = runtime;
    bad_runtime.expert_cache_limit_bytes = 1u;
    bad.runtime_snapshot = &bad_runtime;
    expect_resident_reject("resident effective cache is zero", &bad);
    bad = record;
    bad_sequence = resident;
    memcpy(bad_sequence.sequence.profile_id, "cache-8gib", sizeof("cache-8gib"));
    bad.sequence = &bad_sequence;
    expect_resident_reject("resident profile cannot be relabelled streamed", &bad);
    bad = record;
    bad_sequence = resident;
    bad_sequence.sequence.cache_bytes = 1u;
    bad.sequence = &bad_sequence;
    expect_resident_reject("resident sequence cache is zero", &bad);
    bad = record;
    bad_runtime = runtime;
    bad_runtime.allocations.external_sample.unrelated_process_inventory_stable = false;
    bad.runtime_snapshot = &bad_runtime;
    expect_resident_reject("unstable external inventory is rejected", &bad);
    bad = record;
    bad_metrics = metrics;
    memcpy(bad_metrics.request_id,
           "123e4567-e89b-12d3-a456-426614174009",
           sizeof(bad_metrics.request_id));
    bad.request_metrics = &bad_metrics;
    expect_resident_reject("completion request binding is rejected", &bad);
    bad = record;
    bad_metrics = metrics;
    memcpy(bad_metrics.instance_id,
           "123e4567-e89b-12d3-a456-426614174009",
           sizeof(bad_metrics.instance_id));
    bad.request_metrics = &bad_metrics;
    expect_resident_reject("completion instance binding is rejected", &bad);
    bad = record;
    bad_metrics = metrics;
    bad_metrics.snapshot_seq++;
    bad.request_metrics = &bad_metrics;
    expect_resident_reject("completion snapshot binding is rejected", &bad);
    bad = record;
    bad_metrics = metrics;
    bad_metrics.prompt_tokens = 513u;
    bad.request_metrics = &bad_metrics;
    expect_resident_reject("completion prompt binding is rejected", &bad);
    ds4_bench_qualification_record old_record = {
        .sequence = &resident.sequence,
        .event = record.event, .request_id = record.request_id,
        .repetition_index = record.repetition_index,
        .monotonic_ns = record.monotonic_ns,
        .session_payload_bytes = record.session_payload_bytes,
        .runtime_snapshot = &runtime, .request_metrics = &metrics,
    };
    expect_old_reject("old streamed emitter rejects resident payload/snapshot",
                      &old_record);
    bad = record;
    bad_sequence = resident;
    memcpy(bad_sequence.sequence.profile_id, "cache-8gib", sizeof("cache-8gib"));
    bad_sequence.sequence.cache_bytes = UINT64_C(8589934592);
    bad.sequence = &bad_sequence;
    bad_runtime = runtime;
    bad_runtime.configured_ssd_streaming = true;
    bad_runtime.configured_ssd_streaming_cache_bytes = UINT64_C(8589934592);
    bad_runtime.expert_cache_limit_bytes = UINT64_C(8589934592);
    bad.runtime_snapshot = &bad_runtime;
    expect_resident_reject("resident emitter rejects streamed config/profile/cache",
                           &bad);
    return finish_test();
}
