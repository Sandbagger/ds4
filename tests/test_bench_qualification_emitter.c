#define _POSIX_C_SOURCE 200809L

/* RED-only next-slice seam.  The production emitter/header intentionally does
 * not exist yet; keeping this include unresolved makes the missing behavior a
 * direct, bounded build failure instead of weakening the test with a fake
 * implementation. */
#include "ds4_bench_qualification.h"
#include "ds4_bench_sequence.h"
#include "ds4_runtime.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <poll.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static int failures;

#define CHECK(condition, message) do {                                      \
    if (!(condition)) {                                                     \
        fprintf(stderr, "FAIL: %s (line %d)\n", (message), __LINE__);    \
        failures++;                                                         \
    }                                                                        \
} while (0)

static uint64_t monotonic_ms(void) {
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) return 0;
    return (uint64_t)now.tv_sec * 1000u + (uint64_t)now.tv_nsec / 1000000u;
}

static bool wait_for_line(int fd, char *line, size_t capacity,
                          uint32_t timeout_ms) {
    if (fd < 0 || !line || capacity < 2u) return false;
    const uint64_t deadline = monotonic_ms() + timeout_ms;
    size_t length = 0;
    for (;;) {
        const uint64_t now = monotonic_ms();
        if (now >= deadline) return false;
        const uint64_t remaining = deadline - now;
        struct pollfd descriptor = {
            .fd = fd,
            .events = POLLIN | POLLHUP,
        };
        int polled;
        do {
            polled = poll(&descriptor, 1, (int)remaining);
        } while (polled < 0 && errno == EINTR);
        if (polled <= 0) return false;
        unsigned char byte;
        const ssize_t count = read(fd, &byte, 1u);
        if (count < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) {
            continue;
        }
        if (count <= 0) return false;
        if (length + 1u >= capacity) return false;
        line[length++] = (char)byte;
        if (byte == '\n') {
            line[length] = '\0';
            return true;
        }
    }
}

static bool wait_for_eof(int fd, uint32_t timeout_ms) {
    const uint64_t deadline = monotonic_ms() + timeout_ms;
    unsigned char byte;
    for (;;) {
        const uint64_t now = monotonic_ms();
        if (now >= deadline) return false;
        struct pollfd descriptor = {
            .fd = fd,
            .events = POLLIN | POLLHUP,
        };
        int polled;
        do {
            polled = poll(&descriptor, 1,
                          (int)(deadline - now));
        } while (polled < 0 && errno == EINTR);
        if (polled <= 0) return false;
        const ssize_t count = read(fd, &byte, 1u);
        if (count == 0) return true;
        if (count < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) {
            continue;
        }
        if (count < 0) return false;
        /* There must be exactly the twelve records consumed above. */
        return false;
    }
}

static void fill_runtime(ds4_runtime_wire_snapshot *snapshot) {
    memset(snapshot, 0, sizeof(*snapshot));
    memcpy(snapshot->instance_id,
           "123e4567-e89b-12d3-a456-426614174000",
           sizeof(snapshot->instance_id));
    snapshot->snapshot_seq = 42u;
    snapshot->state = DS4_RUNTIME_WIRE_STATE_READY;
    memcpy(snapshot->build.revision,
           "1111111111111111111111111111111111111111",
           sizeof(snapshot->build.revision));
    memcpy(snapshot->build.backend, "cuda", sizeof("cuda"));
    memcpy(snapshot->build.features[0], "laguna", sizeof("laguna"));
    memcpy(snapshot->build.features[1], "ssd_streaming", sizeof("ssd_streaming"));
    snapshot->build.feature_count = 2u;
    snapshot->executable = (ds4_runtime_file_identity){
        .device = 1u, .inode = 2u, .size_bytes = 3u, .mtime_ns = 4u,
    };
    snapshot->model = (ds4_runtime_file_identity){
        .device = 5u, .inode = 6u, .size_bytes = 7u, .mtime_ns = 8u,
    };
    memcpy(snapshot->model_id, "laguna-s-2.1", sizeof("laguna-s-2.1"));
    memcpy(snapshot->model_family, "laguna", sizeof("laguna"));
    snapshot->configured_context_tokens = 32768u;
    snapshot->configured_prefill_chunk_tokens = 4096u;
    snapshot->configured_session_slots = 1u;
    snapshot->configured_ssd_streaming = true;
    snapshot->configured_ssd_streaming_cache_bytes = UINT64_C(8589934592);
    snapshot->effective_context_tokens = 32768u;
    snapshot->effective_prefill_chunk_tokens = 4096u;
    snapshot->effective_session_slots = 1u;
    snapshot->expert_cache_limit_bytes = UINT64_C(8589934592);
    snapshot->configured_prefill_rows = 4096u;
    snapshot->allocated_prefill_rows = 4096u;
    snapshot->allocations.category_current[DS4_RUNTIME_CATEGORY_EXPERT_CACHE_PAYLOAD] = 1000u;
    snapshot->allocations.category_peak[DS4_RUNTIME_CATEGORY_EXPERT_CACHE_PAYLOAD] = 2000u;
    snapshot->allocations.category_bounds[DS4_RUNTIME_CATEGORY_EXPERT_CACHE_PAYLOAD] = UINT64_C(8589934592);
    snapshot->allocations.category_current[DS4_RUNTIME_CATEGORY_KV_STATE] = 0u;
    snapshot->allocations.category_peak[DS4_RUNTIME_CATEGORY_KV_STATE] = 0u;
    snapshot->allocations.category_bounds[DS4_RUNTIME_CATEGORY_KV_STATE] = UINT64_C(4096);
    snapshot->allocations.qualification_total_current = 1300u;
    snapshot->allocations.qualification_total_bound_bytes = UINT64_C(10000000000);
    snapshot->allocations.qualification_total_peak = 2300u;
    snapshot->allocations.report_current[DS4_RUNTIME_REPORT_MODEL_SOURCE_RESIDENT] = 5000u;
    snapshot->allocations.report_current[DS4_RUNTIME_REPORT_HOST_LIBRARY_UNATTRIBUTED] = 100u;
    snapshot->allocations.report_current[DS4_RUNTIME_REPORT_CUDA_LIBRARY_UNATTRIBUTED] = 200u;
    snapshot->allocations.external_sample.host_library_unattributed_bytes = 100u;
    snapshot->allocations.external_sample.cuda_library_unattributed_bytes = 200u;
    snapshot->allocations.external_sample.unrelated_process_inventory_stable = true;
    for (size_t i = 0; i < 11u; i++) {
        ((uint64_t *)&snapshot->counters)[i] = i + 1u;
    }
}

static void fill_metrics(ds4_runtime_request_metrics *metrics) {
    memset(metrics, 0, sizeof(*metrics));
    memcpy(metrics->request_id,
           "123e4567-e89b-12d3-a456-426614174001",
           sizeof(metrics->request_id));
    memcpy(metrics->instance_id,
           "123e4567-e89b-12d3-a456-426614174000",
           sizeof(metrics->instance_id));
    metrics->snapshot_seq = 42u;
    metrics->prompt_tokens = 512u;
    metrics->generated_tokens = 1u;
    metrics->ttft_present = true;
    metrics->ttft_ns = 100u;
    metrics->prefill_tokens_per_second = 512.0;
    metrics->visible_decode_tokens_per_second = 1.0;
    metrics->wall_time_ns = 200u;
    metrics->terminal_status = DS4_RUNTIME_REQUEST_COMPLETED;
}

static bool has_nonfinite_text(const char *line) {
    return strstr(line, "NaN") != NULL || strstr(line, "Infinity") != NULL ||
           strstr(line, "nan") != NULL || strstr(line, "inf") != NULL;
}

int main(void) {
    ds4_runtime_wire_snapshot runtime;
    ds4_runtime_request_metrics metrics;
    fill_runtime(&runtime);
    fill_metrics(&metrics);

    unsigned char input_bytes[] = "task19 canonical prompt bytes";
    ds4_bench_sequence sequence = {
        .manifest_sha256 =
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        .profile_id = "cache-8gib",
        .cache_bytes = UINT64_C(8589934592),
        .prompt_order_index = 0u,
        .prompt_id = "native-512",
        .prompt_tokens = 512u,
        .input_size_bytes = sizeof(input_bytes) - 1u,
        .input_size = sizeof(input_bytes) - 1u,
        .input_bytes = input_bytes,
        .input_sha256 =
            "f64f809b1c85b78d9e73cc74fc2c9c45bf8130640585b7d9d8663b7b043e5b15",
    };
    (void)sequence;

    int pipe_fds[2] = {-1, -1};
    CHECK(pipe(pipe_fds) == 0, "create lifecycle JSONL pipe");
    if (failures != 0) return 1;
    const int flags = fcntl(pipe_fds[0], F_GETFL, 0);
    CHECK(flags >= 0 && fcntl(pipe_fds[0], F_SETFL, flags | O_NONBLOCK) == 0,
          "make lifecycle reader nonblocking");
    FILE *writer = fdopen(pipe_fds[1], "wb");
    CHECK(writer != NULL, "open buffered lifecycle writer");
    if (!writer) {
        close(pipe_fds[0]);
        close(pipe_fds[1]);
        return 1;
    }

    static const ds4_bench_qualification_event events[] = {
        DS4_BENCH_QUALIFICATION_EVENT_REQUEST_ACCEPTED,
        DS4_BENCH_QUALIFICATION_EVENT_FIRST_TOKEN,
        DS4_BENCH_QUALIFICATION_EVENT_REQUEST_COMPLETE,
    };
    static const char *const request_ids[] = {
        "123e4567-e89b-12d3-a456-426614174001",
        "123e4567-e89b-12d3-a456-426614174002",
        "123e4567-e89b-12d3-a456-426614174003",
        "123e4567-e89b-12d3-a456-426614174004",
    };
    char runtime_json[DS4_RUNTIME_JSON_CAPACITY];
    char metrics_json[DS4_RUNTIME_REQUEST_JSON_CAPACITY];
    char line[DS4_RUNTIME_JSON_CAPACITY * 2u];
    for (uint32_t repetition = 0u; repetition < 4u; repetition++) {
        memcpy(metrics.request_id, request_ids[repetition],
               sizeof(metrics.request_id));
        for (size_t event_index = 0u; event_index < 3u; event_index++) {
            runtime.snapshot_seq =
                UINT64_C(100) + repetition * 3u + event_index;
            metrics.snapshot_seq = runtime.snapshot_seq;
            size_t runtime_length = 0u;
            size_t metrics_length = 0u;
            CHECK(ds4_runtime_wire_snapshot_json(
                      &runtime, runtime_json, sizeof(runtime_json), &runtime_length),
                  "synthetic runtime snapshot serializes canonically");
            if (event_index == 2u) {
                CHECK(ds4_runtime_request_metrics_json(
                          &metrics, metrics_json, sizeof(metrics_json), &metrics_length),
                      "finished request metrics serialize canonically");
            } else {
                metrics_json[0] = '\0';
            }
            CHECK(runtime_length == strlen(runtime_json),
                  "runtime JSON length is exact");
            CHECK(event_index != 2u || metrics_length == strlen(metrics_json),
                  "request JSON length is exact");
            if (failures != 0) break;

            ds4_bench_qualification_record record = {
                .sequence = &sequence,
                .event = events[event_index],
                .request_id = request_ids[repetition],
                .repetition_index = repetition,
                .monotonic_ns =
                    UINT64_C(1000000) + repetition * 100u + event_index,
                .session_payload_bytes = 9000u,
                .runtime_snapshot = &runtime,
                .request_metrics = event_index == 2u ? &metrics : NULL,
            };
            char error[256] = {0};
            CHECK(ds4_bench_qualification_emit_record(
                      writer, &record, error, sizeof(error)),
                  error[0] ? error : "qualification emitter accepted record");
            if (failures != 0) break;
            /* This read occurs while writer is still alive. It is the flush
             * gate: a buffered emitter that omits fflush cannot pass. */
            CHECK(wait_for_line(pipe_fds[0], line, sizeof(line), 1000u),
                  "reader observes one complete JSONL record before writer closes");
            if (failures != 0) break;
            CHECK(line[strlen(line) - 1u] == '\n' &&
                      strlen(line) >= 3u && line[strlen(line) - 2u] == '}',
                  "lifecycle record is one closed JSON object line");
            CHECK(!has_nonfinite_text(line), "lifecycle record has no non-finite values");
            CHECK(strstr(line, runtime_json) != NULL,
                  "lifecycle record reuses canonical nested runtime JSON");
            if (event_index == 2u) {
                CHECK(strstr(line, metrics_json) != NULL,
                      "request_complete reuses canonical request metrics JSON");
                CHECK(strstr(line, "\"terminal_status\"") != NULL,
                      "request_complete carries terminal_status");
            } else {
                CHECK(strstr(line, metrics_json) == NULL,
                      "pre-terminal lifecycle records omit request metrics");
                CHECK(strstr(line, "\"terminal_status\"") == NULL,
                      "pre-terminal lifecycle records omit terminal_status");
            }
            fputs(line, stdout);
        }
        if (failures != 0) break;
    }
    CHECK(fclose(writer) == 0, "close lifecycle writer after records");
    CHECK(wait_for_eof(pipe_fds[0], 1000u),
          "reader observes EOF after exactly twelve flushed records");
    close(pipe_fds[0]);
    return failures == 0 ? 0 : 1;
}
