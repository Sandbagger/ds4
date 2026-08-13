#define _DARWIN_C_SOURCE
#define _POSIX_C_SOURCE 200809L

/* Let this RED-only executable observe absent production symbols at runtime.
 * Rename the normal header declarations so weak_import is the first real
 * declaration on Mach-O; otherwise ld treats the earlier declaration as a
 * required undefined symbol. */
#define ds4_qualification_control_open \
    ds4_qualification_control_open_header_declaration
#define ds4_qualification_control_send_model_fd \
    ds4_qualification_control_send_model_fd_header_declaration
#define ds4_qualification_control_begin_sample \
    ds4_qualification_control_begin_sample_header_declaration
#define ds4_qualification_control_finish_sample \
    ds4_qualification_control_finish_sample_header_declaration
#define ds4_qualification_control_close \
    ds4_qualification_control_close_header_declaration
#include "ds4.h"
#include "ds4_runtime.h"
#undef ds4_qualification_control_open
#undef ds4_qualification_control_send_model_fd
#undef ds4_qualification_control_begin_sample
#undef ds4_qualification_control_finish_sample
#undef ds4_qualification_control_close

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#if defined(__APPLE__)
#define DS4_TEST_WEAK __attribute__((weak_import))
#elif defined(__GNUC__) || defined(__clang__)
#define DS4_TEST_WEAK __attribute__((weak))
#else
#define DS4_TEST_WEAK
#endif

/* Keep this RED executable linkable before the production implementation
 * exists.  Once any symbol is absent, main reports a single explicit RED. */
extern int ds4_qualification_control_open(
    ds4_qualification_control **out,
    int inherited_fd,
    uint32_t timeout_ms,
    char *err,
    size_t errcap) DS4_TEST_WEAK;
extern int ds4_qualification_control_send_model_fd(
    ds4_qualification_control *control,
    int model_fd,
    const ds4_runtime_file_identity *expected_identity,
    char *err,
    size_t errcap) DS4_TEST_WEAK;
extern int ds4_qualification_control_begin_sample(
    ds4_qualification_control *control,
    uint64_t checkpoint_sequence,
    char *err,
    size_t errcap) DS4_TEST_WEAK;
extern int ds4_qualification_control_finish_sample(
    ds4_qualification_control *control,
    uint64_t checkpoint_sequence,
    int model_fd,
    char *err,
    size_t errcap) DS4_TEST_WEAK;
extern void ds4_qualification_control_close(
    ds4_qualification_control *control) DS4_TEST_WEAK;

typedef char ds4_qualification_control_message_size_is_stable[
    sizeof(ds4_qualification_control_message) == 56 ? 1 : -1];

static int failures;

#define CHECK(condition, ...)                                                \
    do {                                                                     \
        if (!(condition)) {                                                  \
            fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);            \
            fprintf(stderr, __VA_ARGS__);                                    \
            fputc('\n', stderr);                                             \
            failures++;                                                      \
        }                                                                    \
    } while (0)

static uint64_t monotonic_ms(void) {
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) return 0;
    return (uint64_t)now.tv_sec * 1000u + (uint64_t)now.tv_nsec / 1000000u;
}

static void sleep_ms(uint32_t milliseconds) {
    struct timespec delay = {
        .tv_sec = (time_t)(milliseconds / 1000u),
        .tv_nsec = (long)(milliseconds % 1000u) * 1000000L,
    };
    while (nanosleep(&delay, &delay) != 0 && errno == EINTR) {}
}

static bool identity_equal(
        const ds4_runtime_file_identity *a,
        const ds4_runtime_file_identity *b) {
    return a && b &&
        a->device == b->device &&
        a->inode == b->inode &&
        a->size_bytes == b->size_bytes &&
        a->mtime_ns == b->mtime_ns;
}

static bool identity_zero(const ds4_runtime_file_identity *identity) {
    static const ds4_runtime_file_identity zero;
    return identity_equal(identity, &zero);
}

static bool identity_from_fd(
        int fd, ds4_runtime_file_identity *identity) {
    struct stat status;
    if (!identity || fstat(fd, &status) != 0 || status.st_size < 0) {
        return false;
    }
    identity->device = (uint64_t)status.st_dev;
    identity->inode = (uint64_t)status.st_ino;
    identity->size_bytes = (uint64_t)status.st_size;
#if defined(__APPLE__)
    identity->mtime_ns =
        (uint64_t)status.st_mtimespec.tv_sec * 1000000000u +
        (uint64_t)status.st_mtimespec.tv_nsec;
#else
    identity->mtime_ns =
        (uint64_t)status.st_mtim.tv_sec * 1000000000u +
        (uint64_t)status.st_mtim.tv_nsec;
#endif
    return true;
}

static int make_model_file(ds4_runtime_file_identity *identity) {
    char path[] = "/tmp/ds4-qualification-control-XXXXXX";
    static const char payload[] = "opened model descriptor identity\n";
    int fd = mkstemp(path);
    if (fd < 0) return -1;
    (void)unlink(path);
    if (fcntl(fd, F_SETFD, FD_CLOEXEC) != 0 ||
        write(fd, payload, sizeof(payload) - 1u) !=
            (ssize_t)(sizeof(payload) - 1u) ||
        lseek(fd, 0, SEEK_SET) != 0 ||
        !identity_from_fd(fd, identity)) {
        (void)close(fd);
        return -1;
    }
    return fd;
}

static bool numeric_name(const char *name) {
    if (!name || !name[0]) return false;
    for (const unsigned char *p = (const unsigned char *)name; *p; p++) {
        if (*p < '0' || *p > '9') return false;
    }
    return true;
}

static size_t open_fd_list(int *fds, size_t capacity) {
    const char *directory = access("/proc/self/fd", R_OK) == 0
        ? "/proc/self/fd" : "/dev/fd";
    DIR *dir = opendir(directory);
    if (!dir) return SIZE_MAX;
    const int scan_fd = dirfd(dir);
    size_t count = 0;
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        if (!numeric_name(entry->d_name)) continue;
        char *end = NULL;
        long value = strtol(entry->d_name, &end, 10);
        if (!end || *end != '\0' || value < 0 || value > INT32_MAX ||
            (int)value == scan_fd) {
            continue;
        }
        if (fds && count < capacity) fds[count] = (int)value;
        count++;
    }
    (void)closedir(dir);
    return count;
}

static bool fd_in_list(int fd, const int *fds, size_t count) {
    for (size_t i = 0; i < count; i++) {
        if (fds[i] == fd) return true;
    }
    return false;
}

static int wait_readable(int fd, int timeout_ms) {
    struct pollfd poll_fd = {
        .fd = fd,
        .events = POLLIN | POLLHUP,
    };
    int result;
    do {
        result = poll(&poll_fd, 1, timeout_ms);
    } while (result < 0 && errno == EINTR);
    return result > 0 ? poll_fd.revents : result;
}

static bool send_all(int fd, const void *data, size_t size) {
    const uint8_t *cursor = data;
    while (size != 0) {
#ifdef MSG_NOSIGNAL
        ssize_t written = send(fd, cursor, size, MSG_NOSIGNAL);
#else
        ssize_t written = send(fd, cursor, size, 0);
#endif
        if (written < 0 && errno == EINTR) continue;
        if (written <= 0) return false;
        cursor += (size_t)written;
        size -= (size_t)written;
    }
    return true;
}

static bool send_control_message(
        int fd,
        uint32_t type,
        uint64_t sequence) {
    const ds4_qualification_control_message message = {
        .protocol_version = DS4_QUALIFICATION_CONTROL_PROTOCOL_VERSION,
        .message_type = type,
        .message_size = sizeof(message),
        .checkpoint_sequence = sequence,
    };
    return send_all(fd, &message, sizeof(message));
}

static bool receive_control_message(
        int fd,
        ds4_qualification_control_message *message,
        int *received_fds,
        size_t received_capacity,
        size_t *received_count) {
    if (!message || !received_count || wait_readable(fd, 1000) <= 0) {
        return false;
    }
    *received_count = 0;
    memset(message, 0, sizeof(*message));
    char ancillary[CMSG_SPACE(sizeof(int) * 4u)];
    memset(ancillary, 0, sizeof(ancillary));
    struct iovec iov = {
        .iov_base = message,
        .iov_len = sizeof(*message),
    };
    struct msghdr header;
    memset(&header, 0, sizeof(header));
    header.msg_iov = &iov;
    header.msg_iovlen = 1;
    header.msg_control = ancillary;
    header.msg_controllen = sizeof(ancillary);

    ssize_t received;
    do {
        received = recvmsg(fd, &header, 0);
    } while (received < 0 && errno == EINTR);
    if (received <= 0 || (header.msg_flags & MSG_CTRUNC)) return false;

    for (struct cmsghdr *item = CMSG_FIRSTHDR(&header);
         item != NULL;
         item = CMSG_NXTHDR(&header, item)) {
        if (item->cmsg_level != SOL_SOCKET ||
            item->cmsg_type != SCM_RIGHTS ||
            item->cmsg_len < CMSG_LEN(0)) {
            return false;
        }
        const size_t data_bytes = item->cmsg_len - CMSG_LEN(0);
        if (data_bytes == 0 || data_bytes % sizeof(int) != 0) return false;
        const size_t count = data_bytes / sizeof(int);
        const int *rights = (const int *)CMSG_DATA(item);
        for (size_t i = 0; i < count; i++) {
            if (*received_count >= received_capacity) {
                (void)close(rights[i]);
                return false;
            }
            received_fds[(*received_count)++] = rights[i];
        }
    }

    size_t offset = (size_t)received;
    while (offset < sizeof(*message)) {
        ssize_t part = recv(fd, (uint8_t *)message + offset,
                            sizeof(*message) - offset, 0);
        if (part < 0 && errno == EINTR) continue;
        if (part <= 0) return false;
        offset += (size_t)part;
    }
    return offset == sizeof(*message);
}

static bool valid_message(
        const ds4_qualification_control_message *message,
        uint32_t expected_type,
        uint64_t expected_sequence) {
    return message &&
        message->protocol_version ==
            DS4_QUALIFICATION_CONTROL_PROTOCOL_VERSION &&
        message->message_type == expected_type &&
        message->message_size == sizeof(*message) &&
        message->reserved == 0 &&
        message->checkpoint_sequence == expected_sequence;
}

typedef struct {
    int pair[2];
    ds4_qualification_control *control;
    int owned_fd;
    size_t baseline_count;
} control_fixture;

static bool fixture_open(control_fixture *fixture, uint32_t timeout_ms) {
    int before[256];
    int after[256];
    memset(fixture, 0, sizeof(*fixture));
    fixture->pair[0] = -1;
    fixture->pair[1] = -1;
    fixture->owned_fd = -1;
    fixture->baseline_count = open_fd_list(NULL, 0);
    if (fixture->baseline_count == SIZE_MAX ||
        socketpair(AF_UNIX, SOCK_STREAM, 0, fixture->pair) != 0) {
        return false;
    }
    for (size_t i = 0; i < 2; i++) {
        if (fcntl(fixture->pair[i], F_SETFD, FD_CLOEXEC) != 0) return false;
    }
    const size_t before_count = open_fd_list(before, 256);
    if (before_count == SIZE_MAX || before_count > 256) return false;

    char error[256] = {0};
    if (ds4_qualification_control_open(
            &fixture->control, fixture->pair[0], timeout_ms,
            error, sizeof(error)) != 0 || !fixture->control) {
        fprintf(stderr, "qualification control open failed: %s\n", error);
        return false;
    }
    const size_t after_count = open_fd_list(after, 256);
    if (after_count != before_count + 1u || after_count > 256) return false;
    size_t added = 0;
    for (size_t i = 0; i < after_count; i++) {
        if (!fd_in_list(after[i], before, before_count)) {
            fixture->owned_fd = after[i];
            added++;
        }
    }
    if (added != 1u || fixture->owned_fd < 0) return false;
    const int flags = fcntl(fixture->owned_fd, F_GETFD);
    return flags >= 0 && (flags & FD_CLOEXEC) != 0;
}

static bool fixture_close(control_fixture *fixture) {
    if (fixture->control) {
        ds4_qualification_control_close(fixture->control);
        fixture->control = NULL;
    }
    if (fixture->pair[0] >= 0) {
        (void)close(fixture->pair[0]);
        fixture->pair[0] = -1;
    }
    if (fixture->pair[1] >= 0) {
        (void)close(fixture->pair[1]);
        fixture->pair[1] = -1;
    }
    const size_t after = open_fd_list(NULL, 0);
    return after != SIZE_MAX && after == fixture->baseline_count;
}

typedef enum {
    WORKER_BEGIN_ONLY = 0,
    WORKER_FULL_SAMPLE = 1,
} worker_kind;

typedef struct {
    pthread_mutex_t mutex;
    ds4_qualification_control *control;
    int model_fd;
    uint64_t sequence;
    worker_kind kind;
    int stage;
    int result;
    bool done;
    uint64_t elapsed_ms;
    char error[256];
} sample_worker;

static void worker_set_stage(sample_worker *worker, int stage) {
    (void)pthread_mutex_lock(&worker->mutex);
    worker->stage = stage;
    (void)pthread_mutex_unlock(&worker->mutex);
}

static int worker_stage(sample_worker *worker, bool *done) {
    (void)pthread_mutex_lock(&worker->mutex);
    const int stage = worker->stage;
    if (done) *done = worker->done;
    (void)pthread_mutex_unlock(&worker->mutex);
    return stage;
}

static void *sample_worker_main(void *argument) {
    sample_worker *worker = argument;
    const uint64_t started = monotonic_ms();
    worker_set_stage(worker, 1);
    int result = ds4_qualification_control_begin_sample(
        worker->control, worker->sequence,
        worker->error, sizeof(worker->error));
    if (result == 0 && worker->kind == WORKER_FULL_SAMPLE) {
        worker_set_stage(worker, 2);
        result = ds4_qualification_control_finish_sample(
            worker->control, worker->sequence, worker->model_fd,
            worker->error, sizeof(worker->error));
    }
    (void)pthread_mutex_lock(&worker->mutex);
    worker->stage = 3;
    worker->result = result;
    worker->done = true;
    worker->elapsed_ms = monotonic_ms() - started;
    (void)pthread_mutex_unlock(&worker->mutex);
    return NULL;
}

static bool worker_init(
        sample_worker *worker,
        ds4_qualification_control *control,
        int model_fd,
        uint64_t sequence,
        worker_kind kind) {
    memset(worker, 0, sizeof(*worker));
    worker->control = control;
    worker->model_fd = model_fd;
    worker->sequence = sequence;
    worker->kind = kind;
    worker->result = -1;
    return pthread_mutex_init(&worker->mutex, NULL) == 0;
}

static bool wait_worker_done(sample_worker *worker, uint32_t timeout_ms) {
    const uint64_t deadline = monotonic_ms() + timeout_ms;
    bool done = false;
    while (monotonic_ms() <= deadline) {
        (void)worker_stage(worker, &done);
        if (done) return true;
        sleep_ms(1);
    }
    return false;
}

static bool run_successful_sample(
        control_fixture *fixture,
        int model_fd,
        const ds4_runtime_file_identity *identity,
        uint64_t sequence) {
    sample_worker worker;
    pthread_t thread;
    bool started = false;
    bool joined = false;
    bool ok = worker_init(
        &worker, fixture->control, model_fd,
        sequence, WORKER_FULL_SAMPLE);
    if (!ok) return false;
    if (pthread_create(&thread, NULL, sample_worker_main, &worker) != 0) {
        (void)pthread_mutex_destroy(&worker.mutex);
        return false;
    }
    started = true;

    ds4_qualification_control_message message;
    int received_fds[4];
    size_t received_count = 0;
    ok = receive_control_message(
             fixture->pair[1], &message,
             received_fds, 4, &received_count) &&
         valid_message(
             &message, DS4_QUALIFICATION_CONTROL_SAMPLE_READY, sequence) &&
         received_count == 0 && identity_zero(&message.model_identity) &&
         worker_stage(&worker, NULL) == 1;
    if (ok) {
        ok = send_control_message(
            fixture->pair[1],
            DS4_QUALIFICATION_CONTROL_SAMPLE_READY_ACK, sequence);
    }
    if (ok) {
        ok = receive_control_message(
                 fixture->pair[1], &message,
                 received_fds, 4, &received_count) &&
             valid_message(
                 &message, DS4_QUALIFICATION_CONTROL_SAMPLE_RESULT,
                 sequence) &&
             received_count == 0 &&
             identity_equal(&message.model_identity, identity) &&
             worker_stage(&worker, NULL) == 2;
    }
    if (ok) {
        ok = send_control_message(
            fixture->pair[1],
            DS4_QUALIFICATION_CONTROL_SAMPLE_RESULT_ACK, sequence);
    }
    if (!ok && fixture->pair[1] >= 0) {
        (void)close(fixture->pair[1]);
        fixture->pair[1] = -1;
    }
    if (started) {
        (void)pthread_join(thread, NULL);
        joined = true;
    }
    ok = ok && joined && worker.result == 0 &&
         worker_stage(&worker, NULL) == 3;
    (void)pthread_mutex_destroy(&worker.mutex);
    return ok;
}

static void test_model_fd_and_blocking_sequence(void) {
    const size_t initial_fds = open_fd_list(NULL, 0);
    ds4_runtime_file_identity model_identity;
    int model_fd = make_model_file(&model_identity);
    CHECK(model_fd >= 0, "create retained model descriptor");
    if (model_fd < 0) return;

    control_fixture fixture;
    bool opened = fixture_open(&fixture, 250);
    CHECK(opened, "open a CLOEXEC duplicate of the inherited Unix socket");
    if (!opened) {
        (void)fixture_close(&fixture);
        (void)close(model_fd);
        return;
    }

    char error[256] = {0};
    int result = ds4_qualification_control_send_model_fd(
        fixture.control, model_fd, &model_identity, error, sizeof(error));
    CHECK(result == 0, "send retained model fd: %s", error);

    ds4_qualification_control_message message;
    int received_fds[4] = {-1, -1, -1, -1};
    size_t received_count = 0;
    bool received = result == 0 && receive_control_message(
        fixture.pair[1], &message, received_fds, 4, &received_count);
    CHECK(received, "receive model SCM_RIGHTS message");
    CHECK(received && valid_message(
              &message, DS4_QUALIFICATION_CONTROL_MODEL_FD, 0),
          "model message has exact protocol/type/size and sequence zero");
    CHECK(received && received_count == 1,
          "model message carries exactly one descriptor (got %zu)",
          received_count);
    CHECK(received && identity_equal(
              &message.model_identity, &model_identity),
          "model message carries the retained fstat identity");
    if (received_count == 1) {
        ds4_runtime_file_identity received_identity;
        CHECK(identity_from_fd(received_fds[0], &received_identity) &&
                  identity_equal(&received_identity, &model_identity),
              "SCM_RIGHTS descriptor fstat matches the opened model");
        CHECK(fcntl(received_fds[0], F_SETFD, FD_CLOEXEC) == 0 &&
                  (fcntl(received_fds[0], F_GETFD) & FD_CLOEXEC) != 0,
              "receiver can retain the delivered model fd CLOEXEC");
    }
    for (size_t i = 0; i < received_count; i++) {
        (void)close(received_fds[i]);
    }

    bool first = received && received_count == 1 &&
        run_successful_sample(
            &fixture, model_fd, &model_identity, 7);
    CHECK(first,
          "READY/ACK and RESULT/ACK block progress across sequence 7");
    bool second = first && run_successful_sample(
        &fixture, model_fd, &model_identity, 8);
    CHECK(second,
          "a strictly increasing checkpoint sequence completes");

    if (second) {
        sample_worker repeat;
        pthread_t thread;
        bool initialized = worker_init(
            &repeat, fixture.control, model_fd, 8, WORKER_BEGIN_ONLY);
        bool started = initialized &&
            pthread_create(&thread, NULL, sample_worker_main, &repeat) == 0;
        CHECK(started, "start repeated-sequence probe");
        if (started) {
            const bool done = wait_worker_done(&repeat, 100);
            const int readable = wait_readable(fixture.pair[1], 0);
            CHECK(done && repeat.result != 0,
                  "repeated checkpoint sequence fails before I/O");
            CHECK(readable == 0,
                  "repeated checkpoint sequence emits no wire message");
            if (!done) {
                (void)close(fixture.pair[1]);
                fixture.pair[1] = -1;
            }
            (void)pthread_join(thread, NULL);
        }
        if (initialized) (void)pthread_mutex_destroy(&repeat.mutex);
    }

    if (fixture.control) {
        ds4_qualification_control_close(fixture.control);
        fixture.control = NULL;
        CHECK(fcntl(fixture.pair[0], F_GETFD) >= 0,
              "control teardown leaves the caller-owned inherited fd open");
    }
    CHECK(fixture_close(&fixture),
          "successful transport teardown leaks no descriptors");
    (void)close(model_fd);
    CHECK(open_fd_list(NULL, 0) == initial_fds,
          "model/control lifecycle restores the process fd baseline");
}

static void test_invalid_endpoints(void) {
    const size_t baseline = open_fd_list(NULL, 0);
    char error[256] = {0};
    ds4_qualification_control *control = (void *)(uintptr_t)1;
    CHECK(ds4_qualification_control_open(
              &control, -1, 100, error, sizeof(error)) != 0 &&
              control == NULL,
          "negative inherited fd fails closed and clears the output");

    int file_fd = open("/dev/null", O_RDONLY);
    CHECK(file_fd >= 0, "open non-socket fixture");
    if (file_fd >= 0) {
        control = (void *)(uintptr_t)1;
        memset(error, 0, sizeof(error));
        CHECK(ds4_qualification_control_open(
                  &control, file_fd, 100, error, sizeof(error)) != 0 &&
                  control == NULL,
              "non-socket inherited fd fails closed");
        (void)close(file_fd);
    }

    int pair[2] = {-1, -1};
    CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, pair) == 0,
          "create zero-timeout socket fixture");
    if (pair[0] >= 0) {
        control = (void *)(uintptr_t)1;
        CHECK(ds4_qualification_control_open(
                  &control, pair[0], 0, error, sizeof(error)) != 0 &&
                  control == NULL,
              "zero timeout fails closed");
        (void)close(pair[0]);
        (void)close(pair[1]);
    }
    CHECK(open_fd_list(NULL, 0) == baseline,
          "invalid endpoint rejection leaks no descriptors");
}

static void test_wrong_ack_sequence(void) {
    control_fixture fixture;
    bool opened = fixture_open(&fixture, 100);
    CHECK(opened, "open wrong-sequence fixture");
    if (!opened) {
        (void)fixture_close(&fixture);
        return;
    }
    sample_worker worker;
    pthread_t thread;
    bool initialized = worker_init(
        &worker, fixture.control, -1, 41, WORKER_BEGIN_ONLY);
    bool started = initialized &&
        pthread_create(&thread, NULL, sample_worker_main, &worker) == 0;
    CHECK(started, "start wrong-sequence worker");
    if (started) {
        ds4_qualification_control_message message;
        int rights[1];
        size_t right_count = 0;
        bool ready = receive_control_message(
            fixture.pair[1], &message, rights, 1, &right_count);
        CHECK(ready && valid_message(
                  &message, DS4_QUALIFICATION_CONTROL_SAMPLE_READY, 41) &&
                  right_count == 0,
              "receive READY before sending the wrong ACK");
        CHECK(send_control_message(
                  fixture.pair[1],
                  DS4_QUALIFICATION_CONTROL_SAMPLE_READY_ACK, 42),
              "send wrong-sequence ACK");
        const bool done = wait_worker_done(&worker, 200);
        CHECK(done && worker.result != 0,
              "wrong ACK sequence fails the bracket safely");
        if (!done) {
            (void)close(fixture.pair[1]);
            fixture.pair[1] = -1;
        }
        (void)pthread_join(thread, NULL);
    }
    if (initialized) (void)pthread_mutex_destroy(&worker.mutex);
    CHECK(fixture_close(&fixture),
          "wrong-sequence failure tears down without fd leaks");
}

static void test_timeout(void) {
    control_fixture fixture;
    bool opened = fixture_open(&fixture, 50);
    CHECK(opened, "open timeout fixture");
    if (!opened) {
        (void)fixture_close(&fixture);
        return;
    }
    sample_worker worker;
    pthread_t thread;
    bool initialized = worker_init(
        &worker, fixture.control, -1, 53, WORKER_BEGIN_ONLY);
    bool started = initialized &&
        pthread_create(&thread, NULL, sample_worker_main, &worker) == 0;
    CHECK(started, "start timeout worker");
    if (started) {
        ds4_qualification_control_message message;
        int rights[1];
        size_t right_count = 0;
        CHECK(receive_control_message(
                  fixture.pair[1], &message, rights, 1, &right_count) &&
                  valid_message(
                      &message, DS4_QUALIFICATION_CONTROL_SAMPLE_READY, 53),
              "receive READY and deliberately withhold its ACK");
        const bool done = wait_worker_done(&worker, 300);
        CHECK(done && worker.result != 0,
              "missing ACK reaches a bounded timeout failure");
        if (!done) {
            (void)close(fixture.pair[1]);
            fixture.pair[1] = -1;
        }
        (void)pthread_join(thread, NULL);
        CHECK(worker.elapsed_ms >= 20 && worker.elapsed_ms <= 250,
              "timeout elapsed is bounded around configured 50ms (got %llu)",
              (unsigned long long)worker.elapsed_ms);
    }
    if (initialized) (void)pthread_mutex_destroy(&worker.mutex);
    CHECK(fixture_close(&fixture),
          "timeout failure tears down without fd leaks");
}

static void test_disconnect(void) {
    control_fixture fixture;
    bool opened = fixture_open(&fixture, 100);
    CHECK(opened, "open disconnect fixture");
    if (!opened) {
        (void)fixture_close(&fixture);
        return;
    }
    (void)close(fixture.pair[1]);
    fixture.pair[1] = -1;
    char error[256] = {0};
    CHECK(ds4_qualification_control_begin_sample(
              fixture.control, 61, error, sizeof(error)) != 0,
          "peer disconnect fails without terminating or blocking the process");
    CHECK(fixture_close(&fixture),
          "disconnect failure tears down without fd leaks");
}

int main(void) {
    if (!ds4_qualification_control_open ||
        !ds4_qualification_control_send_model_fd ||
        !ds4_qualification_control_begin_sample ||
        !ds4_qualification_control_finish_sample ||
        !ds4_qualification_control_close) {
        fprintf(stderr,
                "RED: production qualification-control API is unavailable\n");
        return 1;
    }

    test_invalid_endpoints();
    test_model_fd_and_blocking_sequence();
    test_wrong_ack_sequence();
    test_timeout();
    test_disconnect();
    if (failures != 0) {
        fprintf(stderr, "qualification control: %d failure(s)\n", failures);
        return 1;
    }
    printf("qualification control: all tests passed\n");
    return 0;
}
