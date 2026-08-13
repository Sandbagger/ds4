#if defined(__APPLE__)
#define _DARWIN_C_SOURCE
#else
#define _GNU_SOURCE
#endif
#define _POSIX_C_SOURCE 200809L

#include "ds4_runtime.h"

#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#if !defined(MSG_DONTWAIT)
#error "qualification control requires nonblocking socket message I/O"
#endif
#if !defined(MSG_NOSIGNAL) && !defined(SO_NOSIGPIPE)
#error "qualification control requires per-socket SIGPIPE suppression"
#endif

struct ds4_qualification_control {
    int fd;
    uint32_t timeout_ms;
    uint64_t last_finished_sequence;
    uint64_t active_sequence;
    ds4_runtime_file_identity model_identity;
    bool model_identity_set;
    bool sample_active;
    bool failed;
};

typedef union {
    struct cmsghdr alignment;
    unsigned char bytes[CMSG_SPACE(sizeof(int) * 8u)];
} qualification_control_ancillary;

static void qualification_control_error(
        char *err, size_t errcap, const char *format, ...) {
    if (!err || errcap == 0) return;
    va_list arguments;
    va_start(arguments, format);
    (void)vsnprintf(err, errcap, format, arguments);
    va_end(arguments);
}

static int qualification_control_fail(
        ds4_qualification_control *control,
        char *err,
        size_t errcap,
        const char *format,
        ...) {
    if (control) control->failed = true;
    if (err && errcap != 0) {
        va_list arguments;
        va_start(arguments, format);
        (void)vsnprintf(err, errcap, format, arguments);
        va_end(arguments);
    }
    return -1;
}

static int qualification_control_existing_failure(
        char *err, size_t errcap) {
    qualification_control_error(
        err, errcap, "qualification control is already unsafe");
    return -1;
}

static bool qualification_control_identity_equal(
        const ds4_runtime_file_identity *left,
        const ds4_runtime_file_identity *right) {
    return left && right &&
        left->device == right->device &&
        left->inode == right->inode &&
        left->size_bytes == right->size_bytes &&
        left->mtime_ns == right->mtime_ns;
}

static bool qualification_control_identity_zero(
        const ds4_runtime_file_identity *identity) {
    static const ds4_runtime_file_identity zero;
    return qualification_control_identity_equal(identity, &zero);
}

static int qualification_control_capture_identity(
        int fd,
        ds4_runtime_file_identity *identity,
        char *err,
        size_t errcap) {
    if (!identity) {
        qualification_control_error(
            err, errcap, "model identity output is null");
        return -1;
    }
    memset(identity, 0, sizeof(*identity));

    struct stat status;
    if (fd < 0 || fstat(fd, &status) != 0) {
        const int saved_errno = fd < 0 ? EBADF : errno;
        qualification_control_error(
            err, errcap, "cannot stat opened model descriptor: %s",
            strerror(saved_errno));
        return -1;
    }
    if (!S_ISREG(status.st_mode) || status.st_size < 0) {
        qualification_control_error(
            err, errcap, "opened model descriptor is not a regular file");
        return -1;
    }

#if defined(__APPLE__)
    const time_t mtime_seconds = status.st_mtimespec.tv_sec;
    const long mtime_nanoseconds = status.st_mtimespec.tv_nsec;
#else
    const time_t mtime_seconds = status.st_mtim.tv_sec;
    const long mtime_nanoseconds = status.st_mtim.tv_nsec;
#endif
    const uintmax_t device = (uintmax_t)status.st_dev;
    const uintmax_t inode = (uintmax_t)status.st_ino;
    const uintmax_t size_bytes = (uintmax_t)status.st_size;
    if (device > UINT64_MAX || inode > UINT64_MAX ||
        size_bytes > UINT64_MAX || mtime_seconds < 0 ||
        mtime_nanoseconds < 0 || mtime_nanoseconds >= 1000000000L) {
        qualification_control_error(
            err, errcap, "opened model descriptor identity is out of range");
        return -1;
    }
    const uintmax_t seconds = (uintmax_t)mtime_seconds;
    const uint64_t nanoseconds = (uint64_t)mtime_nanoseconds;
    if (seconds > (UINT64_MAX - nanoseconds) / UINT64_C(1000000000)) {
        qualification_control_error(
            err, errcap, "opened model descriptor mtime overflows");
        return -1;
    }

    identity->device = (uint64_t)device;
    identity->inode = (uint64_t)inode;
    identity->size_bytes = (uint64_t)size_bytes;
    identity->mtime_ns =
        (uint64_t)seconds * UINT64_C(1000000000) + nanoseconds;
    return 0;
}

static int qualification_control_now_ms(uint64_t *out) {
    struct timespec now;
    if (!out || clock_gettime(CLOCK_MONOTONIC, &now) != 0 ||
        now.tv_sec < 0 || now.tv_nsec < 0 || now.tv_nsec >= 1000000000L) {
        return -1;
    }
    const uintmax_t seconds = (uintmax_t)now.tv_sec;
    const uint64_t milliseconds = (uint64_t)now.tv_nsec / UINT64_C(1000000);
    if (seconds > (UINT64_MAX - milliseconds) / UINT64_C(1000)) {
        return -1;
    }
    *out = (uint64_t)seconds * UINT64_C(1000) + milliseconds;
    return 0;
}

static int qualification_control_deadline(
        ds4_qualification_control *control,
        uint64_t *deadline,
        char *err,
        size_t errcap) {
    uint64_t now = 0;
    if (!deadline || qualification_control_now_ms(&now) != 0 ||
        UINT64_MAX - now < control->timeout_ms) {
        return qualification_control_fail(
            control, err, errcap,
            "cannot establish qualification control deadline");
    }
    *deadline = now + control->timeout_ms;
    return 0;
}

static int qualification_control_poll(
        ds4_qualification_control *control,
        short events,
        uint64_t deadline,
        const char *operation,
        char *err,
        size_t errcap) {
    for (;;) {
        uint64_t now = 0;
        if (qualification_control_now_ms(&now) != 0) {
            return qualification_control_fail(
                control, err, errcap,
                "cannot read clock while waiting to %s", operation);
        }
        if (now >= deadline) {
            return qualification_control_fail(
                control, err, errcap,
                "qualification control timed out while waiting to %s",
                operation);
        }
        uint64_t remaining = deadline - now;
        int timeout = remaining > (uint64_t)INT32_MAX
            ? INT32_MAX : (int)remaining;
        if (timeout == 0) timeout = 1;
        struct pollfd descriptor = {
            .fd = control->fd,
            .events = events,
        };
        const int result = poll(&descriptor, 1, timeout);
        if (result < 0 && errno == EINTR) continue;
        if (result < 0) {
            const int saved_errno = errno;
            return qualification_control_fail(
                control, err, errcap,
                "qualification control poll failed while waiting to %s: %s",
                operation, strerror(saved_errno));
        }
        if (result == 0) {
            return qualification_control_fail(
                control, err, errcap,
                "qualification control timed out while waiting to %s",
                operation);
        }
        if ((descriptor.revents & POLLNVAL) != 0) {
            return qualification_control_fail(
                control, err, errcap,
                "qualification control descriptor became invalid while "
                "waiting to %s", operation);
        }
        if ((descriptor.revents & events) != 0) return 0;
        if ((descriptor.revents & (POLLERR | POLLHUP)) != 0) {
            return qualification_control_fail(
                control, err, errcap,
                "qualification control peer disconnected while waiting to %s",
                operation);
        }
    }
}

static int qualification_control_send_flags(void) {
    int flags = 0;
#ifdef MSG_DONTWAIT
    flags |= MSG_DONTWAIT;
#endif
#ifdef MSG_NOSIGNAL
    flags |= MSG_NOSIGNAL;
#endif
    return flags;
}

static int qualification_control_receive_flags(void) {
    int flags = 0;
#ifdef MSG_DONTWAIT
    flags |= MSG_DONTWAIT;
#endif
#ifdef MSG_CMSG_CLOEXEC
    flags |= MSG_CMSG_CLOEXEC;
#endif
    return flags;
}

static int qualification_control_send_remainder(
        ds4_qualification_control *control,
        const unsigned char *bytes,
        size_t size,
        size_t offset,
        uint64_t deadline,
        const char *operation,
        char *err,
        size_t errcap) {
    while (offset < size) {
        if (qualification_control_poll(
                control, POLLOUT, deadline, operation, err, errcap) != 0) {
            return -1;
        }
        const ssize_t written = send(
            control->fd, bytes + offset, size - offset,
            qualification_control_send_flags());
        if (written < 0 &&
            (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) {
            continue;
        }
        if (written < 0) {
            const int saved_errno = errno;
            return qualification_control_fail(
                control, err, errcap,
                "qualification control failed to %s: %s",
                operation, strerror(saved_errno));
        }
        if (written == 0) {
            return qualification_control_fail(
                control, err, errcap,
                "qualification control peer disconnected while trying to %s",
                operation);
        }
        offset += (size_t)written;
    }
    return 0;
}

static int qualification_control_send_message(
        ds4_qualification_control *control,
        const ds4_qualification_control_message *message,
        int passed_fd,
        char *err,
        size_t errcap) {
    uint64_t deadline = 0;
    if (qualification_control_deadline(
            control, &deadline, err, errcap) != 0) {
        return -1;
    }

    if (passed_fd < 0) {
        return qualification_control_send_remainder(
            control, (const unsigned char *)message, sizeof(*message), 0,
            deadline, "send protocol message", err, errcap);
    }

    qualification_control_ancillary ancillary;
    memset(&ancillary, 0, sizeof(ancillary));
    struct iovec vector = {
        .iov_base = (void *)message,
        .iov_len = sizeof(*message),
    };
    struct msghdr header;
    memset(&header, 0, sizeof(header));
    header.msg_iov = &vector;
    header.msg_iovlen = 1;
    header.msg_control = ancillary.bytes;
    header.msg_controllen = CMSG_SPACE(sizeof(int));
    struct cmsghdr *rights = CMSG_FIRSTHDR(&header);
    if (!rights) {
        return qualification_control_fail(
            control, err, errcap,
            "cannot construct qualification model descriptor message");
    }
    rights->cmsg_level = SOL_SOCKET;
    rights->cmsg_type = SCM_RIGHTS;
    rights->cmsg_len = CMSG_LEN(sizeof(int));
    memcpy(CMSG_DATA(rights), &passed_fd, sizeof(passed_fd));

    size_t offset = 0;
    for (;;) {
        if (qualification_control_poll(
                control, POLLOUT, deadline, "send model descriptor",
                err, errcap) != 0) {
            return -1;
        }
        const ssize_t written = sendmsg(
            control->fd, &header, qualification_control_send_flags());
        if (written < 0 &&
            (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) {
            continue;
        }
        if (written < 0) {
            const int saved_errno = errno;
            return qualification_control_fail(
                control, err, errcap,
                "qualification control failed to send model descriptor: %s",
                strerror(saved_errno));
        }
        if (written == 0) {
            return qualification_control_fail(
                control, err, errcap,
                "qualification control peer disconnected while sending "
                "model descriptor");
        }
        if ((size_t)written > sizeof(*message)) {
            return qualification_control_fail(
                control, err, errcap,
                "qualification control sent an invalid model message size");
        }
        offset = (size_t)written;
        break;
    }
    return qualification_control_send_remainder(
        control, (const unsigned char *)message, sizeof(*message), offset,
        deadline, "finish model descriptor message", err, errcap);
}

static void qualification_control_close_received_rights(
        const struct msghdr *header) {
    if (!header) return;
    for (struct cmsghdr *item = CMSG_FIRSTHDR((struct msghdr *)header);
         item != NULL;
         item = CMSG_NXTHDR((struct msghdr *)header, item)) {
        if (item->cmsg_level != SOL_SOCKET ||
            item->cmsg_type != SCM_RIGHTS ||
            item->cmsg_len < CMSG_LEN(0)) {
            continue;
        }
        const size_t bytes = item->cmsg_len - CMSG_LEN(0);
        if (bytes % sizeof(int) != 0) continue;
        const size_t count = bytes / sizeof(int);
        const int *descriptors = (const int *)CMSG_DATA(item);
        for (size_t i = 0; i < count; i++) {
            if (descriptors[i] >= 0) (void)close(descriptors[i]);
        }
    }
}

static int qualification_control_receive_message(
        ds4_qualification_control *control,
        ds4_qualification_control_message *message,
        char *err,
        size_t errcap) {
    uint64_t deadline = 0;
    if (qualification_control_deadline(
            control, &deadline, err, errcap) != 0) {
        return -1;
    }
    memset(message, 0, sizeof(*message));
    size_t offset = 0;
    while (offset < sizeof(*message)) {
        if (qualification_control_poll(
                control, POLLIN, deadline, "receive acknowledgement",
                err, errcap) != 0) {
            return -1;
        }
        qualification_control_ancillary ancillary;
        memset(&ancillary, 0, sizeof(ancillary));
        struct iovec vector = {
            .iov_base = (unsigned char *)message + offset,
            .iov_len = sizeof(*message) - offset,
        };
        struct msghdr header;
        memset(&header, 0, sizeof(header));
        header.msg_iov = &vector;
        header.msg_iovlen = 1;
        header.msg_control = ancillary.bytes;
        header.msg_controllen = sizeof(ancillary.bytes);
        const ssize_t received = recvmsg(
            control->fd, &header, qualification_control_receive_flags());
        if (received < 0 &&
            (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) {
            continue;
        }
        if (received < 0) {
            const int saved_errno = errno;
            return qualification_control_fail(
                control, err, errcap,
                "qualification control failed to receive acknowledgement: %s",
                strerror(saved_errno));
        }
        if (received == 0) {
            return qualification_control_fail(
                control, err, errcap,
                "qualification control peer disconnected while waiting for "
                "acknowledgement");
        }

        const bool ancillary_present =
            header.msg_controllen != 0 ||
            (header.msg_flags & MSG_CTRUNC) != 0;
        if (ancillary_present) {
            qualification_control_close_received_rights(&header);
            return qualification_control_fail(
                control, err, errcap,
                "qualification acknowledgement carried forbidden ancillary "
                "data");
        }
        if ((header.msg_flags & MSG_TRUNC) != 0 ||
            (size_t)received > sizeof(*message) - offset) {
            return qualification_control_fail(
                control, err, errcap,
                "qualification acknowledgement has an invalid size");
        }
        offset += (size_t)received;
    }
    return 0;
}

static int qualification_control_exchange(
        ds4_qualification_control *control,
        uint32_t sent_type,
        uint32_t expected_type,
        uint64_t sequence,
        const ds4_runtime_file_identity *identity,
        char *err,
        size_t errcap) {
    ds4_qualification_control_message sent;
    memset(&sent, 0, sizeof(sent));
    sent.protocol_version = DS4_QUALIFICATION_CONTROL_PROTOCOL_VERSION;
    sent.message_type = sent_type;
    sent.message_size = (uint32_t)sizeof(sent);
    sent.checkpoint_sequence = sequence;
    if (identity) sent.model_identity = *identity;
    if (qualification_control_send_message(
            control, &sent, -1, err, errcap) != 0) {
        return -1;
    }

    ds4_qualification_control_message acknowledgement;
    if (qualification_control_receive_message(
            control, &acknowledgement, err, errcap) != 0) {
        return -1;
    }
    if (acknowledgement.protocol_version !=
            DS4_QUALIFICATION_CONTROL_PROTOCOL_VERSION ||
        acknowledgement.message_type != expected_type ||
        acknowledgement.message_size != sizeof(acknowledgement) ||
        acknowledgement.reserved != 0 ||
        acknowledgement.checkpoint_sequence != sequence ||
        !qualification_control_identity_zero(
            &acknowledgement.model_identity)) {
        return qualification_control_fail(
            control, err, errcap,
            "qualification control received a mismatched acknowledgement");
    }
    return 0;
}

static int qualification_control_validate_socket(
        int fd, char *err, size_t errcap) {
    if (fd < 0) {
        qualification_control_error(
            err, errcap, "qualification control descriptor is invalid");
        return -1;
    }
    int type = 0;
    socklen_t type_size = sizeof(type);
    if (getsockopt(fd, SOL_SOCKET, SO_TYPE, &type, &type_size) != 0 ||
        type_size != sizeof(type) || type != SOCK_STREAM) {
        qualification_control_error(
            err, errcap,
            "qualification control descriptor is not a Unix stream socket");
        return -1;
    }
    struct sockaddr_storage local;
    socklen_t local_size = sizeof(local);
    if (getsockname(fd, (struct sockaddr *)&local, &local_size) != 0 ||
        local_size < sizeof(sa_family_t) || local.ss_family != AF_UNIX) {
        qualification_control_error(
            err, errcap,
            "qualification control descriptor is not a Unix stream socket");
        return -1;
    }
    struct sockaddr_storage peer;
    socklen_t peer_size = sizeof(peer);
    if (getpeername(fd, (struct sockaddr *)&peer, &peer_size) != 0 ||
        peer_size < sizeof(sa_family_t) || peer.ss_family != AF_UNIX) {
        qualification_control_error(
            err, errcap,
            "qualification control Unix socket is not connected");
        return -1;
    }
    return 0;
}

static int qualification_control_duplicate_cloexec(int fd) {
    int duplicate = -1;
#ifdef F_DUPFD_CLOEXEC
    duplicate = fcntl(fd, F_DUPFD_CLOEXEC, 0);
    if (duplicate >= 0) return duplicate;
    if (errno != EINVAL) return -1;
#endif
    duplicate = fcntl(fd, F_DUPFD, 0);
    if (duplicate < 0) return -1;
    const int flags = fcntl(duplicate, F_GETFD);
    if (flags < 0 || fcntl(duplicate, F_SETFD, flags | FD_CLOEXEC) != 0) {
        const int saved_errno = errno;
        (void)close(duplicate);
        errno = saved_errno;
        return -1;
    }
    return duplicate;
}

int ds4_qualification_control_open(
        ds4_qualification_control **out,
        int inherited_fd,
        uint32_t timeout_ms,
        char *err,
        size_t errcap) {
    if (err && errcap != 0) err[0] = '\0';
    if (!out) {
        qualification_control_error(
            err, errcap, "qualification control output is null");
        return -1;
    }
    *out = NULL;
    if (timeout_ms == 0) {
        qualification_control_error(
            err, errcap, "qualification control timeout must be nonzero");
        return -1;
    }
    if (qualification_control_validate_socket(
            inherited_fd, err, errcap) != 0) {
        return -1;
    }

    const int duplicate = qualification_control_duplicate_cloexec(inherited_fd);
    if (duplicate < 0) {
        const int saved_errno = errno;
        qualification_control_error(
            err, errcap,
            "cannot duplicate qualification control descriptor: %s",
            strerror(saved_errno));
        return -1;
    }
#ifdef SO_NOSIGPIPE
    const int enabled = 1;
    if (setsockopt(
            duplicate, SOL_SOCKET, SO_NOSIGPIPE,
            &enabled, sizeof(enabled)) != 0) {
        const int saved_errno = errno;
        (void)close(duplicate);
        qualification_control_error(
            err, errcap,
            "cannot suppress qualification control SIGPIPE: %s",
            strerror(saved_errno));
        return -1;
    }
#endif

    ds4_qualification_control *control = calloc(1, sizeof(*control));
    if (!control) {
        const int saved_errno = errno;
        (void)close(duplicate);
        qualification_control_error(
            err, errcap, "cannot allocate qualification control: %s",
            strerror(saved_errno));
        return -1;
    }
    control->fd = duplicate;
    control->timeout_ms = timeout_ms;
    *out = control;
    return 0;
}

int ds4_qualification_control_send_model_fd(
        ds4_qualification_control *control,
        int model_fd,
        const ds4_runtime_file_identity *expected_identity,
        char *err,
        size_t errcap) {
    if (err && errcap != 0) err[0] = '\0';
    if (!control) {
        qualification_control_error(
            err, errcap, "qualification control is null");
        return -1;
    }
    if (control->failed) {
        return qualification_control_existing_failure(err, errcap);
    }
    if (control->model_identity_set || !expected_identity) {
        return qualification_control_fail(
            control, err, errcap,
            control->model_identity_set
                ? "qualification model descriptor was already sent"
                : "qualification model identity is null");
    }

    ds4_runtime_file_identity observed;
    char identity_error[192] = {0};
    if (qualification_control_capture_identity(
            model_fd, &observed,
            identity_error, sizeof(identity_error)) != 0 ||
        !qualification_control_identity_equal(
            &observed, expected_identity)) {
        return qualification_control_fail(
            control, err, errcap, "%s",
            identity_error[0]
                ? identity_error
                : "qualification model descriptor identity changed");
    }

    ds4_qualification_control_message message;
    memset(&message, 0, sizeof(message));
    message.protocol_version = DS4_QUALIFICATION_CONTROL_PROTOCOL_VERSION;
    message.message_type = DS4_QUALIFICATION_CONTROL_MODEL_FD;
    message.message_size = (uint32_t)sizeof(message);
    message.model_identity = observed;
    if (qualification_control_send_message(
            control, &message, model_fd, err, errcap) != 0) {
        return -1;
    }

    ds4_runtime_file_identity after;
    memset(identity_error, 0, sizeof(identity_error));
    if (qualification_control_capture_identity(
            model_fd, &after, identity_error, sizeof(identity_error)) != 0 ||
        !qualification_control_identity_equal(&after, &observed)) {
        return qualification_control_fail(
            control, err, errcap, "%s",
            identity_error[0]
                ? identity_error
                : "qualification model descriptor identity changed while sent");
    }
    control->model_identity = observed;
    control->model_identity_set = true;
    return 0;
}

int ds4_qualification_control_begin_sample(
        ds4_qualification_control *control,
        uint64_t checkpoint_sequence,
        char *err,
        size_t errcap) {
    if (err && errcap != 0) err[0] = '\0';
    if (!control) {
        qualification_control_error(
            err, errcap, "qualification control is null");
        return -1;
    }
    if (control->failed) {
        return qualification_control_existing_failure(err, errcap);
    }
    if (control->sample_active || checkpoint_sequence == 0 ||
        checkpoint_sequence <= control->last_finished_sequence) {
        return qualification_control_fail(
            control, err, errcap,
            control->sample_active
                ? "qualification sample is already active"
                : "qualification checkpoint sequence is not strictly increasing");
    }
    control->sample_active = true;
    control->active_sequence = checkpoint_sequence;
    return qualification_control_exchange(
        control,
        DS4_QUALIFICATION_CONTROL_SAMPLE_READY,
        DS4_QUALIFICATION_CONTROL_SAMPLE_READY_ACK,
        checkpoint_sequence, NULL, err, errcap);
}

int ds4_qualification_control_finish_sample(
        ds4_qualification_control *control,
        uint64_t checkpoint_sequence,
        int model_fd,
        char *err,
        size_t errcap) {
    if (err && errcap != 0) err[0] = '\0';
    if (!control) {
        qualification_control_error(
            err, errcap, "qualification control is null");
        return -1;
    }
    if (control->failed) {
        return qualification_control_existing_failure(err, errcap);
    }
    if (!control->sample_active ||
        checkpoint_sequence != control->active_sequence) {
        return qualification_control_fail(
            control, err, errcap,
            "qualification sample finish sequence does not match begin");
    }
    if (!control->model_identity_set) {
        return qualification_control_fail(
            control, err, errcap,
            "qualification model descriptor was not sent before sampling");
    }

    ds4_runtime_file_identity observed;
    char identity_error[192] = {0};
    if (qualification_control_capture_identity(
            model_fd, &observed,
            identity_error, sizeof(identity_error)) != 0 ||
        !qualification_control_identity_equal(
            &observed, &control->model_identity)) {
        return qualification_control_fail(
            control, err, errcap, "%s",
            identity_error[0]
                ? identity_error
                : "qualification model descriptor identity changed");
    }
    if (qualification_control_exchange(
            control,
            DS4_QUALIFICATION_CONTROL_SAMPLE_RESULT,
            DS4_QUALIFICATION_CONTROL_SAMPLE_RESULT_ACK,
            checkpoint_sequence, &observed, err, errcap) != 0) {
        return -1;
    }
    control->last_finished_sequence = checkpoint_sequence;
    control->active_sequence = 0;
    control->sample_active = false;
    return 0;
}

void ds4_qualification_control_close(ds4_qualification_control *control) {
    if (!control) return;
    if (control->fd >= 0) (void)close(control->fd);
    control->fd = -1;
    free(control);
}
