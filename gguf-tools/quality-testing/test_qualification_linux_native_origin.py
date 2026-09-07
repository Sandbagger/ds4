#!/usr/bin/env python3
"""Linux-native retained-descriptor qualification transport acceptance tests.

These tests deliberately use one small C ELF fixture.  They do not claim a
DS4, CUDA, model, or GPU qualification result.  They only prove that the
existing host transport can execute a retained descriptor on Linux and that
its authenticated runner observes the real ``/proc/<owned-pid>/exe`` identity.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Iterator

from qualification_artifacts import (
    authenticate_running_executable,
    open_qualification_artifact,
)
from qualification_authenticated import run_authenticated_qualification_child
from qualification_version_probe import QualificationVersionProbe
from test_qualification_records import _expected, _lifecycle_records


SAFE_PATH = "/usr/local/bin:/usr/bin:/bin"
SAFE_ENV = {"PATH": SAFE_PATH, "LANG": "C", "LC_ALL": "C"}
COMPILER_TIMEOUT_SECONDS = 5.0
VERSION_WATCHDOG_SECONDS = 3.0
AUTH_WATCHDOG_SECONDS = 8.0

# The fixture has no dependency on Python, the repository, a model, CUDA, or a
# service.  ``cc`` is invoked by the test only; the qualification child gets
# exactly SAFE_ENV through the parent environment below.
NATIVE_FIXTURE_C = r"""
#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <time.h>
#include <unistd.h>

#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif

#define SAFE_PATH "/usr/local/bin:/usr/bin:/bin"

struct wire_message {
    uint32_t protocol;
    uint32_t message_type;
    uint32_t size;
    uint32_t reserved;
    uint64_t sequence;
    uint64_t device;
    uint64_t inode;
    uint64_t size_bytes;
    uint64_t mtime_ns;
};

_Static_assert(sizeof(struct wire_message) == 56, "unexpected qualification wire ABI");

extern char **environ;

static int write_all(int fd, const void *data, size_t size) {
    const unsigned char *cursor = (const unsigned char *)data;
    while (size != 0) {
        ssize_t written = write(fd, cursor, size);
        if (written < 0 && errno == EINTR) {
            continue;
        }
        if (written <= 0) {
            return -1;
        }
        cursor += (size_t)written;
        size -= (size_t)written;
    }
    return 0;
}

static int recv_all(int fd, void *data, size_t size) {
    unsigned char *cursor = (unsigned char *)data;
    while (size != 0) {
        ssize_t received = recv(fd, cursor, size, 0);
        if (received < 0 && errno == EINTR) {
            continue;
        }
        if (received <= 0) {
            return -1;
        }
        cursor += (size_t)received;
        size -= (size_t)received;
    }
    return 0;
}

static int send_wire(int fd, const struct wire_message *message) {
    const unsigned char *cursor = (const unsigned char *)message;
    size_t remaining = sizeof(*message);
    while (remaining != 0) {
        ssize_t written = send(fd, cursor, remaining, MSG_NOSIGNAL);
        if (written < 0 && errno == EINTR) {
            continue;
        }
        if (written <= 0) {
            return -1;
        }
        cursor += (size_t)written;
        remaining -= (size_t)written;
    }
    return 0;
}

static int environment_is_exact(void) {
    int path_seen = 0;
    int lang_seen = 0;
    int lc_all_seen = 0;
    for (char **entry = environ; entry != NULL && *entry != NULL; ++entry) {
        if (strcmp(*entry, "PATH=" SAFE_PATH) == 0) {
            path_seen = 1;
        } else if (strcmp(*entry, "LANG=C") == 0) {
            lang_seen = 1;
        } else if (strcmp(*entry, "LC_ALL=C") == 0) {
            lc_all_seen = 1;
        } else {
            return 0;
        }
    }
    return path_seen && lang_seen && lc_all_seen;
}

static int executable_label(char *buffer, size_t capacity) {
    ssize_t length = readlink("/proc/self/exe", buffer, capacity - 1);
    if (length >= 0) {
        buffer[length] = '\0';
        if (strstr(buffer, "native-version-") != NULL) {
            return 0;
        }
    }
    /* Kernels normally expose the original dentry through /proc/exe.  If a
       host exposes only the retained /proc/self/fd link, find that exact
       inherited descriptor rather than consulting an environment or argv. */
    for (int fd = 3; fd < 256; ++fd) {
        char descriptor_path[64];
        int path_length = snprintf(descriptor_path, sizeof(descriptor_path),
                                   "/proc/self/fd/%d", fd);
        if (path_length <= 0 || (size_t)path_length >= sizeof(descriptor_path)) {
            continue;
        }
        length = readlink(descriptor_path, buffer, capacity - 1);
        if (length < 0) {
            continue;
        }
        buffer[length] = '\0';
        if (strstr(buffer, "native-version-") != NULL) {
            return 0;
        }
    }
    return -1;
}

static int inherited_descriptors_are_expected(int expected_count, int control_fd) {
    DIR *directory = opendir("/proc/self/fd");
    if (directory == NULL) {
        return 0;
    }
    int directory_fd = dirfd(directory);
    int count = 0;
    int executable_count = 0;
    struct dirent *entry;
    while ((entry = readdir(directory)) != NULL) {
        char *end = NULL;
        errno = 0;
        long parsed = strtol(entry->d_name, &end, 10);
        if (errno != 0 || end == entry->d_name || *end != '\0' ||
            parsed <= 2 || parsed > INT_MAX || parsed == directory_fd) {
            continue;
        }
        int fd = (int)parsed;
        if (fcntl(fd, F_GETFD) < 0) {
            continue;
        }
        ++count;
        if (fd != control_fd) {
            struct stat descriptor_stat;
            if (fstat(fd, &descriptor_stat) == 0 &&
                S_ISREG(descriptor_stat.st_mode) &&
                (descriptor_stat.st_mode & 0111) != 0) {
                ++executable_count;
            }
        }
    }
    closedir(directory);
    return count == expected_count && executable_count == 1;
}

static void short_sleep(void) {
    struct timespec remaining = {2, 0};
    while (nanosleep(&remaining, &remaining) < 0 && errno == EINTR) {
    }
}

static uint64_t last_monotonic_ns = 0;

static int write_record(char *record) {
    const char marker[] = "\"monotonic_ns\":\"";
    char *value = strstr(record, marker);
    if (value == NULL) {
        return -1;
    }
    value += sizeof(marker) - 1;
    char *end = strchr(value, '\"');
    if (end == NULL) {
        return -1;
    }
    struct timespec clock_value;
    if (clock_gettime(CLOCK_MONOTONIC, &clock_value) < 0) {
        return -1;
    }
    uint64_t now = (uint64_t)clock_value.tv_sec * 1000000000ULL +
                   (uint64_t)clock_value.tv_nsec;
    if (now <= last_monotonic_ns) {
        now = last_monotonic_ns + 1;
    }
    last_monotonic_ns = now;
    char decimal[32];
    int length = snprintf(decimal, sizeof(decimal), "%" PRIu64, now);
    if (length <= 0 || (size_t)length >= sizeof(decimal)) {
        return -1;
    }
    if (write_all(STDOUT_FILENO, record, (size_t)(value - record)) < 0 ||
        write_all(STDOUT_FILENO, decimal, (size_t)length) < 0 ||
        write_all(STDOUT_FILENO, end, strlen(end)) < 0) {
        return -1;
    }
    return 0;
}

static int version_main(void) {
    if (!environment_is_exact()) {
        static const char message[] = "native fixture environment mismatch\n";
        (void)write_all(STDERR_FILENO, message, sizeof(message) - 1);
        return 125;
    }
    if (!inherited_descriptors_are_expected(1, -1)) {
        return 124;
    }
    char label[PATH_MAX];
    if (executable_label(label, sizeof(label)) < 0) {
        return 126;
    }
    if (strstr(label, "native-version-timeout") != NULL) {
        static const char prefix[] = "native-timeout-prefix\n";
        (void)write_all(STDOUT_FILENO, prefix, sizeof(prefix) - 1);
        short_sleep();
        return 0;
    }
    if (strstr(label, "native-version-nonzero") != NULL) {
        static const char stdout_prefix[] = "native-nonzero-prefix\n";
        static const char stderr_prefix[] = "native-nonzero-stderr\n";
        (void)write_all(STDOUT_FILENO, stdout_prefix, sizeof(stdout_prefix) - 1);
        (void)write_all(STDERR_FILENO, stderr_prefix, sizeof(stderr_prefix) - 1);
        return 7;
    }
    if (strstr(label, "native-version-malformed") != NULL) {
        static const unsigned char malformed[] = {
            'n', 'o', 't', '-', 'j', 's', 'o', 'n', '\0', 0xff, '\n'
        };
        return write_all(STDOUT_FILENO, malformed, sizeof(malformed)) == 0 ? 0 : 126;
    }
    static const char version[] = "native-elf-version/v1\n";
    return write_all(STDOUT_FILENO, version, sizeof(version) - 1) == 0 ? 0 : 126;
}

static int receive_ack(int endpoint, uint32_t message_type, uint64_t sequence) {
    struct wire_message message;
    if (recv_all(endpoint, &message, sizeof(message)) < 0) {
        return -1;
    }
    return (message.protocol == 1 && message.message_type == message_type &&
            message.size == sizeof(message) && message.reserved == 0 &&
            message.sequence == sequence) ? 0 : -1;
}

static int send_control(int endpoint, uint32_t message_type, uint64_t sequence,
                        const struct wire_message *identity) {
    struct wire_message message;
    memset(&message, 0, sizeof(message));
    message.protocol = 1;
    message.message_type = message_type;
    message.size = sizeof(message);
    message.sequence = sequence;
    if (identity != NULL) {
        message.device = identity->device;
        message.inode = identity->inode;
        message.size_bytes = identity->size_bytes;
        message.mtime_ns = identity->mtime_ns;
    }
    return send_wire(endpoint, &message);
}

static int send_model(int endpoint, int model_fd, const struct wire_message *identity) {
    struct wire_message message;
    memset(&message, 0, sizeof(message));
    message.protocol = 1;
    message.message_type = 1;
    message.size = sizeof(message);
    message.sequence = 0;
    message.device = identity->device;
    message.inode = identity->inode;
    message.size_bytes = identity->size_bytes;
    message.mtime_ns = identity->mtime_ns;

    struct iovec vector = {&message, sizeof(message)};
    char ancillary[CMSG_SPACE(sizeof(int))];
    memset(ancillary, 0, sizeof(ancillary));
    struct msghdr packet;
    memset(&packet, 0, sizeof(packet));
    packet.msg_iov = &vector;
    packet.msg_iovlen = 1;
    packet.msg_control = ancillary;
    packet.msg_controllen = sizeof(ancillary);
    struct cmsghdr *header = CMSG_FIRSTHDR(&packet);
    if (header == NULL) {
        return -1;
    }
    header->cmsg_level = SOL_SOCKET;
    header->cmsg_type = SCM_RIGHTS;
    header->cmsg_len = CMSG_LEN(sizeof(model_fd));
    memcpy(CMSG_DATA(header), &model_fd, sizeof(model_fd));
    ssize_t sent;
    do {
        sent = sendmsg(endpoint, &packet, MSG_NOSIGNAL);
    } while (sent < 0 && errno == EINTR);
    return sent == (ssize_t)sizeof(message) ? 0 : -1;
}

static int argument_value(int argc, char **argv, const char *name,
                          const char **value) {
    int found = 0;
    for (int index = 1; index + 1 < argc; ++index) {
        if (strcmp(argv[index], name) == 0) {
            *value = argv[index + 1];
            ++found;
        }
    }
    return found == 1 && *value != NULL && (*value)[0] != '\0';
}

static int argument_fd(int argc, char **argv, const char *name, int *value) {
    const char *text = NULL;
    if (!argument_value(argc, argv, name, &text)) {
        return 0;
    }
    char *end = NULL;
    errno = 0;
    long parsed = strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed < 3 || parsed > INT_MAX) {
        return 0;
    }
    *value = (int)parsed;
    return 1;
}

static int load_records(const char *path, char **records) {
    FILE *stream = fopen(path, "rb");
    if (stream == NULL) {
        return -1;
    }
    size_t count = 0;
    char *line = NULL;
    size_t capacity = 0;
    ssize_t length;
    while ((length = getline(&line, &capacity, stream)) >= 0) {
        if (count >= 12 || length <= 0 || length > 65536) {
            free(line);
            fclose(stream);
            return -1;
        }
        records[count] = malloc((size_t)length + 1u);
        if (records[count] == NULL) {
            free(line);
            fclose(stream);
            return -1;
        }
        memcpy(records[count], line, (size_t)length);
        records[count][length] = '\0';
        ++count;
    }
    free(line);
    fclose(stream);
    return count == 12 ? 0 : -1;
}

static int control_main(int argc, char **argv) {
    if (!environment_is_exact()) {
        return 125;
    }
    const char *model_path = NULL;
    const char *records_path = NULL;
    int control_fd = -1;
    if (!argument_value(argc, argv, "--model-path", &model_path) ||
        !argument_value(argc, argv, "--records-path", &records_path) ||
        !argument_fd(argc, argv, "--qualification-control-fd", &control_fd)) {
        return 41;
    }
    if (!inherited_descriptors_are_expected(2, control_fd)) {
        return 124;
    }

    char *records[12] = {0};
    if (load_records(records_path, records) < 0) {
        return 42;
    }
    int model_fd = open(model_path, O_RDONLY | O_CLOEXEC);
    if (model_fd < 0) {
        for (size_t index = 0; index < 12; ++index) free(records[index]);
        return 43;
    }
    struct stat model_stat;
    if (fstat(model_fd, &model_stat) < 0) {
        close(model_fd);
        for (size_t index = 0; index < 12; ++index) free(records[index]);
        return 44;
    }
    struct wire_message identity;
    memset(&identity, 0, sizeof(identity));
    identity.device = (uint64_t)model_stat.st_dev;
    identity.inode = (uint64_t)model_stat.st_ino;
    identity.size_bytes = (uint64_t)model_stat.st_size;
    identity.mtime_ns = (uint64_t)model_stat.st_mtim.tv_nsec;
    identity.mtime_ns += (uint64_t)model_stat.st_mtime * 1000000000ULL;

    /* The child offers its opened model descriptor; the parent authenticates
       it and then releases this child with MODEL_FD_ACK. */
    if (send_model(control_fd, model_fd, &identity) < 0) {
        close(model_fd);
        for (size_t index = 0; index < 12; ++index) free(records[index]);
        return 45;
    }
    close(model_fd);
    if (receive_ack(control_fd, 6, 0) < 0) {
        for (size_t index = 0; index < 12; ++index) free(records[index]);
        return 46;
    }

    for (uint64_t sequence = 1; sequence <= 12; ++sequence) {
        if (send_control(control_fd, 2, sequence, NULL) < 0 ||
            receive_ack(control_fd, 3, sequence) < 0 ||
            send_control(control_fd, 4, sequence, &identity) < 0 ||
            receive_ack(control_fd, 5, sequence) < 0 ||
            write_record(records[sequence - 1]) < 0) {
            for (size_t index = 0; index < 12; ++index) free(records[index]);
            return 48;
        }
    }
    for (size_t index = 0; index < 12; ++index) {
        free(records[index]);
    }
    close(control_fd);
    return 0;
}

int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "--version-json") == 0) {
        return version_main();
    }
    return control_main(argc, argv);
}
"""


@contextlib.contextmanager
def _safe_parent_environment() -> Iterator[None]:
    """Make inherited child environment explicit, then restore it exactly."""
    original = dict(os.environ)
    try:
        os.environ.clear()
        os.environ.update(SAFE_ENV)
        yield
    finally:
        os.environ.clear()
        os.environ.update(original)


@contextlib.contextmanager
def _watchdog(seconds: float, label: str) -> Iterator[None]:
    """Bound a native call without replacing its owned-child cleanup."""
    if not hasattr(signal, "SIGALRM") or not hasattr(signal, "setitimer"):
        raise unittest.SkipTest("Linux native fixture needs a POSIX SIGALRM watchdog")

    class _Expired(BaseException):
        pass

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 0.0)

    def expire(_signum: int, _frame: Any) -> None:
        raise _Expired(f"{label} exceeded {seconds}s watchdog")

    signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer != (0.0, 0.0):
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)


def _compile_fixture(root: Path) -> Path:
    """Compile one fixed C source with bounded, credential-free setup."""
    source = root / "native-fixture.c"
    output = root / "native-fixture-template"
    compiler_tmp = root / "compiler-tmp"
    source.write_text(NATIVE_FIXTURE_C, encoding="utf-8")
    compiler_tmp.mkdir()
    compiler = shutil.which("cc", path=SAFE_PATH)
    if compiler is None:
        raise unittest.SkipTest("Linux native-origin fixture requires cc in the safe compiler PATH")
    command = (
        compiler,
        "-std=c11",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-o",
        str(output),
        str(source),
    )
    build_env = dict(SAFE_ENV, TMPDIR=str(compiler_tmp))
    try:
        completed = subprocess.run(
            command,
            cwd=str(root),
            env=build_env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=COMPILER_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise AssertionError("native fixture compiler exceeded its 5s bound") from exc
    if completed.returncode != 0:
        detail = completed.stderr[-4096:].decode("utf-8", "replace")
        raise AssertionError(
            f"native fixture compiler failed ({completed.returncode}); "
            f"fixture/build failure, not a qualification contract failure: {detail}"
        )
    output.chmod(0o700)
    return output


def _materialize(template: Path, directory: Path, name: str) -> Path:
    path = directory / name
    shutil.copyfile(template, path)
    path.chmod(0o700)
    return path


def _write_records(path: Path, records: list[dict[str, Any]]) -> None:
    payload = b"".join(
        (json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")
        for record in records
    )
    path.write_bytes(payload)


@unittest.skipUnless(
    sys.platform == "linux",
    "Linux-only: this fixture intentionally has no Darwin descriptor adapter",
)
class LinuxNativeOriginHostTest(unittest.TestCase):
    """Exercise real Linux ELF/proc identity, not a Python or fake-proc seam."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._fixture = tempfile.TemporaryDirectory(prefix="qualification-linux-native-origin-")
        root = Path(cls._fixture.name)
        try:
            template = _compile_fixture(root)
            cls._template = template
            cls._version_success = _materialize(template, root, "native-version-success")
            cls._version_malformed = _materialize(template, root, "native-version-malformed")
            cls._version_nonzero = _materialize(template, root, "native-version-nonzero")
            cls._version_timeout = _materialize(template, root, "native-version-timeout")
            cls._control = _materialize(template, root, "native-control-good")
            cls._control_other = _materialize(template, root, "native-control-other")
            positive_fd = os.open(cls._version_success, os.O_RDONLY)
            try:
                os.set_inheritable(positive_fd, False)
                with _safe_parent_environment():
                    probe = subprocess.run(
                        [str(cls._version_success), "--version-json"],
                        cwd=str(root),
                        env=SAFE_ENV,
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=2.0,
                        check=False,
                        pass_fds=(positive_fd,),
                    )
            finally:
                os.close(positive_fd)
            if probe.returncode != 0 or probe.stdout != b"native-elf-version/v1\n":
                raise AssertionError(
                    "native fixture positive control failed; "
                    "distinguish compiler/fixture setup from qualification contract"
                )
            if cls._version_success.read_bytes()[:4] != b"\x7fELF":
                raise AssertionError("native fixture compiler did not produce an ELF executable")
        except BaseException:
            cls._fixture.cleanup()
            raise

    @classmethod
    def tearDownClass(cls) -> None:
        cls._fixture.cleanup()

    def _case_dir(self, label: str) -> Path:
        path = Path(self._fixture.name) / label
        path.mkdir()
        return path

    def test_native_version_probe_retains_success_malformed_nonzero_and_timeout(self) -> None:
        cases = (
            ("success", self._version_success, b"native-elf-version/v1\n", "complete"),
            (
                "malformed",
                self._version_malformed,
                b"not-json\x00\xff\n",
                "complete",
            ),
            (
                "nonzero",
                self._version_nonzero,
                b"native-nonzero-prefix\n",
                "child_exit",
            ),
            ("timeout", self._version_timeout, b"native-timeout-prefix\n", "timeout"),
        )
        for label, executable, expected_stdout, expected_reason in cases:
            with self.subTest(case=label):
                # Each invocation has its own fixture-owned copy.  No path in
                # HOME, TMPDIR, XDG, or the repository is changed or removed.
                case_dir = self._case_dir(f"version-{label}")
                case_executable = _materialize(
                    executable, case_dir, f"native-version-{label}"
                )
                # The behavior is selected by the retained executable's basename;
                # this copy remains a real ELF and is opened before probing.
                with open_qualification_artifact(case_executable, executable=True) as artifact:
                    self.assertEqual(os.pread(artifact.fd, 4, 0), b"\x7fELF")
                    probe = QualificationVersionProbe(
                        timeout_ns=(100_000_000 if expected_reason == "timeout" else 1_500_000_000),
                        termination_grace_ns=50_000_000,
                    )
                    with _safe_parent_environment(), _watchdog(
                        VERSION_WATCHDOG_SECONDS, f"native version {label}"
                    ):
                        if expected_reason == "complete":
                            observed_stdout = probe("server", artifact)
                        else:
                            with self.assertRaises(ValueError):
                                probe("server", artifact)
                            observed_stdout = probe.observations[0]["stdout"]
                    self.assertEqual(observed_stdout, expected_stdout)
                    observations = probe.observations
                    self.assertEqual(len(observations), 1)
                    observation = observations[0]
                    self.assertEqual(observation["reason"], expected_reason)
                    self.assertTrue(observation["cleanup_complete"])
                    self.assertIsInstance(observation["pid"], int)
                    self.assertIsInstance(observation["returncode"], int)
                    self.assertFalse(observation["stdout_truncated"])
                    self.assertFalse(observation["stderr_truncated"])
                    if label == "nonzero":
                        self.assertEqual(observation["returncode"], 7)
                        self.assertEqual(observation["stderr"], b"native-nonzero-stderr\n")
                    if label == "malformed":
                        self.assertEqual(observation["stdout"], expected_stdout)
                    # The probe borrows, rather than closes, the retained owner.
                    self.assertEqual(os.pread(artifact.fd, 4, 0), b"\x7fELF")

    def _records_case(self, directory: Path) -> tuple[Path, dict[str, Any]]:
        records = _lifecycle_records()
        expected = _expected(records)
        records_path = directory / "qualification-records.jsonl"
        _write_records(records_path, records)
        return records_path, expected

    def _run_authenticated(
        self,
        directory: Path,
        executable: Path,
        *,
        model: Path,
        capture_before: Any,
        capture_after: Any,
        prepare: Any,
    ) -> Any:
        records_path, expected = self._records_case(directory)
        with (
            open_qualification_artifact(executable, executable=True) as executable_artifact,
            open_qualification_artifact(model) as model_artifact,
            _safe_parent_environment(),
            _watchdog(AUTH_WATCHDOG_SECONDS, "native authenticated control"),
        ):
            return run_authenticated_qualification_child(
                ["--model-path", str(model), "--records-path", str(records_path)],
                expected,
                executable_artifact=executable_artifact,
                model_artifact=model_artifact,
                prepare_descriptor=prepare,
                capture_before=capture_before,
                capture_after=capture_after,
                first_token_timeout_ns=2_000_000_000,
                whole_request_timeout_ns=2_000_000_000,
                idle_timeout_ns=2_000_000_000,
                termination_grace_ns=50_000_000,
                control_timeout_seconds=2.0,
            )

    def test_authenticated_native_success_observes_real_proc_executable(self) -> None:
        directory = self._case_dir("authenticated-success")
        model = directory / "tiny-model.bin"
        model.write_bytes(b"tiny retained model descriptor\0" * 7)
        proc_observations: list[tuple[int, str, tuple[int, int, int, int]]] = []
        callback_pids: list[int] = []

        def proc_identity(pid: int) -> tuple[int, str, tuple[int, int, int, int]]:
            link = f"/proc/{pid}/exe"
            target = os.readlink(link)
            stat_result = os.stat(link)
            return pid, target, (
                stat_result.st_dev,
                stat_result.st_ino,
                stat_result.st_size,
                stat_result.st_mtime_ns,
            )

        executable_path = self._control

        def prepare(pid: int, model_fd: int, evidence: Any) -> dict[str, Any]:
            callback_pids.append(pid)
            observation = proc_identity(pid)
            proc_observations.append(observation)
            executable_stat = executable_path.stat()
            self.assertEqual(observation[2], (
                executable_stat.st_dev, executable_stat.st_ino,
                executable_stat.st_size, executable_stat.st_mtime_ns,
            ))
            self.assertEqual(os.fstat(model_fd).st_ino, evidence.identity.inode)
            return {"native": True, "pid": pid, "model_inode": evidence.identity.inode}

        def before(pid: int, sequence: int) -> dict[str, Any]:
            callback_pids.append(pid)
            observation = proc_identity(pid)
            proc_observations.append(observation)
            return {"pid": pid, "sequence": sequence, "phase": "before"}

        def after(pid: int, sequence: int) -> dict[str, Any]:
            callback_pids.append(pid)
            observation = proc_identity(pid)
            proc_observations.append(observation)
            return {"pid": pid, "sequence": sequence, "phase": "after"}

        result = self._run_authenticated(
            directory,
            executable_path,
            model=model,
            capture_before=before,
            capture_after=after,
            prepare=prepare,
        )
        self.assertEqual(result.transport.reason, "complete")
        self.assertTrue(result.transport.cleanup_complete)
        self.assertEqual(result.transport.returncode, 0)
        self.assertIsNone(result.control_error)
        self.assertEqual(len(result.checkpoints), 12)
        self.assertEqual(len(result.wire_records), 50)
        self.assertEqual(len(callback_pids), len(proc_observations))
        self.assertTrue(proc_observations)
        self.assertEqual({item[0] for item in proc_observations}, {callback_pids[0]})
        expected_executable_identity = os.stat(executable_path)
        expected_identity = (
            expected_executable_identity.st_dev,
            expected_executable_identity.st_ino,
            expected_executable_identity.st_size,
            expected_executable_identity.st_mtime_ns,
        )
        self.assertTrue(all(item[2] == expected_identity for item in proc_observations))
        self.assertTrue(all(item[1] for item in proc_observations))

    def test_authenticated_native_mismatched_executable_is_refused_independently(self) -> None:
        directory = self._case_dir("authenticated-mismatch")
        model = directory / "tiny-model.bin"
        model.write_bytes(b"independent retained model\0" * 5)
        actual_path = _materialize(self._template, directory, "native-control-actual")
        wrong_path = _materialize(self._template, directory, "native-control-wrong")
        mismatch_errors: list[str] = []
        callback_pids: list[int] = []

        def prepare(pid: int, model_fd: int, evidence: Any) -> dict[str, Any]:
            callback_pids.append(pid)
            return {"pid": pid, "inode": evidence.identity.inode}

        def before(pid: int, sequence: int) -> dict[str, Any]:
            callback_pids.append(pid)
            if sequence == 1:
                # This calls production authentication against the real
                # /proc/<owned-pid>/exe.  The wrong owner is independent, so a
                # failure cannot be masked by the successful case above.
                with open_qualification_artifact(wrong_path, executable=True) as wrong:
                    try:
                        authenticate_running_executable(pid, wrong)
                    except ValueError as exc:
                        mismatch_errors.append(str(exc))
                        raise
                self.fail("mismatched executable unexpectedly authenticated")
            return {"pid": pid, "sequence": sequence, "phase": "before"}

        def after(pid: int, sequence: int) -> dict[str, Any]:
            callback_pids.append(pid)
            return {"pid": pid, "sequence": sequence, "phase": "after"}

        result = self._run_authenticated(
            directory,
            actual_path,
            model=model,
            capture_before=before,
            capture_after=after,
            prepare=prepare,
        )
        self.assertTrue(callback_pids)
        self.assertTrue(mismatch_errors)
        self.assertIn("running executable", mismatch_errors[0])
        self.assertEqual(result.transport.reason, "protocol_error")
        self.assertTrue(result.transport.cleanup_complete)
        self.assertIsInstance(result.transport.returncode, int)
        self.assertIsNotNone(result.control_error)
        self.assertLessEqual(len(result.control_error or ""), 4096)


if __name__ == "__main__":
    unittest.main(verbosity=2)
