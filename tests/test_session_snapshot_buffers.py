#!/usr/bin/env python3
"""Host-only native tests for the session-snapshot fmemopen boundary.

The fixture compiles the checked-in ``ds4_session_save_snapshot`` function
body with only the session/payload boundaries stubbed.  It uses the host
libc's real ``fmemopen``/``fwrite``/``fclose`` path; no model, GPU backend, or
whole-engine build is involved.  A successful run is proof of this host's
libc behavior only; it is not a Linux, Metal, CUDA, or GPU proof.  The fixture
probes whether this host appends fmemopen's terminator instead of assuming the
Linux behavior.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DS4_SOURCE_PATH = ROOT / "ds4.c"
DS4_HEADER_PATH = ROOT / "ds4.h"
DS4_SOURCE = DS4_SOURCE_PATH.read_text(encoding="utf-8")
DS4_HEADER = DS4_HEADER_PATH.read_text(encoding="utf-8")
CC = Path("/usr/bin/cc")
FIXTURE_TIMEOUT_SECONDS = 10
PAYLOAD = bytes((0x31, 0x00, 0xA5, 0x42, 0xE7))

# The fixture is deliberately not allowed to inherit repository hooks,
# service selectors, preload paths, or other DOTFILES_* settings.
FIXTURE_ENV = {
    "PATH": "/usr/bin:/bin",
    "LANG": "C",
    "LC_ALL": "C",
}


def _matching_brace(source: str, brace: int) -> int:
    """Find a C brace pair while ignoring comments and literals."""

    if brace < 0 or source[brace] != "{":
        raise AssertionError("brace scanner did not start at an opening brace")
    depth = 0
    index = brace
    state = "code"
    while index < len(source):
        byte = source[index]
        following = source[index + 1] if index + 1 < len(source) else ""
        if state == "code":
            if byte == "/" and following == "/":
                state = "line-comment"
                index += 2
                continue
            if byte == "/" and following == "*":
                state = "block-comment"
                index += 2
                continue
            if byte == '"':
                state = "string"
                index += 1
                continue
            if byte == "'":
                state = "char"
                index += 1
                continue
            if byte == "{":
                depth += 1
            elif byte == "}":
                depth -= 1
                if depth == 0:
                    return index
            index += 1
            continue
        if state == "line-comment":
            if byte in "\r\n":
                state = "code"
            index += 1
            continue
        if state == "block-comment":
            if byte == "*" and following == "/":
                state = "code"
                index += 2
            else:
                index += 1
            continue
        if byte == "\\":
            index += 2
        elif (state == "string" and byte == '"') or (
            state == "char" and byte == "'"
        ):
            state = "code"
            index += 1
        else:
            index += 1
    raise AssertionError("unterminated C brace body")


def _first_code_brace(source: str, start: int) -> int:
    """Return the first opening brace after start outside C literals."""

    index = start
    state = "code"
    while index < len(source):
        byte = source[index]
        following = source[index + 1] if index + 1 < len(source) else ""
        if state == "code":
            if byte == "/" and following == "/":
                state = "line-comment"
                index += 2
                continue
            if byte == "/" and following == "*":
                state = "block-comment"
                index += 2
                continue
            if byte == '"':
                state = "string"
                index += 1
                continue
            if byte == "'":
                state = "char"
                index += 1
                continue
            if byte == "{":
                return index
            index += 1
            continue
        if state == "line-comment":
            if byte in "\r\n":
                state = "code"
            index += 1
            continue
        if state == "block-comment":
            if byte == "*" and following == "/":
                state = "code"
                index += 2
            else:
                index += 1
            continue
        if byte == "\\":
            index += 2
        elif (state == "string" and byte == '"') or (
            state == "char" and byte == "'"
        ):
            state = "code"
            index += 1
        else:
            index += 1
    raise AssertionError("no code brace after function signature")


def _extract_function(source: str, signature: str) -> str:
    """Extract the exact function text, including its signature and braces."""

    start = source.find(signature)
    if start < 0:
        raise AssertionError(f"missing source function {signature}")
    brace = _first_code_brace(source, start + len(signature))
    end = _matching_brace(source, brace)
    return source[start : end + 1]


def _extract_snapshot_record(header: str) -> str:
    """Extract the exact public snapshot record used by the function."""

    match = re.search(
        r"typedef struct \{\s*"
        r"uint8_t \*ptr;\s*"
        r"uint64_t len;\s*"
        r"uint64_t cap;\s*"
        r"\} ds4_session_snapshot;",
        header,
    )
    if match is None:
        raise AssertionError("missing ds4_session_snapshot record definition")
    return match.group(0)


SNAPSHOT_FUNCTION = _extract_function(
    DS4_SOURCE, "int ds4_session_save_snapshot(ds4_session *s,"
)
SNAPSHOT_RECORD = _extract_snapshot_record(DS4_HEADER)


def _fixture_source() -> str:
    """Build a tiny C translation unit around the extracted production body."""

    return f"""#define _GNU_SOURCE
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The production function only needs these two session flags in this fixture. */
typedef struct {{
    bool terminal;
    bool distributed;
}} ds4_session;

{SNAPSHOT_RECORD}

static const uint8_t k_payload[] = {{ 0x31, 0x00, 0xA5, 0x42, 0xE7 }};
static uint64_t g_payload_bytes;
static size_t g_realloc_calls;
static size_t g_last_realloc_size;
static size_t g_fmemopen_calls;
static size_t g_last_fmemopen_size;
static int g_refuse_realloc;
static int g_writer_calls;

static void payload_set_err(char *err, size_t errlen, const char *message) {{
    if (err != NULL && errlen != 0) {{
        (void)snprintf(err, errlen, "%s", message);
    }}
}}

static bool ds4_session_is_logits_only_terminal(const ds4_session *s) {{
    return s != NULL && s->terminal;
}}

static int ds4_session_terminal_error(char *err, size_t errlen) {{
    payload_set_err(err, errlen, "terminal session cannot save a snapshot");
    return 1;
}}

uint64_t ds4_session_payload_bytes(ds4_session *s) {{
    (void)s;
    return g_payload_bytes;
}}

int ds4_session_save_payload(ds4_session *s, FILE *fp, char *err, size_t errlen) {{
    (void)s;
    g_writer_calls++;
    if (fwrite(k_payload, 1, sizeof(k_payload), fp) != sizeof(k_payload)) {{
        payload_set_err(err, errlen, "fixture payload write failed");
        return 1;
    }}
    return 0;
}}

/* Defined before the macro so the spy can delegate only ordinary tiny sizes. */
static void *snapshot_test_realloc(void *ptr, size_t size) {{
    g_realloc_calls++;
    g_last_realloc_size = size;
    /* Huge requests are refused before libc realloc can attempt an allocation. */
    if (g_refuse_realloc || size > 4096u) return NULL;
    return realloc(ptr, size);
}}

static FILE *snapshot_test_fmemopen(void *ptr, size_t size, const char *mode) {{
    g_fmemopen_calls++;
    g_last_fmemopen_size = size;
    return fmemopen(ptr, size, mode);
}}

#define realloc snapshot_test_realloc
#define fmemopen snapshot_test_fmemopen
{SNAPSHOT_FUNCTION}
#undef fmemopen
#undef realloc

/* Darwin's fmemopen may leave the spare byte untouched; Linux commonly writes
 * its terminator on flush.  Probe the host instead of pretending either one. */
static int host_fmemopen_writes_terminator(void) {{
    uint8_t probe[2] = {{ 0xA5, 0xA5 }};
    FILE *fp = fmemopen(probe, sizeof(probe), "wb");
    if (fp == NULL) return -1;
    if (fwrite("x", 1, 1, fp) != 1) {{
        (void)fclose(fp);
        return -1;
    }}
    if (fclose(fp) != 0) return -1;
    return probe[1] == 0;
}}

static void reset_spies(void) {{
    g_payload_bytes = sizeof(k_payload);
    g_realloc_calls = 0;
    g_last_realloc_size = 0;
    g_fmemopen_calls = 0;
    g_last_fmemopen_size = 0;
    g_refuse_realloc = 0;
    g_writer_calls = 0;
}}

static int fail_case(const char *case_name, const char *reason) {{
    (void)fprintf(stderr, "%s: %s\\n", case_name, reason);
    return 1;
}}

#define NEED(CASE, CONDITION, REASON) \\
    do {{ \\
        if (!(CONDITION)) return fail_case((CASE), (REASON)); \\
    }} while (0)

static int run_controls(void) {{
    const char *case_name = "controls";
    reset_spies();
    uint8_t bytes[sizeof(k_payload) + 3];
    (void)memset(bytes, 0xA5, sizeof(bytes));
    FILE *fp = fmemopen(bytes, sizeof(bytes), "wb");
    NEED(case_name, fp != NULL, "host fmemopen unavailable");
    char err[128] = {{0}};
    ds4_session session = {{0}};
    NEED(case_name, ds4_session_save_payload(&session, fp, err, sizeof(err)) == 0,
         "fixture writer did not accept a real memory stream");
    NEED(case_name, fclose(fp) == 0, "host fmemopen close failed");
    NEED(case_name, g_writer_calls == 1, "fixture writer call was not observed");
    NEED(case_name, memcmp(bytes, k_payload, sizeof(k_payload)) == 0,
         "fixture payload oracle was not written exactly");
    int host_terminator = host_fmemopen_writes_terminator();
    NEED(case_name, host_terminator >= 0, "host fmemopen behavior probe failed");
    NEED(case_name, host_terminator == 0 || bytes[sizeof(k_payload)] == 0,
         "host fmemopen probe disagreed with its spare-byte behavior");
    (void)printf("host_fmemopen_terminator=%d\\n", host_terminator);

    /* A separate real file must contain only the payload, never its memory
     * stream's optional terminator.  The literal oracle is independent of
     * the production snapshot allocation and length calculation. */
    reset_spies();
    FILE *disk = fopen("payload-control.bin", "w+b");
    NEED(case_name, disk != NULL, "file oracle could not be opened");
    NEED(case_name, ds4_session_save_payload(&session, disk, err, sizeof(err)) == 0,
         "file oracle writer failed");
    NEED(case_name, fflush(disk) == 0, "file oracle flush failed");
    NEED(case_name, ftell(disk) == (long)sizeof(k_payload),
         "file oracle serialized an unexpected length");
    NEED(case_name, fseek(disk, 0, SEEK_SET) == 0, "file oracle seek failed");
    uint8_t disk_bytes[sizeof(k_payload) + 1] = {{0}};
    NEED(case_name, fread(disk_bytes, 1, sizeof(disk_bytes), disk) == sizeof(k_payload),
         "file oracle length includes extra bytes or is truncated");
    NEED(case_name, memcmp(disk_bytes, k_payload, sizeof(k_payload)) == 0 &&
             memcmp(disk_bytes, bytes, sizeof(k_payload)) == 0,
         "file and memory payload oracles disagree");
    NEED(case_name, fclose(disk) == 0, "file oracle close failed");
    NEED(case_name, remove("payload-control.bin") == 0, "file oracle cleanup failed");

    reset_spies();
    ds4_session_snapshot snap = {{0}};
    memset(err, 0, sizeof(err));
    NEED(case_name, ds4_session_save_snapshot(NULL, &snap, err, sizeof(err)) != 0,
         "invalid-session control unexpectedly succeeded");
    NEED(case_name, strstr(err, "invalid") != NULL,
         "invalid-session control used the wrong diagnostic");
    NEED(case_name, g_realloc_calls == 0 && g_fmemopen_calls == 0 &&
             g_writer_calls == 0,
         "invalid-session control reached a payload or allocator stub");
    return 0;
}}

static int run_fresh(void) {{
    const char *case_name = "fresh";
    reset_spies();
    ds4_session session = {{0}};
    ds4_session_snapshot snap = {{0}};
    char err[128] = {{0}};
    const uint64_t want = (uint64_t)sizeof(k_payload);
    int host_terminator = host_fmemopen_writes_terminator();
    NEED(case_name, host_terminator >= 0, "host fmemopen behavior probe failed");
    NEED(case_name, ds4_session_save_snapshot(&session, &snap, err, sizeof(err)) == 0,
         err[0] == 0 ? "fresh snapshot failed" : err);
    NEED(case_name, snap.len == want, "serialized len is not exact payload bytes");
    NEED(case_name, snap.cap == want + 1, "fresh capacity does not reserve a terminator");
    NEED(case_name, g_realloc_calls == 1 && g_last_realloc_size == (size_t)(want + 1),
         "fresh allocation did not request payload bytes plus one");
    NEED(case_name, g_fmemopen_calls == 1 && g_last_fmemopen_size == (size_t)(want + 1),
         "fresh fmemopen size did not include the terminator slot");
    NEED(case_name, g_writer_calls == 1, "fresh snapshot did not call the writer");
    NEED(case_name, memcmp(snap.ptr, k_payload, sizeof(k_payload)) == 0,
         "final serialized payload byte was not preserved");
    if (host_terminator) {{
        NEED(case_name, snap.ptr[want] == 0,
             "fresh snapshot terminator is not outside payload");
    }}
    free(snap.ptr);
    return 0;
}}

static int run_reuse(void) {{
    const char *case_name = "reuse";
    reset_spies();
    ds4_session session = {{0}};
    const uint64_t cap = (uint64_t)sizeof(k_payload) + 3;
    ds4_session_snapshot snap = {{malloc((size_t)cap), sizeof(k_payload), cap}};
    char err[128] = {{0}};
    NEED(case_name, snap.ptr != NULL, "reuse owner allocation failed");
    (void)memset(snap.ptr, 0xA5, (size_t)cap);
    uint8_t *owner = snap.ptr;
    int host_terminator = host_fmemopen_writes_terminator();
    NEED(case_name, host_terminator >= 0, "host fmemopen behavior probe failed");
    NEED(case_name, ds4_session_save_snapshot(&session, &snap, err, sizeof(err)) == 0,
         err[0] == 0 ? "reuse snapshot failed" : err);
    NEED(case_name, snap.ptr == owner && snap.cap == cap, "capacity was not safely reused");
    NEED(case_name, snap.len == sizeof(k_payload), "reuse len is not exact payload bytes");
    NEED(case_name, g_realloc_calls == 0, "reuse unexpectedly reallocated its owner");
    NEED(case_name, g_fmemopen_calls == 1 &&
             g_last_fmemopen_size == sizeof(k_payload) + 1,
         "reuse fmemopen size did not include the terminator slot");
    NEED(case_name, memcmp(snap.ptr, k_payload, sizeof(k_payload)) == 0,
         "reuse did not preserve the final nonzero payload byte");
    NEED(case_name,
         host_terminator ? snap.ptr[sizeof(k_payload)] == 0
                         : snap.ptr[sizeof(k_payload)] == 0xA5,
         host_terminator ? "host fmemopen terminator was not written outside payload"
                         : "host fmemopen changed the spare byte unexpectedly");
    NEED(case_name, snap.ptr[sizeof(k_payload) + 1] == 0xA5,
         "reuse wrote beyond the reserved terminator slot");
    free(snap.ptr);
    return 0;
}}

static int run_growth(void) {{
    const char *case_name = "growth";
    reset_spies();
    ds4_session session = {{0}};
    ds4_session_snapshot snap = {{malloc(2), 2, 2}};
    char err[128] = {{0}};
    int host_terminator = host_fmemopen_writes_terminator();
    NEED(case_name, host_terminator >= 0, "host fmemopen behavior probe failed");
    NEED(case_name, snap.ptr != NULL, "growth owner allocation failed");
    snap.ptr[0] = 0xC1;
    snap.ptr[1] = 0xC2;
    NEED(case_name, ds4_session_save_snapshot(&session, &snap, err, sizeof(err)) == 0,
         err[0] == 0 ? "growth snapshot failed" : err);
    NEED(case_name, snap.cap == sizeof(k_payload) + 1,
         "growth capacity did not reserve a terminator");
    NEED(case_name, g_realloc_calls == 1 &&
             g_last_realloc_size == sizeof(k_payload) + 1,
         "growth allocation did not request payload bytes plus one");
    NEED(case_name, snap.len == sizeof(k_payload), "growth len is not exact payload bytes");
    NEED(case_name, memcmp(snap.ptr, k_payload, sizeof(k_payload)) == 0,
         "growth did not preserve the final payload byte");
    if (host_terminator) {{
        NEED(case_name, snap.ptr[sizeof(k_payload)] == 0,
             "growth terminator is not outside payload");
    }}
    free(snap.ptr);
    return 0;
}}

static int run_allocation_failure(void) {{
    const char *case_name = "allocation-failure";
    reset_spies();
    g_refuse_realloc = 1;
    ds4_session session = {{0}};
    ds4_session_snapshot snap = {{malloc(2), 2, 2}};
    char err[128] = {{0}};
    NEED(case_name, snap.ptr != NULL, "failure owner allocation failed");
    snap.ptr[0] = 0x9C;
    snap.ptr[1] = 0x3D;
    uint8_t *owner = snap.ptr;
    NEED(case_name, ds4_session_save_snapshot(&session, &snap, err, sizeof(err)) != 0,
         "refused allocation unexpectedly succeeded");
    NEED(case_name, g_realloc_calls == 1 &&
             g_last_realloc_size == sizeof(k_payload) + 1,
         "allocation failure did not probe the bytes-plus-one request");
    NEED(case_name, g_fmemopen_calls == 0 && g_writer_calls == 0,
         "allocation failure reached the memory stream or writer");
    NEED(case_name, snap.ptr == owner && snap.len == 2 && snap.cap == 2,
         "allocation failure changed the snapshot owner state");
    NEED(case_name, snap.ptr[0] == 0x9C && snap.ptr[1] == 0x3D,
         "allocation failure changed existing owner bytes");
    NEED(case_name, strstr(err, "out of memory") != NULL,
         "allocation failure used the wrong diagnostic");
    free(snap.ptr);
    return 0;
}}

static int run_size_max(void) {{
    const char *case_name = "size-max";
    reset_spies();
    ds4_session session = {{0}};
    ds4_session_snapshot snap = {{malloc(8), 8, 8}};
    char err[128] = {{0}};
    NEED(case_name, snap.ptr != NULL, "SIZE_MAX owner allocation failed");
    (void)memset(snap.ptr, 0xD4, 8);
    uint8_t *owner = snap.ptr;
    g_payload_bytes = SIZE_MAX;
    NEED(case_name, ds4_session_save_snapshot(&session, &snap, err, sizeof(err)) != 0,
         "SIZE_MAX payload unexpectedly succeeded");
    NEED(case_name, g_realloc_calls == 0,
         "SIZE_MAX refusal happened after entering the allocator");
    NEED(case_name, g_fmemopen_calls == 0 && g_writer_calls == 0,
         "SIZE_MAX refusal reached the memory stream or writer");
    NEED(case_name, snap.ptr == owner && snap.len == 8 && snap.cap == 8,
         "SIZE_MAX refusal changed the snapshot owner state");
    NEED(case_name, snap.ptr[0] == 0xD4 && snap.ptr[7] == 0xD4,
         "SIZE_MAX refusal changed existing owner bytes");
    NEED(case_name, strstr(err, "too large") != NULL,
         "SIZE_MAX refusal used the wrong diagnostic");
    free(snap.ptr);
    return 0;
}}

static int run_guards(void) {{
    const char *case_name = "guards";
    char err[128] = {{0}};
    ds4_session_snapshot snap;
    ds4_session session;

    reset_spies();
    memset(&snap, 0, sizeof(snap));
    NEED(case_name, ds4_session_save_snapshot(NULL, &snap, err, sizeof(err)) != 0,
         "null session unexpectedly succeeded");
    NEED(case_name, strstr(err, "invalid") != NULL && g_realloc_calls == 0 &&
             g_fmemopen_calls == 0 && g_writer_calls == 0,
         "null session did not stop before side effects");

    reset_spies();
    memset(&session, 0, sizeof(session));
    memset(err, 0, sizeof(err));
    NEED(case_name, ds4_session_save_snapshot(&session, NULL, err, sizeof(err)) != 0,
         "null snapshot unexpectedly succeeded");
    NEED(case_name, strstr(err, "invalid") != NULL && g_realloc_calls == 0 &&
             g_fmemopen_calls == 0 && g_writer_calls == 0,
         "null snapshot did not stop before side effects");

    reset_spies();
    session.distributed = true;
    memset(&snap, 0, sizeof(snap));
    memset(err, 0, sizeof(err));
    NEED(case_name, ds4_session_save_snapshot(&session, &snap, err, sizeof(err)) != 0,
         "distributed snapshot unexpectedly succeeded");
    NEED(case_name, strstr(err, "distributed") != NULL && g_realloc_calls == 0 &&
             g_fmemopen_calls == 0 && g_writer_calls == 0,
         "distributed guard did not stop before side effects");

    reset_spies();
    session.distributed = false;
    memset(&snap, 0, sizeof(snap));
    snap.ptr = malloc(4);
    snap.len = 4;
    snap.cap = 4;
    g_payload_bytes = 0;
    memset(err, 0, sizeof(err));
    NEED(case_name, snap.ptr != NULL, "zero-payload owner allocation failed");
    NEED(case_name, ds4_session_save_snapshot(&session, &snap, err, sizeof(err)) != 0,
         "zero payload unexpectedly succeeded");
    NEED(case_name, strstr(err, "no valid checkpoint") != NULL &&
             g_realloc_calls == 0 && g_fmemopen_calls == 0 && g_writer_calls == 0,
         "zero-payload guard did not stop before side effects");
    NEED(case_name, snap.len == 4 && snap.cap == 4,
         "zero-payload guard changed owner metadata");
    free(snap.ptr);

    reset_spies();
    session.terminal = true;
    memset(&snap, 0, sizeof(snap));
    memset(err, 0, sizeof(err));
    NEED(case_name, ds4_session_save_snapshot(&session, &snap, err, sizeof(err)) != 0,
         "terminal snapshot unexpectedly succeeded");
    NEED(case_name, strstr(err, "terminal") != NULL && g_realloc_calls == 0 &&
             g_fmemopen_calls == 0 && g_writer_calls == 0,
         "terminal guard did not stop before side effects");
    return 0;
}}

int main(int argc, char **argv) {{
    if (argc != 2) {{
        (void)fprintf(stderr, "usage: fixture CASE\\n");
        return 2;
    }}
    if (strcmp(argv[1], "controls") == 0) return run_controls();
    if (strcmp(argv[1], "fresh") == 0) return run_fresh();
    if (strcmp(argv[1], "reuse") == 0) return run_reuse();
    if (strcmp(argv[1], "growth") == 0) return run_growth();
    if (strcmp(argv[1], "allocation-failure") == 0) return run_allocation_failure();
    if (strcmp(argv[1], "size-max") == 0) return run_size_max();
    if (strcmp(argv[1], "guards") == 0) return run_guards();
    (void)fprintf(stderr, "unknown fixture case: %s\\n", argv[1]);
    return 2;
}}
"""


class SessionSnapshotBufferNativeTest(unittest.TestCase):
    """Compile and run the small host-only C body fixture."""

    maxDiff = None
    _tempdir: tempfile.TemporaryDirectory[str]
    _fixture: Path
    _compile_output: subprocess.CompletedProcess[str]

    @classmethod
    def setUpClass(cls) -> None:
        if not CC.is_file() or not os.access(CC, os.X_OK):
            raise unittest.SkipTest(f"required host compiler is unavailable: {CC}")
        session_dir = os.environ.get("RLM_SESSION_DIR")
        artifact_parent = None
        if session_dir:
            artifact_parent = Path(session_dir) / "session-snapshot-fixture"
            artifact_parent.mkdir(parents=True, exist_ok=True)
        cls._tempdir = tempfile.TemporaryDirectory(
            prefix="ds4-session-snapshot-",
            dir=str(artifact_parent) if artifact_parent is not None else None,
        )
        directory = Path(cls._tempdir.name)
        source_path = directory / "fixture.c"
        cls._fixture = directory / "fixture"
        source_path.write_text(_fixture_source(), encoding="utf-8")
        command = [
            str(CC),
            "-std=c11",
            "-Wall",
            "-Wextra",
            "-Werror",
            str(source_path),
            "-o",
            str(cls._fixture),
        ]
        try:
            cls._compile_output = subprocess.run(
                command,
                cwd=directory,
                env=FIXTURE_ENV,
                capture_output=True,
                text=True,
                timeout=FIXTURE_TIMEOUT_SECONDS,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            cls._tempdir.cleanup()
            raise RuntimeError(f"fixture compiler invocation failed: {exc}") from exc
        if cls._compile_output.returncode != 0:
            diagnostics = (
                f"compiler command: {' '.join(command)}\n"
                f"stdout:\n{cls._compile_output.stdout}\n"
                f"stderr:\n{cls._compile_output.stderr}"
            )
            cls._tempdir.cleanup()
            raise RuntimeError(f"fixture compile failed; not a behavioral RED:\n{diagnostics}")

        # These checks establish the real libc/stub controls before any
        # production-behavior assertion is allowed to report RED.
        control = cls._run_fixture("controls")
        cls._fixture_controls = control
        if control.returncode != 0:
            diagnostics = cls._diagnostics(control)
            cls._tempdir.cleanup()
            raise RuntimeError(f"fixture controls failed; not a behavioral RED:\n{diagnostics}")

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "_tempdir"):
            cls._tempdir.cleanup()

    @classmethod
    def _run_fixture(cls, case: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(cls._fixture), case],
            cwd=cls._fixture.parent,
            env=FIXTURE_ENV,
            capture_output=True,
            text=True,
            timeout=FIXTURE_TIMEOUT_SECONDS,
            check=False,
        )

    @staticmethod
    def _diagnostics(result: subprocess.CompletedProcess[str]) -> str:
        return (
            f"exit={result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    def assertFixtureCase(self, case: str) -> None:
        result = self._run_fixture(case)
        self.assertEqual(
            result.returncode,
            0,
            f"fixture case {case} failed:\n{self._diagnostics(result)}",
        )

    def test_fixture_controls_precede_behavioral_cases(self) -> None:
        """The class setup already gates all behavior cases on this control."""
        self.assertFixtureCase("controls")

    def test_fresh_allocation_reserves_terminator_and_exact_len(self) -> None:
        self.assertFixtureCase("fresh")

    def test_capacity_reuse_preserves_final_nonzero_byte(self) -> None:
        self.assertFixtureCase("reuse")

    def test_growth_reserves_terminator(self) -> None:
        self.assertFixtureCase("growth")

    def test_allocation_failure_preserves_owner_state(self) -> None:
        self.assertFixtureCase("allocation-failure")

    def test_size_max_refuses_before_allocator_or_memstream(self) -> None:
        self.assertFixtureCase("size-max")

    def test_invalid_zero_distributed_and_terminal_guards(self) -> None:
        self.assertFixtureCase("guards")


if __name__ == "__main__":
    unittest.main()
