#!/usr/bin/env python3
"""Host-only TDD contract for CUDA model staging span and read safety.

The C++ fixtures below are intentionally small.  They compile the actual
production function bodies extracted from ``ds4_cuda.cu``; they do not copy
those algorithms.  Direct-I/O cases are a host preprocessor simulation, not
proof of Linux ``O_DIRECT`` behavior.
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CUDA_SOURCE = (ROOT / "ds4_cuda.cu").read_text(encoding="utf-8")
SAFE_ENV = {
    "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
    "LANG": "C",
    "LC_ALL": "C",
}

def extract_definition(source: str, signature: str) -> str | None:
    """Extract one definition while ignoring braces in comments and strings."""
    start = source.rfind(signature)
    if start < 0:
        return None
    i = start + len(signature)
    state = "code"
    while i < len(source):
        c = source[i]
        n = source[i + 1] if i + 1 < len(source) else ""
        if state == "code":
            if c == "/" and n == "/": state, i = "line", i + 2; continue
            if c == "/" and n == "*": state, i = "block", i + 2; continue
            if c == '"': state, i = "string", i + 1; continue
            if c == "'": state, i = "char", i + 1; continue
            if c == "{": break
            i += 1
            continue
        if state == "line":
            if c in "\r\n": state = "code"
            i += 1
            continue
        if state == "block":
            if c == "*" and n == "/": state, i = "code", i + 2
            else: i += 1
            continue
        if c == "\\": i += 2
        elif (state == "string" and c == '"') or (state == "char" and c == "'"):
            state, i = "code", i + 1
        else: i += 1
    else:
        raise AssertionError(f"no body brace after {signature}")
    brace = i
    depth = 0
    state = "code"
    while i < len(source):
        c = source[i]
        n = source[i + 1] if i + 1 < len(source) else ""
        if state == "code":
            if c == "/" and n == "/": state, i = "line", i + 2; continue
            if c == "/" and n == "*": state, i = "block", i + 2; continue
            if c == '"': state, i = "string", i + 1; continue
            if c == "'": state, i = "char", i + 1; continue
            if c == "{": depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0: return source[start:i + 1]
            i += 1
            continue
        if state == "line":
            if c in "\r\n": state = "code"
            i += 1
            continue
        if state == "block":
            if c == "*" and n == "/": state, i = "code", i + 2
            else: i += 1
            continue
        if c == "\\": i += 2
        elif (state == "string" and c == '"') or (state == "char" and c == "'"):
            state, i = "code", i + 1
        else: i += 1
    raise AssertionError(f"unterminated definition {signature}")

def code_only(text: str) -> str:
    """Blank comments and literals so source-call assertions cannot match them."""
    out: list[str] = []
    i = 0
    state = "code"
    while i < len(text):
        c = text[i]
        n = text[i + 1] if i + 1 < len(text) else ""
        if state == "code":
            if c == "/" and n == "/": out.extend("  "); state, i = "line", i + 2; continue
            if c == "/" and n == "*": out.extend("  "); state, i = "block", i + 2; continue
            if c == '"': out.append(" "); state, i = "string", i + 1; continue
            if c == "'": out.append(" "); state, i = "char", i + 1; continue
            out.append(c); i += 1; continue
        if state == "line":
            out.append("\n" if c in "\r\n" else " ")
            if c in "\r\n": state = "code"
            i += 1; continue
        if state == "block":
            out.append("\n" if c in "\r\n" else " ")
            if c == "*" and n == "/": out.append(" "); state, i = "code", i + 2
            else: i += 1
            continue
        if c == "\\": out.extend("  "); i += 2; continue
        out.append(" ")
        if (state == "string" and c == '"') or (state == "char" and c == "'"):
            state = "code"
        i += 1
    return "".join(out)

HELPER = extract_definition(CUDA_SOURCE, "static uint64_t cuda_stage_usable_bytes(")
ROUND_DOWN = extract_definition(CUDA_SOURCE, "static uint64_t cuda_round_down(")
ROUND_UP = extract_definition(CUDA_SOURCE, "static uint64_t cuda_round_up(")
STAGE_READ = extract_definition(CUDA_SOURCE, "static int cuda_model_stage_read(")
STREAM_CALLER = extract_definition(CUDA_SOURCE, "static int cuda_model_copy_to_device_streamed(")
RANGE_CALLER = extract_definition(CUDA_SOURCE, "static const char *cuda_model_range_ptr_from_fd(")

HELPER_PREFIX = r"""
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <csignal>
#include <sys/resource.h>
#include <unistd.h>
#ifndef SIZE_MAX
#define SIZE_MAX UINT64_MAX
#endif
"""
HELPER_SUFFIX = r"""
alignas(64) static unsigned char arena[128];
static void configure_process(void) {
    alarm(15);
    struct rlimit limit = {0, 0};
    (void)setrlimit(RLIMIT_CORE, &limit);
}
static void emit(uint64_t value) { std::printf("result=%llu\n", (unsigned long long)value); }
static uint64_t run(const char *name) {
    const uintptr_t top = UINTPTR_MAX;
    if (std::strcmp(name, "unaligned") == 0)
        return cuda_stage_usable_bytes(arena + 3, arena + 16, 64);
    if (std::strcmp(name, "before") == 0)
        return cuda_stage_usable_bytes(arena + 16, arena + 8, 32);
    if (std::strcmp(name, "end") == 0)
        return cuda_stage_usable_bytes(arena + 16, arena + 48, 32);
    if (std::strcmp(name, "outside") == 0)
        return cuda_stage_usable_bytes(arena + 16, arena + 49, 32);
    if (std::strcmp(name, "null-raw") == 0)
        return cuda_stage_usable_bytes(nullptr, arena, 32);
    if (std::strcmp(name, "null-stage") == 0)
        return cuda_stage_usable_bytes(arena, nullptr, 32);
    if (std::strcmp(name, "zero") == 0)
        return cuda_stage_usable_bytes(arena, arena, 0);
    if (std::strcmp(name, "span-overflow") == 0)
        return cuda_stage_usable_bytes(reinterpret_cast<void *>(top - 3),
                                       reinterpret_cast<void *>(top - 3), 8);
    if (std::strcmp(name, "max-end") == 0)
        return cuda_stage_usable_bytes(reinterpret_cast<void *>(top - 7),
                                       reinterpret_cast<void *>(top - 4), 8);
    if (std::strcmp(name, "max-valid") == 0)
        return cuda_stage_usable_bytes(reinterpret_cast<void *>(top),
                                       reinterpret_cast<void *>(top), 1);
    return UINT64_MAX;
}
int main(int argc, char **argv) {
    configure_process();
    if (argc != 2) return 2;
    emit(run(argv[1]));
    return 0;
}
"""

READ_PREFIX = r"""
#include <cstddef>
#include <cstdint>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <csignal>
#include <sys/resource.h>
#include <unistd.h>
#ifndef SIZE_MAX
#define SIZE_MAX UINT64_MAX
#endif

static int g_model_direct_fd = 17;
static int g_model_fd = 23;
static uint64_t g_model_direct_align = 16;
static uint64_t g_model_file_size = 96;
static unsigned char fake_file[128];
static const uint64_t fake_physical_file_bytes = sizeof(fake_file);
static uint64_t fake_virtual_file_base, fake_virtual_file_bytes;
alignas(64) static unsigned char stage_storage[128];
struct fake_span { uintptr_t base; uint64_t bytes; };
static fake_span spans[4];
static size_t span_count;
static int io_calls, direct_calls, buffered_calls, close_calls, direct_closed;
static int control_ok;
static uint64_t last_io_bytes, last_io_offset;
static int last_io_fd;
static int fake_direct_errno, fake_buffered_errno;
static const char *sentinel = reinterpret_cast<const char *>(uintptr_t(0x12345));
static const int entry_errno = ECHILD;

static void register_span(void *base, uint64_t bytes) {
    spans[span_count++] = {reinterpret_cast<uintptr_t>(base), bytes};
}
static bool registered_range(const void *ptr, uint64_t bytes) {
    if (!ptr) return false;
    const uintptr_t address = reinterpret_cast<uintptr_t>(ptr);
    for (size_t i = 0; i < span_count; ++i) {
        if (address >= spans[i].base && address - spans[i].base <= spans[i].bytes &&
            bytes <= spans[i].bytes - (address - spans[i].base)) return true;
    }
    return false;
}
static unsigned char fake_virtual_byte(uint64_t offset) {
    return static_cast<unsigned char>((offset ^ UINT64_C(0xA5)) & UINT64_C(0xff));
}
static int fake_cuda_close(int fd) {
    ++close_calls;
    if (fd == 17) direct_closed = 1;
    return 0;
}
static int fake_cuda_pread_full(int fd, void *buf, uint64_t bytes, uint64_t offset) {
    ++io_calls;
    if (fd == 17) ++direct_calls; else if (fd == 23) ++buffered_calls;
    last_io_fd = fd; last_io_bytes = bytes; last_io_offset = offset;
    if (fd != 17 && fd != 23) { errno = EBADF; return 0; }
    if (fd == 17 && fake_direct_errno) { errno = fake_direct_errno; return 0; }
    if (fd == 23 && fake_buffered_errno) { errno = fake_buffered_errno; return 0; }
    if (offset > g_model_file_size || bytes > g_model_file_size - offset ||
        bytes > static_cast<uint64_t>(SIZE_MAX) || !registered_range(buf, bytes)) {
        errno = EFAULT;
        return 0;
    }
    if (bytes != 0 && offset > UINT64_MAX - (bytes - 1)) {
        errno = EOVERFLOW;
        return 0;
    }
    const bool physical = offset <= fake_physical_file_bytes &&
                          bytes <= fake_physical_file_bytes - offset;
    const bool virtual_file = fake_virtual_file_bytes != 0 &&
                              offset >= fake_virtual_file_base &&
                              offset - fake_virtual_file_base <= fake_virtual_file_bytes &&
                              bytes <= fake_virtual_file_bytes - (offset - fake_virtual_file_base);
    if (!physical && !virtual_file) { errno = EFAULT; return 0; }
    if (physical) {
        if (bytes != 0) std::memcpy(buf, fake_file + static_cast<size_t>(offset), static_cast<size_t>(bytes));
    } else {
        unsigned char *out = static_cast<unsigned char *>(buf);
        for (uint64_t i = 0; i < bytes; ++i) out[i] = fake_virtual_byte(offset + i);
    }
    if (fd == 17) errno = ERANGE;
    return 1;
}
static char *fake_getenv(const char *) { return nullptr; }

"""
READ_SUFFIX = r"""
#undef cuda_pread_full
#undef close
#undef getenv
static void configure_process(void) {
    alarm(15);
    struct rlimit limit = {0, 0};
    (void)setrlimit(RLIMIT_CORE, &limit);
}
static bool pointer_equal(const char *a, const void *b) {
    return b && reinterpret_cast<uintptr_t>(a) == reinterpret_cast<uintptr_t>(b);
}
static bool pointer_plus_equal(const char *value, const void *base, uint64_t delta) {
    if (!base) return false;
    const uintptr_t b = reinterpret_cast<uintptr_t>(base);
    if (delta > static_cast<uint64_t>(UINTPTR_MAX - b)) return false;
    return reinterpret_cast<uintptr_t>(value) == b + static_cast<uintptr_t>(delta);
}
static int payload_byte(const char *payload) {
    if (!registered_range(payload, 1)) return -1;
    return static_cast<unsigned char>(*payload);
}
static void reset_case(uint64_t stage_span = 64) {
    std::memset(stage_storage, 0xCC, sizeof(stage_storage));
    for (size_t i = 0; i < sizeof(fake_file); ++i) fake_file[i] = static_cast<unsigned char>(i + 1);
    g_model_direct_fd = 17; g_model_fd = 23; g_model_direct_align = 16; g_model_file_size = 96;
    fake_virtual_file_base = fake_virtual_file_bytes = 0;
    span_count = 0; register_span(stage_storage + 32, stage_span);
    io_calls = direct_calls = buffered_calls = close_calls = direct_closed = 0;
    last_io_bytes = last_io_offset = 0; last_io_fd = -1;
    fake_direct_errno = fake_buffered_errno = 0; errno = EINTR;
}
static int valid_control(void) {
    reset_case();
    const char *payload = sentinel;
    const int result = cuda_model_stage_read(stage_storage + 32, 64, 0, 4, &payload);
    return result == 1 && pointer_equal(payload, stage_storage + 32) && io_calls == 1 && payload_byte(payload) == 1;
}
static void begin_mutant(uint64_t stage_span = 64) {
    control_ok = valid_control();
    std::printf("control=%d\n", control_ok);
    std::fflush(stdout);
    reset_case(stage_span);
}
static void emit(int result, const char *payload, const void *stage) {
    std::printf("result=%d\n", result);
    std::printf("io_calls=%d\n", io_calls);
    std::printf("direct_calls=%d\n", direct_calls);
    std::printf("buffered_calls=%d\n", buffered_calls);
    std::printf("close_calls=%d\n", close_calls);
    std::printf("direct_closed=%d\n", direct_closed);
    std::printf("direct_fd=%d\n", g_model_direct_fd);
    std::printf("direct_align=%llu\n", (unsigned long long)g_model_direct_align);
    std::printf("last_fd=%d\n", last_io_fd);
    std::printf("last_bytes=%llu\n", (unsigned long long)last_io_bytes);
    std::printf("last_offset=%llu\n", (unsigned long long)last_io_offset);
    std::printf("payload_is_sentinel=%d\n", pointer_equal(payload, sentinel));
    std::printf("payload_is_stage=%d\n", pointer_equal(payload, stage));
    std::printf("payload_is_delta=%d\n", pointer_plus_equal(payload, stage, 3));
    std::printf("payload_byte=%d\n", payload_byte(payload));
    std::printf("errno_restored=%d\n", errno == entry_errno);
}
static int scenario(const char *name) {
    const void *stage = stage_storage + 32;
    if (std::strcmp(name, "valid") == 0) {
        begin_mutant(); const char *payload = sentinel;
        int r = cuda_model_stage_read(const_cast<void *>(stage), 64, 0, 4, &payload);
        emit(r, payload, stage); return 0;
    }
    if (std::strcmp(name, "exact-capacity") == 0) {
        begin_mutant(8); const char *payload = sentinel;
        int r = cuda_model_stage_read(const_cast<void *>(stage), 8, 0, 8, &payload);
        emit(r, payload, stage); return 0;
    }
    if (std::strcmp(name, "direct-success") == 0) {
        begin_mutant(); const char *payload = sentinel; errno = ECHILD;
        int r = cuda_model_stage_read(const_cast<void *>(stage), 64, 19, 5, &payload);
        emit(r, payload, stage); return 0;
    }
    if (std::strcmp(name, "padding-capacity") == 0) {
        begin_mutant(8); const char *payload = sentinel;
        int r = cuda_model_stage_read(const_cast<void *>(stage), 8, 19, 5, &payload);
        emit(r, payload, stage); return 0;
    }
    if (std::strcmp(name, "padding-file") == 0) {
        begin_mutant(); g_model_file_size = 90; const char *payload = sentinel;
        int r = cuda_model_stage_read(const_cast<void *>(stage), 64, 87, 3, &payload);
        emit(r, payload, stage); return 0;
    }
    if (std::strcmp(name, "unsupported-fallback") == 0 || std::strcmp(name, "unsupported-efault") == 0 ||
        std::strcmp(name, "unsupported-enotsup") == 0 || std::strcmp(name, "unsupported-eopnotsupp") == 0) {
        begin_mutant();
        fake_direct_errno = std::strcmp(name, "unsupported-efault") == 0 ? EFAULT :
                            std::strcmp(name, "unsupported-enotsup") == 0 ? ENOTSUP :
                            std::strcmp(name, "unsupported-eopnotsupp") == 0 ? EOPNOTSUPP : EINVAL;
        const char *payload = sentinel;
        int r = cuda_model_stage_read(const_cast<void *>(stage), 64, 19, 5, &payload);
        emit(r, payload, stage); return 0;
    }
    if (std::strcmp(name, "direct-error-fallback") == 0) {
        begin_mutant(); fake_direct_errno = EIO; const char *payload = sentinel;
        int r = cuda_model_stage_read(const_cast<void *>(stage), 64, 19, 5, &payload);
        emit(r, payload, stage); return 0;
    }
    if (std::strcmp(name, "failed-fallback") == 0) {
        begin_mutant(); fake_direct_errno = EIO; fake_buffered_errno = EIO;
        const char *payload = sentinel;
        int r = cuda_model_stage_read(const_cast<void *>(stage), 64, 19, 5, &payload);
        emit(r, payload, stage); return 0;
    }
    if (std::strcmp(name, "zero-read") == 0) {
        begin_mutant(); const char *payload = sentinel;
        int r = cuda_model_stage_read(const_cast<void *>(stage), 64, 19, 0, &payload);
        emit(r, payload, stage); return 0;
    }
    if (std::strcmp(name, "payload-over-capacity") == 0) {
        begin_mutant(96); const char *payload = sentinel;
        int r = cuda_model_stage_read(const_cast<void *>(stage), 4, 0, 8, &payload);
        emit(r, payload, stage); return 0;
    }
    if (std::strcmp(name, "capacity-wrap") == 0) {
        begin_mutant(); const char *payload = sentinel;
        void *bad = reinterpret_cast<void *>(UINTPTR_MAX - 3);
        int r = cuda_model_stage_read(bad, 8, 0, 1, &payload);
        emit(r, payload, bad); return 0;
    }
    if (std::strcmp(name, "interval-wrap") == 0) {
        begin_mutant(); g_model_file_size = UINT64_MAX; const char *payload = sentinel;
        int r = cuda_model_stage_read(const_cast<void *>(stage), 64, UINT64_MAX - 3, 8, &payload);
        emit(r, payload, stage); return 0;
    }
    if (std::strcmp(name, "rounding-overflow") == 0) {
        begin_mutant();
        const uint64_t huge_align = (UINT64_MAX / 2u) + 2u;
        const uint64_t offset = huge_align - 1;
        g_model_direct_align = huge_align; g_model_file_size = UINT64_MAX;
        fake_virtual_file_base = offset; fake_virtual_file_bytes = 2;
        const char *payload = sentinel;
        int r = cuda_model_stage_read(const_cast<void *>(stage), 64, offset, 2, &payload);
        emit(r, payload, stage); return 0;
    }
    if (std::strcmp(name, "null-stage") == 0) {
        begin_mutant(); const char *payload = sentinel;
        int r = cuda_model_stage_read(nullptr, 64, 0, 1, &payload);
        emit(r, payload, nullptr); return 0;
    }
    if (std::strcmp(name, "null-output") == 0) {
        begin_mutant();
        int r = cuda_model_stage_read(const_cast<void *>(stage), 64, 0, 1, nullptr);
        emit(r, sentinel, stage); return 0;
    }
    return 2;
}
int main(int argc, char **argv) {
    configure_process();
    if (argc != 2) return 2;
    return scenario(argv[1]);
}
"""

def helper_fixture() -> str:
    return HELPER_PREFIX + (HELPER or "") + HELPER_SUFFIX

def read_fixture(direct: bool) -> str:
    if STAGE_READ is None or ROUND_DOWN is None or ROUND_UP is None:
        raise AssertionError("missing extracted stage-read or rounding definition")
    simulation = """
#ifdef __linux__
#undef __linux__
#endif
#ifdef O_DIRECT
#undef O_DIRECT
#endif
""" + ("#define __linux__ 1\n#define O_DIRECT 040000\n" if direct else "")
    return READ_PREFIX + simulation + ROUND_DOWN + ROUND_UP + r"""
#define cuda_pread_full fake_cuda_pread_full
#define close fake_cuda_close
#define getenv fake_getenv
""" + STAGE_READ + READ_SUFFIX

class HelperBuildContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory(prefix="ds4-stage-helper-contract-")
        cls.addClassCleanup(cls._tmp.cleanup)
        base = Path(cls._tmp.name)
        cls._source = base / "helper.cc"
        cls._binary = base / "helper"
        cls._source.write_text(helper_fixture(), encoding="utf-8")
        cls._compile = subprocess.run(
            ["c++", "-std=c++17", "-O0", str(cls._source), "-o", str(cls._binary)],
            cwd=ROOT, env=SAFE_ENV, capture_output=True, text=True,
            timeout=15, check=False,
        )
    def test_00_helper_definition_compiles(self) -> None:
        self.assertEqual(self._compile.returncode, 0,
                         "helper fixture compile RED:\n" + self._compile.stderr)
        self.assertIsNotNone(HELPER, "helper body-source RED: definition is absent")
    def _case(self, name: str) -> int:
        if self._compile.returncode != 0:
            self.fail("helper behavior RED: prerequisite fixture compile failed:\n" + self._compile.stderr)
        result = subprocess.run([str(self._binary), name], cwd=ROOT, env=SAFE_ENV,
                                capture_output=True, text=True, timeout=16, check=False)
        if result.returncode < 0:
            self.fail(f"helper case {name} died by signal {-result.returncode}")
        self.assertEqual(result.returncode, 0, result.stderr)
        values = {}
        for line in result.stdout.splitlines():
            key, sep, value = line.partition("=")
            if sep: values[key] = int(value)
        self.assertIn("result", values, result.stdout)
        return values["result"]
    def test_01_unaligned_stage_shrinks_capacity(self) -> None:
        self.assertEqual(self._case("unaligned"), 51)
    def test_02_invalid_before_end_outside_and_null_are_zero(self) -> None:
        for name in ("before", "end", "outside", "null-raw", "null-stage", "zero", "span-overflow"):
            with self.subTest(case=name): self.assertEqual(self._case(name), 0)
    def test_03_safe_max_scalar_boundaries_without_dereference(self) -> None:
        self.assertEqual(self._case("max-end"), 5)
        self.assertEqual(self._case("max-valid"), 1)
    def test_04_helper_body_is_pure_and_does_not_dereference_inputs(self) -> None:
        self.assertIsNotNone(HELPER, "helper body-source RED: definition is absent")
        body = HELPER[HELPER.find("{") + 1:]
        body = code_only(body)
        self.assertNotRegex(body, r"\b(?:malloc|calloc|realloc|free|new|delete)\b")
        self.assertNotRegex(body, r"\b(?:memcpy|memmove|memset|strcpy|strlen)\b")
        self.assertNotRegex(body, r"(?:\*\s*(?:raw|stage)|(?:raw|stage)\s*\[)")

class ExistingStageReadBehaviorContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory(prefix="ds4-stage-read-contract-")
        cls.addClassCleanup(cls._tmp.cleanup)
        cls._fixtures: dict[str, tuple[Path, object]] = {}
        for name, direct in (("direct-enabled", True), ("direct-disabled", False)):
            base = Path(cls._tmp.name); source = base / f"{name}.cc"; binary = base / name
            source.write_text(read_fixture(direct), encoding="utf-8")
            compile_result = subprocess.run(
                ["c++", "-std=c++17", "-O0", str(source), "-o", str(binary)],
                cwd=ROOT, env=SAFE_ENV, capture_output=True, text=True,
                timeout=15, check=False,
            )
            cls._fixtures[name] = (binary, compile_result)
    def _case(self, variant: str, name: str) -> dict[str, int]:
        binary, compile_result = self._fixtures[variant]
        if compile_result.returncode != 0:
            self.fail(f"existing-read behavior RED ({variant}) fixture build:\n{compile_result.stderr}")
        result = subprocess.run([str(binary), name], cwd=ROOT, env=SAFE_ENV,
                                capture_output=True, text=True, timeout=16, check=False)
        values: dict[str, int] = {}
        for line in result.stdout.splitlines():
            key, sep, value = line.partition("=")
            if sep: values[key] = int(value)
        self.assertEqual(values.get("control"), 1,
                         f"existing-read behavior RED ({variant}/{name}) control/stdout={result.stdout!r}")
        if result.returncode < 0:
            self.fail(f"existing-read behavior RED ({variant}/{name}): signal={-result.returncode}; stdout={result.stdout!r}")
        self.assertEqual(result.returncode, 0,
                         f"existing-read behavior RED ({variant}/{name}): {result.stderr}")
        return values
    def test_00_valid_control_runs_actual_body_in_both_host_variants(self) -> None:
        for variant in self._fixtures:
            with self.subTest(variant=variant):
                values = self._case(variant, "valid")
                self.assertEqual(values["result"], 1); self.assertEqual(values["payload_byte"], 1)
    def test_01_direct_success_returns_delta_errno_and_payload(self) -> None:
        values = self._case("direct-enabled", "direct-success")
        self.assertEqual(values["result"], 1); self.assertEqual(values["direct_calls"], 1)
        self.assertEqual(values["buffered_calls"], 0); self.assertEqual(values["payload_is_delta"], 1)
        self.assertEqual(values["payload_byte"], 20); self.assertEqual(values["last_bytes"], 16)
        self.assertEqual(values["last_offset"], 16); self.assertEqual(values["errno_restored"], 1)
    def test_02_padding_that_cannot_fit_falls_back_to_original_buffered_request(self) -> None:
        for name, expected_byte, expected_bytes in (("padding-capacity", 20, 5), ("padding-file", 88, 3)):
            with self.subTest(case=name):
                values = self._case("direct-enabled", name)
                self.assertEqual(values["result"], 1); self.assertEqual(values["direct_calls"], 0)
                self.assertEqual(values["buffered_calls"], 1); self.assertEqual(values["payload_is_stage"], 1)
                self.assertEqual(values["payload_byte"], expected_byte); self.assertEqual(values["last_bytes"], expected_bytes)
    def test_03_all_unsupported_direct_errors_disable_and_fall_back(self) -> None:
        for name in ("unsupported-fallback", "unsupported-efault", "unsupported-enotsup", "unsupported-eopnotsupp"):
            with self.subTest(error=name):
                values = self._case("direct-enabled", name)
                self.assertEqual(values["result"], 1); self.assertEqual(values["direct_calls"], 1)
                self.assertEqual(values["buffered_calls"], 1); self.assertEqual(values["close_calls"], 1)
                self.assertEqual(values["direct_closed"], 1); self.assertEqual(values["direct_fd"], -1)
                self.assertEqual(values["direct_align"], 1); self.assertEqual(values["payload_is_stage"], 1)
                self.assertEqual(values["payload_byte"], 20)
    def test_04_other_direct_error_falls_back_without_disabling_fd(self) -> None:
        values = self._case("direct-enabled", "direct-error-fallback")
        self.assertEqual(values["result"], 1); self.assertEqual(values["direct_calls"], 1)
        self.assertEqual(values["buffered_calls"], 1); self.assertEqual(values["close_calls"], 0)
        self.assertEqual(values["direct_fd"], 17); self.assertEqual(values["direct_align"], 16)
        self.assertEqual(values["payload_is_stage"], 1); self.assertEqual(values["payload_byte"], 20)
    def test_05_failed_read_preserves_output_pointer_in_both_variants(self) -> None:
        for variant in self._fixtures:
            with self.subTest(variant=variant):
                values = self._case(variant, "failed-fallback")
                self.assertEqual(values["result"], 0); self.assertEqual(values["payload_is_sentinel"], 1)
                self.assertEqual(values["close_calls"], 0)
                self.assertEqual(values["io_calls"], 2 if variant == "direct-enabled" else 1)
    def test_06_zero_read_commits_stage_without_io_in_both_variants(self) -> None:
        for variant in self._fixtures:
            with self.subTest(variant=variant):
                values = self._case(variant, "zero-read")
                self.assertEqual(values["result"], 1); self.assertEqual(values["io_calls"], 0)
                self.assertEqual(values["payload_is_stage"], 1)
    def test_07_capacity_and_interval_overflow_are_rejected_before_io(self) -> None:
        for variant in self._fixtures:
            for name in ("payload-over-capacity", "capacity-wrap", "interval-wrap"):
                with self.subTest(variant=variant, case=name):
                    values = self._case(variant, name)
                    self.assertEqual(values["result"], 0); self.assertEqual(values["io_calls"], 0)
                    self.assertEqual(values["payload_is_sentinel"], 1)
    def test_08_rounding_overflow_falls_back_to_original_high_offset_read(self) -> None:
        values = self._case("direct-enabled", "rounding-overflow")
        self.assertEqual(values["result"], 1); self.assertEqual(values["direct_calls"], 0)
        self.assertEqual(values["buffered_calls"], 1); self.assertEqual(values["payload_is_stage"], 1)
        self.assertEqual(values["last_bytes"], 2); self.assertEqual(values["last_offset"], 1 << 63)
        self.assertEqual(values["payload_byte"], 0xA5)
    def test_09_direct_disabled_variant_uses_buffered_read(self) -> None:
        values = self._case("direct-disabled", "direct-success")
        self.assertEqual(values["result"], 1); self.assertEqual(values["direct_calls"], 0)
        self.assertEqual(values["buffered_calls"], 1); self.assertEqual(values["payload_is_stage"], 1)
        self.assertEqual(values["payload_byte"], 20)
    def test_10_null_stage_and_output_are_guarded_in_subprocesses(self) -> None:
        for variant in self._fixtures:
            for name in ("null-stage", "null-output"):
                with self.subTest(variant=variant, case=name):
                    values = self._case(variant, name)
                    self.assertEqual(values["result"], 0); self.assertEqual(values["io_calls"], 0)
                    self.assertEqual(values["payload_is_sentinel"], 1)

    def test_11_exact_capacity_accepts_the_complete_buffered_payload(self) -> None:
        for variant in self._fixtures:
            with self.subTest(variant=variant):
                values = self._case(variant, "exact-capacity")
                self.assertEqual(values["result"], 1); self.assertEqual(values["io_calls"], 1)
                self.assertEqual(values["buffered_calls"], 1); self.assertEqual(values["last_bytes"], 8)
                self.assertEqual(values["payload_is_stage"], 1); self.assertEqual(values["payload_byte"], 1)

class SourceWiringContractTest(unittest.TestCase):
    def _assert_stage_call(self, body: str, raw: str, stage: str, capacity: str) -> None:
        code = code_only(body)
        pattern = (r"cuda_model_stage_read\s*\(\s*" + re.escape(stage) +
                   r"\s*,\s*cuda_stage_usable_bytes\s*\(\s*" + re.escape(raw) +
                   r"\s*,\s*" + re.escape(stage) + r"\s*,\s*" + re.escape(capacity) +
                   r"\s*\)\s*,")
        self.assertEqual(len(re.findall(pattern, code)), 1,
                         "usable-capacity helper must be the actual stage_read second argument")
    def test_stream_selected_call_uses_raw_and_aligned_pool_slots(self) -> None:
        self.assertIsNotNone(STREAM_CALLER)
        self._assert_stage_call(STREAM_CALLER or "", "g_stream_selected_stage_raw[bi]",
                                "g_stream_selected_stage[bi]", "g_stream_selected_stage_bytes")
    def test_model_range_call_uses_raw_and_aligned_pool_slots(self) -> None:
        self.assertIsNotNone(RANGE_CALLER)
        self._assert_stage_call(RANGE_CALLER or "", "g_model_stage_raw[bi]",
                                "g_model_stage[bi]", "g_model_stage_bytes")
if __name__ == "__main__":
    unittest.main(verbosity=2)
