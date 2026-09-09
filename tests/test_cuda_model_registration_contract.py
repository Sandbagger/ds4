"""Host-only behavioral contract for the resident model registration transaction."""
from __future__ import annotations
import os
import signal
import stat
import sys
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
class FixtureInfrastructureError(RuntimeError):
    """A fixture/setup/native defect that must map to process status 125."""
SOURCE_ERROR: Exception | None = None
try:
    CUDA_SOURCE = (ROOT / "ds4_cuda.cu").read_text(encoding="utf-8")
except Exception as exc:
    CUDA_SOURCE = ""
    SOURCE_ERROR = exc
LEGACY_SIGNATURE = (
    'extern "C" int ds4_gpu_set_model_map('
    'const void *model_map, uint64_t model_size)'
)
SIGNATURE = (
    'extern "C" int ds4_gpu_register_model_map_no_copy('
    'const void *model_map, uint64_t model_size)'
)
SAFE_ENV = {
    "PATH": "/opt/homebrew/bin:/usr/bin:/bin",
    "LANG": "C",
    "LC_ALL": "C",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONNOUSERSITE": "1",
}
REQUIRED_METRICS = frozenset({
    "r1", "r2", "result", "register_calls", "unregister_calls", "lookup_calls",
    "physical_registers", "cache_release_calls", "free_calls", "physical_frees",
    "free_owner_observed", "free_live", "released_devices", "free_refusals",
    "physical_unregistrations", "unregister_owner_observed", "released_registrations",
    "unregister_refusals", "device_owned", "request_failed", "copy_path_calls",
    "first_host_is_expected", "first_device_is_expected", "first_size_is_expected",
    "first_device_owned", "first_registered", "first_live", "first_free_live",
    "first_released_devices", "first_released_registrations", "first_free_calls",
    "first_physical_frees", "first_free_refusals", "first_unregister_calls",
    "first_physical_unregistrations", "first_unregister_refusals", "registered", "live",
    "api_errors", "host_is_expected", "size_is_expected", "device_is_expected",
    "device_is_null", "live_host_is_expected", "live_size_is_expected",
    'rescue_device_witnesses', 'rescue_registration_witnesses', 'rescue_device_effects', 'rescue_registration_effects', 'rescue_live_devices', 'rescue_live_registrations', 'rescue_refusal_history_same', 'rescue_remaining_owners', 'rescue_complete',
})
HEALTHY_CONTROL_EXPECTATIONS = {'legacy-healthy-device-rebind': {'result': 1, 'request_failed': 0, 'register_calls': 1, 'unregister_calls': 0, 'lookup_calls': 1, 'physical_registers': 1, 'free_calls': 1, 'physical_frees': 1, 'free_refusals': 0, 'free_owner_observed': 1, 'free_live': 0, 'released_devices': 1, 'physical_unregistrations': 0, 'unregister_refusals': 0, 'copy_path_calls': 0, 'registered': 1, 'live': 1, 'host_is_expected': 1, 'size_is_expected': 1, 'device_is_expected': 1, 'device_owned': 0, 'live_host_is_expected': 1, 'live_size_is_expected': 1}, 'nocopy-healthy-device-rebind': {'result': 1, 'request_failed': 0, 'register_calls': 1, 'unregister_calls': 0, 'lookup_calls': 1, 'physical_registers': 1, 'free_calls': 1, 'physical_frees': 1, 'free_refusals': 0, 'free_owner_observed': 1, 'free_live': 0, 'released_devices': 1, 'physical_unregistrations': 0, 'unregister_refusals': 0, 'copy_path_calls': 0, 'registered': 1, 'live': 1, 'host_is_expected': 1, 'size_is_expected': 1, 'device_is_expected': 1, 'device_owned': 0, 'live_host_is_expected': 1, 'live_size_is_expected': 1}, 'legacy-healthy-registration-rebind': {'result': 1, 'request_failed': 0, 'register_calls': 1, 'unregister_calls': 1, 'lookup_calls': 1, 'physical_registers': 2, 'free_calls': 0, 'physical_frees': 0, 'free_refusals': 0, 'free_owner_observed': 0, 'free_live': 0, 'released_devices': 0, 'physical_unregistrations': 1, 'unregister_refusals': 0, 'unregister_owner_observed': 1, 'released_registrations': 1, 'copy_path_calls': 0, 'registered': 1, 'live': 1, 'host_is_expected': 1, 'size_is_expected': 1, 'device_is_expected': 1, 'device_owned': 0, 'live_host_is_expected': 1, 'live_size_is_expected': 1}}

def _text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)
def _infra(label: str, detail: str, stdout: str = "", stderr: str = "") -> FixtureInfrastructureError:
    return FixtureInfrastructureError(
        f"{label}: {detail}\nstdout={stdout!r}\nstderr={stderr!r}"
    )
def _assert_owned0700(path: Path, label: str) -> None:
    try:
        st = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise _infra(label, f"stat failed: {exc}") from exc
    if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.getuid() or stat.S_IMODE(st.st_mode) != 0o700:
        raise _infra(
            label,
            f"expected owned0700 directory, got uid={st.st_uid} mode={oct(stat.S_IMODE(st.st_mode))}",
        )
def _sandbox_env(paths: dict[str, Path]) -> dict[str, str]:
    env = dict(SAFE_ENV)
    env.update({
        "HOME": str(paths["home"]),
        "TMPDIR": str(paths["tmp"]),
        "XDG_CONFIG_HOME": str(paths["config"]),
        "XDG_CACHE_HOME": str(paths["cache"]),
        "XDG_DATA_HOME": str(paths["data"]),
    })
    return env
_INTERRUPTED_SIGNUM = 0

def _terminate_group(child: subprocess.Popen) -> tuple[str, str]:
    """Stop the still-owned child group before reaping; never signal after reap."""
    if child.returncode is None:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        stdout, stderr = child.communicate(timeout=1)
    except subprocess.TimeoutExpired:
        if child.returncode is None:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        stdout, stderr = child.communicate(timeout=1)
    return _text(stdout), _text(stderr)

def _run_process(argv: list[str], cwd: Path, env: dict[str, str], timeout: float, label: str) -> tuple[str, str]:
    if _INTERRUPTED_SIGNUM:
        raise _infra(label, f"fixture interrupted by signal {_INTERRUPTED_SIGNUM}; no new child")
    child = None
    previous = {}
    def interrupted(signum: int, _frame: object) -> None:
        global _INTERRUPTED_SIGNUM
        _INTERRUPTED_SIGNUM = signum
        raise SystemExit(128 + signum)
    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous[signum] = signal.signal(signum, interrupted)
        child = subprocess.Popen(argv, cwd=str(cwd), env=dict(env),
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 text=True, start_new_session=True)
        stdout, stderr = child.communicate(timeout=timeout)
    except BaseException as exc:
        stdout = _text(getattr(exc, "stdout", None))
        stderr = _text(getattr(exc, "stderr", None))
        if child is not None:
            try:
                stopped_out, stopped_err = _terminate_group(child)
                stdout, stderr = stopped_out or stdout, stopped_err or stderr
            except BaseException as cleanup_exc:
                raise _infra(label, f"child cleanup unproved: {cleanup_exc!r}", stdout, stderr) from cleanup_exc
        raise _infra(label, f"spawn/wait/interruption refused: {exc!r}", stdout, stderr) from exc
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    stdout, stderr = _text(stdout), _text(stderr)
    if child.returncode != 0:
        raise _infra(label, f"native exit {child.returncode}", stdout, stderr)
    return stdout, stderr
def _parse_metrics(stdout: str, stderr: str) -> dict[str, int]:
    values: dict[str, int] = {}
    for line_number, line in enumerate(stdout.splitlines(), 1):
        key, separator, raw_value = line.partition("=")
        if not separator or not key or key in values:
            raise _infra("native metrics", f"malformed or duplicate line {line_number}: {line!r}", stdout, stderr)
        try:
            values[key] = int(raw_value)
        except ValueError as exc:
            raise _infra("native metrics", f"non-integer value on line {line_number}: {line!r}", stdout, stderr) from exc
    missing = sorted(REQUIRED_METRICS - values.keys())
    unexpected = sorted(values.keys() - REQUIRED_METRICS)
    if missing or unexpected:
        raise _infra("native metrics", f"missing={missing!r} unexpected={unexpected!r}", stdout, stderr)
    for key, expected in (("rescue_complete", 1), ("rescue_refusal_history_same", 1),
                          ("rescue_live_devices", 0), ("rescue_live_registrations", 0),
                          ("rescue_remaining_owners", 0), ("copy_path_calls", 0)):
        if values[key] != expected:
            raise _infra("native rescue", f"{key}={values[key]}, expected={expected}", stdout, stderr)
    if (values["rescue_device_witnesses"] != values["rescue_device_effects"] or
            values["rescue_registration_witnesses"] != values["rescue_registration_effects"]):
        raise _infra("native rescue", "physical-effect census differs from witnesses", stdout, stderr)
    return values

def _extract_definition(source: str, signature: str) -> str:
    """Extract one C/C++ definition while ignoring lexical braces."""
    occurrences = source.count(signature)
    if occurrences != 1:
        raise AssertionError(
            f"expected one source function {signature}, found {occurrences}"
        )
    start = source.find(signature)
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
        raise AssertionError("no code brace after function signature")
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
    raise AssertionError("unterminated C function body")

LEGACY_FUNCTION_DEFINITION = ""
FUNCTION_DEFINITION = ""
EXTRACTION_ERROR: Exception | None = SOURCE_ERROR
if EXTRACTION_ERROR is None:
    try:
        LEGACY_FUNCTION_DEFINITION = _extract_definition(CUDA_SOURCE, LEGACY_SIGNATURE)
        FUNCTION_DEFINITION = _extract_definition(CUDA_SOURCE, SIGNATURE)
    except Exception as exc:
        EXTRACTION_ERROR = exc

CPP_PREFIX = r"""#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <time.h>
#include <signal.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>
using cudaError_t = int;
static constexpr cudaError_t cudaSuccess = 0;
static constexpr unsigned cudaHostRegisterMapped = 1u;
static constexpr unsigned cudaHostRegisterReadOnly = 2u;
static constexpr cudaError_t kRegisterFailure = 17;
static constexpr cudaError_t kLookupFailure = 23;
static constexpr cudaError_t kUnregisterFailure = 29;
static constexpr cudaError_t kFreeFailure = 31;
static constexpr cudaError_t kCopyPathUnsupported = 47;
static constexpr int cudaMemcpyHostToDevice = 0;
struct cuda_q8_f32_range { void *device_ptr; };
static std::vector<cuda_q8_f32_range> g_q8_f32_ranges;
static std::unordered_map<uint64_t, size_t> g_q8_f32_by_offset;
static uint64_t g_q8_f32_bytes;
static int g_q8_f16_disabled_after_oom;
static int g_q8_f16_budget_notice_printed;
static const void *g_model_host_base;
static const char *g_model_device_base;
static uint64_t g_model_registered_size;
static int g_model_registered;
static int g_model_device_owned;
static int g_model_range_mapping_supported;
static int g_model_hmm_direct;
static int g_model_fd = -1;
static const void *g_model_fd_host_base;
static int g_model_cache_full;
/* This transaction fixture has no observer or real range owner. */
static int g_model_range_release_failed;
static int fake_range_release_result = 1;
static int cuda_laguna_resident_observer_safe(void) { return 1; }
static int fake_register_calls;
static int fake_unregister_calls;
static int fake_lookup_calls;
static int fake_physical_registrations;
static int fake_physical_unregistrations;
static int fake_cache_release_calls;
static int fake_free_calls;
static int fake_physical_frees;
static int fake_free_refusals;
static int fake_free_owner_observed;
static int fake_unregister_owner_observed;
static int fake_unregister_refusals;
static int fake_copy_path_calls;
static std::unordered_map<const void *, size_t> fake_live_registrations;
static std::unordered_map<const void *, size_t> fake_released_registrations;
static std::unordered_map<const void *, size_t> fake_live_device_allocations;
static std::unordered_map<const void *, size_t> fake_released_devices;
static int fake_api_errors;
static cudaError_t fake_register_result;
static cudaError_t fake_unregister_result;
static cudaError_t fake_lookup_result;
static cudaError_t fake_free_result;
static bool fake_lookup_null;
static int first_host_is_expected;
static int first_device_is_expected;
static int first_size_is_expected;
static int first_device_owned;
static int first_registered;
static int first_live;
static int first_free_live;
static int first_released_devices;
static int first_released_registrations;
static int first_free_calls;
static int first_physical_frees;
static int first_free_refusals;
static int first_unregister_calls;
static int first_physical_unregistrations;
static int first_unregister_refusals;
static void *fake_lookup_device;
class cuda_laguna_compact_legacy_permit {
public:
    bool allowed() const { return true; }
};
static int cuda_stream_selected_cache_release(void) {
    ++fake_cache_release_calls;
    return 1;
}
static int cuda_model_range_release_all(void) {
    ++fake_cache_release_calls;
    if (!fake_range_release_result) g_model_range_release_failed = 1;
    return fake_range_release_result;
}
static int cuda_q8_f16_cache_release_all(void) {
    ++fake_cache_release_calls;
    return 1;
}
static int cuda_q8_f32_cache_release_all(void) { return 1; }
static cudaError_t cudaFree(void *device) {
    ++fake_free_calls;
    auto live = fake_live_device_allocations.find(device);
    if (live == fake_live_device_allocations.end()) { ++fake_api_errors; return 43; }
    if (g_model_device_owned &&
        static_cast<const void *>(g_model_device_base) == device) {
        ++fake_free_owner_observed;
    }
    if (fake_free_result != cudaSuccess) {
        ++fake_free_refusals;
        return fake_free_result;
    }
    const size_t released_bytes = live->second;
    fake_live_device_allocations.erase(live);
    ++fake_physical_frees;
    fake_released_devices[device] = released_bytes;
    return cudaSuccess;
}
static cudaError_t cudaMalloc(void **device, size_t bytes) {
    ++fake_copy_path_calls;
    if (!device || !bytes) { ++fake_api_errors; return kCopyPathUnsupported; }
    *device = nullptr;
    return kCopyPathUnsupported;
}
static cudaError_t cudaMemcpy(void *dst, const void *src, size_t bytes, int kind) {
    ++fake_copy_path_calls;
    if (!dst || !src || !bytes || kind != cudaMemcpyHostToDevice) {
        ++fake_api_errors; return kCopyPathUnsupported;
    }
    return kCopyPathUnsupported;
}
static cudaError_t cudaHostRegister(void *host, size_t bytes, unsigned flags) {
    ++fake_register_calls;
    if (!host || !bytes || flags != (cudaHostRegisterMapped | cudaHostRegisterReadOnly)) {
        ++fake_api_errors; return 39;
    }
    if (fake_register_result != cudaSuccess) return fake_register_result;
    if (fake_live_registrations.count(host)) return 31;
    fake_live_registrations[host] = bytes;
    ++fake_physical_registrations;
    return cudaSuccess;
}
static cudaError_t cudaHostUnregister(void *host) {
    ++fake_unregister_calls;
    auto live = fake_live_registrations.find(host);
    if (live == fake_live_registrations.end()) { ++fake_api_errors; return 37; }
    if (g_model_registered && g_model_host_base == host) {
        ++fake_unregister_owner_observed;
    }
    if (fake_unregister_result != cudaSuccess) {
        ++fake_unregister_refusals;
        return fake_unregister_result;
    }
    const size_t released_bytes = live->second;
    fake_live_registrations.erase(live);
    ++fake_physical_unregistrations;
    fake_released_registrations[host] = released_bytes;
    return cudaSuccess;
}
static cudaError_t cudaHostGetDevicePointer(void **device, void *host, unsigned flags) {
    ++fake_lookup_calls;
    if (!device || !fake_live_registrations.count(host) || flags != 0) {
        ++fake_api_errors; return 41;
    }
    if (fake_lookup_result != cudaSuccess) { *device = nullptr; return fake_lookup_result; }
    *device = fake_lookup_null ? nullptr : fake_lookup_device;
    return cudaSuccess;
}
static const char *cudaGetErrorString(cudaError_t) { return "fake-cuda-error"; }
static cudaError_t cudaGetLastError(void) { return cudaSuccess; }
static char *ds4_contract_getenv(const char *) { return nullptr; }
#define getenv ds4_contract_getenv

"""
CPP_SUFFIX = r"""
#undef getenv
static const void *expected_host;
static const void *expected_device;
static uint64_t expected_size;
static const void *expected_live_host;
static int reset_state(void) {
    if (!fake_live_device_allocations.empty() || !fake_live_registrations.empty()) {
        std::fprintf(stderr,
                     "fixture reset refused with live device=%zu registration=%zu\n",
                     fake_live_device_allocations.size(), fake_live_registrations.size());
        return 0;
    }
    g_model_host_base = nullptr;
    g_model_device_base = nullptr;
    g_model_registered_size = 0;
    g_model_registered = 0;
    g_model_device_owned = 0;
    g_model_range_mapping_supported = 1;
    g_model_hmm_direct = 0;
    g_model_fd = -1;
    g_model_fd_host_base = nullptr;
    g_model_cache_full = 0;
    g_model_range_release_failed = 0;
    fake_range_release_result = 1;
    g_q8_f32_ranges.clear();
    g_q8_f32_by_offset.clear();
    g_q8_f32_bytes = 0;
    g_q8_f16_disabled_after_oom = 0;
    g_q8_f16_budget_notice_printed = 0;
    fake_register_calls = fake_unregister_calls = fake_lookup_calls = 0;
    fake_physical_registrations = fake_physical_unregistrations = 0;
    fake_cache_release_calls = 0;
    fake_free_calls = fake_physical_frees = fake_free_refusals = 0;
    fake_free_owner_observed = fake_unregister_owner_observed = 0;
    fake_unregister_refusals = fake_copy_path_calls = 0;
    fake_released_registrations.clear();
    fake_released_devices.clear();
    fake_api_errors = 0;
    fake_register_result = cudaSuccess;
    fake_unregister_result = cudaSuccess;
    fake_lookup_result = cudaSuccess;
    fake_free_result = cudaSuccess;
    first_host_is_expected = first_device_is_expected = -1;
    first_size_is_expected = first_device_owned = -1;
    first_registered = first_live = first_free_live = -1;
    first_released_devices = first_released_registrations = -1;
    first_free_calls = first_physical_frees = first_free_refusals = -1;
    first_unregister_calls = first_physical_unregistrations = -1;
    first_unregister_refusals = -1;
    fake_lookup_null = false;
    fake_lookup_device = nullptr;
    expected_host = expected_device = expected_live_host = nullptr;
    expected_size = 0;
    return 1;
}
static void seed_registered(const void *host, uint64_t bytes, const void *device) {
    g_model_host_base = host;
    g_model_device_base = static_cast<const char *>(device);
    g_model_registered_size = bytes;
    g_model_registered = 1;
    fake_live_registrations[host] = static_cast<size_t>(bytes);
    fake_physical_registrations = 1;
}
static void seed_device_owned(const void *host, uint64_t bytes, const void *device) {
    g_model_host_base = host;
    g_model_device_base = static_cast<const char *>(device);
    g_model_registered_size = bytes;
    g_model_device_owned = 1;
    fake_live_device_allocations[device] = static_cast<size_t>(bytes);
}
static void emit(int r1, int r2 = -1) {
    std::printf("r1=%d\n", r1);
    std::printf("r2=%d\n", r2);
    std::printf("result=%d\n", r2 >= 0 ? r2 : r1);
    std::printf("request_failed=%d\n", r1 == 0);
    std::printf("register_calls=%d\n", fake_register_calls);
    std::printf("unregister_calls=%d\n", fake_unregister_calls);
    std::printf("lookup_calls=%d\n", fake_lookup_calls);
    std::printf("physical_registers=%d\n", fake_physical_registrations);
    std::printf("cache_release_calls=%d\n", fake_cache_release_calls);
    std::printf("free_calls=%d\n", fake_free_calls);
    std::printf("physical_frees=%d\n", fake_physical_frees);
    std::printf("free_refusals=%d\n", fake_free_refusals);
    std::printf("free_owner_observed=%d\n", fake_free_owner_observed);
    std::printf("copy_path_calls=%d\n", fake_copy_path_calls);
    std::printf("free_live=%zu\n", fake_live_device_allocations.size());
    std::printf("released_devices=%zu\n", fake_released_devices.size());
    std::printf("physical_unregistrations=%d\n", fake_physical_unregistrations);
    std::printf("unregister_refusals=%d\n", fake_unregister_refusals);
    std::printf("unregister_owner_observed=%d\n", fake_unregister_owner_observed);
    std::printf("released_registrations=%zu\n", fake_released_registrations.size());
    std::printf("device_owned=%d\n", g_model_device_owned);
    std::printf("first_host_is_expected=%d\n", first_host_is_expected);
    std::printf("first_device_is_expected=%d\n", first_device_is_expected);
    std::printf("first_size_is_expected=%d\n", first_size_is_expected);
    std::printf("first_device_owned=%d\n", first_device_owned);
    std::printf("first_registered=%d\n", first_registered);
    std::printf("first_live=%d\n", first_live);
    std::printf("first_free_live=%d\n", first_free_live);
    std::printf("first_released_devices=%d\n", first_released_devices);
    std::printf("first_released_registrations=%d\n", first_released_registrations);
    std::printf("first_free_calls=%d\n", first_free_calls);
    std::printf("first_physical_frees=%d\n", first_physical_frees);
    std::printf("first_free_refusals=%d\n", first_free_refusals);
    std::printf("first_unregister_calls=%d\n", first_unregister_calls);
    std::printf("first_physical_unregistrations=%d\n", first_physical_unregistrations);
    std::printf("first_unregister_refusals=%d\n", first_unregister_refusals);
    std::printf("registered=%d\n", g_model_registered);
    std::printf("live=%zu\n", fake_live_registrations.size());
    std::printf("api_errors=%d\n", fake_api_errors);
    std::printf("host_is_expected=%d\n", expected_host && g_model_host_base == expected_host);
    std::printf("size_is_expected=%d\n", g_model_registered_size == expected_size);
    std::printf("device_is_expected=%d\n", expected_device && g_model_device_base == expected_device);
    std::printf("device_is_null=%d\n", g_model_device_base == nullptr);
    auto live = fake_live_registrations.find(expected_live_host);
    std::printf("live_host_is_expected=%d\n", expected_live_host && live != fake_live_registrations.end());
    std::printf("live_size_is_expected=%d\n", live != fake_live_registrations.end() && live->second == static_cast<size_t>(expected_size));
}
static void capture_first(const void *host, const void *device, uint64_t bytes) {
    first_host_is_expected = g_model_host_base == host;
    first_device_is_expected =
        static_cast<const void *>(g_model_device_base) == device;
    first_size_is_expected = g_model_registered_size == bytes;
    first_device_owned = g_model_device_owned;
    first_registered = g_model_registered;
    first_live = static_cast<int>(fake_live_registrations.size());
    first_free_live = static_cast<int>(fake_live_device_allocations.size());
    first_released_devices = static_cast<int>(fake_released_devices.size());
    first_released_registrations =
        static_cast<int>(fake_released_registrations.size());
    first_free_calls = fake_free_calls;
    first_physical_frees = fake_physical_frees;
    first_free_refusals = fake_free_refusals;
    first_unregister_calls = fake_unregister_calls;
    first_physical_unregistrations = fake_physical_unregistrations;
    first_unregister_refusals = fake_unregister_refusals;
}
static int invoke_model_setter(bool no_copy, const void *model_map, uint64_t model_size) {
    return no_copy
        ? ds4_gpu_register_model_map_no_copy(model_map, model_size)
        : ds4_gpu_set_model_map(model_map, model_size);
}
static int scenario_healthy_device_rebind(bool no_copy) {
    if (!reset_state()) return 125;
    const void *old_host = reinterpret_cast<const void *>(0x3900);
    const void *old_device = reinterpret_cast<const void *>(0x3a00);
    const void *new_host = reinterpret_cast<const void *>(0x3b00);
    const void *new_device = reinterpret_cast<const void *>(0x3c00);
    seed_device_owned(old_host, 4096, old_device);
    fake_lookup_device = const_cast<void *>(new_device);
    expected_host = expected_live_host = new_host;
    expected_device = new_device; expected_size = 8192;
    int first = invoke_model_setter(no_copy, new_host, expected_size);
    emit(first); return 0;
}
static int scenario_legacy_registered_rebind(void) {
    if (!reset_state()) return 125;
    const void *old_host = reinterpret_cast<const void *>(0x3d00);
    const void *old_device = reinterpret_cast<const void *>(0x3e00);
    const void *new_host = reinterpret_cast<const void *>(0x3f00);
    const void *new_device = reinterpret_cast<const void *>(0x4000);
    seed_registered(old_host, 4096, old_device);
    fake_lookup_device = const_cast<void *>(new_device);
    expected_host = expected_live_host = new_host;
    expected_device = new_device; expected_size = 8192;
    int first = ds4_gpu_set_model_map(new_host, expected_size);
    emit(first); return 0;
}
static int scenario_device_free_refusal(bool no_copy) {
    if (!reset_state()) return 125;
    const void *old_host = reinterpret_cast<const void *>(0x4100);
    const void *old_device = reinterpret_cast<const void *>(0x4200);
    const void *new_host = reinterpret_cast<const void *>(0x4300);
    const void *new_device = reinterpret_cast<const void *>(0x4400);
    seed_device_owned(old_host, 4096, old_device);
    fake_free_result = kFreeFailure;
    fake_lookup_device = const_cast<void *>(new_device);
    expected_host = expected_live_host = old_host;
    expected_device = old_device; expected_size = 4096;
    int first = invoke_model_setter(no_copy, new_host, 8192);
    capture_first(old_host, old_device, 4096);
    emit(first); return 0;
}
static int scenario_device_free_retry(bool no_copy) {
    if (!reset_state()) return 125;
    const void *old_host = reinterpret_cast<const void *>(0x4500);
    const void *old_device = reinterpret_cast<const void *>(0x4600);
    const void *new_host = reinterpret_cast<const void *>(0x4700);
    const void *new_device = reinterpret_cast<const void *>(0x4800);
    seed_device_owned(old_host, 4096, old_device);
    fake_free_result = kFreeFailure;
    fake_lookup_device = const_cast<void *>(new_device);
    int first = invoke_model_setter(no_copy, new_host, 8192);
    capture_first(old_host, old_device, 4096);
    fake_free_result = cudaSuccess;
    expected_host = expected_live_host = new_host;
    expected_device = new_device; expected_size = 8192;
    int second = invoke_model_setter(no_copy, new_host, 8192);
    emit(first, second); return 0;
}
static int scenario_legacy_registered_unregister_refusal(void) {
    if (!reset_state()) return 125;
    const void *old_host = reinterpret_cast<const void *>(0x5100);
    const void *old_device = reinterpret_cast<const void *>(0x5200);
    const void *new_host = reinterpret_cast<const void *>(0x5300);
    const void *new_device = reinterpret_cast<const void *>(0x5400);
    seed_registered(old_host, 4096, old_device);
    fake_unregister_result = kUnregisterFailure;
    fake_lookup_device = const_cast<void *>(new_device);
    expected_host = expected_live_host = old_host;
    expected_device = old_device; expected_size = 4096;
    int first = ds4_gpu_set_model_map(new_host, 8192);
    capture_first(old_host, old_device, 4096);
    emit(first); return 0;
}
static int scenario_legacy_combined_unregister_retry(void) {
    if (!reset_state()) return 125;
    /* Synthetic boundary only: source review does not prove dual-owner acquisition. */
    const void *old_host = reinterpret_cast<const void *>(0x5500);
    const void *old_device = reinterpret_cast<const void *>(0x5600);
    const void *new_host = reinterpret_cast<const void *>(0x5700);
    const void *new_device = reinterpret_cast<const void *>(0x5800);
    seed_registered(old_host, 4096, old_device);
    g_model_device_owned = 1;
    fake_live_device_allocations[old_device] = 4096;
    fake_free_result = cudaSuccess;
    fake_unregister_result = kUnregisterFailure;
    fake_lookup_device = const_cast<void *>(new_device);
    int first = ds4_gpu_set_model_map(new_host, 8192);
    capture_first(old_host, old_device, 4096);
    fake_unregister_result = cudaSuccess;
    expected_host = expected_live_host = new_host;
    expected_device = new_device; expected_size = 8192;
    int second = ds4_gpu_set_model_map(new_host, 8192);
    emit(first, second); return 0;
}
static int scenario_range_teardown_failure(void) {
    if (!reset_state()) return 125;
    const void *old_host = reinterpret_cast<const void *>(0x1a00);
    const void *old_device = reinterpret_cast<const void *>(0x2a00);
    const void *new_host = reinterpret_cast<const void *>(0x1b00);
    seed_registered(old_host, 2048, old_device);
    expected_host = old_host; expected_device = old_device; expected_size = 2048;
    fake_range_release_result = 0;
    const int first = ds4_gpu_register_model_map_no_copy(new_host, 8192);
    // The old identity must not turn a pending teardown into idempotent success.
    const int second = ds4_gpu_register_model_map_no_copy(old_host, 2048);
    emit(first, second); return 0;
}
static int scenario_register_success(void) {
    if (!reset_state()) return 125;
    const void *map = reinterpret_cast<const void *>(0x1000);
    const void *device = reinterpret_cast<const void *>(0x2000);
    expected_host = expected_live_host = map; expected_device = device; expected_size = 4096;
    fake_lookup_device = const_cast<void *>(device);
    int first = ds4_gpu_register_model_map_no_copy(map, expected_size);
    int second = ds4_gpu_register_model_map_no_copy(map, expected_size);
    emit(first, second); return 0;
}
static int scenario_register_failure(void) {
    if (!reset_state()) return 125;
    const void *map = reinterpret_cast<const void *>(0x1100);
    expected_host = map; expected_size = 4096;
    fake_register_result = kRegisterFailure;
    emit(ds4_gpu_register_model_map_no_copy(map, expected_size)); return 0;
}
static int scenario_lookup_failure(bool null_pointer) {
    if (!reset_state()) return 125;
    const void *map = reinterpret_cast<const void *>(0x1200);
    expected_host = map; expected_size = 4096;
    fake_lookup_result = null_pointer ? cudaSuccess : kLookupFailure;
    fake_lookup_null = null_pointer;
    fake_unregister_result = cudaSuccess;
    emit(ds4_gpu_register_model_map_no_copy(map, expected_size)); return 0;
}
static int scenario_rollback_failure(void) {
    if (!reset_state()) return 125;
    const void *map = reinterpret_cast<const void *>(0x1300);
    expected_host = expected_live_host = map; expected_size = 4096;
    fake_lookup_result = kLookupFailure;
    fake_unregister_result = kUnregisterFailure;
    int first = ds4_gpu_register_model_map_no_copy(map, expected_size);
    int second = ds4_gpu_register_model_map_no_copy(map, expected_size);
    emit(first, second); return 0;
}
static int scenario_replace_release_failure(void) {
    if (!reset_state()) return 125;
    const void *old_host = reinterpret_cast<const void *>(0x1400);
    const void *old_device = reinterpret_cast<const void *>(0x2400);
    const void *new_host = reinterpret_cast<const void *>(0x1500);
    seed_registered(old_host, 2048, old_device);
    expected_host = expected_live_host = old_host; expected_device = old_device; expected_size = 2048;
    fake_unregister_result = kUnregisterFailure;
    emit(ds4_gpu_register_model_map_no_copy(new_host, 8192)); return 0;
}
static int scenario_recovery(void) {
    if (!reset_state()) return 125;
    const void *old_host = reinterpret_cast<const void *>(0x1600);
    const void *old_device = reinterpret_cast<const void *>(0x2600);
    const void *new_host = reinterpret_cast<const void *>(0x1700);
    const void *new_device = reinterpret_cast<const void *>(0x2700);
    seed_registered(old_host, 2048, old_device);
    fake_unregister_result = kUnregisterFailure;
    int first = ds4_gpu_register_model_map_no_copy(new_host, 8192);
    fake_unregister_result = cudaSuccess;
    fake_lookup_device = const_cast<void *>(new_device);
    expected_host = expected_live_host = new_host; expected_device = new_device; expected_size = 8192;
    int second = ds4_gpu_register_model_map_no_copy(new_host, expected_size);
    emit(first, second); return 0;
}
static int scenario_pending_replacement(bool release_succeeds) {
    if (!reset_state()) return 125;
    const void *old_host = reinterpret_cast<const void *>(0x1800);
    const void *new_host = reinterpret_cast<const void *>(0x1900);
    const void *new_device = reinterpret_cast<const void *>(0x2900);
    fake_lookup_result = kLookupFailure;
    fake_unregister_result = kUnregisterFailure;
    int first = ds4_gpu_register_model_map_no_copy(old_host, 2048);
    fake_lookup_result = cudaSuccess;
    fake_lookup_device = const_cast<void *>(new_device);
    fake_unregister_result = release_succeeds ? cudaSuccess : kUnregisterFailure;
    expected_host = expected_live_host = release_succeeds ? new_host : old_host;
    expected_device = release_succeeds ? new_device : nullptr;
    expected_size = release_succeeds ? 8192 : 2048;
    int second = ds4_gpu_register_model_map_no_copy(new_host, 8192);
    emit(first, second); return 0;
}
static int rescue_known_witnesses(void) {
    const int refusal_free_before = fake_free_refusals;
    const int refusal_unregister_before = fake_unregister_refusals;
    const int physical_free_before = fake_physical_frees;
    const int physical_unregister_before = fake_physical_unregistrations;
    const size_t device_witnesses = fake_live_device_allocations.size();
    const size_t registration_witnesses = fake_live_registrations.size();
    /* Faults are disabled only after every scenario has emitted its facts. */
    fake_free_result = cudaSuccess;
    fake_unregister_result = cudaSuccess;
    while (!fake_live_device_allocations.empty()) {
        auto witness = fake_live_device_allocations.begin();
        const void *device = witness->first;
        if (cudaFree(const_cast<void *>(device)) != cudaSuccess) {
            std::fprintf(stderr, "fixture rescue cudaFree failed\n");
            return 125;
        }
        if (g_model_device_owned && g_model_device_base == device) {
            g_model_device_owned = 0;
            g_model_device_base = nullptr;
        }
    }
    while (!fake_live_registrations.empty()) {
        auto witness = fake_live_registrations.begin();
        const void *host = witness->first;
        if (cudaHostUnregister(const_cast<void *>(host)) != cudaSuccess) {
            std::fprintf(stderr, "fixture rescue cudaHostUnregister failed\n");
            return 125;
        }
        if (g_model_registered && g_model_host_base == host) {
            g_model_registered = 0;
            if (!g_model_device_owned) g_model_device_base = nullptr;
        }
    }
    if (!fake_live_device_allocations.empty() || !fake_live_registrations.empty()) {
        std::fprintf(stderr, "fixture rescue left residual live records\n");
        return 125;
    }
    if (fake_free_refusals != refusal_free_before ||
        fake_unregister_refusals != refusal_unregister_before) {
        std::fprintf(stderr, "fixture rescue changed refusal history\n");
        return 125;
    }
    if (fake_physical_frees - physical_free_before !=
            static_cast<int>(device_witnesses) ||
        fake_physical_unregistrations - physical_unregister_before !=
            static_cast<int>(registration_witnesses)) {
        std::fprintf(stderr, "fixture rescue physical effects incomplete\n");
        return 125;
    }
    if (fake_physical_frees != static_cast<int>(fake_released_devices.size()) ||
        fake_physical_unregistrations !=
            static_cast<int>(fake_released_registrations.size()) ||
        fake_api_errors != 0) {
        std::fprintf(stderr, "fixture rescue records are inconsistent\n");
        return 125;
    }
    const int remaining_owners = g_model_device_owned || g_model_registered;
    if (!remaining_owners) {
        g_model_host_base = nullptr;
        g_model_device_base = nullptr;
        g_model_registered_size = 0;
    }
    std::printf("rescue_device_witnesses=%zu\nrescue_registration_witnesses=%zu\n", device_witnesses, registration_witnesses);
    std::printf("rescue_device_effects=%d\nrescue_registration_effects=%d\n", fake_physical_frees - physical_free_before, fake_physical_unregistrations - physical_unregister_before);
    std::printf("rescue_live_devices=%zu\nrescue_live_registrations=%zu\n", fake_live_device_allocations.size(), fake_live_registrations.size());
    std::printf("rescue_refusal_history_same=%d\nrescue_remaining_owners=%d\nrescue_complete=%d\n", fake_free_refusals == refusal_free_before && fake_unregister_refusals == refusal_unregister_before, remaining_owners, !remaining_owners);
    return remaining_owners ? 125 : 0;
}
int main(int argc, char **argv) {
    alarm(10);
    if (argc != 2) return 2;
    int scenario_result = 2;
    if (std::strcmp(argv[1], "legacy-healthy-device-rebind") == 0) scenario_result = scenario_healthy_device_rebind(false);
    if (std::strcmp(argv[1], "nocopy-healthy-device-rebind") == 0) scenario_result = scenario_healthy_device_rebind(true);
    if (std::strcmp(argv[1], "legacy-healthy-registration-rebind") == 0) scenario_result = scenario_legacy_registered_rebind();
    if (std::strcmp(argv[1], "legacy-device-free-refusal") == 0) scenario_result = scenario_device_free_refusal(false);
    if (std::strcmp(argv[1], "nocopy-device-free-refusal") == 0) scenario_result = scenario_device_free_refusal(true);
    if (std::strcmp(argv[1], "legacy-device-free-retry") == 0) scenario_result = scenario_device_free_retry(false);
    if (std::strcmp(argv[1], "nocopy-device-free-retry") == 0) scenario_result = scenario_device_free_retry(true);
    if (std::strcmp(argv[1], "legacy-registered-unregister-refusal") == 0) scenario_result = scenario_legacy_registered_unregister_refusal();
    if (std::strcmp(argv[1], "legacy-combined-unregister-retry") == 0) scenario_result = scenario_legacy_combined_unregister_retry();
    if (std::strcmp(argv[1], "range-teardown-failure") == 0) scenario_result = scenario_range_teardown_failure();
    if (std::strcmp(argv[1], "register-success") == 0) scenario_result = scenario_register_success();
    if (std::strcmp(argv[1], "register-failure") == 0) scenario_result = scenario_register_failure();
    if (std::strcmp(argv[1], "lookup-failure") == 0) scenario_result = scenario_lookup_failure(false);
    if (std::strcmp(argv[1], "lookup-null") == 0) scenario_result = scenario_lookup_failure(true);
    if (std::strcmp(argv[1], "rollback-failure") == 0) scenario_result = scenario_rollback_failure();
    if (std::strcmp(argv[1], "replace-release-failure") == 0) scenario_result = scenario_replace_release_failure();
    if (std::strcmp(argv[1], "recovery") == 0) scenario_result = scenario_recovery();
    if (std::strcmp(argv[1], "pending-recovery") == 0) scenario_result = scenario_pending_replacement(true);
    if (std::strcmp(argv[1], "pending-release-failure") == 0) scenario_result = scenario_pending_replacement(false);
    if (scenario_result != 0) return scenario_result;
    return rescue_known_witnesses();
}
"""
CPP_SOURCE = CPP_PREFIX + LEGACY_FUNCTION_DEFINITION + FUNCTION_DEFINITION + CPP_SUFFIX

class ModelRegistrationContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            if EXTRACTION_ERROR is not None:
                raise _infra("source extraction", repr(EXTRACTION_ERROR))
            admitted = Path(os.environ.get("TMPDIR") or tempfile.gettempdir())
            admitted = Path(os.path.abspath(admitted))
            if not admitted.is_dir():
                raise _infra("fixture setup", f"admitted TMPDIR is not a directory: {admitted}")
            cls._tmp = tempfile.TemporaryDirectory(
                prefix="ds4-model-registration-contract-", dir=str(admitted)
            )
            cls.addClassCleanup(cls._tmp.cleanup)
            base = Path(cls._tmp.name)
            _assert_owned0700(base, "fixture root")
            if Path(os.path.realpath(base.parent)) != Path(os.path.realpath(admitted)):
                raise _infra("fixture setup", "fixture root escaped admitted TMPDIR")
            paths = {name: base / name for name in ("home", "tmp", "config", "cache", "data")}
            for name, path in paths.items():
                path.mkdir(mode=0o700)
                os.chmod(path, 0o700)
                _assert_owned0700(path, f"fixture {name}")
            cls._paths = paths
            cls._env = _sandbox_env(paths)
            cls._source = paths["data"] / "fixture.cc"
            cls._binary = paths["data"] / "fixture"
            cls._source.write_text(CPP_SOURCE, encoding="utf-8")
            _run_process(
                ["c++", "-std=c++17", "-O0", str(cls._source), "-o", str(cls._binary)],
                cwd=paths["data"], env=cls._env, timeout=15.0, label="fixture compiler",
            )
            cls._healthy_snapshots = {}
            for name, expected in HEALTHY_CONTROL_EXPECTATIONS.items():
                stdout, stderr = _run_process(
                    [str(cls._binary), name], cwd=paths["data"], env=cls._env,
                    timeout=10.0, label=f"healthy fixture {name}")
                values = _parse_metrics(stdout, stderr)
                mismatches = {key: (values[key], value)
                              for key, value in expected.items() if values[key] != value}
                if values["api_errors"] != 0 or mismatches:
                    raise _infra("healthy control", f"{name}: api_errors={values['api_errors']} mismatches={mismatches}", stdout, stderr)
                cls._healthy_snapshots[name] = (dict(values), stdout, stderr)
            sys.stderr.write("fixture healthy replacement controls: 3 passed before fault cases\n")
        except FixtureInfrastructureError:
            raise
        except Exception as exc:
            raise _infra("fixture setup", repr(exc)) from exc
    def _case(self, name: str) -> dict[str, int]:
        stdout, stderr = _run_process(
            [str(self._binary), name],
            cwd=self._paths["data"], env=self._env, timeout=10.0,
            label=f"fixture {name}",
        )
        self._last_native_output = (stdout, stderr)
        values = _parse_metrics(stdout, stderr)
        if values["api_errors"] != 0:
            raise _infra("fixture metrics", f"api_errors={values['api_errors']}", stdout, stderr)
        if values["copy_path_calls"] != 0:
            raise _infra(
                "fixture metrics",
                f"unsupported copy path reached: calls={values['copy_path_calls']}",
                stdout,
                stderr,
            )
        return values
    def _healthy_case(self, name: str) -> dict[str, int]:
        values, stdout, stderr = self._healthy_snapshots[name]
        self._last_native_output = (stdout, stderr)
        return dict(values)

    def _assert_fields(self, values: dict[str, int], **expected: int) -> None:
        for key, value in expected.items():
            with self.subTest(field=key):
                self.assertEqual(values.get(key), value)

    def test_00_legacy_healthy_device_rebind_control(self) -> None:
        self._assert_fields(self._healthy_case("legacy-healthy-device-rebind"), result=1,
            request_failed=0, register_calls=1, unregister_calls=0, lookup_calls=1,
            physical_registers=1, free_calls=1, physical_frees=1,
            free_refusals=0, free_owner_observed=1, free_live=0,
            released_devices=1, physical_unregistrations=0,
            unregister_refusals=0, copy_path_calls=0, registered=1, live=1,
            host_is_expected=1, size_is_expected=1, device_is_expected=1,
            device_owned=0, live_host_is_expected=1, live_size_is_expected=1)
    def test_00_no_copy_healthy_device_rebind_control(self) -> None:
        self._assert_fields(self._healthy_case("nocopy-healthy-device-rebind"), result=1,
            request_failed=0, register_calls=1, unregister_calls=0, lookup_calls=1,
            physical_registers=1, free_calls=1, physical_frees=1,
            free_refusals=0, free_owner_observed=1, free_live=0,
            released_devices=1, physical_unregistrations=0,
            unregister_refusals=0, copy_path_calls=0, registered=1, live=1,
            host_is_expected=1, size_is_expected=1, device_is_expected=1,
            device_owned=0, live_host_is_expected=1, live_size_is_expected=1)
    def test_00_legacy_healthy_registration_rebind_control(self) -> None:
        self._assert_fields(self._healthy_case("legacy-healthy-registration-rebind"), result=1,
            request_failed=0, register_calls=1, unregister_calls=1, lookup_calls=1,
            physical_registers=2, free_calls=0, physical_frees=0,
            free_refusals=0, free_owner_observed=0, free_live=0,
            released_devices=0, physical_unregistrations=1,
            unregister_refusals=0, unregister_owner_observed=1,
            released_registrations=1, copy_path_calls=0, registered=1, live=1,
            host_is_expected=1, size_is_expected=1, device_is_expected=1,
            device_owned=0, live_host_is_expected=1, live_size_is_expected=1)

    def test_01_positive_register_and_same_map_idempotence(self) -> None:
        self._assert_fields(self._case("register-success"), r1=1, r2=1, result=1,
            register_calls=1, unregister_calls=0, physical_registers=1, live=1,
            registered=1, host_is_expected=1, size_is_expected=1, device_is_expected=1,
            live_host_is_expected=1, live_size_is_expected=1)
    def test_02_positive_host_register_failure_falls_back(self) -> None:
        self._assert_fields(self._case("register-failure"), result=1,
            register_calls=1, unregister_calls=0, physical_registers=0, live=0,
            registered=0)
    def test_03_lookup_failure_rolls_back_when_unregister_succeeds(self) -> None:
        self._assert_fields(self._case("lookup-failure"), result=1,
            register_calls=1, unregister_calls=1, physical_registers=1, live=0,
            registered=0)
    def test_04_lookup_null_rolls_back_when_unregister_succeeds(self) -> None:
        self._assert_fields(self._case("lookup-null"), result=1,
            register_calls=1, unregister_calls=1, physical_registers=1, live=0,
            registered=0)
    def test_05_lookup_failure_retains_registration_if_rollback_fails(self) -> None:
        self._assert_fields(self._case("rollback-failure"), r1=0, r2=0, result=0,
            register_calls=1, physical_registers=1, live=1, registered=1,
            host_is_expected=1, size_is_expected=1, device_is_null=1,
            live_host_is_expected=1, live_size_is_expected=1)
    def test_06_failed_release_preserves_old_owner_before_replacement(self) -> None:
        self._assert_fields(self._case("replace-release-failure"), result=0,
            register_calls=0, unregister_calls=1, physical_registers=1,
            cache_release_calls=0, live=1, registered=1, host_is_expected=1,
            size_is_expected=1, device_is_expected=1, live_host_is_expected=1,
            live_size_is_expected=1)
    def test_07_recovery_releases_old_then_keeps_only_new_mapping(self) -> None:
        self._assert_fields(self._case("recovery"), r1=0, r2=1, result=1,
            register_calls=1, unregister_calls=2, physical_registers=2,
            live=1, registered=1, host_is_expected=1, size_is_expected=1,
            device_is_expected=1, live_host_is_expected=1, live_size_is_expected=1)
    def test_08_pending_rollback_can_recover_to_a_different_mapping(self) -> None:
        self._assert_fields(self._case("pending-recovery"), r1=0, r2=1,
            register_calls=2, unregister_calls=2, lookup_calls=2, physical_registers=2,
            live=1, registered=1, host_is_expected=1, size_is_expected=1,
            device_is_expected=1, live_host_is_expected=1, live_size_is_expected=1)
    def test_09_pending_rollback_keeps_ownership_when_next_release_fails(self) -> None:
        self._assert_fields(self._case("pending-release-failure"), r1=0, r2=0,
            register_calls=1, unregister_calls=2, lookup_calls=1, physical_registers=1,
            cache_release_calls=3, live=1, registered=1, host_is_expected=1,
            size_is_expected=1, device_is_null=1, live_host_is_expected=1,
            live_size_is_expected=1)

    def test_11_legacy_device_free_refusal_retains_whole_model_owner(self) -> None:
        self._assert_fields(self._case("legacy-device-free-refusal"), result=0,
            request_failed=1, copy_path_calls=0, register_calls=0,
            unregister_calls=0, free_calls=1, free_refusals=1,
            first_free_refusals=1, physical_frees=0, free_owner_observed=1,
            free_live=1, released_devices=0, unregister_refusals=0,
            first_unregister_refusals=0, physical_registers=0, live=0,
            registered=0,
            host_is_expected=1, size_is_expected=1, device_is_expected=1,
            device_owned=1, first_host_is_expected=1,
            first_device_is_expected=1, first_size_is_expected=1,
            first_device_owned=1, first_registered=0, first_free_live=1,
            first_physical_frees=0)
    def test_12_no_copy_device_free_refusal_retains_whole_model_owner(self) -> None:
        self._assert_fields(self._case("nocopy-device-free-refusal"), result=0,
            request_failed=1, copy_path_calls=0, register_calls=0,
            unregister_calls=0, free_calls=1, free_refusals=1,
            first_free_refusals=1, physical_frees=0, free_owner_observed=1,
            free_live=1, released_devices=0, unregister_refusals=0,
            first_unregister_refusals=0, physical_registers=0, live=0,
            registered=0,
            host_is_expected=1, size_is_expected=1, device_is_expected=1,
            device_owned=1, first_host_is_expected=1,
            first_device_is_expected=1, first_size_is_expected=1,
            first_device_owned=1, first_registered=0, first_free_live=1,
            first_physical_frees=0)
    def test_13_legacy_device_free_retry_releases_once_then_rebinds(self) -> None:
        self._assert_fields(self._case("legacy-device-free-retry"), r1=0, r2=1,
            result=1, request_failed=1, copy_path_calls=0, register_calls=1,
            unregister_calls=0, free_calls=2, free_refusals=1,
            first_free_refusals=1, physical_frees=1, free_owner_observed=2,
            free_live=0, unregister_refusals=0, first_unregister_refusals=0,
            released_devices=1, physical_registers=1, live=1, registered=1,
            host_is_expected=1, size_is_expected=1, device_is_expected=1,
            device_owned=0, first_host_is_expected=1,
            first_device_is_expected=1, first_size_is_expected=1,
            first_device_owned=1, first_registered=0, first_free_live=1,
            first_physical_frees=0)
    def test_14_no_copy_device_free_retry_releases_once_then_rebinds(self) -> None:
        self._assert_fields(self._case("nocopy-device-free-retry"), r1=0, r2=1,
            result=1, request_failed=1, copy_path_calls=0, register_calls=1,
            unregister_calls=0, free_calls=2, free_refusals=1,
            first_free_refusals=1, physical_frees=1, free_owner_observed=2,
            free_live=0, unregister_refusals=0, first_unregister_refusals=0,
            released_devices=1, physical_registers=1, live=1, registered=1,
            host_is_expected=1, size_is_expected=1, device_is_expected=1,
            device_owned=0, first_host_is_expected=1,
            first_device_is_expected=1, first_size_is_expected=1,
            first_device_owned=1, first_registered=0, first_free_live=1,
            first_physical_frees=0)
    def test_15_legacy_registration_refusal_retains_old_source(self) -> None:
        self._assert_fields(self._case("legacy-registered-unregister-refusal"), result=0,
            request_failed=1, copy_path_calls=0, register_calls=0,
            unregister_calls=1, unregister_refusals=1,
            first_unregister_refusals=1, free_refusals=0,
            first_free_refusals=0, physical_unregistrations=0,
            unregister_owner_observed=1, released_registrations=0,
            free_calls=0, physical_frees=0, physical_registers=1, live=1,
            registered=1, host_is_expected=1, size_is_expected=1,
            device_is_expected=1, device_owned=0,
            first_host_is_expected=1, first_device_is_expected=1,
            first_size_is_expected=1, first_registered=1, first_live=1,
            first_free_live=0, first_physical_unregistrations=0)
    def test_16_legacy_combined_prefix_retry_does_not_double_free(self) -> None:
        self._assert_fields(self._case("legacy-combined-unregister-retry"), r1=0, r2=1,
            result=1, request_failed=1, copy_path_calls=0, register_calls=1,
            unregister_calls=2, unregister_refusals=1,
            first_unregister_refusals=1, free_refusals=0,
            first_free_refusals=0, physical_unregistrations=1,
            unregister_owner_observed=2,
            released_registrations=1, free_calls=1, physical_frees=1,
            free_owner_observed=1, free_live=0, released_devices=1,
            physical_registers=2, live=1, registered=1, host_is_expected=1,
            size_is_expected=1, device_is_expected=1, device_owned=0,
            first_host_is_expected=1, first_size_is_expected=1,
            first_device_owned=0, first_registered=1, first_live=1,
            first_free_live=0, first_released_devices=1,
            first_physical_frees=1, first_physical_unregistrations=0)

    def test_10_range_teardown_failure_stops_rebind_and_same_map_fast_success(self) -> None:
        self._assert_fields(self._case("range-teardown-failure"), r1=0, r2=0,
            register_calls=0, unregister_calls=1, lookup_calls=0,
            physical_registers=1, live=0, cache_release_calls=2, registered=0,
            host_is_expected=1, size_is_expected=1, device_is_expected=1)

if __name__ == "__main__":
    program = unittest.main(exit=False)
    if program.result.errors:
        raise SystemExit(125)
    if program.result.failures:
        raise SystemExit(1)
    raise SystemExit(0)
