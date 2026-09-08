"""Host-only behavioral contract for the resident model registration transaction."""
from __future__ import annotations
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CUDA_SOURCE = (ROOT / "ds4_cuda.cu").read_text(encoding="utf-8")
SIGNATURE = (
    'extern "C" int ds4_gpu_register_model_map_no_copy('
    'const void *model_map, uint64_t model_size)'
)
SAFE_ENV = {
    "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
    "LANG": "C",
    "LC_ALL": "C",
}
def _extract_definition(source: str, signature: str) -> str:
    """Extract one C/C++ definition while ignoring lexical braces."""
    start = source.find(signature)
    if start < 0:
        raise AssertionError(f"missing source function {signature}")
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

FUNCTION_DEFINITION = _extract_definition(CUDA_SOURCE, SIGNATURE)

CPP_PREFIX = r"""#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
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
static int fake_cache_release_calls;
static std::unordered_map<const void *, size_t> fake_live_registrations;
static int fake_api_errors;
static cudaError_t fake_register_result;
static cudaError_t fake_unregister_result;
static cudaError_t fake_lookup_result;
static bool fake_lookup_null;
static void *fake_lookup_device;
class cuda_laguna_compact_legacy_permit {
public:
    bool allowed() const { return true; }
};
static void cuda_stream_selected_cache_release(void) { ++fake_cache_release_calls; }
static int cuda_model_range_release_all(void) {
    ++fake_cache_release_calls;
    if (!fake_range_release_result) g_model_range_release_failed = 1;
    return fake_range_release_result;
}
static void cuda_q8_f16_cache_release_all(void) { ++fake_cache_release_calls; }
static cudaError_t cudaFree(void *) { return cudaSuccess; }
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
    if (!fake_live_registrations.count(host)) { ++fake_api_errors; return 37; }
    if (fake_unregister_result != cudaSuccess) return fake_unregister_result;
    fake_live_registrations.erase(host);
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
static void reset_state(void) {
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
    fake_physical_registrations = fake_cache_release_calls = 0;
    fake_live_registrations.clear();
    fake_api_errors = 0;
    fake_register_result = cudaSuccess;
    fake_unregister_result = cudaSuccess;
    fake_lookup_result = cudaSuccess;
    fake_lookup_null = false;
    fake_lookup_device = nullptr;
    expected_host = expected_device = expected_live_host = nullptr;
    expected_size = 0;
}
static void seed_registered(const void *host, uint64_t bytes, const void *device) {
    g_model_host_base = host;
    g_model_device_base = static_cast<const char *>(device);
    g_model_registered_size = bytes;
    g_model_registered = 1;
    fake_live_registrations[host] = static_cast<size_t>(bytes);
    fake_physical_registrations = 1;
}
static void emit(int r1, int r2 = -1) {
    std::printf("r1=%d\n", r1);
    std::printf("r2=%d\n", r2);
    std::printf("result=%d\n", r2 >= 0 ? r2 : r1);
    std::printf("register_calls=%d\n", fake_register_calls);
    std::printf("unregister_calls=%d\n", fake_unregister_calls);
    std::printf("lookup_calls=%d\n", fake_lookup_calls);
    std::printf("physical_registers=%d\n", fake_physical_registrations);
    std::printf("cache_release_calls=%d\n", fake_cache_release_calls);
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
static int scenario_range_teardown_failure(void) {
    reset_state();
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
    reset_state();
    const void *map = reinterpret_cast<const void *>(0x1000);
    const void *device = reinterpret_cast<const void *>(0x2000);
    expected_host = expected_live_host = map; expected_device = device; expected_size = 4096;
    fake_lookup_device = const_cast<void *>(device);
    int first = ds4_gpu_register_model_map_no_copy(map, expected_size);
    int second = ds4_gpu_register_model_map_no_copy(map, expected_size);
    emit(first, second); return 0;
}
static int scenario_register_failure(void) {
    reset_state();
    const void *map = reinterpret_cast<const void *>(0x1100);
    expected_host = map; expected_size = 4096;
    fake_register_result = kRegisterFailure;
    emit(ds4_gpu_register_model_map_no_copy(map, expected_size)); return 0;
}
static int scenario_lookup_failure(bool null_pointer) {
    reset_state();
    const void *map = reinterpret_cast<const void *>(0x1200);
    expected_host = map; expected_size = 4096;
    fake_lookup_result = null_pointer ? cudaSuccess : kLookupFailure;
    fake_lookup_null = null_pointer;
    fake_unregister_result = cudaSuccess;
    emit(ds4_gpu_register_model_map_no_copy(map, expected_size)); return 0;
}
static int scenario_rollback_failure(void) {
    reset_state();
    const void *map = reinterpret_cast<const void *>(0x1300);
    expected_host = expected_live_host = map; expected_size = 4096;
    fake_lookup_result = kLookupFailure;
    fake_unregister_result = kUnregisterFailure;
    int first = ds4_gpu_register_model_map_no_copy(map, expected_size);
    int second = ds4_gpu_register_model_map_no_copy(map, expected_size);
    emit(first, second); return 0;
}
static int scenario_replace_release_failure(void) {
    reset_state();
    const void *old_host = reinterpret_cast<const void *>(0x1400);
    const void *old_device = reinterpret_cast<const void *>(0x2400);
    const void *new_host = reinterpret_cast<const void *>(0x1500);
    seed_registered(old_host, 2048, old_device);
    expected_host = expected_live_host = old_host; expected_device = old_device; expected_size = 2048;
    fake_unregister_result = kUnregisterFailure;
    emit(ds4_gpu_register_model_map_no_copy(new_host, 8192)); return 0;
}
static int scenario_recovery(void) {
    reset_state();
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
    reset_state();
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
int main(int argc, char **argv) {
    alarm(15);
    if (argc != 2) return 2;
    if (std::strcmp(argv[1], "range-teardown-failure") == 0) return scenario_range_teardown_failure();
    if (std::strcmp(argv[1], "register-success") == 0) return scenario_register_success();
    if (std::strcmp(argv[1], "register-failure") == 0) return scenario_register_failure();
    if (std::strcmp(argv[1], "lookup-failure") == 0) return scenario_lookup_failure(false);
    if (std::strcmp(argv[1], "lookup-null") == 0) return scenario_lookup_failure(true);
    if (std::strcmp(argv[1], "rollback-failure") == 0) return scenario_rollback_failure();
    if (std::strcmp(argv[1], "replace-release-failure") == 0) return scenario_replace_release_failure();
    if (std::strcmp(argv[1], "recovery") == 0) return scenario_recovery();
    if (std::strcmp(argv[1], "pending-recovery") == 0) return scenario_pending_replacement(true);
    if (std::strcmp(argv[1], "pending-release-failure") == 0) return scenario_pending_replacement(false);
    return 2;
}
"""
CPP_SOURCE = CPP_PREFIX + FUNCTION_DEFINITION + CPP_SUFFIX

class ModelRegistrationContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory(prefix="ds4-model-registration-contract-")
        cls.addClassCleanup(cls._tmp.cleanup)
        base = Path(cls._tmp.name)
        cls._source = base / "fixture.cc"
        cls._binary = base / "fixture"
        cls._source.write_text(CPP_SOURCE, encoding="utf-8")
        result = subprocess.run(
            ["c++", "-std=c++17", "-O0", str(cls._source), "-o", str(cls._binary)],
            cwd=ROOT, env=SAFE_ENV, capture_output=True, text=True, timeout=15, check=False,
        )
        if result.returncode != 0:
            raise RuntimeError("fixture compilation failed before behavioral tests:\n" + result.stderr)
    def _case(self, name: str) -> dict[str, int]:
        result = subprocess.run(
            [str(self._binary), name], cwd=ROOT, env=SAFE_ENV,
            capture_output=True, text=True, timeout=16, check=False,
        )
        self.assertEqual(
            result.returncode, 0,
            f"fixture {name} failed (stdout={result.stdout!r}, stderr={result.stderr!r})",
        )
        values: dict[str, int] = {}
        for line in result.stdout.splitlines():
            key, separator, value = line.partition("=")
            if separator: values[key] = int(value)
        self.assertEqual(values.get("api_errors"), 0, values)
        return values
    def _assert_fields(self, values: dict[str, int], **expected: int) -> None:
        for key, value in expected.items():
            with self.subTest(field=key):
                self.assertEqual(values.get(key), value)

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

    def test_10_range_teardown_failure_stops_rebind_and_same_map_fast_success(self) -> None:
        self._assert_fields(self._case("range-teardown-failure"), r1=0, r2=0,
            register_calls=0, unregister_calls=1, lookup_calls=0,
            physical_registers=1, live=0, cache_release_calls=2, registered=0,
            host_is_expected=1, size_is_expected=1, device_is_expected=1)

if __name__ == "__main__":
    unittest.main()
