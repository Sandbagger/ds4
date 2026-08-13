#!/usr/bin/env python3
"""Boundary tests for DS4's five compact-runtime JSON wire schemas."""

from __future__ import annotations

import copy
import importlib.metadata
import json
import math
import struct
import unittest
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = ROOT / "schemas"
REQUIREMENTS = ROOT / "gguf-tools/quality-testing/requirements-compact-runtime.txt"
DRAFT = "https://json-schema.org/draft/2020-12/schema"
U64_MAX, SAFE_INT = "18446744073709551615", (1 << 53) - 1
SCHEMA_FILES = {
    "version": ("ds4-version-v1.schema.json", "ds4.version/v1"),
    "runtime": ("ds4-runtime-v1.schema.json", "ds4.runtime/v1"),
    "request": ("ds4-runtime-request-v1.schema.json", "ds4.runtime.request/v1"),
    "admission": ("ds4-token-admission-v1.schema.json", "ds4.token-admission/v1"),
    "bundle": ("ds4-laguna-compact-runtime-v1.schema.json", "ds4.laguna.compact-runtime/v1"),
}
PINS = {"jsonschema": "4.25.1", "rfc8785": "0.1.4"}
BACKENDS = ("cpu", "metal", "cuda", "rocm")
STATES = ("starting", "ready", "draining", "unsafe")
TERMINAL = ("completed", "cancelled", "rejected", "recoverable_error", "unsafe_error")
REJECTIONS = ("model_mismatch", "invalid_request", "invalid_output_tokens", "unsupported_tool_choice", "context_overflow")
STATUSES = ("passed", "failed", "invalid")
FAILURES = ("gate_failed", "infrastructure_invalid", "identity_mismatch", "evidence_invalid", "timeout", "process_failed")
VIOLATIONS = (
    "invalid_config", "overflow", "unknown_callsite", "duplicate_callsite",
    "unclassified_callsite", "capacity", "duplicate_id", "address_overflow",
    "undercharge", "overlap", "callsite_bound", "category_bound",
    "owned_total_bound", "report_bound", "qualification_total_bound",
    "relation", "not_live", "live_relation", "external_attribution",
)
# Public names mirror ds4_runtime_category and ds4_runtime_report exactly.
CATEGORIES = ("static_weights", "expert_cache_payload", "cache_metadata_address_tables", "kv_state", "graph_scratch", "pinned_staging", "other_host", "other_cuda")
REPORTS = ("model_mapped_virtual", "model_mapping_registered", "model_source_resident", "host_library_unattributed", "cuda_library_unattributed")
SHA_A, SHA_B, SHA_C = "0123456789abcdef" * 4, "fedcba9876543210" * 4, "1234567890abcdef" * 4
REV_A, REV_B = "0123456789abcdef0123456789abcdef01234567", "76543210fedcba9876543210fedcba9876543210"
UUID_A, UUID_B = "123e4567-e89b-12d3-a456-426614174000", "223e4567-e89b-12d3-a456-426614174001"

try:
    from jsonschema import Draft202012Validator, FormatChecker, ValidationError, validators
except ModuleNotFoundError:
    Draft202012Validator = FormatChecker = ValidationError = validators = None  # type: ignore[assignment,misc]
try:
    import rfc8785
except ModuleNotFoundError:
    rfc8785 = None


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key: {key}")
        result[key] = value
    return result


def loads_strict(payload: str) -> Any:
    def reject(value: str) -> None:
        raise ValueError(f"non-finite JSON value: {value}")
    def finite(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            reject(value)
        return parsed
    return json.loads(payload, object_pairs_hook=_strict_pairs, parse_constant=reject, parse_float=finite)


def stat(seed: int) -> dict[str, str]:
    return {"device": str(seed), "inode": str(seed + 1), "size_bytes": str(seed + 2), "mtime_ns": str(seed + 3)}


def measurement(seed: int) -> dict[str, str]:
    return {"current_bytes": str(seed), "bound_bytes": str(seed + 100), "peak_bytes": str(seed + 10)}


def version() -> dict[str, Any]:
    return {"schema": "ds4.version/v1", "revision": REV_A, "dirty": False, "backend": "cuda", "features": ["laguna", "ssd_streaming"]}


def runtime(violation: bool = False) -> dict[str, Any]:
    return {
        "schema": "ds4.runtime/v1", "instance_id": UUID_A, "snapshot_seq": "7", "state": "ready",
        "build": {key: value for key, value in version().items() if key != "schema"},
        "executable": stat(10),
        "model": {"id": "laguna-s-2.1", "family": "laguna", **stat(20)},
        "config": {"context_tokens": 32768, "prefill_chunk_tokens": 4096, "session_slots": 1, "ssd_streaming": True, "ssd_streaming_cache_bytes": "8589934592"},
        "limits": {"effective_context_tokens": 32768, "effective_prefill_chunk_tokens": 4096, "effective_session_slots": 1, "expert_cache_limit_bytes": "8589934592"},
        "allocations": {
            "categories": {name: measurement(i + 1) for i, name in enumerate(CATEGORIES)},
            "reports": {name: measurement(i + 30) for i, name in enumerate(REPORTS)},
            "owned_total": measurement(60), "qualification_total": measurement(70),
            "configured_prefill_rows": 4096, "allocated_prefill_rows": 4096,
        },
        "counters": {
            "cache_acquire_hits": "11", "cache_acquire_misses": "12", "cache_evictions": "13",
            "model_file_read_operations": "14", "model_file_read_bytes": "15", "model_file_read_ns": "16",
            "host_to_device_bytes": "17", "host_to_device_ns": "18",
            "page_advice_attempts": "19", "page_advice_bytes": "20", "page_advice_failures": "21",
        },
        "violations": [{"code": "category_bound", "latched_snapshot_seq": "6"}] if violation else [],
    }


def request(status: str = "completed", advice: bool = False) -> dict[str, Any]:
    return {
        "schema": "ds4.runtime.request/v1", "request_id": UUID_B,
        "instance_id": UUID_A, "snapshot_seq": "7",
        "prompt_tokens": 22, "generated_tokens": 8 if status != "rejected" else 0,
        "ttft_ns": "1000000" if status == "completed" else None,
        "prefill_tokens_per_second": 120.5, "visible_decode_tokens_per_second": 20.25,
        "wall_time_ns": "2000000", "cache_hits": "1", "cache_misses": "2", "cache_evictions": "0",
        "model_file_read_operations": "3", "model_file_read_bytes": "4", "model_file_read_ns": "5",
        "host_to_device_bytes": "6", "host_to_device_ns": "7",
        "page_advice_attempts": "8", "page_advice_bytes": "9", "page_advice_failures": "0",
        "page_advice_complete_monotonic_ns": "3000000" if advice else None,
        "terminal_status": status,
    }


def admission(fits: bool = True, code: str | None = None) -> dict[str, Any]:
    return {
        "schema": "ds4.token-admission/v1", "model": "laguna-s-2.1",
        "template_revision": "poolside-laguna-s-2.1-native-nothink-v1", "templated_input_tokens": 100,
        "requested_output_tokens": 20, "context_tokens": 128, "fits": fits,
        "rejection_code": code if code is not None else (None if fits else "context_overflow"),
    }


def evidence(name: str, digest: str = SHA_C) -> dict[str, str]:
    return {"path": f"evidence/{name}.json", "sha256": digest}


def scalar(value: Any, kind: str) -> dict[str, Any]:
    return {"kind": kind, "value": value}


def gate(gate_id: str, status: str = "passed", measured: Any = True, threshold: Any = True, unit: str = "boolean", comparison: str = "eq") -> dict[str, Any]:
    kinds = {"boolean": "boolean", "bytes": "uint64", "count": "integer", "ratio": "number", "seconds": "number"}
    kind = kinds[unit]
    return {"gate_id": gate_id, "status": status, "measured": scalar(measured, kind), "threshold": {"comparison": comparison, **scalar(threshold, kind)}, "unit": unit, "evidence": [evidence(gate_id)]}


def profile(status: str = "passed", cache_gib: int = 8) -> dict[str, Any]:
    failure = None if status == "passed" else {"code": "gate_failed" if status == "failed" else "infrastructure_invalid", "message": "profile did not qualify", "evidence": [evidence("profile-failure", SHA_B)]}
    prompt_orders = {8: (512, 2048, 28672, 8192), 12: (2048, 8192, 512, 28672), 16: (8192, 28672, 2048, 512)}
    profile_digests = {8: SHA_A, 12: SHA_B, 16: SHA_C}
    qualification_total = str((cache_gib + 16) << 30)
    return {
        "profile_id": f"cache-{cache_gib}gib", "profile_manifest_sha256": profile_digests[cache_gib],
        "config": {"context_tokens": 32768, "prefill_chunk_tokens": 4096, "session_slots": 1, "ssd_streaming": True, "ssd_streaming_cache_bytes": str(cache_gib << 30)},
        "allocation_plan_sha256": SHA_B,
        "bounds": {
            "categories": {name: str((i + 1) * 1024) for i, name in enumerate(CATEGORIES)},
            "reports": {name: str((i + 20) * 1024) for i, name in enumerate(REPORTS)},
            "owned_non_cache_bytes": "8589934592", "owned_total_bytes": "17179869184",
            "qualification_non_cache_bytes": "17179869184", "qualification_total_bytes": qualification_total,
        },
        "status": status, "gates": [gate("profile-memory-bound", status, "17179869184", qualification_total, "bytes", "lte")],
        "results": {
            "resident_peak_bytes": "68719476736", "streamed_peak_bytes": "25769803776",
            "reduction_bytes": "42949672960", "reduction_ratio": 0.625,
            "benchmark_samples": [
                {
                    "prompt_id": f"native-{prompt_tokens}", "prompt_tokens": prompt_tokens,
                    "repetition": repetition, "mode": "streamed",
                    "request_id": f"223e4567-e89b-12d3-a456-{cache_gib:02d}{sample_index:010d}",
                    "instance_id": f"323e4567-e89b-12d3-a456-{cache_gib:02d}{sample_index // 4:010d}",
                    "snapshot_seq": str(sample_index % 4 + 1),
                    "ttft_ns": "1000000", "wall_time_ns": str(2000000 + sample_index),
                    "visible_decode_tokens_per_second": 20.25,
                    "qualification_total_peak_bytes": qualification_total,
                    "model_inode_resident_bytes": "1048576", "terminal_status": "completed",
                    "request_metrics_sha256": SHA_C, "evidence": [evidence(f"sample-{cache_gib}-{sample_index}")],
                }
                for sample_index, (prompt_tokens, repetition) in enumerate(
                    (pair for prompt_tokens in prompt_orders[cache_gib] for pair in ((prompt_tokens, "cold"), (prompt_tokens, "warm-1"), (prompt_tokens, "warm-2"), (prompt_tokens, "warm-3")))
                )
            ],
            "model_inode_resident_bytes": "1048576",
        },
        "failure": failure,
    }


def bundle(status: str = "passed") -> dict[str, Any]:
    return {
        "schema": "ds4.laguna.compact-runtime/v1", "created_at": "2026-08-13T12:34:56.123456789Z", "status": status,
        "subject": {
            "revision": REV_A, "dirty": False, "version_sha256": SHA_A,
            "binaries": [
                {"role": role, "binary_sha256": digest, "executable": stat(100 + index * 10)}
                for index, (role, digest) in enumerate((("server", SHA_A), ("bench", SHA_B), ("eval", SHA_C)))
            ],
        },
        "host": {
            "hostname": "dgx-spark", "architecture": "aarch64", "kernel_release": "6.11.0",
            "kernel_version": "#1 SMP PREEMPT_DYNAMIC", "cuda_driver_version": "580.126.09",
            "cuda_runtime_version": "13.0", "gpu_uuid": "GPU-123e4567-e89b-12d3-a456-426614174000",
            "filesystem": {"mount_point": "/home", "type": "ext4", "source": "/dev/nvme0n1p2", "device": "259:2", "options": "rw,relatime"},
            "nvme": {"device": "/dev/nvme0n1", "model": "DGX Spark NVMe", "serial": "DS4NVME0001", "firmware_revision": "1.0"},
            "io": {"direct_io": False, "cold_preparation_advice": "madvise_random+posix_fadvise_dontneed+madvise_dontneed+linux_madv_pageout_residual", "runtime_disposal_advice": "madvise_dontneed"},
        },
        "model": {"repository": "poolside/Laguna-S-2.1-GGUF", "revision": REV_B, "filename": "laguna-s-2.1-Q4_K_M.gguf", "size_bytes": "68248759648", "sha256": SHA_C, "device": "259", "inode": "1234567", "mtime_ns": "1786600000000000000", "served_model_id": "laguna-s-2.1"},
        "schemas": [
            {"schema_id": schema_id, "sha256": SHA_A}
            for schema_id in (
                "ds4.version/v1", "ds4.runtime/v1", "ds4.runtime.request/v1",
                "ds4.token-admission/v1", "ds4.laguna.compact-runtime/v1",
                "ds4.compact-runtime-benchmark/v1",
            )
        ],
        "oracle": {"schema": "laguna-resident-promoted-v2", "policy": "single-poolside-v1", "manifest_sha256": SHA_B, "tokenizer_runtime_revision": REV_A, "llama_revision": REV_B},
        "benchmark_manifest": {"schema_id": "ds4.compact-runtime-benchmark/v1", "sha256": SHA_C},
        "global_gates": [gate("global-runtime-bound", status)],
        "profiles": [profile(status, cache_gib) for cache_gib in (8, 12, 16)],
        "evidence_root_sha256": SHA_A,
    }


def variants() -> dict[str, list[dict[str, Any]]]:
    return {
        "version": [version()], "runtime": [runtime(), runtime(True)],
        "request": [request(), request("rejected"), request("cancelled", True)],
        "admission": [admission(), admission(False)],
        "bundle": [bundle(status) for status in STATUSES],
    }


Path = tuple[str | int, ...]


def get(value: Any, path: Path) -> Any:
    for part in path:
        value = value[part]
    return value


def walk(value: Any, path: Path = ()) -> Iterator[tuple[Path, str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = path + (key,)
            yield child_path, key, child
            yield from walk(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from walk(child, path + (index,))


def objects(value: Any, path: Path = ()) -> Iterator[Path]:
    if isinstance(value, dict):
        yield path
        for key, child in value.items():
            yield from objects(child, path + (key,))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from objects(child, path + (index,))


def put(value: Any, path: Path, replacement: Any) -> None:
    get(value, path[:-1])[path[-1]] = replacement


def label(path: Path) -> str:
    return ".".join(map(str, path))


SCHEMAS_PRESENT = all((SCHEMA_DIR / filename).is_file() for filename, _ in SCHEMA_FILES.values())


def exact_number_kind(validator: Any, expected: str, instance: Any, schema: Any) -> Iterator[Any]:
    exact_integer = type(instance) is int
    finite_number = type(instance) in (int, float) and math.isfinite(instance)
    if (expected == "integer" and not exact_integer) or (expected == "number" and not finite_number):
        yield ValidationError(f"{instance!r} is not an exact finite JSON {expected}")


def sorted_unique(validator: Any, enabled: bool, instance: Any, schema: Any) -> Iterator[Any]:
    if enabled and isinstance(instance, list):
        if not all(type(item) is str for item in instance) or len(instance) != len(set(instance)) or instance != sorted(instance):
            yield ValidationError(f"{instance!r} is not sorted and unique")


ExactValidator = (
    validators.extend(Draft202012Validator, {"x-ds4-number-kind": exact_number_kind, "x-ds4-sorted-unique": sorted_unique})
    if validators is not None else None
)


class DependencyTests(unittest.TestCase):
    def test_qualification_dependencies_are_exactly_pinned_and_installed(self) -> None:
        lines = [line.split("#", 1)[0].strip() for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines()]
        pins = dict(line.split("==", 1) for line in lines if line)
        for name, expected in PINS.items():
            with self.subTest(name=name):
                self.assertEqual(pins.get(name), expected)
                try:
                    actual = importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:
                    self.fail(f"missing qualification dependency: {name}=={expected}")
                self.assertEqual(actual, expected)


class SchemaPresenceTests(unittest.TestCase):
    def test_all_five_normative_schemas_exist(self) -> None:
        missing = [str(SCHEMA_DIR / filename) for filename, _ in SCHEMA_FILES.values() if not (SCHEMA_DIR / filename).is_file()]
        self.assertEqual(missing, [])


@unittest.skipUnless(SCHEMAS_PRESENT and ExactValidator is not None, "requires all five schemas and jsonschema")
class SchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schemas, cls.validators = {}, {}
        for name, (filename, _) in SCHEMA_FILES.items():
            schema = loads_strict((SCHEMA_DIR / filename).read_text(encoding="utf-8"))
            cls.schemas[name] = schema
            cls.validators[name] = ExactValidator(schema, format_checker=FormatChecker())

    def valid(self, name: str, value: Any) -> None:
        errors = list(self.validators[name].iter_errors(value))
        self.assertEqual(errors, [], "\n".join(f"{label(tuple(e.absolute_path))}: {e.message}" for e in errors))

    def invalid(self, name: str, value: Any) -> None:
        self.assertTrue(list(self.validators[name].iter_errors(value)), f"{name} accepted {value!r}")

    def test_schema_documents_are_independent_closed_draft_2020_12(self) -> None:
        for name, schema in self.schemas.items():
            with self.subTest(name=name):
                Draft202012Validator.check_schema(schema)
                self.assertEqual(schema["$schema"], DRAFT)
                self.assertEqual(schema["$id"], SCHEMA_FILES[name][1])
            for path, key, node in walk({"root": schema}):
                if key == "$ref":
                    self.assertTrue(node.startswith("#/"), f"external ref at {name}:{label(path)}")
                if isinstance(node, dict) and node.get("type") == "object":
                    self.assertIs(node.get("additionalProperties"), False, f"open object at {name}:{label(path)}")
                    self.assertEqual(set(node.get("required", [])), set(node.get("properties", {})), f"optional property at {name}:{label(path)}")

    def test_valid_fixtures_schema_constants_and_recursive_closure(self) -> None:
        for name, fixture_list in variants().items():
            for index, value in enumerate(fixture_list):
                self.valid(name, value)
                self.assertEqual(value["schema"], SCHEMA_FILES[name][1])
                changed = copy.deepcopy(value); changed["schema"] += ".unknown"; self.invalid(name, changed)
                for path in objects(value):
                    changed = copy.deepcopy(value); get(changed, path)["unexpected_contract_field"] = True
                    with self.subTest(name=name, index=index, unknown=label(path)):
                        self.invalid(name, changed)

    def test_every_fixture_property_is_required_and_nonnullable(self) -> None:
        nullable = {("request", "ttft_ns"), ("request", "page_advice_complete_monotonic_ns"), ("admission", "rejection_code"), ("bundle", "failure")}
        for name, fixture_list in variants().items():
            for value in fixture_list:
                for path, key, current in walk(value):
                    changed = copy.deepcopy(value); del get(changed, path[:-1])[key]; self.invalid(name, changed)
                    if current is not None and (name, key) not in nullable:
                        changed = copy.deepcopy(value); put(changed, path, None); self.invalid(name, changed)

    def test_every_uint64_string_accepts_boundaries_and_rejects_noncanonical_values(self) -> None:
        invalid_values: tuple[Any, ...] = ("00", "01", "+1", "-1", "1e3", "1.0", 1, 1.0, True, "18446744073709551616")
        seen: set[tuple[str, Path]] = set()
        for name, fixture_list in variants().items():
            for value in fixture_list:
                for path, key, current in walk(value):
                    if not isinstance(current, str) or not current.isascii() or not current.isdecimal() or key.endswith(("sha256", "revision")):
                        continue
                    if (name, path) in seen: continue
                    seen.add((name, path))
                    for boundary in ("0", U64_MAX):
                        changed = copy.deepcopy(value); put(changed, path, boundary); self.valid(name, changed)
                    for invalid in invalid_values:
                        changed = copy.deepcopy(value); put(changed, path, invalid); self.invalid(name, changed)
        self.assertGreater(len(seen), 100)

    def test_token_counts_are_bounded_json_integers_not_bools(self) -> None:
        nonnegative = {"generated_tokens", "templated_input_tokens", "requested_output_tokens"}
        positive = {"context_tokens", "prefill_chunk_tokens", "effective_context_tokens", "effective_prefill_chunk_tokens", "configured_prefill_rows", "allocated_prefill_rows", "session_slots", "effective_session_slots", "prompt_tokens"}
        seen = 0
        for name, fixture_list in variants().items():
            value = fixture_list[0]
            for path, key, current in walk(value):
                if key not in nonnegative | positive or type(current) is not int: continue
                seen += 1
                for boundary in ((0, SAFE_INT) if key in nonnegative else (1, SAFE_INT)):
                    changed = copy.deepcopy(value); put(changed, path, boundary); self.valid(name, changed)
                for invalid in (-1, SAFE_INT + 1, 1.0, True, "1", None) + ((0,) if key in positive else ()):
                    changed = copy.deepcopy(value); put(changed, path, invalid); self.invalid(name, changed)
        self.assertGreaterEqual(seen, 15)

    def test_rates_are_finite_nonnegative_json_numbers(self) -> None:
        for key in ("prefill_tokens_per_second", "visible_decode_tokens_per_second"):
            for candidate in (0, 0.5, float.fromhex("0x1.fffffffffffffp+1023")):
                changed = request(); changed[key] = candidate; self.valid("request", changed)
            for candidate in (-1, math.nan, math.inf, -math.inf, "1.0", True, None):
                changed = request(); changed[key] = candidate; self.invalid("request", changed)

    def test_hash_revision_uuid_timestamp_and_sorted_feature_formats(self) -> None:
        for name, fixture_list in variants().items():
            value = fixture_list[0]
            for path, key, current in walk(value):
                if not isinstance(current, str): continue
                width = 64 if key == "sha256" or key.endswith("_sha256") else 40 if key in {"revision", "tokenizer_runtime_revision", "llama_revision"} else 0
                if width:
                    for candidate in (current.upper(), current[:-1], current + "0", "g" * width, "0" * width):
                        changed = copy.deepcopy(value); put(changed, path, candidate); self.invalid(name, changed)
        for name, value, path in (
            ("runtime", runtime(), ("instance_id",)),
            ("request", request(), ("request_id",)),
            ("request", request(), ("instance_id",)),
        ):
            for candidate in (UUID_A.upper(), UUID_A.replace("-", ""), "not-a-uuid", "00000000-0000-0000-0000-000000000000"):
                changed = copy.deepcopy(value); put(changed, path, candidate); self.invalid(name, changed)
        for candidate in ("2026-08-13T12:34:56Z", "2026-08-13T12:34:56.123456789Z"):
            changed = bundle(); changed["created_at"] = candidate; self.valid("bundle", changed)
        for candidate in ("2026-08-13T12:34:56+00:00", "2026-08-13 12:34:56Z", "2026-13-13T12:34:56Z"):
            changed = bundle(); changed["created_at"] = candidate; self.invalid("bundle", changed)
        for name, value, path in (("version", version(), ("features",)), ("runtime", runtime(), ("build", "features"))):
            for candidate in ([], ["laguna"], ["laguna", "ssd_streaming"]):
                changed = copy.deepcopy(value); put(changed, path, candidate); self.valid(name, changed)
            for candidate in (["ssd_streaming", "laguna"], ["laguna", "laguna"], ["laguna", 1], "laguna"):
                changed = copy.deepcopy(value); put(changed, path, candidate); self.invalid(name, changed)

    def test_stable_enums_and_null_coherence(self) -> None:
        for backend in BACKENDS:
            changed = version(); changed["backend"] = backend; self.valid("version", changed)
        for state in STATES:
            changed = runtime(); changed["state"] = state; self.valid("runtime", changed)
        for code in VIOLATIONS:
            changed = runtime(True); changed["violations"][0]["code"] = code; self.valid("runtime", changed)
        for status in TERMINAL: self.valid("request", request(status))
        for code in REJECTIONS: self.valid("admission", admission(False, code))
        for status in STATUSES: self.valid("bundle", bundle(status))
        for bad in ("vulkan", "CUDA", None):
            changed = version(); changed["backend"] = bad; self.invalid("version", changed)
        for bad in ("stopped", "READY", None):
            changed = runtime(); changed["state"] = bad; self.invalid("runtime", changed)
        for bad in ("none", "CATEGORY_BOUND", "unknown"):
            changed = runtime(True); changed["violations"][0]["code"] = bad; self.invalid("runtime", changed)
        for status in TERMINAL:
            for ttft in (None, "1"):
                changed = request(status); changed["ttft_ns"] = ttft; self.valid("request", changed)
        fitted = admission(); fitted["rejection_code"] = "context_overflow"; self.invalid("admission", fitted)
        refused = admission(False); refused["rejection_code"] = None; self.invalid("admission", refused)

    def test_bundle_profile_failure_and_discriminated_scalar_coherence(self) -> None:
        changed = bundle(); changed["profiles"][0]["failure"] = profile("failed")["failure"]; self.invalid("bundle", changed)
        changed = bundle("failed"); changed["profiles"][0]["failure"] = None; self.invalid("bundle", changed)
        for code in FAILURES:
            changed = bundle("failed"); changed["profiles"][0]["failure"]["code"] = code; self.valid("bundle", changed)
        for measured, threshold, unit in ((True, True, "boolean"), ("1", "2", "bytes"), (3, 4, "count"), (0.5, 0.45, "ratio")):
            changed = bundle(); changed["global_gates"] = [gate("typed-scalar", measured=measured, threshold=threshold, unit=unit)]; self.valid("bundle", changed)
        for invalid in (math.nan, math.inf, {"value": 1}, [1], None):
            changed = bundle(); changed["global_gates"][0]["measured"]["value"] = invalid; self.invalid("bundle", changed)
        changed = bundle(); changed["global_gates"][0]["measured"] = scalar("1", "number"); self.invalid("bundle", changed)
        changed = bundle(); changed["global_gates"][0]["threshold"]["kind"] = "uint64"; self.invalid("bundle", changed)

    def test_bundle_mixed_status_propagation_fixtures(self) -> None:
        mixed = bundle("passed")
        mixed["profiles"] = [profile("passed", 8), profile("failed", 12), profile("invalid", 16)]
        self.valid("bundle", mixed)
        failed = bundle("failed")
        failed["global_gates"] = [gate("global-runtime-bound", "passed")]
        failed["profiles"] = [profile("failed", 8), profile("invalid", 12), profile("failed", 16)]
        self.valid("bundle", failed)
        invalid = bundle("invalid")
        invalid["global_gates"] = [gate("global-runtime-bound", "passed")]
        invalid["profiles"] = [profile("invalid", cache_gib) for cache_gib in (8, 12, 16)]
        self.valid("bundle", invalid)
        # Aggregate propagation itself is evaluated by the Task 20 bundle builder;
        # this schema freezes the closed item shapes and failure/status enums.

    def test_bundle_freezes_schema_records_profiles_and_benchmark_samples(self) -> None:
        for key in ("schemas", "profiles"):
            changed = bundle(); changed[key].reverse(); self.invalid("bundle", changed)
            changed = bundle(); changed[key].pop(); self.invalid("bundle", changed)
            changed = bundle(); changed[key][1] = copy.deepcopy(changed[key][0]); self.invalid("bundle", changed)
        changed = bundle(); changed["profiles"][0]["results"]["benchmark_samples"].pop(); self.invalid("bundle", changed)
        changed = bundle(); changed["profiles"][0]["results"]["benchmark_samples"].reverse(); self.invalid("bundle", changed)
        changed = bundle(); changed["profiles"][0]["results"]["benchmark_samples"][0]["prompt_id"] = "native-2048"; self.invalid("bundle", changed)
        for collection, names in (("categories", CATEGORIES), ("reports", REPORTS)):
            changed = bundle(); del changed["profiles"][0]["bounds"][collection][names[0]]; self.invalid("bundle", changed)
            changed = bundle(); changed["profiles"][0]["bounds"][collection]["future_category"] = "0"; self.invalid("bundle", changed)

    def test_allocation_maps_exactly_freeze_current_c_enum_names(self) -> None:
        value = runtime()
        self.assertEqual(tuple(value["allocations"]["categories"]), CATEGORIES)
        self.assertEqual(tuple(value["allocations"]["reports"]), REPORTS)
        for collection, names in (("categories", CATEGORIES), ("reports", REPORTS)):
            for name in names:
                changed = runtime(); del changed["allocations"][collection][name]; self.invalid("runtime", changed)
            changed = runtime(); changed["allocations"][collection]["future_category"] = measurement(1); self.invalid("runtime", changed)

    def test_runtime_model_identity_rejects_whitespace_and_controls(self) -> None:
        for key in ("id", "family"):
            for candidate in ("", " ", "\t", "laguna\n", "laguna\x00"):
                changed = runtime(); changed["model"][key] = candidate; self.invalid("runtime", changed)


@unittest.skipUnless(rfc8785 is not None, "requires rfc8785==0.1.4")
class RFC8785Tests(unittest.TestCase):
    def test_rfc_primitive_string_and_property_order_vector(self) -> None:
        value = {"numbers": [333333333.33333329, 1e30, 4.50, 2e-3, 1e-27], "string": "\u20ac$\u000f\nA'B\"\\\\\"/", "literals": [None, True, False]}
        expected = ('{"literals":[null,true,false],"numbers":[333333333.3333333,1e+30,4.5,0.002,1e-27],"string":"€$\\u000f\\nA\'B\\"\\\\\\\\\\"/"}').encode("utf-8")
        self.assertEqual(rfc8785.dumps(value), expected)

    def test_rfc_appendix_b_number_vectors(self) -> None:
        vectors = (
            ("0000000000000000", "0"), ("8000000000000000", "0"), ("0000000000000001", "5e-324"),
            ("8000000000000001", "-5e-324"), ("7fefffffffffffff", "1.7976931348623157e+308"),
            ("ffefffffffffffff", "-1.7976931348623157e+308"), ("4340000000000000", "9007199254740992"),
            ("c340000000000000", "-9007199254740992"), ("4430000000000000", "295147905179352830000"),
            ("44b52d02c7e14af5", "9.999999999999997e+22"), ("44b52d02c7e14af6", "1e+23"),
            ("44b52d02c7e14af7", "1.0000000000000001e+23"), ("444b1ae4d6e2ef4e", "999999999999999700000"),
            ("444b1ae4d6e2ef4f", "999999999999999900000"), ("444b1ae4d6e2ef50", "1e+21"),
            ("3eb0c6f7a0b5ed8c", "9.999999999999997e-7"), ("3eb0c6f7a0b5ed8d", "0.000001"),
            ("41b3de4355555553", "333333333.3333332"), ("41b3de4355555554", "333333333.33333325"),
            ("41b3de4355555555", "333333333.3333333"), ("41b3de4355555556", "333333333.3333334"),
            ("41b3de4355555557", "333333333.33333343"),
            ("becbf647612f3696", "-0.0000033333333333333333"), ("43143ff3c1cb0959", "1424953923781206.2"),
        )
        for bits, expected in vectors:
            with self.subTest(bits=bits):
                self.assertEqual(rfc8785.dumps(struct.unpack(">d", bytes.fromhex(bits))[0]), expected.encode())

    def test_utf16_property_order_unsigned_utf8_path_order_and_invalid_ijson(self) -> None:
        astral, bmp = "\U00010000", "\ue000"
        self.assertEqual(rfc8785.dumps({bmp: 1, astral: 2}), ('{"' + astral + '":2,"' + bmp + '":1}').encode())
        paths = [f"evidence/{astral}.json", f"evidence/{bmp}.json"]
        self.assertEqual(sorted(paths, key=lambda path: path.encode()), [f"evidence/{bmp}.json", f"evidence/{astral}.json"])
        with self.assertRaisesRegex(ValueError, "duplicate key"): loads_strict('{"a":1,"a":2}')
        for spelling in ("NaN", "Infinity", "-Infinity", "1e9999"):
            with self.assertRaisesRegex(ValueError, "non-finite"): loads_strict('{"value":' + spelling + "}")
        for value in (math.nan, math.inf, -math.inf, "\ud800", "\udead"):
            with self.assertRaises(rfc8785.CanonicalizationError): rfc8785.dumps(value)
        for value in (SAFE_INT + 1, -SAFE_INT - 1):
            with self.assertRaises(rfc8785.CanonicalizationError): rfc8785.dumps(value)
        # 0.1.4 raises UnicodeEncodeError while sorting a surrogate object key;
        # it is still a required fail-closed outcome for input outside I-JSON.
        with self.assertRaises((rfc8785.CanonicalizationError, UnicodeEncodeError)):
            rfc8785.dumps({"\ud800": "value"})


if __name__ == "__main__":
    unittest.main()
