#!/usr/bin/env python3
"""Prepare the Darwin CPU-only heap regression, then exec its verified controller.

Run with an owned, canonical TMPDIR. Configuration and logs remain in the
reported fresh workspace. The runner removes its own compiled fixture tree.
Owned HOME/XDG/TMPDIR leaves are hygiene, not a filesystem or GPU sandbox.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import stat
import sys
import tempfile

SOURCE_CAP = 4 << 20
INPUT_CAP = 16 << 20
JSON_CAP = 1 << 17
DEPENDENCIES = (
    'ds4.c',
    'ds4_distributed.c',
    'ds4_tp.c',
    'ds4_ssd.c',
    'ds4_laguna_stream.c',
    'ds4_runtime.c',
    'ds4_qualification_control.c',
    'ds4_plan_io.c',
    'ds4_laguna_plan.c',
    'ds4_layer_pack.c',
    'ds4.h',
    'ds4_distributed.h',
    'ds4_laguna_plan.h',
    'ds4_laguna_stream.h',
    'ds4_tp.h',
    'ds4_layer_pack.h',
    'ds4_gpu_mgpu.h',
    'ds4_gpu.h',
    'ds4_ssd.h',
    'ds4_runtime.h',
    'ds4_plan_io.h',
    'ds4_gpu_resident.h',
    'ds4_streaming_hotlist.inc',
    'ds4_streaming_hotlist_glm52.inc',
)
PINS = {
    'tests/test_engine_session_lifetime.c': 'a35fbb7f2dc00f571028f995c65e4106b12e6bcfef996168f20a5e36d18c2d25',
    'tests/test_engine_session_lifetime.py': '3619c936a3f7415ba332ab4f3db457793c89954a6f550eb393902e554a30cc66',
    'tests/engine_session_lifetime_controller.py': '5fdf88f748c7a714e39ce16b356cc09afa7b1bddc2109074621ef44d256578ca',
    'tests/engine_session_lifetime.mk': 'f24ff362027767215d461432029e43b7f0b05ba590a9cbf411220e43fe7b14d6',
}
METRICS_SHA256 = '83ab234fa8abbf2c2cea5d983df0789e4840a5782bbc52199f116c671664a855'
CASES = ["empty", "live-one", "live-two", "retained-after-unlock"]
EVIDENCE = "actual_core_DS4_NO_GPU_heap_lifetime_synthetic_pthread_fault"
LIMITS = {"wall_seconds": 240, "cpu_seconds": 240, "compiler_seconds": 120,
          "native_case_seconds": 10, "core_bytes": 0, "file_size_bytes": 64 << 20,
          "log_bytes": 1 << 20, "native_stdout_bytes": 32768,
          "term_grace_seconds": 8, "kill_grace_seconds": 2, "process_ceiling": 4096}


class PreparationError(RuntimeError):
    """Refused setup; never a product regression or qualification result."""


def require(condition, message):
    if not condition:
        raise PreparationError(message)


def _canonical(path):
    return (path.is_absolute() and os.path.normpath(str(path)) == str(path)
            and os.path.realpath(path) == str(path))


def _directory(path, mode=None):
    path = Path(path)
    require(_canonical(path), "directory must be canonical: %s" % path)
    st = os.lstat(path)
    actual_mode = stat.S_IMODE(st.st_mode)
    require(stat.S_ISDIR(st.st_mode) and st.st_uid == os.getuid(),
            "directory type/owner: %s" % path)
    require(mode is None or actual_mode == mode, "directory mode: %s" % path)
    return [st.st_uid, st.st_dev, st.st_ino, actual_mode]


def _identity(st):
    return (st.st_uid, st.st_dev, st.st_ino, st.st_mode, st.st_nlink,
            st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def _read(path, cap=SOURCE_CAP):
    path = Path(path)
    require(_canonical(path), "file must be canonical: %s" % path)
    pre = os.lstat(path)
    require(stat.S_ISREG(pre.st_mode) and pre.st_uid == os.getuid()
            and pre.st_nlink == 1 and 0 <= pre.st_size <= cap,
            "file type/owner/link/size: %s" % path)
    identity = _identity(pre)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        require(_identity(os.fstat(fd)) == identity, "file changed before open")
        chunks, total = [], 0
        while True:
            chunk = os.read(fd, min(65536, cap + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            require(total <= cap, "input read overflow")
        require(_identity(os.fstat(fd)) == identity, "file changed during read")
    finally:
        os.close(fd)
    require(_identity(os.lstat(path)) == identity, "input path changed")
    raw = b"".join(chunks)
    require(len(raw) == pre.st_size, "input size changed")
    return raw, pre


def _record(path, raw, st):
    return {"path": str(path), "bytes": len(raw),
            "lines": len(raw.decode("utf-8").splitlines()),
            "sha256": hashlib.sha256(raw).hexdigest(), "uid": st.st_uid,
            "device": st.st_dev, "inode": st.st_ino,
            "mode": stat.S_IMODE(st.st_mode), "nlink": st.st_nlink}


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: %s" % key)
        result[key] = value
    return result


def _constant(value):
    raise PreparationError("nonfinite JSON value: " + value)


def _json(raw):
    require(len(raw) <= JSON_CAP, "JSON input bound")
    return json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs,
                      parse_constant=_constant)


def _new_json(path, document):
    raw = (json.dumps(document, indent=2, allow_nan=False) + "\n").encode("utf-8")
    require(len(raw) <= JSON_CAP, "generated JSON bound")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        opened = os.fstat(fd)
        require(stat.S_ISREG(opened.st_mode) and opened.st_uid == os.getuid()
                and opened.st_nlink == 1, "new output identity")
        view = memoryview(raw)
        while view:
            count = os.write(fd, view)
            require(count > 0, "short output write")
            view = view[count:]
        os.fsync(fd)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    require(_identity(os.lstat(path)) == _identity(after)
            and after.st_size == len(raw) and stat.S_IMODE(after.st_mode) == 0o600,
            "new output changed")
    observed, st = _read(path, JSON_CAP)
    require(observed == raw, "new output bytes changed")
    return _record(path, observed, st)


def _parents(records):
    for record in records:
        require(_directory(record["path"]) == record["identity"], "parent changed")


def _inputs(records):
    require(len(records) == len({record["path"] for record in records}), "duplicate input")
    total = 0
    for expected in records:
        raw, st = _read(expected["path"], min(SOURCE_CAP, INPUT_CAP - total))
        total += len(raw)
        require(_record(expected["path"], raw, st) == expected, "input changed: " + expected["path"])


def _metrics(raw, fixture):
    document = _json(raw)
    require(set(document) == {"schema", "fixture", "metric_count", "max_key_bytes",
                            "capacity", "evidence_class", "metrics"}, "metric document fields")
    require(document["schema"] == "laguna.heap-engine-session-metric-schema/v1"
            and document["fixture"] == {"sha256": fixture["sha256"]}
            and document["metric_count"] == 367 and document["max_key_bytes"] == 44
            and document["capacity"] == 512
            and document["evidence_class"] == "static_emission_census_not_C_compile_or_native_execution",
            "metric document identity")
    metrics = document["metrics"]
    require(type(metrics) is dict and len(metrics) == 367, "metric count")
    canonical = json.dumps(metrics, sort_keys=True, separators=(",", ":")).encode("utf-8")
    require(hashlib.sha256(canonical).hexdigest() == METRICS_SHA256, "metric definitions changed")
    for key, item in metrics.items():
        require(re.fullmatch(r"[a-z][a-z0-9_]{0,43}", key) is not None
                and item["type"] in ("bool", "uint", "int", "address", "string"), "metric entry")
        if "availability_key" in item:
            require(item["availability_key"] in metrics
                    and metrics[item["availability_key"]]["type"] == "bool", "metric availability")


def prepare(repository_root, output_parent):
    """Generate current checkout identities only. Do not compile or start children."""
    try:
        return _prepare(repository_root, output_parent)
    except PreparationError:
        raise
    except Exception as error:
        refused = PreparationError(repr(error))
        if getattr(error, "workspace", None):
            refused.workspace = error.workspace
        raise refused from error


def _prepare(repository_root, output_parent):
    require(sys.platform == "darwin", "Darwin CPU-only heap regression; platform refused")
    repo, parent = Path(repository_root), Path(output_parent)
    initial = []
    for path in dict.fromkeys((repo, repo / "tests", parent)):
        initial.append({"path": str(path), "identity": _directory(path)})
    # Fixed closure: no scanning, model discovery, inference, or old session paths.
    relatives = list(DEPENDENCIES) + ["tests/test_engine_session_lifetime.c",
        "tests/test_engine_session_lifetime.py", "tests/engine_session_lifetime_controller.py",
        "tests/run_engine_session_lifetime.py", "tests/engine_session_lifetime_metrics.json",
        "tests/engine_session_lifetime.mk"]
    require(len(DEPENDENCIES) == 24 and len(set(relatives)) == 30, "fixed input closure")
    records, content, total = [], {}, 0
    for relative in relatives:
        path = repo / relative
        raw, st = _read(path, min(SOURCE_CAP, INPUT_CAP - total))
        total += len(raw)
        record = _record(path, raw, st)
        if relative in PINS:
            require(record["sha256"] == PINS[relative], "verified asset changed: " + relative)
        records.append(record)
        content[relative] = raw
    by_relative = dict(zip(relatives, records))
    fixture = by_relative["tests/test_engine_session_lifetime.c"]
    _metrics(content["tests/engine_session_lifetime_metrics.json"], fixture)
    _parents(initial)
    _inputs(records)
    workspace = None
    try:
        workspace = Path(tempfile.mkdtemp(prefix="ds4-heap-lifetime-", dir=str(parent)))
        require(workspace.parent == parent, "workspace parent changed")
        parents = initial + [{"path": str(workspace), "identity": _directory(workspace, 0o700)}]
        leaves = {}
        for name in ("home", "tmp", "config", "cache", "data", "logs"):
            path = workspace / name
            path.mkdir(mode=0o700)
            leaves[name] = path
            parents.append({"path": str(path), "identity": _directory(path, 0o700)})
        _parents(parents)
        environment = {"PATH": "/opt/homebrew/bin:/usr/bin:/bin", "LANG": "C", "LC_ALL": "C",
            "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1", "GIT_CONFIG_NOSYSTEM": "1",
            "__CF_USER_TEXT_ENCODING": "0x1F5:0x0:0x0", "CUDA_VISIBLE_DEVICES": ""}
        for key, name in (("HOME", "home"), ("TMPDIR", "tmp"), ("XDG_CONFIG_HOME", "config"),
                          ("XDG_CACHE_HOME", "cache"), ("XDG_DATA_HOME", "data")):
            environment[key] = str(leaves[name])
        preparation = {"schema": "laguna.heap-engine-session-native-static-preparation/v1",
            "repository_root": str(repo), "platform": "Darwin",
            "source_dependencies": [dict(by_relative[rel], relative_path=rel) for rel in DEPENDENCIES],
            "fixture": dict(fixture, relative_path="tests/test_engine_session_lifetime.c"),
            "make_template": by_relative["tests/engine_session_lifetime.mk"],
            "compile_argv": ["/usr/bin/make", "--no-print-directory", "-j1", "-f", "fixture.mk", "all"],
            "native_cases": list(CASES)}
        prep_record = _new_json(workspace / "preparation.json", preparation)
        runner_inputs = records + [prep_record]
        config = {"schema": "laguna.heap-engine-session-cpu-config/v1", "repository_root": str(repo),
            "parents": parents, "inputs": runner_inputs, "preparation_path": prep_record["path"],
            "metric_schema_path": str(repo / "tests/engine_session_lifetime_metrics.json"),
            "environment": environment, "logs_directory": str(leaves["logs"]),
            "logs_identity": _directory(leaves["logs"], 0o700)}
        config_record = _new_json(workspace / "config.json", config)
        controller = by_relative["tests/engine_session_lifetime_controller.py"]
        runner = by_relative["tests/test_engine_session_lifetime.py"]
        admission = {"schema": "laguna.heap-engine-session-cpu-admission/v1", "state": "admitted",
            "evidence_class": EVIDENCE, "repository_root": str(repo), "cwd": str(repo),
            "artifact_root": str(workspace), "log_path": str(workspace / "run.log"),
            "result_path": str(workspace / "result.json"), "parents": parents,
            "environment": environment, "driver": controller, "runner": runner,
            "config": config_record, "inputs": runner_inputs + [config_record],
            "runner_inputs": runner_inputs, "command": ["python3", runner["path"],
                config_record["path"], config_record["sha256"]], "limits": dict(LIMITS),
            "gpu_execution": False, "model_execution": False, "qualification": False}
        admission_record = _new_json(workspace / "admission.json", admission)
        _parents(parents)
        _inputs(admission["inputs"] + [admission_record])
        require(not os.path.lexists(admission["log_path"])
                and not os.path.lexists(admission["result_path"]), "execution outputs must be fresh")
        return {"workspace": str(workspace), "admission_path": admission_record["path"],
                "admission_sha256": admission_record["sha256"], "environment": environment,
                "controller_path": controller["path"]}
    except BaseException as error:
        # Preserve partial metadata. Never delete a caller directory on refusal.
        if workspace is not None:
            error.workspace = str(workspace)
        raise


def _limits():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CPU, (240, 240))
    resource.setrlimit(resource.RLIMIT_FSIZE, (64 << 20, 64 << 20))
    soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
    ceiling = 4096 if hard == resource.RLIM_INFINITY else min(hard, 4096)
    if soft != resource.RLIM_INFINITY:
        ceiling = min(ceiling, soft)
    resource.setrlimit(resource.RLIMIT_NPROC, (ceiling, ceiling))


def main():
    workspace = None
    try:
        require(len(sys.argv) == 1, "usage: run_engine_session_lifetime.py (set canonical owned TMPDIR)")
        require(sys.platform == "darwin", "Darwin CPU-only heap regression; platform refused")
        _limits()
        own_path = Path(__file__).absolute()
        require(_canonical(own_path), "launcher path must be canonical")
        raw_tmp = os.environ.get("TMPDIR")
        parent = Path(raw_tmp) if raw_tmp else Path(tempfile.gettempdir()).resolve(strict=True)
        result = prepare(own_path.parents[1], parent)
        workspace = result["workspace"]
        raw, _ = _read(result["admission_path"], JSON_CAP)
        require(hashlib.sha256(raw).hexdigest() == result["admission_sha256"], "admission changed before exec")
        admission = _json(raw)
        _parents(admission["parents"])
        _inputs(admission["inputs"])
        print("workspace=" + result["workspace"], flush=True)
        print("log_path=" + admission["log_path"], flush=True)
        print("result_path=" + admission["result_path"], flush=True)
        # Keep the existing bounded wait/reaping and actual-exit verification intact.
        os.execve(sys.executable, [sys.executable, result["controller_path"],
                  result["admission_path"], result["admission_sha256"]], result["environment"])
        raise PreparationError("controller exec unexpectedly returned")
    except BaseException as error:
        preserved = getattr(error, "workspace", None) or workspace
        if preserved:
            print("preserved_workspace=" + preserved, file=sys.stderr, flush=True)
        print("heap lifetime setup refused: " + repr(error), file=sys.stderr, flush=True)
        return 125


if __name__ == "__main__":
    raise SystemExit(main())
