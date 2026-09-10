#!/usr/bin/env python3
"""Actual-core heap engine/session lifetime controls. CPU only; not qualification.

Run only with a root-authenticated configuration and its SHA-256. Native setup,
transport, sensor, rescue and cleanup errors are 125; product assertions are 1.
The resident guard and production API are unchanged by this fixture.
"""
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
import unittest

SOURCE_CAP = 4 << 20
FILE_CAP = 64 << 20
LOG_CAP = 1 << 20
STDOUT_CAP = 32768
CASES = ("empty", "live-one", "live-two", "retained-after-unlock")
_INTERRUPTED = False
_CHILD_HANDOFF = False
_REAPING = False
_PRESERVE_TREE = False
_PROCESS_RECORDS = []
_CONTEXT = None

class InfrastructureFailure(RuntimeError):
    """Unproved setup or observation; never a product regression."""

def require(condition, message):
    if not condition:
        raise InfrastructureFailure(message)

def _interrupt(signum, _frame):
    global _INTERRUPTED
    _INTERRUPTED = True
    if not (_CHILD_HANDOFF or _REAPING):
        raise KeyboardInterrupt("interrupted by signal %s" % signum)

def _path_id(st):
    return (st.st_uid, st.st_dev, st.st_ino, st.st_mode, st.st_nlink,
            st.st_size, st.st_mtime_ns, st.st_ctime_ns)

def _dir_id(path, mode=None):
    path = Path(path)
    require(path.is_absolute() and os.path.normpath(str(path)) == str(path)
            and os.path.realpath(path) == str(path), "noncanonical directory")
    st = os.lstat(path)
    require(stat.S_ISDIR(st.st_mode) and st.st_uid == os.getuid(),
            "directory type/owner: %s" % path)
    actual_mode = stat.S_IMODE(st.st_mode)
    require(mode is None or actual_mode == mode, "directory mode: %s" % path)
    return (st.st_uid, st.st_dev, st.st_ino, actual_mode)

def _read_file(path, cap):
    path = Path(path)
    require(path.is_absolute() and os.path.realpath(path) == str(path)
            and os.path.normpath(str(path)) == str(path), "noncanonical file")
    pre = os.lstat(path)
    require(stat.S_ISREG(pre.st_mode) and pre.st_uid == os.getuid()
            and pre.st_nlink == 1 and 0 <= pre.st_size <= cap,
            "file type/owner/link/size: %s" % path)
    identity = _path_id(pre)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        require(_path_id(os.fstat(fd)) == identity, "pre-open drift")
        chunks, total = [], 0
        while True:
            block = os.read(fd, min(65536, cap + 1 - total))
            if not block:
                break
            chunks.append(block)
            total += len(block)
            require(total <= cap, "read overflow")
        require(_path_id(os.fstat(fd)) == identity, "post-read drift")
    finally:
        os.close(fd)
    require(_path_id(os.lstat(path)) == identity, "file path drift")
    raw = b"".join(chunks)
    require(len(raw) == pre.st_size, "read size drift")
    return raw, pre

def _record(path, raw, st, text=True):
    result = {"path": str(path), "bytes": len(raw),
              "sha256": hashlib.sha256(raw).hexdigest(), "uid": st.st_uid,
              "device": st.st_dev, "inode": st.st_ino,
              "mode": stat.S_IMODE(st.st_mode), "nlink": st.st_nlink}
    if text:
        result["lines"] = len(raw.decode("utf-8").splitlines())
    return result

def _authenticate(expected, cap=SOURCE_CAP):
    raw, st = _read_file(expected["path"], cap)
    actual = _record(expected["path"], raw, st)
    require(all(actual.get(k) == v for k, v in expected.items()
                if k != "relative_path"), "input drift: %s" % expected["path"])
    return raw, actual

def _pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, "duplicate JSON key: %s" % key)
        result[key] = value
    return result

def _json(raw):
    def bad_constant(value):
        raise InfrastructureFailure("nonfinite JSON constant: " + value)
    return json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs,
                      parse_constant=bad_constant)

def _write_new(path, raw, mode=0o600):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
    try:
        opened = os.fstat(fd)
        require(stat.S_ISREG(opened.st_mode) and opened.st_nlink == 1
                and opened.st_uid == os.getuid(), "new output identity")
        data = memoryview(raw)
        while data:
            count = os.write(fd, data)
            require(count > 0, "short output write")
            data = data[count:]
        os.fsync(fd)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    require(_path_id(os.lstat(path)) == _path_id(after)
            and after.st_size == len(raw)
            and stat.S_IMODE(after.st_mode) == mode, "new output changed")
    return _record(path, raw, after, text=False)

def _group_gone(pid):
    # Signal 0 is an existence query, not a later teardown signal.
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False

def _stop_child(process):
    global _REAPING, _PRESERVE_TREE
    _REAPING = True
    try:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            if process.poll() is None:
                try:
                    os.killpg(process.pid, sig)
                except ProcessLookupError:
                    pass  # Only wait/poll establishes direct-child reaping.
            try:
                process.wait(timeout=1)
                break
            except subprocess.TimeoutExpired:
                continue
        if process.poll() is None or not _group_gone(process.pid):
            _PRESERVE_TREE = True
            raise InfrastructureFailure("child or descendants not proved reaped")
    finally:
        _REAPING = False

def _child_limits():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (FILE_CAP, FILE_CAP))
    resource.setrlimit(resource.RLIMIT_CPU, (240, 240))
    # Per-user RLIMIT_NPROC is only a ceiling, not group/cgroup isolation.
    soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
    limit = 4096 if hard == resource.RLIM_INFINITY else min(hard, 4096)
    if soft != resource.RLIM_INFINITY:
        limit = min(limit, soft)
    resource.setrlimit(resource.RLIMIT_NPROC, (limit, limit))

def _diagnostic(raw):
    if len(raw) <= 8192:
        return repr(raw)
    return repr(raw[:4096] + b"\n...[bounded excerpt]...\n" + raw[-4096:])

def _run_child(command, cwd, env, label, timeout, stdout_cap):
    global _CHILD_HANDOFF, _PRESERVE_TREE
    require(not _INTERRUPTED, "interrupted before spawn")
    _CONTEXT.check_logs()
    out_path = _CONTEXT.logs / (label + ".stdout")
    err_path = _CONTEXT.logs / (label + ".stderr")
    out_file = err_file = process = None
    row = {"label": label, "argv": command, "pid": None, "reaped": False,
           "group_gone": False, "returncode": None, "spawn_started": False}
    _PROCESS_RECORDS.append(row)
    try:
        _write_new(out_path, b"")
        _write_new(err_path, b"")
        row["stdout_path"], row["stderr_path"] = str(out_path), str(err_path)
        # No truncation; these are the exact fresh, empty owned spools.
        streams = []
        try:
            for path in (out_path, err_path):
                raw, pre = _read_file(path, 0)
                fd = os.open(path, os.O_WRONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                try:
                    require(_path_id(os.fstat(fd)) == _path_id(pre), "spool drift")
                    streams.append(os.fdopen(fd, "wb"))
                except BaseException:
                    os.close(fd)
                    raise
            out_file, err_file = streams
        except BaseException:
            for stream in streams:
                stream.close()
            raise
        require(not _INTERRUPTED, "interrupted before handoff")
        _CHILD_HANDOFF = True
        row["spawn_started"] = True
        try:
            process = subprocess.Popen(command, cwd=str(cwd), env=env,
                stdin=subprocess.DEVNULL, stdout=out_file, stderr=err_file,
                close_fds=True, start_new_session=True, preexec_fn=_child_limits)
            row["pid"] = process.pid
        finally:
            _CHILD_HANDOFF = False
        require(not _INTERRUPTED, "interrupted during handoff")
        out_file.close(); err_file.close(); out_file = err_file = None
        process.wait(timeout=timeout)
        row.update(returncode=process.returncode, reaped=True,
                   group_gone=_group_gone(process.pid))
        if not row["group_gone"]:
            _PRESERVE_TREE = True
            raise InfrastructureFailure("descendants remain after direct-child exit")
        require(not _INTERRUPTED, "interrupted after reaping")
        _CONTEXT.check_logs()
        stdout, out_st = _read_file(out_path, stdout_cap)
        stderr, err_st = _read_file(err_path, LOG_CAP)
        row["stdout"] = _record(out_path, stdout, out_st, text=False)
        row["stderr"] = _record(err_path, stderr, err_st, text=False)
        print("HEAP_PROCESS " + json.dumps(row, sort_keys=True), flush=True)
        require(process.returncode == 0,
                "%s native/setup nonzero (not RED): %s; stdout=%s stderr=%s" %
                (label, process.returncode, _diagnostic(stdout), _diagnostic(stderr)))
        return stdout, stderr
    except BaseException as error:
        if process is None and row["spawn_started"]:
            _PRESERVE_TREE = True
            row["handoff_unproved"] = True
        if process is not None:
            if process.poll() is None:
                try:
                    _stop_child(process)
                except BaseException as reap_error:
                    _PRESERVE_TREE = True
                    row["reap_error"] = repr(reap_error)
            row.update(returncode=process.returncode,
                       reaped=process.poll() is not None,
                       group_gone=_group_gone(process.pid))
            if not row["reaped"] or not row["group_gone"]:
                _PRESERVE_TREE = True
        diagnostics = {}
        for key, path in (("stdout", out_path), ("stderr", err_path)):
            try:
                raw, _ = _read_file(path, LOG_CAP)
                diagnostics[key] = _diagnostic(raw)
            except BaseException as diagnostic_error:
                diagnostics[key] = repr(diagnostic_error)
        row["error"] = repr(error)
        print("HEAP_PROCESS_ERROR " + json.dumps(row, sort_keys=True), flush=True)
        raise InfrastructureFailure("%s: %r; %s" % (label, error, diagnostics)) from error
    finally:
        if out_file is not None:
            out_file.close()
        if err_file is not None:
            err_file.close()

class RunContext:
    def __init__(self, config_path, digest):
        self.root = None
        self.dirs = {}
        self.ids = {}
        self.cleanup_complete = False
        raw, _ = _read_file(config_path, 131072)
        require(re.fullmatch(r"[0-9a-f]{64}", digest) is not None
                and hashlib.sha256(raw).hexdigest() == digest, "configuration digest")
        self.config_raw = raw
        self.config_path = Path(config_path)
        self.config = _json(raw)
        require(set(self.config) == {"schema", "repository_root", "parents",
                "inputs", "preparation_path", "metric_schema_path", "environment",
                "logs_directory", "logs_identity"}, "configuration fields")
        require(self.config["schema"] == "laguna.heap-engine-session-cpu-config/v1",
                "configuration schema")
        require(sys.platform == "darwin", "this build recipe is Darwin only")
        self.repo = Path(self.config["repository_root"])
        self.logs = Path(self.config["logs_directory"])
        self.check_logs()
        require(dict(os.environ) == self.config["environment"], "environment drift")
        raw_tmp = os.environ.get("TMPDIR", "")
        self.tmp = Path(raw_tmp)
        require(str(self.tmp) == raw_tmp, "raw TMPDIR path normalization")
        self.tmp_id = _dir_id(self.tmp, 0o700)
        self.inputs = {}
        require(type(self.config["inputs"]) is list and
                1 <= len(self.config["inputs"]) <= 64, "input count")
        for record in self.config["inputs"]:
            require(set(record) == {"path", "bytes", "lines", "sha256", "uid",
                    "device", "inode", "mode", "nlink"}, "input fields")
            require(record["path"] not in self.inputs, "duplicate input path")
            self.inputs[record["path"]] = record
        require(str(Path(__file__).absolute()) in self.inputs, "runner not authenticated")
        self.authenticate_all()
        self.prep = _json(self.input_bytes[self.config["preparation_path"]])
        self.schema_document = _json(self.input_bytes[self.config["metric_schema_path"]])
        self.metrics = self.schema_document["metrics"]
        require(self.prep["schema"] == "laguna.heap-engine-session-native-static-preparation/v1"
                and self.prep["repository_root"] == str(self.repo)
                and self.prep["platform"] == "Darwin", "build preparation")
        require(self.schema_document["schema"] == "laguna.heap-engine-session-metric-schema/v1"
                and self.schema_document["metric_count"] == 367
                and len(self.metrics) == 367, "metric schema count")
        require(self.prep["fixture"]["sha256"] ==
                self.schema_document["fixture"]["sha256"], "fixture/schema binding")
        require(self.prep["native_cases"] == list(CASES), "native cases")
        require(self.prep["compile_argv"] == ["/usr/bin/make", "--no-print-directory",
                "-j1", "-f", "fixture.mk", "all"], "compiler command")
        require(len(self.prep["source_dependencies"]) == 24, "dependency closure count")
        for key, metric in self.metrics.items():
            require(re.fullmatch(r"[a-z][a-z0-9_]{0,43}", key) is not None
                    and metric["type"] in ("bool", "uint", "int", "address", "string"),
                    "metric schema entry")
            if "availability_key" in metric:
                require(metric["availability_key"] in self.metrics and
                        self.metrics[metric["availability_key"]]["type"] == "bool",
                        "availability schema")
        require([k for k, v in self.metrics.items() if v["type"] == "string"] == ["case"],
                "string metric schema")
        self.binary_record = None
        self.copy_records = {}

    def check_logs(self):
        require(list(_dir_id(self.logs, 0o700)) == self.config["logs_identity"],
                "persistent logs directory changed")

    def authenticate_all(self):
        require(dict(os.environ) == self.config["environment"], "environment changed")
        for parent in self.config["parents"]:
            require(list(_dir_id(parent["path"])) == parent["identity"], "parent changed")
        raw, _ = _read_file(self.config_path, 131072)
        require(raw == self.config_raw, "configuration changed")
        self.input_bytes = {}
        self.current_records = []
        for path, record in self.inputs.items():
            raw, actual = _authenticate(record)
            self.input_bytes[path] = raw
            self.current_records.append(actual)

    def prepare(self):
        require(_dir_id(self.tmp, 0o700) == self.tmp_id, "TMPDIR changed")
        self.root = Path(tempfile.mkdtemp(prefix="ds4-heap-session-", dir=str(self.tmp)))
        self.root_id = _dir_id(self.root, 0o700)
        for name in ("source", "build", "home", "tmp", "config", "cache", "data"):
            path = self.root / name
            path.mkdir(mode=0o700)
            self.dirs[name] = path
            self.ids[name] = _dir_id(path, 0o700)
        tests = self.dirs["source"] / "tests"
        tests.mkdir(mode=0o700)
        self.dirs["source/tests"] = tests
        self.ids["source/tests"] = _dir_id(tests, 0o700)
        require(_dir_id(self.tmp, 0o700) == self.tmp_id, "TMPDIR changed during setup")
        dependency_paths = set()
        for record in self.prep["source_dependencies"] + [self.prep["fixture"]]:
            path = Path(record["path"])
            rel = path.relative_to(self.repo)
            require(str(rel) == record["relative_path"] and ".." not in rel.parts,
                    "relative source path")
            require(path.parent == self.repo or
                    str(rel) == "tests/test_engine_session_lifetime.c", "source scope")
            require(str(path) not in dependency_paths, "duplicate dependency")
            dependency_paths.add(str(path))
            require(all(self.inputs[str(path)].get(k) == v for k,v in record.items()
                        if k != "relative_path"), "preparation input binding")
            dest = self.dirs["source"] / rel
            _write_new(dest, self.input_bytes[str(path)])
            copied, copied_st = _read_file(dest, SOURCE_CAP)
            require(copied == self.input_bytes[str(path)], "copy mismatch")
            self.copy_records[str(dest)] = _record(dest, copied, copied_st)
        template = self.prep["make_template"]
        require(all(self.inputs[template["path"]].get(k) == v
                    for k,v in template.items()), "Make template binding")
        make_path = self.dirs["build"] / "fixture.mk"
        _write_new(make_path, self.input_bytes[template["path"]])
        copied, copied_st = _read_file(make_path, 16384)
        self.copy_records[str(make_path)] = _record(make_path, copied, copied_st)
        self.child_env = {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C",
            "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1",
            "CUDA_VISIBLE_DEVICES": "", "__CF_USER_TEXT_ENCODING": "0x1F5:0x0:0x0"}
        for key, name in (("HOME", "home"), ("TMPDIR", "tmp"),
                ("XDG_CONFIG_HOME", "config"), ("XDG_CACHE_HOME", "cache"),
                ("XDG_DATA_HOME", "data")):
            self.child_env[key] = str(self.dirs[name])
        _run_child(self.prep["compile_argv"], self.dirs["build"], self.child_env,
                   "compile", 120, LOG_CAP)
        self.binary = self.dirs["build"] / "fixture"
        raw, st = _read_file(self.binary, FILE_CAP)
        require(len(raw) > 0 and stat.S_IMODE(st.st_mode) & 0o111
                and os.access(self.binary, os.X_OK), "binary not executable")
        self.binary_record = _record(self.binary, raw, st, text=False)
        print("HEAP_BINARY " + json.dumps(self.binary_record, sort_keys=True), flush=True)
        self.verify_copies()

    def verify_copies(self):
        require(_dir_id(self.tmp, 0o700) == self.tmp_id, "TMPDIR changed")
        require(_dir_id(self.root, 0o700) == self.root_id, "owned root changed")
        for path, record in self.copy_records.items():
            _authenticate(record)
        if self.binary_record is not None:
            raw, st = _read_file(self.binary, FILE_CAP)
            require(_record(self.binary, raw, st, text=False) == self.binary_record,
                    "binary changed")
        for name, identity in self.ids.items():
            require(_dir_id(self.dirs[name], 0o700) == identity, "fixture directory changed")

    def case(self, name):
        require(name in CASES, "unknown native case")
        self.authenticate_all()
        self.verify_copies()
        stdout, stderr = _run_child([str(self.binary), name], self.dirs["build"],
                                   self.child_env, name, 10, STDOUT_CAP)
        print("HEAP_NATIVE " + json.dumps({"case":name, "stdout_hex":stdout.hex(),
              "stdout_sha256":hashlib.sha256(stdout).hexdigest(),
              "stderr_sha256":hashlib.sha256(stderr).hexdigest()}, sort_keys=True), flush=True)
        self.verify_copies()
        self.authenticate_all()
        metrics = parse_metrics(stdout, self.metrics, name)
        validate_observations(metrics, self.metrics, name)
        return metrics

    def cleanup(self):
        global _PRESERVE_TREE
        if self.root is None:
            self.cleanup_complete = True
            return
        if _PRESERVE_TREE:
            raise InfrastructureFailure("preserving fixture tree: " + str(self.root))
        try:
            require(_dir_id(self.tmp, 0o700) == self.tmp_id, "TMPDIR changed before cleanup")
            require(_dir_id(self.root, 0o700) == self.root_id, "owned root changed")
            for name, path in self.dirs.items():
                require(_dir_id(path, 0o700) == self.ids[name], "owned child changed")
            known = {name for name in self.dirs if "/" not in name}
            require({p.name for p in self.root.iterdir()} == known, "unknown root child")
            require(all(not row["spawn_started"] or (row["reaped"] and row["group_gone"])
                        for row in _PROCESS_RECORDS), "unproved process custody")
            # Every recursive mutation is within this fresh, owned, identity-bound tree.
            # Compiler/runtime-created descendants are not arbitrary caller directories.
            shutil.rmtree(self.root)
            require(not os.path.lexists(self.root), "fixture tree remains")
            require(_dir_id(self.tmp, 0o700) == self.tmp_id, "TMPDIR changed during cleanup")
            self.cleanup_complete = True
        except BaseException:
            _PRESERVE_TREE = True
            raise

def parse_metrics(raw, schema, name):
    require(0 < len(raw) <= STDOUT_CAP and raw.endswith(b"\n"), "native framing")
    text = raw.decode("ascii")
    rows = text.split("\n")[:-1]
    require(len(rows) == 367, "native metric row count")
    result = {}
    patterns = {"bool": r"[01]", "uint": r"(?:0|[1-9][0-9]{0,19})",
                "int": r"(?:0|-?[1-9][0-9]{0,9})",
                "address": r"0x(?:0|[1-9a-f][0-9a-f]{0,15})"}
    for row in rows:
        require(row.count("=") == 1, "native metric delimiter")
        key, value = row.split("=")
        require(key in schema and key not in result, "unknown/duplicate native key: " + key)
        kind = schema[key]["type"]
        if kind == "string":
            require(key == "case" and value == name, "native case identity")
            result[key] = value
            continue
        require(re.fullmatch(patterns[kind], value) is not None, "native value type: " + key)
        number = int(value, 16 if kind == "address" else 10)
        require(-(1 << 31) <= number < (1 << 31) if kind == "int"
                else 0 <= number < (1 << 64), "native integer range: " + key)
        result[key] = number
    require(set(result) == set(schema), "native key set")
    for key, entry in schema.items():
        if "availability_key" in entry and not result[entry["availability_key"]]:
            require(result[key] == 0, "unavailable metric has a value: " + key)
    return result

def _infra_equal(actual, expected, label):
    require(actual == expected, "%s: got %r expected %r" % (label, actual, expected))

def _infra_fields(metrics, expected):
    for key, value in expected.items():
        _infra_equal(metrics[key], value, key)

def _unobserved(metrics, prefix):
    for key in metrics:
        if key.startswith(prefix):
            _infra_equal(metrics[key], 0, "unrecorded " + key)

def _available(metrics, name, value=None):
    _infra_equal(metrics[name + "_available"], 1, name + " availability")
    actual = metrics[name + "_value"]
    if value is not None:
        _infra_equal(actual, value, name)
    return actual

def validate_observations(m, schema, name):
    """Validate the experiment, not the desired product outcome."""
    _infra_fields(m, {key: 1 for key in ("setup_ok", "native_cleanup_complete",
        "native_observation_clean", "setup_profile_inactive", "setup_thread_pool_inactive",
        "setup_instance_fd_clear", "setup_engine_empty", "setup_runtime_tracker_unready",
        "configured_backend_cuda", "configured_cache_bytes_set", "configured_compact_runtime",
        "engine_allocated", "engine_size_limit_ok", "session_size_limit_ok",
        "cumulative_heap_limit_ok", "runtime_mutex_initialized", "exact_mutex_initialized",
        "pre_rescue_saved", "rescue_proof_available", "rescue_proof_ok",
        "rescue_final_residual_zero", "rescue_source_effects_unchanged")})
    _infra_fields(m, {key: 0 for key in ("native_return_code", "void_engine_close_status_available",
        "void_engine_close_status", "engine_live_final", "runtime_mutex_live_final",
        "exact_mutex_live_final", "unknown_free_calls", "unknown_mutex_calls",
        "unexpected_allocations", "order_violations", "sticky_errors", "real_lock_failures",
        "real_unlock_failures", "real_destroy_failures", "fault_armed_final")})
    _infra_fields(m, {"configured_context_tokens":32768, "configured_model_fd":-1,
                      "configured_mtp_model_fd":-1,
                      "configured_session_limit":2 if name == "live-two" else 1})
    engine_address = _available(m, "engine_address")
    engine_bytes = _available(m, "engine_bytes")
    require(engine_address > 0 and 0 < engine_bytes <= 4 << 20, "engine allocation witness")
    mutex_addresses = []
    for mutex in ("runtime", "exact"):
        address = _available(m, mutex + "_mutex_address")
        require(engine_address <= address < engine_address + engine_bytes, "embedded mutex address")
        mutex_addresses.append(address)
        _available(m, mutex + "_mutex_init_result", 0)
        _available(m, mutex + "_mutex_destroy_result", 0)
        _infra_equal(m["physical_"+mutex+"_locks"], m["physical_"+mutex+"_unlocks"],
                     mutex + " real lock/unlock balance")
    require(len(set(mutex_addresses)) == 2, "distinct initialized mutex witnesses")
    allocated = 0
    intervals = [(engine_address, engine_address + engine_bytes)]
    session_bytes = 0
    for i in range(2):
        prefix = "session%d_" % i
        acquired = m[prefix + "address_available"]
        allocated += acquired
        _infra_equal(m[prefix + "bytes_available"], acquired, "session size availability")
        _infra_equal(m[prefix + "create_rc_available"], m[prefix + "create_attempted"],
                     "constructor status availability")
        _infra_equal(m[prefix + "live_final"], 0, "session residual")
        if acquired:
            address, size = m[prefix + "address_value"], m[prefix + "bytes_value"]
            require(address > 0 and 0 < size <= 4 << 20, "session allocation witness")
            intervals.append((address, address + size))
            session_bytes += size
        else:
            _infra_fields(m, {prefix + key:0 for key in ("address_value", "bytes_value",
                "source_free_calls", "rescue_free_calls", "pre_rescue_live")})
        if m[prefix + "create_success"]:
            _infra_fields(m, {prefix+key:1 for key in ("create_attempted", "address_available",
                "engine_borrow_available", "reserved_available", "active_after_create_available",
                "test_no_alloc", "backend_payload_empty")})
            _infra_fields(m, {prefix+"create_rc_value":0,
                             prefix+"engine_borrow_value":engine_address})
        if not m[prefix + "create_attempted"]:
            _infra_fields(m, {prefix + key:0 for key in ("create_success", "engine_borrow_available",
                "test_no_alloc", "backend_payload_empty", "reserved_available",
                "active_after_create_available")})
        source, rescue = m[prefix+"source_free_calls"], m[prefix+"rescue_free_calls"]
        _infra_equal(source + rescue, acquired, "session physical free conservation")
        _infra_equal(source, m[prefix+"pre_rescue_source_free_calls"], "session source history")
        delta = _available(m, prefix.rstrip("_")+"_rescue_delta")
        expected = m[prefix+"pre_rescue_live"]
        _infra_equal(m[prefix+"rescue_expected"], expected, "session rescue requirement")
        _infra_equal(delta, expected, "measured session rescue delta")
        _infra_equal(rescue - m[prefix+"pre_rescue_rescue_free_calls"], delta, "session rescue difference")
        _infra_equal(m[prefix+"pre_rescue_rescue_free_calls"], 0, "no early session rescue")
    for i, (low, high) in enumerate(intervals):
        for other_low, other_high in intervals[i+1:]:
            require(high <= other_low or other_high <= low, "independently live allocation overlap")
    _infra_fields(m, {"session_alloc_count":allocated, "core_calloc_calls":allocated,
                      "core_calloc_bytes":session_bytes, "session_bytes_total":session_bytes,
                      "physical_free_calls":1+allocated,
                      "pre_rescue_sessions_live":sum(m["session%d_pre_rescue_live"%i] for i in range(2))})
    require(engine_bytes + session_bytes <= 12 << 20, "cumulative bounded heap")
    for owner, effect, live in (("engine", "free", "engine_live"),
            ("runtime", "destroy", "runtime_mutex_live"), ("exact", "destroy", "exact_mutex_live")):
        source_key = owner + "_source_" + effect + "_calls"
        rescue_key = owner + "_rescue_" + effect + "_calls"
        before_source = m["pre_rescue_" + source_key]
        before_rescue = m["pre_rescue_" + rescue_key]
        expected = m["pre_rescue_" + live]
        _infra_equal(m[source_key] + m[rescue_key], 1, owner + " physical effect conservation")
        _infra_equal(m[source_key], before_source, owner + " source effects preserved")
        _infra_equal(before_rescue, 0, owner + " no early rescue")
        _infra_equal(m["rescue_"+owner+"_expected"], expected, owner + " rescue expected")
        _infra_equal(m["rescue_"+owner+"_delta"], expected, owner + " measured rescue delta")
        _infra_equal(m[rescue_key] - before_rescue, expected, owner + " rescue difference")
    releases = ("release0", "release1", "first_release", "retry_release")
    for prefix in releases:
        if not m[prefix+"_attempted"]:
            _unobserved(m, prefix + "_")
            continue
        _infra_fields(m, {prefix+"_"+key:1 for key in ("call_available", "rc_available",
            "before_engine_live", "before_session_live", "before_active_available",
            "before_reserved_available", "after_engine_live_available", "after_session_live_available")})
        after_engine, after_session = m[prefix+"_after_engine_live"], m[prefix+"_after_session_live"]
        _infra_fields(m, {prefix+"_after_active_available":after_engine,
                          prefix+"_after_reserved_available":after_session,
                          prefix+"_after_owner_match_available":after_session})
        if not after_session:
            _infra_equal(m[prefix+"_after_owner_match"], 0, "dead slot value must not be read")
        rc = m[prefix+"_rc_value"]
        mismatch = rc not in (0,1) or (rc == 1 and after_session) or (rc == 0 and not after_session)
        _infra_equal(m[prefix+"_semantic_mismatch"], int(mismatch), "status/consumption sensor")
    _infra_equal(m["checked_release_calls"], sum(m[p+"_attempted"] for p in releases), "release call census")
    _infra_equal(m["semantic_product_mismatch_count"], sum(m[p+"_semantic_mismatch"] for p in releases),
                 "semantic mismatch census")
    probes = markers = 0
    for prefix in ("close1", "close2", "close3"):
        if not m[prefix+"_recorded"]:
            _unobserved(m, prefix + "_")
            continue
        _infra_fields(m, {prefix+"_"+key:1 for key in ("before_engine_live_available",
            "before_sessions_live_available", "before_reserved_available", "after_engine_live_available",
            "after_sessions_live_available", "after_reserved_available")})
        called, before_live, after_live = m[prefix+"_called"], m[prefix+"_before_engine_live"], m[prefix+"_after_engine_live"]
        before_sessions, after_sessions = m[prefix+"_before_sessions_live_value"], m[prefix+"_after_sessions_live_value"]
        require(0 <= before_sessions <= 2 and 0 <= after_sessions <= 2, "session witness count")
        require(m[prefix+"_before_reserved_value"] <= before_sessions and
                m[prefix+"_after_reserved_value"] <= after_sessions, "reservation/live witness relation")
        _infra_fields(m, {prefix+"_call_available":before_live, prefix+"_called":before_live,
            prefix+"_skipped":1-before_live, prefix+"_before_active_available":called,
            prefix+"_after_active_available":int(bool(called and after_live)),
            prefix+"_after_engine_storage_available":after_live})
        probe = int(bool(called and before_sessions))
        marker = int(bool(probe and not after_live))
        _infra_fields(m, {prefix+"_source_probe":probe, prefix+"_source_free_marker":marker})
        probes += probe
        markers += marker
    _infra_fields(m, {"product_source_probe_count":probes,
                      "product_source_free_marker_count":markers,
                      "product_premature_engine_free":int(markers != 0)})
    require(markers <= m["engine_source_free_calls"], "free marker without physical effect")
    fault = int(name == "retained-after-unlock")
    _infra_fields(m, {key:fault for key in ("fault_arm_count", "fault_selection_count",
        "fault_synthetic_refusal_count", "fault_selected_real_successes",
        "fault_selected_real_result_available", "fault_selected_return_result_available")})
    if not fault:
        _unobserved(m, "first_snapshot_")
        _unobserved(m, "post_close_")
        _unobserved(m, "post_retry_")
        _unobserved(m, "first_release_")
        _unobserved(m, "retry_release_")
        return
    _available(m, "fault_selected_real_result", 0)
    _available(m, "fault_selected_return_result", errno.EPERM)
    _infra_fields(m, {key:1 for key in ("first_snapshot_saved", "first_snapshot_copy_available",
        "first_snapshot_fault_arm_count", "first_snapshot_fault_selection_count",
        "first_snapshot_fault_synthetic_refusal_count", "first_snapshot_fault_real_successes")})
    _available(m, "first_snapshot_fault_real_result", 0)
    _available(m, "first_snapshot_fault_return_result", errno.EPERM)
    _available(m, "first_snapshot_release_rc", _available(m, "first_release_rc"))
    first_engine, first_session = m["first_release_after_engine_live"], m["first_release_after_session_live"]
    _infra_fields(m, {"first_snapshot_engine_live":first_engine,
                      "first_snapshot_session_live":first_session,
                      "first_snapshot_active_available":first_engine,
                      "first_snapshot_reserved_available":first_session,
                      "first_snapshot_source_engine_free_calls":1-first_engine,
                      "first_snapshot_source_session_free_calls":1-first_session})
    if first_engine:
        _available(m, "first_snapshot_active", m["first_release_after_active_value"])
    if first_session:
        _infra_equal(m["first_snapshot_reserved"], m["first_release_after_reserved_value"], "first reservation snapshot")
    require(0 < m["first_snapshot_copy_bytes"] <= 4096, "independent snapshot copy size")
    _available(m, "first_snapshot_equal_before_rescue", 1)
    _available(m, "first_snapshot_equal_after_rescue", 1)
    _infra_equal(m["first_snapshot_preserved"], 1, "first snapshot equality after rescue")
    first_locks = m["first_snapshot_physical_exact_locks"]
    _infra_equal(first_locks, m["first_snapshot_physical_exact_unlocks"], "first real mutex balance")
    require(first_locks > 0 and first_locks <= m["physical_exact_locks"], "first/final physical lock history")
    expected_close = int(bool(first_engine and first_session and m["first_release_rc_value"] == 0
                             and not m["first_release_semantic_mismatch"]))
    _infra_equal(m["close1_recorded"], expected_close, "liveness-gated first close")
    _infra_equal(m["post_close_saved"], expected_close, "post-close snapshot availability")
    if not expected_close:
        _unobserved(m, "post_close_")
    else:
        _infra_fields(m, {"post_close_engine_live_available":1,
            "post_close_engine_live":m["close1_after_engine_live"],
            "post_close_engine_storage_available":m["close1_after_engine_storage_available"],
            "post_close_active_available":m["close1_after_active_available"],
            "post_close_active_value":m["close1_after_active_value"],
            "post_close_runtime_mutex_live":m["close1_after_runtime_mutex_live"],
            "post_close_exact_mutex_live":m["close1_after_exact_mutex_live"]})
        _infra_equal(m["post_close_reserved_available"], m["post_close_session_live"], "post-close reserved availability")
        _infra_equal(m["post_close_session_live"], m["close1_after_sessions_live_value"], "one-session physical witness")
        if m["post_close_session_live"]:
            _infra_equal(m["post_close_reserved"], m["close1_after_reserved_value"], "post-close reservation witness")
    _infra_equal(m["post_retry_saved"], 1, "post-retry recorded witness")
    if m["retry_release_attempted"]:
        engine, session = m["retry_release_after_engine_live"], m["retry_release_after_session_live"]
        active, reserved = m["retry_release_after_active_value"], m["retry_release_after_reserved_value"]
    elif expected_close:
        engine, session = m["post_close_engine_live"], m["post_close_session_live"]
        active, reserved = m["post_close_active_value"], m["post_close_reserved"]
    else:
        engine, session = first_engine, first_session
        active, reserved = m["first_snapshot_active_value"], m["first_snapshot_reserved"]
    _infra_fields(m, {"post_retry_engine_live_available":1, "post_retry_engine_live":engine,
        "post_retry_engine_storage_available":engine, "post_retry_session_live":session,
        "post_retry_active_available":engine, "post_retry_reserved_available":session})
    if engine:
        _infra_equal(m["post_retry_active_value"], active, "post-retry active witness")
    if session:
        _infra_equal(m["post_retry_reserved"], reserved, "post-retry reservation witness")
    if not (expected_close and m["post_close_engine_live"] and m["post_close_session_live"]):
        _unobserved(m, "retry_release_")
    if not (m["retry_release_rc_available"] and m["retry_release_rc_value"] == 1
            and not m["retry_release_after_session_live"] and m["retry_release_after_engine_live"]):
        _unobserved(m, "close2_")
    _unobserved(m, "close3_")
    _unobserved(m, "release0_")
    _unobserved(m, "release1_")


def _parser_controls(schema):
    """Synthetic parser checks only. These are not native product observations."""
    values = {k: ("empty" if v["type"] == "string" else
                  "0x0" if v["type"] == "address" else "0") for k,v in schema.items()}
    def encode(mapping):
        return "".join(k+"="+v+"\n" for k,v in mapping.items()).encode("ascii")
    baseline = encode(values)
    parsed = parse_metrics(baseline, schema, "empty")
    require(len(parsed) == 367 and parsed["case"] == "empty", "parser healthy control")
    rows = baseline.splitlines(keepends=True)
    mutations = [b"".join(rows[:-1]), b"".join(rows[:-1]+[rows[0]]),
                 b"".join(rows[:-1]+[b"unknown_key=0\n"]), baseline[:-1],
                 b"X" * (STDOUT_CAP+1)]
    for key, value in (("setup_ok", "2"), ("engine_bytes_value", "01"),
            ("engine_bytes_value", "18446744073709551616"),
            ("native_return_code", "2147483648"), ("native_return_code", "-0"),
            ("engine_address_value", "0x10000000000000000"),
            ("engine_address_value", "0xA"), ("engine_bytes_value", "1"),
            ("case", "live-one")):
        modified = dict(values)
        modified[key] = value
        mutations.append(encode(modified))
    rejected = 0
    for raw in mutations:
        try:
            parse_metrics(raw, schema, "empty")
        except (InfrastructureFailure, UnicodeDecodeError):
            rejected += 1
        else:
            raise InfrastructureFailure("parser accepted malformed control")
    print("HEAP_PARSER_CONTROLS " + json.dumps({"accepted":1, "rejected":rejected,
          "evidence_class":"synthetic_parser_only"}, sort_keys=True), flush=True)

class EngineSessionLifetimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.controls = set()
        _parser_controls(_CONTEXT.metrics)
        _CONTEXT.prepare()

    def fields(self, m, expected):
        for key, value in expected.items():
            self.assertEqual(m[key], value, key)

    def created(self, m, count):
        self.fields(m, {"session_alloc_count":count, "core_calloc_calls":count})
        for i in range(2):
            prefix = "session%d_" % i
            if i < count:
                self.fields(m, {prefix+key:1 for key in ("create_attempted", "create_success",
                    "create_rc_available", "address_available", "bytes_available",
                    "engine_borrow_available", "reserved_available", "test_no_alloc",
                    "backend_payload_empty", "active_after_create_available")})
                self.fields(m, {prefix+"create_rc_value":0, prefix+"reserved_value":1,
                    prefix+"engine_borrow_value":m["engine_address_value"],
                    prefix+"active_after_create_value":i+1})
            else:
                self.fields(m, {prefix+key:0 for key in ("create_attempted", "create_success",
                    "address_available", "bytes_available", "engine_borrow_available",
                    "reserved_available", "active_after_create_available")})

    def close_state(self, m, prefix, sessions, active, reserved, retained):
        expected = {prefix+"_"+key:1 for key in ("recorded", "called", "call_available",
            "before_engine_live_available", "before_engine_live", "before_sessions_live_available",
            "before_active_available", "before_reserved_available", "before_runtime_mutex_live",
            "before_exact_mutex_live", "after_engine_live_available", "after_sessions_live_available",
            "after_reserved_available")}
        expected.update({prefix+"_skipped":0, prefix+"_before_sessions_live_value":sessions,
            prefix+"_before_active_value":active, prefix+"_before_reserved_value":reserved,
            prefix+"_source_probe":int(sessions != 0), prefix+"_source_free_marker":0,
            prefix+"_after_engine_storage_available":int(retained),
            prefix+"_after_engine_live":int(retained), prefix+"_after_active_available":int(retained),
            prefix+"_after_active_value":active if retained else 0,
            prefix+"_after_sessions_live_value":sessions, prefix+"_after_reserved_value":reserved,
            prefix+"_after_runtime_mutex_live":int(retained), prefix+"_after_exact_mutex_live":int(retained)})
        self.fields(m, expected)

    def release_state(self, m, prefix, before, after, reserved_before, consumed):
        expected = {prefix+"_"+key:1 for key in ("call_available", "attempted", "rc_available",
            "before_engine_live", "before_session_live", "before_active_available",
            "before_reserved_available", "after_engine_live_available", "after_engine_live",
            "after_session_live_available", "after_active_available")}
        expected.update({prefix+"_rc_value":int(consumed), prefix+"_before_active_value":before,
            prefix+"_before_reserved_value":reserved_before, prefix+"_after_active_value":after,
            prefix+"_after_session_live":int(not consumed),
            prefix+"_after_reserved_available":int(not consumed), prefix+"_after_reserved_value":0,
            prefix+"_after_owner_match_available":int(not consumed),
            prefix+"_after_owner_match":int(not consumed), prefix+"_semantic_mismatch":0})
        self.fields(m, expected)

    def source_consumed_all(self, m, sessions, releases, probes):
        self.fields(m, {"engine_source_free_calls":1, "engine_rescue_free_calls":0,
            "runtime_source_destroy_calls":1, "exact_source_destroy_calls":1,
            "runtime_rescue_destroy_calls":0, "exact_rescue_destroy_calls":0,
            "checked_release_calls":releases, "physical_free_calls":sessions+1,
            "product_premature_engine_free":0, "product_source_probe_count":probes,
            "product_source_free_marker_count":0, "semantic_product_mismatch_count":0,
            "pre_rescue_engine_live":0, "pre_rescue_sessions_live":0,
            "pre_rescue_runtime_mutex_live":0, "pre_rescue_exact_mutex_live":0})
        for i in range(2):
            self.fields(m, {"session%d_source_free_calls"%i:int(i<sessions),
                "session%d_rescue_free_calls"%i:0, "session%d_pre_rescue_live"%i:0})

    def test_00_empty_engine_is_consumed_once(self):
        m = _CONTEXT.case("empty")
        self.created(m, 0)
        self.close_state(m, "close1", 0, 0, 0, False)
        self.source_consumed_all(m, 0, 0, 0)
        _unobserved(m, "close2_")
        _unobserved(m, "close3_")
        self.controls.add("empty")

    def test_01_live_session_retains_engine_until_consumed(self):
        require("empty" in self.controls, "empty-engine control not established")
        m = _CONTEXT.case("live-one")
        self.created(m, 1)
        self.close_state(m, "close1", 1, 1, 1, True)
        self.release_state(m, "release0", 1, 0, 1, True)
        self.close_state(m, "close2", 0, 0, 0, False)
        self.source_consumed_all(m, 1, 1, 1)
        _unobserved(m, "release1_")
        _unobserved(m, "close3_")
        self.controls.add("live-one")

    def test_02_both_sessions_must_be_consumed_before_engine(self):
        require({"empty", "live-one"} <= self.controls, "prior healthy controls not established")
        m = _CONTEXT.case("live-two")
        self.created(m, 2)
        self.close_state(m, "close1", 2, 2, 2, True)
        self.release_state(m, "release0", 2, 1, 1, True)
        self.close_state(m, "close2", 1, 1, 1, True)
        self.release_state(m, "release1", 1, 0, 1, True)
        self.close_state(m, "close3", 0, 0, 0, False)
        self.source_consumed_all(m, 2, 2, 2)
        self.controls.add("live-two")

    def test_03_zero_reservations_do_not_end_retained_session_custody(self):
        require(set(CASES[:3]) <= self.controls, "healthy controls required before selected fault")
        m = _CONTEXT.case("retained-after-unlock")
        self.created(m, 1)
        self.release_state(m, "first_release", 1, 0, 1, False)
        self.fields(m, {"first_snapshot_release_rc_available":1, "first_snapshot_release_rc_value":0,
            "first_snapshot_engine_live":1, "first_snapshot_session_live":1,
            "first_snapshot_active_available":1, "first_snapshot_active_value":0,
            "first_snapshot_reserved_available":1, "first_snapshot_reserved":0,
            "first_snapshot_runtime_mutex_live":1, "first_snapshot_exact_mutex_live":1,
            "first_snapshot_source_session_free_calls":0, "first_snapshot_source_engine_free_calls":0})
        self.assertEqual(m["product_premature_engine_free"], 0,
            "engine was physically freed while a retained session remained live after the real unlock")
        self.close_state(m, "close1", 1, 0, 0, True)
        self.fields(m, {"post_close_saved":1, "post_close_engine_live_available":1,
            "post_close_engine_storage_available":1, "post_close_engine_live":1,
            "post_close_session_live":1, "post_close_active_available":1, "post_close_active_value":0,
            "post_close_reserved_available":1, "post_close_reserved":0,
            "post_close_runtime_mutex_live":1, "post_close_exact_mutex_live":1})
        self.release_state(m, "retry_release", 0, 0, 0, True)
        self.fields(m, {"post_retry_saved":1, "post_retry_engine_live_available":1,
            "post_retry_engine_storage_available":1, "post_retry_engine_live":1,
            "post_retry_session_live":0, "post_retry_active_available":1, "post_retry_active_value":0,
            "post_retry_reserved_available":0, "post_retry_reserved":0})
        self.close_state(m, "close2", 0, 0, 0, False)
        self.source_consumed_all(m, 1, 2, 1)

class InfrastructureResult(unittest.TextTestResult):
    def addError(self, test, err):
        global _PRESERVE_TREE
        _PRESERVE_TREE = True
        super().addError(test, err)

class InfrastructureRunner(unittest.TextTestRunner):
    resultclass = InfrastructureResult

TEST_NAMES = (
    "test_00_empty_engine_is_consumed_once",
    "test_01_live_session_retains_engine_until_consumed",
    "test_02_both_sessions_must_be_consumed_before_engine",
    "test_03_zero_reservations_do_not_end_retained_session_custody",
)

def main():
    global _CONTEXT, _PRESERVE_TREE
    result = None
    errors = []
    exit_code = 125
    handlers = {}
    summary = {"schema":"laguna.heap-engine-session-cpu-run/v1",
        "evidence_class":"actual_core_DS4_NO_GPU_heap_lifetime_synthetic_pthread_fault",
        "qualified":False, "tests_run":0, "failures":0, "errors":0,
        "skipped":0, "exit_code":125, "cleanup_complete":False}
    try:
        require(len(sys.argv) == 3, "usage: test_engine_session_lifetime.py CONFIG SHA256")
        for sig in (signal.SIGINT, signal.SIGTERM):
            handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, _interrupt)
        _CONTEXT = RunContext(sys.argv[1], sys.argv[2])
        suite = unittest.TestSuite(EngineSessionLifetimeTest(name) for name in TEST_NAMES)
        result = InfrastructureRunner(verbosity=2).run(suite)
        exit_code = 125 if result.errors else (1 if result.failures else 0)
        require(result.testsRun == 4 and not result.skipped, "incomplete selected test suite")
        _CONTEXT.authenticate_all()
        _CONTEXT.verify_copies()
        require(not _INTERRUPTED, "interrupted run")
    except BaseException as error:
        errors.append(repr(error))
        exit_code = 125
        _PRESERVE_TREE = True
    finally:
        if _CONTEXT is not None:
            try:
                _CONTEXT.cleanup()
            except BaseException as cleanup_error:
                errors.append("cleanup: " + repr(cleanup_error))
                exit_code = 125
            summary["cleanup_complete"] = _CONTEXT.cleanup_complete
            summary["input_records"] = getattr(_CONTEXT, "current_records", [])
            summary["fixture_tree"] = str(_CONTEXT.root) if _CONTEXT.root is not None else None
        if result is not None:
            summary.update(tests_run=result.testsRun, failures=len(result.failures),
                errors=len(result.errors), skipped=len(result.skipped))
        summary.update(exit_code=exit_code, runner_errors=errors,
            interrupted=_INTERRUPTED, preserve_tree=_PRESERVE_TREE,
            processes=_PROCESS_RECORDS)
        if _INTERRUPTED:
            exit_code = summary["exit_code"] = 125
        print("HEAP_RESULT " + json.dumps(summary, sort_keys=True), flush=True)
        for sig, old_handler in handlers.items():
            signal.signal(sig, old_handler)
    return exit_code

if __name__ == "__main__":
    raise SystemExit(main())
