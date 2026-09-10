#!/usr/bin/env python3
"""Bounded CPU-only heap lifetime runner controller. Not a qualification gate."""
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import signal
import stat
import subprocess
import sys
import time

SOURCE_CAP = 4 << 20
FILE_CAP = 64 << 20
LOG_CAP = 1 << 20
_INTERRUPTED = None
_HANDOFF = False
_STOPPING = False
class InfrastructureFailure(RuntimeError):
    """Unproved setup or observation; never a product regression."""

def require(condition, message):
    if not condition:
        raise InfrastructureFailure(message)

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

def _diagnostic(raw):
    if len(raw) <= 8192:
        return repr(raw)
    return repr(raw[:4096] + b"\n...[bounded excerpt]...\n" + raw[-4096:])

def _interrupt(signum, _frame):
    global _INTERRUPTED
    if _INTERRUPTED is None:
        _INTERRUPTED = signum
    if not (_HANDOFF or _STOPPING):
        raise KeyboardInterrupt("controller interrupted by signal %s" % signum)

def _stop(process):
    global _STOPPING
    was_stopping = _STOPPING
    _STOPPING = True
    try:
        for sig, grace in ((signal.SIGTERM,8), (signal.SIGKILL,2)):
            if process.poll() is None:
                try:
                    os.killpg(process.pid, sig)
                except ProcessLookupError:
                    pass
            try:
                process.wait(timeout=grace)
                break
            except subprocess.TimeoutExpired:
                continue
        require(process.poll() is not None and _group_gone(process.pid),
                "runner process/group reaping unproved")
    finally:
        _STOPPING = was_stopping

def _parents(records):
    for record in records:
        require(list(_dir_id(record["path"])) == record["identity"], "parent identity changed")

def _inputs(records):
    require(1 <= len(records) <= 64, "input count")
    actual = []
    seen = set()
    for expected in records:
        require(expected["path"] not in seen, "duplicate input path")
        seen.add(expected["path"])
        _, record = _authenticate(expected)
        actual.append(record)
    return actual

def _validate_runner(raw, returncode, admitted):
    text = raw.decode("utf-8")
    lines = [line[len("HEAP_RESULT "):] for line in text.splitlines()
             if line.startswith("HEAP_RESULT ")]
    require(len(lines) == 1, "exactly one terminal runner record required")
    result = _json(lines[0].encode("utf-8"))
    require(result["schema"] == "laguna.heap-engine-session-cpu-run/v1"
            and result["evidence_class"] ==
                "actual_core_DS4_NO_GPU_heap_lifetime_synthetic_pthread_fault"
            and result["qualified"] is False, "runner evidence class")
    require(type(result["exit_code"]) is int and result["exit_code"] == returncode,
            "terminal record/process exit mismatch")
    if returncode not in (0,1):
        return result
    require(result["tests_run"] == 4 and result["errors"] == 0 and result["skipped"] == 0
            and not result["runner_errors"] and result["interrupted"] is False
            and result["preserve_tree"] is False and result["cleanup_complete"] is True,
            "incomplete/error/cleanup runner result")
    require(type(result["failures"]) is int and
            ((returncode == 0 and result["failures"] == 0) or
             (returncode == 1 and result["failures"] > 0)), "assertion result classification")
    require(result["input_records"] == admitted["runner_inputs"], "runner input record union")
    rows = result["processes"]
    require([row["label"] for row in rows] == ["compile", "empty", "live-one", "live-two",
                                                "retained-after-unlock"], "process sequence")
    require(all(row["spawn_started"] is True and row["reaped"] is True
                and row["group_gone"] is True and row["returncode"] == 0
                and not row.get("error") for row in rows), "native process custody/status")
    return result

def main():
    global _HANDOFF, _STOPPING
    process = stream = None
    owns_log = False
    admission = None
    raw_admission = b""
    result_path = log_path = None
    handlers = {}
    started = time.monotonic()
    record = {"schema":"laguna.heap-engine-session-cpu-execution/v1",
        "state":"preflight", "exit_code":125, "child_returncode":None,
        "child_reaped":False, "child_group_gone":False, "spawn_started":False,
        "gpu_execution":False, "model_execution":False, "qualification":False,
        "filesystem_device_isolation":False, "errors":[]}
    try:
        require(len(sys.argv) == 3 and re.fullmatch(r"[0-9a-f]{64}",sys.argv[2]) is not None,
                "usage: controller ADMISSION SHA256")
        raw_admission, _ = _read_file(Path(sys.argv[1]), 131072)
        require(hashlib.sha256(raw_admission).hexdigest() == sys.argv[2], "admission digest")
        admission = _json(raw_admission)
        require(admission["schema"] == "laguna.heap-engine-session-cpu-admission/v1"
                and admission["state"] == "admitted"
                and admission["evidence_class"] ==
                    "actual_core_DS4_NO_GPU_heap_lifetime_synthetic_pthread_fault",
                "CPU-only execution not admitted")
        require(admission["environment"] == dict(os.environ), "execution environment differs")
        _parents(admission["parents"])
        require(str(Path(__file__).absolute()) == admission["driver"]["path"], "driver path authority")
        _authenticate(admission["driver"])
        _authenticate(admission["config"])
        require(admission["command"] == ["python3", admission["runner"]["path"],
                admission["config"]["path"], admission["config"]["sha256"]], "command authority")
        require(admission["limits"] == {"wall_seconds":240, "cpu_seconds":240,
                "compiler_seconds":120, "native_case_seconds":10, "core_bytes":0,
                "file_size_bytes":FILE_CAP, "log_bytes":LOG_CAP, "native_stdout_bytes":32768,
                "term_grace_seconds":8, "kill_grace_seconds":2,
                "process_ceiling":4096}, "limit authority")
        require(admission["cwd"] == admission["repository_root"], "cwd authority")
        log_path, result_path = Path(admission["log_path"]), Path(admission["result_path"])
        require(log_path.parent == result_path.parent == Path(admission["artifact_root"])
                and log_path != result_path and not os.path.lexists(log_path)
                and not os.path.lexists(result_path), "fresh output authority")
        record["source_inputs_before"] = _inputs(admission["inputs"])
        record.update(admission_sha256=sys.argv[2], command=admission["command"],
                      cwd=admission["cwd"], limits=admission["limits"])
        resource.setrlimit(resource.RLIMIT_CORE,(0,0))
        resource.setrlimit(resource.RLIMIT_CPU,(240,240))
        resource.setrlimit(resource.RLIMIT_FSIZE,(FILE_CAP,FILE_CAP))
        soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
        count_limit = 4096 if hard == resource.RLIM_INFINITY else min(hard,4096)
        if soft != resource.RLIM_INFINITY:
            count_limit = min(soft,count_limit)
        resource.setrlimit(resource.RLIMIT_NPROC,(count_limit,count_limit))
        record["effective_limits"] = {key:list(resource.getrlimit(value)) for key,value in
            (("cpu",resource.RLIMIT_CPU),("file",resource.RLIMIT_FSIZE),
             ("core",resource.RLIMIT_CORE),("processes",resource.RLIMIT_NPROC))}
        for sig in (signal.SIGINT,signal.SIGTERM):
            handlers[sig] = signal.getsignal(sig)
            signal.signal(sig,_interrupt)
        _HANDOFF = True
        try:
            _write_new(log_path,b"")
            owns_log = True
            _, pre = _read_file(log_path,0)
            fd = os.open(log_path,os.O_WRONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            try:
                require(_path_id(os.fstat(fd)) == _path_id(pre), "outer spool identity changed")
                stream = os.fdopen(fd,"wb")
            except BaseException:
                os.close(fd)
                raise
            require(_INTERRUPTED is None, "interrupted before runner spawn")
            record["spawn_started"] = True
            process = subprocess.Popen(admission["command"],cwd=admission["cwd"],
                env=admission["environment"],stdin=subprocess.DEVNULL,
                stdout=stream,stderr=subprocess.STDOUT,close_fds=True,start_new_session=True)
            record["pid"] = process.pid
        finally:
            _HANDOFF = False
        require(_INTERRUPTED is None, "interrupted during runner handoff")
        stream.close(); stream = None
        process.wait(timeout=240)
        record.update(child_returncode=process.returncode,child_reaped=True,
                      child_group_gone=_group_gone(process.pid))
        require(record["child_group_gone"], "runner descendants remain")
        raw_log, log_st = _read_file(log_path,LOG_CAP)
        record["log"] = _record(log_path,raw_log,log_st,text=False)
        record["runner_result"] = _validate_runner(raw_log,process.returncode,admission)
        _parents(admission["parents"])
        require(admission["environment"] == dict(os.environ), "post-run environment drift")
        record["source_inputs_after"] = _inputs(admission["inputs"])
        current_admission, _ = _read_file(Path(sys.argv[1]),131072)
        require(current_admission == raw_admission, "admission changed")
        require(_INTERRUPTED is None, "interrupted after runner exit")
        record.update(state="command_completed",exit_code=process.returncode if process.returncode in (0,1) else 125)
    except BaseException as error:
        record.update(state="infrastructure_refused",exit_code=125)
        record["errors"].append(repr(error))
    finally:
        _STOPPING = True  # Latch repeated signals through final reaping/recording.
        if process is not None:
            if process.poll() is None:
                try:
                    _stop(process)
                except BaseException as cleanup_error:
                    record["errors"].append("reaping: " + repr(cleanup_error))
            _STOPPING = True
            record.update(child_returncode=process.returncode,
                child_reaped=process.poll() is not None,child_group_gone=_group_gone(process.pid))
            if not record["child_reaped"] or not record["child_group_gone"]:
                record.update(state="reaping_unproved",exit_code=125)
        elif record["spawn_started"]:
            record.update(state="handoff_unproved",exit_code=125)
        if stream is not None:
            try:
                stream.close()
            except BaseException as spool_error:
                record["errors"].append("spool close: " + repr(spool_error))
                record["exit_code"] = 125
        record["interrupted_signal"] = _INTERRUPTED
        if _INTERRUPTED is not None:
            record.update(state="interruption_refused",exit_code=125)
        record["elapsed_seconds"] = time.monotonic()-started
        if admission is not None and result_path is not None and owns_log:
            try:
                _parents(admission["parents"])
                _write_new(result_path,(json.dumps(record,indent=2)+"\n").encode("utf-8"))
            except BaseException as publication_error:
                record["errors"].append("result publication: " + repr(publication_error))
                record.update(state="publication_refused",exit_code=125)
        print(json.dumps(record,sort_keys=True),flush=True)
        for sig,handler in handlers.items():
            signal.signal(sig,handler)
    return record["exit_code"]

if __name__ == "__main__":
    raise SystemExit(main())
