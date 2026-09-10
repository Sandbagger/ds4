"""Checkout-packaging controls only; synthetic dependencies are never compiled."""
from contextlib import contextmanager
from pathlib import Path
from unittest import mock
import hashlib
import importlib.util
import json
import os
import re
import stat
import sys
import tempfile
import unittest
ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
LAUNCHER = TESTS / "run_engine_session_lifetime.py"
CONTROLLER = TESTS / "engine_session_lifetime_controller.py"
METRICS = TESTS / "engine_session_lifetime_metrics.json"
TEMPLATE = TESTS / "engine_session_lifetime.mk"
SOURCE_DEPENDENCIES = (
    "ds4.c", "ds4_distributed.c", "ds4_tp.c", "ds4_ssd.c", "ds4_laguna_stream.c",
    "ds4_runtime.c", "ds4_qualification_control.c", "ds4_plan_io.c", "ds4_laguna_plan.c",
    "ds4_layer_pack.c", "ds4.h", "ds4_distributed.h", "ds4_laguna_plan.h",
    "ds4_laguna_stream.h", "ds4_tp.h", "ds4_layer_pack.h", "ds4_gpu_mgpu.h",
    "ds4_gpu.h", "ds4_ssd.h", "ds4_runtime.h", "ds4_plan_io.h", "ds4_gpu_resident.h",
    "ds4_streaming_hotlist.inc", "ds4_streaming_hotlist_glm52.inc")
CASES = ["empty", "live-one", "live-two", "retained-after-unlock"]
PINNED_FILES = {
    "tests/test_engine_session_lifetime.c": (48228, 973, "a35fbb7f2dc00f571028f995c65e4106b12e6bcfef996168f20a5e36d18c2d25"),
    "tests/test_engine_session_lifetime.py": (51356, 965, "3619c936a3f7415ba332ab4f3db457793c89954a6f550eb393902e554a30cc66"),
    "tests/engine_session_lifetime_controller.py": (15721, 344, "5fdf88f748c7a714e39ce16b356cc09afa7b1bddc2109074621ef44d256578ca"),
    "tests/engine_session_lifetime.mk": (1304, 14, "f24ff362027767215d461432029e43b7f0b05ba590a9cbf411220e43fe7b14d6"),
}
METRIC_FINGERPRINT = "83ab234fa8abbf2c2cea5d983df0789e4840a5782bbc52199f116c671664a855"
METRIC_TYPES = {"bool": 229, "uint": 111, "int": 19, "address": 7, "string": 1}
METRIC_SCHEMA = "laguna.heap-engine-session-metric-schema/v1"
FIXTURE_SHA256 = PINNED_FILES["tests/test_engine_session_lifetime.c"][2]
FIXTURE_BYTES, FIXTURE_LINES = PINNED_FILES["tests/test_engine_session_lifetime.c"][:2]
EVIDENCE_CLASS = "actual_core_DS4_NO_GPU_heap_lifetime_synthetic_pthread_fault"
LIMITS = {"wall_seconds": 240, "cpu_seconds": 240, "compiler_seconds": 120, "native_case_seconds": 10, "core_bytes": 0, "file_size_bytes": 64 << 20, "log_bytes": 1 << 20, "native_stdout_bytes": 32768, "term_grace_seconds": 8, "kill_grace_seconds": 2, "process_ceiling": 4096}
CONFIG_FIELDS = {"schema", "repository_root", "parents", "inputs", "preparation_path",
    "metric_schema_path", "environment", "logs_directory", "logs_identity"}
ENV_FIXED = {"PATH": "/opt/homebrew/bin:/usr/bin:/bin", "LANG": "C", "LC_ALL": "C", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1", "GIT_CONFIG_NOSYSTEM": "1", "CUDA_VISIBLE_DEVICES": "", "__CF_USER_TEXT_ENCODING": "0x1F5:0x0:0x0"}
PACKAGE_FILES = (LAUNCHER, CONTROLLER, METRICS, TEMPLATE)
def _identity(st):
    return (st.st_uid, st.st_dev, st.st_ino, st.st_mode, st.st_nlink,
            st.st_size, st.st_mtime_ns, st.st_ctime_ns)

def _read(path, cap=1 << 20, strict=True):
    path = Path(path)
    if not _canonical(path):
        raise RuntimeError("noncanonical fixture input: %s" % path)
    pre = os.lstat(path)
    if (not stat.S_ISREG(pre.st_mode) or pre.st_uid != os.getuid()
            or (strict and pre.st_nlink != 1) or not 0 <= pre.st_size <= cap):
        raise RuntimeError("fixture input type/owner/link/size: %s" % path)
    identity = _identity(pre)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        if _identity(os.fstat(fd)) != identity:
            raise RuntimeError("fixture input changed before open")
        chunks, size = [], 0
        while True:
            chunk = os.read(fd, min(65536, cap + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > cap:
                raise RuntimeError("fixture input read overflow")
        if _identity(os.fstat(fd)) != identity:
            raise RuntimeError("fixture input changed during read")
    finally:
        os.close(fd)
    if _identity(os.lstat(path)) != identity:
        raise RuntimeError("fixture input path changed")
    raw = b"".join(chunks)
    if len(raw) != pre.st_size:
        raise RuntimeError("fixture input size changed")
    return raw, pre

def _record(path, strict=True):
    raw, st = _read(path, strict=strict)
    return {"path": str(path), "bytes": len(raw),
        "lines": len(raw.decode("utf-8").splitlines()),
        "sha256": hashlib.sha256(raw).hexdigest(), "uid": st.st_uid,
        "device": st.st_dev, "inode": st.st_ino, "mode": stat.S_IMODE(st.st_mode),
        "nlink": st.st_nlink}

def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeError("duplicate fixture JSON key: " + key)
        result[key] = value
    return result

def _constant(value):
    raise RuntimeError("nonfinite fixture JSON value: " + value)
def _dir_identity(path):
    st = os.lstat(path)
    if (not _canonical(path) or not stat.S_ISDIR(st.st_mode)
            or st.st_uid != os.getuid()):
        raise RuntimeError("not a canonical owned directory: %s" % path)
    return [st.st_uid, st.st_dev, st.st_ino, stat.S_IMODE(st.st_mode)]
def _canonical(path):
    return (path.is_absolute() and os.path.normpath(str(path)) == str(path)
            and os.path.realpath(path) == str(path))
def _within(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
def _exclusive(path, raw, mode=0o644):
    path.parent.mkdir(mode=0o700, exist_ok=True)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
    try:
        view = memoryview(raw)
        while view:
            count = os.write(fd, view)
            if count <= 0:
                raise AssertionError("short write")
            view = view[count:]
        os.fsync(fd)
    finally:
        os.close(fd)
class EngineSessionLifetimePackagingTest(unittest.TestCase):
    def require_file(self, path):
        self.assertTrue(os.path.lexists(path), "required package file is absent: %s" % path)
        st = os.lstat(path)
        self.assertTrue(stat.S_ISREG(st.st_mode), "package file is not regular: %s" % path)
        self.assertFalse(stat.S_ISLNK(st.st_mode), "package file is symlink: %s" % path)
        self.assertEqual(st.st_nlink, 1, "package file is linked: %s" % path)
        self.assertEqual(st.st_uid, os.getuid(), "package file owner: %s" % path)
    def require_package(self):
        for path in PACKAGE_FILES:
            self.require_file(path)
    def require_pin(self, relative):
        path = ROOT / relative
        self.require_file(path)
        expected_bytes, expected_lines, expected_sha = PINNED_FILES[relative]
        raw, _ = _read(path)
        self.assertEqual((len(raw), len(raw.decode("utf-8").splitlines()),
                          hashlib.sha256(raw).hexdigest()),
                         (expected_bytes, expected_lines, expected_sha), relative)
    @contextmanager
    def sandbox(self):
        raw_tmp = os.environ.get("TMPDIR", "")
        if not raw_tmp or not _canonical(Path(raw_tmp)):
            raise RuntimeError("tests require canonical raw admitted TMPDIR")
        parent = Path(raw_tmp)
        parent_id = _dir_identity(parent)
        if parent_id[-1] != 0o700:
            raise RuntimeError("admitted TMPDIR must be0700")
        temporary = tempfile.TemporaryDirectory(prefix="ds4-lifetime-packaging-", dir=raw_tmp)
        # Detach automatic deletion so a changed root is preserved, not traversed.
        temporary._finalizer.detach()
        root = Path(temporary.name)
        root_id = _dir_identity(root)
        if root.parent != parent or root_id[-1] != 0o700:
            raise RuntimeError("fresh fixture root ownership")
        try:
            yield root
        finally:
            if _dir_identity(parent) != parent_id or _dir_identity(root) != root_id:
                raise RuntimeError("preserving changed fixture root: %s" % root)
            temporary.cleanup()
            if os.path.lexists(root) or _dir_identity(parent) != parent_id:
                raise RuntimeError("fixture cleanup not proved")
    def stage(self, parent):
        self.require_package()
        for relative in PINNED_FILES:
            self.require_pin(relative)
        repo, tests = parent / "checkout", parent / "checkout/tests"
        repo.mkdir(mode=0o700)
        tests.mkdir(mode=0o700)
        for relative in ("tests/run_engine_session_lifetime.py",
                         "tests/engine_session_lifetime_controller.py",
                         "tests/engine_session_lifetime_metrics.json",
                         "tests/engine_session_lifetime.mk",
                         "tests/test_engine_session_lifetime.c",
                         "tests/test_engine_session_lifetime.py"):
            raw, _ = _read(ROOT / relative)
            _exclusive(repo / relative, raw)
        for relative in SOURCE_DEPENDENCIES:
            if relative != "tests/test_engine_session_lifetime.c":
                _exclusive(repo / relative,
                           ("/* synthetic packaging dependency: %s */\n" % relative).encode())
        return repo
    def load_launcher(self, path):
        self.require_file(path)
        spec = importlib.util.spec_from_file_location("engine_session_lifetime_packaging_%x" % id(path), str(path))
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        old = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            spec.loader.exec_module(module)
        finally:
            sys.dont_write_bytecode = old
        self.assertTrue(callable(getattr(module, "prepare", None)))
        return module
    @contextmanager
    def imported_on(self, platform_name):
        with mock.patch.object(sys, "platform", platform_name):
            yield
    def marker_snapshot(self, parent):
        marker = parent / "caller-marker"
        if not os.path.lexists(marker):
            _exclusive(marker, b"caller-owned marker\n", mode=0o600)
        return (tuple(sorted(p.name for p in parent.iterdir())), _read(marker)[0], _record(marker))
    def assert_record_current(self, record, path=None):
        self.assertIsInstance(record, dict)
        actual_path = Path(record["path"])
        if path is not None:
            self.assertEqual(actual_path, path)
        self.assertTrue(_canonical(actual_path), "record path is not canonical")
        actual = _record(actual_path)
        for key, value in actual.items():
            self.assertEqual(record.get(key), value, key)
        # Relocation is checked against the actual owned checkout/workspace.
        # An admitted temporary parent may itself be a Prime artifact directory.
    def json_file(self, path, cap=131072):
        self.require_file(path)
        raw, _ = _read(path, cap)
        return json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs,
                          parse_constant=_constant)
    def assert_dir_leaf(self, path):
        st = os.lstat(path)
        self.assertTrue(stat.S_ISDIR(st.st_mode) and not stat.S_ISLNK(st.st_mode))
        self.assertTrue(_canonical(path))
        self.assertEqual((st.st_uid, stat.S_IMODE(st.st_mode)), (os.getuid(), 0o700))
    def test_00_versioned_assets_preserve_verified_content(self):
        self.require_package()
        for relative in PINNED_FILES:
            self.require_pin(relative)
        document = self.json_file(METRICS)
        self.assertEqual(set(document), {"schema", "fixture", "metric_count", "max_key_bytes",
                         "capacity", "evidence_class", "metrics"})
        self.assertEqual((document["schema"], document["metric_count"], document["max_key_bytes"],
                          document["capacity"], document["evidence_class"]),
                         (METRIC_SCHEMA, 367, 44, 512,
                          "static_emission_census_not_C_compile_or_native_execution"))
        self.assertEqual(document["fixture"], {"sha256": FIXTURE_SHA256})
        metrics = document["metrics"]
        self.assertEqual(len(metrics), 367)
        canonical = json.dumps(metrics, sort_keys=True, separators=(",", ":")).encode()
        self.assertEqual(hashlib.sha256(canonical).hexdigest(), METRIC_FINGERPRINT)
        self.assertEqual({kind: sum(entry.get("type") == kind for entry in metrics.values())
                          for kind in METRIC_TYPES}, METRIC_TYPES)
        self.assertEqual(sum("availability_key" in entry for entry in metrics.values()), 70)
    def test_01_relocated_preparation_binds_current_checkout(self):
        with self.sandbox() as parent:
            repo = self.stage(parent)
            with self.imported_on("darwin"):
                launcher = self.load_launcher(repo / "tests/run_engine_session_lifetime.py")
                result = launcher.prepare(repo, parent)
            self.assertIsInstance(result, dict)
            for key in ("workspace", "admission_path", "admission_sha256", "environment", "controller_path"):
                self.assertIn(key, result)
            workspace, admission_path = Path(result["workspace"]), Path(result["admission_path"])
            controller_path = Path(result["controller_path"])
            self.assertTrue(_canonical(workspace) and workspace.parent == parent)
            self.assertEqual(result["controller_path"], str(repo / "tests/engine_session_lifetime_controller.py"))
            self.assertRegex(result["admission_sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(result["admission_sha256"], hashlib.sha256(_read(admission_path, 131072)[0]).hexdigest())
            self.assertTrue(_canonical(admission_path) and _within(admission_path, workspace))
            self.assertEqual(stat.S_IMODE(os.lstat(workspace).st_mode), 0o700)
            leaves = {name: workspace / name for name in ("home", "tmp", "config", "cache", "data", "logs")}
            for path in leaves.values():
                self.assert_dir_leaf(path)
            self.assertEqual(len({path.stat().st_ino for path in leaves.values()}), 6)
            admission = self.json_file(admission_path)
            self.assertEqual((admission["schema"], admission["state"], admission["repository_root"],
                              admission["cwd"], admission["evidence_class"]),
                             ("laguna.heap-engine-session-cpu-admission/v1", "admitted", str(repo),
                              str(repo), EVIDENCE_CLASS))
            artifact_root = Path(admission["artifact_root"])
            self.assertTrue(_canonical(artifact_root) and _within(artifact_root, workspace))
            self.assertTrue(stat.S_ISDIR(os.lstat(artifact_root).st_mode))
            self.assertEqual(admission["limits"], LIMITS)
            for key in ("gpu_execution", "model_execution", "qualification"):
                if key in admission:
                    self.assertFalse(admission[key])
            expected_env = dict(ENV_FIXED)
            expected_env.update({"HOME": str(leaves["home"]), "TMPDIR": str(leaves["tmp"]),
                                 "XDG_CONFIG_HOME": str(leaves["config"]), "XDG_CACHE_HOME": str(leaves["cache"]),
                                 "XDG_DATA_HOME": str(leaves["data"])})
            self.assertEqual((result["environment"], admission["environment"]), (expected_env, expected_env))
            config_record, config_path = admission["config"], Path(admission["config"]["path"])
            self.assertEqual(config_path.parent, workspace)
            self.assertEqual(stat.S_IMODE(os.lstat(config_path).st_mode), 0o600)
            self.assert_record_current(config_record, config_path)
            config = self.json_file(config_path)
            self.assertEqual(set(config), CONFIG_FIELDS)
            self.assertEqual((config["schema"], config["repository_root"], config["environment"],
                              config["logs_directory"], config["logs_identity"]),
                             ("laguna.heap-engine-session-cpu-config/v1", str(repo), expected_env,
                              str(leaves["logs"]), _dir_identity(leaves["logs"])))
            prep_path, metric_path = Path(config["preparation_path"]), Path(config["metric_schema_path"])
            self.assertEqual(prep_path.parent, workspace)
            self.assertEqual(stat.S_IMODE(os.lstat(prep_path).st_mode), 0o600)
            self.assertEqual(metric_path, repo / "tests/engine_session_lifetime_metrics.json")
            prep = self.json_file(prep_path)
            self.assertEqual((prep["schema"], prep["repository_root"], prep["platform"], prep["native_cases"]),
                             ("laguna.heap-engine-session-native-static-preparation/v1", str(repo), "Darwin", CASES))
            self.assertEqual(prep["compile_argv"], ["/usr/bin/make", "--no-print-directory", "-j1", "-f", "fixture.mk", "all"])
            deps = prep["source_dependencies"]
            self.assertEqual(len(deps), 24)
            self.assertEqual([r["relative_path"] for r in deps], list(SOURCE_DEPENDENCIES))
            for record in deps:
                path = Path(record["path"])
                self.assertEqual(path, repo / record["relative_path"])
                self.assertTrue(_within(path, repo))
                self.assert_record_current(record, path)
            fixture, template = prep["fixture"], prep["make_template"]
            self.assertEqual(fixture["relative_path"], "tests/test_engine_session_lifetime.c")
            self.assert_record_current(fixture, repo / fixture["relative_path"])
            self.assertEqual(fixture["sha256"], FIXTURE_SHA256)
            self.assertEqual(template["path"], str(repo / "tests/engine_session_lifetime.mk"))
            self.assert_record_current(template, repo / "tests/engine_session_lifetime.mk")
            metric_document = self.json_file(metric_path)
            self.assertEqual((metric_document["metric_count"], metric_document["fixture"]["sha256"]), (367, fixture["sha256"]))
            config_inputs = config["inputs"]
            self.assertEqual((len(config_inputs), len({r["path"] for r in config_inputs})), (31, 31))
            for record in config_inputs:
                path = Path(record["path"])
                self.assertTrue(path == prep_path or _within(path, repo))
                self.assert_record_current(record, path)
            admission_inputs = admission["inputs"]
            self.assertEqual((len(admission_inputs), len({r["path"] for r in admission_inputs})), (32, 32))
            self.assertEqual(admission["runner_inputs"], config_inputs)
            self.assertEqual({r["path"] for r in admission_inputs}, {str(config_path)} | {r["path"] for r in config_inputs})
            for record in admission_inputs:
                path = Path(record["path"])
                self.assertTrue(path in (config_path, prep_path) or _within(path, repo))
                self.assert_record_current(record, path)
            self.assertEqual(admission["command"], ["python3", admission["runner"]["path"], str(config_path), config_record["sha256"]])
            self.assertEqual((admission["runner"]["path"], admission["driver"]["path"]),
                             (str(repo / "tests/test_engine_session_lifetime.py"), str(controller_path)))
            by_path = {record["path"]: record for record in admission_inputs}
            self.assertEqual(admission["runner"], by_path[str(repo / "tests/test_engine_session_lifetime.py")])
            self.assertEqual(admission["driver"], by_path[str(controller_path)])
            self.assertEqual(config_record, by_path[str(config_path)])
            expected_parents = {repo, repo / "tests", parent, workspace, *leaves.values()}
            parents = admission["parents"]
            self.assertEqual(len(parents), len(expected_parents))
            self.assertEqual({Path(record["path"]) for record in parents}, expected_parents)
            self.assertEqual(config["parents"], parents)
            for record in parents:
                self.assertEqual(record["identity"], _dir_identity(Path(record["path"])))
            self.assertEqual(artifact_root, workspace)
            log_path, result_path = Path(admission["log_path"]), Path(admission["result_path"])
            self.assertNotEqual(log_path, result_path)
            for path in (log_path, result_path):
                self.assertTrue(_canonical(path) and path.parent == workspace)
                self.assertFalse(os.path.lexists(path))
    def test_02_changed_fixture_refuses_before_workspace_creation(self):
        with self.sandbox() as parent:
            repo = self.stage(parent)
            fixture = repo / "tests/test_engine_session_lifetime.c"
            fixture.unlink()
            _exclusive(fixture, b"changed fixture\n")
            before = self.marker_snapshot(parent)
            with self.imported_on("darwin"):
                launcher = self.load_launcher(repo / "tests/run_engine_session_lifetime.py")
                with self.assertRaises(RuntimeError) as refused:
                    launcher.prepare(repo, parent)
            self.assertEqual(self.marker_snapshot(parent), before)
    def test_03_linked_input_refuses_without_touching_foreign_data(self):
        with self.sandbox() as parent:
            repo = self.stage(parent)
            fixture, foreign = repo / "tests/test_engine_session_lifetime.c", parent / "foreign-data.bin"
            _exclusive(foreign, _read(fixture)[0], mode=0o600)
            fixture.unlink()
            os.link(foreign, fixture)
            foreign_before, before = _record(foreign, strict=False), self.marker_snapshot(parent)
            with self.imported_on("darwin"):
                launcher = self.load_launcher(repo / "tests/run_engine_session_lifetime.py")
                with self.assertRaises(RuntimeError) as refused:
                    launcher.prepare(repo, parent)
            self.assertEqual((self.marker_snapshot(parent), _record(foreign, strict=False)), (before, foreign_before))
    def test_04_unsupported_platform_refuses_before_effects(self):
        with self.sandbox() as parent:
            repo = self.stage(parent)
            before = self.marker_snapshot(parent)
            with self.imported_on("linux"):
                launcher = self.load_launcher(repo / "tests/run_engine_session_lifetime.py")
                with self.assertRaises(RuntimeError) as refused:
                    launcher.prepare(repo, parent)
            self.assertEqual(self.marker_snapshot(parent), before)
    def test_05_make_target_is_explicit_and_darwin_scoped(self):
        makefile = ROOT / "Makefile"
        self.require_file(makefile)
        lines = _read(makefile)[0].decode("utf-8").splitlines()
        self.assertIn("test-engine-session-lifetime", " ".join(l for l in lines if l.startswith(".PHONY:")))
        contexts, stack = [], []
        for line in lines:
            stripped = line.strip()
            condition = re.match(r"^(ifeq|ifneq|ifdef|ifndef)\b", stripped)
            if condition:
                darwin = re.fullmatch(r"(ifeq|ifneq) \(\$\(UNAME_S\),Darwin\)", stripped)
                stack.append((darwin.group(1) == "ifeq") if darwin else None)
            contexts.append(tuple(stack))
            if stripped == "else" and stack:
                if stack[-1] is not None:
                    stack[-1] = not stack[-1]
            elif stripped == "endif":
                self.assertTrue(stack, "unbalanced Make endif")
                stack.pop()
        targets = [i for i, l in enumerate(lines) if re.match(r"^test-engine-session-lifetime\s*:", l)]
        self.assertEqual(len(targets), 1, "one explicit packaging target")
        target = targets[0]
        end = next((i for i in range(target + 1, len(lines))
                    if re.match(r"^[^ \t#].*:", lines[i]) or lines[i].strip() == "endif"), len(lines))
        block = "\n".join(lines[max(0, target - 3):end])
        self.assertEqual(contexts[target], (), "explicit target must not silently vanish on Linux")
        recipes = [line.strip() for line in lines[target + 1:end] if line.startswith("\t")]
        self.assertEqual(recipes, ["python3 tests/run_engine_session_lifetime.py"])
        self.assertRegex(block, r"(?i)darwin")
        self.assertRegex(block, r"(?i)cpu[- ]only")
        resident = [i for i, l in enumerate(lines) if re.match(r"^test-laguna-resident-path\s*:", l)]
        self.assertTrue(resident, "resident aggregate exists")
        additions = [i for i in resident if "test-engine-session-lifetime" in lines[i]]
        self.assertTrue(additions, "resident aggregate includes packaging test")
        self.assertTrue(all(True in contexts[i] and False not in contexts[i] for i in additions),
                        "Linux aggregate changed")
        self.assertFalse(stack, "unclosed Make condition")
if __name__ == "__main__":
    unittest.main()
