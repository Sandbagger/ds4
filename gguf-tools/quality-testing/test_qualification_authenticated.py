#!/usr/bin/env python3
"""Host-only tests for the descriptor-authenticated controlled runner.

The generated producer is the existing bounded control fake.  These tests
only exercise host descriptors, a private simulated proc tree, and local
process supervision; they never contact a model, GPU, service, or network.
"""

from __future__ import annotations

# This import must be first, before importing the existing fake/process
# helpers.  The initial RED must fail here while the seam is absent, before a
# generated fake can start.
from qualification_authenticated import run_authenticated_qualification_child

import contextlib
import hashlib
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest import mock

import compact_runtime_qualify as COMPACT
import qualification_artifacts as ARTIFACTS
import qualification_process as PROCESS
import qualification_controlled as CONTROLLED
import qualification_authenticated as AUTH
import test_qualification_controlled as CONTROLLED_TEST
import test_qualification_process as PROCESS_TEST
from qualification_artifacts import open_qualification_artifact
from test_qualification_records import _expected, _lifecycle_records


POSIX_PROCESS_GROUPS = PROCESS_TEST.POSIX_PROCESS_GROUPS
EVENTS = PROCESS_TEST.EVENTS
WIRE = COMPACT._QUALIFICATION_CONTROL_MESSAGE
RESULT = COMPACT._QUALIFICATION_CONTROL_SAMPLE_RESULT
MODEL_ACK = COMPACT._QUALIFICATION_CONTROL_MODEL_FD_ACK
RESULT_ACK = COMPACT._QUALIFICATION_CONTROL_SAMPLE_RESULT_ACK
FD_ROOT = "/dev/fd" if sys.platform == "darwin" else "/proc/self/fd"


class _WatchdogExpired(BaseException):
    """Raised by the main-thread watchdog if a native call wedges."""


class _AuthenticationProbe:
    """Record authentication boundaries and bind the actual child PID.

    ``proc_root`` is explicitly a simulated private proc tree on every
    platform.  It is not evidence of Linux-native executable origin.
    """

    def __init__(
        self,
        executable: Any,
        model: Any,
        proc_root: Path,
        *,
        running_target: Path,
    ) -> None:
        self.executable = executable
        # Retain the integer while the owner is live; assertions must not
        # recover it from an owner that authentication has invalidated.
        self.executable_fd = executable.fd
        self.model = model
        self.proc_root = proc_root
        self.running_target = running_target
        self.admitted_pids: set[int] = set()
        self.events: list[tuple[Any, ...]] = []
        self.auth_calls: list[tuple[int, Any]] = []
        self.callback_calls: list[tuple[str, int]] = []
        self.original_auth = ARTIFACTS.authenticate_running_executable
        self.original_verify = ARTIFACTS.QualificationArtifact.verify

    def _role(self, owner: Any) -> str:
        if owner is self.executable:
            return "executable"
        if owner is self.model:
            return "model"
        return "other"

    def verify(self, owner: Any) -> None:
        role = self._role(owner)
        self.events.append(("verify", role))
        try:
            self.original_verify(owner)
        except BaseException as exc:
            self.events.append(("verify_error", role, type(exc).__name__))
            raise

    def authenticate(self, pid: int, artifact: Any) -> None:
        role = self._role(artifact)
        self.auth_calls.append((pid, artifact))
        self.events.append(("auth", pid, role))
        # Bind the exact PID supplied by the admitted child transport.  No
        # guessed PID is ever used, and this fixture never waits or reaps.
        if pid not in self.admitted_pids:
            raise AssertionError("authentication supplied a PID that was not admitted")
        pid_directory = self.proc_root / str(pid)
        pid_directory.mkdir(parents=True, exist_ok=True)
        proc_exe = pid_directory / "exe"
        if not proc_exe.exists() and not proc_exe.is_symlink():
            proc_exe.symlink_to(self.running_target)
        try:
            self.original_auth(pid, artifact)
        except BaseException as exc:
            self.events.append(("auth_error", pid, role, type(exc).__name__))
            raise

    def callback(self, name: str, pid: int) -> dict[str, Any]:
        # Capture what the callback could observe before recording its own
        # event.  The outer assertions prove the same facts independently.
        auth_before = sum(event[0] == "auth" for event in self.events)
        verify_before = tuple(
            sorted({event[1] for event in self.events if event[0] == "verify"})
        )
        self.callback_calls.append((name, pid))
        self.events.append(("callback", name, pid))
        return {
            "callback": name,
            "pid": pid,
            "auth_before": auth_before,
            "verify_before": verify_before,
        }


@contextlib.contextmanager
def _authentication_patches(probe: _AuthenticationProbe) -> Iterator[None]:
    """Patch only the authentication seam and the real artifact proc root."""

    def verify(owner: Any) -> None:
        probe.verify(owner)

    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.object(ARTIFACTS, "_PROC_ROOT", probe.proc_root))
        stack.enter_context(
            mock.patch.object(
                ARTIFACTS.QualificationArtifact, "verify", new=verify
            )
        )
        # Support either direct import or module-qualified production use,
        # without replacing the real authenticate_running_executable logic.
        stack.enter_context(
            mock.patch.object(
                ARTIFACTS,
                "authenticate_running_executable",
                side_effect=probe.authenticate,
            )
        )
        if hasattr(AUTH, "authenticate_running_executable"):
            stack.enter_context(
                mock.patch.object(
                    AUTH,
                    "authenticate_running_executable",
                    side_effect=probe.authenticate,
                )
            )
        if hasattr(CONTROLLED, "authenticate_running_executable"):
            stack.enter_context(
                mock.patch.object(
                    CONTROLLED,
                    "authenticate_running_executable",
                    side_effect=probe.authenticate,
                )
            )
        yield


def _adapt_controlled_fake(fake: Path) -> None:
    """Add bounded metadata facts to the existing generated control fake."""

    source = fake.read_text(encoding="utf-8")
    shebang = f"#!{Path(sys.executable).resolve()}"
    if not source.startswith("#!/usr/bin/env python3\n"):
        raise AssertionError("existing controlled fake shebang changed")
    source = source.replace("#!/usr/bin/env python3", shebang, 1)
    inventory = r"""
FD_ROOT = "/dev/fd" if sys.platform == "darwin" else "/proc/self/fd"


def _fd_inventory():
    descriptor_numbers = []
    try:
        # Close the scandir iterator before fstat: the iterator owns a
        # directory descriptor which must not appear in the inventory.
        with os.scandir(FD_ROOT) as iterator:
            for index, entry in enumerate(iterator):
                if index >= 128:
                    raise RuntimeError("inherited fd inventory exceeded 128 entries")
                try:
                    descriptor = int(entry.name)
                except (TypeError, ValueError):
                    continue
                if descriptor > 2:
                    descriptor_numbers.append(descriptor)
    except OSError:
        return []
    result = []
    for descriptor in descriptor_numbers:
        try:
            observed = os.fstat(descriptor)
        except OSError:
            continue
        result.append({
            "fd": descriptor,
            "device": observed.st_dev,
            "inode": observed.st_ino,
        })
    return sorted(result, key=lambda item: item["fd"])
"""
    needle = "\ndef _metadata(**extra):\n"
    if source.count(needle) != 1:
        raise AssertionError("controlled fake metadata seam changed")
    source = source.replace(needle, inventory + needle, 1)
    sid_line = '        "sid": os.getsid(0),\n'
    if source.count(sid_line) != 1:
        raise AssertionError("controlled fake metadata identity changed")
    source = source.replace(
        sid_line,
        sid_line
        + '        "argv": sys.argv[1:],\n'
        + '        "fd_inventory": _fd_inventory(),\n',
        1,
    )
    ready_line = '_metadata(phase="control_ready", control_fd=control_fd)'
    if source.count(ready_line) != 1:
        raise AssertionError("controlled fake control-ready seam changed")
    source = source.replace(
        ready_line,
        '_metadata(phase="control_ready", control_fd=control_fd, '
        'inherited_fd_inventory=_fd_inventory())',
        1,
    )
    write_line = '        handle.write(json.dumps(data, separators=(",", ":")) + "\\n")\n'
    if source.count(write_line) != 1:
        raise AssertionError("controlled fake metadata write changed")
    source = source.replace(
        write_line,
        write_line + "        handle.flush()\n        os.fsync(handle.fileno())\n",
        1,
    )
    fake.write_text(source, encoding="utf-8")
    fake.chmod(0o700)


def _assert_spawn(
    test: unittest.TestCase,
    requests: list[tuple[tuple[str, ...], tuple[int, ...]]],
    executable_fd: int,
    arguments: list[str] | tuple[str, ...],
) -> int:
    test.assertEqual(len(requests), 1)
    command, inherited = requests[0]
    descriptor_path = f"{FD_ROOT}/{executable_fd}"
    test.assertEqual(
        command,
        (descriptor_path, *tuple(arguments), "--qualification-control-fd", command[-1]),
    )
    positions = [
        index
        for index, value in enumerate(command)
        if value == "--qualification-control-fd"
    ]
    test.assertEqual(positions, [len(command) - 2])
    control_fd = command[-1]
    test.assertTrue(control_fd.isdecimal())
    control_number = int(control_fd)
    test.assertGreater(control_number, 2)
    test.assertNotEqual(control_number, executable_fd)
    test.assertEqual(len(inherited), 2)
    test.assertEqual(set(inherited), {executable_fd, control_number})
    return control_number


def _assert_authentication_boundaries(
    test: unittest.TestCase,
    probe: _AuthenticationProbe,
    *,
    final_requires_both_owners: bool = True,
) -> None:
    callbacks = [
        index for index, event in enumerate(probe.events) if event[0] == "callback"
    ]
    test.assertTrue(callbacks, "no user callback actually ran")
    previous: int | None = None
    for index in callbacks:
        segment = probe.events[(previous + 1 if previous is not None else 0):index]
        auth_count = sum(event[0] == "auth" for event in segment)
        model_verify_count = sum(
            event[0] == "verify" and event[1] == "model" for event in segment
        )
        if previous is None:
            test.assertGreaterEqual(
                auth_count, 1,
                f"authentication did not precede callback at event {index}",
            )
            test.assertGreaterEqual(model_verify_count, 1)
        else:
            # The segment must contain one post-callback and one
            # pre-callback authentication/owner check.  Do not infer their
            # relative order: a failed check may stop the second side.
            test.assertGreaterEqual(auth_count, 2)
            test.assertGreaterEqual(model_verify_count, 2)
        roles = {event[1] for event in segment if event[0] == "verify"}
        test.assertTrue({"executable", "model"} <= roles)
        previous = index
    final_segment = probe.events[callbacks[-1] + 1:]
    test.assertTrue(
        any(event[0] == "auth" for event in final_segment),
        "authentication did not run after the final callback",
    )
    final_roles = {event[1] for event in final_segment if event[0] == "verify"}
    if final_requires_both_owners:
        test.assertTrue({"executable", "model"} <= final_roles)
    else:
        test.assertTrue(final_roles)


def _assert_cleanup(
    test: unittest.TestCase,
    result: Any,
    metadata_path: Path,
    token: str,
) -> dict[str, Any]:
    test.assertTrue(result.transport.cleanup_complete)
    metadata = PROCESS_TEST._read_metadata(metadata_path)
    PROCESS_TEST._assert_metadata(test, metadata, token)
    PROCESS_TEST._assert_pid_gone(test, metadata["leader_pid"])
    PROCESS_TEST._assert_group_gone(test, metadata["pgid"])
    return metadata


def _call_bounded(
    arguments: list[str] | tuple[str, ...],
    expected: dict[str, Any],
    *,
    executable_artifact: Any,
    model_artifact: Any,
    probe: _AuthenticationProbe,
    prepare_descriptor: Any,
    capture_before: Any,
    capture_after: Any,
    watchdog_seconds: float = 8.0,
) -> tuple[Any, list[tuple[tuple[str, ...], tuple[int, ...]]]]:
    """Run one call under a main-thread SIGALRM/BaseException watchdog."""

    if not hasattr(signal, "SIGALRM") or not hasattr(signal, "setitimer"):
        raise unittest.SkipTest("authenticated qualification needs POSIX SIGALRM")

    def invoke() -> Any:
        return run_authenticated_qualification_child(
            arguments,
            expected,
            executable_artifact=executable_artifact,
            model_artifact=model_artifact,
            prepare_descriptor=prepare_descriptor,
            capture_before=capture_before,
            capture_after=capture_after,
            first_token_timeout_ns=2_000_000_000,
            whole_request_timeout_ns=4_000_000_000,
            idle_timeout_ns=2_000_000_000,
            termination_grace_ns=100_000_000,
            control_timeout_seconds=2.0,
        )

    requests: list[tuple[tuple[str, ...], tuple[int, ...]]] = []

    def run_with_spawn_observer() -> Any:
        if sys.platform == "darwin":
            descriptor = executable_artifact.fd
            descriptor_path = f"{FD_ROOT}/{descriptor}"
            requested_descriptors: list[int] = []
            current_popen = subprocess.Popen
            bootstrap = (
                "import os,sys;"
                "fd=int(sys.argv[1]);"
                "fd_path=sys.argv[2];"
                "source=os.pread(fd, 1 << 20, 0);"
                "requested_args=sys.argv[3:];"
                "sys.argv=[fd_path,*requested_args];"
                "namespace={'__name__':'__main__','__file__':fd_path};"
                "exec(compile(source.decode('utf-8'), fd_path, 'exec'), namespace, namespace)"
            )

            def descriptor_exec_path(candidate: int) -> str:
                if candidate != descriptor:
                    raise AssertionError("production requested a non-owner descriptor")
                requested_descriptors.append(candidate)
                return descriptor_path

            def adapt_popen(*args: Any, **kwargs: Any) -> Any:
                command_value = args[0] if args else kwargs.get("args")
                if not isinstance(command_value, (list, tuple)):
                    raise AssertionError("production Popen command was not a sequence")
                command = tuple(command_value)
                positions = [
                    index
                    for index, value in enumerate(command)
                    if value == "--qualification-control-fd"
                ]
                if len(positions) != 1 or positions[0] + 1 >= len(command):
                    raise AssertionError("production omitted its dynamic control fd")
                control_fd = command[positions[0] + 1]
                expected_command = (
                    descriptor_path,
                    *tuple(arguments),
                    "--qualification-control-fd",
                    control_fd,
                )
                if command != expected_command:
                    raise AssertionError("production did not forward the exact argv")
                inherited_value = kwargs.get("pass_fds", ())
                inherited = tuple(inherited_value) if inherited_value is not None else ()
                if len(inherited) != 2 or set(inherited) != {descriptor, int(control_fd)}:
                    raise AssertionError(
                        "production inherited descriptors other than executable+control"
                    )
                requests.append((command, inherited))
                adapted = [
                    str(Path(sys.executable).resolve()),
                    "-c",
                    bootstrap,
                    str(descriptor),
                    descriptor_path,
                    *command[1:],
                ]
                if args:
                    process = current_popen(adapted, *args[1:], **kwargs)
                else:
                    adapted_kwargs = dict(kwargs)
                    adapted_kwargs["args"] = adapted
                    process = current_popen(**adapted_kwargs)
                # Record the PID immediately after the real Popen returns;
                # no guessed-PID proc entry can pass the probe admission gate.
                probe.admitted_pids.add(process.pid)
                return process

            with contextlib.ExitStack() as stack:
                stack.enter_context(
                    mock.patch.object(
                        # This is a test-only adapter.  Production remains
                        # Linux-only and the adapter reads bytes from the fd.
                        __import__("qualification_authenticated"),
                        "_descriptor_exec_path",
                        side_effect=descriptor_exec_path,
                    )
                )
                stack.enter_context(
                    mock.patch.object(subprocess, "Popen", side_effect=adapt_popen)
                )
                result = invoke()
            if requested_descriptors != [descriptor]:
                raise AssertionError("Darwin adapter did not check the owner fd")
            return result

        current_popen = subprocess.Popen

        def observe_popen(*args: Any, **kwargs: Any) -> Any:
            command_value = args[0] if args else kwargs.get("args")
            if not isinstance(command_value, (list, tuple)):
                raise AssertionError("production Popen command was not a sequence")
            inherited_value = kwargs.get("pass_fds", ())
            inherited = tuple(inherited_value) if inherited_value is not None else ()
            requests.append((tuple(command_value), inherited))
            process = current_popen(*args, **kwargs)
            # This is the first operation after the real Popen returns.
            probe.admitted_pids.add(process.pid)
            return process

        with mock.patch.object(subprocess, "Popen", side_effect=observe_popen):
            return invoke()

    def watchdog(_signum: int, _frame: Any) -> None:
        raise _WatchdogExpired("run_authenticated_qualification_child exceeded watchdog")

    with _authentication_patches(probe):
        previous_handler = signal.getsignal(signal.SIGALRM)
        previous_timer = signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, watchdog)
        signal.setitimer(signal.ITIMER_REAL, watchdog_seconds)
        try:
            result = run_with_spawn_observer()
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, previous_handler)
            if previous_timer != (0.0, 0.0):
                signal.setitimer(signal.ITIMER_REAL, *previous_timer)
    return result, requests


@unittest.skipUnless(
    POSIX_PROCESS_GROUPS,
    "authenticated qualification needs POSIX process groups",
)
class QualificationAuthenticatedHostTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.records = _lifecycle_records()
        cls.expected = _expected(cls.records)

    def test_full_authenticated_handshake_forwards_fd_argv_and_callback_copies(self) -> None:
        with tempfile.TemporaryDirectory(prefix="qualification-authenticated-model-") as tmp:
            model = Path(tmp) / "admitted-model.gguf"
            payload = b"authenticated model bytes\0" * 257
            model.write_bytes(payload)
            with CONTROLLED_TEST._controlled_fake_case(
                "complete", self.records, model, stderr_burst=32 << 10
            ) as (command, metadata_path, token):
                fake = Path(command[1])
                _adapt_controlled_fake(fake)
                arguments = ["--profile", "cold warm", "--model", "ignored.gguf"]
                with (
                    open_qualification_artifact(fake, executable=True) as executable,
                    open_qualification_artifact(model) as model_artifact,
                    tempfile.TemporaryDirectory(prefix="SIMULATED-PROC-auth-") as proc,
                ):
                    probe = _AuthenticationProbe(
                        executable,
                        model_artifact,
                        Path(proc),
                        running_target=fake,
                    )
                    received_fds: list[int] = []
                    preparation_facts: list[dict[str, Any]] = []
                    before_facts: list[dict[str, Any]] = []
                    after_facts: list[dict[str, Any]] = []

                    def prepare(pid: int, model_fd: int, evidence: Any) -> dict[str, Any]:
                        fact = probe.callback("prepare", pid)
                        received_fds.append(model_fd)
                        fact.update(
                            {
                                "fd_inode": os.fstat(model_fd).st_ino,
                                "sha256": evidence.sha256,
                                "identity": evidence.identity.as_decimal_mapping(),
                            }
                        )
                        preparation_facts.append(fact)
                        return fact

                    def capture_before(pid: int, sequence: int) -> dict[str, Any]:
                        fact = probe.callback("before", pid)
                        fact["sequence"] = sequence
                        before_facts.append(fact)
                        return fact

                    def capture_after(pid: int, sequence: int) -> dict[str, Any]:
                        fact = probe.callback("after", pid)
                        fact["sequence"] = sequence
                        after_facts.append(fact)
                        return fact

                    result, requests = _call_bounded(
                        arguments,
                        self.expected,
                        executable_artifact=executable,
                        model_artifact=model_artifact,
                        probe=probe,
                        prepare_descriptor=prepare,
                        capture_before=capture_before,
                        capture_after=capture_after,
                    )
                    CONTROLLED_TEST._assert_controlled_shape(self, result)
                    self.assertEqual(result.transport.reason, "complete")
                    self.assertEqual(result.transport.returncode, 0)
                    self.assertEqual(len(result.transport.records), 12)
                    self.assertEqual(result.transport.records, tuple(
                        __import__("json").loads(line)
                        for line in result.transport.stdout.splitlines()
                    ))
                    self.assertEqual(len(result.checkpoints), 12)
                    self.assertIsNone(result.control_error)
                    self.assertEqual(len(probe.callback_calls), 25)
                    self.assertEqual(len(received_fds), 1)
                    self.assertEqual(
                        [name for name, _pid in probe.callback_calls],
                        ["prepare"]
                        + [name for sequence in range(1, 13) for name in ("before", "after")],
                    )
                    _assert_authentication_boundaries(self, probe)
                    control_fd = _assert_spawn(
                        self, requests, probe.executable_fd, arguments
                    )
                    metadata = _assert_cleanup(self, result, metadata_path, token)
                    self.assertEqual(metadata["argv"], arguments + [
                        "--qualification-control-fd", str(control_fd)
                    ])
                    inherited = metadata["inherited_fd_inventory"]
                    self.assertEqual(
                        {entry["fd"] for entry in inherited},
                        {executable.fd, control_fd},
                    )
                    executable_stat = os.fstat(executable.fd)
                    self.assertTrue(any(
                        entry["fd"] == executable.fd
                        and entry["device"] == executable_stat.st_dev
                        and entry["inode"] == executable_stat.st_ino
                        for entry in inherited
                    ))
                    self.assertTrue(metadata.get("model_ack"))
                    self.assertEqual(metadata["milestones"], list(range(1, 13)))
                    callback_pids = {pid for _name, pid in probe.callback_calls}
                    auth_pids = {pid for pid, _artifact in probe.auth_calls}
                    self.assertEqual(len(callback_pids), 1)
                    self.assertEqual(callback_pids, auth_pids)
                    self.assertEqual(next(iter(callback_pids)), metadata["leader_pid"])
                    self.assertEqual(result.preparation, preparation_facts[0])
                    for sequence, checkpoint in enumerate(result.checkpoints, 1):
                        self.assertEqual(checkpoint["sequence"], sequence)
                        self.assertTrue(checkpoint["complete"])
                        self.assertEqual(checkpoint["before"], before_facts[sequence - 1])
                        self.assertEqual(checkpoint["after"], after_facts[sequence - 1])
                    for fact in [*preparation_facts, *before_facts, *after_facts]:
                        fact["caller_mutated"] = True
                    self.assertNotIn("caller_mutated", result.preparation)
                    self.assertNotIn("caller_mutated", result.checkpoints[0]["before"])
                    self.assertNotIn("caller_mutated", result.checkpoints[0]["after"])
                    self.assertEqual(len(result.wire_records), 50)
                    self.assertEqual(
                        sum(record["direction"] == "receive" for record in result.wire_records),
                        25,
                    )
                    self.assertEqual(
                        sum(record["direction"] == "send" for record in result.wire_records),
                        25,
                    )
                    for record in result.wire_records:
                        CONTROLLED_TEST._assert_wire_shape(self, record)
                        self.assertTrue(record["complete"])
                        self.assertEqual(len(record["payload"]), WIRE.size)
                    self.assertEqual(result.wire_records[0]["descriptor_count"], 1)
                    self.assertTrue(all(
                        record["descriptor_count"] == 0
                        for record in result.wire_records[1:]
                    ))
                    evidence = result.model_evidence
                    self.assertIsNotNone(evidence)
                    self.assertEqual(evidence.sha256, hashlib.sha256(payload).hexdigest())
                    self.assertEqual(evidence.identity, model_artifact.identity)
                    for fd in received_fds:
                        with self.assertRaises(OSError):
                            os.fstat(fd)
                    # A successful runner borrows both caller owners.
                    self.assertEqual(os.pread(executable.fd, 2, 0), b"#!")
                    self.assertEqual(os.pread(model_artifact.fd, 16, 0), payload[:16])
                    executable.verify()
                    model_artifact.verify()

    def test_model_inode_and_running_executable_mismatch_reject_before_prepare(self) -> None:
        cases = ("model_inode", "running_executable")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory(
                prefix="qualification-authenticated-reject-"
            ) as tmp:
                directory = Path(tmp)
                admitted_model = directory / "admitted-model.gguf"
                model_bytes = b"same bytes, different model inode\0" * 31
                admitted_model.write_bytes(model_bytes)
                wrong_model = directory / "different-inode-model.gguf"
                wrong_model.write_bytes(model_bytes)
                wrong_runner = directory / "wrong-runner"
                wrong_runner.write_bytes(b"a different executable identity\n")
                wrong_runner.chmod(0o700)
                with CONTROLLED_TEST._controlled_fake_case(
                    "complete",
                    self.records,
                    wrong_model if case == "model_inode" else admitted_model,
                ) as (command, metadata_path, token):
                    fake = Path(command[1])
                    _adapt_controlled_fake(fake)
                    with (
                        open_qualification_artifact(fake, executable=True) as executable,
                        open_qualification_artifact(admitted_model) as model_artifact,
                        tempfile.TemporaryDirectory(prefix="SIMULATED-PROC-reject-") as proc,
                    ):
                        probe = _AuthenticationProbe(
                            executable,
                            model_artifact,
                            Path(proc),
                            running_target=(
                                wrong_runner if case == "running_executable" else fake
                            ),
                        )
                        callback_calls: list[str] = []

                        def prepare(pid: int, model_fd: int, evidence: Any) -> dict[str, Any]:
                            del model_fd, evidence
                            callback_calls.append("prepare")
                            probe.callback("prepare", pid)
                            return {"unexpected": "prepare"}

                        def capture_before(pid: int, sequence: int) -> dict[str, Any]:
                            callback_calls.append(f"before-{sequence}")
                            probe.callback("before", pid)
                            return {"unexpected": "before"}

                        def capture_after(pid: int, sequence: int) -> dict[str, Any]:
                            callback_calls.append(f"after-{sequence}")
                            probe.callback("after", pid)
                            return {"unexpected": "after"}

                        result, requests = _call_bounded(
                            ("--model", "same-bytes.gguf", "--profile", case),
                            self.expected,
                            executable_artifact=executable,
                            model_artifact=model_artifact,
                            probe=probe,
                            prepare_descriptor=prepare,
                            capture_before=capture_before,
                            capture_after=capture_after,
                            watchdog_seconds=5.0,
                        )
                        CONTROLLED_TEST._assert_controlled_shape(self, result)
                        self.assertEqual(result.transport.reason, "protocol_error")
                        self.assertTrue(result.control_error)
                        self.assertLessEqual(len(result.control_error), 4096)
                        self.assertEqual(callback_calls, [])
                        self.assertEqual(probe.callback_calls, [])
                        self.assertTrue(probe.auth_calls)
                        metadata = _assert_cleanup(self, result, metadata_path, token)
                        self.assertFalse(metadata.get("model_ack", False))
                        self.assertEqual(metadata.get("milestones"), [])
                        self.assertEqual(len(result.transport.records), 0)
                        self.assertEqual(len(result.wire_records), 1)
                        self.assertEqual(result.wire_records[0]["direction"], "receive")
                        self.assertEqual(result.wire_records[0]["descriptor_count"], 1)
                        self.assertEqual(len(result.wire_records[0]["payload"]), WIRE.size)
                        evidence = result.model_evidence
                        self.assertIsNotNone(evidence)
                        if case == "model_inode":
                            self.assertEqual(evidence.identity.inode, wrong_model.stat().st_ino)
                            self.assertEqual(evidence.sha256, hashlib.sha256(model_bytes).hexdigest())
                        else:
                            self.assertEqual(evidence.identity, model_artifact.identity)
                        _assert_spawn(
                            self,
                            requests,
                            probe.executable_fd,
                            ("--model", "same-bytes.gguf", "--profile", case),
                        )
                        if case == "running_executable":
                            # The real auth helper invalidates the executable
                            # owner on its failed running-origin comparison.
                            with self.assertRaises(ValueError):
                                _ = executable.fd
                            self.assertEqual(os.pread(model_artifact.fd, 8, 0), model_bytes[:8])
                            model_artifact.verify()
                        else:
                            executable.verify()
                            model_artifact.verify()

    def test_final_capture_drift_has_partial_checkpoint_and_no_final_ack(self) -> None:
        for drift in ("executable_path", "model_bytes"):
            with self.subTest(drift=drift), tempfile.TemporaryDirectory(
                prefix="qualification-authenticated-drift-"
            ) as tmp:
                directory = Path(tmp)
                model = directory / "drift-model.gguf"
                payload = b"stable model before final capture\0" * 29
                model.write_bytes(payload)
                with CONTROLLED_TEST._controlled_fake_case(
                    "complete", self.records, model
                ) as (command, metadata_path, token):
                    fake = Path(command[1])
                    _adapt_controlled_fake(fake)
                    replacement = directory / "replacement-runner"
                    if drift == "executable_path":
                        replacement.write_bytes(fake.read_bytes())
                        replacement.chmod(0o700)
                    mutated = (b"MUTATED model after final capture\0" * 29)[: len(payload)]
                    with (
                        open_qualification_artifact(fake, executable=True) as executable,
                        open_qualification_artifact(model) as model_artifact,
                        tempfile.TemporaryDirectory(prefix="SIMULATED-PROC-drift-") as proc,
                    ):
                        probe = _AuthenticationProbe(
                            executable,
                            model_artifact,
                            Path(proc),
                            running_target=fake,
                        )
                        preparation_facts: list[dict[str, Any]] = []
                        before_facts: list[dict[str, Any]] = []
                        after_facts: list[dict[str, Any]] = []

                        def prepare(pid: int, model_fd: int, evidence: Any) -> dict[str, Any]:
                            fact = probe.callback("prepare", pid)
                            fact["sha256"] = evidence.sha256
                            fact["fd_inode"] = os.fstat(model_fd).st_ino
                            preparation_facts.append(fact)
                            return fact

                        def capture_before(pid: int, sequence: int) -> dict[str, Any]:
                            fact = probe.callback("before", pid)
                            fact["sequence"] = sequence
                            before_facts.append(fact)
                            return fact

                        def capture_after(pid: int, sequence: int) -> dict[str, Any]:
                            fact = probe.callback("after", pid)
                            fact["sequence"] = sequence
                            if sequence == 12:
                                if drift == "executable_path":
                                    os.replace(replacement, fake)
                                else:
                                    with model.open("r+b") as handle:
                                        handle.seek(0)
                                        handle.write(mutated)
                                        handle.flush()
                                        os.fsync(handle.fileno())
                            after_facts.append(fact)
                            return fact

                        result, requests = _call_bounded(
                            ["--profile", "final-capture", "--model", "pinned"],
                            self.expected,
                            executable_artifact=executable,
                            model_artifact=model_artifact,
                            probe=probe,
                            prepare_descriptor=prepare,
                            capture_before=capture_before,
                            capture_after=capture_after,
                            watchdog_seconds=8.0,
                        )
                        CONTROLLED_TEST._assert_controlled_shape(self, result)
                        self.assertNotEqual(result.transport.reason, "complete")
                        self.assertEqual(result.transport.reason, "protocol_error")
                        self.assertTrue(result.control_error)
                        self.assertLessEqual(len(result.control_error), 4096)
                        self.assertEqual(len(result.checkpoints), 12)
                        self.assertTrue(all(
                            checkpoint["complete"] for checkpoint in result.checkpoints[:11]
                        ))
                        self.assertFalse(result.checkpoints[-1]["complete"])
                        self.assertEqual(len(before_facts), 12)
                        self.assertEqual(len(after_facts), 12)
                        self.assertEqual(result.preparation, preparation_facts[0])
                        for sequence in range(1, 12):
                            self.assertEqual(
                                result.checkpoints[sequence - 1]["before"],
                                before_facts[sequence - 1],
                            )
                            self.assertEqual(
                                result.checkpoints[sequence - 1]["after"],
                                after_facts[sequence - 1],
                            )
                        self.assertEqual(len(result.transport.records), 11)
                        self.assertEqual(len(result.wire_records), 49)
                        last_wire = result.wire_records[-1]
                        self.assertEqual(last_wire["direction"], "receive")
                        self.assertEqual(last_wire["descriptor_count"], 0)
                        values = WIRE.unpack(last_wire["payload"])
                        self.assertEqual(values[1], RESULT)
                        self.assertEqual(values[4], 12)
                        self.assertFalse(any(
                            record["direction"] == "send"
                            and WIRE.unpack(record["payload"])[1] == RESULT_ACK
                            and WIRE.unpack(record["payload"])[4] == 12
                            for record in result.wire_records
                        ))
                        self.assertEqual(len(probe.callback_calls), 25)
                        _assert_authentication_boundaries(
                            self, probe, final_requires_both_owners=False
                        )
                        _assert_spawn(
                            self,
                            requests,
                            probe.executable_fd,
                            ["--profile", "final-capture", "--model", "pinned"],
                        )
                        metadata = _assert_cleanup(self, result, metadata_path, token)
                        self.assertEqual(metadata["milestones"], list(range(1, 12)))
                        if drift == "executable_path":
                            with self.assertRaises(ValueError):
                                _ = executable.fd
                            model_artifact.verify()
                        else:
                            with self.assertRaises(ValueError):
                                _ = model_artifact.fd
                            executable.verify()
                        self.assertIsNotNone(result.model_evidence)

    def test_interrupt_callback_propagates_after_owned_cleanup_and_borrowed_fds_survive(self) -> None:
        with tempfile.TemporaryDirectory(prefix="qualification-authenticated-interrupt-") as tmp:
            model = Path(tmp) / "interrupt-model.gguf"
            model.write_bytes(b"interrupt model bytes\0" * 23)
            with CONTROLLED_TEST._controlled_fake_case(
                "complete", self.records, model
            ) as (command, metadata_path, token):
                fake = Path(command[1])
                _adapt_controlled_fake(fake)
                with (
                    open_qualification_artifact(fake, executable=True) as executable,
                    open_qualification_artifact(model) as model_artifact,
                    tempfile.TemporaryDirectory(prefix="SIMULATED-PROC-interrupt-") as proc,
                ):
                    probe = _AuthenticationProbe(
                        executable,
                        model_artifact,
                        Path(proc),
                        running_target=fake,
                    )
                    interrupted = KeyboardInterrupt("authenticated callback interrupt")
                    callback_calls: list[tuple[str, int]] = []

                    def prepare(pid: int, model_fd: int, evidence: Any) -> dict[str, Any]:
                        del model_fd, evidence
                        callback_calls.append(("prepare", pid))
                        return probe.callback("prepare", pid)

                    def capture_before(pid: int, sequence: int) -> dict[str, Any]:
                        del sequence
                        callback_calls.append(("before", pid))
                        probe.callback("before", pid)
                        raise interrupted

                    def capture_after(pid: int, sequence: int) -> dict[str, Any]:
                        callback_calls.append(("after", pid))
                        return probe.callback("after", pid)

                    with self.assertRaises(KeyboardInterrupt) as caught:
                        _call_bounded(
                            ["--profile", "interrupt", "--model", "pinned"],
                            self.expected,
                            executable_artifact=executable,
                            model_artifact=model_artifact,
                            probe=probe,
                            prepare_descriptor=prepare,
                            capture_before=capture_before,
                            capture_after=capture_after,
                            watchdog_seconds=5.0,
                        )
                    self.assertIs(caught.exception, interrupted)
                    self.assertEqual(callback_calls, [
                        ("prepare", probe.callback_calls[0][1]),
                        ("before", probe.callback_calls[1][1]),
                    ])
                    self.assertEqual(probe.callback_calls, callback_calls)
                    self.assertTrue(probe.auth_calls)
                    metadata = PROCESS_TEST._read_metadata(metadata_path)
                    PROCESS_TEST._assert_metadata(self, metadata, token)
                    PROCESS_TEST._assert_pid_gone(self, metadata["leader_pid"])
                    PROCESS_TEST._assert_group_gone(self, metadata["pgid"])
                    self.assertTrue(metadata.get("model_ack"))
                    self.assertEqual(metadata.get("ready_sent"), 1)
                    self.assertNotEqual(metadata.get("ready_ack"), 1)
                    self.assertTrue(all(
                        pid == metadata["leader_pid"]
                        for pid, _artifact in probe.auth_calls
                    ))
                    # The runner borrows caller owners even when a callback
                    # raises; neither owner was mutated in this test.
                    self.assertEqual(os.pread(executable.fd, 2, 0), b"#!")
                    self.assertEqual(os.pread(model_artifact.fd, 9, 0), b"interrupt")
                    executable.verify()
                    model_artifact.verify()

    def test_exit_tail_model_drift_preserves_the_completed_raw_prefix(self) -> None:
        with tempfile.TemporaryDirectory(prefix="qualification-authenticated-tail-") as tmp:
            model = Path(tmp) / "tail-model.gguf"
            payload = b"model before owned child exit"
            model.write_bytes(payload)
            with CONTROLLED_TEST._controlled_fake_case(
                "complete", self.records, model,
            ) as (command, metadata_path, token):
                fake = Path(command[1])
                _adapt_controlled_fake(fake)
                with (
                    open_qualification_artifact(fake, executable=True) as executable,
                    open_qualification_artifact(model) as model_artifact,
                    tempfile.TemporaryDirectory(prefix="SIMULATED-PROC-tail-") as proc,
                ):
                    probe = _AuthenticationProbe(
                        executable, model_artifact, Path(proc), running_target=fake,
                    )
                    completed: list[Any] = []
                    native_controlled = AUTH.run_qualification_controlled_child

                    def mutate_after_owned_exit(*args: Any, **kwargs: Any) -> Any:
                        observed = native_controlled(*args, **kwargs)
                        self.assertEqual(observed.transport.reason, "complete")
                        completed.append(observed)
                        # Model a file change after the last live PID check.
                        # The actual native child is already reaped here.
                        with model.open("r+b") as handle:
                            handle.write(b"M")
                        return observed

                    def prepare(pid: int, _fd: int, _evidence: Any) -> Any:
                        return probe.callback("prepare", pid)

                    def before(pid: int, _sequence: int) -> Any:
                        return probe.callback("before", pid)

                    def after(pid: int, _sequence: int) -> Any:
                        return probe.callback("after", pid)

                    with mock.patch.object(
                        AUTH, "run_qualification_controlled_child", mutate_after_owned_exit,
                    ):
                        result, _requests = _call_bounded(
                            ["--profile", "exit-tail"], self.expected,
                            executable_artifact=executable, model_artifact=model_artifact,
                            probe=probe, prepare_descriptor=prepare,
                            capture_before=before, capture_after=after,
                        )
                    self.assertEqual(len(completed), 1)
                    self.assertEqual(result.transport.reason, "protocol_error")
                    self.assertIn("final artifact", result.control_error)
                    self.assertEqual(result.transport.stdout, completed[0].transport.stdout)
                    self.assertEqual(len(result.transport.records), 12)
                    self.assertEqual(result.wire_records, completed[0].wire_records)
                    self.assertEqual(len(result.wire_records), 50)
                    self.assertEqual(result.model_evidence.sha256, hashlib.sha256(payload).hexdigest())
                    _assert_cleanup(self, result, metadata_path, token)
                    with self.assertRaises(ValueError):
                        _ = model_artifact.fd
                    executable.verify()

    def test_invalid_inputs_reject_before_popen_and_borrowed_owners_survive(self) -> None:
        with tempfile.TemporaryDirectory(prefix="qualification-authenticated-preflight-") as tmp:
            directory = Path(tmp)
            executable_path = directory / "runner"
            executable_path.write_text(
                f"#!{Path(sys.executable).resolve()}\nraise SystemExit(0)\n",
                encoding="utf-8",
            )
            executable_path.chmod(0o700)
            model_path = directory / "model.gguf"
            model_path.write_bytes(b"preflight model")
            with (
                open_qualification_artifact(executable_path, executable=True) as executable,
                open_qualification_artifact(model_path) as model_artifact,
                mock.patch.object(subprocess, "Popen") as popen,
            ):
                callback = lambda *args: {"unexpected": args}
                invalid_calls = (
                    ([], ValueError, callback, callback, callback),
                    (["--qualification-control-fd", "7"], ValueError, callback, callback, callback),
                    (["--qualification-control-fd=7"], ValueError, callback, callback, callback),
                    (["--bad", 7], TypeError, callback, callback, callback),
                    (["--valid"], TypeError, None, callback, callback),
                    (["--valid"], TypeError, callback, None, callback),
                    (["--valid"], TypeError, callback, callback, None),
                )
                for arguments, error, prepare, before, after in invalid_calls:
                    with self.subTest(arguments=arguments, error=error.__name__):
                        with self.assertRaises(error):
                            run_authenticated_qualification_child(
                                arguments,
                                self.expected,
                                executable_artifact=executable,
                                model_artifact=model_artifact,
                                prepare_descriptor=prepare,
                                capture_before=before,
                                capture_after=after,
                            )
                popen.assert_not_called()
                self.assertEqual(os.pread(executable.fd, 2, 0), b"#!")
                self.assertEqual(os.pread(model_artifact.fd, len(b"preflight"), 0), b"preflight")
                executable.verify()
                model_artifact.verify()


if __name__ == "__main__":
    unittest.main(verbosity=2)
