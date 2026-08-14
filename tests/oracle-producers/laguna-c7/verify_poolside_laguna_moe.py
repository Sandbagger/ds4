#!/usr/bin/python3
"""Fail-closed verifier for the fixed Laguna C7 Poolside oracle producer."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shlex
import stat
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable


Q4_MANIFEST = Path("tests/test-vectors/laguna-q4-l2-auto/manifest.json")
RESIDUAL_MANIFEST = Path(
    "tests/test-vectors/laguna-moe-residual-auto/manifest.json"
)
SCHEMAS = {
    Q4_MANIFEST: "laguna-q4-l2-auto-fixture/v2",
    RESIDUAL_MANIFEST: "laguna-moe-residual-auto-fixture/v2",
}
PRODUCER_KEYS = {
    "probe",
    "poolside_patch",
    "capture_script",
    "tokens",
    "verifier",
}
NON_PERTURBATION_FILES = {
    "selected_sha256": "layer-01-router-selected.i32",
    "weights_sha256": "layer-01-router-weights.f32",
    "routed_output_sha256": "layer-01-ffn-moe-out.f32",
    "shared_output_sha256": "layer-01-ffn-shared-out.f32",
    "ffn_output_sha256": "layer-01-ffn-out.f32",
    "layer_output_sha256": "layer-01.f32",
}
RESIDUAL_NON_PERTURBATION_KEYS = {
    "routed_output_sha256",
    "shared_output_sha256",
    "ffn_output_sha256",
    "layer_output_sha256",
}
CONSOLIDATED_CAPTURE_KEY = "consolidated_capture_matches_prior_captures"
CLEAN_SUBPROCESS_ENV = {
    "PATH": "/usr/bin:/bin:/usr/local/cuda/bin",
    "LANG": "C",
    "LC_ALL": "C",
    "TMPDIR": "/tmp",
}


class ContractError(RuntimeError):
    """The fixed capture contract was not satisfied."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ContractError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def _sha256_fd(fd: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while True:
        try:
            chunk = os.pread(fd, 8 * 1024 * 1024, offset)
        except OSError as error:
            raise ContractError(f"cannot hash held model fd {fd}: {error}") from error
        if not chunk:
            return digest.hexdigest()
        digest.update(chunk)
        offset += len(chunk)


def _run(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=CLEAN_SUBPROCESS_ENV if env is None else env,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise ContractError(f"command failed ({' '.join(command)}): {detail}")
    return result.stdout


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot read manifest {path}: {error}") from error
    if not isinstance(value, dict):
        raise ContractError(f"manifest is not an object: {path}")
    return value


def discover_repo_root(verifier_path: Path) -> Path:
    resolved = verifier_path.resolve()
    try:
        candidate = resolved.parents[3]
    except IndexError as error:
        raise ContractError(
            "verifier must run from a full DS4 checkout"
        ) from error
    expected = (
        candidate
        / "tests/oracle-producers/laguna-c7/verify_poolside_laguna_moe.py"
    )
    manifests = (candidate / Q4_MANIFEST, candidate / RESIDUAL_MANIFEST)
    if resolved != expected.resolve() or not all(path.is_file() for path in manifests):
        raise ContractError("verifier must run from a full DS4 checkout")
    return candidate


def _cache_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ContractError(f"cannot read CMake cache {path}: {error}") from error
    for line in lines:
        if not line or line.startswith(("#", "//")) or "=" not in line:
            continue
        key_and_type, value = line.split("=", 1)
        key = key_and_type.split(":", 1)[0]
        values[key] = value
    return values


def _bool_cache(value: str | None, key: str) -> bool:
    if value == "ON":
        return True
    if value == "OFF":
        return False
    raise ContractError(f"CMake cache {key} must be ON or OFF")


def _float32_add(left: float, right: float) -> float:
    return struct.unpack("<f", struct.pack("<f", left + right))[0]


def _elf64_dynamic_tags(path: Path) -> set[int]:
    try:
        payload = path.read_bytes()
        if (
            len(payload) < 64
            or payload[:4] != b"\x7fELF"
            or payload[4] != 2
            or payload[5] != 1
        ):
            raise ContractError("probe binary must be little-endian ELF64")
        header = struct.unpack_from("<16sHHIQQQIHHHHHH", payload)
        program_offset = header[5]
        program_entry_size = header[9]
        program_count = header[10]
        if program_count and program_entry_size < 56:
            raise ContractError("probe ELF program-header size is invalid")
        tags: set[int] = set()
        for index in range(program_count):
            offset = program_offset + index * program_entry_size
            if offset < 0 or offset + 56 > len(payload):
                raise ContractError("probe ELF program header is truncated")
            program = struct.unpack_from("<IIQQQQQQ", payload, offset)
            if program[0] != 2:
                continue
            dynamic_offset = program[2]
            dynamic_size = program[5]
            if (
                dynamic_offset < 0
                or dynamic_size % 16
                or dynamic_offset + dynamic_size > len(payload)
            ):
                raise ContractError("probe ELF dynamic table is invalid")
            terminated = False
            for dynamic_entry in range(
                dynamic_offset, dynamic_offset + dynamic_size, 16
            ):
                tag, _ = struct.unpack_from("<qQ", payload, dynamic_entry)
                if tag == 0:
                    terminated = True
                    break
                tags.add(tag)
            if not terminated:
                raise ContractError("probe ELF dynamic table is unterminated")
        return tags
    except ContractError:
        raise
    except (OSError, struct.error) as error:
        raise ContractError(f"cannot inspect probe ELF: {error}") from error


class Verifier:
    def __init__(
        self,
        repo_root: Path,
        *,
        device_query: Callable[[], dict[str, Any]] | None = None,
        compute_process_query: Callable[[], list[int]] | None = None,
        toolchain_query: Callable[[str, Path], dict[str, Any]] | None = None,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.device_query = device_query or self._query_device
        self.compute_process_query = (
            compute_process_query or self._query_compute_processes
        )
        self.toolchain_query = toolchain_query or self._query_toolchain

    def preflight(
        self,
        poolside_src: Path,
        poolside_build: Path,
        model_fd: int,
        *,
        continuity_fd: int,
    ) -> None:
        manifests, model_identity = self._validate_common(model_fd)
        self._validate_source(
            Path(poolside_src), Path(poolside_build), manifests, patched=False
        )
        self._validate_build(Path(poolside_src), Path(poolside_build), manifests)
        self._validate_device(manifests)
        self._validate_gpu_exclusive()
        self._write_model_continuity(continuity_fd, model_identity)

    def captured(
        self,
        poolside_src: Path,
        poolside_build: Path,
        model_fd: int,
        capture_root: Path,
        probe_bin: Path,
        *,
        continuity_fd: int,
    ) -> None:
        manifests, model_identity = self._validate_common(model_fd)
        self._validate_model_continuity(continuity_fd, model_identity)
        self._validate_source(
            Path(poolside_src), Path(poolside_build), manifests, patched=True
        )
        self._validate_build(Path(poolside_src), Path(poolside_build), manifests)
        self._validate_device(manifests)
        self._validate_gpu_exclusive()
        self._validate_probe_binary(Path(probe_bin), manifests)
        self._validate_captures(Path(capture_root), manifests)
        self._validate_residual_semantics(manifests[RESIDUAL_MANIFEST])

    def _manifests(self) -> dict[Path, dict[str, Any]]:
        manifests = {
            relative: _read_json(self.repo_root / relative) for relative in SCHEMAS
        }
        for relative, schema in SCHEMAS.items():
            if manifests[relative].get("schema") != schema:
                raise ContractError(f"unexpected schema in {relative}")
        return manifests

    def _validate_common(
        self, model_fd: int
    ) -> tuple[dict[Path, dict[str, Any]], dict[str, int]]:
        manifests = self._manifests()
        q4 = manifests[Q4_MANIFEST]
        residual = manifests[RESIDUAL_MANIFEST]
        for key in ("poolside_commit", "model", "producer", "execution"):
            if q4.get(key) != residual.get(key):
                raise ContractError(f"manifests disagree on {key}")
        self._validate_producer(q4["producer"])
        model_identity = self._validate_model_fd(model_fd, q4["model"])
        return manifests, model_identity

    def _repo_path(self, relative: str, label: str) -> Path:
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ContractError(f"{label} path must be repository-relative")
        resolved = (self.repo_root / path).resolve()
        try:
            resolved.relative_to(self.repo_root)
        except ValueError as error:
            raise ContractError(f"{label} escapes repository root") from error
        return resolved

    def _validate_producer(self, producer: dict[str, Any]) -> None:
        if set(producer) != PRODUCER_KEYS:
            raise ContractError("producer must pin exactly the fixed C7 assets")
        for key in sorted(PRODUCER_KEYS):
            entry = producer.get(key)
            if not isinstance(entry, dict):
                raise ContractError(f"producer {key} is not an object")
            path = self._repo_path(str(entry.get("path", "")), f"producer {key}")
            try:
                payload = path.read_bytes()
            except OSError as error:
                raise ContractError(f"cannot read producer {key}: {error}") from error
            if len(payload) != entry.get("bytes"):
                raise ContractError(f"producer {key} byte count mismatch")
            if _sha256_bytes(payload) != entry.get("sha256"):
                raise ContractError(f"producer {key} SHA-256 mismatch")

        tokens = producer["tokens"]
        if tokens.get("format") != "little-endian-int32":
            raise ContractError("token format must be little-endian-int32")
        count = tokens.get("count")
        if not isinstance(count, int) or count <= 0:
            raise ContractError("token count must be positive")
        token_path = self._repo_path(tokens["path"], "producer tokens")
        token_bytes = token_path.read_bytes()
        if len(token_bytes) != count * 4:
            raise ContractError("token byte count does not match token count")
        ids = list(struct.unpack(f"<{count}i", token_bytes))
        if ids != tokens.get("ids"):
            raise ContractError("token IDs do not match token bytes")
        for key in ("prompt_sha256", "tokenizer_runtime_commit"):
            value = tokens.get(key)
            if not isinstance(value, str) or not value:
                raise ContractError(f"tokens must pin {key}")

    def _validate_model_fd(
        self, model_fd: int, model: dict[str, Any]
    ) -> dict[str, int]:
        # The shared advisory lock blocks cooperating writers while the shell's
        # inherited read-only descriptor remains open. Uncooperative writers
        # cannot be prevented here, so endpoint hashes plus kernel stat
        # continuity bound that case; a privileged writer can defeat either.
        try:
            access_mode = fcntl.fcntl(model_fd, fcntl.F_GETFL) & os.O_ACCMODE
            fcntl.flock(model_fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            before = os.fstat(model_fd)
        except OSError as error:
            raise ContractError(f"invalid held model fd {model_fd}: {error}") from error
        if access_mode != os.O_RDONLY:
            raise ContractError("held model fd must be read-only")
        if not stat.S_ISREG(before.st_mode):
            raise ContractError("held model fd must name a regular file")
        if before.st_size != model.get("bytes"):
            raise ContractError("held model fd byte count mismatch")
        if _sha256_fd(model_fd) != model.get("sha256"):
            raise ContractError("held model fd SHA-256 mismatch")
        try:
            after = os.fstat(model_fd)
        except OSError as error:
            raise ContractError(f"cannot restat held model fd: {error}") from error
        before_identity = self._model_identity(before)
        if self._model_identity(after) != before_identity:
            raise ContractError("held model changed during endpoint hash")
        return before_identity

    def _model_identity(self, identity: os.stat_result) -> dict[str, int]:
        return {
            "device": identity.st_dev,
            "inode": identity.st_ino,
            "size": identity.st_size,
            "mtime_ns": identity.st_mtime_ns,
            "ctime_ns": identity.st_ctime_ns,
        }

    def _write_model_continuity(
        self, continuity_fd: int, model_identity: dict[str, int]
    ) -> None:
        payload = json.dumps(
            model_identity, sort_keys=True, separators=(",", ":")
        ).encode("ascii")
        try:
            token = os.fstat(continuity_fd)
            if not stat.S_ISREG(token.st_mode):
                raise ContractError("model continuity fd must be a regular file")
            os.ftruncate(continuity_fd, 0)
            written = 0
            while written < len(payload):
                count = os.pwrite(continuity_fd, payload[written:], written)
                if count <= 0:
                    raise ContractError(
                        "model continuity token write made no progress"
                    )
                written += count
            os.fsync(continuity_fd)
        except ContractError:
            raise
        except OSError as error:
            raise ContractError(
                f"cannot write model continuity token: {error}"
            ) from error

    def _validate_model_continuity(
        self, continuity_fd: int, model_identity: dict[str, int]
    ) -> None:
        try:
            token = os.fstat(continuity_fd)
            if not stat.S_ISREG(token.st_mode) or not 1 <= token.st_size <= 4096:
                raise ContractError("model continuity token is invalid")
            payload = os.pread(continuity_fd, token.st_size, 0)
            expected = json.loads(payload)
        except ContractError:
            raise
        except (OSError, json.JSONDecodeError) as error:
            raise ContractError(
                f"cannot read model continuity token: {error}"
            ) from error
        if expected != model_identity:
            raise ContractError("model continuity changed between capture phases")

    def _git(
        self,
        poolside_src: Path,
        *arguments: str,
        env: dict[str, str] | None = None,
    ) -> str:
        environment = {
            **CLEAN_SUBPROCESS_ENV,
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_NO_REPLACE_OBJECTS": "1",
        }
        if env is not None:
            if set(env) != {"GIT_INDEX_FILE"}:
                raise ContractError("noncanonical Git subprocess environment")
            environment.update(env)
        return _run(
            [
                "/usr/bin/git",
                "-c",
                "core.fsmonitor=false",
                "-c",
                "core.untrackedCache=false",
                "-c",
                "core.preloadIndex=false",
                "-C",
                str(poolside_src),
                *arguments,
            ],
            env=environment,
        )

    def _patch_paths(self, patch: Path) -> set[str]:
        paths = set()
        try:
            for line in patch.read_text(encoding="utf-8").splitlines():
                if line.startswith("+++ b/"):
                    paths.add(line[6:])
        except OSError as error:
            raise ContractError(f"cannot read Poolside patch: {error}") from error
        if not paths:
            raise ContractError("Poolside patch has no tracked target")
        return paths

    def _validate_index_file_state(self, poolside_src: Path) -> None:
        index_name = self._git(
            poolside_src, "rev-parse", "--git-path", "index"
        ).strip()
        if not index_name:
            raise ContractError("cannot resolve Poolside index")
        index_path = Path(index_name)
        if not index_path.is_absolute():
            index_path = poolside_src / index_path
        try:
            before = index_path.lstat()
            if not stat.S_ISREG(before.st_mode):
                raise ContractError("Poolside index is not a regular file")
            payload = index_path.read_bytes()
            after = index_path.lstat()
        except ContractError:
            raise
        except OSError as error:
            raise ContractError(f"cannot read Poolside index: {error}") from error
        if self._model_identity(after) != self._model_identity(before):
            raise ContractError("Poolside index changed during inspection")
        if len(payload) < 32 or hashlib.sha1(payload[:-20]).digest() != payload[-20:]:
            raise ContractError("Poolside index checksum mismatch")
        try:
            signature, version, entry_count = struct.unpack_from(
                ">4sII", payload, 0
            )
        except struct.error as error:
            raise ContractError("cannot parse Poolside index header") from error
        if signature != b"DIRC" or version not in {2, 3}:
            raise ContractError("Poolside index format is noncanonical")

        offset = 12
        content_end = len(payload) - 20
        for _ in range(entry_count):
            entry_start = offset
            if entry_start + 62 > content_end:
                raise ContractError("Poolside index entry is truncated")
            flags = struct.unpack_from(">H", payload, entry_start + 60)[0]
            if flags & (0x8000 | 0x4000 | 0x3000):
                raise ContractError(
                    "Poolside source has a hidden index flag "
                    "or noncanonical entry"
                )
            name_length = flags & 0x0FFF
            name_start = entry_start + 62
            if name_length < 0x0FFF:
                name_end = name_start + name_length
                if name_end >= content_end or payload[name_end] != 0:
                    raise ContractError("Poolside index path is malformed")
            else:
                name_end = payload.find(b"\0", name_start, content_end)
                if name_end < 0:
                    raise ContractError("Poolside index path is unterminated")
            unpadded_end = name_end + 1
            offset = entry_start + (
                (unpadded_end - entry_start + 7) // 8 * 8
            )
            if offset > content_end or any(payload[unpadded_end:offset]):
                raise ContractError("Poolside index padding is malformed")

        permitted_extensions = {b"TREE", b"EOIE", b"IEOT"}
        while offset < content_end:
            if offset + 8 > content_end:
                raise ContractError("Poolside index extension is truncated")
            extension, size = struct.unpack_from(">4sI", payload, offset)
            offset += 8
            if extension not in permitted_extensions:
                raise ContractError(
                    "Poolside source has a hidden index flag "
                    "or noncanonical entry"
                )
            offset += size
            if offset > content_end:
                raise ContractError("Poolside index extension is malformed")
        if offset != content_end:
            raise ContractError("Poolside index has trailing data")

    def _tree_entries(self, poolside_src: Path) -> dict[str, tuple[str, str]]:
        object_format = self._git(
            poolside_src, "rev-parse", "--show-object-format"
        ).strip()
        if object_format != "sha1":
            raise ContractError("Poolside repository must use SHA-1 objects")
        output = self._git(
            poolside_src,
            "ls-tree",
            "-r",
            "-z",
            "--full-tree",
            "HEAD",
        )
        entries: dict[str, tuple[str, str]] = {}
        for record in output.split("\0"):
            if not record:
                continue
            try:
                metadata, path = record.split("\t", 1)
                mode, object_type, object_id = metadata.split()
            except ValueError as error:
                raise ContractError("cannot parse Poolside HEAD tree") from error
            if (
                object_type != "blob"
                or mode not in {"100644", "100755", "120000"}
                or len(object_id) != 40
                or any(character not in "0123456789abcdef" for character in object_id)
                or path in entries
            ):
                raise ContractError("Poolside HEAD tree has an unsupported entry")
            entries[path] = (mode, object_id)
        if not entries:
            raise ContractError("Poolside HEAD tree is empty")
        return entries

    def _index_entries(
        self,
        poolside_src: Path,
        *,
        env: dict[str, str] | None = None,
    ) -> dict[str, tuple[str, str]]:
        output = self._git(
            poolside_src, "ls-files", "--stage", "-z", env=env
        )
        entries: dict[str, tuple[str, str]] = {}
        for record in output.split("\0"):
            if not record:
                continue
            try:
                metadata, path = record.split("\t", 1)
                mode, object_id, stage = metadata.split()
            except ValueError as error:
                raise ContractError("cannot parse Poolside index") from error
            if (
                stage != "0"
                or mode not in {"100644", "100755", "120000"}
                or len(object_id) != 40
                or any(character not in "0123456789abcdef" for character in object_id)
                or path in entries
            ):
                raise ContractError("Poolside index has an unsupported entry")
            entries[path] = (mode, object_id)
        return entries

    def _patched_entries(
        self,
        poolside_src: Path,
        patch: Path,
        head_entries: dict[str, tuple[str, str]],
    ) -> dict[str, tuple[str, str]]:
        with tempfile.TemporaryDirectory() as temporary_directory:
            environment = {
                "GIT_INDEX_FILE": str(Path(temporary_directory) / "index")
            }
            self._git(poolside_src, "read-tree", "HEAD", env=environment)
            self._git(
                poolside_src,
                "apply",
                "--cached",
                str(patch),
                env=environment,
            )
            patched_entries = self._index_entries(
                poolside_src, env=environment
            )
        changed = {
            path
            for path in set(head_entries) | set(patched_entries)
            if head_entries.get(path) != patched_entries.get(path)
        }
        if changed != self._patch_paths(patch):
            raise ContractError("pinned patch changes unexpected paths")
        return patched_entries

    def _raw_blob_id(self, path: Path, mode: str) -> str:
        try:
            before = path.lstat()
            if mode == "120000":
                if not stat.S_ISLNK(before.st_mode):
                    raise ContractError("tracked symlink type mismatch")
                payload = os.fsencode(os.readlink(path))
                after = path.lstat()
                if self._model_identity(after) != self._model_identity(before):
                    raise ContractError("tracked symlink changed during hash")
                digest = hashlib.sha1()
                digest.update(f"blob {len(payload)}\0".encode("ascii"))
                digest.update(payload)
                return digest.hexdigest()

            if not stat.S_ISREG(before.st_mode):
                raise ContractError("tracked regular-file type mismatch")
            executable = bool(before.st_mode & 0o111)
            if executable != (mode == "100755"):
                raise ContractError("tracked executable mode mismatch")
            descriptor = os.open(
                path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            )
            try:
                opened = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or (opened.st_dev, opened.st_ino)
                    != (before.st_dev, before.st_ino)
                ):
                    raise ContractError("tracked file changed before hash")
                digest = hashlib.sha1()
                digest.update(f"blob {opened.st_size}\0".encode("ascii"))
                while True:
                    chunk = os.read(descriptor, 8 * 1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
                after = os.fstat(descriptor)
                if self._model_identity(after) != self._model_identity(opened):
                    raise ContractError("tracked file changed during hash")
                path_after = path.lstat()
                if (
                    not stat.S_ISREG(path_after.st_mode)
                    or (path_after.st_dev, path_after.st_ino)
                    != (after.st_dev, after.st_ino)
                    or self._model_identity(path_after)
                    != self._model_identity(after)
                ):
                    raise ContractError("tracked file path changed during hash")
                return digest.hexdigest()
            finally:
                os.close(descriptor)
        except ContractError:
            raise
        except OSError as error:
            raise ContractError(f"cannot hash tracked file {path}: {error}") from error

    def _validate_raw_worktree(
        self,
        poolside_src: Path,
        expected_entries: dict[str, tuple[str, str]],
        *,
        patched: bool,
    ) -> None:
        try:
            source_identity = poolside_src.lstat()
        except OSError as error:
            raise ContractError(f"cannot inspect Poolside source root: {error}") from error
        if not stat.S_ISDIR(source_identity.st_mode):
            raise ContractError("Poolside source root is not a directory")
        checked_directories = {
            poolside_src: self._model_identity(source_identity)
        }
        for relative_name, (mode, expected_id) in expected_entries.items():
            relative = Path(relative_name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ContractError("Poolside HEAD tree path is unsafe")
            parent = poolside_src
            for component in relative.parts[:-1]:
                parent /= component
                if parent in checked_directories:
                    continue
                try:
                    identity = parent.lstat()
                except OSError as error:
                    raise ContractError(
                        f"cannot inspect tracked directory {parent}: {error}"
                    ) from error
                if not stat.S_ISDIR(identity.st_mode):
                    raise ContractError("tracked path has a non-directory ancestor")
                checked_directories[parent] = self._model_identity(identity)
            actual_id = self._raw_blob_id(poolside_src / relative, mode)
            if actual_id != expected_id:
                phase = (
                    "does not equal the pinned patch"
                    if patched
                    else "differs from the pinned commit"
                )
                raise ContractError(f"Poolside raw tracked tree {phase}")
        for directory, expected_identity in checked_directories.items():
            try:
                identity = directory.lstat()
            except OSError as error:
                raise ContractError(
                    f"cannot recheck tracked directory {directory}: {error}"
                ) from error
            if (
                not stat.S_ISDIR(identity.st_mode)
                or self._model_identity(identity) != expected_identity
            ):
                raise ContractError("tracked directory changed during raw tree hash")

    def _validate_source(
        self,
        poolside_src: Path,
        poolside_build: Path,
        manifests: dict[Path, dict[str, Any]],
        *,
        patched: bool,
    ) -> None:
        poolside_src = poolside_src.resolve()
        poolside_build = poolside_build.resolve()
        repository_root = Path(
            self._git(poolside_src, "rev-parse", "--show-toplevel").strip()
        ).resolve()
        if repository_root != poolside_src:
            raise ContractError("Poolside source is not the repository root")
        if self._git(poolside_src, "replace", "-l").strip():
            raise ContractError("Poolside source has replacement refs")
        expected_commit = manifests[Q4_MANIFEST]["poolside_commit"]
        if self._git(poolside_src, "rev-parse", "HEAD").strip() != expected_commit:
            raise ContractError("Poolside source commit mismatch")
        patch_entry = manifests[Q4_MANIFEST]["producer"]["poolside_patch"]
        patch = self._repo_path(patch_entry["path"], "Poolside patch")
        self._validate_index_file_state(poolside_src)
        for flag_view in ("-v", "-f"):
            index_entries = self._git(
                poolside_src, "ls-files", flag_view, "-z"
            ).split("\0")
            hidden_entries = [
                entry
                for entry in index_entries
                if entry and entry[0] != "H"
            ]
            if hidden_entries:
                raise ContractError(
                    "Poolside source has a hidden index flag "
                    "or noncanonical entry"
                )
        head_entries = self._tree_entries(poolside_src)
        index_entries = self._index_entries(poolside_src)
        if index_entries != head_entries:
            detail = (
                "changes beyond the pinned patch"
                if patched
                else "an index differing from the pinned commit"
            )
            raise ContractError(f"Poolside source has {detail}")
        try:
            build_relative = poolside_build.relative_to(poolside_src)
        except ValueError:
            build_prefix = None
        else:
            if build_relative == Path("."):
                raise ContractError("Poolside build must differ from source root")
            build_prefix = build_relative.as_posix().rstrip("/") + "/"
        untracked = [
            entry
            for entry in self._git(
                poolside_src,
                "ls-files",
                "--others",
                "--directory",
                "--no-empty-directory",
                "-z",
            ).split("\0")
            if entry
        ]
        unexpected_untracked = [
            entry
            for entry in untracked
            if build_prefix is None
            or not (
                entry == build_prefix.rstrip("/")
                or entry.startswith(build_prefix)
            )
        ]
        if unexpected_untracked:
            raise ContractError("Poolside source has untracked files")
        if not patched:
            self._git(poolside_src, "apply", "--check", str(patch))
            self._validate_raw_worktree(
                poolside_src, head_entries, patched=False
            )
            return

        self._git(poolside_src, "apply", "--check", "-R", str(patch))
        patched_entries = self._patched_entries(
            poolside_src, patch, head_entries
        )
        self._validate_raw_worktree(
            poolside_src, patched_entries, patched=True
        )

    def _validate_build(
        self,
        poolside_src: Path,
        poolside_build: Path,
        manifests: dict[Path, dict[str, Any]],
    ) -> None:
        execution = manifests[Q4_MANIFEST]["execution"]
        expected = execution["poolside_build"]
        toolchain = execution.get("toolchain")
        self._validate_toolchain(toolchain)
        cache = _cache_values(poolside_build / "CMakeCache.txt")
        home = cache.get("CMAKE_HOME_DIRECTORY")
        if not home or Path(home).resolve() != poolside_src.resolve():
            raise ContractError("CMAKE_HOME_DIRECTORY does not match Poolside source")
        if cache.get("CMAKE_BUILD_TYPE") != expected.get("cmake_build_type"):
            raise ContractError("CMake build type mismatch")
        cache_tools = {
            "CMAKE_COMMAND": "cmake",
            "CMAKE_MAKE_PROGRAM": "make",
            "CMAKE_CXX_COMPILER": "cxx",
            "CMAKE_CUDA_COMPILER": "cuda",
        }
        for cache_key, tool_name in cache_tools.items():
            if cache.get(cache_key) != toolchain[tool_name]["path"]:
                raise ContractError(f"CMake tool path {cache_key} mismatch")
        if cache.get("CMAKE_GENERATOR") != "Unix Makefiles":
            raise ContractError("CMake generator mismatch")
        if execution.get("probe_build", {}).get("compiler") != toolchain["cxx"][
            "path"
        ]:
            raise ContractError("probe compiler does not match pinned C++ tool")
        bool_keys = {
            "ggml_cuda": "GGML_CUDA",
            "ggml_cuda_fa_all_quants": "GGML_CUDA_FA_ALL_QUANTS",
            "ggml_cuda_force_cublas": "GGML_CUDA_FORCE_CUBLAS",
        }
        for manifest_key, cache_key in bool_keys.items():
            if _bool_cache(cache.get(cache_key), cache_key) != expected.get(
                manifest_key
            ):
                raise ContractError(f"CMake option {cache_key} mismatch")
        cxx_flags = shlex.split(cache.get("CMAKE_CXX_FLAGS", "")) + shlex.split(
            cache.get("CMAKE_CXX_FLAGS_RELEASE", "")
        )
        if cxx_flags != expected.get("cxx_flags"):
            raise ContractError("Poolside CXX flags mismatch")
        flags_path = (
            poolside_build
            / "ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/flags.make"
        )
        try:
            lines = flags_path.read_text(encoding="utf-8").splitlines()
        except OSError as error:
            raise ContractError(f"cannot read Poolside CUDA flags: {error}") from error
        values = [
            line.split("=", 1)[1].strip()
            for line in lines
            if line.startswith("CUDA_FLAGS =")
        ]
        if len(values) != 1 or shlex.split(values[0]) != expected.get("cuda_flags"):
            raise ContractError("Poolside CUDA flags mismatch")
        self._validate_generated_recipes(
            poolside_src,
            poolside_build,
            execution.get("generated_recipes"),
        )

    def _query_toolchain(self, name: str, path: Path) -> dict[str, Any]:
        if not path.is_file() or not os.access(path, os.X_OK):
            raise ContractError(f"toolchain {name} executable is unavailable")
        environment = CLEAN_SUBPROCESS_ENV
        common = {
            "path": str(path),
            "realpath": str(path.resolve()),
            "sha256": _sha256_path(path),
        }
        if name == "cmake":
            first = _run([str(path), "--version"], env=environment).splitlines()[0]
            prefix = "cmake version "
            if not first.startswith(prefix):
                raise ContractError("cannot parse cmake version")
            return {**common, "version": first.removeprefix(prefix)}
        if name == "make":
            first = _run([str(path), "--version"], env=environment).splitlines()[0]
            prefix = "GNU Make "
            if not first.startswith(prefix):
                raise ContractError("cannot parse make version")
            return {**common, "version": first.removeprefix(prefix)}
        if name == "cxx":
            version = _run(
                [str(path), "-dumpfullversion"], env=environment
            ).strip()
            target = _run([str(path), "-dumpmachine"], env=environment).strip()
            if not version or not target:
                raise ContractError("cannot parse C++ compiler identity")
            return {**common, "version": version, "target": target}
        if name == "cuda":
            lines = _run([str(path), "--version"], env=environment).splitlines()
            version_line = next(
                (
                    line.strip()
                    for line in lines
                    if line.strip().startswith("Cuda compilation tools, release ")
                ),
                "",
            )
            build_line = next(
                (line.strip() for line in lines if line.strip().startswith("Build ")),
                "",
            )
            prefix = "Cuda compilation tools, release "
            if not version_line or ", V" not in version_line or not build_line:
                raise ContractError("cannot parse CUDA compiler identity")
            release, version = version_line.removeprefix(prefix).split(", V", 1)
            return {
                **common,
                "version": version,
                "release": release,
                "build": build_line.removeprefix("Build "),
            }
        raise ContractError(f"unknown toolchain member: {name}")

    def _validate_toolchain(self, expected: Any) -> None:
        tool_names = {"cmake", "cxx", "cuda", "make"}
        if not isinstance(expected, dict) or set(expected) != tool_names:
            raise ContractError("toolchain must pin cmake, cxx, cuda, and make")
        for name in sorted(tool_names):
            contract = expected[name]
            if not isinstance(contract, dict) or not isinstance(
                contract.get("path"), str
            ):
                raise ContractError(f"toolchain {name} contract is invalid")
            try:
                observed = self.toolchain_query(name, Path(contract["path"]))
            except ContractError:
                raise
            except Exception as error:
                raise ContractError(
                    f"toolchain {name} query failed: {error}"
                ) from error
            if observed != contract:
                raise ContractError(f"toolchain {name} identity mismatch")

    def _validate_generated_recipes(
        self,
        poolside_src: Path,
        poolside_build: Path,
        recipes: Any,
    ) -> None:
        if not isinstance(recipes, dict) or not recipes:
            raise ContractError("generated recipe closure must not be empty")
        source_marker = str(poolside_src.resolve()).encode()
        build_marker = str(poolside_build.resolve()).encode()
        for relative, contract in recipes.items():
            relative_path = Path(relative)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ContractError("generated recipe path is unsafe")
            path = (poolside_build / relative_path).resolve()
            try:
                path.relative_to(poolside_build.resolve())
                payload = path.read_bytes()
            except (OSError, ValueError) as error:
                raise ContractError(
                    f"cannot read generated recipe {relative}: {error}"
                ) from error
            normalized = payload.replace(build_marker, b"$BUILD").replace(
                source_marker, b"$SOURCE"
            )
            expected_contract = {
                "normalized_bytes": len(normalized),
                "normalized_sha256": _sha256_bytes(normalized),
            }
            if contract != expected_contract:
                raise ContractError(f"generated recipe {relative} mismatch")

    def _query_device(self) -> dict[str, Any]:
        output = _run(
            [
                "/usr/bin/nvidia-smi",
                "--query-gpu=name,compute_cap",
                "--format=csv,noheader,nounits",
            ]
        )
        rows = [row.strip() for row in output.splitlines() if row.strip()]
        if len(rows) != 1 or "," not in rows[0]:
            raise ContractError("expected exactly one CUDA device")
        name, capability = (part.strip() for part in rows[0].split(",", 1))
        known_multiprocessors = {("NVIDIA GB10", "12.1"): 48}
        key = (name, capability)
        if key not in known_multiprocessors:
            raise ContractError("unsupported CUDA device identity")
        return {
            "name": name,
            "compute_capability": capability,
            "multiprocessors": known_multiprocessors[key],
        }

    def _validate_device(self, manifests: dict[Path, dict[str, Any]]) -> None:
        expected = manifests[Q4_MANIFEST]["execution"]["device"]
        try:
            actual = self.device_query()
        except ContractError:
            raise
        except Exception as error:
            raise ContractError(f"device query failed: {error}") from error
        if actual != expected:
            raise ContractError(f"device identity mismatch: {actual!r}")

    def _query_compute_processes(self) -> list[int]:
        output = _run(
            [
                "/usr/bin/nvidia-smi",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ]
        )
        processes = []
        for row in output.splitlines():
            value = row.strip()
            if not value:
                continue
            if not value.isascii() or not value.isdecimal() or int(value) <= 0:
                raise ContractError(f"malformed GPU compute-process PID: {value!r}")
            processes.append(int(value))
        return sorted(set(processes))

    def _validate_gpu_exclusive(self) -> None:
        try:
            active = self.compute_process_query()
            if not isinstance(active, list) or any(
                not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0
                for pid in active
            ):
                raise ContractError("GPU compute-process query returned invalid PIDs")
        except Exception as error:
            raise ContractError(
                f"GPU compute-process query failed: {error}"
            ) from error
        if active:
            raise ContractError(f"GPU is not exclusive; active compute PIDs: {active}")

    def _validate_probe_binary(
        self, probe_bin: Path, manifests: dict[Path, dict[str, Any]]
    ) -> None:
        expected = manifests[Q4_MANIFEST]["execution"]["probe_build"]
        try:
            size = probe_bin.stat().st_size
        except OSError as error:
            raise ContractError(f"cannot stat probe binary: {error}") from error
        expected_size = expected.get("binary_bytes")
        if not isinstance(expected_size, int) or size != expected_size:
            raise ContractError("probe binary byte count mismatch")
        if _sha256_path(probe_bin) != expected.get("binary_sha256"):
            raise ContractError("probe binary SHA-256 mismatch")
        if expected.get("runtime_library_path") != "poolside_build/bin":
            raise ContractError("probe runtime library path contract mismatch")
        if expected.get("elf_runpath") != "none":
            raise ContractError("probe ELF RUNPATH contract mismatch")
        dynamic_tags = _elf64_dynamic_tags(probe_bin)
        if dynamic_tags & {15, 29}:
            raise ContractError("probe ELF contains RPATH or RUNPATH")

    def _capture_path(
        self, capture_root: Path, directory: str, source: str, label: str
    ) -> Path:
        for value in (directory, source):
            path = Path(value)
            if path.is_absolute() or ".." in path.parts:
                raise ContractError(f"{label} has unsafe capture path")
        root = capture_root.resolve()
        path = (root / directory / source).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ContractError(f"{label} escapes capture root") from error
        return path

    def _fixture_directory(self, relative_manifest: Path) -> Path:
        return (self.repo_root / relative_manifest).parent

    def _validate_captures(
        self,
        capture_root: Path,
        manifests: dict[Path, dict[str, Any]],
    ) -> None:
        for relative, manifest in manifests.items():
            extractions = manifest.get("extractions")
            files = manifest.get("files")
            if not isinstance(extractions, dict) or set(extractions) != set(files):
                raise ContractError(f"extractions and files disagree in {relative}")
            fixture_dir = self._fixture_directory(relative)
            for name, extraction in extractions.items():
                if extraction.get("producer") != "probe":
                    raise ContractError(f"extraction {name} has unknown producer")
                source = self._capture_path(
                    capture_root,
                    extraction.get("capture_directory", ""),
                    extraction.get("source", ""),
                    f"extraction {name}",
                )
                try:
                    source_payload = source.read_bytes()
                except OSError as error:
                    raise ContractError(
                        f"cannot read source {source.name} for {name}: {error}"
                    ) from error
                if len(source_payload) != extraction.get("source_bytes"):
                    raise ContractError(f"source {source.name} byte count mismatch")
                if _sha256_bytes(source_payload) != extraction.get("source_sha256"):
                    raise ContractError(f"source {source.name} SHA-256 mismatch")
                offset = extraction.get("offset")
                size = extraction.get("bytes")
                if (
                    not isinstance(offset, int)
                    or not isinstance(size, int)
                    or offset < 0
                    or size < 0
                    or offset + size > len(source_payload)
                ):
                    raise ContractError(f"extraction {name} has invalid byte range")
                fixture_path = fixture_dir / name
                try:
                    fixture_payload = fixture_path.read_bytes()
                except OSError as error:
                    raise ContractError(
                        f"cannot read fixture {name}: {error}"
                    ) from error
                file_contract = files[name]
                if len(fixture_payload) != file_contract.get("bytes"):
                    raise ContractError(f"fixture {name} byte count mismatch")
                if _sha256_bytes(fixture_payload) != file_contract.get("sha256"):
                    raise ContractError(f"fixture {name} SHA-256 mismatch")
                if size != len(fixture_payload):
                    raise ContractError(
                        f"extraction {name} size does not match fixture"
                    )
                if source_payload[offset : offset + size] != fixture_payload:
                    raise ContractError(f"extraction {name} does not match fixture")
            self._validate_non_perturbation(capture_root, relative, manifest)

    def _validate_non_perturbation(
        self,
        capture_root: Path,
        relative: Path,
        manifest: dict[str, Any],
    ) -> None:
        capture = manifest.get("capture", {})
        checks = capture.get("non_perturbation", {})
        if not isinstance(checks, dict):
            raise ContractError("non-perturbation record must be an object")
        if checks.get(CONSOLIDATED_CAPTURE_KEY) is not True:
            raise ContractError(
                f"{relative.stem} non-perturbation consolidation is not pinned"
            )

        if relative == Q4_MANIFEST:
            modes = {
                name: value
                for name, value in capture.items()
                if isinstance(value, dict) and "capture_directory" in value
            }
            expected_records = set(modes) | {CONSOLIDATED_CAPTURE_KEY}
            if not modes or set(checks) != expected_records:
                raise ContractError(
                    "q4 non-perturbation capture-mode set mismatch"
                )
            for mode, mode_capture in modes.items():
                mode_checks = checks.get(mode)
                if (
                    not isinstance(mode_checks, dict)
                    or set(mode_checks) != set(NON_PERTURBATION_FILES)
                ):
                    raise ContractError(
                        f"non-perturbation {mode} check set mismatch"
                    )
                self._validate_non_perturbation_hashes(
                    capture_root,
                    str(mode_capture["capture_directory"]),
                    mode_checks,
                    f"non-perturbation {mode}",
                )
            return

        expected_records = RESIDUAL_NON_PERTURBATION_KEYS | {
            CONSOLIDATED_CAPTURE_KEY
        }
        if set(checks) != expected_records:
            raise ContractError("residual non-perturbation check set mismatch")
        self._validate_non_perturbation_hashes(
            capture_root,
            str(capture.get("capture_directory", "")),
            checks,
            "residual non-perturbation",
        )

    def _validate_non_perturbation_hashes(
        self,
        capture_root: Path,
        directory: str,
        checks: dict[str, Any],
        label: str,
    ) -> None:
        for key in sorted(set(checks) - {CONSOLIDATED_CAPTURE_KEY}):
            filename = NON_PERTURBATION_FILES[key]
            path = self._capture_path(
                capture_root, directory, filename, f"{label} {key}"
            )
            if _sha256_path(path) != checks[key]:
                raise ContractError(f"{label} {key} mismatch")

    def _validate_residual_semantics(self, manifest: dict[str, Any]) -> None:
        fixture_dir = self._fixture_directory(RESIDUAL_MANIFEST)
        names = (
            "residual-token0.f32",
            "moe-token0.f32",
            "shared-token0.f32",
            "expected-token0.f32",
        )
        payloads = [(fixture_dir / name).read_bytes() for name in names]
        if len({len(payload) for payload in payloads}) != 1 or len(payloads[0]) % 4:
            raise ContractError("residual fixtures must have equal float32 sizes")
        count = len(payloads[0]) // 4
        residual, moe, shared, expected = (
            struct.unpack(f"<{count}f", payload) for payload in payloads
        )
        intended_mismatches = 0
        legacy_mismatches = 0
        legacy_first = -1
        for index, values in enumerate(zip(residual, moe, shared, expected)):
            residual_value, moe_value, shared_value, expected_value = values
            intended = _float32_add(
                _float32_add(moe_value, shared_value), residual_value
            )
            legacy = _float32_add(
                _float32_add(residual_value, moe_value), shared_value
            )
            expected_bits = struct.pack("<f", expected_value)
            if struct.pack("<f", intended) != expected_bits:
                intended_mismatches += 1
            if struct.pack("<f", legacy) != expected_bits:
                if legacy_first < 0:
                    legacy_first = index
                legacy_mismatches += 1
        oracle = manifest.get("oracle", {})
        expected_oracle = {
            "expression": "(moe + shared) + residual",
            "expected_exact": True,
            "intended_order_mismatches": intended_mismatches,
            "legacy_order": "(residual + moe) + shared",
            "legacy_order_mismatches": legacy_mismatches,
            "legacy_first_mismatch": legacy_first,
        }
        if oracle != expected_oracle:
            raise ContractError("residual association oracle mismatch")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="phase", required=True)
    for phase in ("preflight", "captured"):
        command = phases.add_parser(phase)
        command.add_argument("--poolside-src", required=True)
        command.add_argument("--poolside-build", required=True)
        command.add_argument("--model-fd", required=True, type=int)
        command.add_argument("--continuity-fd", required=True, type=int)
        if phase == "captured":
            command.add_argument("--capture-root", required=True)
            command.add_argument("--probe-bin", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        repo_root = discover_repo_root(Path(__file__))
        verifier = Verifier(repo_root)
        if args.phase == "preflight":
            verifier.preflight(
                Path(args.poolside_src),
                Path(args.poolside_build),
                args.model_fd,
                continuity_fd=args.continuity_fd,
            )
        else:
            verifier.captured(
                Path(args.poolside_src),
                Path(args.poolside_build),
                args.model_fd,
                Path(args.capture_root),
                Path(args.probe_bin),
                continuity_fd=args.continuity_fd,
            )
    except ContractError as error:
        print(f"laguna-c7-provenance: {error}", file=sys.stderr)
        return 1
    print(f"laguna-c7-provenance: {args.phase} PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
