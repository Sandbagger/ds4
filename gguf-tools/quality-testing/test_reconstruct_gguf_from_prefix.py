#!/usr/bin/env python3
"""Contracts for fail-closed GGUF prefix/tensor-payload reconstruction."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
QUALITY_DIR = ROOT / "gguf-tools/quality-testing"
TOOL = QUALITY_DIR / "reconstruct_gguf_from_prefix.py"
ALIGNMENT = 32


def _gguf_string(value: str) -> bytes:
    payload = value.encode("utf-8")
    return struct.pack("<Q", len(payload)) + payload


def _metadata_string(key: str, value: str) -> bytes:
    return _gguf_string(key) + struct.pack("<I", 8) + _gguf_string(value)


def _metadata_u32(key: str, value: int) -> bytes:
    return _gguf_string(key) + struct.pack("<II", 4, value)


def _tensor_info(
    name: str,
    dimensions: tuple[int, ...],
    ggml_type: int,
    relative_offset: int,
) -> bytes:
    return (
        _gguf_string(name)
        + struct.pack("<I", len(dimensions))
        + b"".join(struct.pack("<Q", value) for value in dimensions)
        + struct.pack("<IQ", ggml_type, relative_offset)
    )


def _align(value: int) -> int:
    return (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


TENSORS = (
    ("token_embd.weight", (4,), 0, bytes(range(16))),
    ("blk.0.ffn_gate_exps.weight", (256,), 12, bytes(range(144))),
    ("output_norm.weight", (4,), 30, b"\x11\x22" * 4),
)


def _artifact_bytes(
    *,
    chat_template: str,
    rename_first_tensor: bool = False,
    trailing_bytes: bytes = b"",
) -> tuple[bytes, int]:
    metadata = (
        _metadata_u32("general.alignment", ALIGNMENT),
        _metadata_string("general.name", "synthetic Laguna"),
        _metadata_string("tokenizer.chat_template", chat_template),
    )
    relative_offset = 0
    tensor_directory: list[bytes] = []
    payloads: list[bytes] = []
    for index, (name, dimensions, ggml_type, payload) in enumerate(TENSORS):
        if index == 0 and rename_first_tensor:
            name = "renamed_embd.weight"
        tensor_directory.append(
            _tensor_info(name, dimensions, ggml_type, relative_offset)
        )
        payloads.append(payload)
        relative_offset = _align(relative_offset + len(payload))

    header = struct.pack("<4sIQQ", b"GGUF", 3, len(TENSORS), len(metadata))
    output = bytearray(header + b"".join(metadata) + b"".join(tensor_directory))
    data_offset = _align(len(output))
    output.extend(b"\0" * (data_offset - len(output)))
    for payload in payloads:
        output.extend(payload)
        output.extend(b"\0" * (_align(len(output)) - len(output)))
    output.extend(trailing_bytes)
    return bytes(output), data_offset


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_tool_module():
    sys.path.insert(0, str(QUALITY_DIR))
    try:
        spec = importlib.util.spec_from_file_location("reconstruct_test_module", TOOL)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


class ReconstructGgufFromPrefixTests(unittest.TestCase):
    def _run(
        self,
        old: Path,
        prefix: Path,
        output: Path,
        *,
        old_bytes: bytes,
        new_bytes: bytes,
        old_sha256: str | None = None,
        new_sha256: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(TOOL),
                "--chunk-bytes",
                "17",
                "--old-size",
                str(len(old_bytes)),
                "--old-sha256",
                old_sha256 or _sha256(old_bytes),
                "--new-size",
                str(len(new_bytes)),
                "--new-sha256",
                new_sha256 or _sha256(new_bytes),
                str(old),
                str(prefix),
                str(output),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_reconstructs_exact_artifact_from_corrected_prefix_and_old_tail(self) -> None:
        old_bytes, old_data_offset = _artifact_bytes(chat_template="old")
        new_bytes, new_data_offset = _artifact_bytes(
            chat_template="old" + "x" * 416
        )
        self.assertEqual(new_data_offset - old_data_offset, 416)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / "old.gguf"
            prefix = root / "corrected-prefix.bin"
            output = root / "corrected.gguf"
            old.write_bytes(old_bytes)
            prefix.write_bytes(new_bytes[:new_data_offset])

            completed = self._run(
                old,
                prefix,
                output,
                old_bytes=old_bytes,
                new_bytes=new_bytes,
            )

            self.assertEqual(
                completed.returncode,
                0,
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
            )
            self.assertEqual(output.read_bytes(), new_bytes)
            self.assertEqual(list(root.glob(f".{output.name}.*.tmp")), [])

        report = json.loads(completed.stdout)
        self.assertEqual(report["schema"], "ds4.gguf-prefix-reconstruction/v1")
        self.assertEqual(int(report["old"]["tensor_data_offset"]), old_data_offset)
        self.assertEqual(
            int(report["new_prefix"]["tensor_data_offset"]), new_data_offset
        )
        self.assertEqual(int(report["copied"]["offset_delta"]), 416)
        self.assertEqual(report["output"]["sha256"], _sha256(new_bytes))

    def test_bad_inputs_and_final_digest_leave_no_output_or_temp(self) -> None:
        old_bytes, _ = _artifact_bytes(chat_template="old")
        new_bytes, new_data_offset = _artifact_bytes(
            chat_template="old" + "x" * 416
        )
        wrong_layout, wrong_layout_offset = _artifact_bytes(
            chat_template="old" + "x" * 416,
            rename_first_tensor=True,
        )
        wrong_size_equation, wrong_size_offset = _artifact_bytes(
            chat_template="old" + "x" * 416,
            trailing_bytes=b"x",
        )

        cases = (
            (
                "truncated-prefix",
                new_bytes[: new_data_offset - 1],
                new_bytes,
                None,
                None,
                "tensor data begins beyond",
            ),
            (
                "mismatched-directory",
                wrong_layout[:wrong_layout_offset],
                wrong_layout,
                None,
                None,
                "tensor layouts differ",
            ),
            (
                "wrong-size-equation",
                wrong_size_equation[:wrong_size_offset],
                wrong_size_equation,
                None,
                None,
                "suffix size equation failed",
            ),
            (
                "wrong-old-digest",
                new_bytes[:new_data_offset],
                new_bytes,
                "0" * 64,
                None,
                "old artifact SHA-256 mismatch",
            ),
            (
                "wrong-final-digest",
                new_bytes[:new_data_offset],
                new_bytes,
                None,
                "f" * 64,
                "reconstructed SHA-256 mismatch",
            ),
        )
        for (
            name,
            prefix_bytes,
            expected_new,
            old_digest,
            new_digest,
            error,
        ) in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                old = root / "old.gguf"
                prefix = root / "corrected-prefix.bin"
                output = root / "corrected.gguf"
                old.write_bytes(old_bytes)
                prefix.write_bytes(prefix_bytes)

                completed = self._run(
                    old,
                    prefix,
                    output,
                    old_bytes=old_bytes,
                    new_bytes=expected_new,
                    old_sha256=old_digest,
                    new_sha256=new_digest,
                )

                self.assertEqual(completed.returncode, 2, completed.stdout)
                self.assertEqual(completed.stdout, "")
                self.assertIn(error, completed.stderr)
                self.assertFalse(output.exists())
                self.assertEqual(list(root.glob(f".{output.name}.*.tmp")), [])

    def test_prefix_may_safely_extend_past_tensor_data_offset(self) -> None:
        old_bytes, _ = _artifact_bytes(chat_template="old")
        new_bytes, new_data_offset = _artifact_bytes(
            chat_template="old" + "x" * 416
        )
        extra_prefix_bytes = 13
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / "old.gguf"
            prefix = root / "corrected-prefix.bin"
            output = root / "corrected.gguf"
            old.write_bytes(old_bytes)
            prefix.write_bytes(new_bytes[: new_data_offset + extra_prefix_bytes])

            completed = self._run(
                old,
                prefix,
                output,
                old_bytes=old_bytes,
                new_bytes=new_bytes,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(output.read_bytes(), new_bytes)
            report = json.loads(completed.stdout)
            self.assertEqual(
                int(report["new_prefix"]["size_bytes"]),
                new_data_offset + extra_prefix_bytes,
            )
            self.assertEqual(
                int(report["copied"]["prefix_bytes"]), new_data_offset
            )

    def test_existing_output_is_never_replaced(self) -> None:
        old_bytes, _ = _artifact_bytes(chat_template="old")
        new_bytes, new_data_offset = _artifact_bytes(
            chat_template="old" + "x" * 416
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / "old.gguf"
            prefix = root / "corrected-prefix.bin"
            output = root / "corrected.gguf"
            old.write_bytes(old_bytes)
            prefix.write_bytes(new_bytes[:new_data_offset])
            output.write_bytes(b"do not replace")

            completed = self._run(
                old,
                prefix,
                output,
                old_bytes=old_bytes,
                new_bytes=new_bytes,
            )

            self.assertEqual(completed.returncode, 2)
            self.assertIn("refusing to replace", completed.stderr)
            self.assertEqual(output.read_bytes(), b"do not replace")

    def test_identity_change_during_copy_is_detected_before_publish(self) -> None:
        old_bytes, _ = _artifact_bytes(chat_template="old")
        new_bytes, new_data_offset = _artifact_bytes(
            chat_template="old" + "x" * 416
        )
        module = _load_tool_module()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / "old.gguf"
            prefix = root / "corrected-prefix.bin"
            output = root / "corrected.gguf"
            old.write_bytes(old_bytes)
            prefix.write_bytes(new_bytes[:new_data_offset])
            real_identity = module._file_identity

            def changed_identity(stream):
                identity = real_identity(stream)
                if Path(stream.name) == old:
                    return replace(identity, mtime_ns=identity.mtime_ns + 1)
                return identity

            with mock.patch.object(
                module, "_file_identity", side_effect=changed_identity
            ):
                with self.assertRaisesRegex(
                    module.ReconstructionError, "old artifact changed"
                ):
                    module.reconstruct_artifact(
                        old,
                        prefix,
                        output,
                        old_size=len(old_bytes),
                        old_sha256=_sha256(old_bytes),
                        new_size=len(new_bytes),
                        new_sha256=_sha256(new_bytes),
                        chunk_bytes=17,
                    )

            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(f".{output.name}.*.tmp")), [])

    def test_directory_sync_failure_rolls_back_only_published_inode(self) -> None:
        old_bytes, _ = _artifact_bytes(chat_template="old")
        new_bytes, new_data_offset = _artifact_bytes(
            chat_template="old" + "x" * 416
        )
        module = _load_tool_module()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / "old.gguf"
            prefix = root / "corrected-prefix.bin"
            output = root / "corrected.gguf"
            old.write_bytes(old_bytes)
            prefix.write_bytes(new_bytes[:new_data_offset])

            with mock.patch.object(
                module, "_sync_directory", side_effect=OSError("injected sync failure")
            ):
                with self.assertRaisesRegex(
                    module.ReconstructionError, "cannot publish"
                ):
                    module.reconstruct_artifact(
                        old,
                        prefix,
                        output,
                        old_size=len(old_bytes),
                        old_sha256=_sha256(old_bytes),
                        new_size=len(new_bytes),
                        new_sha256=_sha256(new_bytes),
                        chunk_bytes=17,
                    )

            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(f".{output.name}.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
