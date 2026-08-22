#!/usr/bin/env python3
"""Contract tests for portcheck (staged-capture compare/pin/check)."""

from __future__ import annotations

import struct
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tests" / "oracle-producers"))

import portcheck  # noqa: E402

MANIFEST_PATH = REPO_ROOT / "tests" / "oracle-producers" / "stage-manifests" / "laguna-s21.json"
MANIFEST = portcheck.load_manifest(MANIFEST_PATH)
SHAPE = MANIFEST["shape"]


def stage_width(stage_name, il=None):
    for spec in MANIFEST["stages"]:
        if spec["stage"] == stage_name:
            context = dict(SHAPE)
            if il is not None:
                context["heads"] = portcheck.layer_heads(il, MANIFEST)
            return portcheck.resolve_width(spec["width"], context)
    raise KeyError(stage_name)


def build_capture(root, *, rows=2, detail_layers=(0, 9), corrupt=None,
                  drop=(), extra=()):
    """Write a full staged capture per the laguna-s21 vocabulary."""
    root = Path(root)
    root.mkdir(parents=True)

    def w(name, width, dtype="f32"):
        base = sum(ord(c) for c in name) % 9973
        count = width * rows
        if dtype == "i32":
            values = [(base + i * 7) % SHAPE["n_expert"] for i in range(count)]
            payload = struct.pack(f"<{count}i", *values)
        else:
            values = [(((base + i) * 2654435761) % 65536) / 65536.0 - 0.5
                      for i in range(count)]
            payload = struct.pack(f"<{count}f", *values)
        root.joinpath(name).write_bytes(payload)

    def detail_stages(il):
        heads = portcheck.layer_heads(il, MANIFEST)
        kv = SHAPE["n_head_kv"] * SHAPE["head_dim"]
        stages = [
            ("attn-norm", "f32", SHAPE["n_embd"]),
            ("q-proj", "f32", heads * SHAPE["head_dim"]),
            ("k-proj", "f32", kv),
            ("v-proj", "f32", kv),
            ("gate-proj", "f32", heads),
            ("q-rope", "f32", heads * SHAPE["head_dim"]),
            ("k-rope", "f32", kv),
            ("attn-gated", "f32", heads * SHAPE["head_dim"]),
            ("ffn-inp", "f32", SHAPE["n_embd"]),
            ("ffn-norm", "f32", SHAPE["n_embd"]),
        ]
        if il < SHAPE["leading_dense"]:
            stages.append(("ffn-out", "f32", SHAPE["n_embd"]))
        else:
            stages += [
                ("router-logits", "f32", SHAPE["n_expert"]),
                ("router-selected", "i32", SHAPE["n_expert_used"]),
                ("router-weights", "f32", SHAPE["n_expert_used"]),
                ("ffn-moe-out", "f32", SHAPE["n_embd"]),
                ("ffn-shared-out", "f32", SHAPE["n_embd"]),
            ]
        return [f"layer-{il:02d}-{stage}.{dtype}" for stage, dtype, _ in stages]

    names = ["embd.f32", "logits.f32"]
    names += [f"layer-{il:02d}.f32" for il in range(SHAPE["n_layer"])]
    for il in detail_layers:
        names += detail_stages(il)

    written = {}
    for name in sorted(set(names)):
        spec, il = portcheck.build_index(MANIFEST)[name]
        width = stage_width(spec["stage"], il)
        w(name, width, spec["dtype"])
        written[name] = (width, spec["dtype"])

    if corrupt:
        name, index, delta = corrupt
        path = root.joinpath(name)
        width, dtype = written[name]
        count = width * rows
        base = sum(ord(c) for c in name) % 9973
        if dtype == "i32":
            values = [(base + i * 7) % SHAPE["n_expert"] for i in range(count)]
            values[index] = (values[index] + delta) % SHAPE["n_expert"]
            path.write_bytes(struct.pack(f"<{count}i", *values))
        else:
            values = [(((base + i) * 2654435761) % 65536) / 65536.0 - 0.5
                      for i in range(count)]
            values[index] += delta
            path.write_bytes(struct.pack(f"<{count}f", *values))

    for name in drop:
        root.joinpath(name).unlink()
    for name in extra:
        spec, il = portcheck.build_index(MANIFEST)[name]
        w(name, stage_width(spec["stage"], il), spec["dtype"])
    return root


class WidthResolutionTests(unittest.TestCase):
    def test_layer_class_heads(self):
        self.assertEqual(portcheck.layer_heads(0, MANIFEST), 48)
        self.assertEqual(portcheck.layer_heads(1, MANIFEST), 72)
        self.assertEqual(portcheck.layer_heads(4, MANIFEST), 48)
        self.assertEqual(portcheck.layer_heads(9, MANIFEST), 72)
        self.assertEqual(portcheck.layer_heads(47, MANIFEST), 72)

    def test_stage_widths(self):
        self.assertEqual(stage_width("q-proj", 0), 6144)
        self.assertEqual(stage_width("q-proj", 9), 9216)
        self.assertEqual(stage_width("k-proj"), 1024)
        self.assertEqual(stage_width("gate-proj", 0), 48)
        self.assertEqual(stage_width("gate-proj", 9), 72)
        self.assertEqual(stage_width("router-logits", 9), 256)
        self.assertEqual(stage_width("router-selected", 9), 10)
        self.assertEqual(stage_width("residual"), 3072)
        self.assertEqual(stage_width("logits"), 100352)


class CompareTests(unittest.TestCase):
    def run_compare(self, **builder_kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            ref = build_capture(Path(tmp) / "ref")
            cand = build_capture(Path(tmp) / "cand", **builder_kwargs)
            return portcheck.compare_dirs(ref, cand, MANIFEST, {})

    def test_exact_is_green(self):
        report = self.run_compare()
        self.assertEqual(report["verdict"], "green")
        self.assertIsNone(report["first_divergence"])
        self.assertEqual(report["counts"]["exact"], len(report["rows"]))
        statuses = {row["status"] for row in report["rows"]}
        self.assertEqual(statuses, {"EXACT"})

    def test_single_float_divergence_localized(self):
        name = "layer-09-q-proj.f32"
        report = self.run_compare(corrupt=(name, 5, 0.5))
        self.assertEqual(report["verdict"], "red")
        self.assertEqual(report["first_divergence"], name)
        row = next(r for r in report["rows"] if r["file"] == name)
        self.assertEqual(row["status"], "DIVERGED")
        self.assertGreater(row["max_abs"], 0.25)
        self.assertLess(row["cosine"], 1.0)

    def test_tolerance_close(self):
        name = "layer-00-attn-norm.f32"
        thresholds = {"max_rel_rms": 0.05, "min_cosine": 0.999}
        with tempfile.TemporaryDirectory() as tmp:
            ref = build_capture(Path(tmp) / "ref")
            cand = build_capture(Path(tmp) / "cand", corrupt=(name, 11, 1e-7))
            report = portcheck.compare_dirs(ref, cand, MANIFEST, thresholds)
        row = next(r for r in report["rows"] if r["file"] == name)
        self.assertEqual(row["status"], "CLOSE")
        self.assertEqual(report["verdict"], "green")

    def test_i32_requires_bit_equality(self):
        name = "layer-09-router-selected.i32"
        loose = {"max_rel_rms": 100.0, "min_cosine": -1.0}
        with tempfile.TemporaryDirectory() as tmp:
            ref = build_capture(Path(tmp) / "ref")
            cand = build_capture(Path(tmp) / "cand", corrupt=(name, 2, 1))
            report = portcheck.compare_dirs(ref, cand, MANIFEST, loose)
        row = next(r for r in report["rows"] if r["file"] == name)
        self.assertEqual(row["status"], "DIVERGED")
        self.assertEqual(report["verdict"], "red")

    def test_missing_file_red(self):
        report = self.run_compare(drop=("logits.f32",))
        self.assertEqual(report["counts"]["missing"], 1)
        self.assertEqual(report["verdict"], "red")
        self.assertEqual(report["first_divergence"], "logits.f32")

    def test_extra_file_red(self):
        report = self.run_compare(extra=("layer-20-router-logits.f32",))
        row = next(r for r in report["rows"] if r["file"] == "layer-20-router-logits.f32")
        self.assertEqual(row["status"], "EXTRA")
        self.assertEqual(report["verdict"], "red")

    def test_size_mismatch_red(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref = build_capture(Path(tmp) / "ref")
            cand = build_capture(Path(tmp) / "cand", rows=3)
            report = portcheck.compare_dirs(ref, cand, MANIFEST, {})
        self.assertEqual(report["verdict"], "red")
        size_rows = [r for r in report["rows"] if r["status"] == "SIZE"]
        self.assertTrue(size_rows)

    def test_optional_absence_is_silent(self):
        # attn-o-proj is optional; neither side wrote it -> no row at all.
        report = self.run_compare()
        rows = [r for r in report["rows"] if r["stage"] == "attn-o-proj"]
        self.assertEqual(rows, [])


class GoldenTests(unittest.TestCase):
    def test_pin_check_roundtrip(self):
        import contextlib
        import io

        sink = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            capture = build_capture(Path(tmp) / "capture")
            golden_path = Path(tmp) / "golden.json"
            with contextlib.redirect_stdout(sink):
                golden = portcheck.pin_capture(capture, golden_path, "laguna-s-2.1", False)
            self.assertEqual(golden["model"], "laguna-s-2.1")
            self.assertTrue(golden_path.exists())
            with contextlib.redirect_stdout(sink):
                report = portcheck.check_capture(capture, golden_path)
            self.assertEqual(report["verdict"], "green")

            payload = capture.joinpath("embd.f32").read_bytes()
            capture.joinpath("embd.f32").write_bytes(payload[:-1] + bytes([payload[-1] ^ 0xFF]))
            with contextlib.redirect_stdout(sink):
                report = portcheck.check_capture(capture, golden_path)
            row = next(r for r in report["rows"] if r["file"] == "embd.f32")
            self.assertEqual(row["status"], "CHANGED")
            self.assertEqual(report["verdict"], "red")

    def test_check_detects_missing_and_extra(self):
        import contextlib
        import io

        sink = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            capture = build_capture(Path(tmp) / "capture")
            golden_path = Path(tmp) / "golden.json"
            with contextlib.redirect_stdout(sink):
                portcheck.pin_capture(capture, golden_path, "m", False)
            capture.joinpath("logits.f32").unlink()
            capture.joinpath("stray.bin").write_bytes(b"x")
            with contextlib.redirect_stdout(sink):
                report = portcheck.check_capture(capture, golden_path)
            statuses = {r["file"]: r["status"] for r in report["rows"]}
            self.assertEqual(statuses["logits.f32"], "MISSING")
            self.assertEqual(statuses["stray.bin"], "EXTRA")
            self.assertEqual(report["verdict"], "red")


class CliTests(unittest.TestCase):
    def test_cget_model(self):
        import argparse
        import contextlib
        import io

        stdout = io.StringIO()
        ns = argparse.Namespace(property="model", manifest=str(MANIFEST_PATH))
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(portcheck.cmd_cget(ns), 0)
        self.assertEqual(stdout.getvalue().strip(), "laguna-s-2.1")

    def test_stages_lists_vocabulary(self):
        import contextlib
        import io
        from types import SimpleNamespace

        stdout = io.StringIO()
        ns = SimpleNamespace(manifest=str(MANIFEST_PATH))
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(portcheck.cmd_stages(ns), 0)
        lines = stdout.getvalue().strip().splitlines()
        selected = [line for line in lines if line.startswith("stage=router-selected\t")]
        self.assertEqual(len(selected), 1)
        self.assertIn("dtype=i32", selected[0])
        self.assertIn("width=n_expert_used", selected[0])
        self.assertIn("layers=routed", selected[0])


if __name__ == "__main__":
    unittest.main()
