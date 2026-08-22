#!/usr/bin/env python3
"""Contract tests for model_check and the laguna-s21 model descriptor."""

from __future__ import annotations

import contextlib
import io
import re
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "gguf-tools"))

import model_check as mc  # noqa: E402

DESC_PATH = REPO_ROOT / "gguf-tools" / "models" / "laguna-s21.desc"
DESC = mc.parse_desc(DESC_PATH)

TYPE_IDS = {"F32": 0, "F16": 1, "Q4_0": 2, "Q8_0": 8, "Q2_K": 10,
            "Q3_K": 11, "Q4_K": 12, "Q5_K": 13, "Q6_K": 14}


def enc_str(text):
    payload = text.encode("utf-8")
    return struct.pack("<Q", len(payload)) + payload


def encode_kv(key, value):
    out = bytearray(enc_str(key))
    if isinstance(value, bool):
        out += struct.pack("<IB", 7, int(value))
    elif isinstance(value, int):
        if 0 <= value < 2**32:
            out += struct.pack("<II", 4, value)
        else:
            out += struct.pack("<IQ", 10, value)
    elif isinstance(value, float):
        out += struct.pack("<If", 6, value)
    elif isinstance(value, str):
        out += struct.pack("<I", 8) + enc_str(value)
    elif isinstance(value, list):
        out += struct.pack("<IIQ", 9, 4, len(value))
        for item in value:
            out += struct.pack("<I", item)
    else:
        raise TypeError(f"unsupported kv type {type(value)}")
    return bytes(out)


def write_gguf(path, metadata, tensors):
    """Minimal GGUF v3 writer: header + kv + tensor infos, no data section."""
    buf = bytearray(b"GGUF")
    buf += struct.pack("<IQQ", 3, len(tensors), len(metadata))
    for key in sorted(metadata):
        buf += encode_kv(key, metadata[key])
    for name, dims, ggml_type in tensors:
        buf += enc_str(name)
        buf += struct.pack("<I", len(dims))
        buf += struct.pack(f"<{len(dims)}Q", *dims)
        buf += struct.pack("<IQ", ggml_type, 0)
    path.write_bytes(bytes(buf))
    return path


def synthetic_metadata(desc=DESC):
    meta = {}
    for key, expected in mc.desc_section(desc, "metadata"):
        _, value = mc.parse_expected(expected)
        meta[key] = value
    rule = mc.desc_map(desc, "layer_rule")
    every = int(rule["every"])
    shape = mc.desc_map(desc, "shape")
    heads = [int(rule["swa_heads"]) if il % every == 0 else int(rule["dense_heads"])
             for il in range(int(shape["n_layer"]))]
    meta[rule["head_count_array"]] = heads
    return meta


DIMS_SECTIONS = ("dims.top", "dims.all", "dims.dense", "dims.routed")


def synthetic_tensors(desc=DESC, recipe="signal-q8"):
    templates = []
    for section in ("tensors.top", "tensors.all", "tensors.dense", "tensors.routed"):
        templates += mc.desc_section(desc, section)
    concrete = mc.expand_templates(templates, desc)
    types_map = mc.desc_map(desc, f"types.{recipe}")
    dims_maps = {section: mc.desc_map(desc, section) for section in DIMS_SECTIONS}
    shape_ctx = {key: mc.coerce_int(value) for key, value in mc.desc_map(desc, "shape").items()}
    tensors = []
    for template, instances in concrete.items():
        allowed = types_map.get(template, "F32").split("|")[0]
        dims_spec = next((dims_maps[s][template] for s in DIMS_SECTIONS
                          if template in dims_maps[s]), None)
        for name, il in instances:
            context = dict(shape_ctx)
            if il is not None:
                context["heads"] = mc.layer_heads(il, desc)
            dims = ([mc.resolve_width(part.strip(), context)
                     for part in dims_spec.split(",")]
                    if dims_spec else [1])
            tensors.append((name, dims, TYPE_IDS[allowed]))
    return tensors


def build_model(path, *, recipe="signal-q8", drop=None, meta_patch=None,
                meta_drop=None, retype=None, redim=None, heads_override=None):
    meta = synthetic_metadata()
    if heads_override is not None:
        meta["laguna.attention.head_count"] = heads_override
    if meta_patch:
        meta.update(meta_patch)
    if meta_drop:
        meta.pop(meta_drop, None)
    tensors = synthetic_tensors(recipe=recipe)
    if drop:
        tensors = [entry for entry in tensors if entry[0] != drop]
    if retype:
        name, ggml_type = retype
        tensors = [(n, d, ggml_type if n == name else t) for n, d, t in tensors]
    if redim:
        name, dims = redim
        tensors = [(n, dims if n == name else d, t) for n, d, t in tensors]
    return write_gguf(Path(path), meta, tensors)


class CheckTests(unittest.TestCase):
    def run_check(self, **kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            gguf = build_model(Path(tmp) / "model.gguf", **kwargs)
            return mc.check_model(DESC, mc.read_gguf(gguf))

    def codes(self, report):
        return {code for code, _ in report["findings"]}

    def test_green_signal_q8(self):
        report = self.run_check()
        self.assertEqual(report["verdict"], "green")
        self.assertEqual(report["findings"], [])
        self.assertEqual(report["recipe"], "signal-q8")

    def test_green_legacy(self):
        report = self.run_check(recipe="legacy")
        self.assertEqual(report["verdict"], "green")
        self.assertEqual(report["recipe"], "legacy")

    def test_missing_tensor_red(self):
        report = self.run_check(drop="blk.47.ffn_norm.weight")
        self.assertEqual(self.codes(report), {"missing-tensor"})
        self.assertEqual(report["verdict"], "red")

    def test_wrong_type_red(self):
        report = self.run_check(retype=("blk.3.attn_q.weight", TYPE_IDS["F16"]))
        self.assertEqual(self.codes(report), {"wrong-type"})
        details = [detail for _, detail in report["findings"]]
        self.assertTrue(any("blk.3.attn_q.weight" in detail for detail in details))

    def test_metadata_mismatch_red(self):
        report = self.run_check(meta_patch={"laguna.block_count": 47})
        self.assertEqual(self.codes(report), {"mismatch-key"})

    def test_missing_key_red(self):
        report = self.run_check(meta_drop="laguna.vocab_size")
        self.assertEqual(self.codes(report), {"missing-key"})
        details = [detail for _, detail in report["findings"]]
        self.assertIn("laguna.vocab_size", details)

    def test_head_array_rule_red(self):
        heads = synthetic_metadata()["laguna.attention.head_count"]
        heads[5] = 48
        report = self.run_check(heads_override=heads)
        self.assertEqual(self.codes(report), {"mismatch-key"})
        details = [detail for _, detail in report["findings"]]
        self.assertTrue(any("[5]" in detail for detail in details))

    def test_recipe_unknown_red(self):
        report = self.run_check(retype=("token_embd.weight", TYPE_IDS["F16"]))
        self.assertEqual(self.codes(report), {"recipe-unknown"})
        self.assertEqual(report["recipe"], None)

    def test_wrong_dims_red(self):
        report = self.run_check(redim=("output_norm.weight", [3071]))
        self.assertEqual(self.codes(report), {"wrong-dims"})


class MapAndCgetTests(unittest.TestCase):
    def run_tool(self, func, namespace):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            status = func(namespace)
        return status, stdout.getvalue()

    def test_map_gguf_to_hf(self):
        ns = SimpleNamespace(desc=str(DESC_PATH), hf=None,
                             gguf="blk.9.attn_q.weight")
        status, out = self.run_tool(mc.cmd_map, ns)
        self.assertEqual(status, 0)
        self.assertEqual(out.strip(), "model.layers.9.self_attn.q_proj.weight")

    def test_map_hf_to_gguf(self):
        ns = SimpleNamespace(desc=str(DESC_PATH),
                             hf="model.layers.20.mlp.shared_expert.down_proj.weight",
                             gguf=None)
        status, out = self.run_tool(mc.cmd_map, ns)
        self.assertEqual(status, 0)
        self.assertEqual(out.strip(), "blk.20.ffn_down_shexp.weight")

    def test_cget_model(self):
        ns = SimpleNamespace(property="model", desc=str(DESC_PATH))
        status, out = self.run_tool(mc.cmd_cget, ns)
        self.assertEqual(status, 0)
        self.assertEqual(out.strip(), "laguna-s-2.1")


class EngineSyncTests(unittest.TestCase):
    """Pin the descriptor to ds4.c: profile numbers, validator rules, names."""

    @classmethod
    def setUpClass(cls):
        cls.source = (REPO_ROOT / "ds4.c").read_text()
        match = re.search(r"static const ds4_shape DS4_SHAPE_LAGUNA_S21 = \{(.*?)\};",
                          cls.source, re.S)
        cls.profile_block = match.group(1)

    def engine_fields(self):
        fields = {}
        for m in re.finditer(r"\.(\w+)\s*=\s*([^,\n]+)", self.profile_block):
            raw = m.group(2).strip().rstrip(",")
            if raw.startswith('"'):
                continue
            fields[m.group(1)] = raw[:-1] if raw.endswith("f") else raw
        return fields

    def test_shape_matches_engine_profile(self):
        engine = self.engine_fields()
        for key, value in mc.desc_map(DESC, "shape").items():
            self.assertIn(key, engine, f"desc shape key {key} absent from C profile")
            expected = engine[key]
            if "." in value or "." in expected:
                self.assertAlmostEqual(float(value), float(expected), places=12,
                                       msg=key)
            else:
                self.assertEqual(int(value), int(expected), msg=key)

    def test_layer_rule_matches_validator_interleave(self):
        validator = re.search(
            r"config_validate_laguna_model\(const ds4_model \*m\) \{.*?\n\}",
            self.source, re.S).group(0)
        interleave = re.search(r"\(il % (\d+)u\) == 0 \? (\d+)u : (\d+)u", validator)
        self.assertIsNotNone(interleave, "validator head-interleave expression moved")
        rule = mc.desc_map(DESC, "layer_rule")
        self.assertEqual(int(rule["every"]), int(interleave.group(1)))
        self.assertEqual(int(rule["swa_heads"]), int(interleave.group(2)))
        self.assertEqual(int(rule["dense_heads"]), int(interleave.group(3)))
        self.assertIn('required_u32(m, "laguna.expert_gating_func")', validator)
        self.assertIn("expert_gating_func, 2)", validator)
        self.assertIn('ds4_streq(rope_type, "yarn")', validator)

    def test_tensor_templates_appear_in_engine_source(self):
        for section in ("tensors.top", "tensors.all", "tensors.dense", "tensors.routed"):
            for template, _ in mc.desc_section(DESC, section):
                self.assertIn(f'"{template}"', self.source,
                              f"tensor name {template!r} not bound by ds4.c")

    def test_type_table_matches_engine_enum(self):
        """K-quant ids are 10..15; this pins the reader to ds4.c's DS4_TENSOR_*."""
        block = re.search(r"enum \{\s*DS4_TENSOR_F32\s*= 0,.*?\};",
                          self.source, re.S).group(0)
        engine = dict((name, int(value)) for name, value in
                      re.findall(r"DS4_TENSOR_(\w+)\s*=\s*(\d+)", block))
        self.assertGreaterEqual(len(engine), 8)
        for name, value in engine.items():
            self.assertEqual(mc.type_name(value), name,
                             f"DS4_TENSOR_{name}={value} disagrees with reader table")


if __name__ == "__main__":
    unittest.main()
