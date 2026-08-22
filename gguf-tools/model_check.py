#!/usr/bin/env python3
"""model_check - validate a GGUF against a ds4 model descriptor.

Descriptors are plain-text declarations of what an engine family expects:
shape scalars (engine field names), required metadata key==value pairs,
tensor vocabulary per layer scope, per-recipe dtype tables, and the HF<->GGUF
weight-name correspondence. Checking happens offline against just the GGUF
header, so a 68 GB checkpoint validates in milliseconds without loading it.

Verbs: check / dump / map / cget. Exit codes: 0 green, 1 red, 2 usage.
"""

from __future__ import annotations

import argparse
import re
import struct
import sys
from pathlib import Path

DESC_SCHEMA = "ds4-model-descriptor/v1"

GGUF_VALUE_U8, GGUF_VALUE_I8 = 0, 1
GGUF_VALUE_U16, GGUF_VALUE_I16 = 2, 3
GGUF_VALUE_U32, GGUF_VALUE_I32 = 4, 5
GGUF_VALUE_F32, GGUF_VALUE_BOOL = 6, 7
GGUF_VALUE_STRING, GGUF_VALUE_ARRAY = 8, 9
GGUF_VALUE_U64, GGUF_VALUE_I64, GGUF_VALUE_F64 = 10, 11, 12

FIXED_VALUE_SIZE = {
    GGUF_VALUE_U8: 1, GGUF_VALUE_I8: 1,
    GGUF_VALUE_U16: 2, GGUF_VALUE_I16: 2,
    GGUF_VALUE_U32: 4, GGUF_VALUE_I32: 4,
    GGUF_VALUE_F32: 4, GGUF_VALUE_BOOL: 1,
    GGUF_VALUE_U64: 8, GGUF_VALUE_I64: 8, GGUF_VALUE_F64: 8,
}

# GGML quantization ids, mirrored by the DS4_TENSOR_* enum in ds4.c
# (EngineSyncTests pins this table to that enum).
GGML_TYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K",
    13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS",
    18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S",
    23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64",
    29: "IQ1_M", 30: "BF16",
}

ARRAY_READ_CAP = 65536


class ModelCheckError(RuntimeError):
    pass


class Reader:
    def __init__(self, fh):
        self.fh = fh

    def take(self, fmt):
        size = struct.calcsize(fmt)
        payload = self.fh.read(size)
        if len(payload) != size:
            raise ModelCheckError("truncated GGUF header")
        return struct.unpack("<" + fmt, payload)[0]

    def string(self):
        length = self.take("Q")
        payload = self.fh.read(length)
        if len(payload) != length:
            raise ModelCheckError("truncated GGUF string")
        return payload.decode("utf-8", "replace")


def read_gguf(path):
    """Stream-parse the GGUF header: metadata + tensor infos. No tensor data."""
    path = Path(path)
    fh = open(path, "rb")
    try:
        magic = fh.read(4)
        if magic != b"GGUF":
            raise ModelCheckError(f"{path}: not a GGUF file")
        reader = Reader(fh)
        version = reader.take("I")
        if version < 2:
            raise ModelCheckError(f"{path}: unsupported GGUF version {version}")
        tensor_count = reader.take("Q")
        kv_count = reader.take("Q")

        metadata = {}
        for _ in range(kv_count):
            key = reader.string()
            value_type = reader.take("I")
            metadata[key] = read_value(reader, value_type)

        tensors = {}
        order = []
        for _ in range(tensor_count):
            name = reader.string()
            n_dims = reader.take("I")
            dims = [reader.take("Q") for _ in range(n_dims)]
            ggml_type = reader.take("I")
            offset = reader.take("Q")
            tensors[name] = {"dims": dims, "type": ggml_type, "offset": offset}
            order.append(name)
    finally:
        fh.close()
    return {"path": str(path), "version": version,
            "metadata": metadata, "tensors": tensors, "order": order}


def read_value(reader, value_type):
    if value_type == GGUF_VALUE_STRING:
        return reader.string()
    if value_type == GGUF_VALUE_ARRAY:
        elem_type = reader.take("I")
        count = reader.take("Q")
        if elem_type == GGUF_VALUE_STRING:
            values = []
            for index in range(count):
                value = reader.string()
                if index < ARRAY_READ_CAP:
                    values.append(value)
            return values[:ARRAY_READ_CAP]
        size = FIXED_VALUE_SIZE.get(elem_type)
        if size is None:
            raise ModelCheckError(f"unknown array element type {elem_type}")
        if count > ARRAY_READ_CAP:
            reader.fh.seek(count * size, 1)
            return None
        return [read_value(reader, elem_type) for _ in range(count)]
    if value_type == GGUF_VALUE_BOOL:
        return bool(reader.take("B"))
    if value_type == GGUF_VALUE_U8:
        return reader.take("B")
    if value_type == GGUF_VALUE_I8:
        return reader.take("b")
    if value_type == GGUF_VALUE_U16:
        return reader.take("H")
    if value_type == GGUF_VALUE_I16:
        return reader.take("h")
    if value_type == GGUF_VALUE_U32:
        return reader.take("I")
    if value_type == GGUF_VALUE_I32:
        return reader.take("i")
    if value_type == GGUF_VALUE_F32:
        return reader.take("f")
    if value_type == GGUF_VALUE_U64:
        return reader.take("Q")
    if value_type == GGUF_VALUE_I64:
        return reader.take("q")
    if value_type == GGUF_VALUE_F64:
        return reader.take("d")
    raise ModelCheckError(f"unknown metadata value type {value_type}")


def type_name(ggml_type):
    return GGML_TYPE_NAMES.get(ggml_type, f"T{ggml_type}")


def parse_desc(path):
    path = Path(path)
    sections = {}
    top = {}
    current = None
    current_name = None
    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        heading = re.match(r"\[([A-Za-z0-9_.-]+)\]$", line)
        if heading:
            current_name = heading.group(1)
            current = []
            sections[current_name] = current
            continue
        if current is None:
            if "=" not in line:
                raise ModelCheckError(
                    f"{path}:{lineno}: expected key=value, got {raw!r}")
            key, _, value = line.partition("=")
            top[key.strip()] = value.strip()
            continue
        if current_name == "tensors.top" or current_name.startswith("tensors.all") \
                or current_name.startswith("tensors.dense") \
                or current_name.startswith("tensors.routed"):
            # tensor vocabulary sections hold one bare template per line
            current.append((line, None))
        elif "=" in line:
            separator = "==" if current_name == "metadata" else "="
            key, _, value = line.partition(separator)
            current.append((key.strip(), value.strip()))
        else:
            raise ModelCheckError(
                f"{path}:{lineno}: expected key=value, got {raw!r}")
    for required in ("schema", "arch", "model"):
        if required not in top:
            raise ModelCheckError(f"{path}: missing top-level {required}")
    if top["schema"] != DESC_SCHEMA:
        raise ModelCheckError(
            f"{path}: schema {top['schema']!r} != {DESC_SCHEMA!r}")
    return {"path": str(path), "top": top, "sections": sections}


def desc_section(desc, name):
    return desc["sections"].get(name, [])


def desc_map(desc, name):
    return dict(desc_section(desc, name))


def resolve_width(expression, context):
    total = 0
    for term in str(expression).split("+"):
        value = 1
        for factor in term.split("*"):
            factor = factor.strip()
            value *= context[factor] if factor in context else int(factor)
        total += value
    return total


def layer_heads(il, desc):
    rule = desc_map(desc, "layer_rule")
    cls = "swa" if il % int(rule["every"]) == 0 else "dense"
    return int(rule[f"{cls}_heads"])


def expand_templates(templates, desc):
    """template -> list of (concrete-name, il or None), over the layer scopes."""
    shape = desc_map(desc, "shape")
    leading = int(shape.get("n_leading_dense", 0))
    context = {key: coerce_int(value) for key, value in shape.items()}
    out = {}
    for template, _ in templates:
        if "%u" not in template:
            out.setdefault(template, []).append((template, None))
            continue
        scope = None
        for candidate in ("tensors.all", "tensors.dense", "tensors.routed"):
            if template in desc_map(desc, candidate):
                scope = candidate.split(".")[1]
                break
        for il in range(int(shape["n_layer"])):
            if scope == "dense" and il >= leading:
                continue
            if scope == "routed" and il < leading:
                continue
            name = template.replace("%u", str(il))
            out.setdefault(template, []).append((name, il))
    return out


def coerce_int(value):
    return int(value) if not isinstance(value, str) or "." not in value else int(float(value))


def parse_expected(text):
    text = text.strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        return ("str", text[1:-1])
    if text in ("true", "false"):
        return ("bool", text == "true")
    if re.fullmatch(r"-?\d+", text):
        return ("num", int(text))
    try:
        return ("num", float(text))
    except ValueError:
        return ("str", text)


def values_equal(expected_text, actual):
    kind, expected = parse_expected(expected_text)
    if kind == "bool":
        return isinstance(actual, bool) and actual == expected
    if kind == "num":
        if isinstance(actual, bool) or not isinstance(actual, (int, float)):
            return False
        tolerance = 1e-9 * max(1.0, abs(float(actual)))
        return abs(float(expected) - float(actual)) <= tolerance
    return isinstance(actual, str) and actual == expected


def format_value(value):
    if isinstance(value, list):
        shown = ", ".join(repr(item) for item in value[:3])
        return f"[len={len(value)} {shown}{' …' if len(value) > 3 else ''}]"
    return repr(value)


DISCRIMINATOR_RE = re.compile(r"^(.+)\.type==([A-Za-z0-9_]+)$")


def pick_recipe(desc, tensors):
    for recipe, discriminator in desc_section(desc, "recipes"):
        match = DISCRIMINATOR_RE.match(discriminator)
        if not match:
            raise ModelCheckError(f"recipe {recipe}: bad discriminator {discriminator!r}")
        marker, wanted = match.group(1), match.group(2)
        info = tensors.get(marker)
        if info is not None and type_name(info["type"]) == wanted:
            return recipe
    return None


def check_model(desc, gg):
    meta = gg["metadata"]
    tensors = gg["tensors"]
    findings = []

    def bad(code, detail):
        findings.append((code, detail))

    checks = 0

    if meta.get("general.architecture") != desc["top"]["arch"]:
        bad("arch-mismatch",
            f"general.architecture={meta.get('general.architecture')!r} "
            f"!= {desc['top']['arch']!r}")
    else:
        checks += 1

    for key, expected in desc_section(desc, "metadata"):
        if key not in meta:
            bad("missing-key", key)
            continue
        if values_equal(expected, meta[key]):
            checks += 1
        else:
            bad("mismatch-key",
                f"{key}: expected {expected}, got {format_value(meta[key])}")

    rule = desc_map(desc, "layer_rule")
    array_key = rule.get("head_count_array")
    if array_key:
        heads = meta.get(array_key)
        shape = desc_map(desc, "shape")
        n_layer = int(shape["n_layer"])
        every = int(rule["every"])
        expected_heads = lambda il: (int(rule["swa_heads"]) if il % every == 0
                                     else int(rule["dense_heads"]))
        if not isinstance(heads, list) or len(heads) != n_layer:
            bad("mismatch-key",
                f"{array_key}: expected {n_layer}-entry array, got "
                f"{format_value(heads)}")
        else:
            wrong = [(il, got) for il, got in enumerate(heads)
                     if got != expected_heads(il)]
            if wrong:
                il, got = wrong[0]
                bad("mismatch-key",
                    f"{array_key}[{il}]: expected {expected_heads(il)}, got {got} "
                    f"({len(wrong)} wrong)")
            else:
                checks += 1

    tensor_sections = [("tensors.top", None), ("tensors.all", "all"),
                       ("tensors.dense", "dense"), ("tensors.routed", "routed")]
    templates = [entry for section, _ in tensor_sections
                 for entry in desc_section(desc, section)]
    concrete = expand_templates(templates, desc)

    for template, instances in concrete.items():
        for name, _ in instances:
            if name in tensors:
                checks += 1
            else:
                bad("missing-tensor", name)

    dims_maps = {section: desc_map(desc, section)
                 for section in ("dims.top", "dims.all", "dims.dense", "dims.routed")}
    for template, instances in concrete.items():
        spec = None
        for section in ("dims.top", "dims.all", "dims.dense", "dims.routed"):
            if template in dims_maps[section]:
                spec = dims_maps[section][template]
                break
        if spec is None:
            continue
        for name, il in instances:
            info = tensors.get(name)
            if info is None:
                continue
            context = dict((k, coerce_int(v)) for k, v in desc_map(desc, "shape").items())
            if il is not None:
                context["heads"] = layer_heads(il, desc)
            try:
                expected_dims = [resolve_width(part.strip(), context)
                                 for part in spec.split(",")]
            except (KeyError, ValueError) as error:
                bad("bad-dims-spec", f"{template}: {error}")
                continue
            if info["dims"] != expected_dims:
                bad("wrong-dims",
                    f"{name}: expected {expected_dims}, got {info['dims']}")
            else:
                checks += 1

    recipe = pick_recipe(desc, tensors)
    if recipe is None:
        candidates = ", ".join(d for _, d in desc_section(desc, "recipes")) or "none declared"
        bad("recipe-unknown", f"no recipe matched ({candidates})")
    else:
        types_map = desc_map(desc, f"types.{recipe}")
        for template, allowed_text in types_map.items():
            allowed = set(allowed_text.split("|"))
            for name, _ in concrete.get(template, []):
                info = tensors.get(name)
                if info is None:
                    continue
                got = type_name(info["type"])
                if got not in allowed:
                    bad("wrong-type", f"{name}: {got} not in {allowed_text}")
                else:
                    checks += 1

    return {
        "checks": checks,
        "findings": findings,
        "recipe": recipe,
        "verdict": "red" if findings else "green",
    }


def translate(name, pairs, source_index, target_index):
    for source_template, target_template in pairs:
        pattern = re.compile(
            "^" + re.escape(source_template).replace(re.escape("%u"), r"(\d+)") + "$")
        match = pattern.match(name)
        if match:
            result = target_template
            for group in match.groups():
                result = result.replace("%u", group, 1)
            return result
    raise ModelCheckError(f"name {name!r} matches no {source_index} template "
                          f"(have {len(pairs)} {target_index} mappings)")


def cmd_check(args):
    desc = parse_desc(args.desc)
    gg = read_gguf(args.gguf)
    report = check_model(desc, gg)
    for code, detail in report["findings"][:args.max_findings]:
        print(f"finding={code}\tdetail={detail}")
    hidden = max(0, len(report["findings"]) - args.max_findings)
    print(f"checks={report['checks']}\tfindings={len(report['findings'])}"
          + (f"\thidden={hidden}" if hidden else ""))
    print(f"recipe={report['recipe'] or 'none'}")
    print(f"verdict={report['verdict']}")
    return 0 if report["verdict"] == "green" else 1


def cmd_dump(args):
    gg = read_gguf(args.gguf)
    print(f"version={gg['version']}")
    print(f"tensors={len(gg['tensors'])}")
    for key in sorted(gg["metadata"]):
        value = gg["metadata"][key]
        if value is None:
            print(f"kv[{key}]=<large array>")
        else:
            print(f"kv[{key}]={format_value(value)}")
    if args.tensors:
        for name in gg["order"]:
            info = gg["tensors"][name]
            dims = "x".join(str(d) for d in info["dims"])
            print(f"tensor\t{name}\t{type_name(info['type'])}\t{dims}")
    return 0


def cmd_map(args):
    desc = parse_desc(args.desc)
    pairs = desc_section(desc, "names.hf")
    if args.hf:
        print(translate(args.hf, [(hf, gguf) for gguf, hf in pairs],
                        "hf", "gguf"))
    else:
        print(translate(args.gguf, pairs, "gguf", "hf"))
    return 0


def cmd_cget(args):
    desc = parse_desc(args.desc)
    properties = {
        "path": lambda: str(Path(args.desc)),
        "schema": lambda: desc["top"]["schema"],
        "arch": lambda: desc["top"]["arch"],
        "model": lambda: desc["top"]["model"],
        "n_layer": lambda: desc_map(desc, "shape")["n_layer"],
        "recipes": lambda: "\n".join(name for name, _ in desc_section(desc, "recipes")),
        "templates": lambda: str(len({t for t, _ in desc_section(desc, "tensors.top")
                                      + desc_section(desc, "tensors.all")
                                      + desc_section(desc, "tensors.dense")
                                      + desc_section(desc, "tensors.routed")})),
    }
    getter = properties.get(args.property)
    if getter is None:
        raise ModelCheckError(f"unknown property {args.property!r} "
                              f"(known: {' '.join(sorted(properties))})")
    print(getter())
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="model_check.py",
                                     description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="verb")

    p = sub.add_parser("check", help="validate a GGUF against a descriptor")
    p.add_argument("--desc", required=True)
    p.add_argument("--gguf", required=True)
    p.add_argument("--max-findings", type=int, default=20)
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("dump", help="print GGUF header metadata (+ tensor list)")
    p.add_argument("--gguf", required=True)
    p.add_argument("--tensors", action="store_true")
    p.set_defaults(func=cmd_dump)

    p = sub.add_parser("map", help="translate between HF and GGUF weight names")
    p.add_argument("--desc", required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--hf")
    group.add_argument("--gguf")
    p.set_defaults(func=cmd_map)

    p = sub.add_parser("cget", help="print one bare descriptor property")
    p.add_argument("property")
    p.add_argument("--desc", required=True)
    p.set_defaults(func=cmd_cget)

    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_usage()
        return 2
    try:
        return args.func(args)
    except ModelCheckError as error:
        print(f"model_check: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
