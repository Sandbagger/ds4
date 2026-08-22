#!/usr/bin/env python3
"""portcheck - compare/pin/check staged activation captures between ds4 runs.

Tk-shaped: ensemble verbs, key=value output, stable columns, exit codes
0=green 1=red 2=usage. Captures are flat directories of little-endian
files named per a stage manifest (tests/oracle-producers/stage-manifests/).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from array import array
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "ds4-portcheck-stage-manifest/v1"
GOLDEN_SCHEMA = "ds4-portcheck-golden/v1"
DEFAULT_MANIFEST = Path(__file__).resolve().parent / "stage-manifests" / "laguna-s21.json"
FLOAT_SIZE = 4


class PortcheckError(RuntimeError):
    pass


def load_manifest(path):
    path = Path(path)
    try:
        manifest = json.loads(path.read_text())
    except OSError as error:
        raise PortcheckError(f"{path}: {error.strerror or error}") from error
    except json.JSONDecodeError as error:
        raise PortcheckError(f"{path}: invalid JSON: {error}") from error
    if manifest.get("schema") != SCHEMA:
        raise PortcheckError(f"{path}: schema {manifest.get('schema')!r} != {SCHEMA!r}")
    shape = manifest.get("shape") or {}
    for key in ("n_layer", "leading_dense"):
        value = shape.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise PortcheckError(f"{path}: shape.{key} must be a positive int")
    rule = manifest.get("layer_class_rule") or {}
    if not all(key in rule for key in ("every", "is", "else")):
        raise PortcheckError(f"{path}: layer_class_rule needs every/is/else")
    stages = manifest.get("stages")
    if not stages:
        raise PortcheckError(f"{path}: no stages")
    required = {"stage", "dtype", "width"}
    allowed = required | {"file", "layers", "optional"}
    for spec in stages:
        missing = required - set(spec)
        unknown = set(spec) - allowed
        if missing:
            raise PortcheckError(f"{path}: stage entry missing {sorted(missing)}: {spec}")
        if unknown:
            raise PortcheckError(f"{path}: stage {spec['stage']} unknown keys {sorted(unknown)}")
        spec.setdefault("layers", "detail")
        spec.setdefault("optional", False)
    return manifest


def resolve_width(expression, context):
    total = 0
    for term in str(expression).split("+"):
        value = 1
        for factor in term.split("*"):
            factor = factor.strip()
            value *= context[factor] if factor in context else int(factor)
        total += value
    return total


def layer_heads(il, manifest):
    rule = manifest["layer_class_rule"]
    cls = rule["is"] if il % rule["every"] == 0 else rule["else"]
    key = f"n_head_{cls}"
    shape = manifest["shape"]
    if key not in shape:
        raise PortcheckError(f"shape missing n_head_{cls} (layer class {cls})")
    return shape[key]


def stage_filename(spec, il=None):
    template = spec.get("file") or f"layer-{{ll}}-{spec['stage']}.{spec['dtype']}"
    if "{ll}" in template:
        if il is None:
            raise PortcheckError(f"stage {spec['stage']}: layered template needs a layer")
        return template.replace("{ll}", f"{il:02d}")
    return template


def stage_applies(spec, il, manifest):
    mode = spec["layers"]
    if mode in ("always", "all", "detail"):
        return True
    leading = manifest["shape"]["leading_dense"]
    if mode == "routed":
        return il >= leading
    if mode == "dense":
        return il < leading
    raise PortcheckError(f"stage {spec['stage']}: unknown layers mode {mode!r}")


def build_index(manifest):
    index = {}
    n_layer = manifest["shape"]["n_layer"]
    for spec in manifest["stages"]:
        if spec["layers"] == "always":
            index[stage_filename(spec)] = (spec, None)
            continue
        for il in range(n_layer):
            if stage_applies(spec, il, manifest):
                index[stage_filename(spec, il)] = (spec, il)
    return index


def scan_capture(directory, index):
    directory = Path(directory)
    if not directory.is_dir():
        raise PortcheckError(f"capture dir does not exist: {directory}")
    found, unmatched = {}, []
    for entry in sorted(directory.iterdir()):
        if not entry.is_file():
            continue
        if entry.name in index:
            found[entry.name] = entry
        else:
            unmatched.append(entry.name)
    return found, unmatched


def values_from(payload, dtype, name):
    if dtype == "q8_1":
        return None
    if len(payload) % FLOAT_SIZE:
        raise PortcheckError(f"{name}: size {len(payload)} is not a multiple of {FLOAT_SIZE}")
    values = array("f" if dtype == "f32" else "i")
    values.frombytes(payload)
    if sys.byteorder != "little":
        values.byteswap()
    return values


def f32_metrics(ref_values, cand_values, name):
    sum_delta_sq = sum_ref_sq = sum_cand_sq = dot = 0.0
    max_abs = 0.0
    first_diff = None
    for i, (expected, actual) in enumerate(zip(ref_values, cand_values)):
        expected_f = float(expected)
        actual_f = float(actual)
        if not (math.isfinite(expected_f) and math.isfinite(actual_f)):
            raise PortcheckError(f"{name}: non-finite value at index {i}")
        delta = actual_f - expected_f
        abs_delta = abs(delta)
        if abs_delta > max_abs:
            max_abs = abs_delta
        if delta and first_diff is None:
            first_diff = i
        sum_delta_sq += delta * delta
        sum_ref_sq += expected_f * expected_f
        sum_cand_sq += actual_f * actual_f
        dot += expected_f * actual_f
    count = len(ref_values)
    rms = math.sqrt(sum_delta_sq / count)
    ref_rms = math.sqrt(sum_ref_sq / count)
    relative_rms = rms / max(ref_rms, 1.0e-30)
    denominator = math.sqrt(sum_ref_sq * sum_cand_sq)
    cosine = 1.0 if denominator == 0.0 else dot / denominator
    return {
        "rms": rms,
        "relative_rms": relative_rms,
        "max_abs": max_abs,
        "cosine": cosine,
        "first_diff": first_diff,
    }


def evaluate_row(name, spec, il, ref_entry, cand_entry, thresholds, manifest):
    row = {"file": name, "stage": spec["stage"], "layer": il, "dtype": spec["dtype"]}
    if ref_entry is None:
        row["status"] = "EXTRA"
        return row
    if cand_entry is None:
        row["status"] = "MISSING"
        return row
    ref_bytes = ref_entry.read_bytes()
    cand_bytes = cand_entry.read_bytes()
    if len(ref_bytes) != len(cand_bytes):
        row.update(status="SIZE", ref_bytes=len(ref_bytes), cand_bytes=len(cand_bytes))
        return row
    row["bytes"] = len(ref_bytes)

    width_context = dict(manifest["shape"])
    if il is not None:
        width_context["heads"] = layer_heads(il, manifest)
    width = resolve_width(spec["width"], width_context)
    item_size = None if spec["dtype"] == "q8_1" else FLOAT_SIZE
    if item_size and len(ref_bytes) % (width * item_size):
        row.update(status="WIDTH", width=width)
        return row
    row["width"] = width
    row["rows"] = len(ref_bytes) // (width * item_size) if item_size else None

    if ref_bytes == cand_bytes:
        row.update(status="EXACT", rms=0.0, relative_rms=0.0, max_abs=0.0,
                   cosine=1.0, first_diff=None)
        return row

    if spec["dtype"] in ("i32", "q8_1"):
        first_diff = next((i for i, (a, b) in enumerate(zip(ref_bytes, cand_bytes)) if a != b), None)
        row.update(status="DIVERGED", first_diff=first_diff)
        return row

    metrics = f32_metrics(values_from(ref_bytes, "f32", name),
                          values_from(cand_bytes, "f32", name), name)
    max_rel = thresholds.get("max_rel_rms")
    min_cos = thresholds.get("min_cosine")
    # No thresholds given -> strict mode: any byte difference diverges.
    within = max_rel is not None or min_cos is not None
    if max_rel is not None and metrics["relative_rms"] > max_rel:
        within = False
    if min_cos is not None and metrics["cosine"] < min_cos:
        within = False
    row.update(status="CLOSE" if within else "DIVERGED", **metrics)
    return row


def ordered_names(manifest, index):
    """Forward-pass order: embd, per-layer stage ladder + residual, logits."""
    head, tail, layered = [], [], []
    for spec in manifest["stages"]:
        if spec["layers"] == "always":
            (head if not layered else tail).append(stage_filename(spec))
        else:
            layered.append(spec)
    names = list(head)
    for il in range(manifest["shape"]["n_layer"]):
        for spec in layered:
            if stage_applies(spec, il, manifest):
                names.append(stage_filename(spec, il))
    names.extend(tail)
    return names


def compare_dirs(ref_dir, cand_dir, manifest, thresholds):
    index = build_index(manifest)
    ref_found, ref_unmatched = scan_capture(ref_dir, index)
    cand_found, cand_unmatched = scan_capture(cand_dir, index)

    rows = []
    counts = {"exact": 0, "close": 0, "diverged": 0, "missing": 0,
              "extra": 0, "size": 0, "width": 0}
    first_divergence = None
    present = set(ref_found) | set(cand_found)
    for name in ordered_names(manifest, index):
        if name not in present:
            continue
        spec, il = index[name]
        row = evaluate_row(name, spec, il, ref_found.get(name), cand_found.get(name),
                           thresholds, manifest)
        rows.append(row)
        key = row["status"].lower()
        if key in counts:
            counts[key] += 1
        if row["status"] not in ("EXACT", "CLOSE") and first_divergence is None:
            first_divergence = name
    red = any(counts[k] for k in ("diverged", "missing", "extra", "size", "width"))
    return {
        "schema": "ds4-portcheck-comparison/v1",
        "model": manifest.get("model"),
        "ref": str(ref_dir),
        "cand": str(cand_dir),
        "thresholds": thresholds,
        "rows": rows,
        "unmatched": sorted(set(ref_unmatched) | set(cand_unmatched)),
        "counts": counts,
        "first_divergence": first_divergence,
        "verdict": "red" if red else "green",
    }


def print_table(report):
    header = ["file", "stage", "layer", "dtype", "status", "rows", "rms",
              "relative_rms", "max_abs", "cosine", "first_diff"]
    print("\t".join(header))
    for row in report["rows"]:
        cells = [
            row.get("file"), row.get("stage"),
            "-" if row.get("layer") is None else f"{row['layer']:02d}",
            row.get("dtype"), row.get("status"),
            "-" if row.get("rows") is None else str(row["rows"]),
            fmt_float(row.get("rms")), fmt_float(row.get("relative_rms")),
            fmt_float(row.get("max_abs")), fmt_float(row.get("cosine")),
            "-" if row.get("first_diff") is None else str(row["first_diff"]),
        ]
        print("\t".join(cells))
    counts = report["counts"]
    print("\t".join(
        f"{key}={counts[key]}" for key in
        ("exact", "close", "diverged", "missing", "extra", "size", "width")))
    print(f"unmatched={len(report['unmatched'])}")
    print(f"first_divergence={report['first_divergence'] or 'none'}")
    print(f"verdict={report['verdict']}")


def fmt_float(value):
    return "-" if value is None else f"{value:.9g}"


def pin_capture(capture_dir, out_path, model, force):
    capture_dir = Path(capture_dir)
    out_path = Path(out_path)
    if not capture_dir.is_dir():
        raise PortcheckError(f"capture dir does not exist: {capture_dir}")
    if out_path.exists() and not force:
        raise PortcheckError(f"golden exists (use --force): {out_path}")
    files = []
    for entry in sorted(capture_dir.iterdir()):
        if entry.is_file():
            payload = entry.read_bytes()
            files.append({
                "file": entry.name,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            })
    golden = {
        "schema": GOLDEN_SCHEMA,
        "model": model,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "capture": str(capture_dir),
        "files": files,
    }
    out_path.write_text(json.dumps(golden, indent=2, sort_keys=True) + "\n")
    print(f"files={len(files)}")
    print(f"golden={out_path}")
    print("verdict=green")
    return golden


def check_capture(capture_dir, golden_path):
    capture_dir = Path(capture_dir)
    golden = json.loads(Path(golden_path).read_text())
    if golden.get("schema") != GOLDEN_SCHEMA:
        raise PortcheckError(f"golden schema mismatch: {golden.get('schema')!r}")
    if not capture_dir.is_dir():
        raise PortcheckError(f"capture dir does not exist: {capture_dir}")
    pinned = {entry["file"]: entry for entry in golden["files"]}
    present = {entry.name: entry for entry in capture_dir.iterdir() if entry.is_file()}
    rows = []
    counts = {"exact": 0, "changed": 0, "missing": 0, "extra": 0}
    for name in sorted(set(pinned) | set(present)):
        status = None
        detail_bytes = None
        if name in pinned and name not in present:
            status = "MISSING"
        elif name not in pinned and name in present:
            status = "EXTRA"
        else:
            payload = present[name].read_bytes()
            detail_bytes = len(payload)
            digest = hashlib.sha256(payload).hexdigest()
            if len(payload) != pinned[name]["bytes"] or digest != pinned[name]["sha256"]:
                status = "CHANGED"
            else:
                status = "EXACT"
        counts[status.lower()] += 1
        rows.append({"file": name, "status": status, "bytes": detail_bytes})
    verdict = "red" if counts["changed"] or counts["missing"] or counts["extra"] else "green"
    report = {
        "schema": "ds4-portcheck-golden-check/v1",
        "capture": str(capture_dir),
        "golden": str(golden_path),
        "rows": rows,
        "counts": counts,
        "verdict": verdict,
    }
    for row in rows:
        print(f"file={row['file']}\tstatus={row['status']}\tbytes={row['bytes'] or '-'}")
    print("\t".join(f"{key}={counts[key]}" for key in ("exact", "changed", "missing", "extra")))
    print(f"verdict={verdict}")
    return report


def cmd_compare(args):
    manifest = load_manifest(args.manifest)
    thresholds = {"max_rel_rms": args.max_rel_rms, "min_cosine": args.min_cos}
    report = compare_dirs(args.reference, args.candidate, manifest, thresholds)
    if args.format == "json":
        json.dump(report, sys.stdout, indent=2, allow_nan=False)
        sys.stdout.write("\n")
    else:
        print_table(report)
    return 0 if report["verdict"] == "green" else 1


def cmd_pin(args):
    model = args.model
    if model is None:
        model = load_manifest(args.manifest).get("model") or "unknown"
    pin_capture(args.capture, args.out, model, args.force)
    return 0


def cmd_check(args):
    report = check_capture(args.capture, args.golden)
    return 0 if report["verdict"] == "green" else 1


def cmd_stages(args):
    manifest = load_manifest(args.manifest)
    for spec in manifest["stages"]:
        print("\t".join([
            f"stage={spec['stage']}",
            f"file={spec.get('file') or 'layer-{ll}-' + spec['stage'] + '.' + spec['dtype']}",
            f"dtype={spec['dtype']}",
            f"width={spec['width']}",
            f"layers={spec['layers']}",
            f"optional={str(spec['optional']).lower()}",
        ]))
    return 0


def cmd_cget(args):
    manifest = load_manifest(args.manifest)
    properties = {
        "path": lambda: str(Path(args.manifest)),
        "schema": lambda: manifest["schema"],
        "model": lambda: manifest.get("model") or "unknown",
        "n_layer": lambda: str(manifest["shape"]["n_layer"]),
        "stage_count": lambda: str(len(manifest["stages"])),
    }
    getter = properties.get(args.property)
    if getter is None:
        raise PortcheckError(f"unknown property {args.property!r} "
                             f"(known: {' '.join(sorted(properties))})")
    print(getter())
    return 0


VERBS = {}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="portcheck.py", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="verb")

    p = sub.add_parser("compare", help="diff two staged captures")
    p.add_argument("--reference", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p.add_argument("--max-rel-rms", type=float, default=None)
    p.add_argument("--min-cos", type=float, default=None)
    p.add_argument("--format", choices=("table", "json"), default="table")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("pin", help="hash a green capture into a golden manifest")
    p.add_argument("--capture", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p.add_argument("--model")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_pin)

    p = sub.add_parser("check", help="verify a capture against a golden manifest")
    p.add_argument("--capture", required=True)
    p.add_argument("--golden", required=True)
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("stages", help="list manifest stage vocabulary")
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p.set_defaults(func=cmd_stages)

    p = sub.add_parser("cget", help="print one bare manifest property")
    p.add_argument("property")
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p.set_defaults(func=cmd_cget)

    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_usage()
        return 2
    try:
        return args.func(args)
    except PortcheckError as error:
        print(f"portcheck: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
