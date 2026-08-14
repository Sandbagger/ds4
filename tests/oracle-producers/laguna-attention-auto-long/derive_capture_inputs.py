#!/usr/bin/env python3
"""Derive the exact source and token inputs for Laguna long captures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct


BASE_PROBE_BYTES = 21085
BASE_PROBE_SHA256 = (
    "28825c8f2273ff5fef28d1250d2ec8bb82addbf5b736ec56a2bc99e1e07c9df6"
)
CASE_TOKENS = {
    "layer0_gqa6_512": 512,
    "layer1_gqa9_64": 64,
}


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError(f"base probe does not contain one expected span: {old[:60]!r}")
    return source.replace(old, new, 1)


def derive_probe(base_probe: bytes, token_count: int) -> bytes:
    if len(base_probe) != BASE_PROBE_BYTES or sha256(base_probe) != BASE_PROBE_SHA256:
        raise ValueError("base probe does not match the pinned Poolside probe identity")

    source = base_probe.decode("utf-8")
    source = replace_once(
        source,
        "static constexpr int kTokens = 22;",
        f"static constexpr int kTokens = {token_count};",
    )
    source = replace_once(
        source,
        """static constexpr std::array<int32_t, kTokens> kExpectedTokens = {
    2, 97, 1437, 99, 53225, 3203, 330, 10068, 3612, 31063, 81,
    365, 1161, 15631, 83, 268, 532, 1437, 99, 268, 23, 19,
};

""",
        "",
    )
    source = replace_once(
        source,
        'fail("token file must contain exactly 22 little-endian int32 IDs (88 bytes)");',
        f'fail("token file must contain exactly {token_count} little-endian int32 IDs");',
    )
    source = replace_once(
        source,
        """        if (token != kExpectedTokens[index]) {
            fail("token file does not contain the exact Laguna short prompt at index " +
                 std::to_string(index));
        }
""",
        "",
    )
    source = replace_once(
        source,
        """static constexpr std::array<DetailTarget, 12> kDetailTargets = {{
    {"attn_norm", "attn-norm", DetailLayout::fixed, 3072, 22, 1},
    {"Qcur", "q-proj", DetailLayout::query_flat, 0, 0, 0},
    {"Kcur", "k-proj", DetailLayout::fixed, 1024, 22, 1},
    {"Vcur", "v-proj", DetailLayout::fixed, 1024, 22, 1},
    {"attn_gate_proj", "gate-proj", DetailLayout::query_gate, 0, 0, 0},
    {"Qcur_rope", "q-rope", DetailLayout::query_rope, 0, 0, 0},
    {"Kcur_rope", "k-rope", DetailLayout::fixed, 128, 8, 22},
    {"attn_gated", "attn-gated", DetailLayout::query_flat, 0, 0, 0},
    {"attn_o_proj", "attn-o-proj", DetailLayout::fixed, 3072, 22, 1},
    {"ffn_inp", "ffn-inp", DetailLayout::fixed, 3072, 22, 1},
    {"ffn_norm", "ffn-norm", DetailLayout::fixed, 3072, 22, 1},
    {"ffn_out", "ffn-out", DetailLayout::fixed, 3072, 22, 1},
}};
""",
        """static constexpr std::array<DetailTarget, 5> kDetailTargets = {{
    {"Vcur", "v-proj", DetailLayout::fixed, 1024, kTokens, 1},
    {"attn_gate_proj", "gate-proj", DetailLayout::query_gate, 0, 0, 0},
    {"Qcur_rope", "q-rope", DetailLayout::query_rope, 0, 0, 0},
    {"Kcur_rope", "k-rope", DetailLayout::fixed, 128, 8, kTokens},
    {"attn_gated", "attn-gated", DetailLayout::query_flat, 0, 0, 0},
}};
""",
    )
    source = replace_once(
        source,
        """    if (std::strcmp(name, "embd") == 0) {
        return {TargetKind::embedding, -1};
    }
    // The final Laguna residual is the same tensor first named l_out-47 and
    // then renamed h_nextn when the graph exposes it to speculative drafters.
    if (std::strcmp(name, "h_nextn") == 0) {
        return {TargetKind::layer, kLayers - 1};
    }
    if (std::strcmp(name, "result_output") == 0) {
        return {TargetKind::logits, -1};
    }
""",
        "",
    )
    source = replace_once(
        source,
        """
    static constexpr const char *prefix = "l_out-";
    const size_t prefix_length = std::strlen(prefix);
    if (std::strncmp(name, prefix, prefix_length) != 0) {
        return {};
    }

    const char *suffix = name + prefix_length;
    if (*suffix == '\\0') {
        return {};
    }
    errno = 0;
    char *end = nullptr;
    const long parsed = std::strtol(suffix, &end, 10);
    if (errno != 0 || *end != '\\0' || parsed < 0 || parsed >= kLayers) {
        return {};
    }
    if (std::string(prefix) + std::to_string(parsed) != name) {
        return {};
    }
    return {TargetKind::layer, static_cast<int>(parsed)};
""",
        """
    return {};
""",
    )
    source = replace_once(
        source,
        """    if (!state.embedding_seen) {
        fail("embd callback was not observed");
    }
    for (int layer = 0; layer < kLayers; layer++) {
        if (!state.layer_seen[layer]) {
            fail("l_out callback was not observed for layer " + std::to_string(layer));
        }
    }
""",
        "",
    )
    source = replace_once(
        source,
        """    if (!state.logits_seen) {
        fail("result_output callback was not observed");
    }
""",
        "",
    )
    source = replace_once(
        source,
        """        if (options.detail_layer == 0) {
            std::printf("embedding=embd.f32\\nlayer0_checkpoints=12\\nlayers=48\\n"
                        "logits=logits.f32\\nout=%s\\n",
                        options.out.c_str());
        } else {
            std::printf("embedding=embd.f32\\ndetail_layer=%d\\n"
                        "detail_checkpoints=12\\nlayers=48\\n"
                        "logits=logits.f32\\nout=%s\\n",
                        options.detail_layer, options.out.c_str());
        }
""",
        """        if (options.detail_layer == 0) {
            std::printf("detail_layer=0\\ndetail_checkpoints=5\\nout=%s\\n",
                        options.out.c_str());
        } else {
            std::printf("detail_layer=%d\\ndetail_checkpoints=5\\nout=%s\\n",
                        options.detail_layer, options.out.c_str());
        }
""",
    )
    return source.encode("utf-8")


def derive_tokens(token_prefix: bytes, token_count: int) -> bytes:
    specification = json.loads(token_prefix.decode("utf-8"))
    if specification.get("schema") != "laguna-token-prefix/v1":
        raise ValueError("token prefix has an unsupported schema")
    if specification.get("encoding") != "little-endian-signed-int32":
        raise ValueError("token prefix has an unsupported encoding")
    pattern = specification.get("repeating_pattern")
    available = specification.get("tokens")
    if not isinstance(pattern, list) or not pattern:
        raise ValueError("token prefix repeating_pattern must be a non-empty list")
    if not isinstance(available, int) or available < token_count:
        raise ValueError(f"token prefix does not contain {token_count} tokens")
    values = [pattern[index % len(pattern)] for index in range(token_count)]
    if any(not isinstance(value, int) or not -(2**31) <= value < 2**31 for value in values):
        raise ValueError("token prefix contains a value outside signed int32")
    return struct.pack(f"<{token_count}i", *values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASE_TOKENS, required=True)
    parser.add_argument("--base-probe", type=Path, required=True)
    parser.add_argument("--token-prefix", type=Path, required=True)
    parser.add_argument("--probe-out", type=Path, required=True)
    parser.add_argument("--tokens-out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token_count = CASE_TOKENS[args.case]
    probe = derive_probe(args.base_probe.read_bytes(), token_count)
    tokens = derive_tokens(args.token_prefix.read_bytes(), token_count)
    args.probe_out.write_bytes(probe)
    args.tokens_out.write_bytes(tokens)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
