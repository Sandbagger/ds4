#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    printf 'usage: %s POOLSIDE_SRC POOLSIDE_BUILD MODEL OUT_ROOT\n' "$0" >&2
}

if [[ $# -ne 4 ]]; then
    usage
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "$0")" && pwd)
poolside_src=$1
poolside_build=$2
model=$3
out_root=$4
probe_src="$script_dir/probe_poolside_laguna_moe.cpp"
tokens="$script_dir/short.tokens.i32"
patch="$script_dir/poolside-l2-callback.patch"
poolside_pin=04b2b72cb54048ead292884adbe11f284e3ec950
out_22="$out_root/poolside-c7-moe-auto-22"
out_1="$out_root/poolside-c7-moe-auto-1"
probe_bin=$(mktemp "${TMPDIR:-/tmp}/probe_poolside_laguna_moe.XXXXXX")
patched=0

restore_source() {
    local original_status=$?
    trap - EXIT
    if (( patched )); then
        git -C "$poolside_src" apply --check -R "$patch"
        git -C "$poolside_src" apply -R "$patch"
        patched=0
        cmake --build "$poolside_build" --target llama -j4
    fi
    rm -f -- "$probe_bin"
    test "$(git -C "$poolside_src" rev-parse HEAD)" = "$poolside_pin"
    test -z "$(git -C "$poolside_src" status --porcelain)"
    exit "$original_status"
}
trap restore_source EXIT

test -f "$model"
test "$(git -C "$poolside_src" rev-parse HEAD)" = "$poolside_pin"
test -z "$(git -C "$poolside_src" status --porcelain)"
test ! -e "$out_22"
test ! -e "$out_1"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
mkdir -p -- "$out_root"

git -C "$poolside_src" apply --check "$patch"
git -C "$poolside_src" apply "$patch"
patched=1
cmake --build "$poolside_build" --target llama -j4

g++ -std=c++17 -O2 \
    -I"$poolside_src/include" \
    -I"$poolside_src/ggml/include" \
    "$probe_src" \
    "$poolside_build/bin/libllama.so" \
    "$poolside_build/bin/libggml.so" \
    "$poolside_build/bin/libggml-base.so" \
    -Wl,-rpath,"$poolside_build/bin" \
    -o "$probe_bin"

"$probe_bin" \
    --model "$model" \
    --tokens "$tokens" \
    --out "$out_22" \
    --flash-attn auto \
    --detail-layer 1 \
    --token-count 22

"$probe_bin" \
    --model "$model" \
    --tokens "$tokens" \
    --out "$out_1" \
    --flash-attn auto \
    --detail-layer 1 \
    --token-count 1

printf 'capture_22=%s\ncapture_1=%s\n' "$out_22" "$out_1"
