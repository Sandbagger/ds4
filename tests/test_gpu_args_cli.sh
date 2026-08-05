#!/usr/bin/env bash
# CLI option smoke tests. Run from the repo root via `make test`.
# These do not exercise CUDA or tensor-parallel hardware.
set -uo pipefail

cd "$(dirname "$0")/.."

PASS=0
FAIL=0
LOG=$(mktemp)

ok()   { PASS=$((PASS+1)); echo "ok $1"; }
fail() { FAIL=$((FAIL+1)); echo "FAIL $1"; }

assert_grep() {
    # $1 = name, $2 = pattern, $3 = file
    if grep -q -- "$2" "$3" 2>/dev/null; then ok "$1"; else
        fail "$1 (pattern not in $3)"
        echo "    --- content of $3 ---"
        head -20 "$3" | sed 's/^/    /'
    fi
}

assert_not_grep() {
    # $1 = name, $2 = pattern, $3 = file
    if grep -q -- "$2" "$3" 2>/dev/null; then
        fail "$1 (obsolete pattern found in $3)"
    else
        ok "$1"
    fi
}

# Binaries to check
BINS=(./ds4 ./ds4-server ./ds4-bench ./ds4-agent)
NAMES=(ds4 ds4-server ds4-bench ds4-agent)

# 1: each binary's --help mentions both flags.
for i in "${!BINS[@]}"; do
    name=${NAMES[$i]}; bin=${BINS[$i]}
    if [ ! -x "$bin" ]; then
        fail "$name not built — skipping help check"
        continue
    fi
    "$bin" --help > "$LOG" 2>&1 || true
    assert_grep "$name --help mentions --gpu-vram" "gpu-vram" "$LOG"
    assert_grep "$name --help mentions --gpu-devices" "gpu-devices" "$LOG"
    assert_grep "$name --help mentions --cuda-tensor-parallel" "cuda-tensor-parallel" "$LOG"
    if [ "$name" = "ds4" ]; then
        "$bin" --help distributed > "$LOG" 2>&1 || true
        assert_grep "$name --help distributed mentions --tensor-parallel-token-prefill" \
            "tensor-parallel-token-prefill" "$LOG"
        assert_not_grep "$name --help distributed omits old --tp spellings" "--tp-" "$LOG"
    fi
done

# 2: parser error on syntactically invalid value. For ds4-bench, we
# also pass --prompt-file /dev/null so it doesn't exit on the
# "specify exactly one of --prompt-file or --chat-prompt-file" check
# before the gpu-vram parser is reached.
for i in "${!BINS[@]}"; do
    name=${NAMES[$i]}; bin=${BINS[$i]}
    [ -x "$bin" ] || continue
    if [ "$name" = "ds4-bench" ]; then
        "$bin" --gpu-vram abc -m /dev/null --prompt-file /dev/null > "$LOG" 2>&1
    else
        "$bin" --gpu-vram abc -m /dev/null > "$LOG" 2>&1
    fi
    rc=$?
    if [ $rc -eq 0 ]; then
        fail "$name --gpu-vram abc should exit non-zero (got 0)"
    else
        ok "$name --gpu-vram abc exits non-zero ($rc)"
    fi
    # Confirm the shared value parser was reached, not merely the binary's
    # unknown-option fallback.
    if grep -q -- "--gpu-vram: not a number" "$LOG" 2>/dev/null &&
       ! grep -q "unknown option" "$LOG" 2>/dev/null; then
        ok "$name --gpu-vram abc reaches shared parser"
    else
        fail "$name --gpu-vram abc did not reach shared parser"
        head -10 "$LOG" | sed 's/^/    /'
    fi
done

# 3: count mismatch.
for i in "${!BINS[@]}"; do
    name=${NAMES[$i]}; bin=${BINS[$i]}
    [ -x "$bin" ] || continue
    if [ "$name" = "ds4-bench" ]; then
        "$bin" --gpu-vram 40,12 --gpu-devices 0 -m /dev/null \
            --prompt-file /dev/null > "$LOG" 2>&1
    else
        "$bin" --gpu-vram 40,12 --gpu-devices 0 -m /dev/null > "$LOG" 2>&1
    fi
    rc=$?
    if [ $rc -ne 0 ] &&
       grep -q -- "--gpu-devices count (1) does not match --gpu-vram count (2)" "$LOG" &&
       ! grep -q "unknown option" "$LOG"; then
        ok "$name count-mismatch reaches shared parser ($rc)"
    else
        fail "$name count-mismatch did not reach shared parser"
        head -10 "$LOG" | sed 's/^/    /'
    fi
done

# 4: --cuda --help still works (the flag alone shouldn't break parsing).
for i in "${!BINS[@]}"; do
    name=${NAMES[$i]}; bin=${BINS[$i]}
    [ -x "$bin" ] || continue
    "$bin" --cuda --help > "$LOG" 2>&1 || true
    # Servers may print a usage banner; check help still surfaced.
    if grep -qE "Usage:|usage:|--help" "$LOG"; then
        ok "$name --cuda --help still prints help"
    else
        fail "$name --cuda --help did not print help text"
    fi
done

# 5: --gpu-vram 0 short-circuit. We use ds4 (CLI) specifically because
# it produces predictable stdout/stderr.
if [ -x ./ds4 ]; then
    ./ds4 --gpu-vram 0 -m /dev/null > "$LOG" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        ok "ds4 --gpu-vram 0 exits non-zero (expected: model-load fail)"
    else
        fail "ds4 --gpu-vram 0 returned 0 — unexpected"
    fi
    # The layout line must NOT appear (short-circuit happens before).
    if grep -q "GPU config:" "$LOG"; then
        fail "ds4 --gpu-vram 0 should NOT print GPU layout line"
        head -10 "$LOG" | sed 's/^/    /'
    else
        ok "ds4 --gpu-vram 0 does not print GPU layout (short-circuit reached)"
    fi
fi

# 6: tensor parallelism reuses the distributed role and address options, but
# owns the split and therefore rejects --layers.
if [ -x ./ds4 ]; then
    ./ds4 --metal --tensor-parallel --role coordinator --listen 127.0.0.1 9911 \
        --layers 0:1 -m /dev/null > "$LOG" 2>&1
    rc=$?
    if [ $rc -ne 0 ] && grep -q "always uses one 50/50 worker" "$LOG"; then
        ok "tensor parallel rejects explicit layer slices"
    else
        fail "tensor parallel accepted --layers or returned the wrong error"
    fi

    ./ds4 --tensor-parallel --role worker -m /dev/null > "$LOG" 2>&1
    rc=$?
    if [ $rc -ne 0 ] && grep -q "requires --coordinator HOST PORT" "$LOG"; then
        ok "tensor-parallel worker requires coordinator address"
    else
        fail "tensor-parallel worker returned the wrong missing-address error"
    fi

    ./ds4 --metal --tensor-parallel --role coordinator --listen 127.0.0.1 9911 \
        --transport tcp --tensor-parallel-token-prefill --debug-hash 2 \
        --rdma-device rdma-test --rdma-gid-index 0 \
        --inspect -m /dev/null > "$LOG" 2>&1
    rc=$?
    if [ $rc -ne 0 ] &&
       grep -qE "model file is too small|another ds4 process is already running" "$LOG" &&
       ! grep -q "requires --layers" "$LOG"; then
        ok "tensor-parallel common options reach model loading"
    else
        fail "tensor-parallel common options did not reach model loading"
    fi

    for old_arg in \
        "--tp-coordinator 9911" \
        "--tp-lead 9911" \
        "--tp-coordinator-host 127.0.0.1" \
        "--tp-lead-host 127.0.0.1" \
        "--tp-worker 127.0.0.1 9911" \
        "--tp-transport tcp" \
        "--tp-debug-hash 2" \
        "--tp-token-prefill"
    do
        # Word splitting is intentional: each item contains one old option
        # and its former arguments.
        ./ds4 $old_arg -m /dev/null > "$LOG" 2>&1
        rc=$?
        if [ $rc -ne 0 ] && grep -q "unknown option" "$LOG"; then
            ok "obsolete ${old_arg%% *} is rejected"
        else
            fail "obsolete ${old_arg%% *} was not rejected"
        fi
    done
fi

# 7: --gpu-vram 40,12 layout line.
if [ -x ./ds4 ]; then
    ./ds4 --gpu-vram 40,12 -m /dev/null > "$LOG" 2>&1
    rc=$?
    if grep -q "GPU config: 2 devices \[0,1\] requested, budgets 40,12 GB" "$LOG"; then
        ok "ds4 --gpu-vram 40,12 prints expected layout line"
    else
        fail "ds4 --gpu-vram 40,12 missing or malformed layout line"
        head -10 "$LOG" | sed 's/^/    /'
    fi
fi

# 8: compact-runtime canonical options are shared by every inference binary.
INFERENCE_BINS=(./ds4 ./ds4-server ./ds4-bench ./ds4-agent ./ds4-eval)
INFERENCE_NAMES=(ds4 ds4-server ds4-bench ds4-agent ds4-eval)

run_inference_option() {
    local name=$1
    local bin=$2
    shift 2
    if [ "$name" = "ds4-bench" ]; then
        local planning=false
        local arg
        for arg in "$@"; do
            [ "$arg" = "--qualification-plan" ] && planning=true
        done
        if [ "$planning" = true ]; then
            "$bin" "$@" -m /dev/null > "$LOG" 2>&1
        else
            "$bin" "$@" -m /dev/null --prompt-file /dev/null > "$LOG" 2>&1
        fi
    else
        "$bin" "$@" -m /dev/null > "$LOG" 2>&1
    fi
}

assert_before_model_open() {
    local name=$1
    local expected=$2
    local rc=$3
    if [ "$rc" -eq 2 ] && grep -q -- "$expected" "$LOG" &&
       ! grep -qE "model file is too small|failed to open model|another ds4 process" "$LOG"; then
        ok "$name"
    else
        fail "$name (expected config exit 2 before model open, got $rc)"
        sed -n '1,12p' "$LOG" | sed 's/^/    /'
    fi
}

assert_reaches_model_open() {
    local name=$1
    local rc=$2
    if [ "$rc" -ne 0 ] &&
       grep -qE "model file is too small|failed to open model|another ds4 process" "$LOG" &&
       ! grep -qE "unknown option|must be canonical|cannot be combined|conflicts with" "$LOG"; then
        ok "$name"
    else
        fail "$name (canonical option did not reach model open, got $rc)"
        sed -n '1,12p' "$LOG" | sed 's/^/    /'
    fi
}

for i in "${!INFERENCE_BINS[@]}"; do
    name=${INFERENCE_NAMES[$i]}; bin=${INFERENCE_BINS[$i]}
    if [ ! -x "$bin" ]; then
        fail "$name not built — skipping compact-runtime option checks"
        continue
    fi

    "$bin" --help > "$LOG" 2>&1 || true
    assert_grep "$name --help mentions --ssd-streaming-cache-bytes" \
        "--ssd-streaming-cache-bytes BYTES" "$LOG"

    if [ "$name" = "ds4-bench" ]; then
        run_inference_option "$name" "$bin" \
            --ctx-start 128 --ctx-max 128 --ctx-alloc 1024 \
            --prefill-chunk 1 \
            --ssd-streaming --ssd-streaming-cache-bytes 8589934592
    else
        run_inference_option "$name" "$bin" \
            --ctx 1024 --prefill-chunk 1 \
            --ssd-streaming --ssd-streaming-cache-bytes 8589934592
    fi
    rc=$?
    assert_reaches_model_open "$name accepts canonical cache bytes" "$rc"

    run_inference_option "$name" "$bin" \
        --ssd-streaming --ssd-streaming-cache-experts 8GB
    rc=$?
    assert_reaches_model_open "$name retains legacy cache spelling" "$rc"

    run_inference_option "$name" "$bin" \
        --ssd-streaming --ssd-streaming-cache-bytes 08
    rc=$?
    assert_before_model_open "$name rejects non-canonical cache bytes" \
        "--ssd-streaming-cache-bytes must be canonical" "$rc"

    run_inference_option "$name" "$bin" \
        --ssd-streaming --ssd-streaming-cache-bytes 8589934592 \
        --ssd-streaming-cache-experts 8GB
    rc=$?
    assert_before_model_open "$name rejects canonical/legacy cache mix" \
        "--ssd-streaming-cache-bytes cannot be combined with --ssd-streaming-cache-experts" "$rc"

    run_inference_option "$name" "$bin" \
        --ssd-streaming --ssd-streaming-cache-experts 8GB \
        --ssd-streaming-cache-bytes 8589934592
    rc=$?
    assert_before_model_open "$name rejects legacy/canonical cache mix" \
        "--ssd-streaming-cache-bytes cannot be combined with --ssd-streaming-cache-experts" "$rc"

    run_inference_option "$name" "$bin" \
        --ssd-streaming --ssd-streaming-cache-bytes 18446744073709551615
    rc=$?
    assert_before_model_open "$name rejects impossible exact cache bytes" \
        "--ssd-streaming-cache-bytes is impossible" "$rc"
done

# 9: diagnostic early exits must still pass the shared engine-option preflight.
# /dev/null proves the typed configuration error happened before tokenizer/model
# access; this exact reproduction previously returned the model-open class (1).
if [ -x ./ds4 ]; then
    ./ds4 --dump-tokens -p x -m /dev/null \
        --ssd-streaming \
        --ssd-streaming-cache-bytes 18446744073709551615 \
        > "$LOG" 2>&1
    rc=$?
    assert_before_model_open \
        "ds4 --dump-tokens rejects impossible exact cache before model access" \
        "--ssd-streaming-cache-bytes is impossible" "$rc"
fi

# 10: exact graph-cache pricing requires a declared context, and this first
# qualified compact profile is deliberately single-session until Task 5/14
# account for concurrent graph/KV state.
if [ -x ./ds4-eval ]; then
    run_inference_option ds4-eval ./ds4-eval \
        --cuda --ssd-streaming --ssd-streaming-cache-bytes 8589934592
    rc=$?
    assert_before_model_open \
        "ds4-eval exact graph cache rejects auto context sizing" \
        "--ctx" "$rc"
fi

if [ -x ./ds4 ]; then
    run_inference_option ds4 ./ds4 \
        --metal-graph-prompt-test --ctx 32768 \
        --ssd-streaming --ssd-streaming-cache-bytes 8589934592
    rc=$?
    assert_before_model_open \
        "ds4 exact graph cache rejects direct-allocation diagnostics" \
        "graph diagnostics" "$rc"
fi

if [ -x ./ds4-server ]; then
    run_inference_option ds4-server ./ds4-server \
        --cuda --ctx 32768 --session-slots 1 \
        --ssd-streaming --ssd-streaming-cache-bytes 8589934592
    rc=$?
    if [ "$rc" -ne 0 ] &&
       ! grep -q -- "shared prefill workspace" "$LOG" &&
       ! grep -q -- "--session-slots 1" "$LOG" &&
       grep -qE "model file is too small|failed to open model|another ds4 process|working-set limit is unavailable" "$LOG"; then
        ok "ds4-server exact graph cache accepts one declared session"
    else
        fail "ds4-server exact one-session declaration was refused (got $rc)"
        sed -n '1,12p' "$LOG" | sed 's/^/    /'
    fi

    run_inference_option ds4-server ./ds4-server \
        --cuda --session-slots 2 \
        --ssd-streaming --ssd-streaming-cache-bytes 8589934592
    rc=$?
    assert_before_model_open \
        "ds4-server exact graph cache rejects multiple session slots" \
        "--session-slots" "$rc"

    run_inference_option ds4-server ./ds4-server \
        --cuda --batched-session 2 \
        --ssd-streaming --ssd-streaming-cache-bytes 8589934592
    rc=$?
    assert_before_model_open \
        "ds4-server exact graph cache rejects legacy multi-session alias" \
        "--session-slots" "$rc"

    run_inference_option ds4-server ./ds4-server \
        --cuda --session-slots 2 \
        --ssd-streaming --ssd-streaming-cache-experts 8GB
    rc=$?
    assert_reaches_model_open \
        "ds4-server legacy cache preserves multiple session slots" "$rc"

    run_inference_option ds4-server ./ds4-server \
        --cuda --ctx 32768 --session-slots 1 \
        --gpu-vram 24,24 --gpu-devices 0,1 \
        --ssd-streaming --ssd-streaming-cache-bytes 17179869184
    rc=$?
    assert_before_model_open \
        "ds4-server exact graph cache rejects aggregate multi-GPU budgets" \
        "one CUDA device" "$rc"

    run_inference_option ds4-server ./ds4-server \
        --cuda --ctx 32768 --session-slots 1 \
        --gpu-vram 128 --gpu-devices 3 \
        --ssd-streaming --ssd-streaming-cache-bytes 17179869184
    rc=$?
    assert_before_model_open \
        "ds4-server exact graph cache rejects nonzero single CUDA device" \
        "CUDA device 0" "$rc"
fi

# 11: --session-slots is canonical; --batched-session is an equal-value alias.
if [ -x ./ds4-server ]; then
    ./ds4-server --help > "$LOG" 2>&1 || true
    assert_grep "ds4-server --help mentions --session-slots" \
        "--session-slots N" "$LOG"

    run_inference_option ds4-server ./ds4-server --session-slots 1
    rc=$?
    assert_reaches_model_open "ds4-server accepts --session-slots" "$rc"

    run_inference_option ds4-server ./ds4-server --batched-session 1
    rc=$?
    assert_reaches_model_open "ds4-server retains --batched-session alias" "$rc"

    run_inference_option ds4-server ./ds4-server \
        --session-slots 1 --batched-session 1
    rc=$?
    assert_reaches_model_open "ds4-server accepts equal session-slot aliases" "$rc"

    run_inference_option ds4-server ./ds4-server \
        --session-slots 1 --batched-session 2
    rc=$?
    assert_before_model_open "ds4-server rejects conflicting session-slot aliases" \
        "--session-slots conflicts with --batched-session" "$rc"

    run_inference_option ds4-server ./ds4-server \
        --batched-session 2 --session-slots 1
    rc=$?
    assert_before_model_open "ds4-server rejects reverse session-slot conflict" \
        "--session-slots conflicts with --batched-session" "$rc"
fi

# 12: --qualification-plan is a harness-only option shared by all five
# inference frontends. It is intentionally absent from public help, accepts
# exactly one non-empty path, and rejects malformed provenance before opening
# the model.
assert_qualification_error_before_model_open() {
    local name=$1
    local expected=$2
    local rc=$3
    if [ "$rc" -eq 2 ] && grep -qE -- "$expected" "$LOG" &&
       ! grep -qE "unknown option|model file is too small|failed to open model|another ds4 process" "$LOG"; then
        ok "$name"
    else
        fail "$name (expected qualification-plan exit 2 before model open, got $rc)"
        sed -n '1,12p' "$LOG" | sed 's/^/    /'
    fi
}

PLAN_DIR=$(mktemp -d)
PLAN_PATH="$PLAN_DIR/plan.json"
PLAN_LOCK="$PLAN_DIR/missing/plan.lock"

run_reference_plan_option() {
    local name=$1
    local bin=$2
    local path=$3
    local cache_bytes=$4
    shift 4
    if [ "$name" = "ds4-bench" ]; then
        DS4_LOCK_FILE="$PLAN_LOCK" "$bin" \
            --cuda --ctx-alloc 32768 --prefill-chunk 4096 \
            --ssd-streaming --ssd-streaming-cache-bytes "$cache_bytes" \
            --qualification-plan "$path" "$@" \
            -m /dev/null > "$LOG" 2>&1
    else
        DS4_LOCK_FILE="$PLAN_LOCK" "$bin" \
            --cuda --ctx 32768 --prefill-chunk 4096 \
            --ssd-streaming --ssd-streaming-cache-bytes "$cache_bytes" \
            --qualification-plan "$path" "$@" \
            -m /dev/null > "$LOG" 2>&1
    fi
}

assert_plan_reaches_model_validation_without_runtime_side_effects() {
    local name=$1
    local rc=$2
    if [ "$rc" -eq 2 ] &&
       grep -qE "model file is too small|failed to open model" "$LOG" &&
       ! grep -qE "working-set limit|another ds4 process|failed to open lock file|oom_score_adj|GPU config:|unknown option|not implemented" "$LOG"; then
        ok "$name"
    else
        fail "$name (plan-only path did not reach model validation cleanly, got $rc)"
        sed -n '1,12p' "$LOG" | sed 's/^/    /'
    fi
}

for i in "${!INFERENCE_BINS[@]}"; do
    name=${INFERENCE_NAMES[$i]}; bin=${INFERENCE_BINS[$i]}
    [ -x "$bin" ] || continue

    "$bin" --help > "$LOG" 2>&1 || true
    assert_not_grep "$name --help hides --qualification-plan" \
        "--qualification-plan" "$LOG"

    "$bin" --qualification-plan > "$LOG" 2>&1
    rc=$?
    assert_qualification_error_before_model_open \
        "$name rejects missing qualification-plan path" \
        "missing value for --qualification-plan|--qualification-plan requires an argument" \
        "$rc"

    run_inference_option "$name" "$bin" --qualification-plan ""
    rc=$?
    assert_qualification_error_before_model_open \
        "$name rejects empty qualification-plan path" \
        "--qualification-plan requires a non-empty path" "$rc"

    run_inference_option "$name" "$bin" \
        --qualification-plan /tmp/ds4-qualification-plan.json \
        --qualification-plan /tmp/ds4-qualification-plan.json
    rc=$?
    assert_qualification_error_before_model_open \
        "$name rejects identical duplicate qualification-plan paths" \
        "--qualification-plan may only be specified once" "$rc"

    run_inference_option "$name" "$bin" \
        --qualification-plan /tmp/ds4-qualification-plan-a.json \
        --qualification-plan /tmp/ds4-qualification-plan-b.json
    rc=$?
    assert_qualification_error_before_model_open \
        "$name rejects conflicting qualification-plan paths" \
        "--qualification-plan may only be specified once" "$rc"

    run_reference_plan_option "$name" "$bin" "$PLAN_PATH" 8589934592
    rc=$?
    assert_plan_reaches_model_validation_without_runtime_side_effects \
        "$name reference qualification plan skips runtime startup" "$rc"

    run_reference_plan_option "$name" "$bin" "$PLAN_PATH" 10737418240
    rc=$?
    assert_qualification_error_before_model_open \
        "$name rejects an unfrozen qualification cache profile" \
        "8, 12, or 16 GiB|8/12/16" "$rc"
done

if [ -x ./ds4 ]; then
    run_reference_plan_option ds4 ./ds4 "$PLAN_PATH" 8589934592 --raw
    rc=$?
    assert_qualification_error_before_model_open \
        "ds4 qualification planning rejects ignored CLI options" \
        "qualification-plan.*(cannot be combined|exclusive)" "$rc"
fi
if [ -x ./ds4-server ]; then
    run_reference_plan_option ds4-server ./ds4-server "$PLAN_PATH" 8589934592 --chdir /
    rc=$?
    assert_qualification_error_before_model_open \
        "ds4-server qualification planning rejects ignored service options" \
        "qualification-plan.*(cannot be combined|exclusive)" "$rc"
fi
if [ -x ./ds4-bench ]; then
    run_reference_plan_option ds4-bench ./ds4-bench "$PLAN_PATH" 8589934592 --gen-tokens 1
    rc=$?
    assert_qualification_error_before_model_open \
        "ds4-bench qualification planning rejects benchmark options" \
        "qualification-plan.*(cannot be combined|exclusive)" "$rc"
fi
if [ -x ./ds4-agent ]; then
    run_reference_plan_option ds4-agent ./ds4-agent "$PLAN_PATH" 8589934592 --chdir /
    rc=$?
    assert_qualification_error_before_model_open \
        "ds4-agent qualification planning rejects ignored agent options" \
        "qualification-plan.*(cannot be combined|exclusive)" "$rc"
fi
if [ -x ./ds4-eval ]; then
    run_reference_plan_option ds4-eval ./ds4-eval "$PLAN_PATH" 8589934592 --self-test-extractors
    rc=$?
    assert_qualification_error_before_model_open \
        "ds4-eval qualification planning rejects ignored eval modes" \
        "qualification-plan.*(cannot be combined|exclusive)" "$rc"

    DS4_LOCK_FILE="$PLAN_LOCK" ./ds4-eval \
        --cuda --ctx 32768 --prefill-chunk 4096 \
        --ssd-streaming --ssd-streaming-cache-bytes 8589934592 \
        --qualification-plan="$PLAN_PATH" --self-test-extractors \
        -m /dev/null > "$LOG" 2>&1
    rc=$?
    assert_qualification_error_before_model_open \
        "inline qualification path cannot bypass eval option isolation" \
        "qualification-plan.*(cannot be combined|exclusive)" "$rc"
fi

if [ -x ./ds4 ]; then
    run_reference_plan_option ds4 ./ds4 "$PLAN_DIR/missing/plan.json" 8589934592
    rc=$?
    assert_qualification_error_before_model_open \
        "ds4 rejects a missing qualification-plan parent before model access" \
        "qualification-plan.*(parent|directory)|parent.*(missing|exist)" "$rc"

    run_reference_plan_option ds4 ./ds4 "$PLAN_PATH" 8589934592 --warm-weights
    rc=$?
    assert_qualification_error_before_model_open \
        "ds4 rejects warm scanning in qualification-plan mode" \
        "qualification-plan.*warm|warm.*qualification-plan" "$rc"

    DS4_LOCK_FILE="$PLAN_LOCK" ./ds4 \
        --cuda --ctx 32768 --prefill-chunk 4096 \
        --ssd-streaming --ssd-streaming-cache-bytes 8589934592 \
        --qualification-plan "$PLAN_PATH" --gpu-vram auto \
        -m /dev/null > "$LOG" 2>&1
    rc=$?
    assert_qualification_error_before_model_open \
        "ds4 rejects GPU-layout probes in qualification-plan mode" \
        "qualification-plan.*gpu-vram|gpu-vram.*qualification-plan" "$rc"
fi

rmdir "$PLAN_DIR"

rm -f "$LOG"

echo ""
echo "test_gpu_args_cli: PASS=$PASS FAIL=$FAIL"
if [ $FAIL -gt 0 ]; then
    exit 1
fi
exit 0
