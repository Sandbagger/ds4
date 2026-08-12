#!/usr/bin/env bash
set -euo pipefail

runner_dir=$(CDPATH='' cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(CDPATH='' cd -- "$runner_dir/.." && pwd)
verifier="$repo_root/gguf-tools/quality-testing/compare_laguna_logits.py"
verifier_module_dir="$repo_root/gguf-tools/quality-testing"
fixture="$repo_root/tests/test-vectors/laguna-resident"

pinned_size=68248759648
pinned_sha256=e163b2c98908809a71245d6bb68b2226994d9969cb2a438eccb72196a1c4147a
pinned_contract=a250e43722945e293f6044bc7254c4806d5a7912
pinned_llama=04b2b72cb54048ead292884adbe11f284e3ec950
pinned_capture=cc4fb338556c0895ff985edb5435ae7801be7dcb98c2958dc96a56d34f2c848e
production_lock_path=/tmp/ds4.lock

die() {
    printf 'laguna CUDA gate: %s\n' "$*" >&2
    exit 2
}

assert_production_lock_available() {
    python3 -c '
import fcntl
import os
import sys

path = sys.argv[1]
fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
try:
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(1)
finally:
    os.close(fd)
' "$production_lock_path" || \
        die "production instance lock is already held: $production_lock_path"
}

require_present() {
    local name=$1
    local present=$2
    [ "$present" = present ] || die "$name is required in self-test mode"
}

fd_identity() {
    python3 -c \
        'import os; s = os.fstat(9); o = os.lseek(9, 0, os.SEEK_CUR); print(f"{s.st_dev}:{s.st_ino}:{s.st_size}:{s.st_mtime_ns}:{o}")'
}

assert_retained_identity() {
    local child=$1
    local current
    current=$(fd_identity) || die "cannot inspect retained descriptor after $child"
    if [ "$current" != "$retained_identity" ]; then
        die "retained descriptor identity changed after $child"
    fi
}

cold_prepare_retained_fd() {
    local child=$1
    python3 -c '
import os
if not hasattr(os, "posix_fadvise"):
    raise SystemExit(0)
size = os.fstat(9).st_size
os.posix_fadvise(9, 0, size, os.POSIX_FADV_DONTNEED)
' || die "cannot cold-prepare retained descriptor before $child"
    assert_retained_identity "coldprep-$child"
}

cold_prepare_exact_fd() {
    local child=$1
    assert_production_lock_available
    DS4_LAGUNA_COLD_PREP_LABEL=$child python3 -c '
import sys
sys.path.insert(0, sys.argv[1])
from compact_runtime_qualify import cold_prepare_descriptor_from_plan
try:
    cold_prepare_descriptor_from_plan(9, sys.argv[2], sys.argv[3])
except (OSError, ValueError) as exc:
    print(f"descriptor cold preparation failed: {exc}", file=sys.stderr)
    raise SystemExit(1)
' "$verifier_module_dir" "$qualification_plan" \
        "$qualification_plan_sha256" || \
        die "cannot cold-prepare retained descriptor before $child"
    assert_retained_identity "coldprep-$child"
}

hash_retained_fd() {
    python3 -c '
import sys
sys.path.insert(0, sys.argv[1])
from compare_laguna_logits import ContractError, verify_gguf_fd
try:
    verify_gguf_fd(9, int(sys.argv[2]), sys.argv[3])
except ContractError as exc:
    print(f"retained GGUF verification failed: {exc}", file=sys.stderr)
    raise SystemExit(1)
' "$verifier_module_dir" "$expected_size" "$expected_sha256"
}

if [ "$#" -ne 1 ]; then
    die "usage: tests/run_cuda_laguna_gate.sh resident|streaming|c7|self-test"
fi

mode=$1
case "$mode" in
    resident)
        if [ "${DS4_LAGUNA_GATE_TEST_CHILD_DIR+present}" = present ]; then
            die "DS4_LAGUNA_GATE_TEST_CHILD_DIR is forbidden in resident mode"
        fi
        if [ "${DS4_LAGUNA_GATE_TEST_EXPECTED_SIZE+present}" = present ]; then
            die "DS4_LAGUNA_GATE_TEST_EXPECTED_SIZE is forbidden in resident mode"
        fi
        if [ "${DS4_LAGUNA_GATE_TEST_EXPECTED_SHA256+present}" = present ]; then
            die "DS4_LAGUNA_GATE_TEST_EXPECTED_SHA256 is forbidden in resident mode"
        fi
        [ -n "${DS4_TEST_MODEL:-}" ] || die "DS4_TEST_MODEL is required"
        [ -f "$DS4_TEST_MODEL" ] || die "DS4_TEST_MODEL is not a regular file: $DS4_TEST_MODEL"
        [ -n "${LAGUNA_TOKENIZER_RUNTIME_COMMIT:-}" ] || \
            die "LAGUNA_TOKENIZER_RUNTIME_COMMIT is required"
        model=$DS4_TEST_MODEL
        expected_size=$pinned_size
        expected_sha256=$pinned_sha256
        kernel_child="$repo_root/tests/test_cuda_laguna_kernels"
        model_child="$repo_root/tests/test_cuda_laguna_model"
        [ -x "$kernel_child" ] || die "missing executable: $kernel_child"
        [ -x "$model_child" ] || die "missing executable: $model_child"
        ;;
    c7)
        if [ "${DS4_LAGUNA_GATE_TEST_CHILD_DIR+present}" = present ]; then
            die "DS4_LAGUNA_GATE_TEST_CHILD_DIR is forbidden in c7 mode"
        fi
        if [ "${DS4_LAGUNA_GATE_TEST_EXPECTED_SIZE+present}" = present ]; then
            die "DS4_LAGUNA_GATE_TEST_EXPECTED_SIZE is forbidden in c7 mode"
        fi
        if [ "${DS4_LAGUNA_GATE_TEST_EXPECTED_SHA256+present}" = present ]; then
            die "DS4_LAGUNA_GATE_TEST_EXPECTED_SHA256 is forbidden in c7 mode"
        fi
        [ -n "${DS4_TEST_MODEL:-}" ] || die "DS4_TEST_MODEL is required"
        if [ "$DS4_TEST_MODEL" = ds4flash.gguf ]; then
            die "DS4_TEST_MODEL must be an explicit path, not ds4flash.gguf"
        fi
        [[ $DS4_TEST_MODEL = /* ]] || \
            die "DS4_TEST_MODEL must be an absolute path: $DS4_TEST_MODEL"
        [ -f "$DS4_TEST_MODEL" ] || die "DS4_TEST_MODEL is not a regular file: $DS4_TEST_MODEL"
        [ -n "${LAGUNA_TOKENIZER_RUNTIME_COMMIT:-}" ] || \
            die "LAGUNA_TOKENIZER_RUNTIME_COMMIT is required"
        if [[ ! $LAGUNA_TOKENIZER_RUNTIME_COMMIT =~ ^[0-9a-f]{40}$ ]]; then
            die "LAGUNA_TOKENIZER_RUNTIME_COMMIT must be 40 lowercase hexadecimal characters"
        fi
        model=$DS4_TEST_MODEL
        expected_size=$pinned_size
        expected_sha256=$pinned_sha256
        kernel_child="$repo_root/tests/test_cuda_laguna_kernels"
        model_child="$repo_root/tests/test_cuda_laguna_model"
        stream_child="$repo_root/tests/test_cuda_laguna_stream"
        [ -x "$kernel_child" ] || die "missing executable: $kernel_child"
        [ -x "$model_child" ] || die "missing executable: $model_child"
        [ -x "$stream_child" ] || die "missing executable: $stream_child"
        ;;
    streaming)
        if [ "${DS4_LAGUNA_GATE_TEST_CHILD_DIR+present}" = present ]; then
            die "DS4_LAGUNA_GATE_TEST_CHILD_DIR is forbidden in streaming mode"
        fi
        if [ "${DS4_LAGUNA_GATE_TEST_EXPECTED_SIZE+present}" = present ]; then
            die "DS4_LAGUNA_GATE_TEST_EXPECTED_SIZE is forbidden in streaming mode"
        fi
        if [ "${DS4_LAGUNA_GATE_TEST_EXPECTED_SHA256+present}" = present ]; then
            die "DS4_LAGUNA_GATE_TEST_EXPECTED_SHA256 is forbidden in streaming mode"
        fi
        if [ "${DS4_LOCK_FILE+present}" = present ]; then
            die "DS4_LOCK_FILE is forbidden in streaming mode"
        fi
        [ -n "${DS4_TEST_MODEL:-}" ] || die "DS4_TEST_MODEL is required"
        if [ "$DS4_TEST_MODEL" = ds4flash.gguf ]; then
            die "DS4_TEST_MODEL must be an explicit path, not ds4flash.gguf"
        fi
        [[ $DS4_TEST_MODEL = /* ]] || \
            die "DS4_TEST_MODEL must be an absolute path: $DS4_TEST_MODEL"
        [ -f "$DS4_TEST_MODEL" ] || \
            die "DS4_TEST_MODEL is not a regular file: $DS4_TEST_MODEL"
        [ -n "${LAGUNA_TOKENIZER_RUNTIME_COMMIT:-}" ] || \
            die "LAGUNA_TOKENIZER_RUNTIME_COMMIT is required"
        if [[ ! $LAGUNA_TOKENIZER_RUNTIME_COMMIT =~ ^[0-9a-f]{40}$ ]]; then
            die "LAGUNA_TOKENIZER_RUNTIME_COMMIT must be 40 lowercase hexadecimal characters"
        fi
        [ -n "${DS4_QUALIFICATION_PLAN:-}" ] || \
            die "DS4_QUALIFICATION_PLAN is required"
        [[ $DS4_QUALIFICATION_PLAN = /* ]] || \
            die "DS4_QUALIFICATION_PLAN must be an absolute path: $DS4_QUALIFICATION_PLAN"
        [ -f "$DS4_QUALIFICATION_PLAN" ] || \
            die "DS4_QUALIFICATION_PLAN is not a regular file: $DS4_QUALIFICATION_PLAN"
        if [[ ! ${DS4_QUALIFICATION_PLAN_SHA256:-} =~ ^[0-9a-f]{64}$ ]]; then
            die "DS4_QUALIFICATION_PLAN_SHA256 must be 64 lowercase hexadecimal characters"
        fi
        model=$DS4_TEST_MODEL
        qualification_plan=$DS4_QUALIFICATION_PLAN
        qualification_plan_sha256=$DS4_QUALIFICATION_PLAN_SHA256
        expected_size=$pinned_size
        expected_sha256=$pinned_sha256
        model_child="$repo_root/tests/test_cuda_laguna_model"
        stream_child="$repo_root/tests/test_cuda_laguna_stream"
        qualifier="$verifier_module_dir/compact_runtime_qualify.py"
        [ -x "$model_child" ] || die "missing executable: $model_child"
        [ -x "$stream_child" ] || die "missing executable: $stream_child"
        [ -f "$qualifier" ] || die "missing qualifier module: $qualifier"
        ;;
    self-test)
        require_present DS4_LAGUNA_GATE_TEST_CHILD_DIR \
            "${DS4_LAGUNA_GATE_TEST_CHILD_DIR+present}"
        require_present DS4_LAGUNA_GATE_TEST_EXPECTED_SIZE \
            "${DS4_LAGUNA_GATE_TEST_EXPECTED_SIZE+present}"
        require_present DS4_LAGUNA_GATE_TEST_EXPECTED_SHA256 \
            "${DS4_LAGUNA_GATE_TEST_EXPECTED_SHA256+present}"
        [ -n "${DS4_TEST_MODEL:-}" ] || die "DS4_TEST_MODEL is required"
        [ -f "$DS4_TEST_MODEL" ] || die "DS4_TEST_MODEL is not a regular file: $DS4_TEST_MODEL"
        if [[ ! $DS4_LAGUNA_GATE_TEST_EXPECTED_SIZE =~ ^[1-9][0-9]*$ ]]; then
            die "DS4_LAGUNA_GATE_TEST_EXPECTED_SIZE must be canonical positive decimal"
        fi
        if [[ ! $DS4_LAGUNA_GATE_TEST_EXPECTED_SHA256 =~ ^[0-9a-f]{64}$ ]]; then
            die "DS4_LAGUNA_GATE_TEST_EXPECTED_SHA256 must be 64 lowercase hexadecimal characters"
        fi
        child_dir=$DS4_LAGUNA_GATE_TEST_CHILD_DIR
        [ -d "$child_dir" ] || die "DS4_LAGUNA_GATE_TEST_CHILD_DIR is not a directory: $child_dir"
        model=$DS4_TEST_MODEL
        expected_size=$DS4_LAGUNA_GATE_TEST_EXPECTED_SIZE
        expected_sha256=$DS4_LAGUNA_GATE_TEST_EXPECTED_SHA256
        verifier="$child_dir/compare_laguna_logits.py"
        kernel_child="$child_dir/test_cuda_laguna_kernels"
        model_child="$child_dir/test_cuda_laguna_model"
        [ -x "$verifier" ] || die "missing executable: $verifier"
        [ -x "$kernel_child" ] || die "missing executable: $kernel_child"
        [ -x "$model_child" ] || die "missing executable: $model_child"
        ;;
    *)
        die "unknown mode: $mode"
        ;;
esac

if [ "$mode" = streaming ]; then
    assert_production_lock_available
fi

cd "$repo_root"
exec 9<"$model"
export DS4_TEST_MODEL_FD=9
retained_identity=$(fd_identity) || die "cannot inspect retained descriptor"

if [ "$mode" = self-test ]; then
    hash_retained_fd
    "$verifier" \
        --gguf-fd 9 \
        --gguf-size "$expected_size" \
        --gguf-sha256 "$expected_sha256"
elif [ "$mode" = c7 ] || [ "$mode" = streaming ]; then
    timeout --kill-after=5s 180s python3 "$verifier" \
        --verify-promoted "$fixture" \
        --contract-commit "$pinned_contract" \
        --tokenizer-runtime-commit "$LAGUNA_TOKENIZER_RUNTIME_COMMIT" \
        --llama-commit "$pinned_llama" \
        --capture-manifest-sha256 "$pinned_capture" \
        --gguf-size "$expected_size" \
        --gguf-sha256 "$expected_sha256" \
        --gguf-fd 9
else
    python3 "$verifier" \
        --verify-promoted "$fixture" \
        --contract-commit "$pinned_contract" \
        --tokenizer-runtime-commit "$LAGUNA_TOKENIZER_RUNTIME_COMMIT" \
        --llama-commit "$pinned_llama" \
        --capture-manifest-sha256 "$pinned_capture" \
        --gguf-size "$expected_size" \
        --gguf-sha256 "$expected_sha256" \
        --gguf-fd 9
fi
assert_retained_identity verifier

if [ "$mode" = streaming ]; then
    stream_cases=(
        startup
        cache-validation
        cache-io
        cache-faults
        cache-unsafe
        cache-unsafe-race
        prefill-allocation
        page-advice
        create-unwind-unsafe
        teardown-unsafe
        session-pressure
    )
    for stream_case in "${stream_cases[@]}"; do
        timeout --kill-after=5s 60s "$stream_child" --case "$stream_case"
        assert_retained_identity "stream-$stream_case"
    done

    cold_prepare_exact_fd model-streamed-short
    timeout --kill-after=5s 900s "$model_child" \
        --mode streamed --case short
    assert_retained_identity model-streamed-short

    cold_prepare_exact_fd model-streamed-prefill-8192
    timeout --kill-after=5s 1800s "$model_child" \
        --mode streamed --case prefill-8192
    assert_retained_identity model-streamed-prefill-8192

    cold_prepare_exact_fd model-streamed-warm-stability
    timeout --kill-after=5s 2700s "$model_child" \
        --mode streamed --case warm-stability
    assert_retained_identity model-streamed-warm-stability

    hash_retained_fd
    exit 0
fi

if [ "$mode" = c7 ]; then
    env -u DS4_CUDA_MOE_DECODE_GRAPH \
        timeout --kill-after=5s 60s "$kernel_child" --case all
else
    env -u DS4_CUDA_MOE_DECODE_GRAPH "$kernel_child" --case all
fi
assert_retained_identity kernels

if [ "$mode" = c7 ]; then
    cold_prepare_retained_fd model-streamed-short
    timeout --kill-after=5s 900s "$model_child" \
        --mode streamed --case short
    assert_retained_identity model-streamed-short

    cold_prepare_retained_fd model-streamed-prefill-8192
    timeout --kill-after=5s 1800s "$model_child" \
        --mode streamed --case prefill-8192
    assert_retained_identity model-streamed-prefill-8192

    timeout --kill-after=5s 900s "$model_child"
else
    "$model_child"
fi
assert_retained_identity model

if [ "$mode" = c7 ]; then
    timeout --kill-after=5s 60s "$stream_child" --case startup
    assert_retained_identity stream-startup

    timeout --kill-after=5s 60s "$stream_child" --case teardown-unsafe
    assert_retained_identity stream-teardown-unsafe

    cold_prepare_retained_fd stream-model-startup
    timeout --kill-after=5s 60s "$stream_child" --case model-startup
    assert_retained_identity stream-model-startup

    cold_prepare_retained_fd stream-model-page-advice
    timeout --kill-after=5s 900s "$stream_child" --case model-page-advice
    assert_retained_identity stream-model-page-advice

    cold_prepare_retained_fd stream-model-teardown-unsafe
    timeout --kill-after=5s 60s "$stream_child" --case model-teardown-unsafe
    assert_retained_identity stream-model-teardown-unsafe

    timeout --kill-after=5s 60s "$stream_child" --case prefill-allocation
    assert_retained_identity stream-prefill-allocation
fi

hash_retained_fd
