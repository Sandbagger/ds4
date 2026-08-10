#!/bin/bash -p
set -Eeuo pipefail

usage() {
    printf 'usage: %s POOLSIDE_SRC POOLSIDE_BUILD MODEL OUT_ROOT\n' "$0" >&2
}

capture_environment() {
    exec /usr/bin/env -i \
        PATH=/usr/bin:/bin:/usr/local/cuda/bin \
        LANG=C \
        LC_ALL=C \
        TMPDIR=/tmp \
        "$@"
}

assert_capture_environment() {
    local name
    if [[ ${LAGUNA_C7_SANITIZED:-} != 1 \
        || ${PATH:-} != /usr/bin:/bin:/usr/local/cuda/bin \
        || ${LANG:-} != C \
        || ${LC_ALL:-} != C \
        || ${TMPDIR:-} != /tmp \
        || -z ${PWD:-} \
        || ! ${SHLVL:-} =~ ^[1-9][0-9]*$ ]]; then
        printf 'noncanonical capture environment values\n' >&2
        return 1
    fi
    while IFS= read -r name; do
        case $name in
            LAGUNA_C7_SANITIZED|LANG|LC_ALL|PATH|PWD|SHLVL|TMPDIR) ;;
            *)
                printf 'noncanonical capture environment variable: %s\n' \
                    "$name" >&2
                return 1
                ;;
        esac
    done < <(compgen -e)
}

capture_git() {
    GIT_CONFIG_GLOBAL=/dev/null \
        GIT_CONFIG_NOSYSTEM=1 \
        GIT_NO_REPLACE_OBJECTS=1 \
        /usr/bin/git \
        -c core.fsmonitor=false \
        -c core.untrackedCache=false \
        -c core.preloadIndex=false \
        "$@"
}

gpu_compute_processes() {
    /usr/bin/nvidia-smi \
        --query-compute-apps=pid \
        --format=csv,noheader,nounits
}

assert_gpu_processes() {
    local allowed_pid=${1:-}
    local output
    local pid
    if ! output=$(gpu_compute_processes); then
        printf 'failed to query GPU compute processes\n' >&2
        return 1
    fi
    for pid in $output; do
        if [[ ! $pid =~ ^[1-9][0-9]*$ ]]; then
            printf 'malformed GPU compute-process PID: %s\n' "$pid" >&2
            return 1
        fi
        if [[ -z $allowed_pid || $pid != "$allowed_pid" ]]; then
            printf 'foreign GPU compute-process PID: %s\n' "$pid" >&2
            return 1
        fi
    done
}

run_probe_exclusive() {
    local probe_status=0
    local monitor_status=0
    local finished_pid
    assert_gpu_processes ""
    "$@" &
    active_probe_pid=$!
    (
        while kill -0 "$active_probe_pid" 2>/dev/null; do
            if ! assert_gpu_processes "$active_probe_pid"; then
                kill "$active_probe_pid" 2>/dev/null || true
                exit 1
            fi
            sleep 0.05
        done
    ) &
    active_monitor_pid=$!
    wait "$active_probe_pid" || probe_status=$?
    wait "$active_monitor_pid" || monitor_status=$?
    finished_pid=$active_probe_pid
    active_probe_pid=
    active_monitor_pid=
    assert_gpu_processes "$finished_pid"
    if (( monitor_status != 0 )); then
        return "$monitor_status"
    fi
    return "$probe_status"
}

assert_build_outputs_absent() {
    local directory
    local stale_output
    local target_directories=(
        "$poolside_build/src/CMakeFiles/llama.dir"
        "$poolside_build/ggml/src/CMakeFiles/ggml.dir"
        "$poolside_build/ggml/src/CMakeFiles/ggml-base.dir"
        "$poolside_build/ggml/src/CMakeFiles/ggml-cpu.dir"
        "$poolside_build/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir"
    )
    for directory in "${target_directories[@]}"; do
        [[ -d $directory ]] || continue
        stale_output=$(find "$directory" \( -type f -o -type l \) \
            \( -name '*.o' -o -name '*.obj' -o -name '*.a' \
            -o -name '*.so' -o -name '*.so.*' \) -print -quit)
        if [[ -n $stale_output ]]; then
            printf 'stale build output survived clean: %s\n' "$stale_output" >&2
            return 1
        fi
    done
    if [[ -d $poolside_build/bin ]]; then
        stale_output=$(find "$poolside_build/bin" -maxdepth 1 \
            \( -type f -o -type l \) \
            \( -name 'libllama.so*' -o -name 'libggml*.so*' \) \
            -print -quit)
        if [[ -n $stale_output ]]; then
            printf 'stale linked library survived clean: %s\n' \
                "$stale_output" >&2
            return 1
        fi
    fi
}

rebuild_llama() {
    /usr/bin/cmake --build "$poolside_build" --target clean -j4
    assert_build_outputs_absent
    /usr/bin/cmake --build "$poolside_build" --target llama -j4
}

restore_source() {
    local original_status=$?
    trap - EXIT
    if [[ -n ${active_monitor_pid:-} ]]; then
        kill "$active_monitor_pid" 2>/dev/null || true
        wait "$active_monitor_pid" 2>/dev/null || true
    fi
    if [[ -n ${active_probe_pid:-} ]]; then
        kill "$active_probe_pid" 2>/dev/null || true
        wait "$active_probe_pid" 2>/dev/null || true
    fi
    if (( patched )); then
        capture_git -C "$poolside_src" apply --check -R "$patch"
        capture_git -C "$poolside_src" apply -R "$patch"
        patched=0
        rebuild_llama
        /usr/bin/python3 "$verifier" preflight \
            --poolside-src "$poolside_src" \
            --poolside-build "$poolside_build" \
            --model-fd 9 \
            --continuity-fd 8
    fi
    rm -f -- "$probe_bin"
    exit "$original_status"
}

main() {
    if [[ ${LAGUNA_C7_SANITIZED:-} != 1 ]]; then
        capture_environment \
            LAGUNA_C7_SANITIZED=1 \
            /bin/bash -p "${BASH_SOURCE[0]}" "$@"
    fi
    assert_capture_environment
    if [[ $# -ne 4 ]]; then
        usage
        return 2
    fi

    script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
    repo_root=$(capture_git -C "$script_dir" rev-parse --show-toplevel)
    test "$script_dir" = "$repo_root/tests/oracle-producers/laguna-c7"
    poolside_src=$1
    poolside_build=$2
    model=$3
    out_root=$4
    probe_src="$script_dir/probe_poolside_laguna_moe.cpp"
    tokens="$script_dir/short.tokens.i32"
    patch="$script_dir/poolside-l2-callback.patch"
    verifier="$script_dir/verify_poolside_laguna_moe.py"
    out_22="$out_root/poolside-c7-moe-auto-22"
    out_1="$out_root/poolside-c7-moe-auto-1"
    probe_bin=$(mktemp "${TMPDIR:-/tmp}/probe_poolside_laguna_moe.XXXXXX")
    continuity_file=$(mktemp "${TMPDIR:-/tmp}/laguna_model_continuity.XXXXXX")
    patched=0
    active_probe_pid=
    active_monitor_pid=
    trap restore_source EXIT

    test ! -e "$out_22"
    test ! -e "$out_1"
    assert_gpu_processes ""
    exec 8<>"$continuity_file"
    rm -f -- "$continuity_file"
    exec 9<"$model"
    model_fd_path="/proc/self/fd/9"
    /usr/bin/python3 "$verifier" preflight \
        --poolside-src "$poolside_src" \
        --poolside-build "$poolside_build" \
        --model-fd 9 \
        --continuity-fd 8
    mkdir -p -- "$out_root"

    capture_git -C "$poolside_src" apply --check "$patch"
    capture_git -C "$poolside_src" apply "$patch"
    patched=1
    rebuild_llama

    /usr/bin/c++ -std=c++17 -O2 \
        -I"$poolside_src/include" \
        -I"$poolside_src/ggml/include" \
        "$probe_src" \
        "$poolside_build/bin/libllama.so" \
        "$poolside_build/bin/libggml.so" \
        "$poolside_build/bin/libggml-base.so" \
        -Wl,-rpath-link,"$poolside_build/bin" \
        -o "$probe_bin"

    run_probe_exclusive capture_environment \
        "LD_LIBRARY_PATH=$poolside_build/bin" \
        "$probe_bin" \
        --model "$model_fd_path" \
        --tokens "$tokens" \
        --out "$out_22" \
        --flash-attn auto \
        --detail-layer 1 \
        --token-count 22 \
        8>&-

    run_probe_exclusive capture_environment \
        "LD_LIBRARY_PATH=$poolside_build/bin" \
        "$probe_bin" \
        --model "$model_fd_path" \
        --tokens "$tokens" \
        --out "$out_1" \
        --flash-attn auto \
        --detail-layer 1 \
        --token-count 1 \
        8>&-

    /usr/bin/python3 "$verifier" captured \
        --poolside-src "$poolside_src" \
        --poolside-build "$poolside_build" \
        --model-fd 9 \
        --continuity-fd 8 \
        --capture-root "$out_root" \
        --probe-bin "$probe_bin"

    printf 'capture_22=%s\ncapture_1=%s\n' "$out_22" "$out_1"
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
    main "$@"
fi
