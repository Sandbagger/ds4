#!/bin/sh
set -e

GLM_UNSLOTH_REPO="unsloth/GLM-5.2-GGUF"
GLM_ANTIREZ_REPO="antirez/GLM-5.2-GGUF"
LAGUNA_REPO="poolside/Laguna-S-2.1-GGUF"
LAGUNA_REVISION="e2ccc0579fc18e6ea2362fa25fccbcd470f0e332"
LAGUNA_Q4_SIZE_BYTES="68248760064"
LAGUNA_Q4_SHA256="a34c74e46688122bef83122f4133031bababbefcf57436dde97048c91e2cc6ff"
REPO="antirez/deepseek-v4-gguf"
Q2_IMATRIX_FILE="DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
Q4_IMATRIX_FILE="DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf"
Q2_Q4_IMATRIX_FILE="DeepSeek-V4-Flash-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-fixed.gguf"
PRO_Q2_IMATRIX_FILE="DeepSeek-V4-Pro-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-Instruct-imatrix.gguf"
PRO_Q4_LAYERS00_30_FILE="DeepSeek-V4-Pro-Q4K-Layers00-30.gguf"
PRO_Q4_LAYERS31_OUTPUT_FILE="DeepSeek-V4-Pro-Q4K-Layers-31-output.gguf"
MTP_FILE="DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf"
DSPARK_SUPPORT_FILE="DeepSeek-V4-Flash-DSpark-support.gguf"
GLM_UNSLOTH_Q4_REMOTE_BASE="UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL"
GLM_UNSLOTH_Q4_LOCAL_BASE="GLM-5.2-UD-Q4_K_XL"
GLM_UNSLOTH_Q4_FIRST_FILE="$GLM_UNSLOTH_Q4_LOCAL_BASE-00001-of-00011.gguf"
GLM_ANTIREZ_IQ2XXS_FILE="GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf"
GLM_ANTIREZ_Q2_FILE="GLM-5.2-UD-Q2_K_RoutedQ2K.gguf"
GLM_ANTIREZ_Q4_FILE="GLM-5.2-UD-Q4_K_RoutedQ4K.gguf"
LAGUNA_Q4_FILE="laguna-s-2.1-Q4_K_M.gguf"

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
OUT_DIR=${DS4_GGUF_DIR:-"$ROOT/gguf"}
case "$OUT_DIR" in
    /*) ;;
    *) OUT_DIR="$ROOT/$OUT_DIR" ;;
esac
TOKEN=${HF_TOKEN:-}
HF_REVISION=

usage() {
    cat <<EOF
DwarfStar GGUF downloader

Usage:
  ./download_model.sh q2-imatrix [--token TOKEN]
  ./download_model.sh q2-q4-imatrix [--token TOKEN]
  ./download_model.sh q4-imatrix [--token TOKEN]
  ./download_model.sh pro-q2-imatrix [--token TOKEN]
  ./download_model.sh pro-q4-layers00-30 [--token TOKEN]
  ./download_model.sh pro-q4-layers31-output [--token TOKEN]
  ./download_model.sh pro-q4-split [--token TOKEN]
  ./download_model.sh mtp [--token TOKEN]
  ./download_model.sh dspark-support [--token TOKEN]
  ./download_model.sh glm-unsloth-q4 [--token TOKEN]
  ./download_model.sh glm-antirez-iq2xxs [--token TOKEN]
  ./download_model.sh glm-antirez-q2 [--token TOKEN]
  ./download_model.sh glm-antirez-q4 [--token TOKEN]
  ./download_model.sh laguna-q4 [--token TOKEN]

Targets:

  q2-imatrix
       2-bit routed experts, about 81 GB on disk.
       Recommended model for 96 and 128 GB RAM machines.

  q2-q4-imatrix
       Mixed Flash quant: mostly q2 routed experts, with the last 6 layers
       using q4 routed experts. About 98 GB on disk. Good for higher
       quality inference for 128 GB MacBooks. Works on DGX Spark but loading
       may struggle compared to q2-imatrix.

  q4-imatrix
       4-bit routed experts, about 153 GB on disk.
       Recommended model for machines with 256 GB RAM or more.

  pro-q2-imatrix
       DeepSeek V4 PRO q2 imatrix quant, as a single GGUF file. About 430 GB
       on disk; intended for 512 GB RAM machines.

  pro-q4-layers00-30
       First half of the DeepSeek V4 PRO Q4 routed-expert quant, layers 0..30.
       Use on the coordinator in a two-Mac-Studio distributed run. About 426 GB.

  pro-q4-layers31-output
       Second half of the DeepSeek V4 PRO Q4 routed-expert quant, layers
       31..output. Use on the worker in a two-Mac-Studio distributed run.
       About 412 GB.

  pro-q4-split
       Downloads both PRO Q4 split files into the download directory. About
       838 GB total. This target does not update ./ds4flash.gguf.

  mtp  Optional speculative decoding component, about 3.5 GB on disk.
       It is useful with q2-imatrix, q2-q4-imatrix, and q4-imatrix, but must be
       enabled explicitly with --mtp when running ds4 or ds4-server.

  dspark-support
       Optional DSpark speculative decoding support GGUF, about 6 GB. Enable it
       with --dspark and --mtp when running ds4 or ds4-server.

  glm-unsloth-q4
       GLM 5.2 Unsloth UD-Q4_K_XL quant from unsloth/GLM-5.2-GGUF.
       Downloads all 11 shards and links ./ds4flash.gguf to the first shard.

  glm-antirez-iq2xxs
       GLM 5.2 antirez routed IQ2_XXS GGUF from antirez/GLM-5.2-GGUF.
       Includes Q2_K block 78 and is intended for reduced-memory testing.

  glm-antirez-q2
       GLM 5.2 antirez routed Q2_K GGUF from antirez/GLM-5.2-GGUF.
       About 262 GB on disk.

  glm-antirez-q4
       GLM 5.2 antirez routed Q4_K GGUF from antirez/GLM-5.2-GGUF.
       About 434 GB on disk.

  laguna-q4
       Official imatrix-quantized Laguna S 2.1 Q4_K_M GGUF from Poolside.
       Staged under a revision-qualified immutable path and verified by exact
       size and SHA-256. This staging target does not update ./ds4flash.gguf.

Options:
  --token TOKEN  Hugging Face token. Otherwise HF_TOKEN or the local HF token
                 cache is used if present.

Environment:
  DS4_GGUF_DIR   Directory used for downloaded GGUF files.
                 Default: ./gguf

After main-model downloads except laguna-q4, the script updates:
  ./ds4flash.gguf -> <download directory>/<selected model>

laguna-q4 prints its verified immutable path without changing the active link.

Then the default commands work:
  ./ds4 -p "Hello"
  ./ds4-server --ctx 100000

After downloading mtp, enable it explicitly, for example:
  ./ds4 --mtp <download directory>/$MTP_FILE --mtp-draft 2

After downloading DSpark support, enable it explicitly in greedy mode:
  ./ds4 --dspark --mtp <download directory>/$DSPARK_SUPPORT_FILE --temp 0

PRO and GLM files are downloaded with the official Hugging Face downloader
because they are too large, sharded, or nested for the curl path used by the
smaller DeepSeek Flash GGUF files.
EOF
}

if [ $# -eq 0 ]; then
    usage
    exit 1
fi

MODEL=$1
shift
MODEL_FILES=
LINK_MODEL=1
FORCE_HF_DOWNLOAD=0
FLATTEN_DOWNLOADS=0
LAGUNA_STAGE_MODEL=

case "$MODEL" in
    q2-imatrix) MODEL_FILE=$Q2_IMATRIX_FILE ;;
    q2-q4-imatrix) MODEL_FILE=$Q2_Q4_IMATRIX_FILE ;;
    q4-imatrix) MODEL_FILE=$Q4_IMATRIX_FILE ;;
    pro-q2-imatrix) MODEL_FILE=$PRO_Q2_IMATRIX_FILE ;;
    pro-q4-layers00-30) MODEL_FILE=$PRO_Q4_LAYERS00_30_FILE; LINK_MODEL=0 ;;
    pro-q4-layers31-output) MODEL_FILE=$PRO_Q4_LAYERS31_OUTPUT_FILE; LINK_MODEL=0 ;;
    pro-q4-split)
        MODEL_FILES="$PRO_Q4_LAYERS00_30_FILE $PRO_Q4_LAYERS31_OUTPUT_FILE"
        LINK_MODEL=0
        ;;
    mtp) MODEL_FILE=$MTP_FILE; LINK_MODEL=0 ;;
    dspark-support) MODEL_FILE=$DSPARK_SUPPORT_FILE; LINK_MODEL=0 ;;
    glm-unsloth-q4)
        REPO=$GLM_UNSLOTH_REPO
        MODEL_FILE=$GLM_UNSLOTH_Q4_FIRST_FILE
        MODEL_FILES=
        for part in 00001 00002 00003 00004 00005 00006 00007 00008 00009 00010 00011; do
            MODEL_FILES="$MODEL_FILES $GLM_UNSLOTH_Q4_REMOTE_BASE-${part}-of-00011.gguf"
        done
        FORCE_HF_DOWNLOAD=1
        FLATTEN_DOWNLOADS=1
        ;;
    glm-antirez-q2)
        REPO=$GLM_ANTIREZ_REPO
        MODEL_FILE=$GLM_ANTIREZ_Q2_FILE
        FORCE_HF_DOWNLOAD=1
        ;;
    glm-antirez-iq2xxs)
        REPO=$GLM_ANTIREZ_REPO
        MODEL_FILE=$GLM_ANTIREZ_IQ2XXS_FILE
        FORCE_HF_DOWNLOAD=1
        ;;
    glm-antirez-q4)
        REPO=$GLM_ANTIREZ_REPO
        MODEL_FILE=$GLM_ANTIREZ_Q4_FILE
        FORCE_HF_DOWNLOAD=1
        ;;
    laguna-q4)
        REPO=$LAGUNA_REPO
        MODEL_FILE=$LAGUNA_Q4_FILE
        FORCE_HF_DOWNLOAD=1
        HF_REVISION=$LAGUNA_REVISION
        LINK_MODEL=0
        ;;
    -h|--help|help)
        usage
        exit 0
        ;;
    *)
        echo "Unknown model: $MODEL" >&2
        echo >&2
        usage >&2
        exit 1
        ;;
esac

while [ $# -gt 0 ]; do
    case "$1" in
        --token)
            shift
            if [ $# -eq 0 ]; then
                echo "Missing value after --token" >&2
                exit 1
            fi
            TOKEN=$1
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
    shift
done

if [ -z "$TOKEN" ] && [ -s "$HOME/.cache/huggingface/token" ]; then
    TOKEN=$(cat "$HOME/.cache/huggingface/token")
fi

needs_hf_download() {
    if [ "${FORCE_HF_DOWNLOAD:-0}" -eq 1 ]; then
        return 0
    fi
    case "$1" in
        "$PRO_Q2_IMATRIX_FILE"|"$PRO_Q4_LAYERS00_30_FILE"|"$PRO_Q4_LAYERS31_OUTPUT_FILE")
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

find_hf_command() {
    if command -v hf >/dev/null 2>&1; then
        printf '%s\n' hf
        return 0
    fi
    for dir in "$HOME"/Library/Python/*/bin "$HOME"/.local/bin; do
        if [ -x "$dir/hf" ]; then
            printf '%s\n' "$dir/hf"
            return 0
        fi
    done
    return 1
}

local_download_name() {
    if [ "${FLATTEN_DOWNLOADS:-0}" -eq 1 ]; then
        basename "$1"
    else
        printf '%s\n' "$1"
    fi
}

file_size_bytes() {
    size=
    if size=$(LC_ALL=C stat -f '%z' "$1" 2>/dev/null); then
        :
    elif size=$(LC_ALL=C stat -c '%s' "$1" 2>/dev/null); then
        :
    else
        return 1
    fi
    case "$size" in
        ''|*[!0-9]*) return 1 ;;
    esac
    printf '%s\n' "$size"
}

sha256_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        output=$(sha256sum "$1") || return 1
    elif command -v shasum >/dev/null 2>&1; then
        output=$(shasum -a 256 "$1") || return 1
    else
        echo "Cannot verify Laguna artifact: sha256sum or shasum is required." >&2
        return 1
    fi
    digest=${output%% *}
    case "$digest" in
        ''|*[!0-9a-fA-F]*) return 1 ;;
    esac
    printf '%s\n' "$digest" | tr 'A-F' 'a-f'
}

verify_laguna_q4() {
    artifact=$1
    if [ ! -f "$artifact" ] || [ -L "$artifact" ]; then
        echo "Laguna artifact is not a regular, non-symlink file: $artifact" >&2
        return 1
    fi
    observed_size=$(file_size_bytes "$artifact") || {
        echo "Cannot determine Laguna artifact size: $artifact" >&2
        return 1
    }
    if [ "$observed_size" != "$LAGUNA_Q4_SIZE_BYTES" ]; then
        echo "Laguna artifact size mismatch: $artifact" >&2
        echo "Expected $LAGUNA_Q4_SIZE_BYTES bytes, observed $observed_size." >&2
        return 1
    fi
    observed_sha256=$(sha256_file "$artifact") || {
        echo "Cannot calculate Laguna artifact SHA-256: $artifact" >&2
        return 1
    }
    if [ "$observed_sha256" != "$LAGUNA_Q4_SHA256" ]; then
        echo "Laguna artifact SHA-256 mismatch: $artifact" >&2
        echo "Expected $LAGUNA_Q4_SHA256, observed $observed_sha256." >&2
        return 1
    fi
}

download_laguna_q4_hf() {
    file=$1
    revision_dir="$OUT_DIR/$LAGUNA_REPO/$LAGUNA_REVISION"
    incoming_dir="$revision_dir/.incoming"
    incoming="$incoming_dir/$file"
    out="$revision_dir/$file"
    flat="$OUT_DIR/$file"
    LAGUNA_STAGE_MODEL=$out

    mkdir -p "$revision_dir"
    if [ -e "$flat" ] || [ -L "$flat" ]; then
        echo "Preserving unqualified Laguna artifact; it is not eligible for reuse: $flat" >&2
    fi

    if [ -e "$out" ] || [ -L "$out" ]; then
        if verify_laguna_q4 "$out"; then
            echo "Reusing verified Laguna Q4 artifact: $out"
            return
        fi
        echo "Refusing to replace invalid artifact at immutable Laguna path: $out" >&2
        exit 1
    fi

    HF_CMD=$(find_hf_command || true)
    if [ -z "$HF_CMD" ]; then
        echo "Laguna GGUF downloads require the official Hugging Face CLI." >&2
        echo "Install it with:" >&2
        echo "  python3 -m pip install -U huggingface_hub hf_xet" >&2
        exit 1
    fi

    mkdir -p "$incoming_dir"
    echo "Downloading $file"
    echo "from https://huggingface.co/$REPO"
    echo "revision $HF_REVISION"
    echo "into unpublished staging directory $incoming_dir"
    echo "using $HF_CMD download"
    echo "If the download stops, run the same command again to resume it."

    if [ -n "$TOKEN" ]; then
        "$HF_CMD" download "$REPO" "$file" --revision "$HF_REVISION" --repo-type model --local-dir "$incoming_dir" --token "$TOKEN"
    else
        "$HF_CMD" download "$REPO" "$file" --revision "$HF_REVISION" --repo-type model --local-dir "$incoming_dir"
    fi

    if ! verify_laguna_q4 "$incoming"; then
        echo "Downloaded Laguna artifact was not published; inspect or remove: $incoming" >&2
        exit 1
    fi
    if ! ln "$incoming" "$out"; then
        echo "Cannot publish Laguna artifact without replacing an existing path: $out" >&2
        exit 1
    fi
    rm -f "$incoming"
    echo "Verified and staged Laguna Q4 artifact: $out"
}

download_one_hf() {
    file=$1
    local_file=$(local_download_name "$file")
    out="$OUT_DIR/$local_file"
    hf_out="$OUT_DIR/$file"
    part="$out.part"

    mkdir -p "$(dirname "$out")"

    if [ -s "$out" ]; then
        echo "Already downloaded: $out"
        return
    fi

    if [ -e "$part" ]; then
        echo "Found curl partial download: $part" >&2
        echo "The Hugging Face downloader cannot resume curl .part files." >&2
        echo "Move or remove that partial download before retrying this target." >&2
        exit 1
    fi

    HF_CMD=$(find_hf_command || true)
    if [ -z "$HF_CMD" ]; then
        echo "Large GGUF downloads require the official Hugging Face CLI." >&2
        echo "Install it with:" >&2
        echo "  python3 -m pip install -U huggingface_hub hf_xet" >&2
        exit 1
    fi

    echo "Downloading $file"
    echo "from https://huggingface.co/$REPO"
    if [ -n "$HF_REVISION" ]; then
        echo "revision $HF_REVISION"
    fi
    echo "using $HF_CMD download"
    echo "If the download stops, run the same command again to resume it."

    if [ -n "$TOKEN" ] && [ -n "$HF_REVISION" ]; then
        "$HF_CMD" download "$REPO" "$file" --revision "$HF_REVISION" --repo-type model --local-dir "$OUT_DIR" --token "$TOKEN"
    elif [ -n "$TOKEN" ]; then
        "$HF_CMD" download "$REPO" "$file" --repo-type model --local-dir "$OUT_DIR" --token "$TOKEN"
    elif [ -n "$HF_REVISION" ]; then
        "$HF_CMD" download "$REPO" "$file" --revision "$HF_REVISION" --repo-type model --local-dir "$OUT_DIR"
    else
        "$HF_CMD" download "$REPO" "$file" --repo-type model --local-dir "$OUT_DIR"
    fi

    if [ "$hf_out" != "$out" ] && [ -s "$hf_out" ]; then
        mv "$hf_out" "$out"
        rmdir "$(dirname "$hf_out")" 2>/dev/null || true
    fi

    if [ ! -s "$out" ]; then
        echo "Hugging Face download finished but expected file is missing: $out" >&2
        exit 1
    fi
}

download_one() {
    file=$1
    local_file=$(local_download_name "$file")
    out="$OUT_DIR/$local_file"
    part="$out.part"
    aria2_part="$out.aria2"
    url="https://huggingface.co/$REPO/resolve/main/$file"

    if [ "$MODEL" = "laguna-q4" ]; then
        download_laguna_q4_hf "$file"
        return
    fi

    if needs_hf_download "$file"; then
        download_one_hf "$file"
        return
    fi

    mkdir -p "$(dirname "$out")"

    if [ -e "$aria2_part" ]; then
        echo "Found incomplete aria2 download sidecar: $aria2_part" >&2
        echo "Finish or remove that partial download before using this curl downloader." >&2
        exit 1
    fi

    if [ -s "$out" ]; then
        echo "Already downloaded: $out"
        return
    fi

    echo "Downloading $file"
    echo "from https://huggingface.co/$REPO"
    echo "If the download stops, run the same command again to resume it."

    if [ -n "$TOKEN" ]; then
        curl -fL --progress-meter -C - -H "Authorization: Bearer $TOKEN" -o "$part" "$url"
    else
        curl -fL --progress-meter -C - -o "$part" "$url"
    fi

    mv "$part" "$out"
}

if [ -n "$MODEL_FILES" ]; then
    for file in $MODEL_FILES; do
        download_one "$file"
    done
else
    download_one "$MODEL_FILE"
fi

if [ "$MODEL" = "mtp" ]; then
    echo
    echo "MTP is an optional component for q2-imatrix, q2-q4-imatrix, and q4-imatrix."
    echo "Enable it explicitly, for example:"
    echo "  ./ds4 --mtp $OUT_DIR/$MTP_FILE --mtp-draft 2"
elif [ "$MODEL" = "dspark-support" ]; then
    echo
    echo "DSpark support downloaded. Enable it explicitly in greedy mode:"
    echo "  ./ds4 --dspark -m ./ds4flash.gguf --mtp $OUT_DIR/$DSPARK_SUPPORT_FILE --temp 0"
elif [ "$MODEL" = "laguna-q4" ]; then
    echo
    echo "Verified immutable Laguna Q4 path: $LAGUNA_STAGE_MODEL"
    echo "Staged only; ./ds4flash.gguf was not changed."
elif [ "$MODEL" = "pro-q4-layers00-30" ] || [ "$MODEL" = "pro-q4-layers31-output" ] || [ "$MODEL" = "pro-q4-split" ]; then
    echo
    echo "Downloaded PRO Q4 distributed split file(s). Use them with --layers,"
    echo "for example coordinator layers 0:30 and worker layers 31:output."
elif [ "$LINK_MODEL" -eq 1 ]; then
    cd "$ROOT"
    ln -sfn "$OUT_DIR/$MODEL_FILE" ds4flash.gguf
    echo "Linked ./ds4flash.gguf -> $OUT_DIR/$MODEL_FILE"
fi

echo
echo "Done."
