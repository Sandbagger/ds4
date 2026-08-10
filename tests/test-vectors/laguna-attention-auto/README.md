# Laguna Poolside-AUTO attention oracle

This fixture freezes the inputs and output at Laguna layers 0 and 1's attention
boundaries for the canonical 22-token short prompt. Layer 0 pins the
48-query-head GQA6 AUTO path; layer 1 pins the distinct 72-query-head GQA9 AUTO
path. It separately pins layer 0 token 21's projection inputs and norm weights
so DS4's fused Q/K RMSNorm+RoPE can be tested independently of attention.

The files are raw, little-endian float32 values in GGML's contiguous order:

- `layer-00-q-rope.f32`: `[128, 48, 22]`
- `layer-00-k-rope.f32`: `[128, 8, 22]`
- `layer-00-q-proj-t21.f32`: token 21, `[128, 48, 1]`
- `layer-00-k-proj-t21.f32`: token 21, `[128, 8, 1]`
- `layer-00-q-norm-weight.f32`: `blk.0.attn_q_norm.weight`, `[128]`
- `layer-00-k-norm-weight.f32`: `blk.0.attn_k_norm.weight`, `[128]`
- `layer-00-v-proj.f32`: `[1024, 22]`
- `layer-00-gate-proj.f32`: `[48, 22]`
- `layer-00-attn-gated.f32`: `[6144, 22]`
- `layer-01-q-rope.f32`: `[128, 72, 22]`
- `layer-01-k-rope.f32`: `[128, 8, 22]`
- `layer-01-v-proj.f32`: `[1024, 22]`
- `layer-01-gate-proj.f32`: `[72, 22]`
- `layer-01-attn-gated.f32`: `[9216, 22]`

The capture used Poolside llama.cpp commit
`04b2b72cb54048ead292884adbe11f284e3ec950`, the pinned Laguna GGUF recorded
in `manifest.json`, `--flash-attn auto`, and the diagnostic probe at
`gguf-tools/quality-testing/probe_poolside_laguna_layers.cpp`. The equivalent
capture command is:

```sh
LD_LIBRARY_PATH=/home/will/code/poolside-llama.cpp-laguna/build-c7-diag/bin \
  ./probe_poolside_laguna_layers \
  --model "$LAGUNA_GGUF" \
  --tokens short.tokens.i32 \
  --out poolside-auto \
  --flash-attn auto
```

Two independent AUTO captures produced bit-identical hashes for all 62
diagnostic files. The full prompt geometry is intentional. On the DGX Spark,
`Tq=22`, `Hq:Hkv=48:8`, `D=128`, and Poolside's 256-row padded KV view select
the MMA-F16 `<128, 128, 32, 2>` specialization at layer 0. Layer 1's
`Hq:Hkv=72:8` selects `<128, 128, 32, 1>` and its two-partition reduction.
Smaller or head-repeated synthetic shapes can select or emulate the wrong
arithmetic and are not numerical oracles for these boundaries.

The token-21 Q projection is the 24,576-byte range beginning at byte 516,096
of `layer-00-q-proj.f32`; the K projection is the 4,096-byte range beginning at
byte 86,016 of `layer-00-k-proj.f32`. The two 512-byte weights were copied from
the pinned GGUF tensor payloads after parsing the directory with `parse_gguf()`
from `gguf-tools/mixed/splice_mixed_expert_layers_gguf.py` and asserting shape
`[128]`, GGML type `F32`, and exact size. `manifest.json` records both GGUF
offsets and every promoted-file hash.

The expected token-21 outputs are slices of the existing full
`layer-00-{q,k}-rope.f32` files. Intermediate `Qcur_normed` and `Kcur_normed`
values are intentionally not duplicated: the regression contract is the real
fused public primitive from projection input through RoPE output.
The kernel test resets the projection tensors and exercises both the paired
Q/K API and the single-tensor API independently for Q and K.

`tests/test_cuda_laguna_kernels.c --case prefill-attention-frozen` feeds each
layer's exact Q/K/V/gate bytes to DS4 at `pos0=0` with a 256-row cache, then
compares the complete gated result, every token/head, and one diagnostic
sentinel (layer 0 token 20/head 43; layer 1 token 21/head 71).
`tests/test_cuda_build_contract.py` verifies every fixture size and SHA-256
before the CUDA gate is deployed.
