# Laguna Poolside-AUTO attention oracle

This fixture freezes the four inputs and gated output at Laguna layer 0's
attention boundary for the canonical 22-token short prompt. It exists to test
the DS4 attention primitive independently of the small upstream Q/K
RMSNorm+RoPE difference.

The files are raw, little-endian float32 values in GGML's contiguous order:

- `layer-00-q-rope.f32`: `[128, 48, 22]`
- `layer-00-k-rope.f32`: `[128, 8, 22]`
- `layer-00-v-proj.f32`: `[1024, 22]`
- `layer-00-gate-proj.f32`: `[48, 22]`
- `layer-00-attn-gated.f32`: `[6144, 22]`

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
the MMA-F16 `<128, 128, 32, 2>` specialization. Smaller synthetic shapes can
select another flash-attention kernel and are not an oracle for this boundary.

`tests/test_cuda_laguna_kernels.c --case prefill-attention-frozen` feeds these
exact Q/K/V/gate bytes to DS4 at `pos0=0` with a 256-row cache, then compares
the complete gated result, every token/head, and the diagnostic sentinel at
token 20, head 43. `tests/test_cuda_build_contract.py` verifies every fixture
size and SHA-256 before the CUDA gate is deployed.
