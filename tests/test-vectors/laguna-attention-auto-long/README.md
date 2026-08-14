# Laguna AUTO long-attention oracle

These compact vectors qualify the Poolside CUDA AUTO attention topology that
the 22-token fixture cannot reach:

- layer 0 uses 512 tokens, GQA6, and eight 64-key FATTN iterations;
- layer 1 uses 64 tokens and the GQA9 `ncols=64`, `np=1` topology.

Each fixture retains two adjacent query heads for every query position.  The
heads straddle a GQA-to-KV-head boundary and retain the corresponding two KV
heads for every key position.  Unrelated heads are zero-filled by
`tests/test_cuda_laguna_kernels.c`; attention is head-independent, so every
selected causal row remains the exact Poolside output.  The public production
launcher and full 48:8 or 72:8 grid are still exercised.

The 512-token layer-1 callback capture was attempted first but was abandoned
after a controlled 900-second timeout: inserting callbacks into the first SWA
layer leaves the GPU idle in Poolside's diagnostic scheduler.  The 64-token
capture selects the same GQA9 `ncols=64` arithmetic without that cliff.  It does
not qualify the 513th-token SWA shift; the pinned full-model `swa-513` oracle is
the authority for that boundary.

`manifest.json` pins the exact Poolside/model/device identities, raw captures,
capture-source and binary hashes, byte-level extraction recipes, and compact
files.  The tracked host-only producer
`tests/oracle-producers/laguna-attention-auto-long/derive_capture_inputs.py`
strictly narrows the pinned
`gguf-tools/quality-testing/probe_poolside_laguna_layers.cpp` to the five
required callbacks (`Vcur`, `attn_gate_proj`, `Qcur_rope`, `Kcur_rope`, and
`attn_gated`) and emits the exact source used for either capture.  It also
expands the tracked `swa-prefix-512.tokens.json` token specification into the
exact little-endian int32 input: all 512 IDs for layer 0 or its first 64 IDs for
layer 1.

For example, regenerate the layer-0 inputs without a model, compiler, or GPU:

```sh
python3 tests/oracle-producers/laguna-attention-auto-long/derive_capture_inputs.py \
  --case layer0_gqa6_512 \
  --base-probe gguf-tools/quality-testing/probe_poolside_laguna_layers.cpp \
  --token-prefix tests/oracle-producers/laguna-attention-auto-long/swa-prefix-512.tokens.json \
  --probe-out /tmp/probe_poolside_laguna_512.cpp \
  --tokens-out /tmp/swa-prefix-512.tokens.i32
```

Use `--case layer1_gqa9_64` for the 64-token layer-1 inputs.  The host fixture
test runs both derivations and verifies their byte counts and SHA-256 identities
against the manifest; compiling and running the probe remain explicit capture
steps on the pinned Poolside checkout.
