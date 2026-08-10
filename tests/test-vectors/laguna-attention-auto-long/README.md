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
files.  The source used for each capture was a narrowed derivative of
`gguf-tools/quality-testing/probe_poolside_laguna_layers.cpp`: it requested only
`Vcur`, `attn_gate_proj`, `Qcur_rope`, `Kcur_rope`, and `attn_gated`, accepted
the pinned token prefix, and used the fixed token count recorded per case.
