# laguna-s-2.1 porting ledger

Facts discovered the hard way during the S 2.1 port. Read before touching the
Laguna path; append findings in the same commit that teaches the engine
something new about this model.

## Identity & handles

- `arch=laguna`, model id `laguna-s-2.1` (family `LAGUNA`, variant
  `LAGUNA_S21`); C profile `DS4_SHAPE_LAGUNA_S21`.
- Descriptor: `gguf-tools/models/laguna-s21.desc`
- Stage manifest: `tests/oracle-producers/stage-manifests/laguna-s21.json`

## Architecture quirks

- Interleaved hybrid attention: `il % 4 == 0` -> SWA layer, 48 heads, plain
  RoPE (base 10000, `n_rot_swa=128`), window 512; otherwise full attention,
  72 heads, YaRN (factor 32, base 500000, orig ctx 8192), partial rope
  `n_rot=64`. Predicate: `ds4_laguna_layer_is_swa()`, keyed on per-layer
  head count (`laguna.attention.head_count` GGUF array).
- GQA: 8 kv heads everywhere; head_dim/value_dim 128.
- MoE: 256 experts / 10 used + 1 shared, ff_exp 1024; sigmoid gating
  (`expert_gating_func=2`); routed weight scale 2.5 (sum-normalised);
  exactly one leading dense block (dense FF 12288).
- Tokenizer: GPT-2 BPE from the GGUF with a Laguna pre-tokenizer (LF-run
  split + glm4-style segmenter, max_digits=1); vocab 100352;
  `</assistant>` is a CONTROL turn-end token; eos list `[2, 24]`.
- No hyper-connections, no MLA compressors, no indexer, no hash layers,
  no MTP/NextN: simple `output_norm` + `output` head.

## Checkpoint reality (2026-08)

- The official Poolside download ships under the `Q4_K_M` filename but is
  the **signal-Q8 recipe**: token_embd/output/attn/signal-path weights Q8_0,
  routed experts Q4_K, norms F32 (~63.6 GiB). Discriminator:
  `token_embd.type==Q8_0`.
- Two staged revisions (`706fa697…`, `e2ccc057…`) both validate green
  against the descriptor (2473 checks each, header-only).
- A legacy F16-attn recipe is accepted by the engine but not staged on disk.

## Capture notes

- Diagnostics are CUDA-path only (`laguna_graph_forward_token/batch`);
  CPU forward functions have no checkpoints by design.
- Widths that vary by layer class: q-proj/q-rope/attn-gated scale with
  heads (48 vs 72); gate-proj width IS the head count.
- `router-selected` is `.i32`; MoE capture-buffer stages have `.q8_1`
  companions and only exist under probe diagnostics.
- CUDA decode runs `laguna_attention_decode_fattn_vec_kernel<swa_ring>`;
  GB10 launch policy pins cc 12.1 / 48 SMs in
  `ds4_cuda_laguna_fattn_vec_policy.h`.

## Gotchas that cost us time

- K-quant GGML type ids run **10..15** (Q2_K=10 .. Q6_K=14). An off-by-two
  table mislabels every expert tensor and sends you hunting for a recipe
  mismatch that does not exist. The reader table is pinned to ds4.c's
  `DS4_TENSOR_*` enum by `EngineSyncTests.test_type_table_matches_engine_enum`.
- Capture dirs abort on stale files (`O_EXCL`) — fresh dir every run.
- Dumping syncs + restarts the CUDA command batch; probes assert logits stay
  bitwise identical with observer on/off before trusting any capture.
