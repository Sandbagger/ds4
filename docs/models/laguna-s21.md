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

## 2026-08-24 — long-context memory-budget gate in the qualification harness

Three qualification windows on spark-09fa (121 GiB GB10 unified pool) died in
hard swap-death during `DS4_TEST_MODEL=<laguna Q4_K_M> ./ds4_test`, all in the
model-backed phase, with coder+ds4-server stopped — not residency competition.
`~/window.log` 60 s ticks pin the curve: mid-load (~48 GiB spans covered) the
pool still showed ~20 GiB available; right after "startup model preparation
covered 63.56 GiB of tensor spans" it read 121/121 used, 0 free, 0 buff/cache,
0 available — every large process OOM-dumped, ssh banner-exchange dead. The
box stayed unresponsive ~2 h until the window script's own `timeout` TERM
landed and the restore trap freed it.

Reading: on GB10 the device spans are unified-pool allocations, so model load
(63.56 GiB) + CUDA context + `ds4_session_create(..., 100000)` KV + MoE
staging cannot co-exist with anything else; kswapd reclaims all page cache and
the pool tips over during load itself. The thrash reproduced on an IDLE pool
(~119 GiB available), i.e. true footprint ≈ 1.9× the model file.

Fix: memory-budget gate in `tests/ds4_test.c`, decisions pinned offline by
`--long-context-budget`. Before engine load, `test_long_story_fact_recall`
stats the model and reads `/proc/meminfo:MemAvailable`, skipping with a marker
line unless

    model_bytes + ctx_tokens x 192 KiB (KV reserve) + 16 GiB fixed reserve,
    with 26/20 headroom,

fits into MemAvailable. Headroom is calibrated so the real-file case computes
a ~127 GiB floor — above the whole physical pool of a 128G-class host, which
therefore refuses by construction — while a 160G+-class host runs the case.
Unmeasurable inputs skip (fail-safe). `DS4_TEST_ALLOW_LONG_CONTEXT=1` forces
the attempt. A bounded variant (SSD-streaming profile or explicit small-KV
mode) remains open; this gate only makes the default run honest about not
fitting.
