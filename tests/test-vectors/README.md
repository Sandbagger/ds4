# DeepSeek V4 Flash Test Vectors

These vectors were captured from the official DeepSeek V4 Flash API using
`deepseek-v4-flash`, greedy decoding, thinking disabled, and
`top_logprobs=20`. The hosted API does not expose full logits, so these files
store the best logprob slice the API provides.

Files:

- `prompts/*.txt`: exact user prompts.
- `official/*.official.json`: official API continuations and top-logprobs.
- `official.vec`: compact C-test fixture generated from the official JSON.
- `local-golden.vec`: local top-k/logit fixture captured from a known-sane DS4
  Flash run. It is used to catch substantial backend drift that can keep the
  same greedy token while damaging the logits distribution.

Regenerate official vectors:

```sh
DEEPSEEK_API_KEY=... ./tests/test-vectors/fetch_official_vectors.py
```

Running the fetcher without `--only` also regenerates `official.vec`.

The C runner consumes `official.vec` directly:

```sh
./ds4_test --logprob-vectors
```

GLM 5.2 OpenRouter vectors are kept in a separate directory:

```sh
OPENROUTER_API_KEY=... ./tests/test-vectors/fetch_openrouter_glm_vectors.py

DS4_TEST_MODEL=models/GLM-5.2-UD-Q4_K_XL.gguf \
DS4_TEST_VECTOR_FILE=tests/test-vectors/glm-openrouter/official.vec \
  ./ds4_test --logprob-vectors
```

The same fetcher also writes `tests/test-vectors/glm-openrouter/manifest.tsv`
for `gguf-tools/quality-testing/score_official`.  By default it routes to
OpenRouter `parasail/fp8` with strict parameter matching so top-logprob slices
are present in the fixture.

It also consumes the local golden fixture:

```sh
./ds4_test --local-golden-vectors
```

The Metal SSD-streaming cache-pressure repro for issue #384 is a focused
variant of the official-vector check. It forces a 16GiB routed-expert cache and
runs only the `short_code_completion` case that exposes wrong logits when
layer-batched decode reuses expert-cache buffers before the command buffer has
completed:

```sh
DS4_TEST_MODEL=gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  ./ds4_test --metal-ssd-streaming-cache-pressure
```

The runner opens the normal non-quality path with accelerator-specific fast
routes disabled and pins `DS4_METAL_PREFILL_CHUNK=2048` for this strict
official-vector check.

`official.vec` is intentionally trivial to parse from C: each case points to a
prompt file and each expected token is hex-encoded by bytes. The official JSON
files remain in the tree so the compact fixture can be audited against the raw
API response.

To inspect a local top-logprob dump manually:

```sh
./ds4 --metal --nothink -sys "" --temp 0 -n 4 --ctx 16384 \
  --prompt-file tests/test-vectors/prompts/long_code_audit.txt \
  --dump-logprobs /tmp/long_code_audit.ds4.json \
  --logprobs-top-k 20
```

## Laguna S 2.1 resident-CUDA oracle

`laguna-resident/` is a fail-closed full-model parity contract for
`poolside/Laguna-S-2.1-GGUF` under the single-oracle policy
`single-poolside-v1`. The pinned Poolside `llama.cpp` runtime supplies the
expected final logits. There are exactly four promoted Poolside vectors:
`short`, `swa-513`, `yarn-8193`, and `deep-32768`. The fixture also contains
one eight-ID, teacher-forced continuation for `yarn-8193`.

This is intentionally one full-model oracle. The scalar CUDA references in
`tests/test_cuda_laguna_kernels.c` independently check RMSNorm/RoPE,
attention, KV-ring, and routed-MoE primitives. The serialized-versus-batched
and serialized-versus-mixed session checks in
`tests/test_cuda_laguna_model.c` provide metamorphic assurance for scheduling
and session isolation. Those scalar and metamorphic checks are complementary
assurance layers, not a second full-model oracle.

### Honest limitation and immutable pins

The contract does not detect a model-level error shared by Poolside's runtime
and the captured GGUF conversion. A genuinely independent runtime requires a
new manifest schema and oracle policy; these vectors must not be reinterpreted
as dual-runtime evidence.

The capture, model, runtime, and contract identities are fixed:

- Poolside capture directory:
  `/home/will/artifacts/ds4/laguna-resident/a250e43/poolside-04b2b72`.
- `capture.json` SHA-256 trust anchor:
  `cc4fb338556c0895ff985edb5435ae7801be7dcb98c2958dc96a56d34f2c848e`.
- Model repository: `poolside/Laguna-S-2.1-GGUF`.
- Model revision: `706fa69799926b6afde1af9e24ca2a4923f110a1`.
- Model filename: `laguna-s-2.1-Q4_K_M.gguf`.
- Model size: `68248759648` bytes.
- Model SHA-256:
  `e163b2c98908809a71245d6bb68b2226994d9969cb2a438eccb72196a1c4147a`.
- Poolside `llama.cpp` runtime commit:
  `04b2b72cb54048ead292884adbe11f284e3ec950`.
- Immutable Laguna contract checkpoint:
  `a250e43722945e293f6044bc7254c4806d5a7912`.
- Tokenizer runtime commit captured by the immutable promoted manifest:
  `78bac71d711aab39fbbce6afdcd5f710622aed26`.
  Pass this exact value as `LAGUNA_TOKENIZER_RUNTIME_COMMIT`; do not substitute
  a later branch `HEAD`. The verifier rejects any mismatch.

The fixed source inputs are `cases.json`, `short.txt`,
`generate_benchmark_prompt.py`, and `benchmark-32768.txt`, plus the three
materialized raw prompts. The generator and benchmark SHA-256 values are
`118f1223ad248f845acd0dcb69444f911a3a6843d548db866a73a1106d7c5e3d` and
`aa352ad2890413cf112abc10d7349db3ef4be4c3722e2276f943e7b413a59206`,
respectively. Regenerate and confirm only the deterministic seed before a
separately authorized capture:

```sh
python3 tests/test-vectors/laguna-resident/generate_benchmark_prompt.py \
  --output tests/test-vectors/laguna-resident/benchmark-32768.txt
sha256sum tests/test-vectors/laguna-resident/generate_benchmark_prompt.py \
  tests/test-vectors/laguna-resident/benchmark-32768.txt
```

Each promoted vector is exactly 100,352 little-endian float32 values
(`401408` bytes). The continuation is exactly eight little-endian signed
int32 IDs (`32` bytes). No Metal vector is part of this fixture.

### Promotion and verification

The pinned Poolside capture is already validated at the path above. Do not
perform an ordinary recapture. Re-capture is allowed only when the pinned
capture fails provenance validation or the operator explicitly changes the
model or runtime pins. The Laguna contract has no legacy Metal full-model
capture or verification CLI; use the Poolside commands below.

Promotion from the pinned capture uses the exact command and publishes the
four vectors, one continuation, three materialized prompts, and manifest:

```sh
LAGUNA_MODEL="$LAGUNA_MODEL" \
python3 gguf-tools/quality-testing/compare_laguna_logits.py \
  --ds4 ./ds4 \
  --llama /home/will/artifacts/ds4/laguna-resident/a250e43/poolside-04b2b72 \
  --promote tests/test-vectors/laguna-resident
```

Expected output is exactly:
`promoted=tests/test-vectors/laguna-resident cases=4 vectors=4 oracle=poolside`.
Promotion is no-clobber and records the clean tokenizer runtime commit; do
not use a synthetic checkout.

Run the non-mutating verifier separately with every identity pin:

```sh
python3 gguf-tools/quality-testing/compare_laguna_logits.py \
  --verify-promoted tests/test-vectors/laguna-resident \
  --contract-commit a250e43722945e293f6044bc7254c4806d5a7912 \
  --tokenizer-runtime-commit "$LAGUNA_TOKENIZER_RUNTIME_COMMIT" \
  --llama-commit 04b2b72cb54048ead292884adbe11f284e3ec950 \
  --capture-manifest-sha256 cc4fb338556c0895ff985edb5435ae7801be7dcb98c2958dc96a56d34f2c848e \
  --gguf-size 68248759648 \
  --gguf-sha256 e163b2c98908809a71245d6bb68b2226994d9969cb2a438eccb72196a1c4147a
```

Expected output is exactly:
`verified=tests/test-vectors/laguna-resident cases=4 vectors=4 oracle=poolside`.
Verification rejects missing or extra files, stale provenance, bad sizes or
hashes, token disagreement, and widened thresholds without modifying the
fixture tree.

### Ordered resident-CUDA admission gate

The aggregate target requires the tokenizer runtime pin, verifies the fixture
first, then runs the independent scalar CUDA suite with the graph override
unset, and finally runs the model-backed session contract:

```sh
LAGUNA_TOKENIZER_RUNTIME_COMMIT=78bac71d711aab39fbbce6afdcd5f710622aed26 \
DS4_TEST_MODEL="$LAGUNA_MODEL" \
make test-cuda-laguna-resident
```

Its ordered commands are equivalent to:

```sh
python3 gguf-tools/quality-testing/compare_laguna_logits.py \
  --verify-promoted tests/test-vectors/laguna-resident \
  --contract-commit a250e43722945e293f6044bc7254c4806d5a7912 \
  --tokenizer-runtime-commit "$LAGUNA_TOKENIZER_RUNTIME_COMMIT" \
  --llama-commit 04b2b72cb54048ead292884adbe11f284e3ec950 \
  --capture-manifest-sha256 cc4fb338556c0895ff985edb5435ae7801be7dcb98c2958dc96a56d34f2c848e \
  --gguf-size 68248759648 \
  --gguf-sha256 e163b2c98908809a71245d6bb68b2226994d9969cb2a438eccb72196a1c4147a
env -u DS4_CUDA_MOE_DECODE_GRAPH ./tests/test_cuda_laguna_kernels --case all
DS4_TEST_MODEL="$DS4_TEST_MODEL" ./tests/test_cuda_laguna_model
```

Until resident CUDA engine admission lands, the model binary may stop only at
the existing Laguna engine-admission gate after the verifier and scalar suite
pass. No earlier fixture or policy failure is expected.
