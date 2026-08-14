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

`laguna-resident/` defines a fail-closed, one-full-model-oracle contract for
`poolside/Laguna-S-2.1-GGUF` under policy `single-poolside-v1`. Poolside's
Laguna-capable `llama.cpp` fork supplies the only full-model reference. The
scalar CUDA kernel references and serialized-versus-batched and
serialized-versus-mixed metamorphic session checks are complementary
assurance; they are not second full-model oracles.

The policy has an explicit limitation: it cannot detect a model-level error
shared by Poolside and the captured GGUF conversion. If a genuinely
independent runtime becomes available, introduce a new schema and policy and
recapture rather than reinterpreting these fixtures as dual-oracle evidence.

The exact pins are:

- contract commit: `a250e43722945e293f6044bc7254c4806d5a7912`;
- capture directory:
  `/home/will/artifacts/ds4/laguna-resident/a250e43/poolside-04b2b72`;
- capture manifest SHA-256:
  `cc4fb338556c0895ff985edb5435ae7801be7dcb98c2958dc96a56d34f2c848e`;
- Poolside runtime commit: `04b2b72cb54048ead292884adbe11f284e3ec950`;
- model repository and revision: `poolside/Laguna-S-2.1-GGUF` at
  `706fa69799926b6afde1af9e24ca2a4923f110a1`;
- model file: `laguna-s-2.1-Q4_K_M.gguf`, exactly `68248759648` bytes, with
  SHA-256
  `e163b2c98908809a71245d6bb68b2226994d9969cb2a438eccb72196a1c4147a`;
- promoted tokenizer runtime commit:
  `15c9b92502fed6bc26842e98d11a6347caadb08e`.

The checked-in inputs are `cases.json`, `short.txt`, the deterministic prompt
generator, and its generated `benchmark-32768.txt` seed. Regenerate the seed
and confirm the output before any capture. Their SHA-256 values are
`118f1223ad248f845acd0dcb69444f911a3a6843d548db866a73a1106d7c5e3d` and
`aa352ad2890413cf112abc10d7349db3ef4be4c3722e2276f943e7b413a59206`,
respectively:

```sh
python3 tests/test-vectors/laguna-resident/generate_benchmark_prompt.py \
  --output tests/test-vectors/laguna-resident/benchmark-32768.txt
sha256sum tests/test-vectors/laguna-resident/generate_benchmark_prompt.py \
  tests/test-vectors/laguna-resident/benchmark-32768.txt
```

The pinned capture has already been validated. Do not recapture it during an
ordinary verification or admission run. A new full-model capture requires a
failed provenance check or an explicit model/runtime change plus the external
isolation attestation described by the design.

Promotion consumes that capture and the pinned model through DS4's exact
tokenizer. From the clean canonical DGX checkout used for promotion, run:

```sh
LAGUNA_LLAMA_OUT=/home/will/artifacts/ds4/laguna-resident/a250e43/poolside-04b2b72
LAGUNA_MODEL=/home/will/models/poolside/Laguna-S-2.1-GGUF/706fa69799926b6afde1af9e24ca2a4923f110a1/laguna-s-2.1-Q4_K_M.gguf
LAGUNA_MODEL="$LAGUNA_MODEL" \
python3 gguf-tools/quality-testing/compare_laguna_logits.py \
  --ds4 ./ds4 \
  --llama "$LAGUNA_LLAMA_OUT" \
  --promote tests/test-vectors/laguna-resident
```

Promotion writes exactly four little-endian Poolside float32 vectors,
`short.llama.f32`, `swa-513.llama.f32`, `yarn-8193.llama.f32`, and
`deep-32768.llama.f32`, plus one eight-ID little-endian continuation file,
`yarn-8193.continuation.i32`, and the `laguna-resident-promoted-v2` manifest.
It does not overwrite existing promoted artifacts.

Verification is non-mutating and supplies every provenance pin explicitly.
Read the tokenizer runtime commit from the manifest, then run:

```sh
export LAGUNA_TOKENIZER_RUNTIME_COMMIT="$(
  python3 -c 'import json; print(json.load(open("tests/test-vectors/laguna-resident/manifest.json", encoding="utf-8"))["provenance"]["tokenizer_runtime_commit"])'
)"
test "${#LAGUNA_TOKENIZER_RUNTIME_COMMIT}" -eq 40
python3 gguf-tools/quality-testing/compare_laguna_logits.py \
  --verify-promoted tests/test-vectors/laguna-resident \
  --contract-commit a250e43722945e293f6044bc7254c4806d5a7912 \
  --tokenizer-runtime-commit "$LAGUNA_TOKENIZER_RUNTIME_COMMIT" \
  --llama-commit 04b2b72cb54048ead292884adbe11f284e3ec950 \
  --capture-manifest-sha256 \
    cc4fb338556c0895ff985edb5435ae7801be7dcb98c2958dc96a56d34f2c848e \
  --gguf-size 68248759648 \
  --gguf-sha256 \
    e163b2c98908809a71245d6bb68b2226994d9969cb2a438eccb72196a1c4147a
```

The aggregate target preserves the required gate order: promoted-fixture
verification, unchanged scalar CUDA kernel tests, then the model-backed CUDA
contract. It is deliberately excluded from ordinary `make test`. Run it with
the verified local model:

```sh
export LAGUNA_TOKENIZER_RUNTIME_COMMIT="$(
  python3 -c 'import json; print(json.load(open("tests/test-vectors/laguna-resident/manifest.json", encoding="utf-8"))["provenance"]["tokenizer_runtime_commit"])'
)"
test "${#LAGUNA_TOKENIZER_RUNTIME_COMMIT}" -eq 40
DS4_TEST_MODEL="$LAGUNA_MODEL" make test-cuda-laguna-resident
```

Verification rejects missing or extra files, bad sizes or hashes, stale pins,
token disagreement, or widened gates without modifying the fixture tree. The
legacy dual-runtime CLI is intentionally unsupported: do not pass the removed
Metal, DwarfStar-commit, or DS4-commit options.
