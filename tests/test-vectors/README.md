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

`laguna-resident/` defines a fail-closed, two-oracle model-parity contract for
`poolside/Laguna-S-2.1-GGUF`. The admitted artifact is
`laguna-s-2.1-Q4_K_M.gguf` at revision
`706fa69799926b6afde1af9e24ca2a4923f110a1`, exactly `68248759648` bytes,
with SHA-256
`e163b2c98908809a71245d6bb68b2226994d9969cb2a438eccb72196a1c4147a`.
The independent Poolside reference is pinned to llama.cpp commit
`04b2b72cb54048ead292884adbe11f284e3ec950`.

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

Build the pinned Poolside capture helper and materialize the exact 513, 8193,
and 32768-token raw prompts from the shared seed:

```sh
make -C gguf-tools quality-laguna-logits \
  LLAMA_CPP_DIR="$LLAMA_LAGUNA_DIR" \
  LLAMA_CPP_BUILD="$LLAMA_LAGUNA_DIR/build"
gguf-tools/quality-testing/dump_llama_logits \
  --model "$LAGUNA_MODEL" \
  --cases tests/test-vectors/laguna-resident/cases.json \
  --seed-prompt tests/test-vectors/laguna-resident/benchmark-32768.txt \
  --materialize-prompts tests/test-vectors/laguna-resident \
  --out "$LAGUNA_LLAMA_OUT"
```

Capture the same four frontiers and eight teacher-forced continuation rows
with DwarfStar Metal into `$LAGUNA_METAL_OUT`, then promote only when the two
tokenizations, argmaxes, continuations, and fixed numeric gates agree:

```sh
python3 gguf-tools/quality-testing/compare_laguna_logits.py \
  --ds4 ./ds4 \
  --metal "$LAGUNA_METAL_OUT" \
  --llama "$LAGUNA_LLAMA_OUT" \
  --promote tests/test-vectors/laguna-resident
```

Promotion writes one Metal and one Poolside little-endian float32 vector for
each of `short`, `swa-513`, `yarn-8193`, and `deep-32768`, plus the eight-ID
little-endian continuation and `manifest.json`. It requires centered RMS
`<= 0.02`, centered maximum absolute error `<= 0.10`, top-20 overlap `>= 18`,
and exact argmax and continuation agreement. Materialized prompts and promoted
vectors are intentionally absent until both independent captures pass.

Task 6 admission must run the non-mutating verifier with all runtime and GGUF
pins supplied explicitly:

```sh
python3 gguf-tools/quality-testing/compare_laguna_logits.py \
  --verify-promoted tests/test-vectors/laguna-resident \
  --dwarfstar-commit "$LAGUNA_CONTRACT_COMMIT" \
  --llama-commit 04b2b72cb54048ead292884adbe11f284e3ec950 \
  --gguf-size 68248759648 \
  --gguf-sha256 e163b2c98908809a71245d6bb68b2226994d9969cb2a438eccb72196a1c4147a
```

Verification rejects missing or extra files, bad sizes or hashes, stale pins,
token disagreement, or widened gates without modifying the fixture tree. The
model-backed CUDA target is deliberately excluded from ordinary `make test`;
run it explicitly with the verified local model:

```sh
DS4_TEST_MODEL="$LAGUNA_MODEL" make test-cuda-laguna-model
```
