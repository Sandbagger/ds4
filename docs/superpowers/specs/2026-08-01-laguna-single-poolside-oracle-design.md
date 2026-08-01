# Laguna Single-Poolside Oracle Design

**Date:** 2026-08-01
**Status:** Approved direction; implementation pending
**Branch:** `laguna-s2.1-resident-cuda`
**Existing contract checkpoint:** `a250e43722945e293f6044bc7254c4806d5a7912 test: define resident Laguna CUDA parity contract`

## Decision

Replace the unavailable DwarfStar Metal full-model oracle with one pinned
external full-model oracle: Poolside's Laguna-capable `llama.cpp` fork running
the pinned Q4_K_M GGUF on the DGX Spark.

This is an explicit reduction from two full-model oracles to one. It does not
claim that two `llama.cpp` builds are independent. Confidence comes from four
complementary layers:

1. the pinned Poolside full-model reference;
2. independent scalar references for CUDA primitives;
3. exact tokenizer, provenance, and fixture-integrity checks; and
4. metamorphic session checks comparing batched/mixed execution with the same
   operations serialized.

DS4 CUDA remains the system under test and must never generate its own expected
logits.

## Motivation and constraints

- No high-memory Metal host is available.
- The 68,248,759,648-byte GGUF belongs on the DGX, not the MacBook Air.
- A verified Poolside capture already exists on the DGX at:

  ```text
  /home/will/artifacts/ds4/laguna-resident/a250e43/poolside-04b2b72
  ```

  Its `capture.json` trust anchor is SHA-256
  `cc4fb338556c0895ff985edb5435ae7801be7dcb98c2958dc96a56d34f2c848e`.
  That manifest enumerates the exact allowed capture files and their hashes.

- The model is pinned to revision
  `706fa69799926b6afde1af9e24ca2a4923f110a1` with SHA-256
  `e163b2c98908809a71245d6bb68b2226994d9969cb2a438eccb72196a1c4147a`.
- The oracle runtime is pinned to Poolside `llama.cpp` commit
  `04b2b72cb54048ead292884adbe11f284e3ec950`.
- The existing two-oracle sequence in the Laguna resident-CUDA plan is
  superseded for this branch. Do not preserve an impossible adjacent
  Metal/Poolside fixture commit requirement.

## Assurance model

### Full-model oracle

Poolside `llama.cpp` supplies one final-frontier logit vector for each case:

- `short`
- `swa-513`
- `yarn-8193`
- `deep-32768`

It also supplies eight greedy, teacher-forced continuation IDs after the
YaRN-8193 frontier. Continuation step zero must equal the YaRN final-vector
argmax.

### Independent primitive references

`tests/test_cuda_laguna_kernels.c` remains unchanged as a separate
implementation reference for
weighted RMSNorm/RoPE, decode attention, prefill attention, KV-ring behavior,
and routed MoE. Those scalar routines are independent of Poolside and do not
call the CUDA functions under test, but they share DS4 constants and fixtures
and are not a second model oracle.

### Metamorphic execution checks

`tests/test_cuda_laguna_model.c` retains exact serialized-versus-batched and
serialized-versus-mixed-prefill comparisons. These do not establish model
truth, but they catch scheduling, shared-workspace, and session-isolation
errors that a single final-logit oracle may miss.

### Honest limitation

The contract no longer detects a model-level error shared by Poolside and the
captured GGUF conversion. The manifest and documentation must name the policy
`single-poolside-v1`. If a genuinely independent second runtime becomes
available later, introduce a new schema/policy and recapture; do not silently
reinterpret these fixtures as dual-oracle evidence.

## Fixture schema

Bump the promoted manifest schema to `laguna-resident-promoted-v2` and store
exactly four oracle vectors rather than eight:

```text
short.llama.f32
swa-513.llama.f32
yarn-8193.llama.f32
deep-32768.llama.f32
yarn-8193.continuation.i32
manifest.json
```

The fixed source inputs and three materialized prompts remain in the same
directory. Every `.f32` file contains exactly 100,352 little-endian float32
values (401,408 bytes). The continuation contains exactly eight little-endian
signed int32 IDs (32 bytes).

The manifest has this normative shape; no additional keys are accepted:

```json
{
  "schema": "laguna-resident-promoted-v2",
  "oracle_policy": "single-poolside-v1",
  "vocab_size": 100352,
  "continuation_case": "yarn-8193",
  "continuation_tokens": 8,
  "model": {
    "repository": "poolside/Laguna-S-2.1-GGUF",
    "revision": "706fa69799926b6afde1af9e24ca2a4923f110a1",
    "filename": "laguna-s-2.1-Q4_K_M.gguf",
    "size": 68248759648,
    "sha256": "e163b2c98908809a71245d6bb68b2226994d9969cb2a438eccb72196a1c4147a"
  },
  "oracle": {
    "name": "poolside-llama",
    "runtime_commit": "04b2b72cb54048ead292884adbe11f284e3ec950",
    "capture_manifest_sha256": "cc4fb338556c0895ff985edb5435ae7801be7dcb98c2958dc96a56d34f2c848e"
  },
  "provenance": {
    "contract_commit": "a250e43722945e293f6044bc7254c4806d5a7912",
    "tokenizer_runtime_commit": "<40 lowercase hex characters>",
    "generator_sha256": "<64 lowercase hex characters>",
    "benchmark_sha256": "<64 lowercase hex characters>",
    "poolside_seed_token_count": 61440,
    "ds4_seed_token_count": "<integer at least 32768>"
  },
  "thresholds": {
    "cuda_admission": {
      "centered_rms": 0.04,
      "centered_max_abs": 0.20,
      "top20_overlap": 18,
      "argmax_equal": true,
      "continuation_equal": true
    }
  },
  "cases": [
    {
      "id": "<one fixed case ID>",
      "render": "<chat or raw>",
      "prompt_hex": "<exact UTF-8 bytes as lowercase hex>",
      "prompt_sha256": "<64 lowercase hex characters>",
      "poolside_tokens": ["<exact frontier token IDs>"],
      "ds4_tokens": ["<the same exact frontier token IDs>"],
      "frontier": "<fixed positive integer>",
      "context": "<fixed positive integer>",
      "vector": {
        "file": "<case-id>.llama.f32",
        "sha256": "<64 lowercase hex characters>",
        "argmax": "<valid token ID>"
      }
    }
  ],
  "continuation": {
    "case": "yarn-8193",
    "file": "yarn-8193.continuation.i32",
    "sha256": "<64 lowercase hex characters>",
    "argmax": ["<exactly eight valid token IDs>"]
  }
}
```

`cases` contains exactly the four fixed cases in contract order. The
`contract_commit` is the immutable baseline above. `tokenizer_runtime_commit`
is the clean DS4 repository revision from which the executable used during
promotion was built; it is provenance, not another oracle or a binary
attestation.

Remove Metal runtime fields, Metal vectors, Metal metrics, and the old
oracle-to-oracle promotion thresholds. Verification rejects the v1 dual-oracle
schema rather than guessing how to migrate it.

## Promotion and verification interfaces

Promotion becomes:

```sh
LAGUNA_MODEL="$LAGUNA_MODEL" \
python3 gguf-tools/quality-testing/compare_laguna_logits.py \
  --ds4 ./ds4 \
  --llama "$LAGUNA_LLAMA_OUT" \
  --promote tests/test-vectors/laguna-resident
```

The promotion command must:

1. hash `capture.json` before parsing and require the pinned trust-anchor hash;
2. validate its exact schema, file allowlist, referenced-file hashes, runtime
   pin, and model pin;
3. validate the fixed fixture inputs and deterministic benchmark hash;
4. derive the three materialized prompts from the pinned capture, prove that
   they match deterministic generation and the captured prompt hashes, and
   either publish missing prompts or require existing prompts to be byte-exact;
5. discover the `--ds4` repository's full `HEAD`, require a clean tracked
   worktree and the fixed contract commit as an ancestor, and record that
   `HEAD` as `tokenizer_runtime_commit`;
6. invoke DS4 tokenization for all exact rendered prompts and the benchmark;
7. require DS4 and Poolside token IDs to match at every frontier and for the
   first 32,768 benchmark IDs;
8. require the YaRN final argmax to equal continuation step zero;
9. stage the missing prompts, four vectors, continuation, and manifest on the
   destination filesystem; and
10. publish with an exclusive sibling lock and no-clobber hard links, manifest
    last. Existing vectors, continuation, or manifest make promotion fail.

The canonical DGX branch containing the full contract commit must be fetched
and checked out before promotion; a synthetic checkout that cannot resolve the
commit is not eligible.

Verification becomes:

```sh
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

Verification remains non-mutating and fail-closed. It checks the exact allowed
file set, including rejecting directories and `__pycache__`, all hashes and
sizes, pins, prompt bytes, token arrays, continuation, argmaxes, and fixed
thresholds. The legacy `--metal`, `--dwarfstar-commit`, and `--ds4-commit`
arguments are rejected. Successful commands print exactly:

```text
promoted=<path> cases=4 vectors=4 oracle=poolside
verified=<path> cases=4 vectors=4 oracle=poolside
```

## CUDA admission

For every scenario, CUDA compares its final logits with the one promoted
Poolside vector using fixed ceilings:

```text
centered RMS <= 0.04
centered max absolute error <= 0.20
top-20 overlap >= 18
argmax == Poolside argmax
eight teacher-forced argmax IDs == promoted continuation IDs
```

These limits remain hard-coded. They must not be derived upward from the
observed CUDA disagreement and must not be widened merely to make a fixture
pass.

## Exact-context logits-only API

The `deep-32768` test requires logits after exactly 32,768 prompt tokens in a
32,768-token session. Ordinary `ds4_session_sync()` intentionally reserves one
position for generation and must continue rejecting equality.

Add a dedicated public `ds4_session_sync_logits_only()` API with these rules:

- shorter prompts behave exactly like ordinary sync on every supported mode;
- the exact-context exception is allowed only for local, non-TP,
  non-distributed Laguna CUDA sessions;
- ineligible exact-context calls fail before mutation or any TP/distributed
  message; and
- ordinary generation, server, agent, benchmark, scorer, and logprob paths
  remain unchanged.

An exact-context success permanently marks the session
`logits_only_terminal`. The supported operational surface in that state is
limited to `ds4_session_argmax()`, `ds4_session_argmax_excluding()`,
`ds4_session_top_logprobs()`, `ds4_session_token_logprob()`,
`ds4_session_copy_logits()`, `ds4_session_pos()`, `ds4_session_ctx()`, and
`ds4_session_free()`. The state cannot be cleared except by freeing the
session.

Ordinary or logits-only sync, direct/speculative/batch/mixed decode, rewrite,
rewind, invalidation, logit mutation, layer evaluation, and snapshot/payload
save or load all reject before mutation, export, or dispatch. Void mutators
such as rewind and invalidation become no-ops. A batch containing a terminal
session rejects the whole batch before any member advances. Position, tokens,
logits, and terminal state remain unchanged after every rejected operation.

Use this API only in the full-logit CLI path and the exact-context branch of
the Laguna model test. The model test first proves ordinary sync still rejects
the boundary, then proves logits-only sync succeeds and copies the logits. It
then proves direct decode, rewind followed by decode, rewrite, repeat sync,
snapshot, payload export, and batch decode all reject without changing state.

## Test strategy and sequencing

Every behavior change starts red.

1. Change Python contract tests first so they require one oracle, four vectors,
   v2 schema, the new CLI, exact provenance, non-mutation, and fail-closed
   tamper handling.
2. Implement the comparator/verifier migration and make those tests green.
3. Promote the already-validated, digest-pinned Poolside capture and run the
   same verifier again in a separate non-mutating invocation.
4. Change the C model test to consume four vectors and preserve all existing
   continuation and metamorphic scenarios.
5. Add failing model-independent policy tests and `DS4_TEST_HOOKS` terminal
   state-transition tests. The hooks construct a terminal test session without
   admitting Laguna CUDA in production. Make the policy and rejection tests
   green, then update only the two approved callers.
6. Make `test-cuda-laguna-resident` run the v2 verifier with hard pins before
   either C test. Build on DGX, run the verifier and primitive suites, and
   confirm the model-backed test remains red only at the still-closed CUDA
   engine-admission gate.
7. When resident CUDA admission is implemented, the DGX model-backed test must
   execute the real exact-context success and all terminal-state rejection
   checks. Laguna CUDA admission is not complete until that test passes.

Keep commits focused: contract-test red, comparator green, promoted fixtures,
four-vector C-test red, logits-only policy red, logits-only policy green, and
admission-target wiring. The former two-adjacent-test-commit assertion is
removed from the branch gate because the approved assurance model has changed.

## Failure handling

- Missing, extra, symlinked, malformed, wrong-sized, or wrong-hash contract
  artifacts fail verification.
- A stale capture digest, model, Poolside runtime, contract commit, or DS4
  tokenizer-runtime commit fails promotion and verification.
- Non-finite logits, token disagreement, invalid UTF-8 prompts, or a mismatched
  continuation step zero fail promotion.
- Promotion never overwrites an existing manifest, vector, continuation, or
  materialized prompt. An existing prompt is accepted only when byte-exact.
- The sibling lock and no-clobber publication prevent concurrent promoters
  from racing. A failure removes only files linked by that invocation and
  leaves pre-existing fixed inputs and prompts unchanged.
- A stale partial publication without a manifest fails and lists its paths; it
  requires explicit operator cleanup rather than automatic deletion.
- The DGX model-service stop/restore wrapper remains mandatory for any new
  full-model capture.

## Non-goals

- Treating a second `llama.cpp` revision as an independent oracle.
- Generating expected vectors from DS4 CUDA.
- Re-running the already-valid Poolside capture without a failed provenance
  check or an explicit model/runtime change.
- Generalizing logits-only exact-context support to other model families, TP,
  distributed execution, snapshots, or generation.
- Refactoring unrelated inference or fixture infrastructure.

## Later upgrade path

If a second independent DGX-compatible runtime becomes available, add a new
oracle policy and manifest schema, capture it from the same exact model
artifact and prompts, restore cross-oracle numerical promotion gates, and
recapture the fixtures. The single-Poolside v2 manifest remains auditable
historical evidence rather than being rewritten in place.
