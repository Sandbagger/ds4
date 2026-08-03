# Laguna Single-Poolside Oracle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the impossible Metal/Poolside dual-oracle Laguna contract with one digest-pinned Poolside oracle, promote and verify the v2 fixtures fail-closed, and add the narrow exact-context logits-only session API needed by the 32,768-token CUDA admission case.

**Architecture:** The Python comparator becomes a trust-boundary tool: it hashes the pinned Poolside `capture.json` before parsing, validates every referenced artifact and repository provenance, publishes a v2 single-oracle fixture with no-clobber hard links, and verifies it without mutation. The C session API keeps ordinary sync semantics unchanged, admits exact-context execution only for local non-TP Laguna CUDA, and permanently transitions a successful exact-context session into a read-only terminal state guarded at every mutation, export, and dispatch boundary. The resident-CUDA aggregate target verifies the fixture first, then runs the independent primitive suite and the model-backed contract; until CUDA engine admission lands, only the existing admission gate is expected to remain red.

**Tech Stack:** Python 3 standard library and `unittest`; C11 public session API; GNU Make; CUDA/nvcc on the DGX Spark; Git for immutable provenance.

---

## Scope and fixed decisions

- Work on branch `laguna-s2.1-resident-cuda` from design commit `19ae7d64aaf50b0e3e33fed976aeda4422faa2b6` or a descendant.
- Preserve contract checkpoint `a250e43722945e293f6044bc7254c4806d5a7912` as the immutable `contract_commit` and require it to be an ancestor of the clean tokenizer checkout used for promotion.
- Pin Poolside runtime `04b2b72cb54048ead292884adbe11f284e3ec950`, capture manifest SHA-256 `cc4fb338556c0895ff985edb5435ae7801be7dcb98c2958dc96a56d34f2c848e`, model revision `706fa69799926b6afde1af9e24ca2a4923f110a1`, model size `68248759648`, and model SHA-256 `e163b2c98908809a71245d6bb68b2226994d9969cb2a438eccb72196a1c4147a`.
- Keep `tests/test_cuda_laguna_kernels.c` unchanged; its scalar references remain the independent primitive layer.
- Keep serialized-versus-batched and serialized-versus-mixed checks in `tests/test_cuda_laguna_model.c` bit-exact.
- Do not admit Laguna on CUDA in this plan. The final model-backed command remains intentionally red at the pre-existing engine gate until the resident-CUDA implementation resumes.
- Do not retain compatibility with promoted-v1 or legacy `--metal`, `--dwarfstar-commit`, or `--ds4-commit` arguments.
- Do not recapture Poolside output unless the pinned capture fails provenance validation or the operator explicitly changes the model/runtime pins.

## File responsibility map

- `gguf-tools/quality-testing/test_compare_laguna_logits.py` — executable Python contract tests for v2 promotion, verification, tamper handling, repository provenance, and publication races.
- `gguf-tools/quality-testing/compare_laguna_logits.py` — the only promotion/verifier implementation; owns pins, schema validation, token parity, and atomic publication.
- `tests/test-vectors/laguna-resident/{swa-513.prompt,yarn-8193.prompt,deep-32768.prompt}` — exact prompt bytes materialized from the pinned capture.
- `tests/test-vectors/laguna-resident/{short,swa-513,yarn-8193,deep-32768}.llama.f32` — four promoted Poolside final-logit vectors.
- `tests/test-vectors/laguna-resident/yarn-8193.continuation.i32` — eight teacher-forced Poolside continuation IDs.
- `tests/test-vectors/laguna-resident/manifest.json` — v2 manifest and all immutable provenance.
- `tests/test_cuda_laguna_model.c` — consumes only the four Poolside vectors, preserves metamorphic checks, and exercises real exact-context terminal behavior once CUDA admission opens.
- `tests/test_session_logits_only.c` — new model-independent policy and terminal-state test binary.
- `ds4.h` — declares `ds4_session_sync_logits_only()` and test-only hooks.
- `ds4.c` — owns the terminal flag, exact-context eligibility, pre-dispatch guards, and test-hook session construction.
- `ds4_cli.c` — uses logits-only sync only in the full-logit dump path.
- `Makefile` — builds/runs the model-independent policy test and orders v2 verification before either Laguna CUDA C suite.
- `tests/test-vectors/README.md` — documents the honest single-oracle assurance model and exact DGX commands.

## Commit map

Keep the sequence explicit, even though the two `red` commits intentionally fail their named targets:

1. `test: define single Poolside Laguna fixture contract` — Python contract red.
2. `feat: promote Laguna fixtures from pinned Poolside capture` — comparator/verifier green.
3. `test: add promoted single Poolside Laguna fixtures` — reviewed DGX artifacts.
4. `test: consume one Poolside Laguna model oracle` — four-vector CUDA model contract; still red only at admission.
5. `test: define logits-only terminal session policy` — model-independent C policy red.
6. `feat: add exact-context logits-only sessions` — policy green and two approved callers.
7. `test: gate resident Laguna CUDA with pinned verifier` — ordered target wiring and docs.

### Task 1: Define the v2 Python contract (RED)

**Files:**
- Modify: `gguf-tools/quality-testing/test_compare_laguna_logits.py:1-589`
- Reference: `docs/superpowers/specs/2026-08-01-laguna-single-poolside-oracle-design.md`

- [ ] **Step 1: Replace dual-oracle test constants and helpers**

Replace `DWARFSTAR_COMMIT`, dual-oracle vector builders, native-Metal capture helpers, and v1 manifest construction with these fixed identities:

```python
CONTRACT_COMMIT = "a250e43722945e293f6044bc7254c4806d5a7912"
TOKENIZER_RUNTIME_COMMIT = "0123456789abcdef0123456789abcdef01234567"
LLAMA_COMMIT = "04b2b72cb54048ead292884adbe11f284e3ec950"
CAPTURE_MANIFEST_SHA256 = "cc4fb338556c0895ff985edb5435ae7801be7dcb98c2958dc96a56d34f2c848e"
ORACLE_POLICY = "single-poolside-v1"
PROMOTED_SCHEMA = "laguna-resident-promoted-v2"
```

`write_promoted_fixture()` must write exactly one `<case>.llama.f32` per case and the normative v2 manifest. Its model object uses `filename`, not `file`; each case has `poolside_tokens`, `ds4_tokens`, and singular `vector`; continuation has singular `argmax`.

- [ ] **Step 2: Make verification use only the v2 CLI**

Change `run_verify()` to invoke:

```python
[
    sys.executable,
    str(TOOL),
    "--verify-promoted", str(fixture),
    "--contract-commit", CONTRACT_COMMIT,
    "--tokenizer-runtime-commit", TOKENIZER_RUNTIME_COMMIT,
    "--llama-commit", LLAMA_COMMIT,
    "--capture-manifest-sha256", CAPTURE_MANIFEST_SHA256,
    "--gguf-size", str(GGUF_SIZE),
    "--gguf-sha256", GGUF_SHA256,
]
```

Assert exact success output:

```python
self.assertEqual(
    completed.stdout,
    f"verified={fixture} cases=4 vectors=4 oracle=poolside\n",
)
```

- [ ] **Step 3: Cover the exact promoted file set and non-mutation**

The valid fixture test must assert the directory contains exactly 13 regular files: four fixed source inputs, three materialized prompts, four `.llama.f32` vectors, continuation, and manifest. A directory or `__pycache__` is now an error rather than ignored. Hash the entire tree before and after verification and require equality.

- [ ] **Step 4: Add fail-closed v2 tamper subtests**

Keep each mutation isolated with `subTest`. At minimum cover:

```text
promoted-v1 schema
wrong oracle_policy
missing and extra files
directory, __pycache__, and symlink entries
wrong vector size, hash, argmax, or non-finite value
wrong model revision/filename/size/hash
wrong runtime, capture digest, contract commit, or tokenizer runtime commit
wrong generator/benchmark digest or token count
prompt hex/hash/UTF-8/file bytes mismatch
materialized prompt is not the exact byte prefix of the deterministic benchmark
Poolside/DS4 token mismatch or out-of-range ID
wrong case order/frontier/context
wrong continuation size/hash/argmax/step-zero parity
widened CUDA threshold or an extra manifest key
duplicate JSON keys
```

For every failed verification, assert `tree_digest(fixture)` is unchanged.

- [ ] **Step 5: Build a synthetic pinned Poolside capture helper**

Retain the current capture-v1 shape because the already-validated DGX capture uses it, but generate only the `oracle == "llama"` variant. Its `files` object must match the explicit 21-artifact allowlist rather than defining its own accepted set, and `seed_token_count` must be exactly `61440`. Make the helper return `sha256(capture.json bytes)` so direct function tests can inject the synthetic trust anchor while the production CLI remains hard-pinned.

- [ ] **Step 6: Build a clean temporary DS4 repository helper**

Clone the current repository locally into the test temp directory, place the fake executable as an untracked file, and return both its path and full `HEAD`. Do not synthesize a repository that lacks the real contract ancestor.

```python
subprocess.run(
    ["git", "clone", "--quiet", "--no-hardlinks", str(ROOT), str(repo)],
    check=True,
)
ds4 = repo / "fake-ds4"
ds4.write_text(FAKE_DS4, encoding="utf-8")
ds4.chmod(0o755)
head = subprocess.run(
    ["git", "-C", str(repo), "rev-parse", "HEAD"],
    check=True, capture_output=True, text=True,
).stdout.strip()
```

The fake executable must output the exact captured IDs for each materialized prompt and at least the first 32,768 benchmark IDs.

- [ ] **Step 7: Add promotion success and prompt-publication tests**

Call the implementation function directly with the synthetic capture digest and real contract ancestor. Assert:

- missing materialized prompts are published from the capture;
- pre-existing byte-exact prompts are accepted;
- a mismatched existing prompt fails without mutation;
- output is exactly `promoted=<path> cases=4 vectors=4 oracle=poolside` through the CLI runner;
- a separate verifier invocation succeeds and does not mutate the tree.

- [ ] **Step 8: Add provenance and capture trust-boundary failures**

Test stale capture digest, unknown/extra capture keys, unsafe capture filename, changed referenced file, wrong runtime/model pin, dirty tracked DS4 checkout, unresolved contract commit, contract not ancestor of `HEAD`, and malformed/non-full `HEAD`. Add a focused deterministic-generation negative: change one captured materialized prompt, refresh its capture file hash and the injected synthetic `capture.json` trust anchor so capture-internal integrity still passes, and require promotion to reject because those bytes are no longer the exact prefix of `benchmark-32768.txt`. Patch or inject only internal expected pins; do not add production CLI override flags.

- [ ] **Step 9: Add no-clobber and concurrency tests**

Cover existing vector, continuation, or manifest; stale partial publication without a manifest (stderr lists exact paths); an already-held sibling lock; two concurrent promoters with exactly one winner; and a forced link failure that cleans only paths linked by that invocation while preserving all pre-existing fixed inputs/prompts.

- [ ] **Step 10: Assert legacy CLI arguments are rejected**

Run subprocess cases containing `--metal`, `--dwarfstar-commit`, and `--ds4-commit`; assert argparse exits non-zero and no destination file changes.

- [ ] **Step 11: Run the changed tests and observe the intended failure**

Run:

```sh
python3 gguf-tools/quality-testing/test_compare_laguna_logits.py -v
```

Expected: FAIL because the implementation still accepts promoted-v1/two oracles and lacks the v2 CLI/provenance/publication behavior. Confirm failures are contract mismatches, not syntax or fixture-helper errors.

- [ ] **Step 12: Commit the red contract**

```sh
git add gguf-tools/quality-testing/test_compare_laguna_logits.py
git commit -m "test: define single Poolside Laguna fixture contract"
```

### Task 2: Implement v2 promotion and verification (GREEN)

**Files:**
- Modify: `gguf-tools/quality-testing/compare_laguna_logits.py:1-1137`
- Test: `gguf-tools/quality-testing/test_compare_laguna_logits.py`

- [ ] **Step 1: Replace top-level pins and file-set definitions**

Use the exact spec values and remove `ORACLES`, `PROMOTION_LIMITS`, Metal pins, and native-Metal helpers. Keep separate model constants for the existing capture-v1 (`"file"`) and promoted-v2 (`"filename"`) shapes; converting the capture key would reject the pinned artifact. Split fixture files into immutable checked-in inputs and promotion outputs:

```python
FIXED_INPUT_FILES = {
    "cases.json",
    "short.txt",
    "generate_benchmark_prompt.py",
    "benchmark-32768.txt",
}
MATERIALIZED_PROMPT_FILES = {
    "swa-513.prompt",
    "yarn-8193.prompt",
    "deep-32768.prompt",
}
PROMOTED_VECTOR_FILES = {
    f"{case_id}.llama.f32" for case_id, _, _, _ in CASE_SPECS
}
PROMOTED_OUTPUT_FILES = (
    MATERIALIZED_PROMPT_FILES
    | PROMOTED_VECTOR_FILES
    | {"yarn-8193.continuation.i32", "manifest.json"}
)
```

Set `PROMOTED_MODEL["filename"]` and fixed `CUDA_LIMITS["continuation_equal"] = True` exactly as the v2 schema states. `CAPTURE_MODEL["file"]` retains the same identity under the capture-v1 key name.

- [ ] **Step 2: Hash `capture.json` before parsing it**

Add a byte-oriented JSON loader so the order is observable and testable:

```python
capture_payload = read_bytes(root / "capture.json")
actual_capture_sha = sha256_bytes(capture_payload)
if actual_capture_sha != expected_capture_sha256:
    fail("Poolside capture.json trust-anchor SHA-256 mismatch")
capture = load_json_bytes(capture_payload, root / "capture.json")
```

Only after the digest matches may validation inspect keys, the explicit 21-artifact file allowlist, referenced hashes, runtime commit, model pin, four ordered cases, continuation rows, exact `seed_token_count == 61440`, and step-zero parity. Read every referenced artifact once, validate the hash on that payload, and retain that same payload for promotion; do not validate and later reopen a replaceable path.

- [ ] **Step 3: Tighten file validation**

`actual_files()` must reject every symlink and non-regular entry, including `__pycache__`. `verify_promoted()` must compare the exact v2 set before reading the manifest. Keep vector/continuation little-endian size checks and finite/range checks. Validate threshold member types before equality so Python's `True == 1` does not admit an integer in place of a boolean.

- [ ] **Step 4: Implement exact v2 manifest verification**

Change `verify_promoted()` to accept `contract_commit`, `tokenizer_runtime_commit`, `llama_commit`, `capture_manifest_sha256`, `gguf_size`, and `gguf_sha256`. Require exact keys at every level and reject promoted-v1 rather than migrating it. Verify:

```text
schema == laguna-resident-promoted-v2
oracle_policy == single-poolside-v1
four ordered cases and one singular vector per case
poolside_tokens == ds4_tokens
all prompt bytes/hashes and fixed frontier/context values
continuation binary == manifest.argmax
continuation[0] == yarn vector argmax
fixed CUDA ceilings, including continuation_equal
all model/oracle/provenance pins equal supplied hard pins
```

- [ ] **Step 5: Discover and validate tokenizer repository provenance**

Replace `discover_dwarfstar_commit()` with a helper that resolves the executable, discovers its repository top-level, reads the full `HEAD`, requires a clean tracked worktree, proves the fixed contract commit resolves, and runs `git merge-base --is-ancestor CONTRACT_COMMIT HEAD`. Use `--untracked-files=no` so the built `./ds4` binary does not make a canonical checkout dirty.

Every git command must use `check=False`, capture stderr, and turn failure into `ContractError`. Record `HEAD` as `tokenizer_runtime_commit`; do not accept a caller-supplied promotion commit.

- [ ] **Step 6: Validate deterministic prompts and token parity**

Validate fixed generator and benchmark hashes first and retain the exact benchmark bytes. For each non-short case, take the exact prompt from the pinned capture, require the capture file hash, require `prompt == benchmark[:len(prompt)]` (and reject a prompt longer than the benchmark), and require its token list to equal the corresponding prefix of the 32,768-token case. This byte-prefix check is the deterministic-generation proof; capture hashes alone are not sufficient. Accept an existing destination prompt only if byte-exact. Run DS4 tokenization for all four exact rendered prompts plus the benchmark. Require full per-frontier equality and first-32,768 benchmark equality with Poolside.

- [ ] **Step 7: Build only the v2 manifest and four payloads**

Each case entry must be:

```python
{
    "id": case_id,
    "render": render,
    "prompt_hex": prompt.hex(),
    "prompt_sha256": sha256_bytes(prompt),
    "poolside_tokens": poolside_case["tokens"],
    "ds4_tokens": ds4_tokens,
    "frontier": frontier,
    "context": context,
    "vector": {
        "file": f"{case_id}.llama.f32",
        "sha256": sha256_bytes(poolside_case["logits_payload"]),
        "argmax": poolside_case["argmax"],
    },
}
```

Use the normative `oracle`, `provenance`, `thresholds`, and `continuation` objects from the approved design verbatim. Do not compute cross-oracle metrics.

- [ ] **Step 8: Implement exclusive, no-clobber publication**

Acquire an exclusive sibling lock with atomic `mkdir` before the final destination/stale-partial validation, then hold it through manifest publication. Stage every output in a sibling temporary directory, run `verify_promoted()` on the staged full fixture, and publish with `os.link()` so an existing target raises rather than overwrites. Link materialized prompts only when missing; verify byte-exact existing prompts before staging. Immediately before linking, re-read full repository `HEAD`, recheck tracked cleanliness, and require both to match the provenance used during tokenization. Link manifest last.

Track only links created by the current invocation, including the staged inode identity. On failure, unlink in reverse order only when the destination still names that inode; never delete an externally replaced or pre-existing file. Always remove the invocation's staging directory and lock in `finally`. While holding the lock, if manifest is absent but any vector/continuation output exists, fail and list the stale paths for explicit operator cleanup.

- [ ] **Step 9: Replace the parser and exact output strings**

Promotion accepts only `--ds4`, `--llama`, and `--promote`. Verification accepts only the seven v2 identity arguments. Successful output is exactly:

```text
promoted=<path> cases=4 vectors=4 oracle=poolside
verified=<path> cases=4 vectors=4 oracle=poolside
```

- [ ] **Step 10: Run the focused suite until green**

Run:

```sh
python3 gguf-tools/quality-testing/test_compare_laguna_logits.py -v
python3 gguf-tools/quality-testing/test_generate_laguna_prompt.py -v
```

Expected: all tests PASS with no warnings.

- [ ] **Step 11: Check syntax and accidental legacy surface**

Run:

```sh
python3 -m py_compile gguf-tools/quality-testing/compare_laguna_logits.py
rg -n "metal|dwarfstar|promoted-v1|vectors=8" \
  gguf-tools/quality-testing/compare_laguna_logits.py \
  gguf-tools/quality-testing/test_compare_laguna_logits.py
```

Expected: compile succeeds; grep has no live legacy contract references (negative-test literals are allowed only in the test file).

- [ ] **Step 12: Commit the green implementation**

```sh
git add gguf-tools/quality-testing/compare_laguna_logits.py
git commit -m "feat: promote Laguna fixtures from pinned Poolside capture"
```

### Task 3: Promote the reviewed Poolside capture on the DGX

**Files:**
- Create: `tests/test-vectors/laguna-resident/swa-513.prompt`
- Create: `tests/test-vectors/laguna-resident/yarn-8193.prompt`
- Create: `tests/test-vectors/laguna-resident/deep-32768.prompt`
- Create: `tests/test-vectors/laguna-resident/short.llama.f32`
- Create: `tests/test-vectors/laguna-resident/swa-513.llama.f32`
- Create: `tests/test-vectors/laguna-resident/yarn-8193.llama.f32`
- Create: `tests/test-vectors/laguna-resident/deep-32768.llama.f32`
- Create: `tests/test-vectors/laguna-resident/yarn-8193.continuation.i32`
- Create: `tests/test-vectors/laguna-resident/manifest.json`

- [ ] **Step 1: Fetch the canonical branch on the DGX**

The checkout must contain and resolve `a250e43722945e293f6044bc7254c4806d5a7912`; do not copy files into a synthetic repository.

```sh
git fetch origin laguna-s2.1-resident-cuda
git switch laguna-s2.1-resident-cuda
git pull --ff-only
git cat-file -e a250e43722945e293f6044bc7254c4806d5a7912^{commit}
git merge-base --is-ancestor \
  a250e43722945e293f6044bc7254c4806d5a7912 HEAD
git status --short
```

Expected: all commands succeed and tracked status is clean.

- [ ] **Step 2: Build the tokenizer executable from that clean revision**

Build `./ds4` using the normal DGX CUDA build without editing tracked files. Record:

```sh
LAGUNA_TOKENIZER_RUNTIME_COMMIT="$(git rev-parse HEAD)"
test "${#LAGUNA_TOKENIZER_RUNTIME_COMMIT}" -eq 40
```

- [ ] **Step 3: Confirm the immutable capture trust anchor before promotion**

```sh
sha256sum \
  /home/will/artifacts/ds4/laguna-resident/a250e43/poolside-04b2b72/capture.json
```

Expected: `cc4fb338556c0895ff985edb5435ae7801be7dcb98c2958dc96a56d34f2c848e`.

- [ ] **Step 4: Run promotion once**

```sh
LAGUNA_MODEL="$LAGUNA_MODEL" \
python3 gguf-tools/quality-testing/compare_laguna_logits.py \
  --ds4 ./ds4 \
  --llama /home/will/artifacts/ds4/laguna-resident/a250e43/poolside-04b2b72 \
  --promote tests/test-vectors/laguna-resident
```

Expected exact stdout: `promoted=tests/test-vectors/laguna-resident cases=4 vectors=4 oracle=poolside`.

- [ ] **Step 5: Verify in a separate non-mutating invocation**

Record a digest of the fixture tree, run:

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

Expected exact stdout: `verified=tests/test-vectors/laguna-resident cases=4 vectors=4 oracle=poolside`; the before/after tree digests match.

- [ ] **Step 6: Review manifest and binary sizes**

Require four vectors of exactly 401,408 bytes, one continuation of exactly 32 bytes, no Metal files, and a manifest whose `tokenizer_runtime_commit` equals the captured full `HEAD`.

- [ ] **Step 7: Commit only the promoted artifacts**

```sh
git add \
  tests/test-vectors/laguna-resident/swa-513.prompt \
  tests/test-vectors/laguna-resident/yarn-8193.prompt \
  tests/test-vectors/laguna-resident/deep-32768.prompt \
  tests/test-vectors/laguna-resident/short.llama.f32 \
  tests/test-vectors/laguna-resident/swa-513.llama.f32 \
  tests/test-vectors/laguna-resident/yarn-8193.llama.f32 \
  tests/test-vectors/laguna-resident/deep-32768.llama.f32 \
  tests/test-vectors/laguna-resident/yarn-8193.continuation.i32 \
  tests/test-vectors/laguna-resident/manifest.json
git commit -m "test: add promoted single Poolside Laguna fixtures"
```

### Task 4: Consume four vectors in the CUDA model contract

**Files:**
- Modify: `tests/test_cuda_laguna_model.c:1-626`
- Modify: `ds4.h:358-363`
- Do not modify: `tests/test_cuda_laguna_kernels.c`

- [ ] **Step 1: Collapse the fixture type to one Poolside vector**

Replace:

```c
typedef struct {
    const char *id;
    float *metal;
    float *llama;
} oracle_case;
```

with:

```c
typedef struct {
    const char *id;
    float *poolside;
} oracle_case;
```

Load only `<case>.llama.f32`, free only `poolside`, and retain the eight-ID continuation.

- [ ] **Step 2: Compare CUDA once per scenario**

Rename `compare_session_oracles()` to `compare_session_oracle()` and call it exactly once:

```c
const bool ok = compare_one_oracle(
        scenario, "poolside", actual, fixture->poolside, session_argmax);
```

Keep hard-coded CUDA ceilings unchanged.

- [ ] **Step 3: Preserve all current scenario coverage**

Keep `short`, `swa-513`, `yarn-8193`, eight teacher-forced steps, `deep-32768`, two-session decode batching, and mixed prefill/decode comparison. Do not weaken `memcmp`-based serialized-versus-batched equality.

- [ ] **Step 4: Declare the wished-for API and make the deep expectation red**

Add the public declaration beside ordinary sync:

```c
/* Synchronize a prompt for logits inspection. Prompts shorter than the
 * context use ordinary sync. Exact-context success is restricted to local
 * Laguna CUDA and permanently makes the session logits-only terminal. */
int ds4_session_sync_logits_only(
        ds4_session *s, const ds4_tokens *prompt,
        char *err, size_t errlen);
```

Replace only the `deep-32768` call with a dedicated `run_deep_exact_context()` expectation. Keep `create_and_sync()` ordinary for every shorter case. The deep helper must expect ordinary equality sync to reject unchanged, logits-only sync to succeed, and the real terminal rejection matrix described in Task 6.

- [ ] **Step 5: Update diagnostic text**

The file header and final success line must say one pinned Poolside oracle and `vectors=4`; they must not imply independent Metal evidence.

- [ ] **Step 6: Build and observe the intended red link**

Run:

```sh
make tests/test_cuda_laguna_model
```

Expected now: link failure for the not-yet-implemented `ds4_session_sync_logits_only()`. Confirm the C source itself compiles and no other symbol or fixture error appears. Do not add a stub or bypass the engine gate.

- [ ] **Step 7: Commit the four-vector/red exact-context contract**

```sh
git add tests/test_cuda_laguna_model.c ds4.h
git commit -m "test: consume one Poolside Laguna model oracle"
```

### Task 5: Define the logits-only terminal policy (RED)

**Files:**
- Create: `tests/test_session_logits_only.c`
- Modify: `ds4.h:373-379`
- Modify: `Makefile:50,276-390`

- [ ] **Step 1: Declare narrow `DS4_TEST_HOOKS` helpers**

Expose only enough to construct one synthetic terminal session, one inert control session, evaluate the pure eligibility decision, and fingerprint private state:

```c
#ifdef DS4_TEST_HOOKS
typedef struct {
    int pos;
    bool checkpoint_valid;
    bool logits_only_terminal;
    uint64_t token_hash;
    uint64_t logit_hash;
} ds4_test_session_state;

int ds4_test_logits_only_sync_mode(
        bool native_cuda_build,
        bool laguna, ds4_backend backend,
        bool session_distributed, bool engine_distributed,
        bool transport_tensor_parallel, bool cuda_tensor_parallel,
        int prompt_len, int ctx_size);
int ds4_test_session_create_policy(
        ds4_session **out, int ctx_size, bool terminal);
int ds4_test_session_state(
        const ds4_session *s, ds4_test_session_state *out);
#endif
```

The production API must not expose a way to set or clear terminal state.

- [ ] **Step 2: Write the eligibility matrix test**

Require every non-equal prompt length to delegate to ordinary sync so existing shorter/over-context behavior is exact. For `prompt_len == ctx`, select exact mode only for Laguna + native CUDA + `DS4_BACKEND_CUDA` + no session-distributed state + engine distributed role `NONE` + no transport TP + no CUDA tensor parallelism. Include explicit CPU-only and ROCm-build negatives because ROCm also uses the `DS4_BACKEND_CUDA` enum value.

- [ ] **Step 3: Write the allowed-read test**

On a terminal hook session, prove these remain functional and state-preserving: `argmax`, `argmax_excluding`, `top_logprobs`, `token_logprob`, `copy_logits`, `pos`, and `ctx`. Then free the session through `ds4_session_free()`.

- [ ] **Step 4: Write the scalar mutation/dispatch rejection table**

For each operation, fingerprint before and after, assert rejection/no-op, and require the fingerprint to match:

```text
ordinary sync and repeat logits-only sync
direct eval, internal eval-argmax, and speculative eval
TP speculative cycle
rewrite and common-prefix access
sampling and set_logits
set_power, progress/display/cancel mutation, progress callback dispatch, GPU warmup
layer-slice reset, layer evaluation, and output-head evaluation
rewind and invalidation (void no-ops)
```

Assert the RNG and output buffers supplied to sampling/layer APIs are unchanged when applicable.

- [ ] **Step 5: Write payload/snapshot rejection tests**

Cover payload-size queries, stage/save/load payload, snapshot save/load, layer-payload size/save/load. Check before/after file offsets and sentinel buffers so rejection is proven to happen before temp-file creation, allocation, read, write, accelerator synchronization, or state mutation.

- [ ] **Step 6: Write whole-batch rejection tests**

For ordinary and mixed batches, include a terminal session and at least one inert control session. Put the normal member first and terminal member second so the test proves every member is scanned before member zero advances. Require an error naming terminal state and unchanged fingerprints for every member. Cover terminal as a decode item and as the mixed prefill session.

- [ ] **Step 7: Add the focused Make target**

Compile `tests/test_session_logits_only.o` with `-DDS4_TEST_HOOKS`; link it against `ds4_cpu_test_hooks.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_layer_pack.o`; add `test-session-logits-only-policy` to `.PHONY`; add the binary to `make test` and run it explicitly in that recipe; add its object/binary to `clean`.

- [ ] **Step 8: Run and observe the intended failure**

```sh
make test-session-logits-only-policy
```

Expected: link/test failure because the new API and hooks do not exist. The test source itself must compile cleanly.

- [ ] **Step 9: Commit the red policy**

```sh
git add tests/test_session_logits_only.c ds4.h Makefile
git commit -m "test: define logits-only terminal session policy"
```

### Task 6: Implement exact-context terminal sessions (GREEN)

**Files:**
- Modify: `ds4.c:49207-49260,50227-52459,58607-66838`
- Modify: `ds4_cli.c:744-826`
- Test: `tests/test_cuda_laguna_model.c:300-390,560-610`
- Test: `tests/test_session_logits_only.c`

- [ ] **Step 1: Add private state and central guards**

Add `bool logits_only_terminal;` to `struct ds4_session`. Add small helpers that (a) identify terminal state, (b) write one stable `"session is logits-only terminal"` error, and (c) choose `invalid`, `ordinary`, or `exact` sync mode from model/backend/distributed/TP/length inputs.

- [ ] **Step 2: Implement the public sync split without changing ordinary sync**

Keep `ds4_session_sync()` rejecting `prompt->len >= ctx_size`. Its terminal check must run before TP mirroring.

Implement `ds4_session_sync_logits_only()` so every non-equal length delegates to ordinary sync and preserves its exact TP/distributed/cancellation/error behavior:

```c
if (!s || !prompt || prompt->len != s->ctx_size) {
    return ds4_session_sync(s, prompt, err, errlen);
}
if (ds4_session_is_logits_only_terminal(s)) {
    return ds4_session_terminal_error(err, errlen);
}
if (!ds4_session_exact_logits_only_eligible(s)) {
    return ds4_session_exact_context_error(s, prompt, err, errlen);
}
int rc = ds4_session_sync_internal(
        s, prompt, true, err, errlen);
if (rc == 0) s->logits_only_terminal = true;
return rc;
```

Refactor the internal length check to accept equality only when `allow_exact_context` is true. The exact path must call the internal local prefill directly—never the TP-aware wrapper. Eligibility must require Laguna, a native non-ROCm CUDA build, `engine->backend == DS4_BACKEND_CUDA`, no session or engine distributed mode, `!engine->tp.active`, and `!engine->cuda_tensor_parallel`, all before any message or mutation. Add a defensive terminal check inside the internal sync path too.

- [ ] **Step 3: Guard public mutation, export, and dispatch entry points**

Put terminal checks at the outermost boundary, before any TP/distributed send, callback, temporary file, `memset(out, 0, sizeof(*out))`, file read/write, GPU synchronization, allocation, or checkpoint mutation. At minimum guard:

```text
ds4_session_sync / sync_logits_only / rewrite_from_common
ds4_session_sample / set_logits / gpu_warmup / eval
internal ds4_session_eval_argmax and ds4_session_eval_internal
ds4_sessions_eval_batch / eval_batch_with_prefill
ds4_session_eval_speculative_argmax / tp_spec_cycle
ds4_session_invalidate / rewind
ds4_session_layer_slice_reset / eval_layer_slice / eval_output_head_from_hc
ds4_session_payload_bytes / stage_payload / save_payload / load_payload
ds4_session_save_snapshot / load_snapshot
ds4_session_layer_payload_bytes / save_layer_payload / load_layer_payload
```

Void mutators become no-ops. Integer/result APIs return their normal error sentinel. Sampling must reject before changing RNG. Stage/snapshot APIs must reject before changing caller-owned outputs.

- [ ] **Step 4: Make batch rejection all-or-nothing**

Before the `count == 1` shortcut or any ordinary validation/backend dispatch, scan all decode sessions for terminal state. Do the same for both the prefill session and every decode item in mixed batches. If any is terminal, return the terminal error and touch no member.

- [ ] **Step 5: Keep only the approved read surface active**

Do not guard `argmax`, `argmax_excluding`, `top_logprobs`, `token_logprob`, `copy_logits`, `pos`, `ctx`, or `free`. For other mutable/accessor session APIs not used by those reads (`set_power`, progress/cancel setters/reporting, `common_prefix`, `sample`, `prefill_cap`, `tokens`), return a neutral error/empty value or no-op in terminal state so no unsupported operational path remains.

- [ ] **Step 6: Implement test-only construction and fingerprinting**

Under `DS4_TEST_HOOKS`, construct a zeroed session with a static inert CPU engine, deterministic checkpoint tokens/logits, allocated buffers, and optional terminal flag. Construct the terminal fixture at `pos < ctx` so tests prove guards depend on the terminal bit rather than accidental context exhaustion. The hook must not alter production admission. `ds4_session_free()` must release it normally. Fingerprint all private state needed to prove no mutation.

- [ ] **Step 7: Switch only the full-logit CLI caller**

In `run_logits_dump()`, replace `ds4_session_sync()` with `ds4_session_sync_logits_only()`. Do not change logprob dump, generation, server, agent, benchmark, scorer, or any other call site.

- [ ] **Step 8: Add the real deep-context model scenario**

The red `run_deep_exact_context()` from Task 4 already expresses the caller contract. Once the implementation links, it must create a 32,768-token session, assert ordinary sync fails with position/logits unchanged, assert logits-only sync succeeds, compare the Poolside vector, and capture terminal state.

In that real session, prove direct decode, rewind followed by decode, rewrite, ordinary repeat sync, logits-only repeat sync, snapshot save, staged payload export, and a batch containing the terminal session all reject while position and copied logits remain identical. Verify the independent nonterminal batch member also remains unchanged.

- [ ] **Step 9: Run focused tests**

```sh
make test-session-logits-only-policy
```

Expected: PASS with no warnings.

- [ ] **Step 10: Run local regression/build checks**

```sh
make ds4_test
./ds4_test --server
make ds4 ds4-server ds4-agent ds4-bench ds4-eval
make tests/test_cuda_laguna_model \
     tests/test_cuda_session_batch \
     tests/test_cuda_mixed_batch
```

Expected: targeted regression and all builds PASS. Run bare `make test` only on a host intentionally provisioned for its full model/GPU groups; otherwise record that environmental omission rather than weakening tests.

- [ ] **Step 11: Build and exercise the DGX model contract**

```sh
make tests/test_cuda_laguna_model
DS4_TEST_MODEL="$LAGUNA_MODEL" ./tests/test_cuda_laguna_model
```

Expected until admission lands: failure only at engine open. Once resident CUDA admission is implemented later, all exact-context success and terminal rejection checks must execute and pass.

- [ ] **Step 12: Commit the green API and callers**

```sh
git add ds4.c ds4_cli.c
git commit -m "feat: add exact-context logits-only sessions"
```

### Task 7: Gate the resident-CUDA target with the pinned verifier

**Files:**
- Modify: `Makefile:50,268-365`
- Modify: `tests/test-vectors/README.md:81-161`

- [ ] **Step 1: Rewire aggregate prerequisites to binaries**

Do not keep `test-cuda-laguna-resident` dependent on the two phony run targets; Make would run those before its recipe. Instead depend on the two built binaries and run all three gates in recipe order:

```make
test-cuda-laguna-resident: tests/test_cuda_laguna_kernels tests/test_cuda_laguna_model
	@test -n "$(LAGUNA_TOKENIZER_RUNTIME_COMMIT)" || \
	  { echo "LAGUNA_TOKENIZER_RUNTIME_COMMIT is required" >&2; exit 1; }
	python3 gguf-tools/quality-testing/compare_laguna_logits.py \
	  --verify-promoted tests/test-vectors/laguna-resident \
	  --contract-commit a250e43722945e293f6044bc7254c4806d5a7912 \
	  --tokenizer-runtime-commit "$(LAGUNA_TOKENIZER_RUNTIME_COMMIT)" \
	  --llama-commit 04b2b72cb54048ead292884adbe11f284e3ec950 \
	  --capture-manifest-sha256 cc4fb338556c0895ff985edb5435ae7801be7dcb98c2958dc96a56d34f2c848e \
	  --gguf-size 68248759648 \
	  --gguf-sha256 e163b2c98908809a71245d6bb68b2226994d9969cb2a438eccb72196a1c4147a
	env -u DS4_CUDA_MOE_DECODE_GRAPH ./tests/test_cuda_laguna_kernels --case all
	DS4_TEST_MODEL="$(DS4_TEST_MODEL)" ./tests/test_cuda_laguna_model
```

- [ ] **Step 2: Update the fixture documentation**

Rewrite the Laguna section to say:

- one full-model Poolside oracle under policy `single-poolside-v1`;
- scalar CUDA kernel references and metamorphic session checks are complementary assurance, not second full-model oracles;
- the honest shared-runtime/conversion limitation;
- the pinned capture path/hash and all model/runtime/contract pins;
- exact promotion, verification, and aggregate target commands;
- four vectors and one eight-ID continuation;
- no ordinary recapture and no legacy Metal CLI.

- [ ] **Step 3: Run static documentation/CLI checks**

```sh
rg -n "two-oracle|Metal/llama|vectors=8|--dwarfstar-commit|--metal" \
  tests/test-vectors/README.md \
  gguf-tools/quality-testing/compare_laguna_logits.py \
  tests/test_cuda_laguna_model.c \
  Makefile
```

Expected: no stale live contract text.

- [ ] **Step 4: Run the complete ordered DGX gate**

```sh
export LAGUNA_TOKENIZER_RUNTIME_COMMIT="$(
  python3 -c 'import json; print(json.load(open("tests/test-vectors/laguna-resident/manifest.json", encoding="utf-8"))["provenance"]["tokenizer_runtime_commit"])'
)"
test "${#LAGUNA_TOKENIZER_RUNTIME_COMMIT}" -eq 40
DS4_TEST_MODEL="$LAGUNA_MODEL" make test-cuda-laguna-resident
```

Expected before CUDA admission:

1. verifier prints the exact `verified=tests/test-vectors/laguna-resident cases=4 vectors=4 oracle=poolside` line;
2. `test_cuda_laguna_kernels --case all` passes unchanged;
3. the model binary loads all v2 fixtures and fails only at the existing CUDA engine-admission diagnostic (`Laguna S 2.1 currently requires --metal`).

Any earlier failure is a blocker. Do not reinterpret the expected admission failure as fixture or policy success.

- [ ] **Step 5: Run branch hygiene checks**

```sh
git diff --check
git status --short
git log --oneline --decorate -7
```

Expected: no whitespace errors; only intended Makefile/docs changes remain; commit order matches the map above.

- [ ] **Step 6: Commit target wiring and docs**

```sh
git add Makefile tests/test-vectors/README.md
git commit -m "test: gate resident Laguna CUDA with pinned verifier"
```

## Completion boundary and later admission work

This plan is complete when the v2 fixture/verifier and terminal API tests are green, promoted artifacts are committed, the primitive CUDA suite passes, and the aggregate target reaches only the unchanged Laguna CUDA engine-admission gate. Do not remove or bypass that gate here.

When resident CUDA engine admission is implemented in the follow-on plan, rerun `test-cuda-laguna-resident`. Admission is not complete until the model binary executes all four Poolside comparisons, eight teacher-forced continuation checks, both metamorphic batch checks, real `deep-32768` exact-context success, and every real terminal-state rejection check.
