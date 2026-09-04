# Laguna Compact CUDA SSD-Streaming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the pinned Poolside Laguna S 2.1 Q4_K_M artifact numerically correct on one DGX Spark CUDA device, then replace whole-model registration/residency with a genuinely bounded SSD-streaming path that publishes a machine-verifiable compact-runtime qualification bundle.

**Architecture:** First complete the already-approved single-Poolside oracle plan and make its resident-CUDA target green without admitting streaming. Then add two narrow modules: `ds4_laguna_stream` owns pure-C tensor/range, allocation-plan, page-range, grouping, and cache-state policy; `ds4_runtime` owns categorized counters, immutable snapshots, wire records, and build identity. CUDA mechanics remain in `ds4_cuda.cu` behind an engine-lifetime opaque compact context: it attaches the model fd/mapping without registering it, copies only ledger-approved static ranges, allocates fixed cache/staging storage once, and resolves all compact weights strictly. `ds4.c` owns Laguna graph scheduling and admission; Python qualification tooling owns cold preparation, exact-inode measurement, schemas, canonical evidence, and atomic publication. DS4 remains a foreground inference process and never manages peer services, ports, or co-residency.

**Tech Stack:** C11; CUDA C++/nvcc; POSIX `pread`, `madvise`, `posix_fadvise`, `mincore`, `fstat`, and signals on Linux; Python 3 qualification tooling and `unittest`; JSON Schema Draft 2020-12; RFC 8785 canonical JSON; GNU Make; the DGX Spark GB10 and its local NVMe.

---

## Scope, prerequisites, and execution rules

- Work in `/private/tmp/ds4-laguna-single-poolside-oracle-plan` on branch `laguna-s2.1-resident-cuda`.
- Treat [the compact-runtime design](../specs/2026-08-02-laguna-compact-cuda-streaming-design.md) as normative. Do not weaken a numerical, allocation, page-cache, protocol, or evidence gate to make a run pass.
- Execute every task in [the single-Poolside oracle plan](2026-08-01-laguna-single-poolside-oracle.md) first. That plan owns fixture promotion, four Poolside vectors, the eight-token continuation, and exact-context terminal sessions. Do not duplicate or redesign them here.
- Use `@superpowers-ruby:test-driven-development` for every code task: write the named failing test, run it and observe the intended failure, add only enough implementation to pass, rerun focused and regression checks, then commit.
- Use the pinned model only:

  ```text
  repository = poolside/Laguna-S-2.1-GGUF
  revision   = 706fa69799926b6afde1af9e24ca2a4923f110a1
  file       = laguna-s-2.1-Q4_K_M.gguf
  size       = 68248759648
  sha256     = e163b2c98908809a71245d6bb68b2226994d9969cb2a438eccb72196a1c4147a
  ```

- Set these once on the DGX before model-backed tasks:

  ```sh
  export LAGUNA_MODEL=/absolute/path/to/laguna-s-2.1-Q4_K_M.gguf
  ```

- After Task 0 promotes the fixture, derive its immutable tokenizer provenance
  from the verified manifest rather than from the branch's later `HEAD`:

  ```sh
  export LAGUNA_TOKENIZER_RUNTIME_COMMIT="$(
    python3 -c 'import json; print(json.load(open("tests/test-vectors/laguna-resident/manifest.json", encoding="utf-8"))["provenance"]["tokenizer_runtime_commit"])'
  )"
  test "${#LAGUNA_TOKENIZER_RUNTIME_COMMIT}" -eq 40
  ```

  Every later resident/streamed target uses this exported capture-time value.
  Recomputing it with `git rev-parse HEAD` after fixture promotion is invalid.

- Do not use Metal as a build or acceptance dependency. Preserve portable compilation where touched, but all model-backed acceptance runs are CUDA on the DGX Spark.
- Do not use `drop_caches`, daemonize DS4, choose a deployment port, start/stop peer models, add systemd units, or make a co-residency claim.
- Keep every commit single-concern. Do not combine cache policy, CUDA I/O, graph integration, public wire contracts, or qualification publication into one change.
- Qualification outputs are generated evidence, not ordinary source fixtures. Commit schemas, harnesses, tests, and documentation; do not commit a host-specific evidence bundle unless the operator explicitly asks.

## Critical dependency traps

The current branch has three tempting but invalid shortcuts. Preserve these as explicit regression assertions:

1. `ds4_gpu_set_model_map_spans()` currently calls `ds4_gpu_set_model_map()`, which either copies or `cudaHostRegister`s the complete model mapping. It is not a compact attachment API.
2. `ds4_sessions_eval_batch_cuda()` and the mixed prefill/decode path currently treat every non-GLM CUDA model as DeepSeek. Laguna must be excluded from those native graph paths before resident CUDA admission opens.
3. The CUDA selected-expert cache is currently a replace-on-layer buffer whose budget setters return zero/do nothing. It is not the fixed engine-lifetime `(layer_id, expert_id)` cache required by the design.

## Integration checkpoints

- **A — Resident oracle:** the existing `test-cuda-laguna-resident` target is completely green on the pinned artifact; streaming remains rejected.
- **B — Compact startup:** a streamed engine starts with zero whole-map registration/copy bytes and zero routed payload bytes resident at startup; every non-routed lookup is strict.
- **C — Streamed oracle:** short, SWA-513, YaRN-8193, deep-32768, continuation, serialized/batched, and serialized/mixed comparisons all pass at unchanged Poolside ceilings with 4,096 allocated prefill rows.
- **D — Bounded runtime:** fault, cancellation, page-advice, warm-growth, and two-session pressure tests reconcile and remain inside declared bounds.
- **E — Stable handoff:** all five normative wire schemas, runtime/request telemetry, token admission, protocol, lifecycle, benchmark, and eval contracts pass.
- **F — Qualification:** the immutable 8/12/16-GiB curve is complete and the canonical bundle, evidence index, and external sidecar verify.

## File responsibility map

- `ds4_laguna_stream.h`, `ds4_laguna_stream.c` — pure-C compact-runtime policy: tensor ledger records, exact expert ranges, allocation-plan arithmetic, slot state/victim selection, deterministic grouping, saturating counters, and inward page-range/union math.
- `ds4_runtime.h`, `ds4_runtime.c` — categorized allocation tracker, simultaneous peaks, hard-bound violations, model/executable identity, build information, runtime/request snapshots, UUID/sequence helpers, and JSON serialization.
- `ds4.c` — builds the Laguna ledger from bound tensor roles, stores the compact context on the engine, threads exact prefill rows, routes resident versus streamed graph calls, performs admission, and exposes runtime snapshots.
- `ds4.h` — public engine/session snapshot and request-metrics interfaces required by server/bench/eval.
- `ds4_gpu.h`, `ds4_cuda.cu` — opaque compact CUDA context, strict static-range placement, tracked CUDA/pinned allocations, exact-range reads/uploads, cache slot events/refcounts, streamed routed kernels, and test-only fault hooks.
- `ds4_ssd.h`, `ds4_ssd.c` — exact positive decimal byte parsing and compatibility parsing only; no policy selection.
- `ds4_cli.c`, `ds4_server.c`, `ds4_agent.c`, `ds4_bench.c`, `ds4_eval.c`, `ds4_help.c` — stable flags, version output, runtime/admission endpoints, per-request metrics, corrected benchmark/eval evidence, and foreground lifecycle.
- `tests/test_laguna_stream.c` — CPU tests for the pure policy and allocation arithmetic.
- `tests/test_cuda_laguna_stream.c` — CUDA startup, allocation, cache, I/O, fault, cancellation, advice, and pressure integration tests.
- `tests/test_cuda_laguna_model.c` — resident/streamed Poolside oracle modes and continuation/metamorphic checks.
- `tests/test_runtime.c`, `tests/test_runtime_contract.py`, `tests/test_laguna_server_contract.py` — C snapshot/serializer tests, schema boundary tests, and live child-process HTTP/signal tests.
- `schemas/*.schema.json` — benchmark manifest plus the five normative downstream wire schemas.
- `gguf-tools/quality-testing/compact_runtime_qualify.py` — immutable manifest builder, cold preparation, exact-inode measurement, benchmark runner, identity binding, gate evaluator, canonical bundle builder/verifier, and atomic publisher.
- `gguf-tools/quality-testing/test_compact_runtime_qualify.py` — manifest, invalid-run, artifact-binding, canonicalization, evidence-union, tamper, and publication tests.
- `Makefile`, `README.md`, `CONTRIBUTING.md`, `tests/test-vectors/README.md` — build wiring, focused targets, user-facing flags, and exact DGX commands.

## Commit map

1. `feat: admit resident Laguna on CUDA`
2. `test: freeze compact benchmark manifest contract`
3. `feat: add exact compact runtime options`
4. `feat: validate Laguna compact tensor ledger`
5. `feat: plan and track bounded Laguna allocations`
6. `feat: attach compact Laguna models without whole-map registration`
7. `feat: define deterministic Laguna expert cache policy`
8. `feat: back Laguna cache slots with bounded CUDA IO`
9. `feat: stream Laguna routed decode through fixed cache`
10. `feat: group over-capacity Laguna prefill`
11. `feat: allocate Laguna prefill scratch at the configured cap`
12. `feat: make compact runtime page disposal exact and measured`
13. `test: measure exact-inode and external compact footprint`
14. `test: gate Laguna warm stability and session pressure`
15. `test: freeze compact runtime wire schemas`
16. `feat: expose DS4 build identity and runtime snapshots`
17. `feat: expose request metrics and exact token admission`
18. `feat: lock compact server protocol and lifecycle semantics`
19. `feat: report qualification-safe benchmark and eval evidence`
20. `feat: run and publish canonical Laguna qualification`
21. `docs: add compact Laguna qualification runbook`
22. `docs: record compact Laguna qualification`

### Task 0: Complete and verify the single-Poolside prerequisite

**Files:**
- Execute: `docs/superpowers/plans/2026-08-01-laguna-single-poolside-oracle.md`
- Verify: `tests/test-vectors/laguna-resident/manifest.json`
- Verify: `gguf-tools/quality-testing/compare_laguna_logits.py`
- Verify: `tests/test_session_logits_only.c`
- Verify: `tests/test_cuda_laguna_model.c`
- Verify: `Makefile`

- [ ] **Step 1: Execute all seven prerequisite tasks and commits**

Follow the prerequisite plan, with one correction: its final Task 7 command must use the capture-time `tokenizer_runtime_commit` from the promoted manifest, not the then-current branch `HEAD`. Preserve/export that value immediately after promotion and use it for every later verifier invocation. Stop if fixture provenance, token parity, comparator thresholds, terminal-state policy, or the primitive CUDA suite is red.

- [ ] **Step 2: Confirm the intended handoff state**

```sh
export LAGUNA_TOKENIZER_RUNTIME_COMMIT="$(
  python3 -c 'import json; print(json.load(open("tests/test-vectors/laguna-resident/manifest.json", encoding="utf-8"))["provenance"]["tokenizer_runtime_commit"])'
)"
test "${#LAGUNA_TOKENIZER_RUNTIME_COMMIT}" -eq 40
DS4_TEST_MODEL="$LAGUNA_MODEL" make test-cuda-laguna-resident
```

Expected: the promoted fixture verifies, `test_cuda_laguna_kernels --case all` passes, and the model test fails only with the unchanged diagnostic that Laguna currently requires Metal. Any earlier failure belongs to the prerequisite plan.

- [ ] **Step 3: Record the prerequisite revision**

```sh
git rev-parse HEAD
git status --short
```

Expected: a clean worktree and a revision containing the seven prerequisite commits. Copy that revision into the implementation trace/PR notes; do not edit this plan with a moving hash.

### Task 1: Admit resident Laguna on CUDA safely

**Files:**
- Modify: `ds4.c:57418-57468,58645-58656,65839-66045`
- Modify: `tests/test_cuda_laguna_model.c:357-557`
- Modify: `Makefile:348-365`
- Modify: `README.md` Laguna support paragraph

- [ ] **Step 1: Add resident family-dispatch assertions**

Extend the existing two-session and mixed-session model tests so they require Laguna to use the correctness fallback, never `metal_graph_encode_token_raw_swa()` or `metal_graph_eval_mixed_prefill_decode()`. Under `DS4_TEST_HOOKS`, add counters for Laguna fallback calls and assert both counters advance.

- [ ] **Step 2: Run the RED resident target**

```sh
DS4_TEST_MODEL="$LAGUNA_MODEL" make test-cuda-laguna-resident
```

Expected: verifier and primitive tests pass; model-backed execution fails at the Metal-only engine gate.

- [ ] **Step 3: Exclude Laguna from DeepSeek-native CUDA batching**

In `ds4_sessions_eval_batch_cuda()` and `ds4_sessions_eval_batch_with_prefill_cuda()`, require both `!ds4_session_is_glm(...)` and `!ds4_session_is_laguna(...)` before entering the DeepSeek-native path. Laguna then uses the established serialized correctness fallback, which preserves private KV/checkpoint state.

- [ ] **Step 4: Open resident CUDA only**

Change Laguna engine validation to accept `DS4_BACKEND_METAL` or `DS4_BACKEND_CUDA`, keep `e->ssd_streaming` rejected, keep multi-GPU/TP/distributed/slice rejection, and make diagnostics say “graph backend” rather than “Metal”. Do not touch whole-map CUDA setup in this task; it is the like-for-like resident baseline.

- [ ] **Step 5: Reach checkpoint A and run regressions**

```sh
DS4_TEST_MODEL="$LAGUNA_MODEL" make test-cuda-laguna-resident
make cuda-regression
make test
```

Expected: all three commands pass; the resident model test reports four Poolside vectors, eight continuation tokens, and both metamorphic session checks.

- [ ] **Step 6: Commit**

```sh
git add ds4.c tests/test_cuda_laguna_model.c Makefile README.md
git commit -m "feat: admit resident Laguna on CUDA"
```

### Task 2: Freeze the compact benchmark manifest contract

**Files:**
- Create: `schemas/compact-runtime-benchmark-v1.schema.json`
- Create: `gguf-tools/quality-testing/compact_runtime_qualify.py`
- Create: `gguf-tools/quality-testing/test_compact_runtime_qualify.py`
- Modify: `Makefile`
- Reference: `tests/test-vectors/laguna-resident/generate_benchmark_prompt.py`
- Reference: `tests/test-vectors/laguna-resident/benchmark-32768.txt`

- [ ] **Step 1: Write RED manifest tests**

Cover exact rendered prompt bytes and SHA-256 at 512/2048/8192/28672 native-template tokens, output ceiling 512, temperature 0, seed 1, all remaining sampling values, stop sequences, tokenizer/template revision, cache order `[8,12,16]` GiB, counterbalanced prompt order, one cold plus exactly three warm repetitions, 45-minute whole-request and 15-minute TTFT timeouts, all host/device/filesystem identity fields, and the four required eval IDs. Reject unknown keys, placeholders, reordered profiles, missing prompt bytes, and hashes computed after result fields exist.

- [ ] **Step 2: Run the RED suite**

```sh
python3 gguf-tools/quality-testing/test_compact_runtime_qualify.py -v
```

Expected: import/file errors naming the missing schema and manifest builder.

- [ ] **Step 3: Implement only manifest build/verify**

Add `build_manifest()`, `validate_manifest()`, and `manifest_sha256()` using strict duplicate-key rejection, finite JSON values, explicit allowlists, and canonical input ordering. Expose only `manifest build --model PATH --output FILE` and `manifest verify --manifest FILE` at this stage; do not add process execution or bundle publication yet. The builder must write the complete manifest to a temporary file before any benchmark result is visible.

- [ ] **Step 4: Make the manifest suite green**

```sh
python3 gguf-tools/quality-testing/test_compact_runtime_qualify.py -v
python3 -m json.tool schemas/compact-runtime-benchmark-v1.schema.json >/dev/null
```

Expected: all manifest tests pass and the schema parses.

- [ ] **Step 5: Wire and commit**

Add `test-laguna-compact-python` to `Makefile`, running the unittest file directly.

```sh
git add schemas/compact-runtime-benchmark-v1.schema.json \
  gguf-tools/quality-testing/compact_runtime_qualify.py \
  gguf-tools/quality-testing/test_compact_runtime_qualify.py Makefile
git commit -m "test: freeze compact benchmark manifest contract"
```

### Task 3: Add exact compact-runtime options

**Files:**
- Modify: `ds4.h:132-184`
- Modify: `ds4_ssd.h`, `ds4_ssd.c`
- Create: `tests/test_laguna_stream.c`
- Modify: `ds4_cli.c:1886-1910`
- Modify: `ds4_server.c:12970-13040`
- Modify: `ds4_agent.c` option parser
- Modify: `ds4_bench.c:287-311`
- Modify: `ds4_eval.c:1603-1627`
- Modify: `ds4_help.c:146-197`
- Modify: `tests/test_gpu_args_cli.sh`
- Modify: `Makefile`

- [ ] **Step 1: Test strict decimal byte parsing and option conflicts**

Add `ds4_parse_positive_u64_decimal()` tests for `1`, `8589934592`, and `18446744073709551615`; reject zero, sign characters, whitespace, suffixes, leading zeroes, overflow, and trailing junk. Add CLI smoke cases requiring canonical `--ssd-streaming-cache-bytes BYTES` on every inference binary and `--session-slots N` on the server. Passing both a canonical option and its compatibility alias with different values must exit `2` before model open.

- [ ] **Step 2: Observe RED**

```sh
make tests/test_laguna_stream && ./tests/test_laguna_stream --case options
./tests/test_gpu_args_cli.sh
```

Expected: the unit build or new assertions fail because canonical parsers/options do not exist.

- [ ] **Step 3: Implement canonical fields and compatibility aliases**

Add exact-value/set booleans to `ds4_engine_options`. `--ssd-streaming-cache-bytes` sets the byte field and exact flag. `--session-slots` maps to the existing server slot count; retain `--batched-session` as a deprecated equal-value alias. Keep `--ssd-streaming-cache-experts` for compatibility, but qualification rejects its use.

- [ ] **Step 4: Remove silent rewriting for exact values**

When the exact byte flag is set, reject an unsafe or impossible value with exit `2`; never overwrite it in the `ds4_streaming_manual_cache_safe_bytes()` path. Runtime must later report `effective_cache_limit == configured_cache_limit`. Preserve legacy behavior only for the deprecated spelling outside qualification.

- [ ] **Step 5: Build all parsers and run focused tests**

```sh
make tests/test_laguna_stream ds4 ds4-server ds4-agent ds4-bench ds4-eval
./tests/test_laguna_stream --case options
./tests/test_gpu_args_cli.sh
```

Expected: all pass; help shows canonical spellings; invalid canonical values return `2` without model-loading diagnostics.

- [ ] **Step 6: Commit**

```sh
git add ds4.h ds4_ssd.h ds4_ssd.c tests/test_laguna_stream.c \
  ds4_cli.c ds4_server.c ds4_agent.c ds4_bench.c ds4_eval.c ds4_help.c \
  tests/test_gpu_args_cli.sh Makefile
git commit -m "feat: add exact compact runtime options"
```

### Task 4: Validate the Laguna compact tensor ledger

**Files:**
- Create: `ds4_laguna_stream.h`
- Create: `ds4_laguna_stream.c`
- Modify: `ds4.c:4446-6808,57380-57418`
- Modify: `tests/test_laguna_stream.c`
- Modify: `Makefile`

- [ ] **Step 1: Define the pure descriptor contract in tests**

Use synthetic tensor descriptors to require exactly one class per tensor: `STATIC`, `ROUTED_EXPERT`, or `METADATA`. Routed records must bind `(layer, expert, gate/up/down)` exact source ranges. Cover valid Laguna layout, duplicate classification, unclassified tensor, overlap, truncation, integer overflow, wrong projection, inconsistent expert sizes, out-of-range layer/expert, and a quantized non-routed tensor that must remain `STATIC`.

- [ ] **Step 2: Observe RED**

```sh
make tests/test_laguna_stream && ./tests/test_laguna_stream --case ledger
```

Expected: compile failure for the missing ledger API.

- [ ] **Step 3: Add minimal pure-C types and validation**

Define `ds4_laguna_tensor_desc`, `ds4_laguna_tensor_range`, `ds4_laguna_expert_entry`, and `ds4_laguna_ledger`. Provide:

```c
bool ds4_laguna_ledger_build(
    ds4_laguna_ledger *out,
    const ds4_laguna_tensor_desc *tensors,
    size_t n_tensors,
    uint64_t file_size,
    uint64_t device_alignment,
    char *err,
    size_t errlen);
void ds4_laguna_ledger_free(ds4_laguna_ledger *ledger);
```

Victim/cache policy does not belong in this commit.

- [ ] **Step 4: Build real descriptors from bound roles**

Immediately after `weights_bind()` and Laguna layout validation, enumerate `ds4_model.tensors`. Determine routed identity from the exact `ffn_gate_exps`, `ffn_up_exps`, and `ffn_down_exps` pointers bound by `weights_bind_laguna_layer()`, not from quantization type or name substring. Fail engine open before GPU initialization if any model tensor is absent, duplicated, overlapping, or outside the opened file.

- [ ] **Step 5: Verify synthetic and pinned inspect paths**

```sh
make tests/test_laguna_stream && ./tests/test_laguna_stream --case ledger
make cpu
./ds4 --inspect -m "$LAGUNA_MODEL"
```

Expected: CPU policy tests pass; inspect prints exact static/routed/metadata byte totals and the maximum aligned entry stride without allocating CUDA memory.

- [ ] **Step 6: Wire objects and commit**

Add `ds4_laguna_stream.o` to `CORE_OBJS`, `CPU_CORE_OBJS`, the ROCm override, object dependencies, test link lines, and `clean`.

```sh
git add ds4_laguna_stream.h ds4_laguna_stream.c ds4.c \
  tests/test_laguna_stream.c Makefile
git commit -m "feat: validate Laguna compact tensor ledger"
```

### Task 5: Plan and track bounded Laguna allocations

**Files:**
- Create: `ds4_runtime.h`
- Create: `ds4_runtime.c`
- Modify: `ds4_laguna_stream.h`, `ds4_laguna_stream.c`
- Modify: `ds4.h`, `ds4.c`
- Modify: `ds4_cli.c`, `ds4_server.c`, `ds4_agent.c`, `ds4_bench.c`, `ds4_eval.c`
- Modify: `tests/test_laguna_stream.c`
- Modify: `tests/test_gpu_args_cli.sh`
- Modify: `Makefile`

- [ ] **Step 1: Test allocation-plan arithmetic, attribution, and simultaneous peaks**

Cover static aligned bytes, `slot_count=floor(configured_cache_bytes/slot_stride_bytes)`, charged slot padding, cache metadata/address tables, KV, graph/scratch, four fixed pinned staging buffers, other host/CUDA call sites, and exact sum reconciliation. Require 8/12/16-GiB profiles to fit 24/28/32-GiB total ceilings with non-cache at most 16 GiB. Test uint64 saturation, overlapping address ranges, duplicate physical attribution, unclassified call site, category overrun, total overrun, and resident reduction thresholds of at least 32 GiB and 0.45.

- [ ] **Step 2: Observe RED**

```sh
make tests/test_laguna_stream && ./tests/test_laguna_stream --case allocation
```

Expected: missing plan/tracker symbols or failed assertions.

- [ ] **Step 3: Implement plan and tracker primitives**

Define the exact categories from the design and a tracker that updates `owned_total_current` after every event and `owned_total_peak` from that simultaneous sum. Every physical host/CUDA allocation event also records its base address, requested/charged size, category, physical domain, allocation call-site ID, and registration/managed relationship in a copy-only attribution table. Registration bytes are metadata only; managed allocation is charged once to `OTHER_CUDA`; mapped virtual bytes are never physically charged. Reject overlapping or multiply charged physical identities. A violation latches permanently and returns an unsafe status.

- [ ] **Step 4: Add pre-allocation qualification-plan output**

Add harness-only `--qualification-plan FILE` to the shared engine options and
parse it in `ds4`, `ds4-server`, `ds4-agent`, `ds4-bench`, and `ds4-eval`.
Reject a duplicate/conflicting occurrence with exit `2` before model open and
keep it out of ordinary public help. After model/ledger validation but before
`ds4_gpu_init()` or any model-lifetime allocation, write the immutable plan
and SHA-256 using temporary-file + `fsync` + rename. Include the opened-model
stat identity, ledger digest, every exact tensor range/class, the normalized
union of safe full-page cold-preparation ranges, unavoidable
metadata/shared-boundary bytes, all category/call-site bounds, and requested
total bound. Reject placeholders, an unclassified allocation call site, a
requested profile that cannot fit its declared total, or any plan-path failure
with exit `2`. This is the machine-readable ledger evidence consumed by Task
13; the harness never reparses GGUF names independently.

- [ ] **Step 5: Make policy tests green**

```sh
make tests/test_laguna_stream && ./tests/test_laguna_stream --case allocation
./tests/test_gpu_args_cli.sh
make cpu
```

Expected: all allocation arithmetic tests pass and CPU binaries link both new modules.

- [ ] **Step 6: Wire objects and commit**

Add `ds4_runtime.o` everywhere `ds4_laguna_stream.o` was added in Task 4.

```sh
git add ds4_runtime.h ds4_runtime.c ds4_laguna_stream.h \
  ds4_laguna_stream.c ds4.h ds4.c ds4_cli.c ds4_server.c ds4_agent.c \
  ds4_bench.c ds4_eval.c tests/test_laguna_stream.c \
  tests/test_gpu_args_cli.sh Makefile
git commit -m "feat: plan and track bounded Laguna allocations"
```

### Task 6: Attach compact models without whole-map registration

**Files:**
- Modify: `ds4_gpu.h:90-110`
- Modify: `ds4_cuda.cu:541-800,3124-3405,27079-27105`
- Modify: `ds4.c:57700-58135`
- Create: `tests/test_cuda_laguna_stream.c`
- Modify: `Makefile`

- [ ] **Step 1: Add RED compact-startup CUDA cases**

Add `--case startup` assertions for model attachment identity, exact static range copies, strict lookup hits, strict routed misses, zero whole-map registered bytes, zero whole-map copied bytes, zero routed payload bytes at startup, no opportunistic range allocation, and clean teardown. Inject overlapping/truncated ranges and all known full-map/cache environment options; each must fail before a model allocation.

- [ ] **Step 2: Observe RED on the DGX**

```sh
make tests/test_cuda_laguna_stream
./tests/test_cuda_laguna_stream --case startup
```

Expected: missing compact-context API or assertions showing full-map registration.

- [ ] **Step 3: Add an opaque engine-lifetime CUDA context**

Declare in `ds4_gpu.h`:

```c
typedef struct ds4_gpu_laguna_compact ds4_gpu_laguna_compact;
int ds4_gpu_laguna_compact_create(
    ds4_gpu_laguna_compact **out,
    int model_fd,
    const void *model_map,
    uint64_t model_size,
    const ds4_laguna_ledger *ledger,
    const ds4_laguna_allocation_plan *plan,
    ds4_runtime_tracker *tracker);
void ds4_gpu_laguna_compact_destroy(ds4_gpu_laguna_compact *ctx);
```

Creation records fd/base/size without `cudaHostRegister`, allocates one tracked static CUDA slab, copies only ledger-approved static ranges, and installs strict offset lookups. Exactly one compact context may be active per process.

- [ ] **Step 4: Make compact resolution fail closed**

When the active model map belongs to the compact context, the single-GPU branch of `cuda_resolve_weight_ptr()` must use strict static/cache lookup. A miss returns `NULL`; it never calls `cuda_model_range_ptr()`, registers a page, allocates an arena, copies from the mmap, or falls back to resident/HMM access.

- [ ] **Step 5: Select compact startup only for Laguna streaming**

In `ds4_engine_open()`, preserve the resident CUDA baseline. For Laguna plus canonical streaming, construct the context from the validated ledger/plan and bypass `ds4_gpu_set_model_map()`, `_range()`, `_spans()`, accelerator tensor caches, and warm-weight paths. Reject `DS4_CUDA_COPY_MODEL`, `DS4_CUDA_COPY_MODEL_CHUNKED`, `DS4_CUDA_WEIGHT_CACHE`, `DS4_CUDA_WEIGHT_PRELOAD`, or any equivalent whole-map option before creation.

- [ ] **Step 6: Reach checkpoint B**

```sh
make tests/test_cuda_laguna_stream
./tests/test_cuda_laguna_stream --case startup
DS4_TEST_MODEL="$LAGUNA_MODEL" ./tests/test_cuda_laguna_stream --case model-startup
```

Expected: both pass; the pinned model reports exact static bytes, zero routed startup bytes, and both whole-map counters as zero.

- [ ] **Step 7: Commit**

```sh
git add ds4_gpu.h ds4_cuda.cu ds4.c tests/test_cuda_laguna_stream.c Makefile
git commit -m "feat: attach compact Laguna models without whole-map registration"
```

### Task 7: Define deterministic Laguna expert-cache policy

**Files:**
- Modify: `ds4_laguna_stream.h`, `ds4_laguna_stream.c`
- Modify: `tests/test_laguna_stream.c`

- [ ] **Step 1: Write RED state-machine tests**

Test keys `(layer_id, expert_id)`, states `EMPTY/LOADING/READY/IN_USE`, generation changes, in-flight refcounts, publish-after-completion, hit reuse across token steps, saturating `route_hotness`, monotonic `last_used`, and exact victim ordering: lowest hotness, oldest last-used, then lowest key. Cover duplicate acquire, all-pinned refusal, stale completion generation, failed load rollback, cancellation in every state, and teardown with in-flight references.

- [ ] **Step 2: Test deterministic grouping separately**

Given more unique selected experts than slots, require stable first-occurrence groups no larger than the available slots, with identical input producing byte-identical groups. Reject a cache smaller than one token's maximum per-layer selected set.

- [ ] **Step 3: Observe RED**

```sh
make tests/test_laguna_stream && ./tests/test_laguna_stream --case cache-policy
```

Expected: missing cache-policy symbols.

- [ ] **Step 4: Implement pure state only**

Add `ds4_laguna_cache_policy_init/acquire/publish/pin/unpin/fail/cancel/drain` and grouping helpers. They own no threads, fds, CUDA pointers, or allocation. Every transition validates invariants and returns `RECOVERABLE`, `UNSAFE`, or success explicitly.

- [ ] **Step 5: Make tests green and commit**

```sh
make tests/test_laguna_stream
./tests/test_laguna_stream --case cache-policy
git add ds4_laguna_stream.h ds4_laguna_stream.c tests/test_laguna_stream.c
git commit -m "feat: define deterministic Laguna expert cache policy"
```

### Task 8: Back fixed cache slots with bounded CUDA I/O

**Files:**
- Modify: `ds4_gpu.h`
- Modify: `ds4_cuda.cu:126-174,1458-1738,20866-20910,22880-23090,27382-27560`
- Modify: `tests/test_cuda_laguna_stream.c`
- Modify: `Makefile`

- [ ] **Step 1: Add RED allocation, reuse, and fault cases**

Add `--case cache-io` and `--case cache-faults`. Assert cache payload/staging allocations occur exactly once at context creation, stay at or below the configured byte ceiling, and never grow under pressure. The second acquire of the same key must be a hit with zero model-file bytes. Inject `EINTR`, EOF, short read, hard I/O error, CUDA copy failure, event-record failure, event-completion failure, cancellation, and teardown in `LOADING/IN_USE`.

- [ ] **Step 2: Observe RED**

```sh
make tests/test_cuda_laguna_stream
./tests/test_cuda_laguna_stream --case cache-io --case cache-faults
```

Expected: current replace-on-layer cache reallocates/reloads or lacks the fault hooks.

- [ ] **Step 3: Allocate fixed payload and staging from the plan**

Inside `ds4_gpu_laguna_compact`, allocate `slot_count * slot_stride_bytes` once plus exactly the planned number and size of pinned staging buffers/events. Charge every allocation through `ds4_runtime_tracker`; no `cudaMalloc`, `cudaMallocManaged`, `cudaHostAlloc`, or registration in the compact path may bypass a categorized wrapper.

- [ ] **Step 4: Implement exact-range load and publication**

Use the opened model fd and `pread` loop; retry `EINTR` at the same offset and fail on zero/short terminal reads. Copy gate/up/down payload into the fixed slot, record a completion event, and publish the key only after event success. A recoverable failure restores `EMPTY`, releases pins/capacity, increments typed counters, and never calls a resident resolver. An invariant-restoration failure latches unsafe state.

- [ ] **Step 5: Make fault tests green**

```sh
make tests/test_cuda_laguna_stream
./tests/test_cuda_laguna_stream --case cache-io --case cache-faults
./tests/test_cuda_laguna_stream --case startup
```

Expected: all pass; tracked current returns to baseline after teardown and peak never exceeds plan.

- [ ] **Step 6: Commit**

```sh
git add ds4_gpu.h ds4_cuda.cu tests/test_cuda_laguna_stream.c Makefile
git commit -m "feat: back Laguna cache slots with bounded CUDA IO"
```

### Task 9: Stream Laguna routed decode through the fixed cache

**Files:**
- Modify: `ds4_gpu.h`
- Modify: `ds4_cuda.cu` compact-context lookup and acquire/release paths
- Modify: `ds4.c:47430-47798,60043-60091,61693-61724`
- Modify: `tests/test_cuda_laguna_stream.c`
- Modify: `tests/test_cuda_laguna_model.c:357-557`
- Modify: `Makefile`

- [ ] **Step 1: Add RED streamed-decode oracle cases**

Add explicit `--mode resident|streamed` and `--case short|continuation` selectors to the model test. In streamed mode require the canonical byte cache option, assert every selected routed tensor resolves from an engine-lifetime slot, and compare the short logits plus all eight teacher-forced continuation tokens against the same promoted Poolside evidence and unchanged ceilings used by resident mode. Add a compact test hook that fails if any routed pointer came from the static slab, full model mapping, managed memory, or a per-request allocation.

- [ ] **Step 2: Observe RED on the DGX**

```sh
make tests/test_cuda_laguna_model tests/test_cuda_laguna_stream
DS4_TEST_MODEL="$LAGUNA_MODEL" \
  ./tests/test_cuda_laguna_model --mode streamed --case short
```

Expected: compact startup succeeds, then the first routed lookup fails closed because decode has not acquired the selected experts.

- [ ] **Step 3: Add a typed compact execution result**

Define a small result enum shared across the graph/GPU boundary: success, request-cancelled, recoverable cache I/O/CUDA load failure, and unsafe invariant failure. Do not collapse these into a generic `false` or silently retry through resident lookup; later server work depends on the distinction.

- [ ] **Step 4: Acquire, publish, pin, execute, and release each decode layer**

After Laguna routing selects the ledger/model-declared expert set for a layer
(`n_expert_used == 10` for the pinned Laguna S 2.1 artifact):

1. validate the admitted set and increment each selected key's saturating hotness once;
2. acquire or load every selected `(layer_id, expert_id)` entry;
3. map its gate/up/down tensor ranges to the fixed slot projections;
4. pin the entries before issuing routed CUDA kernels;
5. record/synchronize the last consumer event before unpinning; and
6. run post-upload source-page disposal only after the upload completion point.

Use one monotonic use sequence for deterministic `last_used`. Hits reuse the published slot across token steps. A recoverable load error unwinds the complete layer set and returns the typed request error; an unsafe result latches the runtime violation.

- [ ] **Step 5: Prove strict resolution and cache reuse**

```sh
DS4_TEST_MODEL="$LAGUNA_MODEL" \
  ./tests/test_cuda_laguna_stream --case cache-io
DS4_TEST_MODEL="$LAGUNA_MODEL" \
  ./tests/test_cuda_laguna_model --mode streamed --case short
DS4_TEST_MODEL="$LAGUNA_MODEL" \
  ./tests/test_cuda_laguna_model --mode streamed --case continuation
DS4_TEST_MODEL="$LAGUNA_MODEL" \
  ./tests/test_cuda_laguna_model --mode resident --case all
```

Expected: streamed short and continuation pass the resident ceilings; the continuation produces cache hits and no fallback counter; resident mode remains unchanged.

- [ ] **Step 6: Commit**

```sh
git add ds4_gpu.h ds4_cuda.cu ds4.c \
  tests/test_cuda_laguna_stream.c tests/test_cuda_laguna_model.c Makefile
git commit -m "feat: stream Laguna routed decode through fixed cache"
```

### Task 10: Group over-capacity Laguna prefill

**Files:**
- Modify: `ds4_laguna_stream.h`, `ds4_laguna_stream.c`
- Modify: `ds4_gpu.h`, `ds4_cuda.cu`
- Modify: `ds4.c:47821-48250,59435-59590,60043-60091`
- Modify: `tests/test_laguna_stream.c`
- Modify: `tests/test_cuda_laguna_stream.c`
- Modify: `tests/test_cuda_laguna_model.c`

- [ ] **Step 1: Add RED over-capacity and determinism tests**

Construct a batch whose per-layer unique expert working set exceeds `slot_count` while each individual token's selected set fits. Assert stable first-occurrence grouping, no group larger than the available slots, no overflow allocation, no pinned-entry eviction, identical group traces across repeated runs, and cancellation at every group boundary. Add a numerical test comparing grouped prefill with resident serialized prefill at 512 and 2,048 tokens.

- [ ] **Step 2: Observe RED**

```sh
make tests/test_laguna_stream tests/test_cuda_laguna_stream tests/test_cuda_laguna_model
./tests/test_laguna_stream --case grouping
DS4_TEST_MODEL="$LAGUNA_MODEL" \
  ./tests/test_cuda_laguna_stream --case grouped-prefill
```

Expected: pure grouping passes only once wired from Task 7, while CUDA prefill fails because it attempts to make the complete selected set resident together.

- [ ] **Step 3: Process deterministic groups without changing accumulation order**

For each Laguna routed layer, derive groups from stable token-row/expert first occurrence. For one group at a time, acquire and pin its entries, evaluate only the matching token/expert pairs, accumulate into the same output rows in the original token-row then selected-expert order, synchronize, and release before slot reuse. Keep route weights and accumulation precision identical to resident execution. Never reorder reductions by cache hit state.

- [ ] **Step 4: Make cancellation and failure group-safe**

Check cancellation before starting a group and after its final CUDA completion, not while submitted work can still reference a slot. On any recoverable failure, drain submitted events, undo unpublished entries, release all group pins, and leave prior completed groups' temporary accumulation inaccessible to the caller. Unsafe cleanup failure latches the process violation.

- [ ] **Step 5: Run grouped regressions**

```sh
./tests/test_laguna_stream --case grouping --case cache-policy
DS4_TEST_MODEL="$LAGUNA_MODEL" \
  ./tests/test_cuda_laguna_stream --case grouped-prefill --case cache-faults
DS4_TEST_MODEL="$LAGUNA_MODEL" \
  ./tests/test_cuda_laguna_model --mode streamed --case short
DS4_TEST_MODEL="$LAGUNA_MODEL" \
  ./tests/test_cuda_laguna_model --mode streamed --case swa-513
```

Expected: every case passes, grouped and ungrouped runs accept the same logits, and allocation peaks do not change with working-set size.

- [ ] **Step 6: Commit**

```sh
git add ds4_laguna_stream.h ds4_laguna_stream.c ds4_gpu.h ds4_cuda.cu \
  ds4.c tests/test_laguna_stream.c tests/test_cuda_laguna_stream.c \
  tests/test_cuda_laguna_model.c
git commit -m "feat: group over-capacity Laguna prefill"
```

### Task 11: Allocate Laguna prefill scratch at the configured cap

**Files:**
- Modify: `ds4.c:35201-35280,47250-47430,48279-48329,58624-58656,59435-59590`
- Modify: `ds4_runtime.h`, `ds4_runtime.c`
- Modify: `tests/test_laguna_stream.c`
- Modify: `tests/test_cuda_laguna_stream.c`
- Modify: `tests/test_cuda_laguna_model.c`

- [ ] **Step 1: Add RED exact-row tests**

Require the plan, graph estimator, session, and runtime tracker to agree that `--ctx 32768 --prefill-chunk 4096` configures and allocates exactly 4,096 Laguna prefill rows. Cover chunks 1, 4,096, context-sized legacy allocation, zero, greater-than-context, and multiplication overflow. Add model-backed 8,192-token and deep-32,768 terminal cases that must process multiple 4,096-row chunks without changing the durable 32K KV allocation.

- [ ] **Step 2: Observe RED**

```sh
make tests/test_laguna_stream tests/test_cuda_laguna_stream tests/test_cuda_laguna_model
./tests/test_laguna_stream --case prefill-plan
DS4_TEST_MODEL="$LAGUNA_MODEL" \
  ./tests/test_cuda_laguna_stream --case prefill-allocation
```

Expected: runtime allocation reports the current hard-coded 16,384 rows or the session reports context size instead of the configured chunk.

- [ ] **Step 3: Thread one validated `prefill_rows` value end to end**

Change Laguna graph allocation to accept the already-validated configured cap:

```c
static bool laguna_graph_alloc(
    ds4_laguna_gpu_graph *g,
    uint32_t ctx_size,
    uint32_t prefill_rows,
    ds4_runtime_tracker *tracker);
```

Derive `prefill_rows` once from the exact option, reject invalid values before session mutation, use it in the immutable allocation plan and graph estimator, allocate every row-shaped buffer from it, and set both `s->prefill_cap` and `g->prefill_cap` to it. Keep total context/KV at 32,768. Remove the `min(ctx, 16384)` and `s->prefill_cap = ctx` rewrites.

- [ ] **Step 4: Reconcile tracker bytes**

Charge graph/scratch allocations to their declared call sites and assert the byte-exact total equals the pre-allocation plan. The test must detect even a single buffer still sized from context or 16,384 rows.

- [ ] **Step 5: Reach checkpoint C**

```sh
./tests/test_laguna_stream --case prefill-plan --case allocation
DS4_TEST_MODEL="$LAGUNA_MODEL" \
  ./tests/test_cuda_laguna_stream --case prefill-allocation
DS4_TEST_MODEL="$LAGUNA_MODEL" \
  ./tests/test_cuda_laguna_model --mode streamed --case all
```

Expected: configured and allocated rows both equal 4,096; short, SWA-513, YaRN-8193, deep-32768, continuation, serialized/batched, and serialized/mixed checks all pass at unchanged ceilings.

- [ ] **Step 6: Commit**

```sh
git add ds4.c ds4_runtime.h ds4_runtime.c tests/test_laguna_stream.c \
  tests/test_cuda_laguna_stream.c tests/test_cuda_laguna_model.c
git commit -m "feat: allocate Laguna prefill scratch at the configured cap"
```

### Task 12: Make compact runtime page disposal exact and measured

**Files:**
- Modify: `ds4_laguna_stream.h`, `ds4_laguna_stream.c`
- Modify: `ds4_runtime.h`, `ds4_runtime.c`
- Modify: `ds4_cuda.cu:1458-1488,1656-1740`
- Modify: `tests/test_laguna_stream.c`
- Modify: `tests/test_cuda_laguna_stream.c`

- [ ] **Step 1: Add RED page-range and counter tests**

For synthetic page sizes and tensor ranges, require the advised interval to contain only full pages wholly inside a safe range. Cover unaligned starts/ends, one shared page, exact pages, adjacent/overlapping ranges, empty ranges, overflow, and union deduplication. Require separate attempted/successful/failed call and byte counters, `errno` failure buckets, touched eligible unique pages, and advised unique pages. A failed advice call counts as attempted but not successful. With an injected exact-residency sampler, prove that the pre-advice source charge is `max(exact_sample, prior_post_advice_residency + unique_pages_touched_since)`, saturates at model size, updates the simultaneous qualification peak before advice, and cannot be erased by a lower post-advice sample.

- [ ] **Step 2: Observe RED**

```sh
make tests/test_laguna_stream tests/test_cuda_laguna_stream
./tests/test_laguna_stream --case page-ranges
./tests/test_cuda_laguna_stream --case page-advice
```

Expected: current helpers round outward, ignore return codes, and advise before upload synchronization.

- [ ] **Step 3: Implement inward range and union policy**

Put overflow-safe full-page intersection and interval-union logic in `ds4_laguna_stream`. Register dynamically touched source intervals when a static copy or routed read begins, but make an interval eligible for disposal only after its final host-to-device event and consumer safety point. Keep a process-lifetime union for coverage and a since-last-sample union for conservative high-water accounting. Shared metadata/tensor boundary pages remain excluded.

- [ ] **Step 4: Replace fire-and-forget advice**

Make compact runtime disposal operate on the exact opened fd/mapping and capture `posix_fadvise`/`posix_madvise` results. At compact attachment and every quiescent post-advice point, sample the exact mapping with `mincore` and retain the last post-advice resident-byte count. Immediately before each advice call, compute the conservative value from that retained count plus the since-last-sample touched union, optionally take a complete exact sample, charge the larger value as `model_source_resident_bytes`, and update `qualification_total_peak` before any page can be discarded. Only after CUDA completion, that charge, and the advice attempt may DS4 take/store the next post-advice sample and clear the since-last-sample union. Do not mutate process-global cache state. Update the runtime snapshot after each attempt and set `page_advice_complete_monotonic_ns` only after final request advice and synchronization complete.

- [ ] **Step 5: Run advice/fault regressions**

```sh
./tests/test_laguna_stream --case page-ranges
./tests/test_cuda_laguna_stream --case page-advice --case cache-faults
DS4_TEST_MODEL="$LAGUNA_MODEL" \
  ./tests/test_cuda_laguna_stream --case model-page-advice
```

Expected: attempted coverage equals the touched eligible union, successful bytes are nonzero on the model case, failures are zero, the recorded pre-advice source high-water is no smaller than the conservative charge, and injected failures remain visible rather than being reported as eviction.

- [ ] **Step 6: Commit**

```sh
git add ds4_laguna_stream.h ds4_laguna_stream.c ds4_runtime.h \
  ds4_runtime.c ds4_cuda.cu tests/test_laguna_stream.c \
  tests/test_cuda_laguna_stream.c
git commit -m "feat: make compact runtime page disposal exact and measured"
```

### Task 13: Measure exact-inode residency and external footprint

**Files:**
- Modify: `ds4_runtime.h`, `ds4_runtime.c`
- Modify: `ds4_gpu.h`, `ds4_cuda.cu`
- Modify: `ds4.h`, `ds4.c`
- Create: `tests/test_runtime.c`
- Modify: `tests/test_cuda_laguna_stream.c`
- Modify: `gguf-tools/quality-testing/compact_runtime_qualify.py`
- Modify: `gguf-tools/quality-testing/test_compact_runtime_qualify.py`
- Modify: `schemas/compact-runtime-benchmark-v1.schema.json`
- Modify: `Makefile`

- [x] **Step 1: Add RED sparse-file and attribution-fixture tests**

Build sparse temporary GGUF-like files with page-aligned and shared-boundary tensor ranges. Test qualification-only cold preparation over every safe full page, exact `st_dev/st_ino/st_size/st_mtime_ns` binding, no symlink traversal, metadata/shared-boundary exclusion, duplicate-range unioning, `mincore` bit counting, advice failures, identity changes before/after measurement, and a derived unavoidable-residency value above 2 GiB. Add recorded `/proc/self/smaps` fixtures with model-inode VMAs, tracked host/pinned/managed ranges, overlapping tracked ranges, shared-library/stack/heap PSS, and malformed/overflow fields. Add synthetic process-scoped NVML inventories in which DS4 already owns CUDA context/library bytes before its first tracked model allocation, tracked allocations later grow, an unrelated process changes only between checkpoints, and NVML reports missing/unknown bytes. Mock only syscall/CUDA/NVML inputs in unit tests; keep all range/de-duplication arithmetic real.

- [x] **Step 2: Observe RED**

```sh
make tests/test_runtime tests/test_cuda_laguna_stream
./tests/test_runtime --case external-attribution
python3 gguf-tools/quality-testing/test_compact_runtime_qualify.py -v
```

Expected: missing cold-preparation, smaps/CUDA attribution, or external-checkpoint APIs.

- [x] **Step 3: Implement descriptor-bound cold preparation**

Verify the qualification-plan digest and its opened-model identity, then consume its normalized safe full-page range union; do not rediscover tensor roles in Python. Open the pinned model without following symlinks, `fstat` it, issue advice without `drop_caches`, synchronize an exact-inode `mincore` sample, and `fstat` again. Emit eligible/attempted/successful/failed call and byte counts plus errno buckets. Treat identity change, advice failure, incomplete coverage, ledger/plan mismatch, or an unavoidable bound above 2 GiB as invalid evidence.

- [x] **Step 4: Produce de-duplicated external-memory samples**

Before launching the child, the harness records a device-UUID-scoped NVML
process inventory without creating a CUDA context. Inside DS4, before any
model-lifetime allocation, record the device UUID, own PID, tracked-allocation
baseline, and NVML process bytes if a CUDA context already exists. At every
synchronized qualification checkpoint, query NVML's compute-process API for
the current DS4 PID, record `cudaMemGetInfo` only as a device-wide cross-check,
and parse `/proc/self/smaps`. Exclude exactly once:

1. model VMAs matching the opened descriptor's device/inode, whose resident pages are charged separately by `mincore`;
2. tracked host, pinned, registered, and managed ranges from the allocation attribution table; and
3. CUDA/managed bytes already physically charged by the internal tracker.

Define `cuda_library_unattributed = nvml_bytes_for_ds4_pid -
tracked_cuda_physical_current`, never from a post-context free-memory baseline.
This charges CUDA context/library bytes even when they exist before the first
tracked model allocation. Define `host_library_unattributed` as remaining
non-model PSS after address-range de-duplication. A negative gap, unknown or
missing NVML process usage, UUID/PID mismatch, overlap, missing tracked VMA,
parser error, duplicate attribution, or either value above 512 MiB is a
reconciliation failure. Update `qualification_total_current =
owned_total_current + cuda_library_unattributed +
host_library_unattributed + model_source_resident_bytes` and its simultaneous
peak at the checkpoint. Expose the raw process-scoped NVML, `cudaMemGetInfo`,
smaps identity/counters, and reconciled values through
`ds4_runtime_snapshot`.

- [x] **Step 5: Detect unrelated CUDA baseline changes**

Freeze one NVML API/version and the full pre-child per-process inventory in the
manifest. At every checkpoint, compare every non-DS4 PID's allocation with
that frozen pre-child inventory as well as with the immediately-before and
immediately-after inventory. A process appearing, disappearing, or changing
bytes anywhere from pre-child baseline through the checkpoint makes the
sample infrastructure-invalid, even if it is stable during the narrow
checkpoint window. The harness neither charges nor forgives those bytes.
Match the DS4 PID, GPU UUID, and build identity explicitly; use DS4's
process-scoped NVML value as the CUDA attribution source.

- [x] **Step 6: Make cold-preparation and live attribution tests green**

```sh
./tests/test_runtime --case external-attribution
DS4_TEST_MODEL="$LAGUNA_MODEL" \
  ./tests/test_cuda_laguna_stream --case external-attribution \
  --case model-page-advice
python3 gguf-tools/quality-testing/test_compact_runtime_qualify.py -v
make test-laguna-compact-python
```

Expected: sparse-file coverage, tamper/identity, boundary-page, 2-GiB cap, conservative-high-water, smaps de-duplication, CUDA reconciliation, unrelated-process invalidation, and both 512-MiB ceiling cases pass.

- [x] **Step 7: Commit**

```sh
git add ds4_runtime.h ds4_runtime.c ds4_gpu.h ds4_cuda.cu ds4.h ds4.c \
  tests/test_runtime.c tests/test_cuda_laguna_stream.c \
  gguf-tools/quality-testing/compact_runtime_qualify.py \
  gguf-tools/quality-testing/test_compact_runtime_qualify.py \
  schemas/compact-runtime-benchmark-v1.schema.json Makefile
git commit -m "test: measure exact-inode and external compact footprint"
```

### Task 14: Gate warm stability and two-session pressure

**Files:**
- Modify: `tests/test_cuda_laguna_stream.c`
- Modify: `tests/test_cuda_laguna_model.c`
- Modify: `gguf-tools/quality-testing/compact_runtime_qualify.py`
- Modify: `gguf-tools/quality-testing/test_compact_runtime_qualify.py`
- Modify: `Makefile`

- [x] **Step 1: Add RED warm-growth and pressure cases**

After one warm-up, run three create/prefill/decode/free cycles and require every current owned category to return within 64 MiB of the first post-warm result with no monotonically growing category. Run the same prompt cold then warm; require identical accepted output, increased cache hits, and routed model-file bytes no greater than cold. Because the canonical exact-cache profile deliberately admits one live graph session, exercise the additional pressure shape as a separate synthetic 4K/two-logical-actor cache profile with interleaved misses, forced eviction, one cancellation, and a batch working set larger than `slot_count`; do not present it as public two-session graph support.

- [x] **Step 2: Observe RED**

```sh
make tests/test_cuda_laguna_stream tests/test_cuda_laguna_model
DS4_TEST_MODEL="$LAGUNA_MODEL" \
  ./tests/test_cuda_laguna_stream --case session-pressure
DS4_TEST_MODEL="$LAGUNA_MODEL" \
  ./tests/test_cuda_laguna_model --mode streamed --case warm-stability
```

Expected: at least one lifetime counter, cache pin, or allocation teardown assertion is absent or fails under forced pressure.

- [x] **Step 3: Fix lifetime ownership at the narrowest seam**

Make engine-lifetime cache allocations survive session churn, while all session/request graph, KV, pins, temporary grouping state, cancellation state, and request telemetry return to their declared baseline. Keep engine lifetime and session lifetime categories distinct. Do not clear monotonic cache/I/O counters between sessions.

- [x] **Step 4: Add the focused CUDA target**

Define `make test-cuda-laguna-streaming` to run pure policy, compact CUDA startup/I/O/fault/advice/stability/pressure, and streamed oracle suites. Keep `test-cuda-laguna-resident` separate so the baseline remains independently executable.

- [x] **Step 5: Reach checkpoint D**

Checkpoint D's Laguna-scoped acceptance closed in a guarded DGX Spark
maintenance window spanning 2026-08-12 and 2026-08-13, under the explicit
amendment to the original generic-regression criterion described below. The
tested source was exact revision
`e554d0fb4fab8b891e4913b23aaa977c1eb3836e`, exported with `git archive` into
the fresh directory
`/tmp/ds4-laguna-task14-e554d0fb4fab8b891e4913b23aaa977c1eb3836e`;
no remote worktree patching was used. `DS4_LOCK_FILE` was absent. After the
target's pure-policy prerequisite and before the runner's verifier, cold
preparation, or CUDA children, the runner proved `/tmp/ds4.lock` available;
each model child used that production lock.

The retained model descriptor was bound to the 68,248,759,648-byte Laguna
artifact with SHA-256
`e163b2c98908809a71245d6bb68b2226994d9969cb2a438eccb72196a1c4147a`.
Cold preparation used the inode-bound plan at `/tmp/ds4-task14-plan.json`,
SHA-256 `21b0836316e92c8386fc76cbd4069ec6fe99ff03e7691e5ad07a4a7c11edd8a4`,
and tokenizer runtime commit
`15c9b92502fed6bc26842e98d11a6347caadb08e`. The runner cold-prepared the
retained descriptor's plan-eligible ranges after the verifier and before its
first CUDA child, and again after its final descriptor rehash and before the
following CUDA process. Exact-inode sampling proved those eligible pages cold
while permitting the plan-declared unavoidable coverage.

Before the focused gate, the separate live external-attribution target passed
27 assertions. The focused streaming gate then passed every pure-policy and
synthetic compact case, including the two-logical-actor pressure case, followed
by streamed `short`, `prefill-8192`, and `warm-stability`. The 8,192-token run
used two real 4,096-token graph calls with all 146 live allocation records and
every owned category byte-identical across chunks. Cold and first-warm sessions
each passed Poolside logit tolerances with matching Poolside/session argmax; the
first warm rerun added cache hits and did not increase routed model-file reads.
Three later churn cycles preserved argmax, returned graph/KV current ownership
to zero, and restored the 22-record engine baseline. `cuda-regression` also
passed the dedicated NVML warm-up-plus-four capture FD-stability case, the
original 693-assertion startup/global-FD unwind case, long-context smoke, and
the Laguna kernel suite. At the recorded checkpoints after each top-level gate,
the frozen non-DS4 GPU peer PID/name/byte inventory was unchanged. The
production service was restarted afterward, reacquired `/tmp/ds4.lock`, served
its model inventory and a nonempty chat response, and left the non-DS4 peer
inventory unchanged across restoration. The maintenance transcript is
`/tmp/task14-e554d0f-maintenance-attempt-1.log` on the DGX.

Original planned sequence (the focused gate and `cuda-regression` passed; the
literal final command did not):

```sh
DS4_TEST_MODEL="$LAGUNA_MODEL" \
DS4_QUALIFICATION_PLAN="$LAGUNA_QUALIFICATION_PLAN" \
DS4_QUALIFICATION_PLAN_SHA256="$LAGUNA_QUALIFICATION_PLAN_SHA256" \
LAGUNA_TOKENIZER_RUNTIME_COMMIT="$LAGUNA_TOKENIZER_RUNTIME_COMMIT" \
  make test-cuda-laguna-streaming
make cuda-regression
make test
```

Original planned acceptance: all three commands pass; pressure changes
I/O/timing/eviction counters only, every slot/pin/capacity invariant
reconciles, and no accepted output changes. Observed: the first two commands
passed. For checkpoint D, the unavailable model-backed generic target was
explicitly replaced by the non-model Make-recipe equivalent below; this
amendment does not claim that literal `make test` passed.

Observed qualification caveat: the literal final `make test` is not recorded
as passing. A fresh archive has no gitignored `ds4flash.gguf`, while its bare
`./ds4_test` recipe still requires a provisioned resident model. Substituting
the production DeepSeek model and forcing `DS4_TEST_SSD_STREAMING=1` exercised
a pre-existing generic DeepSeek quality/SSD-selected-cache path, outside the
Laguna compact gate, and failed; Task 14 did not change that path. The remaining
commands in the `make test` recipe were therefore run with all model,
SSD-streaming, and CUDA-tuning overrides absent and with only bare
`./ds4_test` replaced by `./ds4_test --server`; it exited zero, including the
evaluation/agent/server checks, allocation and placement tests, every Laguna
pure-policy case, 169 external-attribution assertions, 6,519 Laguna-plan
assertions, 138 CLI assertions, and sampling parity. This is recorded as the
non-model Make-recipe equivalent, not as a successful literal `make test`.
Provisioning a supported resident fixture for that generic model suite, and
separately fixing its CUDA SSD-quality path, remain follow-up work and do not
invalidate the focused Laguna and `cuda-regression` evidence above.

- [x] **Step 6: Commit**

```sh
git add tests/test_cuda_laguna_stream.c tests/test_cuda_laguna_model.c \
  gguf-tools/quality-testing/compact_runtime_qualify.py \
  gguf-tools/quality-testing/test_compact_runtime_qualify.py Makefile
git commit -m "test: gate Laguna warm stability and session pressure"
```

### Task 15: Freeze the compact-runtime wire schemas

**Files:**
- Create: `schemas/ds4-version-v1.schema.json`
- Create: `schemas/ds4-runtime-v1.schema.json`
- Create: `schemas/ds4-runtime-request-v1.schema.json`
- Create: `schemas/ds4-token-admission-v1.schema.json`
- Create: `schemas/ds4-laguna-compact-runtime-v1.schema.json`
- Create: `gguf-tools/quality-testing/compact_runtime_schema.py`
- Create: `gguf-tools/quality-testing/requirements-compact-runtime.txt`
- Create: `tests/test_runtime_contract.py`
- Modify: `Makefile`

- [x] **Step 1: Write RED schema-boundary tests**

Load all five schemas with Draft 2020-12 validation and assert their canonical `$id`/`schema` constants. Cover every required field, recursive `additionalProperties: false`, missing/null distinctions, sorted feature arrays, lowercase SHA-256, RFC 3339 timestamps, UUIDs, status enums, and stable rejection/error codes. For every uint64 decimal-string field, accept `0` and `18446744073709551615` and reject leading zeroes, signs, exponent/decimal notation, JSON numbers, and `18446744073709551616`. Token counts remain bounded JSON integers; rates remain finite JSON numbers.

- [x] **Step 2: Add canonical-JSON dependency and conformance tests**

Pin `jsonschema` and `rfc8785` versions in the qualification-only requirements file. Test the RFC 8785 implementation against the RFC's number/string/property-order vectors plus duplicate-key, non-finite-number, lone-surrogate, and unsigned UTF-8 path-order cases. DS4's C response serializers need valid JSON, but bundle canonicalization stays in the Python harness.

- [x] **Step 3: Observe RED**

```sh
python3 tests/test_runtime_contract.py -v
```

Expected: missing schemas and dependency instructions.

- [x] **Step 4: Define exact closed schemas**

Transcribe the approved wire contracts without optional catch-all objects:

- `ds4.version/v1` — exactly `schema`, `revision`, `dirty`, `backend`, `features`;
- `ds4.runtime/v1` — identity, config, limits, allocations, counters, and violations in one snapshot;
- `ds4.runtime.request/v1` — one request's tokens, timing, rates, deltas, terminal status, and nullable advice-completion timestamp;
- `ds4.token-admission/v1` — exact templated/requested/context counts, fit, and nullable rejection code; and
- `ds4.laguna.compact-runtime/v1` — subject/host/model/schema bindings, oracle, immutable manifest, global gates, all profiles, and evidence root.

Use reusable `$defs` only inside a schema file so each distributed schema validates independently.
Use the shared strict-parser/Draft-2020-12 profile for the two `x-ds4-*`
keywords; raw Draft validation alone does not enforce exact JSON number kinds
or lexically sorted feature arrays.

- [x] **Step 5: Make schema tests green and wire the target**

```sh
python3 tests/test_runtime_contract.py -v
for schema in schemas/ds4-*-v1.schema.json; do
  python3 -m json.tool "$schema" >/dev/null
done
make test-laguna-compact-contract
```

Expected: all valid boundaries pass, all unknown/overflow/noncanonical cases fail, and each schema is independently valid.

- [x] **Step 6: Commit**

```sh
git add schemas/ds4-version-v1.schema.json schemas/ds4-runtime-v1.schema.json \
  schemas/ds4-runtime-request-v1.schema.json \
  schemas/ds4-token-admission-v1.schema.json \
  schemas/ds4-laguna-compact-runtime-v1.schema.json \
  gguf-tools/quality-testing/compact_runtime_schema.py \
  gguf-tools/quality-testing/requirements-compact-runtime.txt \
  tests/test_runtime_contract.py Makefile
git commit -m "test: freeze compact runtime wire schemas"
```

### Task 16: Expose DS4 build identity and runtime snapshots

**Files:**
- Modify: `Makefile`
- Modify: `ds4_runtime.h`, `ds4_runtime.c`
- Modify: `ds4.h`, `ds4.c:2470-2525`
- Create: `ds4_build_info.h`, `ds4_build_info.c`
- Create: `ds4_qualification_control.c`
- Modify: `ds4_gpu.h`, `ds4_cuda.cu`
- Modify: `ds4_cli.c`, `ds4_server.c`, `ds4_agent.c`, `ds4_bench.c`, `ds4_eval.c`
- Modify: `ds4_help.c`
- Modify: `tests/test_runtime.c`
- Create: `tests/test_version_json.py`
- Create: `tests/test_runtime_endpoint_contract.py`
- Create: `tests/test_qualification_control.c`
- Create: `tests/test_qualification_control_contract.py`
- Create: `tests/test_qualification_control_cli_contract.py`
- Create: `tests/test_task16_gate_contract.py`
- Modify: `tests/test_cuda_laguna_stream.c`, `tests/test_cuda_build_contract.py`

- [x] **Step 1: Add RED build/snapshot tests**

Require every inference binary's `--version-json` to exit `0` before opening a model and validate as `ds4.version/v1`. Test clean/dirty revisions, exact 40-hex revision, backend `cpu|metal|cuda|rocm`, sorted unique compiled features, and no additional fields. In C tests, require one process-lifetime UUID and one process-global `snapshot_seq` that is strictly increasing across engine instances until it saturates at `UINT64_MAX`, internally consistent allocation totals, executable stat identity, retained opened-model stat identity, exact configured/effective values, and immutable historical violations. Add a Unix-socketpair test for a hidden qualification control fd: exactly one opened model descriptor plus its stat identity must arrive with `SCM_RIGHTS`; external-sample ready/ack/result/ack messages must use a strictly increasing checkpoint sequence and block model progress while the parent brackets its inventories; an invalid/non-socket fd, wrong sequence, timeout, or disconnect fails qualification safely.

- [x] **Step 2: Observe RED**

```sh
make tests/test_runtime ds4 ds4-server ds4-agent ds4-bench ds4-eval
./tests/test_runtime --case external-attribution
uv run --with-requirements \
  gguf-tools/quality-testing/requirements-compact-runtime.txt \
  python tests/test_runtime_contract.py -v
./ds4-server --version-json
```

Expected: missing build macros, runtime serializer, or option.

- [x] **Step 3: Stamp reproducible build facts**

Have the Makefile pass revision, dirty state, selected backend, and compiled feature set into one `ds4_build_info` implementation. Feature sorting happens at construction, not ad hoc per binary. `--version-json` must be handled immediately after argument parsing and before model-path validation or CUDA initialization. A qualification harness rejects `dirty=true`, non-CUDA backend, or missing `laguna`/`ssd_streaming`.

- [x] **Step 4: Retain executable and opened-model identity**

At process startup, open/stat the running image through `/proc/self/exe` on Linux and record device, inode, size, and nanosecond mtime. Preserve the model fd identity obtained at engine open rather than reconstructing it from the path. Add harness-only `--qualification-control-fd N` to the common inference options. When present, DS4 sends a duplicated opened model descriptor and exact stat identity to that inherited Unix socket with `SCM_RIGHTS` before model allocation, then retains its own descriptor normally. Passing the hidden fd transfers that inherited endpoint to the engine-open wrapper: it creates the live close-on-exec duplicate and consumes the original on every return path. After sending exactly one reference to the retained opened-model descriptor, DS4 waits for a sequence-zero `MODEL_FD_ACK` carrying the same identity. The parent hashes with pre/post `fstat`, may cold-prepare that exact descriptor through the preparation callback, rechecks it, and only then acknowledges, so child validation/allocation cannot begin early. The child grants this preparation a distinct 15-minute model-ack budget; READY/RESULT barrier acknowledgements retain the 30-second budget. Keep the socket open as a checkpoint barrier: after CUDA synchronization and before external sampling, DS4 sends `sample_ready(snapshot_seq)` and waits while the parent takes the frozen-baseline/before inventory; after sampling it sends `sample_result(snapshot_seq, identity)` and waits while the parent takes the after inventory, then resumes only on the matching acknowledgement. CUDA holds the compact execution lock from READY through RESULT_ACK, so checkpoint work and concurrent runtime snapshots remain blocked until the corresponding parent acknowledgement. The Python qualifier hashes the received model descriptor with pre/post `fstat`; ordinary runtime consumers never receive a path or fd number. Close-on-exec, timeout, disconnect, and teardown tests prove neither endpoint leaks or leaves CUDA work pinned. This private control channel is evidence plumbing, not a public wire schema.

- [x] **Step 5: Serialize one coherent runtime snapshot**

Add `ds4_engine_runtime_snapshot()` and a serializer that takes the tracker/cache/page counters under one snapshot boundary, increments `snapshot_seq` once, and emits every required `ds4.runtime/v1` section. For compact CUDA, one combined capture holds both the compact execution lock and compact-state lock across counter copying and tracker wire capture; publication then allocates the process-global sequence exactly once. Thus no endpoint can combine counters and ownership from different execution epochs. Byte/duration/counter values use checked canonical decimal strings. Add `GET /v1/runtime` and make `/v1/models` return the same canonical model ID/family and opened-file identity used by the runtime snapshot. A healthy process has an empty violations array; a latched bound violation remains visible until exit.

- [x] **Step 6: Make identity and live endpoint tests green**

```sh
make test-laguna-runtime-identity
make test-cuda-build-contract
env -u DS4_LOCK_FILE \
  DS4_TEST_MODEL="$LAGUNA_MODEL" \
  make test-cuda-laguna-qualification-control
# Exact plan-bound cold preparation must run before and after the managed child.
env -u DS4_LOCK_FILE \
  DS4_TEST_MODEL="$LAGUNA_MODEL" \
  DS4_RUNTIME_SERVER_START_TIMEOUT=900 \
  uv run --with-requirements \
    gguf-tools/quality-testing/requirements-compact-runtime.txt \
    python tests/test_runtime_endpoint_contract.py -v --live ./ds4-server
```

Expected: all JSON validates; model/executable stats match the opened descriptors; repeated runtime reads increase `snapshot_seq` without resetting counters.

- [x] **Step 7: Commit**

Task 16 landed as a focused RED/green/hardening series from `b99533d` through
`6ef4123`; it was not collapsed into a mega-commit. The exact accepted revision
and its build, live CUDA, endpoint, restoration, and transcript identities are
recorded below.

Task 16 closed on DGX Spark on 2026-08-13 against exact clean code revision
`6ef4123b2c282ce53de790f0a0adf1e32e9010be`. It was exported with
`git archive` (SHA-256
`70da13a7f0fbdaf47f3dea3eee9cc91aca98697fce8b22432b875f099ad89c97`)
into the fresh directory
`/tmp/ds4-laguna-task16-6ef4123b2c282ce53de790f0a0adf1e32e9010be`;
no remote source patching was used. The acceptance transcript is
`/tmp/task16-6ef4123-maintenance-attempt-1.log`, SHA-256
`46f4c059d2326e1d9c1981b113b9f3262965778955381f3cafd7a470301ea716`.
The build reported `ds4.version/v1` with that revision, `dirty=false`,
`backend=cuda`, and sorted features `laguna,ssd_streaming`.

The guarded maintenance flow left `DS4_LOCK_FILE` absent, stopped only the
production DS4 service, proved the canonical `/tmp/ds4.lock` available before
the live stages, and kept the three unrelated GPU peer PID/name/byte tuples
unchanged. It bound the opened model descriptor to device `66306`, inode
`16794939`, mtime-ns `1785523774395107433`, size `68,248,759,648`, and SHA-256
`e163b2c98908809a71245d6bb68b2226994d9969cb2a438eccb72196a1c4147a`.
Four descriptor-bound cold preparations ran before external attribution,
before the two-process qualification-control target, before the managed
endpoint, and once more after that endpoint. They used plan
`/tmp/ds4-task14-plan.json`, SHA-256
`21b0836316e92c8386fc76cbd4069ec6fe99ff03e7691e5ad07a4a7c11edd8a4`.
Each attempted and completed `68,242,178,048` advice bytes with zero failed
calls, then measured `6,361,088` resident bytes, below the plan-declared
`6,582,272` unavoidable bytes. The pre-CUDA capacity check measured
`49,946,324,992` bytes available against a `25,769,803,776` qualification
bound plus an `8,589,934,592` reserve.

The host gate passed 210 runtime-attribution assertions, the qualification
transport C suite, 7 integration contracts, 4 hidden-CLI contracts, 78 parent
qualifier tests, 7 build-identity tests, 20 normative schema tests, 8 endpoint
contracts with the managed live case intentionally skipped, and 61 CUDA
source/build contracts. The hardware phase then passed external attribution (27
assertions), two same-engine READY/RESULT transactions and teardown (47), the
intentional control-disconnect/UNSAFE path (27, including the expected broken
pipe), and the managed live runtime endpoint (9). The live endpoint bound the
reported executable and opened model identities, kept build/model facts stable,
and advanced `snapshot_seq` across repeated reads.

The hardware RED-to-GREEN trail is retained as part of the evidence. Revision
`34c6c82` stopped before outage on a compile error: the compact counter helper
was used before declaration. Revision `645e0eb` was an invalid infrastructure
attempt, not a CUDA or product result: a temporary-git build-identity probe
inherited the outer exact build stamps through nested Make and failed before
outage. Revision `895a612` supplied the required live hardware RED: external
attribution passed 27 assertions, then qualification-control success failed 6
of 47 teardown assertions because attributed host/CUDA external-report currents
remained latched, baseline reconciliation failed, and fail-closed teardown
retained owners. Revision `3fac2a0` fixed that production accounting defect
without weakening the baseline and made success pass all 47 assertions. One
`3fac2a0` run was separately invalidated fail-closed by live embedding traffic
changing an unrelated peer from 620 to 626 MiB; exact peer-byte equality was
retained. Its stable fresh-process retry reached the expected disconnect EPIPE
but failed 1 of 26 assertions because the test compared preserved historical
tracker provenance with a fresh zero observation generation. Production had
correctly latched unsafe/external attribution without releasing or fabricating
ownership. The test-only correction at `6ef4123` compares the full tracker and
active-record state while normalizing only the expected violation transition;
the clean rerun passed all 27 disconnect assertions. These invalid and RED
attempts are not counted as acceptance.

After the final cold preparation, production restarted as PID `164601`,
reacquired the canonical lock, served the expected Flash/Pro model inventory,
and completed an independent chat request. The unrelated peer inventory was
byte-identical to its pre-window baseline; no test child remained and the
kernel recorded no OOM, Xid, NVRM, GPU fault, or segfault. The real-CUDA child
was driven by the native control harness, while the Python parent independently
tests descriptor hashing, exact preparation-before-model-ACK, deadlines, and
failure cleanup against protocol fixtures. Task 20 retains ownership of the
end-to-end Python qualification launcher/publication run.

### Task 17: Expose request metrics and exact token admission

**Files (implemented):**
- Modify: `Makefile`, `ds4.h`, `ds4.c`, `ds4_gpu.h`, `ds4_cuda.cu`
- Modify: `ds4_kvstore.h`, `ds4_kvstore.c`, `ds4_laguna_plan.c`,
  `ds4_laguna_stream.c`, `ds4_runtime.h`, `ds4_runtime.c`, `ds4_server.c`
- Modify: `tests/ds4_test.c`, `tests/test_cuda_build_contract.py`,
  `tests/test_cuda_laguna_stream.c`, `tests/test_gpu_args_cli.sh`,
  `tests/test_laguna_plan.c`, `tests/test_laguna_stream.c`,
  `tests/test_runtime.c`
- Create: `tests/test_laguna_server_contract.py`,
  `tests/test_laguna_server_live_contract.py`,
  `tests/test_session_request_attribution_api.c`,
  `tests/test_task17_output_ceiling_contract.py`

- [x] **Step 1: Add RED request-metrics tests**

Start a server child and require a server-generated request ID at acceptance for OpenAI Chat, Responses, and Anthropic requests. Non-streaming responses must contain one `ds4.runtime.request/v1` object; the final streaming usage event must contain the same fields before the protocol terminator. Assert exact prompt/generated token counts, request-scoped cache/I/O deltas, TTFT from acceptance to first emitted token, prefill and visible-decode rates, wall time, terminal status, and nullable/final page-advice completion. Test counter saturation and a request ending before first token.

- [x] **Step 2: Add RED side-effect-free admission tests**

POST the same logical model/messages/tools/tool-choice request to `/v1/token-admission` and inference. Cover exact fit, one-token overflow, zero/negative/non-integer output, malformed tools, unsupported `tool_choice=required`, mismatched model family, unknown field, native-template revision, and hidden-reasoning/tool/stop tokens sharing one output ceiling. Snapshot session count/KV/cache state before and after admission and require no mutation.

- [x] **Step 3: Observe RED**

```sh
make tests/test_runtime ds4-server
DS4_TEST_MODEL="$LAGUNA_MODEL" \
  python3 tests/test_laguna_server_contract.py \
    --server ./ds4-server --case metrics --case admission
```

Expected: request IDs are allocated too late, metrics are process aggregates or absent, and no admission route exists.

- [x] **Step 4: Factor one parse/render/admit path**

Allocate `request_id` immediately after HTTP acceptance. Refactor the existing protocol parsers and Laguna native-template renderer into a pure prepare function that returns canonical model identity, rendered tokens, requested output, and a stable rejection code without creating or mutating a session. Use it for `POST /v1/token-admission` and call the exact same context-fit predicate again immediately before inference session mutation. Never truncate or silently reduce the requested output.

- [x] **Step 5: Thread request-scoped accounting through execution**

Create a `ds4_runtime_request_context` when the request ID is accepted and pass
its pointer explicitly through session prefill/decode, routing/cache acquire,
model-file reads, H2D uploads, grouped execution, and page-advice calls. Under
the same synchronization that updates each process-lifetime counter, update
the initiating request's saturating counter. The request that owns a cache load
owns its read/H2D bytes and time; a concurrent waiter records its own
hit/wait/status but does not inherit the loader's bytes. Page-advice work keeps
the request identity attached to its touched-range set through final advice.
Never infer request deltas by subtracting process-global snapshots.

Record acceptance, prefill completion, first emitted token, final
visible/generated counts, final advice completion, and terminal status on that
same context. Add a two-slot interleaving test with disjoint reads plus a shared
in-flight cache load and prove each response receives only its own metrics
while process counters reconcile to the physical operations.

- [x] **Step 6: Emit metrics in all three protocols**

Add the request ID and metrics object to each non-streaming response and to the final usage event for Chat Completions, Responses, and Anthropic streaming. Preserve each protocol's native terminator and usage fields. If no page advice applied, emit JSON `null`; otherwise the timestamp must be after final synchronization.

- [x] **Step 7: Make server-contract tests green**

```sh
./tests/test_runtime --case request-metrics
python3 tests/test_laguna_server_contract.py \
  --server ./ds4_test --case metrics --case admission
DS4_TEST_MODEL="$LAGUNA_MODEL" \
  python3 tests/test_laguna_server_live_contract.py \
    --live ./ds4-server
DS4_TEST_MODEL="$LAGUNA_MODEL" \
  make test-cuda-laguna-request-counters
```

Expected: schema validation passes for every shape; overflow and malformed requests return stable 4xx results before session mutation; exact-fit inference remains accepted.

- [x] **Step 8: Commit**

Task 17 landed as 35 focused RED/green/hardening commits from `2390e82`
through `97ee3b6` (18 `test:`, 13 `feat:`, and 4 `fix:` commits), excluding
the interleaved Task 16 documentation commit `7aafdc8`; it was not collapsed
into a mega-commit. The exact accepted revision and its build, host-contract,
live-CUDA/server, restoration, and transcript identities are recorded below.

Task 17 closed on DGX Spark on 2026-08-14 against exact clean code revision
`97ee3b60314cfb9d13b71afddc280c393710ece0` (tree
`ffa8fe70d2e063ca139ff428077c9cae5fe85905`). It was exported with
`git archive` (SHA-256
`bb92057e197475438b195bbca793db1ec9e022b3fadd69a822a355672aaee730`)
into the fresh directory
`/tmp/ds4-laguna-task17-97ee3b60314cfb9d13b71afddc280c393710ece0`;
no remote source patching was used. The exact runner had SHA-256
`f491a322a2217b9f655af9a2f492dfe7f8d4976490d405093e62a6f17bd00eec`.
The acceptance transcript is
`/tmp/task17-97ee3b6-maintenance-188009.log`, SHA-256
`239f13475b927bbb34f007d0e8cc20684497009610f0f150a74f462b5003ff4b`
(639 lines, 160,367 bytes). Every candidate binary reported
`ds4.version/v1` with the accepted revision, `dirty=false`, `backend=cuda`,
and sorted features `laguna,ssd_streaming`.

The implementation allocates a request UUID at HTTP acceptance, shares one
side-effect-free prepare/admit result between `POST /v1/token-admission` and
the final pre-mutation inference check, and never truncates an accepted output
limit. Request contexts carry saturating cache, model-I/O, H2D, timing, token,
terminal-status, and observed/final page-advice facts through explicit
attributed operations. Cache-load owners receive the physical read/H2D work;
same-key later rows receive logical hits without inheriting physical bytes.
Chat Completions, Responses, and Anthropic responses publish exactly one
`ds4.runtime.request/v1` object in their native non-streaming response or final
streaming usage event while preserving protocol IDs, native usage, and
terminator order.

The fresh accepted plan is
`/tmp/ds4-task17-97ee3b6-two-session-plan.json`, SHA-256
`41c7d1d03e36e3cb2f38250451bd2522b1270d62409fe52b077a41317ef99bbb`.
Its 65-byte sidecar, retained descriptor identity, and every bound were
rechecked before execution. Profile `cache-8gib-sessions-2` fixes CUDA,
32,768 context tokens, 4,096 prefill rows, two sessions, an 8,589,934,592-byte
effective cache, 3,372,220,416 KV bytes, 3,074,105,360 graph bytes, and a
30,064,771,072-byte qualification bound. The guarded capacity check observed
50,102,583,296 bytes available against that bound plus an 8,589,934,592-byte
reserve. Three descriptor-bound cold preparations each attempted and completed
68,242,178,048 bytes across 671 calls with zero failed calls; measured
residency never exceeded 6,500,352 bytes, below the declared 6,582,272
unavoidable bytes.

Before outage, the archive passed 63 request-metrics assertions, 210 external
attribution assertions, 78 parent-qualifier tests, 7 build-identity tests, 20
normative schema tests, 9 endpoint tests with 1 intentional live skip, 27 pure
admission/metrics server tests, and 12 live-launcher tests with 3 intentional
model-backed skips. It also passed 64 CUDA source/build contracts, 127 Laguna
option assertions, 290 allocation assertions, 6,522 plan checks, 5 output
ceiling tests, and the server unit suite. A 60-second quiet window then proved
zero running/waiting peer requests, stable prompt/generated/success counters,
no established port-8003 client, and byte-identical unrelated peer
PID/name/allocation tuples before stopping production.

The hardware phase proved the canonical lock free, cold-prepared the retained
model descriptor, and passed `request-counters` (27 assertions). That fresh
process exercised same-key A/B ownership, disjoint D/E ownership, exact sums of
the eleven process counters, final page-advice barriers, and reused-session C
isolation. The live-server harness then passed all 12 tests in 751.639 seconds.
Its three model-backed cases proved exact executable/model/two-session runtime
identity, pure exact-fit and one-token-overflow admission, and Chat
Completions, Responses, and Anthropic streaming/non-streaming terminal
metrics. Native usage matched the request snapshot, request UUIDs remained
distinct from protocol IDs, snapshot sequences advanced, real visible output
had non-null TTFT, and each protocol placed its sole metrics object at its
native terminal boundary. A final runtime read remained ready with no
violations.

After the last cold preparation, production restarted as PID `193894` with
`Result=success`, `NRestarts=0`, and `ExecMainCode/Status=0/0`. It reacquired
the same canonical lock (device `66306`, inode `524639`), restored the exact
executable, working directory, and NUL-delimited argv, served exactly the Flash
and Pro model inventory, and completed an independent non-empty chat request.
The three unrelated vLLM peers remained exactly PID `134571`/68,817 MiB, PID
`134597`/626 MiB, and PID `134604`/1,683 MiB. No candidate process or
established port-8003 client remained, and the maintenance-window kernel log
contained no OOM, killed-process, Xid, NVRM, GPU-fault, or segfault record.
Only then did the transcript emit
`TASK17_ACCEPTANCE_GREEN revision=97ee3b60314cfb9d13b71afddc280c393710ece0`.

Three non-acceptance runs are retained as qualification evidence. Attempt 1
(`/tmp/task17-1ba20c4-maintenance-178664.log`, SHA-256
`d1a5d67eb76e0ea076eaa6e7db8d867b9078867dc5a013129a243d11d2dbe0b1`)
stopped before outage because a clean archive had not built the host
`tests/test_laguna_stream` binary; an untracked local binary had masked that
prerequisite during runner review. Attempt 2
(`/tmp/task17-1ba20c4-maintenance-181192.log`, SHA-256
`85b8ca538a1fd1a8b83fe2cf82e1b7807f80be238eeeb15adcf42c073c834a31`)
generated the correct plan but a runner regex searched for a literal
backslash-n in its valid checksum sidecar, so no live test ran; restoration was
operationally healthy but its evidence gate also rejected stochastic chat text
equality. Attempt 3
(`/tmp/task17-1ba20c4-maintenance-184393.log`, SHA-256
`e2074dcdf3f9bad824d95e2732d3a4a563fd0d84e3ae3cd3915509b1a74d4d55`)
reached the real CUDA gate and supplied the final RED: its test fixture omitted
the build identity that enables runtime snapshots. The missing baseline caused
9 of 27 failures, including every A/B check and the D/E and C process-delta
reconciliations; D/E and C request-local physical-ownership checks still
passed. Production restored cleanly and no live HTTP test ran. Commit
`97ee3b6` added the missing fixture identity plus a host source
contract pinning identity declaration, option wiring, engine open, and baseline
snapshot order. These infrastructure/test-harness failures are not counted as
product acceptance. The reusable lessons are now executable: clean-archive
prerequisites are explicit, evidence parsers bind actual byte formats, health
witnesses avoid stochastic content, and every snapshot-driven hardware test
wires the identity required to make the snapshot observable.

### Task 18: Lock compact server protocol and lifecycle semantics

**Files (implemented):**
- Modify: `Makefile`, `ds4.c`, `ds4_cuda.cu`, `ds4_gpu.h`,
  `ds4_runtime.c`, `ds4_server.c`
- Modify: `tests/ds4_test.c`, `tests/test_laguna_server_contract.py`,
  `tests/test_runtime.c`
- Create: `tests/test_task18_cli_contract.py`,
  `tests/test_task18_cuda_failure_contract.py`,
  `tests/test_task18_cuda_failure_source_contract.py`,
  `tests/test_task18_failure_contract.py`,
  `tests/test_task18_lifecycle_contract.py`

- [x] **Step 1: Add RED protocol matrix tests**

For Chat Completions, Responses, and Anthropic messages, cover streaming/non-streaming equivalence, `tool_choice=auto`, `tool_choice=none`, stable unsupported `required`, chunked/multiple tool calls, malformed-call rejection, continuation after real tool results, and a request naming the wrong model family. Compare visible text, reasoning separation, tagged tool calls, finish status, token counts, request ID, and final runtime metrics between streaming and non-streaming forms.

- [x] **Step 2: Add RED HTTP/fault classification tests**

Require invalid input, overflow, protocol errors, and unsupported values to return stable structured 4xx bodies before session mutation while the process remains healthy. Inject recoverable compact read/upload failures and require 503 only after every slot/pin is restored. Inject invariant violations and require a structured 500 if headers are unsent, otherwise an abruptly terminated stream, followed by process exit `1`. None may fall back to resident execution. The model-free boundary fixtures are paired with a dedicated four-process real-CUDA/HTTP fault target whose evidence is derived from live compact snapshots before the response action.

- [x] **Step 3: Add RED foreground/signal tests**

Launch the server as a child process and verify:

- normal idle or safely completed first-`TERM` drain exits `0`;
- invalid invocation/configuration exits `2` before model allocation;
- startup/model/CUDA/internal unsafe failure exits `1`;
- `SIGINT` retains `130`;
- a second/forced `SIGTERM`, or a first `SIGTERM` before a safe response can complete, retains `143`; and
- after first `TERM`, new requests are rejected while admitted CUDA work drains to a safe point and releases pins.

Also assert DS4 remains foreground and never forks a daemon, restarts itself, signals peers, changes global page-cache state, or chooses an alternate deployment port.

- [x] **Step 4: Observe RED**

```sh
make ds4_test ds4-server
python3 tests/test_laguna_server_contract.py \
  --server ./ds4_test --case protocol -v
python3 tests/test_task18_failure_contract.py --server ./ds4_test -v
python3 tests/test_task18_lifecycle_contract.py --server ./ds4_test -v
python3 tests/test_task18_cli_contract.py \
  --live-server ./ds4-server --test-server ./ds4_test -v
python3 tests/test_task18_cuda_failure_source_contract.py -v
```

Expected: the protocol, typed failure, lifecycle, CLI, and physical-evidence seams fail independently before their corresponding implementation commits. The real-CUDA failure target remains outside ordinary `make test` and requires the pinned retained model descriptor.

- [x] **Step 5: Implement explicit server and process state transitions**

Separate accepting, draining, signal-unsafe-draining, unsafe-draining, and forced-exit states. First `TERM` closes admission and requests cooperative cancellation/safe completion; it does not free cache state referenced by CUDA. Preserve the originating signal when a safe response/drain cannot complete or a second signal forces exit. Map compact execution results to restored 503 versus unsafe 500/abrupt termination exactly once at the HTTP boundary. Final request page advice remains sealable after a recoverable prefill failure, while event-completion uncertainty retains one unpublished `LOADING` owner with zero execution-visible references and poisons the cache.

- [x] **Step 6: Make the Task 18 matrix green and prove its stable-handoff slice**

```sh
make test-laguna-runtime-identity
make test-laguna-server-contract
make test-laguna-plan test-runtime-request test-cuda-build-contract
./ds4_test --server
./tests/test_laguna_stream --case options
./tests/test_laguna_stream --case allocation
python3 tests/test_task17_output_ceiling_contract.py -v
```

On the guarded DGX maintenance run, additionally execute the three low-level cache-fault cases, the two-session request-counter target, the four-process physical CUDA/HTTP failure target, and the standard 12-case live server contract against the retained pinned model descriptor.

Expected: protocol, admission, runtime, lifecycle, CLI, source/build, physical CUDA fault, and live HTTP contracts pass without service-control behavior. This closes Task 18's protocol/lifecycle portion of stable handoff; Task 19 still owns the benchmark/eval portion of checkpoint E, so checkpoint E remains open.

- [x] **Step 7: Commit**

Task 18 landed as 31 focused RED/green/hardening commits from `8db582f`
through `8a1d186` (24 `test:`, 2 `feat:`, and 5 `fix:` commits). It was not
collapsed into a mega-commit. The exact accepted revision and its build,
host-contract, physical-CUDA/server, restoration, and transcript identities
are recorded below.

Task 18 closed on DGX Spark on 2026-08-14 against exact clean code revision
`8a1d1862b9e3bed56ca2ac1d291f224c6faeab88` (tree
`063262d4b2e91a8c75e81ced4d753d21c38dab4f`). It was exported with
`git archive` (SHA-256
`d5bafaa440190d70a9dea51b0ad96fbc630f5781db35ff01bc09618376128453`)
into the fresh directory
`/tmp/ds4-laguna-task18-8a1d1862b9e3bed56ca2ac1d291f224c6faeab88`;
no remote source patching was used. The exact runner had SHA-256
`e9ac2b12400379bf8a24904b53ae38c6bfd00a92021c83bc812ab28e927b1131`.
The acceptance transcript is
`/tmp/task18-8a1d186-maintenance-250011.log`, SHA-256
`2b7c2e22c0dcc36705d56073a12ae5f48fe0092e90043e2bcc03fb01eb48dd25`
(768 lines, 204,373 bytes). Every candidate binary reported
`ds4.version/v1` with the accepted revision, `dirty=false`, `backend=cuda`,
and sorted features `laguna,ssd_streaming`.

The implementation gives the foreground server explicit accepting, draining,
signal-unsafe-draining, unsafe-draining, and forced-exit transitions. Normal
first-`TERM` drains preserve exit `0`; unsafe or incomplete signal-driven
drains retain the originating signal status; second signals force their exact
status; internal unsafe failure exits `1`; invalid invocation remains `2`.
Chat Completions, Responses, and Anthropic share stable parsing and terminal
semantics across streaming and non-streaming forms. Recoverable compact
`pread`/CUDA-copy failures restore cache state before returning 503 and leave
the same process accepting. Unsafe event-completion failures return a
structured 500 before headers, while post-header request-barrier failures
terminate the stream without a false terminal record and then exit `1`.

The fresh accepted plan is
`/tmp/ds4-task18-8a1d186-two-session-plan.json`, SHA-256
`41c7d1d03e36e3cb2f38250451bd2522b1270d62409fe52b077a41317ef99bbb`.
Its sidecar, retained descriptor identity, and every bound were checked before
execution. Profile `cache-8gib-sessions-2` fixes CUDA, 32,768 context tokens,
4,096 prefill rows, two sessions, an 8,589,934,592-byte effective cache,
3,372,220,416 KV bytes, 3,074,105,360 graph bytes, and a
30,064,771,072-byte qualification bound. The guarded capacity check observed
51,161,837,568 bytes available against that bound plus an 8,589,934,592-byte
reserve. Four descriptor-bound cold preparations each attempted and completed
68,242,178,048 bytes across 671 calls with zero failed calls; their measured
residence was 6,483,968, 6,492,160, 6,492,160, and 6,492,160 bytes,
respectively, each below the declared 6,582,272 unavoidable bytes.

Before outage, the archive passed 36 admission/metrics/protocol tests, 3 typed
failure tests, 11 lifecycle tests, 5 CLI tests, and 9 launcher tests with 3
intentional model-backed skips. It also passed 65 CUDA source/build contracts,
6 main-session recovery contracts, 9 Task 18 physical-source contracts, the
runtime-identity gates, 6,522 plan checks, 290 allocation assertions, 127
option assertions, 5 output-ceiling tests, and the server unit suite. A
60-second quiet window then proved stable production and peer counters,
byte-identical unrelated peer inventories, the canonical production lock
still owned, and no established port-8003 client before production was
stopped.

During outage, the three low-level cache fault/unsafe cases passed, followed by
the 27-assertion two-session request-counter target. The physical server suite
passed all four cases in 604.855 seconds: recoverable `pread` and CUDA-copy
failures restored state before 503 and a same-PID follow-up inference;
event-completion uncertainty produced a pre-header 500, retained one
unpublished zero-ref `LOADING` owner, and exited `1`; and a post-visible-frame
request-barrier failure omitted every terminal protocol record before exiting
`1`. The standard live-server harness then passed all 12 cases in 780.696
seconds, covering exact model/runtime identity, admission, and Chat
Completions, Responses, and Anthropic streaming/non-streaming behavior.

After final cold preparation, production restarted as PID `255535` with
`Result=success`, `NRestarts=0`, and `ExecMainCode/Status=0/0`. It reacquired
the same canonical lock (device `66306`, inode `524639`), restored the exact
executable, working directory, and NUL-delimited argv, served exactly the
Flash and Pro model inventory, and completed an independent non-empty chat
health request. The three unrelated vLLM peers remained exactly
`212855/620 MiB`, `212900/1683 MiB`, and `213554/68625 MiB`. No candidate
process or established port-8003 client remained, and the maintenance-window
kernel log contained no OOM, killed-process, Xid, NVRM, GPU-fault, or segfault
record. Only then did the transcript emit:

```text
TASK18_ACCEPTANCE_GREEN revision=8a1d1862b9e3bed56ca2ac1d291f224c6faeab88 archive_sha256=d5bafaa440190d70a9dea51b0ad96fbc630f5781db35ff01bc09618376128453 plan_sha256=41c7d1d03e36e3cb2f38250451bd2522b1270d62409fe52b077a41317ef99bbb at=2026-08-14T20:08:50+02:00
```

Three non-acceptance runs are retained as qualification evidence. Attempt 1
(`/tmp/task18-02212b0-maintenance-239058.log`, SHA-256
`bd947147f2f06f490cef78f11a24d8bf8414ca83e31e6ddad52b0de2eaaf5d06`,
263 lines, 19,370 bytes) stopped before outage because a clean real-CUDA build
proved that `ds4_gpu_laguna_compact` lacked the
`request_barrier_unsafe_failures` storage required by its public snapshot.
Commits `8a6ba88` and `064df5f` pinned and added that ABI state. Production was
never stopped.

Attempt 2 (`/tmp/task18-064df5f-maintenance-240219.log`, SHA-256
`60ced0f6d7e5d7a2b61c9908e3742a6b47592a9d318fbf4bb0c28d79df406f71`,
576 lines, 61,818 bytes) stopped before outage when the missing-model CLI
fixture inherited the canonical production lock held by PID `219291`; it
therefore exited `2` at lock acquisition instead of exercising the intended
startup/model failure exit `1`. Commits `6416aab` and `dedea25` gave the Task
18 CLI harness a private lock and pinned the ambient-held-lock regression.
Production was never stopped.

Attempt 3 (`/tmp/task18-dedea25-maintenance-243158.log`, SHA-256
`f27d1994acef2188fe936491cd769da1db71a91d1484e41f0aaba07555b0504b`,
783 lines, 174,732 bytes) passed every host gate, the 60-second quiet window,
fresh-plan verification, all three low-level CUDA failure cases, and the
27-assertion request-counter target. Its fresh plan had SHA-256
`41c7d1d03e36e3cb2f38250451bd2522b1270d62409fe52b077a41317ef99bbb`;
capacity was 51,214,737,408 bytes against the 30,064,771,072-byte bound plus
reserve. The four-case physical suite ran in 342.854 seconds: the post-visible
request-barrier unsafe case passed, while recoverable `pread` and CUDA-copy
failures returned 500 instead of 503 and the event-completion oracle expected
a positive refcount from an unpublished `LOADING` owner.

The two recoverable failures had restored their cache slots, but their terminal
page-advice seal required `prefill_complete`; a failed prefill has only
`prefill_started`, so the barrier incorrectly reclassified both as unsafe.
Commits `c340540` and `8a1d186` defined and fixed that chronology without
inventing prefill completion or bypassing final advice. Commit `42e51bf`
corrected the independent event oracle: an event-completion-uncertain load
owner remains `LOADING` and poisoned but has zero references until publication,
so a positive refcount would falsely claim execution visibility. The trap
restored production cleanly as PID `248873`, preserved all three peer
PID/allocation tuples, and left no candidate or port-8003 client; no acceptance
marker was emitted. These non-acceptance runs are not counted as product
acceptance. Their reusable lessons are now executable in the clean CUDA ABI
gate, private-lock CLI fixture, failed-prefill advice-barrier unit test, and
physical zero-ref unsafe-owner oracle.

### Task 19: Report qualification-safe benchmark and eval evidence

**Files:**
- Modify: `ds4_bench.c:487-545,671-815`
- Modify: `ds4_eval.c:98-128,1045-1080,1514-1650,4040-4065`
- Modify: `gguf-tools/quality-testing/compact_runtime_qualify.py`
- Modify: `gguf-tools/quality-testing/test_compact_runtime_qualify.py`
- Create: `tests/test_bench_eval_contract.py`
- Modify: `Makefile`

- [ ] **Step 1: Add RED benchmark-field and milestone tests**

Require machine-readable samples to distinguish serialized session payload from actual live KV allocation. Replace or deprecate ambiguous `kvcache_bytes` with `session_payload_bytes` and add `kv_allocated_bytes` from the runtime tracker. Require request ID, runtime metrics, configured/allocated prefill rows, cache ceiling/current/peak, simultaneous qualification total, exact-inode residency, external-attribution sample, and resident/streamed mode in every qualification sample.

Add qualification-only `--qualification-sequence FILE`. It accepts one
already-validated manifest slice containing one prompt and exactly four
repetitions, keeps one engine process alive, and emits flushed JSONL lifecycle
records `request_accepted`, `first_token`, and `request_complete` for the cold
request followed by three warm requests. Records bind repetition index,
monotonic timestamp, and request ID. Reject any sequence count/order/input that
does not match the immutable manifest before model allocation.

- [ ] **Step 2: Add RED stable eval-selection tests**

Add a repeatable stable `--case-id` selector and machine-readable result mode. Require exactly these four IDs in the manifest order:

```text
recNu3MXkvWUzHZr9
001b51d76b4d422988f2c11f104a2c6c
aime2025-01
compsec-076
```

Each output record must bind case ID, answer, grade, terminal status, request/runtime identity, and evidence digest. Reject unknown/duplicate IDs and index-only selection.

- [ ] **Step 3: Observe RED**

```sh
python3 tests/test_bench_eval_contract.py -v
```

Expected: ambiguous KV header, absent runtime evidence, or no stable-ID selection.

- [ ] **Step 4: Implement benchmark/eval records without log parsing**

Read all allocation/request fields through the public runtime APIs added in Tasks 16–17. Flush milestone JSONL immediately so the parent qualifier can enforce TTFT and whole-request deadlines while the child is still running. Keep human output if useful, but make the qualification JSON/JSONL format closed and deterministic. The harness must reject missing, duplicated, out-of-order, non-finite, or schema-invalid records.

- [ ] **Step 5: Compare like-for-like resident and streamed evidence**

Have the harness execute identical prompt/sampling/template inputs in both modes and compare:

- promoted oracle vectors and eight continuation token IDs at the existing thresholds; and
- the four-case eval `(answer, grade, terminal_status)` vectors for exact equality.

The four cases need only terminate and match resident; correctness on all four is not a gate. Keep the complete 92-case run optional and nonblocking.

- [ ] **Step 6: Make evidence tests green**

```sh
python3 tests/test_bench_eval_contract.py -v
make ds4-bench ds4-eval
DS4_TEST_MODEL="$LAGUNA_MODEL" \
  python3 gguf-tools/quality-testing/compact_runtime_qualify.py \
    smoke-eval --model "$LAGUNA_MODEL" --eval-bin ./ds4-eval \
    --case-id recNu3MXkvWUzHZr9 \
    --case-id 001b51d76b4d422988f2c11f104a2c6c \
    --case-id aime2025-01 --case-id compsec-076
```

Expected: resident and streamed answer/grade vectors match exactly and every sample carries unambiguous live allocation evidence.

- [ ] **Step 7: Commit**

```sh
git add ds4_bench.c ds4_eval.c \
  gguf-tools/quality-testing/compact_runtime_qualify.py \
  gguf-tools/quality-testing/test_compact_runtime_qualify.py \
  tests/test_bench_eval_contract.py Makefile
git commit -m "feat: report qualification-safe benchmark and eval evidence"
```

### Task 20: Run and publish canonical Laguna qualification

**Implementation progress (2026-09-04):** `4248699` adds the bounded
`qualification_records.py` streaming parser and shared validator CLI shim.
`b1d35bd` adds the pure `qualification_evidence.py` metadata index builder;
`26abf29` wires its test target into `test-laguna-compact-python`. Each helper
passes 12 host tests from a pinned worktree and has independent code review.

The parser binds a streamed slice and validates its 12 records incrementally.
The index builder checks the exact **declared** reference/observation union,
path/SHA/uint64-string syntax and canonical bytes; it does not inspect files,
check symlinks, authenticate observed bytes or collect references from a bundle.
Neither helper launches children, enforces real deadlines, authenticates a
running model/executable, evaluates gates, retries or publishes a bundle.
The runner and publication steps below therefore remain unchecked. See
[port-observability notes](../../spikes/2026-09-04-laguna-port-observability.md)
for diagnostic reuse, verification commands and the next implementation seam.

**Files:**
- Modify: `gguf-tools/quality-testing/compact_runtime_qualify.py`
- Modify: `gguf-tools/quality-testing/test_compact_runtime_qualify.py`
- Create: `gguf-tools/quality-testing/qualification_records.py`
- Create: `gguf-tools/quality-testing/test_qualification_records.py`
- Modify: `tests/validate_bench_qualification_json.py`
- Create: `gguf-tools/quality-testing/qualification_evidence.py`
- Create: `gguf-tools/quality-testing/test_qualification_evidence.py`
- Modify: `schemas/ds4-laguna-compact-runtime-v1.schema.json`
- Modify: `Makefile`

- [ ] **Step 1: Add RED qualification-runner tests**

Use fake foreground child binaries and a fake monotonic clock to require this
exact orchestration: validate/hash the immutable manifest before results,
capture a resident baseline, then visit streamed cache profiles in 8/12/16-GiB
order and each profile's frozen counterbalanced prompt order. For every
`(mode, profile, prompt)` slice, cold-prepare first, launch a new process, and
consume exactly one cold plus three consecutive warm repetitions from that
same child. Assert acceptance-to-first-token and acceptance-to-completion
deadlines, process-group termination/reaping, partial-evidence preservation,
one and only one retry for evidenced infrastructure-invalid runs, no retry for
a valid failed gate, and no mutation/reordering of the manifest.

- [ ] **Step 2: Add RED identity, status, and evidence-union tests**

Cover clean build/revision digest binding, qualification binary digest/stat, `/proc/<pid>/exe` identity, opened model descriptor identity before/after hashing and after the final sample, canonical served-model ID agreement across `/v1/models` and `/v1/runtime`, every consumed schema ID/content digest, immutable profile manifests, stable gate IDs, and `passed|failed|invalid` propagation. Require all 8/12/16-GiB profiles to remain present; bundle status is passed only when all global gates pass and at least one profile passes.

- [ ] **Step 3: Add RED canonical publication/tamper tests**

Require the referenced evidence set to equal the exact union of paths in global/profile gates. Reject missing, extra, duplicate, size/digest-mismatched files; symlinks; absolute paths; empty/`.`/`..` components; non-normal POSIX paths; control characters; and invalid UTF-8. Sort distinct paths by unsigned UTF-8 bytes, build canonical `evidence-index.json` entries with exactly `path`, `size_bytes`, `sha256`, and hash its RFC 8785 bytes. Exclude the bundle, sidecar, and evidence index from the referenced union.

- [ ] **Step 4: Observe RED**

```sh
python3 gguf-tools/quality-testing/test_compact_runtime_qualify.py -v
```

Expected: runner/process-control and bundle build/verify/publish tests fail because only manifest construction exists.

- [ ] **Step 5: Implement the immutable qualification runner**

Add a `run` subcommand with explicit `--manifest`, `--model`, `--server-bin`,
`--bench-bin`, `--eval-bin`, and `--evidence-dir` arguments. It must:

1. validate and hash the manifest and all five schemas before creating the
   evidence directory;
2. verify each binary's clean CUDA version/build identity and hash/stat it;
3. obtain and verify the pre-allocation plan/ledger for the resident baseline
   and every streamed profile;
4. use a private inherited Unix socket to receive DS4's opened model fd with
   `SCM_RIGHTS`, hash that descriptor with pre/post `fstat`, and match runtime
   identity;
5. run the resident oracle/protocol/eval/footprint baseline;
6. for every frozen streamed slice, perform Task 13 cold preparation, launch
   `ds4-bench --qualification-sequence` as a new foreground process, consume
   its flushed milestone JSONL, enforce the 15-minute TTFT and 45-minute
   whole-request deadlines from each acceptance record, and collect exactly
   one cold plus three warm records;
7. take the synchronized exact-inode/external-attribution/NVML inventory
   checkpoints required by the manifest, charge DS4's process-scoped NVML
   bytes, and reject any non-DS4 inventory difference from the frozen
   pre-child baseline or the narrow before/after sample; and
8. preserve stdout, stderr, control records, runtime/request snapshots, advice
   samples, oracle/eval records, exit status, and timeout diagnostics as
   content-addressable raw evidence.

Use `start_new_session=True` and a bounded `TERM`/safe-drain then `KILL`
cleanup so every timed-out child and descendant is reaped. The runner writes
only per-attempt evidence/status records; it does not publish a bundle or alter
the manifest. It never uses `drop_caches` and never starts/stops unrelated
services.

- [ ] **Step 6: Implement gate and profile evaluation**

Encode every approved global/profile gate as a stable ID with status, measured value, threshold, unit, and nonempty content-addressed evidence references. A valid threshold miss is `failed` and is never relabelled infrastructure noise. Permit one evidenced infrastructure-invalid retry without changing the manifest; a second invalid run remains `invalid`. No aggregate score can offset a red required gate.

- [ ] **Step 7: Bind running artifacts through descriptors**

Hash the model fd received from DS4's qualification control socket with `fstat` before/after, hash the binary before launch, match the running `/proc/<pid>/exe` stat, and repeat model/executable checks after the final sample. Abort publication on any identity drift. The bundle subject/model sections contain the pinned repository/revision/file/size/SHA plus the runtime identities, never path-only claims.

- [ ] **Step 8: Implement canonical verify-then-publish**

Build and validate the bundle in memory, canonicalize with RFC 8785, write the evidence index, recompute its root, reread/verify every evidence file, and validate the final bundle schema. Write the bundle to a same-directory temporary file, `fsync` it, atomically rename, `fsync` the directory, then write an external `<bundle>.sha256` sidecar for the exact final bytes using the same durability sequence. The verifier must independently reproduce both hashes.

- [ ] **Step 9: Make runner and publication tests green**

```sh
python3 gguf-tools/quality-testing/test_compact_runtime_qualify.py -v
make test-laguna-compact-python test-laguna-compact-contract
```

Expected: fake-child ordering/timeout/retry tests pass, valid bundles reproduce byte-for-byte, and every tamper/path/identity/status fixture fails closed before publication.

- [ ] **Step 10: Commit**

```sh
git add gguf-tools/quality-testing/compact_runtime_qualify.py \
  gguf-tools/quality-testing/test_compact_runtime_qualify.py \
  schemas/ds4-laguna-compact-runtime-v1.schema.json Makefile
git commit -m "feat: run and publish canonical Laguna qualification"
```

### Task 21: Document and run the compact Laguna qualification

**Files:**
- Modify: `README.md`
- Modify: `CONTRIBUTING.md`
- Modify: `tests/test-vectors/README.md`
- Generate outside the worktree: benchmark manifest, qualification evidence directory, bundle, and sidecar

- [ ] **Step 1: Write the DGX reference-run guide**

Document the pinned Poolside artifact, CUDA-only build, clean-build requirement, 32K total context, exact 4K prefill, one session slot, canonical byte option, 8/12/16-GiB fixed profile order, four prompt lengths, cold/new-process plus exactly three same-process warm repetitions, 45-minute request timeout, 15-minute TTFT timeout, and the four stable eval IDs. State explicitly that `drop_caches`, legacy whole-map options, deprecated expert-count qualification, daemonization, port selection, peer eviction, and co-residency claims are outside DS4.

- [ ] **Step 2: Commit the pre-run guide**

```sh
git add README.md CONTRIBUTING.md tests/test-vectors/README.md
git commit -m "docs: add compact Laguna qualification runbook"
git status --short
```

Expected: the guide is committed and the worktree is clean before the
qualification revision is built.

- [ ] **Step 3: Build from the clean committed CUDA revision**

```sh
git status --short
make clean
make CUDA=1 ds4 ds4-server ds4-bench ds4-eval \
  tests/test_cuda_laguna_model tests/test_cuda_laguna_stream
./ds4-server --version-json
```

Expected: worktree is clean before build; version JSON reports `dirty=false`, `backend="cuda"`, and sorted features including `laguna` and `ssd_streaming`.

- [ ] **Step 4: Run every deterministic preflight**

```sh
export LAGUNA_TOKENIZER_RUNTIME_COMMIT="$(
  python3 -c 'import json; print(json.load(open("tests/test-vectors/laguna-resident/manifest.json", encoding="utf-8"))["provenance"]["tokenizer_runtime_commit"])'
)"
test "${#LAGUNA_TOKENIZER_RUNTIME_COMMIT}" -eq 40
make test
make cuda-regression
DS4_TEST_MODEL="$LAGUNA_MODEL" make test-cuda-laguna-resident
DS4_TEST_MODEL="$LAGUNA_MODEL" make test-cuda-laguna-streaming
make test-laguna-compact-python test-laguna-compact-contract
```

Expected: all pass before the long curve starts.

- [ ] **Step 5: Freeze and verify the manifest before results**

```sh
python3 gguf-tools/quality-testing/compact_runtime_qualify.py \
  manifest build --model "$LAGUNA_MODEL" \
  --output /absolute/path/to/laguna-qualification/compact-runtime-benchmark-v1.json
python3 gguf-tools/quality-testing/compact_runtime_qualify.py \
  manifest verify \
  --manifest /absolute/path/to/laguna-qualification/compact-runtime-benchmark-v1.json
test ! -e /absolute/path/to/compact-runtime-evidence
```

Expected: the four prompts/token counts/hashes, profile/prompt order, sampling,
timeouts, identities, and eval IDs are frozen and hashed before an evidence
directory exists.

- [ ] **Step 6: Run the resident baseline and immutable streamed curve**

```sh
python3 gguf-tools/quality-testing/compact_runtime_qualify.py \
  run \
  --manifest /absolute/path/to/laguna-qualification/compact-runtime-benchmark-v1.json \
  --model "$LAGUNA_MODEL" \
  --server-bin ./ds4-server \
  --bench-bin ./ds4-bench \
  --eval-bin ./ds4-eval \
  --evidence-dir /absolute/path/to/compact-runtime-evidence
```

Expected: the runner captures the like-for-like resident oracle,
protocol/eval/footprint baseline, then every streamed `(8,12,16 GiB × frozen
prompt order)` slice with cold preparation, one fresh child, and exactly one
cold plus three warm repetitions. It enforces both deadlines, records every
profile as passed/failed/invalid, and applies the one-invalid-retry rule
without tuning or reordering.

- [ ] **Step 7: Verify every gate and publish**

```sh
python3 gguf-tools/quality-testing/compact_runtime_qualify.py \
  verify \
  --manifest /absolute/path/to/laguna-qualification/compact-runtime-benchmark-v1.json \
  --evidence-dir /absolute/path/to/compact-runtime-evidence
python3 gguf-tools/quality-testing/compact_runtime_qualify.py \
  publish \
  --manifest /absolute/path/to/laguna-qualification/compact-runtime-benchmark-v1.json \
  --evidence-dir /absolute/path/to/compact-runtime-evidence \
  --output /absolute/path/to/ds4-laguna-compact-runtime-v1.json
python3 gguf-tools/quality-testing/compact_runtime_qualify.py \
  verify-bundle /absolute/path/to/ds4-laguna-compact-runtime-v1.json
```

Expected: all global gates pass, every profile is represented, at least one profile passes, passed profiles satisfy numerical/protocol/page/bound/performance gates, and the sidecar digest verifies the final canonical bytes.

- [ ] **Step 8: Run optional regression evidence separately**

If time permits, run the complete 92-case GPQA/SuperGPQA/AIME/COMPSEC set and attach it as explicitly optional evidence. A miss does not change compact-runtime qualification status.

- [ ] **Step 9: Update docs with only reproducible outcomes**

Record the qualification revision, manifest digest, bundle sidecar digest, passed profile IDs, measured footprint/reduction/decode values, and exact rerun command. Do not call a profile co-resident, select a deployment port, or recommend Flash/Laguna lifecycle changes; those decisions belong to the downstream Dotfiles design.

- [ ] **Step 10: Run final verification and commit**

```sh
make test
make cuda-regression
make test-laguna-compact-python test-laguna-compact-contract
git diff --check
git status --short
```

Expected: verification is green and only the intended outcome-documentation changes remain.

```sh
git add README.md CONTRIBUTING.md tests/test-vectors/README.md
git commit -m "docs: record compact Laguna qualification"
```

## Completion boundary

This plan is complete only when checkpoint F produces a schema-valid canonical bundle whose external sidecar verifies, all global gates pass, and at least one of the mandatory 8/12/16-GiB profiles is `passed`. The deliverable is a truthful DS4 compact-runtime qualification artifact and stable runtime interface. It does not deploy Laguna, choose a port, evict or retain Flash, manage ensemble co-residency, or modify Dotfiles; those actions begin only under the approved downstream Dotfiles plan after it consumes a passed bundle.
