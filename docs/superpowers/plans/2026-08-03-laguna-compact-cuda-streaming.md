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

**Post-success constructor unlock handoff, actual-core CPU controls:**
`ds4_session_create` now passes its caller output slot to checked cleanup after
successful construction followed by a reported tracker-unlock failure. It still
returns 2. Successful cleanup clears the slot after physical free; refused
cleanup leaves the original live session in that slot for cleanup only. Keep the
slot and borrowed engine alive until checked cleanup consumes it. Do not use a
failed output for evaluation or let later cleanup erase the constructor failure.
Other constructor failure paths gain no retained-owner guarantee from this fix.

`tests/test_session_constructor_handoff.c` includes the actual `ds4.c` translation
unit and private engine/session ABI with `DS4_NO_GPU` and `DS4_TEST_HOOKS`. The
existing no-allocation hook still performs one real session calloc. The fixture
uses real libc and pthread effects, not copied constructor/release algorithms.
Each of its three named cases is a fresh process with an alarm and zero core
limit. The injected tracker unlock physically succeeds before reporting one
synthetic error; the selected cleanup lock refuses without acquiring the mutex.
These are control-flow signals, not genuine pthread failure states.

Before the production edit, the unchanged checked-release baseline
`95b5f029f817a0fdbac759b319795f9307790df6` passed `success` and
`unlock-cleanup-consumed`, but `unlock-cleanup-retained` lost the live output.
The fixture recorded that failure before using its known-live witness for actual
checked test teardown. All control, retry, physical-free, mutex-destroy and
cleanliness checks passed; the missing caller handoff alone returned 1. After
the one-branch fix, all three cases return 0: the failed constructor keeps its
cleanup-only output and checked retry consumes it exactly once.

The fixture refuses every allocation after the first before libc, preserves
sticky failures and current/live state until physical free, and refuses mutex
destruction while an owner or reservation remains. Non-handoff sensor, setup,
cleanup, compile and link defects are not feature RED. The isolated target
`make test-session-constructor-handoff` now joins `test-laguna-resident-path`
and `test`; its actual-header prerequisites avoid stale private-ABI builds.
Working host validation passed the prior 13 targets plus this native target,
seven selected source methods, and the synthetic record compositions. Independent
review found no regression in the bounded change or fixture. These results are
not an immutable or CUDA compile seal for the later constructor revision.
The earlier checked-release baseline has its own completed host and full CUDA
compile-only seals; no produced CUDA product, GPU or model was executed.

Caller/engine custody remains open across the bounded 18-call audit. Callers
avoid ordinary use after nonzero create, but can discard the owner or use legacy
void cleanup. A retained container can coexist with an active count of zero after
release-unlock refusal; count is not cleanup reachability. Partial graph failure,
reservation acquisition/precontainer custody, fatal allocation errors, full
engine/source-context/bootstrap ownership, first-event resident observation and
late exit/publication propagation are separate unfinished work. Production graph
callers remain LEGACY and the resident qualification guard remains closed.
Cold plus three warm runs and the sixteen-slice verified publication remain due.

Reusable rule: **a failed constructor must not erase the only cleanup owner.
Prove the handoff with the actual ABI where possible, record a lost handoff before
fixture rescue, and keep cleanup success distinct from successful execution.**

**Checked session release, host-tested custody increment:**
`ds4_session_free_checked(ds4_session **owner)` returns 0 while retaining the
session handle and remaining cleanup state. It returns 1 and clears the caller
slot only after physically freeing the container. NULL owner storage is refused;
an empty slot is already consumed. The caller must keep the slot outside the
payload, retain the borrowed engine, and serialize same-session operations.
After any attempt the handle is cleanup-only. A later successful retry does not
erase the first failure for eventual exit/publication.

The existing C release body now uses one checked entry and a thin legacy void
projection. Completed distributed/raw-pointer and graph-ready state is cleared
before a later refusal. Existing backend helpers already zero their containers;
they are not duplicated or refactored. Laguna lock/free/unlock refusals retain the
session, including successful graph cleanup followed by failed unlock. Exact-cache
reservation reconciliation remains last: missing authority/lock or zero-count
refuses unchanged; a valid decrement and flag clear occur under the mutex before
unlock. Failed unlock retains the container without allowing a second decrement.
The old private void constructor/generation rollback helper is unchanged.

Genuine missing-feature RED was 10 methods/18 failures/0 errors before production
edits. The corrected fixture now passes all 10 methods. It compiles the actual
marked C helper/release/wrapper, real public header and separate C caller, with
**fake private session/engine shapes and backend/TP/distributed/lock effects**.
Tiny real libc owners and forwarding sensors check physical-free-before-retirement,
published handles, full current/live sums, allocation attempts, retained retries,
and reservation state at fake unlock entry. This is not a combined real session,
engine, graph and runtime execution, nor a real pthread failure model. Rejected
fixture syntax/linkage/oracle/scoping defects are not feature RED.

Prior host controls pass: 24 observer, 22 tensor, 21 graph, 14 raw-host, 57 legacy
CUDA-ownership, six frontend, seven snapshot and 27 bench/eval methods; runtime,
plan/parser/emitter checks and both twelve-record synthetic compositions also
pass. Seven selected source checks preserve lock/mutation/unlock boundaries.
The two existing free-body extractors now inspect the checked implementation;
the graph refusal still requires a failure return before common cleanup.
Independent review found no blocker within this bounded increment. These working
host checks do not claim exact-version immutable/full-CUDA seals. The earlier
raw-host baseline
`39937a892c5242d5e3d2da3e72e5a66514dce2c3` already has its own immutable host and
full NVCC `sm_121` compile/four-link seals; they do not cover this later change.

This increment does not repair failed-constructor handoff, engine/source-context
custody, late-exit/publication propagation or complete resident allocation
coverage. Void backend leaves retain their existing coverage limits. The engine
session count is not a list of reachable cleanup owners. Production graph callers
remain LEGACY and the qualification guard remains closed. No produced CUDA
product, GPU or model was executed; cold/three-warm and sixteen-slice published
qualification are still outstanding.

Reusable rule: **retryable cleanup needs an observable status, a surviving owner
handle and explicit completed-prefix state. Test consumed state using observations
made before physical free, never by dereferencing the consumed container.**

**Explicit raw-host owners, host-tested increment:**
`ds4_gpu_laguna_resident_host_calloc/free` now owns raw HOST payloads through a
C-safe caller-held handle. Namespace `0x48` is distinct from a tensor descriptor's
`0x52`, even when both records have HOST domain and OTHER_HOST category. Only
ledger arrays and the seven explicit engine/model/bootstrap/vocab/session/tracker/
serializer callsites are accepted. This primitive uses the existing resident lock
and requires exact attached identity, but no initialized GPU or CUDA API call.
It does not integrate any engine/parser/graph caller or infer admission.

Checked multiplication, record storage/slot capacity and producer-ID preflights
precede libc. Successful calloc is recorded at its actual base before publishing
the handle. Over-bound insertion preserves the real transient peak; rollback
physically frees before retiring its live record. Checked free authenticates the
live ID, namespace, site/category/domain, size and base, and refuses live dependent
relations before touching the payload. Legitimate free/retirement works while
unsafe; observer end now refuses outstanding raw-host owners as well as tensors.
C free has no failure return, so it does not need a private CUDA-style quarantine.
Caller-owned handle/tracker/record/callsite storage must remain outside the payload
and survive cleanup; arbitrary corruption or concurrent direct mutations are not
supported. Early bootstrap authority and full source/context lifetime remain open.

Root-reviewed RED preceded production edits: 14 methods/31 missing source/header/
Make failures. First focused GREEN passes all 14 methods with actual C++ observer/
host bodies, separate real runtime C, and a C11 header/link probe. Forwarding libc
sensors use tiny real allocations, count every free-order violation and every fake
CUDA boundary, and check zeroed contents, actual currents/peaks, classification,
capacity/IDs/tombstones, identity, restored malformed handles, registration/retry,
and checked final detachment. The other-producer namespace control uses a labeled
HOST record setup through the actual runtime, not a tensor API or GPU execution.
The 64-bit size branch is an overflow test, not claimed narrow-size_t execution.

Independent source review found no blocking runtime defect. Root strengthened
its two LOW oracle gaps: every site's record is checked individually, and every
physical free checks the complete current total as well as a live record. All 14
methods still pass. Prior host regressions and five selected source checks pass.
Exact-commit host and full NVCC verification remain separate pending steps.
No produced CUDA product, GPU or model was executed. The production qualification
guard remains closed. There is no new proof of memory fit, numerical correctness,
throughput or published qualification.

Reusable rule: **bind writes to parsed full scope paths and stop on a missing
expected parent; do not create a guessed worktree.** Static scaffold/configuration
failures are not feature RED. Keep physical cleanup, record retirement and sticky
qualification failure as separate facts, including when cleanup succeeds.

**Explicit graph owner composition, host-tested increment:**
The graph now selects `LEGACY` or `RESIDENT` explicitly. Both existing production
allocation callers remain `LEGACY`; neither tracker presence nor a noncompact
profile opens resident execution. Native mode uses embedded typed owners and
own-graph slot bindings, not a second heap allocation or legacy `0x4f` charges.
The canonical 28 scratch plus 96 K/V tensors require 124 owner entries and 248
native records. The preflight budgets reusable records and monotonic producer IDs
separately, checks attached identity before tracker access, and rejects unsupported
Metal/ROCm branches without linking CUDA native-owner implementations.

Native cleanup releases in allocation order, clears only successful aliases, and
retains failed/unvisited owners, tracker and graph storage for a checked retry.
Copied/foreign slot bindings refuse before dereference. Observer attachment is a
read-only identity query even while unsafe; cleanup may detach successfully while
preserving the violation and peaks. Legacy physical-free order and reverse record
retirement remain separate. Callers check local graph cleanup; session failure
returns before common storage is freed. The public void session/engine teardown
chain and failed-session custody are still unfinished, not a publication gate.

Root-reviewed feature RED preceded production edits: 20 methods/38 failures at
missing graph/bridge/build seams. First GREEN: all 20 methods using the actual C
graph block, actual C++ native observer/tensor functions and separate real runtime
C TU. Oracles cover exact currents/peaks, four owner cycles, partial rollback,
private unrecorded rollback, cleanup retry, identity and slot refusals. Planner
and data I/O are labeled host-fixture stubs. Backend controls select preprocessor
branches after real platform headers; they are not Metal/ROCm device tests.
Wrong-site cases mutate metadata after valid attachment and prove changed-config
refusal, not admission of alternate tracker configurations. Prior owner, resident,
legacy CUDA, runtime, snapshot and bench/eval host regressions and five selected
source checks pass. Independent review found no in-scope native graph blocker.
A further actual-runtime legacy-retirement-refusal control brings the graph suite
to 21 methods: aliases are already null before failed record retirement, retry
does not repeat physical frees, and the first violation persists. The control's
initial report-bound confound was retained and corrected before acceptance.
Exact graph revision `e2bc9c2dd8f76e519fb31157d4171258f28865af` now has immutable
host and full NVCC `sm_121` compile/four-link seals. The prior tensor revision
`f47d094c45754ef393716e741226815db1de1242` retains its own version-scoped seals.
Neither seal covers the later raw-host increment above. No produced CUDA product,
GPU or model was executed. Production resident qualification remains blocked on the remaining
reachable owners, engine authority, authenticated snapshots and checked late
failure through child exit/publication. This is not allocation fit, numerical
correctness, throughput or published qualification evidence.

Reusable rule: **compose physical owners with stable caller-local bindings; do
not copy a live owner container or replace failed cleanup with metadata reset.**
A syntactically repaired scaffold is not feature RED. First validate its language,
forwarding sensors and cleanup oracles; preserve rejected revisions separately.

**Explicit resident tensor owners, host-tested increment:**
`ds4_gpu_resident.h` exposes a CUDA-only allocation/checked-free handle without
changing the ordinary tensor layout. Each tensor has two actual namespace `0x52`
owners: its CPU descriptor at `OTHER_HOST_SESSION`, and CUDA storage at
`KV_STATE` or `GRAPH_SCRATCH`. Both record slots and producer IDs are checked
before allocation. The descriptor event precedes the CUDA allocation. Failed
allocation preserves transient peaks and either rolls back or retains the only
private retry handle; no failed driver result is relabeled as a successful event.

Checked free authenticates IDs before dereferencing a descriptor, checks live
relations/current device/synchronization, and preserves both owners on failure.
Physical release precedes each record's retirement. Successful cleanup zeros the
handle but does not clear a sticky violation. A wrong attached-tracker identity
refuses without mutating either tracker. Caller quiescence and retained tracker
storage remain required; this is not an arbitrary-memory-corruption boundary.

Generic CUDA tensor allocation/view/free paths refuse while attached. Generic
use before attachment sets a one-way fresh-process fence, even after cleanup.
Tensor data I/O and compute remain usable; their reachable lazy allocations are
separate coverage work. Metal/ROCm implementations and tensor layouts are unchanged.
All 37 explicit GPU-header Make rules now depend on the shared header, including
CPU and test-hook objects. The benchmark dependency oracle retains exact equality
with the new prerequisite rather than weakening the check.

RED preceded implementation: 22 methods failed at missing ABI/body/build wiring.
The root reviewed and repaired fixture semantics before execution. GREEN: 22 new
tensor methods plus 24 prior-owner methods use extracted native functions and the
real tracker against fake CUDA. C99/C++17 include-order probes and the broader
resident/legacy CUDA/runtime/source/snapshot/bench-eval host controls pass.
Independent review preserves the limits: no full CUDA translation-unit build or
GPU/model execution is established for this tensor increment by these host tests.

Production remains blocked. Graph/KV caller composition, remaining host/managed/
registration owners, selected-expert cache and other raw allocation paths,
explicit engine attachment, authenticated external snapshots and late-cleanup
exit/publication refusal are unfinished. Earlier owner notes below predate these
tensor primitives, not these outstanding caller and admission requirements.
Reusable rules: **an allocation descriptor is a physical owner, not free metadata;
shared-header changes require dependency-oracle updates outside backend-named
tests as well as build-rule changes.**

**Native resident owner events, bounded first slice (2026-09-08):**
CUDA scratch, raw pinned model-stage reservations and device weight arenas now
use event-time observation helpers. Sites 1/11/22 issue namespace `0x52` owners
only after successful physical allocation. Capacity, ID and classification failures
refuse before driver calls. Physical free precedes tracker retirement; failed
free/rollback keeps the owner and a sticky violation. A never-returned allocation
can retain a private teardown handle even if recording or the driver result fails.
No failure can turn cached attribution back into a valid snapshot.

Scratch growth refuses rather than returning a retained undersized slab. Arena
release retires a successful prefix, retains failed/unvisited owners and blocks
model rebinding until explicit cleanup succeeds. Raw stage alignment slack and
full arena reservations are charged, not only their logical views. Compact and
resident attachment are mutually exclusive, but this borrowed-tracker seam is
**not engine attachment, admission, a complete owner inventory or authentication**.

Test-first host evidence: 24 actual-observer/real-tracker methods under a fake
CUDA driver, 57 legacy CUDA contract methods, 283 runtime assertions and 32 source
sampler checks. The owned runtime fixtures preserve real retained-FD replacement
and fork identity checks inside a private test directory. Independent fixture
review plus root review kept async/CUDA and remaining-owner limits explicit;
pinned release/relation controls distinguish report-only registration from owned
charges. The new host target is in the resident/default aggregate.

This is still partial integration: graph/KV/tensors, managed/registration/host
owners, remaining caches, authenticated external capture, resident snapshots and
admitted end-to-end execution are unfinished. Host tests do not prove a new full
NVCC build, GPU execution, model correctness, memory fit or throughput. Production
resident qualification keeps its pre-startup refusal. The reusable rule is:
**record allocation events and retire only after physical release; a checkpoint
inventory cannot recover transient or failed-owner peaks.**

**Bounded qualification decoder (2026-09-08):**
The shared resident/streamed benchmark runner now executes up to the authenticated
512-token cap. It uses greedy native argmax and `ds4_token_is_stop`, not the
ordinary CSV benchmark's EOS-excluding policy. Later native stops complete with
actual output counts; a stop before any output refuses instead of inventing a
first-token record. Four fresh sessions still share one engine and twelve
successful milestone records. Only the first output performs the first-token
checkpoint; later decode failures stop before barrier/finish/completion.

Root reviewed and corrected fixture oracles before RED (512 is a cap, not EOS
suppression; a repetition-2/token-257 failure has eight milestones and three
first-visible marks). RED preceded implementation. The host lifecycle now covers
four full-length repetitions, EOS and a distinct native stop after three outputs,
zero-output refusal, late-token failure, and exact typed cleanup. Both real
emitter compositions, six frontend methods, the resident plan and existing
27 bench/eval regressions pass. Independent source review found no correctness
finding. The one-token smoke at `51ddbfa` is superseded, but production resident
execution remains gated: native owner coverage, authenticated snapshots and
admitted end-to-end qualification are not established by fake-backend tests.

**Resident benchmark lifecycle groundwork (2026-09-08):**
The explicit `ds4-bench --qualification-resident-sequence` selects the typed
resident parser and emitter. It shares the existing one-engine/four-fresh-session
lifecycle without changing the streamed record API or treating schema/profile
strings as mode authority. All twelve accepted/first-token/completion records
pass the real resident serializer and separate consumer in a fake-backend host
composition; streamed composition still emits twelve valid streamed records.
At this historical `51ddbfa` milestone the shared Task 19 runner generated
one non-EOS token per repetition; the bounded decoder increment above supersedes
that smoke-only execution, not its native-accounting evidence limits.
Parser/emitter/checkpoint/snapshot failures stop and free the current session,
engine, and selected sequence. Fixed resident argv rejects even matching or
ignored benchmark/streaming overrides and duplicate model/backend selectors.

`test-laguna-resident-path` is now in `make test`: the lifecycle, both typed
compositions, six frontend/aggregate test methods, production translation-unit
compile, and resident plan pass. The real CPU-only frontend verifies an explicit
production refusal before NVML/model/engine work until native resident
allocation events and authenticated snapshots exist. The test-backend macro
is not production readiness. The 27 existing bench/eval tests, trusted parser
(57 checks), resident parser (193), and resident emitter (1188) also pass on the
host. No new Linux/NVCC/GPU/model result is claimed for this increment.

Reusable rule: a typed emitter validates/serializes observations; it does not
create or authenticate them. Keep an unfinished native producer visibly
unavailable even when fake-backend lifecycle and reservation tests are green.
Task 20 still requires native ownership accounting, admitted resident execution,
full schedule composition, gates/retries, and verified atomic publication.

**Resident reservation-plan groundwork (2026-09-08):**
`ds4_laguna_resident.{h,c}` now defines a separate 32K/4K/one-session
reservation plan, not a rewritten compact-cache plan. It preserves exact
ledger/KV/graph geometry, admits device/managed/pinned owner envelopes, and
reserves no compact expert cache. Model mapping and registration remain
report-only; qualification totals add only owned allocations and measured
source/host/CUDA external reports. The tracker starts with no observations;
plan bounds must never populate current or peak measurements.

The missing-header RED preceded implementation. `test-laguna-resident-plan`
passes 608 host assertions, including real reference-tracker ownership,
relations, retained peaks, independent invalid inputs and zeroed overflow
outputs. Independent review found no blocking arithmetic/classification
issue. Native callers must supply a builder-produced retained-model ledger;
this helper validates scalar geometry, not model/array provenance. Resident
native events must explicitly use its aggregate pinned11/device22/managed26
IDs rather than forwarding absent compact pool/workspace IDs. No allocator
hook, native resident snapshot, admitted run or GPU qualification is implied.

**Snapshot-buffer repair (2026-09-07):** The recovered
`test-session-snapshot-buffers` target exercises the production
`ds4_session_save_snapshot` body with the host libc memory stream and a small
stub payload. Before the fix, its controls passed and five of seven cases
failed. The writer now reserves one byte outside the serialized payload for a
possible `fmemopen` terminator, rejects `SIZE_MAX` before allocation, and keeps
the existing terminal/distributed-session guards. All seven focused tests pass
on macOS. This verifies the host allocation/stream boundary, not a Linux,
whole-engine, model, Metal, or CUDA qualification result.

**CUDA allocation-safety integration (2026-09-07):** The runner's native safety
changes were selected independently of its competing wrapper/schema changes.
Staging reads now use the span remaining after pointer alignment and reject
capacity/address/file-interval overflow before I/O. Checked staging release
retains failed and unvisited owners for retry, and arena allocation rejects
unrepresentable alignment or invalid existing geometry before allocation.
The three recovered host contract targets ran RED with passing existing-body
controls, then passed 19/20/8 tests after the patch. These use extracted C++
bodies and fake CUDA calls, not NVCC, GPU execution, async-quiescence proof, or
model qualification. Admitted schema binding and descriptor-relative evidence
verification are retained. Task 20 and public `run` remain incomplete.

**Frozen execution schedule (2026-09-07):** `qualification_schedule.py` now
materializes exactly sixteen immutable slice inputs: the four canonical
resident prompts, then the 8/12/16-GiB streamed profiles in their frozen prompt
orders. Each slice carries its trusted record kind, profile/prompt/index,
manifest digest, exact sequence bytes and sequence digest. The helper validates
and detaches the manifest and reuses the existing mode-specific builders.
Six new tests passed after the missing-module RED; the four existing resident
sequence controls also pass. `test-qualification-schedule` is part of the pinned
Python aggregate. This is deterministic runner input, not admission, process
execution, retry/gate policy, native residency accounting, or a qualification
verdict. No public `run` or publication behavior changes in this increment.

**Native platform proof (2026-09-07):** Clean commit
`c57b062a90c9729f231fa7dc49fb7618eab5420d` compiled all four production
programs (`ds4`, `ds4-server`, `ds4-bench`, `ds4-eval`) with NVCC 13.0.88
and `CUDA_ARCH=sm_121` on DGX GB10. The authenticated source archive, clean
Git tree, complete compiler/link log and output binary digests were retained
outside the worktree. This was compile-only; none of those programs ran.
All 54 snapshot/staging/ownership/arena host regressions also pass on Linux
(7/19/20/8), including real glibc memory streams. CUDA calls in those host
fixtures remain fake; the same 54 tests pass on macOS.

A separate root-reviewed, tiny-file Linux fixture now passes three tests:
real retained-descriptor ELF version transport (success, raw malformed bytes,
nonzero exit and timeout), authenticated twelve-checkpoint control with real
`/proc/<owned-pid>/exe`, and independent wrong-inode executable refusal.
`test-qualification-linux-native-origin` uses the pinned requirements and
explicitly skips on non-Linux; no fake proc tree or Darwin adapter is used.
The fixture uses legacy `input_admission=None`, so this proves Linux host
transport/origin, not admitted-schema provenance or native DS4 runtime identity.
No GPU/model, resident allocation/snapshot, numerical, throughput, gate or
publication acceptance is claimed. Those remain Task 20's completion boundary.

Reusable native-test rule: compile and execute a small real ELF for Linux
`/proc`/descriptor semantics; retain failing controls and raw observations.
Keep model-FD evidence separate from executable identity. Verify actual resource
and access boundaries rather than treating sandbox configuration admission as
proof. A native compile and host transport pass cannot replace a GPU/model run.

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
The runner and publication steps below therefore remain unchecked.

**Supervisor progress (2026-09-05):** `9b07a35` adds incremental, deep-copied
validated-record draining (19 parser tests). `qualification_supervisor.py`
adds pure lifecycle/parent-clock deadline monitoring (14 host tests), including
coalesced late milestones, clock regression from launch onward, complete-stream
closure and retained partial observations. `192ed3e` closes a reviewed late
parent-receipt bypass without discarding coalesced observations.

**Process transport progress (2026-09-05):** `qualification_process.py` now
launches one foreground child in a new session, drains both capped raw streams,
enforces milestone deadlines, and preserves validated partial observations.
Fifteen native fake-process tests cover success, protocol/exit/output errors,
phase timeouts, descendant-held pipes, interrupts and bounded group cleanup.
`8a551d3` separates released resources from EOF after I/O failure. The transport
checks deadlines even when observing a nonzero exit, and freezes producer time
only after both child exit and both pipe EOFs are observed; later cleanup grace
cannot manufacture an exit timeout. EOF remains necessary for `complete`.
The parent
uses `waitid(WNOWAIT)` to reserve the direct-child PID until all group signaling
ends, then reaps and proves group disappearance. Darwin needs a small libc
`waitid` bridge because this Python build omits `os.waitid`; zombie-only groups
can reject signals with `EPERM`, so actual release—not signal acknowledgement—is
the cleanup authority. These are transport facts, not qualification verdicts.
`QualificationControl.wire_records` now retains bounded raw message and
failure-prefix observations (six host tests, including interrupt propagation
from deadline/readiness clocks). A full-frame flag is not a
protocol verdict. The 64-KiB lifetime budget rejects an over-budget ACK before
sending it, and received rights close on failure. The process owner is shared
through one context so the control-FD path can reuse its tested cleanup rather
than duplicate process supervision. `qualification_controlled.py` now uses
that owner for exactly one inherited control descriptor, descriptor-bound model
preparation, twelve READY/RESULT brackets, and a strict trailing-control EOF
check. Four generated-child host tests cover simultaneous pipe backpressure,
snapshot/ACK ordering, retained preparation failures, milestone timeout while
waiting for control, and unexpected-rights closure. Clock interrupts propagate
rather than turning into protocol errors. These are tiny-file host tests, not
model, CUDA, executable/runtime-authentication or qualification acceptance.
Full orchestration, gates/retries, authentication and publication remain
unfinished; the CLI `run` rejection is unchanged. At `0092010`, immutable
verification passed 58 focused tests, 201 aggregate Python tests and the native
host harnesses. Independent review found no confirmed defect. The next active
seam is immutable input/schema/build admission and retained artifact identity.
The shared regular-file opener now rejects FIFOs without blocking, and model
hashing reads only the initial size with pre/post identity checks (94 qualifier
tests, including two new RED-driven file-boundary regressions).
`qualification_artifacts.py` now owns nofollow, read-only file descriptors,
checks byte caps before hashing, and detects retained-inode/path/change-time
drift. Six host tests include proc-exe identity simulation, real file growth,
replacement, and same-size content changes with a restored mtime. Proc-exe
matching does not establish PID ownership; the controlled transport must hold
the PID, and live Linux process authentication remains unqualified here.
`qualification_admission.py` now reads and pins the manifest and six schema
files, rejects nonlocal schema references and invalid/oversized inputs, matches
the model inode/content, and validates three same-revision clean-CUDA version
responses before yielding four artifact owners. Five host tests cover byte
hashes, ordering, drift, rejection/cleanup and final context-exit verification.
At `4782539`, immutable host verification passed 130 focused tests, 214
aggregate Python tests and the native host harnesses; independent review found
no confirmed defect. Admission's required callback still treats injected version
bytes as trusted claims, not native-origin proof.

**Native version-probe progress (2026-09-05):**
`QualificationVersionProbe` now provides that callback by executing the retained
artifact descriptor with exactly `--version-json`. It borrows exactly one FD,
caps both raw streams at 64 KiB, applies one fixed deadline, and checks the
artifact again after owned cleanup. Its detached observations survive errors
and interrupts, including interruption during context admission. The shared
process owner records release proof even when cleanup is interrupted; merely
reaching `finally` is not proof. Seven host tests pass, plus the 30 affected
process/control/artifact/admission tests. Pinned verification and review of this
increment remain pending at commit time.

Descriptor execution is Linux-only. Measured Darwin `/dev/fd` attempts failed
with `EACCES` for both a script and a native interpreter. The Darwin tests use
an explicit Python-FD-content adapter for supervision only, plus an unadapted
fail-closed/no-launch test. They do **not** prove native Linux origin or Linux
`waitid` behavior. No mutable-path execution fallback exists. Full orchestration,
runtime authentication, gates/retries and bundle publication remain unfinished;
`run` still refuses incomplete execution.

At `81a8388`, pinned verification passed 137 focused tests, 221 aggregate Python
tests and the native host harnesses; independent review found no confirmed
defect. `qualification_authenticated.py` now composes controlled execution with
retained model/executable owners. Only the executable FD and owned control
socket are inherited. The running executable and received model descriptor
must match the pinned inputs before preparation and before/after every sample
callback, including the final callback before its RESULT_ACK. File drift in
the exit tail rejects the result without discarding its completed raw prefix.
Six host cases cover success, identity mismatches before model ACK, final-ACK
refusal on drift, exit-tail drift, interruption and preflight rejection. The
proc tree is simulated on every host; Darwin also uses the explicit FD-content
adapter. These tests do not qualify Linux-native running-executable origin.
Pinned verification/review of this increment remain pending at commit time.
Resident-first orchestration, runtime-schema binding, cold preparation and
snapshot collection, gates/retry policy, filesystem evidence authentication and
atomic bundle publication remain unfinished.


**Resident producer dependency (2026-09-05):** At `83cb9b7`, pinned checks passed
143 focused tests, 227 aggregate Python tests and the native host harnesses;
independent review found no confirmed defect. Source inspection then confirmed
that the current native plan writer and controlled benchmark are streamed-only.
Resident engine open builds the tensor ledger, but does not initialize the
qualification allocation tracker or publish `ds4.runtime/v1` snapshots. The
human startup estimate and one-pass smoke eval are not substitutes for the
resident plan, observed footprint and same-child cold/warm baseline required
below. No native resident/model/GPU run was made during this inspection.

Keep the existing streamed sequence/parser/emitter contracts unchanged. Add a
separate, trusted resident sequence boundary before connecting native resident
allocation/snapshot production: `ds4.resident-qualification-sequence/v1`,
`profile_id=resident`, `cache_bytes=0`, `mode=resident`, canonical prompt order
512/2048/8192/28672, and the same fixed one-cold/three-warm sampling contract.
Host parser/builder tests must reject cross-mode substitutions in both
directions. This boundary alone will not implement resident allocation
accounting, execute a baseline, enable CLI `run`, or qualify CUDA behavior.


**Resident sequence progress (2026-09-05):** RED `7c94a40` recorded the missing
Python module and C resident type/API. The shared bounded C parser now has a
separate, trusted-only resident entrypoint and owner type; it never selects the
mode from file contents. Python uses a separate public resident builder and the
shared fixed byte formatter. All four canonical prompt positions and the
zero-cache resident profile are bound to the manifest/input/sequence digests.
Root verification passed four Python tests and 193 native C checks, including
mixed schema/profile/mode rejection in both directions, canonical-zero, input
size and decoded-input limits. The existing streamed parser, trusted parser,
and 27 benchmark/eval contract tests passed without changing their contracts.
Pinned verification and independent review remain pending at commit time.

The allocation review also confirmed that ordinary resident CUDA first tries
registered host model pages and can fall back to a device range cache. Neither
that fallback nor optional dequantized caches may be silently treated as zero
allocation. Resident graph setup also currently chooses up to 16384 prefill
rows rather than the compact plan's 4096 rows. The next native producer work
must bind actual allocation paths and declared geometry before publishing a
resident runtime snapshot. Startup estimates are not measured footprint.
No resident baseline execution, native CUDA qualification, or full CLI `run`
implementation follows from the sequence boundary.


**Resident raw emitter progress (2026-09-05):** At `440d095`, pinned checks passed
147 focused and 231 aggregate Python tests plus native harnesses; independent
review found no confirmed defect. RED `e2b65b3` then required a distinct resident
record type and emitter. `ds4_bench_resident_qualification_emit_record()` now
uses the shared bounded formatter/validators through a resident-only entrypoint.
It requires the resident profile/order, zero expert-cache config/bound/current/
peak, 32K context, 4K configured/effective/allocated prefill geometry, one session,
and stable external inventory. Request completion must bind matching metrics.
Registered model pages, static/device allocations, and observed resident/source
footprint are copied without replacement by zero or a startup estimate.

The repaired native fixture passes 48 independent emissions and 1188 checks;
the existing streamed emitter/lifecycle/composition/production compile and 27
benchmark/eval Python contracts also pass. Each negative fixture starts from a
valid record and independently reset nested objects. Both pipe ends are
nonblocking and a 15-second alarm bounds the whole host fixture. The emitted
objects are raw observations, not a complete resident twelve-record consumer,
engine allocation tracker, native snapshot producer, or executed baseline.
Pinned verification and independent review of this increment remain pending
at commit time. No CLI `run`, resident model/GPU execution, or CUDA qualification
is claimed.

**Resident raw consumer progress (2026-09-05):** The emitter at `dc5b643`
subsequently passed 50 affected/focused and 231 aggregate Python tests plus
native harnesses; independent review found no confirmed defect. RED `aa9d0cd`
then required `QualificationResidentRecordStream` and `validate_resident_record`
in `qualification_resident_records.py`, with the separate
`ds4.bench.resident-qualification/v1` JSON schema. The public resident boundary
selects its own schema/profile/order and cache-zero/4096-row restrictions. The
existing streamed boundary and its schema remain strict and separate.

Private parsing/lifecycle code is shared without temporarily changing shared
schema paths, profile maps, or byte limits. Each slice binds all six expected
values, accepts exactly twelve LF-terminated records, and enforces the existing
1-MiB record, 12-MiB stream and 64-level JSON bounds. Copied drain/finish results
preserve valid partial evidence without clearing sticky failure. Matching raw
footprint and completion metrics are retained; they are not qualification gates.

The model-free fixture now compiles the native emitter and feeds its actual
bytes as four complete slices. Unlike the independent raw-emission fixture,
each repetition uses accepted/first/completion snapshots at base+1/base+2/base+4
with request metrics at base+3. Before pinning this increment, 11 resident, 19
streamed and 14 deadline tests, plus native emitter/lifecycle/composition and
production-compile harnesses, passed. Pinned aggregate verification and
independent review remain pending at commit time.

This consumer is not yet wired into a resident deadline/process/control runner.
The admission schema registry remains the existing six bundle-required schemas;
authenticated loading/binding of consumed record schemas still needs integration.
Actual resident allocation tracking, plan/source-page accounting, engine runtime
snapshots, controlled baseline execution, gates/retries and atomic publication
remain unfinished. No model/GPU run, native CUDA qualification or CLI `run` is
claimed.

**Shared native model-page sampler progress (2026-09-05):** The consumer at
`44861f6` passed pinned 71 focused and 242 aggregate Python tests plus native
harnesses, with independent review clear. RED `2ab0285` then required the native
`ds4_runtime_model_source_resident_bytes()` API. The existing compact CUDA
sampler now calls this shared runtime helper instead of owning a separate
`mincore` loop. This makes actual source-page measurement available to resident
production without copying compact-only allocation bounds.

The helper uses at most 65536 residency bytes on the stack per batch, requires
the actual system page size, rejects alignment/rounding/full-address-interval
overflow, and commits output only after every batch succeeds. Resident final
partial file pages are charged as full physical pages. It neither touches
model contents nor owns the mapping/FD; the caller must hold their lifetime.
A multi-batch sample is an observation, not a globally atomic page snapshot.

Before pinning, the new native harness passed 32 checks. It measures an
immediately-unlinked, touched two-page host mapping against an independent
`mincore` result, then separately interposes system calls for bounded large-span
arithmetic, status-bit masking and transactional failure tests. Setup failures
must fail the fixture; an oversized fake syscall must not overrun the vector.
The C++ linkage/signature check reaches the new C symbol. Existing host checks
passed 128 Python tests plus runtime/Laguna/emitter/lifecycle/composition harnesses.
The CUDA build-contract suite uses host checks, not a CUDA compilation or GPU run.
Pinned aggregate verification and independent review remain pending at commit time.

The native audit also confirmed that resident startup rejects an explicit
prefill chunk, allocates up to 16384 graph rows, and caps execution chunks at
512 rows. Those paths are not a truthful configured/effective/allocated 4096-row
qualification baseline. No resident geometry, CUDA cache policy or allocation
bounds are changed by this sampler. Resident allocation-plan/tracker attachment,
external attribution, engine snapshots and controlled execution remain required.
No full CLI `run`, native Linux/CUDA execution or model qualification is claimed.

**No-copy registration ownership progress (2026-09-05):** At `f873014`,
pinned verification passed 128 focused and 242 aggregate Python tests plus the
native host checks; independent review found no confirmed defect. RED `e9d8d46`
then compiled the real no-copy registration function in a host-only fake CUDA
fixture. Both positive controls passed, while seven fault/recovery cases exposed
37 field failures. The fixture counts live registrations independently of the
production ownership flags and allows more than one mapping to remain live.

`ds4_gpu_register_model_map_no_copy()` now checks prior registration release
before destroying caches or replacing its owner. A successful registration is
owned before device-pointer lookup. Failed or null lookup requires rollback;
a failed rollback returns failure while retaining the registered host base/size
and a null device pointer. Same-map reuse cannot turn that unresolved state into
success. A later different-map call can recover after releasing the old owner.
The existing nonregistered fallback remains available after successful rollback.
All nine host cases now pass, including failed-release preservation and recovery
from a failed rollback. `test-cuda-model-registration-contract` is included by
`test-cuda-build-contract` and the aggregate `test` target.

This fixes only the no-copy registration transaction. Generic CUDA cleanup,
range-cache releases, arena reservation, optional Q8 caches, tracker attachment,
resident 4096-row geometry, external attribution and native snapshot production
are not repaired or supplied by this increment. The host fixture does not
compile a CUDA translation unit or execute a GPU/model. Pinned regression
verification and independent review remain pending at commit time. Task 20 and
CLI `run` remain incomplete.

**Explicit CUDA prefill geometry progress (2026-09-05):** At `d5b85e1`,
pinned checks passed 137 focused and 251 aggregate Python tests plus native host
checks; independent registration review found no confirmed defect. RED `730cf4d`
then exposed the existing row policies through host test hooks. The native
geometry case compiled and failed 14 of 28 assertions, with the legacy sizing
controls intact. Four source-wiring tests also failed at the real admission,
allocation, dispatch and both estimator definitions.

Explicit Laguna CUDA prefill rows now select the graph capacity and session
step size through shared tested helpers. Both GPU-enabled and `DS4_NO_GPU`
estimator branches use the same capacity rule. The prefill restriction permits
an explicit CUDA override, but does not grant new Metal/CPU backend support or
weaken compact plan admission. Compact allocation retains its plan rows. With
32K context and explicit 4096 rows, the graph sizing path reports KV 1686110208
plus scratch 1537052680, totaling 3223162888 bytes. The host sequence selects
seven 4096-row steps for the canonical 28672-token prompt.

An effective CUDA prefill value of zero retains the legacy up-to-16384-row
allocation and up-to-512-row resident dispatch. Nonzero effective values are
honored, including an existing tensor-parallel default of 512; that path's
allocation can shrink accordingly. Invalid configured rows above context yield
zero capacity and fail before graph allocation. This is geometry selection,
not a new resident allocation bound or full backend/profile admission policy.

The geometry and prior prefill-plan cases pass 53 host assertions in both the
GPU-enabled C host build and a separate `DS4_NO_GPU` build; neither launches GPU
work. The four source-wiring tests pass. The geometry case is included by
`test-laguna-stream` and the main `test` recipe. Pinned regression verification and independent
review remain pending at commit time. CUDA kernels are unchanged. No model/GPU
execution, numerical parity, throughput acceptance, complete resident inventory,
tracker/snapshot producer, or full Task 20 runner is claimed; `run` still refuses
incomplete execution.

**Evidence-file authentication progress (2026-09-05):**
`qualification_evidence_files.py` verifies a supplied canonical index against
bounded regular files using descriptor-relative no-follow traversal. It checks
exact file union, sizes, streamed hashes, pre/post descriptor identity and a
final metadata/tree pass. Missing/extra files, symlinks (including root spellings
`link/` and `link/.`), special files, malformed/noncanonical indexes and observed
mutation/replacement are rejected. The index builder stays declaration-only.
`make test-qualification-evidence-files` runs the 28 host tests in the pinned
Python environment and is part of `test-laguna-compact-python`; the existing
index target now uses that environment too. Precision tests distinguish the
post-read identity boundary from hash rejection, force an identical-byte
rewrite between scans, and prove descriptor release on nested read/scan errors.

This is file-byte authentication, not native acceptance or gate provenance.
The caller still must derive references from the exact union of all gates,
authenticate the supplied bundle/index, and reverify before durable publication.
Reserved regular bundle/sidecar/index files are excluded, not authenticated by
this helper. Its v0 limits are a 1-MiB index, 4096 evidence files, 64 MiB per
file, 256 MiB total, 8192 directory entries and 32 path components. No `run`,
`verify` or `publish` CLI has been admitted; model/executable control-channel
composition, gate/retry evaluation and publication remain open. No GPU or model
run is implied by the host tests.

See [port-observability notes](../../spikes/2026-09-04-laguna-port-observability.md)
for diagnostic reuse, verification commands and the next implementation seam.

**Authenticated record-kind wiring progress (2026-09-05):** The host-only
qualification composition now accepts an explicit keyword-only `record_kind`
with a fixed streamed default and selects only the trusted streamed or resident
record stream. Resident and streamed slices can share the same retained model
and executable owners; returned data remains observations, not a qualification
verdict. Existing schema-path consumption is unchanged, and authenticated
consumed-schema admission remains follow-on. The native resident plan/tracker/
snapshot producer is still missing. Proc origin remains simulated, and the
Darwin descriptor-content adapter remains test-only. Public `run` remains
rejected.

**Record-schema admission progress (2026-09-06):** Admission now requires,
retains, and hashes both benchmark record schemas alongside the six core
schemas. Only the two record schemas may use the exact runtime and request
schema filenames as cross-document `$ref` targets; core schemas remain
local-only, and resource retrieval remains disabled. The existing bounded
owners and lifetime checks cover all eight files before and after version
probes and through context exit. This does not yet bind the record parsers'
path-based reads to admitted snapshots. That consumption binding remains the
next integration step; native qualification and public `run` remain absent.

**Files:**
- Modify: `gguf-tools/quality-testing/compact_runtime_qualify.py`
- Modify: `gguf-tools/quality-testing/test_compact_runtime_qualify.py`
- Create: `gguf-tools/quality-testing/qualification_records.py`
- Create: `gguf-tools/quality-testing/qualification_resident_records.py`
- Create: `schemas/ds4-bench-resident-qualification-v1.schema.json`
- Create: `gguf-tools/quality-testing/test_qualification_records.py`
- Modify: `tests/validate_bench_qualification_json.py`
- Create: `gguf-tools/quality-testing/qualification_evidence.py`
- Create: `gguf-tools/quality-testing/test_qualification_evidence.py`
- Create: `gguf-tools/quality-testing/qualification_evidence_files.py`
- Create: `gguf-tools/quality-testing/test_qualification_evidence_files.py`
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

1. validate and hash the manifest and all six schemas required by the bundle
   schema (the five runtime/result schemas plus the benchmark-manifest schema)
   before creating the evidence directory;
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

**Admitted-schema consumption progress (2026-09-07):** Both fixed record
streams and the existing monitor/process/controlled/authenticated chain now
accept keyword-only `input_admission=None`. A supplied handle must be the exact
live admission type with a matching manifest binding; authenticated execution
also requires its exact retained bench and model owner objects. Admission
retains four parsed record/runtime/request documents from the already-read,
hash-checked bytes and exposes detached copies. The consumers use a private
in-memory map, with local references resolved in their owning document and
only the two fixed runtime/request dependencies allowed across documents.
They do not reopen schema paths or use an ambient cache in admitted mode.
Live owner checks bracket record validation, finish, and authenticated
callbacks; final exit-tail drift retains the bounded observation prefix as a
protocol error. Existing strict numeric, scalar, flattened, lifecycle, ACK,
deadline, descriptor-inheritance and cleanup boundaries remain in place.
The two new host suites are wired into `test-laguna-compact-python` with the
pinned runtime requirements. GREEN and independent review are pending at
commit time. `None` remains the legacy path, not schema-bound qualification.

Reusable rule: admitting a schema hash is not consumption binding. Pass its
retained admission explicitly, validate only detached authenticated bytes,
and recheck the owners at the existing evidence/ACK boundaries. Do not infer
native origin, gate provenance, or safe publication from schema binding.
Native resident plan/tracker/snapshots, qualification and public `run` remain
unfinished; this increment does not run a model/GPU or publish a verdict.
