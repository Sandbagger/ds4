# Laguna port observability and reusable numerical debugging

Date: 2026-09-04
Status: host-first tools implemented; no new CUDA qualification or deployment.
Base: Task 19 `dd6d628570d9d7c2f2d4999a84c5c9132572922f`.

## Decision

Use differential numerical debugging before broad mechanistic interpretability.
Keep the trusted model/oracle, exact tokens, tensor layout, compiler/math mode,
and invocation fixed. Locate the first differing semantic boundary, then reduce
it to a small replayable operation. Diagnostics do not relax the existing
Poolside tolerances, choose a faster default, or qualify the runtime.

Do not build a second activation framework. This tree already contains:

- `gguf-tools/quality-testing/probe_poolside_laguna_layers.cpp` and
  `compare_laguna_layers.py`: embedding, residual-layer, layer-0 substage and
  logit captures; exact hashes plus RMS, relative RMS, max error and cosine.
- `tests/oracle-producers/laguna-c7/compare_laguna_moe_execution.py`:
  router/expert-stage comparisons, bit coordinates, counterfactual routed-sum
  replay and microscope binding.
- `tests/oracle-producers/laguna-c7/token513-layer1-comparison.json`:
  a retained token-513 investigation. It distinguishes observed F32 operands
  from **unobserved** Poolside internal Q8_1 bytes; inferred values are not
  relabelled as captures.
- `tests/test_laguna_token513_comparison_report.py`: release versus null-hook
  and null-hook versus active-hook transparency contracts.
- `schemas/ds4-bench-qualification-v1.schema.json` and the Task 19 benchmark
  emitter: typed records for Task 20, not a completed qualification runner.

The MoE files live under `tests/oracle-producers/laguna-c7`, not beside the
layer comparator. Their absence from a Task 19 diff does not mean they are
absent from its inherited tree.

## Small host-only changes

1. `4248699`: a bounded incremental qualification-record parser. Reuse the existing
   strict schema/field checks; bind every record to the expected manifest,
   sequence, profile and prompt; preserve one instance and four ordered
   repetitions. Fail closed on truncation, changed identities, malformed JSON,
   excessive depth/bytes and extra records. This is only the input-validation
   part of Task 20. Process supervision, deadlines, retry classification,
   model/executable descriptor identity, gate evaluation and publication remain
   separate work. Parsed synthetic records never become hardware evidence.
2. `b467bc6`: bit-level first-mismatch coordinates in the existing layer comparator.
   Report the original flat index and, where the shape is known, token and
   channel, including slices. Preserve existing metric and first-divergence
   semantics. An exact-bit difference is not a tolerance failure. Reject
   nonfinite inputs even when identical bytes would otherwise take the fast
   equality path.

3. `b1d35bd`: a pure evidence-index metadata builder. It checks the exact
   declared reference/observation union, deduplicates shared equal-digest
   references and rejects conflicts, duplicate observations, reserved paths,
   invalid UTF-8/POSIX paths, malformed hashes and noncanonical uint64 strings.
   It sorts paths by unsigned UTF-8 bytes and returns RFC 8785 bytes without a
   trailing newline. It does not access files, authenticate their contents,
   check symlinks, collect references from a bundle or issue a gate verdict.

The parser and index each pass 12 host tests. The activation change passes 8
new and 4 existing tests. All three changes passed independent, bounded code
review and root verification from pinned detached worktrees. These are
synthetic host checks, not fresh model-oracle comparisons or CUDA qualification.

Reproduce the focused checks from the repository root with the pinned project
environment:

```sh
uv run --with-requirements gguf-tools/quality-testing/requirements-compact-runtime.txt \
  make test-qualification-records test-qualification-evidence test-laguna-layer-diagnostics
```

`test-laguna-compact-python` includes all three targets plus the existing
qualifier suite. `test-laguna-compact-contract` runs the separate exhaustive
schema corpus. The latter passed at `b467bc6` (20 tests); the qualifier suite
at that commit also passed (92 tests). The combined code at `26abf29` passed
155 Python tests with no skips plus the host emitter, lifecycle, composition,
production-translation-unit, sequence and stable-case harnesses:

```sh
uv run --offline --with-requirements gguf-tools/quality-testing/requirements-compact-runtime.txt \
  make -j2 test-laguna-compact-python test-bench-eval-contract \
  test-bench-qualification-emitter test-bench-qualification-lifecycle \
  test-bench-qualification-composition test-bench-qualification-production-compile \
  test-bench-sequence test-bench-sequence-trusted
```

The 20-test exhaustive schema result belongs to `b467bc6`, not a fresh run on
`26abf29`. Verification logs retain exact tested revisions. These host commands
use fixtures/fakes; passing them does not establish live CUDA, descriptor,
NVML, memory-curve, real-model or deployment acceptance.

Use the existing capture-directory contract for a diagnostic report:

```sh
python3 gguf-tools/quality-testing/compare_laguna_layers.py \
  --reference "$REFERENCE_CAPTURE" --candidate "$CANDIDATE_CAPTURE" --format json
```

`first_mismatch` is null or contains the absolute `flat_index`, known-width
`token_index`/`element_index`, canonical `reference_bits`/`candidate_bits`, and
finite numeric values. Unknown-width logits retain null token/element indexes.
The table format adds `first_mismatch=...` lines without changing its existing
TSV columns. Signed zero can be a bit mismatch with zero numerical error.

## Next techniques, ordered by immediate value

### 1. Boundary capture, followed by first-divergence replay

Start with residual boundaries; capture only the first suspect layer's Q/K/V,
RoPE, normalization, gating, router logits, selected expert IDs/weights and
expert projections. Use exact token IDs and layout metadata, never prompt text
alone. At discontinuities, compare IDs separately from float tolerances.
Prioritize the existing SWA-513 and long-context/YaRN cases over a new broad
instrumentation surface. The current generic layer probe has a fixed short
shape; do not claim it already handles arbitrary models or token counts.

A capture must identify model/export/GGUF, converter and quantization, oracle
and candidate commits, executable/compiler/CUDA/driver/GPU, math mode,
tokenizer/template, tokens, shape/strides/layout and artifact hashes. Keep
thresholds preregistered. Capture overhead can change CUDA ordering: compare
release, hook-present-but-disabled and hook-active runs before relying on the
observed values. Diagnostic timing is not performance evidence.

### 2. Operation microscopes and controlled interventions

The existing Q4_K/token-513 microscope and routed-sum counterfactuals are the
right pattern. Extract exact inputs and weights into tiny fixtures and compare
reference, scalar/high-precision and optimized calculations. High precision is
an arithmetic diagnostic, not a replacement for the model's numerical oracle.
Inspect quantization boundaries directly where possible. Keep unavailable
internal values explicitly unavailable.

Activation patching can then test whether replacing one suspect activation
with its reference value removes a downstream discrepancy. Patch the smallest
boundary, preserve control runs and do not ship patching in the inference path.
A successful patch localizes a contribution; it does not prove uniqueness or
establish model quality. Full live intervention hooks are not part of this
host-only change.

### 3. CUDA correctness and performance tools, on tiny cases

Compute Sanitizer `memcheck`, `racecheck`, `initcheck` and `synccheck` complement
numerical comparisons: memory/synchronization correctness is not arithmetic
parity. Nsight Systems is for copies, launches and synchronization; Nsight
Compute is for one kernel's memory, instruction and occupancy costs. Use small
deterministic replay fixtures and narrow capture windows. Verify exact installed
versions and GB10/SM121 support before use. Do not infer compatibility from a
marketing architecture name. These tools have not been run by this change.

## Mechanistic interpretability: useful, but a separate claim

PyTorch hooks, NNsight and TransformerLens offer activation access and
interventions in supported Python models. They are not drop-in hooks for the
custom DS4 C/CUDA/GGUF runtime, and native Laguna support has not been
established. Use them on a verified reference adapter if that becomes useful.
Causal tracing/activation patching can help localization; circuit discovery,
large patch sweeps and sparse autoencoders are deferred until numerical
fidelity and an actual behavioral question justify them.

Primary references:

- PyTorch module hooks: <https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook>
- PyTorch FX: <https://docs.pytorch.org/docs/stable/fx.html>
- NNsight interventions: <https://nnsight.net/documentation/intervention/>
- TransformerLens patching: <https://transformerlensorg.github.io/TransformerLens/generated/code/transformer_lens.patching.html>
- Compute Sanitizer: <https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html>
- Nsight Systems: <https://docs.nvidia.com/nsight-systems/UserGuide/index.html>
- Nsight Compute: <https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html>
- Causal tracing (ROME): <https://arxiv.org/abs/2202.05262>

## Worktree safety

Task 19 is 44 commits ahead of `laguna-s2.1-main-integration` at `c63e6d7`.
The older resident-CUDA line at `a6e49db` diverges from integration (20 versus
471 commits from their common ancestor). Do not wholesale merge it to recover
a diagnostic. Review any selected commit against the integrated code and prove
it on an immutable candidate. Existing source worktrees were clean at the
start; detached verification binaries were left untouched. New implementation
is isolated on `feature/laguna-task20-runner`.

## Next Task 20 implementation seam

Keep the existing `run` rejection until a tested orchestration path replaces it.
The current helpers do not complete any whole runner/publication step.

1. Expose validated lifecycle milestones to a fake-clock supervisor without
   weakening the parser's feed/finish contract. Test acceptance-to-first-token
   and whole-request deadlines, child exit, truncated streams, bounded output,
   TERM/KILL/reaping, partial evidence and the single infrastructure-invalid
   retry rule before wiring a real child.
2. Reuse `QualificationControl` for the descriptor-bound handshake and immutable
   manifest/sequence builders for resident-first, profile and prompt ordering.
   A matched JSON identity is not a running executable/model descriptor check.
3. Collect the exact references from schema-validated global/profile evidence;
   authenticate file observations through safe descriptors before calling the
   index builder. Then implement gate/status propagation and atomic,
   fsynced bundle/sidecar publication with tamper tests.
4. Only after host orchestration is complete, obtain separate current-revision
   CUDA parity and real DGX 8/12/16-GiB sweep evidence. Historical passes and
   diagnostic captures cannot waive this. Deployment remains a separate action.

The reusable pattern is **same boundary, exact provenance, first divergence,
small replay, controlled intervention**. Typed metadata and content hashes are
useful inputs to that workflow, not substitutes for observed and authenticated
runtime evidence.
