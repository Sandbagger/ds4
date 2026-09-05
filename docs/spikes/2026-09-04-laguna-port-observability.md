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

## Control-FD host checkpoint (2026-09-05)

The fixed control session now reuses the owned process transport. It pumps
stdout/stderr and milestone deadlines during model-descriptor and checkpoint
waits, freezes callback observations, preserves raw bounded wire messages and
partial checkpoints, and rejects trailing bytes/rights before accepting EOF.
Four native tests use tiny files and generated foreground children. This is
not executable/runtime authentication, full runner acceptance or a GPU run.

Two fixture rules came from actual failures in this increment:

- An import-error RED proves the entry point is absent, not that unexecuted
  test bodies are correct. Check callback forwarding and the real wire/record
  order before implementing against a new scaffold.
- Publish owned-child metadata by same-directory write/replace. TERM can land
  during a metadata refresh. Assert wire transfer from captured wire bytes,
  not from a later child metadata write that cleanup may prevent.

`QualificationControl.wire_records[*].complete` means a complete frame was
transferred. It does not mean the frame was accepted, authenticated or useful
as qualification evidence. The 64-KiB lifetime cap is checked before sending
an ACK that could release the child beyond the retained evidence budget.

## Immutable admission host increment

Descriptor owners now enforce nofollow regular-file opens, initial-size hash
bounds, pre-hash JSON byte caps, retained path/inode checks and change-time
guards. A real regression showed that a same-size rewrite followed by restored
mtime bypasses the four exported stat fields; change-time remains an internal
owner guard, not a new bundle field. Input admission freezes exact schema,
manifest and version-response bytes and rechecks held inputs before/after each
trusted version callback and at normal context exit. Native version origin,
full orchestration, gates/retries and publication still need integration.

Negative-test rules from this increment:

- Do not put `self.fail()` inside `assertRaises(Exception)`: it catches its own
  AssertionError and can pass when the invalid operation succeeds.
- Require evidence that an expected failure path actually ran. An empty list
  of observed callback/descriptor events does not prove their cleanup.
- Keep fixture patches active while the implementation runs, not just while
  constructing input. Mutate a test file through a separate writable handle;
  a failed write to the read-only owner is not a real growth race.

## Descriptor-only version collection

The admission callback now has a Linux-only owned implementation in
`qualification_version_probe.py`. It returns raw bytes; admission still owns
JSON, clean-CUDA and revision validation. The lifecycle and raw-version paths
share one private child owner with separate monitors and byte caps.

Two real interrupt regressions shaped the owner boundary: cleanup can release
all resources yet propagate an interrupt before its return value is assigned,
and an admitted child can produce evidence before a context yields its owner.
Keep the owner before entering; then record actual reaping, group absence and
closed-stream proof in the cleanup exit path. Never infer release just because
a cleanup block ran. Preserve the original interrupt and capped raw prefixes.

Darwin returned EACCES for `/dev/fd` execution of both a shebang program and the
native project interpreter. Production therefore fails closed outside Linux.
The host-test-only adapter reads the retained Python fake's FD; it validates
supervision and FD handoff, not kernel-native executable origin. Adapters must
assert the **entire requested command**, not synthesize a correct flag while
silently ignoring a wrong production argument. Native Linux and real-model
qualification remain separate evidence requirements.

## Authenticated control-boundary composition

`qualification_authenticated.py` checks retained input owners, the live owned
PID's executable, and the received model FD around each preparation/snapshot
callback. A rejection must happen before its ACK releases the child. The
controller still owns socket/FD cleanup and raw observations; authentication
does not replace its earliest transport failure with a qualification verdict.

The host fixtures use a simulated proc tree whose allowed PIDs come only from
actual Popen returns. They must not manufacture an acceptable proc leaf for any
PID supplied by production. Capture the executable FD integer before a failing
check invalidates its owner; assertions must not reopen a failed owner. Close
an FD-directory iterator before counting inherited descriptors, or the fixture
counts its own scan. To prove before-and-after callback checks, require both
sides between successive callbacks, not merely one intervening authentication.

The first native run also caught fixture-only errors: a 64-KiB stderr burst plus
later milestone diagnostics correctly exceeded the existing 64-KiB cap, and
an eight-byte read could not equal the nine-byte word `preflight`. The tests
were corrected; production byte caps were not weakened.

## Resident and streamed sequence separation

The trusted entrypoint, not an authenticated file's contents, chooses the
sequence kind. Matching hashes authenticate bytes; they do not authorize a
resident file at a streamed boundary or the reverse. Tests must authenticate
the mutated bytes again when checking schema/profile/cache/mode rejection,
otherwise an early digest mismatch can hide missing semantic validation.

The two C boundaries share file/size/hash/base64/cleanup code. The two Python
builders share the fixed formatter, but retain separate public profile and
prompt-order selection. A separate resident type and schema avoid adding an
ambiguous mutable mode to existing streamed callers. This supplies no resident
allocation or runtime evidence by itself.

## Independent rejection fixtures and resident raw evidence

Reset each rejection case from a known-valid record, including nested snapshot,
sequence, metrics, and pointers. Reusing scratch state can leave an earlier
invalid request ID or missing metrics in place, making later hash, geometry or
cache tests pass without testing their intended rule. Mutate independent
configuration fields separately. A multi-field cross-mode rejection does not
prove each discriminator.

The resident emitter reuses runtime serialization and copies observed physical
footprint even when the expert-cache limit is zero. A zero streaming cache does
not mean zero resident CUDA allocations or zero registered model pages. Native
allocation collection and parent-side lifecycle/schema validation are not
supplied by the emitter alone.

## Resident record consumption and rejection boundaries

The resident consumer shares private JSONL framing/lifecycle mechanics with the
streamed consumer, not mutable public mode selection. Fixed public boundaries
select their own schema and profile rules. Never temporarily swap the streamed
module's schema paths, profile maps or limits to parse resident evidence.

The host fixture feeds actual native-emitted bytes for four complete resident
slices. Snapshot gaps matter: accepted, first token, request metrics and complete
runtime snapshots occupy four successive sequence values, but only three are
lifecycle records. Four sets of independent raw emissions do not prove that
complete twelve-record slices are consumable.

A malformed single-line input test must assert rejection during `feed()`, not
merely accept any eventual error from `finish()`. An incomplete stream will fail
finish even if the bad line or ignored byte cap was accepted. Establish valid
baseline bytes first, isolate each mutant and each failed parser instance, and
prove depth rejection precedes JSON decoding. Draining a valid prefix does not
make a failed stream recoverable or a completed qualification run.

This is raw host-side consumption, not native allocation/snapshot production,
authenticated record-schema admission, resident process supervision, a baseline
execution, or a qualification verdict. Those integration boundaries remain open.

## Reuse measured model-page accounting

The source-page sampler is now shared by the native runtime and the compact
CUDA caller. Keep its allocation-free bounded batches and full final-page
charge. Require the actual system page size and validate the complete pointer
interval before a syscall; power-of-two/alignment checks alone do not prevent
undercounting with a wrong page size or wrapping the last span. Publish the
caller output only after all batches succeed.

Separate physical host evidence from simulated syscall evidence. A touched
tiny mapping plus independent `mincore` proves actual host page accounting;
interposed large spans prove batch geometry and error handling without a huge
mapping. Neither proves a native resident CUDA allocation inventory. Fake
syscalls must reject oversized vectors before writing them, and setup errors
must fail rather than silently skip the meaningful test.

Resident memory bounds still require the actual registration/cache/graph paths.
Do not reuse the compact plan's hard-coded external envelopes or report the
resident 512-row execution/16384-row allocation as a configured 4096-row run.

## Preserve registration ownership through failed pointer lookup

Acquiring a host registration and obtaining its device pointer are separate
operations. Own the registration immediately after acquisition. If pointer
lookup fails or returns null, prove rollback before reporting the unregistered
fallback. Failed rollback retains the host base/size and ownership flag, with no
usable device-pointer claim. An identical-map fast path must not convert that
unresolved state to success. Before replacing an old registration, a failed
release must preserve its metadata and prevent cache teardown/new registration.

The no-copy function's host fixture extracts and compiles the real C++ function,
not a reimplementation of its algorithm. Keep a separate fake registry that can
hold multiple live mappings, validate CUDA-call arguments, and prove accepted
controls before failure cases. Test both preservation and later recovery from
pointer-lookup rollback failure. Register temporary-directory cleanup before
compilation so a setup error cannot leak fixture files.

This is a narrow transaction contract, not proof that every legacy CUDA free or
unregister path preserves ownership. In particular, generic model/range cleanup
still uses best-effort release. Resident bounds also cannot use the range-cache
payload scalar as arena reservation: the normal arena chunk is 1792 MiB, the
copy path uses four 64-MiB staging buffers plus alignment, and optional Q8 caches
consume additional memory. Those paths still need real inventory/bounds before
native resident runtime evidence is available.
