# Laguna upstream convergence amendment

**Status:** approved GB10/CUDA-only implementation amendment; canonical
qualification is blocked until the corrected artifact is promoted. Metal and
cross-backend checkpoint compatibility are outside this milestone.

**Date:** 2026-08-15

**Applies to:**
[2026-08-03-laguna-compact-cuda-streaming.md](2026-08-03-laguna-compact-cuda-streaming.md)
at local revision `fe971ff4fdf6087f7487a98b16758e83e0d7899b`.

## Decision

Keep the current branch as the compact-runtime correctness and qualification
candidate. Do not reset it onto, merge, or bulk cherry-pick antirez/ds4 PR
#594 before the corrected Q4 qualification. The branch contains 460 commits
after its resident-CUDA planning base and has working SSD streaming, accounting,
request/runtime evidence, lifecycle control, and canonical publication code
that PR #594 deliberately does not implement.

Stop the current Task 21 run. Its Poolside artifact revision is superseded.
No result produced from revision `706fa69799926b6afde1af9e24ca2a4923f110a1`
may be called canonical or published. First promote the corrected artifact,
rebuild every model-bound oracle and manifest, reconfirm the already-correct
CUDA KV-ring invariant, and rerun the existing qualification stack.

Before spending a maintenance window on the bounded-memory curve, establish an
efficient resident Q4 baseline. Treat exact PR #594 head
`7005761d1e4a53ff50c8e2b033d33c375fdb6297` as an immutable same-host
performance reference, not as a literal merge/rebase base. The current branch
must first pass the corrected Poolside oracle and then remain within the frozen
resident-performance margin at every context frontier. If it does not, adapt
only narrow, oracle-gated Q4 CUDA primitives before running SSD streaming.

After the truthful compact-runtime bundle exists, converge upstreamability in
a separate worktree. Prefer PR #594's eventual upstream-main merge commit as
the integration base; while it remains open, manually port the selected narrow
primitives onto a chosen current-upstream branch. Replay the compact layers
additively and keep the qualified current branch as the differential oracle.
Q2/Q3, DFlash, and mixed-precision target expansion remain separate follow-ups.

## Prior-art audit

As of 2026-08-15, antirez/ds4 has a real Laguna line, but none of the relevant
pull requests is merged:

| Upstream work | State | What it contributes | Disposition here |
| --- | --- | --- | --- |
| `laguna-s2.1` (`448d569`) | branch | Antirez's native model, Metal graph, protocol/template support, revised Q4 layout, and later Q2/Q3 work | Already the ancestry of the local resident work through `7e3dbef`; do not reimplement |
| [PR #594](https://github.com/antirez/ds4/pull/594), head `7005761` | open draft | Full-resident, one-GPU CUDA; mature Q4/Q2/Q3 kernels, Blackwell paths, DFlash, and GB10 measurements; explicitly no SSD streaming | Same-host resident performance reference after corrected Q4 oracle promotion; later upstream convergence source |
| [PR #613](https://github.com/antirez/ds4/pull/613), head `ceb4685` | open | Corrected official Poolside GGUF revision | Immediate blocking input |
| [PR #614](https://github.com/antirez/ds4/pull/614), head `b388b8c` | open | Metal oversized-prefill KV-ring correction, exact 1024/512 regression, checkpoint payload ABI v3 | No GB10 implementation work; local CUDA was race-free from its first commit |
| [PR #633](https://github.com/antirez/ds4/pull/633), head `4f4c724` | open | APEX IQ4_XS/Q6_K and official BF16 mixed-precision targets plus metadata-driven RoPE | Defer until corrected Q4 compact qualification passes |
| [PR #634](https://github.com/antirez/ds4/pull/634), head `fefbcb7` | closed, unmerged draft | Metal/MLXFast and DFlash experiments | No action |

The two CUDA implementations are separate development lines: neither contains
the other. Current local `HEAD` and PR #594 share merge base `0a7ad776`; the
two sides have 493 and 178 unique commits respectively. Current local `HEAD`
and `upstream/laguna-s2.1` share `7e3dbef`; the two sides have 492 and 15
unique commits. A wholesale merge would mix model work, months of unrelated
upstream changes, and the entire compact-runtime stack in the same conflict
resolution.

## Corrected artifact identity

The canonical Q4 target is now:

```text
repository = poolside/Laguna-S-2.1-GGUF
revision   = e2ccc0579fc18e6ea2362fa25fccbcd470f0e332
file       = laguna-s-2.1-Q4_K_M.gguf
size       = 68248760064
sha256     = a34c74e46688122bef83122f4133031bababbefcf57436dde97048c91e2cc6ff
```

It is 416 bytes larger than the currently pinned file. PR #633 reports
numerically identical weights and identifies the delta as GGUF metadata/chat
template repair. That is useful evidence, not permission to relabel the old
fixtures. Full-file identity, tensor offsets, embedded template metadata,
tokenized prompts, oracle provenance, served behavior, and every qualification
manifest remain artifact-bound and must be regenerated or explicitly proved
reusable. The delta is thirteen 32-byte GGUF alignment units, so tensor
types/shapes/counts and slot geometry are expected to remain stable while every
absolute source offset and descriptor identity changes; the comparison must
measure this rather than infer it.

## Current progress retained

The current branch is not redundant with upstream:

- resident CUDA admission and Poolside numerical oracle are implemented;
- compact tensor ledger, immutable allocation plan, and categorized runtime
  tracker are implemented;
- compact attachment avoids whole-model CUDA registration/copy;
- bounded `pread` plus H2D, deterministic fixed expert cache, streamed decode,
  grouped prefill, and exact 4096-row scratch are implemented;
- exact page disposal, `mincore`/smaps/NVML attribution, runtime/request
  snapshots, token admission, server lifecycle, and qualification-control fd
  protocol are implemented;
- Task 19 benchmark/eval evidence is implemented (`a916572` through
  `8ac4fc1`) and its eight host contract tests pass;
- Task 20 runner/publication is implemented (`203e474` through `12115b3`,
  with later hardening) and its 97 qualifier tests pass; and
- Task 21's pre-run guide exists (`b84ad97`), but there is no canonical DGX
  curve, bundle, sidecar, or outcome commit. Checkpoint F is open.

PR #594 is materially ahead in resident CUDA kernel maturity, mixed target
formats, Blackwell specialization, and DFlash. Its reported performance is not
directly comparable with the local measurements, but the gap is large enough
that a same-host, same-artifact A/B is mandatory before selecting the long-term
runtime. It does not replace the compact capacity mode or its verifier. Its
routed-MoE path is specifically not a drop-in donor: it resolves whole
contiguous 256-expert arrays and uses pre-local-oracle Q8_K/SwiGLU weighting
without the later Poolside column-L2 rescale and slot-descriptor semantics.
Importing it would bypass both compact lookup and the accepted arithmetic
repairs. The first credible isolated donor is its split-history decode
attention, and even that changes reduction order and must pass the promoted
Poolside oracle.

PR #614 adds no GB10-critical implementation work. Its CUDA-relevant invariant
is already present locally:
`laguna_commit_kv_f16_kernel()` skips staged rows older than the newest
`cache_cap`, and `tests/test_cuda_laguna_kernels.c` constructs the same expected
newest-row ring. That behavior existed from the local CUDA implementation's
first commit, so checkpoints created by this GB10 path do not carry the Metal
race that motivated ABI v3. Keep ABI v2 for this milestone; Metal repair,
cross-backend checkpoint import, and a shared ABI bump are explicitly deferred.

## Amended implementation order

Every code change below remains failing-test first. Keep commits focused and
keep the current qualified candidate recoverable throughout.

### Phase 0 — contain the invalid run

- [ ] Do not resume, verify, or publish evidence from the interrupted
  `706fa697…` run.
- [ ] Preserve its logs only as infrastructure-invalid diagnostics, clearly
  outside the canonical evidence directory.
- [ ] Before another GPU job, prove the old transient qualification unit and
  all descendants are gone, recover the DGX, and verify the production service
  with both model identity and a tiny generation request.
- [ ] Run only model-independent host/source tests while production owns the
  large Flash fixture. Do not invoke a generic `make test` whose linked model
  fixture can fault the production GGUF into unified memory.

### Phase 1 — prove and pin the corrected artifact

- [ ] Download `e2ccc057…` into a new immutable path. Do not overwrite or
  relabel the old file.
- [ ] Verify the exact size and SHA-256 above before DS4 opens it.
- [ ] Add a GGUF comparison test/tool that records metadata, tensor name/type/
  shape, offsets and lengths, plus a per-tensor payload digest for both files.
- [ ] Prove whether tensor payload bytes are identical. Offset equality is not
  assumed because the metadata prefix changed. If any tensor payload differs,
  invalidate and recapture every low-level model-derived fixture.
- [ ] Add RED pin tests for the corrected revision/size/SHA and explicit
  rejection of `706fa697…`, then update downloader, C planning/streaming
  constants, schemas, qualifier, gates, producer contracts, and documentation.
  Do not mechanically rewrite generated fixture provenance; retain old
  captures/manifests as explicitly superseded history.
- [ ] Because no canonical compact-runtime v1 bundle has been published,
  correct the v1 artifact pin in place. If an external v1 is discovered before
  this change lands, preserve it and issue v2 instead of silently changing v1.

### Phase 2 — re-promote the Poolside oracle

- [ ] Capture the corrected embedded chat template and its thinking/tool-call
  semantics; reconcile it with DS4's explicit native think/nothink rendering.
- [ ] Re-run the Poolside capture and tokenizer parity for short, SWA-513,
  YaRN-8193, deep-32768, continuation, terminal, batched, and mixed cases.
- [ ] Promote a new manifest through the existing strict producer/verifier
  path. Record new model, runtime, tokenizer, prompt, and capture identities.
- [ ] Regenerate the tensor ledger, bootstrap and per-profile qualification
  plans and digests, safe full-page cold-preparation union, and every opened-fd
  descriptor binding. A three-constant pin swap is not sufficient.
- [ ] Re-run resident CUDA parity before admitting streamed comparison.
- [ ] Reuse a low-level fixture only when the Phase 1 tensor-payload evidence
  proves its complete source range identical and its provenance records both
  artifacts. Regenerate behavior/eval fixtures unconditionally.

### Phase 3 — reconfirm the GB10 CUDA-ring invariant

- [ ] Run the existing CUDA oversized/multi-wrap prefill contract against the
  corrected artifact. Strengthen it only if the upstream 1024-token/512-row
  case exposes an untested edge.
- [ ] Keep checkpoint payload ABI v2. Do not port Metal code or invalidate
  checkpoints produced by the already-correct GB10 path.

### Phase 4 — establish the efficient resident-Q4 gate

- [ ] In isolated clean exports, build exact PR #594 head and the current
  candidate against the same corrected Q4 file. Bind the two binary hashes,
  exact source revisions, model identity, host identity, and pre-child GPU
  inventory in a no-clobber evidence directory.
- [ ] Immediately before timing, require the current candidate to pass the
  corrected resident CUDA oracle. Correctness is an admission prerequisite,
  never a trade for speed.
- [ ] Run the implementations serially through the stable normal
  `ds4-bench` CSV interface: fresh resident-CUDA process per implementation,
  the pinned `speed-bench/promessi_sposi.txt` prompt, 32K allocation, 4096-row
  prefill chunks, 256 generated tokens, and exact 2048, 4096, 8192, 16384, and
  28672-token frontiers. Do not use compact/streaming flags in this gate.
- [ ] Reject the reference run unless its 2K/4K/8K prefill and steady-decode
  values are at least 80% of PR #594's recorded Q4 numbers. Require candidate
  prefill and steady decode to reach at least 90% of the same-run reference at
  every frontier; a mean or median must not hide one red context length.
- [ ] If the unchanged candidate fails, stop before compact measurement. First
  adapt PR #594's split-history decode attention (`bdf47ff`, `021ca4e`,
  `8984bcc`) behind a rollback switch and rerun the corrected oracle plus the
  entire resident gate. Consider a Poolside-preserving equivalent of its fused
  Q8 projections only if profiling still identifies launch overhead. Do not
  import its contiguous-expert routed-MoE path.

### Phase 5 — rerun Tasks 19 and 20 on corrected provenance

- [ ] Run model-independent source, schema, qualifier, and fake-child suites
  before the maintenance window.
- [ ] Export a clean revision to an isolated DGX directory whose default-model
  fixture cannot resolve to the production Flash GGUF.
- [ ] During one explicit maintenance window, stop production, run the focused
  corrected Laguna resident and streamed gates, then run the immutable Task 20
  schedule. Only one model-bearing process may run at a time.
- [ ] Before freezing the long curve, run a short corrected-artifact
  28K-context streamed steady-decode preflight against the already-admitted
  efficient resident candidate. The existing 0.5 visible tok/s threshold is a
  compact-mode viability floor, not a resident-efficiency target. If it misses,
  stop before the 12 streamed slices and profile the streaming/cache path; do
  not weaken the resident gate or conflate the failure with core CUDA speed.
- [ ] Restore production in an idempotent cleanup path and verify identity,
  generation liveness, peer inventory, and kernel logs before accepting any
  result.

### Phase 6 — complete Task 21

- [ ] Freeze a new benchmark manifest only after the corrected oracle and
  clean CUDA binaries pass.
- [ ] Run the resident baseline and all mandatory 8/12/16-GiB streamed slices
  without tuning, reordering, or changing retry policy.
- [ ] Verify the exact evidence union, publish the canonical bundle and
  sidecar, and record the reproducible outcome.
- [ ] Checkpoint F closes only if all global gates and at least one profile
  pass on `e2ccc057…`.

### Phase 7 — converge upstream after the compact proof

- [ ] If PR #594 has merged, create the upstreaming branch from that merge
  commit. If it remains open, create the branch from the selected current
  upstream main and manually port its narrow Q4 primitives using `7005761` as
  the reference; do not transplant its 178-commit side wholesale. Replay the
  local work in logical stacks: corrected oracle/admission; compact
  ledger/runtime; compact attachment/cache/I/O; streamed graph;
  evidence/protocol/lifecycle; qualification tooling.
- [ ] Start with independently useful primitives such as split-history decode
  attention. Do not import PR #594's routed-MoE implementation: rework any
  later MoE optimization around the local Poolside-exact Q8_1, column-L2
  rescale, post-down weighting/residual association, and slot descriptors.
  Every fast kernel must consume exact resident tensors or pinned cache-slot
  spans; it must never recover whole-map registration or bypass request/cache
  accounting.
- [ ] Differentially test old local resident, upstream resident, converged
  resident, and converged streamed paths on the corrected artifact. Preserve
  oracle ceilings and the canonical capacity bounds before using performance
  to choose a winner.
- [ ] Run `git range-diff` over each replayed stack and keep the current branch
  until the converged branch reproduces its bundle and improves or preserves
  the controlled performance baseline.

### Deferred expansion

- [ ] All Metal Laguna changes, cross-backend checkpoint import, and ABI v3.
- [ ] Non-GB10 CUDA portability hardening, including `ac1a187`'s PTX-version
  guard for lower-virtual-architecture builds.
- [ ] PR #594/#633 Q2/Q3, IQ4_XS/Q6_K, BF16, and DFlash support.
- [ ] Multi-GPU, tensor parallel, ROCm, and distributed Laguna execution.
- [ ] Deployment, port assignment, peer eviction, or co-residency claims.

## Acceptance boundary

This amendment is complete when the corrected-Q4 candidate first clears the
same-host resident performance gate, then produces a canonical compact bundle,
and finally the separate upstream-convergence branch reproduces its correctness
and capacity gates. This order prevents bounded-memory machinery from masking
an inefficient core implementation while preserving streaming as an additive
capacity mode.
