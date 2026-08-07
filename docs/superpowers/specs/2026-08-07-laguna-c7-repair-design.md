# Laguna C7 Repair Checkpoint Design

**Status:** Approved by the operator on 2026-08-07 through the instruction to
execute the adversarial review's recommended repair checkpoint.

**Scope:** Repair the confirmed Tasks 0--7 safety and evidence gaps before Task
8 adds asynchronous SSD reads and CUDA cache ownership. This checkpoint does
not implement streamed decode, grouped prefill, the 4K graph allocation, page
disposal, server lifecycle, or deployment.

## Objective

Make checkpoint B fail closed and evidence-bearing:

- no cache teardown may invalidate an active load owner;
- no legacy CUDA registration, copy, arena, or selected-expert cache path may
  run while a compact context is creating, active, or destroying;
- every already-live compact engine/model/vocabulary allocation must be
  represented in the bounded runtime tracker before compact CUDA allocation;
- the compact context must retain the opened model descriptor and reject a
  mapping that is not backed by the same file identity;
- the checked JSON Schema and strict Python validator must reject the same
  schema-expressible invalid manifests; and
- the resident and compact DGX gates must hash the actual opened test artifact
  and run both synthetic and pinned-model startup.

## Approaches considered

### 1. Guard individual callers

Add checks only to the legacy entry points identified by the adversarial
review. This is the smallest textual diff, but it remains vulnerable whenever
another legacy allocator or alias mapping is added. Rejected because compact
mode's invariant is process-wide.

### 2. Fail closed at shared lifecycle and evidence boundaries (selected)

Treat any non-idle compact lifecycle state as exclusive for all legacy model
placement APIs. Keep compact resolution as the sole exception for the exact
context mapping. Refuse drain while either `LOADING` or `IN_USE` exists. Bind
and retain model identity once, replay the exact still-live pre-plan allocation
inventory, and make schema/artifact gates validate their real inputs. This is
narrow, testable, and composes with Tasks 8--11.

### 3. Split compact CUDA into a separate backend/process

A separate CUDA backend or helper process would eliminate most global-state
sharing. It is a much larger architectural change, duplicates current CUDA
kernels, and is not required by the approved design. Deferred unless later
usage proves the singleton boundary unmaintainable.

## Runtime design

### Cache drain

`ds4_laguna_cache_policy_drain()` is a non-destructive readiness check followed
by clearing reusable state. It first audits every slot. If any slot is
`LOADING` or `IN_USE`, it returns `RECOVERABLE` without changing slots, reverse
maps, generations, or counters. Only `READY` slots may be cleared by a
successful drain. A regression holds a `LOAD_OWNER` handle across drain and
requires byte-identical policy state.

### Compact CUDA exclusivity

The compact lifecycle predicate has two separate questions:

1. Is legacy placement allowed? Only when the process state is `IDLE`.
2. May this lookup resolve from compact storage? Only when state is `ACTIVE`
   and the mapping is the context's exact mapping.

All whole-map, range/span, arena, q8, warm-cache, and selected-expert legacy
entry points use the first predicate. An alternate mapping of the same inode is
not an escape hatch. Rejected attempts increment the existing violation/test
counters without allocating or copying bytes.

### Model descriptor and mapping identity

The compact context duplicates the opened descriptor with close-on-exec and
owns that duplicate until successful destruction. It requires `fstat` to show
a regular file whose exact size and device/inode match `model_size`, the ledger,
and the engine's captured identity. On Linux it validates that the complete
supplied address interval is covered by readable file-backed VMAs whose
device/inode and file offsets match the duplicated descriptor. Creation rejects
anonymous, `PROT_NONE`, mixed-inode, post-map-truncated, or offset-shifted
mappings before CUDA allocation. The engine mapping remains owned by the
engine; the context-owned descriptor makes later `pread` lifetime independent
of caller fd reuse.

### Allocation tracking

The engine can inventory every pre-plan allocation that remains live: the
engine object, model metadata table, model tensor table, vocabulary token
array, and the two vocabulary hash-table arrays. Once the plan initializes the
runtime tracker, DS4 replays this exact live inventory before compact CUDA
creation and routes subsequent compact allocations directly through the
tracker. Temporary parsing allocations that are already freed are not physical
members of the simultaneous compact footprint and need no historical record.

The file mapping remains a non-additive model-mapping record. Replay fails on
an unclassified call site, overlap, capacity, or bound mismatch. Released
tracker records are reusable tombstones so bounded allocation churn cannot
exhaust the record table merely because historical events occurred; monotonic
event counters remain separate from live attribution records.

## Qualification and provenance design

### Frozen manifest schema

Qualification pins a real Draft 2020-12 validator. A shared mutation corpus is
run through both the checked schema and `validate_manifest()` and requires
equivalent accept/reject results for every schema-expressible constraint.
Canonical uint64 strings are bounded at `18446744073709551615`; base64 uses a
canonical-bit/padding pattern; model paths are absolute and end in the pinned
filename. JSON lexical distinctions that disappear after parsing are enforced
by the strict duplicate-preserving input layer and explicitly documented as
outside JSON Schema's value model.

### Opened artifact binding

The C7 runner opens the actual model once and retains that descriptor across
all verifier and test child processes. The verifier hashes the inherited
descriptor with before/after `fstat`. Resident and compact tests use a hidden
test-only engine option that duplicates and maps that same inherited descriptor;
they never reopen the path for execution. The runner verifies the retained
descriptor identity after each child and rehashes it after the final child, so
a path swap cannot separate verified bytes from executed bytes.

The qualification-plan writer records an observed digest produced from its own
retained opened descriptor, not a literal labelled as observed evidence. A
hash mismatch or identity change fails before compact CUDA allocation. The
general Task 16 qualification-control fd protocol remains future work; this
checkpoint pulls forward only the test runner's descriptor handoff needed to
make checkpoint B truthful.

### Required C7 DGX gate

Add a named `test-cuda-laguna-c7` target. It fails if the pinned model path or
capture-time tokenizer revision is absent, then runs in order:

1. promoted fixture verification bound to the actual model bytes;
2. resident Laguna kernels and model oracle;
3. synthetic compact startup; and
4. pinned-model compact startup.

The general host test target remains model-independent. `cuda-regression`
continues to be the lightweight CUDA suite; checkpoint B is claimed only from
the explicit C7 target.

## Validation hardening

Qualification-plan validation additionally proves each expert view uses the
correct parent for its layer/projection and the exact offset
`parent_offset + expert * projection_expert_bytes`. This closes the reviewed
machine-evidence mutation without changing the valid ledger builder.

## Error handling and teardown

- Unsafe compact state never falls back to resident execution.
- A failed descriptor duplicate, VMA identity check, artifact hash, inventory
  replay, or tracker reconciliation aborts before the first compact CUDA
  allocation.
- Compact destruction returns a typed `OK`, `RECOVERABLE`, or `UNSAFE` status.
  `ACTIVE` transitions to `DESTROYING` before synchronization. A synchronization
  or injected pre-commit failure releases nothing, remains `DESTROYING`, and is
  `RECOVERABLE` for an explicit retry. Successful synchronization is the
  teardown commit point: state becomes `RELEASING`, and no inference or retry
  may begin. CUDA/host/tracker/mapping records are then released in dependency
  order and the duplicated descriptor is closed last. Any post-commit release
  failure is `UNSAFE`, leaves the process fail-closed, and retains engine-owned
  model/ledger memory until process exit; it is never described as recoverable.
  Only complete release transitions to `IDLE` and returns `OK`.
- Concurrent lookup/use versus engine destruction remains governed by the
  later server draining lifecycle in Task 18; the C7 gate does not claim that
  serving lifecycle is complete.

## Test strategy

Every production change begins with a focused failing regression:

- `LOADING` drain preservation;
- alternate-map and legacy selected-cache rejection during compact activity;
- mismatched fd/mmap, `PROT_NONE`, post-map truncation, and caller-fd reuse;
- injected pre-commit synchronization failure that returns `RECOVERABLE`,
  preserves byte-identical ownership in `DESTROYING`, and succeeds on retry;
- injected post-commit release failure that returns `UNSAFE`, remains
  non-`IDLE`, rejects retry and every legacy placement path, and retains the
  engine-owned model/ledger state without a second release attempt;
- live pre-plan allocation reconciliation and tracker tombstone reuse;
- expert-offset mutation rejection;
- real-schema mutation equivalence; and
- inherited-descriptor hash mismatch, path swap immunity, and missing-input
  failure in the C7 Make target.

Host verification runs policy, plan, runtime, Python, CLI, build-contract,
ASan, and UBSan suites. DGX verification builds an archive of the exact
revision and runs synthetic startup. The pinned-model C7 target runs only in a
controlled window with peer inference services intentionally stopped; the
repair does not authorize DS4 to manage those services.

## Acceptance boundary

This checkpoint is accepted when all new RED tests have been observed failing,
then pass with the implementation; all prior Tasks 0--7 tests remain green;
both recoverable and unsafe teardown branches satisfy their ownership/state
assertions; and the named C7 DGX target passes on the pinned artifact. It still
does not make compact Laguna inference usable: Tasks 8--11 remain responsible
for fixed cache I/O, streamed execution, grouping, and the 4K scratch shape.
