# Laguna Compact CUDA SSD-Streaming Runtime Design

**Date:** 2026-08-02
**Status:** Specification review approved; operator review pending
**DS4 baseline:** `b257bb6fde1c6ba6d8c541bc76952fbd1fb83ae7`
**Prerequisite:** [Laguna Single-Poolside Oracle Design](2026-08-01-laguna-single-poolside-oracle-design.md)

## Decision

Implement a numerically correct, hard-bounded CUDA SSD-streaming runtime for
Poolside Laguna S 2.1. The primary DS4 objective is to reduce Laguna's runtime
footprint by keeping non-routed tensors resident and loading routed experts
through a bounded cache instead of registering or copying the complete GGUF.

DS4 owns the compact inference process. It reports exact configuration,
allocation, cache, I/O, protocol, and benchmark evidence so a downstream
deployment can decide whether the achieved footprint is useful.

Making Laguna eligible for a multi-model deployment motivates this work, but
is not a DS4 acceptance claim. DS4 does not know which other models exist and
does not start, stop, protect, route, or compare them. It neither manages nor
claims co-residency.

## Success boundary

DS4 answers:

> Can this pinned Laguna artifact run correctly on CUDA through a genuinely
> streamed, bounded data path, and what footprint/performance curve does that
> path achieve?

A downstream deployment answers:

> Is one of those qualified profiles small and useful enough for this host and
> its other workloads?

An upstream success may therefore be rejected downstream. Conversely, a
downstream co-residency observation cannot waive an upstream numerical or
allocation-bound failure.

## Scope

### In scope

- the pinned revised Poolside Laguna Q4_K_M GGUF on one CUDA device;
- routed/non-routed tensor classification and exact range accounting;
- one-time non-routed tensor placement in owned device allocations;
- bounded SSD reads, pinned staging, and an engine-lifetime expert cache;
- configurable total context, prefill chunk, session slots, and expert-cache
  byte ceiling without silent clamping or growth;
- safe per-file/per-range page-cache advice after completed uploads;
- Laguna chat, reasoning, tagged-tool, continuation, and server semantics;
- versioned build/model/configuration and runtime telemetry;
- a machine-verifiable compact-runtime qualification bundle;
- resident-versus-streamed correctness and footprint comparison; and
- generic DS4 benchmark and capability-regression evidence.

### Out of scope

- fixed ports, endpoint slots, aliases, or planner roles;
- systemd units, wrappers, cgroups, restart or OOM policy;
- leases, pins, fencing, prewarming, timers, or resident-set transitions;
- knowledge of Qwen, Flash, embedders, rerankers, ASR, or any peer process;
- host-wide co-residency, swap, PSI, or peer-latency acceptance thresholds;
- choosing a deployment's cache size or prompt/output reserve policy;
- planning evals, worktrees, executors, operator intervention, or promotion;
- dotfiles traces, bus events, infrastructure docs, or model registries;
- Metal implementation changes, tensor parallelism, DFlash, MTP, or
  speculative decoding; and
- claiming that a qualified process fits a particular ensemble.

## Pinned artifact and oracle prerequisite

```text
repository: poolside/Laguna-S-2.1-GGUF
revision:   706fa69799926b6afde1af9e24ca2a4923f110a1
file:       laguna-s-2.1-Q4_K_M.gguf
size:       68248759648 bytes
sha256:     e163b2c98908809a71245d6bb68b2226994d9969cb2a438eccb72196a1c4147a
```

The approved `single-poolside-v1` resident-CUDA oracle contract is a hard
prerequisite. The inspected DS4 baseline still has the superseded fixture
shape, and Laguna SSD streaming remains rejected. A current green target must
not be reinterpreted as satisfying the approved design.

This design supersedes one ownership detail in that prerequisite: capture
isolation is an external laboratory precondition. The deployment/operator
harness supplies an isolation attestation and owns any peer-service
stop/restore action. DS4 captures or verifies model evidence but never manages
another service.

Resident and streamed CUDA must each pass the same promoted Poolside evidence:

```text
cases                         = short, swa-513, yarn-8193, deep-32768
centered RMS                  <= 0.04
centered max absolute error   <= 0.20
top-20 overlap                >= 18
argmax                        == Poolside argmax
teacher-forced continuation  == all 8 promoted token IDs
```

Streaming receives no wider numerical tolerance.

## Reference qualification profile

The first qualification curve uses the following parameters:

```text
backend                    = CUDA / one GB10 device
allocated context          = 32768 total tokens
session slots              = 1
prefill chunk              = 4096 tokens
model layout               = official revised Q4_K_M
expert-cache sweep         = 8 GiB, 12 GiB, 16 GiB
benchmark prompt lengths   = 512, 2048, 8192, 28672 tokens
```

These are reproducible benchmark inputs, not deployment policy. DS4 accepts an
explicit cache ceiling in bytes and reports the effective value. It does not
select a winner from the curve. A downstream consumer may select one qualified
profile or reject all of them.

The 32K value is a total sequence allocation. Ordinary inference rejects any
request whose exact templated input plus requested output cannot fit; it never
silently truncates. The `deep-32768` oracle uses the terminal-logits API and is
not a server/decode claim.

### Immutable reference-run manifest

Before qualification output is visible, the harness writes and hashes
`compact-runtime-benchmark-v1.json`. The checked-in schema rejects unknown or
placeholder values and requires:

- the exact rendered prompt bytes and SHA-256 for 512, 2,048, 8,192, and 28,672
  native-template tokens;
- maximum generated tokens `512`, temperature `0`, seed `1`, all remaining
  sampling fields, stop sequences, and tokenizer/template revision;
- cache ceilings in the fixed order 8, 12, then 16 GiB and a fixed
  counterbalanced prompt-length order per profile;
- for every `(profile, prompt length)`, qualification-only file-cold
  preparation, then a new process run followed by three consecutive
  same-process warm repetitions;
- a 45-minute whole-request timeout and 15-minute first-token timeout measured
  from request acceptance, with medians taken over exactly the three warm
  repetitions;
- host/kernel, CUDA driver/runtime, GPU UUID, filesystem/mount, NVMe identity,
  and direct-I/O/advice mode; and
- the four `ds4-eval` cases `recNu3MXkvWUzHZr9`,
  `001b51d76b4d422988f2c11f104a2c6c`, `aime2025-01`, and `compsec-076`.

Before launching the fresh process, the qualification harness opens the pinned
model inode, applies qualification-only cold-preparation advice over every safe
full page in the model tensor ranges, and obtains a synchronized exact-inode
`mincore` measurement at or below the source-residency bound. Global
`drop_caches` is forbidden. The four-case capability smoke passes when all
requests terminate normally and the streamed answer/grade vector exactly
matches the like-for-like resident answer/grade vector; answering all four
correctly is not a release gate.

Every required 8/12/16-GiB profile appears in the bundle with
`passed`, `failed`, or `invalid`. One evidenced infrastructure-invalid run may
be repeated once without changing the manifest. A second invalid run remains
`invalid`; a valid threshold failure remains `failed` and is never retried as
noise. The bundle is usable when all global gates pass and at least one profile
is `passed`. Downstream consumers may select only a passed profile.

## Streaming data path

At startup DS4 builds one immutable tensor-range table from the validated GGUF
layout. Every tensor is classified as one of:

- non-routed and copied once to an owned device allocation;
- routed expert payload addressed by exact file range; or
- metadata/address state with a separately reported allocation.

The CUDA SSD path never registers, pins, or copies the full 68-GB mapping.
Routed expert payload is absent from resident device allocations at startup.
It is read by exact GGUF range into bounded pinned staging and then into fixed
cache slots. Any legacy whole-model registration or copy option combined with
Laguna streaming fails before model allocation.

The runtime fails closed on invalid, overlapping, truncated, or misclassified
ranges. It must not fall back from the compact path to a resident path after
startup or allocation failure.

## Expert-cache contract

The cache lives for the engine lifetime and is keyed by
`(layer_id, expert_id)`. One entry contains exactly the quantized gate, up, and
down projection payloads for that expert. At startup, the tensor-range ledger
derives every entry's source bytes, aligned payload bytes, and the maximum
aligned entry size. Fixed slots use that maximum as their stride:

```text
slot_count          = floor(configured_cache_bytes / slot_stride_bytes)
cache_payload_bound = slot_count * slot_stride_bytes
cache_payload_bound <= configured_cache_bytes
```

An entry smaller than the stride leaves padding inside its slot; padding is
charged to cache payload rather than hidden elsewhere. Cache metadata, address
tables, staging, and allocator overhead are bounded and reported separately.

Eviction follows an explicit deterministic form of the existing DS4 Metal
policy. `route_hotness` is a saturating unsigned 64-bit selection count,
incremented once for each expert selected by a successfully admitted routing
decision. `last_used` is a process-monotonic sequence number. Evict by:

1. lowest route hotness first;
2. oldest `last_used` as the tie-breaker; and
3. lowest `(layer_id, expert_id)` as the final tie-breaker.

An entry referenced by in-flight CUDA work or the current expert group is
never an eviction candidate.

If one batched layer selects more unique experts than fit, DS4 processes
deterministic expert groups through the fixed slots and synchronizes before
reuse. It never creates an overflow routed allocation. Startup fails when the
configured cache cannot hold one token's maximum selected-expert set for a
layer.

Each slot has an explicit `EMPTY`, `LOADING`, `READY`, or `IN_USE` state plus a
generation and in-flight reference count. A key is published only after the
exact range read completes and its host-to-device completion event succeeds.
Only `READY` can become a hit; `IN_USE` cannot be selected for eviction.

`EINTR` retries the same offset, while EOF/short read, other I/O failure, CUDA
copy/event failure, request cancellation, or session teardown drains or waits
for submitted CUDA work, removes the unpublished key, releases every pin, and
returns the slot to `EMPTY`. A recoverable load failure returns a request-local
server error after restoring cache invariants. Failure to restore those
invariants is unsafe and terminates the process. Neither path falls back to a
resident/full-map allocation or leaks usable slot capacity.

Required invariants:

- `cache_payload_current <= cache_payload_peak <= effective_cache_limit`;
- `effective_cache_limit <= configured_cache_limit`;
- a hit reuses the same engine-lifetime entry across token steps;
- no routed payload allocation exists outside declared cache/staging categories;
- pressure changes I/O, timing, and eviction counters, never accepted logits;
  and
- serialized, batched, and mixed-session execution stays inside the promoted
  numerical ceilings.

Fault-injection tests cover short reads, `EINTR`, hard I/O errors, CUDA
copy/event failure, cancellation in every slot state, and teardown with
in-flight work. In addition to the one-session reference curve, a 4K-context,
two-session pressure profile forces interleaved misses, eviction, cancellation,
and batch working sets larger than the slot count. It must preserve the same
state, allocation, and numerical invariants; it is not a performance gate.

## Allocation and context contract

Before model allocation, qualification mode emits and hashes an immutable
allocation plan. It derives disjoint bounds for:

- static/non-routed weights;
- expert-cache payload;
- cache metadata and address tables;
- KV state;
- CUDA graph and scratch;
- pinned staging;
- other owned host buffers; and
- other owned CUDA buffers.

Static bounds are the aligned sizes of the classified tensor ledger. Cache and
staging bounds follow the fixed-slot formula and maximum in-flight load count.
KV and graph/scratch bounds are computed from exact context, session, model,
and prefill dimensions. Metadata and both `other` categories name every
allocation call site and a numeric bound; an unclassified call site fails
qualification.

Every model-lifetime host/CUDA allocation in the compact path goes through the
tracked allocator and is assigned one physical-domain category.
`cudaMallocManaged` is charged once, at its full requested size, to
`other_cuda`; its host-visible VMA is not charged again. `cudaHostAlloc` is
charged to its host category. `cudaHostRegister` reports registration bytes but
does not add a second allocation charge: its underlying host allocation or
file-backed resident pages already carry the physical charge. In streamed
mode, `model_mapping_registered_bytes` is zero.

The GGUF virtual mapping is reported as `model_mapped_virtual_bytes` but has no
physical charge. Its exact-inode resident bytes are a separate additive
category, `model_source_resident_bytes`, explicitly excluded from
`owned_total_current` before it is added once by the qualification formula. The streamed profile's 2-GiB
source-residency allowance is inside, not additive to, the 24/28/32-GiB profile
total. A resident baseline uses the same rule: registered-model bytes are
non-additive metadata and the corresponding resident file pages are charged
once as model-source residency.

After every allocation, free, register, and unregister event, DS4 computes:

```text
owned_total_current = sum(category_current at this event)
owned_total_peak    = max(previous_peak, owned_total_current)
```

This simultaneous peak is not the sum of independently observed category
peaks. The internal tensor/allocation plan and event ledger reconcile
byte-exactly. At synchronized qualification checkpoints, allocator-visible
CUDA usage and `/proc/<pid>/smaps` PSS are also sampled. The PSS decomposition
excludes the exact model inode and VMAs already attributed to tracked CUDA,
managed, registered, or host allocations. Remaining non-model PSS is
`host_library_unattributed`; remaining allocator-visible CUDA usage is
`cuda_library_unattributed`. Each has a 512-MiB ceiling and neither may contain
bytes already charged elsewhere.

The benchmark manifest freezes the process-scoped CUDA attribution method and
its pre-model baseline. Evidence of an unrelated CUDA allocator changing that
baseline during a sample makes the sample infrastructure-invalid; the harness
does not charge or forgive another process's allocation as DS4 footprint.

The single deduplicated formula used for every resident/streamed bound and
reduction comparison is:

```text
qualification_total_current = owned_total_current
                            + cuda_library_unattributed
                            + host_library_unattributed
                            + model_source_resident_bytes
```

`qualification_total_peak` is the maximum of that simultaneous value over
allocation events and synchronized residency checkpoints. A reconciliation
gap, duplicate attribution, or unattributed ceiling breach fails
qualification.

For the 32K/4K/one-session reference curve, the predeclared non-cache
qualification bound is at most 16 GiB. Therefore the hard simultaneous
qualification-total bounds are 24, 28, and 32 GiB for the 8, 12, and 16-GiB
cache profiles respectively. The allocation plan must fit its profile bound
before startup; every current value and the simultaneous peak must remain
inside both category and total bounds.
Crossing a bound is an unsafe runtime state: the request fails, the violation
is recorded, and the process exits nonzero rather than continuing with an
invalid compactness claim.

The CUDA graph honors the configured prefill cap. For the reference profile,
both configured and allocated rows equal 4,096. Retaining the current
16,384-row allocation shape, roughly 5.7 GiB instead of roughly 1.4 GiB, fails
qualification.

The qualification bundle reports resident and streamed peaks at the same
context/prefill/session profile plus:

```text
reduction_bytes = resident_qualification_total_peak
                - streamed_qualification_total_peak
reduction_ratio = reduction_bytes / resident_qualification_total_peak
```

For each passing reference profile, `reduction_bytes` must be at least 32 GiB
and `reduction_ratio` at least `0.45`. DS4 reports the achieved reduction and
enforces this compact-runtime floor; it does not decide whether even a passing
24/28/32-GiB profile is sufficient for a particular host ensemble.

## Page-cache contract

Global cache mutation such as `drop_caches` is forbidden.

The qualification harness and live runtime have separate advice domains:

- **Cold preparation:** before process launch, the harness may advise every
  full page wholly contained in safe tensor ranges on the pinned model inode.
  It excludes GGUF metadata and shared boundary pages, records
  `coldprep_eligible_unique_bytes`, and verifies eviction with `mincore`.
- **Runtime disposal:** after initial non-routed copies and after each
  routed-expert upload completes and CUDA work is synchronized, DS4 advises
  every full source page in that dynamically touched range. The denominator is
  the union `runtime_touched_eligible_unique_bytes`, not every expert in the
  model.

A shared boundary page remains resident. Neither advice domain races an
in-flight upload or discards a page required by another live tensor.

Cold-preparation evidence and runtime telemetry separately report attempted,
successful, and failed calls/bytes, unique eligible/advised page counts, and
`errno` by failure class. Qualification requires 100% attempted coverage of
both `coldprep_eligible_unique_bytes` and the final
`runtime_touched_eligible_unique_bytes`, nonzero successful bytes in each
domain, and zero failed calls. A successful syscall is not treated as proof of
eviction.

Qualification records resident pages for the exact opened model inode with
`mincore` at a synchronized quiescent point after advice and before the next
request. Aggregate host file-cache counters are not a substitute. The model
ledger derives unavoidable metadata/shared-boundary pages and records that
value, but the total quiescent exact-inode residency ceiling is also capped at
2 GiB. The last two identical warm repetitions may differ by at most 256 MiB,
and the latter cannot exceed the cap. The measurement includes source pages
for non-routed tensors copied at startup as well as routed-expert pages.

Qualification also captures source residency immediately before each runtime
advice operation. If a complete `mincore` sample would disturb the measured
path, it conservatively charges the prior post-advice residency plus the unique
pages touched since then. The larger observed/charged value updates
`model_source_resident_bytes` and therefore `qualification_total_peak`; a
transient file-page high-water cannot disappear behind successful post-request
advice.

## Protocol and foreground-process contract

DS4 preserves Laguna's native chat template, visible/reasoning separation,
tagged tool calls, and continuation after real tool results.

The compact path passes the same supported server shapes as resident CUDA:

- OpenAI Chat Completions and Responses;
- Anthropic messages;
- streaming and non-streaming equivalence;
- `tool_choice=auto` and `tool_choice=none`;
- chunked and multiple tool calls;
- malformed-call rejection; and
- continuation after a tool result.

`tool_choice=required` is not silently treated as `auto`; it returns a stable
unsupported-value error until DS4 genuinely enforces it. A request naming a
model family that does not match the loaded engine is rejected.

DS4 remains a foreground process. It never daemonizes, restarts itself, starts
another model, kills peers, or alters host-global caches. First `TERM` begins a
graceful drain. Process exit status is stable:

```text
0  graceful drain and exit
2  invalid invocation or configuration
1  startup, model, CUDA, or unsafe-runtime failure
```

Signals otherwise retain signal-derived exit status.

HTTP outcomes are separate from process exit status:

- invalid input, unsupported options, overflow, and protocol errors return a
  stable 4xx body before session mutation; the process remains healthy;
- a recoverable expert read/upload error returns 503 after the slot is restored
  to `EMPTY`; the process remains healthy; and
- a violated allocation/cache invariant or irrecoverable CUDA/cache state is
  unsafe: if headers are unsent, return a structured 500, otherwise terminate
  the stream, then drain and exit `1`.

Cancellation and `TERM` stop admitting requests, wait for submitted CUDA work
to reach a safe point, release pins, and then exit. A signal received before a
safe response can complete still retains its signal-derived process status.

## Stable downstream handoff

The compact runtime exposes DS4-owned facts without log scraping.

The following checked-in JSON Schemas are normative and use
`additionalProperties: false`:

```text
schemas/ds4-version-v1.schema.json
schemas/ds4-runtime-v1.schema.json
schemas/ds4-runtime-request-v1.schema.json
schemas/ds4-token-admission-v1.schema.json
schemas/ds4-laguna-compact-runtime-v1.schema.json
```

All potentially full-width unsigned 64-bit byte counts, durations, counters,
and `snapshot_seq` values are canonical decimal strings matching
`0|[1-9][0-9]*`; the schemas additionally enforce the uint64 maximum. Token
counts that are bounded below `2^53` are JSON integers. Rates are JSON numbers
with their token/second unit in the field name, timestamps are UTC RFC 3339
strings, and SHA-256 values are lowercase 64-character hex strings. Counters
saturate rather than wrap. One `/v1/runtime` response is an internally
consistent snapshot with a monotonically increasing process-lifetime
`snapshot_seq`; counters reset only on process start.

### Build and runtime identity

`--version-json` validates against `ds4.version/v1` and contains exactly
`schema`, 40-hex `revision`, boolean `dirty`, `backend`, and sorted `features`.
The qualification build requires `dirty=false`, `backend="cuda"`, and features
`laguna` plus `ssd_streaming`. `/v1/models` reports the canonical loaded model
identity.

The stable compact-runtime flags are:

```text
--ctx <positive tokens>
--prefill-chunk <positive tokens>
--session-slots <positive count>
--ssd-streaming
--ssd-streaming-cache-bytes <positive uint64 bytes>
```

These values never auto-grow or silently clamp. The existing
`--ssd-streaming-cache-experts` spelling may remain as deprecated compatibility
but is not part of the handoff contract and cannot appear in qualification.

`GET /v1/runtime` returns `ds4.runtime/v1` with required top-level keys
`schema`, `instance_id`, `snapshot_seq`, `state`, `build`, `executable`,
`model`, `config`, `limits`, `allocations`, `counters`, and `violations`.
`instance_id` is a process-lifetime UUID. The required content is:

- build revision, dirty flag, backend, and compiled features;
- executable device, inode, size, and mtime from the running image;
- canonical model ID/family plus opened-file device, inode, size, and mtime;
- effective context, prefill chunk, session slots, streaming state, and exact
  expert-cache limit;
- current/bound/peak bytes for every allocation category plus simultaneous
  internal and external totals;
- monotonic process-lifetime cache hit/miss/eviction counters;
- model-file read operations/bytes/time and host-to-device bytes/time;
- page-advice attempts/bytes/failures;
- configured and allocated prefill rows; and
- an array of active or historical hard-bound violations, empty for a healthy
  qualified process.

Every inference request returns its server-generated `request_id`. A
`ds4.runtime.request/v1` metrics object in the non-streaming response, or the
final streaming usage event, carries that request's prompt and total generated
tokens, TTFT, prefill rate, `visible_decode_tokens_per_second`, wall time,
cache/I/O deltas, and terminal status. It also records
`page_advice_complete_monotonic_ns` after the final
post-request advice and synchronization, or null when no advice applied. It is
not inferred by differencing two unrelated runtime snapshots.

The current `ds4-bench` `kvcache_bytes` value is snapshot/session payload size,
not live KV allocation. Rename or supplement it as
`session_payload_bytes`; report actual live allocation as
`kv_allocated_bytes`.

### Exact token admission

`POST /v1/token-admission` is side-effect free and accepts the same logical
`model`, `messages`, `tools`, and `tool_choice` fields as chat inference plus an
explicit `requested_output_tokens`. It applies the same Laguna tokenizer and
native chat template used by inference. Its response validates as
`ds4.token-admission/v1` with exactly `schema`, `model`, `template_revision`,
`templated_input_tokens`, `requested_output_tokens`, `context_tokens`, `fits`,
and nullable `rejection_code`:

- canonical model identity and native-template revision;
- exact templated-input and requested-output token counts;
- allocated context tokens;
- `fits` as a boolean; and
- a stable rejection code when the model, request shape, or output count is
  invalid.

`requested_output_tokens` means every generated token: hidden reasoning,
visible text, tool syntax, and stop tokens share that one ceiling. The
operation never creates or mutates a session. Downstream policy chooses any
additional reserve and includes it in the requested-output count; DS4 only
decides whether that exact total fits. The subsequent inference request still
performs the same admission check, so this endpoint cannot bypass enforcement.

### Qualification bundle

The qualification harness emits a JSON document conforming to
`ds4.laguna.compact-runtime/v1`. Its required top-level keys are `schema`,
`created_at`, `status`, `subject`, `host`, `model`, `schemas`, `oracle`,
`benchmark_manifest`, `global_gates`, `profiles`, and `evidence_root_sha256`.

`subject` binds the DS4 revision, clean `--version-json` digest, qualification
binary SHA-256, and running executable stat identity. `model` binds the GGUF
repository, revision, filename, size, SHA-256, opened-file stat identity, and
canonical served-model ID returned by `/v1/models` and `/v1/runtime`.
The harness hashes the model through DS4's opened descriptor, checks `fstat`
before and after hashing, and fails on identity change. It hashes the executable
before launch, then matches `/proc/<pid>/exe` device/inode/size/mtime to the
same file. It repeats model-descriptor and executable identity checks after the
final qualification sample. `schemas` records the schema IDs and content
hashes for every wire surface consumed by the run.

Each `global_gates[]` item has a stable `gate_id`, `status` from
`passed|failed|invalid`, and nonempty content-addressed evidence references.
Each `profiles[]` item has:

- `profile_id` and `profile_manifest_sha256`, the digest of canonical
  configuration plus bounds before results exist;
- exact `config` values and the allocation-plan digest;
- numeric per-category and simultaneous-total `bounds`;
- `status` from `passed|failed|invalid`;
- every required gate with measured values, threshold values, units, and
  evidence SHA-256 references; and
- resident/streamed peaks, reduction bytes/ratio, benchmark samples, exact-inode
  residency, and terminal failure information.

The bundle's `status` is `passed` only when all global gates pass and at least
one profile passes. A profile passes only when every gate required for that
profile passes; no aggregate score can compensate for a red gate. Failed and
invalid profiles remain in the bundle but are not qualified deployment inputs.

JSON is canonicalized with RFC 8785. The evidence set is exactly the union of
relative paths referenced anywhere in `global_gates[]` and `profiles[]`; the
bundle, sidecar, and evidence index are excluded. Paths must be UTF-8
POSIX-relative normal form with no symlinks, empty/`.`/`..` component, or
control character.

The harness sorts distinct paths by unsigned UTF-8 byte order and writes
`evidence-index.json` as an array of objects containing exactly `path`,
`size_bytes`, and `sha256` of the raw file bytes. It canonicalizes that array
with RFC 8785 and defines:

```text
evidence_root_sha256 = SHA256(canonical evidence-index.json bytes)
```

The verifier rejects a missing, extra, duplicate, size-mismatched, or
digest-mismatched referenced file. The canonical bundle is written to a
temporary file, fsynced, atomically renamed, and accompanied by a
`<bundle>.sha256` sidecar containing the digest of the final bytes. That sidecar
digest is the external trust anchor copied into a deployment manifest; the
bundle does not self-attest.

The bundle is evidence, not a service-control instruction. It contains no port,
peer model, residency class, or deployment decision.

## Qualification gates

### Numerical and metamorphic

- resident and streamed CUDA pass all promoted oracle ceilings and continuation;
- scalar CUDA primitives remain green;
- serialized versus batched and serialized versus mixed execution stay within
  the promoted ceilings; and
- cold, warm, and pressure states do not change accepted outputs.

### True streaming and bounds

- full-model registered bytes and full-model device-copy bytes are zero;
- routed expert bytes resident at startup are zero;
- every cache and allocation invariant remains true at each allocation event;
- the internal allocation plan/event ledger reconciles byte-exactly, each
  external unattributed category is at most 512 MiB, and the simultaneous
  `qualification_total_peak` is at most 24/28/32 GiB for the 8/12/16-GiB
  profile;
- 4,096 configured prefill rows allocate 4,096 rows; and
- every passing profile reduces the like-for-like resident peak by at least
  32 GiB and 45%.

### Page cache, failure, and stability

- reference qualification attempts 100% of qualification-only cold-preparation
  pages and 100% of the runtime touched-page union, has nonzero successful
  advice bytes and zero failures in both domains, and its synchronized
  exact-inode residency is at most the model-derived bound, which itself may
  not exceed 2 GiB;
- the final two identical warm repetitions differ by at most 256 MiB of
  exact-inode residency and the latter remains under the cap;
- a repeated identical workload increases cache hits and reads no more routed
  expert bytes than its cold predecessor; and
- after warm-up, three create/prefill/decode/free cycles leave current owned
  bytes within 64 MiB of the first post-warm cycle, with no monotonically
  growing allocation category;
- every injected read, CUDA, cancellation, and teardown failure restores all
  slots/pins/capacity or exits as unsafe without resident fallback; and
- the two-session pressure profile preserves cache-state, bound, and numerical
  invariants.

### Protocol and usability

- all supported request and tool-continuation shapes pass;
- overflow, invalid configuration, and malformed input fail in their declared
  class before session mutation;
- every passed profile completes every manifest sample at 512, 2K, 8K, and 28K
  on the recorded GB10/NVMe host within 45 minutes, TTFT is at most 15 minutes,
  and its exact three-warm-run median steady visible decode is at least 0.5
  token/second; and
- all four deterministic `ds4-eval` requests terminate normally and streamed
  answers/grades equal resident answers/grades.

The complete 92-case GPQA/SuperGPQA/AIME/COMPSEC run is optional, nonblocking
regression evidence. It is not a planner or co-residency gate.

## Test-first implementation sequence

1. Complete the approved single-Poolside resident-CUDA prerequisite.
2. Add failing benchmark-manifest/schema validation and artifact-binding tests.
3. Add failing tests for tensor classification, exact ranges, no-whole-map
   startup, numeric profile bounds, and byte-exact allocation ledger/telemetry.
4. Add failing cache identity, slot-state, reuse, eviction,
   batch-over-capacity, hard-bound, synchronization, two-session, and injected
   failure/cancellation tests.
5. Add failing 4K graph-allocation, advice-coverage, exact-inode residency, and
   repeated-growth tests.
6. Implement the minimum streamed CUDA path that makes the numerical,
   metamorphic, cache, allocation, and page-cache tests green.
7. Add failing normative-schema, version/runtime/request-metrics,
   token-admission, foreground-process, and protocol tests; implement the stable
   handoff.
8. Run the immutable reference profile curve and atomically publish the bundle,
   canonical evidence index, and sidecar digest.

## Rejected alternatives

### Put service management in DS4

That would couple an inference runtime to one host's process manager, endpoint
layout, peer models, and residency policy. DS4 remains a foreground process;
deployment systems manage it.

### Make ensemble co-residency a DS4 release gate

DS4 cannot define another workload's acceptable headroom, latency regression,
or lifecycle. It reports a truthful bounded footprint; the deployment makes
the system-level decision.

### Reuse the resident/full-map CUDA setup

Whole-map registration or copying defeats the primary objective even if Linux
later reclaims pages. The compact path must be compact by construction.

### Use only raw RSS or `nvidia-smi`

Neither proves which allocations DS4 owns, whether the complete model was
registered, or whether file pages accumulate. The allocation ledger, runtime
telemetry, and exact-inode evidence are all required.
