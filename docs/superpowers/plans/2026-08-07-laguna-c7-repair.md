# Laguna C7 Repair Checkpoint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the confirmed Tasks 0--7 safety, accounting, schema, and provenance gaps before implementing Laguna streamed cache I/O.

**Architecture:** Compact mode becomes a process-wide fail-closed CUDA lifecycle with typed teardown and an owned model descriptor. Pure policy and evidence validators reject active loads, exhausted attribution storage, wrong expert subranges, schema drift, and unverified artifacts. A retained descriptor runner binds the resident and compact DGX gates to the exact same opened model bytes.

**Tech Stack:** C99, CUDA C++, POSIX/Linux `/proc/self/maps`, Python 3, Draft 2020-12 JSON Schema, Make, shell, ASan/UBSan.

**Normative design:** `docs/superpowers/specs/2026-08-07-laguna-c7-repair-design.md`

---

### Task 1: Preserve active cache loads during drain

**Files:**
- Modify: `tests/test_laguna_stream.c:1719-1785`
- Modify: `ds4_laguna_stream.c:651-679`
- Modify: `ds4_laguna_stream.h:328-335`

- [ ] **Step 1: Add the failing LOADING-drain regression**

Extend `test_cache_pins_and_drain()` with a fresh one-slot fixture. Acquire a
missing key and require `LOAD_OWNER`, copy the complete slot array and reverse
map, route-hotness array, and sequence counter, then assert:

```c
CHECK(ds4_laguna_cache_policy_drain(&f.policy) ==
          DS4_LAGUNA_CACHE_RECOVERABLE &&
      memcmp(slots_before, f.slots, sizeof(slots_before)) == 0 &&
      memcmp(maps_before, f.entry_to_slot, sizeof(maps_before)) == 0 &&
      memcmp(hotness_before, f.route_hotness, sizeof(hotness_before)) == 0 &&
      f.policy.sequence == sequence_before,
      "drain preserves an active load owner");
```

- [ ] **Step 2: Run RED and commit the regression**

Run:

```sh
make tests/test_laguna_stream
./tests/test_laguna_stream --case cache-policy
```

Expected: FAIL because drain returns `OK` and clears the `LOADING` slot.

```sh
git add tests/test_laguna_stream.c
git commit -m "test: preserve Laguna load owners during drain"
```

- [ ] **Step 3: Make drain fail non-destructively**

Scan all slots before mutation. Treat both `LOADING` and `IN_USE` as active.
Return `RECOVERABLE` immediately if either exists. Clear only `READY` slots on
the successful path. Document this readiness contract in the header.

- [ ] **Step 4: Run GREEN and commit**

```sh
make tests/test_laguna_stream
./tests/test_laguna_stream --case cache-policy
git add ds4_laguna_stream.c ds4_laguna_stream.h
git commit -m "fix: keep Laguna loads live through drain"
```

### Task 2: Reuse bounded runtime-attribution storage

**Files:**
- Modify: `tests/test_laguna_stream.c:2210-2495`
- Modify: `ds4_runtime.h:104-132`
- Modify: `ds4_runtime.c:79-95,402-418`
- Modify: `ds4_cuda.cu:475,877-891`

- [ ] **Step 1: Add the failing tombstone-reuse test**

Initialize a tracker with capacity one. In namespace `0x10`, allocate sequence
1, release it, then allocate sequence 2 at a disjoint address. Require status
`OK`, `record_count == 1`, one live record for sequence 2, correct current
bytes, preserved peak bytes, and no violation. In fresh fixtures require the
historical sequence 1 and a never-issued-but-older sequence to fail closed,
while namespace `0x11` may independently start at sequence 1. Require sequence
zero to fail. IDs use the high byte as a producer namespace and the low 56
bits as a strictly increasing sequence.

- [ ] **Step 2: Run RED and commit**

```sh
make tests/test_laguna_stream
./tests/test_laguna_stream --case allocation
```

Expected: the second allocation latches `DS4_RUNTIME_VIOLATION_CAPACITY`.

```sh
git add tests/test_laguna_stream.c
git commit -m "test: recycle Laguna runtime attribution slots"
```

- [ ] **Step 3: Reuse dead records without weakening ID uniqueness**

Add `issued_sequence_high_water[256]` to the tracker. `append_record()` splits
each nonzero ID into its high-byte namespace and low-56-bit sequence, rejects
sequence zero or a sequence at/below that namespace's high-water mark, then
selects the first non-live record as a tombstone. Only append and increment
`record_count` if no tombstone exists. After a slot is secured, raise the
namespace high-water mark, zero the selected record, and fill the new ID.
Peaks remain tracker-level monotonic values, not record history. Change the
compact producer from the ledger's `0x4c` namespace to `0x4d`, and assign its
reserved IDs in actual append order: offsets, static slab, then mapping. This
is an exact bounded uniqueness rule across independent producers, not a global
ordering assumption across today's non-monotonic namespaces.

- [ ] **Step 4: Commit GREEN and verify the compact producer on DGX**

```sh
make tests/test_laguna_stream
./tests/test_laguna_stream --case allocation
git add ds4_runtime.h ds4_runtime.c ds4_cuda.cu
git commit -m "fix: reuse released Laguna attribution records"
green_revision="$(git rev-parse HEAD)"
green_dir="/tmp/ds4-laguna-tracker-green-${green_revision}"
ssh dgx-spark "mkdir -p '${green_dir}'"
git archive --format=tar "${green_revision}" |
  ssh dgx-spark "tar -xf - -C '${green_dir}'"
ssh dgx-spark \
  "cd '${green_dir}' && make tests/test_cuda_laguna_stream && ./tests/test_cuda_laguna_stream --case startup"
```

This CUDA run is required because the host allocation test does not compile
`ds4_cuda.cu`; it proves the new compact namespace and append-order ID
assignment do not trip the tracker.

### Task 3: Bind expert evidence to exact layer/projection subranges

**Files:**
- Modify: `ds4_laguna_stream.h:61-72`
- Modify: `ds4_laguna_stream.c:1179-1183`
- Modify: `ds4_laguna_plan.c:408-436,575-617,925-955`
- Modify: `tests/test_laguna_plan.c`
- Modify: `tests/test_laguna_stream.c`

- [ ] **Step 1: Add RED mutations for expert number, layer, and projection**

In the qualification-plan fixture, mutate expert 1 so its gate/up/down source
offsets point at expert 0. Separately swap a gate parent with an UP parent and
swap a complete parent triplet across layers. Each
`ds4_laguna_qualification_plan_validate()` call must fail without mutating the
fixture.

- [ ] **Step 2: Run RED and commit**

```sh
make tests/test_laguna_plan tests/test_laguna_stream
./tests/test_laguna_plan
./tests/test_laguna_stream --case ledger
```

Expected: at least the wrong-expert-offset mutation is accepted.

```sh
git add tests/test_laguna_plan.c tests/test_laguna_stream.c
git commit -m "test: bind Laguna expert evidence to exact ranges"
```

- [ ] **Step 3: Retain routed identity in tensor ranges**

Add `routed_layer` and `routed_projection` to
`ds4_laguna_tensor_range`. The ledger builder copies both descriptor fields;
non-routed ranges use `UINT32_MAX` and `DS4_LAGUNA_ROUTED_PROJECTION_NONE`.
Serialize these fields for routed tensor ranges so the immutable plan preserves
the validation inputs.

- [ ] **Step 4: Validate exact expert views**

For each view require the parent's retained layer/projection to match the
entry and view role. Compute with checked multiplication/addition:

```c
expected = parent->source_offset +
           entry->expert * ledger->routed_projection_expert_bytes;
```

Require `view->source_offset == expected`, exact source bytes, and the existing
device bounds/order. Update the frozen expected ledger JSON.

- [ ] **Step 5: Run GREEN and commit**

```sh
make tests/test_laguna_plan tests/test_laguna_stream
./tests/test_laguna_plan
./tests/test_laguna_stream --case ledger --case allocation
git add ds4_laguna_stream.h ds4_laguna_stream.c ds4_laguna_plan.c
git commit -m "fix: validate exact Laguna expert evidence"
```

### Task 4: Make the checked benchmark schema enforceable

**Files:**
- Create: `gguf-tools/quality-testing/requirements-compact-runtime.txt`
- Modify: `schemas/compact-runtime-benchmark-v1.schema.json`
- Modify: `gguf-tools/quality-testing/test_compact_runtime_qualify.py`
- Modify: `gguf-tools/quality-testing/compact_runtime_qualify.py`
- Modify: `Makefile`

- [ ] **Step 1: Add RED Draft 2020-12 equivalence tests**

Pin `jsonschema==4.25.1` in the qualification requirements. Load the schema with
`Draft202012Validator`, call `check_schema`, and validate a real manifest plus
a mutation corpus through both the schema and `validate_manifest()`.

The schema-expressible corpus must reject:

- `18446744073709551616` and larger canonical decimal strings;
- `0` in every field using the positive-uint64 definition;
- invalid padding, invalid alphabet, and noncanonical low bits in base64;
- relative model paths, repeated-slash paths, and absolute paths with the wrong
  basename; and
- every existing unknown/missing/null mutation.

Keep lexical `0` versus `0.0` in a separate strict-parser test because JSON
Schema validates number values rather than source spelling.

- [ ] **Step 2: Run RED and commit**

```sh
uv run --with-requirements \
  gguf-tools/quality-testing/requirements-compact-runtime.txt \
  python gguf-tools/quality-testing/test_compact_runtime_qualify.py -v
```

Expected: schema/custom mismatch for uint64 overflow, base64, and relative path.

```sh
git add gguf-tools/quality-testing/requirements-compact-runtime.txt \
  gguf-tools/quality-testing/test_compact_runtime_qualify.py
git commit -m "test: enforce the compact benchmark schema"
```

- [ ] **Step 3: Tighten schema-expressible constraints and the normal gate**

Use this exact bounded uint64 decimal pattern:

```text
^(?:0|[1-9][0-9]{0,18}|1[0-7][0-9]{18}|18[0-3][0-9]{17}|184[0-3][0-9]{16}|1844[0-5][0-9]{15}|18446[0-6][0-9]{14}|184467[0-3][0-9]{13}|1844674[0-3][0-9]{12}|184467440[0-6][0-9]{10}|1844674407[0-2][0-9]{9}|18446744073[0-6][0-9]{8}|1844674407370[0-8][0-9]{6}|18446744073709[0-4][0-9]{5}|184467440737095[0-4][0-9]{4}|18446744073709550[0-9]{3}|18446744073709551[0-5][0-9]{2}|1844674407370955160[0-9]|1844674407370955161[0-5])$
```

Use canonical base64:

```text
^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/][AQgw]==|[A-Za-z0-9+/]{2}[AEIMQUYcgkosw048]=)?$
```

Use the absolute pinned-basename model path pattern:

```text
^/(?:[^/]+/)*laguna-s-2[.]1-Q4_K_M[.]gguf$
```

Define positive uint64 with `allOf`: the bounded uint64 `$ref` plus
`{"not":{"const":"0"}}`; run overflow and zero mutations against both
ordinary and positive fields. Tighten the Python model-path validator to reject
`//` so it agrees with the canonical schema path, and keep it as the
independent strict implementation. Change
`test-laguna-compact-python` itself to run `uv run --with-requirements
gguf-tools/quality-testing/requirements-compact-runtime.txt python ...`; do
not rely on callers to pre-activate an environment. Add a short diagnostic if
`uv` or the pinned validator dependency is unavailable.

- [ ] **Step 4: Run GREEN and commit**

```sh
make test-laguna-compact-python
make test
git add schemas/compact-runtime-benchmark-v1.schema.json \
  gguf-tools/quality-testing/compact_runtime_qualify.py Makefile
git commit -m "fix: align compact schema with its validator"
```

### Task 5: Hash the opened descriptor in qualification plans

**Files:**
- Modify: `ds4_plan_io.h`
- Modify: `ds4_plan_io.c`
- Modify: `ds4_laguna_plan.h`
- Modify: `ds4_laguna_plan.c`
- Modify: `ds4.c`
- Modify: `tests/test_plan_io.c`
- Modify: `tests/test_laguna_plan.c`
- Modify: `Makefile`

- [ ] **Step 1: Add RED descriptor-hash and plan-provenance tests**

Add `ds4_plan_io_sha256_fd()` tests using a temporary regular file. Require the
known `abc` digest, unchanged caller file offset, exact expected size,
before/after identity stability, rejection of directories/truncation, and a
cleared output on failure.

The production signature is:

```c
bool ds4_plan_io_sha256_fd(
    int fd, uint64_t expected_size,
    char out[DS4_PLAN_IO_SHA256_HEX_SIZE],
    char *err, size_t errcap);
```

For the identity-change test, build `tests/test_plan_io` and `ds4_plan_io.c`
with `DS4_PLAN_IO_TEST_HOOKS`. A test-only chunk callback runs after the first
successful `pread`; it deterministically calls `ftruncate`/`futimens` on the
same open file before hashing continues. The normal object contains no hook.

Change the frozen plan expectation from a literal `expected_sha256` to an
observed `sha256`. Require plan validation to reject a missing, malformed, or
non-pinned observed digest.

- [ ] **Step 2: Run RED and commit**

```sh
make tests/test_plan_io tests/test_laguna_plan
./tests/test_plan_io
./tests/test_laguna_plan
```

Expected: missing fd hash API and unchanged `expected_sha256` output.

```sh
git add tests/test_plan_io.c tests/test_laguna_plan.c
git commit -m "test: bind Laguna plans to opened model bytes"
```

- [ ] **Step 3: Implement streaming fd SHA-256**

Expose a pread-based SHA-256 helper that hashes exactly `expected_size` bytes,
does not change the descriptor offset, and compares regular-file device,
inode, size, and nanosecond mtime before and after. Reuse the existing internal
SHA-256 context; do not allocate the model size.

- [ ] **Step 4: Thread the observed digest into the plan**

Add `model_sha256` to `ds4_laguna_qualification_plan_input`. Validate lowercase
64-hex and equality with the pinned Laguna digest. The plan-only writer hashes
the same retained `model.fd` after layout validation and before serialization,
then publishes `"sha256"` from that observed value.

- [ ] **Step 5: Run GREEN and commit**

```sh
make tests/test_plan_io tests/test_laguna_plan
./tests/test_plan_io
./tests/test_laguna_plan
git add ds4_plan_io.h ds4_plan_io.c ds4_laguna_plan.h \
  ds4_laguna_plan.c ds4.c Makefile
git commit -m "fix: hash Laguna plan model descriptors"
```

### Task 6: Bind resident and compact tests to one retained model fd

**Files:**
- Modify: `ds4.h`
- Modify: `ds4.c:2606-2790`
- Modify: `tests/test_cuda_laguna_model.c`
- Modify: `tests/test_cuda_laguna_stream.c`
- Modify: `gguf-tools/quality-testing/compare_laguna_logits.py`
- Modify: `gguf-tools/quality-testing/test_compare_laguna_logits.py`
- Create: `tests/test_cuda_laguna_gate_runner.py`
- Create: `tests/run_cuda_laguna_gate.sh`
- Modify: `Makefile`

- [ ] **Step 1: Add RED inherited-fd tests**

Add a Python helper test that hashes a passed descriptor, preserves its offset,
rejects size/hash mismatch, and detects an identity change. Add a behavioral
runner test with staged fake verifier/kernel/model children: every child
`fstat(9)`, the first child renames the pathname and installs different bytes,
and all later children must still report the original inode and bytes. Source
checks alone are insufficient. Update both C model tests to require that
`DS4_TEST_MODEL_FD`, when present, is used by engine options.

- [ ] **Step 2: Run RED and commit**

```sh
python3 gguf-tools/quality-testing/test_compare_laguna_logits.py -v
python3 tests/test_cuda_build_contract.py -v
python3 tests/test_cuda_laguna_gate_runner.py -v
```

Expected: missing inherited-fd verifier/engine contract and runner.

```sh
git add gguf-tools/quality-testing/test_compare_laguna_logits.py \
  tests/test_cuda_build_contract.py tests/test_cuda_laguna_gate_runner.py \
  tests/test_cuda_laguna_model.c \
  tests/test_cuda_laguna_stream.c
git commit -m "test: retain one Laguna artifact across CUDA gates"
```

- [ ] **Step 3: Add the hidden engine descriptor input**

Append `qualification_model_fd` and `qualification_model_fd_set` to
`ds4_engine_options`; no CLI parser exposes them. `model_open()` duplicates
the supplied fd with close-on-exec and maps that duplicate instead of reopening
the path. Normal callers retain path behavior. Tests parse only the trusted
`DS4_TEST_MODEL_FD` harness variable.

- [ ] **Step 4: Add the retained-fd runner and verifier support**

`tests/run_cuda_laguna_gate.sh` validates resident inputs, opens fd 9 once in
that shell, captures identity, invokes the Python verifier with `--gguf-fd 9`,
and exports `DS4_TEST_MODEL_FD=9` to every direct C child. It checks identity
after each child and rehashes after the final child. Replace the existing
multi-recipe-line resident target with the single invocation
`tests/run_cuda_laguna_gate.sh resident`; Make only builds its prerequisites.
The script has a distinct `self-test` mode for
`DS4_LAGUNA_GATE_TEST_CHILD_DIR`, `DS4_LAGUNA_GATE_TEST_EXPECTED_SIZE`, and
`DS4_LAGUNA_GATE_TEST_EXPECTED_SHA256`, allowing a small fake-child fixture.
`resident` rejects all three variables before opening fd 9 and always uses the
pinned 68,248,759,648-byte identity; the behavioral test proves both rejection
and `self-test` pathname-swap immunity. C7 mode and its stricter input checks
land test-first in Task 9.

- [ ] **Step 5: Run GREEN host contracts and commit**

```sh
python3 gguf-tools/quality-testing/test_compare_laguna_logits.py -v
python3 tests/test_cuda_build_contract.py -v
python3 tests/test_cuda_laguna_gate_runner.py -v
make cpu
git add ds4.h ds4.c gguf-tools/quality-testing/compare_laguna_logits.py \
  tests/test_cuda_laguna_gate_runner.py tests/run_cuda_laguna_gate.sh Makefile
git commit -m "fix: execute Laguna gates from one retained descriptor"
```

### Task 7: Make the compact CUDA lifecycle globally fail closed

**Files:**
- Modify: `ds4_gpu.h:103-140`
- Modify: `ds4_cuda.cu:429-540,684-1080,1464-1495,3936-4085,23783-23916,28390-28430`
- Modify: `ds4.h`
- Modify: `ds4.c:60179-60192`
- Modify: `tests/test_cuda_laguna_stream.c`
- Modify: `tests/test_runtime_cpp_link.cc`
- Modify: `Makefile`

- [ ] **Step 1: Drive and land the typed API seam**

First update `tests/test_runtime_cpp_link.cc` and add compile-contract calls for
the identity-bearing create signature,
typed destroy result, public test lifecycle enum, expanded snapshot, and the
two distinct test hooks. Also require in `ds4.h` a test-only engine-close
observation `{destroy_result, engine_retained, gpu_cleanup_before,
gpu_cleanup_after}` and require a generic-cleanup attempt counter hook in
`ds4_gpu.h`. Commit that API RED and verify that the exact DGX revision fails
to build. Then implement only the seam: accept but do not yet enforce the
identity, return a typed result around today's teardown behavior, publish the
snapshot fields/counters, expose a zeroed close observation, and make the new
failure hooks inert. Existing unsafe fallback/retry behavior deliberately
remains for the behavioral RED.

```sh
make tests/test_runtime_cpp_link
git add tests/test_cuda_laguna_stream.c tests/test_runtime_cpp_link.cc
git commit -m "test: require typed Laguna compact lifecycle"
api_red_revision="$(git rev-parse HEAD)"
api_red_dir="/tmp/ds4-laguna-c7-api-red-${api_red_revision}"
ssh dgx-spark "mkdir -p '${api_red_dir}'"
git archive --format=tar "${api_red_revision}" |
  ssh dgx-spark "tar -xf - -C '${api_red_dir}'"
ssh dgx-spark "cd '${api_red_dir}' && make tests/test_cuda_laguna_stream"
```

Expected: a compile/link failure naming the missing typed contract. Implement
the seam, update every call site, run the existing startup case, and commit:

```sh
make tests/test_runtime_cpp_link
git add ds4_gpu.h ds4.h ds4_cuda.cu ds4.c
git commit -m "refactor: expose the Laguna compact lifecycle seam"
api_green_revision="$(git rev-parse HEAD)"
api_green_dir="/tmp/ds4-laguna-c7-api-green-${api_green_revision}"
ssh dgx-spark "mkdir -p '${api_green_dir}'"
git archive --format=tar "${api_green_revision}" |
  ssh dgx-spark "tar -xf - -C '${api_green_dir}'"
ssh dgx-spark \
  "cd '${api_green_dir}' && make tests/test_cuda_laguna_stream && ./tests/test_cuda_laguna_stream --case startup"
```

- [ ] **Step 2: Add runnable RED lifecycle, mapping, and bypass cases**

With the seam buildable, extend synthetic startup with:

- a second mapping and an alias mapping of the same inode; assert every current
  placement API family: `ds4_gpu_set_model_fd`,
  `ds4_gpu_set_model_fd_for_map`, `ds4_gpu_set_model_map`,
  `ds4_gpu_register_model_map_no_copy`, `ds4_gpu_set_model_map_range`,
  `ds4_gpu_set_model_map_spans`, `ds4_gpu_register_support_map`,
  `ds4_gpu_device_cache_tensors`, `ds4_gpu_device_cache_support_tensors`,
  `ds4_gpu_cache_model_range`, `ds4_gpu_cache_q8_f16_range`,
  `ds4_gpu_preload_q4_expert_tables`, and the begin/prepare/seed selected-expert
  cache entry points. Exercise internal range, arena, Q8-F16, Q8-F32, and warm
  resolution through public wrappers. Each rejection increments the dedicated
  compact-rejection counter but changes no legacy allocation/copy, cache,
  tracker, or placement counter;
- fd A with mmap B, expected identity A with a same-size fd B, an anonymous
  map, `PROT_NONE`, mixed-inode adjacent VMAs, a gap, post-map truncation, and
  a shifted file offset. Every case must fail before the static CUDA allocation
  attempt counter changes. Also close/reuse the caller fd after successful
  creation and prove the context still reads its owned duplicate; and
- a pre-commit synchronization failure returning `RECOVERABLE`, preserving
  ownership, followed by a successful retry.

Add separate `--case teardown-unsafe`. Inject the first post-commit release
failure and require `UNSAFE`, `RELEASING`, no retry/release repetition, no
legacy placement, and retained engine-owned records.

Add `--case model-teardown-unsafe` for the controlled pinned-model window. It
opens an engine from `DS4_TEST_MODEL_FD`, injects the post-commit failure, and
calls `ds4_engine_close()`. Test-only close observation must prove the engine
was retained, the compact state remains `RELEASING`, and the generic
`ds4_gpu_cleanup()` call counter did not advance. This case runs as its own
process so deliberate unsafe state cannot poison another case.

- [ ] **Step 3: Commit runnable RED and record the exact DGX failures**

```sh
git add tests/test_cuda_laguna_stream.c Makefile
git commit -m "test: close compact CUDA lifecycle escapes"
red_revision="$(git rev-parse HEAD)"
red_dir="/tmp/ds4-laguna-c7-red-${red_revision}"
ssh dgx-spark "mkdir -p '${red_dir}'"
git archive --format=tar "${red_revision}" |
  ssh dgx-spark "tar -xf - -C '${red_dir}'"
ssh dgx-spark "cd '${red_dir}' && make tests/test_cuda_laguna_stream"
ssh dgx-spark "cd '${red_dir}' && ./tests/test_cuda_laguna_stream --case startup"
ssh dgx-spark "cd '${red_dir}' && ./tests/test_cuda_laguna_stream --case teardown-unsafe"
```

Expected: the binary builds, reaches the new assertions, and fails mapping,
legacy-exclusivity, recoverable, and unsafe teardown behavior. Preserve the
output in the task trace; do not amend the RED commit.

In the controlled peer-services-stopped window, also open fd 9 in one remote
shell and observe the engine integration RED:

```sh
ssh dgx-spark \
  "cd '${red_dir}' && exec 9<'${LAGUNA_MODEL}' && DS4_TEST_MODEL='${LAGUNA_MODEL}' DS4_TEST_MODEL_FD=9 ./tests/test_cuda_laguna_stream --case model-teardown-unsafe"
```

- [ ] **Step 4: Split legacy permission from compact lookup**

Legacy placement is allowed only in `IDLE`. Compact lookup succeeds only for
the exact active mapping. Guard every legacy model/cache entry before every
mutation except incrementing the dedicated compact-rejection counter. Legacy
allocation/copy/cache/placement counters remain unchanged, including for the
selected-expert loader and support mapping.

- [ ] **Step 5: Own and validate descriptor identity**

Change the create prototype to:

```c
int ds4_gpu_laguna_compact_create(
    ds4_gpu_laguna_compact **out, int model_fd, const void *model_map,
    uint64_t model_size, const ds4_laguna_file_identity *model_identity,
    const ds4_laguna_ledger *ledger,
    const ds4_laguna_allocation_plan *plan, ds4_runtime_tracker *tracker);
```

Pass `&e->model.identity` at the engine call site. Duplicate the fd with
close-on-exec. Before CUDA allocation, require the duplicate's regular-file
device, inode, exact size, and mtime to equal that engine-captured identity.
Parse `/proc/self/maps` into ordered VMAs and require readable coverage of
`[model_map, model_map + model_size)` with no gaps. For each covered segment,
match the maps hex major/minor to `major(st_dev)`/`minor(st_dev)`, match its
decimal inode, and require:

```text
vma_file_offset + (covered_start - vma_start)
    == covered_start - model_map
```

Store only the duplicate for future `pread`; close/caller-fd reuse cannot
affect it.

Creation validation failures close the duplicate immediately. A
pre-publication unwind closes it last only after every acquired owner releases
successfully; an unwind release failure latches fail-closed state and retains
the duplicate. Post-commit `UNSAFE` also retains it for process-exit cleanup.

- [ ] **Step 6: Implement typed two-phase teardown**

Expose this exact result API:

```c
typedef enum {
    DS4_GPU_LAGUNA_DESTROY_OK = 0,
    DS4_GPU_LAGUNA_DESTROY_RECOVERABLE = 1,
    DS4_GPU_LAGUNA_DESTROY_UNSAFE = 2,
} ds4_gpu_laguna_destroy_status;

ds4_gpu_laguna_destroy_status ds4_gpu_laguna_compact_destroy(
    ds4_gpu_laguna_compact *ctx);
```

Under test hooks expose this public test-visible lifecycle type (distinct from
the private atomic's representation):

```c
typedef enum {
    DS4_GPU_LAGUNA_LIFECYCLE_IDLE = 0,
    DS4_GPU_LAGUNA_LIFECYCLE_CREATING,
    DS4_GPU_LAGUNA_LIFECYCLE_ACTIVE,
    DS4_GPU_LAGUNA_LIFECYCLE_DESTROYING,
    DS4_GPU_LAGUNA_LIFECYCLE_RELEASING,
} ds4_gpu_laguna_lifecycle;
```

Expose distinct
`ds4_gpu_test_laguna_compact_fail_sync_once()` and
`ds4_gpu_test_laguna_compact_fail_release_once()` injections. Extend the
snapshot with lifecycle (`IDLE`, `CREATING`, `ACTIVE`, `DESTROYING`,
`RELEASING`), all owner-live flags, sync-attempt count, release-attempt count,
and rejection count. A pre-commit sync failure returns `RECOVERABLE`; compare
owner pointers/flags, descriptor identity, allocation data, and tracker records
byte-for-byte with the pre-destroy snapshot, while separately requiring the
intentional `ACTIVE` to `DESTROYING` state change and incremented sync-attempt
counter. Release attempts remain zero and one retry succeeds. Successful sync
is the commit point and
transitions to `RELEASING`; the first injected release failure returns
`UNSAFE`. A retry is rejected and cannot increase release attempts or repeat a
release. Close the duplicate fd last on full success only; retain it in the
post-commit unsafe state.

`ds4_engine_close()` branches on the result: it may retry one `RECOVERABLE`
result, but `UNSAFE` or a second `RECOVERABLE` logs and returns while retaining
the engine. Neither result may route through `ds4_gpu_cleanup()`; generic GPU
cleanup runs only after compact destroy returned `OK`.

Under `DS4_TEST_HOOKS`, expose a close observation containing the destroy
result, whether the engine was retained, and generic GPU-cleanup attempts
before/after. The pinned `model-teardown-unsafe` subprocess asserts this
engine-level branch; direct-context teardown alone is insufficient.

- [ ] **Step 7: Commit GREEN and run that exact revision on DGX**

```sh
git add ds4_gpu.h ds4_cuda.cu ds4.c
git commit -m "fix: make compact CUDA lifecycle fail closed"
green_revision="$(git rev-parse HEAD)"
green_dir="/tmp/ds4-laguna-c7-green-${green_revision}"
ssh dgx-spark "mkdir -p '${green_dir}'"
git archive --format=tar "${green_revision}" |
  ssh dgx-spark "tar -xf - -C '${green_dir}'"
ssh dgx-spark "cd '${green_dir}' && make tests/test_cuda_laguna_stream"
ssh dgx-spark "cd '${green_dir}' && ./tests/test_cuda_laguna_stream --case startup"
ssh dgx-spark "cd '${green_dir}' && ./tests/test_cuda_laguna_stream --case teardown-unsafe"
```

In the same controlled window, run the pinned engine branch from retained fd 9:

```sh
ssh dgx-spark \
  "cd '${green_dir}' && exec 9<'${LAGUNA_MODEL}' && DS4_TEST_MODEL='${LAGUNA_MODEL}' DS4_TEST_MODEL_FD=9 ./tests/test_cuda_laguna_stream --case model-teardown-unsafe"
```

Expected: all three cases pass. If they do not, make a new focused fix commit and
archive/run that new exact revision; never patch the DGX review directory.

### Task 8: Reconcile every live C7 allocation

**Files:**
- Modify: `ds4.h`
- Modify: `ds4.c:36395-36420,58690-58875,60907-60980`
- Modify: `ds4_runtime.h`
- Modify: `ds4_runtime.c`
- Modify: `tests/test_laguna_stream.c`
- Modify: `tests/test_cuda_laguna_stream.c:885-934`

- [ ] **Step 1: Drive the generic replay helper RED**

Add this generic descriptor and helper to `ds4_runtime.h`:

```c
typedef struct {
    uint64_t allocation_id;
    uint32_t callsite_id;
    uint64_t base;
    uint64_t requested_bytes;
    uint64_t charged_bytes;
} ds4_runtime_owned_descriptor;

ds4_runtime_status ds4_runtime_tracker_replay_owned(
    ds4_runtime_tracker *tracker,
    const ds4_runtime_owned_descriptor *owners, size_t owner_count,
    bool *owner_live);
```

The helper sets each `owner_live[i]` only after its allocation record commits.
Pure tests replay all six valid descriptors and require exact records,
category totals, and simultaneous totals. Separate fresh fixtures reject an
unknown/unclassified callsite, address overlap, insufficient record capacity,
and callsite/category/total bound mismatch.

```sh
make tests/test_laguna_stream
```

Expected: link failure for the absent replay helper. Commit the complete API
RED, including the declaration that the test compiled against:

```sh
git add ds4_runtime.h tests/test_laguna_stream.c
git commit -m "test: replay live Laguna allocations"
```

- [ ] **Step 2: Implement and GREEN the generic replay helper**

Implement the helper as a thin ordered application of the existing checked
tracker allocation API. Preserve committed `owner_live` flags on later
failure so startup unwind can release exactly what became live.

```sh
make tests/test_laguna_stream tests/test_runtime_cpp_link
./tests/test_laguna_stream --case allocation
./tests/test_runtime_cpp_link
git add ds4_runtime.c
git commit -m "feat: replay live runtime allocations"
```

- [ ] **Step 3: Drive and land the engine snapshot seam**

Under test hooks declare the copy-only live accessor
`ds4_test_engine_laguna_runtime_snapshot(engine, snapshot, records, capacity,
required)` and process-local close observer
`ds4_test_laguna_last_close_snapshot(snapshot)`. Add the model-startup calls
and commit this API RED; it must fail to link. Then implement only the access
seam: the live accessor copies today's tracker, while the close observer
returns false until close capture is implemented. Commit the seam so the same
test binary now builds but has not yet gained inventory behavior.

Also expose an independent test-only
`ds4_test_engine_laguna_live_owners(engine, owners, capacity, required)` that
copies six `{base, bytes, callsite_id}` tuples directly from the live private
engine/model/vocabulary pointers and counts. It must compute those tuples from
the owner fields on each call, not reuse the replay descriptor array. The
model-startup test compares tracker records with these independently observed
owners, so a wrong tracked pointer or size cannot pass by self-report.

```sh
git add ds4.h tests/test_cuda_laguna_stream.c
git commit -m "test: expose Laguna engine allocation snapshots"
snapshot_red_revision="$(git rev-parse HEAD)"
snapshot_red_dir="/tmp/ds4-laguna-inventory-api-red-${snapshot_red_revision}"
ssh dgx-spark "mkdir -p '${snapshot_red_dir}'"
git archive --format=tar "${snapshot_red_revision}" |
  ssh dgx-spark "tar -xf - -C '${snapshot_red_dir}'"
ssh dgx-spark \
  "cd '${snapshot_red_dir}' && make tests/test_cuda_laguna_stream"
```

Expected: link failure naming the absent accessors. After observing it,
implement the accessors and commit:

```sh
git add ds4.c
git commit -m "test: expose copy-only Laguna allocation snapshots"
```

- [ ] **Step 4: Run the exact engine-inventory RED on DGX**

In a controlled peer-services-stopped window, `--case model-startup` must
require call sites `OTHER_HOST_ENGINE`,
`OTHER_HOST_MODEL`, and `OTHER_HOST_VOCAB`, with bases/bytes matching the six
live owners plus ledger and compact records. After close, the close observer
must report no active record and no violation. Archive the exact accessor-seam
revision and run:

```sh
red_revision="$(git rev-parse HEAD)"
red_dir="/tmp/ds4-laguna-c7-inventory-red-${red_revision}"
ssh dgx-spark "mkdir -p '${red_dir}'"
git archive --format=tar "${red_revision}" |
  ssh dgx-spark "tar -xf - -C '${red_dir}'"
ssh dgx-spark \
  "cd '${red_dir}' && make tests/test_cuda_laguna_stream && exec 9<'${LAGUNA_MODEL}' && DS4_TEST_MODEL='${LAGUNA_MODEL}' DS4_TEST_MODEL_FD=9 ./tests/test_cuda_laguna_stream --case model-startup"
```

Expected: only ledger/static records exist; engine/model/vocabulary call sites
are absent and no close observation exists. The binary must build, reach those
assertions, and fail. Preserve the output; do not patch the DGX directory.

- [ ] **Step 5: Replay the exact live inventory**

After tracker initialization and before compact CUDA creation, build and replay
these exact descriptors, using namespace `0x4e` sequences 1--6 as stable IDs:

1. `e` / `sizeof(*e)` as `OTHER_HOST_ENGINE`;
2. `model.kv` / `model.n_kv * sizeof(model.kv[0])` and `model.tensors` /
   `model.n_tensors * sizeof(model.tensors[0])` as `OTHER_HOST_MODEL`; and
3. `vocab.token` / `vocab.n_vocab * sizeof(vocab.token[0])`,
   `vocab.token_to_id.entry` / `vocab.token_to_id.cap *
   sizeof(vocab.token_to_id.entry[0])`, and `vocab.merge_rank.entry` /
   `vocab.merge_rank.cap * sizeof(vocab.merge_rank.entry[0])` as
   `OTHER_HOST_VOCAB`.

Use checked size arithmetic before constructing any descriptor. Store six IDs
and six live flags on the engine. Adapt compact ownership reconciliation to
recognize inventory plus ledger records rather than assuming exactly three
live records. Release the three vocabulary records immediately before
`vocab_free`, the two model records immediately before `model_close`, and the
engine record immediately before `free(e)`. Capture the test close snapshot
after that final release.

- [ ] **Step 6: Commit GREEN and verify that exact revision**

```sh
make tests/test_laguna_stream tests/test_runtime_cpp_link
./tests/test_laguna_stream --case allocation
./tests/test_runtime_cpp_link
git add ds4.c
git commit -m "fix: track every live Laguna C7 allocation"
green_revision="$(git rev-parse HEAD)"
green_dir="/tmp/ds4-laguna-c7-inventory-green-${green_revision}"
ssh dgx-spark "mkdir -p '${green_dir}'"
git archive --format=tar "${green_revision}" |
  ssh dgx-spark "tar -xf - -C '${green_dir}'"
ssh dgx-spark \
  "cd '${green_dir}' && make tests/test_cuda_laguna_stream && exec 9<'${LAGUNA_MODEL}' && DS4_TEST_MODEL='${LAGUNA_MODEL}' DS4_TEST_MODEL_FD=9 ./tests/test_cuda_laguna_stream --case model-startup"
```

Expected: pure allocation tests and pinned model startup pass. Fix and archive
a new exact revision rather than editing the DGX directory if the CUDA case is
not green.

### Task 9: Wire and run the required C7 qualification gate

**Files:**
- Modify: `tests/test_cuda_build_contract.py`
- Modify: `tests/test_cuda_laguna_gate_runner.py`
- Modify: `tests/run_cuda_laguna_gate.sh`
- Modify: `Makefile`
- Modify: `README.md`
- Modify: `docs/superpowers/plans/2026-08-03-laguna-compact-cuda-streaming.md`

- [ ] **Step 1: Add and commit RED C7 input/gate tests**

Extend the behavioral runner and Make contract tests. C7 mode must reject:

- unset or default `DS4_TEST_MODEL=ds4flash.gguf`, even if that file exists;
- a nonexistent model path; and
- an unset, empty, or non-40-lowercase-hex
  `LAGUNA_TOKENIZER_RUNTIME_COMMIT`; and
- each of `DS4_LAGUNA_GATE_TEST_CHILD_DIR`,
  `DS4_LAGUNA_GATE_TEST_EXPECTED_SIZE`, and
  `DS4_LAGUNA_GATE_TEST_EXPECTED_SHA256`, singly and together, before fd 9 is
  opened.

Require `test-cuda-laguna-c7` in `.PHONY`, with CUDA binaries as prerequisites
and exactly one recipe process: `tests/run_cuda_laguna_gate.sh c7`.

```sh
python3 tests/test_cuda_build_contract.py -v
python3 tests/test_cuda_laguna_gate_runner.py -v
```

Expected: C7 mode/target or at least one validation is absent.

```sh
git add tests/test_cuda_build_contract.py tests/test_cuda_laguna_gate_runner.py
git commit -m "test: require explicit Laguna C7 gate inputs"
```

- [ ] **Step 2: Implement C7 mode and commit GREEN before qualification**

Add the named target and validation. The one retained-fd runner process performs
fixture verification, resident kernels/oracle, synthetic startup, unsafe
teardown in its own subprocess, pinned model startup, and
`model-teardown-unsafe` in a final isolated subprocess. This final engine-level
unsafe case is mandatory after Task 8's close/inventory changes; no Make recipe
line opens or reopens the model path between children.

```sh
python3 tests/test_cuda_build_contract.py -v
python3 tests/test_cuda_laguna_gate_runner.py -v
git add Makefile tests/run_cuda_laguna_gate.sh
git commit -m "test: add the Laguna C7 qualification gate"
```

- [ ] **Step 3: Run host and sanitizer regression**

```sh
uv run --with-requirements \
  gguf-tools/quality-testing/requirements-compact-runtime.txt make test
make -B -j4 \
  CFLAGS='-O1 -g -Wall -Wextra -std=c99 -fsanitize=address,undefined -fno-omit-frame-pointer' \
  CXXFLAGS='-O1 -g -Wall -Wextra -std=c++11 -fsanitize=address,undefined -fno-omit-frame-pointer' \
  tests/test_laguna_stream tests/test_laguna_plan tests/test_plan_io
./tests/test_laguna_stream
./tests/test_laguna_plan
./tests/test_plan_io
git diff --check
```

Expected: every host suite passes with no sanitizer finding or whitespace
error.

- [ ] **Step 4: Run the committed DGX C7 target in a controlled window**

First confirm peer inference services are intentionally stopped; DS4 does not
stop them. The gate implementation is already committed by Step 2. Archive
that exact `HEAD` to a new revision-named DGX directory, build there, then run:

```sh
gate_revision="$(git rev-parse HEAD)"
gate_dir="/tmp/ds4-laguna-c7-${gate_revision}"
ssh dgx-spark "mkdir -p '${gate_dir}'"
git archive --format=tar "${gate_revision}" |
  ssh dgx-spark "tar -xf - -C '${gate_dir}'"
export LAGUNA_TOKENIZER_RUNTIME_COMMIT="$(
  python3 -c 'import json; print(json.load(open("tests/test-vectors/laguna-resident/manifest.json"))["provenance"]["tokenizer_runtime_commit"])'
)"
ssh dgx-spark \
  "cd '${gate_dir}' && DS4_TEST_MODEL='${LAGUNA_MODEL}' LAGUNA_TOKENIZER_RUNTIME_COMMIT='${LAGUNA_TOKENIZER_RUNTIME_COMMIT}' make test-cuda-laguna-c7"
```

Expected: fixture verification, resident oracle, synthetic startup, unsafe
teardown isolation, and pinned-model compact startup all pass from one retained
descriptor.

- [ ] **Step 5: Document the observed checkpoint and commit**

Document the named gate, exact tested revision and observed result, and state
clearly that original streaming-plan Tasks 8--21 remain unimplemented. If the
gate required a code fix, commit that fix and rerun the new exact revision
before recording success.

```sh
git add README.md \
  docs/superpowers/plans/2026-08-03-laguna-compact-cuda-streaming.md
git commit -m "test: require the Laguna C7 DGX checkpoint"
```

- [ ] **Step 6: Request final adversarial review**

Review the complete range from `96d9634` through `HEAD` for spec compliance,
CUDA lifetime safety, tracker reconciliation, artifact binding, and test-gate
truthfulness. Fix every Critical or Important finding before proceeding to
original streaming-plan Task 8.
