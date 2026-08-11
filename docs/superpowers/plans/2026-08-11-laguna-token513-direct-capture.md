# Laguna token-513 layer-1 direct capture

Status: routed-expert causal diagnosis complete; router attribution remains open.

This report records the direct Poolside/DS4 token-513, layer-1 comparison.
Every comparison is bitwise and follows semantic execution order rather than
sorting by error magnitude.

## Pinned run identity

- Model: `laguna-s-2.1-Q4_K_M.gguf`
  - bytes: `68248759648`
  - SHA-256: `e163b2c98908809a71245d6bb68b2226994d9969cb2a438eccb72196a1c4147a`
- Prefix: 512 little-endian I32 tokens
  - bytes: `2048`
  - SHA-256: `569aa6394783e0f17558db421ba26480d7a530d44dd2219bcc9aac2c09a3b559`
  - resume token: `3612` at position 512
- Device: NVIDIA GB10, compute capability 12.1, driver 580.126.09
- Poolside commit: `04b2b72cb54048ead292884adbe11f284e3ec950`
- DS4 accepted-capture code state: `1d009d4f134af0f069730702a6247c077e18fdbd`
- DS4 capture hook commit: `6718196b28f1270300cec2fb980d81cd886208c0`

The complete content-addressed run envelope is
[`token513-layer1-run.json`](../../../tests/oracle-producers/laguna-c7/token513-layer1-run.json),
SHA-256
`c25eb57c4b84859856b42e1f3b2333e5fa38dface30da5a25e9a7d8004ab9a3e`.
It binds every comparator input to its byte count and SHA-256, plus the
model, prefix, device, runtime commits, producer binaries, and release control.

The durable ordered result is
[`token513-layer1-comparison.json`](../../../tests/oracle-producers/laguna-c7/token513-layer1-comparison.json),
SHA-256
`17614d19cce3f95941b4d8a8babd3bcfc7ae791086efbacc2a4155bc2d37b708`.

Raw bundles remain in scratch storage rather than Git:

- Poolside complete rerun:
  `/tmp/poolside-laguna-token513-complete.CVy1F1` on the capture host and
  `/private/tmp/laguna-direct-analysis.EV2V7r/poolside-complete` locally.
- DS4 accepted capture:
  `/tmp/ds4-laguna-direct513-moe-capture2` on the capture host and
  `/private/tmp/laguna-direct-analysis.EV2V7r/ds4-repaired` locally.

Both exact producers and their run recipes are tracked. The raw-bundle
retention limitation is explicit in the run envelope.

## Acceptance gates

Three DS4 executions were compared at token 513:

| Control | Result |
|---|---|
| release-linked probe vs hook cubin with null capture | bit-exact logits |
| hook cubin with null capture vs active capture | bit-exact logits |
| logits payload | 401,408 bytes, SHA-256 `e837bc5b...d21c2` |

This closes both observer-store perturbation and compile-time hook-cubin
perturbation for the token-513 logits.

The complete Poolside rerun reproduced every previously pinned Poolside
artifact hash, then added the missing FFN norm, column-L2, and F32 down-input
boundaries. The diagnostic callback patch was reversed afterward; the
Poolside source checkout was clean, the GPU process list was empty, and all
six ensemble services remained inactive with PID 0.

Pre-expert input binding:

| Operand | Poolside | DS4 | Evidence |
|---|---|---|---|
| FFN-normalized F32 | `eabe89d1...85ef0e` | `eabe89d1...85ef0e` | runtime bytes exact |
| expert input Q8_1 | not graph-exposed | `8df5b6ef...d7de0a` | DS4 runtime bytes equal microscope fixture |
| Q4_K expert-144 gate row 0 | pinned model row `b42e00f4...70fe5` | same pinned model | fixture/model identity |

Poolside's runtime Q8_1 bytes were not directly observed: Poolside creates
them inside the CUDA Q4_K multiply rather than as a graph callback tensor.
The microscope binds Poolside's byte-exact F32 input to its pinned
quantization implementation, while DS4's runtime Q8_1 bytes are captured
directly. Thus the dual-runtime Q8 identity is inferred from pinned Poolside
quantization semantics, not claimed as two observed byte arrays.

An earlier DS4 capture was rejected because its FFN input hash was
`a166f996...a94808`. The recovered worktree had omitted the existing
generalized Poolside Q8 Stream-K prefill and long AUTO MMA64 attention parity
paths. Restoring those paths produced the accepted exact FFN input above.
Nothing downstream from the rejected input is used here.

## Ordered comparison

| Stage | Exact values | Unequal values | First unequal coordinate |
|---|---:|---:|---|
| FFN norm | 3072 / 3072 | 0 | none |
| router logits | 92 / 256 | 164 | expert index 0 |
| selected IDs | 10 / 10 | 0 | none |
| router weights | 8 / 10 | 2 | slot 0, expert 144 |
| gate | 1557 / 10240 | 8683 | slot 0, expert 144, row 0 |
| up | 1465 / 10240 | 8775 | slot 0, expert 144, row 0 |
| SwiGLU | 750 / 10240 | 9490 | slot 0, expert 144, row 1 |
| column L2 | 7 / 10 | 3 | slot 1, expert 15, row 0 |
| F32 down input | 797 / 10240 | 9443 | slot 0, expert 144, row 1 |
| down-input Q8_1 | unavailable | unavailable | Poolside runtime tensor not exposed |
| down | 4384 / 30720 | 26336 | slot 0, expert 144, row 0 |
| weighted down | 4222 / 30720 | 26498 | slot 0, expert 144, row 0 |
| routed sum | 382 / 3072 | 2690 | row 0 |
| shared expert | 3072 / 3072 | 0 | none |
| combined MoE | 664 / 3072 | 2408 | row 0 |

The newly captured column-L2 and F32 down-input boundaries occur after the
gate projection in the execution DAG. They therefore do not change the first
routed-expert result.

First overall arithmetic inequality:

```text
stage:         router logits
expert index:  0
Poolside:      0xbea377ba
DS4:           0xbea377b8
```

The selected expert IDs remain exact:

```text
144, 15, 106, 165, 240, 108, 253, 102, 34, 226
```

Only routing-weight slots 0 and 7 differ:

```text
slot 0: Poolside 0x3ebff8c4, DS4 0x3ebff8c3
slot 7: Poolside 0x3e70edd2, DS4 0x3e70edd5
```

## First routed-expert inequality

```text
expert:       144
projection:   gate
row:          0

Poolside:     0xbdcdf5ed
DS4 capture:  0xbdcdf5ef

microscope:
    Poolside topology -> 0xbdcdf5ed
    DS4 topology      -> 0xbdcdf5ef
    FP64 reference    -> -0.10056671364048952

operand hashes:
    Q4_K row          b42e00f452c044c1cf8679b5340a9a0738854576ad7aa294076ff4a3ed870fe5
    Q8_1 activation   8df5b6ef5738aed267bccb6afd3b0deec21cd44e3a09908e88356cfa24d7de0a
```

The generic one-row microscope rerun produced:

```text
DS4 topology      bits=0xbdcdf5ef abs_error=2.3713312202744419e-08
Poolside topology bits=0xbdcdf5ed abs_error=8.8121510088967625e-09
```

The comparator now fails closed unless the microscope token, layer,
projection, selected slot, expert, row, direct Poolside bits, direct DS4 bits,
fixture output, and both oracle values all bind to the first routed mismatch.
This establishes the causal explanation for expert 144 gate row 0. It does
not claim to explain the earlier router-logit divergence.

## Router mechanism

The router does not share the expert Q4_K/Q8_1 primitive:

- Router weight and input are F32; no Q4_K unpack or Q8_1 quantization occurs.
- Poolside's F32 MMVF assigns adjacent `float2` pairs to 256 threads, reduces
  with warp XOR shuffles, then reduces eight warp sums with another XOR tree.
- DS4 assigns thread `t` the scalar indices `t, t+256, ...` and uses a
  shared-memory binary tree with strides 128 through 1.
- Routed gate/up/down projections instead use Q4_K weights and runtime Q8_1
  activations.

The 164 unequal router logits are consistent with their differing F32 reduction topologies,
but that attribution is not yet conclusive because the
router has not been run through an equivalent controlled one-row microscope.
The top-k IDs are exact and only two of ten weights differ, so router work
remains lower priority than the routed-expert reduction.

## Routing-weight versus expert-arithmetic contribution

The comparator replays the captured raw down projections and weights with
explicit slot-order binary32 multiplication and addition. It fails if replay
does not reproduce every stored weighted value and routed-sum value.
Both runtimes replayed bit-exact:

- weighted values: 30,720 / 30,720 each
- routed-sum values: 3,072 / 3,072 each

Relative to the Poolside routed sum:

| Counterfactual | Unequal rows | Max absolute delta | RMS delta | L2 delta |
|---|---:|---:|---:|---:|
| DS4 weights, Poolside expert outputs | 1851 / 3072 | `7.450580596923828e-09` | `8.133701849961412e-10` | `4.508155154480043e-08` |
| Poolside weights, DS4 expert outputs | 2682 / 3072 | `1.1175870895385742e-08` | `1.8270042716640243e-09` | `1.0126285517335878e-07` |
| DS4 weights and expert outputs | 2690 / 3072 | `1.4901161193847656e-08` | `1.931828639288176e-09` | `1.0707281135244065e-07` |

Expert arithmetic is the larger contribution in this capture: its L2 delta
is about 2.25 times the routing-only delta. Both contributions are measurable.

## Verification and residual limits

- The tracked 512+1 Poolside producer captures FFN norm, router, all F32 expert
  boundaries, routed sum, shared expert, combined MoE, and final logits.
- The ordered comparator covers every observable boundary and explicitly
  records the internal Poolside down-input Q8_1 boundary as unavailable.
- The comparator rejects stale manifests, altered captures, replay failures,
  selected-expert mismatches, direct-oracle mismatches, and microscope origins
  that are not the first routed mismatch.
- Release-linked DS4, hook-null DS4, and hook-active DS4 token-513 logits are
  bit-exact.
- The generic microscope matches both pinned topology outputs.
- The raw tensor bundles are scratch artifacts rather than Git blobs; their
  hashes and exact producers are durable.
- The release/null/active verdicts are content-addressed in the run envelope,
  but their raw console logs are not committed; the tracked probe reruns them.
- The restored long-prefill production paths are exercised by the pinned
  512-token case. A CUDA regression sweep over ragged token counts, nonzero
  starting positions, and cache wrap remains separate hardening work.
- Router arithmetic attribution remains a hypothesis pending a router-specific
  reduction microscope.

The actionable implementation target is therefore narrow: a
Poolside-compatible Q4_K/Q8_1 reduction topology for the routed expert
projection. Router parity should be measured separately rather than folded
into that kernel change.
