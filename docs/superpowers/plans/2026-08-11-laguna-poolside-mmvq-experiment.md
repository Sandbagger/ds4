# Laguna Poolside-order MMVQ experiment

Status: production implementation qualified as an opt-in experiment; the
default remains unchanged.

The narrow Poolside-order path succeeds at its direct objective. At frozen
token 513, layer 1, every routed gate, up, activation, rescale input, and raw
down-projection value is bit-exact to Poolside. Across a 32-token continuation,
it preserves every greedy token while removing 93.7% of DS4's excess
teacher-forced NLL relative to Poolside. On GB10, the combined execution of the
two affected kernels is 2.01 times faster in Nsight Systems, and the earlier
three-pair decode benchmark shows no throughput regression.

It is not promoted to the default in this change. The router remains different,
one frozen final-logit L2 result becomes 5.02% worse, and task-level
planner/orchestrator evaluation requires an external verified coding-task
harness that this repository does not currently provide.

The durable structured result is
[`token513-poolside-mmvq-experiment.json`](../../../tests/oracle-producers/laguna-c7/token513-poolside-mmvq-experiment.json),
SHA-256
`b295a8af6b6c5cc1094fb873f3b9d9d0901180ec809d6b6f7d30d2802a6d93ed`.

## Feature gate and scope

Set:

```text
DS4_MM_VQ_REDUCTION=poolside
```

The value is read once during CUDA initialization. Unset, empty, and unknown
values retain the existing serial reduction. The experimental path additionally
requires all of the following:

- one token;
- force-resident GLM Q4_K routed MoE;
- Q8_1 activation;
- 256 total experts, 10 selected experts;
- dimensions 3072 -> 1024 -> 3072.

Router projection, activation quantization, column-L2 rescaling, shared expert,
multi-token MMQ, and every default path are unchanged.

This is a shape gate, not a model-hash gate. An opt-in GLM Q4_K model with the
same expert shape would also use it. It also covers any one-token routed
invocation, including a one-token sync or checkpoint call, rather than carrying
an explicit decode-phase identity. That is acceptable for the experimental
environment switch but must remain explicit if this is later exposed as a
general runtime option.

## Pinned identity and controls

- Model: 68,248,759,648 bytes, SHA-256
  `e163b2c98908809a71245d6bb68b2226994d9969cb2a438eccb72196a1c4147a`.
- Prefix: 512 little-endian I32 tokens, SHA-256
  `569aa6394783e0f17558db421ba26480d7a530d44dd2219bcc9aac2c09a3b559`.
- Resume token: 3612 at position 512.
- Device: NVIDIA GB10, compute capability 12.1, driver 580.126.09.
- Pinned Poolside commit: `04b2b72cb54048ead292884adbe11f284e3ec950`.
- Final DS4 experiment commit: `1d2106150868aa4fcc32332f8da021fac24d99a8`.
- Final `ds4_cuda.cu` SHA-256:
  `3937d54a4c539b57e0973d2c6fca2d7f20cfa1cf38a8d8d102fb06566cc8512d`.

The final release-linked Nsight probes reproduced the accepted control logits:

```text
serial     e837bc5b6ad4dbfe74e17a1d4e7552eaae6756c46f30c4548c00d63c6e7d21c2
candidate  e077258890d64d369e81fa72940049f0b69a9293fb46bd92364a81ed90fff98f
```

All post-reboot GPU runs were exclusive. The Poolside oracle, DS4 serial
behavior, DS4 candidate behavior, serial profile, and candidate profile ran one
at a time, with an empty compute-process check between them. Peak DS4 residency
was 96,532 MiB. The ensemble GPU services remained inactive afterward.

## Direct numerical result

The router is deliberately unchanged:

| Boundary | Result versus Poolside |
|---|---:|
| FFN norm | exact |
| router logits | 164 / 256 unequal, L2 `1.1615e-6` |
| selected experts | exact 10 / 10 |
| routing weights | 2 / 10 unequal, L2 `5.3727e-8` |

The affected expert path changes as intended:

| Boundary | Serial unequal | Candidate unequal | Serial L2 | Candidate L2 |
|---|---:|---:|---:|---:|
| gate | 8,683 | 0 | `1.7964e-6` | 0 |
| up | 8,775 | 0 | `1.8131e-6` | 0 |
| SwiGLU | 9,490 | 0 | `1.3960e-7` | 0 |
| column L2 | 3 | 0 | `3.1610e-8` | 0 |
| down input | 9,443 | 0 | `2.8465e-2` | 0 |
| raw down | 26,336 | 0 | `2.8836e-7` | 0 |
| weighted down | 26,498 | 5,822 | `9.8731e-8` | `3.4728e-8` |
| routed sum | 2,690 | 1,851 | `1.0707e-7` | `4.5082e-8` |
| combined MoE | 2,408 | 1,393 | `1.1243e-7` | `5.5573e-8` |
| layer 1 output | 1,052 | 403 | `1.8313e-7` | `9.1098e-8` |

The candidate gate capture has the exact Poolside SHA-256
`5563a95ae9d46efe1e451414ae0134535a8ac483f39f35a6b3e2592b8e621140`.
The residual beginning at weighted down is precisely the unchanged routing
weight difference.

This improvement is not monotonic through the network. Layers 1 through 3
improve by roughly 40-50%; the pre-existing layer-4 bifurcation then dominates.
The candidate improves much of layers 33 through 45, but is worse at layers 46
and 47. Final-logit L2 is:

```text
serial     35.63575131923614
candidate  37.42493333725273  (+5.02%)
```

Top-1 at token 513 remains 8473 in Poolside, serial DS4, and candidate DS4.
This single final-logit result is why local arithmetic parity alone is
insufficient evidence for promotion.

## Behavioral result

The tracked Poolside producer generated 32 full-vocabulary logit vectors and a
strict-argmax continuation. Every vector was finite and every argmax reproduced
the emitted token. Its output-set SHA-256 is
`41328b85ebcdfc9563247c9430c03923a900f0de035714b4b2d70956a024061f`;
the continuation SHA-256 is
`24c8392c6936174264048072bcdef4c6ccfb79f26e8ddca7e5d03b9f7f31db07`.
The set hash consumes files in lexicographic basename order as binary records
`UTF-8 basename || NUL || raw SHA-256(file contents)`.

Both DS4 modes followed that entire continuation exactly:

| Measure | Poolside | Serial DS4 | Candidate DS4 |
|---|---:|---:|---:|
| greedy matching prefix | 32 | 32 | 32 |
| teacher top-1 matches | 32 | 32 | 32 |
| teacher NLL total | `0.0008953540` | `0.0010390626` | `0.0009044693` |
| excess NLL versus Poolside | 0 | `0.0001437086` | `0.0000091153` |
| top-20 set overlap, mean | 20 | 18.375 | 18.469 |
| top-20 set overlap, minimum | 20 | 12 | 15 |
| same-position top-20 IDs, mean | 20 | 8.688 | 8.219 |

The candidate removes 93.657% of serial DS4's excess NLL. It does not uniformly
improve top-20 order: mean set overlap rises slightly and the worst overlap
improves, while same-position agreement falls slightly. The defensible claim is
therefore closer Poolside teacher-forced likelihood on this frozen 32-step
continuation with unchanged greedy behavior, not statistical calibration or
universal rank parity.

The Poolside producer binary and shared-library SHA-256 identities are recorded
in the structured result. DS4 behavior metadata is retained locally with SHA-256
`cd7d6eae850821b753ffe01f2aa795166108e5141cb01ca030f5da97ccaf1dcb`.
The two behavior wall times are not compared because the second run had a much
warmer model-startup cache.

## GB10 performance

The earlier three paired `ds4-bench` runs reported:

| Measure | Serial median | Candidate median | Change |
|---|---:|---:|---:|
| steady decode | 1.56 tok/s | 1.62 tok/s | +3.85% |
| first token | 624.264 ms | 606.748 ms | -2.81% |

Their raw CSV files were on reboot-cleared scratch storage; the values survive
in the experiment transcript. The final, locally retained Nsight reports are
the stronger kernel-level evidence:

| Kernel | Mode | Launches | Grid / block | Mean | Total |
|---|---|---:|---|---:|---:|
| gate + up | serial | 47 | `4x10x1 / 256x1x1` | 339.088 us | 15.937 ms |
| gate + up | candidate | 47 | `1024x10x1 / 32x4x1` | 263.949 us | 12.406 ms |
| down | serial | 47 | `12x1x1 / 256x1x1` | 563.681 us | 26.493 ms |
| down | candidate | 47 | `768x1x1 / 32x4x1` | 185.751 us | 8.730 ms |

The affected path keeps exactly 94 launches per decoded token. Combined time
falls from 42.430 ms to 21.136 ms, a 2.007x speedup and 50.19% reduction.

Using the required complete Q4_K weight blocks, including their scales, `d`, and
`dmin`, effective payload throughput rises from 58.80 to 118.04 GB/s. This is
not a measured DRAM bandwidth figure: it cannot distinguish cache traffic from
DRAM traffic and omits activations, outputs, and other traffic from its
numerator.

Nsight launch records provide the following resource data:

| Kernel | Mode | Registers/thread | Static shared/block |
|---|---|---:|---:|
| gate + up | serial | 96 | 0 B |
| gate + up | candidate | 62 | 768 B |
| down | serial | 43 | 0 B |
| down | candidate | 48 | 1,536 B |

Actual achieved occupancy and hardware DRAM counters are unavailable because
Nsight Compute profiling is disabled for this user
(`RmProfilingAdminOnly` / `ERR_NVGPUCTRPERM`). No occupancy percentage is
inferred from launch resources alone.

The candidate down kernel still serializes ten expert slots with two block
barriers per slot. It reproduces Poolside's per-dot reduction order, not
Poolside's complete cross-expert scheduling topology. Despite that
correctness-first structure, it is substantially faster on GB10.

## Why the four-warp reduction changes the bits

The serial kernel assigns one CUDA thread to a row and accumulates its Q4_K
blocks linearly. The candidate distributes Poolside's Q4_K fragments across
four warps, adds warps 1-3 into warp 0 lane-by-lane, then applies the same XOR
tree at offsets 16, 8, 4, 2, and 1. Binary32 addition is non-associative, so
this parenthesization determines the result. The generic microscope and direct
graph capture prove the exact bit outcome for the frozen row.

This should be described as Poolside-compatible ordering, not as a generally
more accurate reduction. The frozen row happens to be closer to its FP64
reference, but one row cannot establish that property globally.

Poolside's internal runtime Q8_1 bytes remain unobserved. The controlled case
binds identical F32 input, pinned Poolside quantization semantics, captured DS4
Q8_1 bytes, and an exact Poolside arithmetic result. That is sufficient for
this production experiment, but it remains a provenance caveat rather than a
claim of dual-runtime byte capture.

The full production CUDA path is runtime-qualified by the pinned probes and
retained artifacts, not by an executable CI regression. CI enforces the source,
dispatch, and report contracts, but cannot replay the full-model comparison
without the Laguna weights and a compatible NVIDIA GPU.

## Decision and next work

Keep `DS4_MM_VQ_REDUCTION=poolside` available and opt-in. Do not change the
default yet.

The independent F32 router reduction microscope and the available DS4 quality
and capability proxies are now complete; see
[`2026-08-11-laguna-router-quality-proxies.md`](2026-08-11-laguna-router-quality-proxies.md).
The router row-0 inequality is explained by F32 reduction topology, while the
100-prompt result is mixed and the four-case capability smoke preserves answers
but changes every exact output. Neither proxy is a planner/orchestrator
evaluation. Promotion still requires an external verified coding-task harness
under the intended compact runtime, not solely token-513 parity.

Raw captures, behavior logits, JSON outputs, and Nsight reports remain under
`/private/tmp/laguna-mmvq-analysis.3n8mgM`. They are scratch artifacts rather
than Git blobs. The structured result records hashes for the key retained
outputs and profiles plus an aggregate Poolside behavior-set hash; it does not
hash every intermediate tensor. The tracked producers make the runs
reproducible.
