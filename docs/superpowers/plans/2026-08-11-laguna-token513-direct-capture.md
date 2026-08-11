# Laguna token-513 layer-1 direct capture

Status: causal diagnosis complete on NVIDIA GB10 CUDA.

This report records the direct DS4 observer run requested after the Laguna C7
repair.  Comparisons are bitwise and follow the semantic execution DAG.  They
are not sorted by error magnitude.

## Pinned run identity

- Model: `laguna-s-2.1-Q4_K_M.gguf`
  - bytes: `68248759648`
  - SHA-256: `e163b2c98908809a71245d6bb68b2226994d9969cb2a438eccb72196a1c4147a`
- Prefix: first 512 tokens of the pinned `swa-513` case
  - bytes: `2048`
  - SHA-256: `569aa6394783e0f17558db421ba26480d7a530d44dd2219bcc9aac2c09a3b559`
  - resume token: `3612`
- Device: NVIDIA GB10, compute capability 12.1, driver 580.126.09
- Poolside source commit: `04b2b72cb54048ead292884adbe11f284e3ec950`
- DS4 capture implementation commit: `6718196`
- Ordered comparator commit: `ad0b9a9`
- Capture probe binary SHA-256:
  `3ba5b42be2064fe4cb21e353ee3f37fa58f86f752d5bdd588098f34671d9ef8e`
- Microscope binary SHA-256:
  `6dd45acc7ab64209672d7eb62300697a36ffb40a18a55989ddd34ac3bcd96212`
- Release DS4 binary SHA-256:
  `cabd5bdb4d4aab8e38413e200dfa3c2d8ed5aa07c42903a2e0e460185694b7d7`

The probe ran under an `env -i` environment with only `PATH`, `HOME`, and
`LC_ALL`.  The coder, embedding, reranking, DS4 server, and ASR services were
all inactive and `nvidia-smi` reported no compute processes before each CUDA
run.

Raw artifacts on the capture host at the time of this report:

- Poolside MoE: `/tmp/poolside-laguna-resume513-moe.9nnWwy`
- Poolside layer input: `/tmp/poolside-laguna-resume513-layer1.QYbh7H`
- Accepted DS4 capture: `/tmp/ds4-laguna-direct513-moe-capture2`
- Rejected DS4 capture: `/tmp/ds4-laguna-direct513-moe-capture1`

The durable ordered result is
[`token513-layer1-comparison.json`](../../../tests/oracle-producers/laguna-c7/token513-layer1-comparison.json).
Its SHA-256 is
`7d67d8c1cd9e4b7617e19a06c00597361a3998ff147db5302d52efb6d5543c82`.

## Acceptance gates

The active observer and a control session produced byte-identical token-513
logits.  The comparison then required both pre-expert operands to match the
generic microscope fixture:

| Operand | Poolside | DS4 | Result |
|---|---|---|---|
| FFN-normalized F32 input | `eabe89d1...85ef0e` | `eabe89d1...85ef0e` | byte-exact |
| Q8_1 activation | fixture `8df5b6ef...d7de0a` | `8df5b6ef...d7de0a` | byte-exact |

An initial capture was rejected before router analysis because its FFN input
hash was `a166f996...a94808`.  The recovered worktree had omitted two existing,
uncommitted parity changes: generalized Poolside Q8 Stream-K prefill and the
long AUTO MMA64 attention path.  Restoring those exact changes produced the
accepted input hashes above.  The rejected capture is not evidence for any
downstream arithmetic claim.

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
| down | 4384 / 30720 | 26336 | slot 0, expert 144, row 0 |
| weighted down | 4222 / 30720 | 26498 | slot 0, expert 144, row 0 |
| routed sum | 382 / 3072 | 2690 | row 0 |
| shared expert | 3072 / 3072 | 0 | none |
| combined MoE | 664 / 3072 | 2408 | row 0 |

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

The generic microscope rerun reported:

```text
serial   bits=0xbdcdf5ef abs_error=2.3713312202744419e-08
poolside bits=0xbdcdf5ed abs_error=8.8121510088967625e-09
```

The direct capture and the generic one-row microscope therefore identify the
same two results from the same quantized operands.  Replacing a kernel and
watching logits improve is not needed for this causal claim.

## Router mechanism

The router is a separate primitive from the expert MMVQ:

- Router weight and input are F32; no Q4_K unpack or Q8_1 quantization occurs.
- Poolside's F32 MMVF assigns adjacent `float2` pairs to 256 threads, reduces
  with warp XOR shuffles, then reduces the eight warp sums with another XOR
  tree.
- DS4 assigns thread `t` the scalar indices `t, t+256, ...` and uses a shared
  memory binary tree with strides 128 through 1.
- Routed expert gate/up/down projections use Q4_K weights and runtime Q8_1
  activations instead.

Thus the router and expert differences are two concrete kernels, not one
shared Q4_K/Q8_1 helper.  They are members of the same broader numerical class:
reduction topology changes F32 rounding.  Router top-k remains stable here, so
router work is lower priority than the expert reduction.

## Routing-weight versus expert-arithmetic contribution

The captured raw down projections and weights were replayed with an explicit
slot-order F32 multiply and F32 add.  Replay reproduced all 30,720 weighted
values and all 3,072 routed-sum values bit-for-bit for both runtimes.

Relative to the Poolside routed sum:

| Counterfactual | Unequal rows | Max absolute delta | RMS delta | L2 delta |
|---|---:|---:|---:|---:|
| DS4 weights, Poolside expert outputs | 1851 / 3072 | `7.45058059692e-09` | `8.13370184996e-10` | `4.50815515448e-08` |
| Poolside weights, DS4 expert outputs | 2682 / 3072 | `1.11758708954e-08` | `1.82700427166e-09` | `1.01262855173e-07` |
| DS4 weights and expert outputs | 2690 / 3072 | `1.49011611938e-08` | `1.93182863929e-09` | `1.07072811352e-07` |

Expert arithmetic is the larger contribution in this capture (about 2.25x the
routing-only L2 delta), while both are measurable.

## Verification

- Host source and comparator contracts: 33 / 33 passed.
- Release DS4, capture probe, and generic microscope compiled with NVCC.
- All five binaries linking `ds4_cuda_test_hooks.o` linked against the CUDA
  test-hook object without unresolved or duplicate symbols.
- Direct probe: `PASS token=513 layer=1 ... nonperturbing=bit-exact`.
- Generic microscope: Poolside and DS4 topology bits both matched their pinned
  oracle values.

The non-perturbation check compares null capture with active capture inside the
test-hook cubin.  It does not claim a direct bitwise comparison between the
release cubin and the test-hook cubin; the release build is separately proven
to compile.
