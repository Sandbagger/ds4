# Laguna router microscope and quality proxies

Status: router row-0 cause established; Poolside-order MMVQ remains opt-in;
serial remains the default.

The independent F32 microscope reproduces the first captured router
inequality exactly: DS4 order produces `0xbea377b8`, while Poolside's
two-stage XOR order produces the direct Poolside bit `0xbea377ba`. This
explains the captured token-513, layer-1, row-0 scalar. It does not prove that
every one of the 164 unequal router logits has no additional cause, and it does
not justify changing the production router.

The broader evidence is mixed. On the 100-prompt model-quality proxy,
Poolside-order MMVQ has 51 per-case NLL wins versus 49 for serial, but its
token-weighted average NLL is 0.239% worse and its average target-prefix length
is 0.25 token lower. On the four-case capability proxy, both modes pass all
four selected questions and extract the same answers, while none of their four
reasoning outputs is byte-identical.

These results provide no basis to promote Poolside order or characterize
Laguna as a better planner. They also do not establish a general regression:
the aggregate changes are small, the case-level effects are mixed, and the
measurements come from proxies rather than a representative task population.

The durable structured result is
[`token513-router-quality-proxies.json`](../../../tests/oracle-producers/laguna-c7/token513-router-quality-proxies.json),
SHA-256
`01ad339b95312bb96c43efa0f2380d6d94329331ec79820051b8271acf23c72b`.

## Router microscope

The microscope uses byte-identical frozen operands:

- 3,072-element FFN-normalized input, SHA-256
  `eabe89d1d9a4bdc660e5759c2a20d347d4dedaec1e617a44f3244cfe7985ef0e`;
- row 0 of `blk.1.ffn_gate_inp.weight`, SHA-256
  `4b2e76f429c40ab67023a7500cd2eb25e0fd820de9d550025642b172452b16b1`.

| Reduction | Value | Bits | Absolute error versus sequential C binary64 |
|---|---:|---:|---:|
| DS4 production order | `-0.3192727565765381` | `0xbea377b8` | `1.7770e-8` |
| Poolside order | `-0.31927281618118286` | `0xbea377ba` | `7.7375e-8` |

Poolside's diagnostic path mirrors the pinned production topology: adjacent
`float2` products, 256 threads, XOR reduction within each warp, followed by an
XOR reduction of eight warp sums. Its result matches the direct Poolside graph
capture. The unchanged DS4 kernel matches the direct DS4 capture.

The DS4 value is closer to the sequential C binary64 calculation for this one
scalar. That is not a general accuracy or model-quality result. The fixture's
host calculation differs in the last binary64 bits
(`-0.31927273880611085` versus the C executable's
`-0.31927273880611073`), so neither value is described as an exact mathematical
oracle.

The production decision is therefore deliberately asymmetric: the microscope
closes the row-0 causal question, but the router arithmetic stays unchanged.
The selected ten experts were already exact in the direct capture, and the
observed DS4 scalar is not less accurate against the higher-precision
reference.

## 100-prompt model-quality proxy

`score_official` teacher-forces 100 tracked, at-most-24-token continuations
collected from hosted `poolside/laguna-s-2.1`. Both local runs use the same
Q4_K_M model, 4,096-token context, full residency, and direct CUDA execution.
Serial ran first and Poolside order second; each mode ran once.

| Measure | Serial | Poolside order | Difference |
|---|---:|---:|---:|
| scored target tokens | 2,342 | 2,342 | 0 |
| average NLL, lower is better | `0.237165561` | `0.237731846` | `+0.000566285` (`+0.239%`) |
| per-case total-NLL wins | 49 | 51 | Poolside `+2` |
| hosted first-match flags | 90 | 90 | identical flags in 100 cases |
| average teacher-forced greedy LCP | `11.090` | `10.840` | `-0.250` |

The case-level distribution is why the result is described as mixed. The
Poolside-minus-serial per-case total-NLL delta has mean `+0.013262` and median
`-0.005032`. Poolside improves LCP in two cases, serial improves it in seven,
and 91 tie. These curated prompts are not a random population, so no population
confidence claim is attached to those descriptive statistics.

The first-match row is an unchanged-path control, not candidate-quality
evidence. Prompt synchronization uses multi-token prefill, while the
experimental path requires exactly one token. Consequently, each case's first
scored position bypasses Poolside MMVQ; only the remaining 2,242 scored
positions exercise the candidate path. The TSV retains only a boolean
match-versus-hosted flag, not each predicted first-token ID.

Similarly, `greedy_lcp` is the prefix for which the local argmax at each
teacher-forced target position equals the hosted target. It is not a retained
free-running continuation. Poolside's hosted endpoint supplied no token
logprobs, so the zero-filled API columns are unavailable sentinels rather than
perfect agreement.

The serial and Poolside TSV SHA-256 values are respectively
`4e368fbdc0e93f475b28bf7d9ba4f4c4c58b0244a7b5290f3affe05fadfcaaae`
and
`247166cd9b3780a1fbef41ebe14c830f1776007b59290b372da4e51a54fff649`.
The structured result additionally binds the scorer binary, comparator,
manifest, and an aggregate hash over all 301 tracked fixture files.

## Four-case capability proxy

The second run uses `ds4-eval` with greedy decoding, no thinking mode, 32,768
context tokens, and a 512-token reply ceiling. It selects embedded case indices
`1,2,3,76`: the first GPQA, SuperGPQA, AIME, and COMPSEC cases. This is a smoke
selection, not a representative benchmark.

`ds4-eval` keeps the full suite denominator in its trace even when a sequence
is selected. The result is four requested, four attempted, four gradeable, and
88 not run. It must not be reported as a score out of 92.

| Case | Expected | Serial | Poolside order | Serial / candidate tokens | Output exact |
|---|---|---|---|---:|---:|
| GPQA `recNu3MXkvWUzHZr9` | B | B, pass | B, pass | 324 / 313 | no |
| SuperGPQA `001b51d76b4d422988f2c11f104a2c6c` | C | C, pass | C, pass | 103 / 114 | no |
| AIME `aime2025-01` | 70 | 70, pass | 70, pass | 297 / 277 | no |
| COMPSEC `compsec-076` | 17-20 | 20, pass | 20, pass | 298 / 345 | no |

The stable extracted answers are reassuring, but four between-mode output
changes with four stable answers cannot rank capability. They are consistent
with the reduction change being behaviorally active outside the frozen
token-513 probe. Each mode ran once, so there is no within-mode repeat to
separate that observation from run-to-run nondeterminism.

Wall time is not compared. Serial ran before Poolside order, startup cache
state differed, and the modes generated different token counts. Host RSS and a
point-in-time CUDA allocation likewise do not qualify memory or co-residency.
Both runs used full residency without SSD streaming.

The serial and Poolside trace SHA-256 values are respectively
`2a5ea28485570db376e89fc68e82c6cea80b2b056c2fa31955d63946cf37ba23`
and
`284c50088059f9cad45cd2d86fffb80d877968a61440504eb2175d6c7a994d66`.
Per-case hashes cover the exact byte count declared by each counted
`MODEL_OUTPUT` block.

## Scope and decision

This is not a planner/orchestrator evaluation. Neither proxy exercises tool
selection, candidate-plan comparison, code edits, tests, multi-turn state, or
orchestration, and neither compares Laguna with Flash. No coding-task verifier
was used. The full-residency numerical runs also say nothing about the intended
compact-runtime co-residency envelope.

Decision:

- keep serial as the default;
- keep `DS4_MM_VQ_REDUCTION=poolside` available as an opt-in experiment;
- leave production router arithmetic unchanged;
- make no Laguna-versus-Flash or planner-quality claim;
- require a verified coding/planner harness under the intended compact DS4
  runtime before either promoting Poolside order or making the later
  Laguna-versus-Flash replacement decision.

Raw TSVs, traces, stdout, stderr, and timing records are retained under
`/private/tmp/laguna-mmvq-analysis-20260811`. They are scratch artifacts rather
than Git blobs. The structured result binds the TSVs, traces, stdout, stderr,
binaries, and key manifests and records the reproduction commands; timing
records are retained only as non-comparable diagnostics.
