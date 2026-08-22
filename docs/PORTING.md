# Model porting & onboarding runbook

The tools that make onboarding a new model mechanical instead of
archaeological. Everything here is offline; only step 2 needs GPU time.

## The loop

1. **Pre-flight the GGUF** — validate metadata, tensor vocabulary, dims, and
   dtype recipe against a descriptor before any engine load.
2. **Capture staged activations** from ds4 and the reference runtime.
3. **Compare** captures stage-by-stage; the first diverging row localizes the bug.
4. **Pin a green capture** as a golden manifest so future rebuilds regress-check cheaply.
5. **Record quirks** in `docs/models/<name>.md`.

## 1. Descriptor pre-flight

Descriptors live in `gguf-tools/models/<name>.desc` (plain text: `[shape]`
uses exact `ds4_shape` field names, `[metadata]` pins required GGUF keys,
`tensors/dims` sections declare the weight vocabulary, `types.<recipe>`
declares dtype expectations, `names.hf` maps HF<->GGUF tensor names).

```sh
python3 gguf-tools/model_check.py check --desc gguf-tools/models/laguna-s21.desc \
    --gguf path/to/model.gguf          # exit 0=green 1=findings 2=usage
python3 gguf-tools/model_check.py dump --gguf model.gguf --tensors   # authoring aid
python3 gguf-tools/model_check.py map --desc <desc> \
    --gguf blk.9.attn_q.weight         # -> HF name, and --hf for reverse
```

Header-only parse: a 68 GB checkpoint checks in ~0.2 s. New model => copy an
existing desc, edit values, keep it pinned to the C profile via
`tests/test_model_check.py::EngineSyncTests`.

## 2. Capturing staged activations

Laguna dumps fire when the build has `DS4_TEST_HOOKS` and `DS4_LAGUNA_DIAG_DIR`
is set; detail-layer stages additionally need `DS4_LAGUNA_DIAG_LAYER=N`.
Both decode (`laguna_graph_forward_token`) and batch
(`laguna_graph_forward_batch`) paths carry the same checkpoint ladder;
embd/logits always dump when the dir is set.

```sh
DS4_LAGUNA_DIAG_DIR=/tmp/cap-ds4-l9 DS4_LAGUNA_DIAG_LAYER=9 ./ds4 -m model.gguf ...
```

Reference-side producers and the rigorous capture drivers live in
`tests/oracle-producers/` (`laguna-c7/` holds the poolside llama.cpp pair).
Observer perturbation is guarded upstream: probe builds require bitwise-equal
logits with diagnostics off/on.

Gotchas: dump dirs are created `O_EXCL` (a stale dir aborts the run — always
use a fresh path); `attn-o-proj` is skipped on legacy F16-attn recipes; the
MoE capture-buffer stages exist only under probe diagnostics.

## 3. Comparing captures

```sh
python3 tests/oracle-producers/portcheck.py compare \
    --reference /tmp/cap-ref --candidate /tmp/cap-cand            # strict
portcheck.py compare --reference R --candidate C \
    --max-rel-rms 0.04 --min-cosine 0.99                          # tolerance mode
portcheck.py compare --format json ...                            # machine report
```

Output is forward-pass ordered (`embd`, per-layer detail ladder, residual,
`logits`), one TSV row per file plus key=value summary:

```
exact=75	close=0	diverged=0	missing=0	extra=0	size=0	width=0
first_divergence=layer-09-q-proj.f32
verdict=red
```

Exit codes: 0 green, 1 red, 2 usage. Stage widths come from the manifest
(`stage-manifests/<model>.json`) — widths are expressions over shape params
with `heads` resolved per layer class, so a mis-sized stage reports `WIDTH`
instead of comparing garbage. `i32`/`q8_1` companions are byte-exact only.
Unknown files are counted as `unmatched` and never fail a run silently.

`compare_laguna_layers.py` remains for the fixed c7 fixture contract; prefer
portcheck for new work — it derives everything from the manifest.

## 4. Golden manifests

```sh
portcheck.py pin --capture /tmp/cap-green --out tests/oracle-producers/goldens/laguna-s21.json
portcheck.py check --capture /tmp/cap-rebuild --golden tests/oracle-producers/goldens/laguna-s21.json
```

Goldens store sizes + SHA-256 only (cheap to commit), so they catch rebuild
drift bit-exactly. Cross-runtime numeric comparison stays in `compare`.

## 5. Adding a new model checklist

1. Copy nearest `.desc`; fill `[shape]` from the C profile; extend
   `EngineSyncTests` if a new profile was added to ds4.c.
2. Copy nearest stage manifest; fix widths/rules; `portcheck.py stages` sanity-reads it.
3. Pre-flight every candidate GGUF with `model_check.py check`.
4. Port kernels behind red/green kernel tests; use capture+compare for graph-level parity.
5. Pin a golden once green; write `docs/models/<name>.md`.
