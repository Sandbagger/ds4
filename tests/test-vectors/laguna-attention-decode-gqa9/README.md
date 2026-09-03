# Laguna full-window GQA9 decode oracle

This fixture is a narrow, exact regression for the public
`ds4_gpu_laguna_store_attention_tensor` wrapper at the first Laguna sliding-
window boundary: position 512, `key_start=1`, 512 active keys, 72 query heads,
8 KV heads, and head dimension 128.

The source is the authenticated `512 + token 3612` full-model diagnostic against
Poolside llama.cpp commit `04b2b72cb54048ead292884adbe11f284e3ec950` on
NVIDIA GB10. Poolside retains absolute cache rows 0 through 512 and pads its
F16 attention view to 768 rows. Its active SWA window is rows 1 through 512;
row 0 and rows 513 through 767 are masked. DS4 stores those same active values
in a 512-row ring, with Poolside row 512 at DS4 row 0. The diagnostic preserves
the full Poolside physical cache and proves that the normalized active K/V
bytes equal DS4 byte-for-byte.

To keep the checked-in payload small, the fixture stores query head 0, KV head
0 across all 512 DS4 ring rows, gate head 0, and Poolside output head 0. The
device test repeats that authenticated stream across all 72 query heads and 8
KV heads. It poisons cache row 0, derives the current K/V input from the
captured post-store row, calls the production wrapper, then requires the full
cache and every output byte to match. This repeated-head construction tests the
four-partition reduction and ring-slot schedule, but not per-head diversity.

Runtime launch tracing pins Poolside's AUTO route to
`flash_attn_ext_vec<128,1,F16,F16,false>` with grid `(1,4,72)`, block
`(32,4,1)`, followed by `flash_attn_combine_results<128>`. The expected output
is captured, not recomputed with a host softmax. `manifest.json` pins all model,
tokenizer, source, binary, cache, extraction, and payload hashes.

This operation-level fixture cannot qualify the model. The promoted
`laguna-resident/swa-513.llama.f32` Poolside oracle remains authoritative.
