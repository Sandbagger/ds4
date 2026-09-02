# Laguna token-513 vector-attention oracle

This fixture is a narrow regression for the public
`ds4_gpu_laguna_store_attention_tensor` production wrapper. It fixes the first
long-context decode shape at which the pinned full-model oracle diverges:
position 512 with 513 causal keys, 48 query heads, 8 KV heads, and head
dimension 128.

The input is generated in the CUDA test from integer formulas recorded in
`manifest.json`; no model bytes are needed or tracked. Q uses an explicit
rounded F32 reciprocal followed by multiplication, matching the production
test binary's `-ffast-math` arithmetic without depending on an implicit compiler
rewrite. The expected 6,144 F32 values came from Poolside llama.cpp commit
`04b2b72cb54048ead292884adbe11f284e3ec950` on the NVIDIA GB10. The standalone
GGML graph used Q `[128,1,48,1]` and padded K/V `[128,768,8,1]` views with byte
strides `[2,2048,256]`. Runtime launch tracing proved AUTO selected
`flash_attn_ext_vec<128,1,F16,F16,false>` with grid `(1,3,48)`, block
`(32,4,1)`, followed by `flash_attn_combine_results<128>`.

Only rows 0 through 512 are logical. The 255-row padded tail is masked. The
wrapper stores the deterministic current K/V into row 512 before attention, so
the test covers the real KV-store-plus-decode boundary. Gate 100 is exactly
representable and amplifies the tiny reduction-order seam without changing the
attention topology. The frozen output is Poolside's ungated F32 output followed
by one round-to-nearest F32 multiplication by 100. `manifest.json` pins all
input hashes, producer/source/library identities, launch geometry, and the
promoted file hash.

This synthetic vector is a regression microscope, not a substitute for the
pinned Laguna full-model oracle. A green fixture only proves this wrapper shape;
`swa-513` remains authoritative for production qualification.
