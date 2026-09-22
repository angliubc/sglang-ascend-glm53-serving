# SGLang patches (Ascend 910B, GLM-5.3 GlmMoeDsa production tree)

35 diffs against the stock sglang tree inside
`quay.io/ascend/sglang:main-cann9.0.0-910b` (sglang 0.5.19.dev). Extracted
from the production serving tree that ran the EP32×DP4×NEXTN deployment
(2026-09). Filenames mirror the tree path with `/`→`__`.

Apply:

```bash
# extract stock tree from the image once per node
docker create --name probe quay.io/ascend/sglang:main-cann9.0.0-910b
docker cp probe:/sgl-workspace/sglang/python/sglang /data/models/sglang_full/
docker rm probe

# apply all patches (each file is a standalone unified diff, -p1 relative to
# the sglang package root)
cd /data/models/sglang_full/sglang
for p in <repo>/sglang-patches/*.patch; do patch -p1 --dry-run < $p || echo "SKIP $p"; done
# then without --dry-run for those that apply

# serve with the overlay mounted:
#   -v /data/models/sglang_full/sglang:/sgl-workspace/sglang/python/sglang
```

## Groups

### MoE W4A8 grouped-matmul fast path (the big perf win)
- `srt__hardware_backend__npu__quantization__moe_methods.py.patch` —
  int4→int8 expansion for w2 (K<N: nibble pairs expanded along N, lossless
  since int4 range is −8..7) paired with a 2-D bf16 scale, taking the
  skip-empty-group fast kernel: ~1.7ms → 63µs per op. Env-gated
  (`W4A8_W2_INT8=1`); w13 variant (`W4A8_W13_INT8=1`) as fallback for the
  int4+int64 fast kernel that fails in-server graph capture. Includes the
  microbenchmark-measured rationale in comments.
- `srt__hardware_backend__npu__moe__matmul.py.patch` — env-gated numeric
  sanity print (`W4A8_NUMCHECK=1`) on grouped-matmul outputs (skip during
  graph capture); used to validate the fast path on real weights.

### modelslim quantized-weight loading
- `srt__layers__quantization__modelslim__modelslim.py.patch` — 3 fixes:
  fused layer-level MoE descriptor keys (layer-level `*.weight` W8A8 entries),
  graceful fallback to unquantized BF16 MoE for appended (MTP/nextn) layers
  beyond the quantized set (instead of ValueError), and no-scheme → None
  return letting FusedMoE use its unquantized method.
- `srt__models__deepseek_common__deepseek_weight_loader.py.patch` — skip
  `kv_b_proj` post-load processing on hybrid layers that don't have one.

### Memory pools & KV cache (hybrid DSA/mamba architecture)
- `srt__mem_cache__memory_pool.py.patch` (+106 lines) — hybrid-arch pool
  changes incl. env-gated W4A8 experiment hooks.
- `srt__mem_cache__kv_cache_configurator.py.patch`,
  `srt__mem_cache__pool_host__{dsa,mamba,mla}.py.patch`,
  `srt__mem_cache__hybrid_cache__hybrid_pool_assembler.py.patch`,
  `srt__mem_cache__unified_cache__unified_tree_core.py.patch`,
  `srt__mem_cache__unified_radix_cache.py.patch`,
  `srt__mem_cache__index_key_cache.py.patch`,
  `srt__model_executor__pool_configurator.py.patch`,
  `srt__hardware_backend__npu__memory_pool_npu.py.patch` — the hybrid pool
  assembly for DSA sparse-attention + mamba hybrid models, radix-cache
  interplay, and per-arch pool sizing.
- `srt__layers__attention__hybrid_linear_attn_backend.py.patch` — forward
  metadata delegation for hybrid backends (the wrapper never produces
  metadata; the full-attn child owns it).
- `srt__hardware_backend__npu__attention__ascend_backend.py.patch` (+403/-74)
  — the largest patch: DSA/sparse attention NPU paths, NoPE guards
  (`rotary_emb is not None`), FIA rope-kwargs conditionals, per-mode forward
  routing.
- `srt__hardware_backend__npu__modules__deepseek_v2_attention_mla_npu.py.patch`
  — MLA NPU module fixes incl. verify-path paged attention.
- `srt__layers__attention__dsa__dsa_indexer.py.patch`,
  `srt__layers__communicator.py.patch`,
  `srt__layers__rotary_embedding__base.py.patch` — DSA indexer and
  communicator adjustments for the hybrid arch.

### Speculative decoding (NEXTN/EAGLE)
- `srt__speculative__eagle_worker_v2.py.patch`,
  `srt__speculative__spec_utils.py.patch`,
  `srt__models__deepseek_nextn.py.patch` (env-gated DSA debug),
  `kernels__ops__speculative__cache_locs.py.patch` — draft-chain fixes
  (0907/0910: cache-locs handling for spec slots).

### Serving / API layer
- `srt__entrypoints__openai__serving_chat.py.patch` (+99) — reasoning/
  tool-call output handling, SSE streaming behavior fixes (drip-feed when
  the client gateway over-buffers).
- `srt__entrypoints__http_server.py.patch` — streaming response flags.
- `srt__managers__tokenizer_manager.py.patch`,
  `srt__sampling__penaltylib__repetition_penalty.py.patch`,
  `srt__constrained__reasoner_grammar_backend.py.patch` (+41: minimum-think-
  tokens guard, env `SGLANG_MIN_THINK_TOKENS`, blocks empty-thinking turns
  that degenerate into verbatim context copying),
  `srt__constrained__xgrammar_backend.py.patch`,
  `srt__environ.py.patch` (the env definition for the above),
  `srt__configs__model_config.py.patch`,
  `srt__model_loader__weight_utils.py.patch`,
  `srt__hardware_backend__npu__utils.py.patch`.

## Verification discipline

Every patch above ran a real production traffic mix (multi-agent workloads,
concurrent chat, tool-calling). The env-gated experimental hooks
(`W4A8_*`, `SGLANG_DEBUG_DSA`) default OFF — the production path is the
ungated code.

Two independent gotchas worth restating:
- Patches were developed against a specific daily build; re-verify every
  hunk applies cleanly against your image before trusting a silent-fail
  `patch` (the `--dry-run` loop above prints SKIP for stale anchors).
- The tree contains no `models/glm5_next*` or KDA files — that's the
  separate experimental GLM-5.3-Flash line, deliberately excluded.
