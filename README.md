# sglang-ascend-glm53-serving

**GLM-5.3 (GlmMoeDsa, 256 routed experts, DSA sparse attention) production inference on Ascend 910B with SGLang** — a converged, measured, agent-consumable skill package: EP32 × DP-attention × NEXTN topology, cross-node DeepEP strategy patch, W4A8C8 MoE fast-kernel patches, benchmark harness, and quantization artifact verification.

Measured (4 × Atlas 800T A2, 32 × 910B 64GB, RoCE, 2026-09): **1,552 tok/s aggregate decode at 1,024 concurrent** (2.1 tok/s single-stream baseline), perfectly linear per-DP-group scaling, converged over 18 ignition iterations.

## This repo is agent-readable

The entry point is [`SKILL.md`](SKILL.md) — structured as a skill package (fact card → decision table → deep references). Any agent can load it directly:

- **Fact card**: topology, flags, env, throughput, READY time in one table
- **10 load-bearing conclusions** (each a measured failure mode, not a guess)
- **Triage decision table**: symptom → first check
- `scripts/`, `patches/`, `references/` follow skill-package conventions

## Layout

```
SKILL.md                     agent entry: fact card + decisions + triage
scripts/deploy_ep32_dp4.py   converged 4-node launcher (boot#1-#18 rationale in docstring)
scripts/bench_ep32.py        concurrency scaling ladder (the throughput table's source)
scripts/bench_decode.py      SSE step-time probe (median gap = step time, tokens/event ≈ accept len)
scripts/gsm8k_eval.py        accuracy regression (catches acceptance-rate losses throughput misses)
scripts/verify_quant_output.py  safetensors header-only dtype-coverage auditor (EXIT=0 ≠ compressed)
build.sh + Dockerfile       one-click image build (extract stock tree -> apply patches -> build)
patches/sglang/              35 per-file diffs vs stock main-cann9.0.0-910b tree (patch -p3, verified byte-exact reproduction)
patches/deep_ep/ep_strategy.py  cross-node EP strategy modes (drop-in replacement)
references/                  deep docs: deployment rationale / patch groups / deep_ep verdict / benchmarks / ops gotchas
```

## The one-paragraph version

Stock Ascend SGLang cannot serve GLM-5.3 across nodes at production throughput: DeepEP's normal-mode dispatch op (`aclnnNotifyDispatchA2`) is single-node-only, its LL=alltoall decode path measures 0.05 tok/s, and the W4A8 grouped-matmul path reads the full expert set per op (1.7 ms). This repo's three enablers: a 14-line `StrategyMap` patch mixing prefill ALLTOALL (HCCL alltoallv) with LL native ops; an int4→int8 w2 expansion + 2-D bf16 scale taking the skip-empty-group kernel (63 µs); and the DP4×TP8-attention + EP32-MoE topology with all the memory/slot/dispatch-token constraints mapped (KV pool three-wall budget, DDT=256 cap, mr/mamba/KV lockstep).

## Requirements

- 4 × 910B nodes (aarch64 — pull images with `--platform linux/arm64`), RoCE
- `quay.io/ascend/sglang:main-cann9.0.0-910b` (sglang 0.5.19.dev, CANN 9.0.0)
- GLM-5.3 W4A8C8 weights (ModelScope `Eco-Tech/GLM-5.3-w8a8c8`) — 910B does not support FP8

## License

Apache-2.0. Patches derive from SGLang's Apache-2.0 sources.
