# deep_ep patch: cross-node EP strategy modes for Ascend

`ep_strategy.py` — drop-in replacement for the `deep_ep` package vendored in
`quay.io/ascend/sglang:main-cann9.0.0-910b` (CANN 9.0.0). Adds two entries to
`StrategyMap.strategy_map`:

```python
("alltoall_ops"):     (NormalStrategy.ALLTOALL,    LowLatencyStrategy.OPS),     # 0911
("alltoall_default"): (NormalStrategy.ALLTOALL,    LowLatencyStrategy.DEFAULT), # 0911
```

The stock map only offers uniform (Normal, LL) pairs. On a 4-node EP32
deployment neither stock combination works:

- **Normal=DEFAULT** (`aclnnNotifyDispatchA2`): the fused A2 op is
  **single-node-only** — tiling-fails cross-node at every chunk size tested
  (512/组 and 256/组). Normal must be ALLTOALL (pure HCCL alltoallv).
- **LL=ALLTOALL**: decode runs ~300 cross-node collectives per step —
  measured **0.05 tok/s**. LL must use an op-based or native path.
- **LL=OPS** (`MoeLowLatencyDispatchV2`): works single-node, but
  aicore-exception'd (0x26) cross-node at DDT=256 in our runs (tiling passed,
  execution failed) — hence the `alltoall_default` combination (LL=DEFAULT =
  deep_ep C++ native, RDMA buffer + vendored ops) that production converged
  on with `DEEP_USE_MODE=alltoall_default`.

Everything else in the package is stock (verified: `__init__.py` identical,
only this file differs).

## Install

Mount over the container's copy (no image rebuild):

```
-v /data/deep_ep_patched/deep_ep:/usr/local/python3.11.15/lib/python3.11/site-packages/deep_ep:ro
```

Or bake into the image. Then set:

```
-e DEEP_USE_MODE=alltoall_default
-e SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256
```

## Provenance

Measured on 4 × Atlas 800T A2 (32 × 910B), GLM-5.3 W4A8C8, sglang 0.5.19.dev,
2026-09: the boot chain (#2 → #18) that converged on this mode is documented
in `deployment/README.md`. Throughput with this patch: 1,552 tok/s aggregate
decode at 1,024 concurrent.
