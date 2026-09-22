# GLM-5.3 (355B-A32B MoE) Production Inference on Ascend 910B with SGLang

Production-grade serving of the full **GLM-5.3** model (GlmMoeDsa, 256 routed experts, DSA sparse attention) on a **4-node Ascend 910B cluster** (32 NPUs, 64GB HBM each), built on SGLang with DeepEP expert parallelism, DP-attention, W4A8C8 weight-only-plus quantized weights, and NEXTN (MTP) speculative decoding.

**Sustained measured throughput: 1,552 tok/s aggregate decode at 1,024 concurrent requests; 948 tok/s at 256 concurrent** — perfectly linear per-DP-group scaling from batch 1 to 256, saturating only on the slot+KV+dispatch-token triad, not compute.

```
              ┌─ DP group 0 (node 1, TP8 attention) ─┐
 clients ──►  │  ... x4 DP groups ...                │  ──  EP32 MoE (DeepEP)  ──►  8 experts/card
              └─ DP group 3 (node 4, TP8 attention) ─┘        across all 32 NPUs
```

## Why this exists

GLM-5.3 (GlmMoeDsa) shipped with no Ascend serving recipe. The stock `quay.io/ascend/sglang` image runs its dense-attention paths, but a production deployment needed three things the image didn't ship:

1. **Cross-node expert parallelism**: stock DeepEP's normal-mode dispatch (`aclnnNotifyDispatchA2`) tiling-fails across nodes; we patch `StrategyMap` with two mode combinations that route prefill through pure-HCCL alltoallv and decode through the vendored low-latency ops (see [`deep-ep-patch/`](deep-ep-patch/)).
2. **W4A8C8 quantized MoE fast-path**: the stock grouped-matmul path reads the full expert weight set per token (~1.7 ms/op); our patch expands int4→int8 for the w2 GEMM and pairs it with a 2-D bf16 scale, taking the skip-empty-group fast kernel (63 µs) — env-gated, off by default (`W4A8_W2_INT8=1`).
3. **DP-attention × EP topology**: 4 attention copies (per-node TP8, zero cross-node attention comm) + one MoE copy (EP32). This mirrors the official vLLM-A2 recipe; SGLang's `a2a=none` path cannot express it (EP must equal TP).

## Repository layout

| Directory | Contents |
|---|---|
| [`deployment/`](deployment/) | One-shot 4-node launcher (`deploy_ep32_dp4.py`), full flag-by-flag rationale |
| [`sglang-patches/`](sglang-patches/) | 35 per-file diffs against the stock `ascend-sglang` tree (apply with `patch -p1` inside the image's sglang source) |
| [`deep-ep-patch/`](deep-ep-patch/) | Patched `deep_ep/ep_strategy.py` — mount over the container's site-packages copy (or bake into the image) |
| [`benchmark/`](benchmark/) | Concurrency scaling harness, SSE step-time prober, GSM8K eval |
| [`quantization/`](quantization/) | safetensors header-only dtype-coverage verifier — proves quantization actually compressed the artifact (EXIT=0 is not proof) |

## Hardware & software baseline

- 4 × Atlas 800T A2 (Kunpeng aarch64), 8 × 910B NPUs (64GB) per node, RoCE interconnect
- Base image: `quay.io/ascend/sglang` CANN 9.0.0 910b daily (`main-cann9.0.0-910b`, sglang 0.5.19.dev)
- Weights: GLM-5.3 W4A8C8 (Eco-Tech release or msmodelslim-quantized), ~390GB
- deep_ep: vendored in the image; patched as above

## Quick start

```bash
# 1) Pull the base image — 910B is aarch64, x86 hosts MUST specify the platform
docker pull --platform linux/arm64 quay.io/ascend/sglang:main-cann9.0.0-910b
docker image inspect <img> --format '{{.Architecture}}'   # must be arm64

# 2) Bake the overlay: sglang patches + deep_ep patch into the image, or
#    bind-mount the patched tree at runtime (see deployment/README.md)

# 3) Launch (on each node, with its rank):
python3 deployment/deploy_ep32_dp4.py <rank 0-3>   # rank0 = master node

# 4) Wait for READY (~5-6 min: ~4 min weight load + graph capture), then
curl http://<master>:8077/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"glm-5","messages":[{"role":"user","content":"Hello"}]}'
```

## Measured performance

Concurrency scaling (512-token streaming responses, barrier-synced, temp=0):

| Concurrency | Aggregate tok/s | Notes |
|---|---|---|
| 1 | 2.1 | single-stream baseline (accept ~2.5-3.2/4) |
| 32 | 60.6 | |
| 128 | 235.1 | |
| 256 | 457.5 | slot+KV triad starts binding |
| 512 | 861.4 | |
| 1024 | **1552.5** | peak measured |
| 2048 | 1572.5 | saturated; queueing does not degrade throughput (latency doubles) |

Single-stream by task type: essay 32-33 tok/s, math 39-41 tok/s (NEXTN 3/1/4, accept length 2.5-3.2 of 4).

## Key engineering findings (all measured, 2026-09)

These are the load-bearing constraints discovered the hard way; details in [`docs/`](docs/).

1. **`a2a=none` (EP=1) is the only stock EP mode on Ascend; DeepEP forces EP==TP.** Cross-node MoE needs the patched deep_ep modes: prefill normal-mode = HCCL alltoallv (the `aclnnNotifyDispatchA2` fused op is single-node-only — tiling-fails cross-node at every chunk size we tested), decode low-latency = vendored `MoeLowLatencyDispatchV2/CombineV2` ops.
2. **`--dp 4 --enable-dp-attention` divides `max_running_requests` by DP internally** (moe_hook.py) — global 1024 = 256/group. Raising it requires mamba cache (≥mr/4 per group) and KV pool headroom in lockstep.
3. **The three memory walls for a 160k KV pool**: 950k→graph-capture OOM; 880k→HCCL lazy allocation failure (561000); 740k→first-request op workspace OOM (207001). 680k + ~5GB headroom is the stable point. (~49KB/token: KDA hybrid layers allocate latent buffers for all 45 layers.)
4. **`chunked-prefill-size` scales with per-card compute, not just halved from H800**: 910B bf16 ≈ 376T vs H800 ≈ 989T → 0.38× → 8192×0.38 ≈ 3100 → 2048 works. At 4096, `HcclAllToAllV` workspace (~global_batch×32×scale ≈ 1.6GB) OOMs against the ~1.5GB left after deep_ep's 32-rank C++ buffer eats ~20GB of the 21.7GB post-pool remainder.
5. **DDT (dispatch tokens per rank) caps at 256**: the aclnn `MoeDistributeDispatchV2` op's batchsize check. `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256`.
6. **`/health` 200 is not "serving"**: it never runs forward. First real request exercises prefill-only paths; always probe with a real generation before declaring READY.
7. **Overlap schedule wedged at first request on this build** (32 threads stuck at dp_attn sync behind in-flight forward) — disabled; serial has observable batch-by-batch progress and eager+overlap showed no gain on 910B.
8. **Draft (NEXTN) weights inherit the target model's quantization**: the full GLM-5.3 ckpt's layer.78 is W8A8 — passing `--speculative-draft-model-quantization unquant` makes the loader hunt for `weight_offset`/scale keys that don't exist (KeyError in deepseek_weight_loader). Omit the flag.
9. **NEXTN aclnn batch safety line is 1024**: 192/group × draft 4 = 768 is stable; mr=1024 pushes aicore timeout crashes.
10. **Cross-node shape divergence crashes with cryptic broadcast errors**: "AclNN 64 and 32 cannot broadcast" = one node was not synced to the latest config. Verify `docker inspect .Config.Cmd` on ALL nodes before firing.

## What is NOT included

- GLM-5.3-Flash (`glm5_next`) porting work — a separate, still-experimental line (output-quality issues unresolved as of 2026-09).
- Model weights. W4A8C8 GLM-5.3 from ModelScope (`Eco-Tech/GLM-5.3-w8a8c8`) or quantize yourself with msmodelslim (see `quantization/`).
- The locally-built docker image referenced by the launcher (16.6GB); rebuild via the patches here against the public base image.

## License

Apache-2.0 (matching SGLang). The patches in this repo are derived from SGLang's Apache-2.0 sources.
