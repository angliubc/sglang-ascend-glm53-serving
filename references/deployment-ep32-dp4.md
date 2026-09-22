# Deployment: GLM-5.3 EP32 × DP-attention × NEXTN on 4×910B nodes

The launcher `deploy_ep32_dp4.py` is the converged result of **18 ignition
attempts** (boot#1–#18). Each flag below carries a measured failure it prevents.
Read this before trimming anything.

## Topology

```
4 nodes × 8 × 910B (64GB) = 32 NPUs, RoCE interconnect
  tp=32, dp=4, --enable-dp-attention, --moe-a2a-backend deepep
  → attention: 4 DP groups, each TP8 WITHIN one node (zero cross-node attn comm)
  → MoE: one EP32 copy (per-card 8/256 experts)
  → NEXTN speculative (3 steps / topk 1 / 4 draft tokens)
API: rank0 node :8077, model name "glm-5"
```

## Launcher mechanics

`python3 deploy_ep32_dp4.py <rank 0-3>` (rank0 = master):

- **Self-contained**: the standard Atlas 800T A2 (910B) device mappings
  (`/dev/davinci0-7`, `/dev/davinci_manager`, `/dev/hisi_hdc`,
  `/dev/devmm_svm`) and driver binds (driver/firmware/ascend_install.info)
  are inlined in the launcher — identical on every node, no container-
  template JSON needed. The internal-only binds from the reference deployment
  (`/common`, queue_schedule, per-model patch mounts) are deliberately
  excluded.
- Removes any stale container, waits for it to fully exit (30×1s poll —
  leftover HBM contexts from a half-dead container cause the next launch to
  fail with "no GPU memory for KV cache").
- Fires `docker run` with 3 retries.
- Launch order: all four nodes in quick succession is fine (TCPStore waits);
  workers may start before rank0.

## The flag-by-flag rationale

| Flag | Value | Why (measured failure it prevents) |
|---|---|---|
| `--tp 32 --dp 4 --enable-dp-attention` | | The topology. dp×attn_tp=8=node-local attention. |
| `--moe-a2a-backend deepep` | | Cross-node EP. `a2a=none` caps EP at 1. |
| `--quantization modelslim` | | W4A8C8 weights carry their own scheme descriptors; this tells the loader to honor them. |
| `--mem-fraction-static 0.87` | | Higher with EP32: deep_ep's 32-rank C++ buffer (~20GB of the 21.7GB post-pool space) must fit. 0.92 (no-EP value) OOMs. |
| `--max-total-tokens 160000` | | Per-DP-group KV pool. The three walls (910B, this arch): 950k→capture OOM / 880k→HCCL 561000 / 740k→first-request 207001. |
| `--context-length 96000` | | Must be ≥ the longest request you accept; requests longer than this are rejected cleanly. |
| `--max-running-requests 768` | | Global; internally ÷4 per DP group = 256/group. **aclnn batch safety line is 1024** (192/group × draft 4 = 768 stable; 1024 → aicore timeout crashes). |
| `--max-mamba-cache-size 256` | | Must be ≥ mr/dp (256) else `aclnnInplaceCopy` broadcast crash. |
| `--chunked-prefill-size 2048` | | Scaled from H800's 8192 by compute ratio (376T/989T≈0.38×→~3100→2048). At 4096: `HcclAllToAllV` workspace ~1.6GB > the ~1.5GB left → 207001. |
| `--watchdog-timeout 900` | | First-request triton JIT runs minutes; default 300 kills a healthy first run mid-sampler. |
| `--page-size 64` | | DSA KV pool hard-assert. |
| `--disable-radix-cache` | | Untested with dp-attention on this build; isolated the topology variable first. |
| `--disable-overlap-schedule` | | Overlap wedged at first request (32 threads stuck at dp_attn.py:452 sync behind in-flight forward; no Decode line for 35 min, watchdog never fired). No measured gain on 910B anyway. |
| `--cuda-graph-max-bs-decode 16` | | Speculative decode graphs only. |
| `--speculative-algorithm NEXTN` 3/1/4 | | The ckpt's built-in MTP layer (layer.78). Higher steps didn't pay: accept 2.5-3.2/4. |
| *(no)* `--speculative-draft-model-quantization unquant` | | The full-ckpt MTP layer is W8A8 — `unquant` draft build KeyErrors on `weight_offset` in deepseek_weight_loader. Let draft inherit modelslim quant. |
| `--reasoning-parser glm45 --tool-call-parser glm47` | | GLM-5.3's native think/tool-call format. Without them, tool calls leak into content. |
| `--skip-server-warmup` | | Warmup exercises a prefill path before you've watched the boot; do manual text probes instead. |

Environment (set on the docker line, see launcher):

| Env | Value | Why |
|---|---|---|
| `DEEP_USE_MODE=alltoall_default` | | Patched deep_ep mode: prefill normal = ALLTOALL (HCCL alltoallv, cross-node-safe), decode LL = DEFAULT (deep_ep C++ native + vendored `MoeLowLatencyDispatchV2`). |
| `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` | 256 | aclnn `MoeDistributeDispatchV2` batchsize cap. |
| `TASK_QUEUE_ENABLE=1` / `HCCL_OP_EXPANSION_MODE=AIV` | | Ascend engine scheduling defaults for serving. |
| `ASCEND_CUSTOM_OPP_PATH` | `.../opp/vendors/customize` | Vendored custom ops (deep_ep's hwcomputing kernels). |
| `TRITON_CACHE_DIR` | mounted path | Else every restart re-JITs KDA kernels (~5 min). |

## The boot chain that produced this config

Abbreviated; full comments are in the launcher docstring.

- **#1**: `--language-model-only` rejected — GlmMoeDsa is a pure LM (flag is for
  mm archs). Removed.
- **#2**: `ascend_fuseep` backend died at first request — its fused ops
  (`aclnnDispatchFFNCombine`/`fused_deep_moe`) are absent from CANN 9.0.0's
  libopapi. Stock deepep's `MoeLowLatencyDispatchV2/CombineV2` ARE vendored →
  use stock + patch.
- **#4/#5**: `aclnnNotifyDispatchA2` tiling-fails cross-node (both 512/组和
  256/组 chunks) → the A2 fused op is single-node-only → normal-mode must be
  ALLTOALL.
- **#6/#7/#8**: chunked-prefill 8192/4096 → 207001 OOM in `HcclAllToAllV`
  workspace; 2048/1024 passed. Compute-ratio scaling confirmed.
- **#9**: KV cut 100k→40k fixed the OOM but wedged at first request with
  overlap schedule on → `--disable-overlap-schedule`.
- **#10**: `DEEP_USE_MODE=alltoall` — prefill crossed 78 layers' alltoallv
  fine, but LL=alltoall decode = **0.05 tok/s** (~300 cross-node collectives
  per decode step) → patched StrategyMap adds mixed modes.
- **#11**: DDT 1024 → `MoeDistributeDispatchV2` "batchsize is invalid" (first
  LL call) → 256.
- **#15**: one node unsynced (mr=128 vs 256) → "AclNN 64 and 32 cannot
  broadcast". Always md5 + Cmd spot-check all four nodes before firing.
- **#14–#18**: throughput ladder to the 1552 tok/s ceiling; perfectly linear
  per-DP-group to 256/group, ceiling = slot+KV+DDT triad.

## Readiness protocol

```bash
# health (poll the CODE, not the body — this build returns 200 with empty body)
until [ "$(curl -s -o /dev/null -w '%{http_code}' http://<master>:8077/health)" = 200 ]; do sleep 15; done

# REAL probe — /health never runs forward
curl http://<master>:8077/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"glm-5","messages":[{"role":"user","content":"写一篇约500字的短文"}],"max_tokens":512}'
# healthy single-stream ≈ 32-41 tok/s (usage.completion_tokens / wall)
```

READY reference: weight load ~4 min + verify-graph capture (29-32s) + draft
graphs (~8s) ≈ 5-6 min total.

## Scaling measurements

See repo root README. Reproduce with:

```bash
python3 benchmark/bench_ep32.py 1 2 4 8 ... 1024   # concurrency ladder
```

## Rollback

Keep the previous production launcher untouched (this repo's launcher was
developed alongside a `deploy_glm53_tp32_noep.py` monolithic config; the EP32
line was the experiment, TP32-noEP the rollback). On failure: `docker rm -f
glm53-ep32dp4` on all nodes, re-fire the known-good launcher.
