---
name: sglang-ascend-glm53-serving
description: GLM-5.3 (GlmMoeDsa 256E) 在昇腾 910B 四节点集群上的 SGLang 生产推理：EP32×DP-attention×NEXTN 定型配方、跨节点 DeepEP 策略补丁、W4A8C8 MoE 快核补丁、35 文件生产树 patch 集、并发/精度基准、量化产物验证。当要在 910B/A2 NPU 上部署 GLM-5.3 或同族 GlmMoeDsa 架构、遇到 Ascend 跨节点专家并行/DeepEP 模式选择、W4A8 分组矩阵乘慢、KV 池三道墙、dp-attention 参数联动、或需要复现 1552 tok/s@1024 并发配置时使用。
version: 1.0.0
license: Apache-2.0
metadata:
  hardware: "4 × Atlas 800T A2 (Kunpeng aarch64), 8 × Ascend 910B 64GB/node, RoCE"
  software: "quay.io/ascend/sglang main-cann9.0.0-910b (sglang 0.5.19.dev, CANN 9.0.0)"
  weights: "GLM-5.3 W4A8C8 (~390GB), ModelScope Eco-Tech/GLM-5.3-w8a8c8"
  measured: "2026-09, 18 boot iterations, production traffic validated"
---

# GLM-5.3 on Ascend 910B — SGLang 生产推理 skill

## 事实卡（一眼定夺，全部实测）

| 项 | 值 |
|---|---|
| 拓扑 | tp32 / dp4 / enable-dp-attention / deepep；4 节点 × 8 NPU |
| 注意力 | 4 个 DP 组，每组节点内 TP8 → **跨节点零注意力通信** |
| MoE | EP32 单份（每卡 8/256 专家），DeepEP |
| 投机 | NEXTN 3/1/4（ckpt 内置 layer.78 MTP，**不加** `--speculative-draft-model-quantization unquant`） |
| 单流 | essay 32-33 / math 39-41 tok/s（usage 口径） |
| 聚合 | 256 并发 948 → 1024 并发 **1552 tok/s**（2048 饱和 1572，排队不降吞吐只升延迟） |
| READY | ~5-6 min（权重 ~4min + verify 图 29-32s + draft 图 ~8s） |
| 关键 env | `DEEP_USE_MODE=alltoall_default` + `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256` |

## 目录

| 路径 | 内容 |
|---|---|
| `scripts/deploy_ep32_dp4.py` | 定型启动器（读 docker-inspect 模板构造容器；18 次 boot 的收敛结果，flag 依据全在 docstring） |
| `patches/sglang/` | 35 个 per-file diff，对 stock `main-cann9.0.0-910b` 树，`patch -p1` 应用 |
| `patches/deep_ep/ep_strategy.py` | 跨节点 EP 策略补丁（drop-in 替换容器内 deep_ep） |
| `scripts/bench_ep32.py` | 并发阶梯（README 表的来源） |
| `scripts/bench_decode.py` | SSE 步时探针：median gap=步时，tokens/event≈accept len（A/B 必须同 prompt 同温度） |
| `scripts/gsm8k_eval.py` | 精度回归（吞吐数字抓不到 accept 率劣化） |
| `scripts/verify_quant_output.py` | 量化产物 dtype 覆盖审计（EXIT=0≠压缩成功） |
| `references/` | 深度文档：部署依据 / patch 分组说明 / deep_ep 判决 / 基准方法 / 运维坑 |

## 快速部署

```bash
docker pull --platform linux/arm64 quay.io/ascend/sglang:main-cann9.0.0-910b   # 910B=aarch64, 必须显式指定
# 抽 stock 树 → 应用 patches/sglang/*.patch → 放 deep_ep 补丁（详见 references/sglang-patches.md）
OVERLAY_DIR=/path/to/overlay WEIGHTS_DIR=/path/to/models python3 scripts/deploy_ep32_dp4.py <rank 0-3>  # 四节点各跑
# READY 判据：/health 200 之后必须再发真实生成请求（/health 不跑 forward）
```

## 核心结论（决策时直接引用）

1. **跨节点 EP 唯一可行模式 = prefill ALLTOALL（HCCL alltoallv）+ decode LL DEFAULT（deep_ep C++ 原生 + vendored MoeLowLatencyDispatchV2）**。stock 策略表只有均匀组合，均不可用：Normal=DEFAULT 的 `aclnnNotifyDispatchA2` 跨节点 tiling 必挂（单节点专用算子，512/组与 256/组 chunk 均实锤）；LL=ALLTOALL decode 实测 **0.05 tok/s**（每步 ~300 次跨节点集合通信）；LL=OPS 跨节点 DDT=256 下 aicore 0x26。→ `patches/deep_ep/` + `DEEP_USE_MODE=alltoall_default`。
2. **a2a=none 仅支持 ep=1；DeepEP 强制 ep==tp**。要表达 ep<tp / DP-attention 独立 EP 只有 deepep 路线。
3. **KV 池三道墙（GLM-5.3 hybrid 架构，~49KB/token，45 层全配 latent buffer）**：95 万→图捕获 OOM；88 万→HCCL 惰性分配失败(561000)；74 万→首请求算子 workspace OOM(207001)。68-16 万/DP组 + ~5G 余量 = 稳态。
4. **chunked-prefill 按算力比缩放，不是对半**：910B bf16 ~376T vs H800 ~989T = 0.38× → 8192×0.38≈3100 → 取 2048。4096 时 HcclAllToAllV workspace（~global_batch×32×scale≈1.6GB）打爆 deep_ep 32-rank C++ buffer 吃剩的 ~1.5GB → 207001。
5. **DDT 上限 256**（aclnn MoeDistributeDispatchV2 batchsize 检查）。mr、mamba cache、KV 三者联动：mr 全局 1024（内部 ÷dp=256/组，aclnn batch 安全线 1024，192/组×draft4=768 稳）；mamba cache ≥ mr/dp 否则 aclnnInplaceCopy broadcast 崩；每 slot 另需 ~620 tok KV。
6. **draft 量化跟随目标模型**：全量 ckpt layer.78 本身是 W8A8，加 `unquant` 会在 deepseek_weight_loader KeyError `weight_offset`。
7. **overlap schedule 在此 build 首请求 wedged**（32 线程卡 dp_attn.py:452 sync，watchdog 不触发）→ `--disable-overlap-schedule`；910B 上 eager+overlap 无实测收益。
8. **npu-smi HBM 对容器内分配全盲**；显存预检唯一可信口径 = 容器内 `torch_npu.mem_get_info`（引擎同款 API）。
9. **207001 ≠ 缺二进制**：多为算子内 OOM/tiling 失败误报。判别法：同 op+shape 在空闲 NPU probe 容器通过、在 serving 挂 → 查内存不查算子。
10. **多节点诊断先看对端**：CANN 异步栈指向的是下一个 sync 点的旁观者；跨节点 TP/DP 一节点死，幸存者挂收集通信的表象。`ASCEND_LAUNCH_BLOCKING=1` 拿准确栈（debug 专用，延迟×2）。

## 排障决策表

| 症状 | 首查 |
|---|---|
| /health 200 但首请求崩/挂 | prefill-only 路径；py-spy scheduler 看 JIT（`linalg_to_bin_enable_npu_compile`）vs 真挂；watchdog 900 |
| 首请求 OOM 207001 | KV 池尺寸（墙#3）、chunked-prefill workspace、TRITON_CACHE_DIR 持久化 |
| 图捕获 `capture_bs=[]` | 投机+dp-attention：decode 图 bs 必须为 attn_tp 倍数；running/graph <32 必死 |
| "AclNN X and Y cannot broadcast" | 某节点配置未同步 → 四节点 `docker inspect .Config.Cmd` 比对后再 fire |
| decode 0.0x tok/s | DEEP_USE_MODE 落到 LL=ALLTOALL；确认 patched deep_ep 挂载生效 |
| 首 token 即乱码 | W4A8 scale_bias/scale 处理路径（references/sglang-patches.md MoE 组）；`W4A8_NUMCHECK=1` 打数值 |
| kill 后重启报 no GPU memory | 旧容器 HBM 未释放，等完全退出（启动器内置 30×1s 轮询） |

## 边界（不含）

- **GLM-5.3-Flash（glm5_next 架构）移植线不在本仓库**——输出质量问题截至 2026-09 未定案，生产不可用。
- 不含模型权重；W4A8C8 用 ModelScope `Eco-Tech/GLM-5.3-w8a8c8` 或 msmodelslim 自量化（fused MoE 勿走 llm_ptq API，见 references/quantization-verification.md）。
- 910B(A2) **不支持 FP8/MXFP8**；可用 W8A8 / W4A8 / W4A16(AWQ/GPTQ)。

## 详细参考

- 部署 flag-by-flag 依据 + boot#1-#18 实验链：`references/deployment-ep32-dp4.md`
- 35 个 patch 分组与用途：`references/sglang-patches.md`
- deep_ep 跨节点策略判决：`references/deep-ep-cross-node-ep.md`
- 基准方法与参考数字：`references/benchmarks.md`
- 量化产物审计：`references/quantization-verification.md`
- 910B 运维坑全集：`references/ascend-910b-operational-gotchas.md`
