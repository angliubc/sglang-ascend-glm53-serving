"""4-node DP-attention x EP32 launcher for full GLM-5.3 (GlmMoeDsa, 256E) on Ascend 910B.

Topology: attention 4 copies (per-node TP8, zero cross-node attention comm)
+ MoE one copy (EP32 global, per-card 8/256 experts). Converged over 18
ignition attempts (2026-09); every flag carries a measured failure it
prevents — read the rationale in references/deployment-ep32-dp4.md before
trimming anything.

Self-contained: standard 910B device/driver mounts are inlined below (they
are identical on every Atlas 800T A2 node). No container-template JSON needed.

Usage (each node, rank 0-3; rank0 = master):
  IMAGE=sglang-ascend-glm53:ep32 \
  WEIGHTS_DIR=/data/models \
  TRITON_CACHE_HOST=/data/models/triton_cache \
  python3 deploy_ep32_dp4.py <rank> [master_ip]

  IMAGE             baked image from ../build.sh (recommended), or the stock
                    quay.io/ascend/sglang:main-cann9.0.0-910b (then also set
                    SGLANG_OVERLAY=<patched sglang tree> and
                    DEEP_EP_PATCH_DIR=<dir containing deep_ep/ep_strategy.py>)
  WEIGHTS_DIR       host dir containing GLM-5.3-w8a8c8 (default /data/models)
  TRITON_CACHE_HOST host dir for the persistent triton cache (default
                    /data/models/triton_cache; created if missing — else every
                    restart re-JITs KDA kernels, ~5 min)
  MODEL_NAME        served model name (default glm-5)
  PORT              API port on rank0 (default 8077)

Flag rationale (measured):
  --dp 4 --enable-dp-attention   THE topology. a2a=none caps EP at 1; DeepEP
      --moe-a2a-backend deepep   forces ep==tp; this combo = 4 node-local
                                 TP8 attention groups + 1 EP32 MoE copy.
  --mem-fraction-static 0.87     deep_ep's 32-rank C++ buffer (~20GB of the
                                 21.7GB post-pool space) must fit; 0.92 OOMs.
  --max-total-tokens 160000      per-DP-group KV pool. Three walls (GLM-5.3
                                 hybrid arch, ~49KB/token): 950k -> capture
                                 OOM; 880k -> HCCL 561000; 740k -> first-
                                 request 207001.
  --max-running-requests 768     global; internally /dp = 256/group. aclnn
                                 batch safety line 1024 (192/group x draft4
                                 = 768 stable; 1024 -> aicore timeout).
  --max-mamba-cache-size 256     must be >= mr/dp else aclnnInplaceCopy
                                 broadcast crash.
  --chunked-prefill-size 2048    scaled from H800's 8192 by compute ratio
                                 (910B ~376T vs H800 ~989T = 0.38x). At 4096:
                                 HcclAllToAllV workspace ~1.6GB > ~1.5GB left
                                 -> 207001.
  --watchdog-timeout 900         first-request triton JIT runs minutes;
                                 default 300 kills a healthy first run.
  --page-size 64                 DSA KV pool hard-assert.
  --disable-overlap-schedule     overlap wedged at first request on this build
                                 (32 threads stuck at dp_attn.py:452 sync);
                                 no measured gain on 910B anyway.
  --cuda-graph-max-bs-decode 16  speculative decode graphs only.
  --speculative-algorithm NEXTN 3/1/4: ckpt's built-in MTP layer (layer.78).
                                 Do NOT pass --speculative-draft-model-
                                 quantization unquant: the full-ckpt MTP layer
                                 is W8A8; unquant draft build KeyErrors on
                                 weight_offset in deepseek_weight_loader.
  --reasoning-parser glm45 --tool-call-parser glm47
                                 GLM-5.3 native think/tool-call format; without
                                 them tool calls leak into content.
  --skip-server-warmup           warmup exercises a prefill path before you
                                 have watched the boot; probe manually.

Env on the docker line:
  DEEP_USE_MODE=alltoall_default          patched deep_ep mode: prefill
                                         Normal=ALLTOALL (HCCL alltoallv;
                                         aclnnNotifyDispatchA2 tiling-fails
                                         cross-node), decode LL=DEFAULT
                                         (deep_ep C++ native + vendored
                                         MoeLowLatencyDispatchV2). LL=ALLTOALL
                                         measured 0.05 tok/s.
  SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256
                                         aclnn MoeDistributeDispatchV2
                                         batchsize cap.
  TASK_QUEUE_ENABLE=1 / HCCL_OP_EXPANSION_MODE=AIV
                                         Ascend serving defaults.
  ASCEND_CUSTOM_OPP_PATH                  vendored custom ops (deep_ep's
                                         hwcomputing kernels) — set by the
                                         base image; re-set defensively.

Throughput ladder (512-tok streaming, barrier-synced, bench_ep32.py):
  N=1:2.1  32:60.6  128:235.1  256:457.5  512:861.4  1024:1552.5
  2048(saturated):1572.5 tok/s  <- ceiling = slot+KV+DDT triad, NOT compute.
  Single stream: essay 32-33 / math 39-41 tok/s (accept 2.5-3.2/4).

Lessons (boot#14-#18):
  - DP1/2/3 scheduler logs live in THEIR OWN nodes' containers, not rank0's.
    round_robin works (8 probes -> 2/2/2/2 across groups).
  - "AclNN 64 and 32 cannot broadcast" = a node was NOT synced (old mr/mamba
    values) -> dp-attention global shapes diverge. Verify .Config.Cmd on ALL
    nodes before firing.
  - raising max_running requires BOTH mamba cache (>= mr/4 per group) AND KV
    pool (+~620 tok per slot).
  - sustained throughput at saturation == peak: queueing does not degrade
    the pipeline; latency doubles instead (p50 657s @ N=2048).
"""
import os
import subprocess
import sys
import time

rank = int(sys.argv[1]) if len(sys.argv) > 1 else -1
assert rank in (0, 1, 2, 3), "usage: deploy_ep32_dp4.py <rank 0-3> [master_ip]"

MASTER_IP = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("MASTER_IP", "10.0.0.1")
IMAGE = os.environ.get("IMAGE", "sglang-ascend-glm53:ep32")
WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR", "/data/models")
MODEL_DIRNAME = os.environ.get("MODEL_DIRNAME", "GLM-5.3-w8a8c8")
MODEL_NAME = os.environ.get("MODEL_NAME", "glm-5")
PORT = os.environ.get("PORT", "8077")
DIST_PORT = os.environ.get("DIST_PORT", "6310")
TRITON_CACHE_HOST = os.environ.get("TRITON_CACHE_HOST", "/data/models/triton_cache")
SGLANG_OVERLAY = os.environ.get("SGLANG_OVERLAY", "")  # patched tree, if IMAGE is stock
DEEP_EP_PATCH_DIR = os.environ.get("DEEP_EP_PATCH_DIR", "")  # dir with deep_ep/ep_strategy.py
CONTAINER = os.environ.get("CONTAINER", "glm53-ep32dp4")

# --- standard Atlas 800T A2 (910B) mounts — identical on every node ---------
DEVICES = [
    "davinci0", "davinci1", "davinci2", "davinci3",
    "davinci4", "davinci5", "davinci6", "davinci7",
    "davinci_manager", "hisi_hdc", "devmm_svm",
]
DRIVER_BINDS = [
    "/usr/local/Ascend/driver:/usr/local/Ascend/driver",
    "/usr/local/Ascend/firmware:/usr/local/Ascend/firmware",
    "/etc/ascend_install.info:/etc/ascend_install.info",
    "/usr/local/sbin:/usr/local/sbin",
]
# Internal-only binds from the reference deployment (/common, queue_schedule,
# per-model patch mounts) are deliberately NOT included.

os.makedirs(TRITON_CACHE_HOST, exist_ok=True)

# --- remove stale container, wait for full exit (HBM release) ---------------
subprocess.run(["docker", "rm", "-f", CONTAINER],
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
for _ in range(30):
    left = subprocess.run(["docker", "ps", "-a", "--format", "{{.Names}}"],
                          capture_output=True, text=True).stdout.split()
    if CONTAINER not in left:
        break
    time.sleep(1)

cmd = ["docker", "run", "-d",
       "--name", CONTAINER,
       "--privileged",                       # required for NPU visibility
       "--security-opt", "label=disable",    # ditto
       "--network", "host",
       "--shm-size", "32g"]
for d in DEVICES:
    cmd += ["--device", f"/dev/{d}"]
for b in DRIVER_BINDS:
    cmd += ["-v", b]
cmd += ["-v", f"{WEIGHTS_DIR}:/model"]
cmd += ["-v", f"{TRITON_CACHE_HOST}:/sgl-triton-cache",
        "-e", "TRITON_CACHE_DIR=/sgl-triton-cache"]
if SGLANG_OVERLAY:
    cmd += ["-v", f"{SGLANG_OVERLAY}:/sgl-workspace/sglang/python/sglang"]
if DEEP_EP_PATCH_DIR:
    cmd += ["-v", f"{DEEP_EP_PATCH_DIR}/deep_ep:/usr/local/python3.11.15/lib/python3.11/site-packages/deep_ep:ro"]

cmd += ["-e", "ASCEND_CUSTOM_OPP_PATH=/usr/local/Ascend/cann-9.0.0/opp/vendors/customize",
        "-e", "TASK_QUEUE_ENABLE=1",
        "-e", "HCCL_OP_EXPANSION_MODE=AIV",
        "-e", "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256",
        "-e", "DEEP_USE_MODE=alltoall_default"]

cmd += [IMAGE, "python3", "-m", "sglang.launch_server",
        "--model-path", f"/model/{MODEL_DIRNAME}",
        "--served-model-name", MODEL_NAME,
        "--trust-remote-code",
        "--tp", "32", "--dp", "4", "--enable-dp-attention",
        "--moe-a2a-backend", "deepep",
        "--nnodes", "4", "--node-rank", str(rank),
        "--dist-init-addr", f"{MASTER_IP}:{DIST_PORT}",
        "--quantization", "modelslim",
        "--mem-fraction-static", "0.87",
        "--max-total-tokens", "160000",
        "--context-length", "96000",
        "--max-running-requests", "768",
        "--max-mamba-cache-size", "256",
        "--chunked-prefill-size", "2048",
        "--watchdog-timeout", "900",
        "--page-size", "64",
        "--disable-radix-cache",
        "--disable-overlap-schedule",
        "--cuda-graph-max-bs-decode", "16",
        "--speculative-algorithm", "NEXTN",
        "--speculative-num-steps", "3",
        "--speculative-eagle-topk", "1",
        "--speculative-num-draft-tokens", "4",
        # NO --speculative-draft-model-quantization: see docstring.
        "--reasoning-parser", "glm45",
        "--tool-call-parser", "glm47",
        "--skip-server-warmup",
        "--host", "0.0.0.0", "--port", PORT]

for attempt in range(3):
    try:
        subprocess.run(cmd, check=True)
        break
    except subprocess.CalledProcessError:
        if attempt == 2:
            raise
        subprocess.run(["docker", "rm", "-f", CONTAINER],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2)

print(f"[rank {rank}] container up — watch: docker logs -f {CONTAINER}")
if rank == 0:
    print(f"readiness: curl http://{MASTER_IP}:{PORT}/health  (then a REAL generation —")
    print("            /health never runs forward; see SKILL.md triage table)")
