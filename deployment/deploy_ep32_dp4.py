"""EXPERIMENT (0911 ang, user-approved): 4-node DP-attention x EP32 for full GLM-5.3 (GlmMoeDsa, 256E).
Topology replication of the vLLM A2 recipe: attention 4 copies (per-node TP8,
zero cross-node attention comm) + MoE one copy (EP32 global, per-card 8/256
experts). Base = production canonical deploy_glm53_tp32_noep.py, changed ONLY:
  + --dp 4 --enable-dp-attention --moe-a2a-backend deepep
    (tree auto-forces ep_size -> tp_size(32) for spanning backends;
    ascend_fuseep was boot#2's pick but DIED on first request: its fused ops
    aclnnDispatchFFNCombine/fused_deep_moe are absent from this image's CANN
    9.0.0 libopapi AND deep_ep's vendored hwcomputing pkg -> use stock deepep
    path whose ops MoeLowLatencyDispatchV2/CombineV2 ARE vendored)
  + DEEP_USE_MODE=ops (0907 EP16 recipe: fixes w8a8 LL-layout garble)
  - EAGLE/MTP draft, HiCache(+env), radix cache  (untested combos w/ dp-attn,
    isolate the topology variable for ignition #1)
  - mem-fraction 0.92->0.87 (no effect: KV pool is bound by max-total-tokens,
    both boots allocated identical 99968-token/10.23GB pools)
  - max-total-tokens 100000->40000 (boot#9 root cause: deep_ep 32-rank C++
    buffer (nvl+rdma hints, auto mode) eats ~20GB of the 21.69GB left after
    pools -> ~1.5GB remains; HcclAllToAllV (attn reduce_scatter + MoE
    alltoallv) needs ~global_batch*32*scale ~= 1.6GB at chunk 4096 -> 207001.
    Empirical threshold: boots #4/#5 (global 2048/1024 = 800/400MB) PASSED
    reduce_scatter, #6/#7/#8 (8192/4096 >= 1.6GB) all died. Freeing 6GB of
    KV pool gives ~7.8GB headroom -> 4096 fits. Ignition test doesn't need
    100k-token KV)
  - chunked-prefill 4096->2048 (boot#10, user directive: scale from H800's
    8192 by COMPUTE not just halve - 910B per-card bf16 ~376T vs H800 ~989T
    = 0.38x -> 8192*0.38 ~= 3100 -> 2048; boots #4/#5 (global 2048/1024)
    also empirically passed reduce_scatter while 4096 OOM'd pre-KV-cut)
  + --disable-overlap-schedule (boot#9 verdict: OOM GONE w/ KV cut [32 ranks
    survived first forward for 35+min, AICore 94-97% busy] but wedged in
    first request: all 32 host threads stuck at dp_attn.py:452 sync
    all_gather queued behind in-flight forward; overlap's async enqueue lets
    scheduler race ahead into next sync while forward's collectives grind
    [JIT per-layer + idle/real rank shape divergence], no Decode line in 35min
    & watchdog 900s never fired. Serial loop = observable batch-by-batch
    progress; eager+overlap has no measured gain on 910B anyway)
  + DEEP_USE_MODE=alltoall -> alltoall_ops + patched deep_ep mount (boot#10
    verdict: prefill alltoallv WORKS cross-node [78 layers crossed, tokens
    streamed] but LL=ALLTOALL decode = 0.05 tok/s [gen throughput logged]:
    ~300 cross-node collectives per decode step. 0907 EP16 recipe's ops mode
    has the fast LL (aclnn MoeLowLatencyDispatchV2) but its Normal side
    (DEFAULT=aclnnNotifyDispatchA2) tiling-fails cross-node [boots #4/#5].
    Patched StrategyMap adds mode alltoall_ops=(Normal=ALLTOALL, LL=OPS):
    patched pkg at $OVERLAY_DIR/deep_ep_patched/deep_ep (see deep-ep-patch/ in this repo),
    mounted ro over container site-packages/deep_ep)
  + DDT 1024->256 (boot#11: LL=ops MoeDistributeDispatchV2 tiling-fail
    "batchsize is invalid" in DECODE path (first LL call ever reached); 0907
    EP16 PASS recipe ran DDT=256 + DEEP_USE_MODE=ops. The aclnn A2 op's
    batchsize check likely caps at 256)
    (boot#4/#5 first request died on aclnnNotifyDispatchA2 tiling ret=-1 at BOTH
    512/组 and 256/组 chunks -> batch<=256 hypothesis falsified; vendored A2 fused
    op likely single-node-only (0907 EP16 recipe was single-node PASS). DEEP_USE_MODE
    =alltoall swaps normal dispatch to alltoallv (pure HCCL, internode OK, no
    tiling limit); LL buffer asserts <=1024, decode-only under auto mode)
  - max-total-tokens 720000->100000 (ignition-conservative) (per-DP-group scheduler pool; KV per card
    is 4x production's since attn_tp=8 shards KV 8-way not 32-way)
  - NO --language-model-only: GlmMoeDsa is pure-LM (flag supports only mm archs
    like Glm5Next; boot #1 died on resolve_once ValueError without it)
Usage: OVERLAY_DIR=... WEIGHTS_DIR=... python3 deploy_ep32_dp4.py <rank 0-3>   (rank0 = master node)
Rollback: production = your known-good launcher (untouched).

THROUGHPUT LADDER (0911/0912, boot#14 -> #18, bench=/tmp/bench_ep32.py, 512-tok
streaming, barrier-synced):
  N=1:2.1  8:14.9  16:30.5  32:60.6  64:120.4  128:235.1  256:457.5
  512:861.4  1024:1552.5  2048(saturated):1572.5 tok/s  <-- ceiling
  PERFECTLY linear per-DP-group (batch 1->256, 2.1->440 tok/s/group, zero
  saturation); ceiling = slot+KV+DDT triad, NOT compute:
  1. max_running 256/group (=1024 global; moe_hook.py:397 divides by dp=4)
  2. KV 160000/group: batch256 x ~620tok = 0.82 usage at steady state
  3. DDT 256: decode dispatch token/rank caps at LL buffer 256 -> batch>256/组
     would exceed the SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256 that
     fixed boot#11's tiling failure. KV 320k would also not fit (pool-end
     avail mem only 15.5GB w/ 160k).
  LESSONS (boot#14-#18):
  - "all requests go to DP0" was a DIAGNOSIS BLIND SPOT: DP1/2/3 scheduler
    logs live in THEIR OWN nodes' containers, not .31's. round_robin works
    perfectly (8 probes -> 2/2/2/2 across groups).
  - boot#15 crash "AclNN 64 and 32 cannot broadcast" = .31 was NOT synced
    (old mr=128/mamba=32) while .32/33/34 had 256/64 -> dp-attention global
    tensor shapes diverge across groups. ALWAYS verify .Config.Cmd on ALL 4
    nodes before firing (md5 + Cmd spot-check).
  - raising max_running requires BOTH mamba cache (>= mr/4 per group, else
    aclnnInplaceCopy broadcast crash) AND KV pool (+~620tok per slot).
  - sustained throughput at saturation == peak (1572 vs 1552): queueing does
    not degrade the pipeline; latency doubles instead (p50 657s @ N=2048)."""
import json, os, pathlib, subprocess, sys, time

rank = int(sys.argv[1]); assert rank in (0, 1, 2, 3)

# Paths you must provide (env vars): CONTAINER_TEMPLATE = a `docker inspect`
# JSON of a known-good Ascend sglang container on this node (supplies the
# load-bearing device mappings + driver binds). OVERLAY_DIR = dir holding the
# patched sglang tree + deep_ep_patched/. WEIGHTS_DIR = parent of the model dir.
OVERLAY_DIR = os.environ.get('OVERLAY_DIR', '/data/models/mtp_experiment_0907')
WEIGHTS_DIR = os.environ.get('WEIGHTS_DIR', '/data/models')
original = json.loads(pathlib.Path(os.environ.get('CONTAINER_TEMPLATE', os.path.join(OVERLAY_DIR, 'running-node0-0907.json'))).read_text())[0]
name = 'glm53-ep32dp4'
subprocess.run(['docker', 'rm', '-f', name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
for _ in range(30):
    left = subprocess.run(['docker', 'ps', '-a', '--format', '{{.Names}}'],
                          capture_output=True, text=True).stdout.split()
    if name not in left:
        break
    time.sleep(1)
cmd = ['docker', 'run', '-d', '--name', name, '--network', 'host', '--shm-size', '32g', ]
hc = original['HostConfig']
if hc.get('Privileged'):
    cmd += ['--privileged']
for opt in hc.get('SecurityOpt') or []:
    cmd += ['--security-opt', opt]
for dev in hc.get('Devices') or []:
    cmd += ['--device', dev['PathOnHost'] + ':' + dev['PathInContainer']]
for mount in original['Mounts']:
    if mount['Type'] != 'bind':
        continue
    cmd += ['-v', mount['Source'] + ':' + mount['Destination'] + ('' if mount['RW'] else ':ro')]
for env in original['Config']['Env']:
    if env.split('=', 1)[0] in {'ASCEND_LAUNCH_BLOCKING'}:
        continue
    cmd += ['-e', env]
cmd += ['-e', 'ASCEND_CUSTOM_OPP_PATH=/usr/local/Ascend/cann-9.0.0/opp/vendors/customize']
cmd += ['-e', 'TASK_QUEUE_ENABLE=1']
cmd += ['-e', 'HCCL_OP_EXPANSION_MODE=AIV']
cmd += ['-e', 'SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256']
cmd += ['-e', 'DEEP_USE_MODE=alltoall_default']
cmd += ['-v', os.path.join(OVERLAY_DIR, 'deep_ep_patched/deep_ep') + ':/usr/local/python3.11.15/lib/python3.11/site-packages/deep_ep:ro']
cmd += ['-v', os.path.join(WEIGHTS_DIR, 'GLM-5.3-w8a8c8') + ':/model/GLM-5.3-w8a8c8:ro']
cmd += [original['Config']['Image'], 'python3', '-m', 'sglang.launch_server',
        '--model-path', os.environ.get('MODEL_PATH', '/model/GLM-5.3-w8a8c8'), '--served-model-name', 'glm-5',
        '--trust-remote-code', '--tp', '32', '--dp', '4', '--enable-dp-attention',
        '--moe-a2a-backend', 'deepep',
        '--nnodes', '4', '--node-rank', str(rank),
        '--dist-init-addr', '<MASTER_IP>:6310', '--quantization', 'modelslim',
        '--mem-fraction-static', '0.87', '--max-total-tokens', '160000',
        '--context-length', '96000', '--max-running-requests', '768',
        '--max-mamba-cache-size', '256', '--chunked-prefill-size', '2048',
        '--watchdog-timeout', '900', '--page-size', '64',
        '--disable-radix-cache', '--disable-overlap-schedule',
        '--cuda-graph-max-bs-decode', '16',
        '--speculative-algorithm', 'NEXTN',
        '--speculative-num-steps', '3',
        '--speculative-eagle-topk', '1',
        '--speculative-num-draft-tokens', '4',
        # v2 0902: unquant REMOVED -- GLM-5.3 full ckpt layer.78 is W8A8
        # (weight_offset/scale present); unquant-built draft has no such params
        # -> KeyError 'model.decoder.mlp.shared_experts.gate_up_proj.weight_offset'
        # in deepseek_weight_loader.py:323. Let draft follow modelslim quant like
        # the target model. (0907 Flash line's unquant used a separately
        # extracted bf16 layer45, different situation.)
        '--reasoning-parser', 'glm45', '--tool-call-parser', 'glm47',
        '--enable-strict-thinking',
        '--skip-server-warmup',
        '--host', '0.0.0.0', '--port', '8077']
for attempt in range(3):
    try:
        subprocess.run(cmd, check=True)
        break
    except subprocess.CalledProcessError:
        if attempt == 2:
            raise
        subprocess.run(['docker', 'rm', '-f', name],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2)
