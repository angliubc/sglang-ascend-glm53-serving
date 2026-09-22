# Ascend 910B serving: operational gotchas (all measured)

Working notes from running production SGLang inference on 4×Atlas 800T A2.
Each item cost at least one debugging round; none are in any upstream doc
we could find.

## Health & readiness

- **`/health` 200 ≠ serving.** The health handler never runs forward. Both
  the DSA-CP assert and several OOM classes crashed only on the FIRST REAL
  request after a green health check. Always probe with a real generation.
- This build's `/health` returns 200 with an EMPTY body — poll the status
  code, never string-match "healthy".
- "no GPU memory for KV cache" right after killing a previous container =
  old HBM not yet released. Wait for the container to fully exit (the
  launcher polls 30×1s).

## Memory

- **npu-smi HBM-Usage is blind to in-container NPU allocations.** A
  container holding 40GB/card for 3 hours showed ~3.4GB "used" in npu-smi
  the whole time. The only trustworthy preflight is the driver view from
  inside a container:
  `docker exec <c> python3 -c "import torch,torch_npu;print(min(torch.npu.mem_get_info(i)[0] for i in range(8))/1e9)"`
  (idle ≈ 65GB/card; engine needs ≥55GB).
- **CANN 207001 "Binary get function by entry Failed" ≠ missing binary.**
  Usually an OOM/tiling allocation failure INSIDE the op, misreported. Tell:
  the exact op+shape passes in an idle-NPU probe container but fails
  in-serving → bound the memory, don't hunt for binaries.
- `_check_tp_memory_balance` crash = some node's HBM is held by a foreign
  process. `grep "Load weight begin. avail mem"` across nodes' logs: a
  node-uniform deficit = foreign container on that node.
- **Three memory walls for KV pool sizing** (GLM-5.3 hybrid arch, EP32):
  950k tokens → graph-capture OOM; 880k → HCCL lazy-allocation failure
  (error 561000); 740k → first-request op-workspace OOM (207001). Budget
  ~49KB/token (latent buffers allocated for all 45 layers) and keep ~5GB
  headroom.

## Errors that lie

- **CANN async stack traces blame the bystander.** The reported failing op
  is whichever enqueue hits the next sync point after the device died — a
  plain `F.linear` on the PEER node can manifest as an mHC op locally.
  Always pull the peer node's `grep -A40 "hit an exception"` before
  theorizing. `ASCEND_LAUNCH_BLOCKING=1` gives accurate stacks (debug only —
  roughly doubles op latency).
- **Clocks are not clocks.** Container logs and `docker inspect StartedAt`
  are UTC on UTC-hosts-but-Shanghai-tz-Asia setups; scripts logging inside
  containers can be off by ±8-16h from both host and each other. Anchor
  timelines on per-host `date` and file mtimes, never in-log HH:MM.
- **`docker exec X pgrep ... | head -2 || echo dead` never fires** — `||`
  binds head's exit code. Check `docker ps` first, or redirect so the
  exec failure is visible.

## Triton / JIT

- First-request triton JIT looks like a hang (no scheduler logs, curl
  timeout, containers alive). `py-spy dump` on the scheduler shows
  `linalg_to_bin_enable_npu_compile_A2_A3`. Fix: persistent `TRITON_CACHE_DIR`
  on a mounted path + `--watchdog-timeout 900` (default 300 kills the first
  healthy run).
- **Multi-rank cache write races**: N scheduler ranks first-executing the
  same `@triton.autotune` kernel compile simultaneously into the SHARED
  cache; raced writes corrupt binaries and UNRELATED later GEMMs fail with
  207001 or hang on a rank subset. Precompile single-process per node before
  serving (shape buckets: autotune keys are shape-dependent).

## Containers & cluster ops

- **`--privileged --security-opt label=disable` are required for NPU
  visibility.** Without them every image — including known-good ones — fails
  identically with `torch.npu.is_available()=False`.
- 910B is aarch64: multi-arch registries serve the PULLING host's arch.
  `docker pull --platform linux/arm64` + verify `.Architecture` BEFORE any
  17GB transfer. Binary wheels must be `*_aarch64.whl`.
- pip in air-gapped containers: `pip install --no-index --no-deps` — plain
  pip probes PyPI and hangs per-package.
- Node-local `/data` is NOT shared across nodes — stage wheels/scripts per
  node.
- `pkill -f <pattern>` over ssh matches your own remote `bash -c` wrapper
  (exit 255 self-kill, or phantom "still alive" verdicts). Use
  `pgrep -af "pat[t]ern"` or kill by exact PID.
- **Never scp over a script that's running on the target** — bash reads by
  offset; the running instance executes garbage. Kill → copy → fire.
- Multi-node TP: a dead peer's collectives hang the survivor silently; check
  the PEER's container/logs before diagnosing the node that reported the
  error.
- TP16 paired restarts: rank0 first, 12s, then rank1. Fully-parallel
  two-node launches intermittently miss the gloo rendezvous window.

## Ascend engine specifics

- `a2a=none` expert parallel caps at EP=1; DeepEP forces EP==TP. Expressing
  ep<tp requires the deepep backend (see deep-ep-patch/).
- ServerArgs validation passing ≠ runtime support: two flag combinations
  passed `prepare_server_args` and died at graph-capture / first-request.
  **Every new parallelism combination must be validated with a real request**
  (garbled output also counts as "successfully started").
- Speculative decode with dp-attention: decode graph batch sizes must be
  multiples of attn_tp (gathered buffer alignment).
- DSA/NSA models need `--page-size 64` (KV pool hard-assert).
