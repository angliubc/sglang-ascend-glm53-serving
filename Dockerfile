# GLM-5.3 (GlmMoeDsa) serving image for Ascend 910B — patches baked in.
#
# Build inputs (assembled by ./build.sh):
#   build/sglang/          stock sglang tree from the base image + patches/sglang/*.patch
#   build/ep_strategy.py   patched deep_ep strategy (patches/deep_ep/ep_strategy.py)
#
# COPY-only layers => cross-arch docker build works WITHOUT qemu
# (safe to build on an x86 host; the result still only RUNS on aarch64/910B).
FROM quay.io/ascend/sglang:main-cann9.0.0-910b

# SGLang tree with the 35 production patches (MoE W4A8 fast path, modelslim
# quantized loading, hybrid DSA/mamba pools, NEXTN draft chain, serving layer).
COPY build/sglang/ /sgl-workspace/sglang/python/sglang/

# deep_ep cross-node EP strategy modes (only changed file in the package).
COPY build/ep_strategy.py /usr/local/python3.11.15/lib/python3.11/site-packages/deep_ep/ep_strategy.py

# Production mode for 4-node EP32: prefill Normal=ALLTOALL (HCCL alltoallv),
# decode LL=DEFAULT (deep_ep C++ native + vendored MoeLowLatencyDispatchV2).
ENV DEEP_USE_MODE=alltoall_default \
    SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256

# Persist triton JIT cache at a predictable mount point (bind a host dir here,
# or every restart re-JITs KDA kernels, ~5 min).
ENV TRITON_CACHE_DIR=/sgl-triton-cache
VOLUME /sgl-triton-cache

LABEL org.opencontainers.image.title="sglang-ascend-glm53-serving" \
      org.opencontainers.image.description="GLM-5.3 (GlmMoeDsa) EP32 x DP-attention x NEXTN on Ascend 910B: cross-node DeepEP patch + W4A8 MoE fast path (1552 tok/s @1024 concurrent)" \
      org.opencontainers.image.licenses="Apache-2.0"
