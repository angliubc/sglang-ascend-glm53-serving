#!/bin/bash
# One-click build: sglang-ascend-glm53:ep32 (GLM-5.3 Ascend 910B serving image).
#
# What it does:
#   1. ensures the base image exists (aarch64 — pulls with --platform on any host)
#   2. extracts the stock sglang tree from the base image
#   3. applies patches/sglang/*.patch (35 files, verified clean against
#      main-cann9.0.0-910b; re-verify on image updates — a failed anchor prints SKIP)
#   4. stages the patched deep_ep ep_strategy.py
#   5. docker build (COPY-only layers; no qemu needed even on x86 hosts)
#
# Run on a 910B node (aarch64) or any docker host — the image only RUNS on 910B.
set -euo pipefail
cd "$(dirname "$0")"

BASE=quay.io/ascend/sglang:main-cann9.0.0-910b
TAG=${TAG:-sglang-ascend-glm53:ep32}
PROBE=glm53-build-probe

echo "== [1/5] base image =="
if ! docker image inspect "$BASE" >/dev/null 2>&1; then
  docker pull --platform linux/arm64 "$BASE"
fi
ARCH=$(docker image inspect "$BASE" --format '{{.Architecture}}')
echo "base arch: $ARCH"
[ "$ARCH" = "arm64" ] || { echo "!! base is $ARCH, not arm64 — 910B is aarch64; re-pull with --platform linux/arm64" >&2; exit 1; }

echo "== [2/5] extract stock sglang tree =="
rm -rf build
mkdir -p build
docker rm -f "$PROBE" >/dev/null 2>&1 || true
docker create --name "$PROBE" "$BASE" >/dev/null
docker cp "$PROBE":/sgl-workspace/sglang/python/sglang build/
docker rm -f "$PROBE" >/dev/null
echo "stock tree: $(find build/sglang -name '*.py' | wc -l) py files"

echo "== [3/5] apply sglang patches =="
OK=0; SKIP=0
cd build
for p in ../patches/sglang/*.patch; do
  if patch --dry-run -p3 < "$p" >/dev/null 2>&1; then
    patch -s -p3 < "$p"
    OK=$((OK+1))
  else
    echo "  SKIP (anchor miss — image drift?): $p"
    SKIP=$((SKIP+1))
  fi
done
cd ..
echo "applied: $OK / skipped: $SKIP"
[ "$SKIP" -eq 0 ] || echo "!! skipped patches — the image's tree has drifted from main-cann9.0.0-910b; review before serving" >&2

echo "== [4/5] stage deep_ep patch =="
# ep_strategy.py is the ONLY file that differs from the package vendored in the
# base image (verified by full-dir diff) — a single-file overwrite is complete.
cp patches/deep_ep/ep_strategy.py build/ep_strategy.py

echo "== [5/5] docker build =="
export DOCKER_BUILDKIT=0   # FROM references a registry image; buildkit mirror quirks aside, classic is predictable
docker build -t "$TAG" .

echo
echo "DONE: $TAG"
echo "  next: IMAGE=$TAG WEIGHTS_DIR=<weights parent> TRITON_CACHE_HOST=<cache dir> \\"
echo "        python3 scripts/deploy_ep32_dp4.py <rank 0-3> <master_ip>   (on each node)"
