# Quantization artifacts verification

## verify_quant_output.py

Header-only dtype-coverage verifier for safetensors quantization outputs.
Run after EVERY msmodelslim (or any) quantization — **EXIT=0 / `.quant_done`
markers do NOT prove the product is compressed**:

> The GLM-5.3-Flash W8A8 artifact (2026-08-31) measured 529 int8 linears
> (~7GB) + 608.8GB BF16 experts = 620GB ≈ the original BF16 size —
> undeployable on 8×64GB — while every progress marker claimed success.
> Root cause: msmodelslim's `llm_ptq` API only wraps `nn.Linear`; fused MoE
> experts (raw 3-D Parameters, e.g. `gate_up_proj [E, 2I, H]`) are silently
> skipped.

Reads only each file's 8-byte little-endian header-length prefix + JSON
header — seconds even for 600GB artifacts, no torch needed, runs over ssh:

```bash
python3 verify_quant_output.py /path/to/quantized_model_dir
# remote, without copying the script:
ssh root@host 'python3 - /data/models/MODEL-W8A8/' < verify_quant_output.py
```

Exit 1 when BF16 > 40% of tensor bytes (usable as a gate in automated
launchers). Output: per-module-category × dtype byte distribution — check
that `routed_experts` shows int8/int4 bytes dominating.

## Sourcing GLM-5.3 quantized weights

- Ready-made W4A8C8: ModelScope `Eco-Tech/GLM-5.3-w8a8c8` / `GLM-5.3-w8a8c8`
  (what the production deployment ran).
- Self-quantizing: msmodelslim on the BF16 checkpoint. Prefer the
  `modelslim_v1` declarative-yaml pipeline for fused-MoE architectures (the
  `llm_ptq` API skips fused experts — see above). Check the upstream repo's
  `lab_practice/` for arch-specific yaml templates.
- 910B (A2) does NOT support FP8/MXFP8 — W8A8/W4A8/W4A16(AWQ/GPTQ) only.
