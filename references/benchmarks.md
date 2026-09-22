# Benchmarks

All scripts are endpoint-generic (point them at any OpenAI-compatible server).

## bench_ep32.py — concurrency scaling ladder

```bash
python3 bench_ep32.py <N>          # N concurrent 512-token generations
# edit URL/model at the top for your endpoint
```

Barrier-synced fan-out of N requests (no stagger), usage-based token counts,
reports aggregate tok/s + latency percentiles. This is the harness behind the
repo README's scaling table (2.1 tok/s @1 → 1,552 @1024).

## bench_decode.py — SSE step-time probe (A/B comparator)

```bash
python3 bench_decode.py http://<host>:<port> <n_concurrent> <max_tokens> <model>
```

Under MTP/EAGLE speculative decoding each SSE event ≈ one verify step's
tokens, so: **median inter-event gap = step time** (the cross-task comparable
engine metric, stall-resistant) and **tokens/event ≈ accept length**
(task-dependent — A/B comparisons must use the same prompt+temperature).

## gsm8k_eval.py — accuracy eval

```bash
# data prep (once)
curl -sS -L "https://huggingface.co/datasets/gsm8k/resolve/main/main/test-00000-of-00001.parquet" -o /tmp/gsm8k_test.parquet
python3 -c "import pyarrow.parquet as pq,json;t=pq.read_table('/tmp/gsm8k_test.parquet');[print(json.dumps(r)) for r in t.to_pylist()]" > gsm8k_test.jsonl

python3 gsm8k_eval.py --host <host> --port <port> --model glm-5 --concurrent 4 --num 1319
# long-context mode: --long-context (stuffs a ~44K-token KB into the system prompt)
```

Reports accuracy, throughput, latency p50/p95/p99. Use it to verify a
deployment change didn't cost quality — throughput numbers alone don't catch
acceptance-rate regressions.

## Measured reference points

GLM-5.3 W4A8C8, EP32×DP4×NEXTN, 4×910B nodes (2026-09):

| N concurrent | agg tok/s |
|---|---|
| 1 | 2.1 (essay 32-33, math 39-41 usage-based) |
| 32 | 60.6 |
| 128 | 235.1 |
| 256 | 457.5 |
| 512 | 861.4 |
| 1024 | 1552.5 |
| 2048 | 1572.5 (saturated; queueing absorbs, latency doubles) |
