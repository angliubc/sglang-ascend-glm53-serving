#!/usr/bin/env python3
"""GSM8K eval for SGLang-served models. Supports short/long-context modes.

Usage:
  # Short context, 4 concurrent, 100 questions
  python3 gsm8k_eval.py --host 127.0.0.1 --port 8000 --model glm5.3 --concurrent 4 --num 100

  # Long context (~44K tok KB per question), 4 concurrent, full 1319
  python3 gsm8k_eval.py --host 127.0.0.1 --port 8000 --model glm5.3 --concurrent 4 --num 1319 --long-context

  # Thinking ON (default), thinking OFF
  python3 gsm8k_eval.py --host 127.0.0.1 --port 8000 --model glm5.3 --concurrent 6 --no-thinking

Requires: gsm8k_test.jsonl in same dir or /tmp/. Download:
  curl -sS -L "https://huggingface.co/datasets/gsm8k/resolve/main/main/test-00000-of-00001.parquet" -o /tmp/gsm8k_test.parquet
  python3 -c "import pyarrow.parquet as pq,json;t=pq.read_table('/tmp/gsm8k_test.parquet');[print(json.dumps(r)) for r in t.to_pylist()]" > gsm8k_test.jsonl
"""
import json, time, re, statistics, argparse, concurrent.futures, http.client, os

def load_qs(path, n):
    qs = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            m = re.search(r'####\s*(.+)', r["answer"])
            ans = m.group(1).strip().replace(",", "") if m else ""
            qs.append({"q": r["question"], "a": ans})
            if len(qs) >= n:
                break
    return qs

def gen_kb(topic, n_chars):
    out = []
    for i in range(n_chars // 180):
        s = i % 20
        out.append(
            f"[{topic}-S{s}] Ref {i}: Mathematical reference section {s}. "
            f"Algebra: factoring, quadratic formula, Vieta's, AM-GM. "
            f"Number theory: divisibility, modular arithmetic, GCD/LCM. "
            f"Combinatorics: nCr, permutations, Inclusion-Exclusion, Pigeonhole. "
            f"Geometry: Pythagorean, area formulas, similar triangles, circles. "
            f"Rates: d=rt, work rate, mixture problems. "
            f"Verification: substitute answer back. Check units. "
        )
    return "".join(out)[:n_chars]

def extract(text):
    if not text:
        return ""
    m = re.search(r'####\s*([0-9.,\-]+)', text)
    if m:
        return m.group(1).replace(",", "").strip()
    nums = re.findall(r'[-+]?\d+\.?\d*', text)
    return nums[-1] if nums else ""

def send(idx, question, correct, host, port, model, kb, thinking):
    messages = []
    if kb:
        messages.append({"role": "system", "content": "You are a math solver. Solve step by step. "
            "After reasoning, write final answer on last line as '#### <number>'.\n\n=== Reference ===\n" + kb})
    messages.append({"role": "user", "content": question})

    payload = {"model": model, "messages": messages, "max_tokens": 65536, "temperature": 0.1}
    if thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": True}

    body = json.dumps(payload).encode()
    conn = http.client.HTTPConnection(host, port, timeout=600)
    t0 = time.time()
    try:
        conn.request("POST", "/v1/chat/completions", body=body,
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        raw = resp.read().decode()
        lat = time.time() - t0
        conn.close()
        if resp.status != 200:
            return {"i": idx, "ca": correct, "ma": "", "ok": False,
                    "lat": lat, "tok": 0, "err": f"HTTP{resp.status}: {raw[:80]}"}
        data = json.loads(raw)
        msg = data["choices"][0]["message"]
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
        u = data.get("usage", {})
        ma = extract(content) or extract(reasoning)
        return {"i": idx, "ca": correct, "ma": ma, "ok": ma == correct,
                "lat": lat, "tok": u.get("completion_tokens", 0), "err": None}
    except Exception as e:
        lat = time.time() - t0
        try: conn.close()
        except: pass
        return {"i": idx, "ca": correct, "ma": "", "ok": False,
                "lat": lat, "tok": 0, "err": str(e)[:60]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="glm5.3")
    ap.add_argument("--concurrent", type=int, default=4)
    ap.add_argument("--num", type=int, default=1319)
    ap.add_argument("--long-context", action="store_true", help="Add ~44K token KB to system prompt")
    ap.add_argument("--no-thinking", action="store_true", help="Disable thinking mode")
    ap.add_argument("--data", default="", help="Path to gsm8k_test.jsonl")
    args = ap.parse_args()

    data_file = args.data or next(
        (p for p in ["gsm8k_test.jsonl", "/tmp/gsm8k_test.jsonl"]
         if os.path.exists(p)), "gsm8k_test.jsonl")
    qs = load_qs(data_file, args.num)
    thinking = not args.no_thinking
    kbs = [gen_kb(f"Math{i}", 150000) for i in range(args.concurrent)] if args.long_context else [None] * args.concurrent

    print(f"{'='*80}")
    print(f"  GSM8K: {len(qs)} questions | {args.concurrent} concurrent | ", end="")
    print(f"long-context ({150000//1000}K char KB)" if args.long_context else "short context", end="")
    print(f" | thinking: {'ON' if thinking else 'OFF'}")
    print(f"  Model: {args.model} @ {args.host}:{args.port}")
    print(f"{'='*80}\n")

    all_res = [None] * len(qs)
    t0 = time.time()
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrent) as pool:
        futs = {pool.submit(send, i, q["q"], q["a"], args.host, args.port, args.model,
                            kbs[i % args.concurrent], thinking): i for i, q in enumerate(qs)}
        for f in concurrent.futures.as_completed(futs):
            r = f.result()
            all_res[r["i"]] = r
            done += 1
            mark = "OK" if r["ok"] else "XX"
            err = f" ERR:{r['err'][:20]}" if r.get("err") else ""
            print(f"  [{done:4d}/{len(qs)}] #{r['i']:4d} {r['ca']}->{r['ma'] or '?'} "
                  f"{mark} {r['lat']:.0f}s {r['tok']}tok{err}", flush=True)
    wall = time.time() - t0

    ok = [r for r in all_res if r["ok"]]
    bad = [r for r in all_res if not r["ok"]]
    errs = sum(1 for r in all_res if r.get("err"))
    total_tok = sum(r["tok"] for r in all_res)
    lats = sorted(r["lat"] for r in all_res)
    p50 = lats[len(lats)//2] if lats else 0
    p95 = lats[min(int(len(lats)*0.95), len(lats)-1)] if lats else 0
    p99 = lats[min(int(len(lats)*0.99), len(lats)-1)] if lats else 0

    print(f"\n{'='*80}")
    print(f"  GSM8K Results ({args.concurrent} concurrent, {len(qs)} questions)")
    print(f"{'='*80}")
    print(f"  Accuracy:   {len(ok)}/{len(all_res)} = {len(ok)/len(all_res)*100:.1f}%")
    print(f"  Errors:     {errs}")
    print(f"  Total tok:  {total_tok}")
    print(f"  Wall:       {wall:.1f}s ({wall/60:.1f} min)")
    print(f"  Throughput: {total_tok/max(wall,0.1):.1f} tok/s agg")
    if lats:
        print(f"  Latency:    min={lats[0]:.1f} p50={p50:.1f} avg={statistics.mean(lats):.1f} "
              f"p95={p95:.1f} p99={p99:.1f} max={lats[-1]:.1f}s")
    if bad:
        print(f"\n  Wrong ({len(bad)}):")
        for r in bad[:15]:
            print(f"    #{r['i']:4d} expected={r['ca']} got={r['ma']} ({r['lat']:.0f}s {r['tok']}tok)")
        if len(bad) > 15:
            print(f"    ... and {len(bad)-15} more")
    print(f"{'='*80}")
    with open("/tmp/gsm8k_eval_results.json", "w") as f:
        json.dump(all_res, f, indent=2)
    print(f"  Results saved: /tmp/gsm8k_eval_results.json")

if __name__ == "__main__":
    main()
