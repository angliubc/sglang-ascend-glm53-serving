import json, urllib.request, time, sys, concurrent.futures as cf

URL = 'http://127.0.0.1:8077/v1/chat/completions'
PROMPTS = [
    '写一篇关于人工智能在医疗领域应用的短文，300字左右。',
    '解释量子计算的基本原理，并说明它与经典计算的区别。',
    '用Python实现一个快速排序，并逐行注释。',
    '总结三国演义的主要情节脉络和核心人物关系。',
    '分析全球气候变化的主要原因及应对策略。',
    '写一个关于时间旅行的科幻微小说。',
    '详细说明TCP三次握手的过程及每一步的作用。',
    '比较儒家与道家思想的核心差异。',
    '设计一个简单的数据库表结构来管理图书馆藏书。',
    '论述机器学习中的过拟合问题及缓解方法。',
    '解释深度学习中Transformer架构的注意力机制。',
    '写一段鼓励学生坚持学习的演讲稿。',
]
def ask(p, max_tokens=512):
    body = json.dumps({"model":"glm-5","messages":[{"role":"user","content":p}],"max_tokens":max_tokens,"temperature":0}).encode()
    req = urllib.request.Request(URL, data=body, headers={'Content-Type':'application/json'})
    t0 = time.time()
    d = json.loads(urllib.request.urlopen(req, timeout=1800).read())
    u = d['usage']
    return u['completion_tokens'], time.time()-t0, (d['choices'][0]['message'].get('content') or '')[:40]

N = int(sys.argv[1]) if len(sys.argv) > 1 else 4
prompts = (PROMPTS * (N//len(PROMPTS) + 1))[:N]
import threading
barrier = threading.Barrier(N)
def ask_b(p):
    barrier.wait()
    return ask(p)
t0 = time.time()
with cf.ThreadPoolExecutor(N) as ex:
    results = list(ex.map(ask_b, prompts))
wall = time.time() - t0
tot = sum(r[0] for r in results)
ok = sum(1 for r in results if r[0] > 0)
lat = sorted(r[1] for r in results)
print(f'N={N} ok={ok}/{N} wall={wall:.0f}s tokens={tot} agg={tot/wall:.1f} tok/s '
      f'lat p50={lat[len(lat)//2]:.0f}s max={lat[-1]:.0f}s')
for r in results[:3]: print(' sample:', r[0], f'{r[1]:.0f}s', r[2].replace(chr(10),' ')[:36])
