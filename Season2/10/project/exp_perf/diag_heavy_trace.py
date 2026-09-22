"""diag_heavy_trace.py —— 诊断 heavy_cpu 组 trace 窗口 3 倍于墙时的问题。

vfy_heavy_cpu 报 profiler_window 4385 ms 对 wall_total 1429 ms，
busy 1242 ms（86.9%）对 data 占 61.6%，两个口径互相矛盾。
区间并集不可能超过窗口，所以窗口本身或事件集合有问题。
本脚本重跑该配置的 profiling 段，把 trace 存盘并分析：
  - 所有 cat == Trace 的根事件
  - 各类事件数与 ts 跨度
  - kernel 事件按时间聚类成几组（看看到底录进了几步）
只读分析，不改 perf_bench。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, ".")
from exp_perf.perf_bench import (  # noqa: E402
    CorpusWindowDataset,
    _calibrate_cpu_work,
    _run_cpu_work,
    build_model,
    load_corpus_ids,
)
from exp_perf.perf_harness import export_trace_once  # noqa: E402
from exp_hf.contract import contract_loss  # noqa: E402

TRACE_OUT = Path("results/Season2/10/_diag_heavy_trace.json")


class Args:
    hidden = 384
    layers = 6
    heads = 6
    head_dim = 64
    seq = 256


def main() -> None:
    device = "cuda"
    ids, vocab = load_corpus_ids("exp_scale/data")
    state = _calibrate_cpu_work(8.0)
    print("calibrated cpu work: n=%s measured=%s ms" % (state["n"], state["measured_ms"]))
    ds = CorpusWindowDataset(ids, 256, 60 * 16, slow_ms=0.0, heavy_cpu=True,
                             heavy_ms=8.0, vocab_size=vocab)
    loader = torch.utils.data.DataLoader(ds, batch_size=16, num_workers=0,
                                         shuffle=False, drop_last=True)
    args = Args()
    model, _ = build_model(vocab, args, "eager", device, False)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    model.train()
    it = iter(loader)

    from torch.profiler import ProfilerActivity, profile

    walls = []
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            b = next(it)   # heavy 负载在 __getitem__ 里，这里不再叠加
            b = b.to(device)
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.float16):
                loss = contract_loss(model(b[:, :-1]), b[:, 1:])
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            torch.cuda.synchronize()
            walls.append((time.perf_counter() - t0) * 1000)

    trace, _ = export_trace_once(prof, workdir=TRACE_OUT.parent)
    TRACE_OUT.write_text(json.dumps(trace), encoding="utf-8")
    print("wall_total = %.1f ms (10 steps)" % sum(walls))

    evs = trace.get("traceEvents", [])
    roots = [e for e in evs if e.get("cat") == "Trace"]
    print("root Trace events: %d" % len(roots))
    for e in roots[:5]:
        print("   ts=%s dur=%s name=%s" % (e.get("ts"), e.get("dur"), e.get("name")))

    from collections import defaultdict
    by_cat = defaultdict(lambda: [0, 0.0, 0.0, None, None])
    for e in evs:
        c = e.get("cat") or "?"
        d = e.get("dur")
        ts = e.get("ts")
        rec = by_cat[c]
        rec[0] += 1
        if d is not None:
            rec[1] += d
            rec[2] = max(rec[2], d)
        if ts is not None:
            rec[3] = ts if rec[3] is None else min(rec[3], ts)
            end = ts + (d or 0)
            rec[4] = end if rec[4] is None else max(rec[4], end)
    print("\ncat 统计（事件数 / dur合计ms / dur最大ms / ts起 / ts止）：")
    for c, r in sorted(by_cat.items(), key=lambda kv: -kv[1][1]):
        print("  %-22s n=%-7d sum=%-10.1f max=%-9.2f span=[%s, %s]"
              % (c, r[0], r[1] / 1000, r[2] / 1000, r[3], r[4]))

    kern = [e for e in evs if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
            and e.get("ts") is not None and e.get("dur") is not None]
    kern.sort(key=lambda e: e["ts"])
    print("\nkernel 事件 %d 个，跨度 %.1f ms"
          % (len(kern), (kern[-1]["ts"] + kern[-1]["dur"] - kern[0]["ts"]) / 1000))
    # 按 50 ms 以上的间隔聚类，看录进了几"步"
    groups = 1
    gaps = []
    for a, b in zip(kern, kern[1:]):
        g = b["ts"] - (a["ts"] + a["dur"])
        if g > 50000:
            groups += 1
            gaps.append(round(g / 1000, 1))
    print("kernel 聚类组数 = %d（间隔>50ms），前 15 个间隔 ms: %s" % (groups, gaps[:15]))


if __name__ == "__main__":
    main()
