"""diag_trace.py —— 诊断 chrome trace 里 gpu_busy 超过 wall 的原因（10 篇）。

inject_no_pin 报出 busy 2293.39 ms 对 wall 652.77 ms（351.3%）。
区间并集在数学上不可能超过 trace 的时间跨度，所以 trace 里必然存在
异常长的事件，或者事件跨度远超计时的 10 步窗口。

本脚本重跑该配置，把 trace 按 cat 分类统计：
  - 每类的事件数、dur 合计、dur 最大值
  - trace 的时间跨度（max(ts+dur) - min(ts)）
  - 与 wall 窗口的对比
  - 最长的 10 个事件

不猜，直接看数据。
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile
from torch.utils.data import DataLoader

# 本脚本从仓库的 .float-writing/ 下拷到 WSL 项目根目录运行，
# 但为了两边都能跑，同时把当前工作目录加进 sys.path。
sys.path.insert(0, os.getcwd())
sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp_perf.perf_bench import (  # noqa: E402
    CorpusWindowDataset,
    build_model,
    load_corpus_ids,
)
from exp_hf.contract import contract_loss  # noqa: E402

FAULT = sys.argv[1] if len(sys.argv) > 1 else "no_pin"
OPTS = sys.argv[2] if len(sys.argv) > 2 else "workers"
STEPS = 10


class Args:
    hidden = 384
    layers = 6
    heads = 6
    head_dim = 64
    seq = 256


def main() -> None:
    args = Args()
    device = "cuda"
    ids, vocab = load_corpus_ids("exp_scale/data")
    opts = set(OPTS.split(",")) if OPTS and OPTS != "none" else set()

    ds = CorpusWindowDataset(ids, args.seq, STEPS * 16 + 64)
    nw = 4 if "workers" in opts else 0
    loader = DataLoader(ds, batch_size=16, num_workers=nw, shuffle=False,
                        drop_last=True,
                        pin_memory=("pin" in opts) and FAULT != "no_pin",
                        persistent_workers=(nw > 0 and "persistent" in opts),
                        prefetch_factor=(4 if nw > 0 and "prefetch" in opts else None))

    attn = "sdpa" if "sdpa" in opts else "eager"
    model, _ = build_model(vocab, args, attn, device, False)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    model.train()

    it = iter(loader)

    def one_step():
        nonlocal it
        try:
            b = next(it)
        except StopIteration:
            it = iter(loader)
            b = next(it)
        b = b.to(device, non_blocking=False)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = contract_loss(model(b[:, :-1]), b[:, 1:])
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    for _ in range(3):
        one_step()
    torch.cuda.synchronize()

    walls = []
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    tmp.close()
    trace_path = Path(tmp.name)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(STEPS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            one_step()
            torch.cuda.synchronize()
            walls.append((time.perf_counter() - t0) * 1000)

    # 必须在 with 外导出：kineto_results 是在 profiler stop（__exit__）时
    # 才填充的，在 with 内调用会拿到 None 并抛 AttributeError。
    # perf_bench 里也是在 with 外导出，所以那边正常。
    prof.export_chrome_trace(str(trace_path))

    wall_total = sum(walls)
    tr = json.loads(trace_path.read_text(encoding="utf-8"))
    events = tr.get("traceEvents", [])

    print(f"fault={FAULT} opts={OPTS} steps={STEPS}")
    print(f"wall_total = {wall_total:.2f} ms  (每步均值 {wall_total/STEPS:.2f} ms)")
    print(f"trace 事件总数 = {len(events)}")

    # 按 cat 分类
    by_cat = defaultdict(lambda: {"n": 0, "dur_us": 0.0, "max_us": 0.0,
                                  "max_name": "", "min_ts": None, "max_end": None})
    for e in events:
        cat = e.get("cat")
        if cat is None:
            continue
        d = by_cat[cat]
        d["n"] += 1
        dur = float(e.get("dur") or 0.0)
        ts = e.get("ts")
        d["dur_us"] += dur
        if dur > d["max_us"]:
            d["max_us"] = dur
            d["max_name"] = str(e.get("name", ""))[:60]
        if ts is not None:
            ts = float(ts)
            d["min_ts"] = ts if d["min_ts"] is None else min(d["min_ts"], ts)
            end = ts + dur
            d["max_end"] = end if d["max_end"] is None else max(d["max_end"], end)

    print("\n%-22s %7s %12s %12s %10s  %s"
          % ("cat", "事件数", "dur合计ms", "单个最大ms", "跨度ms", "最大事件名"))
    for cat, d in sorted(by_cat.items(), key=lambda kv: -kv[1]["dur_us"]):
        span = ((d["max_end"] - d["min_ts"]) / 1000.0
                if d["min_ts"] is not None else 0.0)
        print("%-22s %7d %12.2f %12.2f %10.2f  %s"
              % (cat[:22], d["n"], d["dur_us"] / 1000.0, d["max_us"] / 1000.0,
                 span, d["max_name"][:44]))

    # GPU 侧事件的区间并集与跨度
    GPU_CATS = ("kernel", "gpu_memcpy", "gpu_memset")
    iv = []
    for e in events:
        if e.get("cat") not in GPU_CATS:
            continue
        dur = e.get("dur")
        ts = e.get("ts")
        if dur is None or ts is None:
            continue
        iv.append((float(ts), float(ts) + float(dur), str(e.get("name", "")),
                   float(dur)))
    iv.sort()
    if iv:
        span_ms = (iv[-1][1] - iv[0][0]) / 1000.0
        union = 0.0
        cur_s, cur_e = iv[0][0], iv[0][1]
        for s, e, _, _ in iv[1:]:
            if s <= cur_e:
                cur_e = max(cur_e, e)
            else:
                union += cur_e - cur_s
                cur_s, cur_e = s, e
        union += cur_e - cur_s
        print(f"\nGPU 侧事件数 = {len(iv)}")
        print(f"GPU 事件时间跨度 = {span_ms:.2f} ms")
        print(f"GPU 区间并集     = {union/1000.0:.2f} ms")
        print(f"wall 窗口        = {wall_total:.2f} ms")
        print(f"并集/wall        = {union/1000.0/wall_total:.3f}")
        if span_ms > wall_total * 1.2:
            print("  !! trace 跨度远超 wall 窗口：trace 里含计时窗口之外的事件")
        print("\n最长的 10 个 GPU 事件：")
        for s, e, name, dur in sorted(iv, key=lambda x: -x[3])[:10]:
            print(f"  {dur/1000.0:9.3f} ms  {name[:66]}")

    trace_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
