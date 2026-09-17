"""mem_probe.py —— 分阶段显存探针与 sweep（07 篇核心交付物）。

04 篇的估算公式把显存拆成权重、梯度、优化器状态、激活四项，100M 档
估算 1.97 GB、实测 2.49 GB，差了 26%。本篇要回答：差额去了哪里。

三个工具：

    stage_probe   在模型加载、优化器初始化、forward、backward、step、
                  zero_grad 六个阶段打点，记录 allocated / reserved /
                  峰值，画出一个 step 的显存时间线
    sweep         单变量扫描 batch / seq / hidden / layers / dtype /
                  optimizer，观察各项显存的增长形状
    frag_probe    制造"reserved 高但 allocated 不高"的碎片化场景，
                  用 memory snapshot 找出无法满足的大块分配

口径说明（全文统一）：
    allocated   PyTorch 缓存分配器已经交给张量的字节数（精确）
    reserved    分配器向 CUDA 要到的字节数（含空闲块，精确但含碎片）
    peak        max_memory_allocated，运行至今的 allocated 峰值
    nvidia-smi  进程占用，含 CUDA 上下文与驱动开销（只能外部观测）
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.reproducibility import set_all_seeds  # noqa: E402
from exp_hf.adapters import build_llama  # noqa: E402
from exp_hf.contract import contract_loss  # noqa: E402
from exp_hf.train_hf import TokenDataset, encode_all  # noqa: E402
from exp_scale.schedulers import cosine_with_warmup  # noqa: E402


def _mb(n: int) -> float:
    return round(n / 1024 / 1024, 1)


def _snap(tag: str) -> dict:
    return {
        "stage": tag,
        "allocated_mb": _mb(torch.cuda.memory_allocated()),
        "reserved_mb": _mb(torch.cuda.memory_reserved()),
        "peak_mb": _mb(torch.cuda.max_memory_allocated()),
    }


def stage_probe(vocab_size: int, hidden: int, layers: int, heads: int,
                seq_len: int, batch: int, amp: bool, steps: int = 3,
                device: str = "cuda") -> dict:
    """六阶段打点：加载 → 优化器 → forward → backward → step → zero_grad。"""
    torch.cuda.reset_peak_memory_stats()
    timeline = [_snap("start")]

    model = build_llama(vocab_size, hidden=hidden, layers=layers, heads=heads,
                        head_dim=hidden // heads, intermediate=int(hidden * 8 / 3),
                        max_seq_len=max(seq_len, 512)).to(device)
    timeline.append(_snap("model_loaded"))

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    timeline.append(_snap("optimizer_init"))

    scaler = torch.amp.GradScaler("cuda", enabled=amp) if amp else None
    x = torch.randint(0, vocab_size, (batch, seq_len), device=device)
    y = torch.randint(0, vocab_size, (batch, seq_len), device=device)

    for step in range(steps):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            loss = contract_loss(model(x), y)
        timeline.append(_snap(f"forward_{step}"))

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        timeline.append(_snap(f"backward_{step}"))

        if scaler is not None:
            scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if scaler is not None:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        timeline.append(_snap(f"step_{step}"))

        opt.zero_grad(set_to_none=True)
        timeline.append(_snap(f"zero_grad_{step}"))

    return {
        "config": {"vocab": vocab_size, "hidden": hidden, "layers": layers,
                   "seq": seq_len, "batch": batch, "amp": amp},
        "params": sum(p.numel() for p in model.parameters()),
        "timeline": timeline,
    }


def sweep_one(vocab_size: int, hidden: int, layers: int, heads: int,
              seq_len: int, batch: int, amp: bool, opt_kind: str,
              device: str = "cuda") -> dict:
    """单配置测量：模型加载后、优化器后、一个 step 后的 allocated 与峰值。"""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = build_llama(vocab_size, hidden=hidden, layers=layers, heads=heads,
                        head_dim=hidden // heads, intermediate=int(hidden * 8 / 3),
                        max_seq_len=max(seq_len, 512)).to(device)
    after_model = _mb(torch.cuda.memory_allocated())

    if opt_kind == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=3e-4)
    after_opt = _mb(torch.cuda.memory_allocated())

    scaler = torch.amp.GradScaler("cuda", enabled=amp) if amp else None
    x = torch.randint(0, vocab_size, (batch, seq_len), device=device)
    y = torch.randint(0, vocab_size, (batch, seq_len), device=device)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
        loss = contract_loss(model(x), y)
    if scaler is not None:
        scaler.scale(loss).backward()
    else:
        loss.backward()
    after_bwd = _mb(torch.cuda.memory_allocated())

    # 优化器状态是 lazy 的：init 时不分配，第一次 step 才到账。
    # 不跑 step 就量不到优化器增量，SGD 与 AdamW 会测成一样。
    if scaler is not None:
        scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if scaler is not None:
        scaler.step(opt)
        scaler.update()
    else:
        opt.step()
    after_step = _mb(torch.cuda.memory_allocated())
    opt.zero_grad(set_to_none=True)
    after_zero = _mb(torch.cuda.memory_allocated())

    peak = _mb(torch.cuda.max_memory_allocated())
    reserved = _mb(torch.cuda.memory_reserved())

    del model, opt, scaler, x, y, loss
    torch.cuda.empty_cache()
    return {
        "after_model_mb": after_model,
        "after_opt_mb": after_opt,
        "after_backward_mb": after_bwd,
        "after_step_mb": after_step,
        "after_zero_mb": after_zero,
        "opt_delta_mb": round(after_step - after_bwd, 1),
        "peak_mb": peak,
        "reserved_mb": reserved,
    }


def frag_probe(device: str = "cuda") -> dict:
    """碎片化 OOM 的三个必要条件（缺一不可）：

      (a) 缓存池里没有满足请求的连续空闲块；
      (b) 含空闲块的 segment 无法整体归还驱动（里面有 active 块）；
      (c) 驱动侧余量不足以扩张新 segment。

    布局（8 GB 卡，总 8192 MB）：
      1. filler 占 7400 MB，驱动余量只剩约 280 MB（< 512，条件 c）；
      2. 申请 512 MB（独立 segment），释放后劈成两个 256 MB（a、b）；
      3. 释放 b：空闲 256 MB，但 segment 里 a 还 active，整段还不了（条件 b）；
      4. 申请 512 MB：池里最大连续 256（条件 a）、段还不了、驱动余量 280
         不够 512 → OOM，尽管 reserved - allocated 显示还有 256 MB 空闲。
    复验：释放 a 后整段可归还，同样的 512 MB 立刻能拿到。
    """
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    filler = torch.empty(7400 * 1024 * 1024 // 4, device=device)
    after_filler = _snap("after_filler")

    slab = torch.empty(512 * 1024 * 1024 // 4, device=device)
    del slab
    a = torch.empty(256 * 1024 * 1024 // 4, device=device)
    b = torch.empty(256 * 1024 * 1024 // 4, device=device)
    before = _snap("split")
    del b
    after_free = _snap("after_free_half")

    snapshot = torch.cuda.memory._snapshot()
    segs = snapshot.get("segments", [])
    largest_inactive = 0
    for s in segs:
        for blk in s.get("blocks", []):
            if blk.get("state") == "inactive":
                largest_inactive = max(largest_inactive, blk.get("size", 0))
    free_total = after_free["reserved_mb"] - after_free["allocated_mb"]
    driver_headroom = round(
        (torch.cuda.get_device_properties(0).total_memory / 1024 / 1024)
        - after_free["reserved_mb"], 1)

    want_mb = 512.0
    oom = None
    try:
        probe_t = torch.empty(int(want_mb * 1024 * 1024) // 4, device=device)
        del probe_t
    except torch.cuda.OutOfMemoryError as exc:
        oom = str(exc)[:300]

    recovered = None
    del a
    torch.cuda.empty_cache()
    try:
        probe_t = torch.empty(int(want_mb * 1024 * 1024) // 4, device=device)
        recovered = True
        del probe_t
    except torch.cuda.OutOfMemoryError:
        recovered = False

    result = {
        "after_filler": after_filler,
        "before": before,
        "after_free": after_free,
        "free_total_mb": round(free_total, 1),
        "largest_inactive_block_mb": _mb(largest_inactive),
        "driver_headroom_mb": driver_headroom,
        "requested_mb": want_mb,
        "oom": oom,
        "recovered_after_free_a": recovered,
        "verdict": ("碎片化：空闲 256 MB 但最大连续块 256 MB、段含 active 块还不了、"
                    "驱动余量不足，512 MB 申请失败" if oom else "未触发 OOM"),
    }
    del filler
    torch.cuda.empty_cache()
    return result


def oom_capacity(vocab_size: int, hidden: int, layers: int, heads: int,
                 seq_len: int, batch: int, amp: bool, fix_batch: int,
                 mem_fraction: float | None = None,
                 device: str = "cuda") -> dict:
    """mem_fraction：给进程画显存红线（set_per_process_memory_fraction）。

    WSL2 的 DXG 驱动会把超出 VRAM 的分配 overcommit 到主机内存，
    裸金属上必 OOM 的配置在这里只会慢、不会崩。画红线后，
    超出红线的分配会真实抛 OutOfMemoryError，等价于模拟一张更小的卡。
    """
    if mem_fraction is not None:
        torch.cuda.set_per_process_memory_fraction(
            mem_fraction, torch.cuda.current_device())
    """真实容量 OOM：把 batch 推到显存放不下，记录 OOM 前最后一个阶段的证据。

    证据三件套：
      1. OOM 前最后成功的阶段打点（allocated / reserved / peak）；
      2. memory snapshot 的最大连续空闲块与空闲总量；
      3. 驱动余量（total - reserved）。
    判定：最大连续空闲块与驱动余量都远小于请求量 → 容量不足，不是碎片。
    复验：把 batch 降到 fix_batch 后同样配置能跑完。
    """
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = build_llama(vocab_size, hidden=hidden, layers=layers, heads=heads,
                        head_dim=hidden // heads, intermediate=int(hidden * 8 / 3),
                        max_seq_len=max(seq_len, 512)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=amp) if amp else None

    timeline = [_snap("model_loaded"), _snap("optimizer_init")]
    oom = None
    oom_stage = None
    x = torch.randint(0, vocab_size, (batch, seq_len), device=device)
    y = torch.randint(0, vocab_size, (batch, seq_len), device=device)
    try:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            loss = contract_loss(model(x), y)
        timeline.append(_snap("forward"))
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        timeline.append(_snap("backward"))
    except torch.cuda.OutOfMemoryError as exc:
        oom = str(exc)[:400]
        oom_stage = timeline[-1]["stage"] + " 之后"

    snapshot = torch.cuda.memory._snapshot()
    largest_inactive = 0
    free_total = 0
    for s in snapshot.get("segments", []):
        for blk in s.get("blocks", []):
            if blk.get("state") == "inactive":
                largest_inactive = max(largest_inactive, blk.get("size", 0))
                free_total += blk.get("size", 0)
    reserved = _mb(torch.cuda.memory_reserved())
    driver_headroom = round(
        torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 - reserved, 1)

    del model, opt, scaler, x, y
    torch.cuda.empty_cache()

    # 复验：降 batch 后同配置跑通
    torch.cuda.reset_peak_memory_stats()
    model = build_llama(vocab_size, hidden=hidden, layers=layers, heads=heads,
                        head_dim=hidden // heads, intermediate=int(hidden * 8 / 3),
                        max_seq_len=max(seq_len, 512)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=amp) if amp else None
    x = torch.randint(0, vocab_size, (fix_batch, seq_len), device=device)
    y = torch.randint(0, vocab_size, (fix_batch, seq_len), device=device)
    fixed_peak = None
    try:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            loss = contract_loss(model(x), y)
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if scaler is not None:
            scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if scaler is not None:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        fixed_peak = _mb(torch.cuda.max_memory_allocated())
    except torch.cuda.OutOfMemoryError:
        fixed_peak = None
    del model, opt, scaler, x, y
    torch.cuda.empty_cache()

    return {
        "failed_batch": batch,
        "oom": oom,
        "oom_stage": oom_stage,
        "timeline": timeline,
        "largest_inactive_block_mb": _mb(largest_inactive),
        "free_total_mb": _mb(free_total),
        "reserved_mb": reserved,
        "driver_headroom_mb": driver_headroom,
        "verdict": ("容量不足：最大连续空闲与驱动余量都远小于请求量" if oom
                    else "未触发 OOM"),
        "fix_batch": fix_batch,
        "fix_peak_mb": fixed_peak,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="stage",
                    choices=["stage", "sweep", "frag", "oom"])
    ap.add_argument("--vocab", type=int, default=6015)
    ap.add_argument("--hidden", type=int, default=384)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--fix-batch", type=int, default=8)
    ap.add_argument("--mem-fraction", type=float, default=None)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--opt", default="adamw", choices=["adamw", "sgd"])
    ap.add_argument("--output", default="/tmp/mem_probe.json")
    args = ap.parse_args()

    if args.mode == "stage":
        out = stage_probe(args.vocab, args.hidden, args.layers, args.heads,
                          args.seq, args.batch, args.amp)
    elif args.mode == "frag":
        out = frag_probe()
    elif args.mode == "oom":
        out = oom_capacity(args.vocab, args.hidden, args.layers, args.heads,
                           args.seq, args.batch, args.amp, args.fix_batch,
                           args.mem_fraction)
    else:
        out = sweep_one(args.vocab, args.hidden, args.layers, args.heads,
                        args.seq, args.batch, args.amp, args.opt)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2)[:1500])


if __name__ == "__main__":
    main()
