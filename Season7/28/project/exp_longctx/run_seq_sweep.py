"""run_seq_sweep.py —— E1 序列长度扫描：显存与 step time 随 S 的增长。

用法：python run_seq_sweep.py --out out/e1_sweep.json [--ckpt-gc]
报告：峰值显存、step time、tokens/s；OOM 如实记录为边界。
"""

from __future__ import annotations

import argparse
import time

import torch
from torch.utils.checkpoint import checkpoint

import longctx_common as lc
from tiny_rope import TinyRoPE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seqs", type=int, nargs="+",
                    default=[512, 1024, 2048, 4096, 8192, 16384])
    ap.add_argument("--gc", action="store_true", help="梯度检查点")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = args.device

    torch.manual_seed(0)
    model = TinyRoPE(vocab=lc.VOCAB).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    rows = []
    for S in args.seqs:
        try:
            xs, ys = lc.make_batch(args.batch, S, max(6, S // 85), seed=S)
            xs, ys = xs.to(device), ys.to(device)
            if device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            if args.gc:
                logits = checkpoint(lambda t: model(t), xs, use_reentrant=False)
            else:
                logits = model(xs)
            loss = torch.nn.functional.cross_entropy(
                logits.flatten(0, 1), ys.flatten(0, 1), ignore_index=-100)
            loss.backward()
            opt.step()
            opt.zero_grad()
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            rows.append({
                "seq": S, "gc": args.gc, "step_s": round(dt, 3),
                "tokens_per_s": round(S * args.batch / dt, 1),
                "peak_mem_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1)
                if device.startswith("cuda") else None,
                "oom": False,
            })
            print(f"S={S} gc={args.gc} {rows[-1]}")
            del logits, loss, xs, ys
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            rows.append({"seq": S, "gc": args.gc, "oom": True})
            print(f"S={S} gc={args.gc} OOM")
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    out = {
        "experiment": f"e1_sweep_{'gc' if args.gc else 'full'}",
        "framework": "plain-pytorch(tiny_rope)",
        "device": device,
        "rows": rows,
        "labels": {"REAL": "单卡真实运行"},
    }
    lc.save_json(out, args.out)


if __name__ == "__main__":
    main()
