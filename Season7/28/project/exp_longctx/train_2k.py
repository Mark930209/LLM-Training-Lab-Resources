"""train_2k.py —— 2k 基线训练（长程关联检索任务，rope=none）。

用法：python train_2k.py --out out/train_2k.json --ckpt out/ckpt_2k.pt
"""

from __future__ import annotations

import argparse
import time

import torch

import longctx_common as lc
from tiny_rope import TinyRoPE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--pairs", type=int, default=24)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = args.device
    torch.manual_seed(lc.TRAIN_SEED)
    model = TinyRoPE(vocab=lc.VOCAB, rope_mode="none", rope_s=1.0).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1.0, (i + 1) / 50))

    t0 = time.perf_counter()
    loss_curve = []
    model.train()
    for i in range(args.steps):
        xs, ys = lc.make_batch(args.batch, args.seq, args.pairs,
                               lc.TRAIN_SEED + 1000 + i,
                               query_depth=None)   # 随机深度，防位置捷径
        xs, ys = xs.to(device), ys.to(device)
        logits = model(xs)
        loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), ys.flatten(0, 1), ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        loss_curve.append(float(loss))
        if (i + 1) % 100 == 0:
            print(f"step {i+1} loss {loss_curve[-1]:.4f}")
    train_s = time.perf_counter() - t0

    torch.save({"model": model.state_dict(), "steps": args.steps,
                "seq": args.seq, "pairs": args.pairs}, args.ckpt)

    grid = {}
    for S in (512, 1024, 2048):
        for d in (0.1, 0.5, 0.9):
            grid[f"{S}@{d}"] = lc.needle_eval(model, device, S, d,
                                              n_pairs=max(6, S // 85))
    out = {
        "experiment": "train_2k",
        "framework": "plain-pytorch(tiny_rope)",
        "rope": "none",
        "train_len": args.seq,
        "steps": args.steps,
        "loss_first": loss_curve[0],
        "loss_last": loss_curve[-1],
        "loss_curve": loss_curve,
        "needle_2k_grid": grid,
        "short_loss": lc.short_task_loss(model, device),
        "train_seconds": round(train_s, 2),
        "peak_mem_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1)
        if device.startswith("cuda") else None,
        "labels": {"REAL": "单卡真实运行"},
    }
    lc.save_json(out, args.out)
    print("[train_2k]", {k: round(v, 3) for k, v in grid.items()},
          "short", round(out["short_loss"], 4))


if __name__ == "__main__":
    main()
