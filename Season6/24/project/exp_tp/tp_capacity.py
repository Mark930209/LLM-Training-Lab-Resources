"""tp_capacity.py —— 24 篇 E5：宽层容量对照。

一层 hidden 12288 / FFN 4× 的 MLP（≈1.2 G 参数，23 篇实测 8.0 B/param
≈ 9.7 GB 全参 AdamW 状态）：
    --mode full     单卡全量训练（预算钉死 mem-fraction），3070 预期 OOM
                    （真实异常记录进报告），4090 预期训通。
    --mode sharded  2 GPU TP（fc1 切行、fc2 切列），每 rank 持一半，
                    预期两 rank 均训通（REAL 容量结论）。

用法：
    python tp_capacity.py --mode full --out results/xxx.json          （单进程）
    torchrun --nproc_per_node=1 ... tp_capacity.py --mode sharded ... （两机）
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import torch.distributed as dist

from tp_common import CommAccount, env_rank_world, run_meta, save_report
from tp_layers import RefMLP, TPMLP, bind_comm


def probe(tag, device):
    if not torch.cuda.is_available():
        return {"stage": tag}
    return {
        "stage": tag,
        "allocated_mb": round(torch.cuda.memory_allocated() / 1024 / 1024, 1),
        "reserved_mb": round(torch.cuda.memory_reserved() / 1024 / 1024, 1),
        "peak_mb": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["full", "sharded"], required=True)
    ap.add_argument("--hidden", type=int, default=12288)
    ap.add_argument("--inter", type=int, default=49152)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--mem-fraction", type=float, default=0.9)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rank, world = env_rank_world()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        torch.cuda.set_per_process_memory_fraction(args.mem_fraction,
                                                   device=local_rank)
    if args.mode == "sharded" and world > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")

    comm = CommAccount(world if args.mode == "sharded" else 1)
    if args.mode == "sharded":
        bind_comm(comm, world, rank)

    stages = []
    report = {
        "meta": run_meta(rank, world, {**vars(args),
                                       "budget_note": "mem-fraction 钉死预算"}),
        "mode": args.mode,
        "stages": stages,
    }
    try:
        torch.manual_seed(42)
        if args.mode == "full":
            model = RefMLP(args.hidden, args.inter,
                           bias=False).to(device, torch.bfloat16)
        else:
            model = TPMLP(args.hidden, args.inter, world, rank,
                          bias=False).to(device, torch.bfloat16)
        stages.append(probe("model_loaded", device))
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        stages.append(probe("optim_init", device))

        torch.manual_seed(7)
        losses = []
        t0 = time.perf_counter()
        for s in range(args.steps):
            x = torch.randn(args.batch, args.seq, args.hidden,
                            device=device, dtype=torch.bfloat16)
            if args.mode == "sharded" and world > 1:
                # 两机 RNG 不一致（torch 2.14 vs 2.11），输入从 rank0 广播
                dist.broadcast(x, src=0)
            y = model(x)
            loss = y.float().pow(2).mean()
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            losses.append(round(loss.item(), 6))
            if s == 0:
                stages.append(probe("step0_done", device))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        stages.append(probe("train_done", device))
        report.update({
            "ok": True,
            "losses": losses,
            "wall_s": round(wall, 3),
            "comm": comm.summary() if args.mode == "sharded" else None,
        })
    except RuntimeError as e:
        msg = str(e)
        stages.append(probe("failed", device))
        report.update({
            "ok": False,
            "error_type": "RuntimeError",
            "error_msg": (msg if len(msg) < 500 else msg[:500] + "..."),
            "oom": "out of memory" in msg.lower(),
        })
    save_report(args.out.replace(".json", f".r{rank}.json"), report)


if __name__ == "__main__":
    main()
