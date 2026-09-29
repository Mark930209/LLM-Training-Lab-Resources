"""tp_bench.py —— 24 篇 E4：单卡 vs 2 GPU TP 的耗时/显存/通信基准 + 链路基准。

两个口径：
1. mlpsweep  hidden ∈ {1024, 1536, 2048, 4096} 的 TP-MLP（fc1 col + fc2 row）
   fwd+bwd 耗时 vs 同卡单卡 RefMLP；两机异构如实分别记录，不混算加速比。
2. link      纯 AllReduce 延迟 vs payload 尺寸，实测有效带宽，
   用来和 ring 理论下界对账——裁决"TP 慢是实现差还是带宽顶死"。

用法：torchrun 两机（world=2）或单进程（world=1 单卡对照）。
    python tp_bench.py --mode mlp|link --out results/xxx.json
"""

from __future__ import annotations

import argparse
import os
import statistics
import time

import torch
import torch.distributed as dist

from tp_common import (CommAccount, env_rank_world, run_meta,
                       ring_lower_bound, save_report)
from tp_layers import RefMLP, TPMLP, bind_comm

HIDDENS = [1024, 1536, 2048, 4096]
LINK_PAYLOADS = [1 << 20, 1 << 22, 1 << 24]  # 元素数（bf16：2/8/32 MiB）


def bench(fn, warmup: int = 3, iters: int = 10) -> float:
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return round(statistics.median(ts) * 1000, 3)  # ms


def mlp_sweep(world, rank, device, batch, seq):
    rows = []
    for h in HIDDENS:
        inter = h * 4
        torch.manual_seed(42)
        ref = RefMLP(h, inter, bias=False).to(device, torch.bfloat16)
        tp = TPMLP(h, inter, world, rank, bias=False).to(device, torch.bfloat16)
        # TP 权重取参考切片，保证 FLOPs 可比
        with torch.no_grad():
            shard = inter // world
            tp.fc1.weight.copy_(ref.fc1.weight[rank * shard:
                                               (rank + 1) * shard])
            tp.fc2.weight.copy_(ref.fc2.weight[:, rank * shard:
                                               (rank + 1) * shard])
        # x 必须开梯度：真实训练里输入梯度的 AllReduce 每步都会走，
        # requires_grad=False 会把通信量少记一半（本篇实测教训）。
        # 广播放在开梯度之前，避免对 grad 叶子做原地 collective。
        x = torch.randn(batch, seq, h, device=device, dtype=torch.bfloat16)
        if world > 1:
            dist.broadcast(x, src=0)   # 两机 RNG 不一致，输入统一
        x.requires_grad_(True)

        def run_ref():
            y = ref(x)
            y.float().pow(2).mean().backward()
            ref.zero_grad(set_to_none=True)

        def run_tp():
            y = tp(x)
            y.float().pow(2).mean().backward()
            tp.zero_grad(set_to_none=True)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t_ref = bench(run_ref)
        ref_peak = (torch.cuda.max_memory_allocated() >> 20
                    if torch.cuda.is_available() else 0)
        # 计时前就绑通信：否则计时迭代里 collective 不发生，时间失真
        bind_comm(CommAccount(world), world, rank)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t_tp = bench(run_tp)
        tp_peak = (torch.cuda.max_memory_allocated() >> 20
                   if torch.cuda.is_available() else 0)
        # 通信账单独采集一步：计时迭代会重复累加事件
        comm = CommAccount(world)
        bind_comm(comm, world, rank)
        run_tp()
        rows.append({
            "hidden": h, "inter": inter, "batch": batch, "seq": seq,
            "ref_ms": t_ref, "tp_ms": t_tp,
            "tp_over_ref": round(t_tp / t_ref, 3),
            "ref_peak_mb": ref_peak, "tp_peak_mb": tp_peak,
            "comm": comm.summary(),
        })
    return rows


def link_bench(world, rank, device):
    rows = []
    if world < 2:
        return rows
    for n in LINK_PAYLOADS:
        t = torch.randn(n, device=device, dtype=torch.bfloat16)

        def run():
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

        ms = bench(run, warmup=5, iters=20)
        payload = n * 2
        rows.append({
            "elements": n,
            "payload_mb": round(payload / 1024 / 1024, 2),
            "all_reduce_ms": ms,
            "ring_lower_bound_mb": round(
                ring_lower_bound(payload, world) / 1024 / 1024, 2),
            "effective_bus_mb_s": round(
                ring_lower_bound(payload, world) / (ms / 1000) / 1024 / 1024, 1),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["mlp", "link"], default="mlp")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rank, world = env_rank_world()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    if world > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)

    if args.mode == "mlp":
        rows = mlp_sweep(world, rank, device, args.batch, args.seq)
    else:
        rows = link_bench(world, rank, device)

    report = {
        "meta": run_meta(rank, world, {**vars(args),
                                       "hiddens": HIDDENS,
                                       "link_payloads": LINK_PAYLOADS}),
        "rows": rows,
    }
    save_report(args.out.replace(".json", f".r{rank}.json"), report)


if __name__ == "__main__":
    main()
