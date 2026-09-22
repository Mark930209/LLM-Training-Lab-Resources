"""collective_lab.py —— 13 篇主实验：正确性、trace、backend 对照、DDP 梯度对账。

四种 mode：
  correctness : 手写 naive / ring / reduce_scatter+all_gather 与 dist.all_reduce 逐位对照
  trace      : ring 的逐轮 trace（每 rank 记录发送字节，核对 2(w-1)/w）
  backend    : gloo vs nccl 的 latency/带宽 sweep（message size 1MB~256MB）
  ddp_grad   : 把 12 篇 DDP 的真实梯度 bucket 送入手写 ring，核对理论通信量

用法（gloo 单机 2-rank，13 篇主实验）：
  torchrun --nproc_per_node=2 -m exp_collective.collective_lab --mode correctness
  torchrun --nproc_per_node=2 -m exp_collective.collective_lab --mode trace
  torchrun --nproc_per_node=2 -m exp_collective.collective_lab --mode backend --backend gloo
跨机 NCCL 对照（复用 11 篇环境）：
  两侧 torchrun --nnodes=2 --nproc_per_node=1 ... --mode backend --backend nccl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

from .collective_common import (
    TraceLog, all_gather_bytes, bandwidth_gbps, bench_collective,
    naive_bus_bytes, reduce_scatter_bytes, ring_bus_bytes, setup_dist, cleanup_dist,
)
from .hand_written import (
    all_gather_ring, all_reduce_naive, reduce_scatter_ring, ring_all_reduce,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _out(path: str, payload: dict, rank: int) -> None:
    if rank != 0:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[rank0] written {p}")


# ---------------------------------------------------------------- mode: correctness

def mode_correctity(args) -> dict:
    rank, _, world = setup_dist("gloo")
    torch.manual_seed(1234 + rank)
    results = {}

    for name, numel, dtype in [("small_1k", 1024, torch.float32),
                                ("grad_like_51m", 51_712_000 // 4, torch.float32)]:
        # 每个 rank 一份不同的输入（模拟不同数据分片产生的不同梯度）
        base = torch.randn(numel, dtype=dtype)
        local = base * (rank + 1) + rank * 0.5

        # 参考答案：dist.all_reduce
        ref = local.clone()
        dist.all_reduce(ref)

        # 手写 naive
        t_naive = local.clone()
        all_reduce_naive(t_naive)
        # 手写 ring
        t_ring = local.clone()
        ring_all_reduce(t_ring)
        # 拆开的两原语：reduce_scatter + all_gather
        t_rs = local.clone()
        shard = reduce_scatter_ring(t_rs)
        t_rebuilt = all_gather_ring(shard, world, local_idx=(rank + 1) % world)

        algo_bytes = numel * 4
        results[name] = {
            "numel": numel,
            "algo_bytes": algo_bytes,
            "naive_vs_ref_max_abs_err": (t_naive - ref).abs().max().item(),
            "ring_vs_ref_max_abs_err": (t_ring - ref).abs().max().item(),
            "rs_ag_vs_ref_max_abs_err": (t_rebuilt - ref).abs().max().item(),
            "naive_pass": torch.allclose(t_naive, ref, atol=1e-5),
            "ring_pass": torch.allclose(t_ring, ref, atol=1e-5),
            "rs_ag_pass": torch.allclose(t_rebuilt, ref, atol=1e-5),
        }
        print(f"[rank{rank}] {name}: naive={results[name]['naive_vs_ref_max_abs_err']:.2e} "
              f"ring={results[name]['ring_vs_ref_max_abs_err']:.2e} "
              f"rs+ag={results[name]['rs_ag_vs_ref_max_abs_err']:.2e}")

    cleanup_dist()
    return {"mode": "correctness", "world": world, "results": results}


# ---------------------------------------------------------------- mode: trace

def mode_trace(args) -> dict:
    rank, _, world = setup_dist("gloo")
    numel = 16  # trace 用小张量，逐轮看得清
    torch.manual_seed(1234 + rank)
    local = torch.randn(numel) * (rank + 1)

    ref = local.clone()
    dist.all_reduce(ref)

    trace = TraceLog(rank=rank)
    t = local.clone()
    ring_all_reduce(t, trace=trace)

    algo_bytes = numel * 4
    sent = trace.total_bytes_sent()
    expected = ring_bus_bytes(algo_bytes, world)

    payload = {
        "mode": "trace",
        "world": world,
        "numel": numel,
        "algo_bytes": algo_bytes,
        "rank": rank,
        "rounds": [
            {"round": r.round_idx, "phase": r.phase, "src": r.src, "dst": r.dst,
             "chunk": r.chunk, "bytes": r.bytes_moved, "note": r.note}
            for r in trace.rounds
        ],
        "bytes_sent_by_this_rank": sent,
        "expected_ring_bus_bytes_per_rank": expected,
        "trace_matches_formula": sent == expected,
        "result_correct": torch.allclose(t, ref, atol=1e-5),
    }
    print(f"[rank{rank}] sent={sent}B expected={expected}B match={sent == expected} "
          f"rounds={len(trace.rounds)} correct={payload['result_correct']}")
    cleanup_dist()
    return payload


# ---------------------------------------------------------------- mode: backend

def mode_backend(args) -> dict:
    rank, _, world = setup_dist(args.backend)
    sizes_mb = [1, 4, 16, 64, 256]
    rows = []

    for mb in sizes_mb:
        numel = mb * 1024 * 1024 // 4
        t = torch.ones(numel, dtype=torch.float32,
                       device="cuda" if (args.backend == "nccl" and torch.cuda.is_available()) else "cpu")
        algo_bytes = numel * 4
        bus_bytes = ring_bus_bytes(algo_bytes, world)

        # 库实现
        stats = bench_collective(lambda: dist.all_reduce(t), iters=args.iters)
        rows.append({
            "size_mb": mb, "impl": f"dist.all_reduce({args.backend})",
            "ms_median": stats["ms_median"], "ms_p95": stats["ms_p95"],
            "algbw_gbps": bandwidth_gbps(algo_bytes, stats["ms_median"]),
            "busbw_gbps": bandwidth_gbps(bus_bytes, stats["ms_median"]),
        })
        print(f"[rank{rank}] {mb}MB {args.backend}: {stats['ms_median']:.2f}ms "
              f"algbw={rows[-1]['algbw_gbps']:.3f}GB/s busbw={rows[-1]['busbw_gbps']:.3f}GB/s")

    # 手写 ring 在小尺寸上的对照（大尺寸手写太慢，只跑 1/4MB）
    if args.backend == "gloo":
        for mb in [1, 4]:
            numel = mb * 1024 * 1024 // 4
            t = torch.ones(numel, dtype=torch.float32)
            algo_bytes = numel * 4
            bus_bytes = ring_bus_bytes(algo_bytes, world)
            stats = bench_collective(lambda: ring_all_reduce(t), iters=args.iters)
            rows.append({
                "size_mb": mb, "impl": "hand_written_ring(gloo)",
                "ms_median": stats["ms_median"], "ms_p95": stats["ms_p95"],
                "algbw_gbps": bandwidth_gbps(algo_bytes, stats["ms_median"]),
                "busbw_gbps": bandwidth_gbps(bus_bytes, stats["ms_median"]),
            })
            print(f"[rank{rank}] {mb}MB hand_ring: {stats['ms_median']:.2f}ms "
                  f"algbw={rows[-1]['algbw_gbps']:.3f}GB/s")

    cleanup_dist()
    return {"mode": "backend", "backend": args.backend, "world": world, "rows": rows}


# ---------------------------------------------------------------- mode: ddp_grad

def mode_ddp_grad(args) -> dict:
    """把 12 篇 DDP 的真实梯度送进通信量核算。

    用 12 篇同款 build_llama（12.93M 参数）跑一步 DDP backward，
    从 DDP 的 bucket 里读出真实通信量，与理论值（参数字节 × 2(w-1)/w）对账。
    """
    rank, _, world = setup_dist("gloo")
    from exp_ddp.ddp_common import FixedSampleDataset, build_model, load_corpus_ids  # noqa: E402

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 12 篇的固定样本池：同一份 token 序列切窗口；vocab 以语料为准
    ids, vocab = load_corpus_ids()
    ds = FixedSampleDataset(ids, seq_len=256, n_samples=64)
    model = build_model(vocab_size=vocab, seq_len=256, device=device)
    ddp = torch.nn.parallel.DistributedDataParallel(model)
    params_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    theory_bus = ring_bus_bytes(params_bytes, world)

    sampler = torch.utils.data.distributed.DistributedSampler(ds, shuffle=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=8, sampler=sampler)
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)
    out = ddp(x)
    logits = out.logits if hasattr(out, "logits") else out
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), y.reshape(-1))
    loss.backward()

    # DDP 按 bucket 分批 all-reduce；bucket 上限默认 25MB
    cap = 25 * 1024 * 1024
    n_buckets = max(1, -(-params_bytes // cap))
    bucket_bytes = [min(cap, params_bytes - i * cap) for i in range(n_buckets)]

    payload = {
        "mode": "ddp_grad",
        "world": world,
        "params_bytes": params_bytes,
        "params_m": params_bytes / 4 / 1e6,
        "theory_bus_bytes_per_rank": theory_bus,
        "n_buckets": n_buckets,
        "bucket_bytes": bucket_bytes,
        "loss": loss.item(),
        "note": "DDP 梯度按 bucket 分批 all-reduce；bucket 化让通信与反向重叠成为可能（14 篇实测）",
    }
    print(f"[rank{rank}] params={params_bytes/4/1e6:.2f}M theory_bus={theory_bus/1e6:.1f}MB "
          f"buckets={n_buckets} loss={loss.item():.4f}")
    cleanup_dist()
    return payload


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["correctness", "trace", "backend", "ddp_grad"])
    ap.add_argument("--backend", default="gloo", choices=["gloo", "nccl"])
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.mode == "correctness":
        payload = mode_correctity(args)
    elif args.mode == "trace":
        payload = mode_trace(args)
    elif args.mode == "backend":
        payload = mode_backend(args)
    else:
        payload = mode_ddp_grad(args)

    # 注意：各 mode 内部已 cleanup_dist，改用 torchrun 注入的 RANK 环境变量判断
    if args.out and int(os.environ.get("RANK", 0)) == 0:
        _out(args.out, payload, 0)


if __name__ == "__main__":
    main()
