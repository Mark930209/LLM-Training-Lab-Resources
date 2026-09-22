"""bench_train.py —— 14 篇主实验：分段计时训练循环 + straggler 注入。

三种 mode：
  single : 单进程基线，逐步记录分段耗时
  ddp    : DDP 训练，同一口径记录；--stragger-ms 给指定 rank 注入延迟
  sweep  : 模型规模 × per-rank batch 扫描，找 compute/comm 临界点

用法（gloo 单机 2-rank）：
  python -m exp_bench.bench_train --mode single --steps 30 --global-batch 16
  torchrun --nproc_per_node=2 -m exp_bench.bench_train --mode ddp --steps 30 --global-batch 16
  torchrun --nproc_per_node=2 -m exp_bench.bench_train --mode ddp --steps 30 --global-batch 16 --straggler-ms 50
跨机 NCCL（复用 11 篇环境）：
  两侧 torchrun --nnodes=2 --nproc_per_node=1 ... --mode ddp --backend nccl
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.distributed as dist

from .bench_common import (StepTimer, SyntheticTokenDataset,
                           comm_bytes_estimate, comm_ms_estimate,
                           write_report)


def setup(backend: str, need_dist: bool):
    if need_dist and not dist.is_initialized():
        dist.init_process_group(backend=backend)
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and need_dist:
        torch.cuda.set_device(int(__import__("os").environ.get("LOCAL_RANK", 0)) % torch.cuda.device_count())
    return rank, world, device


def build_model(vocab: int, hidden: int, layers: int, device: str):
    """12 篇同款小 Llama 的极简版：性能实验只需要一个真实的 Transformer。"""
    from exp_ddp.ddp_common import build_model as build_llama
    return build_llama(vocab_size=vocab, seq_len=256, device=device,
                       hidden=hidden, layers=layers, heads=6, head_dim=64)


def make_loader(global_batch: int, world: int, ddp: bool, vocab: int, seq: int, steps: int):
    per_rank = global_batch // world
    # 数据集要够大：warmup + steps 个 batch，且能被 world 整除
    n = max(global_batch * (steps + 10), 512)
    n = (n // world) * world
    ds = SyntheticTokenDataset(n_samples=n, seq_len=seq, vocab_size=vocab)
    if ddp:
        sampler = torch.utils.data.distributed.DistributedSampler(ds, shuffle=False, drop_last=True)
        return torch.utils.data.DataLoader(ds, batch_size=per_rank, sampler=sampler, num_workers=0, drop_last=True)
    return torch.utils.data.DataLoader(ds, batch_size=global_batch, shuffle=False, num_workers=0)


def train_loop(args, ddp: bool) -> dict:
    rank, world, device = setup(args.backend, need_dist=ddp)
    torch.manual_seed(42)

    model = build_model(vocab=args.vocab, hidden=args.hidden, layers=args.layers, device=device)
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model)
    params_bytes = sum(p.numel() * p.element_size() for p in (model.module if ddp else model).parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    loader = make_loader(args.global_batch, world, ddp, args.vocab, args.seq, args.steps + args.warmup)
    timer = StepTimer(device)

    # warmup：不计入统计（口径之一：含/不含 warmup，审计脚本会重算）
    warmup_steps = args.warmup
    it = iter(loader)
    for _ in range(warmup_steps):
        x, y = next(it)
        x, y = x.to(device), y.to(device)
        out = model(x)
        logits = out.logits if hasattr(out, "logits") else out
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        opt.step()
        opt.zero_grad()

    steps = []
    it = iter(loader)
    for s in range(args.steps):
        timer.start()
        x, y = next(it)
        x, y = x.to(device), y.to(device)
        timer.mark_data()

        out = model(x)
        logits = out.logits if hasattr(out, "logits") else out
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        timer.mark_fwd()

        loss.backward()
        timer.mark_bwd()

        opt.step()
        opt.zero_grad()
        timer.mark_opt()

        # straggler 注入：指定 rank 在每步末尾睡 straggler_ms
        if args.straggler_ms > 0 and rank == args.straggler_rank:
            if device == "cuda":
                torch.cuda.synchronize()
            time.sleep(args.straggler_ms / 1000)
        # DDP 下用 barrier 让 straggler 的延迟传导到所有 rank
        if ddp and args.straggler_ms > 0:
            dist.barrier()
        timer.mark_eval()
        if device == "cuda":
            torch.cuda.synchronize()  # events 必须完成后才能 elapsed_time
        timer.finish()
        steps.append(dict(step=s, **timer.record))

    if dist.is_initialized():
        dist.barrier()

    total_ms = [st["total_ms"] for st in steps]
    total_ms.sort()
    n = len(total_ms)
    payload = {
        "mode": "ddp" if ddp else "single",
        "backend": args.backend,
        "world": world,
        "rank": rank,
        "config": {"hidden": args.hidden, "layers": args.layers, "seq": args.seq,
                   "global_batch": args.global_batch, "steps": args.steps,
                   "warmup": warmup_steps, "straggler_ms": args.straggler_ms},
        "params_bytes": params_bytes,
        "params_m": params_bytes / 4 / 1e6,
        "theory_comm_bytes": comm_bytes_estimate(params_bytes, world) if ddp else 0,
        "steps": steps,
        "total_ms_median": total_ms[n // 2],
        "total_ms_p95": total_ms[int(n * 0.95) - 1 if n > 1 else 0],
        "total_ms_mean": sum(total_ms) / n,
        "tok_per_s": args.global_batch * args.seq / (total_ms[n // 2] / 1000),
    }
    print(f"[rank{rank}] {'ddp' if ddp else 'single'} w={world} median={payload['total_ms_median']:.1f}ms "
          f"tok/s={payload['tok_per_s']:.0f} params={payload['params_m']:.2f}M")
    if dist.is_initialized():
        dist.destroy_process_group()
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["single", "ddp"])
    ap.add_argument("--backend", default="gloo", choices=["gloo", "nccl"])
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--global-batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--vocab", type=int, default=2048)
    ap.add_argument("--hidden", type=int, default=384)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--straggler-ms", type=float, default=0)
    ap.add_argument("--straggler-rank", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    payload = train_loop(args, ddp=(args.mode == "ddp"))
    if args.out:
        write_report(args.out, payload)


if __name__ == "__main__":
    main()
