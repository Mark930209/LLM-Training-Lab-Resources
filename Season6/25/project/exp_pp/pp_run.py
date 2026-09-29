"""pp_run.py —— 25 篇实验驱动：per-block 探针 / 2-stage pipeline 运行。

用法（torchrun 两机，或单进程自检）：
    python pp_run.py --mode probe --out results/Season6/25/probe.json
    python pp_run.py --mode run --schedule gpipe|1f1b --stage-blocks 4 \
        --micro 4 --out results/Stage6/25/xxx.json

--stage-blocks k：stage0 持 block 0..k-1，stage1 持 k..7（8 块栈）。
run 模式自带 parity 验收（输出/输入梯度/参数梯度 vs 单进程整模型）与
单卡参考计时（PP vs 单卡对照）。
"""

from __future__ import annotations

import argparse
import os
import statistics
import time

import torch
import torch.distributed as dist

from pp_common import (P2PBytes, Timeline, env_rank_world, parity,
                       run_meta, save_report, theoretical_bubble)
from pp_model import FF_MULTS, broadcast_params, build_stack, count_params, stack_forward
from pp_runtime import Stage, run_pp, run_reference, timed_run


def probe_blocks(stack, hidden: int, seq: int) -> list[dict]:
    rows = []
    x = torch.randn(1, seq, hidden, device="cuda")
    for i, blk in enumerate(stack):
        def fwd():
            blk.zero_grad(set_to_none=True)
            y = blk(x)
            y.pow(2).mean().backward()

        for _ in range(3):
            fwd()
        torch.cuda.synchronize()
        ts = []
        for _ in range(10):
            t0 = time.perf_counter()
            fwd()
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        rows.append({"block": i, "ff_mult": FF_MULTS[i],
                     "ms": round(statistics.median(ts) * 1000, 3)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["probe", "run"], required=True)
    ap.add_argument("--schedule", choices=["gpipe", "1f1b"], default="gpipe")
    ap.add_argument("--stage-blocks", type=int, default=4)
    ap.add_argument("--micro", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=1536)
    ap.add_argument("--heads", type=int, default=24)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--backend", choices=["nccl", "gloo"], default="nccl",
                    help="gloo = 单卡共享 SCALED：NCCL 不支持同 GPU 多进程互发")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rank, world = env_rank_world()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    # SCALED：多进程共享单卡时 local_rank 会超出本机 GPU 数，按实际数取模；
    # gloo 后端强制 CPU（NCCL 不支持同 GPU 多进程 send/recv）
    use_cuda = torch.cuda.is_available() and args.backend == "nccl"
    dev_idx = (local_rank if use_cuda
               and local_rank < torch.cuda.device_count() else 0)
    device = f"cuda:{dev_idx}" if use_cuda else "cpu"
    if use_cuda:
        torch.cuda.set_device(dev_idx)
    if world > 1 and not dist.is_initialized():
        dist.init_process_group(args.backend)

    stack = build_stack(args.hidden, args.heads).to(device)
    broadcast_params(stack)
    n_params = count_params(stack)

    if args.mode == "probe":
        rows = probe_blocks(stack, args.hidden, args.seq)
        save_report(args.out.replace(".json", f".r{rank}.json"), {
            "meta": run_meta(rank, world, {**vars(args), "n_params": n_params}),
            "rows": rows,
        })
        return

    # ---- run 模式：2-stage 用 --stage-blocks k 切；N-stage 均匀切
    n_blk = len(stack)
    if world == 2:
        k = args.stage_blocks
        my_blocks = list(range(0, k)) if rank == 0 else list(range(k, n_blk))
    else:
        per = n_blk // world
        my_blocks = list(range(rank * per, (rank + 1) * per))
    stage = Stage(stack, my_blocks, rank, world)

    # 输入与损失系数：rank0 生成后广播（两机 RNG 不一致，24 篇教训）
    torch.manual_seed(31337)
    xs_cat = torch.randn(args.micro, 1, args.seq, args.hidden, device=device)
    gcoef = torch.randn(args.micro, 1, args.seq, args.hidden, device=device)
    if world > 1:
        dist.broadcast(xs_cat, src=0)
        dist.broadcast(gcoef, src=0)
    xs = [xs_cat[i].detach().requires_grad_(True)
          for i in range(args.micro)]

    tl = Timeline()
    p2p = P2PBytes()
    # P2P 预热：unbatched send/recv 首次调用会新建 NCCL communicator，
    # 这笔一次性开销不能混进计时（实测 m1 壁钟 1.11s 里近半是它）
    if world > 1:
        warm = torch.ones(1, args.seq, args.hidden, device=device)
        # 偶数 rank 先发：预热 (0,1)(2,3) 相邻对的 P2P communicator
        if rank % 2 == 0:
            if rank + 1 < world:
                dist.send(warm, rank + 1)
                dist.recv(warm, rank + 1)
        else:
            dist.recv(warm, rank - 1)
            dist.send(warm, rank - 1)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    wall_pp, loss_pp = timed_run(lambda: run_pp(stage, xs, gcoef,
                                                args.schedule, tl, p2p))
    peak_pp = (torch.cuda.max_memory_allocated() >> 20
               if torch.cuda.is_available() else 0)
    pp_grads = {n: p.grad.detach().clone() for n, p in stack.named_parameters()
                if p.grad is not None}

    # ---- 单进程参考（同卡本地）：parity + 单卡计时
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    wall_ref, (loss_ref, ref_grads, ref_xgrads) = timed_run(
        lambda: run_reference(stack, xs, gcoef))
    peak_ref = (torch.cuda.max_memory_allocated() >> 20
                if torch.cuda.is_available() else 0)

    own = {n for n, _ in stack.named_parameters()
           if any(n.startswith(f"{i}.") for i in my_blocks)}
    p_grads = {}
    for n in sorted(own):
        if n in pp_grads and n in ref_grads:
            p_grads[n] = parity(pp_grads[n], ref_grads[n])

    # rank0 还比输入梯度（各 micro-batch）；末 stage 比 loss
    extra = {}
    if rank == 0:
        xparity = []
        for i in range(args.micro):
            xparity.append(parity(xs[i].grad, ref_xgrads[i]))
        extra["x_grad_parity"] = xparity
    if stage.last:
        extra["loss_parity"] = parity(
            torch.tensor([loss_pp]), torch.tensor([loss_ref]))
        extra["loss_pp"] = round(loss_pp, 6)
        extra["loss_ref"] = round(loss_ref, 6)

    ok_all = (all(v["within_tol"] for v in p_grads.values())
              and all(v["within_tol"] for v in extra.get("x_grad_parity", []))
              and extra.get("loss_parity", {"within_tol": True})["within_tol"])

    save_report(args.out.replace(".json", f".r{rank}.json"), {
        "meta": run_meta(rank, world, {**vars(args), "n_params": n_params,
                                       "my_blocks": my_blocks}),
        "pp_wall_s": round(wall_pp, 4),
        "ref_wall_s": round(wall_ref, 4),
        "pp_over_ref": round(wall_pp / wall_ref, 4),
        "peak_pp_mb": peak_pp,
        "peak_ref_mb": peak_ref,
        "timeline": tl.summary(),
        "theory_bubble": round(theoretical_bubble(world, args.micro), 4),
        "p2p": p2p.summary(),
        "parity_weight_grads": p_grads,
        "parity_max_maxabs": max([v["maxabs"] for v in p_grads.values()]
                                 or [0.0]),
        "parity_pass": bool(ok_all),
        **extra,
    })


if __name__ == "__main__":
    main()
