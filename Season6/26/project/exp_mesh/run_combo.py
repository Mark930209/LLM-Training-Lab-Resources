"""run_combo.py —— 26 篇组合并行驱动：DP×TP×PP + parity + 失败变体。

用法（torchrun；2-GPU 两机 REAL 或单机 4-rank SCALED）：
    python run_combo.py --dp 2 --tp 1 --pp 1 --micro 2 --out results/x.json
    python run_combo.py --dp 2 --tp 2 --pp 1 --bug dupdata --out ...
    python run_combo.py --dp 1 --tp 2 --pp 2 --ckpt-dir DIR --out ...

验收三件套：rank 映射表（报告内）、输出/梯度 parity、checkpoint 往返。
SUM 归约口径：reference 定义为全批损失之和，DP 归约用 SUM 不用 MEAN。
"""

from __future__ import annotations

import argparse
import hashlib
import os
import time

import torch
import torch.distributed as dist

from mesh import Mesh, build_groups
from mm_runtime import (RefStage, TPStage, _CopyToTP, _ReduceFromTP,
                        _ref_block_params, broadcast_params)


def parity(a: torch.Tensor, b: torch.Tensor, atol=2e-4, rtol=1e-4) -> dict:
    a32, b32 = a.detach().float(), b.detach().float()
    diff = (a32 - b32).abs()
    maxabs = diff.max().item()
    scale = b32.abs().max().clamp_min(1e-6).item()
    return {"maxabs": maxabs, "norm_maxabs": round(maxabs / scale, 6),
            "within_tol": bool(maxabs <= atol + rtol * scale)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dp", type=int, default=1)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--pp", type=int, default=1)
    ap.add_argument("--micro", type=int, default=2)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=384)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--backend", choices=["gloo", "nccl"], default="gloo")
    ap.add_argument("--bug", choices=["none", "meshswap", "dupdata", "nometa"],
                    default="none")
    ap.add_argument("--ckpt-dir", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rank, world = int(os.environ.get("RANK", 0)), \
        int(os.environ.get("WORLD_SIZE", 1))
    if world > 1 and not dist.is_initialized():
        dist.init_process_group(args.backend)
    # 设备层：nccl 后端要求 GPU 张量（两机 REAL），gloo 走 CPU（SCALED）
    use_cuda = (args.backend == "nccl" and torch.cuda.is_available())
    dev_idx = int(os.environ.get("LOCAL_RANK", 0))
    if use_cuda:
        torch.cuda.set_device(dev_idx)
    device = f"cuda:{dev_idx}" if use_cuda else "cpu"
    mesh = Mesh(args.dp, args.tp, args.pp, world)
    groups = build_groups(mesh, rank, swap_tp=(args.bug == "meshswap"))
    tp_members, tp_group = groups["tp"]
    pp_chain, _ = groups["pp"]
    dp_members, dp_group = groups["dp"]
    _CopyToTP.group = _ReduceFromTP.group = tp_group
    _CopyToTP.world = _ReduceFromTP.world = args.tp

    # ---- 权重：rank0 生成后广播（随机数纪律）
    params_list = [_ref_block_params(args.hidden, args.heads, 20260930 + i)
                   for i in range(args.blocks)]
    for p in params_list:
        for k in p:
            p[k] = p[k].to(device)
            if world > 1:
                dist.broadcast(p[k], src=0)

    # ---- 数据：全局批 = dp × batch × micro，rank0 生成后广播
    n_global = args.dp * args.batch * args.micro
    torch.manual_seed(31337)
    xs_all = torch.randn(n_global, 1, args.seq, args.hidden, device=device)
    gcoef = torch.randn(n_global, 1, args.seq, args.hidden, device=device)
    if world > 1:
        dist.broadcast(xs_all, src=0)
        dist.broadcast(gcoef, src=0)
    dp_idx, pp_idx, tp_idx = mesh.coords(rank)
    lo = dp_idx * (args.batch * args.micro)
    if args.bug == "dupdata":
        lo = 0                            # 所有 dp rank 拿同一片：数据重复
    xs_local = [xs_all[lo + i] for i in range(args.batch * args.micro)]
    gc_local = [gcoef[lo + i] for i in range(args.batch * args.micro)]
    data_sha = hashlib.sha256(
        torch.cat([x.flatten()[:64] for x in xs_local]).cpu().numpy()
        .tobytes()).hexdigest()[:12]

    # ---- 本 stage 的块
    per = args.blocks // args.pp
    my_blocks = list(range(pp_idx * per, (pp_idx + 1) * per))
    stage = TPStage(args.hidden, args.heads,
                    [params_list[i] for i in my_blocks], args.tp, tp_idx)
    first, last = (pp_idx == 0), (pp_idx == args.pp - 1)
    nxt = pp_chain[pp_chain.index(rank) + 1] if not last else None
    prv = pp_chain[pp_chain.index(rank) - 1] if not first else None

    def fwd_bwd():
        for p in stage.parameters():
            p.grad = None
        loss_val = 0.0
        acts = []
        sends: list = []

        def isend_keep(buf, dst):
            # gloo 的 isend().wait() 实为阻塞：交错下会 send-send 交叉死锁
            # （25 篇同款坑），延迟 wait，缓冲引用随句柄一起持有
            sends.append((dist.isend(buf, dst), buf))
            if len(sends) > 4:
                h, _ = sends.pop(0)
                h.wait()

        for i in range(len(xs_local)):
            x = xs_local[i].clone().requires_grad_(True)
            if not first:
                buf = torch.empty_like(x)
                dist.recv(buf, prv)
                x = buf.detach().requires_grad_(True)
            y = stage(x)
            if not last:
                isend_keep(y.contiguous(), nxt)
            else:
                l = (y * gc_local[i]).sum()
                loss_val += float(l.detach())
                l.backward()
                if prv is not None:
                    isend_keep(x.grad.contiguous(), prv)
            acts.append((x, y))
        if not last:
            for i in range(len(acts)):
                x, y = acts[i]
                g = torch.empty_like(y)
                dist.recv(g, nxt)
                torch.autograd.backward(y, g)
                if prv is not None:
                    isend_keep(x.grad.contiguous(), prv)
        while sends:
            h, _ = sends.pop(0)
            h.wait()
        return loss_val

    t0 = time.perf_counter()
    loss_local = fwd_bwd()
    if args.tp > 1 or args.dp > 1:
        pass
    # ---- DP 梯度 SUM 归约
    if args.dp > 1:
        for p in stage.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, group=dp_group)
    # ---- DP loss 求和（pp 末段才有 loss）
    loss_t = torch.tensor([loss_local if last else 0.0], device=device)
    if args.dp > 1:
        dist.all_reduce(loss_t, group=dp_group)
    wall = time.perf_counter() - t0

    # ---- 参考：单进程全量模型（所有块，同权重同数据）
    ref_stage = RefStage(args.hidden, args.heads, params_list)
    for p in ref_stage.parameters():
        p.grad = None
    loss_ref_local = 0.0
    for i in range(len(xs_all)):
        x = xs_all[i].clone().requires_grad_(True)
        y = ref_stage(x)
        l = (y * gcoef[i]).sum()
        loss_ref_local += float(l.detach())
        l.backward()
    # 参考 loss 是全批损失，每个 rank 算出的都一样，不再 DP 归约
    # （归约会重复计数成 dp 倍）
    loss_ref_t = torch.tensor([loss_ref_local if last else 0.0], device=device)

    # ---- parity：本 stage 参数梯度（名字映射配对，TP 分片对参考切片）
    TP2REF = {"q.weight": "qw", "q.bias": "qb", "k.weight": "kw",
              "k.bias": "kb", "v.weight": "vw", "v.bias": "vb",
              "o.weight": "ow", "o.bias": "ob",
              "fc1.weight": "fc1w", "fc1.bias": "fc1b",
              "fc2.weight": "fc2w", "fc2.bias": "fc2b"}

    def pair_parity(tp_mod, ref_mod, tp_i, tp_n):
        out = {}
        ref_map = {}
        for n, r in ref_mod.named_parameters():
            parts = n.split(".", 2)
            ref_map[f"{parts[1]}.{parts[2]}"] = r
        for n, p in tp_mod.named_parameters():
            parts = n.split(".", 2)
            tail = TP2REF.get(parts[2], parts[2])
            # stage 内局部块号 → 全局块号（PP 下两者不同，配对错就成
            # "拿 stage1 的块 0 比参考的块 0"——值全错但 shape 全对）
            gid = my_blocks[int(parts[1])]
            r = ref_map.get(f"{gid}.{tail}")
            if r is None or p.grad is None or r.grad is None:
                continue
            key = f"{gid}.{tail}"
            if p.shape == r.shape:
                out[key] = parity(p.grad, r.grad)
            elif p.dim() == 2 and p.shape[1] == r.shape[1] // tp_n:
                # row 层权重：按列（输入维）切
                shard = r.shape[1] // tp_n
                out[key] = parity(
                    p.grad, r.grad[:, tp_i * shard:(tp_i + 1) * shard])
            elif p.dim() == 2:
                # col 层权重：按行（输出维）切
                shard = r.shape[0] // tp_n
                out[key] = parity(
                    p.grad, r.grad[tp_i * shard:(tp_i + 1) * shard])
            else:
                shard = r.shape[0] // tp_n
                out[key] = parity(
                    p.grad, r.grad[tp_i * shard:(tp_i + 1) * shard])
        return out

    pg = pair_parity(stage, ref_stage, tp_idx, args.tp)

    ok_loss = (not last) or parity(loss_t, loss_ref_t)["within_tol"]
    ok_grads = all(v["within_tol"] for v in pg.values())

    report = {
        "config": vars(args),
        "mesh": {"dp": args.dp, "tp": args.tp, "pp": args.pp,
                 "map": mesh.table()},
        "self_check": mesh.self_check(),
        "rank": rank,
        "tp_members": tp_members,
        "pp_chain": pp_chain,
        "dp_members": dp_members,
        "data_sha": data_sha,
        "loss_combo": round(float(loss_t[0]), 6),
        "loss_ref": round(float(loss_ref_t[0]), 6),
        "parity_loss": parity(loss_t, loss_ref_t),
        "parity_grads": pg,
        "parity_grads_max_maxabs": max([v["maxabs"] for v in pg.values()]
                                       or [0.0]),
        "pass": bool(ok_loss and ok_grads),
        "wall_s": round(wall, 4),
    }

    if args.ckpt_dir:
        from ckpt_mesh import save_ckpt, load_ckpt
        if os.environ.get("CKPT_MODE", "save") == "save":
            report["ckpt"] = save_ckpt(args.ckpt_dir, stage, mesh, rank,
                                       my_blocks,
                                       no_meta=(args.bug == "nometa"))
        else:
            report["ckpt"] = load_ckpt(args.ckpt_dir, stage, mesh, rank,
                                       my_blocks)

    out = args.out.replace(".json", f".r{rank}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    import json
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
