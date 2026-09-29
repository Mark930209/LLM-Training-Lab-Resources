"""tp_parity.py —— 24 篇 E0~E3 + E6 的 parity 验收器。

用法（单进程自检 world=1，或 torchrun 两机 world=2）：
    python tp_parity.py --level linear-col|linear-row|mlp|attn|block \
        --bug none|bias|splitdim [--dtype fp32|bf16] --out results/xxx.json

验收判据（正文口径）：
    TP 改变了求和顺序，只能容差 parity，不能逐位（12 篇 DDP 门禁不适用）。
    fp32：绝对+相对组合容差 maxabs <= 1e-5 + 1e-4*max|ref| 判 PASS；bf16 只报数不判门。
    bug 变体预期 FAIL，并给出可诊断的签名：
        bias  变体 → 输出差 (world-1) 倍 bias（bias 被加了 world 次）
        split 变体 → shape 全对、maxrel 巨大（静默数值错误）
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import torch.distributed as dist

from tp_common import CommAccount, env_rank_world, parity, run_meta, save_report
from tp_layers import (RefAttention, RefBlock, RefMLP, _ScatterToTPRegion,
                       bind_comm, tp_attn_from_ref, tp_block_from_ref,
                       tp_mlp_from_ref, tp_from_ref_linear_col,
                       tp_from_ref_linear_row)

TOL_FP32 = 1e-4


def build(level, hidden, inter, heads, world, rank, bug, device, dtype):
    """构建参考模块与对应 TP 模块（TP 权重 = 参考权重切片，严格对应）。"""
    torch.manual_seed(20260929)
    if level == "linear-col":
        ref = torch.nn.Linear(hidden, hidden, bias=True).to(device)
        tp = tp_from_ref_linear_col(ref, world, rank,
                                    split_bug=(bug == "splitdim"))
    elif level == "linear-row":
        ref = torch.nn.Linear(hidden, hidden, bias=True).to(device)
        tp = tp_from_ref_linear_row(ref, world, rank, bug=bug)
    elif level == "mlp":
        ref = RefMLP(hidden, inter).to(device)
        tp = tp_mlp_from_ref(ref, world, rank, bug=bug)
    elif level == "attn":
        ref = RefAttention(hidden, heads).to(device)
        tp = tp_attn_from_ref(ref, world, rank, bug=bug)
    elif level == "block":
        ref = RefBlock(hidden, inter, heads).to(device)
        tp = tp_block_from_ref(ref, world, rank, bug=bug)
    else:
        raise ValueError(level)
    return ref.to(dtype), tp.to(dtype)


def grad_snapshot(module):
    return {n: p.grad.detach().clone() for n, p in module.named_parameters()
            if p.grad is not None}


def weight_parity(tp_snap, ref_snap, world, rank):
    """梯度对账：column 切行（out）、row 切列（in），与参考切片对齐。"""
    out = {}
    for name, gref in ref_snap.items():
        if name not in tp_snap:
            continue
        gtp = tp_snap[name]
        if gtp.shape == gref.shape:
            out[name] = parity(gtp, gref)
        elif gtp.dim() == 1:
            shard = gref.shape[0] // world
            sl = gref[rank * shard:(rank + 1) * shard]
            out[name] = parity(gtp, sl)
        elif gtp.shape[0] == gref.shape[0]:
            # 输入维分片（row parallel weight）：按列切
            shard = gref.shape[1] // world
            sl = gref[:, rank * shard:(rank + 1) * shard]
            out[name] = parity(gtp, sl)
        else:
            # 输出维分片（column parallel weight）：按行切
            shard = gref.shape[0] // world
            sl = gref[rank * shard:(rank + 1) * shard, :]
            out[name] = parity(gtp, sl)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", required=True)
    ap.add_argument("--bug", default="none",
                    choices=["none", "bias", "splitdim"])
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--hidden", type=int, default=1536)
    ap.add_argument("--inter", type=int, default=4096)
    ap.add_argument("--heads", type=int, default=24)
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
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16

    comm = CommAccount(world)
    bind_comm(comm, world, rank)

    ref, tp = build(args.level, args.hidden, args.inter, args.heads,
                    world, rank, args.bug, device, dtype)

    # 随机输入必须跨 rank 一致：两机 torch 版本不同，同 seed 生成的
    # 随机流不同（2.14 vs 2.11 实测），混进 AllReduce 就是静默错误。
    torch.manual_seed(31337)
    xb = torch.randn(args.batch, args.seq, args.hidden, device=device,
                     dtype=dtype)
    torch.manual_seed(777)
    gcoef = torch.randn(args.batch, args.seq, args.hidden, device=device,
                        dtype=torch.float32)
    if world > 1:
        dist.broadcast(xb, src=0)
        dist.broadcast(gcoef, src=0)
    x_ref = xb.clone().requires_grad_(True)
    x_tp = xb.clone().requires_grad_(True)

    # ---- 参考路径
    y_ref = ref(x_ref)
    (y_ref.float() * gcoef).sum().backward()
    ref_grads = grad_snapshot(ref)
    ref_xgrad = x_ref.grad.detach().clone()
    y_ref_d = y_ref.detach().clone()

    # ---- TP 路径（输出为本地分片时，损失系数也要切同样的片：
    #      分片损失之和 = 全量损失，反向才能对齐参考梯度）
    # 独立 row-parallel 的输入要先切成本地分片（组合模块里这一步由
    # column 层的输出分片天然完成）
    t0 = time.perf_counter()
    x_in = (_ScatterToTPRegion.apply(x_tp) if args.level == "linear-row"
            else x_tp)
    y_tp = tp(x_in)
    if y_tp.shape[-1] != gcoef.shape[-1]:
        shard_c = gcoef.shape[-1] // world
        gcoef_tp = gcoef[..., rank * shard_c:(rank + 1) * shard_c]
    else:
        gcoef_tp = gcoef
    (y_tp.float() * gcoef_tp).sum().backward()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_tp = time.perf_counter() - t0
    tp_grads = grad_snapshot(tp)

    # ---- 输出 parity：row 结尾的输出是复制全量；col 结尾是本地分片
    if y_tp.shape == y_ref_d.shape:
        p_out = parity(y_tp, y_ref_d)
    else:
        shard = y_ref_d.shape[-1] // world
        sl = y_ref_d[..., rank * shard:(rank + 1) * shard]
        p_out = parity(y_tp, sl)

    report = {
        "meta": run_meta(rank, world, {**vars(args), "dtype": args.dtype}),
        "wall_tp_s": round(t_tp, 4),
        "comm": comm.summary(),
        "parity_output": p_out,
        "parity_input_grad": parity(x_tp.grad, ref_xgrad),
        "parity_weight_grads": weight_parity(tp_grads, ref_grads, world, rank),
    }

    # ---- bug 签名：bias 被加了 world 次
    if args.bug == "bias":
        b = None
        for m in tp.modules():
            if getattr(m, "bias_before_reduce", False) and m.bias is not None:
                b = m.bias
        if b is not None:
            report["bias_signature"] = {
                "world": world,
                "max_bias_abs": b.detach().abs().max().item(),
                "expected_diff_floor": (world - 1) * b.detach().abs().max().item(),
            }
    expect_fail = args.bug != "none"
    oks = [p_out["within_tol"]] + [v["within_tol"] for v in
                                   report["parity_weight_grads"].values()]
    oks += [report["parity_input_grad"]["within_tol"]]
    passed = bool(all(oks)) if args.dtype == "fp32" else None
    report["verdict"] = {
        "expect_fail": expect_fail,
        "passed": passed,
        "consistent_with_expectation": (passed == (not expect_fail))
        if passed is not None else None,
    }
    save_report(args.out.replace(".json", f".r{rank}.json"), report)


if __name__ == "__main__":
    main()
