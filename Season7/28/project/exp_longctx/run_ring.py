"""run_ring.py —— E3 上下文并行（Ring 风格分块 attention）两进程实验。

每个 rank 持有序列的一半（Q_i, K_i, V_i），KV 沿 ring 轮转 P 步：
每步用收到的 KV 块算局部 Q 的部分 attention（带正确 causal/块偏置），
softmax 统计量在线合并（max/sumexp/加权 value），通信只传 KV 块。

报告：与单卡全量 attention 的 parity（容差 2e-4 + rtol 1e-4）、每步通信字节、
step time。用法（torchrun 2 进程，gloo=CPU 或 nccl=GPU）：
    torchrun --nproc_per_node=2 run_ring.py --seq 8192 --out out/e3_ring.json
"""

from __future__ import annotations

import argparse
import math
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

import longctx_common as lc


def full_attn(q, k, v):
    """单卡参照：全量 causal attention（q/k/v: [B,H,S,D]）。"""
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


def ring_attn(q_local, k_local, v_local, rank, world, log_bytes):
    """在线合并 softmax 统计量的 Ring attention（2 块简化实现）。"""
    B, H, Sq, D = q_local.shape
    Sk = k_local.shape[2]
    out = torch.zeros_like(q_local)
    m_prev = torch.full((B, H, Sq, 1), -float("inf"), device=q_local.device)
    l_prev = torch.zeros(B, H, Sq, 1, device=q_local.device)
    acc = torch.zeros_like(q_local)

    k_recv, v_recv = k_local.clone(), v_local.clone()
    for step in range(world):
        owner = (rank + step) % world           # 当前 KV 块的原始 rank
        # 该 KV 块覆盖的全局位置 [owner*Sk, owner*Sk+Sk)
        lo, hi = owner * Sk, owner * Sk + Sk
        qpos = torch.arange(rank * Sq, rank * Sq + Sq, device=q_local.device)
        kpos = torch.arange(lo, hi, device=q_local.device)
        allow = kpos[None, :] <= qpos[:, None]   # [Sq, Sk] causal
        scores = (q_local.float() @ k_recv.float().transpose(-1, -2)) / math.sqrt(D)
        scores = scores.masked_fill(~allow[None, None], -float("inf"))
        m = scores.amax(dim=-1, keepdim=True)
        m = torch.maximum(m_prev, m)
        p = torch.exp(scores - m)
        l = torch.exp(m_prev - m) * l_prev + p.sum(dim=-1, keepdim=True)
        acc = acc * torch.exp(m_prev - m) + p @ v_recv.float()
        m_prev, l_prev = m, l
        # ring 传递 KV（isend/irecv 保序；字节记账）
        nxt, prv = (rank + 1) % world, (rank - 1) % world
        kb, vb = k_recv.clone(), v_recv.clone()
        sk = [torch.empty_like(kb), torch.empty_like(vb)]
        rk = [torch.empty_like(kb), torch.empty_like(vb)]
        ops = [dist.P2POp(dist.isend, kb, nxt), dist.P2POp(dist.isend, vb, nxt),
               dist.P2POp(dist.irecv, rk[0], prv), dist.P2POp(dist.irecv, rk[1], prv)]
        for w in dist.batch_isend_irecv(ops):
            w.wait()
        log_bytes += kb.numel() * 4 * 2      # 发送 KV 两个张量，fp32
        k_recv, v_recv = rk[0], rk[1]
    return (acc / l_prev).type_as(q_local)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=8192)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", choices=["gloo", "nccl"], default="nccl")
    args = ap.parse_args()

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    dist.init_process_group(args.backend)
    use_cuda = args.backend == "nccl" and torch.cuda.is_available()
    dev = f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}" if use_cuda else "cpu"
    if use_cuda:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    assert args.seq % world == 0
    S = args.seq
    torch.manual_seed(20260930)
    B, H, D = 1, 16, 16
    q = torch.randn(B, H, S, D, device=dev)
    k = torch.randn(B, H, S, D, device=dev)
    v = torch.randn(B, H, S, D, device=dev)

    sl = S // world
    q_l = q[:, :, rank * sl:(rank + 1) * sl].contiguous()
    k_l = k[:, :, rank * sl:(rank + 1) * sl].contiguous()
    v_l = v[:, :, rank * sl:(rank + 1) * sl].contiguous()

    log_bytes = 0
    if use_cuda:
        torch.cuda.synchronize()
    import time
    wt0 = time.perf_counter()
    out_l = ring_attn(q_l, k_l, v_l, rank, world, 0)
    ring_s = time.perf_counter() - wt0

    ref = full_attn(q, k, v)[:, :, rank * sl:(rank + 1) * sl]
    diff = (out_l.float() - ref.float()).abs().max()
    scale = ref.float().abs().max().clamp_min(1e-6)
    tol = 2e-4 + 1e-4 * scale.item()
    result = {
        "experiment": f"e3_ring_{args.backend}",
        "seq": S, "world": world, "backend": args.backend,
        "kv_bytes_per_rank_per_step": sl * H * D * 4 * 2 * B,
        "ring_steps": world,
        "comm_bytes_per_rank_total": sl * H * D * 4 * 2 * B * (world - 1),
        "ring_seconds": round(ring_s, 4),
        "maxabs": diff.item(),
        "within_tol": bool(diff.item() <= tol),
        "labels": {"REAL": "两进程真实运行"},
    }
    if rank == 0:
        lc.save_json(result, args.out)
    print(f"[ring r{rank}] maxabs {diff.item():.3e} tol {tol:.3e} "
          f"comm {result['comm_bytes_per_rank_total']} B")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
