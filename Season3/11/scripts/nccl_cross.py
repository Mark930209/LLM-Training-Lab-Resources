"""跨机 NCCL 最小验证脚本。

两个 rank 分别跑在两台机器的两块 GPU 上，验证三件事：
  1. init_process_group(nccl) 能否完成握手
  2. all_reduce 的数学结果是否正确（rank0 出 1.0，rank1 出 2.0，求和应为 3.0）
  3. 大张量 all_reduce 的实际带宽与延迟

用法（两台机器各跑一次）：
  torchrun --nnodes=2 --node_rank=0 --nproc_per_node=1 \
      --master_addr=<rank0 的 tailscale IP> --master_port=29500 nccl_cross.py
"""

import os
import time

import torch
import torch.distributed as dist


def log(rank, msg):
    print(f"[rank{rank}] {msg}", flush=True)


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.cuda.set_device(local_rank)
    dev = torch.device(f"cuda:{local_rank}")

    name = torch.cuda.get_device_name(local_rank)
    log(rank, f"world_size={world} device={dev} gpu={name}")
    log(rank, f"torch={torch.__version__} nccl={torch.cuda.nccl.version()}")

    # --- 1. 正确性：all_reduce SUM ---
    t = torch.tensor([rank + 1.0], device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    expected = float(world * (world + 1) / 2)
    ok = abs(t.item() - expected) < 1e-6
    log(rank, f"all_reduce SUM -> {t.item():.6f} (expected {expected:.6f}) {'OK' if ok else 'FAIL'}")

    # --- 2. 双向点对点 ---
    if rank == 0:
        buf = torch.tensor([12345.0], device=dev)
        dist.send(buf, dst=1)
        recv = torch.zeros(1, device=dev)
        dist.recv(recv, src=1)
        log(rank, f"p2p echo -> {recv.item():.1f}")
    else:
        recv = torch.zeros(1, device=dev)
        dist.recv(recv, src=0)
        dist.send(recv, dst=0)
        log(rank, f"p2p got -> {recv.item():.1f}")

    # --- 3. 带宽：不同尺寸 all_reduce ---
    log(rank, "=== all_reduce 带宽 ===")
    for mb in (1, 4, 16, 64, 256):
        n = mb * 1024 * 1024 // 4
        x = torch.ones(n, device=dev)
        dist.barrier()
        torch.cuda.synchronize()

        # 预热
        for _ in range(3):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        dist.barrier()

        iters = 20 if mb <= 16 else 8
        t0 = time.perf_counter()
        for _ in range(iters):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        per_iter_ms = dt / iters * 1000
        # ring all-reduce 的总线字节数 = 2*(world-1)/world * size
        algo_bytes = n * 4
        bus_bytes = algo_bytes * 2 * (world - 1) / world
        # dt 是 iters 次迭代的总时间：带宽 = 单次字节数 * 迭代数 / 总时间
        algbw = algo_bytes * iters / dt / 1e9
        busbw = bus_bytes * iters / dt / 1e9
        log(
            rank,
            f"{mb:>4} MB  {per_iter_ms:8.2f} ms/iter  "
            f"algbw={algbw:7.3f} GB/s  busbw={busbw:7.3f} GB/s",
        )

    dist.barrier()
    log(rank, "DONE")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
