"""hand_written.py —— 手写集合通信：reduce+broadcast 与 Ring AllReduce。

只用 dist.send / dist.recv / dist.broadcast 三个点对点原语实现，
每一轮记录 trace。正确性判据：与 dist.all_reduce 的结果逐位一致。

Ring AllReduce 的两阶段（本篇的核心机制）：
  阶段一 ReduceScatter：w-1 轮，每轮每个 rank 把一个 chunk 发给右邻居并累加。
    结束后每个 rank 持有一个"完整归约好的 chunk"（自己负责的那份）。
  阶段二 AllGather：w-1 轮，每轮把完整 chunk 沿环传一圈。
    结束后每个 rank 持有完整的归约结果。

通信量（每 rank）：
  ReduceScatter 发送 (w-1)/w 份，AllGather 发送 (w-1)/w 份，
  合计 2(w-1)/w 份 —— 与 naive 的"rank0 收发 (w-1) 份"相比，
  环上没有热点，每 rank 的负载是均匀的。
"""

from __future__ import annotations

import torch
import torch.distributed as dist

try:
    from .collective_common import RoundTrace, TraceLog
except ImportError:  # 直接 python hand_written.py 调试时
    from collective_common import RoundTrace, TraceLog


# ---------------------------------------------------------------- naive：reduce + broadcast

def all_reduce_naive(tensor: torch.Tensor, dst: int = 0, trace: TraceLog | None = None) -> torch.Tensor:
    """最直观的 AllReduce：全部发给 rank0，rank0 求和后广播。

    语义正确，但 rank0 是瓶颈：它接收 (w-1) 份、发送 (w-1) 份，
    其他 rank 各只收发 1 份。w 大时 rank0 的链路被压满，其他 rank 空等。
    """
    rank = dist.get_rank()
    world = dist.get_world_size()
    algo_bytes = tensor.numel() * tensor.element_size()

    # reduce 阶段：所有非 dst rank 发给 dst
    if rank == dst:
        acc = tensor.clone()
        for src in range(world):
            if src == dst:
                continue
            buf = torch.empty_like(tensor)
            dist.recv(buf, src=src)
            acc += buf
            if trace is not None:
                trace.add(RoundTrace(round_idx=src, phase="reduce",
                                     src=src, dst=dst, chunk=-1,
                                     bytes_moved=algo_bytes,
                                     note=f"rank{src} -> rank{dst} 整块"))
        tensor.copy_(acc)
    else:
        dist.send(tensor, dst=dst)
        if trace is not None:
            trace.add(RoundTrace(round_idx=rank, phase="reduce",
                                 src=rank, dst=dst, chunk=-1,
                                 bytes_moved=algo_bytes,
                                 note=f"rank{rank} -> rank{dst} 整块"))

    # broadcast 阶段：dst 把结果广播给所有人
    dist.broadcast(tensor, src=dst)
    if trace is not None:
        trace.add(RoundTrace(round_idx=0, phase="broadcast",
                             src=dst, dst=-1, chunk=-1,
                             bytes_moved=algo_bytes if rank == dst else 0,
                             note=f"rank{dst} 广播整块" + ("（本 rank 发送）" if rank == dst else "（本 rank 接收）")))
    return tensor


# ---------------------------------------------------------------- 手写 Ring AllReduce

def ring_all_reduce(tensor: torch.Tensor, trace: TraceLog | None = None) -> torch.Tensor:
    """手写 Ring AllReduce。要求 world <= chunk 数（本篇 w=2 或 4，chunk=8）。

    阶段一 ReduceScatter（w-1 轮）：
      第 r 轮，rank i 把 chunk (i - r) % w 发给右邻居 (i+1)%w，
      并把收到的 chunk 累加到自己的对应 chunk 上。
      经过 w-1 轮，rank i 手里的 chunk (i+1) % w 已累加了全部 w 份。
    阶段二 AllGather（w-1 轮）：
      第 r 轮，rank i 把"已完整"的 chunk 沿环传给右邻居。
      经过 w-1 轮，所有 rank 拿到全部完整 chunk。
    """
    world = dist.get_world_size()
    rank = dist.get_rank()
    assert tensor.numel() % world == 0, "演示版要求 numel 能被 world 整分"
    chunks = list(tensor.chunk(world, dim=0))
    chunk_bytes = chunks[0].numel() * tensor.element_size()
    right = (rank + 1) % world

    # ---- 阶段一：ReduceScatter ----
    for r in range(world - 1):
        send_idx = (rank - r) % world
        recv_idx = (rank - 1 - r) % world
        send_buf = chunks[send_idx].clone()
        req = dist.isend(send_buf, dst=right)
        recv_buf = torch.empty_like(chunks[recv_idx])
        dist.recv(recv_buf, src=(rank - 1) % world)
        req.wait()
        chunks[recv_idx] += recv_buf
        if trace is not None:
            trace.add(RoundTrace(round_idx=r, phase="reduce_scatter",
                                 src=rank, dst=right, chunk=send_idx,
                                 bytes_moved=chunk_bytes,
                                 note=f"轮{r}: rank{rank} 把 chunk{send_idx} 发给 rank{right}，累加 chunk{recv_idx}"))

    # ---- 阶段二：AllGather ----
    for r in range(world - 1):
        send_idx = (rank + 1 - r) % world
        recv_idx = (rank - r) % world
        send_buf = chunks[send_idx].clone()
        req = dist.isend(send_buf, dst=right)
        recv_buf = torch.empty_like(chunks[recv_idx])
        dist.recv(recv_buf, src=(rank - 1) % world)
        req.wait()
        chunks[recv_idx] = recv_buf.clone()
        if trace is not None:
            trace.add(RoundTrace(round_idx=r, phase="all_gather",
                                 src=rank, dst=right, chunk=send_idx,
                                 bytes_moved=chunk_bytes,
                                 note=f"轮{r}: rank{rank} 把完整 chunk{send_idx} 发给 rank{right}，接收 chunk{recv_idx}"))

    tensor.copy_(torch.cat(chunks, dim=0))
    return tensor


# ---------------------------------------------------------------- 拆开的两个原语

def reduce_scatter_ring(tensor: torch.Tensor, trace: TraceLog | None = None) -> torch.Tensor:
    """只做 Ring 的阶段一。返回本 rank 负责的那份完整 chunk（1/w 大小）。

    DDP 的梯度同步在 bucket 化后实际用的就是这个原语：
    各 rank 交换后各持有一份不同的"归约好的分片"。
    """
    world = dist.get_world_size()
    chunks = list(tensor.chunk(world, dim=0))
    chunk_bytes = chunks[0].numel() * tensor.element_size()
    rank = dist.get_rank()
    right = (rank + 1) % world

    for r in range(world - 1):
        send_idx = (rank - r) % world
        recv_idx = (rank - 1 - r) % world
        send_buf = chunks[send_idx].clone()
        req = dist.isend(send_buf, dst=right)
        recv_buf = torch.empty_like(chunks[recv_idx])
        dist.recv(recv_buf, src=(rank - 1) % world)
        req.wait()
        chunks[recv_idx] += recv_buf
        if trace is not None:
            trace.add(RoundTrace(round_idx=r, phase="reduce_scatter",
                                 src=rank, dst=right, chunk=send_idx,
                                 bytes_moved=chunk_bytes, note=""))

    return chunks[(rank + 1) % world]


def all_gather_ring(local_chunk: torch.Tensor, world: int,
                    local_idx: int | None = None, trace: TraceLog | None = None) -> torch.Tensor:
    """只做 Ring 的阶段二。把每个 rank 的 local_chunk 拼回完整张量。

    local_idx 是这份 shard 在完整张量里的位置。reduce_scatter_ring 返回的是
    (rank+1)%w 位置的 chunk，必须显式传入，否则拼回去会错位（本篇真实踩坑）。

    FSDP 的参数 materialize 用的就是这个原语：
    各 rank 各持 1/w 份参数，AllGather 后短暂持有完整参数做前向。
    """
    rank = dist.get_rank()
    if local_idx is None:
        local_idx = rank
    chunks = [None] * world
    chunks[local_idx] = local_chunk
    right = (rank + 1) % world

    for r in range(world - 1):
        # 轮转起点是 local_idx：第 r 轮发送 (local_idx - r) % w 位置的 chunk，
        # 接收 (local_idx - 1 - r) % w 位置。local_idx == rank 时退化为标准轮转。
        send_idx = (local_idx - r) % world
        recv_idx = (local_idx - 1 - r) % world
        send_buf = chunks[send_idx].clone()
        req = dist.isend(send_buf, dst=right)
        recv_buf = torch.empty_like(local_chunk)
        dist.recv(recv_buf, src=(rank - 1) % world)
        req.wait()
        chunks[recv_idx] = recv_buf.clone()
        if trace is not None:
            trace.add(RoundTrace(round_idx=r, phase="all_gather",
                                 src=rank, dst=right, chunk=send_idx,
                                 bytes_moved=local_chunk.numel() * local_chunk.element_size(), note=""))

    return torch.cat(chunks, dim=0)
