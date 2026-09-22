"""collective_common.py —— 13 篇 Collective Lab 的公共件。

设计要点（服务于"通信量可复算"这个唯一判据）：

1. 手写实现只用 dist.send / dist.recv / dist.broadcast 这三个点对点原语，
   不调用 dist.all_reduce —— 否则就是拿库验证库，什么也证明不了。

2. 每个手写实现的每一轮都记录 trace：谁发给谁、多少字节、归约了什么。
   trace 是本篇的核心交付，读者要能逐轮核对 Ring 的分块流动。

3. 通信量口径：algo_bytes = 张量本身的字节数（逻辑上必须移动的数据）；
   bus_bytes = 实际过网字节数。Ring AllReduce 的 bus_bytes =
   algo_bytes * 2*(w-1)/w，这个系数在 14 篇的 timeline 分解里直接复用。

4. 计时口径：torch.cuda.synchronize 后取 perf_counter，iters 次取中位数。
   11 篇的教训（带宽公式多除一次 iters）在这里从设计上排除：
   ms_per_iter 与 bytes_per_iter 先各自算好，带宽 = bytes_per_iter / (ms/1000)。
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import torch
import torch.distributed as dist


# ---------------------------------------------------------------- 进程组

def setup_dist(backend: str = "gloo") -> tuple[int, int, int]:
    """初始化进程组。13 篇的通信实验不依赖 GPU，gloo 足够；
    NCCL 对照组在跨机环境跑（backend 由命令行传入）。"""
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    return dist.get_rank(), int(os.environ.get("LOCAL_RANK", 0)), dist.get_world_size()


def cleanup_dist() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------- trace

@dataclass
class RoundTrace:
    """一轮通信的记录。ring 的每一轮、reduce+broadcast 的每一步都记一条。"""
    round_idx: int
    phase: str          # "reduce" / "gather" / "reduce_scatter" / "all_gather" / "broadcast"
    src: int            # 发送方 rank
    dst: int            # 接收方 rank（-1 表示 broadcast 到所有 rank）
    chunk: int          # chunk 编号（-1 表示整块）
    bytes_moved: int    # 本轮该 rank 实际发送的字节数
    note: str = ""


@dataclass
class TraceLog:
    """单个 rank 视角的完整 trace。"""
    rank: int
    rounds: list[RoundTrace] = field(default_factory=list)

    def add(self, r: RoundTrace) -> None:
        self.rounds.append(r)

    def total_bytes_sent(self) -> int:
        return sum(r.bytes_moved for r in self.rounds)


# ---------------------------------------------------------------- 通信量口径

def ring_bus_bytes(algo_bytes: int, world: int) -> int:
    """Ring AllReduce 的总线字节数：每个 rank 发送 2*(w-1)/w 份张量。
    w=2 时系数为 1，algbw == busbw（11 篇已实测验证）。"""
    return int(algo_bytes * 2 * (world - 1) / world)


def reduce_scatter_bytes(algo_bytes: int, world: int) -> int:
    """ReduceScatter：每个 rank 发送 (w-1)/w 份。"""
    return int(algo_bytes * (world - 1) / world)


def all_gather_bytes(algo_bytes: int, world: int) -> int:
    """AllGather：每个 rank 发送 (w-1)/w 份。"""
    return int(algo_bytes * (world - 1) / world)


def naive_bus_bytes(algo_bytes: int, world: int) -> int:
    """reduce+broadcast 的总线字节数：reduce 阶段 w-1 份 + broadcast 阶段 w-1 份，
    都经过 rank0，rank0 的发送/接收量是 (w-1) 份，其他 rank 是 1 份。
    返回的是"每 rank 平均"口径，用于与 ring 对比。"""
    return int(algo_bytes * 2 * (world - 1) / world)


# ---------------------------------------------------------------- 计时

def bench_collective(fn, iters: int = 20, warmup: int = 5) -> dict:
    """计时一个集合操作。fn() 执行一次操作。返回中位数与 P95。

    11 篇教训的防御：带宽不由本函数计算，只返回 ms；
    字节数由调用方按口径给出，带宽 = bytes / (ms/1000) 在报告层算，
    公式只写一处，杜绝"多除一次 iters"这类错误。
    """
    for _ in range(warmup):
        fn()
    dist.barrier()
    times = []
    for _ in range(iters):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    dist.barrier()
    times.sort()
    n = len(times)
    return {
        "ms_median": times[n // 2],
        "ms_p95": times[int(n * 0.95) - 1 if n > 1 else 0],
        "ms_min": times[0],
        "iters": iters,
    }


def bandwidth_gbps(bytes_per_iter: int, ms: float) -> float:
    """唯一带宽公式：单次字节数 / 单次耗时。"""
    return bytes_per_iter / (ms / 1000) / 1e9
