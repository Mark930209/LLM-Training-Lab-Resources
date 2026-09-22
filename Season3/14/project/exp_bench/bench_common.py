"""bench_common.py —— 14 篇 Distributed Benchmark Harness 的公共件。

设计要点（服务于"口径先行"这个唯一原则）：

1. 每一步记录分段耗时：data / fwd / bwd / opt / eval / total。
   原始日志落盘后，口径审计脚本用同一批数据重算多种 speedup，
   证明"口径能把结论改到什么程度"。

2. 性能基准用合成数据（固定 seed 的随机 token），不依赖语料文件。
   12 篇已验证训练等价；本篇测的是时间，不是 loss。
   合成数据下 data 段接近 0，这本身是数据点（数据不是瓶颈）。

3. 计时用 CUDA events（GPU 侧）+ perf_counter（CPU 侧）双口径。
   跨机 NCCL 下 bwd 段包含 all-reduce；通信量用 13 篇的公式
   （参数字节 × 2(w-1)/w ÷ 实测带宽）估算，与 total 对账。

4. speedup 的定义写死在代码里，不允许调用方随意换：
   speedup = T_single(global_batch=B) / T_dual(global_batch=B)
   强扩展：全局 batch 不变，卡数增加。
   弱扩展：每卡 batch 不变，全局 batch 随卡数增加。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch


# ---------------------------------------------------------------- 合成数据

class SyntheticTokenDataset(torch.utils.data.Dataset):
    """固定 seed 的随机 token 数据集。性能基准不需要真实语料。"""

    def __init__(self, n_samples: int, seq_len: int, vocab_size: int, seed: int = 42):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randint(0, vocab_size, (n_samples, seq_len), generator=g)
        self.y = torch.randint(0, vocab_size, (n_samples, seq_len), generator=g)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# ---------------------------------------------------------------- 分段计时

class StepTimer:
    """单步分段计时。CUDA 可用时用 events 测 GPU 段，否则用 perf_counter。"""

    def __init__(self, device: str):
        self.device = device
        self.use_cuda = device.startswith("cuda")
        self._ev = {}
        self.record = {}

    def _mark(self, name: str):
        if self.use_cuda:
            self._ev[name] = torch.cuda.Event(enable_timing=True)
            self._ev[name].record()
        else:
            self._ev[name] = time.perf_counter()

    def _elapsed(self, a: str, b: str) -> float:
        if self.use_cuda:
            torch.cuda.synchronize()  # 确保两个 event 都已完成
            return self._ev[a].elapsed_time(self._ev[b])
        return (self._ev[b] - self._ev[a]) * 1000

    def start(self):
        self._mark("t0")

    def mark_data(self):
        self._mark("t_data")

    def mark_fwd(self):
        self._mark("t_fwd")

    def mark_bwd(self):
        self._mark("t_bwd")

    def mark_opt(self):
        self._mark("t_opt")

    def mark_eval(self):
        self._mark("t_eval")

    def finish(self) -> dict:
        self._mark("t1")
        r = {
            "data_ms": self._elapsed("t0", "t_data"),
            "fwd_ms": self._elapsed("t_data", "t_fwd"),
            "bwd_ms": self._elapsed("t_fwd", "t_bwd"),
            "opt_ms": self._elapsed("t_bwd", "t_opt"),
            "eval_ms": self._elapsed("t_opt", "t_eval"),
            "total_ms": self._elapsed("t0", "t1"),
        }
        self.record = r
        return r


# ---------------------------------------------------------------- 口径定义

SPEEDUP_DEFINITION = (
    "speedup = T_single(global_batch=B) / T_dual(global_batch=B)，"
    "强扩展下 B 不变。所有口径变体都在 caliber_audit.py 里显式列出。"
)


def comm_bytes_estimate(params_bytes: int, world: int) -> int:
    """13 篇口径：Ring AllReduce 每 rank 发送 2(w-1)/w 份张量。"""
    return int(params_bytes * 2 * (world - 1) / world)


def comm_ms_estimate(params_bytes: int, world: int, bandwidth_gbps: float) -> float:
    """通信时间估算：字节 ÷ 带宽。带宽取 13 篇实测（跨机 NCCL ~0.10 GB/s）。"""
    return comm_bytes_estimate(params_bytes, world) / (bandwidth_gbps * 1e9) * 1000


# ---------------------------------------------------------------- 报告

def write_report(path: str, payload: dict) -> None:
    if int(os.environ.get("RANK", 0)) != 0:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[rank0] written {p}")
