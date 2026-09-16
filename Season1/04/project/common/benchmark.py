"""benchmark —— 量化采集：step time / 显存峰值 / 吞吐。

统一接口 run_experiment(config) 返回标准化结果 dict，
后续所有专题的 benchmark 表都从这里自动生成，保证口径一致。

测量纪律：
    - 前 N 步是预热（CUDA 上下文初始化、cudnn autotune），必须丢弃
    - step time 用 torch.cuda.synchronize() 后再计时，否则测的是入队时间
    - 显存用 max_memory_allocated（张量实际占用），不是 nvidia-smi（含上下文开销）
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable

import torch


@dataclass
class BenchResult:
    """标准化 benchmark 结果，字段与系列公共实验接口一致。"""
    loss: float
    step_time_ms: float
    tokens_per_sec: float
    peak_memory_mb: float
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StepTimer:
    """逐步计时器：自动丢弃预热步，CUDA 下同步后计时。"""

    def __init__(self, warmup: int = 10, device: str = "cuda"):
        self.warmup = warmup
        self.device = device
        self._times: list[float] = []
        self._step = 0
        self._t0: float | None = None

    def start(self) -> None:
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        self._t0 = time.perf_counter()

    def stop(self) -> None:
        if self._t0 is None:
            return
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - self._t0) * 1000
        if self._step >= self.warmup:  # 丢弃预热步
            self._times.append(elapsed_ms)
        self._step += 1
        self._t0 = None

    @property
    def avg_ms(self) -> float:
        return sum(self._times) / len(self._times) if self._times else float("nan")


def measure_peak_memory_mb() -> float:
    """读取本次运行的显存峰值（MB）；CPU 下返回 0。"""
    if torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated() / 1e6, 1)
    return 0.0


def run_benchmark(train_step: Callable[[], float], steps: int, tokens_per_step: int,
                  device: str = "cuda", warmup: int = 10) -> BenchResult:
    """通用 benchmark 循环：train_step 返回本步 loss，其余由本函数采集。

    参数:
        train_step: 无参函数，执行一步训练并返回 loss（float）。
        steps: 总步数。
        tokens_per_step: 每步处理的 token 数（批大小 × 序列长度），用于算吞吐。
        device: "cuda" 或 "cpu"。
        warmup: 预热步数，这些步不计入平均。
    """
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    timer = StepTimer(warmup=warmup, device=device)
    last_loss = float("nan")
    for _ in range(steps):
        timer.start()
        last_loss = train_step()
        timer.stop()

    avg_ms = timer.avg_ms
    tps = round(tokens_per_step / (avg_ms / 1000)) if avg_ms == avg_ms else 0  # NaN 检查
    return BenchResult(
        loss=round(last_loss, 4),
        step_time_ms=round(avg_ms, 2),
        tokens_per_sec=tps,
        peak_memory_mb=measure_peak_memory_mb(),
    )
