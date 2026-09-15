"""metrics —— 指标口径统一：loss / 困惑度 / 显存对账。

系列所有文章的指标都从这里取，保证跨篇可比。
显存对账函数用于鉴别"假性 OOM"：nvidia-smi 显示的占用包含 CUDA 上下文与缓存，
torch 实际分配的张量显存往往小得多，两个数字对不上是正常现象，不是泄漏。
"""

from __future__ import annotations

import math
from typing import Any

import torch


def perplexity(loss: float) -> float:
    """困惑度 = exp(交叉熵 loss)。loss 是自然对数口径。"""
    return math.exp(loss)


def memory_report(device: int = 0) -> dict[str, Any]:
    """显存三方对账：torch 分配 / torch 保留 / 驱动层实际可用。

    返回 MB 口径，用于文章中的显存对账表。
    """
    if not torch.cuda.is_available():
        return {"error": "CUDA 不可用"}
    free_b, total_b = torch.cuda.mem_get_info(device)
    return {
        "torch_allocated_mb": round(torch.cuda.memory_allocated(device) / 1e6, 1),
        "torch_reserved_mb": round(torch.cuda.memory_reserved(device) / 1e6, 1),
        "torch_peak_allocated_mb": round(torch.cuda.max_memory_allocated(device) / 1e6, 1),
        "driver_free_mb": round(free_b / 1e6, 1),
        "driver_total_mb": round(total_b / 1e6, 1),
    }


def param_count_mb(model: torch.nn.Module) -> dict[str, float]:
    """模型参数量与 fp32/fp16 权重体积（MB），显存估算的最小单元。"""
    n = sum(p.numel() for p in model.parameters())
    return {
        "params_million": round(n / 1e6, 2),
        "fp32_mb": round(n * 4 / 1e6, 1),
        "fp16_mb": round(n * 2 / 1e6, 1),
    }
