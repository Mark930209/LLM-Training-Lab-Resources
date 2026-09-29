"""hparam_bridge.py —— 复用 20 篇 exp_hparam 的 LR 调度，避免重复实现。

21 篇的 PYTHONPATH 含 20 篇工程目录（数据底座、架构、调度全部同源），
这里只做一层稳定转发：20 篇的 lr_at（cosine/linear/constant 三衰减）。
"""
from __future__ import annotations

from exp_hparam.hparam_metrics import lr_at, lr_range_at  # noqa: F401  (20 篇)

__all__ = ["lr_at", "lr_range_at"]
