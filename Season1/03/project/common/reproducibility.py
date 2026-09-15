"""reproducibility —— seed 管理：可复现的第一件事。

同 seed 两次运行结果逐位一致，是环境可复现性的最小充分验证。
随机源有三处：Python random、NumPy、PyTorch（CPU 与 CUDA 各自独立）。
只设 torch.manual_seed 而漏掉 CUDA 或 DataLoader worker，是"复现失败"的最常见根因。
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_all_seeds(seed: int, deterministic: bool = False) -> None:
    """一次性固定全部随机源。

    参数:
        seed: 随机种子。
        deterministic: 是否开启 cuDNN 确定性模式。开启后同 seed 可逐位复现，
            但部分算子会报错或变慢；冒烟/对照实验建议开启，性能 benchmark 建议关闭。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # 同时作用于 CPU 与 CUDA
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """DataLoader 的 worker_init_fn：多进程数据加载的随机源也要固定。

    用法: DataLoader(..., worker_init_fn=seed_worker, generator=g)
    其中 g = torch.Generator(); g.manual_seed(seed)
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
