"""naive_attn.py —— 显式物化 N×N 分数矩阵的 attention 参考实现（09 篇基线）。

存在的意义不是"能用"，而是把 SDPA 内部被融掉的东西摊开给读者看：
分数矩阵 QK^T 有多大、softmax 要再存一份、causal mask 怎么加。
07 篇的显存账里 activation 是一个系数，这里能看到那个系数的一部分来历。

与 torch.nn.functional.scaled_dot_product_attention 数学等价，
correctness 模式会逐元素比对输出与梯度。
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def naive_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = True,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """物化 N×N 分数矩阵的 attention。

    输入形状 (B, H, N, D)，与 SDPA 一致。返回 (B, H, N, D)。

    每一步都显式落一个张量，这正是 FlashAttention 要消掉的东西：
      scores  (B, H, N, N)  —— 第一份 N×N
      masked  (B, H, N, N)  —— 加 mask 时再一份（这里用 in-place 省掉）
      probs   (B, H, N, N)  —— softmax 输出，第二份 N×N
    """
    b, h, n, d = q.shape
    scale = 1.0 / math.sqrt(d)

    # 第一份 N×N：分数矩阵
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    if is_causal:
        # in-place 写 mask，不再分配第二份 N×N；
        # 真实框架里 mask 常是独立张量，那样峰值还要再加一份
        mask = torch.full((n, n), float("-inf"), device=q.device, dtype=q.dtype)
        mask = torch.triu(mask, diagonal=1)
        scores = scores + mask

    # 第二份 N×N：softmax 输出
    probs = F.softmax(scores, dim=-1)

    if dropout_p > 0.0:
        probs = F.dropout(probs, p=dropout_p)

    return torch.matmul(probs, v)


def score_matrix_bytes(b: int, h: int, n: int, dtype: torch.dtype) -> float:
    """一份 (B,H,N,N) 分数矩阵占多少 MB。

    naive 至少要两份（scores + probs），这个函数给单份，
    文章里用它解释"为什么 N 翻倍、显存翻四倍"。
    """
    itemsize = torch.finfo(dtype).bits // 8 if dtype.is_floating_point else dtype.itemsize
    return b * h * n * n * itemsize / 1024 / 1024
