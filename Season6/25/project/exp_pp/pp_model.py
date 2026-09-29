"""pp_model.py —— 25 篇的实验模型：块间 FFN 倍率不等的 Transformer 栈。

设计要点：8 个 block 的 FFN 倍率取 [4,2,3,4,2,3,4,2]，逐块耗时天然不等，
"层数均衡 ≠ 时间均衡"的根源就在这里。权重在 rank0 初始化后广播，
保证两机权重逐位一致（24 篇 RNG 教训）。
"""

from __future__ import annotations

import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

FF_MULTS = [4, 2, 3, 4, 2, 3, 4, 2]


class Block(nn.Module):
    def __init__(self, hidden: int, heads: int, ff_mult: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden)
        self.q = nn.Linear(hidden, hidden)
        self.k = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, hidden)
        self.proj = nn.Linear(hidden, hidden)
        self.ln2 = nn.LayerNorm(hidden)
        self.fc1 = nn.Linear(hidden, hidden * ff_mult)
        self.fc2 = nn.Linear(hidden * ff_mult, hidden)
        self.heads = heads
        self.dh = hidden // heads
        self.scale = 1.0 / math.sqrt(self.dh)

    def forward(self, x):
        b, s, h = x.shape
        t = self.ln1(x)
        q = self.q(t).view(b, s, self.heads, self.dh).transpose(1, 2)
        k = self.k(t).view(b, s, self.heads, self.dh).transpose(1, 2)
        v = self.v(t).view(b, s, self.heads, self.dh).transpose(1, 2)
        att = torch.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        o = (att @ v).transpose(1, 2).reshape(b, s, h)
        x = x + self.proj(o)
        t = self.ln2(x)
        return x + self.fc2(F.gelu(self.fc1(t)))


def build_stack(hidden: int, heads: int, ff_mults=FF_MULTS) -> nn.ModuleList:
    torch.manual_seed(20260930)
    return nn.ModuleList([Block(hidden, heads, m) for m in ff_mults])


def broadcast_params(stack: nn.ModuleList) -> None:
    """rank0 初始化、广播到所有 rank：两机 torch 版本不同，同 seed 不同随机流。"""
    if not dist.is_initialized():
        return
    for p in stack.parameters():
        dist.broadcast(p.data, src=0)


def stack_forward(stack: nn.ModuleList, x: torch.Tensor,
                  blocks: range | list[int] | None = None) -> torch.Tensor:
    for i, blk in enumerate(stack):
        if blocks is not None and i not in blocks:
            continue
        x = blk(x)
    return x


def count_params(stack: nn.ModuleList) -> int:
    return sum(p.numel() for p in stack.parameters())
