"""tiny_rope.py —— 28 篇 Long Context Lab 的小型 llama 风格模型（RoPE 可插拔）。

结构：token emb + N × (RMSNorm → MHA causal → RMSNorm → SwiGLU MLP) + lm_head。
RoPE 扩展方式（--rope）：
    none ：直接外推（训练位置外的角度继续按原频率走）
    pi   ：位置插值（position interpolation，位置除以缩放因子 s）
    ntk  ：NTK-aware（base 乘 s^(d/(d-2))，位置不动）
    yarn ：YaRN 类（分频段混合插值 + 同款 base 缩放，带 ramp 平滑）

维度与 27 篇一致（dim 256、6 层、16 头），RoPE 手术必须自持实现。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True)
                                       + self.eps)
        return (norm * self.weight.float()).type_as(x)


def rope_freqs(dim: int, base: float = 10000.0):
    """原始 RoPE 频率（每个头维度一对）。"""
    return 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))


class RoPE(nn.Module):
    """带扩展方式的 RoPE。s = 目标长度 / 训练长度（扩展倍数）。"""

    def __init__(self, head_dim: int, mode: str = "none", s: float = 1.0,
                 base: float = 10000.0, yarn_ramp: float = 0.25):
        super().__init__()
        self.mode = mode
        self.s = max(s, 1.0)
        self.head_dim = head_dim
        inv_freq = rope_freqs(head_dim, base)
        if mode == "ntk" and self.s > 1.0:
            # NTK-aware：高频少动、低频多动
            inv_freq = rope_freqs(head_dim, base * (self.s ** (head_dim / (head_dim - 2))))
        elif mode == "yarn" and self.s > 1.0:
            inv_freq = rope_freqs(head_dim, base * (self.s ** (head_dim / (head_dim - 2))))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor):
        """x: [B, H, S, D]；positions: [S]（long）。"""
        if self.mode == "pi" and self.s > 1.0:
            pos = positions.float() / self.s
        elif self.mode == "yarn" and self.s > 1.0:
            pos = positions.float() / self.s   # 低频段已并入 base 缩放
        else:
            pos = positions.float()
        ang = pos[:, None] * self.inv_freq[None, :]      # [S, D/2]
        cos, sin = ang.cos(), ang.sin()
        x1, x2 = x[..., 0::2], x[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x1 * sin + x2 * cos
        return out


class MHA(nn.Module):
    def __init__(self, dim: int, n_heads: int, rope: RoPE):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)
        self.rope = rope

    def forward(self, x, positions, attn_mask=None):
        B, S, _ = x.shape
        qkv = self.qkv(x).view(B, S, 3, self.n_heads, self.head_dim)
        q, k, v = (qkv[:, :, i].transpose(1, 2) for i in range(3))  # [B,H,S,D]
        q = self.rope(q, positions)
        k = self.rope(k, positions)
        if attn_mask is None:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(B, S, -1)
        return self.o(out)


class Block(nn.Module):
    def __init__(self, dim: int, n_heads: int, ffn: int, rope: RoPE):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = MHA(dim, n_heads, rope)
        self.norm2 = RMSNorm(dim)
        self.gate = nn.Linear(dim, ffn, bias=False)
        self.up = nn.Linear(dim, ffn, bias=False)
        self.down = nn.Linear(ffn, dim, bias=False)

    def forward(self, x, positions, attn_mask=None):
        x = x + self.attn(self.norm1(x), positions, attn_mask)
        h = self.norm2(x)
        x = x + self.down(F.silu(self.gate(h)) * self.up(h))
        return x


class TinyRoPE(nn.Module):
    def __init__(self, vocab: int = 512, dim: int = 256, n_layers: int = 6,
                 n_heads: int = 16, ffn: int = 512,
                 rope_mode: str = "none", rope_s: float = 1.0,
                 base: float = 10000.0):
        super().__init__()
        rope = RoPE(dim // n_heads, mode=rope_mode, s=rope_s, base=base)
        self.emb = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList(
            [Block(dim, n_heads, ffn, rope) for _ in range(n_layers)])
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)

    def forward(self, ids: torch.Tensor, attn_mask=None) -> torch.Tensor:
        B, S = ids.shape
        pos = torch.arange(S, device=ids.device)
        x = self.emb(ids)
        for blk in self.blocks:
            x = blk(x, pos, attn_mask)
        return self.head(self.norm(x))
