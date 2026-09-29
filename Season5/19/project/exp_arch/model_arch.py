"""model_arch.py —— 组件化 Transformer：每个现代架构组件都可独立开关。

19 篇的核心工具。04 篇的 SuperMiniGPT 已内置 RMSNorm/SwiGLU/RoPE（带开关），
但只覆盖"有/无"，不覆盖"换成另一种实现"。本篇要对照的是现代组件各自替换掉
了什么经典组件，所以每个轴都要有"经典档"与"现代档"两个可选实现：

  norm_type   : "rms"（现代，RMSNorm）      | "layer"（经典，LayerNorm 带 bias）
  norm_pos    : "pre"（现代，先 norm 再子层）| "post"（经典 GPT-2，子层后 norm）
  ffn_type    : "swiglu"（现代，8/3 扩展 3 矩阵）| "gelu"（经典，4x 扩展 2 矩阵）
  pos_enc     : "rope"（现代，旋转进 q/k）   | "learned"（经典，绝对位置 embedding）
  attn_type   : "mha"（kv_heads=heads）| "gqa"（kv_heads=heads/g）| "mqa"（kv_heads=1）

参数量随组件变化（LayerNorm 多 bias、GELU FFN 2 矩阵 vs SwiGLU 3 矩阵、
learned 位置编码多 seq_len×hidden），这正是"组件收益里混着工程量重新分配"
的物质基础。等参数量对照靠调 hidden 把参数量拉回基准，见 arch_metrics.py。

所有实现自己写，不用 nn.TransformerEncoder，保证每个组件的行为可追溯。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- 归一化

class RMSNorm(nn.Module):
    """RMSNorm：均方根归一化，不减均值、无 bias，只有 hidden 个 weight 参数。"""

    def __init__(self, hidden: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return self.weight * x * rms


def build_norm(norm_type: str, hidden: int) -> nn.Module:
    """按 norm_type 建归一化层。layer 用 nn.LayerNorm（带 bias，2×hidden 参数）。"""
    if norm_type == "rms":
        return RMSNorm(hidden)
    if norm_type == "layer":
        return nn.LayerNorm(hidden)
    raise ValueError(f"未知 norm_type: {norm_type}")


# ---------------------------------------------------------------- 位置编码

class RoPE(nn.Module):
    """旋转位置编码：把位置旋转进 q/k，相对位置体现在旋转角差上。无参数。"""

    def __init__(self, head_dim: int, max_seq_len: int = 1024, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, heads, T, head_dim)
        T = x.shape[2]
        cos = self.cos[:T].unsqueeze(0).unsqueeze(0)
        sin = self.sin[:T].unsqueeze(0).unsqueeze(0)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x1 * sin + x2 * cos
        return out


# ---------------------------------------------------------------- 注意力（MHA/GQA/MQA）

class Attention(nn.Module):
    """多头因果自注意力，支持 MHA / GQA / MQA（由 kv_heads 控制）。

    kv_heads == heads 是 MHA；kv_heads == 1 是 MQA；中间是 GQA。
    q 投影恒为 hidden×hidden；k/v 投影为 hidden×(kv_heads×head_dim)，
    GQA/MQA 下 k/v 参数与 KV cache 都按 kv_heads/heads 比例缩小。
    """

    def __init__(self, hidden: int, heads: int, kv_heads: int,
                 pos_enc: str = "rope", max_seq_len: int = 1024,
                 rope_base: float = 10000.0):
        super().__init__()
        assert hidden % heads == 0, "hidden 必须能被 heads 整除"
        assert heads % kv_heads == 0, "heads 必须能被 kv_heads 整除（GQA 分组）"
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = hidden // heads
        if pos_enc == "rope":
            # RoPE 按奇偶维配对旋转，head_dim 必须是偶数（19 篇实测踩坑：
            # hidden=378/heads=6 → head_dim=63，forward 形状不匹配崩溃）
            assert self.head_dim % 2 == 0, \
                f"RoPE 要求 head_dim 为偶数，实得 {self.head_dim}（hidden={hidden}, heads={heads}）"
        self.groups = heads // kv_heads          # 每个 kv 头被几个 q 头共享
        self.kv_dim = kv_heads * self.head_dim

        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(hidden, self.kv_dim, bias=False)
        self.proj = nn.Linear(hidden, hidden, bias=False)

        self.pos_enc = pos_enc
        if pos_enc == "rope":
            self.rope = RoPE(self.head_dim, max_seq_len=max_seq_len, base=rope_base)
        elif pos_enc == "learned":
            pass                                  # learned 位置 embedding 在模型顶层加
        else:
            raise ValueError(f"未知 pos_enc: {pos_enc}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.kv_heads, self.head_dim).transpose(1, 2)

        if self.pos_enc == "rope":
            q, k = self.rope(q), self.rope(k)

        # GQA/MQA：把 kv 头复制到 q 头数（expand 不占额外显存，repeat 才占）
        if self.groups > 1:
            k = k.repeat_interleave(self.groups, dim=1)
            v = v.repeat_interleave(self.groups, dim=1)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


# ---------------------------------------------------------------- 前馈（SwiGLU / GELU）

class SwiGLU(nn.Module):
    """门控前馈：W2( SiLU(W1 x) ⊙ W3 x )，3 个矩阵，扩展比默认 8/3。"""

    def __init__(self, hidden: int, expansion: float = 8 / 3):
        super().__init__()
        ffn_dim = int(expansion * hidden)
        ffn_dim = (ffn_dim + 7) // 8 * 8
        self.w1 = nn.Linear(hidden, ffn_dim, bias=False)
        self.w3 = nn.Linear(hidden, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class GELUFFN(nn.Module):
    """经典两层前馈：W2( GELU(W1 x) )，2 个矩阵，扩展比默认 4。"""

    def __init__(self, hidden: int, expansion: float = 4.0):
        super().__init__()
        ffn_dim = int(expansion * hidden)
        self.w1 = nn.Linear(hidden, ffn_dim, bias=True)
        self.w2 = nn.Linear(ffn_dim, hidden, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.gelu(self.w1(x)))


def build_ffn(ffn_type: str, hidden: int, expansion: float | None = None) -> nn.Module:
    if ffn_type == "swiglu":
        return SwiGLU(hidden, expansion if expansion is not None else 8 / 3)
    if ffn_type == "gelu":
        return GELUFFN(hidden, expansion if expansion is not None else 4.0)
    raise ValueError(f"未知 ffn_type: {ffn_type}")


# ---------------------------------------------------------------- Block（pre/post-norm）

class Block(nn.Module):
    """一个 transformer block，支持 pre-norm 与 post-norm 两种残差结构。

    pre-norm（现代）：x = x + attn(norm1(x)); x = x + ffn(norm2(x))
        残差主干是干净的 x，深层稳定，几乎不需 warmup。
    post-norm（经典 GPT-2/原始 Transformer）：x = norm1(x + attn(x)); x = norm2(x + ffn(x))
        残差主干上叠了 norm，深层梯度尺度累积，需要 warmup 与更小心的初始化。
    """

    def __init__(self, hidden: int, heads: int, kv_heads: int,
                 norm_type: str = "rms", norm_pos: str = "pre",
                 ffn_type: str = "swiglu", pos_enc: str = "rope",
                 max_seq_len: int = 1024, rope_base: float = 10000.0,
                 ffn_expansion: float | None = None):
        super().__init__()
        self.norm_pos = norm_pos
        self.norm1 = build_norm(norm_type, hidden)
        self.norm2 = build_norm(norm_type, hidden)
        self.attn = Attention(hidden, heads, kv_heads, pos_enc=pos_enc,
                              max_seq_len=max_seq_len, rope_base=rope_base)
        self.ffn = build_ffn(ffn_type, hidden, ffn_expansion)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_pos == "pre":
            x = x + self.attn(self.norm1(x))
            x = x + self.ffn(self.norm2(x))
        else:  # post
            x = self.norm1(x + self.attn(x))
            x = self.norm2(x + self.ffn(x))
        return x


# ---------------------------------------------------------------- 顶层模型

class ArchGPT(nn.Module):
    """组件化 GPT：token embedding (+ learned 位置) → N×Block → norm → lm_head。

    权重共享：lm_head 与 tok_emb 同一矩阵（与 04/17/18 篇同口径）。
    pos_enc="learned" 时额外有一个 pos_emb（seq_len×hidden，不共享）。
    """

    def __init__(self, vocab_size: int, hidden: int = 384, layers: int = 6,
                 heads: int = 6, kv_heads: int | None = None, seq_len: int = 256,
                 norm_type: str = "rms", norm_pos: str = "pre",
                 ffn_type: str = "swiglu", pos_enc: str = "rope",
                 rope_base: float = 10000.0, ffn_expansion: float | None = None,
                 tie_weights: bool = True):
        super().__init__()
        if kv_heads is None:
            kv_heads = heads
        self.seq_len = seq_len
        self.pos_enc = pos_enc
        self.vocab_size = vocab_size
        self.hidden = hidden

        self.tok_emb = nn.Embedding(vocab_size, hidden)
        if pos_enc == "learned":
            self.pos_emb = nn.Embedding(seq_len, hidden)
        else:
            self.pos_emb = None

        self.blocks = nn.ModuleList(
            Block(hidden, heads, kv_heads, norm_type=norm_type, norm_pos=norm_pos,
                  ffn_type=ffn_type, pos_enc=pos_enc, max_seq_len=seq_len,
                  rope_base=rope_base, ffn_expansion=ffn_expansion)
            for _ in range(layers)
        )
        self.norm_f = build_norm(norm_type, hidden)
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)
        if tie_weights:
            self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        assert T <= self.seq_len, f"序列长度 {T} 超过声明的 {self.seq_len}"
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            pos = torch.arange(T, device=idx.device).unsqueeze(0)
            x = x + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        return self.lm_head(x)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.seq_len:]
            logits = self(ctx)[:, -1] / temperature
            if top_k is not None:
                kth = torch.topk(logits, top_k).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


def kv_heads_for(attn_type: str, heads: int) -> int:
    """把 attn_type 字符串映射到 kv_heads 数。gqa 取 heads 的一半（至少 1）。"""
    if attn_type == "mha":
        return heads
    if attn_type == "mqa":
        return 1
    if attn_type == "gqa":
        return max(1, heads // 2)
    raise ValueError(f"未知 attn_type: {attn_type}")
