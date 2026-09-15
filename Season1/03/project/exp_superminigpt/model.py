"""model.py —— SuperMiniGPT：每个组件都自己写，不用 nn.Transformer。

与 02 篇冒烟实验的区别：那里用 nn.TransformerEncoder 验证环境通不通，
这里逐个组件手写，每个组件都带消融开关（use_xxx=False 就去掉它），
让"这个组件解决什么问题"变成可以跑出来的事实，而不是背下来的结论。

组件清单（每行的注释说明它解决什么）：
    CausalSelfAttention  只看过去不看未来；没有因果掩码，预测就是在抄答案
    RoPE                 位置信息；没有它，"字序"对模型不可见
    RMSNorm              稳定训练的归一化；比 LayerNorm 少一组参数、少一次均值
    SwiGLU               前馈层；门控让"哪些信息值得传下去"变成可学习的
    Residual             深层网络的梯度通路；没有它三层以上基本训不动
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """多头自注意力 + 因果掩码 + RoPE。

    消融开关 use_mask=False 时去掉因果掩码：
    模型在第 t 个位置能"看到"第 t+1 个字符的标签，loss 异常低但生成乱码，
    这是本篇最重要的失败实验（信息泄露）。
    """

    def __init__(self, hidden: int, heads: int, use_mask: bool = True,
                 use_rope: bool = True, max_seq_len: int = 1024):
        super().__init__()
        assert hidden % heads == 0, "hidden 必须能被 heads 整除"
        self.heads = heads
        self.head_dim = hidden // heads
        self.qkv = nn.Linear(hidden, hidden * 3, bias=False)
        self.proj = nn.Linear(hidden, hidden, bias=False)
        self.use_mask = use_mask
        self.rope = RoPE(self.head_dim, max_seq_len=max_seq_len, use_rope=use_rope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # 拆头：(B, T, C) → (B, heads, T, head_dim)，每头独立算注意力
        q = q.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.heads, self.head_dim).transpose(1, 2)

        q, k = self.rope(q), self.rope(k)   # 位置信息旋转进 q/k
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if self.use_mask:
            # 因果掩码：位置 t 只允许看 0..t，未来位置设为 -inf
            mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
            att = att.masked_fill(mask, float("-inf"))
        att = F.softmax(att, dim=-1)

        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class RoPE(nn.Module):
    """旋转位置编码：把位置信息旋转进 q/k 向量。

    相对位置体现在旋转角差上：位置 m 和 n 的 q/k 内积只依赖 m-n。
    消融开关 use_rope=False 时不加任何位置编码，
    模型看到的每个位置都一样，"顺序"信息完全丢失。
    """

    def __init__(self, head_dim: int, max_seq_len: int = 1024, use_rope: bool = True):
        super().__init__()
        self.use_rope = use_rope
        if not use_rope:
            return
        # 频率按维度对数分布：低维转得快（管近处），高维转得慢（管远处）
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)                      # (max_seq_len, head_dim/2)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, heads, T, head_dim)，在第 2 维（序列维）上加旋转
        if not self.use_rope:
            return x
        T = x.shape[2]
        cos = self.cos[:T].unsqueeze(0).unsqueeze(0)          # (1,1,T,D/2)
        sin = self.sin[:T].unsqueeze(0).unsqueeze(0)
        x1, x2 = x[..., 0::2], x[..., 1::2]                   # 奇偶维拆开
        out = torch.empty_like(x)
        out[..., 0::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x1 * sin + x2 * cos
        return out


class RMSNorm(nn.Module):
    """RMSNorm：均方根归一化，不减均值、不加偏置。

    与 LayerNorm 的差别是少一次均值计算和一组偏置参数，效果接近，
    现代 LLM（Llama/Qwen 系）普遍用它。这里自己写一遍就明白它有多简单。
    """

    def __init__(self, hidden: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return self.weight * x * rms


class SwiGLU(nn.Module):
    """门控前馈层：FFN(x) = W2( SiLU(W1 x) ⊙ W3 x )。

    门控（⊙ 那一项）让网络可以学习"哪路信息放行、哪路抑制"，
    同参数量下比普通两层 MLP 收敛更快。隐藏维度惯例取 8/3*hidden
    并对齐到 8 的倍数（参数量与 4*hidden 的普通 FFN 大致持平）。
    """

    def __init__(self, hidden: int, expansion: float = 8 / 3):
        super().__init__()
        ffn_dim = int(expansion * hidden)
        ffn_dim = (ffn_dim + 7) // 8 * 8  # 对齐到 8 倍数
        self.w1 = nn.Linear(hidden, ffn_dim, bias=False)  # 门控支路
        self.w3 = nn.Linear(hidden, ffn_dim, bias=False)  # 值支路
        self.w2 = nn.Linear(ffn_dim, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    """一个 transformer block：attention → residual → FFN → residual。

    pre-norm 结构（先 norm 再进子层）：残差主干上是"干净的" x，
    深层训练更稳，现代实现全部采用。
    消融开关 use_residual=False 时去掉两条残差通路：
    梯度只能穿过 12 个线性层的链式乘积，深层训练基本失效——
    这是"残差连接为什么是 transformer 刚需"的可运行证明。
    """

    def __init__(self, hidden: int, heads: int, use_mask: bool = True,
                 use_rope: bool = True, use_residual: bool = True):
        super().__init__()
        self.use_residual = use_residual
        self.norm1 = RMSNorm(hidden)
        self.attn = CausalSelfAttention(hidden, heads, use_mask=use_mask, use_rope=use_rope)
        self.norm2 = RMSNorm(hidden)
        self.ffn = SwiGLU(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            x = x + self.attn(self.norm1(x))   # attention 分支带残差
            x = x + self.ffn(self.norm2(x))    # FFN 分支带残差
        else:
            x = self.attn(self.norm1(x))
            x = self.ffn(self.norm2(x))
        return x


class SuperMiniGPT(nn.Module):
    """SuperMiniGPT v0：token embedding → N × block → norm → lm_head。

    权重共享：embedding 与 lm_head 用同一矩阵（词表大时省掉一份
    vocab×hidden 的参数，小模型上对 loss 也有帮助）。
    """

    def __init__(self, vocab_size: int, hidden: int = 128, layers: int = 4,
                 heads: int = 4, seq_len: int = 128,
                 use_mask: bool = True, use_rope: bool = True, use_residual: bool = True):
        super().__init__()
        self.seq_len = seq_len
        self.tok_emb = nn.Embedding(vocab_size, hidden)
        self.blocks = nn.ModuleList(
            Block(hidden, heads, use_mask=use_mask, use_rope=use_rope,
                  use_residual=use_residual)
            for _ in range(layers)
        )
        self.norm_f = RMSNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)
        # 权重共享（tie weights）
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
        assert T <= self.seq_len, f"序列长度 {T} 超过训练时声明的 {self.seq_len}"
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        return self.lm_head(x)  # (B, T, vocab)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
        """自回归生成：每次只取最后一个位置的 logits，采一个字，拼回去再滚一遍。

        temperature 控制分布陡峭程度（<1 保守，>1 发散）；
        top_k 只在概率最高的 k 个字里采样，压制乱码。
        """
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
