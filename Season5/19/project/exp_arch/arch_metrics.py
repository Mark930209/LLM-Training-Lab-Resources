"""arch_metrics.py —— 架构对照的三个静态口径：参数量、KV cache、每步 FLOPs。

等参数量对照的换算规则在这里单处定义（18 篇的教训：公式只写一处）。
全部是解析式，跑一次模型都不用；训练侧实测在 pipeline.py 里对照这些解析值。
"""

from __future__ import annotations

import math


def ffn_dim(ffn_type: str, hidden: int, expansion: float | None = None) -> int:
    """FFN 中间维（SwiGLU 对齐到 8 的倍数，与 model_arch.SwiGLU 同规则）。"""
    if expansion is None:
        expansion = 8 / 3 if ffn_type == "swiglu" else 4.0
    d = int(expansion * hidden)
    if ffn_type == "swiglu":
        d = (d + 7) // 8 * 8
    return d


def count_params(vocab: int, hidden: int, layers: int, heads: int, kv_heads: int,
                 seq_len: int, norm_type: str, ffn_type: str, pos_enc: str,
                 ffn_expansion: float | None = None, tie: bool = True) -> dict:
    """按组件配置解析参数量，返回逐项分解（与 model_arch 实建模型对账）。

    attention : q(hidden²) + k,v(hidden×kv_dim) + proj(hidden²)
    ffn       : swiglu 3×hidden×d（无 bias）；gelu 2×hidden×d + d + hidden（有 bias）
    norm      : rms 每层 2×hidden + 最终 hidden；layer 翻倍（weight+bias）
    embedding : vocab×hidden（tie 时 lm_head 不另计）+ learned 位置 seq_len×hidden
    """
    kv_dim = kv_heads * (hidden // heads)
    attn = hidden * hidden + 2 * hidden * kv_dim + hidden * hidden
    d = ffn_dim(ffn_type, hidden, ffn_expansion)
    if ffn_type == "swiglu":
        ffn = 3 * hidden * d
    else:
        ffn = 2 * hidden * d + d + hidden          # GELU FFN 带 bias
    norm_each = hidden if norm_type == "rms" else 2 * hidden
    norms = layers * 2 * norm_each + norm_each     # 每 block 两个 + 最终一个
    emb = vocab * hidden + (seq_len * hidden if pos_enc == "learned" else 0)
    head = 0 if tie else vocab * hidden
    total = emb + layers * (attn + ffn) + norms + head
    return {
        "embedding": emb, "attention": layers * attn, "ffn": layers * ffn,
        "norms": norms, "lm_head": head, "total": total,
    }


def solve_hidden_for_params(target: int, vocab: int, layers: int, heads: int,
                            kv_heads: int, seq_len: int, norm_type: str,
                            ffn_type: str, pos_enc: str,
                            ffn_expansion: float | None = None) -> int:
    """等参数量对照：给定目标总参数，反解该组件配置下可用的最大 hidden。

    一维单调递增函数，直接二分（解析求根要解含 floor/对齐的方程，不值得）。
    hidden 只取 **2×heads 的倍数**：一要被 heads 整除（注意力分头），二要
    保证 head_dim 为偶数——RoPE 把 head_dim 按奇偶维配对旋转（x[..., 0::2]
    与 x[..., 1::2]），奇数 head_dim 会形状不匹配直接崩（实测踩坑：
    hidden=378 / heads=6 → head_dim=63，layernorm 格 forward 崩溃）。
    """
    step = 2 * heads
    lo, hi = step, 4096
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        mid = mid // step * step                    # 对齐到 2×heads 倍数
        if mid < step:
            lo = (lo + hi) // 2 + 1
            continue
        n = count_params(vocab, mid, layers, heads, kv_heads, seq_len,
                         norm_type, ffn_type, pos_enc, ffn_expansion)["total"]
        if n <= target:
            best = mid
            lo = mid + step
        else:
            hi = mid - step
    if best is None:
        raise ValueError(f"目标参数量 {target} 太小，hidden 无法取到 {step}")
    return best


def kv_cache_mib(layers: int, kv_heads: int, head_dim: int, seq_len: int,
                 batch: int = 1, dtype_bytes: int = 4) -> float:
    """推理期 KV cache 显存（MiB）：2(K,V) × layers × kv_heads × head_dim × seq × batch × dtype。

    GQA/MQA 的收益在这里显形：kv_heads 缩小几倍，KV cache 就缩小几倍。
    """
    n = 2 * layers * kv_heads * head_dim * seq_len * batch * dtype_bytes
    return round(n / 1024 / 1024, 2)


def step_flops(hidden: int, layers: int, vocab: int, seq_len: int, batch: int,
               heads: int, kv_heads: int, ffn_type: str,
               ffn_expansion: float | None = None) -> float:
    """每步训练 FLOPs 估算（等算力对照的口径）。

    两部分：
    1. 矩阵乘：C ≈ 6 × 参与矩阵乘的参数 × token 数（前向 2、反向 4 的惯例）。
       参与矩阵乘的参数 = 每层 attention(q/k/v/proj) + FFN，加 embedding/lm_head
       的 vocab×hidden（tie 只算一次矩阵乘）。norm 与逐元素运算不计（占比 <1%）。
    2. attention 分数：QK^T 与 AV 各 2×layers×seq²×hidden×batch（前向），
       训练计入反向共 ×3。
    """
    tokens = seq_len * batch
    head_dim = hidden // heads
    kv_dim = kv_heads * head_dim
    d = ffn_dim(ffn_type, hidden, ffn_expansion)
    attn_p = 2 * hidden * hidden + 2 * hidden * kv_dim
    ffn_p = 3 * hidden * d if ffn_type == "swiglu" else 2 * hidden * d
    params_mm = layers * (attn_p + ffn_p) + vocab * hidden
    flops_mm = 6 * params_mm * tokens
    flops_attn = 3 * 2 * 2 * layers * seq_len * seq_len * hidden * batch
    return flops_mm + flops_attn
