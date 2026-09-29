"""20 篇超参实验的纯函数：规模反解、吞吐缩放、步数与时间估算、缩放关系。

全部可离线单测。时间估算用 19 篇实测 tok/s 作基准，按 hidden^2 反比缩放
（吞吐主要受每 token 计算量支配，计算量 ~ hidden^2）。估算值标 ESTIMATE，
真实时长以运行日志为准。
"""
from __future__ import annotations

import math

SEQ_LEN = 256


def n_params_dense(hidden: int, n_layer: int, vocab: int, ffn_mult: float = 2.67,
                   tie: bool = True, ffn_type: str = "swiglu") -> int:
    """稠密 GPT 参数量，与 19 篇 exp_arch.count_params 完全同口径。

    SwiGLU 的 ffn_dim 取 int(8*hidden/3)（与 model_arch 层构造一致），
    其余激活取 int(ffn_mult*hidden)。tie=True 时 embedding 只算一份
    （wte 与 lm_head 共享权重）。19 篇 modern（hidden384/6层/vocab6120/tie）
    实测 12,971,904，本函数须逐位复现。
    """
    attn = 4 * hidden * hidden
    if ffn_type == "swiglu":
        ffn_dim = (int(8 * hidden / 3) + 7) // 8 * 8   # 对齐到 8 的倍数，与 model_arch 同规则
        ffn = 3 * hidden * ffn_dim
    else:
        ffn_dim = int(ffn_mult * hidden)
        ffn = 2 * hidden * ffn_dim + ffn_dim + hidden   # GELU FFN 带 bias
    norm = 2 * hidden                 # 每 block 两个 RMSNorm
    emb = vocab * hidden
    if not tie:
        emb += vocab * hidden
    return n_layer * (attn + ffn + norm) + emb + hidden


def tok_per_sec(hidden: int, ref_hidden: int, ref_tps: float) -> float:
    """按 hidden^2 反比缩放吞吐（ESTIMATE）。"""
    return ref_tps * (ref_hidden / hidden) ** 2


def train_seconds(token_budget: int, hidden: int, ref_hidden: int, ref_tps: float) -> float:
    """单次训练估算秒数（ESTIMATE）。"""
    return token_budget / tok_per_sec(hidden, ref_hidden, ref_tps)


def total_steps(token_budget: int, batch_size: int) -> int:
    return max(1, round(token_budget / (batch_size * SEQ_LEN)))


# ---- 缩放关系（正文要检验的两条）----
def lr_sqrt_scale(lr_base: float, batch_base: int, batch_new: int) -> float:
    """平方根缩放：LR ∝ sqrt(batch)。"""
    return lr_base * math.sqrt(batch_new / batch_base)


def lr_linear_scale(lr_base: float, batch_base: int, batch_new: int) -> float:
    """线性缩放：LR ∝ batch。"""
    return lr_base * (batch_new / batch_base)


# ---- Chinchilla 算力预算 C ≈ 6 N D ----
def flops_6nd(n_params: int, n_tokens: int) -> float:
    return 6.0 * n_params * n_tokens


def tokens_for_flops(flops_budget: float, n_params: int) -> float:
    """给定算力与参数量，反解训练 token 数 D = C / (6N)。"""
    return flops_budget / (6.0 * n_params)


def chinchilla_optimal_params(flops_budget: float) -> float:
    """Chinchilla 最优点 N ∝ C^0.5，系数取 N≈sqrt(C)/6^0.5/ 经验比例。

    Chinchilla: N_opt ≈ (C / (6 * 20))^{0.5}（D=20N 时）。返回参数量。
    """
    return math.sqrt(flops_budget / (6.0 * 20.0))


def fit_power_law(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """对 y = a * x^b 取对数做最小二乘，返回 (a, b)。用于 scaling law 拟合。"""
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    n = len(lx)
    mx = sum(lx) / n
    my = sum(ly) / n
    num = sum((lx[i] - mx) * (ly[i] - my) for i in range(n))
    den = sum((lx[i] - mx) ** 2 for i in range(n))
    b = num / den
    a = math.exp(my - b * mx)
    return a, b


def predict_loss(a: float, b: float, n_params: int) -> float:
    return a * (n_params ** b)


def rel_error(pred: float, actual: float) -> float:
    return abs(pred - actual) / abs(actual) * 100


def non_emb_params(hidden: int, n_layer: int) -> int:
    """非 embedding 参数量（Chinchilla 口径，18 篇 _non_emb_params 同款）。

    SwiGLU 8/3x 下每层 attention 4H² + FFN 3×H×(8H/3)=8H² = 12H²，
    norm 每层 2H + 最终 H。tied embedding（vocab×H）不计入——
    小模型里词表参数会占主导，用总参数会把"模型规模"污染成"词表规模"。
    """
    return 12 * hidden * hidden * n_layer + hidden * (2 * n_layer + 1)


def lr_at(step: int, total: int, lr: float, warmup_frac: float,
          decay: str = "cosine") -> float:
    """LR 调度：线性 warmup + 三种衰减（17 篇 _lr_at 的 cosine 同款扩展）。

    cosine  : lr × 0.5 × (1 + cos(π·prog))   —— 17/19 篇基准
    linear  : lr × (1 − prog)
    constant: lr（warmup 后恒定）
    """
    warmup = max(1, int(total * warmup_frac))
    if step < warmup:
        return lr * step / warmup
    prog = (step - warmup) / max(1, total - warmup)
    if decay == "cosine":
        return lr * 0.5 * (1 + math.cos(math.pi * prog))
    if decay == "linear":
        return lr * (1 - prog)
    if decay == "constant":
        return lr
    raise ValueError(f"unknown decay: {decay}")


def lr_range_at(step: int, total: int, lr_min: float, lr_max: float) -> float:
    """LR range test 的指数上升调度：lr_min × (lr_max/lr_min)^(step/(total-1))。"""
    if total <= 1:
        return lr_min
    return lr_min * (lr_max / lr_min) ** (step / (total - 1))
