"""schedulers.py —— 学习率调度与显存估算：真正的训练程序需要的第一批工程件。

03 篇的玩具循环用常数学习率，玩具够用；放大到 100M 后它会出问题：
    - 训练初期梯度方向还不稳，大 lr 会把权重推飞（需要 warmup 慢慢升温）
    - 训练后期已经接近最优点，大 lr 会在附近震荡下不去（需要 cosine 衰减）
本模块把这两个需求实现成可配置的调度器，并给出"提前知道会不会 OOM"的估算器。
"""

from __future__ import annotations

import math


def cosine_with_warmup(step: int, total_steps: int, warmup_steps: int,
                       lr: float, min_lr_ratio: float = 0.1) -> float:
    """线性 warmup + 余弦退火。

    前 warmup_steps 步：lr 从 0 线性升到 lr（让梯度方向先稳定下来）。
    之后：按余弦曲线从 lr 降到 lr*min_lr_ratio（后期收敛到最优点附近）。

    返回该步应使用的学习率。
    """
    if step < warmup_steps:
        return lr * (step + 1) / max(warmup_steps, 1)
    if step >= total_steps:
        return lr * min_lr_ratio
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr * (min_lr_ratio + (1.0 - min_lr_ratio) * coeff)


def estimate_model_params(vocab_size: int, hidden: int, layers: int,
                          ffn_dim: int | None = None,
                          tie_weights: bool = True) -> int:
    """估算 Transformer 参数量。

    逐项拆开算，而不是"跑一遍再说"：
        attention  qkv + proj          4 * h^2
        FFN        SwiGLU 三矩阵        3 * h * ffn
        norm       2 * h 每组 * 2 组    （RMSNorm 只有 weight）
        embedding  vocab * h；tie 时 lm_head 复用
    """
    ffn_dim = ffn_dim or int(8 / 3 * hidden)
    per_layer_attn = 4 * hidden * hidden
    per_layer_ffn = 3 * hidden * ffn_dim
    per_layer_norm = 4 * hidden
    per_layer = per_layer_attn + per_layer_ffn + per_layer_norm
    emb = vocab_size * hidden * (1 if tie_weights else 2)
    return per_layer * layers + emb + hidden  # +hidden 是 final norm


def estimate_training_memory_mb(params: int, seq_len: int, batch_size: int,
                                hidden: int, layers: int, vocab_size: int,
                                dtype_bytes: int = 4,
                                amp: bool = True) -> dict:
    """估算训练显存（MB），逐项拆开，让"为什么 OOM"可以算出来。

    组成（AdamW + AMP 场景）：
        模型权重      params * 4（fp32 主权重；AMP 下 forward 用 fp16 副本）
        梯度          params * 4
        优化器状态    params * 8（AdamW 的 exp_avg 与 exp_avg_sq，各 fp32）
        激活          batch * seq * hidden * layers * k（k 约为 10~20，随实现浮动）
        其他          logits/临时缓冲，按激活的 20% 估

    这套估算的用途不是精确预测，而是在启动前判断"这个配置要不要 OOM"，
    以及 OOM 时知道该动哪个旋钮（batch / seq_len / amp）。
    """
    mb = 1024 * 1024
    weights = params * 4
    grads = params * 4
    opt_state = params * 8
    act_bytes = dtype_bytes // 2 if amp else dtype_bytes
    activations = batch_size * seq_len * hidden * layers * 16 * act_bytes
    peaks = int((weights + grads + opt_state + activations) * 0.2)
    total = weights + grads + opt_state + activations + peaks
    return {
        "weights_mb": round(weights / mb, 1),
        "grads_mb": round(grads / mb, 1),
        "optimizer_mb": round(opt_state / mb, 1),
        "activations_mb": round(activations / mb, 1),
        "overhead_mb": round(peaks / mb, 1),
        "total_mb": round(total / mb, 1),
    }