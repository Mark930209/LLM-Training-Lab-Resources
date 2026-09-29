"""stability_metrics.py —— 21 篇数值稳定性的纯函数：精度范围、spike 检测、
告警提前量、退化判据。全部可离线单测，不依赖 GPU。

数值范围是解析式（IEEE 754 定义），spike/告警/退化判据是算法，训练侧实测在
pipeline.py 里对照这些函数。
"""
from __future__ import annotations

import math

# ---- IEEE 754 各精度的表示范围（解析常量，REFERENCE：IEEE 754-2019）----
# (最小正规数, 最大有限值, 尾数位含隐含位)
FLOAT_RANGE = {
    "fp32": (2**-126, (2 - 2**-23) * 2**127, 24),
    "fp16": (2**-14, (2 - 2**-10) * 2**15, 11),      # 最大有限值 = 65504
    "bf16": (2**-126, (2 - 2**-7) * 2**127, 8),      # 指数位同 fp32，范围一致
    "fp8_e4m3": (2**-6, 448.0, 4),      # E4M3：4 位指数 3 位尾数（无 inf 编码，最大 448）
    "fp8_e5m2": (2**-14, 57344.0, 3),   # E5M2：5 位指数 2 位尾数（最大 57344）
}


def max_finite(dtype: str) -> float:
    return FLOAT_RANGE[dtype][1]


def min_normal(dtype: str) -> float:
    return FLOAT_RANGE[dtype][0]


def overflow_of(dtype: str, value: float) -> bool:
    """value 在该精度下是否上溢（超过最大有限值 → inf）。"""
    return abs(value) > max_finite(dtype)


def underflow_of(dtype: str, value: float) -> bool:
    """value 在该精度下是否下溢到零（次正规区以下）。"""
    return 0 < abs(value) < min_normal(dtype) / 2   # 粗略：低于最小正规数一半即视作危险


def mantissa_bits(dtype: str) -> int:
    return FLOAT_RANGE[dtype][2]


def precision_epsilon(dtype: str) -> float:
    """该精度的机器 epsilon（2^-(尾数位-1)）。"""
    return 2.0 ** (-(mantissa_bits(dtype) - 1))


# ---- spike 检测 ----
def detect_spike(loss: float, history: list[float], k: float = 3.0) -> bool:
    """当前 loss 是否构成 spike：超过历史窗口中位数的 k 倍，或 NaN/inf。"""
    if loss is None or math.isnan(loss) or math.isinf(loss):
        return True
    if not history:
        return False
    med = median(history)
    return loss > med * k


def median(vals: list[float]) -> float:
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


def alarm_threshold(grad_norms: list[float], k: float = 5.0) -> float:
    """告警阈值 = 基线窗口 grad_norm 中位数 × k。"""
    return median(grad_norms) * k


def first_alarm_step(grad_norms: list[float], thresh: float, skip: int = 0,
                     sustain: int = 1) -> int | None:
    """grad_norm 首次"持续"超阈值的步号（前置指标报警点）。

    skip：跳过前 N 步（基线/初始化窗口）。训练初期 grad_norm 有初始化瞬态
    （实测前 5 步 7.6/6.2/5.4/5.9/3.2，远高于稳态 ~0.58），不跳过会把瞬态
    误报为告警（alarm_step=0）。基线窗口本身用于算阈值，不用于报警。

    sustain：要求从该步起连续 sustain 步都超阈值才算"持续告警"，过滤零星
    误报。实测 ramp 场景里 warmup 尾部有单步瞬时超阈（step 52/56）随即回落，
    若用"首次超阈"会把它当告警、算出虚高的提前量（lead=181）；真正的发散前兆
    是 grad_norm 持续爬升（bf16 实测 step 216 起 9.6→24.7→13.9→139），
    sustain=3 能锁定真前兆、排除零星误报。
    """
    n = len(grad_norms)
    for i in range(skip, n):
        g = grad_norms[i]
        if g is None or not (g > thresh):
            continue
        # 检查从 i 起连续 sustain 步是否都超阈
        ok = True
        for j in range(i, min(i + sustain, n)):
            gj = grad_norms[j]
            if gj is None or not (gj > thresh):
                ok = False
                break
        if ok:
            return i
    return None


def first_spike_step(losses: list[float], k: float = 3.0, window: int = 50) -> int | None:
    """loss 首次构成 spike 的步号（实际发散点）。"""
    for i, l in enumerate(losses):
        hist = losses[max(0, i - window):i]
        if detect_spike(l, hist, k):
            return i
    return None


def lead_steps(alarm_step: int | None, spike_step: int | None) -> int | None:
    """告警提前量 = spike 步 − 告警步（正数表示前置指标提前预警）。"""
    if alarm_step is None or spike_step is None:
        return None
    return spike_step - alarm_step


# ---- 静默退化判据 ----
def is_silent_degradation(train_losses: list[float], eval_ppls: list[float],
                          split_frac: float = 0.4) -> bool:
    """静默退化：切换点之后 train loss 持续下降，但 eval ppl 持续上升。

    split_frac 之前是正常训练，之后进入记忆化。判据：后半段 train loss
    末值 < 首值（在降），同时 eval ppl 末值 > 首值（在升）。
    """
    n = len(train_losses)
    split = int(n * split_frac)
    if split >= n - 1:
        return False
    tail_train = train_losses[split:]
    tail_eval = eval_ppls[split:] if len(eval_ppls) > split else eval_ppls
    if not tail_train or not tail_eval:
        return False
    train_falling = tail_train[-1] < tail_train[0]
    eval_rising = tail_eval[-1] > tail_eval[0]
    return train_falling and eval_rising


def degradation_gap_pct(eval_ppls: list[float], split_frac: float = 0.4) -> float:
    """退化幅度：后半段 eval ppl 从首值到末值上升的百分比。"""
    n = len(eval_ppls)
    split = int(n * split_frac)
    tail = eval_ppls[split:] if split < n else eval_ppls
    if len(tail) < 2 or tail[0] == 0:
        return 0.0
    return (tail[-1] - tail[0]) / tail[0] * 100


# ---- 处置策略的步数损失 ----
def steps_lost(strategy: str, spike_steps: list[int], total_steps: int,
               ckpt_every: int = 40) -> int:
    """各处置策略因 spike 损失的有效步数（粗略口径，供对照）。

    skip_bad          : 每个 spike 丢 1 步
    lower_lr          : 不丢步，但 LR 降低后续收敛变慢（这里记 0 丢步）
    rollback_rewarmup : 回退到最近 checkpoint，丢 (spike_step % ckpt_every) + re-warmup
    tighten_clip      : 不丢步（clip 更紧只影响更新幅度）
    """
    if strategy == "skip_bad":
        return len(spike_steps)
    if strategy == "rollback_rewarmup":
        lost = 0
        for s in spike_steps:
            lost += s % ckpt_every          # 回退到最近 ckpt 丢的步
        return lost
    return 0
