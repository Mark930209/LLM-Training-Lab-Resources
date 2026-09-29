"""exp_stability 纯函数单测（CPU 即可，不依赖 GPU）。"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stability_metrics import (  # noqa: E402
    FLOAT_RANGE, max_finite, min_normal, overflow_of, underflow_of,
    mantissa_bits, precision_epsilon, detect_spike, median,
    alarm_threshold, first_alarm_step, first_spike_step, lead_steps,
    is_silent_degradation, degradation_gap_pct, steps_lost,
)


def test_float_ranges():
    # fp16 最大有限值 65504（IEEE 754 half）
    assert abs(max_finite("fp16") - 65504) < 1e-6
    # bf16 与 fp32 指数位相同 → 最小正规数一致、最大有限值同量级（尾数位不同故略小）
    assert min_normal("bf16") == min_normal("fp32")
    assert max_finite("bf16") < max_finite("fp32")          # 尾数 7 位 < 23 位
    assert max_finite("bf16") / max_finite("fp32") > 0.99   # 但同量级（都 ~3.4e38）
    # bf16 尾数 8 位（含隐含位）< fp16 的 11 位：范围大但精度低
    assert mantissa_bits("bf16") < mantissa_bits("fp16")
    # fp8 E4M3 最大 448
    assert abs(max_finite("fp8_e4m3") - 448.0) < 1e-6


def test_overflow_underflow():
    assert overflow_of("fp16", 70000.0)          # > 65504 → inf
    assert not overflow_of("fp16", 60000.0)
    assert not overflow_of("bf16", 70000.0)      # bf16 装得下
    assert underflow_of("fp16", 1e-8)            # < 6.1e-5 → 0
    assert not underflow_of("bf16", 1e-8)        # bf16 范围同 fp32
    assert not underflow_of("fp16", 0.001)


def test_precision_epsilon():
    assert abs(precision_epsilon("fp32") - 2**-23) < 1e-15
    assert abs(precision_epsilon("fp16") - 2**-10) < 1e-12
    assert abs(precision_epsilon("bf16") - 2**-7) < 1e-12
    # bf16 精度比 fp16 粗 8 倍
    assert precision_epsilon("bf16") / precision_epsilon("fp16") == 8


def test_median():
    assert median([3, 1, 2]) == 2
    assert median([4, 1, 3, 2]) == 2.5
    assert median([]) == 0.0


def test_detect_spike():
    hist = [1.0] * 20
    assert detect_spike(3.5, hist, k=3.0)        # > 3× 中位数
    assert not detect_spike(2.9, hist, k=3.0)
    assert detect_spike(float("nan"), hist)      # NaN 即 spike
    assert detect_spike(float("inf"), hist)
    assert not detect_spike(1.0, [])             # 无历史不判


def test_alarm_and_lead():
    # 基线 grad_norm ≈ 1.0，第 30 步起飙到 20，loss 第 35 步才 spike
    gnorms = [1.0] * 30 + [20.0] * 10
    losses = [1.0] * 35 + [10.0] * 5
    thresh = alarm_threshold(gnorms[:50], k=5.0)
    assert abs(thresh - 5.0) < 1e-9              # 中位数 1.0 × 5
    a = first_alarm_step(gnorms, thresh)
    s = first_spike_step(losses, k=3.0)
    assert a == 30 and s == 35
    assert lead_steps(a, s) == 5                 # 提前 5 步预警
    assert lead_steps(None, s) is None


def test_silent_degradation():
    # 后半段：train loss 降、eval ppl 升 → 静默退化
    train = [3.0, 2.8, 2.6, 2.4, 2.0, 1.5, 1.0, 0.7, 0.5, 0.3]
    ev = [400, 395, 390, 388, 390, 400, 420, 450, 490, 540]
    assert is_silent_degradation(train, ev, split_frac=0.4)
    # 正常训练：两者都降 → 不是退化
    ev_ok = [400, 380, 360, 340, 320, 300, 280, 260, 240, 220]
    assert not is_silent_degradation(train, ev_ok, split_frac=0.4)
    gap = degradation_gap_pct(ev, split_frac=0.4)
    # 后半段从 390 升到 540
    assert abs(gap - (540 - 390) / 390 * 100) < 0.01


def test_steps_lost():
    spikes = [60, 120, 180]
    assert steps_lost("skip_bad", spikes, 300) == 3
    assert steps_lost("lower_lr", spikes, 300) == 0
    assert steps_lost("tighten_clip", spikes, 300) == 0
    # rollback：每步丢 step % ckpt_every；60%40=20, 120%40=0, 180%40=20 → 40
    assert steps_lost("rollback_rewarmup", spikes, 300, ckpt_every=40) == 40


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        fn()
        passed += 1
        print(f"PASS {fn.__name__}")
    print(f"\n{passed}/{len(fns)} tests passed")
