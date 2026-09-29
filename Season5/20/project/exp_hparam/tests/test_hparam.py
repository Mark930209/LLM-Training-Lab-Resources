"""exp_hparam 纯函数单测（CPU 即可，不依赖 GPU）。"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hparam_metrics import (  # noqa: E402
    n_params_dense, tok_per_sec, train_seconds, total_steps,
    lr_sqrt_scale, lr_linear_scale, flops_6nd, tokens_for_flops,
    chinchilla_optimal_params, fit_power_law, predict_loss, rel_error,
    non_emb_params, lr_at, lr_range_at,
)


def test_n_params_scales_with_hidden():
    p256 = n_params_dense(256, 6, 6120)
    p384 = n_params_dense(384, 6, 6120)
    p512 = n_params_dense(512, 6, 6120)
    assert p256 < p384 < p512
    # hidden 翻倍，非 embedding 部分约 4 倍
    assert p512 > p256 * 2


def test_n_params_matches_19_modern():
    # 19 篇 modern：hidden 384, 6 层, vocab 6120, SwiGLU 2.67 → 12,971,904
    p = n_params_dense(384, 6, 6120, 2.67)
    assert p == 12971904, f"应等于 19 篇 modern 实测参数量，实得 {p}"


def test_tok_per_sec_inverse_square():
    ref = tok_per_sec(384, 384, 54682)
    assert abs(ref - 54682) < 1e-6
    half = tok_per_sec(768, 384, 54682)
    assert abs(half - 54682 / 4) < 1e-6


def test_train_seconds_positive():
    s = train_seconds(2_000_000, 384, 384, 54682)
    assert abs(s - 2_000_000 / 54682) < 1e-6


def test_total_steps():
    # 2M token / (batch16 × seq256) = 488
    assert total_steps(2_000_000, 16) == 488
    assert total_steps(1_000_000, 16) == 244


def test_lr_scaling_relations():
    # batch 16→64（4 倍）：线性 ×4，平方根 ×2
    assert abs(lr_linear_scale(0.0003, 16, 64) - 0.0012) < 1e-9
    assert abs(lr_sqrt_scale(0.0003, 16, 64) - 0.0006) < 1e-9


def test_flops_6nd_roundtrip():
    c = flops_6nd(1_000_000, 2_000_000)
    assert abs(c - 6 * 1e6 * 2e6) < 1
    d = tokens_for_flops(c, 1_000_000)
    assert abs(d - 2_000_000) < 1e-3


def test_chinchilla_optimal():
    # C=6ND, D=20N → C=120N² → N=sqrt(C/120)
    c = 1.2e16
    n = chinchilla_optimal_params(c)
    assert abs(n - math.sqrt(c / 120)) < 1


def test_fit_power_law_recovers_exponent():
    # 构造 y = 2 * x^-0.1，拟合应还原 a≈2, b≈-0.1
    xs = [1e6, 4e6, 1.6e7, 6.4e7]
    a_true, b_true = 2.0, -0.1
    ys = [a_true * (x ** b_true) for x in xs]
    a, b = fit_power_law(xs, ys)
    assert abs(b - b_true) < 1e-6
    assert abs(a - a_true) < 1e-6


def test_predict_and_rel_error():
    a, b = 2.0, -0.1
    pred = predict_loss(a, b, 1e7)
    assert abs(pred - 2.0 * (1e7 ** -0.1)) < 1e-9
    assert abs(rel_error(90, 100) - 10.0) < 1e-9


def test_non_emb_params_formula():
    # 18 篇口径：12H²L + H(2L+1)。hidden384/6层 → 12×147456×6 + 384×13
    ne = non_emb_params(384, 6)
    assert ne == 12 * 384 * 384 * 6 + 384 * 13 == 10621824
    # 总参数 = 非 embedding + tied embedding（vocab×hidden）
    total = n_params_dense(384, 6, 6120)
    assert total - ne == 6120 * 384


def test_lr_at_warmup_and_decays():
    total, lr, wu = 100, 0.001, 0.1
    # warmup 段线性：step 5 / warmup 10 → 一半
    assert abs(lr_at(5, total, lr, wu) - 0.0005) < 1e-12
    # warmup 结束点 = 峰值
    assert abs(lr_at(10, total, lr, wu) - lr) < 1e-9
    # cosine 中点（prog=0.5）= lr/2
    mid = 10 + (total - 10) // 2
    assert abs(lr_at(mid, total, lr, wu, "cosine") - lr * 0.5) < 2e-3
    # linear 中点 = lr/2
    assert abs(lr_at(mid, total, lr, wu, "linear") - lr * 0.5) < 2e-3
    # constant 恒为 lr
    assert abs(lr_at(mid, total, lr, wu, "constant") - lr) < 1e-12
    # 末点：cosine/linear → 0，constant → lr
    assert lr_at(total - 1, total, lr, wu, "cosine") < 1e-5
    assert lr_at(total - 1, total, lr, wu, "linear") < lr * 0.02
    assert abs(lr_at(total - 1, total, lr, wu, "constant") - lr) < 1e-12
    # 与 17 篇 _lr_at 同款验证：cosine 是默认
    assert abs(lr_at(30, total, lr, wu) - lr_at(30, total, lr, wu, "cosine")) < 1e-15


def test_lr_range_at_exponential():
    # 起点 = lr_min，终点 = lr_max，中点 = 几何均值
    assert abs(lr_range_at(0, 100, 3e-5, 3e-2) - 3e-5) < 1e-12
    assert abs(lr_range_at(99, 100, 3e-5, 3e-2) - 3e-2) < 1e-9
    # total=99 时 step=49 的指数恰为 49/98=0.5 → 几何均值
    mid = lr_range_at(49, 99, 3e-5, 3e-2)
    assert abs(mid - math.sqrt(3e-5 * 3e-2)) < 1e-9
    # 单调上升
    curve = [lr_range_at(s, 100, 3e-5, 3e-2) for s in range(100)]
    assert all(curve[i] < curve[i + 1] for i in range(99))


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        fn()
        passed += 1
        print(f"PASS {fn.__name__}")
    print(f"\n{passed}/{len(fns)} tests passed")
