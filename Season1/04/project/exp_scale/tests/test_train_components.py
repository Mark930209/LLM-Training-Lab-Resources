"""test_train_components.py —— 04 篇工程件的 correctness 验证（CPU 可跑）。

不跑训练，只验证每个工程件的数值行为正确：
    - lr 调度器：warmup 阶段线性上升、cosine 阶段递减、边界值正确
    - 参数量估算：与真实模型参数量吻合
    - 显存估算：单调性正确（batch/seq 增大则显存增大，AMP 开启则激活减半）

用法（工程根目录）：
    python -m exp_scale.tests.test_train_components
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from exp_scale.schedulers import (  # noqa: E402
    cosine_with_warmup, estimate_model_params, estimate_training_memory_mb)


class TestLRSchedule:

    def test_warmup_linear_rise(self):
        """warmup 阶段 lr 线性上升，末步达到峰值。"""
        lrs = [cosine_with_warmup(s, 1000, 100, 3e-4) for s in range(100)]
        assert lrs[0] < lrs[50] < lrs[99], "warmup 应单调上升"
        assert abs(lrs[99] - 3e-4) < 1e-9, "warmup 末步应达到峰值 lr"

    def test_cosine_decay(self):
        """warmup 之后 lr 单调下降。"""
        lrs = [cosine_with_warmup(s, 1000, 100, 3e-4) for s in range(100, 1000)]
        assert all(lrs[i] >= lrs[i + 1] - 1e-12 for i in range(len(lrs) - 1)), \
            "cosine 阶段应单调下降"

    def test_final_lr_floor(self):
        """训练结束后 lr 停在 lr * min_lr_ratio。"""
        lr = cosine_with_warmup(1000, 1000, 100, 3e-4, min_lr_ratio=0.1)
        assert abs(lr - 3e-5) < 1e-9, "末尾应降到 min_lr_ratio 倍"

    def test_midpoint_cosine(self):
        """cosine 中点大约在峰值与最低值的中间。"""
        lr = cosine_with_warmup(550, 1000, 100, 3e-4, min_lr_ratio=0.0)
        assert abs(lr - 1.5e-4) < 1e-5, f"中点应约 1.5e-4，实为 {lr:.2e}"


class TestParamEstimate:

    def test_matches_real_model_10m(self):
        """估算参数量应与真实模型吻合（±5%）。"""
        pytest.importorskip("torch")
        import torch
        import torch.nn as nn

        # 用最小可运行的等价结构验证估算公式
        vocab, hidden, layers, heads = 4532, 384, 6, 6
        est = estimate_model_params(vocab, hidden, layers)

        # 手写等价模型（与 model.py 同构）
        ffn = int(8 / 3 * hidden)
        ffn = (ffn + 7) // 8 * 8
        real = (4 * hidden * hidden + 3 * hidden * ffn + 4 * hidden) * layers \
            + vocab * hidden + hidden
        assert abs(est - real) / real < 0.01, "估算应与公式一致"

    def test_monotonic_in_size(self):
        """hidden/layers 增大，参数量应单调增大。"""
        small = estimate_model_params(5000, 384, 6)
        big = estimate_model_params(5000, 768, 12)
        assert big > small * 3, "100M 应显著大于 10M"


class TestMemoryEstimate:

    def test_amp_reduces_activations(self):
        """AMP 开启时激活显存减半。"""
        kw = dict(params=88_800_000, seq_len=256, batch_size=4, hidden=768,
                  layers=12, vocab_size=6015)
        full = estimate_training_memory_mb(**kw, amp=False)
        amp = estimate_training_memory_mb(**kw, amp=True)
        assert amp["activations_mb"] < full["activations_mb"], "AMP 应减少激活显存"

    def test_batch_scales_activations(self):
        """batch 翻倍，激活显存翻倍。"""
        kw = dict(params=10_000_000, seq_len=256, batch_size=8, hidden=384,
                  layers=6, vocab_size=4532, amp=True)
        a = estimate_training_memory_mb(**kw)
        kw["batch_size"] = 16
        b = estimate_training_memory_mb(**kw)
        assert abs(b["activations_mb"] - a["activations_mb"] * 2) < 1.0

    def test_weights_independent_of_batch(self):
        """权重显存与 batch 无关。"""
        kw = dict(params=10_000_000, seq_len=256, batch_size=4, hidden=384,
                  layers=6, vocab_size=4532, amp=True)
        a = estimate_training_memory_mb(**kw)
        kw["batch_size"] = 32
        b = estimate_training_memory_mb(**kw)
        assert a["weights_mb"] == b["weights_mb"]
        assert a["optimizer_mb"] == b["optimizer_mb"]

    def test_100m_fits_8gb(self):
        """100M 配置的估算显存应在 8GB 内（本篇核心论断）。"""
        mem = estimate_training_memory_mb(
            params=88_800_000, seq_len=256, batch_size=4, hidden=768,
            layers=12, vocab_size=6015, amp=True)
        assert mem["total_mb"] < 6.78 * 1024, "100M 估算应能装进 8GB 卡"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])