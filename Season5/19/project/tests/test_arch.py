"""test_arch.py —— exp_arch 离线单测（小配置、CPU、快速）。

覆盖：
- 组件全组合 forward 不崩、输出形状正确
- count_params 解析式与实建模型参数量逐格一致（等参数量对照的地基）
- solve_hidden_for_params 反解结果不超目标且最大化
- GQA/MQA 的 KV cache 按 kv_heads 比例缩小
- learned 位置编码与 RoPE 都能训一步（loss 有限）
- post-norm 结构 forward 正常
"""

from __future__ import annotations

import itertools
import unittest

import torch

from exp_arch import arch_metrics as am
from exp_arch.model_arch import ArchGPT, kv_heads_for

VOCAB = 500
BASE = dict(hidden=64, layers=2, heads=4, seq_len=32)


def _cfg(**over):
    c = dict(norm_type="rms", norm_pos="pre", ffn_type="swiglu",
             pos_enc="rope", attn_type="mha")
    c.update(over)
    return c


class TestForwardCombinations(unittest.TestCase):
    def test_all_axis_combinations_forward(self):
        x = torch.randint(0, VOCAB, (2, 16))
        for norm_type, norm_pos, ffn_type, pos_enc, attn_type in itertools.product(
                ("rms", "layer"), ("pre", "post"), ("swiglu", "gelu"),
                ("rope", "learned"), ("mha", "gqa", "mqa")):
            kv = kv_heads_for(attn_type, BASE["heads"])
            with self.subTest(norm_type=norm_type, norm_pos=norm_pos,
                              ffn_type=ffn_type, pos_enc=pos_enc, attn_type=attn_type):
                m = ArchGPT(VOCAB, kv_heads=kv, norm_type=norm_type,
                            norm_pos=norm_pos, ffn_type=ffn_type,
                            pos_enc=pos_enc, **BASE)
                out = m(x)
                self.assertEqual(out.shape, (2, 16, VOCAB))
                self.assertTrue(torch.isfinite(out).all())

    def test_generate_runs(self):
        m = ArchGPT(VOCAB, kv_heads=2, **BASE)
        idx = torch.randint(0, VOCAB, (1, 4))
        out = m.generate(idx, max_new_tokens=3, top_k=10)
        self.assertEqual(out.shape[1], 7)


class TestParamCount(unittest.TestCase):
    def test_analytic_matches_actual_all_cells(self):
        for norm_type, ffn_type, pos_enc, attn_type in itertools.product(
                ("rms", "layer"), ("swiglu", "gelu"), ("rope", "learned"),
                ("mha", "gqa", "mqa")):
            kv = kv_heads_for(attn_type, BASE["heads"])
            with self.subTest(norm_type=norm_type, ffn_type=ffn_type,
                              pos_enc=pos_enc, attn_type=attn_type):
                m = ArchGPT(VOCAB, kv_heads=kv, norm_type=norm_type,
                            ffn_type=ffn_type, pos_enc=pos_enc, **BASE)
                analytic = am.count_params(
                    VOCAB, BASE["hidden"], BASE["layers"], BASE["heads"], kv,
                    BASE["seq_len"], norm_type, ffn_type, pos_enc)["total"]
                self.assertEqual(analytic, m.n_params())

    def test_gelu_ffn_bias_counted(self):
        # GELU FFN 带 bias：解析式必须计入 d + hidden
        n = am.count_params(VOCAB, 64, 2, 4, 4, 32, "rms", "gelu", "rope")
        m = ArchGPT(VOCAB, ffn_type="gelu", **BASE)
        self.assertEqual(n["total"], m.n_params())


class TestEqualParams(unittest.TestCase):
    def test_solve_hidden_under_target_and_maximal(self):
        target = am.count_params(VOCAB, 64, 2, 4, 4, 32, "rms", "swiglu", "rope")["total"]
        for norm_type, ffn_type, pos_enc, attn_type in itertools.product(
                ("rms", "layer"), ("swiglu", "gelu"), ("rope", "learned"),
                ("mha", "gqa", "mqa")):
            kv = kv_heads_for(attn_type, 4)
            h = am.solve_hidden_for_params(target, VOCAB, 2, 4, kv, 32,
                                           norm_type, ffn_type, pos_enc)
            n = am.count_params(VOCAB, h, 2, 4, kv, 32, norm_type, ffn_type,
                                pos_enc)["total"]
            n_up = am.count_params(VOCAB, h + 8, 2, 4, kv, 32, norm_type,
                                   ffn_type, pos_enc)["total"]
            self.assertLessEqual(n, target)
            self.assertGreater(n_up, target)   # 再大一档就超（最大化）
            self.assertEqual(h % 8, 0)         # 2×heads 倍数（head_dim 偶数）

    def test_solved_hidden_head_dim_even(self):
        # 回归：反解结果的 head_dim 必须是偶数（RoPE 奇偶配对要求）
        target = am.count_params(VOCAB, 64, 2, 4, 4, 32, "rms", "swiglu", "rope")["total"]
        for heads in (4, 6, 8):
            h = am.solve_hidden_for_params(target, VOCAB, 2, heads, heads, 32,
                                           "layer", "gelu", "learned")
            self.assertEqual((h // heads) % 2, 0, f"heads={heads} hidden={h}")

    def test_rope_odd_head_dim_asserts(self):
        # 奇数 head_dim + RoPE 必须报错而不是静默崩溃
        with self.assertRaises(AssertionError):
            ArchGPT(VOCAB, hidden=66, layers=1, heads=6, seq_len=16,
                    pos_enc="rope")

    def test_modern_self_recovers(self):
        # modern 格自己反解自己：hidden 应回到 64
        target = am.count_params(VOCAB, 64, 2, 4, 4, 32, "rms", "swiglu", "rope")["total"]
        h = am.solve_hidden_for_params(target, VOCAB, 2, 4, 4, 32,
                                       "rms", "swiglu", "rope")
        self.assertEqual(h, 64)


class TestKVCache(unittest.TestCase):
    def test_gqa_mqa_shrink(self):
        mha = am.kv_cache_mib(6, 6, 64, 256)
        gqa = am.kv_cache_mib(6, 3, 64, 256)
        mqa = am.kv_cache_mib(6, 1, 64, 256)
        self.assertAlmostEqual(gqa, mha / 2, places=2)
        self.assertAlmostEqual(mqa, mha / 6, places=2)


class TestTrainStep(unittest.TestCase):
    def test_learned_pos_and_postnorm_step(self):
        for cfg in (_cfg(pos_enc="learned"), _cfg(norm_pos="post"),
                    _cfg(norm_type="layer", ffn_type="gelu", pos_enc="learned",
                         norm_pos="post")):
            kv = kv_heads_for(cfg.pop("attn_type"), BASE["heads"])
            m = ArchGPT(VOCAB, kv_heads=kv, **BASE, **cfg)
            opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
            x = torch.randint(0, VOCAB, (2, 16))
            y = torch.randint(0, VOCAB, (2, 16))
            loss = torch.nn.functional.cross_entropy(
                m(x).reshape(-1, VOCAB), y.reshape(-1))
            loss.backward()
            opt.step()
            self.assertTrue(torch.isfinite(loss), f"loss 非有限: {cfg}")


if __name__ == "__main__":
    unittest.main()
