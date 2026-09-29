"""test_synthetic.py —— 合成语料生成器离线单测（不联网）。"""

from __future__ import annotations

import unittest

from exp_data import dedup, synthetic


class TestSynthetic(unittest.TestCase):
    def test_uniform_docs_are_distinct(self):
        texts = synthetic.make_corpus(50, "uniform", doc_chars=300)
        self.assertEqual(len(texts), 50)
        self.assertEqual(len(set(texts)), 50)  # 全部互不相同

    def test_skewed_has_near_duplicate_group(self):
        texts = synthetic.make_corpus(100, "skewed", doc_chars=300, skew_ratio=0.3)
        skew_group = texts[:30]
        # 近重复组共享同一基底：前缀高度重合
        prefix = skew_group[0][:200]
        shared = sum(1 for t in skew_group if t.startswith(prefix[:100]))
        self.assertEqual(shared, 30)

    def test_skewed_minhash_removes_near_duplicates(self):
        texts = synthetic.make_corpus(60, "skewed", doc_chars=400, skew_ratio=0.5)
        cfg = dedup.MinHashConfig(num_perm=64, threshold=0.8, shingle_k=5)
        res = dedup.minhash_dedup(texts, cfg)
        # 30 条近重复只应保留 1 条，其余 29 条被判重；30 条随机各保留
        self.assertLessEqual(res["n_out"], 60 - 29 + 2)
        self.assertGreaterEqual(res["duplicates"], 25)

    def test_deterministic_with_same_seed(self):
        a = synthetic.make_corpus(20, "skewed", seed=123)
        b = synthetic.make_corpus(20, "skewed", seed=123)
        self.assertEqual(a, b)

    def test_moderate_docs_share_base_but_stay_distinct(self):
        texts = synthetic.make_corpus(40, "moderate", doc_chars=400,
                                      shared_frac=0.8)
        self.assertEqual(len(set(texts)), 40)  # 正文不同，整篇互不相同
        # 全部共享同一基底前缀（doc_chars × shared_frac）
        prefix = texts[0][:int(400 * 0.8)]
        self.assertTrue(all(t.startswith(prefix) for t in texts))

    def test_moderate_verifications_grow_superlinearly(self):
        # moderate 分布同桶碰撞但绝大多数互不判重，验证次数应随规模平方增长。
        # 注意 MinHash 是概率估计：shared_frac=0.8 时理论 Jaccard≈0.67，
        # 64 perm 的估计噪声（σ≈0.06）偶尔把个别对推过 0.8 阈值造成误判重，
        # 因此断言用"高留存率"而不是"全部保留"。
        cfg = dedup.MinHashConfig(num_perm=64, threshold=0.8, shingle_k=5)
        small = synthetic.make_corpus(200, "moderate", doc_chars=400,
                                      shared_frac=0.8)
        large = synthetic.make_corpus(400, "moderate", doc_chars=400,
                                      shared_frac=0.8)
        r_small = dedup.minhash_dedup(small, cfg)
        r_large = dedup.minhash_dedup(large, cfg)
        # 规模翻倍，验证次数应明显超过翻倍（平方项）
        self.assertGreater(r_large["verifications"], r_small["verifications"] * 2)
        # 绝大多数保留（误判重率 < 10%）
        self.assertGreaterEqual(r_small["n_out"], 180)
        self.assertGreaterEqual(r_large["n_out"], 360)


if __name__ == "__main__":
    unittest.main()
