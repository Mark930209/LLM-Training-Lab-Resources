"""test_dedup.py —— dedup 模块离线单测（不联网、不下载）。

覆盖：
- 精确去重：完全重复删除、规整后重复（空白差异）也删除
- MinHash：近似重复被判重、明显不同文本保留
- shingle：短文本与空文本的边界
- 分桶倾斜统计字段存在且自洽
"""

from __future__ import annotations

import unittest

from exp_data import dedup


class TestExactDedup(unittest.TestCase):
    def test_removes_exact_duplicates(self):
        texts = ["文档内容甲" * 30, "文档内容乙" * 30, "文档内容甲" * 30]
        res = dedup.exact_dedup(texts)
        self.assertEqual(res["n_in"], 3)
        self.assertEqual(res["n_out"], 2)
        self.assertEqual(res["duplicates"], 1)

    def test_whitespace_normalized_duplicate(self):
        a = "这是 一段   文本" * 20
        b = "这是 一段 文本" * 20  # 规整后与 a 相同
        res = dedup.exact_dedup([a, b])
        self.assertEqual(res["n_out"], 1)

    def test_empty_input(self):
        res = dedup.exact_dedup([])
        self.assertEqual(res["n_in"], 0)
        self.assertIsNone(res["doc_retention"])


class TestShingles(unittest.TestCase):
    def test_short_text_single_shingle(self):
        self.assertEqual(dedup._shingles("abc", 5), {"abc"})

    def test_empty_text(self):
        self.assertEqual(dedup._shingles("", 5), set())

    def test_k_shingle_count(self):
        sh = dedup._shingles("abcdefg", 3)
        self.assertEqual(len(sh), 5)  # abc bcd cde def efg


class TestMinHashDedup(unittest.TestCase):
    CFG = dedup.MinHashConfig(num_perm=64, threshold=0.7, shingle_k=3)

    def test_near_duplicate_detected(self):
        # base 用内容多样的长文本，保证唯一 shingle 足够多；
        # near 只在结尾追加一句，新增 shingle 占比可忽略，Jaccard 仍高于阈值。
        base = "".join(f"第{i}句关于数据管线的不同内容，" for i in range(60))
        near = base + "结尾多了一句。"
        diff = "".join(f"另一篇{i}讨论模型训练的不同段落，" for i in range(60))
        res = dedup.minhash_dedup([base, near, diff], self.CFG)
        self.assertEqual(res["n_in"], 3)
        # base 保留；near 与 base 近似应被判重；diff 保留
        self.assertEqual(res["duplicates"], 1)
        self.assertEqual(res["n_out"], 2)

    def test_distinct_texts_kept(self):
        # 用确定性的随机字符序列，保证 5 篇文档的 shingle 集合差异足够大，
        # 不会因共享大量相同字词而被 MinHash 误判为近似重复。
        import random

        pool = "数据管线去重分片抽取过滤语种识别训练模型评测语料网页正文模板导航清洗留存"
        texts = []
        for i in range(5):
            rng = random.Random(2026 + i)
            texts.append("".join(rng.choice(pool) for _ in range(400)))
        res = dedup.minhash_dedup(texts, self.CFG)
        self.assertEqual(res["n_out"], 5)
        self.assertEqual(res["duplicates"], 0)

    def test_bucket_skew_fields(self):
        texts = ["".join(f"共享{i}开头的内容片段，" for _ in range(30)) + f"变体{i}"
                 for i in range(6)]
        res = dedup.minhash_dedup(texts, self.CFG)
        skew = res["bucket_skew"]
        self.assertIn("n_buckets", skew)
        self.assertIn("max_bucket", skew)
        self.assertIn("skew_ratio", skew)
        self.assertGreaterEqual(skew["max_bucket"], 1)

    def test_empty_text_preserved(self):
        res = dedup.minhash_dedup(["", "正常文本内容" * 20], self.CFG)
        # 空文本不进 LSH，直接保留
        self.assertEqual(res["n_out"], 2)


if __name__ == "__main__":
    unittest.main()
