"""test_recipe.py —— exp_recipe 离线单测（不联网、不训练、不读大语料）。

用小型合成文本驱动，覆盖：
- 域切分确定性、评测集与训练池不重叠
- 重复注入的数量、比例与确定性
- 配方制备的过滤/去重/配比三阶段与报告字段
- 多样性指标对重复的敏感性（去重过激会拉低 unique n-gram 的机制反证）
"""

from __future__ import annotations

import unittest

from exp_recipe import corpus, recipe


def _mk_nl(n_paras=40, para_len=300):
    return "\n".join("第%d段" % i + "内容" * para_len for i in range(n_paras))


def _mk_code_docs(n=25):
    return [(f"mod{i}.py", f"def f{i}():\n    return {i}\n" + "# 注释\n" * 60)
            for i in range(n)]


class TestCorpusSplit(unittest.TestCase):
    def test_split_is_deterministic_and_disjoint(self):
        nl = _mk_nl()
        code = _mk_code_docs()
        d1 = corpus.split_domains(nl, code)
        d2 = corpus.split_domains(nl, code)
        self.assertEqual(d1["nl_eval"], d2["nl_eval"])
        self.assertEqual(len(d1["code_eval"]), len(d2["code_eval"]))
        # 评测集不进训练池
        self.assertNotIn(d1["nl_eval"][:200], d1["nl_train"])
        eval_names = {n for n, _ in d1["code_eval"]}
        train_names = {n for n, _ in d1["code_train"]}
        self.assertEqual(eval_names & train_names, set())

    def test_fingerprint_stable(self):
        d = corpus.split_domains(_mk_nl(), _mk_code_docs())
        fp1 = corpus.eval_fingerprint(d)
        fp2 = corpus.eval_fingerprint(d)
        self.assertEqual(fp1, fp2)
        self.assertEqual(len(fp1["nl_eval_sha256"]), 16)


class TestDuplicateInjection(unittest.TestCase):
    def test_injection_counts_and_determinism(self):
        docs = ["文档%d" % i + "内容" * 100 for i in range(100)]
        pool1, s1 = corpus.inject_duplicates(docs)
        pool2, s2 = corpus.inject_duplicates(docs)
        self.assertEqual(pool1, pool2)  # 固定 seed → 确定性
        self.assertEqual(s1["n_original"], 100)
        self.assertEqual(s1["n_exact_copies"], 10)
        self.assertEqual(s1["n_near_copies"], 10)
        self.assertEqual(s1["n_total"], 120)
        self.assertEqual(len(pool1), 120)

    def test_near_copy_differs_from_original(self):
        docs = ["原始文档内容" * 50]
        pool, _ = corpus.inject_duplicates(docs, exact_ratio=0.0, near_ratio=1.0)
        self.assertEqual(len(pool), 2)
        self.assertNotEqual(pool[0], pool[1])  # 扰动副本与原文不同


class TestRecipePrepare(unittest.TestCase):
    def setUp(self):
        self.nl_docs = ["小说段落%d" % i + "汉字内容" * 80 for i in range(60)]
        self.code_docs = ["def g%d():" % i + "    pass\n" * 5 + "x = %d\n" % i
                          for i in range(60)]

    def test_mix_ratio_respected(self):
        spec = recipe.RecipeSpec("t", nl_ratio=0.8, dedup="none",
                                 quality=False, char_budget=20000)
        res = recipe.prepare_recipe(spec, self.nl_docs, self.code_docs)
        mix = res["report"]["stages"]["mix"]
        self.assertAlmostEqual(mix["actual_nl_ratio"], 0.8, delta=0.15)
        self.assertIn("text_sha256", res["report"])

    def test_exact_dedup_removes_injected_copies(self):
        docs = ["唯一文档%d" % i + "内容" * 60 for i in range(30)]
        docs += [docs[0], docs[1]]  # 两个精确副本
        spec = recipe.RecipeSpec("t", nl_ratio=1.0, dedup="exact",
                                 quality=False, char_budget=100000)
        res = recipe.prepare_recipe(spec, docs, [])
        self.assertEqual(res["report"]["stages"]["dedup"]["nl"]["duplicates"], 2)

    def test_quality_stage_records_reasons(self):
        docs = self.nl_docs + ["短"]  # 一条过短
        spec = recipe.RecipeSpec("t", nl_ratio=1.0, dedup="none",
                                 quality=True, char_budget=100000)
        res = recipe.prepare_recipe(spec, docs, [])
        q = res["report"]["stages"]["quality"]["nl"]
        self.assertGreaterEqual(q["reason_counts"].get("too_short", 0), 1)


class TestDiversity(unittest.TestCase):
    def test_repetition_lowers_unique_ratio(self):
        unique = "".join("字符%d" % i for i in range(500))
        repeated = "同一句话重复很多遍。" * 200
        m_u = recipe.diversity_metrics(unique)
        m_r = recipe.diversity_metrics(repeated)
        self.assertGreater(m_u["unique_4gram_ratio"], m_r["unique_4gram_ratio"])


if __name__ == "__main__":
    unittest.main()
