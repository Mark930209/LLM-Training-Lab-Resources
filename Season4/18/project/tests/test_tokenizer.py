"""test_tokenizer.py —— exp_tokenizer 离线单测（小语料、小词表，快速）。

覆盖：
- BPE 与 Unigram 两条训练路线都能产出可加载的 tokenizer
- 数字切分策略 merge vs single 的实际差异（数字被合并还是逐位）
- 压缩率随词表增大而改善（bytes_per_token 下降）
- embedding 参数占比解析式与手算一致
- 覆盖率度量对 SentencePiece 空白规范化鲁棒（不虚报为 0）
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from exp_tokenizer import tok_metrics, tok_train


def _build_corpus() -> str:
    """构造多样化小语料：足够多的唯一字符/子词，让词表档位与数字策略的差异显现。

    真实 sweep 用 19MB 两域语料；单测用这个小但多样的语料，避免"同几行重复
    300 遍"导致唯一字符太少（Unigram 词表上不去、BPE 大小词表压缩率相同）。
    """
    import random
    rng = random.Random(20260924)
    hanzi = "".join(chr(c) for c in range(0x4E00, 0x4E00 + 600))  # 600 个不同汉字
    words = ["model", "token", "vocab", "compress", "embed", "matrix", "vector",
             "tensor", "gradient", "batch", "epoch", "layer", "attention", "norm",
             "the", "quick", "brown", "fox", "jumps", "lazy", "dog", "data", "code"]
    lines = []
    # 中文：随机汉字组合，保证唯一性
    for _ in range(400):
        seg = "".join(rng.choice(hanzi) for _ in range(rng.randint(8, 30)))
        lines.append(seg + "。\n")
    # 英文：随机词组合
    for _ in range(300):
        seg = " ".join(rng.choice(words) for _ in range(rng.randint(6, 20)))
        lines.append(seg + ".\n")
    # 数字：多种长度
    for _ in range(200):
        nums = " ".join(str(rng.randint(0, 999999)) for _ in range(rng.randint(3, 8)))
        lines.append(nums + "\n")
    # 代码：带缩进与符号
    for i in range(200):
        lines.append(f"def func_{i}(arg):\n    return arg * {i} + {i % 7}\n")
    return "".join(lines)


_CORPUS = _build_corpus()


class TestTrainRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.corpus_file = Path(cls.tmp.name) / "corpus.txt"
        cls.corpus_file.write_text(_CORPUS, encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_bpe_train_and_load(self):
        info = tok_train.train_bpe(self.corpus_file, 600, "merge", self.tmp.name)
        self.assertEqual(info["algo"], "bpe")
        self.assertGreater(info["actual_vocab"], 0)
        tok = tok_train.load_tokenizer(info)
        ids = tok_train.encode_text(tok, info, "中文测试 123")
        self.assertGreater(len(ids), 0)

    def test_unigram_train_and_load(self):
        # byte_fallback 下 Unigram 词表需 ≥ 唯一字符数 + 256 byte pieces + 元符号
        info = tok_train.train_unigram(self.corpus_file, 1000, "merge", self.tmp.name)
        self.assertEqual(info["algo"], "unigram")
        tok = tok_train.load_tokenizer(info)
        ids = tok_train.encode_text(tok, info, "中文测试 123")
        self.assertGreater(len(ids), 0)


class TestDigitStrategy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.corpus_file = Path(cls.tmp.name) / "corpus.txt"
        # 数字密集语料，让两种策略的差异显现
        cls.corpus_file.write_text(
            ("数字 12345 67890 24680 13579 11111 22222 33333\n" * 400)
            + ("文本内容 abc def ghi\n" * 400), encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_single_splits_digits_finer_than_merge(self):
        text = "99999"
        m = tok_train.train_bpe(self.corpus_file, 500, "merge", self.tmp.name)
        s = tok_train.train_bpe(self.corpus_file, 500, "single", self.tmp.name)
        tok_m, tok_s = tok_train.load_tokenizer(m), tok_train.load_tokenizer(s)
        n_merge = len(tok_train.encode_text(tok_m, m, text))
        n_single = len(tok_train.encode_text(tok_s, s, text))
        # 逐位切分的 token 数应不少于合并切分
        self.assertGreaterEqual(n_single, n_merge)


class TestCompression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.corpus_file = Path(cls.tmp.name) / "corpus.txt"
        cls.corpus_file.write_text(_CORPUS, encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_bigger_vocab_compresses_better(self):
        sample = _CORPUS[:20000]
        small = tok_train.train_bpe(self.corpus_file, 400, "merge", self.tmp.name)
        big = tok_train.train_bpe(self.corpus_file, 1200, "merge", self.tmp.name)
        ts, tb = tok_train.load_tokenizer(small), tok_train.load_tokenizer(big)
        bpt_small = tok_metrics.compression_metrics(ts, small, sample)["bytes_per_token"]
        bpt_big = tok_metrics.compression_metrics(tb, big, sample)["bytes_per_token"]
        # 词表越大，每 token 承载的字节越多（bytes_per_token 越大 = 压缩越好）
        self.assertGreater(bpt_big, bpt_small)


class TestEmbeddingShare(unittest.TestCase):
    def test_share_matches_manual(self):
        # vocab=6120 hidden=384：embedding = 6120×384 = 2,350,080
        res = tok_metrics.embedding_param_share(6120, 384, 6, 6)
        self.assertEqual(res["embedding_params"], 6120 * 384)
        self.assertGreater(res["embedding_share"], 0)
        self.assertLess(res["embedding_share"], 1)

    def test_bigger_vocab_bigger_share(self):
        s = tok_metrics.embedding_param_share(8000, 384, 6, 6)["embedding_share"]
        b = tok_metrics.embedding_param_share(64000, 384, 6, 6)["embedding_share"]
        self.assertGreater(b, s)  # 词表越大，embedding 占比越高


class TestCoverageRobust(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.corpus_file = Path(cls.tmp.name) / "corpus.txt"
        cls.corpus_file.write_text(_CORPUS, encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_unigram_coverage_not_falsely_zero(self):
        # SentencePiece 折叠空白，覆盖率会低于 BPE，但不应虚报为接近 0
        info = tok_train.train_unigram(self.corpus_file, 1000, "merge", self.tmp.name)
        tok = tok_train.load_tokenizer(info)
        cov = tok_metrics.coverage_metrics(tok, info, _CORPUS[:3000])
        self.assertGreater(cov["roundtrip_similarity"], 0.7)

    def test_bpe_coverage_near_one(self):
        info = tok_train.train_bpe(self.corpus_file, 600, "merge", self.tmp.name)
        tok = tok_train.load_tokenizer(info)
        cov = tok_metrics.coverage_metrics(tok, info, _CORPUS[:3000])
        self.assertGreater(cov["roundtrip_similarity"], 0.95)


class TestParamBudget(unittest.TestCase):
    """固定参数预算的解析式：词表挤占模型容量的数学形态。"""

    def test_non_emb_matches_17_real(self):
        # 17 篇实测 12,971,904 = 6120×384（embedding，权重共享只计一份）
        # + 非 embedding 部分；解析式必须与实测一字不差
        from exp_tokenizer import pipeline
        non_emb = pipeline._non_emb_params(384, 6)
        self.assertEqual(6120 * 384 + non_emb, 12971904)

    def test_fixed_budget_recovers_17_hidden(self):
        # 词表 6120、目标 12,971,904 时反解 hidden 应回到 384 附近
        from exp_tokenizer import pipeline
        res = pipeline._fixed_budget_hidden(6120, 12971904, 6)
        self.assertIn(res["hidden"], (383, 384))
        self.assertLessEqual(res["total_params"], 12971904)

    def test_bigger_vocab_smaller_hidden(self):
        from exp_tokenizer import pipeline
        h8 = pipeline._fixed_budget_hidden(8000, 12971904, 6)["hidden"]
        h64 = pipeline._fixed_budget_hidden(64000, 12971904, 6)["hidden"]
        self.assertGreater(h8, h64)  # 词表越大，同预算下 hidden 越小

    def test_bpc_manual(self):
        from exp_tokenizer import pipeline
        # loss=2.0、1000 token、2000 字符 → 2.0×1000/2000/ln2 = 1/ln2
        import math
        self.assertAlmostEqual(pipeline._bpc(2.0, 1000, 2000),
                               1.0 / math.log(2), places=4)
        self.assertIsNone(pipeline._bpc(None, 1000, 2000))


class TestTokenizerAdapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.corpus_file = Path(cls.tmp.name) / "corpus.txt"
        cls.corpus_file.write_text(_CORPUS, encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_adapter_encode_passthrough(self):
        from exp_tokenizer import pipeline
        info = tok_train.train_bpe(self.corpus_file, 600, "merge", self.tmp.name)
        tok = tok_train.load_tokenizer(info)
        ad = pipeline._TokenizerAdapter(tok, info)
        self.assertEqual(ad.vocab_size, info["actual_vocab"])
        self.assertEqual(ad.encode("中文测试 123"),
                         tok_train.encode_text(tok, info, "中文测试 123"))


if __name__ == "__main__":
    unittest.main()
