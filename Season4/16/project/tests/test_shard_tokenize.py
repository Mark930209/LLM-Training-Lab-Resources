"""test_shard_tokenize.py —— shard/tokenize_par 模块离线单测（不联网）。

重点覆盖提纲要求的失败案例：文档边界口径不一致时，boundary_audit 必须
报 mismatch（loss 看不出来的静默错位，审计要能看出来）。
"""

from __future__ import annotations

import tempfile
import unittest

from exp_data import shard, tokenize_par


class TestShard(unittest.TestCase):
    def test_write_and_stream_read_roundtrip(self):
        texts = [f"文档{i}的内容，" * 30 for i in range(25)]
        with tempfile.TemporaryDirectory() as tmp:
            w = shard.write_shards(texts, tmp, rows_per_shard=10)
            self.assertEqual(w["n_docs"], 25)
            self.assertEqual(w["n_shards"], 3)  # 10+10+5
            r = shard.stream_read_benchmark(tmp)
            self.assertEqual(r["n_docs"], 25)
            self.assertGreater(r["docs_per_sec"], 0)

    def test_empty_input_writes_nothing_bad(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = shard.write_shards([], tmp, rows_per_shard=10)
            self.assertEqual(w["n_docs"], 0)


class TestTokenize(unittest.TestCase):
    def test_char_tokenizer_and_parallel(self):
        texts = ["甲乙丙丁" * 20, "子丑寅卯" * 20]
        spec = tokenize_par.build_char_tokenizer(texts)
        res = tokenize_par.tokenize_parallel(texts, spec, workers=1)
        self.assertEqual(res["n_docs"], 2)
        self.assertEqual(res["doc_lengths"], [80, 80])
        # 词表 = 8 个不同字符 + EOS
        self.assertEqual(spec.vocab_size, 9)
        self.assertEqual(spec.eos_id, 8)

    def test_pack_stream_eos(self):
        token_lists = [[1, 2, 3], [4, 5]]
        stream = tokenize_par.pack_stream(token_lists, eos_id=9,
                                          eos_between_docs=True)
        self.assertEqual(stream, [1, 2, 3, 9, 4, 5, 9])
        stream2 = tokenize_par.pack_stream(token_lists, eos_id=9,
                                           eos_between_docs=False)
        self.assertEqual(stream2, [1, 2, 3, 4, 5])


class TestBoundaryAudit(unittest.TestCase):
    def _make_case(self, n_docs=12, doc_len=120):
        texts = ["".join(chr(ord("a") + (i + j) % 26) for j in range(doc_len))
                 for i in range(n_docs)]
        spec = tokenize_par.build_char_tokenizer(texts)
        res = tokenize_par.tokenize_parallel(texts, spec, workers=1)
        return res, spec

    def test_eos_true_passes(self):
        res, spec = self._make_case()
        stream = tokenize_par.pack_stream(res["token_lists"], spec.eos_id, True)
        audit = tokenize_par.boundary_audit(
            stream, res["doc_lengths"], spec.eos_id, True,
            seq_len=100, n_probes=10)
        self.assertEqual(audit["verdict"], "ok")
        self.assertEqual(audit["mismatches"], 0)

    def test_eos_false_reports_mismatch(self):
        # 失败案例：无 EOS 直接拼接，窗口跨文档且无边界标记 → mismatch
        res, spec = self._make_case()
        stream = tokenize_par.pack_stream(res["token_lists"], spec.eos_id, False)
        audit = tokenize_par.boundary_audit(
            stream, res["doc_lengths"], spec.eos_id, False,
            seq_len=100, n_probes=10)
        self.assertEqual(audit["verdict"], "boundary_mismatch")
        self.assertGreater(audit["mismatches"], 0)


if __name__ == "__main__":
    unittest.main()
