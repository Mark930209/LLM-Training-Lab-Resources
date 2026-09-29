"""test_quality_langid.py —— quality/langid 模块离线单测（不联网）。"""

from __future__ import annotations

import unittest

from exp_data import langid, quality


class TestQuality(unittest.TestCase):
    CFG = quality.QualityConfig(min_chars=50, max_symbol_ratio=0.30,
                                max_repeat_line_ratio=0.30)

    def test_clean_text_passes(self):
        text = "这是一段足够长的正常文本，" * 6
        res = quality.check(text, self.CFG)
        self.assertTrue(res.passed)
        self.assertEqual(res.reasons, [])

    def test_too_short(self):
        res = quality.check("太短", self.CFG)
        self.assertFalse(res.passed)
        self.assertIn("too_short", res.reasons)

    def test_high_symbol_ratio(self):
        text = "$$%%##@@!!" * 20
        res = quality.check(text, self.CFG)
        self.assertFalse(res.passed)
        self.assertIn("high_symbol_ratio", res.reasons)

    def test_high_repeat_line(self):
        text = "\n".join(["导航首页"] * 10 + ["正文内容一段"] * 2)
        res = quality.check(text, self.CFG)
        self.assertFalse(res.passed)
        self.assertIn("high_repeat_line", res.reasons)

    def test_symbol_ratio_ignores_whitespace(self):
        self.assertEqual(quality.symbol_ratio("   "), 0.0)
        self.assertGreater(quality.symbol_ratio("abc$$$"), 0.4)

    def test_filter_batch_counts(self):
        texts = ["这是一段足够长的正常文本，" * 6, "短", "$$%%##@@!!" * 20]
        out = quality.filter_batch(texts, self.CFG)
        self.assertEqual(out["n_in"], 3)
        self.assertEqual(out["n_out"], 1)
        self.assertEqual(out["reason_counts"].get("too_short"), 1)


class TestLangId(unittest.TestCase):
    CFG = langid.LangIdConfig(target_langs=("en", "zh-cn"), min_prob=0.5,
                              min_chars_for_detect=20)

    def test_english_detected_ok(self):
        res = langid.detect(
            "This is a reasonably long English sentence used for language detection.",
            self.CFG)
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["lang"], "en")

    def test_too_short_not_detected(self):
        res = langid.detect("hi", self.CFG)
        self.assertEqual(res["status"], "too_short")

    def test_filter_batch_keeps_target(self):
        texts = [
            "This is a reasonably long English sentence used for language detection.",
            "bonjour",  # too short
        ]
        out = langid.filter_batch(texts, self.CFG)
        self.assertEqual(out["n_in"], 2)
        self.assertEqual(out["n_kept"], 1)
        self.assertEqual(out["status_counts"].get("too_short"), 1)


if __name__ == "__main__":
    unittest.main()
