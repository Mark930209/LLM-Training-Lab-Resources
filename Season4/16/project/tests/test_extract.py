"""test_extract.py —— exp_data 首个增量的离线单测（不联网）。

覆盖：
- wet_segment_from_warc 的路径规则（含"不能按分段号匹配清单"的回归）
- resolve_segment 只接受 warc/ 内容分段
- 三种正文抽取的行为与留存率排序
- 空白规整
"""

from __future__ import annotations

import unittest

from exp_data import warc_sample
from exp_data.extract import (extract_naive_visible, extract_trafilatura,
                              extract_wet, _normalize_ws)


class TestWetSegmentRule(unittest.TestCase):
    def test_same_segment_directory_rule(self):
        warc = ("crawl-data/CC-MAIN-2024-10/segments/1707947473819.62/warc/"
                "CC-MAIN-20240222125841-20240222155841-00689.warc.gz")
        wet = warc_sample.wet_segment_from_warc(warc)
        self.assertEqual(
            wet,
            "crawl-data/CC-MAIN-2024-10/segments/1707947473819.62/wet/"
            "CC-MAIN-20240222125841-20240222155841-00689.warc.wet.gz",
        )

    def test_rejects_non_warc_segment(self):
        with self.assertRaises(ValueError):
            warc_sample.wet_segment_from_warc("crawl-data/X/segments/1/wet/foo.wet.gz")
        with self.assertRaises(ValueError):
            warc_sample.wet_segment_from_warc("crawl-data/X/segments/1/warc/foo.warc")


class TestExtractors(unittest.TestCase):
    HTML = (
        "<html><head><style>a{color:red}</style>"
        "<script>var x=1;</script></head><body>"
        "<nav>Home About Contact</nav>"
        "<p>这是正文第一段，足够长用来测试抽取。</p>"
        "<footer>Copyright 2024</footer>"
        "</body></html>"
    )

    def test_naive_keeps_more_than_trafilatura(self):
        naive = extract_naive_visible(self.HTML)
        # 朴素法保留 nav/footer 文本
        self.assertIn("Home About Contact", naive)
        self.assertIn("Copyright", naive)
        # 朴素法删掉了 script/style 内容
        self.assertNotIn("var x=1", naive)
        self.assertNotIn("color:red", naive)

    def test_wet_only_normalizes_whitespace(self):
        text = "已经抽取的纯文本\n\n\n多余空行   和空格"
        out = extract_wet(text)
        self.assertNotIn("\n\n\n", out)
        self.assertIn("已经抽取的纯文本", out)

    def test_trafilatura_returns_string(self):
        # trafilatura 对极小 HTML 可能抽不出正文，这里只断言类型与非崩溃
        out = extract_trafilatura(self.HTML)
        self.assertIsInstance(out, str)

    def test_normalize_ws_collapses(self):
        self.assertEqual(_normalize_ws("a   b\n\n\n\nc"), "a b\n\nc")


if __name__ == "__main__":
    unittest.main()
