"""test_pii_manifest.py —— pii/manifest 模块离线单测（不联网）。"""

from __future__ import annotations

import unittest

from exp_data import manifest, pii


class TestPii(unittest.TestCase):
    def test_email_scrubbed(self):
        text = "联系我们 support@example.com 获取详情"
        out, hits = pii.scrub(text)
        self.assertIn("<EMAIL>", out)
        self.assertNotIn("support@example.com", out)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].kind, "email")

    def test_phone_scrubbed(self):
        text = "客服电话 13812345678 或 +86 139-1234-5678 均可"
        out, hits = pii.scrub(text)
        self.assertNotIn("13812345678", out)
        self.assertEqual(len(hits), 2)
        self.assertTrue(all(h.kind == "phone" for h in hits))

    def test_id_number_scrubbed(self):
        text = "身份证号 110101199003074518 已登记"
        out, hits = pii.scrub(text)
        self.assertIn("<ID>", out)
        self.assertEqual(len(hits), 1)

    def test_clean_text_untouched(self):
        text = "这是一段没有任何个人信息的正常文本。"
        out, hits = pii.scrub(text)
        self.assertEqual(out, text)
        self.assertEqual(hits, [])

    def test_scrub_batch_stats(self):
        texts = ["联系 a@b.com", "正常文本一段", "电话 13812345678"]
        res = pii.scrub_batch(texts)
        self.assertEqual(res["n_in"], 3)
        self.assertEqual(res["docs_with_pii"], 2)
        self.assertEqual(res["total_hits"], 2)
        self.assertEqual(len(res["cleaned_texts"]), 3)


class TestManifest(unittest.TestCase):
    def test_config_sha256_stable_and_order_free(self):
        cfg_a = {"x": 1, "y": {"z": 2}}
        cfg_b = {"y": {"z": 2}, "x": 1}
        self.assertEqual(manifest.config_sha256(cfg_a),
                         manifest.config_sha256(cfg_b))

    def test_stage_row_retention(self):
        row = manifest.stage_row("quality", 100, 70, 1.5)
        self.assertEqual(row["doc_retention"], 0.7)
        self.assertEqual(row["elapsed_sec"], 1.5)

    def test_stage_row_zero_input(self):
        row = manifest.stage_row("empty", 0, 0)
        self.assertIsNone(row["doc_retention"])

    def test_build_manifest_fields(self):
        mf = manifest.build_manifest(
            source={"segment": "seg.warc.gz"},
            stages=[manifest.stage_row("extract", 10, 8)],
            cfg={"a": 1},
            code_sha256={"pipeline.py": "abc"},
            truth_label="REAL",
        )
        for key in ("created_at", "pipeline_version", "truth_label",
                    "source", "stages", "config_sha256", "code_sha256"):
            self.assertIn(key, mf)
        self.assertEqual(mf["truth_label"], "REAL")
        self.assertEqual(len(mf["config_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
