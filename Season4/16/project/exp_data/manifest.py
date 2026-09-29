"""manifest.py —— 数据血缘清单：各阶段留存率、配置哈希与来源。

manifest 是"这批数据从哪来、经过什么、每步删了多少"的单一事实来源。
17 篇配方消融、18 篇 tokenizer 训练、30 篇精确恢复都要读它：没有 manifest，
下游只能信任文件名，口径漂移无从发现（核心判断的落点）。

字段：
  created_at / pipeline_version : 时间与代码版本（code_sha256 摘要）
  source                        : crawl、分段、采样参数
  stages                        : 每阶段 n_in/n_out/留存率/耗时
  config_sha256                 : 全链路配置的哈希（配置即口径）
  truth_label                   : REAL / SIMULATED（本批数据的性质）
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any


def config_sha256(cfg: dict) -> str:
    """对配置做规范化 JSON 后取 SHA-256：配置即口径，口径要可哈希。"""
    canonical = json.dumps(cfg, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_manifest(source: dict, stages: list[dict], cfg: dict,
                   code_sha256: dict[str, str], truth_label: str = "REAL",
                   extra: dict[str, Any] | None = None) -> dict:
    """组装 manifest。stages 是有序列表，每项含 name/n_in/n_out/elapsed_sec。"""
    code_digest = hashlib.sha256(
        json.dumps(code_sha256, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "pipeline_version": code_digest,
        "truth_label": truth_label,
        "source": source,
        "stages": stages,
        "config_sha256": config_sha256(cfg),
        "code_sha256": code_sha256,
    }
    if extra:
        manifest["extra"] = extra
    return manifest


def stage_row(name: str, n_in: int, n_out: int,
              elapsed_sec: float | None = None, **kw: Any) -> dict:
    """生成一条阶段记录，自动算留存率。"""
    row = {
        "name": name,
        "n_in": n_in,
        "n_out": n_out,
        "doc_retention": round(n_out / n_in, 4) if n_in else None,
    }
    if elapsed_sec is not None:
        row["elapsed_sec"] = elapsed_sec
    row.update(kw)
    return row
