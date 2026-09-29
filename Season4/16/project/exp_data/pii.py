"""pii.py —— PII 检测与移除（邮箱、电话、证件号）。

规则式检测：正则命中即替换为占位符，并记录每类命中次数。
误伤率核对方式：把替换前后的文本各抽样若干条，人工可核对占位符上下文
（本模块输出命中片段的前后 20 字符窗口，供审计而不是直接进语料）。

边界说明：规则式 PII 检测召回有限（变体写法、非标准格式会漏），
工业管线会叠加 NER 模型；本篇只演示规则层的口径与代价，不声称完备。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# 邮箱：常规 local@domain 形式
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
# 中国大陆手机号：11 位、1 开头，允许 +86 前缀与分隔符
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?86[\s\-]?)?1[3-9]\d(?:[\s\-]?\d){8}(?!\d)")
# 中国大陆身份证号：18 位（末位可为 X），带出生日期合法性粗校验
_ID_RE = re.compile(r"(?<!\d)\d{6}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx](?!\d)")

_PATTERNS = {
    "email": (_EMAIL_RE, "<EMAIL>"),
    "phone": (_PHONE_RE, "<PHONE>"),
    "id_number": (_ID_RE, "<ID>"),
}


@dataclass
class PiiHit:
    kind: str
    # 命中片段的前后窗口（审计用），不保存命中原文
    context_before: str
    context_after: str


def scrub(text: str, context_window: int = 20) -> tuple[str, list[PiiHit]]:
    """替换文本中的 PII，返回 (清洗后文本, 命中记录列表)。"""
    hits: list[PiiHit] = []
    for kind, (pattern, placeholder) in _PATTERNS.items():
        def _replace(m: re.Match, _kind: str = kind) -> str:
            start, end = m.span()
            hits.append(PiiHit(
                kind=_kind,
                context_before=text[max(0, start - context_window):start],
                context_after=text[end:end + context_window],
            ))
            return placeholder
        text = pattern.sub(_replace, text)
    return text, hits


def scrub_batch(texts: list[str], keep_samples: int = 5) -> dict:
    """批量清洗，返回留存统计与少量命中上下文样本（审计用）。"""
    cleaned: list[str] = []
    kind_counts: dict[str, int] = {}
    docs_with_pii = 0
    samples: list[dict] = []
    for t in texts:
        out, hits = scrub(t)
        cleaned.append(out)
        if hits:
            docs_with_pii += 1
        for h in hits:
            kind_counts[h.kind] = kind_counts.get(h.kind, 0) + 1
            if len(samples) < keep_samples:
                samples.append({
                    "kind": h.kind,
                    "context_before": h.context_before,
                    "context_after": h.context_after,
                })
    n_in = len(texts)
    return {
        "n_in": n_in,
        "docs_with_pii": docs_with_pii,
        "doc_pii_rate": round(docs_with_pii / n_in, 4) if n_in else None,
        "kind_counts": kind_counts,
        "total_hits": sum(kind_counts.values()),
        "samples": samples,
        "cleaned_texts": cleaned,
    }
