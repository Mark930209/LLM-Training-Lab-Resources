"""extract.py —— 正文抽取对照：WET 直用 vs HTML trafilatura vs 朴素去标签。

三种方法回答同一个问题：从原始网页拿到"可训练文本"，不同抽取口径差多少。
- wet           : Common Crawl 官方 WET 已抽取文本（直用，代表"省事"路线）
- trafilatura   : 从 HTML 抽主正文，自带 boilerplate 去除（代表"认真抽"路线）
- naive_visible : 朴素去标签（未过滤基线，近似"什么都不做"，用来暴露噪声占比）

留存率 = 抽取后字符数 / 原始 HTML 字符数；噪声占比用朴素法与 trafilatura 的差衡量。
"""

from __future__ import annotations

import re

_TAG_RE = re.compile(r"<script[^>]*>.*?</script>|<style[^>]*>.*?</style>|<[^>]+>",
                     re.DOTALL | re.IGNORECASE)
_WS_RE = re.compile(r"[ \t\r\f\v]+")
_BLANK_RE = re.compile(r"\n{3,}")


def extract_wet(text: str) -> str:
    """WET 已是抽取后的纯文本，只做空白规整，不再去标签。"""
    return _normalize_ws(text)


def extract_trafilatura(html: str) -> str:
    """用 trafilatura 从 HTML 抽主正文（含导航/页脚/广告去除）。"""
    import trafilatura

    out = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=True,
        favor_recall=False,
    )
    return _normalize_ws(out or "")


def extract_naive_visible(html: str) -> str:
    """朴素去标签：删 script/style 与所有标签，保留可见文本。未做任何质量过滤。"""
    text = _TAG_RE.sub(" ", html)
    # 常见 HTML 实体还原（够采样对照用，不追求完备）
    for ent, ch in (("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
                    ("&quot;", '"'), ("&#39;", "'"), ("&nbsp;", " ")):
        text = text.replace(ent, ch)
    return _normalize_ws(text)


def _normalize_ws(text: str) -> str:
    text = _WS_RE.sub(" ", text)
    text = _BLANK_RE.sub("\n\n", text)
    return text.strip()


EXTRACTORS = {
    "wet": extract_wet,
    "trafilatura": extract_trafilatura,
    "naive_visible": extract_naive_visible,
}
