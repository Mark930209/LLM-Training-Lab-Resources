"""langid.py —— 语种识别与阈值过滤。

用 langdetect 对每条文本判语种，保留目标语种且置信度达标的文档。
langdetect 基于字符 n-gram 概率，对短文本与混合语种不稳，因此：
- 只对达到最小长度的文本判语种（过短的交给 quality 的长度规则处理）；
- 记录置信度，低于阈值的标为 uncertain，单独计数而不是直接丢。

真实数据管线里语种识别常用 fastText 的 lid.176 模型（更准、更快），
这里用纯 Python 的 langdetect 以免额外下载模型；两者口径差异在文章里说明。
"""

from __future__ import annotations

from dataclasses import dataclass

# langdetect 需要确定性时固定 seed
try:
    from langdetect import DetectorFactory, detect_langs  # type: ignore

    DetectorFactory.seed = 42
    _LANGDETECT_OK = True
except Exception:  # pragma: no cover - 依赖缺失时降级
    _LANGDETECT_OK = False


@dataclass
class LangIdConfig:
    target_langs: tuple[str, ...] = ("en", "zh-cn", "zh-tw")
    min_prob: float = 0.80
    min_chars_for_detect: int = 40


def detect(text: str, cfg: LangIdConfig) -> dict:
    """判一条文本的语种，返回 {lang, prob, status}。

    status: ok（目标语种且达标）/ other（非目标语种）/ uncertain（置信度不足）
            / too_short（太短不判）/ unavailable（依赖缺失）
    """
    if not _LANGDETECT_OK:
        return {"lang": None, "prob": None, "status": "unavailable"}
    if len(text) < cfg.min_chars_for_detect:
        return {"lang": None, "prob": None, "status": "too_short"}
    try:
        langs = detect_langs(text)
    except Exception:
        return {"lang": None, "prob": None, "status": "uncertain"}
    if not langs:
        return {"lang": None, "prob": None, "status": "uncertain"}
    top = langs[0]
    lang = str(top.lang).lower()
    prob = float(top.prob)
    if prob < cfg.min_prob:
        return {"lang": lang, "prob": round(prob, 4), "status": "uncertain"}
    if lang in cfg.target_langs:
        return {"lang": lang, "prob": round(prob, 4), "status": "ok"}
    return {"lang": lang, "prob": round(prob, 4), "status": "other"}


def filter_batch(texts: list[str], cfg: LangIdConfig) -> dict:
    """对一批文本判语种，返回保留集与语种/状态分布。"""
    kept: list[str] = []
    status_counts: dict[str, int] = {}
    lang_counts: dict[str, int] = {}
    for t in texts:
        res = detect(t, cfg)
        status_counts[res["status"]] = status_counts.get(res["status"], 0) + 1
        if res["lang"]:
            lang_counts[res["lang"]] = lang_counts.get(res["lang"], 0) + 1
        if res["status"] == "ok":
            kept.append(t)
    n_in = len(texts)
    return {
        "n_in": n_in,
        "n_kept": len(kept),
        "doc_retention": round(len(kept) / n_in, 4) if n_in else None,
        "status_counts": status_counts,
        "lang_counts": lang_counts,
        "kept_texts": kept,
    }
