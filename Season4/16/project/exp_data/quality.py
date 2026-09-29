"""quality.py —— 规则质量过滤：长度、符号率、重复行。

每条文本逐规则判定，返回是否通过及触发原因，便于统计"每道过滤删掉多少"。
规则是启发式的，阈值在 config.yaml 的 quality 段；这里只做判定，不改文本。

留存率口径：通过条数 / 输入条数（条级），以及通过字符数 / 输入字符数（字符级）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# 符号字符：非字母、非数字、非中日韩、非常见空白与基本标点
_SYMBOL_RE = re.compile(
    r"[^\w\u4e00-\u9fff\s.,!?;:'\"()\[\]{}\-–—…。，！？；：、（）【】“”‘’]",
    re.UNICODE,
)
_LINE_SPLIT_RE = re.compile(r"\r?\n")


@dataclass
class QualityConfig:
    min_chars: int = 200
    max_symbol_ratio: float = 0.30
    max_repeat_line_ratio: float = 0.30


@dataclass
class QualityResult:
    passed: bool
    reasons: list[str] = field(default_factory=list)


def symbol_ratio(text: str) -> float:
    """符号字符占比（分母为非空白字符数，避免长空白拉低比例）。"""
    non_ws = re.sub(r"\s", "", text)
    if not non_ws:
        return 0.0
    symbols = _SYMBOL_RE.findall(non_ws)
    return len(symbols) / len(non_ws)


def repeat_line_ratio(text: str) -> float:
    """重复行占比：出现多于一次的行 / 总行数（导航/模板的典型信号）。"""
    lines = [ln.strip() for ln in _LINE_SPLIT_RE.split(text) if ln.strip()]
    if len(lines) <= 1:
        return 0.0
    counts: dict[str, int] = {}
    for ln in lines:
        counts[ln] = counts.get(ln, 0) + 1
    repeated = sum(c for c in counts.values() if c > 1)
    return repeated / len(lines)


def check(text: str, cfg: QualityConfig) -> QualityResult:
    """逐规则判定一条文本，返回是否通过与全部触发原因。"""
    reasons: list[str] = []
    if len(text) < cfg.min_chars:
        reasons.append("too_short")
    if symbol_ratio(text) > cfg.max_symbol_ratio:
        reasons.append("high_symbol_ratio")
    if repeat_line_ratio(text) > cfg.max_repeat_line_ratio:
        reasons.append("high_repeat_line")
    return QualityResult(passed=not reasons, reasons=reasons)


def filter_batch(texts: list[str], cfg: QualityConfig) -> dict:
    """对一批文本跑质量过滤，返回条级/字符级留存率与各原因计数。"""
    n_in = len(texts)
    chars_in = sum(len(t) for t in texts)
    passed_texts: list[str] = []
    reason_counts: dict[str, int] = {}
    for t in texts:
        res = check(t, cfg)
        if res.passed:
            passed_texts.append(t)
        for r in res.reasons:
            reason_counts[r] = reason_counts.get(r, 0) + 1
    n_out = len(passed_texts)
    chars_out = sum(len(t) for t in passed_texts)
    return {
        "n_in": n_in,
        "n_out": n_out,
        "doc_retention": round(n_out / n_in, 4) if n_in else None,
        "chars_in": chars_in,
        "chars_out": chars_out,
        "char_retention": round(chars_out / chars_in, 4) if chars_in else None,
        "reason_counts": reason_counts,
        "passed_texts": passed_texts,
    }
