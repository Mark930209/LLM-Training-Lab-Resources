"""recipe.py —— 配方制备：领域配比 × 去重强度 × 质量过滤。

去重与质量过滤直接复用 16 篇 exp_data 的实现（同一套代码 = 同一个口径，
这正是"阶段口径对齐"的实践：17 篇不重写去重，而是把 16 篇的工具当库用）。

配方定义（config.yaml 的 recipes 段）：
  mix        : nl/code 两域文档的混合比例（按字符预算配比）
  dedup      : none / exact / minhash08 / minhash05（0.5 是故意的过激档）
  quality    : off / on（on 用 16 篇三规则，代码域单独放宽符号率阈值）

制备产物：每个配方一份 token 池文本 + 制备报告（各阶段留存率、注入统计、
配置哈希），全部落盘可复算。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

# 16 篇工具复用（PYTHONPATH 需同时含 15/16/17 三个 project 目录）
from exp_data import dedup as dedup16
from exp_data import quality as quality16

# 质量过滤阈值：自然语言域用 16 篇默认；代码域符号率高是常态，单独放宽
NL_QUALITY = quality16.QualityConfig(min_chars=200, max_symbol_ratio=0.30,
                                     max_repeat_line_ratio=0.30)
CODE_QUALITY = quality16.QualityConfig(min_chars=200, max_symbol_ratio=0.60,
                                       max_repeat_line_ratio=0.50)
# 失败案例二用：把 16 篇的自然语言阈值原样开到代码域
CODE_QUALITY_STRICT = NL_QUALITY


@dataclass
class RecipeSpec:
    name: str
    nl_ratio: float          # 自然语言域字符占比（code = 1 - nl_ratio）
    dedup: str               # none | exact | minhash08 | minhash05
    quality: bool            # 是否开质量过滤
    char_budget: int         # 配方总字符预算


def _dedup_docs(docs: list[str], mode: str) -> tuple[list[str], dict]:
    """按去重档位处理文档池，返回 (留存文档, 统计)。"""
    if mode == "none":
        return docs, {"mode": "none", "n_in": len(docs), "n_out": len(docs),
                      "removed": 0}
    if mode == "exact":
        res = dedup16.exact_dedup(docs)
        kept = res.pop("kept_texts")
        return kept, {"mode": "exact", **res}
    threshold = {"minhash08": 0.8, "minhash05": 0.5}[mode]
    cfg = dedup16.MinHashConfig(num_perm=128, threshold=threshold, shingle_k=5)
    res = dedup16.minhash_dedup(docs, cfg)
    kept = res.pop("kept_texts")
    return kept, {"mode": mode, "threshold": threshold,
                  **{k: v for k, v in res.items()}}


def prepare_recipe(spec: RecipeSpec, nl_docs: list[str],
                   code_docs: list[str]) -> dict:
    """制备一个配方：过滤 → 去重 → 按预算配比混合。

    返回 {text, report}；report 含各阶段留存率与配置哈希，进制备档案。
    """
    report: dict = {"recipe": spec.name, "nl_ratio": spec.nl_ratio,
                    "dedup": spec.dedup, "quality": spec.quality,
                    "char_budget": spec.char_budget, "stages": {}}

    # 阶段 1：质量过滤（分域阈值）
    if spec.quality:
        nl_res = quality16.filter_batch(nl_docs, NL_QUALITY)
        code_res = quality16.filter_batch(code_docs, CODE_QUALITY)
        nl_kept, code_kept = nl_res["passed_texts"], code_res["passed_texts"]
        report["stages"]["quality"] = {
            "nl": {k: v for k, v in nl_res.items() if k != "passed_texts"},
            "code": {k: v for k, v in code_res.items() if k != "passed_texts"},
        }
    else:
        nl_kept, code_kept = list(nl_docs), list(code_docs)
        report["stages"]["quality"] = {"skipped": True}

    # 阶段 2：去重（分域执行，避免跨域误判）
    nl_dedup, nl_stat = _dedup_docs(nl_kept, spec.dedup)
    code_dedup, code_stat = _dedup_docs(code_kept, spec.dedup)
    report["stages"]["dedup"] = {"nl": nl_stat, "code": code_stat}

    # 阶段 3：按字符预算配比混合
    nl_budget = int(spec.char_budget * spec.nl_ratio)
    code_budget = spec.char_budget - nl_budget

    def take_budget(docs: list[str], budget: int) -> list[str]:
        out, total = [], 0
        for d in docs:
            if total >= budget:
                break
            out.append(d)
            total += len(d)
        return out

    nl_part = take_budget(nl_dedup, nl_budget)
    code_part = take_budget(code_dedup, code_budget)
    mixed = nl_part + code_part
    text = "\n".join(mixed)
    report["stages"]["mix"] = {
        "nl_docs": len(nl_part), "nl_chars": sum(len(d) for d in nl_part),
        "code_docs": len(code_part), "code_chars": sum(len(d) for d in code_part),
        "total_chars": len(text),
        "actual_nl_ratio": round(sum(len(d) for d in nl_part) / max(1, len(text)), 4),
    }
    report["text_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return {"text": text, "report": report}


def diversity_metrics(text: str) -> dict:
    """多样性代理指标：unique n-gram 比例（字符级 4-gram 与 8-gram）。"""
    def uniq_ratio(n: int) -> float:
        grams = [text[i:i + n] for i in range(len(text) - n + 1)]
        if not grams:
            return 0.0
        return round(len(set(grams)) / len(grams), 4)
    return {"unique_4gram_ratio": uniq_ratio(4),
            "unique_8gram_ratio": uniq_ratio(8),
            "chars": len(text)}


def save_report(path: str, report: dict) -> None:
    from pathlib import Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                 encoding="utf-8")
