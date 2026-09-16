#!/usr/bin/env python3
"""analyze_text_quality.py —— 对采样文本做客观指标统计。

用于回答一个常见疑问："看上去小模型和大模型写得差不多"。
肉眼看短文本确实难分，因为差别不在"像不像古文"，而在更长程的结构
（回目、诗赞、说书人套语）和跨书适应力。本脚本把能算的都算出来，
不能算的明确标注"需人工判读"，不硬凑趋势。

指标说明：
    - 记忆率：文本里 n-gram 有多少能在训练语料中原样找到。值越高说明
      模型在背原文，值越低说明它在生成。
    - 结构标记：章回体的格式特征（第 N 回、且听下回分解、正是：、
      有诗为证）。这不是词句能力，是文体格式，必须见过足量同类文本才能归纳。
    - 病句信号：连续重复字、重复词、连续标点。

用法：
    python scripts/analyze_text_quality.py --samples ../../results/Season1/04/samples.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 结构标记：章回体特有的格式特征
STRUCTURE_MARKERS = [
    ("回目（第 N 回）", r"第[一二三四五六七八九十百千0-9]+回"),
    ("且听下回分解", r"下回分解"),
    ("诗赞引导（正是：）", r"正是[：:]"),
    ("诗赞引导（有诗为证）", r"有诗为证"),
    ("说书人套语", r"话表|话分两头|却说|且说"),
]

# 病句信号
BAD_MARKERS = [
    ("连续重复字（≥3）", r"(.)\1{2,}"),
    ("重复词（2 字）", r"(.{2})\1"),
    ("连续标点", r"[。，、！？：；]{2,}"),
]


def norm(text: str) -> str:
    """去掉空白与换行，保留标点。"""
    return re.sub(r"\s+", "", text)


def memorize_rate(text: str, corpus: str, n: int = 8) -> float:
    """n-gram 记忆率：文本的 n-gram 有多大比例能在语料里原样找到。"""
    t, c = norm(text), norm(corpus)
    if len(t) < n:
        return 0.0
    pool = {c[i:i + n] for i in range(len(c) - n)}
    windows = [t[i:i + n] for i in range(len(t) - n + 1)]
    return sum(1 for w in windows if w in pool) / len(windows)


def count_markers(text: str, markers) -> int:
    return sum(len(re.findall(pat, text)) for _, pat in markers)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--corpus-dir", default=None,
                    help="语料目录（默认 DevResources/Season1/04/project/exp_scale/data）")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[3]
    corpus_dir = Path(args.corpus_dir) if args.corpus_dir else \
        Path(__file__).resolve().parent.parent / "project/exp_scale/data"
    corpora = {}
    for name, key in (("corpus_small", "small"), ("corpus_large", "large")):
        p = corpus_dir / f"{name}.txt"
        if p.exists():
            corpora[key] = p.read_text(encoding="utf-8")

    data = json.loads(Path(args.samples).read_text(encoding="utf-8"))
    print(f"{'模型':<16}{'参数M':>7}{'困惑度':>8}{'记忆率':>8}"
          f"{'结构标记':>9}{'病句信号':>9}")
    print("-" * 60)
    rows = []
    for m in data["models"]:
        corpus = corpora.get(m["corpus"], "")
        mems, structs, bads = [], [], []
        for o in m["outputs"]:
            t = o["text"]
            mems.append(memorize_rate(t, corpus))
            structs.append(count_markers(t, STRUCTURE_MARKERS))
            bads.append(count_markers(t, BAD_MARKERS))
        row = {
            "label": m["label"],
            "params_million": m["params_million"],
            "perplexity": round(math.exp(m["best_val"]), 1),
            "memorize_rate": round(sum(mems) / len(mems), 3),
            "structure_markers": sum(structs),
            "bad_markers": sum(bads),
        }
        rows.append(row)
        print(f"{row['label']:<16}{row['params_million']:>7.2f}"
              f"{row['perplexity']:>8.1f}{row['memorize_rate']:>8.3f}"
              f"{row['structure_markers']:>9}{row['bad_markers']:>9}")

    note = ("\n注：结构标记与记忆率是客观统计；'读起来顺不顺'仍需人工判读，\n"
            "     短文本（180 token）四档差别很小，要看出差距需要更长的生成。")
    print(note)
    out = {"models": rows, "note": note.strip()}
    out_path = Path(args.samples).with_name("text_quality.json")
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"已写入 {out_path}")


if __name__ == "__main__":
    main()