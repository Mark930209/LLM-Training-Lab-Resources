#!/usr/bin/env python3
"""retrieval_qa.py —— 检索式问答（对照生成式模型的"会编"）。

思路：把 100 回原文切成检索单元（每回一段），用 BM25 找与问题最相关的回，
返回原文片段作为答案。它不会"生成"，只会"找到并引用"，因此事实准确——
代价是只能回答原文里明说的内容，答不了需要概括推理的问题。

这正是与生成模型的对照：
    生成模型：流畅，但会编（问"第五十难是什么"可能一本正经胡说）
    检索系统：准确，但只会摘录，不会总结，换个问法找不到就答不上

BM25 纯 Python 实现，无第三方依赖，便于读者读懂每个公式。中文按字符
bigram 切 token（无分词依赖），对检索够用。

用法：
    python scripts/retrieval_qa.py --chapters-dir exp_superminigpt/data/chapters \
        --query "孙悟空是怎么出生的"
    python scripts/retrieval_qa.py --chapters-dir ... --benchmark qa_set.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path


def tokenize(text: str) -> list[str]:
    """中文按字符 bigram + 保留 ASCII 词。无分词依赖，检索够用。"""
    text = re.sub(r"\s+", "", text)
    tokens = re.findall(r"[a-zA-Z]+", text)
    han = re.findall(r"[\u4e00-\u9fff]", text)
    tokens += [a + b for a, b in zip(han, han[1:])]   # 相邻字组成 bigram
    return tokens


class BM25:
    """标准 BM25（k1=1.5, b=0.75）。"""

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = docs
        self.doc_len = [len(d) for d in docs]
        self.avgdl = sum(self.doc_len) / len(docs) if docs else 0
        self.tf = [Counter(d) for d in docs]
        self.df: Counter = Counter()
        for d in docs:
            for term in set(d):
                self.df[term] += 1
        self.N = len(docs)

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1 + (self.N - df + 0.5) / (df + 0.5))

    def score(self, query_terms: list[str]) -> list[float]:
        scores = [0.0] * self.N
        for term in query_terms:
            idf = self._idf(term)
            for i in range(self.N):
                f = self.tf[i].get(term, 0)
                if f == 0:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                scores[i] += idf * f * (self.k1 + 1) / denom
        return scores


CN_NUM = {"零": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
          "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def cn_to_int(s: str) -> int:
    """中文数字（一~九十九）转 int。回目用中文数字，基准集用阿拉伯数字。"""
    s = s.replace("第", "").replace("回", "").strip()
    if s.isdigit():
        return int(s)
    if "十" not in s:
        return CN_NUM.get(s, -1)
    if s == "十":
        return 10
    tens, _, ones = s.partition("十")
    return CN_NUM.get(tens, 1) * 10 + (CN_NUM.get(s2, 0) if (s2 := s[len(tens) + 1:]) else 0)


def chapter_no(title: str) -> int:
    m = re.search(r"第[一二三四五六七八九十百零]+回", title)
    return cn_to_int(m.group(0)) if m else -1


def load_chapters(chapters_dir: Path) -> list[dict]:
    out = []
    for p in sorted(chapters_dir.glob("*.txt")):
        text = p.read_text(encoding="utf-8")
        title = text.splitlines()[0] if text else p.name
        out.append({"file": p.name, "title": title, "text": text,
                    "no": chapter_no(title)})
    return out


def answer(bm25: BM25, chapters: list[dict], query: str, top_k: int = 3,
           excerpt_chars: int = 200) -> list[dict]:
    scores = bm25.score(tokenize(query))
    ranked = sorted(range(len(chapters)), key=lambda i: scores[i], reverse=True)[:top_k]
    results = []
    for rank, i in enumerate(ranked, 1):
        ch = chapters[i]
        # 抽取问题关键词在原文中最密集的一段作为答案摘录
        snippet = best_excerpt(ch["text"], query, excerpt_chars)
        results.append({"rank": rank, "chapter": ch["title"], "no": ch.get("no", -1),
                        "score": round(scores[i], 2), "excerpt": snippet})
    return results


def best_excerpt(text: str, query: str, n: int) -> str:
    """在章节里找与 query 词重叠最高的窗口。"""
    qterms = set(tokenize(query))
    lines = text.splitlines()
    best, best_overlap = "", -1
    window = "\n".join(lines[:1])
    joined = "".join(lines)
    step = max(1, n // 3)
    for start in range(0, max(1, len(joined) - n), step):
        seg = joined[start:start + n]
        overlap = len(qterms & set(tokenize(seg)))
        if overlap > best_overlap:
            best_overlap, best = overlap, seg
    return best.strip()[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chapters-dir", default="exp_superminigpt/data/chapters")
    ap.add_argument("--query", default=None)
    ap.add_argument("--benchmark", default=None, help="问题集 JSON：[{id,question,expected}]")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    chapters = load_chapters(Path(args.chapters_dir))
    bm25 = BM25([tokenize(c["text"]) for c in chapters])
    print(f"检索库就绪：{len(chapters)} 回")

    if args.query:
        for r in answer(bm25, chapters, args.query, args.top_k):
            print(f"\n#{r['rank']}  {r['chapter']}  (score {r['score']})")
            print(f"    {r['excerpt']}")

    if args.benchmark:
        qs = json.loads(Path(args.benchmark).read_text(encoding="utf-8"))
        results = []
        for item in qs:
            hits = answer(bm25, chapters, item["question"], args.top_k)
            # 命中判定：期望回号（阿拉伯数字）与检索回号（中文数字转换）比对
            expected = cn_to_int(item.get("expected_chapter", ""))
            hit = any(h.get("no") == expected for h in hits)
            results.append({**item, "top1": hits[0]["chapter"],
                            "topk_titles": [h["chapter"] for h in hits],
                            "hit": hit, "excerpt": hits[0]["excerpt"]})
            print(f"[{'HIT ' if hit else 'MISS'}] {item['question']} → {hits[0]['chapter']}")
        out = {"total": len(results), "hit_at_k": sum(r["hit"] for r in results),
               "results": results}
        print(f"\n检索命中率 hit@{args.top_k}: {out['hit_at_k']}/{out['total']}")
        if args.output:
            Path(args.output).write_text(
                json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
