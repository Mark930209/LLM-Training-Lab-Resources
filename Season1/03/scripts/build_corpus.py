#!/usr/bin/env python3
"""build_corpus.py —— 西游记语料库构建流水线。

从维基文库（公版，吴承恩《西遊記》百回本）下载 100 回 wikitext，
清洗成训练可用的纯文本，并产出结构化索引。这是"构建自己的小语料库"
的完整示范：下载 → 清洗 → 繁转简 → 切分 → 索引 → 统计。

产出（默认写到 --out-dir）：
    xiyouji_raw/          每回的原始 wikitext（可追溯，供复查清洗）
    xiyouji.txt           百回清洗合并后的训练语料（生成模型用）
    chapters/NNN.txt      每回单独文件（检索问答的检索单元）
    chapters.json         结构化索引：回数 / 回目 / 简体正文长度（整体信息）
    corpus_stats.json     语料统计：总字符 / 词表大小 / 每回长度分布

用法（WSL2 内，工程根目录）：
    python scripts/build_corpus.py --out-dir exp_superminigpt/data
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://zh.wikisource.org/w/api.php"
MAIN_PAGE = "西遊記"
UA = "LLMTrainingLab-corpus-builder/1.0 (educational; contact: lab)"


def _api_get(params: dict, retries: int = 5) -> dict:
    qs = urllib.parse.urlencode({**params, "format": "json"})
    req = urllib.request.Request(f"{API}?{qs}", headers={"User-Agent": UA})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            # 429/503 限流：按 Retry-After 或指数退避等待后重试
            if e.code in (429, 503) and attempt < retries - 1:
                wait = int(e.headers.get("Retry-After", 0) or (2 ** attempt * 3))
                time.sleep(min(wait, 30))
                continue
            raise
    raise RuntimeError("重试耗尽")


def fetch_wikitext(page: str) -> str:
    d = _api_get({"action": "parse", "page": page, "prop": "wikitext",
                  "redirects": "1"})
    if "parse" not in d:
        raise RuntimeError(f"API 未返回 parse（页 {page}）: "
                           f"{json.dumps(d, ensure_ascii=False)[:300]}")
    return d["parse"]["wikitext"]["*"]


def list_chapter_links() -> list[str]:
    """从主页面解析出 100 个回子页标题（形如 西遊記/第001回）。"""
    wikitext = fetch_wikitext(MAIN_PAGE)
    subs = re.findall(r"\[\[(/第\d{3}回)", wikitext)
    return [f"{MAIN_PAGE}{s}" for s in dict.fromkeys(subs)]


# ---------- 清洗 ----------

def clean_wikitext(wt: str) -> str:
    """wikitext → 纯正文。顺序敏感，先处理带内容的模板再删空模板。"""
    # 1) {{另|主形|异形}} 取主形（注音/校勘用）
    wt = re.sub(r"\{\{另\|([^|}]+)\|[^}]*\}\}", r"\1", wt)
    # 2) 删除其余模板 {{...}}（header/footer/检索/Textquality 等，含跨行）
    wt = re.sub(r"\{\{[^{}]*\}\}", "", wt, flags=re.S)
    # 3) [[A|B]] 取 B，[[A]] 取 A（内链）
    wt = re.sub(r"\[\[[^|\]]+\|([^\]]+)\]\]", r"\1", wt)
    wt = re.sub(r"\[\[([^\]]+)\]\]", r"\1", wt)
    # 4) 去 HTML 标签（<br> 转换行）
    wt = wt.replace("<br>", "\n")
    wt = re.sub(r"<[^>]+>", "", wt)
    # 5) 行首 : 是维基引用缩进，去掉标记保留文字
    wt = re.sub(r"^:+", "", wt, flags=re.M)
    # 6) 去多余空白行、行首尾空格；保留段落单换行
    lines = [ln.strip() for ln in wt.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


def extract_title(wt: str) -> str:
    m = re.search(r"\|\s*section\s*=\s*(.+)", wt)
    if not m:
        return ""
    return re.sub(r"<br>", " ", m.group(1)).strip()


# ---------- 繁转简 ----------

_CONVERTER = None
def to_simplified(text: str) -> str:
    global _CONVERTER
    if _CONVERTER is None:
        from opencc import OpenCC
        _CONVERTER = OpenCC("t2s")
    return _CONVERTER.convert(text)


def build(out_dir: Path, delay: float = 0.3, limit: int = 0) -> dict:
    raw_dir = out_dir / "xiyouji_raw"
    chap_dir = out_dir / "chapters"
    raw_dir.mkdir(parents=True, exist_ok=True)
    chap_dir.mkdir(parents=True, exist_ok=True)

    links = list_chapter_links()
    if limit:
        links = links[:limit]
    print(f"发现 {len(links)} 回，开始下载与清洗…")

    chapters = []
    all_parts = []
    for i, page in enumerate(links, 1):
        num = re.search(r"第(\d{3})回", page).group(1)
        raw_path = raw_dir / f"{num}.wikitext"
        if raw_path.exists():
            wt = raw_path.read_text(encoding="utf-8")
        else:
            wt = fetch_wikitext(page)
            raw_path.write_text(wt, encoding="utf-8")
            time.sleep(delay)

        title = to_simplified(clean_wikitext(extract_title(wt)))  # 标题同样清洗+转简体
        body = to_simplified(clean_wikitext(wt))
        (chap_dir / f"{num}.txt").write_text(f"{title}\n\n{body}\n", encoding="utf-8")
        chapters.append({"no": int(num), "title": title, "chars": len(body)})
        all_parts.append(f"{title}\n\n{body}")
        if i % 10 == 0:
            print(f"  已处理 {i}/{len(links)} 回")

    full = "\n\n".join(all_parts) + "\n"
    (out_dir / "xiyouji.txt").write_text(full, encoding="utf-8")
    (out_dir / "chapters.json").write_text(
        json.dumps(chapters, ensure_ascii=False, indent=2), encoding="utf-8")

    vocab = sorted(set(full))
    stats = {
        "chapters": len(chapters),
        "total_chars": len(full),
        "vocab_size": len(vocab),
        "min_chapter_chars": min(c["chars"] for c in chapters),
        "max_chapter_chars": max(c["chars"] for c in chapters),
        "avg_chapter_chars": round(sum(c["chars"] for c in chapters) / len(chapters)),
    }
    (out_dir / "corpus_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print("语料统计:", json.dumps(stats, ensure_ascii=False))
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="exp_superminigpt/data")
    ap.add_argument("--delay", type=float, default=0.3)
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 回（调试用）")
    args = ap.parse_args()
    build(Path(args.out_dir), args.delay, args.limit)
