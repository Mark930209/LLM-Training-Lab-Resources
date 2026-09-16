#!/usr/bin/env python3
"""build_corpus.py —— 四大名著语料库构建流水线（04 篇）。

在 03 篇单书（西游记）构建器基础上泛化到多书：同一套"下载 → 清洗 →
繁转简 → 切分 → 索引 → 统计"流程，配置驱动地处理四大名著，
产出可对比的"小语料 vs 大语料"两档训练语料。

为什么做这件事：03 篇用 73.8 万字符训练 3.53M 参数，1400 步就过拟合。
04 篇要把模型放大到千万级，语料必须同步放大——本脚本就是"数据规模"
这一课的可复现工具。

为什么选四大名著：四部都是明清白话章回小说，与 03 篇的西游记同源，
风格统一。小语料（西游记）是大语料（四大名著）的真子集，对照时
只变"数据量"这一个变量，归因干净。

产出（默认写到 --out-dir）：
    raw/<book>/NNN.wikitext   每回原始 wikitext（可追溯）
    books/<book>.txt          单书清洗合并文本
    corpus_small.txt          小语料（西游记，与 03 篇对齐）
    corpus_large.txt          大语料（四大名著合并）
    chapters/<book>/NNN.txt   每回单独文件（检索单元）
    chapters.json             结构化索引：书/回/标题/字数
    corpus_stats.json         语料统计：各书字数、总字数、词表大小

用法（WSL2 内，工程根目录）：
    python scripts/build_corpus.py --out-dir exp_scale/data
    python scripts/build_corpus.py --out-dir exp_scale/data --books xiyouji
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://zh.wikisource.org/w/api.php"
UA = "LLMTrainingLab-corpus-builder/1.0 (educational; contact: lab)"

# 书目配置：主页面 → 子页前缀过滤 → 输出名
# 结构差异来自维基文库各书的编排方式，探测后固化（见 .float-writing/04-scale/）
# 四大名著：明清白话章回小说，与 03 篇西游记同源，风格统一
BOOKS: dict[str, dict] = {
    "xiyouji": {
        "page": "西遊記",
        "pattern": r"^第\d{3}回$",       # 第001回 ~ 第100回
        "title": "西游记",
    },
    "hongloumeng": {
        "page": "紅樓夢",
        "pattern": r"^第\d{3}回$",       # 第001回 ~ 第120回
        "title": "红楼梦",
    },
    "sanguoyanyi": {
        "page": "三國演義",
        "pattern": r"^第\d{3}回$",       # 过滤 序/凡例/读法
        "title": "三国演义",
    },
    "shuihuzhuan": {
        "page": "水滸傳 (120回本)",       # 主页面是版本消歧页，取 120 回本
        "pattern": r"^第\d{3}回$",       # 过滤 序/引首
        "title": "水浒传",
    },
}

# 小语料 = 与 03 篇对齐的单书；大语料 = 四大名著
SMALL_BOOKS = ["xiyouji"]


def _api_get(params: dict, retries: int = 6) -> dict:
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


def list_subpages(page: str, pattern: str) -> list[str]:
    """从主页面解析出符合 pattern 的子页标题（形如 西遊記/第001回）。"""
    wikitext = fetch_wikitext(page)
    subs = re.findall(r"\[\[/([^\]|]+)", wikitext)
    keep = [s for s in dict.fromkeys(subs) if re.match(pattern, s.strip())]
    return [f"{page}/{s.strip()}" for s in keep]


# ---------- 清洗（与 03 篇同一套逻辑，保证两篇语料口径一致） ----------

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


def build_book(key: str, out_dir: Path, delay: float = 0.3,
               limit: int = 0) -> dict:
    """构建单书：下载 → 清洗 → 繁转简 → 切分 → 落盘。返回该书统计。"""
    spec = BOOKS[key]
    raw_dir = out_dir / "raw" / key
    chap_dir = out_dir / "chapters" / key
    raw_dir.mkdir(parents=True, exist_ok=True)
    chap_dir.mkdir(parents=True, exist_ok=True)

    links = list_subpages(spec["page"], spec["pattern"])
    if limit:
        links = links[:limit]
    print(f"[{key}] {spec['title']}: 发现 {len(links)} 个回/卷，开始下载…",
          flush=True)

    chapters = []
    parts = []
    for i, page in enumerate(links, 1):
        num = page.rsplit("/", 1)[-1]
        raw_path = raw_dir / f"{num}.wikitext"
        if raw_path.exists():
            wt = raw_path.read_text(encoding="utf-8")
        else:
            wt = fetch_wikitext(page)
            raw_path.write_text(wt, encoding="utf-8")
            time.sleep(delay)

        title = to_simplified(clean_wikitext(extract_title(wt)))
        body = to_simplified(clean_wikitext(wt))
        (chap_dir / f"{num}.txt").write_text(f"{title}\n\n{body}\n",
                                             encoding="utf-8")
        chapters.append({"book": key, "no": num, "title": title,
                         "chars": len(body)})
        parts.append(f"{title}\n\n{body}")
        if i % 20 == 0:
            print(f"  [{key}] 已处理 {i}/{len(links)}", flush=True)

    full = "\n\n".join(parts) + "\n"
    (out_dir / "books" / f"{key}.txt").parent.mkdir(parents=True, exist_ok=True)
    (out_dir / "books" / f"{key}.txt").write_text(full, encoding="utf-8")

    stats = {
        "book": key,
        "title": spec["title"],
        "chapters": len(chapters),
        "total_chars": len(full),
        "vocab_size": len(set(full)),
    }
    print(f"[{key}] 完成: {stats['chapters']} 回/卷, "
          f"{stats['total_chars']:,} 字符", flush=True)
    return {"stats": stats, "chapters": chapters, "text": full}


def build(out_dir: Path, books: list[str], delay: float = 0.3,
          limit: int = 0) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_chapters: list[dict] = []
    book_texts: dict[str, str] = {}
    book_stats: list[dict] = []

    for key in books:
        r = build_book(key, out_dir, delay=delay, limit=limit)
        all_chapters.extend(r["chapters"])
        book_texts[key] = r["text"]
        book_stats.append(r["stats"])

    # 小语料：与 03 篇对齐的单书
    small = "\n\n".join(book_texts[k] for k in SMALL_BOOKS if k in book_texts)
    (out_dir / "corpus_small.txt").write_text(small + "\n", encoding="utf-8")

    # 大语料：全部书目合并
    large = "\n\n".join(book_texts[k] for k in books)
    (out_dir / "corpus_large.txt").write_text(large + "\n", encoding="utf-8")

    (out_dir / "chapters.json").write_text(
        json.dumps(all_chapters, ensure_ascii=False, indent=2),
        encoding="utf-8")

    stats = {
        "books": book_stats,
        "small_chars": len(small),
        "large_chars": len(large),
        "small_vocab": len(set(small)),
        "large_vocab": len(set(large)),
        "total_chapters": len(all_chapters),
    }
    (out_dir / "corpus_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== 语料统计 ===")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="exp_scale/data")
    ap.add_argument("--books", default="all",
                    help="逗号分隔的书目 key，或 all")
    ap.add_argument("--delay", type=float, default=0.3)
    ap.add_argument("--limit", type=int, default=0, help="每书只处理前 N 回（调试）")
    args = ap.parse_args()

    keys = list(BOOKS) if args.books == "all" else args.books.split(",")
    for k in keys:
        if k not in BOOKS:
            raise SystemExit(f"未知书目: {k}（可选: {', '.join(BOOKS)}）")
    build(Path(args.out_dir), keys, args.delay, args.limit)
