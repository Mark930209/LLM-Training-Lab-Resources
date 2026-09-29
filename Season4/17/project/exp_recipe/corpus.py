"""corpus.py —— 17 篇领域语料加载、评测集切分与重复注入。

两个领域（全部本地，不下载）：
  - nl   : 四大名著语料（04 篇产物，中文自然语言域）
  - code : Python 3.12 标准库源码（代码域）

评测集切分口径（核心判断"阶段口径对齐"的第一道）：
  - 每个域切出固定评测集，永不进入任何配方、永不参与重复注入；
  - 切分用确定性规则（位置/排序），不依赖随机数，跨 run 可复现。

重复注入：
  本地语料没有天然近重复，按已知比例向训练池注入精确副本与扰动副本
  （模拟转载与改稿），让去重强度有真实作用对象。注入比例写入制备报告。
"""

from __future__ import annotations

import hashlib
import random
import re
import sysconfig
from pathlib import Path

# 评测集切分参数（确定性，不用随机）
NL_EVAL_CHARS = 200_000        # 小说域评测集：中段固定 200K 字符
CODE_EVAL_EVERY = 20           # 代码域评测集：按文件名排序每 20 个取 1 个
CODE_DOC_MIN_CHARS = 800       # 过短的源文件不进语料（注释头/空 init）

# 重复注入参数
DUP_EXACT_RATIO = 0.10         # 精确副本占训练池文档数的 10%
DUP_NEAR_RATIO = 0.10          # 扰动副本占 10%
NEAR_MUTATE_RATE = 0.02        # 扰动副本：2% 字符替换
NEAR_SEED = 20260924


def load_nl_corpus(corpus_path: str | Path) -> str:
    """加载四大名著语料全文。"""
    return Path(corpus_path).read_text(encoding="utf-8")


def load_code_docs() -> list[tuple[str, str]]:
    """收集 Python 标准库源文件，返回 [(文件名, 内容)]，按文件名排序。"""
    stdlib = Path(sysconfig.get_paths()["stdlib"])
    docs: list[tuple[str, str]] = []
    for path in sorted(stdlib.rglob("*.py")):
        rel = path.relative_to(stdlib).as_posix()
        # 排除测试与示例目录
        if re.search(r"(^|/)(test|tests|idle_test|lib2to3)/", rel):
            continue
        if "/test" in rel or rel.startswith("test"):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if len(text) >= CODE_DOC_MIN_CHARS:
            docs.append((rel, text))
    return docs


def split_domains(nl_text: str, code_docs: list[tuple[str, str]]) -> dict:
    """把两个域各切成 train / eval 两部分，切分规则确定性可复现。"""
    # nl：中段固定区间做评测集，其余做训练池
    mid = len(nl_text) // 2
    nl_eval = nl_text[mid - NL_EVAL_CHARS // 2: mid + NL_EVAL_CHARS // 2]
    nl_train = nl_text[:mid - NL_EVAL_CHARS // 2] + nl_text[mid + NL_EVAL_CHARS // 2:]

    # code：每 20 个文件取 1 个做评测集
    code_eval = [d for i, d in enumerate(code_docs) if i % CODE_EVAL_EVERY == 0]
    code_train = [d for i, d in enumerate(code_docs) if i % CODE_EVAL_EVERY != 0]

    return {
        "nl_train": nl_train,
        "nl_eval": nl_eval,
        "code_train": code_train,   # [(name, text)]
        "code_eval": code_eval,
    }


def split_nl_train_docs(nl_train: str, chunk_chars: int = 4000) -> list[str]:
    """把小说训练池按段落边界切成文档块，便于按文档去重与配比。"""
    paras = nl_train.split("\n")
    docs: list[str] = []
    buf: list[str] = []
    size = 0
    for p in paras:
        buf.append(p)
        size += len(p) + 1
        if size >= chunk_chars:
            docs.append("\n".join(buf))
            buf, size = [], 0
    if buf:
        docs.append("\n".join(buf))
    return docs


def _mutate(text: str, rng: random.Random, rate: float) -> str:
    """扰动副本：按 rate 比例把字符替换为同域随机字符（模拟改稿转载）。"""
    chars = list(text)
    n_mut = max(1, int(len(chars) * rate))
    pool = "的了是在和有我人这中大为上个国不以到说时要就出会也你对开年"
    for _ in range(n_mut):
        i = rng.randrange(len(chars))
        chars[i] = pool[rng.randrange(len(pool))]
    return "".join(chars)


def inject_duplicates(docs: list[str], exact_ratio: float = DUP_EXACT_RATIO,
                      near_ratio: float = DUP_NEAR_RATIO,
                      seed: int = NEAR_SEED) -> tuple[list[str], dict]:
    """向文档池注入精确副本与扰动副本，返回 (新池, 注入统计)。

    注入是确定性的（固定 seed），注入数量与比例写入统计，进制备报告。
    """
    rng = random.Random(seed)
    n = len(docs)
    n_exact = int(n * exact_ratio)
    n_near = int(n * near_ratio)
    exact_idx = [rng.randrange(n) for _ in range(n_exact)]
    near_idx = [rng.randrange(n) for _ in range(n_near)]
    injected = [docs[i] for i in exact_idx]
    injected += [_mutate(docs[i], rng, NEAR_MUTATE_RATE) for i in near_idx]
    pool = docs + injected
    rng.shuffle(pool)
    stats = {
        "n_original": n,
        "n_exact_copies": n_exact,
        "n_near_copies": n_near,
        "n_total": len(pool),
        "injected_ratio": round((n_exact + n_near) / n, 4),
        "near_mutate_rate": NEAR_MUTATE_RATE,
        "seed": seed,
    }
    return pool, stats


def eval_fingerprint(domains: dict) -> dict:
    """评测集指纹：内容 SHA-256，保证所有 run 评的是同一份数据。"""
    def sha(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return {
        "nl_eval_sha256": sha(domains["nl_eval"]),
        "nl_eval_chars": len(domains["nl_eval"]),
        "code_eval_sha256": sha("\n".join(t for _, t in domains["code_eval"])),
        "code_eval_docs": len(domains["code_eval"]),
    }
