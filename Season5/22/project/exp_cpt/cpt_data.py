"""exp_cpt.data —— CPT 语料/评测集/领域词表构建。

复用 17 篇 exp_recipe.corpus 的语料加载与确定性切分（train/eval 指纹跨篇一致），
在 Qwen tokenizer 上产出：训练流（领域+replay 混合）、固定文本评测集、
扩词表候选（代码高频被拆散的标识符）。

评测集口径：按固定字符切片定义（不按 token 截断），保证跨 tokenizer 可比——
报告 ppl 之外同时报 nats/char（tokenization 不变），扩词表前后才是有效对比。
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

# 复用 17 篇数据底座（exp_recipe.corpus），路径相对本文件解析
_DEV_RESOURCES = Path(__file__).resolve().parents[4]
_RECIPE_PROJECT = _DEV_RESOURCES / "Season4" / "17" / "project"
if str(_RECIPE_PROJECT) not in sys.path:
    sys.path.insert(0, str(_RECIPE_PROJECT))

from exp_recipe.corpus import (  # noqa: E402
    eval_fingerprint,
    load_code_docs,
    load_nl_corpus,
    split_domains,
)

# 训练流取样：从训练池均匀抽文档拼接（避免只取前缀偏向前半部分文件）
STREAM_DOC_CHARS = 4000
IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]{3,23}")


def load_domains(nl_path: str | Path) -> dict:
    """加载两大域并按 17 篇规则切 train/eval（确定性）。"""
    nl_text = load_nl_corpus(nl_path)
    code_docs = load_code_docs()
    return split_domains(nl_text, code_docs)


def fingerprint(domains: dict) -> dict:
    """评测集指纹（复用 17 篇 eval_fingerprint，跨篇可核对）。"""
    return eval_fingerprint(domains)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def split_to_docs(text: str, chunk_chars: int = STREAM_DOC_CHARS) -> list[str]:
    """把长文本按段落边界切成文档块（与 17 篇 split_nl_train_docs 同法）。"""
    docs: list[str] = []
    buf: list[str] = []
    size = 0
    for p in text.split("\n"):
        buf.append(p)
        size += len(p) + 1
        if size >= chunk_chars:
            docs.append("\n".join(buf))
            buf, size = [], 0
    if buf:
        docs.append("\n".join(buf))
    return docs


def sample_docs(docs: list[str], total_chars: int) -> list[str]:
    """从文档列表均匀取样到约 total_chars 字符（确定性，覆盖全语料范围）。"""
    if not docs:
        raise ValueError("empty docs")
    if total_chars <= 0:
        raise ValueError("total_chars must be positive")
    need = max(1, total_chars // STREAM_DOC_CHARS)
    if need >= len(docs):
        return list(docs)
    step = len(docs) / need
    picked = [docs[min(len(docs) - 1, int(i * step))] for i in range(need)]
    return picked


def stream_source_docs(domains: dict, code_chars: int = 1_000_000,
                       nl_chars: int = 800_000) -> dict:
    """训练流的源文档（均匀取样；预算 200K token 用不完，留足余量）。"""
    code_docs = [t for _, t in domains["code_train"]]
    nl_docs = split_to_docs(domains["nl_train"])
    return {
        "code": sample_docs(code_docs, code_chars),
        "nl": sample_docs(nl_docs, nl_chars),
    }


def eval_texts(domains: dict, nl_chars: int = 24_000,
               code_chars: int = 36_000) -> dict[str, str]:
    """固定文本评测集（按字符切片定义，跨 tokenizer 可比）。

    nl：评测集文本前 nl_chars 字符；code：code_eval 文档按文件名序拼接
    后取前 code_chars 字符。约 12K token 量级。
    """
    nl_text = domains["nl_eval"][:nl_chars]
    code_text = "\n".join(t for _, t in domains["code_eval"])[:code_chars]
    return {"nl": nl_text, "code": code_text}


def tokenize(text: str, tokenizer) -> list[int]:
    """整段文本 → token id 列表（无 special token）。

    分块编码：整段 1M+ 字符一次 encode 会触发 HF 的
    "longer than maximum sequence length" 警告（模型 max_pos 131072），
    与实际用法无关（训练流只切窗使用），分块消音。
    """
    ids: list[int] = []
    step = 40000
    for i in range(0, len(text), step):
        ids.extend(tokenizer.encode(text[i:i + step], add_special_tokens=False))
    return ids


def build_stream(domain_docs: list[str], replay_docs: list[str], tokenizer,
                 budget: int, replay_ratio: float) -> list[int]:
    """token 级混合流（混合方式对照用；真实 CPT 不这么混）。"""
    from cpt_metrics import mix_token_ids

    domain_ids = tokenize("\n".join(domain_docs), tokenizer)
    replay_ids = tokenize("\n".join(replay_docs), tokenizer)
    return mix_token_ids(domain_ids, replay_ids, replay_ratio, budget)


def build_doc_stream(domain_docs: list[str], replay_docs: list[str], tokenizer,
                     budget: int, replay_ratio: float) -> list[int]:
    """文档级混合流（主方案，同真实 CPT 的 replay 混合方式）。

    整篇文档为单位按目标比例交错：哪一侧的 token 份额落后于目标就补哪一侧
    一整篇文档，保证（a）比例收敛到 replay_ratio（误差 < 一篇文档长度），
    （b）文档内部 token 连续（不制造代码/散文逐 token 交错的非自然上下文）。
    文档不够预算时循环取样；一侧为空由调用校验拒绝。
    """
    if budget <= 0:
        raise ValueError("budget must be positive")
    if not 0.0 <= replay_ratio <= 1.0:
        raise ValueError("replay_ratio must be in [0, 1]")
    if not domain_docs and replay_ratio < 1.0:
        raise ValueError("empty domain docs")
    if not replay_docs and replay_ratio > 0.0:
        raise ValueError("empty replay docs")

    dom_ids = [tokenize(d, tokenizer) for d in domain_docs]
    rep_ids = [tokenize(d, tokenizer) for d in replay_docs]
    stream: list[int] = []
    dom_used = rep_used = 0
    di = ri = 0
    while len(stream) < budget:
        # 份额落后于目标的一侧补一整篇文档（两侧都循环取，按份额收敛）
        if replay_ratio >= 1.0:
            behind_rep = True
        else:
            behind_rep = rep_used < len(stream) * replay_ratio
        if rep_ids and behind_rep:
            doc = rep_ids[ri % len(rep_ids)]
            ri += 1
            rep_used += len(doc)
        else:
            doc = dom_ids[di % len(dom_ids)]
            di += 1
            dom_used += len(doc)
        stream.extend(doc)
    return stream[:budget]


def extract_domain_tokens(code_docs: list[str], tokenizer, n: int,
                          min_count: int = 30) -> dict:
    """扩词表候选：代码高频标识符里被 Qwen 词表拆散（≥2 子词）的前 n 个。

    返回 new_tokens（新增字符串）、subword_ids（每个新 token 的旧子词 id 列表，
    供 subword_avg 初始化）、stats（压缩率等）。
    """
    text = "\n".join(code_docs)
    counts: dict[str, int] = {}
    for m in IDENT_RE.finditer(text):
        w = m.group(0)
        counts[w] = counts.get(w, 0) + 1

    candidates: list[tuple[str, int]] = []
    for w, c in counts.items():
        if c < min_count:
            continue
        ids = tokenizer.encode(w, add_special_tokens=False)
        if len(ids) >= 2:          # 已是单 token 的不用扩
            candidates.append((w, c))
    candidates.sort(key=lambda x: (-x[1], x[0]))

    new_tokens = [w for w, _ in candidates[:n]]
    subword_ids = [tokenizer.encode(w, add_special_tokens=False)
                   for w in new_tokens]
    raw_subtokens = sum(len(s) for s in subword_ids)
    stats = {
        "n_candidates": len(candidates),
        "n_new": len(new_tokens),
        "total_count": sum(c for _, c in candidates[:n]),
        "subtokens_before": raw_subtokens,
        "subtokens_after": len(new_tokens),
        "compression": round(raw_subtokens / max(1, len(new_tokens)), 3),
        "top10": [(w, c) for w, c in candidates[:10]],
    }
    return {"new_tokens": new_tokens, "subword_ids": subword_ids, "stats": stats}


def data_report(domains: dict, src_docs: dict, eval_t: dict,
                tokenizer) -> dict:
    """数据侧可复现性报告（指纹 + 规模 + 评测集字符数）。"""
    code_train_chars = sum(len(t) for t in src_docs["code"])
    nl_train_chars = sum(len(t) for t in src_docs["nl"])
    rep = fingerprint(domains)
    rep.update({
        "code_train_docs_total": len(domains["code_train"]),
        "code_stream_docs": len(src_docs["code"]),
        "code_stream_chars": code_train_chars,
        "nl_stream_docs": len(src_docs["nl"]),
        "nl_stream_chars": nl_train_chars,
        "code_stream_sha": _sha("\n".join(src_docs["code"])),
        "nl_stream_sha": _sha("\n".join(src_docs["nl"])),
        "eval_nl_chars": len(eval_t["nl"]),
        "eval_code_chars": len(eval_t["code"]),
        "eval_nl_tokens": len(tokenize(eval_t["nl"], tokenizer)),
        "eval_code_tokens": len(tokenize(eval_t["code"], tokenizer)),
        "tokenizer_len": len(tokenizer),
    })
    return rep
