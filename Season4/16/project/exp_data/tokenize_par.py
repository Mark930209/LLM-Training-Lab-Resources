"""tokenize_par.py —— 并行 tokenize 与文档边界口径。

提纲要求的失败案例在这里复现：离线 tokenize 的特殊 token、截断与文档边界
口径与训练端读取不一致时，loss 曲线看不出任何异常，但样本边界已经错位——
一个训练样本可能前半截是文档 A、后半截是文档 B，模型在学"跨文档拼接"的
伪相关。

两种边界口径：
  - eos_between_docs=True ：文档之间插入 EOS token，训练端按 EOS 复位
    attention/position（或至少能识别边界）。
  - eos_between_docs=False：文档直接首尾相接。训练端如果仍按"每个样本
    是独立文档"的假设读取，边界就静默错位。

检测器 boundary_audit 不依赖 loss：它从 token 流里按记录的文档长度重建
边界，抽查若干窗口，验证窗口内 token 是否全部属于同一文档。口径不一致时
audit 直接报 mismatch，这就是"loss 看不出来、审计能看出来"的证据。

tokenizer 用 04 篇同款 char 级口径（本文重点是边界而不是词表，词表留给
18 篇）：字符 → id 的映射从语料现场构建，EOS 取词表外专用 id。
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass


@dataclass
class TokenizerSpec:
    vocab_size: int
    eos_id: int


def build_char_tokenizer(texts: list[str]) -> TokenizerSpec:
    """从文档集现场构建 char 词表；EOS 用 len(vocab) 作为专用 id。"""
    vocab: set[str] = set()
    for t in texts:
        vocab.update(t)
    vocab_size = len(vocab) + 1  # +1 给 EOS
    return TokenizerSpec(vocab_size=vocab_size, eos_id=len(vocab))


def _tokenize_one(args: tuple[str, dict]) -> list[int]:
    text, stoi = args
    return [stoi[ch] for ch in text if ch in stoi]


def tokenize_parallel(texts: list[str], spec: TokenizerSpec,
                      workers: int = 8) -> dict:
    """并行 tokenize 一批文档，返回 token 流与每文档长度（边界重建用）。"""
    # 现场重建 stoi（与 build_char_tokenizer 同口径）
    vocab: set[str] = set()
    for t in texts:
        vocab.update(t)
    stoi = {ch: i for i, ch in enumerate(sorted(vocab))}

    started = time.perf_counter()
    if workers > 1 and len(texts) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            token_lists = list(pool.map(_tokenize_one, [(t, stoi) for t in texts]))
    else:
        token_lists = [_tokenize_one((t, stoi)) for t in texts]
    elapsed = time.perf_counter() - started

    n_tokens = sum(len(tl) for tl in token_lists)
    return {
        "token_lists": token_lists,
        "doc_lengths": [len(tl) for tl in token_lists],
        "n_docs": len(texts),
        "n_tokens": n_tokens,
        "elapsed_sec": round(elapsed, 4),
        "tokens_per_sec": round(n_tokens / elapsed, 1) if elapsed > 0 else None,
        "vocab_size": spec.vocab_size,
        "eos_id": spec.eos_id,
    }


def pack_stream(token_lists: list[list[int]], eos_id: int,
                eos_between_docs: bool) -> list[int]:
    """把文档 token 列表打包成连续训练流。"""
    stream: list[int] = []
    for tl in token_lists:
        stream.extend(tl)
        if eos_between_docs:
            stream.append(eos_id)
    return stream


def boundary_audit(stream: list[int], doc_lengths: list[int], eos_id: int,
                   eos_between_docs: bool, seq_len: int = 256,
                   n_probes: int = 20) -> dict:
    """从 token 流重建文档边界并抽查窗口，验证训练端假设是否成立。

    训练端假设：每个 seq_len 窗口内的 token 属于同一文档（或以 EOS 分隔）。
    audit 按打包口径重建每个 token 的文档归属，抽查 n_probes 个窗口：
      - eos_between_docs=True 时，窗口内出现 EOS 即视为有边界标记，合格；
      - eos_between_docs=False 时，窗口若跨越两个文档且无 EOS，判 mismatch。
    """
    # 重建每个位置的文档归属
    owner: list[int] = []
    for doc_idx, length in enumerate(doc_lengths):
        owner.extend([doc_idx] * length)
        if eos_between_docs:
            owner.append(-1)  # EOS 位置，不属于任何文档

    mismatches = 0
    probes = 0
    stride = max(1, (len(stream) - seq_len) // max(1, n_probes))
    for start in range(0, max(1, len(stream) - seq_len), stride):
        if probes >= n_probes:
            break
        window_owner = owner[start:start + seq_len]
        docs_in_window = {o for o in window_owner if o != -1}
        has_eos = any(stream[p] == eos_id for p in range(start, min(start + seq_len, len(stream))))
        probes += 1
        if len(docs_in_window) > 1 and not (eos_between_docs and has_eos):
            mismatches += 1

    return {
        "seq_len": seq_len,
        "n_probes": probes,
        "mismatches": mismatches,
        "eos_between_docs": eos_between_docs,
        "verdict": "ok" if mismatches == 0 else "boundary_mismatch",
    }
