"""tok_metrics.py —— tokenizer 静态指标：压缩率、覆盖率、长尾分布。

这些指标不需要训练模型就能算，是词表 sweep 的第一层证据：
  - 压缩率：bytes/token 与 chars/token，词表越大压缩越好
  - 覆盖率：评测文本里能被无损编码的字符比例（byte-level BPE 恒为 1，
    Unigram 靠 byte fallback 也接近 1，char 级受词表限制）
  - 长尾分布：每个 token 在训练语料里的出现次数，低于阈值的占比。
    这是"词表越大、参数花在没被训练的位置"的直接量化。
"""

from __future__ import annotations

import difflib
from collections import Counter

from .tok_train import decode_ids, encode_text


def compression_metrics(tokenizer, info: dict, text: str) -> dict:
    """对一段文本算压缩率与编码统计。"""
    ids = encode_text(tokenizer, info, text)
    n_tokens = len(ids)
    n_bytes = len(text.encode("utf-8"))
    n_chars = len(text)
    return {
        "n_tokens": n_tokens,
        "n_chars": n_chars,
        "n_bytes": n_bytes,
        "bytes_per_token": round(n_bytes / n_tokens, 3) if n_tokens else None,
        "chars_per_token": round(n_chars / n_tokens, 3) if n_tokens else None,
        "compression_ratio": round(n_chars / n_tokens, 3) if n_tokens else None,
    }


def coverage_metrics(tokenizer, info: dict, text: str,
                     max_chars: int = 5000) -> dict:
    """覆盖率：编码再解码后与原文的还原相似度。

    用 difflib.SequenceMatcher 做最优对齐，对 SentencePiece 的前导/词间
    空格差异鲁棒：逐字符位置对比会因 ▁ 还原成空格而整体错位，把无损
    还原虚报成大量损失（冒烟实测 Unigram 覆盖率被压到 0.07）。byte-level
    BPE 与 byte-fallback Unigram 的还原相似度都应接近 1.0。长文本截取前
    max_chars 字符，避免 SequenceMatcher 的 O(n^2) 开销。
    """
    text = text[:max_chars]
    ids = encode_text(tokenizer, info, text)
    decoded = decode_ids(tokenizer, info, ids)
    similarity = difflib.SequenceMatcher(None, text, decoded).ratio()
    return {
        "roundtrip_similarity": round(similarity, 4),
        "exact_after_strip": text.strip() == decoded.strip(),
        "len_text": len(text),
        "len_decoded": len(decoded),
    }


def token_frequency_distribution(tokenizer, info: dict, texts: list[str],
                                 low_freq_threshold: int = 5) -> dict:
    """统计训练语料里每个 token 的出现次数，给出长尾分布。

    长尾占比 = 出现次数 < 阈值的 token 种类数 / 词表实际使用的 token 种类数。
    词表越大，长尾占比越高：新增的 token 多是低频组合，训练时几乎不被更新。
    """
    counter: Counter = Counter()
    for t in texts:
        counter.update(encode_text(tokenizer, info, t))
    used = len(counter)
    low = sum(1 for c in counter.values() if c < low_freq_threshold)
    counts = sorted(counter.values())
    return {
        "used_token_types": used,
        "low_freq_types": low,
        "low_freq_ratio": round(low / used, 4) if used else None,
        "low_freq_threshold": low_freq_threshold,
        "median_count": counts[len(counts) // 2] if counts else 0,
        "max_count": counts[-1] if counts else 0,
    }


def embedding_param_share(vocab_size: int, hidden: int, layers: int,
                          heads: int, tied: bool = True) -> dict:
    """算 embedding 参数占总参数的比例（解析式，与 04 篇同口径）。

    SuperMiniGPT 结构：tok_emb(vocab×hidden) + layers×block + norm + lm_head。
    tied=True 时 lm_head 与 tok_emb 共享权重，embedding 只算一份。
    每个 block：attention qkv(hidden×3hidden) + proj(hidden×hidden)
              + SwiGLU 3×(hidden×ffn)，ffn=round(8/3×hidden)
              + 2×RMSNorm(2hidden)
    """
    ffn = round(8 / 3 * hidden)
    per_block = (hidden * 3 * hidden + hidden * hidden      # attention
                 + 3 * hidden * ffn                          # SwiGLU
                 + 2 * hidden)                               # 2 RMSNorm
    backbone = layers * per_block + hidden                   # + final norm
    emb = vocab_size * hidden                                # tied: 只算一份
    total = emb + backbone
    return {
        "vocab_size": vocab_size,
        "embedding_params": emb,
        "backbone_params": backbone,
        "total_params": total,
        "embedding_share": round(emb / total, 4),
    }
