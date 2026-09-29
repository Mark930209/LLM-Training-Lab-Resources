"""dedup.py —— 精确去重 + MinHash LSH 近重复去重。

核心判断的落点：去重是整条管线里唯一随语料量**超线性**变贵的环节。
- 精确去重：对规整后的文本取哈希，O(n) 时间、O(unique) 内存。
- MinHash LSH：把每条文本的 shingle 集合压成签名，按 band 分桶，只在同桶内
  两两比对。桶数固定时，单桶文档数随语料量增长，桶内两两比对是 O(k²)，
  于是总代价超线性。分桶倾斜（某些桶挤进大量近似文档）会把这个平方项放大。

本模块记录每桶文档数分布，用来暴露倾斜；倾斜桶是 OOM/超时的先兆。
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass

_WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """精确去重前的规整：压空白、去首尾。大小写保留（中文无大小写，英文按需）。"""
    return _WS_RE.sub(" ", text).strip()


def _shingles(text: str, k: int) -> set[str]:
    """字符级 k-shingle 集合（中文按字切，与系列 char 口径一致）。"""
    t = _normalize(text)
    if len(t) < k:
        return {t} if t else set()
    return {t[i:i + k] for i in range(len(t) - k + 1)}


def exact_dedup(texts: list[str]) -> dict:
    """精确去重：规整后哈希，保留首次出现。返回留存与耗时。"""
    started = time.perf_counter()
    seen: set[str] = set()
    kept: list[str] = []
    dup = 0
    for t in texts:
        h = hashlib.sha256(_normalize(t).encode("utf-8")).hexdigest()
        if h in seen:
            dup += 1
            continue
        seen.add(h)
        kept.append(t)
    elapsed = time.perf_counter() - started
    n_in = len(texts)
    return {
        "n_in": n_in,
        "n_out": len(kept),
        "duplicates": dup,
        "doc_retention": round(len(kept) / n_in, 4) if n_in else None,
        "elapsed_sec": round(elapsed, 4),
        "kept_texts": kept,
    }


@dataclass
class MinHashConfig:
    num_perm: int = 128
    threshold: float = 0.8
    shingle_k: int = 5


def minhash_dedup(texts: list[str], cfg: MinHashConfig) -> dict:
    """MinHash LSH 近重复去重（带签名验证）。

    对每条文本建 MinHash 签名插入 LSH；查询到同桶候选后，用完整签名的
    Jaccard 逐一验证，达到阈值才判重。验证是必要的：单个 band 碰撞只是
    候选，不验证会把"碰巧同桶"的不同文档误删。

    代价结构（核心判断的机制）：
    - 签名构建：O(n · num_perm · shingles)，线性；
    - 桶内验证：对落在同一桶的文档两两比对。桶大小 k 随语料量增长时，
      验证代价是 O(k²)——这就是去重超线性的来源。分桶倾斜（大量文档共享
      模板前缀 → 挤进同几个桶）会把平方项放大。
    - 内存：每条保留文档的签名（num_perm × 8B + 对象开销）常驻，
      语料越大占用越高，放大到 TB 级必须先分片。
    """
    from datasketch import MinHash, MinHashLSH

    started = time.perf_counter()
    lsh = MinHashLSH(threshold=cfg.threshold, num_perm=cfg.num_perm)
    kept: list[str] = []
    kept_sigs: dict[str, object] = {}
    dup = 0
    verifications = 0
    bucket_sizes: dict[str, int] = {}

    for i, t in enumerate(texts):
        sh = _shingles(t, cfg.shingle_k)
        if not sh:
            # 空文本直接保留（交给 quality 的长度规则处理），不进 LSH
            kept.append(t)
            continue
        mh = MinHash(num_perm=cfg.num_perm)
        for s in sh:
            mh.update(s.encode("utf-8"))
        key = f"doc_{i}"
        candidates = lsh.query(mh)
        is_dup = False
        for cand in candidates:
            verifications += 1
            if mh.jaccard(kept_sigs[cand]) >= cfg.threshold:
                is_dup = True
                break
        if is_dup:
            dup += 1
        else:
            lsh.insert(key, mh)
            kept_sigs[key] = mh
            kept.append(t)
        # 统计该文档落到的桶（LSH 内部按 band 哈希分桶）
        for band_key in _band_keys(mh, cfg):
            bucket_sizes[band_key] = bucket_sizes.get(band_key, 0) + 1

    elapsed = time.perf_counter() - started
    n_in = len(texts)
    sizes = sorted(bucket_sizes.values(), reverse=True)
    skew = {
        "n_buckets": len(sizes),
        "max_bucket": sizes[0] if sizes else 0,
        "top5_bucket": sizes[:5],
        "mean_bucket": round(sum(sizes) / len(sizes), 2) if sizes else 0.0,
        # 倾斜度：最大桶 / 平均桶；越大说明分布越不均，平方项越危险
        "skew_ratio": round((sizes[0] / (sum(sizes) / len(sizes))), 2) if sizes else 0.0,
    }
    return {
        "n_in": n_in,
        "n_out": len(kept),
        "duplicates": dup,
        "verifications": verifications,
        "doc_retention": round(len(kept) / n_in, 4) if n_in else None,
        "elapsed_sec": round(elapsed, 4),
        "config": {"num_perm": cfg.num_perm, "threshold": cfg.threshold,
                   "shingle_k": cfg.shingle_k},
        "bucket_skew": skew,
        "kept_texts": kept,
    }


def _band_keys(mh, cfg: MinHashConfig) -> list[str]:
    """近似还原 LSH 的 band 分桶键，用于统计桶大小分布。

    datasketch 的 MinHashLSH 把 num_perm 个哈希值切成若干 band，每个 band 内
    的哈希片段一致才进同桶。这里用签名的分段哈希近似该分桶，只为统计倾斜，
    不参与去重判定（判定由 lsh.query 完成）。
    """
    import struct

    h = mh.hashvalues  # numpy uint64 array, length = num_perm
    n = len(h)
    # 与 datasketch 默认一致的 band 数：threshold 决定 b 与 r 的切分，这里取近似
    b = max(1, int(round(cfg.num_perm / 4)))
    r = max(1, n // b)
    keys = []
    for band in range(b):
        chunk = h[band * r:(band + 1) * r]
        if len(chunk) == 0:
            continue
        digest = hashlib.md5(struct.pack(f"<{len(chunk)}Q", *[int(x) for x in chunk])).hexdigest()
        keys.append(f"b{band}_{digest[:12]}")
    return keys
