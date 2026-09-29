"""shard.py —— parquet 分片写入与流式读取吞吐。

把过滤/去重后的文档写成固定行数的 parquet 分片，并测量训练端视角的
流式读取吞吐（MB/s 与 docs/s）。分片是"管线产物"与"训练输入"的交接面：
交接面上的口径（schema、行序、文档边界字段）不一致，就是核心判断说的
"阶段之间口径不对齐"。

schema 固定为：
  doc_id      : 分片内唯一（shard 序号 × rows_per_shard + 行号）
  text        : 清洗后正文
  char_len    : 字符数
  source_seg  : 来源 WARC 分段名（血缘）
  stage_flags : 该文档通过的阶段标记（JSON 字符串，血缘）
"""

from __future__ import annotations

import json
import time
from pathlib import Path


def write_shards(texts: list[str], out_dir: str | Path, rows_per_shard: int,
                 source_seg: str = "synthetic") -> dict:
    """把文档写成 parquet 分片，返回分片清单与写入统计。"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    shards: list[dict] = []
    total_bytes = 0
    for shard_idx in range(0, max(1, (len(texts) + rows_per_shard - 1) // rows_per_shard)):
        chunk = texts[shard_idx * rows_per_shard:(shard_idx + 1) * rows_per_shard]
        if not chunk:
            break
        table = pa.table({
            "doc_id": [shard_idx * rows_per_shard + i for i in range(len(chunk))],
            "text": chunk,
            "char_len": [len(t) for t in chunk],
            "source_seg": [source_seg] * len(chunk),
            "stage_flags": [json.dumps({"extract": 1, "quality": 1, "langid": 1})] * len(chunk),
        })
        path = out_dir / f"shard-{shard_idx:05d}.parquet"
        pq.write_table(table, path, compression="zstd")
        size = path.stat().st_size
        total_bytes += size
        shards.append({"shard": path.name, "rows": len(chunk), "bytes": size})

    elapsed = time.perf_counter() - started
    return {
        "n_docs": len(texts),
        "n_shards": len(shards),
        "rows_per_shard": rows_per_shard,
        "total_bytes": total_bytes,
        "write_elapsed_sec": round(elapsed, 4),
        "write_mbps": round(total_bytes / elapsed / 1e6, 2) if elapsed > 0 else None,
        "shards": shards,
    }


def stream_read_benchmark(shard_dir: str | Path, max_shards: int | None = None) -> dict:
    """模拟训练端流式读取：逐分片、逐行迭代，测吞吐。

    训练端只关心"能按顺序流出来多少 docs/s、多少 MB/s"；这里不构造
    Tensor，只测 IO 与反序列化，避免把 tokenize 混进读取口径。
    """
    import pyarrow.parquet as pq

    shard_dir = Path(shard_dir)
    paths = sorted(shard_dir.glob("shard-*.parquet"))
    if max_shards is not None:
        paths = paths[:max_shards]

    started = time.perf_counter()
    n_docs = 0
    n_bytes = 0
    for path in paths:
        n_bytes += path.stat().st_size
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=256, columns=["text"]):
            n_docs += batch.num_rows
    elapsed = time.perf_counter() - started
    return {
        "n_shards": len(paths),
        "n_docs": n_docs,
        "bytes": n_bytes,
        "elapsed_sec": round(elapsed, 4),
        "docs_per_sec": round(n_docs / elapsed, 1) if elapsed > 0 else None,
        "mb_per_sec": round(n_bytes / elapsed / 1e6, 2) if elapsed > 0 else None,
    }
