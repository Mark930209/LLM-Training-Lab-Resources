"""warc_sample.py —— Common Crawl WARC/WET 采样下载与流式解析。

用 HTTP Range 只取分段开头的一段字节，流式喂给 warcio，避免下载整个
约 1 GB 的分段。截断处最后一条不完整记录会被安全跳过，已产出的记录有效。

只读公开数据，不写任何本地标识；分段路径是 Common Crawl 公开信息。
"""

from __future__ import annotations

import io
import json
import urllib.request
from typing import Iterator

from warcio.archiveiterator import ArchiveIterator

DEFAULT_UA = "llm-training-lab-16/1.0 (research; educational)"


def _http_range(url: str, max_bytes: int, timeout: int = 60) -> bytes:
    """带 Range 的 GET，最多取 max_bytes 字节。"""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": DEFAULT_UA, "Range": f"bytes=0-{max_bytes - 1}"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (https only)
        return resp.read(max_bytes)


def resolve_segment(index_url: str, query_url: str, timeout: int = 30) -> str:
    """从 CDX 索引解析一个真实 WARC 内容分段的完整路径。

    CDX 对同一查询会返回多种分段（``warc/`` 内容、``crawldiagnostics/`` 诊断等）；
    只有 ``warc/`` 内容分段才有对应的 WET，这里筛出第一条 ``warc/`` 分段。
    """
    url = f"{index_url}?url={query_url}&output=json&limit=40"
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line:
                continue
            filename = json.loads(line).get("filename", "")
            if "/warc/" in filename and filename.endswith(".warc.gz"):
                return filename
    raise RuntimeError("CDX 查询没有返回任何 warc/ 内容分段")


def wet_segment_from_warc(warc_segment: str) -> str:
    """由 WARC 分段路径推出同段 WET 路径。

    Common Crawl 的目录约定：WET 与对应 WARC 在**同一分段目录**下，只把
    ``warc/`` 换成 ``wet/``、文件名 ``.warc.gz`` 换成 ``.warc.wet.gz``。
    注意不能按分段号去 ``wet.paths.gz`` 清单里匹配——清单里同分段号的 WET
    可能来自另一个分段目录（日期段不同），记录 ID 完全不相交。
    """
    if "/warc/" not in warc_segment or not warc_segment.endswith(".warc.gz"):
        raise ValueError(f"不是 warc/ 内容分段路径: {warc_segment}")
    return warc_segment.replace("/warc/", "/wet/").replace(".warc.gz", ".warc.wet.gz")


def stream_records(data: bytes) -> Iterator:
    """把截断的 gzip 字节流解析成 warcio 记录，安全跳过末尾不完整记录。"""
    buf = io.BytesIO(data)
    try:
        for record in ArchiveIterator(buf):
            yield record
    except Exception:
        # 截断处最后一条记录不完整，warcio 会抛错；此前产出的记录仍然有效。
        return


def fetch_warc_html(base_url: str, segment: str, max_bytes: int,
                    max_records: int, timeout: int = 60) -> dict[str, str]:
    """取 WARC response 记录的 HTML 正文，按 WARC-Record-ID 索引。"""
    data = _http_range(f"{base_url}/{segment}", max_bytes, timeout)
    out: dict[str, str] = {}
    for record in stream_records(data):
        if record.rec_type != "response":
            continue
        rid = record.rec_headers.get("WARC-Record-ID")
        body = record.content_stream().read()
        if rid and body:
            out[rid] = body.decode("utf-8", errors="replace")
        if len(out) >= max_records:
            break
    return out


def fetch_wet_text(base_url: str, wet_segment: str, max_bytes: int,
                   max_records: int, timeout: int = 60) -> dict[str, str]:
    """取 WET conversion 记录的已抽取文本，按 WARC-Refers-To（原 WARC 记录 ID）索引。"""
    data = _http_range(f"{base_url}/{wet_segment}", max_bytes, timeout)
    out: dict[str, str] = {}
    for record in stream_records(data):
        if record.rec_type != "conversion":
            continue
        refers = record.rec_headers.get("WARC-Refers-To")
        body = record.content_stream().read()
        if refers and body:
            out[refers] = body.decode("utf-8", errors="replace")
        if len(out) >= max_records:
            break
    return out
