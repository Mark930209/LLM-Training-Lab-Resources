"""pipeline.py —— exp_data 统一入口（run_experiment 风格）。

当前已接线模式：
  sample_extract : WARC/WET 采样 → 三种正文抽取对照 → 留存率与噪声占比（REAL）

后续模式（quality/dedup/shard/tokenize/full）按增量逐个接线，键位见 config.yaml。

所有数字来自真实运行；采样规模由 config 控制，全量外推在文章里标 SCALED。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import yaml

from . import dedup, langid, manifest, pii, quality, shard, tokenize_par, warc_sample
from .extract import EXTRACTORS


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _code_sha256() -> dict[str, str]:
    here = Path(__file__).parent
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (here / "pipeline.py", here / "warc_sample.py",
                  here / "extract.py", here / "quality.py", here / "langid.py",
                  here / "dedup.py", here / "shard.py", here / "tokenize_par.py",
                  here / "pii.py", here / "manifest.py")
    }


def _fetch_corpus(cfg: dict) -> dict:
    """采样并解析 WARC/WET，返回对照集（同时有 HTML 与 WET 的记录）。"""
    src = cfg["source"]
    timeout = int(src.get("timeout_sec", 60))
    segment = src.get("segment") or warc_sample.resolve_segment(
        src["index_url"], src["query_url"], timeout
    )
    wet_segment = warc_sample.wet_segment_from_warc(segment)
    html_by_id = warc_sample.fetch_warc_html(
        src["base_url"], segment, int(src["max_bytes"]),
        int(src["max_records"]), timeout,
    )
    wet_by_id = warc_sample.fetch_wet_text(
        src["base_url"], wet_segment, int(src["max_bytes"]),
        int(src["max_records"]), timeout,
    )
    common_ids = [rid for rid in html_by_id if rid in wet_by_id]
    return {
        "segment": segment,
        "wet_segment": wet_segment,
        "html_by_id": html_by_id,
        "wet_by_id": wet_by_id,
        "common_ids": common_ids,
    }


def mode_sample_extract(cfg: dict) -> dict:
    methods = cfg["extract"]["methods"]

    started = time.perf_counter()
    corpus = _fetch_corpus(cfg)
    html_by_id = corpus["html_by_id"]
    wet_by_id = corpus["wet_by_id"]
    common_ids = corpus["common_ids"]

    per_method: dict[str, dict] = {m: {"n": 0, "raw_chars": 0, "out_chars": 0,
                                       "retention": [], "empty": 0} for m in methods}
    for rid in common_ids:
        html = html_by_id[rid]
        wet = wet_by_id[rid]
        raw_len = len(html)
        for method in methods:
            text = EXTRACTORS[method](wet if method == "wet" else html)
            bucket = per_method[method]
            bucket["n"] += 1
            bucket["raw_chars"] += raw_len
            bucket["out_chars"] += len(text)
            if text:
                bucket["retention"].append(len(text) / raw_len if raw_len else 0.0)
            else:
                bucket["empty"] += 1

    summary = {}
    for method, b in per_method.items():
        ret = b["retention"]
        summary[method] = {
            "n_docs": b["n"],
            "empty_docs": b["empty"],
            "raw_chars": b["raw_chars"],
            "out_chars": b["out_chars"],
            "overall_retention": round(b["out_chars"] / b["raw_chars"], 4) if b["raw_chars"] else None,
            "median_retention": round(statistics.median(ret), 4) if ret else None,
            "mean_retention": round(statistics.fmean(ret), 4) if ret else None,
        }

    # 噪声占比：朴素去标签相对 trafilatura 多保留的字符比例（越高说明 boilerplate 越多）
    noise = None
    if "naive_visible" in summary and "trafilatura" in summary:
        nv = summary["naive_visible"]["out_chars"]
        tf = summary["trafilatura"]["out_chars"]
        if nv:
            noise = round((nv - tf) / nv, 4)

    elapsed = round(time.perf_counter() - started, 3)
    return {
        "mode": "sample_extract",
        "segment": corpus["segment"],
        "wet_segment": corpus["wet_segment"],
        "common_docs": len(common_ids),
        "methods": summary,
        "boilerplate_noise_ratio": noise,
        "metadata": {
            "python": platform.python_version(),
            "elapsed_sec": elapsed,
            "code_sha256": _code_sha256(),
            "config": {"source": cfg["source"], "extract": cfg["extract"]},
        },
        "note": "采样子集 REAL；overall_retention 是字符级留存率，"
                "boilerplate_noise_ratio 越高说明朴素法保留的模板/导航噪声越多。",
    }


def _run_three_stages(cfg: dict) -> dict:
    """采样 → 抽取 → 质量过滤 → 语种识别，返回各阶段结果与最终文本。

    filter 与 dedup 模式共用这一段，保证两个模式的前三阶段口径完全一致
    （核心判断：阶段之间的口径对齐）。
    """
    method = cfg.get("filter", {}).get("extract_method", "trafilatura")
    corpus = _fetch_corpus(cfg)
    html_by_id = corpus["html_by_id"]
    wet_by_id = corpus["wet_by_id"]

    extracted: list[str] = []
    raw_chars = 0
    empty_docs = 0
    for rid in corpus["common_ids"]:
        html = html_by_id[rid]
        raw_chars += len(html)
        text = EXTRACTORS[method](wet_by_id[rid] if method == "wet" else html)
        if text:
            extracted.append(text)
        else:
            empty_docs += 1

    qcfg = quality.QualityConfig(**cfg.get("quality", {}))
    qres = quality.filter_batch(extracted, qcfg)

    lcfg = langid.LangIdConfig(
        target_langs=tuple(cfg.get("langid", {}).get("target_langs", ("en", "zh-cn", "zh-tw"))),
        min_prob=float(cfg.get("langid", {}).get("min_prob", 0.80)),
        min_chars_for_detect=int(cfg.get("langid", {}).get("min_chars_for_detect", 40)),
    )
    lres = langid.filter_batch(qres["passed_texts"], lcfg)

    stages = {
        "extract": {
            "n_in": len(corpus["common_ids"]),
            "n_out": len(extracted),
            "empty_docs": empty_docs,
            "raw_chars": raw_chars,
            "out_chars": sum(len(t) for t in extracted),
            "char_retention": round(sum(len(t) for t in extracted) / raw_chars, 4) if raw_chars else None,
        },
        "quality": {
            "n_in": qres["n_in"],
            "n_out": qres["n_out"],
            "doc_retention": qres["doc_retention"],
            "char_retention": qres["char_retention"],
            "reason_counts": qres["reason_counts"],
        },
        "langid": {
            "n_in": lres["n_in"],
            "n_kept": lres["n_kept"],
            "doc_retention": lres["doc_retention"],
            "status_counts": lres["status_counts"],
            "lang_counts": lres["lang_counts"],
        },
    }
    return {
        "corpus": corpus,
        "method": method,
        "stages": stages,
        "final_texts": lres["kept_texts"],
    }


def mode_filter(cfg: dict) -> dict:
    """采样 → 抽取 → 质量过滤 → 语种识别，逐阶段报告留存率。"""
    started = time.perf_counter()
    three = _run_three_stages(cfg)
    corpus = three["corpus"]
    elapsed = round(time.perf_counter() - started, 3)
    n_in = len(corpus["common_ids"])
    n_final = three["stages"]["langid"]["n_kept"]
    return {
        "mode": "filter",
        "segment": corpus["segment"],
        "extract_method": three["method"],
        "stages": three["stages"],
        "overall": {
            "n_in": n_in,
            "n_final": n_final,
            "doc_retention": round(n_final / n_in, 4) if n_in else None,
        },
        "metadata": {
            "python": platform.python_version(),
            "elapsed_sec": elapsed,
            "code_sha256": _code_sha256(),
            "config": {"source": cfg["source"], "filter": cfg.get("filter", {}),
                       "quality": cfg.get("quality", {}), "langid": cfg.get("langid", {})},
        },
        "note": "采样子集 REAL；逐阶段留存率用于定位是哪一道过滤删掉了数据。",
    }


def mode_dedup(cfg: dict) -> dict:
    """三阶段过滤后接精确去重 + MinHash LSH 近重复去重。

    核心判断落点：去重是唯一随语料量超线性变贵的环节。这里记录精确去重与
    MinHash 各自的留存率、耗时，以及 MinHash 的分桶倾斜（最大桶/平均桶），
    倾斜桶是放大到 TB 级时 OOM/超时的先兆。
    """
    started = time.perf_counter()
    three = _run_three_stages(cfg)
    corpus = three["corpus"]
    texts = three["final_texts"]

    dcfg = cfg.get("dedup", {})
    exact_res = dedup.exact_dedup(texts)

    mh_cfg_raw = dcfg.get("minhash", {})
    mh_thresholds = mh_cfg_raw.get("thresholds", [0.8])
    mh_results = {}
    for th in mh_thresholds:
        mcfg = dedup.MinHashConfig(
            num_perm=int(mh_cfg_raw.get("num_perm", 128)),
            threshold=float(th),
            shingle_k=int(mh_cfg_raw.get("shingle_k", 5)),
        )
        res = dedup.minhash_dedup(exact_res["kept_texts"], mcfg)
        # 不保留 kept_texts 进 JSON（太大），只留统计
        res.pop("kept_texts", None)
        mh_results[str(th)] = res

    elapsed = round(time.perf_counter() - started, 3)
    n_in = len(corpus["common_ids"])
    return {
        "mode": "dedup",
        "segment": corpus["segment"],
        "extract_method": three["method"],
        "stages": three["stages"],
        "dedup": {
            "input_docs": len(texts),
            "exact": {k: v for k, v in exact_res.items() if k != "kept_texts"},
            "minhash_by_threshold": mh_results,
        },
        "overall": {
            "n_in": n_in,
            "n_after_filter": len(texts),
            "n_after_exact": exact_res["n_out"],
        },
        "metadata": {
            "python": platform.python_version(),
            "elapsed_sec": elapsed,
            "code_sha256": _code_sha256(),
            "config": {"source": cfg["source"], "filter": cfg.get("filter", {}),
                       "quality": cfg.get("quality", {}), "langid": cfg.get("langid", {}),
                       "dedup": dcfg},
        },
        "note": "采样子集 REAL；minhash_by_threshold 的 bucket_skew.skew_ratio "
                "越大说明分桶越不均，是超线性代价与 OOM 的先兆。",
    }


def mode_dedup_scale(cfg: dict) -> dict:
    """合成分布规模扫描：复现去重的超线性代价与分桶倾斜（SIMULATED）。

    对 uniform / skewed 两种分布、若干规模档，分别跑精确去重与 MinHash，
    记录耗时、判重数、验证次数与桶倾斜。skewed 档的近重复组挤进同几个桶，
    桶内两两验证的平方项被放大——这是 TB 级去重 OOM/超时的先兆。
    """
    from . import synthetic

    started = time.perf_counter()
    scfg = cfg.get("dedup_scale", {})
    sizes = scfg.get("sizes", [500, 1000, 2000])
    distributions = scfg.get("distributions", ["uniform", "skewed"])
    mh_cfg_raw = cfg.get("dedup", {}).get("minhash", {})
    mcfg = dedup.MinHashConfig(
        num_perm=int(mh_cfg_raw.get("num_perm", 128)),
        threshold=float(scfg.get("threshold", 0.8)),
        shingle_k=int(mh_cfg_raw.get("shingle_k", 5)),
    )

    runs = []
    for dist in distributions:
        for n in sizes:
            corpus = synthetic.make_corpus(
                n, dist,
                doc_chars=int(scfg.get("doc_chars", 600)),
                skew_ratio=float(scfg.get("skew_ratio", 0.3)),
                shared_frac=float(scfg.get("shared_frac", 0.80)),
                seed=int(scfg.get("seed", 20260924)),
            )
            exact = dedup.exact_dedup(corpus)
            mh = dedup.minhash_dedup(corpus, mcfg)
            mh.pop("kept_texts", None)
            exact.pop("kept_texts", None)
            runs.append({
                "distribution": dist,
                "n_docs": n,
                "exact": {k: exact[k] for k in
                          ("n_out", "duplicates", "elapsed_sec")},
                "minhash": {
                    "n_out": mh["n_out"],
                    "duplicates": mh["duplicates"],
                    "verifications": mh["verifications"],
                    "elapsed_sec": mh["elapsed_sec"],
                    "bucket_skew": mh["bucket_skew"],
                },
            })

    elapsed = round(time.perf_counter() - started, 3)
    return {
        "mode": "dedup_scale",
        "truth_label": "SIMULATED",
        "runs": runs,
        "metadata": {
            "python": platform.python_version(),
            "elapsed_sec": elapsed,
            "code_sha256": _code_sha256(),
            "config": {"dedup_scale": scfg, "minhash": mh_cfg_raw},
        },
        "note": "合成分布 SIMULATED：uniform 是均匀基线；skewed 的近重复组"
                "同桶但被早判重（验证次数仍线性，展示桶占用倾斜）；moderate "
                "同桶碰撞但互不判重，验证次数随规模平方增长（超线性代价的"
                "直接形态）。对比三者的 verifications 与 elapsed_sec 随规模"
                "的增长倍率即可区分线性与平方。",
    }


def mode_shard_tokenize(cfg: dict) -> dict:
    """三阶段 + 精确去重 → parquet 分片 → 流式读取 → 并行 tokenize → 边界审计。

    失败案例复现：eos_between_docs=False 时文档直接首尾相接，训练端若按
    "每窗口单文档"假设读取，boundary_audit 会报 mismatch；True 时窗口内
    有 EOS 边界标记，audit 通过。两种口径的 loss 都看不出异常，只有审计
    能暴露——这就是"阶段口径不对齐"的静默形态。
    """
    import tempfile

    started = time.perf_counter()
    three = _run_three_stages(cfg)
    corpus = three["corpus"]
    texts = three["final_texts"]

    # 精确去重（分片前最后一道）
    exact = dedup.exact_dedup(texts)
    final_texts = exact["kept_texts"]

    # 分片写入 + 流式读取
    scfg = cfg.get("shard", {})
    with tempfile.TemporaryDirectory(prefix="exp_data_shards_") as tmpdir:
        write_res = shard.write_shards(
            final_texts, tmpdir,
            rows_per_shard=int(scfg.get("rows_per_shard", 1000)),
            source_seg=corpus["segment"].split("/")[-1],
        )
        read_res = shard.stream_read_benchmark(tmpdir)

    # 并行 tokenize
    tcfg = cfg.get("tokenize", {})
    spec = tokenize_par.build_char_tokenizer(final_texts)
    tok_res = tokenize_par.tokenize_parallel(
        final_texts, spec, workers=int(tcfg.get("workers", 8)))

    # 两种边界口径的打包与审计
    audits = {}
    for eos_flag in (True, False):
        stream = tokenize_par.pack_stream(
            tok_res["token_lists"], spec.eos_id, eos_flag)
        audit = tokenize_par.boundary_audit(
            stream, tok_res["doc_lengths"], spec.eos_id, eos_flag,
            seq_len=int(tcfg.get("seq_len", 256)),
            n_probes=int(tcfg.get("audit_probes", 20)))
        audits[str(eos_flag)] = {"stream_tokens": len(stream), **audit}

    elapsed = round(time.perf_counter() - started, 3)
    return {
        "mode": "shard_tokenize",
        "segment": corpus["segment"],
        "stages": three["stages"],
        "dedup_exact": {k: v for k, v in exact.items() if k != "kept_texts"},
        "shard_write": {k: v for k, v in write_res.items() if k != "shards"},
        "shard_read": read_res,
        "tokenize": {k: v for k, v in tok_res.items()
                     if k not in ("token_lists", "doc_lengths")},
        "boundary_audits": audits,
        "metadata": {
            "python": platform.python_version(),
            "elapsed_sec": elapsed,
            "code_sha256": _code_sha256(),
            "config": {"shard": scfg, "tokenize": tcfg},
        },
        "note": "采样子集 REAL；boundary_audits 的 False 口径若报 "
                "boundary_mismatch，即复现'loss 看不出、审计能看出'的"
                "文档边界静默错位。",
    }


def mode_full(cfg: dict) -> dict:
    """全链路收口：三阶段 → 精确去重 → PII → 分片 → tokenize → 边界审计 → manifest。

    manifest 是这批数据的血缘清单：来源、每阶段留存率、配置哈希、代码哈希。
    下游（17 配方、18 tokenizer、30 恢复）只信 manifest，不信文件名。
    """
    import tempfile

    started = time.perf_counter()
    three = _run_three_stages(cfg)
    corpus = three["corpus"]
    texts = three["final_texts"]

    stage_rows = [
        manifest.stage_row("extract", three["stages"]["extract"]["n_in"],
                           three["stages"]["extract"]["n_out"]),
        manifest.stage_row("quality", three["stages"]["quality"]["n_in"],
                           three["stages"]["quality"]["n_out"]),
        manifest.stage_row("langid", three["stages"]["langid"]["n_in"],
                           three["stages"]["langid"]["n_kept"]),
    ]

    # 精确去重
    exact = dedup.exact_dedup(texts)
    stage_rows.append(manifest.stage_row("dedup_exact", exact["n_in"],
                                         exact["n_out"], exact["elapsed_sec"]))
    texts = exact["kept_texts"]

    # PII 清洗
    pii_res = pii.scrub_batch(texts)
    stage_rows.append(manifest.stage_row("pii", pii_res["n_in"], pii_res["n_in"],
                                         None, hits=pii_res["total_hits"],
                                         docs_with_pii=pii_res["docs_with_pii"]))
    texts = pii_res["cleaned_texts"]

    # 分片 + 流式读取
    scfg = cfg.get("shard", {})
    with tempfile.TemporaryDirectory(prefix="exp_data_full_") as tmpdir:
        write_res = shard.write_shards(
            texts, tmpdir,
            rows_per_shard=int(scfg.get("rows_per_shard", 1000)),
            source_seg=corpus["segment"].split("/")[-1],
        )
        read_res = shard.stream_read_benchmark(tmpdir)

    # 并行 tokenize + 边界审计（只用正确口径 EOS=True）
    tcfg = cfg.get("tokenize", {})
    spec = tokenize_par.build_char_tokenizer(texts)
    tok_res = tokenize_par.tokenize_parallel(
        texts, spec, workers=int(tcfg.get("workers", 8)))
    stream = tokenize_par.pack_stream(
        tok_res["token_lists"], spec.eos_id, True)
    audit = tokenize_par.boundary_audit(
        stream, tok_res["doc_lengths"], spec.eos_id, True,
        seq_len=int(tcfg.get("seq_len", 256)),
        n_probes=int(tcfg.get("audit_probes", 20)))

    elapsed = round(time.perf_counter() - started, 3)
    code_hashes = _code_sha256()
    mf = manifest.build_manifest(
        source={
            "crawl_index": cfg["source"]["index_url"],
            "segment": corpus["segment"],
            "wet_segment": corpus["wet_segment"],
            "max_bytes": cfg["source"]["max_bytes"],
            "max_records": cfg["source"]["max_records"],
        },
        stages=stage_rows,
        cfg=cfg,
        code_sha256=code_hashes,
        truth_label="REAL",
        extra={
            "shard_write": {k: v for k, v in write_res.items() if k != "shards"},
            "shard_read": read_res,
            "tokenize": {k: v for k, v in tok_res.items()
                         if k not in ("token_lists", "doc_lengths")},
            "boundary_audit": audit,
        },
    )
    return {
        "mode": "full",
        "segment": corpus["segment"],
        "stages": three["stages"],
        "dedup_exact": {k: v for k, v in exact.items() if k != "kept_texts"},
        "pii": {k: v for k, v in pii_res.items()
                if k not in ("cleaned_texts", "samples")},
        "pii_samples": pii_res["samples"],
        "shard_write": {k: v for k, v in write_res.items() if k != "shards"},
        "shard_read": read_res,
        "tokenize": {k: v for k, v in tok_res.items()
                     if k not in ("token_lists", "doc_lengths")},
        "boundary_audit": audit,
        "manifest": mf,
        "metadata": {
            "python": platform.python_version(),
            "elapsed_sec": elapsed,
            "code_sha256": code_hashes,
        },
        "note": "采样子集 REAL；manifest 是这批数据的血缘清单，"
                "config_sha256 锁定口径，stages 记录每步留存率。",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["sample_extract", "filter", "dedup",
                             "dedup_scale", "shard_tokenize", "full"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.mode == "sample_extract":
        payload = mode_sample_extract(cfg)
    elif args.mode == "filter":
        payload = mode_filter(cfg)
    elif args.mode == "dedup":
        payload = mode_dedup(cfg)
    elif args.mode == "dedup_scale":
        payload = mode_dedup_scale(cfg)
    elif args.mode == "shard_tokenize":
        payload = mode_shard_tokenize(cfg)
    elif args.mode == "full":
        payload = mode_full(cfg)
    else:  # pragma: no cover
        raise ValueError(f"未接线的模式: {args.mode}")

    text = json.dumps(payload, ensure_ascii=False, indent=1)
    print(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
