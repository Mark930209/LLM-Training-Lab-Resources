"""eval_lab.py —— 15 篇主实验：口径对照、多 seed 方差、污染扫描。

四种 mode：
  caliber    : 同一 checkpoint 在 zero/few-shot × likelihood/generation 下的分数对照
  variance   : 多 seed 重复评测，报告均值 ± 标准差
  contamination : n-gram 污染扫描（评测集 vs 训练语料）
  report     : 生成"最小可复现评测报告"字段清单

模型：04 篇的语料规模 checkpoint（或随机初始化对照）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import time
from pathlib import Path

import torch

from .eval_common import (build_mini_eval, evaluation_set_sha256,
                          ngram_contamination, predict_generation,
                          predict_likelihood, score_generation, write_report)


def load_model_and_tokenizer(ckpt: str | None, device: str):
    """加载 04 篇的 SuperMiniGPT checkpoint（03 篇架构）或随机初始化。"""
    import os
    from exp_scale.model import SuperMiniGPT

    # 词表与语料：与 04 篇训练一致
    tok = None
    for cand in ["exp_scale/data/corpus_large.txt", "exp_scale/data/corpus_small.txt"]:
        if os.path.exists(cand):
            text = open(cand, encoding="utf-8").read()
            from exp_scale.data import CharTokenizer
            tok = CharTokenizer(text)
            break
    if tok is None:
        raise FileNotFoundError("找不到语料文件，CharTokenizer 需要语料构建")

    if ckpt:
        # 04 篇 checkpoint 含自定义 Config；torch 2.6+ 默认 weights_only=True。
        # 自家产物，信任来源，显式放行（读者复现外部 checkpoint 时不要照抄）。
        import sys
        sys.path.insert(0, ".")
        try:
            from common.config import Config  # noqa: F401
            torch.serialization.add_safe_globals([Config])
        except ImportError:
            pass
        state = torch.load(ckpt, map_location=device, weights_only=False)
        cfg = state["config"]["experiment"]
        model = SuperMiniGPT(vocab_size=state["model"]["tok_emb.weight"].shape[0],
                             hidden=cfg["hidden"], layers=cfg["layers"],
                             heads=cfg["heads"], seq_len=cfg["seq_len"])
        model.load_state_dict(state["model"])
    else:
        model = SuperMiniGPT(vocab_size=6015, hidden=384, layers=6,
                             heads=6, seq_len=256)
    model = model.to(device)
    model.eval()
    return model, tok


def mode_caliber(args) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    total_started = time.perf_counter()
    load_started = total_started
    model, tok = load_model_and_tokenizer(args.ckpt, device)
    if device == "cuda":
        torch.cuda.synchronize()
    model_loaded = time.perf_counter()
    samples = build_mini_eval()

    results = {}
    question_results = []
    scoring_started = time.perf_counter()
    # likelihood 判分（只对 classification 有效）
    cls = [(idx, sample) for idx, sample in enumerate(samples)
           if sample["task"] == "classification"]
    likelihood_results = []
    for idx, sample in cls:
        outcome = predict_likelihood(model, tok, sample, device)
        likelihood_results.append(outcome)
        question_results.append({
            "sample_index": idx,
            "task": sample["task"],
            "scoring": "likelihood",
            "prompt": sample["prompt"],
            "options": sample["options"],
            **outcome,
        })
    acc = sum(item["correct"] for item in likelihood_results) / len(cls)
    results["classification_likelihood"] = {"acc": acc, "n": len(cls)}

    # generation 判分（全部任务），seed 固定
    for task in ["classification", "completion", "arithmetic"]:
        subset = [(idx, sample) for idx, sample in enumerate(samples)
                  if sample["task"] == task]
        if not subset:
            continue
        outcomes = []
        for idx, sample in subset:
            outcome = predict_generation(model, tok, sample, device, seed=42)
            outcomes.append(outcome)
            question_results.append({
                "sample_index": idx,
                "task": sample["task"],
                "scoring": "generation",
                "prompt": sample["prompt"],
                "options": sample.get("options"),
                **outcome,
            })
        acc = sum(item["correct"] for item in outcomes) / len(subset)
        results[f"{task}_generation"] = {"acc": acc, "n": len(subset), "seed": 42}

    if device == "cuda":
        torch.cuda.synchronize()
    finished = time.perf_counter()
    peak_memory_mib = {
        "allocated": round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1),
        "reserved": round(torch.cuda.max_memory_reserved() / (1024 ** 2), 1),
    } if device == "cuda" else {"allocated": None, "reserved": None}
    code_files = [Path(__file__), Path(__file__).with_name("eval_common.py")]
    tokenizer_source = next(
        (Path(candidate).name for candidate in [
            "exp_scale/data/corpus_large.txt", "exp_scale/data/corpus_small.txt"
        ] if os.path.exists(candidate)),
        None,
    )
    tokenizer_vocab = getattr(tok, "stoi", None)
    tokenizer_vocab_sha256 = None
    if tokenizer_vocab is not None:
        vocab_bytes = json.dumps(
            sorted(tokenizer_vocab.items(), key=lambda item: item[1]),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        tokenizer_vocab_sha256 = hashlib.sha256(vocab_bytes).hexdigest()
    hardware = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    payload = {"mode": "caliber", "ckpt": args.ckpt, "results": results,
               "metadata": {
                   "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
                   "tokenizer": {
                       "class": type(tok).__name__,
                       "source": tokenizer_source,
                       "vocab_size": getattr(tok, "vocab_size", None),
                       "vocab_sha256": tokenizer_vocab_sha256,
                   },
                   "evaluation_set": {
                       "sha256": evaluation_set_sha256(samples),
                       "rows": len(samples),
                       "shots": 0,
                   },
                   "scoring": {
                       "likelihood": "mean conditional log-probability per encoded option token; out-of-vocabulary characters are skipped",
                       "generation": {
                           "max_new_tokens": 16,
                           "temperature": 0.8,
                           "top_k": 20,
                           "seed": 42,
                       },
                   },
                   "hardware": {
                       "device": hardware,
                       "pytorch": str(torch.__version__),
                       "cuda_runtime": torch.version.cuda,
                       "python": platform.python_version(),
                   },
                   "peak_memory_mib": peak_memory_mib,
                   "timing_seconds": {
                       "model_and_tokenizer_load": round(model_loaded - load_started, 3),
                       "scoring": round(finished - scoring_started, 3),
                       "total": round(finished - total_started, 3),
                   },
                   "code_sha256": {
                       path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in code_files
                   },
               },
               "question_results": question_results,
               "note": "同一 checkpoint，两种判分方式。分数不同是口径不同，不是模型不同。"}
    print(json.dumps(results, ensure_ascii=False, indent=1))
    return payload


def mode_variance(args) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load_model_and_tokenizer(args.ckpt, device)
    samples = build_mini_eval()
    seeds = list(range(args.seeds))

    per_task = {}
    for task in ["completion", "arithmetic"]:
        subset = [s for s in samples if s["task"] == task]
        accs = []
        for seed in seeds:
            acc = sum(score_generation(model, tok, s, device, seed=seed) for s in subset) / len(subset)
            accs.append(acc)
        mean = statistics.mean(accs)
        std = statistics.stdev(accs) if len(accs) > 1 else 0.0
        per_task[task] = {
            "accs": accs,
            "mean": mean,
            "std": std,
            "seed_range": [min(accs), max(accs)],
        }

    payload = {"mode": "variance", "ckpt": args.ckpt, "seeds": seeds,
               "per_task": per_task,
               "note": "多 seed 结果描述固定评测集上的生成波动，不是总体置信区间；likelihood 判分确定性。"}
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "accs"}
                      for k, v in per_task.items()}, ensure_ascii=False, indent=1))
    return payload


def mode_contamination(args) -> dict:
    import os
    corpus = None
    for cand in ["exp_scale/data/corpus_large.txt", "exp_scale/data/corpus.txt"]:
        if os.path.exists(cand):
            corpus = open(cand, encoding="utf-8").read()
            break
    if corpus is None:
        raise FileNotFoundError("找不到训练语料")

    samples = build_mini_eval()
    result = ngram_contamination(samples, corpus, n=args.ngram)
    payload = {"mode": "contamination", "ngram": args.ngram,
               "contamination_rate": result["contamination_rate"],
               "n_contaminated": sum(1 for h in result["samples"] if h["contaminated"]),
               "n_total": len(samples),
               "n_scannable": result["n_scannable"],
               "n_unscannable": result["n_unscannable"],
               "hits": [h for h in result["samples"] if h["contaminated"]][:20]}
    print(f"污染率: {result['contamination_rate']:.1%} "
          f"({payload['n_contaminated']}/{result['n_scannable']} 可扫描；"
          f"{result['n_unscannable']} 条短样本无法判定)")
    return payload


def mode_report(args) -> dict:
    fields = {
        "model": "checkpoint 相对路径与精确参数量",
        "tokenizer": "tokenizer 类、来源文件名、词表大小与 token→ID 映射 SHA-256",
        "eval_set": "规范化 JSON 的 SHA-256 与题数",
        "shots": "few-shot 数量与示例来源",
        "scoring": "likelihood / generation（含采样参数与 seed）",
        "seeds": "重复次数",
        "metrics": "各任务正确数/题数与准确率；多 seed 报均值、样本标准差和观测范围，不作为总体置信区间",
        "question_results": "逐题索引、任务、prompt/options、预测、标准答案、正确性与 generation 原始输出",
        "contamination": "n-gram 扫描的 n 与命中率",
        "hardware": "GPU 型号、PyTorch/CUDA 与 Python 版本",
        "peak_memory": "CUDA 峰值 allocated/reserved 显存（MiB）；CPU 运行时为 null",
        "timing": "模型/Tokenizer 加载、评分和总耗时",
        "evaluation_code": "eval_lab.py 与 eval_common.py 的 SHA-256",
    }
    payload = {"mode": "report", "fields": fields,
               "note": "缺任何一项，分数不可比。这是 15 篇之后所有评测报告的强制格式。"}
    print(json.dumps(fields, ensure_ascii=False, indent=1))
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["caliber", "variance", "contamination", "report"])
    ap.add_argument("--ckpt", default=None, help="04 篇 checkpoint 路径；缺省用随机初始化")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--ngram", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.mode == "caliber":
        payload = mode_caliber(args)
    elif args.mode == "variance":
        payload = mode_variance(args)
    elif args.mode == "contamination":
        payload = mode_contamination(args)
    else:
        payload = mode_report(args)

    if args.out:
        write_report(args.out, payload)


if __name__ == "__main__":
    main()
