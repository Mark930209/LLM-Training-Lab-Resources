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
import json
import statistics

import torch

from .eval_common import (build_mini_eval, ngram_contamination,
                          score_generation, score_likelihood, write_report)


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
    model, tok = load_model_and_tokenizer(args.ckpt, device)
    samples = build_mini_eval()

    results = {}
    # likelihood 判分（只对 classification 有效）
    cls = [s for s in samples if s["task"] == "classification"]
    acc = sum(score_likelihood(model, tok, s, device) for s in cls) / len(cls)
    results["classification_likelihood"] = {"acc": acc, "n": len(cls)}

    # generation 判分（全部任务），seed 固定
    for task in ["classification", "completion", "arithmetic"]:
        subset = [s for s in samples if s["task"] == task]
        if not subset:
            continue
        acc = sum(score_generation(model, tok, s, device, seed=42) for s in subset) / len(subset)
        results[f"{task}_generation"] = {"acc": acc, "n": len(subset), "seed": 42}

    payload = {"mode": "caliber", "ckpt": args.ckpt, "results": results,
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
        per_task[task] = {"accs": accs, "mean": mean, "std": std,
                          "ci95": [mean - 1.96 * std, mean + 1.96 * std]}

    payload = {"mode": "variance", "ckpt": args.ckpt, "seeds": seeds,
               "per_task": per_task,
               "note": "generation 判分受采样影响，必须报方差；likelihood 判分确定性、方差为 0。"}
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
               "hits": [h for h in result["samples"] if h["contaminated"]][:20]}
    print(f"污染率: {result['contamination_rate']:.1%} "
          f"({payload['n_contaminated']}/{len(samples)})")
    return payload


def mode_report(args) -> dict:
    fields = {
        "model": "checkpoint 路径与参数量",
        "tokenizer": "词表来源与版本",
        "eval_set": "评测集版本哈希与条数",
        "shots": "few-shot 数量与示例来源",
        "scoring": "likelihood / generation（含采样参数与 seed）",
        "seeds": "重复次数",
        "metrics": "各任务正确率、均值、标准差、置信区间",
        "contamination": "n-gram 扫描的 n 与命中率",
        "hardware": "GPU 型号与单次评测耗时",
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
