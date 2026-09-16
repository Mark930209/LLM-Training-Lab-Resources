#!/usr/bin/env python3
"""collect_long_samples.py —— 采集长文本续写（每段 600 token）。

为什么需要独立于 collect_samples.py：

    180 token 的短样本看不出模型间差别。四档模型在 180 token 内都能写出
    像样的文言片段，肉眼看不出高下；把生成长度提到 600 token，差距才暴露：
    有的模型能撑到回目诗赞，有的中途开始重复同一个词。

    生成长度不是"更长的同样东西"，而是对模型长程建模能力的直接压力测试。

用法：
    python scripts/collect_long_samples.py --out ../../results/Season1/04/long_samples.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exp_scale.data import load_corpus  # noqa: E402
from exp_scale.model import SuperMiniGPT  # noqa: E402

RUNS = Path.home() / "llm-training-lab/runs"
DATA = Path.home() / "llm-training-lab/exp_scale/data"

MODELS = [
    ("12.36M · 小语料", "scale_main_scale_10m_20260916_090305", "small"),
    ("12.93M · 大语料", "scale_main_scale_10m_large_20260916_103221", "large"),
    ("35.32M · 大语料", "scale_main_scale_30m_20260916_090558", "large"),
    ("89.57M · 大语料", "scale_main_scale_100m_20260916_091537", "large"),
]

# 三个提示词分别指向三本书，可以顺带观察模型是否"串书"
PROMPTS = ["却说那", "话说林黛玉", "玄德曰"]

GEN_TOKENS = 600
TEMPERATURE = 0.8
TOP_K = 40


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gen-tokens", type=int, default=GEN_TOKENS)
    args = ap.parse_args()

    torch.manual_seed(42)
    results = {"prompts": PROMPTS, "temperature": TEMPERATURE,
               "top_k": TOP_K, "gen_tokens": args.gen_tokens, "models": []}

    for label, run, corpus in MODELS:
        ckpt = RUNS / run / "ckpt_best.pt"
        if not ckpt.exists():
            ckpt = RUNS / run / "ckpt_last.pt"
        if not ckpt.exists():
            print(f"[SKIP] {label}: 无 checkpoint")
            continue
        ck = torch.load(ckpt, map_location="cuda", weights_only=False)
        exp = ck["config"]["experiment"]
        tok, _, _ = load_corpus(DATA, exp["seq_len"], corpus=corpus)
        model = SuperMiniGPT(tok.vocab_size, exp["hidden"], exp["layers"],
                             exp["heads"], exp["seq_len"]).cuda()
        model.load_state_dict(ck["model"])
        model.eval()

        entry = {"label": label, "run": run, "corpus": corpus,
                 "params_million": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
                 "best_val": ck.get("best_val"), "outputs": []}
        for prompt in PROMPTS:
            ids = tok.encode(prompt) or [0]
            ctx = torch.tensor([ids], dtype=torch.long, device="cuda")
            with torch.no_grad():
                out = model.generate(ctx, args.gen_tokens,
                                     temperature=TEMPERATURE, top_k=TOP_K)
            entry["outputs"].append({"prompt": prompt,
                                     "text": tok.decode(out[0].tolist())})
        results["models"].append(entry)
        del model
        torch.cuda.empty_cache()
        print(f"[done] {label}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"已写入 {out_path}")


if __name__ == "__main__":
    main()