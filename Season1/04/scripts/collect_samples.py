#!/usr/bin/env python3
"""collect_samples.py —— 用同一提示词让各档模型续写，采集真实输出。

用于 WebUI 演示与文章配图：切换不同模型，看同一个提示词续写出什么。
所有输出都是真实推理结果，不得人工修饰。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path.home() / "llm-training-lab"))

from exp_scale.data import load_corpus  # noqa: E402
from exp_scale.model import SuperMiniGPT  # noqa: E402

RUNS = Path.home() / "llm-training-lab/runs"
OUT = Path("/mnt/d/DocProjects/LearnLLMFromDraft/results/Season1/04/samples.json")

# 四组模型：标签 → run 目录匹配模式 + 语料档位
MODELS = [
    ("10M · 小语料", "scale_main_scale_10m_20260916_090305", "small"),
    ("10M · 大语料", "scale_main_scale_10m_large_20260916_103221", "large"),
    ("30M · 大语料", "scale_main_scale_30m_20260916_090558", "large"),
    ("100M · 大语料", "scale_main_scale_100m_20260916_091537", "large"),
]

# 同一个提示词（base model 是续写器，给开头让它接）
PROMPTS = ["却说那", "话说", "次日"]

GEN_TOKENS = 180
TEMPERATURE = 0.8
TOP_K = 40


def main() -> None:
    results = {"prompts": PROMPTS, "temperature": TEMPERATURE,
               "top_k": TOP_K, "gen_tokens": GEN_TOKENS, "models": []}

    for label, run_name, corpus in MODELS:
        run_dir = RUNS / run_name
        ckpt = run_dir / "ckpt_best.pt"
        if not ckpt.exists():
            ckpt = run_dir / "ckpt_last.pt"
        if not ckpt.exists():
            print(f"[SKIP] {label}: 无 checkpoint ({run_dir})")
            continue

        ck = torch.load(ckpt, map_location="cuda", weights_only=False)
        cfg = ck.get("config")
        exp = cfg["experiment"]
        tok, _, _ = load_corpus(
            Path.home() / "llm-training-lab/exp_scale/data",
            exp["seq_len"], corpus=corpus)

        model = SuperMiniGPT(tok.vocab_size, exp["hidden"], exp["layers"],
                             exp["heads"], exp["seq_len"]).cuda()
        model.load_state_dict(ck["model"])
        model.eval()
        n_params = sum(p.numel() for p in model.parameters())

        entry = {"label": label, "run": run_name, "corpus": corpus,
                 "params_million": round(n_params / 1e6, 2),
                 "vocab_size": tok.vocab_size,
                 "best_val": ck.get("best_val"), "outputs": []}
        print(f"\n=== {label} (params {n_params/1e6:.2f}M, "
              f"vocab {tok.vocab_size}, best_val {ck.get('best_val')}) ===")

        for prompt in PROMPTS:
            ids = tok.encode(prompt)
            ctx = torch.tensor([ids], dtype=torch.long, device="cuda")
            with torch.no_grad():
                out = model.generate(ctx, GEN_TOKENS,
                                     temperature=TEMPERATURE, top_k=TOP_K)
            text = tok.decode(out[0].tolist())
            entry["outputs"].append({"prompt": prompt, "text": text})
            print(f"  [{prompt}] → {text[:120]}...")

        results["models"].append(entry)
        del model
        torch.cuda.empty_cache()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n已写入 {OUT}")


if __name__ == "__main__":
    main()