#!/usr/bin/env python3
"""diagnose_init.py —— 诊断不同规模下的初始 loss，定位放大后 loss 异常的原因。

理论预期：随机初始化下 loss ≈ ln(vocab_size) ≈ 8.70（大语料 6015 词表）。
实测 10M（hidden=384, 12 层以下）接近预期，100M（768×12 层）却有 30+，
说明 logits 尺度随深度放大——这是"玩具代码放大后暴露的问题"。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path.home() / "llm-training-lab"))

from exp_scale.data import load_corpus  # noqa: E402
from exp_scale.model import SuperMiniGPT  # noqa: E402

CFGS = [("10M", 384, 6, 6), ("30M", 576, 8, 8), ("100M", 768, 12, 12)]


def main() -> None:
    tok, tr, _ = load_corpus(Path.home() / "llm-training-lab/exp_scale/data",
                             256, corpus="large")
    loader = torch.utils.data.DataLoader(tr, batch_size=4, shuffle=True)
    x, y = next(iter(loader))
    print(f"vocab={tok.vocab_size}  ln(vocab)={math.log(tok.vocab_size):.3f}")

    for name, h, layers, heads in CFGS:
        torch.manual_seed(42)
        m = SuperMiniGPT(tok.vocab_size, h, layers, heads, 256).cuda()
        with torch.no_grad():
            logits = m(x.cuda())
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), y.cuda().reshape(-1))
            # 关键指标：logits 的标准差（尺度）
            print(f"{name:>5} hidden={h:3d} layers={layers:2d} "
                  f"init_loss={loss.item():7.4f}  "
                  f"logits_std={logits.std().item():6.3f}  "
                  f"logits_absmax={logits.abs().max().item():7.2f}")
        del m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()