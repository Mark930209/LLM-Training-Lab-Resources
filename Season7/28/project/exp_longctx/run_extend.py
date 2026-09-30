"""run_extend.py —— RoPE 扩展对照：2k 基线零样本扩到 4k/8k/16k。

用法：python run_extend.py --ckpt out/ckpt_2k.pt --out out/e2_extend.json

四方式：none（直接外推）/ pi / ntk / yarn；评测 = 捞针网格 + 短文本回测。
失败案例 1：某方式长检索提升而短测退化（如实记录）。
"""

from __future__ import annotations

import argparse

import torch

import longctx_common as lc
from tiny_rope import TinyRoPE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = args.device

    blob = torch.load(args.ckpt, weights_only=False)
    train_len = blob["seq"]

    results = {}
    for mode in ("none", "pi", "ntk", "yarn"):
        entry = {"needle": {}, "short_loss": None}
        for S in (4096, 8192, 16384):
            s = S / train_len
            torch.manual_seed(0)
            model = TinyRoPE(vocab=lc.VOCAB, rope_mode=mode,
                             rope_s=(1.0 if mode == "none" else s)).to(device)
            model.load_state_dict(blob["model"])
            for d in (0.1, 0.5, 0.9):
                entry["needle"][f"{S}@{d}"] = lc.needle_eval(
                    model, device, S, d, n_pairs=max(12, S // 85))
            del model
        # 短文本回测：短位置上的行为变化（PI 会压缩位置 → 退化来源）
        torch.manual_seed(0)
        model = TinyRoPE(vocab=lc.VOCAB, rope_mode=mode,
                         rope_s=(1.0 if mode == "none" else 8.0)).to(device)
        model.load_state_dict(blob["model"])
        entry["short_loss"] = lc.short_task_loss(model, device)
        entry["short_needle_512"] = lc.needle_eval(model, device, 512, 0.5,
                                                   n_pairs=6)
        results[mode] = entry
        print(mode, "short", round(entry["short_loss"], 4),
              "needle16k@0.5", entry["needle"].get("16384@0.5"))

    out = {
        "experiment": "e2_rope_extend",
        "framework": "plain-pytorch(tiny_rope)",
        "train_len": train_len,
        "results": results,
        "labels": {"REAL": "单卡真实运行"},
    }
    lc.save_json(out, args.out)


if __name__ == "__main__":
    main()
