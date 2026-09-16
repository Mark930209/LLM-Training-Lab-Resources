#!/usr/bin/env python3
"""eval_cross_book.py —— 跨书留出困惑度：测模型对"没见过的书"的适应力。

为什么需要这个脚本：

    val loss 只在一个 train/val split 上算。大语料里四本书按 9:1 切分后，
    val 集里混着四本书的片段，所以"大语料模型的 val loss 更低"不能直接说明
    它学会了四本书——只能说它在自己见过的四本书的留出段上更好。

    要验证"语料扩大到底带来了什么"，必须换一把尺子：把每一本书单独拿出来，
    用同一段文本、同一个口径算 loss，看模型在自己训练过的书和没见过的书上的
    差距。差距小说明它学到了跨书的共性，差距大说明它只会自己那一本。

用法：

    python scripts/eval_cross_book.py --out ../../results/Season1/04/cross_book.json

输出：每个模型的四本书 loss 与困惑度对照表（真实推理，不得人工修饰）。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exp_scale.data import CharTokenizer  # noqa: E402
from exp_scale.model import SuperMiniGPT  # noqa: E402

RUNS = Path.home() / "llm-training-lab/runs"
DATA = Path.home() / "llm-training-lab/exp_scale/data"

# 四档模型：标签 → (run 目录名, 训练时用的语料档)
MODELS = [
    ("12.36M · 小语料", "scale_main_scale_10m_20260916_090305", "small"),
    ("12.93M · 大语料", "scale_main_scale_10m_large_20260916_103221", "large"),
    ("35.32M · 大语料", "scale_main_scale_30m_20260916_090558", "large"),
    ("89.57M · 大语料", "scale_main_scale_100m_20260916_091537", "large"),
]

BOOKS = ["xiyouji", "hongloumeng", "sanguoyanyi", "shuihuzhuan"]
CN = {"xiyouji": "西游记", "hongloumeng": "红楼梦",
      "sanguoyanyi": "三国演义", "shuihuzhuan": "水浒传"}
# 小语料模型只训练过这本；其余三本是它没见过的
SEEN_BY_SMALL = "xiyouji"


def eval_loss(model, tok, text: str, seq_len: int,
              stride: int = 128, max_windows: int = 60) -> float:
    """在给定文本上算平均交叉熵（自然对数，与训练同口径）。

    从文本开头顺次切窗口，不走训练时的 split，这样"见过的书"与
    "没见过的书"用的是同一把尺子，数字可以直接比。
    """
    ids = tok.encode(text)
    if len(ids) <= seq_len + 1:
        return float("nan")
    total, count, start = 0.0, 0, 0
    model.eval()
    with torch.no_grad():
        while start + seq_len + 1 <= len(ids) and count < max_windows:
            chunk = torch.tensor([ids[start:start + seq_len + 1]],
                                 dtype=torch.long, device="cuda")
            x, y = chunk[:, :-1], chunk[:, 1:]
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
            total += loss.item()
            count += 1
            start += stride
    return total / max(count, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="结果 JSON 输出路径")
    ap.add_argument("--window-chars", type=int, default=20000,
                    help="每本书取多少字符（取中段，避开各书风格特殊的开头）")
    ap.add_argument("--max-windows", type=int, default=60,
                    help="每本书最多算多少个窗口（限时用）")
    args = ap.parse_args()

    torch.manual_seed(42)
    books_dir = DATA / "books"
    samples = {}
    for b in BOOKS:
        t = (books_dir / f"{b}.txt").read_text(encoding="utf-8")
        mid = len(t) // 2
        samples[b] = t[mid:mid + args.window_chars]

    out = []
    for label, run, corpus in MODELS:
        ckpt = RUNS / run / "ckpt_best.pt"
        if not ckpt.exists():
            ckpt = RUNS / run / "ckpt_last.pt"
        if not ckpt.exists():
            print(f"[SKIP] {label}: 无 checkpoint")
            continue
        ck = torch.load(ckpt, map_location="cuda", weights_only=False)
        exp = ck["config"]["experiment"]
        # tokenizer 必须与该模型训练时一致（词表不同则 id 对不上）
        fname = "corpus_small.txt" if corpus == "small" else "corpus_large.txt"
        tok = CharTokenizer((DATA / fname).read_text(encoding="utf-8"))
        model = SuperMiniGPT(tok.vocab_size, exp["hidden"], exp["layers"],
                             exp["heads"], exp["seq_len"]).cuda()
        model.load_state_dict(ck["model"])

        row = {"label": label, "run": run, "corpus": corpus,
               "params_million": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
               "best_val": ck.get("best_val"),
               "seen_books": ([SEEN_BY_SMALL] if corpus == "small" else BOOKS),
               "losses": {}}
        for b in BOOKS:
            row["losses"][b] = round(eval_loss(model, tok, samples[b],
                                               exp["seq_len"],
                                               max_windows=args.max_windows), 4)
        out.append(row)
        del model
        torch.cuda.empty_cache()
        print(f"[done] {label}")

    # 打印对照表：见过 vs 没见过
    print()
    print(f"{'模型':<16}" + "".join(f"{CN[b]:>11}" for b in BOOKS)
          + f"{'没见过-见过':>13}")
    print("-" * 73)
    for r in out:
        vals = [r["losses"][b] for b in BOOKS]
        if r["corpus"] == "small":
            gap = max(vals[1:]) - vals[0]
        else:
            gap = max(vals) - min(vals)
        ppl = [f"{math.exp(v):.1f}" for v in vals]
        print(f"{r['label']:<16}" + "".join(f"{v:>11.4f}" for v in vals)
              + f"{gap:>13.4f}")
        print(f"{'  ↑困惑度':<16}" + "".join(f"{p:>11}" for p in ppl)
              + f"{'':>13}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\n已写入 {out_path}")


if __name__ == "__main__":
    main()