#!/usr/bin/env python3
"""estimate_memory.py —— 显存账：启动前算出"会不会 OOM"。

不跑训练，纯计算。这是 04 篇"先算账再动手"的可复现工具：
把 100M 模型在 8GB 卡上的显存逐项拆开，让"为什么 OOM"和
"该调哪个参数"变成可以算出来的事实。

用法（WSL2 内，工程根目录）：
    python scripts/estimate_memory.py
    python scripts/estimate_memory.py --hidden 768 --layers 12 --batch 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exp_scale.schedulers import (  # noqa: E402
    estimate_model_params, estimate_training_memory_mb)

# 三档模型配置（与 config_*.yaml 对齐）
CONFIGS = {
    "10M":  dict(hidden=384, layers=6,  heads=6,  batch=16, seq=256),
    "30M":  dict(hidden=576, layers=8,  heads=8,  batch=8,  seq=256),
    "100M": dict(hidden=768, layers=12, heads=12, batch=4,  seq=256),
}
VOCAB = 5000  # 四大名著 char-level 词表量级


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--vocab", type=int, default=VOCAB)
    ap.add_argument("--amp", action="store_true", default=True)
    args = ap.parse_args()

    if args.hidden:
        cfgs = {"custom": dict(hidden=args.hidden, layers=args.layers,
                               heads=args.layers, batch=args.batch,
                               seq=args.seq)}
    else:
        cfgs = CONFIGS

    rows = []
    for name, c in cfgs.items():
        params = estimate_model_params(args.vocab, c["hidden"], c["layers"])
        mem = estimate_training_memory_mb(
            params, c["seq"], c["batch"], c["hidden"], c["layers"],
            args.vocab, amp=args.amp)
        row = {"name": name, "params_million": round(params / 1e6, 2),
               "batch": c["batch"], "seq": c["seq"], **mem}
        rows.append(row)
        print(f"\n=== {name} (hidden={c['hidden']}, layers={c['layers']}, "
              f"batch={c['batch']}, seq={c['seq']}) ===")
        print(f"  参数量: {row['params_million']}M")
        print(f"  权重:     {mem['weights_mb']:>8.1f} MB")
        print(f"  梯度:     {mem['grads_mb']:>8.1f} MB")
        print(f"  优化器:   {mem['optimizer_mb']:>8.1f} MB")
        print(f"  激活:     {mem['activations_mb']:>8.1f} MB")
        print(f"  其他:     {mem['overhead_mb']:>8.1f} MB")
        print(f"  ────────────────────────")
        print(f"  合计:     {mem['total_mb']:>8.1f} MB "
              f"({mem['total_mb']/1024:.2f} GB)")

    print("\n=== 8GB 卡（实际可用约 6.78GB）判读 ===")
    for r in rows:
        gb = r["total_mb"] / 1024
        verdict = "可跑" if gb < 6.0 else ("临界" if gb < 6.78 else "会 OOM")
        print(f"  {r['name']:>6}: {gb:.2f} GB → {verdict}")

    out = Path("memory_estimate.json")
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n结果已写入 {out}")


if __name__ == "__main__":
    main()