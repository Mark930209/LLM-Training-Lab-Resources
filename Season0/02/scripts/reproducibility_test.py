#!/usr/bin/env python3
"""reproducibility_test.py —— LLM Training Lab 通用复现性验证脚本

用途：验证"同 seed 两次运行逐位一致，异 seed 结果发散"，
     这是环境可复现性的最小充分验证（文章 02 篇 §5.3 对照实验）。

用法：
    python reproducibility_test.py [--steps 50] [--output reproducibility.json]

原理：固定 seed 后，模型初始化与数据生成的随机流完全确定；
     两次独立进程运行（而非同进程重跑）才能证明复现不依赖进程内状态。
     因此本脚本用 subprocess 把同一训练跑三遍：seedA-1、seedA-2、seedB。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys

# 单次运行的训练代码（作为子进程执行，保证进程级独立）
RUNNER = r"""
import json, sys, torch, torch.nn as nn
seed, steps, hidden = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
torch.manual_seed(seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
if dev == "cuda":
    torch.use_deterministic_algorithms(False)  # 冒烟级复现；严格逐位一致需 cuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
model = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 10)).to(dev)
opt = torch.optim.SGD(model.parameters(), lr=0.01)
g = torch.Generator(device="cpu").manual_seed(seed)
losses = []
for _ in range(steps):
    x = torch.randn(32, hidden, generator=g).to(dev)
    y = torch.randint(0, 10, (32,), generator=g).to(dev)
    loss = nn.functional.cross_entropy(model(x), y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    losses.append(round(loss.item(), 8))
print(json.dumps(losses))
"""


def run_once(seed: int, steps: int, hidden: int) -> list[float]:
    out = subprocess.run(
        [sys.executable, "-c", RUNNER, str(seed), str(steps), str(hidden)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def digest(losses: list[float]) -> str:
    return hashlib.sha256(json.dumps(losses).encode()).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seed-a", type=int, default=42)
    ap.add_argument("--seed-b", type=int, default=43)
    ap.add_argument("--output", default="reproducibility.json")
    args = ap.parse_args()

    print(f"运行 3 次独立进程: seedA={args.seed_a} ×2, seedB={args.seed_b} ×1, 步数={args.steps}")
    a1 = run_once(args.seed_a, args.steps, args.hidden)
    a2 = run_once(args.seed_a, args.steps, args.hidden)
    b1 = run_once(args.seed_b, args.steps, args.hidden)

    ha1, ha2, hb1 = digest(a1), digest(a2), digest(b1)
    same_seed_identical = a1 == a2
    diff_seed_diverged = a1 != b1

    print(f"seedA 第1次 loss 摘要: {ha1}  首loss={a1[0]}  末loss={a1[-1]}")
    print(f"seedA 第2次 loss 摘要: {ha2}  首loss={a2[0]}  末loss={a2[-1]}")
    print(f"seedB 第1次 loss 摘要: {hb1}  首loss={b1[0]}  末loss={b1[-1]}")
    print(f"同 seed 逐位一致: {same_seed_identical}")
    print(f"异 seed 结果发散: {diff_seed_diverged}")

    result = {
        "seedA运行1": {"摘要": ha1, "首loss": a1[0], "末loss": a1[-1]},
        "seedA运行2": {"摘要": ha2, "首loss": a2[0], "末loss": a2[-1]},
        "seedB运行1": {"摘要": hb1, "首loss": b1[0], "末loss": b1[-1]},
        "同seed逐位一致": same_seed_identical,
        "异seed发散": diff_seed_diverged,
        "步数": args.steps,
        "seedA": args.seed_a,
        "seedB": args.seed_b,
        "完整loss曲线_seedA": a1,
        "完整loss曲线_seedB": b1,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"结果已写入 {args.output}")

    if same_seed_identical and diff_seed_diverged:
        print("REPRODUCIBILITY_OK")
    else:
        print("REPRODUCIBILITY_FAILED: 检查是否有未受 seed 控制的随机源")
        sys.exit(1)


if __name__ == "__main__":
    main()
