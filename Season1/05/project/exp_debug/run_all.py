"""run_all.py —— 05 篇实验矩阵一键运行：基线 + 12 个故障 + 续训对照。

实验矩阵（对应提纲）：
    Baseline        健康训练 300 步，采集全部诊断指标
    Data faults     label_shift / label_shuffle / dup_batch / val_leak
    Update faults   lr_zero / lr_huge / no_clip / amp_overflow
    Resume faults   完整恢复 vs 逐项漏恢复（opt / scaler / rng / sched）
    Seed faults     resume_rng 即 seed 对照（RNG 指纹 + 后续 batch 差异）

总耗时估算（RTX 3070，10M 档，300 步 ≈ 20s/组）：
    14 组 × ~20s ≈ 5 分钟；续训对照 6 组 × ~20s ≈ 2 分钟。合计 < 10 分钟。

用法（在 project/ 目录下）：
    python -m exp_debug.run_all --config exp_scale/config_10m.yaml --out results_debug.json
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import config as cfg_mod  # noqa: E402
from exp_debug import train_debug  # noqa: E402
from exp_debug.fail_modes import DATA_FAULTS, RESUME_FAULTS, UPDATE_FAULTS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--resume-steps", type=int, default=150,
                    help="续训对照的断点步数")
    ap.add_argument("--out", default="results_debug.json")
    args = ap.parse_args()

    cfg = cfg_mod.load_config(args.config, [])
    results = {}

    # ---- 1. 健康基线 ----
    print("== baseline ==")
    results["baseline"] = train_debug.run(cfg, args.steps)

    # ---- 2. 数据故障（silent 类主角）----
    for fault in DATA_FAULTS:
        print(f"== {fault} ==")
        results[fault] = train_debug.run(cfg, args.steps, fault=fault)

    # ---- 3. 优化故障 ----
    for fault in UPDATE_FAULTS:
        print(f"== {fault} ==")
        results[fault] = train_debug.run(cfg, args.steps, fault=fault)

    # ---- 4. 续训对照：完整恢复 vs 逐项漏恢复 ----
    with tempfile.TemporaryDirectory() as td:
        ckpt = str(Path(td) / "mid.pt")
        print(f"== 连续训练 {args.resume_steps}+{args.steps - args.resume_steps} 步（对照基准）==")
        results["continuous"] = train_debug.run(
            cfg, args.resume_steps, save_ckpt=ckpt)
        # 完整恢复的续训：应与连续训练一致
        print("== resume_full ==")
        results["resume_full"] = train_debug.run(
            cfg, args.steps, resume=ckpt)
        # 逐项漏恢复
        for fault in RESUME_FAULTS:
            print(f"== {fault} ==")
            results[fault] = train_debug.run(
                cfg, args.steps, fault=fault, resume=ckpt)

    Path(args.out).write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"全部 {len(results)} 组结果已写入 {args.out}")


if __name__ == "__main__":
    main()