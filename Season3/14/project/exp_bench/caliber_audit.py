"""caliber_audit.py —— 14 篇口径审计：同一批原始日志，五种口径算 speedup。

提纲的核心实验：不改一行训练代码，只换统计口径，
speedup 能从"接近线性"漂到"双卡更慢"。

五种口径：
  A  per-rank batch 当全局 batch（12 篇 fault_global_batch 的性能版）
  B  含 warmup 的均值
  C  不含 warmup 的均值
  D  不含 warmup 的中位数（本篇标准口径）
  E  不含 warmup 的 P95（尾部口径）

用法：先跑 single 和 ddp 落盘，再：
  python -m exp_bench.caliber_audit --single results/.../single.json --dual results/.../ddp.json
"""

from __future__ import annotations

import argparse
import json

from .bench_common import write_report


def load(path: str) -> dict:
    return json.load(open(path, encoding="utf-8"))


def caliber_speedups(single: dict, dual: dict) -> dict:
    s_steps = single["steps"]
    d_steps = dual["steps"]
    warmup_s = single["config"].get("warmup", 0)
    warmup_d = dual["config"].get("warmup", 0)

    def total(steps):
        return [st["total_ms"] for st in steps]

    def trimmed(steps, w):
        return total(steps)[w:]

    s_all, d_all = total(s_steps), total(d_steps)
    s_trim, d_trim = trimmed(s_steps, warmup_s), trimmed(d_steps, warmup_d)

    def mean(xs):
        return sum(xs) / len(xs)

    def median(xs):
        xs = sorted(xs)
        return xs[len(xs) // 2]

    def p95(xs):
        xs = sorted(xs)
        return xs[int(len(xs) * 0.95) - 1 if len(xs) > 1 else 0]

    out = {}

    # 口径 A：把 per-rank batch 当全局 batch。
    # dual 的全局 batch 实际是 single 的 2 倍（每卡 16），工作量翻倍。
    # 错误算法：直接比"每卡看到的 batch"的耗时 → speedup 虚高
    gb_s = single["config"]["global_batch"]
    gb_d = dual["config"]["global_batch"]
    per_rank_d = gb_d // dual["world"]
    # 正确 speedup（同全局 batch）
    out["correct_same_global_batch"] = mean(s_trim) / mean(d_trim) if gb_s == gb_d else None
    # 口径 A 的错误：dual 每卡只处理 gb_d/world，拿它当"等效 batch 16"
    # 如果 dual 配置就是 per_rank=16（gb=32），错误口径会拿它跟 single gb=16 比
    out["A_per_rank_as_global"] = (mean(s_trim) / mean(d_trim)) * (gb_d / gb_s) if gb_d != gb_s else None

    out["B_mean_with_warmup"] = mean(s_all) / mean(d_all)
    out["C_mean_no_warmup"] = mean(s_trim) / mean(d_trim)
    out["D_median_no_warmup"] = median(s_trim) / median(d_trim)
    out["E_p95_no_warmup"] = p95(s_trim) / p95(d_trim)

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", required=True)
    ap.add_argument("--dual", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    single, dual = load(args.single), load(args.dual)
    speedups = caliber_speedups(single, dual)

    vals = [v for v in speedups.values() if v is not None]
    payload = {
        "single_file": args.single,
        "dual_file": args.dual,
        "single_global_batch": single["config"]["global_batch"],
        "dual_global_batch": dual["config"]["global_batch"],
        "dual_world": dual["world"],
        "speedups": speedups,
        "speedup_range": [min(vals), max(vals)],
        "note": "同一批原始日志，五种口径的 speedup 取值区间。区间越宽，说明口径越先于优化失效。",
    }
    print(json.dumps(payload, ensure_ascii=False, indent=1))
    if args.out:
        write_report(args.out, payload)


if __name__ == "__main__":
    main()
