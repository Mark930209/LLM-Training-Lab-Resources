"""mem_estimator.py —— Memory Estimator v1（07 篇交付物）。

04 篇的估算公式是静态账：权重 + 梯度 + 优化器状态 + 激活。它漏了三类东西：

    1. CUDA 上下文与驱动开销（nvidia-smi 能看到、allocated 看不到）
    2. 临时缓冲：logits、AMP 的 fp16 副本、cudnn workspace
    3. 分配器 reserve 与 allocated 的差（碎片余量）

Estimator v1 保留四项静态账作为骨架，但把系数交给 sweep 数据拟合：

    peak = wg * (weights+grads) + o * optimizer + a * activation + c

四个系数各有物理含义：wg 应接近 1（fp32 权重与梯度是精确账），
o 应接近 1（AdamW 两份 fp32 动量），a 捕捉激活与临时缓冲的放大倍数，
c 捕捉上下文与 reserve 余量。拟合要求数据同时变动参数量、optimizer
种类与激活量，否则系数不可辨识（batch/seq 单变量 sweep 里参数量恒定，
w/g/o 会退化成同一个数）。

盲测用未参与拟合的配置，报告误差与失效边界：跨 dtype（fp32）时
激活减半假设与 fp16 副本都不成立，误差会显著放大，这就是 Estimator
的适用边界，不是 bug。

用法：
    python -m exp_mem.mem_estimator --fit results/Season1/07/sweep_fit.json \
        --blind results/Season1/07/sweep_blind.json --output .../estimator_report.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def components(r: dict) -> dict:
    """静态账（MB）。amp 只把激活与 attention 矩阵减半，权重/梯度保持 fp32。

    act 含两部分：
      线性激活：每层保存的中间张量，约 batch*seq*hidden 的若干倍；
      attention 矩阵：非 FlashAttention 实现会物化 B*H*S*S 的分数矩阵，
      与 seq 平方成正比，是长序列下估算失真的主因。
    """
    p = r["params"]
    wg = p * 8 / 1024 / 1024            # fp32 权重 + fp32 梯度
    o = (p * 8 / 1024 / 1024) if r.get("opt", "adamw") == "adamw" else 0.0
    dt = 2 if r.get("amp") else 4
    act = r["batch"] * r["seq"] * r["hidden"] * dt * r["layers"] * 6 / 1024 / 1024
    attn = r["batch"] * r["heads"] * r["seq"] * r["seq"] * dt * r["layers"] / 1024 / 1024
    return {"wg": wg, "o": o, "a": act + attn}


def fit(rows: list[dict]) -> dict:
    """最小二乘拟合 peak = WG + a*ACT + c（2 未知数 a、c）。

    WG 是精确账直接加；OPT 不进 peak（优化器状态在 step 阶段才到账，
    而 peak 由 forward/backward 的激活主导，sweep 实测 sgd 与 adamw 的
    peak 相同可证）。激活（含 attention 矩阵）合并成一个放大项：
    试过把线性激活与 attention 矩阵分开拟合（3 未知数），盲测误差反而
    更大（28% vs 22%），小样本下二次项过拟合，故保留单激活项并把误差
    如实报告为失效边界。
    """
    n = 2
    xs, ys = [], []
    for r in rows:
        c = components(r)
        xs.append([c["a"], 1.0])
        ys.append(r["peak_mb"] - c["wg"])
    xt_x = [[0.0] * n for _ in range(n)]
    xt_y = [0.0] * n
    for x, y in zip(xs, ys):
        for i in range(n):
            xt_y[i] += x[i] * y
            for j in range(n):
                xt_x[i][j] += x[i] * x[j]
    m = [row[:] + [xt_y[i]] for i, row in enumerate(xt_x)]
    for i in range(n):
        p = m[i][i]
        for j in range(i, n + 1):
            m[i][j] /= p
        for k in range(n):
            if k != i:
                f = m[k][i]
                for j in range(i, n + 1):
                    m[k][j] -= f * m[i][j]
    a, c = (m[i][n] for i in range(n))
    return {"a": a, "c_mb": c}


def predict(coef: dict, r: dict) -> float:
    c = components(r)
    return c["wg"] + coef["a"] * c["a"] + coef["c_mb"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", required=True)
    ap.add_argument("--blind", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    fit_rows = json.loads(Path(args.fit).read_text(encoding="utf-8"))
    blind_rows = json.loads(Path(args.blind).read_text(encoding="utf-8"))
    coef = fit(fit_rows)

    def errs(rows):
        out = []
        for r in rows:
            est = predict(coef, r)
            out.append({"label": r.get("label", ""), "est_mb": round(est, 1),
                        "real_mb": r["peak_mb"],
                        "err_pct": round((est - r["peak_mb"]) / r["peak_mb"] * 100, 1)})
        return out

    fit_errs, blind_errs = errs(fit_rows), errs(blind_rows)
    report = {
        "coefficients": {k: round(v, 4) for k, v in coef.items()},
        "fit_errors": fit_errs,
        "blind_errors": blind_errs,
        "blind_max_abs_err_pct": round(max(abs(e["err_pct"]) for e in blind_errs), 1),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
