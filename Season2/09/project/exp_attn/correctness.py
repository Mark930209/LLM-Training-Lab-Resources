"""correctness.py —— naive 与 SDPA 各 backend 的数值一致性（09 篇）。

比速度之前必须先比正确性，否则"快"没有意义。这里对每个 backend 检查两件事：

  1. 前向输出：与 naive 参考实现的最大绝对误差
  2. 反向梯度：对 q/k/v 三个输入的梯度最大绝对误差

容差按 dtype 分档。fp16 的累加误差本来就大，用统一容差会误判；
fp32 用更严的容差。容差值不是拍脑袋定的，correctness 模式会先把
实测误差打出来，文章里的阈值取自实测分布再留余量。

用法：
    python -m exp_attn.correctness
    python -m exp_attn.correctness --seq 512 --dtype fp16 --out results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from exp_attn.naive_attn import naive_attention

BACKENDS = {
    "math": SDPBackend.MATH,
    "efficient": SDPBackend.EFFICIENT_ATTENTION,
    "flash": SDPBackend.FLASH_ATTENTION,
    "cudnn": SDPBackend.CUDNN_ATTENTION,
}

# 容差按 dtype 分档：fp16 尾数只有 10 位，N 越大累加误差越大
TOL = {
    torch.float32: {"fwd": 1e-5, "bwd": 1e-4},
    torch.float16: {"fwd": 2e-2, "bwd": 5e-2},
    torch.bfloat16: {"fwd": 8e-2, "bwd": 2e-1},
}


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    """在 fp32 下比较，避免用低精度去度量低精度的误差。"""
    return float((a.float() - b.float()).abs().max().item())


def _run_one(fn, q, k, v, causal):
    """跑一次前向 + 反向，返回输出与三份梯度。"""
    q = q.detach().clone().requires_grad_(True)
    k = k.detach().clone().requires_grad_(True)
    v = v.detach().clone().requires_grad_(True)
    out = fn(q, k, v, causal)
    # 用同一个标量损失触发反向，保证各 backend 的梯度可比
    out.sum().backward()
    return out.detach(), q.grad.detach(), k.grad.detach(), v.grad.detach()


def check(seq: int, heads: int, head_dim: int, batch: int, dtype: torch.dtype,
          causal: bool) -> dict:
    torch.manual_seed(0)
    shape = (batch, heads, seq, head_dim)
    q = torch.randn(shape, device="cuda", dtype=dtype)
    k = torch.randn(shape, device="cuda", dtype=dtype)
    v = torch.randn(shape, device="cuda", dtype=dtype)

    ref_out, ref_gq, ref_gk, ref_gv = _run_one(naive_attention, q, k, v, causal)
    tol = TOL[dtype]

    report = {
        "config": {
            "batch": batch, "heads": heads, "seq": seq, "head_dim": head_dim,
            "dtype": str(dtype).replace("torch.", ""), "causal": causal,
        },
        "tolerance": tol,
        "backends": {},
    }

    for name, be in BACKENDS.items():
        try:
            with sdpa_kernel(be):
                out, gq, gk, gv = _run_one(
                    lambda a, b, c, cs: F.scaled_dot_product_attention(
                        a, b, c, is_causal=cs),
                    q, k, v, causal,
                )
        except Exception as exc:  # noqa: BLE001 - 不可用本身就是结论
            report["backends"][name] = {
                "status": "unavailable",
                "error": f"{type(exc).__name__}: {exc}"[:220],
            }
            continue

        fwd = _max_abs_diff(out, ref_out)
        bwd = max(_max_abs_diff(gq, ref_gq),
                  _max_abs_diff(gk, ref_gk),
                  _max_abs_diff(gv, ref_gv))
        report["backends"][name] = {
            "status": "ok",
            "fwd_max_abs_diff": round(fwd, 8),
            "bwd_max_abs_diff": round(bwd, 8),
            "fwd_pass": fwd <= tol["fwd"],
            "bwd_pass": bwd <= tol["bwd"],
            "pass": fwd <= tol["fwd"] and bwd <= tol["bwd"],
        }

    ok = [n for n, r in report["backends"].items() if r.get("pass")]
    report["summary"] = {
        "passed": ok,
        "unavailable": [n for n, r in report["backends"].items()
                        if r["status"] == "unavailable"],
        "failed": [n for n, r in report["backends"].items()
                   if r["status"] == "ok" and not r.get("pass")],
    }
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    ap.add_argument("--non-causal", action="store_true")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
             "fp32": torch.float32}[args.dtype]
    report = check(args.seq, args.heads, args.head_dim, args.batch, dtype,
                   not args.non_causal)

    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
