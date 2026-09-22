"""attn_bench.py —— Attention backend 与 shape sweep（09 篇核心交付物）。

回答三个问题：
  1. 同一个 shape 下，naive / math / efficient / flash / cudnn 各花多少时间、多少显存
  2. 序列长度从短到长，差距在哪个点开始拉开
  3. head_dim、dtype、causal 变化时，哪些 backend 直接不可用或静默回退

每次测量都记录三件事：时间、峰值显存、**实际生效的 backend**。
第三项是关键——只报时间不报 backend，读者无法判断"快"是因为换了实现，
还是因为 profiler 抓到的根本不是同一个 kernel。

显存口径沿用 07 篇：max_memory_allocated，单位 MB。

用法（WSL 项目根目录）：
    python -m exp_attn.attn_bench --mode backend
    python -m exp_attn.attn_bench --mode seq --impls naive,math,efficient,flash
    python -m exp_attn.attn_bench --mode shape
    python -m exp_attn.attn_bench --mode dtype
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exp_attn.backend_probe import detect_active_backend  # noqa: E402
from exp_attn.naive_attn import naive_attention, score_matrix_bytes  # noqa: E402

SDPA_BACKENDS = {
    "math": SDPBackend.MATH,
    "efficient": SDPBackend.EFFICIENT_ATTENTION,
    "flash": SDPBackend.FLASH_ATTENTION,
    "cudnn": SDPBackend.CUDNN_ATTENTION,
}

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def _mb(n: int) -> float:
    return round(n / 1024 / 1024, 1)


def _run_impl(impl: str, q, k, v, causal: bool):
    """按 impl 名字跑一次前向，返回输出张量。"""
    if impl == "naive":
        return naive_attention(q, k, v, is_causal=causal)
    with sdpa_kernel(SDPA_BACKENDS[impl]):
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


def measure(impl: str, batch: int, heads: int, seq: int, head_dim: int,
            dtype: torch.dtype, causal: bool, iters: int = 20,
            warmup: int = 5) -> dict:
    """测一个 (impl, shape) 组合的前向/反向时间与峰值显存。"""
    torch.manual_seed(0)
    shape = (batch, heads, seq, head_dim)
    rec = {
        "impl": impl, "batch": batch, "heads": heads, "seq": seq,
        "head_dim": head_dim, "dtype": str(dtype).replace("torch.", ""),
        "causal": causal,
        "score_matrix_mb": round(score_matrix_bytes(batch, heads, seq, dtype), 1),
    }

    try:
        # 预热：首次调用含 autotune 与 kernel 编译，不计入
        for _ in range(warmup):
            q = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
            k = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
            v = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
            _run_impl(impl, q, k, v, causal).sum().backward()
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001 - 跑不动本身就是结论
        rec.update({"status": "unavailable",
                    "error": f"{type(exc).__name__}: {exc}"[:220]})
        return rec

    fwd_times, bwd_times = [], []
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    try:
        for _ in range(iters):
            q = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
            k = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
            v = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
            torch.cuda.synchronize()

            s = torch.cuda.Event(enable_timing=True)
            m = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)

            s.record()
            out = _run_impl(impl, q, k, v, causal)
            m.record()
            out.sum().backward()
            e.record()
            torch.cuda.synchronize()

            fwd_times.append(s.elapsed_time(m))
            bwd_times.append(m.elapsed_time(e))
    except torch.cuda.OutOfMemoryError as exc:
        rec.update({"status": "oom", "error": str(exc)[:220]})
        return rec

    rec.update({
        "status": "ok",
        "fwd_ms_median": round(statistics.median(fwd_times), 3),
        "bwd_ms_median": round(statistics.median(bwd_times), 3),
        "total_ms_median": round(statistics.median(fwd_times)
                                 + statistics.median(bwd_times), 3),
        "fwd_ms_min": round(min(fwd_times), 3),
        "peak_mb": _mb(torch.cuda.max_memory_allocated()),
        "reserved_mb": _mb(torch.cuda.memory_reserved()),
    })

    # 实际生效的 backend：naive 是自己写的，其余强制启用后用 profiler 确认。
    # 必须传 forced=impl，否则探测走默认路径，PyTorch 的偏好会把
    # efficient/cudnn 全标成 flash（本篇踩过这个坑）。
    if impl == "naive":
        rec["active_backend"] = "naive-materialized"
    else:
        det = detect_active_backend(batch, heads, seq, head_dim, dtype, causal,
                                    forced=impl)
        rec["active_backend"] = det["active_backend"]
        rec["backend_evidence"] = det["evidence"]
        rec["forced_honored"] = det["forced_honored"]
    return rec


def sweep(mode: str, impls: list[str], args) -> list[dict]:
    """按 mode 生成 shape 组合并逐个测量。"""
    dtype = DTYPES[args.dtype]
    rows = []

    if mode == "backend":
        for impl in impls:
            rows.append(measure(impl, args.batch, args.heads, args.seq,
                                args.head_dim, dtype, not args.non_causal,
                                args.iters))

    elif mode == "seq":
        for seq in args.seq_list:
            for impl in impls:
                rows.append(measure(impl, args.batch, args.heads, seq,
                                    args.head_dim, dtype, not args.non_causal,
                                    args.iters))

    elif mode == "shape":
        for hd in args.head_dim_list:
            for impl in impls:
                rows.append(measure(impl, args.batch, args.heads, args.seq,
                                    hd, dtype, not args.non_causal, args.iters))

    elif mode == "dtype":
        for dt_name in args.dtype_list:
            for impl in impls:
                rows.append(measure(impl, args.batch, args.heads, args.seq,
                                    args.head_dim, DTYPES[dt_name],
                                    not args.non_causal, args.iters))

    elif mode == "causal":
        for causal in (True, False):
            for impl in impls:
                rows.append(measure(impl, args.batch, args.heads, args.seq,
                                    args.head_dim, dtype, causal, args.iters))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("backend", "seq", "shape", "dtype", "causal"),
                    default="backend")
    ap.add_argument("--impls", type=str, default="naive,math,efficient,flash,cudnn")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--dtype", choices=tuple(DTYPES), default="fp16")
    ap.add_argument("--non-causal", action="store_true")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--seq-list", type=str, default="128,256,512,1024,2048,4096")
    ap.add_argument("--head-dim-list", type=str, default="32,64,128,256")
    ap.add_argument("--dtype-list", type=str, default="fp16,bf16,fp32")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    impls = [s.strip() for s in args.impls.split(",") if s.strip()]
    seq_list = [int(s) for s in args.seq_list.split(",")]
    args.seq_list = seq_list
    args.head_dim_list = [int(s) for s in args.head_dim_list.split(",")]
    args.dtype_list = [s.strip() for s in args.dtype_list.split(",")]

    rows = sweep(args.mode, impls, args)
    report = {
        "mode": args.mode,
        "device": torch.cuda.get_device_name(0),
        "capability": "%d.%d" % torch.cuda.get_device_capability(0),
        "torch": torch.__version__,
        "iters": args.iters,
        "rows": rows,
    }

    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
