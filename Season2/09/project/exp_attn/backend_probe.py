"""backend_probe.py —— 判断 SDPA 实际走了哪个 backend（09 篇交付工具）。

核心问题："我调用了 scaled_dot_product_attention" 和 "我跑了 FlashAttention"
是两件事。默认路径下 PyTorch 自己挑 backend，挑不到 flash 就静默换成
memory-efficient 或 math，不报错、不警告，只有速度不一样。

两种检测手段，互相印证：
  1. 可用性探测：用 sdpa_kernel 逐个强制启用，看哪些不抛异常
     —— 回答"这张卡 + 这个 shape 支持哪些 backend"
  2. profiler 内核名识别：跑一次默认路径，抓 CUDA kernel 名字
     —— 回答"刚才那一次实际用的是哪个"

用法：
    python -m exp_attn.backend_probe
    python -m exp_attn.backend_probe --dtype fp32 --seq 1024
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.profiler import ProfilerActivity, profile

BACKENDS = {
    "flash": SDPBackend.FLASH_ATTENTION,
    "efficient": SDPBackend.EFFICIENT_ATTENTION,
    "cudnn": SDPBackend.CUDNN_ATTENTION,
    "math": SDPBackend.MATH,
}

# profiler 抓到的 CUDA kernel 名字里的特征子串 → backend 名
# 不同 torch/CUDA 版本命名有差异，所以每个 backend 给多个候选
KERNEL_SIGNATURES = {
    "flash": ("flash_fwd", "flash_bwd", "fmha_v2", "FlashAttn"),
    "efficient": ("cutlass", "mem_efficient", "efficient_attention", "fmha_cutlassF", "xformers"),
    "cudnn": ("cudnn", "flash_attention_cudnn"),
    # math 没有专属 kernel，它退化成一串通用算子，靠"出现 softmax/gemm 且无上面三类"判定
}
MATH_SIGNATURES = ("softmax", "gemm", "cutlass_", "vectorized_elementwise", "reduce_kernel")


def _make_inputs(b: int, h: int, n: int, d: int, dtype: torch.dtype, causal: bool):
    torch.manual_seed(0)
    shape = (b, h, n, d)
    q = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=False)
    k = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=False)
    v = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=False)
    return q, k, v


def probe_availability(b: int, h: int, n: int, d: int, dtype: torch.dtype,
                       causal: bool) -> dict:
    """逐个强制启用 backend，记录哪些真的能跑。"""
    q, k, v = _make_inputs(b, h, n, d, dtype, causal)
    result = {}
    for name, be in BACKENDS.items():
        try:
            with sdpa_kernel(be):
                out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
            torch.cuda.synchronize()
            result[name] = {"available": True, "dtype": str(out.dtype)}
        except Exception as exc:  # noqa: BLE001 - 探针要留下全部失败原因
            result[name] = {
                "available": False,
                "error": f"{type(exc).__name__}: {exc}"[:220],
            }
    return result


def classify_kernel(kname: str) -> str | None:
    """按 kernel 名字判定 backend，判不出返回 None。"""
    for backend, sigs in KERNEL_SIGNATURES.items():
        if any(s.lower() in kname.lower() for s in sigs):
            return backend
    return None


def detect_active_backend(b: int, h: int, n: int, d: int, dtype: torch.dtype,
                          causal: bool, forced: str | None = None) -> dict:
    """跑一次前向，用 profiler 抓 kernel 名字判定实际 backend。

    forced=None 时走默认路径，回答"PyTorch 自己会挑哪个"；
    forced 给 backend 名时，在 sdpa_kernel 上下文里跑，回答"我强制的这个
    真的生效了吗"。两者必须分开——默认路径永远报 PyTorch 的偏好，
    拿它去标注强制运行的结果会把 efficient/cudnn 全标成 flash。
    """
    q, k, v = _make_inputs(b, h, n, d, dtype, causal)

    def _once():
        if forced is None:
            return F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        with sdpa_kernel(BACKENDS[forced]):
            return F.scaled_dot_product_attention(q, k, v, is_causal=causal)

    # 预热，避免首次调用的 autotune/编译内核混进采样
    for _ in range(3):
        _once()
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(5):
            _once()
        torch.cuda.synchronize()

    cuda_names = []
    for ev in prof.events():
        # 只取真正落到 device 上的 kernel
        if str(ev.device_type) == "DeviceType.CUDA" and ev.self_device_time_total > 0:
            cuda_names.append(ev.name)
    if not cuda_names:
        # 某些版本 events() 的 device_type 表现不同，退回用 key_averages
        for ka in prof.key_averages():
            if ka.self_device_time_total > 0 and classify_kernel(ka.key):
                cuda_names.extend([ka.key] * max(1, int(ka.count)))

    counts = Counter(cuda_names)
    votes = Counter()
    for kname, c in counts.items():
        backend = classify_kernel(kname)
        if backend:
            votes[backend] += c

    if votes:
        active = votes.most_common(1)[0][0]
        evidence = "kernel-name"
    else:
        # 没有任何 flash/efficient/cudnn 特征 kernel，说明走的是 math 分解路径
        has_math_ops = any(
            any(s.lower() in kname.lower() for s in MATH_SIGNATURES)
            for kname in counts
        )
        active = "math" if has_math_ops else "unknown"
        evidence = "absence-of-fused-kernel" if has_math_ops else "no-signal"

    out = {
        "active_backend": active,
        "evidence": evidence,
        "backend_votes": dict(votes),
        "distinct_kernels": len(counts),
        "top_kernels": [
            {"name": n_[:110], "count": c} for n_, c in counts.most_common(6)
        ],
    }
    if forced is not None:
        # 强制与实测不一致就是静默回退，这是本篇要抓的现象
        out["forced"] = forced
        out["forced_honored"] = (active == forced)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    ap.add_argument("--non-causal", action="store_true")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    causal = not args.non_causal

    report = {
        "config": {
            "batch": args.batch, "heads": args.heads, "seq": args.seq,
            "head_dim": args.head_dim, "dtype": args.dtype, "causal": causal,
        },
        "device": torch.cuda.get_device_name(0),
        "capability": "%d.%d" % torch.cuda.get_device_capability(0),
        "torch": torch.__version__,
        "availability": probe_availability(args.batch, args.heads, args.seq,
                                           args.head_dim, dtype, causal),
        "default_path": detect_active_backend(args.batch, args.heads, args.seq,
                                              args.head_dim, dtype, causal),
    }

    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        from pathlib import Path
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
