"""e2e_step.py —— 把 attention 的局部收益放回完整训练 step 复验（09 篇）。

microbenchmark 里 attention kernel 快 2 倍，不等于训练快 2 倍。
一个 step 里还有 QKV 投影、FFN、norm、优化器、数据搬运。
attention 占比越小，kernel 收益被稀释得越厉害。

本模块跑真实训练 step（复用 06 篇的 build_llama 与 08 篇的配置口径），
对照三种 attention 实现（名字用 HF 的 attn_implementation 取值）：
    eager              HF 的显式实现，物化 N×N 分数矩阵（等价于本篇 naive 基线）
    sdpa               HF 走 torch SDPA，backend 由 PyTorch 自己挑
    flash_attention_2  需要装 flash-attn 包，没装会明确报错，这本身是条边界

每个实现记录：step 时间、attention 段占比、峰值显存、tok/s、loss。
loss 必须在同一批数据上对齐，否则速度没有可比性。

用法（WSL 项目根目录）：
    python -m exp_attn.e2e_step --impls eager,sdpa
    python -m exp_attn.e2e_step --impls eager,sdpa --seq 2048 --steps 30
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.reproducibility import set_all_seeds  # noqa: E402
from exp_hf.adapters import build_llama  # noqa: E402
from exp_hf.contract import contract_loss  # noqa: E402


def _mb(n: int) -> float:
    return round(n / 1024 / 1024, 1)


# reserved 超过整卡这个比例就怀疑 WSL2 已把显存溢出到主机内存。
# 07 篇实测 8 GB 卡可用约 6.78 GB（占 85%），取 0.60 留余量。
SPILLOVER_WARN_RATIO = 0.60


def build_model(impl: str, vocab: int, hidden: int, layers: int, heads: int,
                head_dim: int, seq: int):
    """按 attn_implementation 搭模型。flash 没装包时 HF 会抛错，原样传出。"""
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=hidden * 8 // 3,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=heads,
        head_dim=head_dim,
        max_position_embeddings=max(seq, 1024),
        tie_word_embeddings=True,
        use_cache=False,
        attn_implementation=impl,
    )
    return LlamaForCausalLM(cfg)


def run(impl: str, args) -> dict:
    set_all_seeds(args.seed)
    rec = {"impl": impl, "seq": args.seq, "batch": args.batch,
           "layers": args.layers, "hidden": args.hidden, "steps": args.steps}

    try:
        model = build_model(impl, args.vocab, args.hidden, args.layers,
                            args.heads, args.head_dim, args.seq)
    except Exception as exc:  # noqa: BLE001 - 装不上 flash-attn 是真实边界
        rec.update({"status": "unavailable",
                    "error": f"{type(exc).__name__}: {exc}"[:300]})
        return rec

    # AMP 的正确姿势：主权重留 fp32，靠 autocast 在前向里降精度。
    # 把整个模型 .to(fp16) 会让参数与梯度都是 fp16，GradScaler.unscale_
    # 直接抛 "Attempting to unscale FP16 gradients"（本篇实测踩到）。
    model = model.to(args.device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95),
                            weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=args.use_amp)

    n_params = sum(p.numel() for p in model.parameters())
    rec["params_m"] = round(n_params / 1e6, 2)
    rec["attn_impl_reported"] = getattr(
        model.config, "_attn_implementation", None)

    # 固定数据：所有 impl 消费完全相同的 token，loss 才可比
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    batches = []
    for _ in range(args.steps + args.warmup):
        ids = torch.randint(0, args.vocab, (args.batch, args.seq + 1), generator=g)
        batches.append(ids.to(args.device))

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    step_times, losses = [], []
    try:
        for i, ids in enumerate(batches):
            x, y = ids[:, :-1], ids[:, 1:]
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=args.torch_dtype,
                                enabled=args.use_amp):
                out = model(x)
                loss = contract_loss(out, y)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            torch.cuda.synchronize()
            dt = time.perf_counter() - t0

            if i >= args.warmup:
                step_times.append(dt)
                losses.append(float(loss.item()))
    except torch.cuda.OutOfMemoryError as exc:
        rec.update({"status": "oom", "error": str(exc)[:220],
                    "peak_mb": _mb(torch.cuda.max_memory_allocated())})
        return rec

    med = statistics.median(step_times)
    toks = args.batch * args.seq
    peak_mb = _mb(torch.cuda.max_memory_allocated())
    reserved_mb = _mb(torch.cuda.memory_reserved())
    rec.update({
        "status": "ok",
        "step_ms_median": round(med * 1000, 2),
        "step_ms_min": round(min(step_times) * 1000, 2),
        "step_ms_max": round(max(step_times) * 1000, 2),
        "tok_per_s": round(toks / med, 0),
        "peak_mb": peak_mb,
        "reserved_mb": reserved_mb,
        "final_loss": round(losses[-1], 4),
        "mean_loss": round(statistics.mean(losses), 4),
        "grad_norm_last": round(float(grad_norm), 4),
    })

    # WSL2 溢出污染检测：reserved 逼近整卡容量时，DXG 会把超额请求溢出到
    # 主机内存，step 时间里混进 PCIe 传输，此时测出的"加速比"不是 kernel
    # 差异。09 篇实测踩过：seq 2048 eager reserved 6662 MB（占整卡 81%），
    # step 24064 ms，对 sdpa 算出 272 倍，画红线后直接 OOM——那 272 倍是
    # 溢出代价，不是 FlashAttention 的收益。任何超过阈值的行都必须带此标记。
    total_mb = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
    reserve_ratio = reserved_mb / total_mb
    rec["reserved_fraction_of_card"] = round(reserve_ratio, 3)
    if reserve_ratio > SPILLOVER_WARN_RATIO:
        spread = max(step_times) / max(min(step_times), 1e-9)
        rec["spillover_suspect"] = True
        rec["spillover_warning"] = (
            "reserved 占整卡 %.0f%%，WSL2 下 DXG 可能已把超额显存溢出到主机"
            "内存，step 时间含 PCIe 传输，本行加速比不可当作 kernel 收益；"
            "step 时间极差 %.1f 倍亦为抖动证据。裸金属上此配置可能直接 OOM，"
            "需用 --mem-fraction 画红线复核。" % (reserve_ratio * 100, spread)
        )

    # attention 占 step 的比例：用 profiler 按 kernel 名归类，
    # 这是"kernel 加速比 ≠ 端到端加速比"的量化依据，不能只靠推断。
    # 包在 try 里：溢出配置在训练循环后已贴近容量上限，再跑一次
    # 带 profiler 的前向反向可能 OOM，那会丢掉整条记录（含溢出标记）。
    # 标记比占比重要，占比测不到就留 None。
    try:
        rec["attn_share"] = attention_share(model, args, impl)
    except torch.cuda.OutOfMemoryError:
        rec["attn_share"] = {
            "impl": impl,
            "attn_fraction": None,
            "note": "profiling 阶段 OOM，无法测 attention 占比；"
                    "该配置已贴近容量上限，本身就是溢出信号",
        }
        torch.cuda.empty_cache()
    return rec


ATTN_KERNEL_SIGS = ("flash", "fmha", "cutlass", "attention", "sdp", "mem_eff")
# eager 路径下 attention 由通用算子拼成，靠模块名而不是 kernel 名归类
EAGER_ATTN_MARKERS = ("attn", "attention", "matmul", "softmax", "bmm")


def attention_share(model, args, impl: str) -> dict:
    """测 attention 相关 kernel 占一个 step 的 device 时间比例。"""
    from torch.profiler import ProfilerActivity, profile

    g = torch.Generator(device="cpu").manual_seed(args.seed + 999)
    ids = torch.randint(0, args.vocab, (args.batch, args.seq + 1),
                        generator=g).to(args.device)
    x, y = ids[:, :-1], ids[:, 1:]

    for _ in range(2):  # 预热
        with torch.autocast("cuda", dtype=args.torch_dtype, enabled=args.use_amp):
            contract_loss(model(x), y).backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        with torch.autocast("cuda", dtype=args.torch_dtype, enabled=args.use_amp):
            contract_loss(model(x), y).backward()
        torch.cuda.synchronize()

    total = 0.0
    attn = 0.0
    for ka in prof.key_averages():
        t = float(ka.self_device_time_total)
        if t <= 0:
            continue
        total += t
        low = ka.key.lower()
        if impl == "eager":
            hit = any(s in low for s in ("softmax", "baddbmm", "bmm"))
        else:
            hit = any(s in low for s in ATTN_KERNEL_SIGS)
        if hit:
            attn += t

    return {
        "impl": impl,
        "device_time_ms_total": round(total / 1000, 2),
        "device_time_ms_attn": round(attn / 1000, 2),
        "attn_fraction": round(attn / total, 4) if total > 0 else None,
        "note": ("eager 按 softmax/bmm 归类，是下界；"
                 "sdpa 按融合 kernel 名归类" if impl == "eager"
                 else "按融合 attention kernel 名归类"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--impls", type=str, default="eager,sdpa")
    ap.add_argument("--vocab", type=int, default=6015)
    ap.add_argument("--hidden", type=int, default=384)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    ap.add_argument("--mem-fraction", type=float, default=None,
                    help="画显存红线模拟裸金属（WSL2 下超额会溢出到主机内存而不是 OOM）")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    args.device = "cuda"
    args.torch_dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    args.use_amp = args.dtype == "fp16"

    if args.mem_fraction is not None:
        torch.cuda.set_per_process_memory_fraction(args.mem_fraction, 0)

    impls = [s.strip() for s in args.impls.split(",") if s.strip()]
    rows = [run(impl, args) for impl in impls]

    report = {
        "device": torch.cuda.get_device_name(0),
        "capability": "%d.%d" % torch.cuda.get_device_capability(0),
        "torch": torch.__version__,
        "dtype": args.dtype,
        "mem_fraction": args.mem_fraction,
        "rows": rows,
    }

    # 端到端加速比只在都跑通的组合之间算
    ok = {r["impl"]: r for r in rows if r["status"] == "ok"}
    if "eager" in ok:
        base = ok["eager"]["step_ms_median"]
        for impl, r in ok.items():
            r["speedup_vs_eager"] = round(base / r["step_ms_median"], 3)

    # 基线自己被溢出污染时，加速比整体不可信，必须在报告里明说
    if ok.get("eager", {}).get("spillover_suspect"):
        report["speedup_unreliable"] = (
            "eager 基线 reserved 占整卡 "
            "%.0f%%，step 时间含 WSL2 主机内存溢出的 PCIe 传输，"
            "speedup_vs_eager 不是 kernel 收益，不可引用。"
            % (ok["eager"]["reserved_fraction_of_card"] * 100)
        )

    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
