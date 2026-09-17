"""opt_bench.py —— 显存优化基准（08 篇核心交付物）。

07 篇解释了显存花在哪里；本篇回答：目标配置 OOM 时先改哪一项，
能省多少显存，付出多少速度与收敛代价。

六种优化开关（可组合）：
    accum      小 batch + 梯度累积（用时间换 batch，不改计算图）
    amp        fp16 混合精度（激活与临时张量减半）
    ckpt       activation checkpointing（用重算换激活）
    lowstate   fp16 优化器状态（状态字节减半，自实现 AdamW16）
    offload    优化器状态 CPU offload（用 PCIe 传输换 GPU 容量）
    setnone    zero_grad(set_to_none=True)（释放梯度张量而非填零）

每个组合记录：分阶段 peak（复用 07 的口径）、tok/s、最终 loss。
另有两个对照模式：
    parity     同 token 预算下 baseline 与组合方案的 loss 对照
    oom        原始配置真实 OOM 的阶段证据

用法（WSL 项目根目录）：
    python -m exp_opt.opt_bench --mode bench --opts baseline
    python -m exp_opt.opt_bench --mode bench --opts accum,amp,ckpt,setnone
    python -m exp_opt.opt_bench --mode oom
    python -m exp_opt.opt_bench --mode parity --opts accum,amp,ckpt,lowstate,setnone
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.reproducibility import set_all_seeds  # noqa: E402
from exp_hf.adapters import build_llama  # noqa: E402
from exp_hf.contract import contract_loss  # noqa: E402


def _mb(n: int) -> float:
    return round(n / 1024 / 1024, 1)


def _snap(tag: str) -> dict:
    return {
        "stage": tag,
        "allocated_mb": _mb(torch.cuda.memory_allocated()),
        "reserved_mb": _mb(torch.cuda.memory_reserved()),
        "peak_mb": _mb(torch.cuda.max_memory_allocated()),
    }


class AdamW16(torch.optim.Optimizer):
    """AdamW 的 fp16 状态版：动量 m/v 存 fp16，状态字节减半。

    主权重保持 fp32（数值口径不变），只把状态降精度。
    这是"低状态优化器"的最小可复现实现，不依赖 bitsandbytes。
    """

    def __init__(self, params, lr=3e-4, betas=(0.9, 0.95), eps=1e-8,
                 weight_decay=0.1):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p, dtype=torch.float16)
                    state["v"] = torch.zeros_like(p, dtype=torch.float16)
                state["step"] += 1
                m, v = state["m"], state["v"]
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                # 计算用 fp32 视图，避免 fp16 累加误差进入更新量
                mf = m.float()
                vf = v.float()
                bc1 = 1 - b1 ** state["step"]
                bc2 = 1 - b2 ** state["step"]
                denom = (vf / bc2).sqrt_().add_(group["eps"])
                upd = (mf / bc1) / denom
                if group["weight_decay"]:
                    upd.add_(p, alpha=group["weight_decay"])
                p.add_(upd, alpha=-group["lr"])
        return None


class OffloadAdamW(torch.optim.AdamW):
    """优化器状态 CPU offload：step 前搬回 GPU，step 后搬回 CPU。

    简化版 ZeRO-Offload：只 offload 状态，不 offload 梯度与参数。
    代价是每步两次 PCIe 传输，状态越大传输越贵。
    """

    def __init__(self, params, **kw):
        super().__init__(params, **kw)
        self._cpu = True

    def _to(self, device):
        for _, state in self.state.items():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device, non_blocking=False)
        self._cpu = device == torch.device("cpu")

    @torch.no_grad()
    def step(self, closure=None):
        self._to(torch.device("cuda"))
        try:
            return super().step(closure)
        finally:
            self._to(torch.device("cpu"))


class CkptBlock(nn.Module):
    """把单个 Transformer block 包成 activation checkpointing 单元。"""

    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def forward(self, *args, **kwargs):
        if self.training:
            return torch_checkpoint(self.block, *args,
                                    use_reentrant=False, **kwargs)
        return self.block(*args, **kwargs)


def build_model(vocab_size, hidden, layers, heads, seq_len, use_ckpt, device):
    model = build_llama(vocab_size, hidden=hidden, layers=layers, heads=heads,
                        head_dim=hidden // heads, intermediate=int(hidden * 8 / 3),
                        max_seq_len=max(seq_len, 512))
    if use_ckpt:
        model.model.layers = nn.ModuleList(
            CkptBlock(b) for b in model.model.layers)
    return model.to(device)


def bench(opts: set, vocab_size: int, hidden: int, layers: int, heads: int,
          seq_len: int, batch: int, steps: int,
          mem_fraction: float | None = None, device: str = "cuda") -> dict:
    if mem_fraction is not None:
        torch.cuda.set_per_process_memory_fraction(
            mem_fraction, torch.cuda.current_device())
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    set_all_seeds(42, deterministic=False)

    use_amp = "amp" in opts
    use_accum = "accum" in opts
    use_ckpt = "ckpt" in opts
    use_low = "lowstate" in opts
    use_off = "offload" in opts
    set_none = "setnone" in opts

    accum = 4 if use_accum else 1
    micro_batch = max(1, batch // accum)

    model = build_model(vocab_size, hidden, layers, heads, seq_len, use_ckpt, device)
    timeline = [_snap("model_loaded")]

    if use_off:
        opt = OffloadAdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    elif use_low:
        opt = AdamW16(model.parameters(), lr=3e-4, weight_decay=0.1)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    timeline.append(_snap("optimizer_init"))

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    t0 = time.perf_counter()
    tokens = 0
    model.train()
    oom = None
    oom_step = None
    try:
        for step in range(steps):
            # 每步新 batch：loss 才有对照意义（同 batch 复用会变成背单批）
            x = torch.randint(0, vocab_size, (micro_batch, seq_len), device=device)
            y = torch.randint(0, vocab_size, (micro_batch, seq_len), device=device)
            opt.zero_grad(set_to_none=set_none)
            for _ in range(accum):
                with torch.autocast(device_type="cuda", dtype=torch.float16,
                                    enabled=use_amp):
                    loss = contract_loss(model(x), y) / accum
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            timeline.append(_snap(f"backward_{step}"))
            if scaler is not None:
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if scaler is not None:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            tokens += micro_batch * seq_len * accum
    except torch.cuda.OutOfMemoryError as exc:
        oom = str(exc)[:200]
        oom_step = step
    wall = time.perf_counter() - t0
    timeline.append(_snap("done"))

    peak = _mb(torch.cuda.max_memory_allocated())
    reserved = _mb(torch.cuda.memory_reserved())
    del model, opt, scaler, x, y
    torch.cuda.empty_cache()
    return {
        "opts": sorted(opts),
        "steps": steps if oom is None else oom_step,
        "batch_effective": micro_batch * accum,
        "wall_time_s": round(wall, 2),
        "tok_per_s": round(tokens / wall, 0) if wall > 0 else 0,
        "peak_mb": peak,
        "reserved_mb": reserved,
        "final_loss": round(loss.item() * accum, 4) if oom is None else None,
        "oom": oom,
        "timeline": timeline[:6] + timeline[-2:],
    }


def oom_evidence(vocab_size, hidden, layers, heads, seq_len, batch,
                 mem_fraction: float | None = None,
                 device: str = "cuda") -> dict:
    """原始配置真实 OOM：记录失败发生在哪个阶段。

    WSL2 的 DXG 驱动会把超出 VRAM 的分配 overcommit 到主机内存，
    裸金属上必 OOM 的配置在这里只会慢、不会崩。mem_fraction 给进程
    画显存红线，超出即真实抛 OutOfMemoryError，等价于模拟更小的卡。
    """
    if mem_fraction is not None:
        torch.cuda.set_per_process_memory_fraction(
            mem_fraction, torch.cuda.current_device())
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = build_model(vocab_size, hidden, layers, heads, seq_len, False, device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    timeline = [_snap("model_loaded"), _snap("optimizer_init")]
    oom = None
    stage = None
    x = torch.randint(0, vocab_size, (batch, seq_len), device=device)
    y = torch.randint(0, vocab_size, (batch, seq_len), device=device)
    try:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=False):
            loss = contract_loss(model(x), y)
        timeline.append(_snap("forward"))
        loss.backward()
        timeline.append(_snap("backward"))
        opt.step()
        timeline.append(_snap("step"))
    except torch.cuda.OutOfMemoryError as exc:
        oom = str(exc)[:300]
        stage = timeline[-1]["stage"] + " 之后"
    snapshot = torch.cuda.memory._snapshot()
    largest_inactive, free_total = 0, 0
    for s in snapshot.get("segments", []):
        for b in s.get("blocks", []):
            if b.get("state") == "inactive":
                largest_inactive = max(largest_inactive, b.get("size", 0))
                free_total += b.get("size", 0)
    out = {
        "config": {"hidden": hidden, "layers": layers, "seq": seq_len,
                   "batch": batch, "amp": False},
        "oom": oom,
        "oom_stage": stage,
        "timeline": timeline,
        "largest_inactive_block_mb": _mb(largest_inactive),
        "free_total_mb": _mb(free_total),
        "reserved_mb": _mb(torch.cuda.memory_reserved()),
    }
    del model, opt, x, y
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="bench", choices=["bench", "oom", "parity"])
    ap.add_argument("--opts", default="baseline")
    ap.add_argument("--vocab", type=int, default=6015)
    ap.add_argument("--hidden", type=int, default=768)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--heads", type=int, default=12)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--mem-fraction", type=float, default=None)
    ap.add_argument("--output", default="/tmp/opt_bench.json")
    args = ap.parse_args()

    if args.mode == "oom":
        out = oom_evidence(args.vocab, args.hidden, args.layers, args.heads,
                           args.seq, args.batch, args.mem_fraction)
    else:
        opts = set() if args.opts == "baseline" else set(args.opts.split(","))
        out = bench(opts, args.vocab, args.hidden, args.layers, args.heads,
                    args.seq, args.batch, args.steps, args.mem_fraction)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "timeline"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
