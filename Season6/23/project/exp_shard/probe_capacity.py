"""probe_capacity.py —— 23 篇容量靶标校准（固化版，学 22 篇 probe_mem）。

回答一个问题：目标模型在 3070 8 GB 卡上，DDP 全量状态到底放不放得下？
分片后又放不放得下？

四步：
1. build_llama 放大配置构建模型并转 bf16（22 篇"bf16 全参 AdamW"口径），
   count_params 数去重参数量；
2. 实测 AdamW 优化器状态的 dtype 与 bytes/param（不假设，跟 PyTorch 版本走）；
3. state_account 静态账 + 实测三档 bs/seq 的五阶段峰值（DDP 等价的全量
   状态路径），对照 8 GB 定界值；
4. WSL 超配说明：8 GB 卡上分配超物理显存不会 OOM 而是溢出降速（22 篇实测
   tps 1340→408）。--mem-fraction 把 CUDA 分配器预算钉在物理容量（复现
   裸机行为、拿真实 OOM traceback），0 表示不钉。

输出 probe_capacity.json：静态账 + 实测峰值 + 采纳的容量靶标。

用法（rank0 WSL）：
    PYTHONPATH=. python exp_shard/probe_capacity.py --out results/probe_capacity.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exp_shard.shard_common import (  # noqa: E402
    StateAccount, StageProbe, count_params, run_meta)


def build_model(vocab_size: int, hidden: int, layers: int,
                inter: int, seq_len: int, dtype=torch.bfloat16):
    from exp_hf.adapters import build_llama
    heads = max(1, hidden // 64)
    model = build_llama(vocab_size=vocab_size, hidden=hidden, layers=layers,
                        heads=heads, head_dim=64, intermediate=inter,
                        max_seq_len=seq_len)
    return model.to(dtype)


def optim_state_bytes(model: torch.nn.Module) -> dict:
    """实测 AdamW 状态 dtype 与 bytes/param（初始化后统计）。"""
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    n = count_params(model)
    for p in model.parameters():
        if p.grad is None:
            p.grad = torch.zeros_like(p)
    opt.step()
    state_bytes = 0
    dtypes: set[str] = set()
    for state in opt.state.values():
        for v in state.values():
            if torch.is_tensor(v):
                state_bytes += v.numel() * v.element_size()
                dtypes.add(str(v.dtype))
    opt.zero_grad(set_to_none=True)
    return {"optim_state_mb": round(state_bytes / 1024 / 1024, 1),
            "optim_dtype": sorted(dtypes),
            "bytes_per_param": round(state_bytes / n, 2) if n else 0}


def probe_one(model_fn, vocab: int, bs: int, seq: int,
              device: str = "cuda") -> dict | None:
    """全量状态（DDP 等价）单步实测：加载、优化器、fwd、bwd、step 五阶段。"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    probe = StageProbe(device=device)
    try:
        model = model_fn().to(device)
        probe.snap("model_loaded")
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        probe.snap("optim_init")
        ids = torch.randint(0, vocab, (bs, seq + 1), device=device)
        x, t = ids[:, :-1], ids[:, 1:]
        out = model(x, labels=t)
        probe.snap("forward")
        out.loss.backward()
        probe.snap("backward")
        opt.step()
        probe.snap("step")
        opt.zero_grad(set_to_none=True)
        probe.snap("zero_grad")
        peak = probe.records[-1]["peak_mb"]
        ok = True
    except torch.cuda.OutOfMemoryError:
        peak, ok = None, False
        probe.snap("OOM")
    del model, opt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"bs": bs, "seq": seq, "ok": ok, "peak_mb": peak,
            "stages": probe.records}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/probe_capacity.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--vocab", type=int, default=5120)
    ap.add_argument("--hidden", type=int, default=1536)
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--inter", type=int, default=4096)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--mem-fraction", type=float, default=0.9,
                    help="把分配器预算钉在物理显存的比例，0=不钉")
    args = ap.parse_args()

    if args.mem_fraction > 0 and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.mem_fraction)

    def model_fn():
        return build_model(args.vocab, args.hidden, args.layers,
                           args.inter, args.seq_len)

    model = model_fn()
    n_params = count_params(model)
    param_dtype = str(next(model.parameters()).dtype)
    optim_info = optim_state_bytes(model)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # bytes/param 以实测为准：参数 + 梯度（同 dtype）+ 优化器状态
    p_bytes = torch.finfo(torch.float32).bits // 8 if "float32" in param_dtype else 2
    per_param = p_bytes * 2 + optim_info["bytes_per_param"]
    account_rows = {
        "ddp": StateAccount(n_params, 1, per_param).as_dict(),
        "zero1": StateAccount(n_params, 2, per_param, True, False, False).as_dict(),
        "zero2": StateAccount(n_params, 2, per_param, True, True, False).as_dict(),
        "zero3": StateAccount(n_params, 2, per_param, True, True, True).as_dict(),
    }

    # 3070 的 8 GB 定界：WSL 下 CUDA 上下文与驱动开销约 0.5~0.8 GB，
    # 可用按 7.2 GB（7373 MiB）留余量。
    trials = [probe_one(model_fn, args.vocab, bs, seq, args.device)
              for bs, seq in ((2, 512), (2, 1024), (4, 512))]

    out = {
        "mode": "probe_capacity",
        "config": {"vocab": args.vocab, "hidden": args.hidden,
                   "layers": args.layers, "inter": args.inter,
                   "seq_len": args.seq_len, "dtype": param_dtype,
                   "mem_fraction": args.mem_fraction},
        "n_params": n_params,
        "optim_state": optim_info,
        "state_accounts": account_rows,
        "trials": trials,
        "budget_mb": 7373,
        "meta": run_meta(0, 1, vars(args)),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    for t in trials:
        print(f"bs={t['bs']} seq={t['seq']}: "
              f"{'peak ' + str(t['peak_mb']) + ' MiB' if t['ok'] else 'OOM'}")
    print(f"n_params={n_params/1e6:.1f}M dtype={param_dtype} "
          f"per_param={per_param}B optim={optim_info}")
    print(f"ddp_state={account_rows['ddp']['state_mb']} MiB  "
          f"zero2_state={account_rows['zero2']['state_mb']} MiB  "
          f"zero3_state={account_rows['zero3']['state_mb']} MiB")
    print(f"done -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

