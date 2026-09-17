"""harness_on_hf.py —— 05 篇 Correctness Harness 在 HF 模型上的复用验证（06 篇）。

05 篇的门禁依赖 model.named_parameters()、p.grad 与张量字节三个接口，
与模型是不是 HF 的无关。本脚本把 Harness 挂到 llama+BPE 的短训上，
验证两件事：

    1. 健康训练下 Harness 不报错、不误报（harness_ok 为 True，alerts 为空）；
    2. 记录 HF 路径的基线指标（初始 loss、grad_norm、update_ratio），
       作为后续故障实验的校准参照（06 篇提纲 §6.1 的要求）。

用法（WSL 项目根目录）：
    python -m exp_hf.harness_on_hf --steps 300 --amp
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.reproducibility import set_all_seeds  # noqa: E402
from exp_debug.correctness_harness import HarnessConfig, HarnessRunner  # noqa: E402
from exp_debug.diagnostics import snapshot_params, update_ratio  # noqa: E402
from exp_hf.adapters import build_llama  # noqa: E402
from exp_hf.contract import contract_loss  # noqa: E402
from exp_hf.tokenize_bpe import load_fast_tokenizer, train_bpe  # noqa: E402
from exp_hf.train_hf import TokenDataset, encode_all  # noqa: E402
from exp_scale.schedulers import cosine_with_warmup  # noqa: E402


def run(args) -> dict:
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    set_all_seeds(args.seed, deterministic=False)

    data_dir = Path(args.data_dir)
    corpus_text = (data_dir / "corpus_large.txt").read_text(encoding="utf-8")
    json_path = data_dir / "tokenizer_bpe.json"
    train_bpe(data_dir / "corpus_large.txt", json_path)
    fast = load_fast_tokenizer(json_path)
    vocab_size = fast.vocab_size

    ids = encode_all(fast, "bpe", corpus_text)
    cut = int(len(ids) * 0.9)
    train_ds = TokenDataset(ids[:cut], args.seq_len)
    val_ds = TokenDataset(ids[cut:], args.seq_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            drop_last=True, num_workers=0)

    model = build_llama(vocab_size, hidden=384, layers=6, heads=6, head_dim=64,
                        intermediate=1024, max_seq_len=max(args.seq_len, 512)).to(device)

    # Harness 配置：阈值先用 05 篇的默认值，跑完用基线实测校准
    harness = HarnessRunner(HarnessConfig(vocab_size=vocab_size))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=0.1, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp) if device == "cuda" else None

    history = {"grad_norm": [], "update_ratio": []}
    init_loss = None
    step = 0
    t0 = time.perf_counter()

    while step < args.steps:
        for x, y in train_loader:
            if step >= args.steps:
                break
            x, y = x.to(device), y.to(device)
            harness.check_batch(x)
            before = snapshot_params(model)

            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=args.amp):
                loss = contract_loss(model(x), y)
            if init_loss is None:
                init_loss = loss.item()
                harness.check_initial_loss(init_loss)
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # AMP 下必须先 unscale 再 clip，否则 grad_norm 是放大后的值
            if scaler is not None:
                scaler.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            harness.check_grad(float(grad_norm), step)
            history["grad_norm"].append(round(float(grad_norm), 4))

            lr_now = cosine_with_warmup(step, args.steps, 100, args.lr)
            for g in opt.param_groups:
                g["lr"] = lr_now
            if scaler is not None:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            ur = harness.check_update(model, before, step, scaler=scaler)
            history["update_ratio"].append(round(ur, 6))
            opt.zero_grad(set_to_none=True)
            step += 1

        if step % args.eval_every == 0 or step >= args.steps:
            model.eval()
            xb, yb = next(iter(val_loader))
            xb, yb = xb.to(device), yb.to(device)
            with torch.no_grad():
                vl = contract_loss(model(xb), yb).item()
            model.train()
            harness.check_val_loss(vl, loss.item(), step)

    wall = time.perf_counter() - t0
    result = {
        "model": "llama",
        "tokenizer": "bpe",
        "steps": step,
        "wall_time_s": round(wall, 2),
        "init_loss_measured": round(init_loss, 4),
        "init_loss_theory": round(math.log(vocab_size), 4),
        "grad_norm_steady": history["grad_norm"][-1],
        "update_ratio_steady": history["update_ratio"][-1],
        "harness_ok": harness.ok,
        "harness_alerts": harness.state.messages(),
        "harness_summary": harness.summary(),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-dir", default="exp_scale/data")
    ap.add_argument("--output", default="/tmp/harness_on_hf.json")
    args = ap.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
