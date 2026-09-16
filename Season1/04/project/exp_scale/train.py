"""train.py —— 可复用的训练程序（04 篇核心交付物）。

与 03 篇演示循环的差别，就是本篇要讲的"工程件"：

    03 演示循环                      04 可复用的训练程序
    ─────────────────────────────    ─────────────────────────────
    常数 lr                          warmup + cosine 调度
    单步 batch                       gradient accumulation（小显存补大 batch）
    fp32                             AMP 混合精度（省显存 + 提速）
    只存权重                         checkpoint 含优化器/步数/调度器状态
    不能续训                         resume 从任意 checkpoint 继续
    无吞吐统计                       step time / tok/s / 显存峰值全采集

用法（在 project/ 目录下）：
    python -m exp_scale.train --config exp_scale/config_10m.yaml
    python -m exp_scale.train --config exp_scale/config_100m.yaml --mode no_amp
    python -m exp_scale.train --config exp_scale/config_100m.yaml --resume runs/xxx/ckpt.pt
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import config as cfg_mod, logging as log_mod  # noqa: E402
from common.benchmark import measure_peak_memory_mb  # noqa: E402
from common.reproducibility import set_all_seeds  # noqa: E402
from exp_scale.data import load_corpus  # noqa: E402
from exp_scale.model import SuperMiniGPT  # noqa: E402
from exp_scale.schedulers import (  # noqa: E402
    cosine_with_warmup, estimate_model_params, estimate_training_memory_mb)


@torch.no_grad()
def estimate_loss(model, loader, device, iters: int = 20,
                  amp: bool = False) -> float:
    """验证集 loss：判断"模型在学还是在背"的唯一标尺。"""
    model.eval()
    losses = []
    for i, (x, y) in enumerate(loader):
        if i >= iters:
            break
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=amp):
            logits = model(x)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def build_optimizer(model, lr: float, weight_decay: float):
    """AdamW：把权重衰减与梯度更新解耦（03 篇已用，这里保持口径一致）。"""
    return torch.optim.AdamW(model.parameters(), lr=lr,
                             weight_decay=weight_decay, betas=(0.9, 0.95))


def save_checkpoint(path: Path, model, opt, scaler, step: int, cfg: dict,
                    best_val: float) -> None:
    """保存完整训练状态：不只是权重，还要能"接着训"。

    03 篇只存 model.state_dict()，恢复后优化器动量、学习率进度全丢，
    续训等于重新开始。可复用的训练程序必须把这三样一起存。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": step,
        "best_val": best_val,
        "config": cfg,
    }, path)


def load_checkpoint(path: Path, model, opt, scaler, device: str) -> tuple[int, float]:
    """恢复训练状态，返回 (step, best_val)。"""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if opt is not None and ckpt.get("optimizer") is not None:
        opt.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt.get("step", 0), ckpt.get("best_val", float("inf"))


def run(cfg: dict, mode: str = "main", resume: str | None = None) -> dict:
    exp, hw = cfg["experiment"], cfg["hardware"]
    device = hw["device"]
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("config 要求 cuda 但 GPU 不可用；显式设置 hardware.device=cpu")
    set_all_seeds(exp["seed"], deterministic=False)

    # ---- 工程件开关（每个 mode 只关一个，保证归因干净）----
    use_amp = mode != "no_amp"
    use_sched = mode != "no_sched"
    accum = 1 if mode == "no_accum" else exp.get("grad_accum", 1)

    corpus = exp.get("corpus", "small")
    tok, train_ds, val_ds = load_corpus(Path(__file__).parent / "data",
                                        exp["seq_len"], corpus=corpus)
    bs = hw["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              drop_last=True, num_workers=hw.get("num_workers", 2))
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            drop_last=True, num_workers=hw.get("num_workers", 2))

    model = SuperMiniGPT(tok.vocab_size, exp["hidden"], exp["layers"],
                         exp["heads"], exp["seq_len"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_params_est = estimate_model_params(tok.vocab_size, exp["hidden"],
                                         exp["layers"])
    opt = build_optimizer(model, exp["lr"], exp.get("weight_decay", 0.1))
    # AMP 的 GradScaler：fp16 梯度下溢时自动放大 loss 尺度
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if device == "cuda" else None

    rl = log_mod.RunLogger("runs", f"scale_{mode}_{exp['name']}")
    rl.write_env_card(log_mod.env_card())
    cfg_mod.snapshot(cfg_mod.Config(cfg), rl.run_dir)

    mem_est = estimate_training_memory_mb(
        n_params, exp["seq_len"], bs, exp["hidden"], exp["layers"],
        tok.vocab_size, amp=use_amp)
    rl.log.info("mode=%s params=%.2fM(est %.2fM) vocab=%d bs=%d accum=%d amp=%s",
                mode, n_params / 1e6, n_params_est / 1e6, tok.vocab_size, bs,
                accum, use_amp)
    rl.log.info("显存估算: %s", json.dumps(mem_est, ensure_ascii=False))

    start_step, best_val = 0, float("inf")
    if resume:
        start_step, best_val = load_checkpoint(Path(resume), model, opt, scaler,
                                               device)
        rl.log.info("从 checkpoint 恢复: step=%d best_val=%.4f", start_step, best_val)

    history = {"train": [], "val": [], "lr": []}
    step = start_step
    t_start = time.perf_counter()
    nan_at = None
    eval_every, log_every = exp["eval_every"], exp["log_every"]
    total_steps = exp["steps"]
    tokens_per_step = bs * exp["seq_len"] * accum

    model.train()
    opt.zero_grad(set_to_none=True)
    micro = 0
    running_loss = 0.0
    loss_count = 0     # 本次日志窗口内累计的 micro-batch 数（不等于 step 数）
    while step < total_steps:
        for x, y in train_loader:
            if step >= total_steps:
                break
            x, y = x.to(device), y.to(device)

            # ---- 工程件：AMP 前向 ----
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=use_amp):
                logits = model(x)
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
                # ---- 工程件：gradient accumulation ----
                # 把 accum 个 micro-batch 的梯度累加，等效于大 batch
                loss = loss / accum

            if scaler is not None and use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # 还原成未缩放的原始 loss 记账（loss 本身已除过 accum）
            running_loss += loss.item() * accum
            loss_count += 1
            micro += 1
            if micro < accum:
                continue
            micro = 0

            # ---- 工程件：梯度裁剪（放大后更重要）----
            if scaler is not None and use_amp:
                scaler.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                       exp.get("grad_clip", 1.0))

            # ---- 工程件：lr schedule ----
            lr_now = (cosine_with_warmup(step, total_steps, exp["warmup_steps"],
                                         exp["lr"], exp.get("min_lr_ratio", 0.1))
                      if use_sched else exp["lr"])
            for g in opt.param_groups:
                g["lr"] = lr_now

            if scaler is not None and use_amp:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1

            if not math.isfinite(running_loss) and nan_at is None:
                nan_at = step
            if step % log_every == 0:
                # 除以实际累计的 micro-batch 数，而不是 log_every：
                # accum > 1 时两者相差 accum 倍，除错会让 loss 虚高（本篇真实踩过的坑）
                avg = running_loss / max(loss_count, 1)
                history["train"].append(round(avg, 4))
                history["lr"].append(round(lr_now, 8))
                rl.log_metrics(step=step, train_loss=round(avg, 4),
                               lr=lr_now, grad_norm=round(float(grad_norm), 4))
                rl.log.info("step %d/%d loss %.4f lr %.2e gnorm %.2f",
                            step, total_steps, avg, lr_now, float(grad_norm))
                running_loss = 0.0
                loss_count = 0
            if step % eval_every == 0 or step == total_steps:
                vl = estimate_loss(model, val_loader, device, amp=use_amp)
                history["val"].append(round(vl, 4))
                rl.log_metrics(step=step, val_loss=round(vl, 4))
                rl.log.info("step %d val_loss %.4f", step, vl)
                if vl < best_val:
                    best_val = vl
                    save_checkpoint(rl.run_dir / "ckpt_best.pt", model, opt,
                                    scaler, step, cfg, best_val)
            if nan_at is not None:
                break
        if nan_at is not None:
            break

    wall_s = time.perf_counter() - t_start
    steps_done = step - start_step
    final_val = history["val"][-1] if history["val"] else float("nan")
    peak_mb = measure_peak_memory_mb()

    save_checkpoint(rl.run_dir / "ckpt_last.pt", model, opt, scaler, step, cfg,
                    best_val)

    samples = {}
    if nan_at is None:
        ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
        for temp, tag in [(1.0, "temp1.0"), (0.8, "temp0.8")]:
            out = model.generate(ctx.clone(), exp.get("gen_tokens", 200),
                                 temperature=temp, top_k=40)
            samples[tag] = tok.decode(out[0].tolist())

    result = {
        "mode": mode,
        "corpus": corpus,
        "params_million": round(n_params / 1e6, 2),
        "params_est_million": round(n_params_est / 1e6, 2),
        "vocab_size": tok.vocab_size,
        "batch_size": bs,
        "grad_accum": accum,
        "effective_batch": bs * accum,
        "amp": use_amp,
        "schedule": use_sched,
        "lr": exp["lr"],
        "final_val_loss": final_val,
        "best_val_loss": best_val,
        "initial_train_loss": history["train"][0] if history["train"] else float("nan"),
        "nan_at_step": nan_at,
        "wall_time_s": round(wall_s, 1),
        "steps_done": steps_done,
        "step_time_ms": round(wall_s * 1000 / max(steps_done, 1), 1),
        "tokens_per_sec": round(tokens_per_step * steps_done / max(wall_s, 1e-9)),
        "peak_memory_mb": peak_mb,
        "mem_estimate": mem_est,
        "train_curve": history["train"],
        "val_curve": history["val"],
        "lr_curve": history["lr"],
        "sample": samples.get("temp0.8", ""),
        "sample_temp1.0": samples.get("temp1.0", ""),
    }
    rl.log_metrics(**{k: v for k, v in result.items() if not isinstance(v, list)})
    run_dir = rl.close()
    result["run_dir"] = str(run_dir)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", default="main",
                    choices=["main", "no_amp", "no_sched", "no_accum"])
    ap.add_argument("--resume", default=None, help="checkpoint 路径，续训用")
    ap.add_argument("--output", default=None, help="结果 JSON 落盘路径")
    ap.add_argument("--set", action="append", default=[],
                    help="覆盖配置，如 --set experiment.steps=100")
    args = ap.parse_args()

    cfg = cfg_mod.load_config(args.config, args.set)
    result = run(cfg, args.mode, args.resume)
    print(json.dumps({k: v for k, v in result.items()
                      if not isinstance(v, list)}, ensure_ascii=False, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()