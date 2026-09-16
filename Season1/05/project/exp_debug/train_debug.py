"""train_debug.py —— 05 篇主实验程序：健康基线 + 故障注入，一次跑完。

与 04 篇 train.py 的关系：复用它的模型、数据、优化器、调度器，
只把训练循环换成带诊断面板的版本，并接入 fail_modes 故障开关。

用法（在 project/ 目录下）：
    # 健康基线（300 步，采集全部诊断指标）
    python -m exp_debug.train_debug --config exp_scale/config_10m.yaml --steps 300

    # 注入单个故障
    python -m exp_debug.train_debug --config exp_scale/config_10m.yaml --steps 300 --fault label_shift

    # 续训对照实验（先跑 150 步存 checkpoint，再从断点续 150 步）
    python -m exp_debug.train_debug --config exp_scale/config_10m.yaml --steps 150 --save-ckpt /tmp/a.pt
    python -m exp_debug.train_debug --config exp_scale/config_10m.yaml --steps 300 --resume /tmp/a.pt
    python -m exp_debug.train_debug --config exp_scale/config_10m.yaml --steps 300 --resume /tmp/a.pt --fault resume_opt
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
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import config as cfg_mod, logging as log_mod  # noqa: E402
from common.reproducibility import set_all_seeds  # noqa: E402
from exp_debug.diagnostics import (  # noqa: E402
    batch_dup_rate, data_fingerprint, dataset_overlap, expected_initial_loss,
    first_nonfinite_step, opt_checksum, param_checksum, rng_checksum,
    snapshot_params, update_ratio)
from exp_debug.fail_modes import (  # noqa: E402
    FAULT_STAGES, apply_fault, fault_clip_enabled, fault_lr_multiplier)
from exp_scale.data import LMDataset, load_corpus  # noqa: E402
from exp_scale.model import SuperMiniGPT  # noqa: E402
from exp_scale.schedulers import cosine_with_warmup  # noqa: E402


class LeakyValDataset(Dataset):
    """val_leak 故障：把 train 段前 10% 的数据拼进 val 集。"""

    def __init__(self, val_ids: torch.Tensor, leak_ids: torch.Tensor,
                 block_size: int):
        self.data = torch.cat([leak_ids, val_ids])
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) - self.block_size - 1

    def __getitem__(self, idx: int):
        chunk = self.data[idx : idx + self.block_size + 1]
        return chunk[:-1], chunk[1:]


@torch.no_grad()
def estimate_loss(model, loader, device, amp: bool = False,
                  iters: int = 20) -> float:
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
    return sum(losses) / max(len(losses), 1)


def run(cfg: dict, steps: int, fault: str | None = None,
        resume: str | None = None, save_ckpt: str | None = None,
        probe_every: int = 50) -> dict:
    exp, hw = cfg["experiment"], cfg["hardware"]
    device = hw["device"]
    set_all_seeds(exp["seed"], deterministic=False)

    use_amp = device == "cuda"
    corpus = exp.get("corpus", "large")
    tok, train_ds, val_ds = load_corpus(Path(__file__).parent.parent /
                                         "exp_scale" / "data",
                                         exp["seq_len"], corpus=corpus)

    # val_leak 在数据集构建时注入：train 前 10% 拼进 val
    if fault == "val_leak":
        full_ids = train_ds.data
        leak_ids = full_ids[: len(full_ids) // 10]
        val_ds = LeakyValDataset(val_ds.data, leak_ids, exp["seq_len"])

    bs = hw["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                           drop_last=True, num_workers=0)

    # 数据集重叠体检：val_leak 的直接证据（健康划分 = 0）
    overlap = dataset_overlap(train_ds, val_ds)

    model = SuperMiniGPT(tok.vocab_size, exp["hidden"], exp["layers"],
                         exp["heads"], exp["seq_len"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=exp["lr"],
                            weight_decay=exp.get("weight_decay", 0.1),
                            betas=(0.9, 0.95))
    # amp_overflow：初始 scale 拉到 2^48，前几十步 inf 梯度全部跳步
    init_scale = 2 ** 48 if fault == "amp_overflow" else 2 ** 16
    scaler = (torch.amp.GradScaler("cuda", enabled=use_amp,
                                   init_scale=init_scale) if use_amp else None)

    rl = log_mod.RunLogger("runs", f"debug_{fault or 'baseline'}_{exp['name']}")
    rl.write_env_card(log_mod.env_card())

    # ---- 健康基线体检：初始 loss 应接近 ln(vocab) ----
    init_theory = expected_initial_loss(tok.vocab_size)
    x0, y0 = next(iter(train_loader))
    x0, y0 = x0.to(device), y0.to(device)
    with torch.no_grad():
        logits = model(x0)
        init_loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), y0.reshape(-1)).item()

    # ---- 恢复阶段 ----
    start_step = 0
    if resume:
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        # 伪续训：按故障开关决定恢复哪些状态
        if fault not in ("resume_opt",):
            if ckpt.get("optimizer"):
                opt.load_state_dict(ckpt["optimizer"])
        if fault not in ("resume_scaler",) and scaler is not None:
            if ckpt.get("scaler"):
                scaler.load_state_dict(ckpt["scaler"])
        if fault not in ("resume_rng",):
            if ckpt.get("rng_cpu") is not None:
                torch.set_rng_state(ckpt["rng_cpu"].cpu())
            if ckpt.get("rng_cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state(ckpt["rng_cuda"].cpu())
        if fault not in ("resume_sched",):
            start_step = ckpt.get("step", 0)
        # resume_* 故障的对照基准：完整恢复的续训
        if fault in FAULT_STAGES and FAULT_STAGES[fault]["stage"] == "resume":
            if fault == "resume_opt":
                pass  # 权重已恢复，优化器状态丢弃
            elif fault == "resume_scaler":
                pass
            elif fault == "resume_rng":
                pass
            elif fault == "resume_sched":
                pass
        else:
            start_step = ckpt.get("step", 0)

    # ---- 训练循环（带诊断面板）----
    history = {"train": [], "val": [], "lr": [], "grad_norm": [],
               "update_ratio": [], "fingerprints": [], "batch_dup": []}
    step = start_step
    nan_at = None
    clip = fault_clip_enabled(fault or "", exp.get("grad_clip", 1.0))
    lr_mult = fault_lr_multiplier(fault or "")
    t0 = time.perf_counter()

    model.train()
    opt.zero_grad(set_to_none=True)
    while step < steps:
        for x, y in train_loader:
            if step >= steps:
                break
            x, y = x.to(device), y.to(device)

            # ---- 数据阶段故障注入 ----
            if fault in FAULT_STAGES and FAULT_STAGES[fault]["stage"] == "data":
                x, y = apply_fault(fault, x, y)

            fp = data_fingerprint(x, y)
            history["batch_dup"].append(batch_dup_rate(x))
            before = snapshot_params(model)

            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=use_amp):
                logits = model(x)
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), y.reshape(-1))

            if scaler is not None and use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if scaler is not None and use_amp:
                scaler.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

            lr_now = cosine_with_warmup(step, steps, exp["warmup_steps"],
                                        exp["lr"] * lr_mult,
                                        exp.get("min_lr_ratio", 0.1))
            for g in opt.param_groups:
                g["lr"] = lr_now

            if scaler is not None and use_amp:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1

            ur = update_ratio(model, before)
            if not math.isfinite(loss.item()) and nan_at is None:
                nan_at = step
            if step % 10 == 0:
                history["train"].append(round(loss.item(), 4))
                history["lr"].append(lr_now)
                history["grad_norm"].append(round(float(grad_norm), 4))
                history["update_ratio"].append(f"{ur:.2e}")
                history["fingerprints"].append(fp)
            if step % 100 == 0 or step == steps:
                vl = estimate_loss(model, val_loader, device, amp=use_amp)
                history["val"].append(round(vl, 4))
                rl.log.info("step %d loss %.4f val %.4f gnorm %.2f ur %.2e",
                            step, loss.item(), vl, float(grad_norm), ur)
            if nan_at is not None:
                break
        if nan_at is not None:
            break

    wall_s = time.perf_counter() - t0

    # ---- 终态体检 ----
    final_val = history["val"][-1] if history["val"] else float("nan")
    result = {
        "fault": fault or "baseline",
        "stage": FAULT_STAGES.get(fault, {}).get("stage", "none"),
        "loud": FAULT_STAGES.get(fault, {}).get("loud", False),
        "params_million": round(n_params / 1e6, 2),
        "vocab_size": tok.vocab_size,
        "init_loss_theory": round(init_theory, 4),
        "init_loss_measured": round(init_loss, 4),
        "steps_done": step - start_step,
        "start_step": start_step,
        "final_train_loss": history["train"][-1] if history["train"] else float("nan"),
        "final_val_loss": final_val,
        "best_val_loss": min(history["val"]) if history["val"] else float("nan"),
        "nan_at_step": nan_at,
        "first_nonfinite": first_nonfinite_step(history["train"]),
        "param_checksum": param_checksum(model),
        "opt_checksum": opt_checksum(opt),
        "rng_checksum": rng_checksum(),
        "final_grad_norm": history["grad_norm"][-1] if history["grad_norm"] else None,
        "final_update_ratio": history["update_ratio"][-1] if history["update_ratio"] else None,
        "dup_fingerprint_rate": _dup_rate(history["fingerprints"]),
        "batch_dup_rate_mean": (sum(history["batch_dup"]) /
                                 max(len(history["batch_dup"]), 1)),
        "dataset_overlap": overlap,
        "wall_time_s": round(wall_s, 1),
        "train_curve": history["train"],
        "val_curve": history["val"],
        "grad_norm_curve": history["grad_norm"],
        "update_ratio_curve": history["update_ratio"],
    }

    if save_ckpt:
        ck = {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "step": step,
            "rng_cpu": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        }
        torch.save(ck, save_ckpt)
        result["saved_ckpt"] = save_ckpt

    rl.log_metrics(**{k: v for k, v in result.items() if not isinstance(v, list)})
    run_dir = rl.close()
    result["run_dir"] = str(run_dir)
    return result


def _dup_rate(fingerprints: list[str]) -> float:
    """指纹重复率：dup_batch 注入后接近 0.5，健康训练约 0。"""
    if not fingerprints:
        return 0.0
    return 1 - len(set(fingerprints)) / len(fingerprints)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--fault", default=None, choices=list(FAULT_STAGES))
    ap.add_argument("--resume", default=None)
    ap.add_argument("--save-ckpt", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()

    cfg = cfg_mod.load_config(args.config, args.set)
    result = run(cfg, args.steps, args.fault, args.resume, args.save_ckpt)
    print(json.dumps({k: v for k, v in result.items()
                      if not isinstance(v, list)}, ensure_ascii=False, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()