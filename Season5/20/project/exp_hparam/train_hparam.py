"""train_hparam.py —— 20 篇超参实验的训练循环。

与 19 篇 train_arch 同口径（同 AdamW、同 clip 1.0、同 dataset/loader seeding、
同 evaluate_ppl），差别在 LR 调度可配（三种衰减 + warmup 比例）、支持梯度累积、
支持 LR range test 模式、发散早停。架构固定 19 篇 modern，用 exp_arch.build_model。
"""

from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F

from exp_recipe.train_eval import TrainConfig, evaluate_ppl
from exp_scale.data import LMDataset

from .hparam_metrics import lr_at, lr_range_at


def build_modern(hidden: int, base: dict, vocab: int):
    """建 19 篇 modern 架构（五轴固定），只变 hidden。"""
    from exp_arch.train_arch import build_model
    from exp_arch.model_arch import kv_heads_for
    arch = base["arch"]
    spec = {
        "hidden": hidden,
        "norm_type": arch["norm_type"], "norm_pos": arch["norm_pos"],
        "ffn_type": arch["ffn_type"], "pos_enc": arch["pos_enc"],
        "attn_type": arch["attn_type"],
    }
    b = {"layers": base["layers"], "heads": base["heads"], "seq_len": base["seq_len"]}
    spec["kv_heads"] = kv_heads_for(arch["attn_type"], base["heads"])
    return build_model(spec, vocab, b)


def train_hparam(train_ids: torch.Tensor, model, cfg: TrainConfig,
                 accum: int = 1, decay: str = "cosine",
                 diverge_thresh: float | None = None) -> dict:
    """固定 token 预算训练。accum=梯度累积步数，decay=衰减策略。

    发散早停：loss NaN 或 > diverge_thresh 时停止，diverged=True。
    返回 train_sec/peak_memory_mib/tokens_per_sec/final_train_loss/total_steps/
    n_params/diverged。
    """
    torch.manual_seed(cfg.seed)
    device = cfg.device if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    loader = torch.utils.data.DataLoader(
        LMDataset(train_ids, cfg.seq_len), batch_size=cfg.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(cfg.seed), drop_last=True)

    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    tokens_per_step = cfg.batch_size * cfg.seq_len
    # 优化器步数 = token 预算 / (有效 batch token)；有效 batch = batch × accum
    opt_steps = max(1, cfg.token_budget // (tokens_per_step * accum))

    started = time.perf_counter()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model.train()
    data_iter = iter(loader)
    opt.zero_grad(set_to_none=True)
    last_loss = None
    diverged = False
    micro = 0
    for opt_step in range(opt_steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(opt_step, opt_steps, cfg.lr, cfg.warmup_frac, decay)
        for _ in range(accum):
            try:
                xb, yb = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                xb, yb = next(data_iter)
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
            (loss / accum).backward()
            last_loss = loss.item()
            micro += 1
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if diverge_thresh is not None and (
                last_loss is None or math.isnan(last_loss) or last_loss > diverge_thresh):
            diverged = True
            break
    train_sec = time.perf_counter() - started

    peak_mib = round(torch.cuda.max_memory_allocated() / 1024**2, 1) if device == "cuda" else None
    steps_done = opt_step + 1   # 循环至少跑一次（opt_steps ≥ 1），发散时为已完成的步数
    return {
        "n_params": n_params,
        "train_tokens": micro * tokens_per_step,
        "opt_steps": steps_done,
        "final_train_loss": round(last_loss, 4) if last_loss is not None and not math.isnan(last_loss) else None,
        "train_sec": round(train_sec, 2),
        "tokens_per_sec": round(micro * tokens_per_step / train_sec, 1) if train_sec > 0 else None,
        "peak_memory_mib": peak_mib,
        "diverged": diverged,
        "device": device,
    }


def lr_range_test(train_ids: torch.Tensor, model, cfg: TrainConfig,
                  lr_min: float, lr_max: float) -> dict:
    """LR range test：单次训练内 LR 指数上升，记录每步 (lr, loss) 曲线。

    用于找可用 LR 区间与发散拐点（loss 开始上升处）。不做衰减、不 early-stop，
    跑满 token 预算，把整条 loss-lr 曲线返回给上层分析。
    """
    torch.manual_seed(cfg.seed)
    device = cfg.device if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    loader = torch.utils.data.DataLoader(
        LMDataset(train_ids, cfg.seq_len), batch_size=cfg.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(cfg.seed), drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr_min, weight_decay=cfg.weight_decay)
    tokens_per_step = cfg.batch_size * cfg.seq_len
    total_steps = max(2, cfg.token_budget // tokens_per_step)

    curve = []
    model.train()
    data_iter = iter(loader)
    started = time.perf_counter()
    for step in range(total_steps):
        lr = lr_range_at(step, total_steps, lr_min, lr_max)
        for g in opt.param_groups:
            g["lr"] = lr
        try:
            xb, yb = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            xb, yb = next(data_iter)
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        lv = loss.item()
        curve.append({"step": step, "lr": round(lr, 8),
                      "loss": round(lv, 4) if not math.isnan(lv) else None})
    train_sec = time.perf_counter() - started
    return {"curve": curve, "total_steps": total_steps,
            "train_sec": round(train_sec, 2), "device": device}


def eval_hparam(model, tokenizer, nl_eval: str, code_eval: str, cfg: TrainConfig) -> dict:
    """两域留出困惑度（复用 17 篇 evaluate_ppl）。"""
    nl_ids = torch.tensor(tokenizer.encode(nl_eval), dtype=torch.long)
    code_ids = torch.tensor(tokenizer.encode(code_eval), dtype=torch.long)
    return {"nl": evaluate_ppl(model, nl_ids, cfg),
            "code": evaluate_ppl(model, code_ids, cfg)}
