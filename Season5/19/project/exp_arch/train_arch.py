"""train_arch.py —— 组件化模型的固定预算训练（19 篇专用）。

训练循环与 17 篇 train_eval 同口径（同 lr 调度、同 clip、同评测窗口规则），
差别只在模型：这里建 ArchGPT（组件可配），17 篇建 SuperMiniGPT（组件固定）。
evaluate_ppl 直接复用 17 篇的实现（它只依赖 model 的 forward，与模型类型无关）。
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from exp_recipe.train_eval import TrainConfig, evaluate_ppl, _lr_at

from .model_arch import ArchGPT, kv_heads_for


def build_model(spec: dict, vocab_size: int, base: dict) -> ArchGPT:
    """按消融配置建模型。spec 含组件五轴 + hidden（等参数量反解结果）。"""
    return ArchGPT(
        vocab_size=vocab_size,
        hidden=spec["hidden"],
        layers=base["layers"],
        heads=base["heads"],
        kv_heads=kv_heads_for(spec.get("attn_type", "mha"), base["heads"]),
        seq_len=base["seq_len"],
        norm_type=spec.get("norm_type", "rms"),
        norm_pos=spec.get("norm_pos", "pre"),
        ffn_type=spec.get("ffn_type", "swiglu"),
        pos_enc=spec.get("pos_enc", "rope"),
        rope_base=spec.get("rope_base", 10000.0),
        ffn_expansion=spec.get("ffn_expansion"),
    )


def train_arch(train_ids: torch.Tensor, model: ArchGPT, cfg: TrainConfig) -> dict:
    """固定 token 预算训练。与 17 篇 train_recipe 的训练循环逐行同口径。

    返回 train_sec / peak_memory_mib / tokens_per_sec / final_train_loss /
    total_steps / n_params。
    """
    torch.manual_seed(cfg.seed)
    device = cfg.device if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    from exp_scale.data import LMDataset
    dataset = LMDataset(train_ids, cfg.seq_len)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(cfg.seed), drop_last=True)

    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    tokens_per_step = cfg.batch_size * cfg.seq_len
    total_steps = max(1, cfg.token_budget // tokens_per_step)

    started = time.perf_counter()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model.train()
    data_iter = iter(loader)
    step = 0
    last_loss = None
    while step < total_steps:
        try:
            xb, yb = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            xb, yb = next(data_iter)
        xb, yb = xb.to(device), yb.to(device)
        for g in opt.param_groups:
            g["lr"] = _lr_at(step, total_steps, cfg)
        logits = model(xb)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last_loss = loss.item()
        step += 1
    train_sec = time.perf_counter() - started

    peak_mib = None
    if device == "cuda":
        peak_mib = round(torch.cuda.max_memory_allocated() / 1024**2, 1)

    return {
        "n_params": n_params,
        "train_tokens": step * tokens_per_step,
        "total_steps": total_steps,
        "final_train_loss": round(last_loss, 4) if last_loss is not None else None,
        "train_sec": round(train_sec, 2),
        "tokens_per_sec": round(step * tokens_per_step / train_sec, 1),
        "peak_memory_mib": peak_mib,
        "device": device,
    }


def eval_arch(model: ArchGPT, tokenizer, nl_eval: str, code_eval: str,
              cfg: TrainConfig) -> dict:
    """两域留出困惑度（复用 17 篇 evaluate_ppl，评测窗口默认 cfg.seq_len）。"""
    nl_ids = torch.tensor(tokenizer.encode(nl_eval), dtype=torch.long)
    code_ids = torch.tensor(tokenizer.encode(code_eval), dtype=torch.long)
    return {
        "nl": evaluate_ppl(model, nl_ids, cfg),
        "code": evaluate_ppl(model, code_ids, cfg),
    }
