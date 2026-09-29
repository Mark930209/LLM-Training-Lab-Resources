"""train_eval.py —— 固定 token 预算训练 + 两域留出困惑度 + 多样性。

复用 04 篇的 SuperMiniGPT 与 CharTokenizer（同一套模型代码 = 同一个口径）。

公平对照的三条口径（核心判断"阶段口径对齐"在训练环节的体现）：
  1. tokenizer 固定：用两域训练池并集建一次词表，所有配方共用。否则不同
     配方词表不同、切分粒度不同，困惑度不可比。
  2. token 预算固定：每个配方训练相同的 token 数（steps × batch × block），
     比的是"同样喂这么多 token，哪种配方学得好"，不是"谁训得久"。
  3. 评测集固定且不进配方：nl_eval / code_eval 永不参与训练与重复注入，
     指纹（SHA-256）写进每份结果，跨 run 核对评的是同一份数据。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from exp_scale.data import CharTokenizer, LMDataset
from exp_scale.model import SuperMiniGPT


@dataclass
class TrainConfig:
    hidden: int = 384
    layers: int = 6
    heads: int = 6
    seq_len: int = 256
    batch_size: int = 16
    token_budget: int = 2_000_000   # 每个配方训练相同的 token 数
    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_frac: float = 0.1
    seed: int = 20260924
    device: str = "cuda"


def build_fixed_tokenizer(nl_train: str, code_train_text: str) -> CharTokenizer:
    """用两域训练池并集建固定词表，所有配方共用这一个 tokenizer。"""
    return CharTokenizer(nl_train + "\n" + code_train_text)


def _lr_at(step: int, total: int, cfg: TrainConfig) -> float:
    warmup = max(1, int(total * cfg.warmup_frac))
    if step < warmup:
        return cfg.lr * step / warmup
    prog = (step - warmup) / max(1, total - warmup)
    return cfg.lr * 0.5 * (1 + math.cos(math.pi * prog))


@torch.no_grad()
def evaluate_ppl(model, token_ids: torch.Tensor, cfg: TrainConfig,
                 block_size: int | None = None) -> dict:
    """在留出集上算平均交叉熵与困惑度（滑窗，不重叠）。

    评测窗口必须 ≤ 模型声明的 seq_len，否则 RoPE/因果 mask 越界（模型
    forward 里有断言）。默认取 cfg.seq_len，保证评测口径与训练口径对齐。
    """
    model.eval()
    device = next(model.parameters()).device
    if block_size is None:
        block_size = cfg.seq_len
    total_loss, n_tokens = 0.0, 0
    for start in range(0, max(1, len(token_ids) - block_size - 1), block_size):
        chunk = token_ids[start:start + block_size + 1]
        if len(chunk) < block_size + 1:
            break
        x = chunk[:-1].unsqueeze(0).to(device)
        y = chunk[1:].unsqueeze(0).to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                               reduction="sum")
        total_loss += loss.item()
        n_tokens += y.numel()
    model.train()
    if n_tokens == 0:
        return {"loss": None, "ppl": None, "n_tokens": 0}
    mean_loss = total_loss / n_tokens
    return {"loss": round(mean_loss, 4), "ppl": round(math.exp(mean_loss), 2),
            "n_tokens": n_tokens}


def train_recipe(recipe_text: str, tokenizer: CharTokenizer,
                 nl_eval: str, code_eval: str, cfg: TrainConfig) -> dict:
    """训练一个配方并评测两域困惑度。返回完整结果字典。"""
    torch.manual_seed(cfg.seed)
    device = cfg.device if torch.cuda.is_available() else "cpu"

    train_ids = torch.tensor(tokenizer.encode(recipe_text), dtype=torch.long)
    dataset = LMDataset(train_ids, cfg.seq_len)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(cfg.seed), drop_last=True)

    model = SuperMiniGPT(vocab_size=tokenizer.vocab_size, hidden=cfg.hidden,
                         layers=cfg.layers, heads=cfg.heads,
                         seq_len=cfg.seq_len).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)

    tokens_per_step = cfg.batch_size * cfg.seq_len
    total_steps = max(1, cfg.token_budget // tokens_per_step)

    started = time.perf_counter()
    if cfg.device == "cuda" and torch.cuda.is_available():
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
    train_sec = round(time.perf_counter() - started, 2)

    peak_mib = None
    if device == "cuda":
        peak_mib = round(torch.cuda.max_memory_allocated() / 1024**2, 1)

    # 两域留出困惑度（同一 tokenizer，OOV 字符被 encode 跳过，口径一致）
    nl_ids = torch.tensor(tokenizer.encode(nl_eval), dtype=torch.long)
    code_ids = torch.tensor(tokenizer.encode(code_eval), dtype=torch.long)
    nl_ppl = evaluate_ppl(model, nl_ids, cfg)
    code_ppl = evaluate_ppl(model, code_ids, cfg)

    return {
        "n_params": n_params,
        "vocab_size": tokenizer.vocab_size,
        "train_tokens": step * tokens_per_step,
        "total_steps": total_steps,
        "final_train_loss": round(last_loss, 4) if last_loss is not None else None,
        "train_sec": train_sec,
        "peak_memory_mib": peak_mib,
        "device": device,
        "nl_eval": nl_ppl,
        "code_eval": code_ppl,
    }
