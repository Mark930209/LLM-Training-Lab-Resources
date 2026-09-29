"""train_cpt.py —— 22 篇的 CPT 训练循环：Qwen2.5-0.5B 全参数续训。

与 20/21 篇的自有小模型不同，本篇直接在真实开源底座上做 CPT（bf16 全参
AdamW，8GB 卡实测 bs=2 seq=512 峰值 6337 MiB / 1340 tok/s）。职责：

1. 两段式学习率（re-warmup + cosine 衰减，cpt_metrics.lr_two_stage）
2. 训练流 = 领域数据 + replay 混合（cpt_data.build_stream，token 级均匀撒入）
3. 周期性双评测（领域/通用 ppl + nats/char），画遗忘曲线
4. 遗忘量化：参数相对底座平均变化量、表示相似度（固定 probe 输入的末层
   hidden states 余弦）
5. 扩词表三种 embedding 初始化（random/mean/subword_avg）
6. 周期 checkpoint（失败案例的回退恢复用）

所有 eval 文本按固定字符切片定义；跨 tokenizer 对比只看 nats/char。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from cpt_metrics import (
    eval_window_plan,
    lr_two_stage,
    param_mean_change,
    representation_similarity,
)


@dataclass
class CPTConfig:
    lr_peak: float = 3e-5
    replay_ratio: float = 0.2
    rewarm: bool = True
    rewarm_frac: float = 0.05
    floor_frac: float = 0.1
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    batch_size: int = 2
    seq_len: int = 512
    token_budget: int = 200000
    eval_every: int = 40
    eval_max_tokens: int = 6000
    seed: int = 20260929
    diverge_thresh: float = 20.0
    checkpoint_every: int = 0      # 0=不存；失败实验用
    device: str = "cuda"
    tag: str = ""


# ---------------------------------------------------------------- 模型与词表


def load_model(model_dir: str, device: str = "cuda"):
    """加载 Qwen2.5-0.5B（bf16）与 tokenizer。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16)
    model.to(device)
    return model, tok


def apply_vocab_expansion(model, tok, new_tokens: list[str],
                          subword_ids: list[list[int]], method: str,
                          rng: np.random.Generator) -> dict:
    """扩词表：把新 token 加进 tokenizer，并按 method 初始化对应 embedding 行。

    Qwen 词表 151643，模型 embedding 矩阵 151936 行（293 行余量）——新 token
    的 id 落在既有矩阵内，无需 resize，但旧行取值来自预训练 checkpoint，
    必须显式覆盖为实验初始化值。tied embedding：输出头共享同一矩阵。
    """
    old_len = len(tok)
    added = tok.add_tokens(new_tokens)
    emb = model.get_input_embeddings().weight
    if old_len + added > emb.shape[0]:
        raise ValueError(f"need {old_len + added} rows, have {emb.shape[0]}")

    from cpt_metrics import init_new_embeddings

    old_np = emb.data[:old_len].detach().float().cpu().numpy()
    new_np = init_new_embeddings(method, old_np, subword_ids, rng)
    with torch.no_grad():
        emb[old_len:old_len + added] = torch.from_numpy(new_np).to(emb.dtype).to(emb.device)
    return {"old_len": old_len, "added": added, "method": method,
            "new_vocab": old_len + added}


# ---------------------------------------------------------------- 评测


def evaluate_text(model, tok, text: str, seq_len: int, device: str = "cuda",
                  max_tokens: int | None = None,
                  score_tail: bool = False) -> dict:
    """固定文本评测：返回 ppl、nats/token、nats/char。

    文本按 seq_len 切窗，labels 与输入同序列（HF 内部移位一次）。
    默认只计分完整窗（每窗 seq_len 个预测），与既有结果 JSON 的 ppl
    口径逐位一致；score_tail=True 时把尾部不足整窗的残段作短窗补计
    （首 token 作上下文），除全文首 token 外全覆盖，此时 nats/char 的
    分子与分母对应同一文本，跨 tokenizer 严格可比。
    max_tokens 只截取前若干 token（训练中快速评测用），nats/char 按
    token 比例折算字符数；正式对比一律用全量文本。
    """
    ids = tok.encode(text, add_special_tokens=False)
    if max_tokens is not None:
        ids = ids[:max_tokens]
    if len(ids) < seq_len + 1:
        raise ValueError(f"text too short: {len(ids)} tokens < seq_len+1")
    plan = eval_window_plan(len(ids), seq_len, score_tail=score_tail)
    total_nll, total_tok = 0.0, 0
    model.eval()
    with torch.no_grad():
        for i in range(plan["n_windows"]):
            # 每窗 seq_len+1 个 id：HF labels 内部移位一次，loss 覆盖 seq_len 个预测
            chunk = ids[i * seq_len: (i + 1) * seq_len + 1]
            t = torch.tensor([chunk], device=device)
            out = model(t, labels=t)
            total_nll += out.loss.item() * seq_len
            total_tok += seq_len
        if plan["tail_preds"] > 0:
            # 尾窗补计：首 token 作上下文，预测其余 tail_len-1 个
            tail = ids[plan["n_windows"] * seq_len:]
            t = torch.tensor([tail], device=device)
            out = model(t, labels=t)
            total_nll += out.loss.item() * plan["tail_preds"]
            total_tok += plan["tail_preds"]
    nats_tok = total_nll / total_tok
    # 字符数：用原始文本对应比例（max_tokens 截断时按 token 比例折算）
    full_ids = tok.encode(text, add_special_tokens=False)
    frac = len(ids) / len(full_ids)
    n_chars = max(1, int(len(text) * frac))
    return {
        "ppl": math.exp(nats_tok),
        "nats_per_tok": nats_tok,
        "nats_per_char": total_nll / n_chars,
        "tokens": total_tok,
        "chars": n_chars,
    }


def capture_probe(model, probe_ids: list[int], device: str = "cuda") -> np.ndarray:
    """固定 probe 输入的末层 hidden states（表示相似度的参照/对照）。"""
    x = torch.tensor([probe_ids], device=device)
    model.eval()
    with torch.no_grad():
        out = model(x, output_hidden_states=True)
    return out.hidden_states[-1][0].float().cpu().numpy()


# ---------------------------------------------------------------- 参数漂移


def snapshot_params(model) -> dict[str, torch.Tensor]:
    """参数快照（CPU），load_state_dict 可直接回灌。

    用 state_dict() 而非 named_parameters()：tied embedding 的 lm_head.weight
    在 named_parameters 里被去重隐藏，但 load_state_dict 严格模式要求它在场。
    """
    return {n: p.detach().cpu().clone() for n, p in model.state_dict().items()}


def param_drift(base: dict[str, torch.Tensor], model) -> dict:
    """参数相对底座的平均变化量（按元素数加权）+ 分块明细。

    分块：embed_tokens / layers.0..N（逐层）/ norm / lm_head——
    逐层明细用来回答"遗忘主要发生在哪些层"。
    """
    total_numel, weighted = 0, 0.0
    block_acc: dict[str, list[tuple[float, int]]] = {}
    for n, p in model.named_parameters():
        cur = p.detach().float().cpu()
        chg = param_mean_change(base[n].float().numpy(), cur.numpy())
        weighted += chg * p.numel()
        total_numel += p.numel()
        if n.startswith("model.layers."):
            blk = "layers." + n.split(".")[2]
        else:
            blk = n.split(".")[0] if not n.startswith("model.") else n.split(".")[1]
        block_acc.setdefault(blk, []).append((chg, p.numel()))
    blocks = {}
    for blk, items in block_acc.items():
        n_sum = sum(k for _, k in items)
        blocks[blk] = round(sum(c * k for c, k in items) / n_sum, 6)
    return {"mean_change": weighted / total_numel, "by_block": blocks}


# ---------------------------------------------------------------- 训练循环


def train_cpt(model, tok, stream_ids: list[int], cfg: CPTConfig,
              eval_texts: dict[str, str], probe_ids: list[int],
              base_probe: np.ndarray | None = None,
              base_params: dict | None = None) -> dict:
    """一轮 CPT。返回训练曲线、双评测曲线、遗忘量化、（可选）checkpoint。"""
    device = cfg.device
    # 一步 = batch_size 行 × (seq_len+1) 个连续 token（行内标签同序列，
    # HF 内部移位一次 → 每行 seq_len 个预测）
    row_len = cfg.seq_len + 1
    step_tokens = cfg.batch_size * cfg.seq_len
    total_steps = cfg.token_budget // (cfg.batch_size * row_len)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr_peak,
                            weight_decay=cfg.weight_decay)

    # 训练流 → 窗口
    windows = []
    for i in range(total_steps):
        seg = stream_ids[i * cfg.batch_size * row_len:
                         (i + 1) * cfg.batch_size * row_len]
        rows = [seg[j * row_len: (j + 1) * row_len]
                for j in range(cfg.batch_size)]
        windows.append(torch.tensor(rows, device=device))

    lrs, losses, eval_curve = [], [], []
    checkpoints: dict[int, dict] = {}
    diverged_at = None
    peak_mib = 0.0
    t0 = time.perf_counter()
    model.train()
    torch.cuda.reset_peak_memory_stats()

    for step in range(1, total_steps + 1):
        lr = lr_two_stage(step, total_steps, cfg.lr_peak,
                          cfg.rewarm_frac, cfg.floor_frac, cfg.rewarm)
        for g in opt.param_groups:
            g["lr"] = lr
        batch = windows[step - 1]
        out = model(batch, labels=batch)
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        lrs.append(lr)
        losses.append(loss.item())
        if math.isnan(loss.item()) or loss.item() > cfg.diverge_thresh:
            diverged_at = step
            break
        if (cfg.checkpoint_every and step % cfg.checkpoint_every == 0
                and len(checkpoints) < 6):
            # 封顶 6 个：失败格 ~10 步内发散足够用；若未发散防内存爆
            checkpoints[step] = snapshot_params(model)
        if step % cfg.eval_every == 0 or step == total_steps:
            quick = {}
            for dom, txt in eval_texts.items():
                r = evaluate_text(model, tok, txt, cfg.seq_len, device,
                                  max_tokens=cfg.eval_max_tokens)
                quick[dom] = round(r["ppl"], 4)
            eval_curve.append({"step": step, "tokens_seen": step * step_tokens,
                               "ppl": quick})
            model.train()

    wall = time.perf_counter() - t0
    if torch.cuda.is_available():
        peak_mib = torch.cuda.max_memory_allocated() / 1024 ** 2

    # 终评（全量固定文本）+ 遗忘量化
    final_eval = {dom: evaluate_text(model, tok, txt, cfg.seq_len, device)
                  for dom, txt in eval_texts.items()}
    result = {
        "tag": cfg.tag,
        "config": {"lr_peak": cfg.lr_peak, "replay_ratio": cfg.replay_ratio,
                   "rewarm": cfg.rewarm, "token_budget": cfg.token_budget,
                   "total_steps": total_steps, "seed": cfg.seed},
        "train_loss": {"first": round(losses[0], 4),
                       "final": round(losses[-1], 4),
                       "min": round(min(losses), 4),
                       "curve": [round(v, 4) for v in losses]},
        "lr_curve": [round(v, 10) for v in lrs],
        "eval_curve": eval_curve,
        "final_eval": {k: {m: round(v, 6) if isinstance(v, float) else v
                           for m, v in r.items()} for k, r in final_eval.items()},
        "diverged_at": diverged_at,
        "wall_s": round(wall, 1),
        "peak_mib": round(peak_mib, 1),
        "tps": round(cfg.token_budget / max(wall, 1e-9), 1),
        "steps_done": len(losses),
    }
    if base_probe is not None:
        cur_probe = capture_probe(model, probe_ids, device)
        result["repr_sim"] = round(representation_similarity(base_probe, cur_probe), 6)
    if base_params is not None:
        drift = param_drift(base_params, model)
        result["param_drift"] = {k: (round(v, 6) if isinstance(v, float) else v)
                                 for k, v in drift.items()}
    if checkpoints:
        result["checkpoints"] = checkpoints   # 内存对象，序列化前取用
    return result
