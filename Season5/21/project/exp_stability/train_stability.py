"""train_stability.py —— 21 篇的训练循环：三种精度、坏 batch 注入、
步级指标采集（grad_norm/update_ratio/loss_scale）、spike 处置策略。

与 17/19/20 篇同口径（同 AdamW、同 dataset seeding、同 evaluate_ppl），
差别在：autocast 精度包装、GradScaler（fp16）、注入与处置钩子、步级指标。
"""

from __future__ import annotations

import math
import time
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from exp_recipe.train_eval import TrainConfig, evaluate_ppl
from exp_scale.data import LMDataset

from .hparam_bridge import lr_at   # 复用 20 篇的调度（cosine/linear/constant）
from .stability_metrics import detect_spike, median


def build_modern(hidden: int, base: dict, vocab: int):
    """建 19 篇 modern 架构（与 20 篇 train_hparam.build_modern 同款）。"""
    from exp_arch.train_arch import build_model
    from exp_arch.model_arch import kv_heads_for
    arch = base["arch"]
    spec = {
        "hidden": hidden,
        "norm_type": arch["norm_type"], "norm_pos": arch["norm_pos"],
        "ffn_type": arch["ffn_type"], "pos_enc": arch["pos_enc"],
        "attn_type": arch["attn_type"],
        "kv_heads": kv_heads_for(arch["attn_type"], base["heads"]),
    }
    b = {"layers": base["layers"], "heads": base["heads"], "seq_len": base["seq_len"]}
    return build_model(spec, vocab, b)


def _autocast_ctx(dtype: str, device: str):
    if device != "cuda":
        return nullcontext()
    if dtype == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if dtype == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()   # fp32


def train_stability(train_ids: torch.Tensor, model, cfg: TrainConfig, *,
                    dtype: str = "fp32", clip: float | None = 1.0,
                    decay: str = "cosine",
                    inject_every: int | None = None, bad_factor: float = 1.0,
                    corrupt_frac: float = 0.0, vocab_size: int | None = None,
                    lr_ramp: dict | None = None,
                    strategy: str | None = None, spike_k: float = 3.0,
                    ckpt_every: int = 40, log_every: int = 1,
                    memorize_from: int | None = None, subset_frac: float = 1.0,
                    eval_fn=None, eval_every: int = 0) -> dict:
    """统一训练循环。

    dtype        : fp32 / fp16（GradScaler）/ bf16（autocast 无 scaler）
    clip         : 梯度裁剪阈值，None = 关闭
    inject_every : 每 N 步注入一次坏 batch（异常样本）
    corrupt_frac : 坏 batch 里被替换成随机错误 token 的目标比例（模拟脏标签）。
                   这是 spike 的主注入机制——污染目标让 cross_entropy 真实飙到
                   ~log(vocab)，loss 尖峰可见、梯度方向有害。bad_factor 仅作
                   额外的梯度放大（对 AdamW 效果弱，因自适应归一化会吸收单次放大）。
    vocab_size   : 污染目标时随机 token 的上界（必填当 corrupt_frac>0）
    lr_ramp      : LR 渐进攀升场景 {"from": 安全值, "to": 失稳值, "start_frac": 开始攀升的步数比例}。
                   模拟真实训练里 LR 过高导致的渐进失稳：grad_norm 先逐步爬升（前兆），
                   若干步后 loss 爆炸（后果）——告警提前量的正样本。启用时覆盖 lr_at 调度。
    strategy     : None / skip_bad / lower_lr / rollback_rewarmup / tighten_clip
    memorize_from: 从该步起只用前 subset_frac 的数据（静默退化场景）
    eval_fn      : 回调（step, model）→ eval ppl，用于退化曲线
    """
    torch.manual_seed(cfg.seed)
    device = cfg.device if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    full_ids = train_ids
    if memorize_from is not None and subset_frac < 1.0:
        sub_ids = train_ids[:int(len(train_ids) * subset_frac)]
    else:
        sub_ids = None

    def make_loader(ids):
        return torch.utils.data.DataLoader(
            LMDataset(ids, cfg.seq_len), batch_size=cfg.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(cfg.seed), drop_last=True)

    loader = make_loader(full_ids)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == "fp16" and device == "cuda"))

    tokens_per_step = cfg.batch_size * cfg.seq_len
    total_steps = max(1, cfg.token_budget // tokens_per_step)

    metrics = []          # 步级指标（grad_norm/update_ratio/loss_scale/loss）
    scale_history = []    # fp16 的 loss scale 历史（独立于 log_every，供 scale_adjusts）
    loss_hist = []        # spike 检测窗口
    ckpt = None           # rollback 用（state_dict 浅拷贝到 CPU）
    handled_inject = set()  # rollback 后隔离的注入步（否则重训到同步号再遇同一坏
                            # batch → 再 spike → 再回退，无限循环，实测卡死 14 分钟）
    rewarm_left = 0       # re-warmup 剩余步数（>0 时 LR 从 0 线性升回）
    rewarm_total = 0
    skipped = 0
    rollbacks = 0
    lr_now = cfg.lr
    clip_now = clip
    diverged = False
    started = time.perf_counter()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model.train()
    data_iter = iter(loader)
    step = 0
    last_loss = None
    eval_curve = []

    while step < total_steps:
        # 静默退化切换
        if memorize_from is not None and step == memorize_from and sub_ids is not None:
            loader = make_loader(sub_ids)
            data_iter = iter(loader)

        for g in opt.param_groups:
            if lr_ramp is not None and step >= int(total_steps * lr_ramp["start_frac"]):
                rs = int(total_steps * lr_ramp["start_frac"])
                prog = (step - rs) / max(1, total_steps - rs)
                g["lr"] = lr_ramp["from"] + (lr_ramp["to"] - lr_ramp["from"]) * prog
            elif rewarm_left > 0:
                # re-warmup：回退后 LR 从 0 线性升回当前调度值，避免二次冲击
                g["lr"] = lr_at(step, total_steps, lr_now, cfg.warmup_frac, decay) \
                    * (rewarm_total - rewarm_left) / rewarm_total
            else:
                g["lr"] = lr_at(step, total_steps, lr_now, cfg.warmup_frac, decay)
        if rewarm_left > 0:
            rewarm_left -= 1

        try:
            xb, yb = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            xb, yb = next(data_iter)
        xb, yb = xb.to(device), yb.to(device)

        is_bad = (inject_every is not None and step > 0
                  and step % inject_every == 0 and step not in handled_inject)
        # 异常样本注入：污染目标 token（模拟脏标签），让 cross_entropy 真实飙升到
        # ~log(vocab)。这是 spike 的主机制——loss 尖峰可见、梯度方向有害。
        # bad_factor 仅作额外梯度放大（对 AdamW 效果弱，自适应归一化吸收单次放大）。
        if is_bad and corrupt_frac > 0 and vocab_size is not None:
            mask = torch.rand(yb.shape, device=device) < corrupt_frac
            rand_tok = torch.randint(0, vocab_size, yb.shape, device=device)
            yb = torch.where(mask, rand_tok, yb)

        # 前一步参数快照（update_ratio 用）
        with torch.no_grad():
            p_norm_before = sum(p.detach().float().pow(2).sum().item()
                                for p in model.parameters()) ** 0.5

        with _autocast_ctx(dtype, device):
            logits = model(xb)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))

        eff_loss = loss * bad_factor if is_bad else loss

        opt.zero_grad(set_to_none=True)
        scaler.scale(eff_loss).backward()

        # grad_norm（fp16 先 unscale 再量，与 clip 同口径；scaler.step 不会重复 unscale）
        if dtype == "fp16" and device == "cuda":
            scaler.unscale_(opt)
        if clip_now is not None:
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_now).item()
        else:
            gn = math.sqrt(sum(p.grad.detach().float().pow(2).sum().item()
                               for p in model.parameters() if p.grad is not None))

        lv = loss.item()
        spike = detect_spike(lv, loss_hist[-50:], spike_k)

        # 处置策略
        acted = None
        if strategy and spike:
            if strategy == "skip_bad":
                if dtype == "fp16" and device == "cuda":
                    scaler.update()      # inf 梯度会让 scaler 降 scale（跳批语义）
                    scale_history.append(scaler.get_scale())
                skipped += 1
                acted = "skip"
                loss_hist.append(lv)
                if log_every and step % log_every == 0:
                    metrics.append({"step": step, "loss": round(lv, 4),
                                    "grad_norm": round(gn, 4), "spike": True,
                                    "acted": acted,
                                    "loss_scale": scaler.get_scale() if dtype == "fp16" else None})
                step += 1
                continue
            if strategy == "lower_lr":
                lr_now *= 0.5
                acted = f"lr→{lr_now:.2e}"
            elif strategy == "tighten_clip":
                clip_now = 0.1
                acted = "clip→0.1"
            elif strategy == "no_action":
                acted = "none"      # 对照组：记录但不处置，坏更新照常落地
            elif strategy == "rollback_rewarmup":
                if ckpt is not None:
                    model.load_state_dict(ckpt["model"])
                    if ckpt["opt"] is not None:
                        opt.load_state_dict(ckpt["opt"])
                    step = ckpt["step"]
                    loss_hist = loss_hist[:step]
                    rollbacks += 1
                    # 隔离下一个注入步：重训到同步号不再注入同一坏 batch
                    # （否则回退→重训→同步号再遇坏 batch→再回退，无限循环，
                    # 实测卡死 14 分钟）；现实语义 = 定位并跳过问题数据
                    if inject_every:
                        nxt = step + (inject_every - step % inject_every)
                        if step % inject_every == 0:
                            nxt = step
                        handled_inject.add(nxt)
                    # re-warmup：LR 从 0 线性升回，避免回退后立即大 LR 二次冲击
                    rewarm_total = max(1, int(total_steps * cfg.warmup_frac))
                    rewarm_left = rewarm_total
                    acted = f"rollback→{step}"
                    data_iter = iter(loader)
                    # 回退后跳过本步更新：坏 batch 的梯度不得落到刚恢复的参数上。
                    # fp16 必须 scaler.update() 重置 unscale 状态（本步已为量 grad_norm
                    # 调过 unscale_，不走 step/update 的话下次迭代再 unscale_ 会报
                    # "already been called"）
                    opt.zero_grad(set_to_none=True)
                    if dtype == "fp16" and device == "cuda":
                        scaler.update()
                        scale_history.append(scaler.get_scale())
                    if log_every and step % log_every == 0:
                        metrics.append({"step": step, "loss": round(lv, 4),
                                        "grad_norm": round(gn, 4), "spike": True,
                                        "acted": acted,
                                        "loss_scale": scaler.get_scale() if dtype == "fp16" else None})
                    step += 1
                    continue

        scaler.step(opt)
        scaler.update()
        if dtype == "fp16" and device == "cuda":
            scale_history.append(scaler.get_scale())

        with torch.no_grad():
            p_norm_after = sum(p.detach().float().pow(2).sum().item()
                               for p in model.parameters()) ** 0.5
        update_ratio = (p_norm_after - p_norm_before) / max(p_norm_before, 1e-12)

        loss_hist.append(lv)
        last_loss = lv
        if math.isnan(lv) or math.isinf(lv) or lv > 20.0:
            diverged = True
            metrics.append({"step": step, "loss": None, "grad_norm": round(gn, 4),
                            "spike": True, "acted": acted, "diverged": True})
            break

        if log_every and step % log_every == 0:
            metrics.append({
                "step": step, "loss": round(lv, 4), "grad_norm": round(gn, 4),
                "update_ratio": round(update_ratio, 6), "spike": spike,
                "acted": acted, "bad_inject": is_bad,
                "loss_scale": scaler.get_scale() if dtype == "fp16" and device == "cuda" else None,
            })

        # checkpoint（rollback 策略用）
        if strategy == "rollback_rewarmup" and ckpt_every and (step + 1) % ckpt_every == 0:
            ckpt = {"step": step + 1,
                    "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    "opt": None}
            try:
                ckpt["opt"] = opt.state_dict()
            except Exception:
                ckpt["opt"] = None

        # 退化曲线评测
        if eval_fn is not None and eval_every and step > 0 and step % eval_every == 0:
            p = eval_fn(step, model)
            if p is not None:
                eval_curve.append({"step": step, "eval_ppl": p,
                                   "train_loss": round(lv, 4)})
            model.train()

        step += 1

    train_sec = time.perf_counter() - started
    peak_mib = round(torch.cuda.max_memory_allocated() / 1024**2, 1) if device == "cuda" else None
    scale_adjusts = None
    if dtype == "fp16" and device == "cuda":
        scale_adjusts = sum(1 for i in range(1, len(scale_history))
                            if scale_history[i] != scale_history[i - 1])

    return {
        "dtype": dtype, "clip": clip, "strategy": strategy,
        "final_train_loss": round(last_loss, 4) if last_loss is not None and not math.isnan(last_loss) else None,
        "diverged": diverged,
        "total_steps": step + (0 if diverged else 0),
        "train_sec": round(train_sec, 2),
        "tokens_per_sec": round(step * tokens_per_step / train_sec, 1) if train_sec > 0 else None,
        "peak_memory_mib": peak_mib,
        "skipped_batches": skipped, "rollbacks": rollbacks,
        "loss_scale_adjusts": scale_adjusts,
        "spike_count": sum(1 for m in metrics if m.get("spike")),
        "metrics": metrics if log_every else [],
        "eval_curve": eval_curve,
        "device": device,
    }


def eval_stability(model, tokenizer, nl_eval: str, code_eval: str, cfg: TrainConfig) -> dict:
    nl_ids = torch.tensor(tokenizer.encode(nl_eval), dtype=torch.long)
    code_ids = torch.tensor(tokenizer.encode(code_eval), dtype=torch.long)
    return {"nl": evaluate_ppl(model, nl_ids, cfg),
            "code": evaluate_ppl(model, code_ids, cfg)}
