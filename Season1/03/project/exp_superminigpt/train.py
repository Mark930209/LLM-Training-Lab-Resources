"""exp_superminigpt/train.py —— SuperMiniGPT v0 训练入口。

四组实验一次跑完（按文章推进链排序）：
    main       完整模型在真实语料上训练（主实验）
    no_mask    去掉因果掩码（失败实验：信息泄露，loss 虚低但生成乱码）
    no_rope    去掉位置编码（消融：顺序信息丢失，loss 明显更差）
    overscale  学习率放大 10 倍（失败实验：NaN）

用法（在 project/ 目录下）：
    python -m exp_superminigpt.train --mode main
    python -m exp_superminigpt.train --mode no_mask
    python -m exp_superminigpt.train --mode all --output results.json
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
from common.reproducibility import set_all_seeds  # noqa: E402
from exp_superminigpt.data import load_corpus  # noqa: E402
from exp_superminigpt.model import SuperMiniGPT  # noqa: E402


@torch.no_grad()
def estimate_loss(model, loader, device, iters: int = 20) -> float:
    """验证集 loss：判断"模型在学还是在背"的唯一标尺。"""
    model.eval()
    losses = []
    for i, (x, y) in enumerate(loader):
        if i >= iters:
            break
        logits = model(x.to(device))
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), y.to(device).reshape(-1))
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def run(cfg: dict, mode: str = "main") -> dict:
    exp, hw = cfg["experiment"], cfg["hardware"]
    device = hw["device"]
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("config 要求 cuda 但 GPU 不可用；显式设置 hardware.device=cpu")
    set_all_seeds(exp["seed"], deterministic=False)

    corpus = exp.get("corpus", "shakespeare")
    tok, train_ds, val_ds = load_corpus(Path(__file__).parent / "data",
                                        exp["seq_len"], corpus=corpus)
    train_loader = DataLoader(train_ds, batch_size=hw["batch_size"], shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=hw["batch_size"], shuffle=False,
                            drop_last=True)

    use_mask = mode != "no_mask"
    use_rope = mode != "no_rope"
    use_residual = mode != "no_residual"
    lr = exp["lr"] * (10.0 if mode == "overscale" else 1.0)

    model = SuperMiniGPT(tok.vocab_size, exp["hidden"], exp["layers"], exp["heads"],
                    exp["seq_len"], use_mask=use_mask, use_rope=use_rope,
                    use_residual=use_residual).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)

    rl = log_mod.RunLogger("runs", f"superminigpt_{mode}")
    rl.write_env_card(log_mod.env_card())
    cfg_mod.snapshot(cfg_mod.Config(cfg), rl.run_dir)
    rl.log.info("mode=%s params=%.2fM vocab=%d lr=%g",
                mode, n_params / 1e6, tok.vocab_size, lr)

    history = {"train": [], "val": []}
    step = 0
    t_start = time.perf_counter()
    nan_at = None
    eval_every, log_every = exp["eval_every"], exp["log_every"]
    total_steps = exp["steps"]

    model.train()
    while step < total_steps:
        for x, y in train_loader:
            if step >= total_steps:
                break
            logits = model(x.to(device))
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), y.to(device).reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1

            if not math.isfinite(loss.item()) and nan_at is None:
                nan_at = step          # NaN 一旦出现，训练已无意义，记下位置
            if step % log_every == 0:
                history["train"].append(round(loss.item(), 4))
                rl.log.info("step %d/%d loss %.4f", step, total_steps, loss.item())
            if step % eval_every == 0 or step == total_steps:
                vl = estimate_loss(model, val_loader, device)
                history["val"].append(round(vl, 4))
                rl.log.info("step %d val_loss %.4f", step, vl)
            if nan_at is not None:
                break
        if nan_at is not None:
            break

    wall_s = time.perf_counter() - t_start
    final_val = history["val"][-1] if history["val"] else float("nan")

    # 保存权重：训练产物必须可复用（问答/继续训练都依赖它）
    ckpt_path = rl.run_dir / "model.pt"
    torch.save({"model": model.state_dict(), "vocab_size": tok.vocab_size,
                "config": {"hidden": exp["hidden"], "layers": exp["layers"],
                           "heads": exp["heads"], "seq_len": exp["seq_len"]}},
               ckpt_path)
    rl.log.info("权重已保存: %s", ckpt_path)

    # 生成样本：训练前后的对比是"能学习"最直观的证据
    samples = {}
    ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
    if nan_at is None:
        for temp, tag in [(1.0, "temp1.0"), (0.8, "temp0.8")]:
            out = model.generate(ctx.clone(), exp["gen_tokens"], temperature=temp, top_k=40)
            samples[tag] = tok.decode(out[0].tolist())

    result = {
        "mode": mode,
        "params_million": round(n_params / 1e6, 2),
        "vocab_size": tok.vocab_size,
        "lr": lr,
        "final_val_loss": final_val,
        "best_val_loss": min(history["val"]) if history["val"] else float("nan"),
        "initial_train_loss": history["train"][0] if history["train"] else float("nan"),
        "nan_at_step": nan_at,
        "wall_time_s": round(wall_s, 1),
        "train_curve": history["train"],
        "val_curve": history["val"],
        "sample": samples.get("temp0.8", ""),
        "sample_temp1.0": samples.get("temp1.0", ""),
    }
    rl.log_metrics(**{k: v for k, v in result.items() if not isinstance(v, list)})
    run_dir = rl.close()

    out = dict(result)
    out["run_dir"] = str(run_dir)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="exp_superminigpt/config.yaml")
    ap.add_argument("--mode", default="main",
                    choices=["main", "no_mask", "no_rope", "no_residual", "overscale", "all"])
    ap.add_argument("--override", action="append", default=[])
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    base = cfg_mod.load_config(args.config, args.override)
    modes = ["main", "no_mask", "no_rope", "overscale"] if args.mode == "all" else [args.mode]

    results = {}
    for m in modes:
        import copy
        results[m] = run(copy.deepcopy(dict(base)), mode=m)

    print(json.dumps(results, ensure_ascii=False, indent=2))
    if args.output:
        Path(args.output).write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
