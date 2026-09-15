#!/usr/bin/env bash
# nan_hunt.sh —— 03 篇失败案例补跑：去梯度裁剪 + 更大学习率，制造真实 NaN
# 对照组：overscale（lr 3e-3，有 clip 1.0，收敛）→ 说明 clip 的保护作用
# 实验组：lr 1e-2 无裁剪 → 预期若干步内 loss 变 NaN
set -euo pipefail
cd ~/llm-training-lab
source .venv/bin/activate
export PYTHONPATH=.

OUT=/mnt/d/DocProjects/LearnLLMFromDraft/results/Season1/03

python - <<PY
import json, sys
sys.path.insert(0, ".")
from pathlib import Path
from exp_superminigpt import train as T

cfg = {
    "experiment": {"name": "supersuperminigpt_nan", "seed": 42, "steps": 3000, "warmup_steps": 0,
                    "hidden": 192, "layers": 6, "heads": 6, "seq_len": 128,
                    "lr": 5e-2, "eval_every": 200, "log_every": 50, "gen_tokens": 200},
    "hardware": {"device": "cuda", "batch_size": 32, "num_workers": 2},
}
# 临时关闭梯度裁剪：train.py 的 run() 内置 clip，这里复制其逻辑并去掉
import torch, torch.nn as nn, math, time
from torch.utils.data import DataLoader
from common import config as cfg_mod, logging as log_mod
from common.reproducibility import set_all_seeds
from exp_superminigpt.data import load_corpus
from exp_superminigpt.model import SuperMiniGPT

exp, hw = cfg["experiment"], cfg["hardware"]
set_all_seeds(exp["seed"], deterministic=False)
tok, train_ds, val_ds = load_corpus(Path("exp_superminigpt/data"), exp["seq_len"])
train_loader = DataLoader(train_ds, batch_size=hw["batch_size"], shuffle=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=hw["batch_size"], shuffle=False, drop_last=True)

model = SuperMiniGPT(tok.vocab_size, exp["hidden"], exp["layers"], exp["heads"], exp["seq_len"]).to("cuda")
opt = torch.optim.AdamW(model.parameters(), lr=exp["lr"], weight_decay=0.1)
rl = log_mod.RunLogger("runs", "supersuperminigpt_nan_lr5e-2_noclip")
rl.write_env_card(log_mod.env_card())

history, nan_at = [], None
step = 0
model.train()
while step < exp["steps"]:
    for x, y in train_loader:
        if step >= exp["steps"]:
            break
        logits = model(x.to("cuda"))
        loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.to("cuda").reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # 注意：没有 clip_grad_norm_
        opt.step()
        step += 1
        lv = loss.item()
        if step % 10 == 0 or not math.isfinite(lv):
            history.append((step, round(lv, 4)))
            rl.log.info("step %d loss %.4f", step, lv)
        if not math.isfinite(lv) and nan_at is None:
            nan_at = step
        if nan_at is not None:
            break
    if nan_at is not None:
        break

out = {"mode": "nan_lr5e-2_noclip", "lr": exp["lr"], "clip": "none",
       "nan_at_step": nan_at, "loss_trace": history[:30],
       "conclusion": ("NaN 出现在 step %d，此时 loss 轨迹见 loss_trace" % nan_at) if nan_at
                      else "3000 步未出现 NaN（学习率仍不够大）"}
rl.log.info("result: %s", out)
rl.close()
Path("$OUT").mkdir(parents=True, exist_ok=True)
Path("$OUT/nan_experiment.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=2))
PY
echo "NAN_HUNT_DONE"
