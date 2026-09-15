#!/usr/bin/env bash
# smoke_model.sh —— 03 篇 SuperMiniGPT 代码冒烟：forward/backward/generate 各跑一遍
set -euo pipefail
cd ~/llm-training-lab
source .venv/bin/activate
export PYTHONPATH=.

python - <<'PY'
import sys, torch
sys.path.insert(0, ".")
from exp_superminigpt.model import SuperMiniGPT

torch.manual_seed(42)
m = SuperMiniGPT(vocab_size=65, hidden=192, layers=6, heads=6, seq_len=128)
n = sum(p.numel() for p in m.parameters())
print(f"params={n/1e6:.2f}M")

x = torch.randint(0, 65, (2, 64))
logits = m(x)
print("forward OK:", tuple(logits.shape))

loss = torch.nn.functional.cross_entropy(
    logits.reshape(-1, 65), torch.randint(0, 65, (2, 64)).reshape(-1))
loss.backward()
print("backward OK: loss=%.4f" % loss.item())

out = m.generate(torch.zeros((1, 1), dtype=torch.long), 20, temperature=1.0, top_k=40)
print("generate OK: len =", out.shape[1])

# 消融开关检查
m2 = SuperMiniGPT(vocab_size=65, use_mask=False, use_rope=False)
print("ablation flags OK")
print("SMOKE_OK")
PY
