#!/usr/bin/env bash
# sync_and_smoke.sh —— 同步改名后代码到 WSL 并冒烟验证
set -euo pipefail
cd ~/llm-training-lab
rm -rf exp_minigpt exp_superminigpt
cp -r /mnt/d/DocProjects/LearnLLMFromDraft/DevResources/Season1/03/project/exp_superminigpt .
mkdir -p scripts
cp /mnt/d/DocProjects/LearnLLMFromDraft/DevResources/Season1/03/scripts/*.py scripts/ 2>/dev/null || true
cp /mnt/d/DocProjects/LearnLLMFromDraft/DevResources/Season1/03/scripts/qa_set.json scripts/ 2>/dev/null || true

source .venv/bin/activate
export PYTHONPATH=.

python - <<'PY'
import torch
from exp_superminigpt.model import SuperMiniGPT

m = SuperMiniGPT(vocab_size=65, hidden=192, layers=6, heads=6, seq_len=128)
x = torch.randint(0, 65, (2, 32))
print("forward OK:", tuple(m(x).shape))
n = sum(p.numel() for p in m.parameters())
print(f"params={n/1e6:.2f}M")
out = m.generate(torch.zeros((1,1), dtype=torch.long), 20)
print("generate OK:", out.shape[1])
PY

# 语料加载验证（西游记路径）
python - <<'PY'
from pathlib import Path
from exp_superminigpt.data import load_corpus
tok, tr, va = load_corpus(Path("exp_superminigpt/data"), 128, corpus="xiyouji")
print(f"xiyouji vocab={tok.vocab_size} train={len(tr)} val={len(va)}")
PY
echo "SMOKE_OK"