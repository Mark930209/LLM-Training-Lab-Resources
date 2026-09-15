#!/usr/bin/env bash
# quick_train.sh —— 03 篇训练循环快速验证（200 步，确认 loss 下降与语料加载）
set -euo pipefail
cd ~/llm-training-lab
source .venv/bin/activate
export PYTHONPATH=.

python -m exp_superminigpt.train --mode main \
  --override experiment.steps=200 \
  --override experiment.eval_every=100 \
  --override experiment.log_every=25 \
  --override experiment.gen_tokens=100 \
  --output /mnt/d/DocProjects/LearnLLMFromDraft/results/Season1/03/quick_train.json
