#!/usr/bin/env bash
# train_xiyouji.sh —— 西游记语料训练（main + 采样验证）
# 前置：exp_superminigpt/data/xiyouji.txt 已由 build_corpus.py 产出
set -euo pipefail
cd ~/llm-training-lab
source .venv/bin/activate
export PYTHONPATH=.

OUT=/mnt/d/DocProjects/LearnLLMFromDraft/results/Season1/03
mkdir -p "$OUT"

echo "===== 西游记语料训练（3000 步）====="
python -m exp_superminigpt.train --config exp_superminigpt/config_xiyouji.yaml --mode main \
  --output "$OUT/xiyouji_train.json"

echo "===== Shakespeare 对照（同架构，仅换语料）====="
python -m exp_superminigpt.train --config exp_superminigpt/config.yaml --mode main \
  --override experiment.steps=3000 \
  --output "$OUT/shakespeare_3000.json"

echo "TRAIN_DONE"