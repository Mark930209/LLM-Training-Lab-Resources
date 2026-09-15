#!/usr/bin/env bash
# collect_data.sh —— 03 篇全量实验采集：main + 三组消融/失败实验
# 产出落盘 results/Season1/03/（文章全部数字的出处）
set -uo pipefail
cd ~/llm-training-lab
source .venv/bin/activate
export PYTHONPATH=.

OUT=/mnt/d/DocProjects/LearnLLMFromDraft/results/Season1/03
mkdir -p "$OUT"

echo "===== 全量四组实验（约 5~10 分钟）====="
python -m exp_superminigpt.train --mode all \
  --override experiment.gen_tokens=400 \
  --output "$OUT/experiments.json"

echo "===== 环境卡片 ====="
cp runs/$(ls -t runs | head -1)/env_card.txt "$OUT/env_card.txt" 2>/dev/null || true
cat "$OUT/env_card.txt" 2>/dev/null || true

echo "COLLECT_DONE"
