#!/usr/bin/env bash
# quick_train.sh —— 快速验证训练程序（100 步，约 1 分钟）
#
# 已验证：forward/backward/AMP/accumulation/schedule/checkpoint 全流程
set -uo pipefail
cd "$HOME/llm-training-lab" || exit 1

echo "=== 10M + 大语料（100 步）==="
./.venv/bin/python -m exp_scale.train --config exp_scale/config_10m.yaml \
    --set experiment.steps=100 --set experiment.eval_every=50 2>&1 | tail -20

echo ""
echo "=== 100M + 大语料（100 步，验证 8GB 能不能跑）==="
./.venv/bin/python -m exp_scale.train --config exp_scale/config_100m.yaml \
    --set experiment.steps=100 --set experiment.eval_every=50 2>&1 | tail -20