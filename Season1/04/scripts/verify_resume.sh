#!/usr/bin/env bash
# verify_resume.sh —— resume 验证：中断 → 恢复 → loss 连续性
#
# 可复用的训练程序必须能"接着训"。03 篇只存权重，恢复后优化器动量、
# 学习率进度全丢，续训等于重新开始。本脚本验证 04 的 checkpoint
# 是否真的能无缝续训。
#
# 验证方法：
#   1. 训 200 步，存 checkpoint
#   2. 从 checkpoint 恢复，再训 200 步
#   3. 对照：一次训 400 步的 loss 曲线
#   若恢复后的曲线与连续训练吻合，说明 resume 正确。
#
# 用法：bash scripts/verify_resume.sh
set -euo pipefail

OUT="${OUT:-/mnt/d/DocProjects/LearnLLMFromDraft/results/Season1/04}"
mkdir -p "$OUT"

echo "=== 1. 训 200 步，存 checkpoint ==="
./.venv/bin/python -m exp_scale.train --config exp_scale/config_10m.yaml \
    --set experiment.steps=200 --set experiment.eval_every=100 \
    --output "$OUT/resume_part1.json" 2>&1 | tail -8

CKPT=$(ls -td runs/scale_main_scale_10m_* | head -1)/ckpt_last.pt
echo "checkpoint: $CKPT"

echo "=== 2. 从 checkpoint 恢复，再训 200 步 ==="
./.venv/bin/python -m exp_scale.train --config exp_scale/config_10m.yaml \
    --set experiment.steps=400 --set experiment.eval_every=100 \
    --resume "$CKPT" --output "$OUT/resume_part2.json" 2>&1 | tail -8

echo "=== 3. 对照：一次训 400 步 ==="
./.venv/bin/python -m exp_scale.train --config exp_scale/config_10m.yaml \
    --set experiment.steps=400 --set experiment.eval_every=100 \
    --output "$OUT/resume_continuous.json" 2>&1 | tail -8

echo "=== 完成，对照 resume_part2.json 与 resume_continuous.json 的 val_curve ==="