#!/usr/bin/env bash
# collect_fill_gaps.sh —— 补齐 2×2 对照矩阵的缺失格（主采集脚本跑完后运行）
#
# 背景：config_10m.yaml 默认 corpus=small，导致首次 scale_10m 实际跑的是
#       "10M × 小语料"（已归档为 corpus_10m_small.json）。三个规模对照
#       必须统一用大语料，只变模型大小，因此需要补跑：
#         B  10M  × 大语料（与 A 同为 3000 步，只变数据）
#         C  100M × 小语料（与 D 同为 8000 步，只变数据，预期过拟合最猛）
#
# 用法：bash scripts/collect_fill_gaps.sh
set -uo pipefail

OUT="${OUT:-/mnt/d/DocProjects/LearnLLMFromDraft/results/Season1/04}"
mkdir -p "$OUT"

run() {
    local name="$1"; shift
    echo ""
    echo "########## [$name] $(date '+%H:%M:%S') ##########"
    ./.venv/bin/python -m exp_scale.train "$@" --output "$OUT/$name.json" \
        2>&1 | tail -30
    echo "########## [$name] done $(date '+%H:%M:%S') ##########"
}

echo "===== 补齐对照矩阵 $(date) ====="

# B: 10M × 大语料（3000 步，与 A 对齐）
run scale_10m_large --config exp_scale/config_10m.yaml \
    --set experiment.name=scale_10m_large --set experiment.corpus=large \
    --set experiment.steps=3000

# C: 100M × 小语料（8000 步，与 D 对齐；数据不放大时 100M 过拟合最狠）
run scale_100m_small --config exp_scale/config_100m.yaml \
    --set experiment.name=scale_100m_small --set experiment.corpus=small \
    --set experiment.steps=8000

echo ""
echo "===== 补齐完成 $(date) ====="