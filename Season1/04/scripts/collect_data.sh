#!/usr/bin/env bash
# collect_data.sh —— 04 篇实验数据采集（全量，约 2 小时）
#
# 设计原则：每个实验只变一个变量，且"证明什么"决定在哪个规模上做：
#   - 规模效应（10M/30M/100M）→ 大语料，测参数量/显存/速度/收敛
#   - 数据规模效应（小语料 vs 大语料）→ 固定 10M 模型，只换语料
#   - AMP 消融 → 在 100M 上做（显存收益在最大模型上最明显）
#   - schedule / accumulation 消融 → 在 30M 上做（收敛差异需要长训练，
#     30M 跑得起更长步数，比 100M 更划算）
#
# 用法：
#   bash scripts/collect_data.sh            # 全量（约 2 小时）
#   bash scripts/collect_data.sh --quick    # 快速验证（每档 100 步）
set -uo pipefail

OUT="${OUT:-/mnt/d/DocProjects/LearnLLMFromDraft/results/Season1/04}"
LOG_DIR="$HOME/llm-training-lab/exp_scale/logs"
QUICK="${1:-}"
mkdir -p "$OUT" "$LOG_DIR"

run() {
    local name="$1"; shift
    echo ""
    echo "########## [$name] $(date '+%H:%M:%S') ##########"
    ./.venv/bin/python -m exp_scale.train "$@" --output "$OUT/$name.json" \
        2>&1 | tail -30
    echo "########## [$name] done $(date '+%H:%M:%S') ##########"
}

if [[ "$QUICK" == "--quick" ]]; then
    run quick_10m  --config exp_scale/config_10m.yaml  --set experiment.steps=100 \
        --set experiment.eval_every=50
    run quick_100m --config exp_scale/config_100m.yaml --set experiment.steps=100 \
        --set experiment.eval_every=50
    exit 0
fi

echo "===== 开始全量采集 $(date) ====="

# ---- 1. 规模效应：10M / 30M / 100M（大语料）----
run scale_10m  --config exp_scale/config_10m.yaml
run scale_30m  --config exp_scale/config_30m.yaml
run scale_100m --config exp_scale/config_100m.yaml

# ---- 2. 数据规模效应：固定 10M，小语料 vs 大语料 ----
# 2×2 矩阵：{10M, 100M} × {小语料, 大语料}
#   本脚本负责 scale_10m（10M×大）、corpus_small（10M×小）
#   缺失的 100M×小 反例格由 collect_fill_gaps.sh 补跑
# 小语料过拟合更快，加密验证频率以捕捉拐点
run corpus_small --config exp_scale/config_10m.yaml \
    --set experiment.name=corpus_small --set experiment.corpus=small \
    --set experiment.steps=3000 --set experiment.eval_every=100

# ---- 3. AMP 消融（100M，显存收益最明显）----
run no_amp --config exp_scale/config_100m.yaml --mode no_amp \
    --set experiment.name=no_amp --set experiment.steps=3000

# ---- 4. schedule 消融（30M，收敛差异需要长训练）----
run no_sched --config exp_scale/config_30m.yaml --mode no_sched \
    --set experiment.name=no_sched

# ---- 5. accumulation 消融（30M，等效 batch 对 loss 的影响）----
run no_accum --config exp_scale/config_30m.yaml --mode no_accum \
    --set experiment.name=no_accum

echo ""
echo "===== 全量采集完成 $(date) ====="