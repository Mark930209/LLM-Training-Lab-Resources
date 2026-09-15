#!/usr/bin/env bash
# sync_project.sh —— 把 Windows 侧工程同步到 WSL 训练目录（幂等）
set -euo pipefail
SRC=/mnt/d/DocProjects/LearnLLMFromDraft/DevResources/Season1/03/project
DST=~/llm-training-lab
cp -r "$SRC"/. "$DST"/
# 语料构建脚本与检索脚本也放进工程 scripts/，方便读者复现
mkdir -p "$DST/scripts"
cp /mnt/d/DocProjects/LearnLLMFromDraft/DevResources/Season1/03/scripts/*.py "$DST/scripts/" 2>/dev/null || true
cp /mnt/d/DocProjects/LearnLLMFromDraft/DevResources/Season1/03/scripts/qa_set.json "$DST/scripts/" 2>/dev/null || true
echo "SYNC_OK: $(ls "$DST/exp_superminigpt" | tr '\n' ' ')"