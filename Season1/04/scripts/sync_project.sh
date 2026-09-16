#!/usr/bin/env bash
# sync_project.sh —— Windows 侧工程同步到 WSL 训练目录
#
# 用法（Windows PowerShell 或 WSL 内均可）：
#   bash scripts/sync_project.sh
set -euo pipefail

SRC="/mnt/d/DocProjects/LearnLLMFromDraft/DevResources/Season1/04"
DST="$HOME/llm-training-lab"

echo "同步 $SRC → $DST"
mkdir -p "$DST/exp_scale" "$DST/scripts"

# 工程代码（不含 data/，语料单独构建）
cp -r "$SRC/project/common" "$DST/" 2>/dev/null || true
cp "$SRC/project/exp_scale/"*.py "$DST/exp_scale/"
cp "$SRC/project/exp_scale/"*.yaml "$DST/exp_scale/"
cp "$SRC/scripts/"*.py "$DST/scripts/" 2>/dev/null || true
cp "$SRC/scripts/"*.sh "$DST/scripts/" 2>/dev/null || true

echo "同步完成。语料需单独构建："
echo "  ./.venv/bin/python exp_scale/build_corpus.py --out-dir exp_scale/data"