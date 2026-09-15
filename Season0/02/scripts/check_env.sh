#!/usr/bin/env bash
# check_env.sh —— LLM Training Lab 通用环境体检脚本
#
# 用途：一键输出本机环境卡片（硬件/驱动/CUDA/torch/uv/磁盘），
#       可直接粘贴给大模型 agent 排错，或附在问题报告里。
# 特性：只读、幂等、非交互；任何一项缺失不中断，标"未检测到"继续。
#
# 用法：
#   bash check_env.sh [--project-dir DIR]
#   默认项目目录 ~/llm-training-lab

set -uo pipefail

# 注意：bash 变量名必须用 ASCII（非 ASCII 变量名会被当作命令名执行）
PROJ_DIR="$HOME/llm-training-lab"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-dir) PROJ_DIR="$2"; shift 2 ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
done

分隔() { echo "----------------------------------------"; }

echo "# 环境体检报告  $(date '+%Y-%m-%d %H:%M:%S')"
分隔

echo "## 1. 操作系统与内核"
uname -a
if grep -qi microsoft /proc/version 2>/dev/null; then
  echo "运行环境: WSL2"
else
  echo "运行环境: 原生 Linux"
fi
分隔

echo "## 2. GPU 与驱动（nvidia-smi）"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "未检测到 nvidia-smi（无 NVIDIA GPU，或 WSL 直通未生效）"
fi
分隔

echo "## 3. CPU 与内存"
grep -m1 'model name' /proc/cpuinfo || echo "未检测到 CPU 信息"
echo "CPU 核数: $(nproc)"
free -h | head -2
分隔

echo "## 4. 磁盘"
df -h "$HOME" | tail -1
分隔

echo "## 5. Python 工具链"
echo "系统 python3: $(python3 --version 2>&1 || echo 未安装)"
if command -v uv >/dev/null 2>&1 || [[ -x "$HOME/.local/bin/uv" ]]; then
  export PATH="$HOME/.local/bin:$PATH"
  echo "uv: $(uv --version)"
else
  echo "uv: 未安装"
fi
分隔

echo "## 6. 项目虚拟环境与 PyTorch"
if [[ -x "$PROJ_DIR/.venv/bin/python" ]]; then
  "$PROJ_DIR/.venv/bin/python" - <<'PY'
import sys
print("python:", sys.version.split()[0])
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda build:", torch.version.cuda)
    print("cudnn:", torch.backends.cudnn.version())
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print("device:", torch.cuda.get_device_name(0))
        print("vram GB:", round(p.total_memory / 1e9, 2))
        print("capability:", f"{p.major}.{p.minor}")
        free_b, total_b = torch.cuda.mem_get_info(0)
        print(f"实际可用显存 GB: {round(free_b/1e9,2)} / {round(total_b/1e9,2)}")
except ImportError:
    print("torch: 未安装")
PY
else
  echo "未找到 $PROJ_DIR/.venv（先运行 setup_env.sh）"
fi
分隔

echo "## 7. 依赖锁文件"
if [[ -f "$PROJ_DIR/requirements.lock.txt" ]]; then
  echo "存在: $PROJ_DIR/requirements.lock.txt（$(wc -l < "$PROJ_DIR/requirements.lock.txt") 个包）"
else
  echo "不存在（setup_env.sh 会自动生成）"
fi
分隔

echo "体检完成。把以上全部内容粘贴给 agent，配合报错原文即可获得排错建议。"
