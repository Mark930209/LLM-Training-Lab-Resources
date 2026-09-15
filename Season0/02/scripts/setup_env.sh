#!/usr/bin/env bash
# setup_env.sh —— LLM Training Lab 通用训练环境初始化脚本
#
# 适用：WSL2 Ubuntu 22.04/24.04 或原生 Ubuntu；NVIDIA GPU（CPU-only 机器同样可用）
# 特性：幂等（重复运行安全）、非交互（可被 agent/CI 直接调用）、自验证
#
# 用法：
#   bash setup_env.sh [选项]
#
# 选项（全部可省略，默认值即可用）：
#   --project-dir DIR   项目根目录（默认 ~/llm-training-lab）
#   --python-ver VER    Python 版本（默认 3.12）
#   --mirror URL        pip/uv 镜像源（默认清华；海外机器传 https://pypi.org/simple）
#   --packages "PKG..." 要安装的包（默认 "torch numpy pyyaml"）
#   --skip-apt          跳过 apt 系统包安装（无 sudo 权限时）
#
# 示例：
#   bash setup_env.sh --project-dir ~/lab --packages "torch numpy datasets"

set -euo pipefail

# ---------- 默认参数 ----------
# 注意：bash 变量名必须用 ASCII（非 ASCII 变量名在 POSIX locale 下会被当作命令名执行），
# 命令行参数保留中文
PROJ_DIR="$HOME/llm-training-lab"
PY_VER="3.12"
MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"
PKGS="torch numpy pyyaml"
SKIP_APT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-dir) PROJ_DIR="$2"; shift 2 ;;
    --python-ver)  PY_VER="$2"; shift 2 ;;
    --mirror)      MIRROR="$2"; shift 2 ;;
    --packages)    PKGS="$2"; shift 2 ;;
    --skip-apt)    SKIP_APT=1; shift ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
done

echo "== 0. 参数确认 =="
echo "项目目录: $PROJ_DIR"
echo "Python:   $PY_VER"
echo "镜像:     $MIRROR"
echo "依赖:     $PKGS"

echo "== 1. 系统依赖（apt，幂等） =="
if [[ $SKIP_APT -eq 0 ]]; then
  sudo apt-get update -qq
  # python3-venv：Ubuntu 24.04 最小安装缺失会导致 venv 创建失败（ensurepip 不可用）
  sudo apt-get install -y -qq "python${PY_VER}-venv" python3-pip curl
else
  echo "跳过 apt（--跳过apt）"
fi

echo "== 2. 安装 uv（幂等：已存在则跳过） =="
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
else
  echo "uv 已安装: $(uv --version)"
fi

echo "== 3. 配置 uv 镜像（国内网络必须，直连 pypi.org 易超时） =="
mkdir -p "$HOME/.config/uv"
cat > "$HOME/.config/uv/uv.toml" <<EOF
[[index]]
url = "$MIRROR"
default = true
EOF
cat "$HOME/.config/uv/uv.toml"

echo "== 4. 创建项目虚拟环境（--clear 保证非交互） =="
mkdir -p "$PROJ_DIR"
cd "$PROJ_DIR"
uv venv --clear --python "$PY_VER" .venv
echo "venv: $PROJ_DIR/.venv"

echo "== 5. 安装依赖 =="
# shellcheck disable=SC2086
uv pip install --python "$PROJ_DIR/.venv/bin/python" --index-url "$MIRROR" $PKGS

echo "== 6. 验证（torch + CUDA） =="
"$PROJ_DIR/.venv/bin/python" - <<'PY' | tee /tmp/torch_verify.log
import torch
print("torch:", torch.__version__)
print("cuda build:", torch.version.cuda)
print("cudnn:", torch.backends.cudnn.version())
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device count:", torch.cuda.device_count())
    print("device:", torch.cuda.get_device_name(0))
    p = torch.cuda.get_device_properties(0)
    print("vram GB:", round(p.total_memory / 1e9, 2))
    print("capability:", f"{p.major}.{p.minor}")
    print("sm count:", p.multi_processor_count)
else:
    print("提示: 未检测到 CUDA。若机器有 NVIDIA GPU，检查 torch.version.cuda 是否为 None（CPU 版）")
PY

echo "== 7. 锁定依赖版本 =="
uv pip freeze --python "$PROJ_DIR/.venv/bin/python" > "$PROJ_DIR/requirements.lock.txt"
echo "lock 文件: $PROJ_DIR/requirements.lock.txt（前 10 行）"
head -10 "$PROJ_DIR/requirements.lock.txt"

echo ""
echo "初始化完成。激活环境: source $PROJ_DIR/.venv/bin/activate"
echo "SETUP_DONE"
