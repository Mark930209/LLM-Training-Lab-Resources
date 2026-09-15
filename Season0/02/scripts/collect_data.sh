#!/usr/bin/env bash
# collect_data.sh —— 02 篇全部实验数据一键采集（WSL2 内运行）
#
# 前置：setup_env.sh 已成功（~/llm-training-lab/.venv 内有 torch）
# 产出：全部写入 /mnt/d/DocProjects/LearnLLMFromDraft/results/Season0/02/
#
# 用法：
#   bash collect_data.sh [--project-dir ~/llm-training-lab]

set -uo pipefail

# 注意：bash 变量名必须用 ASCII（非 ASCII 变量名会被当作命令名执行）
PROJ_DIR="$HOME/llm-training-lab"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-dir) PROJ_DIR="$2"; shift 2 ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
done

OUT_DIR="/mnt/d/DocProjects/LearnLLMFromDraft/results/Season0/02"
SCRIPT_DIR="/mnt/d/DocProjects/LearnLLMFromDraft/DevResources/Season0/02/scripts"
CODE_DIR="/mnt/d/DocProjects/LearnLLMFromDraft/DevResources/Season0/02/project"
PY="$PROJ_DIR/.venv/bin/python"

mkdir -p "$OUT_DIR"

echo "===== 1/6 torch 验证 ====="
"$PY" - <<'PY' | tee "$OUT_DIR/torch-verify.log"
import torch
print("torch:", torch.__version__)
print("cuda build:", torch.version.cuda)
print("cudnn:", torch.backends.cudnn.version())
print("nccl:", torch.cuda.nccl.version() if torch.cuda.is_available() else "N/A")
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device count:", torch.cuda.device_count())
    print("device:", torch.cuda.get_device_name(0))
    p = torch.cuda.get_device_properties(0)
    print("vram GB:", round(p.total_memory / 1e9, 2))
    print("capability:", f"{p.major}.{p.minor}")
    print("sm count:", p.multi_processor_count)
    free_b, total_b = torch.cuda.mem_get_info(0)
    print(f"实际可用显存 GB: {round(free_b/1e9,2)} / {round(total_b/1e9,2)}")
PY

echo "===== 2/6 依赖锁 ====="
export PATH="$HOME/.local/bin:$PATH"
uv pip freeze --python "$PY" > "$OUT_DIR/requirements.lock.txt"
wc -l "$OUT_DIR/requirements.lock.txt"

echo "===== 3/6 环境卡片 ====="
bash "$SCRIPT_DIR/check_env.sh" --project-dir "$PROJ_DIR" | tee "$OUT_DIR/env_card.txt" >/dev/null
echo "已写入 env_card.txt"

echo "===== 4/6 冒烟训练（common 骨架统一接口） ====="
cd "$CODE_DIR"
"$PY" -c "import yaml" 2>/dev/null || uv pip install --python "$PY" pyyaml -q
PYTHONPATH="$CODE_DIR" "$PY" -m exp_smoke.run --config exp_smoke/config.yaml \
  --output "$OUT_DIR/smoke_train.json" | tee "$OUT_DIR/smoke_train.log"

echo "===== 5/6 复现性对照 ====="
"$PY" "$SCRIPT_DIR/reproducibility_test.py" --steps 50 --output "$OUT_DIR/reproducibility.json" \
  | tee "$OUT_DIR/reproducibility.log"

echo "===== 6/6 CPU 降级路线验证（L0 读者可用性） ====="
PYTHONPATH="$CODE_DIR" "$PY" -m exp_smoke.run --config exp_smoke/config.yaml \
  --override hardware.device=cpu --override experiment.steps=30 \
  --output "$OUT_DIR/smoke_train_cpu.json" | tail -5

echo ""
echo "全部采集完成。产出文件："
ls -la "$OUT_DIR"
echo "COLLECT_DONE"
