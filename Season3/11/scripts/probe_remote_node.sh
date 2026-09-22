#!/usr/bin/env bash
# 远端节点能力探测：GPU、PyTorch、NCCL、网络。
# 用法： ssh <host> 'bash -s' < probe_remote_node.sh
set -u
export PATH=/usr/lib/wsl/lib:$PATH

echo "=== host ==="
hostname
uname -r

echo "=== gpu ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 | head -4
echo "cuda devices via nvidia-smi -L:"
nvidia-smi -L 2>&1 | head -4

echo "=== cpu / mem ==="
echo "cores: $(nproc)"
free -g | head -2

echo "=== python ==="
for py in python3 ~/miniforge3/bin/python; do
  if command -v "$py" >/dev/null 2>&1 || [ -x "$py" ]; then
    echo "--- $py ---"
    "$py" - <<'PYEOF' 2>&1 | head -12
import sys
print("python", sys.version.split()[0])
try:
    import torch
    print("torch", torch.__version__)
    print("cuda_build", torch.version.cuda)
    print("cuda_available", torch.cuda.is_available())
    print("device_count", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("device0", torch.cuda.get_device_name(0))
        print("capability", torch.cuda.get_device_capability(0))
    import torch.distributed as d
    print("nccl_available", d.is_nccl_available())
    print("gloo_available", d.is_gloo_available())
except ImportError as e:
    print("NO_TORCH:", e)
PYEOF
  fi
done

echo "=== uv / venv ==="
command -v uv >/dev/null 2>&1 && echo "uv: $(uv --version)" || echo "uv: not found"
ls -d ~/llm-training-lab 2>/dev/null || echo "no ~/llm-training-lab"

echo "=== network ==="
echo "ip addrs:"
ip -4 -o addr show 2>/dev/null | awk '{print $2, $4}' | head -8
echo "default route:"
ip route 2>/dev/null | head -3

echo "=== torchrun ==="
command -v torchrun >/dev/null 2>&1 && echo "torchrun: $(command -v torchrun)" || echo "torchrun: not found"

echo "=== done ==="
