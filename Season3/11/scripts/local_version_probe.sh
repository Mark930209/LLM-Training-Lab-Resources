#!/usr/bin/env bash
# 本机 WSL 环境探测：torch / CUDA / 运行时 NCCL + Hyper-V 防火墙状态提示。
set -u
echo "=== local venv ==="
~/llm-training-lab/.venv/bin/python - <<'PYEOF'
import torch, ctypes
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
lib = ctypes.CDLL("libnccl.so.2")
v = ctypes.c_int()
lib.ncclGetVersion(ctypes.byref(v))
print("runtime_nccl", v.value)
PYEOF
echo "=== cross-nccl dir ==="
ls ~/cross-nccl 2>/dev/null || echo "no ~/cross-nccl"
