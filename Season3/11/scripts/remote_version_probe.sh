#!/usr/bin/env bash
# 远端环境版本探测：torch / CUDA / 运行时 NCCL（含 LD_PRELOAD 对齐验证）。
set -u
export PATH=/usr/lib/wsl/lib:$PATH

echo "=== baseline (no preload) ==="
~/miniforge3/bin/python - <<'PYEOF'
import torch, ctypes
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
print("compile_time_nccl", torch.cuda.nccl.version())
lib = ctypes.CDLL("libnccl.so.2")
v = ctypes.c_int()
lib.ncclGetVersion(ctypes.byref(v))
print("runtime_nccl", v.value)
PYEOF

echo "=== with LD_PRELOAD nccl2307 ==="
LD_PRELOAD=$HOME/nccl2307/lib/libnccl.so.2 ~/miniforge3/bin/python - <<'PYEOF'
import torch, ctypes
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
t = torch.ones(4, device="cuda").sum().item()
print("cuda_tensor_ok", t == 4.0)
lib = ctypes.CDLL("libnccl.so.2")
v = ctypes.c_int()
lib.ncclGetVersion(ctypes.byref(v))
print("runtime_nccl", v.value)
PYEOF

echo "=== cross-nccl dir ==="
ls ~/cross-nccl
