#!/usr/bin/env bash
# rank1（远端 RTX 4090）：mirrored 模式 LAN 直连跨机 NCCL 验证。
# LD_PRELOAD 把运行时 NCCL 对齐到 2.30.7（编译期是 2.28.9，不对齐会 bootstrap 失败）。
set -u
cd ~/cross-nccl
export PATH=/usr/lib/wsl/lib:$PATH
: "${RANK0_LAN_IP:?Set RANK0_LAN_IP to the rank0 mirrored LAN address}"

LD_PRELOAD=$HOME/nccl2307/lib/libnccl.so.2 \
NCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0 NCCL_IB_DISABLE=1 \
NCCL_DEBUG=WARN \
~/miniforge3/bin/python -m torch.distributed.run \
  --nnodes=2 --node_rank=1 --nproc_per_node=1 \
  --master_addr="$RANK0_LAN_IP" --master_port=29501 nccl_cross.py \
  2>&1 | tee rank1_mirrored.log
