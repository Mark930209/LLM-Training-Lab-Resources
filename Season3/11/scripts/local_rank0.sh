#!/usr/bin/env bash
# rank0（本机 RTX 3070）：mirrored 模式 LAN 直连跨机 NCCL 验证。
set -u
cd ~/cross-nccl
export PATH=/usr/lib/wsl/lib:$PATH
: "${RANK0_LAN_IP:?Set RANK0_LAN_IP to the rank0 mirrored LAN address}"

NCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0 NCCL_IB_DISABLE=1 \
NCCL_DEBUG=WARN \
~/llm-training-lab/.venv/bin/python -m torch.distributed.run \
  --nnodes=2 --node_rank=0 --nproc_per_node=1 \
  --master_addr="$RANK0_LAN_IP" --master_port=29501 nccl_cross.py \
  2>&1 | tee rank0_mirrored.log
