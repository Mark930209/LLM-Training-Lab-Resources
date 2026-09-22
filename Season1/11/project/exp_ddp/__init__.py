"""exp_ddp —— 11 篇 DDP Lab。

ddp_common      进程组、固定样本序列数据集、模型构建、校验和工具
ddp_train       torchrun 入口：single / ddp / gradsync / resume 四模式 + 故障注入
sampler_audit   DistributedSampler 分片审计：覆盖率、重复数、set_epoch 顺序
parity_check    单卡与双卡三级对齐：每步梯度、loss 轨迹、最终参数
fail_modes_ddp  故障文档与辅助函数（sum reduction、独立 shuffle 等）

传输选型（实测）：NCCL 拒绝单卡 2 rank（ncclInvalidUsage），gloo 可以，
且 gloo 能驱动 CUDA 模型做 DDP（梯度 all-reduce 逐位正确）。本篇正确性
实验全部走"单卡 GPU 训练 + gloo 通信"，跨机 NCCL 另测（见 ddp_train
的 --backend nccl 与 rendezvous 参数）。
"""
