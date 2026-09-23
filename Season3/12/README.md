# DevResources/Season3/12 —— DDP Lab（分布式正确性）

对应文章：`Articles/Season3/12_同一份训练任务，怎样改成双卡 DDP 而不改变结果？.md`

## 本篇解决什么问题

把单卡脚本套上 `torchrun` 和 DDP，程序能启动并不代表训练等价。数据、梯度、batch、随机数、checkpoint 要怎样处理，双卡结果才能与单卡基线对齐？

核心判断：DDP 的本质是每个 rank 保留完整模型、读取不同数据，再把梯度同步成一致结果。正确性首先取决于全局 batch 和数据分片语义，速度由 14 篇测量。

## 包内容

```text
12/
├── README.md                  # 本文件
└── project/
    └── exp_ddp/
        ├── ddp_common.py        # 进程组（gloo/nccl）、FixedSampleDataset、param_checksum、聚合工具
        ├── ddp_train.py         # torchrun 入口：single/ddp/gradsync/resume 四模式 + 5 个故障注入
        ├── sampler_audit.py     # DistributedSampler 分片审计（纯 CPU）：覆盖率/重复/set_epoch
        ├── parity_check.py      # 单卡 vs 双卡三级对齐门禁（一条命令）
        ├── xnode_check.py       # 单卡 vs 两机 NCCL：节点、设备、语料、轨迹与参数核验
        ├── fail_modes_ddp.py    # 5 个故障的文档与预期偏差表
        ├── analyze_faults.py    # 故障偏差 vs 基线的汇总分析（采集期工具）
        └── analyze_noise_floor.py # 噪声底测量：等价配置两次跑的参数差
```

## 前置：本篇包依赖前序篇的模块

`exp_ddp` 复用 06 篇的 `build_llama` 与 `contract_loss`、04 篇的 `CharTokenizer`，并依赖 02 篇骨架。本篇包只含**新增**的 `exp_ddp/`，运行前需累积：

| 需要的模块 | 来自 | 用途 |
|---|---|---|
| `common/` | 02 篇包 | seed / config / 留痕 / benchmark / 指标口径 |
| `exp_scale/` | 04 篇包 | `CharTokenizer`、四大名著语料 |
| `exp_hf/` | 06 篇包 | `build_llama` 模型构建、`contract_loss` |
| `exp_ddp/` | 本篇包 | DDP 三级对齐门禁 + sampler 审计 + 故障注入 |

```bash
cp -r DevResources/Season3/12/project/exp_ddp ~/llm-training-lab/
cd ~/llm-training-lab
```

## 传输选型（重要）

**主实验是两机 NCCL DDP**：两台 WSL 主机各有一张 GPU，每 rank 独占一张卡，固定全局 batch 16 和 30 步，用 `xnode_check` 对单卡与两个 DDP rank 落盘数据逐项验收。已归档 `nccl_xnode_parity_30.json` 的 `overall_pass` 为 true。两台机器的端口、地址与接口由运行时环境指定，不写入跟踪文件。

只有一张 GPU 时可用 gloo 进行单机预检：

- NCCL 拒绝单卡 2 rank（`ncclInvalidUsage`），要求每 rank 独占一张 GPU。
- gloo 能驱动 CUDA 模型做 DDP：前向反向在 GPU 上真实跑，梯度 all-reduce 经 CPU 中转。
- gloo 的单机验证不能代替独立节点、设备和 NCCL 后端的两机验收。

单卡能复现算法和 sampler 审计，不能复现双卡训练的环境与性能结论。

## 快速使用

```bash
cd ~/llm-training-lab

# 旧单卡 gloo 预检（一条命令跑单进程 + 2-rank DDP 并比对）
./.venv/bin/python -m exp_ddp.parity_check --steps 30 --global-batch 16 \
    --out results/Season3/12/parity.json

# 梯度同步观测：证明 all-reduce 发生在 backward 中
./.venv/bin/torchrun --nproc_per_node=2 --master_port=29542 -m exp_ddp.ddp_train \
    --mode gradsync --global-batch 16 --out results/Season3/12/gradsync.json

# sampler 审计（纯 CPU，不需要多卡，秒级）
./.venv/bin/python -m exp_ddp.sampler_audit --n-samples 480 --world 2 --batch 8 --epochs 3 \
    --out results/Season3/12/sampler_audit.json

# 故障注入（5 个）
./.venv/bin/torchrun --nproc_per_node=2 --master_port=29545 -m exp_ddp.ddp_train \
    --mode ddp --fault no_sampler --steps 30 --global-batch 16 --out ...

# checkpoint + resume 对齐
./.venv/bin/torchrun --nproc_per_node=2 -m exp_ddp.ddp_train --mode ddp --steps 15 \
    --schedule-total 30 --save-ckpt --ckpt-dir runs/collect_ckpts --out ...
./.venv/bin/torchrun --nproc_per_node=2 -m exp_ddp.ddp_train --mode resume --steps 15 \
    --schedule-total 30 --ckpt-dir runs/collect_ckpts --out ...
```

`--mode` 四选一：`single` / `ddp` / `gradsync` / `resume`。`--fault` 可选 `no_sampler` `global_batch_misconfig` `sum_reduction` `no_set_epoch` `all_rank_save`。`--backend` 可选 `gloo`（默认，单卡可跑）/ `nccl`（需每 rank 一卡）。

## 结果文件

`results/Season3/12/`。两机与历史单机记录分别保留：

| 文件 | 内容 |
|---|---|
| `nccl_xnode_single_30.json` / `nccl_xnode_rank0_30.json` / `nccl_xnode_rank1_30.json` | 单卡与两机 NCCL 两个 rank 的原始 30 步记录 |
| `nccl_xnode_parity_30.json` | 节点/设备/语料/初始权重/两 rank 轨迹门禁，通过；单卡对 DDP 逐张量最大相对误差 4.3223e-5（容差 1e-4） |
| `parity.json` | 历史 gloo 预检：init_checksum 相同、loss 轨迹差约 1e-6；旧脚本的 4.33e-5 只对应最大绝对差所在张量，不是全模型逐张量最大相对误差 |
| `gradsync.json` / `_rank1.json` | DDP 梯度 vs 单进程全 batch 梯度差 1.64e-7，证明同步在 backward 中 |
| `sampler_audit.json` | 覆盖率 1.0、重复 0、同步零重叠；set_epoch 顺序逐 epoch 变化 |
| `fault_deviation.json` | 各故障 loss 轨迹偏差 vs 噪声底 |
| `resume_parity.json` | 分段跑 vs 连续跑参数差 2.79e-6，PASS |
| `xnode_nccl_attempt.json` | 早期 NAT 环境的失败记录，现已由上述两机训练证据取代 |
| `ddp_none` / `single_none` / `fault_*` / `continuous_30` 等 | 各组训练结果 |

## 等价判据：容差比对，不是 checksum 相等

配置完全相同的两次历史 gloo DDP 跑（`ddp_none` 与 `continuous_30`），final loss 六位小数一致，但 `final_checksum` 不同。旧校验器记录的 2.99e-6 只针对最大绝对差所在张量，不是完整噪声分布；参数低位差异的来源尚未单独定位。

checksum 相同提供了逐位相同的强信号；不同不等于训练失败。新两机门禁逐张量计算最大相对误差并检查 1e-4 容差，历史噪声值只作粗略量级参考。静默故障的 loss 偏差 0.09~0.19，远大于这些旧对照中的数值误差。

## 五个故障（四个静默）

| 故障 | loss 轨迹最大偏差 | 显性/静默 |
|---|---|---|
| no_sampler（样本重复 2 次） | 0.0999 | 静默（loss 反而更低） |
| global_batch_misconfig（全局 32 当 16） | 0.1900 | 静默（末步只差 0.002） |
| sum_reduction（未除 world_size） | 17933（首步即炸） | 显性 |
| no_set_epoch（漏调） | 0.088（第 10 步起） | 静默（偏差精确出现在 epoch 边界） |
| all_rank_save（全 rank 写） | — | 见下 |

`all_rank_save` 的诚实修正：受控测试连跑 6 次文件全部可加载、无损坏。因为 DDP 让各 rank 状态相同，写相同字节到相同偏移，本地文件系统上竞态恰好无害。真实代价是 world_size 倍冗余 I/O + 共享文件系统（NFS）上的潜在竞态 + 无 barrier。**"测试时没出问题"正是它危险的原因。**

## 已知边界

- 新两机 NCCL 记录来自异构 3070/4090、单次运行，不代表同型号多卡；每 rank 训练墙钟也不是全局同步的完整作业计时，性能比较以 14 篇正式 benchmark 为准。
- 旧 gloo 的 wall_s（DDP 7.75s 对单进程 6.51s）只说明共享单卡的行为。早期 NAT 失败记录属于排查历史，不是当前状态。
- 全程 fp32 不用 AMP：GradScaler 跳步会引入额外状态，让逐位对齐判据变模糊。
