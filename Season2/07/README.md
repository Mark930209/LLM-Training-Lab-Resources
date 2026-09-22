# DevResources/Season2/07 —— 显存去哪了

对应文章：`Articles/Season2/07_训练显存为什么总比公式算得多：追踪一个 step 的显存去向.md`

## 本篇解决什么问题

公式算的是参数 + 梯度 + 优化器状态，实测峰值总比它多。多出来的是什么？

本篇建立**显存四口径**（allocated / reserved / nvidia-smi / 理论账），用阶段探针追踪一个 step 内显存的去向，并给出峰值估算器。

核心判断：显存不是一个数，是四个口径；对不上账往往不是算错了，是在比不同的东西。

## 包内容

```text
07/
├── README.md                  # 本文件
└── project/
    └── exp_mem/
        ├── mem_probe.py       # 四模式探针：stage（时间线）/ sweep（扫描）/ frag（碎片）/ oom（红线）
        └── mem_estimator.py   # 峰值估算器：peak = W·G + a·ACT + c，系数由实测拟合
```

## 前置：本篇包依赖前序篇的模块

`exp_mem` 复用 06 篇的 `build_llama` 与 `contract_loss`、04 篇的数据，并依赖 02 篇骨架。本篇包只含**新增**的 `exp_mem/`，运行前需累积：

| 需要的模块 | 来自 | 用途 |
|---|---|---|
| `common/` | 02 篇包 | seed / config / 留痕 / benchmark / 指标口径 |
| `exp_scale/` | 04 篇包 | 数据加载、语料 |
| `exp_hf/` | 06 篇包 | `build_llama` 模型构建、`contract_loss` |
| `exp_mem/` | 本篇包 | 显存四口径探针 + 峰值估算器 |

```bash
cp -r DevResources/Season2/07/project/exp_mem ~/llm-training-lab/
cd ~/llm-training-lab
```

## 快速使用

```bash
cd ~/llm-training-lab

# 阶段时间线：一个 step 内显存怎么涨怎么落（100M 档 + AMP）
./.venv/bin/python -m exp_mem.mem_probe --mode stage --hidden 768 --layers 12 \
    --heads 12 --seq 256 --batch 8 --amp --output results/Season2/07/stage_100m_amp.json

# 扫描：batch / seq / hidden / layers / 精度 / 优化器逐项变
./.venv/bin/python -m exp_mem.mem_probe --mode sweep --hidden 384 --layers 6 \
    --seq 256 --batch 16 --output results/Season2/07/sweep_batch_16.json

# 碎片探测
./.venv/bin/python -m exp_mem.mem_probe --mode frag --output results/Season2/07/frag_probe.json

# OOM 红线：WSL2 下复现真 OOM（见"已知边界"）
./.venv/bin/python -m exp_mem.mem_probe --mode oom --mem-fraction 0.35 --hidden 768 \
    --layers 12 --seq 256 --batch 8 --output results/Season2/07/oom_redline.json

# 峰值估算器：拟合系数 + 盲测误差
./.venv/bin/python -m exp_mem.mem_estimator --output results/Season2/07/estimator_report.json
```

`--mode` 四选一：`stage` / `sweep` / `frag` / `oom`。`--opt` 可选 `adamw` / `sgd`。`--mem-fraction` 画显存红线（WSL2 复现 OOM 用）。

## 结果文件

`results/Season2/07/`（20 个）：

| 文件 | 内容 |
|---|---|
| `stage_100m_amp.json` | 时间线：加载 343.6/380 → step_0 尖峰 1410.8/1890/peak 1754.3 → 稳态 1410.2/1818.1 |
| `sweep_batch_{2,4,8,16}.json` | peak 267.2 / 273.9 / 467.9 / 844.8 MB |
| `sweep_seq_{64,128,256,512}.json` | peak 273.8 / 467.9 / 844.8 / 1580.7 MB |
| `sweep_fp32.json` / `sweep_sgd.json` | fp32 1103.1 对 AMP 844.8；SGD 844.8 对 AdamW 945.3 |
| `sweep_hidden576.json` / `sweep_hidden768.json` / `sweep_layers12.json` | 结构维度扫描 |
| `sweep_fit.json` / `sweep_blind.json` | 估算器拟合集与盲测集 |
| `estimator_report.json` | 系数 a=2.8257、c=142.77 MB；盲测最大误差 20.7% |
| `oom_redline.json` / `oom_capacity.json` | 红线下的真 OOM 证据 |
| `frag_probe.json` | 256MB 大块分配 100% 成功，largest_inactive 256MB，无碎片恶化 |

`stage_100m_amp.json` 里 `optimizer_init` 与 `model_loaded` 完全相同，这是**优化器状态延迟分配**的直接证据：AdamW 的 m/v 在第一次 `step()` 时才分配，不在构造时。

## 估算器的诚实边界

两系数模型 `peak = W·G + a·ACT + c`：

- 拟合集最大误差 4.9%，但 **b4 档误差 34.6%**（估 368.6 对实测 273.9）
- 盲测最大误差 20.7%

这个误差没有藏起来。估算器用于量级判断（"会不会 OOM"），不用于精确预算。小 batch 下固定开销 c 的占比高，是误差的主要来源。

## 已知边界：WSL2 的显存溢出

**这是本篇最重要的环境发现**。WSL2 的 DXG 内核驱动（Windows 的 GPU 内核驱动，WSL2 里的 CUDA 请求都经它转发给真实显卡）会把超额显存请求**悄悄溢出到主机内存**：

- seq 1024 / batch 8 的前向分配 17,854 MB，在 8 GB 卡上**不 OOM，只是变慢**
- 裸金属（不经虚拟化、直接装 Linux 的物理机）上这个配置必然崩溃

所以在 WSL2 里无法直接复现显存不足的故障。修法是用 `torch.cuda.set_per_process_memory_fraction` 给进程画一条硬上限（红线），超出即触发真实 `CUDA out of memory`。本篇统一用 `--mem-fraction 0.35`，对应 8 GB 卡上约 2.80 GiB 可用上限，等价于模拟一张约 3 GB 的显卡。

裸金属用户不需要这个参数，显存不足时系统本身就会报错。

这个机制在 09 篇再次咬人：seq 2048 的 eager attention 跑出 272 倍"加速比"，核查发现 reserved 占整卡 81.3%，是溢出导致的假数据。10 篇的工具已内置防护（reserved 超 60% 自动标 `spillover_suspect`）。
