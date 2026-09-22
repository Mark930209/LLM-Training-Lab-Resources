# DevResources/Season1/08 —— 显存优化六项交换

对应文章：`Articles/Season1/08_目标模型放不进显存时，哪些优化真的值得用？.md`

## 本篇解决什么问题

目标配置（04 篇的 100M 档，89.59M 参数）在受限环境下 OOM。六项显存优化手段，哪些真的值得用，代价是什么？

核心判断：显存优化是资源交换，交换只在被换的那种资源是约束时才划算。本篇用红线机制在受控条件下复现真 OOM，再逐项测每项交换省多少显存、掉多少吞吐。

## 包内容

```text
08/
├── README.md                  # 本文件
└── project/
    └── exp_opt/
        └── opt_bench.py       # 三模式：bench（六项交换逐项测）/ oom（红线 OOM 证据）/ parity（同 token 预算对照）
```

六项交换：AMP 混合精度、gradient accumulation、activation checkpointing、fp16 优化器状态（lowstate）、CPU offload、set_to_none。

## 前置：本篇包依赖前序篇的模块

`exp_opt` 复用 06 篇的 `build_llama` 与 `contract_loss`，并依赖 02 篇骨架。本篇包只含**新增**的 `exp_opt/`，运行前需累积：

| 需要的模块 | 来自 | 用途 |
|---|---|---|
| `common/` | 02 篇包 | seed / config / 留痕 / benchmark / 指标口径 |
| `exp_hf/` | 06 篇包 | `build_llama` 模型构建、`contract_loss` |
| `exp_opt/` | 本篇包 | 六项显存交换 + OOM 红线 + parity |

```bash
cp -r DevResources/Season1/08/project/exp_opt ~/llm-training-lab/
cd ~/llm-training-lab
```

## 快速使用

```bash
cd ~/llm-training-lab

# baseline：目标配置在红线下 OOM
./.venv/bin/python -m exp_opt.opt_bench --mode bench --opts baseline --hidden 768 \
    --layers 12 --heads 12 --seq 256 --batch 8 --steps 50 --mem-fraction 0.35 \
    --output results/Season1/08/bench_baseline_oom.json

# 单项与组合交换
./.venv/bin/python -m exp_opt.opt_bench --mode bench --opts amp --mem-fraction 0.35 \
    --hidden 768 --layers 12 --heads 12 --seq 256 --batch 8 --steps 50 \
    --output results/Season1/08/bench_amp.json

./.venv/bin/python -m exp_opt.opt_bench --mode bench --opts accum,amp,setnone --mem-fraction 0.35 ...
./.venv/bin/python -m exp_opt.opt_bench --mode bench --opts amp,ckpt --mem-fraction 0.35 ...
./.venv/bin/python -m exp_opt.opt_bench --mode bench --opts amp,ckpt,lowstate --mem-fraction 0.35 ...

# OOM 红线证据
./.venv/bin/python -m exp_opt.opt_bench --mode oom --mem-fraction 0.35 ...

# parity：同 token 预算下对照，保证可比
./.venv/bin/python -m exp_opt.opt_bench --mode parity ...
```

`--mode` 三选一：`bench` / `oom` / `parity`。`--opts` 用逗号组合：`amp` `accum` `ckpt` `lowstate` `offload` `setnone`。`--mem-fraction 0.35` 是红线（约 2.80 GiB），模拟一张约 3 GB 的卡。

## 结果文件

`results/Season1/08/`（12 个），bench 矩阵核心数字：

| 配置 | peak MB | tok/s | final_loss | 结论 |
|---|---|---|---|---|
| `bench_baseline_oom` | 2610.7 | 1329 | — | 红线下 step 1 即 OOM |
| `bench_amp` | 2768.6 | 2419 | — | AMP 单独仍 OOM（峰值反升，见下） |
| `bench_accum_amp_setnone` | 1870.2 | 8901 | 8.8085 | 通过 |
| `bench_amp_ckpt` | 1801.3 | 11723 | 8.8015 | 通过，吞吐最高 |
| `bench_amp_ckpt_lowstate` | 1473.1 | 13968 | **NaN** | 省最多但 fp16 优化器状态下溢 |
| `bench_..._offload_accum_setnone` | 1741.6 | 3088 | 8.8085 | offload 慢 4.4 倍且 peak 没更低 |

另含 `oom_original.json`、`oom_redline.json`、`parity.json`、`parity_offload_*.json`。

## 两个真实失败案例

1. **fp16 优化器状态 NaN**：`lowstate` 把 AdamW 的 m/v 存成 fp16，省显存最多（1473.1 MB），但 final_loss = NaN。根因是 fp16 优化器状态在长训练里下溢。省显存换来了不可用。
2. **offload 不值得**：全家桶含 CPU offload 跑出 3088 tok/s，比不 offload 慢 4.4 倍，peak 反而没更低（1741.6 对 1801.3）。PCIe 传输的代价吃掉了容量收益。这是"offload 不值得"的实测证据，不是推断。

## 已知边界

- **AMP 单独 peak 反升**（2768.6 对 baseline 2610.7）：AMP 省的是激活，但 GradScaler 与 fp32 主权重的额外开销在小模型上盖过了收益。AMP 主要省时间不省显存，这个反直觉结果在 04 篇也出现过。
- 红线 0.35 是为在 8 GB 卡上复现 100M 档 OOM 而设的模拟约束，不是真实硬件限制。裸金属上目标配置是否 OOM 取决于实际显存。
- 本篇的交换在"显存是约束"时成立。10 篇会回测：显存宽裕时，ckpt 与 accum 是纯吞吐亏损（0.84x / 0.38x）。两篇结论不矛盾，适用条件不同。
