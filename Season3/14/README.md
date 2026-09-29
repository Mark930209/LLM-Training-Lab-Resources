# DevResources/Season3/14 —— Distributed Benchmark Harness（多卡扩展效率测量）

对应文章：`Articles/Season3/14_两张 GPU 为什么没有快一倍：怎样正确测量多卡扩展效率.md`

## 本篇解决什么问题

增加一张 GPU 之后，一样多的 token 到底多久训完？为什么真实加卡反而更慢？

核心判断：speedup 不是一个数，而是一族数。取哪段样本（含不含 warmup）、用哪个统计量（均值/中位数/P95）、batch 定义是否对齐，三个口径选择各自都能改写结论。口径没锁死之前，任何"双卡快了多少"的说法都不成立。

## 包内容

```text
14/
├── README.md                  # 本文件
└── project/
    └── exp_bench/
        ├── bench_common.py    # 口径定义（写死）、合成数据集、分段计时器、通信量估算、报告落盘
        ├── bench_train.py     # single/ddp 两种 mode、straggler 注入、逐步 event 与墙钟、各 rank 单独落盘
        └── caliber_audit.py   # 同一批日志五种口径重算 speedup
```

## 前置：本篇包依赖前序篇的模块

`exp_bench` 复用 12 篇 `exp_ddp` 的模型构建，跨机主档复用 11 篇的双机 NCCL 环境。运行前需累积：

| 需要的模块 | 来自 | 用途 |
|---|---|---|
| `common/` | 02 篇包 | seed / config / 留痕 / benchmark / 指标口径 |
| `exp_scale/` | 04 篇包 | `CharTokenizer`（模型构建链依赖） |
| `exp_hf/` | 06 篇包 | `build_llama` 模型构建 |
| `exp_ddp/` | 12 篇包 | `build_model`（11.41M / 44.06M 两档小 Llama） |
| `exp_bench/` | 本篇包 | 分段计时 + 口径审计 |

```bash
cp -r DevResources/Season3/14/project/exp_bench ~/llm-training-lab/
cd ~/llm-training-lab
```

## 传输选型（重要）

本篇有三档实验，传输路径各不相同，**结论不能跨档挪用**：

| 档位 | 路径 | 能测什么 | 不能测什么 |
|---|---|---|---|
| 单卡旧档 | 两个 gloo rank 共享同一张 3070 | 口径审计：同一批日志五种统计口径的 speedup 漂移 | 双卡收益——混有单卡争用与 CPU 中转 |
| 两机主档 | 每 rank 独占一张卡，NCCL 跨机 | 异构双机对单卡的真实墙钟比（本篇主结论） | 同型号双卡或更快互联的表现 |
| 扩展档 | 多次重复 + 通信 profiler + straggler 注入 | 尚未采集，不生成误差条 | — |

- 共享单卡的 gloo 记录（`single_g16` / `ddp_g16` / `ddp_g16_big` 等旧命名 JSON）保留作口径反例，**不充当 GPU 扩展效率**。
- 跨机主档需要 11 篇的双机网络环境（mirrored + NCCL 版本对齐），本篇不重复搭。
- 没有 GPU 时可用 gloo + CPU 跑通脚本，但 CPU 计时路径改用 `perf_counter`，不能与 CUDA event 数字直接比较。

## 快速使用

```bash
cd ~/llm-training-lab

# 单卡旧档：单进程基线 + 共享单卡 DDP（复现口径陷阱）
./.venv/bin/python -m exp_bench.bench_train --mode single \
    --steps 30 --global-batch 16 --out results/Season3/14/single_g16.json
./.venv/bin/torchrun --nproc_per_node=2 --master_port=29571 \
    -m exp_bench.bench_train --mode ddp --steps 30 --global-batch 16 \
    --out results/Season3/14/ddp_g16.json

# 口径审计：同一批日志，五种口径重算 speedup
./.venv/bin/python -m exp_bench.caliber_audit \
    --single results/Season3/14/single_g16.json \
    --dual results/Season3/14/ddp_g16.json \
    --out results/Season3/14/caliber_audit.json

# straggler 注入：给 rank1 每步加 20 ms 固定延迟，barrier 让延迟传导
./.venv/bin/torchrun --nproc_per_node=2 --master_port=29572 \
    -m exp_bench.bench_train --mode ddp --steps 30 --global-batch 16 \
    --straggler-ms 20 --out results/Season3/14/ddp_g16_straggler20.json

# 两机主档（复用 11 篇环境，两侧各执行一次）
./.venv/bin/torchrun --nnodes=2 --nproc_per_node=1 --node_rank=$RANK \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    -m exp_bench.bench_train --mode ddp --backend nccl \
    --steps 30 --global-batch 16 --out results/Season3/14/nccl_xnode_ddp_g16.json
```

`--mode` 二选一：`single` / `ddp`。`--backend` 可选 `gloo`（默认，单机可跑）/ `nccl`（需每 rank 一卡）。`--straggler-ms` 与 `--straggler-rank` 控制延迟注入。

## 结果文件

`results/Season3/14/`。两机主档与旧单卡档分别保留：

| 文件 | 内容 |
|---|---|
| `nccl_xnode_3070_g16.json` / `nccl_xnode_4090_g16.json` | 单卡基线：30 步墙钟 1.599 s / 0.625 s |
| `nccl_xnode_3070_g8.json` / `nccl_xnode_4090_g8.json` | 弱扩展单卡参照：每卡 batch 8 |
| `nccl_xnode_ddp_rank0_g16.json` / `nccl_xnode_ddp_rank1_g16.json` | 两机 NCCL DDP 各 rank 墙钟 12.957 s / 14.152 s；取较慢 rank 作保守完成时间代理 |
| `single_g16.json` / `ddp_g16.json` | 旧单卡档：单进程 58.2 ms/步 vs 共享单卡 DDP 464.2 ms/步 |
| `single_g16_big.json` / `ddp_g16_big.json` | 44.06M 档：倍数从约 8.0 增至 10.9 |
| `single_g32.json` / `ddp_g32.json` | 全局 batch 32，用于口径 A 的混比反例 |
| `ddp_g16_straggler20.json` | rank1 注入 20 ms 延迟后的记录 |
| `caliber_audit.json` / `caliber_audit_A.json` | 五种口径重算结果：B~E 落在 0.108~0.126，口径 A 的 0.214 是构造指标 |

## 已知边界

- **两机主档是单次运行**，没有重复采样误差条，暂不能确定尾部波动和通信占比。下一步需按相同口径补至少 3 次运行及离散度。
- **两端计时起点没有全局同步**：`measured_wall_s` 各自从首个正式步骤起算，不是整个作业的端到端墙钟。
- **backward 段混有计算与通信**：CUDA event 没有拆开两者，也没有同步后的 bucket profiler 证据，因此不能把单卡与双卡的 bwd 差值直接当作 NCCL 传输时间。
- 旧脚本的统计缺陷如实保留在结果里：`total_ms_median` 取 `n//2`（偶数样本的上中位，非严格中位数）、`total_ms_p95` 用 `int(n*0.95)-1` 作零基下标（30 步时选中第 28 个值，按最近秩定义应取第 29 个）。文章中的 P95 已按最近秩重新计算，不沿用 JSON 旧字段。
- `bucket_cap_mb`、`static_graph`、`gradient_as_bucket_view` 会改变调度或内存使用，没有对照实验时不能断言它们提高了重叠或吞吐。
- 合成 token 的 data 段不足 1 ms，不代表真实语料加载的性能。
- 4/8 GPU 尚无实测曲线，不生成外推结论。
