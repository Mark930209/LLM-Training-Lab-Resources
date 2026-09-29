# DevResources/Season3/13 —— Collective Lab（手写集合通信与通信量核算）

对应文章：`Articles/Season3/13_多卡训练时梯度是怎么合并的：从手写 AllReduce 到 NCCL.md`

## 本篇解决什么问题

DDP 同步后的梯度是怎样在 rank 之间流动的？同步一次要移动多少字节？

核心判断：Ring AllReduce 可以拆成 ReduceScatter + AllGather 两阶段，每 rank 发送量为张量大小的 $2(w-1)/w$ 倍；这个数可以用逐轮 trace 逐字节核对，而不是背公式。naive 的集中归约结果同样正确，但流量全压在 rank0 上，负载不均。

## 包内容

```text
13/
├── README.md                  # 本文件
└── project/
    └── exp_collective/
        ├── collective_common.py   # 通信量口径（ring_bus_bytes 等）、trace 数据结构、计时与带宽公式（单处定义）
        ├── hand_written.py        # naive / Ring / ReduceScatter / AllGather 四个手写实现，全部带逐轮 trace
        └── collective_lab.py      # 四种 mode：correctness / trace / backend / ddp_grad
```

## 前置：本篇包依赖前序篇的模块

`exp_collective` 的 `ddp_grad` 模式复用 12 篇 `exp_ddp` 的模型与数据构建，其余三个模式零依赖。运行前需累积：

| 需要的模块 | 来自 | 用途 |
|---|---|---|
| `common/` | 02 篇包 | seed / config / 留痕 / benchmark / 指标口径 |
| `exp_scale/` | 04 篇包 | `CharTokenizer`、四大名著语料 |
| `exp_hf/` | 06 篇包 | `build_llama` 模型构建 |
| `exp_ddp/` | 12 篇包 | `build_model`、`FixedSampleDataset`、`load_corpus_ids`（仅 ddp_grad 模式） |
| `exp_collective/` | 本篇包 | 手写集合通信 + 通信量核算 |

```bash
cp -r DevResources/Season3/13/project/exp_collective ~/llm-training-lab/
cd ~/llm-training-lab
```

## 传输选型（重要）

**主实验是单机 2-rank gloo，不需要 GPU**：手写实现只用 `dist.send` / `dist.recv` / `dist.broadcast` 三个点对点原语，不调用 `dist.all_reduce`（否则就是拿库验证库）。gloo 传 CPU 张量，两个进程可以在同一台机器（甚至同一张卡所在主机）上跑。

- `backend` 模式的 sweep 测的是 CPU 张量经 gloo 的耗时，**不是 CUDA 张量带宽**；即使机器有 GPU 也不改变这条路径。
- 跨机 NCCL 对照复用 11 篇的双机环境（`--backend nccl`，两侧 `torchrun --nnodes=2 --nproc_per_node=1`），本篇不重复搭环境。
- NCCL 内部可能按拓扑选择 ring/tree 算法、切分 channel；手写 Ring 的正确性只证明这条实现路径在所测输入下有效，不能代替对 NCCL 内部调度的观察。

## 快速使用

```bash
cd ~/llm-training-lab

# 正确性：三种手写实现与 dist.all_reduce 逐位对照（约 1 分钟）
./.venv/bin/torchrun --nproc_per_node=2 --master_port=29551 \
    -m exp_collective.collective_lab --mode correctness \
    --out results/Season3/13/correctness.json

# 逐轮 trace：16 元素小张量，核对每 rank 发送字节 = 2(w-1)/w × 张量大小
./.venv/bin/torchrun --nproc_per_node=2 --master_port=29552 \
    -m exp_collective.collective_lab --mode trace \
    --out results/Season3/13/trace.json

# backend sweep：1~256 MB，dist.all_reduce 与手写 Ring 的耗时对照（约 10 分钟）
./.venv/bin/torchrun --nproc_per_node=2 --master_port=29553 \
    -m exp_collective.collective_lab --mode backend --backend gloo \
    --out results/Season3/13/backend_gloo.json

# DDP 梯度对账：跑一步真实 DDP backward，按参数字节核对理论通信量
./.venv/bin/torchrun --nproc_per_node=2 --master_port=29554 \
    -m exp_collective.collective_lab --mode ddp_grad \
    --out results/Season3/13/ddp_grad.json
```

## 结果文件

`results/Season3/13/`：

| 文件 | 内容 |
|---|---|
| `correctness.json` | 1K 元素与 51.7 MB 两档，naive / ring / rs+ag 对 `dist.all_reduce` 的最大绝对误差均为 0.0（当前配置逐位一致；浮点加法不满足结合律，不能推广到任意拓扑） |
| `trace.json` / `trace_rank0.json` | rank0 视角逐轮 trace：reduce_scatter 32 B + all_gather 32 B = 64 B，与公式 2(w-1)/w × 64 B 逐字节吻合 |
| `backend_gloo.json` | gloo CPU 张量 1~256 MB sweep：有效吞吐约 0.10~0.12 GB/s，每档多次迭代取耗时中位数（旧记录未存 iters 字段） |
| `ddp_grad.json` | 12.93M 参数一步 DDP backward：理论通信量 51.7 MB（w=2 系数为 1）；bucket 字节按 25 MiB 上限估算，脚本未读取运行时 bucket |
| `ddp_grad_seeded_20260924.json` | 修正注释并固定 seed 后的重跑记录：字节数与 bucket 切段同原记录，loss 8.761（原记录未固定 seed，loss 8.779 属初始化波动） |

## 已知边界

- 全部主实验在 w=2 下完成。$2(w-1)/w$ 系数在 w=4、8 时为 1.5、1.75，是公式换算，不是八卡实测。
- `ddp_grad` 的 bucket 切段是按 25 MiB 上限的估算；DDP 真实 bucket 布局按梯度就绪顺序分组，脚本没有读取，也没有采集通信与反向重叠的证据。
- 普通 DDP 的梯度同步是 all_reduce，backward 后各 rank 持有完整梯度；ReduceScatter 是 Ring 的中间阶段和分片训练（ZeRO/FSDP 类）的原语，不能混为一谈（源码注释已按此修正）。
- gloo 的 CPU 张量吞吐与 11 篇跨机 NCCL 的 GPU 张量吞吐处于相近量级，但路径不同，不能据此认定瓶颈相同。
