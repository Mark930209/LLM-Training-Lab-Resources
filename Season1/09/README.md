# DevResources/Season1/09 —— Attention Kernel Lab

对应文章：`Articles/Season1/09_FlashAttention 为什么更快：从 Attention 矩阵到显存读写.md`

## 本篇解决什么问题

FlashAttention 的浮点运算量比朴素实现更多，却跑得更快。如果瓶颈在算力，这件事不该发生。

核心判断：attention kernel 的差距不在 FLOPs，在中间那个 N×N 矩阵要不要反复进出显存。本篇用算术强度（SCALED 推导）与五种实现对照（REAL 实测）两条路验证。

## 包内容

```text
09/
├── README.md                  # 本文件
└── project/
    ├── dump_io_arithmetic.py  # 把 io_arithmetic 的推导写成 results JSON（SCALED）
    └── exp_attn/
        ├── naive_attn.py          # 手写 naive 实现：显式物化完整 N×N 分数矩阵，对照基线
        ├── backend_probe.py       # SDPA backend 可用性探测 + 内核名识别 + forced_honored 自证
        ├── correctness.py         # 五种实现的数值一致性（对 naive 的误差）
        ├── attn_bench.py          # 五种 sweep：backend / seq / shape / dtype / causal
        ├── e2e_step.py            # 端到端单步：kernel 级收益稀释到完整训练程序
        ├── io_arithmetic.py       # HBM 流量与算术强度推导（SCALED，非实测）
        └── summarize.py           # 汇总成对照表
```

五种实现 = SDPA 的四个 backend（math / efficient / flash / cudnn）+ 手写 naive。

## 前置：本篇包依赖前序篇的模块

`exp_attn` 复用 06 篇的 `build_llama` 与 `contract_loss`，并依赖 02 篇骨架。本篇包只含**新增**的 `exp_attn/`，运行前需累积：

| 需要的模块 | 来自 | 用途 |
|---|---|---|
| `common/` | 02 篇包 | seed / config / 留痕 / benchmark / 指标口径 |
| `exp_hf/` | 06 篇包 | `build_llama` 模型构建、`contract_loss` |
| `exp_attn/` | 本篇包 | 五种 attention 实现 + backend 探测 + IO 推导 |

```bash
cp -r DevResources/Season1/09/project/exp_attn ~/llm-training-lab/
cp DevResources/Season1/09/project/dump_io_arithmetic.py ~/llm-training-lab/exp_attn/
cd ~/llm-training-lab
```

## 快速使用

```bash
cd ~/llm-training-lab

# backend 探测：这张卡上哪些 backend 可用，强制指定是否真的生效
./.venv/bin/python -m exp_attn.backend_probe --seq 1024 --batch 2 --heads 8 --dtype fp16 \
    --out results/Season1/09/probe_seq1024_fp16.json

# 数值一致性：五种实现对 naive 的误差
./.venv/bin/python -m exp_attn.correctness --seq 256 --dtype fp16 \
    --out results/Season1/09/correctness_seq256_fp16.json

# backend 对照（文章 1.2 节的表）
./.venv/bin/python -m exp_attn.attn_bench --mode backend --seq 1024 --batch 2 --heads 8 --dtype fp16 \
    --out results/Season1/09/backend_seq1024_fp16.json

# 四种 sweep
./.venv/bin/python -m exp_attn.attn_bench --mode seq --dtype fp16 --out results/Season1/09/sweep_seq_fp16.json
./.venv/bin/python -m exp_attn.attn_bench --mode shape --dtype fp16 --out results/Season1/09/sweep_headdim_fp16.json
./.venv/bin/python -m exp_attn.attn_bench --mode dtype --out results/Season1/09/sweep_dtype.json

# 端到端：kernel 收益在完整训练程序里被稀释多少
./.venv/bin/python -m exp_attn.e2e_step --seq 1024 --dtype fp16 --out results/Season1/09/e2e_seq1024_fp16.json

# 算术强度推导（SCALED）
./.venv/bin/python -m exp_attn.dump_io_arithmetic --out results/Season1/09/io_arithmetic.json
```

`--mode` 五选一：`backend` / `seq` / `shape` / `dtype` / `causal`。`--impls` 可指定实现子集。

## 结果文件

`results/Season1/09/`（15 个）：

| 文件 | 内容 |
|---|---|
| `backend_seq1024_fp16.json` | 五种实现对照：naive 1.975 / math 4.926 / efficient 0.575 / flash 0.425 / cudnn 0.513 ms；peak 154.3 / 292.3 / 38.3 / 38.4 / 40.4 MB |
| `probe_seq1024_fp16.json` / `probe_seq1024_fp32.json` | fp16 四 backend 全可用；fp32 下 flash/cudnn 不可用 |
| `correctness_seq256_fp16.json` | 四 backend 对 naive 误差均为 fp16 1 ULP |
| `sweep_seq_fp16.json` | 显存倍率 naive/flash 从 4.0x（seq 1024）涨到 20.1x（seq 4096） |
| `sweep_headdim_fp16.json` | head_dim 256 时 efficient 退化 9.7 倍、cudnn 不可用 |
| `backend_seq1024_fp16_causal.json` / `_noncausal.json` | causal 对照两边方向相反 |
| `e2e_seq{256,512,1024,2048}_fp16.json` | kernel 级 5.1x 稀释到端到端 1.76x（seq 1024）；seq 256 差异在噪声内 |
| `e2e_seq2048_redline060.json` | 画红线 0.60 后的重测（溢出防护触发） |
| `io_arithmetic.json` | 算术强度 10.7 FLOP/byte 对机器平衡 45（fp32）/ 181（fp16 tensor），SCALED |

## 本篇最有价值的失败案例

seq 2048 的端到端跑出 **272 倍"加速比"**。核查确认是 WSL2 DXG 显存溢出：eager 路径 reserved 占整卡 81.3%，画红线 0.60 后直接 OOM，同配置三次测量 24064 / 8186 / 115138 ms，差 14 倍。

溢出防护在本篇真实触发：工具自己算出 1505 倍，又自己标上 `speedup_unreliable` 判定不可引用。**benchmark 出现数量级异常，先怀疑测量环境，再怀疑结论。**

## 两个工具 bug（已修）

1. **backend 检测跑默认路径**：第一版把强制指定的 backend 全标成 flash。修法是加 `forced` 参数与 `forced_honored` 自证字段。
2. **`model.to(fp16)` 让 GradScaler 报错**：`unscale_` 抛 ValueError。修法是 fp32 主权重 + autocast，不直接转模型精度。

## 已知边界

- `io_arithmetic.json` 是 **SCALED**（按实现结构做的算术推导），不是实测。只有比值可用，绝对值不可当实测引用。
- 五种实现的对照在 sm86（RTX 3070）上做的。flash 要求 sm80 以上，fp32 下不可用；换架构结果会变。
- head_dim 256 的退化是这张卡上的实测，不同 cudnn 版本行为可能不同。
