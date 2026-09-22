# DevResources/Season2/10 —— Single GPU Performance Lab

对应文章：`Articles/Season2/10_显存省下来了，为什么 GPU 还是吃不满？.md`

## 本篇解决什么问题

模型放得下了（08 篇），attention 也换成融合 kernel 了（09 篇），GPU utilization 仍然忽高忽低。一个训练 step 到底在等数据、等 CPU、等 kernel，还是被同步和周期任务打断？

核心判断：单卡性能不是 GPU 算得快不快，而是整条流水线有没有让 GPU 停下来等。要量的是 **GPU 空洞**（一步墙上时间里没有 kernel 在跑的部分），不是 nvidia-smi 的 utilization。

## 包内容

```text
10/
├── README.md                  # 本文件
└── project/
    └── exp_perf/
        ├── perf_harness.py    # 阶段打点（perf_counter 口径）、step 分布、GPU 空洞测量（chrome trace 区间并集）
        ├── perf_bench.py      # 端到端训练 + 5 种故障注入 + 11 个优化开关 + --repeats 重复测量
        ├── perf_report.py     # 7 张表：含 sanity 自检表（busy > wall 即标不可用）
        ├── diag_trace.py      # 诊断：gpu_busy 超过 wall 的根因排查（瞬态 trace 损坏）
        └── diag_heavy_trace.py # 诊断：profiler 窗口被撑大的排查
```

两个 diag 脚本是采集期排查测量错误用的，文章第 2、8 章的踩坑叙述就是靠它们拿到证据。

## 前置：本篇包依赖前序篇的模块

`exp_perf` 复用 06 篇的 `build_llama` 与 `contract_loss`，并依赖 02 篇骨架。本篇包只含**新增**的 `exp_perf/`，运行前需累积：

| 需要的模块 | 来自 | 用途 |
|---|---|---|
| `common/` | 02 篇包 | seed / config / 留痕 / benchmark / 指标口径 |
| `exp_hf/` | 06 篇包 | `build_llama` 模型构建、`contract_loss` |
| `exp_perf/` | 本篇包 | GPU 空洞度量 + 阶段打点 + 故障注入 |

```bash
cp -r DevResources/Season2/10/project/exp_perf ~/llm-training-lab/
cd ~/llm-training-lab
```

## 快速使用

```bash
cd ~/llm-training-lab

# baseline：未优化端到端 + GPU 空洞
./.venv/bin/python -m exp_perf.perf_bench --mode baseline --steps 60 --warmup 8 --profile \
    --repeats 3 --out results/Season2/10/rep_baseline.json

# 故障注入（5 种）：让 GPU 等数据 / 等 CPU / 同步日志
./.venv/bin/python -m exp_perf.perf_bench --mode inject --fault slow_data --slow-data-ms 8 --profile --repeats 3 --out ...
./.venv/bin/python -m exp_perf.perf_bench --mode inject --fault heavy_cpu --profile --repeats 3 --out ...
./.venv/bin/python -m exp_perf.perf_bench --mode inject --fault sync_log --log-every 1 --sync-log-ms 20 --profile --repeats 3 --out ...

# 单项优化（11 个开关）
./.venv/bin/python -m exp_perf.perf_bench --mode optimize --opts compile --profile --repeats 3 --out ...

# 组合瀑布
./.venv/bin/python -m exp_perf.perf_bench --mode optimize --opts workers,pin,prefetch,persistent,nonblock,sdpa,fused,setnone,compile --profile --repeats 3 --out ...

# 报告（含 sanity 自检表）
./.venv/bin/python -m exp_perf.perf_report results/Season2/10
```

`--mode` 三选一：`baseline` / `inject` / `optimize`。`--fault` 可选 `slow_data` `heavy_cpu` `no_pin` `sync_log`。`--opts` 可组合 `workers` `pin` `prefetch` `persistent` `nonblock` `sdpa` `fused` `setnone` `compile` `ckpt` `accum`。`--repeats N` 同配置跑 N 轮取中位数。

## 结果文件

`results/Season2/10/`（88 个）。**命名约定**：`rep_*` 是 3 轮中位数口径（文章只引用这些），无前缀的是单次跑（只用于历史踩坑叙述与趋势扫描）。

核心证据链（`rep_baseline` / `rep_slow_data` / `rep_slow_data_workers`）：

| 组 | step 中位 ms | tok/s | busy | data 占比 |
|---|---|---|---|---|
| baseline | 59.332 | 69035 | 61.6% | 0.43% |
| 注入慢数据（sleep 8ms，workers=0） | 207.794 | 19712 | 28.4% | 62.77% |
| 慢数据 + 并行加载 | 59.400 | 68957 | 60.9% | 4.21% |

优化瀑布（`rep_combo_*`）：baseline 69035 → combo_no_compile 98398（+42.5%）→ combo_all 154524 tok/s（2.24x，peak 762.5 MB）。

## 本篇踩过的测量坑（全部已修，工具里留有记录）

1. **CUDA event 测不到 CPU 阻塞**：data 阶段恒为 0。改用 perf_counter + 每段 synchronize。
2. **gpu_busy 连错两版**：v1 全表累加报 151.4%，v2 只过滤 CUDA 报 493.1%。根因是 `record_function` 注解的 device_type 也是 CUDA 且聚合子 kernel。正确口径是 chrome trace 的 kernel 区间**并集**。
3. **heavy_cpu / sync_log 注入量级不足**：第一版 heavy_cpu 注入几十微秒（想模拟 8ms），sync_log 只有 synchronize（在每段都同步的 harness 里免费）。改为运行时自校准，实测毫秒数写进 `inject` 字段。
4. **瞬态 trace 损坏**：no_pin 曾报 busy 351.3%，heavy_cpu 曾报窗口撑大 3 倍。重跑即恢复。加了窗口自检（window > wall×1.5）。

## 本篇的独有发现：GPU 被数据饿着会降频

注入慢数据后，gemm kernel 单次耗时 70.1 → 99.2 us（1.41x）。nvidia-smi 时钟采样显示 SM 时钟从 1725/1950 MHz 掉到 330~555 MHz。机制：GPU 空闲间隙掉到低功耗态，下一个 kernel 来不及爬回高频。修好数据供给后 kernel 连续到达，时钟保持高频，单次耗时回到 68.9 us。

所以"kernel 时间不变"是错的。正确表述：主效应是空转（step 3.5x），次效应是降频（kernel 1.4x）。

## 已知边界

- **单次测量不可信**：run-to-run 方差可达 1.85 倍。关键组必须 `--repeats 3`，极差比超 1.15 标 `unstable`，文章只能当量级引用。
- **WSL2 溢出防护**：reserved 超整卡 60% 的数据不可用于速度对比（09 篇教训），工具自动标 `spillover_suspect`。本篇全部组 reserved 占 16%~17%，未触发。
- 模型只有 12.93M 参数，optimizer 阶段占比比大模型高，比例不能外推。baseline 的 busy 61.6% 偏低有 kernel 碎的因素，大模型上会更高。
- torch.compile 只测默认模式，max-autotune 在 46 SM 上不可用（inductor 报 `Not enough SMs`）。
