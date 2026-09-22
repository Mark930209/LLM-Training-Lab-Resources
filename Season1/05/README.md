# DevResources/Season1/05 —— 训练正确性体检

对应文章：`Articles/Season1/05_模型训练完就可以了吗——那些隐藏在训练背后的问题.md`

## 本篇解决什么问题

loss 在下降、程序没有报错，是否足以证明训练正确？

不够。本篇给训练循环装上诊断面板，再**人为注入 13 类故障**，看哪些会报错、哪些只让指标悄悄失真。最有价值的发现是静默故障：`label_shift` 把训练任务换成"跳一个字预测"，train loss 降到 0.45（看起来是奇迹），val loss 却涨到 9.85（比随机初始化的 8.70 还差）。

核心判断：训练诊断不是背报错原因，而是先建立健康基线，再用观测证据逐步缩小故障范围。

## 包内容

```text
05/
├── README.md                  # 本文件
└── project/
    └── exp_debug/
        ├── diagnostics.py          # 诊断面板：batch 重复率 / 数据集重叠数 / update_ratio / 三类校验和
        ├── fail_modes.py           # 13 类故障开关（data 4 / update 5 / resume 4），含静默/报错分类
        ├── correctness_harness.py  # 五阶段正确性门禁（只观测不干预）
        ├── train_debug.py          # 主程序：带诊断面板的训练循环，--fault 指定注入项
        ├── run_all.py              # 一键跑完整矩阵（16 组）
        └── README.md               # 模块内说明
```

## 前置：本篇包依赖前序篇的模块

`exp_debug` 复用 04 篇的模型、数据、优化器与调度器，并依赖 02 篇的可复现骨架。本篇包只含**新增**的 `exp_debug/`，运行前需要把这些模块累积到同一个工作目录：

| 需要的模块 | 来自 | 用途 |
|---|---|---|
| `common/` | 02 篇包 | seed 管理、config、留痕、benchmark、指标口径 |
| `exp_scale/` | 04 篇包 | 模型、char tokenizer、语料加载、lr 调度 |
| `exp_debug/` | 本篇包 | 诊断面板 + 故障注入 + 正确性门禁 |

累积方式（WSL2 内，`~/llm-training-lab` 是 02 篇建好的工作目录）：

```bash
cp -r DevResources/Season1/05/project/exp_debug ~/llm-training-lab/
cd ~/llm-training-lab
```

语料沿用 04 篇的 `exp_scale/data/`（四大名著），不需要重新下载。

## 快速使用

```bash
cd ~/llm-training-lab

# 健康基线（不注入故障），300 步
./.venv/bin/python -m exp_debug.train_debug --config exp_scale/config_10m.yaml \
    --steps 300 --output results/Season1/05/baseline.json

# 注入单个故障
./.venv/bin/python -m exp_debug.train_debug --config exp_scale/config_10m.yaml \
    --steps 300 --fault label_shift --output results/Season1/05/fault_label_shift.json

# 一键跑完整矩阵（13 故障 + 3 对照 = 16 组，约 8 分钟）
./.venv/bin/python -m exp_debug.run_all --config exp_scale/config_10m.yaml \
    --steps 300 --out results/Season1/05/results_debug.json
```

`--fault` 可选值见下表。不传即健康 baseline。

## 13 类故障的注入实现

文章里每个故障都对应一段真实代码，不是概念描述：

| 故障 | 注入方式 | 显性/静默 |
|---|---|---|
| `label_shift` | `torch.roll` 把 target 移 3 位 | 静默（train 假降，val 暴涨） |
| `label_shuffle` | `randperm` 重排 target | 静默（loss 停在字符频率熵 6.3680） |
| `dup_batch` | 前半 batch 覆盖后半 | 静默 |
| `val_leak` | `LeakyValDataset` 把 10% train 拼进 val | 静默（短训练不显形） |
| `lr_zero` | lr × 0 | 静默（loss 不动） |
| `lr_huge` | lr × 100 | 显性（NaN@132） |
| `no_clip` | 跳过 `clip_grad_norm_` | 诚实不告警（单独无害） |
| `lr_huge_noclip` | lr × 100 且不裁剪 | 显性（NaN@109，更早） |
| `amp_overflow` | `GradScaler(init_scale=2^48)` 强制跳步 | 静默 |
| `resume_opt` | 恢复时跳过 optimizer 状态 | 静默 |
| `resume_scaler` | 恢复时跳过 scaler | 诚实不告警（逐位一致） |
| `resume_rng` | 恢复时跳过 RNG 状态 | 静默 |
| `resume_sched` | 恢复时跳过 scheduler（warmup 重跑） | **最危险**：val 4.92 比正确续训 5.35 还好 |

## 结果文件

`results/Season1/05/`：

| 文件 | 内容 |
|---|---|
| `results_debug.json` | 完整矩阵（16 组）：每组含 train/val loss、告警列表、诊断指标 |
| `clean_small.json` | val_leak 的干净对照组（小语料长训练 2000 步，val 4.53） |
| `leak_small.json` | val_leak 的泄漏组（同配置，val 2.89） |

明星数据：`label_shift` 的 train 0.45 / val 9.85 剪刀差；`label_shuffle` 停在 6.3680（实测字符频率熵，定量解释"随机标签 loss 也会降"）；`resume_sched` 的伪续训（指标变好掩盖故障）。

## 正确性门禁的硬约束

`correctness_harness.py` **只观测不干预**：接入前后基线 train 5.1396 / val 5.3229 / param_checksum d577125c47b4 逐位一致。门禁不能改变被观测的训练。

这个约束来自一个真实教训：门禁初版缺 `check_val_loss`，导致 `label_shift` 抓不到（train loss 降得太漂亮，单看 train 一切正常）。**诊断指标必须先自证能抓到注入的故障，再写进文章。**

## 已知边界

- 13 个故障中 11 个静默，2 个（`no_clip`、`resume_scaler`）诚实不告警。"不告警"本身是结论，不是漏检。
- `val_leak` 在 300 步下不显形，必须用 2000 步长训练对照才能看出真实泛化差距（1.97 被抹成 0.33）。
- 配置沿用 04 篇 10M 档（12.93M 参数，四大名著语料），实测峰值 1233.6 MB。
- 本篇不深挖 OOM、吞吐瓶颈与多卡死锁：OOM 见 07/08 篇，GPU 利用率见 10 篇，多卡故障见 Season 3。
