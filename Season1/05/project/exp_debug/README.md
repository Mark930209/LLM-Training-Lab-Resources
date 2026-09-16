# Season 1 / 05：训练正确性体检（exp_debug）

复用 04 篇的模型、语料与训练程序，给训练循环补最小诊断面板，
再用 `fail_mode` 开关注入 14 类故障，回答一个问题：

> loss 在下降、程序没有报错，是否足以证明训练正确？

## 目录结构

```text
exp_debug/
├── fail_modes.py     # 故障注入框架：14 个开关 + 阶段/静默分类
├── diagnostics.py    # 诊断面板：重复率 / 重叠数 / 更新量 / 校验和
├── train_debug.py    # 主实验程序（基线 + 单故障 + 续训对照）
└── run_all.py        # 一键跑完整实验矩阵（16 组）
```

## 快速开始

```bash
# 在 project/ 目录下（依赖 04 篇的 exp_scale/ 与 common/）
python -m exp_debug.run_all --config exp_scale/config_10m.yaml --out results_debug.json

# 单个故障
python -m exp_debug.train_debug --config exp_scale/config_10m.yaml --steps 300 --fault label_shift

# 续训对照
python -m exp_debug.train_debug --config exp_scale/config_10m.yaml --steps 150 --save-ckpt /tmp/mid.pt
python -m exp_debug.train_debug --config exp_scale/config_10m.yaml --steps 300 --resume /tmp/mid.pt
python -m exp_debug.train_debug --config exp_scale/config_10m.yaml --steps 300 --resume /tmp/mid.pt --fault resume_opt
```

全套 16 组实验在 RTX 3070 上约 8 分钟（每组 300 步约 20 秒）。

## 实验矩阵实测结果（REAL，10M × 大语料，300 步）

| 故障 | 阶段 | 静默/报错 | 关键现象 |
|---|---|---|---|
| baseline | - | - | train 5.14 / val 5.32 / ur 3.06e-04 |
| label_shift | data | 静默 | train 0.45（假奇迹）/ val 9.85（比随机还差） |
| label_shuffle | data | 静默 | loss 停在 6.29 ≈ 字符频率熵 6.37 |
| dup_batch | data | 静默 | batch 内重复率 0.5 直接抓到 |
| val_leak | data | 静默 | 重叠数 1000/1000；长训练泄漏组 val 2.89 vs 干净组 4.53 |
| lr_zero | update | 静默 | update_ratio = 0，loss 卡 8.81 |
| lr_huge | update | 报错 | NaN @ step 132（裁剪拖后 131 步） |
| no_clip | update | 静默 | 单独无害（5.23 vs 5.14），诚实记录 |
| lr_huge_noclip | update | 报错 | NaN @ step 109，裁剪确实在挡爆炸 |
| amp_overflow | update | 静默 | scale 2^48 跳步，loss 5.19，学习被拖慢 |
| resume_opt | resume | 静默 | val 5.37 vs 5.35，opt 校验和不一致 |
| resume_scaler | resume | 静默 | 与完整恢复逐位一致（scale 是 2 的幂，诚实边界） |
| resume_rng | resume | 静默 | val 5.42，后续 batch 序列不同 |
| resume_sched | resume | 静默 | val 4.92 反而更好——伪续训最危险的形态 |

## 故障清单

| 故障 | 阶段 | 静默/报错 | 现象要点 |
|---|---|---|---|
| label_shift | data | 静默 | loss 下降但任务已换，val 不降 |
| label_shuffle | data | 静默 | loss 卡在 ln(vocab) 附近 |
| dup_batch | data | 静默 | 指纹重复率 ~0.5，有效数据减半 |
| val_leak | data | 静默 | val loss 虚低，失去意义 |
| lr_zero | update | 静默 | update_ratio = 0，loss 不动 |
| lr_huge | update | 报错 | loss 爆炸 / NaN（step 132） |
| no_clip | update | 静默 | 本规模下单独无害（诚实记录） |
| lr_huge_noclip | update | 报错 | 对照 lr_huge，看裁剪挡住了什么 |
| amp_overflow | update | 静默 | scale 2^48：inf 梯度跳步，参数不动 |
| resume_opt | resume | 静默 | 续训 loss 跳变，opt 校验和不一致 |
| resume_scaler | resume | 静默 | AMP 续训 scale 重置 |
| resume_rng | resume | 静默 | 后续 batch 序列不同 |
| resume_sched | resume | 静默 | lr 进度重置，warmup 重跑 |

## 诊断面板（Correctness Harness 核心）

| 阶段 | 指标 | 健康值 | 检测什么 |
|---|---|---|---|
| 数据 | batch 内重复率 / 数据集重叠数 | ≈ 0 | dup_batch / val_leak |
| 前向 | 初始 loss vs ln(vocab) | 偏差 < 0.5 | 结构 / 数据 bug |
| 反向 | grad_norm | 有限、量级稳定 | 爆炸 / 消失 / NaN |
| 更新 | update_ratio = ‖Δw‖/‖w‖ | ~1e-3 | lr_zero / 过大更新 / 跳步 |
| 恢复 | param/opt/rng 校验和 | 与连续训练一致 | 伪续训 |

## 与 04 篇的关系

- 模型、语料、优化器、调度器全部复用 `exp_scale/`，零改动。
- 本篇新增的 `exp_debug/` 只做两件事：诊断面板 + 故障注入。
- 06 篇接入 Hugging Face 模型后，本篇的 Harness 作为回归门禁复用。