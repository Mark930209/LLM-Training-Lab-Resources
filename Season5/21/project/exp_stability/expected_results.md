# expected_results.md —— exp_stability 预期输出范围

> 21 篇 Numerical Stability Lab 的基线文档：每个模式跑完后对照本文档判读。
> 冒烟测试已验证 range/precision/spike/lr_ramp/recovery；完整矩阵实测基线在
> `results/Season5/21/stability_all_*.json` 落盘核验后回填。

## 环境（REAL）

- WSL2 Ubuntu 24.04，RTX 3070 8GB（Ampere：fp16/bf16 硬件支持，FP8 无），
  PyTorch 2.14.0+cu130，Python 3.12.3
- 数据底座与 17/18/19/20 篇同口径（指纹跨篇核对）；架构固定 19 篇 modern
  （rms+pre+swiglu+rope+mha，h384/6层/heads6）；超参基准取 20 篇结论
  （lr 3e-4、b16、warmup 0.1；衰减本篇按场景选）

## range（数值范围演示，REAL 张量实测）

fp16 上溢：70000 → **Infinity**（max finite 65504）；下溢：1e-8 → **0.0**。
bf16：70000 → 70144（有损但有限）；1e-8 → 1.0e-8（指数范围同 fp32）。
精度损失：bf16 `1+2^-9` → 1.0（尾数 8 位舍掉）；fp16 `1+2^-12` → 1.0（尾数 11 位）。
FP8 e4m3/e5m2 只做解析式范围表（REFERENCE，无硬件运行路径）。

## precision（6 格，REAL，已落盘 stability_all_20260928.json 初版）

| 格 | loss | nl_ppl | code_ppl | tok/s | peak MiB | scale_adj |
|---|---|---|---|---|---|---|
| fp32_clipon | 3.3585 | 585.43 | 11.16 | 41659 | 1461.2 | — |
| fp32_clipoff | 3.7282 | 668.47 | 15.11 | 39593 | 1460.3 | — |
| fp16_clipon | 3.3520 | 584.81 | 11.24 | 61616 | 1219.3 | 0 |
| fp16_clipoff | 3.7283 | 668.48 | 15.11 | 52274 | 1219.3 | 0 |
| bf16_clipon | 3.3527 | 584.53 | 11.23 | 61461 | 1218.1 | — |
| bf16_clipoff | 3.7281 | 668.48 | 15.11 | 54651 | 1218.1 | — |

判读要点：
- fp16/bf16 与 fp32 质量差在噪声带内（nl 584.5~585.4 / 668.5），**精度不损质量**；
  吞吐 +48%（61.6k vs 41.7k tok/s）、显存 −17%（1219 vs 1461 MiB）。
- clip on/off 差异巨大（nl 585 vs 668，+14%）：**lr 3e-4 下 clip 是质量必需**，
  不是可选项（clipoff 三精度 loss 几乎相同 3.728——clip 效应在精度之上）。
- fp16 scale_adjusts=0：正常训练下 GradScaler 无需调整（scale 稳定 65536），
  调整只发生在 spike/溢出场景——这是"fp16 需要 loss scaling 机制"的对照证据。
- bf16 无需 scaler（指数范围同 fp32，不上溢），代码更简单——主流大模型选 bf16 的原因。

## spike（3 格，REAL，冒烟已验证注入生效）

**初版注入失败的三层根因（真实踩坑，文章素材）**：
1. `loss × bad_factor` 只放大 backward 的梯度，记录的 loss 是未放大值 → detect_spike 看不到；
2. AdamW 自适应归一化（m/√v）吸收单次梯度放大 → loss×100 对 Adam 基本无效；
3. fp16 GradScaler 遇 inf 梯度自动跳步 → 进一步保护模型。
三重保护下 spike 三格全 0。**修法**：污染目标 token（corrupt_frac=1.0，模拟脏标签），
cross_entropy 真实飙到 ~2×log(vocab)≈13.5（基线 4.45），三精度全部触发。

冒烟实测（lr×5=1.5e-3、关 clip、每 60 步注入）：
- 坏 batch 步 loss 13.5/12.8/12.4/11.9（基线中位 4.45，坏/基线 2.67~3.04×）；
  grad_norm 4.5/3.3/2.8/3.0（基线中位 0.58，坏/基线 4.8~7.7×）。
- **loss 判据天然脆弱**：自然波动最大已达基线 1.97×，3× 阈值漏检 3/4 个坏 batch；
  spike_detect_k=2.5（自然最大 1.97 与坏最小 2.67 之间）才能全捕获。
- **grad_norm 判据分离干净**：4× 阈值零漏检零误报——前置指标优于 loss 的定量证据。
- 脏数据 spike 的 loss 与 grad_norm **同步**触发（lead≈0）：本场景验证"可复现、
  可定位"，提前量由 lr_ramp 场景给出。

## lr_ramp（3 格，REAL，冒烟已验证）

LR 从 3e-4 线性攀升到 0.15（20% 步数后开始，constant 衰减）。冒烟实测：
- **bf16（梯度爆炸型发散）**：step 216 起 grad_norm 9.6→24.7→13.9→139 持续爬升，
  loss 到 228 才爆炸（8.61）→ 234 NaN。**grad_norm 前兆领先 loss 爆炸约 12 步**。
- **fp16（突发溢出型发散）**：发散前一步 grad_norm 0.23 完全正常，loss 直接 NaN
  （step 133/156/207 视 seed 与调度）——**无 grad_norm 前兆**：fp16 溢出发生在
  前向瞬间（激活→inf），来不及在梯度上留下痕迹。这是"fp16 需要 loss scaling +
  inf 检测跳步"的实测依据，也是"grad_norm 监控不是万能"的诚实边界。
- **fp32**：lr 0.15 仍不发散（grad_norm 0.3~0.5 平稳）——pre-norm 架构 + fp32
  最稳（19 篇 stress 已证 pre-norm 稳定性，此处再证）。
- 告警判读三要素（冒烟调出的方法论）：① 基线窗口取 ramp 前稳定段（[ramp_start-8,
  ramp_start)），不能用 warmup 尾部（gn 1.9~4.8 会压低阈值→误报）；② sustain=3
  （连续 3 步超阈才算告警，过滤零星单步误报）；③ 阈值 = 基线中位 × 4。
- 判读纪律：**lead 只在梯度爆炸型发散里为正**（bf16 ~12 步）；突发溢出型
  lead=None 是诚实结果，不硬凑。

## recovery（5 格，REAL，冒烟已验证 5 策略分化）

同一 spike 场景（脏数据注入、lr×5、关 clip、fp16），5 策略对照。
**冒烟实测（nl_ppl，6 次 spike）**：

| 策略 | nl_ppl | vs no_action | 机制 |
|---|---|---|---|
| skip_bad | **523.08** | −9.2%（最优）| 6 个坏 batch 全跳过，零损伤落地 |
| rollback_rewarmup | 562.19 | −2.4% | 回退干净 ckpt（rb=6）+ re-warmup，但丢 40~60 步进度 |
| no_action（对照）| 576.28 | — | 坏更新照常落地，损伤基线 |
| tighten_clip | 675.77 | **+17.3%（更差）**| clip 0.1 过紧，正常梯度也被砍，欠拟合 |
| lower_lr | 680.93 | **+18.2%（更差）**| 6 次各 LR×0.5 → 缩到 1/64，严重欠拟合 |

判读要点（核心结论）：**最佳处置是精准移除坏数据（skip_bad），全局压制训练
（lower_lr/tighten_clip）反而比不处置更差**——过度矫正的损伤超过 spike 本身。
lower_lr 每次 spike 都砍半 LR，6 次后 LR 缩到 1/64，欠拟合损伤远超它避免的
spike 损伤；tighten_clip 把 clip 收到 0.1，正常步的梯度也被过度裁剪。
策略优劣必须与 no_action 对照 + 噪声带判定；"处置造成的步数损失"
（skip/rollback 的代价）与最终质量一起报告。

**rollback 无限循环踩坑（真实，已修）**：初版回退到 ckpt（step 40）后重训到
step 60 又遇同一坏 batch（inject_every=60，60%60==0）→ 再 spike → 再回退，
step 永远卡在 40~60，实测卡死 14 分钟。修法：① handled_inject 集合隔离触发
spike 的注入步（现实语义=定位并跳过问题数据）；② 回退后跳过本步更新（坏梯度
不落刚恢复的参数）+ fp16 补 scaler.update() 重置 unscale 状态（否则下次迭代
unscale_ 报 "already been called"）；③ 真正的 re-warmup（LR 从 0 线性升回）。

## silent（1 格，REAL，冒烟已验证退化复现）

记忆化子集：50% 步后只用 1% 数据（15K token ≈ 4 个 batch），train loss 下降
（曲线"更好看"），留出 eval ppl 持续上升 = 静默退化。eval_every=20 画退化曲线。

**首跑失败与修复（真实踩坑）**：初版 10% 子集（150K token）+ 40% 切换点，
silent_degradation=False、gap=−17.64%——150K token 对 h384 模型太大，366 步内
记不住，切换后 eval ppl 仍在降（欠拟合主导，记忆化没发生）。修法：子集缩到
1%（15K token，百步级可背完）+ 切换点推到 50%（留 183 步给记忆化发生）。

冒烟实测（修复后）：degraded=**True**，gap=**+22.15%**。剪刀差清晰：
切换点（step 183）前 train_loss 6.83→4.32 / eval_ppl 2568→586.5 同向降；
切换点后 train_loss 4.32→3.93（继续降，"曲线更好看"）而 eval_ppl
586.5→750.63（持续升 +28%）。只盯 loss 曲线的人会在 step 360 宣布训练成功。

判据：is_silent_degradation（切换点后 train loss 降而 eval ppl 升）+
degradation_gap_pct（切换点前后 eval ppl 差）。

## 硬校验（完整矩阵落盘后逐项过）

- [ ] 指纹与 17/18/19/20 篇一致
- [ ] vocab 6120
- [ ] precision 6 格 + spike 3 格 + lr_ramp 3 格 + recovery 5 格 + silent 1 格 = 18 格
- [ ] spike 三格 spike_count ≥ 4（注入生效）
- [ ] lr_ramp bf16 diverged=True 且 alarm 有正 lead
- [ ] recovery no_action 与 skip_bad 的 nl 差超噪声带
- [ ] silent 的 silent_degradation=True
- [ ] fp16 溢出型发散如实记录 alarm=None（不硬凑提前量）
