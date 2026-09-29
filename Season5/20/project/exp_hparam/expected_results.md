# expected_results.md —— exp_hparam 预期输出范围

> 20 篇 HParam Lab 的基线文档：每个模式跑完后对照本文档判读。
> 预期范围来自 plan 估算（ESTIMATE）与已落盘的 noise 冒烟测试（REAL）；
> 完整矩阵实测基线在 `results/Season5/20/hparam_all_20260926.json` 落盘核验后回填。

## 环境（REAL）

- WSL2 Ubuntu 24.04，RTX 3070 8GB，PyTorch 2.14.0+cu130，Python 3.12.3
- venv：`~/llm-training-lab/.venv`；PYTHONPATH 需含 20/19/17/16/04 五个工程目录
- 数据底座与 17/18/19 篇同口径：corpus_large.txt（9.25M 字符）、17 篇基准配方
  nl50_dedup08_q（char_budget 1.5M）、char 级词表 6120、指纹跨篇一致

## 显存边界（REAL，探针实测，batch16/seq256 单步峰值）

| hidden | params | peak MiB | 判定 |
|---|---|---|---|
| 384 | 12,971,904 | 1349.2 | 安全 |
| 516 | 22,335,060 | 1668.4 | 安全 |
| 636 | 33,024,300 | 1938.6 | 安全 |
| 768 | 47,177,472 | 2251.3 | 安全 |
| 888 | 62,221,272 | 2578.7 | 安全 |
| 1032 | 83,010,984 | 2935.3 | 安全 |

判读要点：seq_len 256 下激活占比小，显存随 hidden 增长远慢于 hidden²；
8GB 卡全部档位安全，**显存不是本篇瓶颈**（与直觉相反，值得在文章里写）。

## plan（离线估算，ESTIMATE）

完整矩阵 60 格约 29 分钟（1717 s）。各段：noise 4 格 73 s、lr_range 1 格 4 s、
sweep 24 格 439 s、warmup_decay 9 格 165 s、grad_accum 4 格 73 s、
scaling 4 格 265 s（h768 单格 146 s 最贵）、compute_budget 4 格 7 s、
mup 8 格 164 s、predict_larger 2 格 528 s（h1032 单格 264 s）。

判读要点：估算用 19 篇实测 tok/s 按 hidden² 缩放；真实时长以运行日志为准，
偏差 ±30% 内属正常（估算不含评测与数据装载）。

## noise（噪声带，REAL，冒烟测试已落盘 `hparam_noise_20260926.json`）

4 seeds（20260926/1001/1002/1003），1M token 预算，h384 基准配置：

| seed | final_loss | nl_ppl | code_ppl |
|---|---|---|---|
| 20260926 | 4.0352 | 587.37 | 11.84 |
| 1001 | 3.9294 | 591.19 | 10.96 |
| 1002 | 4.5406 | 581.19 | 10.92 |
| 1003 | 4.9880 | 587.04 | 11.50 |

噪声带：nl 3.81% / code 5.79%（极差/均值）。

判读要点：
- 1M token 预算下噪声带比 19 篇 2M 预算（nl 3.85%/code 2.54%）的 code 域更宽，
  符合"token 越少越噪"；本篇全部单变量判定以**本篇噪声带**为标尺，不跨篇借用。
- final_loss 的 seed 间差（3.93~4.99）远大于 ppl 的差：最后一个 batch 的 loss
  是单点值，天然高噪；判读一律用留出集 ppl，不用 final_loss。
- 同 seed 复现一致（all 模式与冒烟测试逐值相同）：noise 段是 all 首段，
  首格 RNG 状态与独立冒烟测试一致，故逐位复现。
- **同配置格不逐位复现（已知工程行为，如实记录）**：all 模式里同配置三格
  （noise 基准 / sweep_b16_lr0.0003 / wd_w0.1_cosine）nl 在 587~596 浮动。
  原因：build_modern 的随机初始化在 train_hparam 的 torch.manual_seed 之前
  执行，模型初值取决于该格之前跑过多少格（全局 RNG 状态），与 19 篇
  train_arch 同模式。差异被噪声带吸收，不影响任何判定；但不得声称
  "同配置逐位可复现"，只能声称"同配置在噪声带内一致"。若未来需要逐位
  确定性，把 manual_seed 提到 build_model 之前并重跑全矩阵（本篇数据
  落盘后不改代码，避免 code_sha256 审计链断裂）。

## sweep（24 格，REAL，采集中）

预期形态：每 batch 档 ppl 随 LR 呈单峰（U 形），大 LR × 大 batch 可能发散
（diverged=True，早停记录，失败案例①素材）。

**已观察（b8/b16/b32 三档，REAL）**：
- b8：nl 最优 lr=0.0006（493.46），单峰干净
- b16：nl 最优 lr=0.0003（595.59），但与 lr=0.0006（606.75）差 1.87% < 噪声带 3.81%
- b32：nl 最优 lr=0.0006（675.13），平台更宽
- 三档 argmin 非单调（0.0006→0.0003→0.0006）

判读要点（方法论，数据逼出来的）：
- 单点 argmin 对噪声敏感（极值点附近平坦是数学必然），**不得**直接用 argmin
  做缩放判定；主判定量用"近似最优平台中心"（与最优差在噪声带内的 LR 集合的
  log 空间几何中心），拟合指数 p（center ∝ batch^p）：p≈1 线性缩放，
  p≈0.5 平方根缩放，p≤0 方向相反。
- 平台宽度本身是结论：宽平台 = 超参不敏感，调参时间可以省。
- 若平台过宽导致 p 不可信，如实写"缩放信号被噪声带淹没"（失败案例④），
  给补救方向（增大 token 预算 / 多 seed 平均），不硬凑规律。
- **边界删失（已从 b64 段确认）**：b64 档 nl_ppl 随 LR 单调下降未见底
  （1586→913→782→753），最优 LR 触网格上限，平台中心被截断有偏。
  删失档从缩放指数拟合中排除（核验脚本 4c 组已实现）；删失方向本身
  是素材（大 batch 需要更大 LR，与缩放规则定性一致）。
- 跨 batch 只比平台中心位置，不比 ppl 绝对值（固定 token 预算下 batch 越大
  步数越少，绝对值不可比；这是检验缩放规则的标准设置，Goyal 2017 同款假设）。

## warmup_decay / grad_accum / scaling / compute_budget / mup / predict_larger

PENDING —— 完整矩阵落盘核验后回填实测基线与判读要点。

预期检查点：
- warmup_decay：**预期已被部分实测推翻**——w0.02_constant nl 499.44 vs cosine 573.85
  （低 13%，超带）。机制是欠拟合区的训练预算效应（constant 把更多预算花在
  高 LR 做功），非 constant 本质更优；收敛区通常反转（REFERENCE）。行文必须
  写透这个区分。核验脚本 4d 组判定 constant 是否跨 warmup 档系统性占优，
  并对照 warmup 比例本身的影响（cosine 内各档极差 vs 噪声带）
- grad_accum：b32_a1 vs b16_a2 vs b8_a4 的 nl 差应在噪声带内（等效），
  b8_a1（有效 batch 不同）应超带
- scaling：L(N)=a·N^b 拟合，b 预期为负（loss 随 N 下降）；外推 1032 的
  预测误差是失败案例②的判定点
- compute_budget：4 档 D/N=2.3~36.1，实测最优与 Chinchilla(20) 的距离；
  小模型 + 小语料下最优点预期偏向更大 D/N（过拟合压力小、步数多）
- mup：最优 LR 随宽度移动方向 vs 1/w、1/√w 规则
- predict_larger：迁移 LR vs 直搬 LR 的 ppl 差，回扣核心判断

## 硬校验（数据落盘后逐项过）

- [ ] 指纹与 17/18/19 篇一致
- [ ] vocab 6120
- [ ] 60 格齐全（4+1+24+9+4+4+4+8+2）
- [ ] noise 同 seed 与冒烟测试复现一致
- [ ] 发散格 diverged=True 如实记录（不美化）
- [ ] 全部 ppl 非负、非 NaN（发散格除外）
- [ ] scaling 拟合 b < 0
- [ ] compute_budget 各档 FLOPs ≈ 6e12（±5%）
