# DevResources/Season1/04 —— 真正的训练程序 + 四大名著语料库

对应文章：`Articles/Season1/第二个实验：当我们加大语料规模时，会发生什么？.md`

## 本篇解决什么问题

承接 03 篇的结论——语料不够，所以模型不够好。本篇把语料库从单本《西游记》
（72.3 万字符）扩到四大名著（309.1 万字符），用真实实验回答**加大语料规模会带来
什么变化、代价是什么**；同时把 03 篇的演示代码补成一套可复用的训练程序。

## 包内容

```text
04/
├── README.md                  # 本文件
├── scripts/
│   ├── build_corpus.py        # 四大名著语料库构建（抓取→清洗→繁转简→切分）
│   ├── estimate_memory.py     # 显存账：启动前算"会不会 OOM"
│   ├── diagnose_init.py       # 初始 loss 诊断（放大后 logits 尺度检查）
│   ├── collect_samples.py     # 采集各模型对同一提示词的续写输出（demo 数据）
│   ├── serve_demo.py          # 模型对比演示 WebUI（零依赖，只用标准库）
│   ├── collect_data.sh        # 主实验采集（约 90 分钟）
│   ├── collect_fill_gaps.sh   # 补齐 2×2 对照矩阵缺失格
│   ├── verify_resume.sh       # resume 验证：中断→恢复→曲线连续
│   ├── quick_train.sh         # 快速验证（100 步）
│   └── sync_project.sh        # Windows 侧工程同步到 WSL
└── project/
    ├── common/                # 与 02/03 篇完全同一套骨架（seed/config/logging/benchmark）
    └── exp_scale/
        ├── config_10m.yaml    # 10M 档（hidden 384 / 6 层）
        ├── config_30m.yaml    # 30M 档（hidden 576 / 8 层）
        ├── config_100m.yaml   # 100M 档（hidden 768 / 12 层）
        ├── model.py           # SuperMiniGPT 架构（与 03 完全一致，只放大规模）
        ├── data.py            # char tokenizer + 小/大语料切换
        ├── schedulers.py      # lr 调度（warmup+cosine）+ 参数量/显存估算
        ├── train.py           # 训练程序：AMP/accum/schedule/checkpoint/resume
        ├── expected_results.md
        ├── tests/             # correctness 测试（10 项，CPU 可跑）
        └── data/              # 全部语料数据（读者无需重新下载）
            ├── corpus_small.txt   # 小语料：西游记 723,296 字符
            ├── corpus_large.txt   # 大语料：四大名著 3,091,487 字符
            ├── books/             # 四本书的单书合并文本
            ├── chapters/          # 每回单独文件（4 本书共 460 回/卷）
            ├── raw/               # 每回原始 wikitext（清洗可追溯）
            ├── chapters.json      # 结构化索引
            └── corpus_stats.json  # 语料统计
```

## 演示 WebUI（零依赖）

切换不同模型，输入同一个提示词，并排看续写结果：

```bash
cd ~/llm-training-lab
./.venv/bin/python scripts/serve_demo.py
# 浏览器打开 http://localhost:7860
```

只用 Python 标准库（`http.server`），不装 Gradio/Flask。默认加载四档模型
（10M×小语料 / 10M×大语料 / 30M×大语料 / 100M×大语料），显存有限时
按需加载、只缓存当前一个。

## 03 → 04：玩具代码补了哪些工程件

| 工程件 | 03 玩具循环 | 04 真正的训练程序 |
|---|---|---|
| 学习率 | 常数 | warmup + cosine 退火 |
| batch | 单步 batch | gradient accumulation（小显存补大 batch）|
| 精度 | fp32 | AMP 混合精度（省显存 + 提速）|
| 梯度 | clip_grad_norm | 同上，但放大后更关键 |
| checkpoint | 只存权重 | 含优化器/步数/best_val，**可续训** |
| resume | 不支持 | 从任意 checkpoint 继续 |
| 吞吐统计 | 无 | step time / tok/s / 显存峰值全采集 |
| 显存预估 | 无 | `estimate_memory.py` 先算账再跑 |

## 语料库：小语料 vs 大语料

| 档位 | 内容 | 字符数 | 词表 |
|---|---|---|---|
| small | 西游记 | 723,296 | 4,532 |
| large | 四大名著（西游+红楼+三国+水浒） | 3,091,487 | 6,015 |

**为什么选四大名著**：四部都是明清白话章回小说，与 03 篇西游记同源，风格统一；
小语料是大语料的真子集，对照时只变"数据量"一个变量，归因干净。

**为什么不用四书五经**：实测四书五经合计仅约 40 万字符（比 03 单书还少），
且是文言，与 03 白话风格不统一。

## 快速使用（WSL2 内）

```bash
# 前置：02 篇环境（~/llm-training-lab + .venv + torch cuda）

# 1. 同步工程
bash scripts/sync_project.sh

# 2. 构建语料库（约 10 分钟，含维基文库限流退避）
./.venv/bin/python exp_scale/build_corpus.py --out-dir exp_scale/data

# 3. 显存账（不跑训练，秒级）
./.venv/bin/python scripts/estimate_memory.py

# 4. 快速验证（100 步，约 1 分钟）
bash scripts/collect_data.sh --quick

# 5. 主实验采集（约 90 分钟）
bash scripts/collect_data.sh

# 6. 补齐 2×2 对照矩阵缺失格（约 40 分钟）
bash scripts/collect_fill_gaps.sh

# 7. resume 验证
bash scripts/verify_resume.sh

# 8. 采集演示数据 + 启动 WebUI
./.venv/bin/python scripts/collect_samples.py
./.venv/bin/python scripts/serve_demo.py
```

## 硬件三档

| 档位 | 硬件 | 能做什么 |
|---|---|---|
| 最低 | 8GB 级 CUDA GPU | 10M~100M 训练（本篇实测：100M 峰值 2.5GB）|
| 推荐 | 16~24GB 级 | 100M 更大 batch / 更长序列，无需 gradient accumulation |
| 增强 | 40GB+ | 更大模型（300M+）或多卡 |
| 无 GPU | CPU | 代码正确性验证（`hardware.device=cpu`，steps 降到 100）|

## 实验设计（每个实验只变一个变量）

### 2×2 对照矩阵（核心）

| 格 | 模型 | 语料 | 证明什么 |
|---|---|---|---|
| A | 12.36M | 小 72.3万 | baseline |
| B | 12.36M | **大 309.1万** | **纯数据效应**（模型不变）|
| C | 89.57M | 小 72.3万 | 反例：模型放大但数据没跟上 |
| D | 89.57M | 大 309.1万 | 最终目标 |

### 工程件消融

| 实验 | 模型 | 语料 | 证明什么 |
|---|---|---|---|
| no_amp | 100M | 大 | AMP 的速度与显存收益 |
| no_sched | 30M | 大 | schedule 对收敛的作用 |
| no_accum | 30M | 大 | 等效 batch 对 loss 的影响 |

## 设计要点

- **每个工程件都是"为什么需要 + 怎么实现 + 验证"**：不是 API 罗列，而是放大模型后
  真实出现的需求（显存不够→AMP，batch 太小→accumulation，损失震荡→schedule）。
- **失败开关内置**：`--mode no_amp/no_sched/no_accum` 关掉单个工程件，不用改代码。
- **先算账再跑**：`estimate_memory.py` 把显存逐项拆开，让"为什么 OOM"可计算。
- **可续训**：checkpoint 含优化器状态与步数，resume 后曲线连续（03 做不到）。
- **复用 03 架构**：模型代码零改动，只改 config 里的 hidden/layers/heads——
  "同一套代码，改配置就能放大"的直接体现。

## 真实失败案例（本篇实测）

1. **gradient accumulation 的 loss 记账坑**：`accum=4` 时日志按 `log_every` 归一化，
   loss 虚高 4 倍（实测 34.6 vs 真实 7.98）。只在启用 accumulation 时出现，`accum=1` 不暴露。
2. **初始 loss 诊断**：放大后若初始 loss 远超 ln(vocab)（大语料理论值 8.70），
   说明 logits 尺度异常，`diagnose_init.py` 可直接定位。
3. **`nohup` 不脱离会话**：WSL 会话结束会带走后台进程，必须 `setsid nohup`。