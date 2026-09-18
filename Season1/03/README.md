# DevResources/Season1/03 —— SuperMiniGPT v0 资源包

对应文章：`Articles/Season1/03_第一个实验：从零训练一个SuperMiniGPT.md`

## 包内容

```text
03/
├── README.md                  # 本文件
├── scripts/
│   ├── build_corpus.py        # 西游记语料库构建流水线（抓取→清洗→繁转简→切分→统计）
│   ├── retrieval_qa.py        # BM25 检索问答（纯 Python，中文 bigram，无分词依赖）
│   ├── qa_set.json            # 27 题事实问答基准（expected_chapter 经语料反查验证）
│   ├── smoke_model.sh         # 模型冒烟：forward/backward/generate 三连验证
│   ├── quick_train.sh         # 200 步快速训练（验证训练循环与语料加载）
│   ├── train_xiyouji.sh       # 西游记语料训练 + Shakespeare 对照（各 3000 步）
│   ├── collect_data.sh        # 英文消融四组实验（main/no_mask/no_rope/overscale）
│   ├── gen_qa.sh              # 生成模型答事实问题实测（加载 checkpoint 采样）
│   ├── nan_hunt.sh            # NaN 狩猎实验（lr 上限探索，无梯度裁剪）
│   ├── sync_project.sh        # Windows 侧工程同步到 WSL 训练目录
│   └── sync_and_smoke.sh      # 同步 + 冒烟验证一体脚本
└── project/
    ├── common/                # 与 02 篇完全同一套骨架（seed/config/logging/benchmark/metrics）
    └── exp_superminigpt/
        ├── config.yaml        # Shakespeare 语料配置
        ├── config_xiyouji.yaml# 西游记语料配置（只改 corpus 与语料相关项）
        ├── data.py            # char tokenizer（OOV 容错）+ 滑动窗口 + 双语料加载
        ├── model.py           # SuperMiniGPT：手写 attention/RoPE/RMSNorm/SwiGLU/residual
        ├── train.py           # 训练入口：6 种 mode + checkpoint 保存
        └── data/              # 全部语料数据（读者无需重新下载即可复现）
            ├── xiyouji.txt        # 西游记百回合并纯文本（723,296 字符，训练用）
            ├── chapters/          # 每回一个文件（001.txt~100.txt，检索单元）
            ├── chapters.json      # 结构化索引：回数/回目/字数
            ├── corpus_stats.json  # 语料统计
            ├── xiyouji_raw/       # 每回原始 wikitext（清洗可追溯）
            └── tiny_shakespeare.txt # 英文对照语料（Karpathy char-rnn 公开数据）
```

## 快速使用（WSL2 内）

```bash
# 前置：02 篇环境（~/llm-training-lab + .venv + torch cuda）
# 把 project/ 内容拷入 ~/llm-training-lab/ 后：

# 1. 模型冒烟（10 秒）
bash scripts/smoke_model.sh

# 2. 西游记训练 + 英文对照（约 4 分钟）
bash scripts/train_xiyouji.sh

# 3. 生成模型答事实问题（看它怎么"编"）
bash scripts/gen_qa.sh

# 4. BM25 检索问答基准（27 题，hit@3）
python scripts/retrieval_qa.py --chapters-dir exp_superminigpt/data/chapters \
    --benchmark scripts/qa_set.json

# 5. 英文消融四组（约 7 分钟）
bash scripts/collect_data.sh

# 从零重建语料库（可选，data/ 已含全部数据；约 2 分钟，含限流退避）
python scripts/build_corpus.py --out-dir exp_superminigpt/data
```

## 消融开关说明

`exp_superminigpt/train.py --mode <mode>`，每个 mode 对应模型里的一个消融开关：

| mode | 关掉的东西 | 实测现象（数据见 results/Season1/03/） |
|---|---|---|
| main | 无（完整模型） | 英文 val 1.2769 采样可读；中文 val 4.5708(best) 后过拟合回升 |
| no_mask | 因果掩码 | val 0.0172 虚低 74 倍，生成乱码（信息泄露） |
| no_rope | 位置编码 | val 1.3352，拼写明显劣化 |
| no_residual | 残差连接 | val 3.3485 纹丝不动，训练失效 |
| overscale | 无（lr×10） | 收敛正常（稳定性反直觉，见文章讨论） |

## 设计要点

- **每个组件手写**：不用 nn.Transformer，attention/RoPE/RMSNorm/SwiGLU 全部自己写，
  每个约 20~40 行，注释写清"解决什么、没有它会怎样"。
- **消融开关内置**：失败实验用开关不用改代码（对应 experiment-rules 的 fail_mode 原则）。
- **双语料同框架**：config 只改 `experiment.corpus` 一项即可切换中英文语料，
  模型代码零改动——"数据不同、框架相同"的直接体现。
- **语料全流程可复现**：原始 wikitext 落盘（xiyouji_raw/），清洗可追溯；
  build_corpus.py 幂等，已下载的回不重复请求。
- **复用 02 篇骨架**：common/ 原样复用，config 分离、seed 管理、留痕在训练里全部生效。
- **验证集是标尺**：90/10 切分，val loss 是判断"在学还是在背"的唯一客观依据；
  中文语料的过拟合拐点（约 1400 步）是它存在的直接证据。
