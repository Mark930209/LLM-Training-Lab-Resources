# DevResources/Season4/18 —— Tokenizer Lab：词表大小 sweep

对应文章：`Articles/Season4/18_怎样训练自己的 Tokenizer：词表大小对显存、速度和效果的影响.md`

## 本篇解决什么问题

03 篇以来一直在用现成的 char 级 tokenizer，而词表大小、切分算法与数字/代码
的处理方式直接决定了同样 token 预算下模型见到多少文本、embedding 吃掉多少
参数、某类任务能不能做对。本篇自己训 tokenizer，并用可量化的指标给词表设计
定出一个有依据的区间。

核心判断：词表大小同时抬高压缩率和 embedding 参数占比，两者收益方向相反。
固定参数预算下存在一个区间，越过之后新增的词表容量主要落在长尾 token 上，
而这些 token 在训练语料中出现次数过少，等于把参数花在几乎没被训练的位置。

## 包内容

```text
18/
├── README.md                  # 本文件
└── project/
    ├── exp_tokenizer/
    │   ├── __init__.py
    │   ├── config.yaml        # sweep 配置（算法 × 词表档位 × 数字策略）
    │   ├── tok_train.py       # BPE（HF tokenizers）与 Unigram（SentencePiece）训练
    │   ├── tok_metrics.py     # 压缩率、覆盖率、长尾分布、embedding 参数占比
    │   ├── pipeline.py        # 统一入口：sweep（静态指标）与 train（训练侧验证）
    │   └── expected_results.md
    └── tests/
        └── test_tokenizer.py  # 离线单测（小语料、小词表）
```

## 跨篇模块复用

语料与评测集切分复用 17 篇 `exp_recipe.corpus`（同一把尺子：四大名著 +
Python 标准库两域，评测集永不参与 tokenizer 训练）。运行时 `PYTHONPATH`
需同时含 18/17/16 三个 project 目录与 04 篇训练工程。

## 运行（本机 WSL，CPU 训练 tokenizer）

```bash
cd ~/llm-training-lab
P18="/mnt/d/.../DevResources/Season4/18/project"
P17="/mnt/d/.../DevResources/Season4/17/project"
P16="/mnt/d/.../DevResources/Season4/16/project"
export PYTHONPATH="$P18:$P17:$P16:$HOME/llm-training-lab"
PY="$HOME/llm-training-lab/.venv/bin/python"
CFG="$P18/exp_tokenizer/config.yaml"

# 词表 sweep：BPE/Unigram × 4 档词表 × 2 种数字策略，算静态指标
# 结果一律直写 /mnt/d 挂载路径（WSL /tmp 跨重启不可靠，见踩坑 5）
"$PY" -m exp_tokenizer.pipeline --mode sweep --config "$CFG" \
  --out /mnt/d/.../results/Season4/18/tokenizer_sweep_20260924.json

# 训练侧验证：固定 token 预算与配方，只换词表（RTX 3070）
# 实测单组 38 s（BPE 8k）~ 1764 s（BPE 64k，吞吐塌方），全矩阵约 71 分钟
# PYTHONPATH 需另加 04 篇工程目录（SuperMiniGPT/LMDataset）
P04="/mnt/d/.../DevResources/Season1/04/project"
export PYTHONPATH="$P18:$P17:$P16:$P04:$HOME/llm-training-lab"
"$PY" -m exp_tokenizer.pipeline --mode train --config "$CFG" \
  --out /mnt/d/.../results/Season4/18/tokenizer_train_20260924.json
```

train 模式的两条控制口径：固定架构（hidden 384/6/6，与 17 篇同档，词表越大
总参数越多）与固定总参数（解析解不训练：把总参数锁在 17 篇 10M 档
12,971,904，反解每档词表可用的 hidden）。跨 tokenizer 比效果用 BPC
（bits per char = loss × tokens / chars / ln2），ppl 是每 token 口径不可横比。

## 依赖

`tokenizers` 0.23.2（BPE，Rust 实现）、`sentencepiece` 0.2.2（Unigram）。
均已装入 `~/llm-training-lab/.venv`（uv 管理，清华源）。

## 真实性标签

- 词表 sweep 的压缩率/覆盖率/长尾/embedding 占比：`REAL`（本机 CPU 训练）
- 训练侧验证（词表对显存/吞吐/困惑度的影响）：`REAL`（RTX 3070）
- 大模型上 embedding 占比与吞吐变化：`SCALED`（参数量公式 + 小规模实测外推）
- 主流开源模型词表配置：`REFERENCE`

## 已知实现细节（踩坑记录）

1. **tokenizers 0.23.2 的预切分**：没有 `pre_tokenizers.Regex`，GPT-2 风格
   正则预切分要用 `pre_tokenizers.Split(Regex(pattern), "isolated")`，
   `Regex` 在 `tokenizers` 顶层。decoder 用 `decoders.ByteLevel()`，不是
   `pre_tokenizers.ByteLevel()`。
2. **SentencePiece 往返解码**：必须用 `detokenize` 而不是 `decode`，后者
   不还原 `▁` 空白标记。
3. **SentencePiece 默认折叠空白**：normalizer 把换行与缩进折叠成单空格，
   对代码（Python 缩进是语义）有损；byte-level BPE 逐字节无损。覆盖率
   度量用 `difflib.SequenceMatcher` 做鲁棒对齐，逐字符位置对比会因空格
   错位把无损还原虚报成大量损失（冒烟实测覆盖率被压到 0.07）。
4. **Unigram 词表下限**：byte_fallback 开启时 vocab_size 必须 ≥ 唯一字符数
   + 256 byte pieces + 元符号，否则训练直接报错。
5. **WSL /tmp 不可作结果落盘点**：首轮 sweep 全 16 组跑完后 WSL 虚拟机自动
   停止，`/tmp/tok18_sweep.json` 与日志一并消失（/tmp 在 ext4 根分区上，
   丢失机制未完全查明，但事实是文件没了）。长任务的 `--out` 一律直写
   `/mnt/d` 挂载路径；词表产物（work_dir）丢失时 train 模式会自动重训。
