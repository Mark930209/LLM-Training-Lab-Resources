# DevResources/Season1/06 —— Hugging Face 接口契约

对应文章：`Articles/Season1/06_把训练程序接进 Hugging Face 生态：五条接口契约.md`

## 本篇解决什么问题

前几篇的训练程序自成一套，与外面的生态不通用。本篇把它改造成认 Hugging Face 那套通用约定的样子，改完能与生态互操作。

**先澄清一个命名误会**：本篇的 HF 模型是从零训练的（`LlamaForCausalLM(cfg)`，config 是唯一输入，随机初始化，不下载权重）。`PreTrainedModel`、`save_pretrained` 里的 "pretrained" 指的是**接口与格式标准**，与权重是否来自预训练无关。语料仍是四大名著，初始 loss 9.082 ≈ ln(8192) 的随机理论值就是证据。

核心判断：改造的代价取决于训练循环对模型的依赖有多少条，把这些依赖一条条列出来就是"契约"。本篇列出五条。

## 包内容

```text
06/
├── README.md                  # 本文件
└── project/
    └── exp_hf/
        ├── contract.py         # 五条契约的实现 + contract_loss（全系列唯一 loss 口径）
        ├── adapters.py         # SuperMiniGPTAdapter / SuperMiniGPTConfig / build_llama / sidecar 存取
        ├── tokenize_bpe.py     # 四大名著训 BPE 8192 + char/BPE 对照
        ├── train_hf.py         # 通用入口：--model / --tokenizer / --save-hf / --resume-hf / --fault
        ├── parity_test.py      # 改造前后等价性验证
        ├── harness_on_hf.py    # 把 05 篇的正确性门禁接到 HF 路径上
        ├── fail_modes_hf.py    # 三个接口层故障
        └── tokenizer_bpe.json  # 训练好的 BPE 分词器（8192 词表）
```

`contract_loss` 是全系列后续所有篇的 loss 口径来源（07–11 篇都 import 它），保证"换模型不会偷偷换掉 loss 的定义"。

## 前置：本篇包依赖前序篇的模块

`exp_hf` 复用 04 篇的模型与数据、05 篇的正确性门禁，并依赖 02 篇骨架。本篇包只含**新增**的 `exp_hf/`，运行前需累积：

| 需要的模块 | 来自 | 用途 |
|---|---|---|
| `common/` | 02 篇包 | seed / config / 留痕 / benchmark / 指标口径 |
| `exp_scale/` | 04 篇包 | 模型、char tokenizer、语料、lr 调度 |
| `exp_debug/` | 05 篇包 | `harness_on_hf` 复用的正确性门禁 |
| `exp_hf/` | 本篇包 | HF 接口契约 + build_llama + BPE |

```bash
cp -r DevResources/Season1/06/project/exp_hf ~/llm-training-lab/
cd ~/llm-training-lab
```

依赖库：本篇首次引入 `transformers` 与 `tokenizers`（用 uv 装，venv 里没有 pip）：

```bash
uv pip install --python ./.venv/bin/python transformers tokenizers
```

## 快速使用

```bash
cd ~/llm-training-lab

# 三种模型走同一个训练循环（验证契约成立）
./.venv/bin/python -m exp_hf.train_hf --model superminigpt --tokenizer char --steps 300 --output results/Season1/06/train_char_baseline.json
./.venv/bin/python -m exp_hf.train_hf --model superminigpt-hf --tokenizer char --steps 300 --output results/Season1/06/train_hf_adapter_char.json
./.venv/bin/python -m exp_hf.train_hf --model llama --tokenizer char --steps 300 --output results/Season1/06/train_llama_char.json
./.venv/bin/python -m exp_hf.train_hf --model llama --tokenizer bpe --steps 300 --output results/Season1/06/train_llama_bpe.json

# 改造前后等价性（四个 0.0）
./.venv/bin/python -m exp_hf.parity_test --output results/Season1/06/parity.json

# 存/读 HF 标准格式 + 续训
./.venv/bin/python -m exp_hf.train_hf --model llama --tokenizer bpe --steps 150 --save-hf runs/hf_ckpt --output results/Season1/06/save_hf_150.json
./.venv/bin/python -m exp_hf.train_hf --model llama --tokenizer bpe --steps 300 --resume-hf runs/hf_ckpt --output results/Season1/06/resume_hf_full.json

# 05 篇门禁接到 HF 路径
./.venv/bin/python -m exp_hf.harness_on_hf --output results/Season1/06/harness_on_hf.json

# 接口层故障
./.venv/bin/python -m exp_hf.train_hf --model llama --tokenizer bpe --fault double_shift --output results/Season1/06/fault_double_shift.json
```

`--model` 可选 `superminigpt`（03 篇原始实现，参考基准）/ `superminigpt-hf`（套 HF 外壳）/ `llama`（HF 标准架构）。`--tokenizer` 可选 `char` / `bpe`。

## 结果文件

`results/Season1/06/`（13 个）：

| 文件 | 内容 |
|---|---|
| `parity.json` | 改造前后等价性，四项误差全 0.0 |
| `train_char_baseline.json` / `train_hf_adapter_char.json` | 逐位一致，val 5.3205 |
| `train_llama_char.json` / `train_llama_bpe.json` | val 5.3119 / 7.2501 |
| `save_hf_150.json` / `resume_hf_full.json` | 续训 val 7.2677 对直训 7.2501 |
| `llama_roundtrip.json` | 存取往返误差 0.0 |
| `harness_on_hf.json` | 门禁五阶段全过 |
| `tokenizer_compare.json` | char/BPE 对照（压缩比 0.988→0.675） |
| `fault_*.json` | 三个接口层故障（double_shift / vocab_mismatch / resume_weights_only） |

## 调试踩出的三个真实的坑

1. **tied weights 不声明就报错**：`tie_word_embeddings=True` 时 `save_pretrained` 需要显式声明，否则直接抛错。
2. **RoPE 的 cos/sin 是 non-persistent buffer**：HF 的 meta 实例化会丢掉它，导致前向 NaN。改为 persistent 并随 safetensors 存取。
3. **AMP 下先 clip 后 unscale 让 grad_norm 读成放大值**：实测读出 258726，真实约 3.95。修法是 clip 前补 `scaler.unscale_`。这个 bug 同时影响 04/05 篇的口径，在本篇修正。

第三个坑说明：**指标口径错误不是性能问题，是正确性问题**。门禁阈值只有在正确口径下才有意义，而这类 bug 只在挂了门禁之后才暴露。05 篇的产物在 06 篇第一次真正发挥了拦截作用。

## 已知边界

- BPE 词表 8192 是在四大名著上训的，不是通用词表。压缩比对照只在这个语料内有效。
- `--fault vocab_mismatch` 的显性方向（BPE id 喂 char 模型）在 CPU 上触发：CUDA 的 gather 越界是 device-side assert，异常虽可捕获但会污染整个 CUDA 上下文，后续实验全废。
- 本篇不涉及加载预训练权重微调，那是 Season 6 的事。
