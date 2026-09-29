# DevResources/Season4/17 —— 数据配方消融实验室

对应文章：`Articles/Season4/17_训练数据怎样配比：去重强度、质量过滤与领域比例的消融实验.md`

## 本篇解决什么问题

16 篇产出了能清洗语料的管线，但"清洗"本身有很多档：去重做多狠、质量过滤
开不开、两个领域按什么比例混。这些配方选择对模型的影响，凭直觉说不清，
要用固定算力下的对照实验量出来。本篇在 CPU 上完成数据处理、在 8 GB 级
单卡上用 10M 小模型做固定 token 预算的对照训练，用两域留出困惑度与多样性
指标判断配方好坏。

核心判断：数据处理的每一步都在同时改变信息密度和多样性，两者最优点不重合。
去重和过滤提高密度，做过头会削掉长尾；配比决定模型的能力分布，不是各
领域效果的简单叠加。

## 包内容

```text
17/
├── README.md                  # 本文件
└── project/
    ├── exp_recipe/
    │   ├── __init__.py
    │   ├── config.yaml        # 配方矩阵 + 训练超参 + 语料路径
    │   ├── corpus.py          # 两域语料加载、评测集切分、重复注入
    │   ├── recipe.py          # 配方制备：过滤→去重→配比（复用 16 篇 exp_data）
    │   ├── train_eval.py      # 固定 token 预算训练 + 两域留出困惑度
    │   ├── pipeline.py        # 统一入口：prepare / matrix
    │   └── expected_results.md
    └── tests/
        └── test_recipe.py     # 离线单测（不联网、不训练）
```

## 跨篇模块复用

本篇不重写去重与质量过滤，而是把 16 篇的 `exp_data` 当库用（同一套代码 =
同一个口径）。运行时 `PYTHONPATH` 需同时含三个工程目录：

- 17 篇 `exp_recipe`（本篇）
- 16 篇 `exp_data`（去重、质量过滤）
- 04 篇 `exp_scale`（SuperMiniGPT 模型、CharTokenizer、LMDataset）

## 运行（本机 WSL，RTX 3070 节点）

```bash
cd ~/llm-training-lab
P17="/mnt/d/.../DevResources/Season4/17/project"
P16="/mnt/d/.../DevResources/Season4/16/project"
export PYTHONPATH="$P17:$P16:$HOME/llm-training-lab"
PY="$HOME/llm-training-lab/.venv/bin/python"
CFG="$P17/exp_recipe/config.yaml"

# 只制备配方（不训练），检查留存率/配比/多样性
"$PY" -m exp_recipe.pipeline --mode prepare --config "$CFG" --out-dir /tmp/recipe_prepare

# 制备 + 训练 + 两域评测全部配方（主实验，约 10 分钟）
"$PY" -m exp_recipe.pipeline --mode matrix --config "$CFG" --out-dir /tmp/recipe_matrix
```

## 公平对照的三条口径

1. **tokenizer 固定**：用两域训练池并集建一次词表，所有配方共用，否则
   词表不同、困惑度不可比。
2. **token 预算固定**：每个配方训练相同 token 数（约 488 步），比的是
   "同样喂这么多 token，哪种配方学得好"。
3. **评测集固定且不进配方**：nl_eval / code_eval 永不参与训练与重复注入，
   指纹（SHA-256）写进每份结果。

## 真实性标签

- 配方制备、训练、两域困惑度：`REAL`（本机 RTX 3070 + CPU）
- 重复注入是合成的（本地语料无天然近重复），注入比例写入制备报告，如实标注
- 大语料上的配方效果外推：`SCALED`（本篇不做，留给后续）

## 依赖

复用 16 篇已装的 `datasketch`、`pyarrow` 等；训练用 04 篇环境的 PyTorch
2.14.0+cu130。无新增依赖。
