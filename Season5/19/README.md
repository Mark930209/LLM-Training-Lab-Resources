# DevResources/Season5/19 —— Modern Architecture Lab：组件化架构消融

对应文章：`Articles/Season5/19_现代大模型换掉了哪些结构：归一化、激活函数与注意力头的取舍.md`

## 本篇解决什么问题

03 篇的 SuperMiniGPT 已经内置了 RMSNorm、SwiGLU、RoPE（各带开关），当前主流
开源模型也是这套骨架。但这些组件各自替换掉了什么经典实现、单项收益有多大、
叠加之后是否还成立，"大家都这么写"不等于"每项都被验证过"。本篇把五个架构轴
做成可独立切换的组件化模型，在等参数量与等算力两套控制下逐项消融。

核心判断：这些组件中真正稳定带来质量提升的少，多数收益来自工程量的重新分配
（显存、吞吐或数值稳定性）；单项消融的收益不可简单相加；任何不控制参数量或
算力的对照，都会把实现差异误读成架构收益。

叙事校准：不是"从经典升级到现代"，而是"把现代骨架逐项关掉/换回经典档做对照"。

## 五个消融轴

| 轴 | 现代档（基准） | 经典档 | 参数量影响 |
|---|---|---|---|
| norm_type | RMSNorm（无 bias） | LayerNorm（带 bias） | 每处 norm 多 hidden 参数 |
| norm_pos | pre-norm | post-norm（GPT-2） | 无 |
| ffn_type | SwiGLU 8/3x 三矩阵 | GELU 4x 两矩阵（带 bias） | 近似持平 |
| pos_enc | RoPE（无参数） | learned 绝对位置 | 多 seq_len×hidden |
| attn_type | MHA | GQA（kv=heads/2）/ MQA（kv=1） | k/v 投影按比例缩小 |

## 包内容

```text
19/
├── README.md                  # 本文件
└── project/
    ├── exp_arch/
    │   ├── __init__.py
    │   ├── model_arch.py      # 组件化 ArchGPT：五轴全可配，全部自己写
    │   ├── arch_metrics.py    # 参数量/KV cache/每步 FLOPs 解析式 + 等参数量反解
    │   ├── train_arch.py      # 固定预算训练（与 17 篇同口径）+ 两域评测
    │   ├── pipeline.py        # plan（只算不训）/ train（逐格训练 + stress）
    │   ├── config.yaml        # 9 格消融矩阵 + 等算力缩放 + stress 大 LR 对照
    │   └── expected_results.md
    └── tests/
        └── test_arch.py       # 8 项离线单测（96 种组件组合 forward 等）
```

## 跨篇模块复用

语料切分、固定词表（char 级 6120）、训练配方（nl50+mh08+q）、训练循环口径
全部复用 17 篇 `exp_recipe`；模型基座结构对齐 04 篇 SuperMiniGPT。运行时
`PYTHONPATH` 需含 19/17/16/04 四个 project 目录。

## 两套控制口径

- **等参数量**：以 modern 格（hidden 384）的 12,971,904 为目标，其余格用
  `solve_hidden_for_params` 二分反解 hidden（只取 2×heads 倍数，保证 head_dim
  为偶数——RoPE 奇偶配对要求），参数量落在目标以下 1.1%~5.6%（对齐约束的
  固有代价，如实报告）。
- **等算力**：总 FLOPs = modern 格 step_flops × token_budget，每格按
  step_flops 反比缩放步数（391~516 步），容差 2%。

## 运行（本机 WSL，RTX 3070）

```bash
P19="/mnt/d/.../DevResources/Season5/19/project"
P17="/mnt/d/.../DevResources/Season4/17/project"
P16="/mnt/d/.../DevResources/Season4/16/project"
P04="/mnt/d/.../DevResources/Season1/04/project"
export PYTHONPATH="$P19:$P17:$P16:$P04"
PY="$HOME/llm-training-lab/.venv/bin/python"
CFG="$P19/exp_arch/config.yaml"

# 计划模式：只算不训（hidden 反解、参数分解、KV cache、FLOPs、等算力步数）
"$PY" -m exp_arch.pipeline --mode plan --config "$CFG" \
  --out /mnt/d/.../results/Season5/19/arch_plan_20260925.json

# 训练模式：9 格等参数量训练 + stress 大 LR 对照（约 10~15 分钟）
"$PY" -m exp_arch.pipeline --mode train --config "$CFG" \
  --out /mnt/d/.../results/Season5/19/arch_train_20260925.json
```

结果 JSON 一律 `--out` 直写 `/mnt/d` 挂载路径（18 篇教训：WSL /tmp 不可靠）。

## 真实性标签

- 消融矩阵训练、stress 对照、KV cache 与 FLOPs 解析式：`REAL`（本机 RTX 3070，解析式经单测与实建模型逐格对账）
- 规模放大后各组件收益的变化：`SCALED`（小规模实测外推，文章标注）
- 主流开源模型的组件组合（Llama/Qwen/Gemma 等）：`REFERENCE`

## 已知实现细节

1. **GQA 的 kv 复制用 `repeat_interleave`**：heads 必须能被 kv_heads 整除；
   expand 视图在 attention 矩阵乘里同样有效，但 repeat 语义更直白，小模型上
   显存差异可忽略。
2. **post-norm 用同一套初始化**：没有为 post-norm 单独调初始化/warmup，
   发散与否本身就是实验结果（提纲要求的稳定性对照）。
3. **等参数量反解只取 2×heads 倍数**：hidden 必须被 heads 整除（注意力分头），
   且 head_dim 必须为偶数（RoPE 把 head_dim 按奇偶维配对旋转，奇数 head_dim
   直接形状不匹配崩溃；实测踩坑：hidden=378/heads=6 → head_dim=63，layernorm
   格 forward 崩溃，修正后反解到 372）；因此各格参数量落在目标以下
   1.1%~5.6%，不是精确相等。model_arch.Attention 已加断言拦截奇数 head_dim。
4. **stress 发散判据**：final_train_loss 为 NaN / >20 或评测 ppl 为空。
   判据写在 pipeline 里，不靠人工看曲线。
