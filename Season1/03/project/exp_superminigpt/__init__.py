"""exp_superminigpt —— 第一个实验：从零训练一个 SuperMiniGPT（03 篇实验包）。

模块:
    data   char-level tokenizer + Tiny Shakespeare 语料 + 滑动窗口数据集
    model  SuperMiniGPT：手写 attention / RoPE / RMSNorm / SwiGLU / residual
    train  训练入口：main / no_mask / no_rope / overscale 四组实验
"""
