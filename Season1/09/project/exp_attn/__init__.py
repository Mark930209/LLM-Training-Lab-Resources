"""exp_attn —— 09 篇 Attention Kernel Lab。

naive_attn      物化 N×N 分数矩阵的参考实现（基线，用来对照显存与正确性）
backend_probe   判断 SDPA 实际走了哪个 backend（可用性探测 + profiler 内核名识别）
correctness     naive vs SDPA 各 backend 的输出与梯度数值一致性
attn_bench      backend / seq / head_dim / dtype sweep，测时间、峰值、实际 backend
e2e_step        把 microbenchmark 收益放回完整训练 step 复验
"""
