"""exp_perf —— 10 篇 Single GPU Performance Lab。

perf_harness   阶段打点、step time 分布、GPU 空洞测量（替代只报平均值的 StepTimer）
perf_bench     端到端训练 + 故障注入 + 优化开关，产出时间线与瀑布数据
timeline       用 profiler 画 CPU/H2D/forward/backward/optimizer/periodic 时间线
"""
