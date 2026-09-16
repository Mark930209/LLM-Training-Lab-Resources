"""common —— LLM Training Lab 可复现实验骨架。

四个模块对应可复现的四件具体事：
    reproducibility  seed 管理（同 seed 跑两次结果一样）
    config           配置与代码分离（换硬件只改配置）
    logging          每次运行留痕（旧记录可对账）
    benchmark        量化采集（step time / 显存 / 吞吐）

设计目标不只是"同机复现"，还包括"跨机迁移"：换到更大显存的卡或云 GPU 时，
代码不变、config 只改硬件相关项、lock 保证依赖一致、留痕保证历史记录可对照。
"""

__all__ = ["reproducibility", "config", "logging", "benchmark", "metrics"]
