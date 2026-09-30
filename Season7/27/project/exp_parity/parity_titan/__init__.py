"""parity_titan —— 27 篇 TorchTitan 迁移腿（自定义模块包）。

作为 torchtitan 的扩展模块使用：
    python -m torchtitan.train --module parity_titan --config parity_tiny

迁移要点（对应文章"隐性语义"清单）：
- 数据管道：FixedDataLoader 喂与基线逐字节相同的批次（不信 seed，读落盘文件）；
- 初始化：PARITY_INIT_FILE 设置时冻结权重注入（monkeypatch init_states），
  不设置则走框架原生初始化 = 失败案例 1 的病灶腿；
- 优化器/调度：implementation="for-loop"（fused kernel 会引入单 ulp 差），
  LRSchedulersContainer(warmup_steps=20) 与基线同类同参；
- loss 口径：trainer 原生 CE(sum)/global_valid_tokens，与基线一致。
"""
