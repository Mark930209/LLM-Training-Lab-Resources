"""exp_shard —— 23 篇 State Sharding Lab。

ZeRO-1/2/3 与 FSDP 的分片选择、step 内参数生命周期、wrap/reshard/prefetch
sweep、sharded checkpoint。容量结论只在两机各 rank 独占 GPU 时标 REAL。
"""
