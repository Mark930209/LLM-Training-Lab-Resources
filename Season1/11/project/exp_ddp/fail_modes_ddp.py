"""fail_modes_ddp.py —— 11 篇故障注入的文档与预期偏差。

每个故障都遵守 10 篇的教训：注入必须自证生效（ddp_train 把实际效果量写进
结果文件的 inject 字段），且只动一个变量。

故障清单（与 ddp_train.ALL_FAULTS 对应）：

1. no_sampler —— 不用 DistributedSampler，各 rank 独立 shuffle 全池
   现象：程序正常跑，loss 也在降，但每个样本每 epoch 被训练 world 次，
   等效全局 batch 里全是重复样本；与单卡基线的 loss 轨迹逐步偏离。
   根因：DataLoader 默认 shuffle 不知道分布式环境，两个 rank 各自遍历全池。
   自证：inject.sample_duplication_factor = world；sampler_audit 的
   no_sampler 节给出重复抽取数与最大单样本命中次数。

2. global_batch_misconfig —— 把"全局 batch"当成"每卡 batch"
   现象：每卡 batch=B，实际全局 batch=world×B。与单卡 B 基线对不齐：
   梯度噪声更小、等效学习率相对更大，loss 轨迹系统性偏离。
   根因：DDP 的语义是"每 rank 一份数据"，全局 batch = per_rank × world。
   自证：inject.effective_global_batch = B × world。

3. sum_reduction —— loss 用 sum 且不除 world_size
   现象：每 rank 梯度放大 (B/2) 倍，all-reduce mean 后仍比单卡大 (B/2) 倍，
   等效学习率爆炸，loss 快速发散或跑飞（clip 只能挡爆炸不能根治）。
   根因：等价数学要求 mean reduction；sum 破坏了"梯度 = 单卡全 batch 梯度"。
   自证：final_grad_norm 与 loss 轨迹相对基线的偏离量级。

4. no_set_epoch —— 多 epoch 训练从不调 sampler.set_epoch
   现象：每个 epoch 的 shuffle 顺序完全相同，模型反复按同一顺序过数据。
   短程 loss 看不出异常，多 epoch 长训练才显形（顺序固化影响收敛）。
   根因：DistributedSampler 的 shuffle 种子 = seed + epoch，不调 set_epoch
   则 epoch 恒为初始值。
   自证：sampler_audit 的 set_epoch 节——without 组所有 epoch 顺序全同。

5. all_rank_save —— 所有 rank 同时 torch.save 到同一路径
   实测（诚实修正）：连跑 6 次，文件全部可加载、无损坏。根因是 DDP 让各
   rank 状态完全相同，两个 rank 写的是相同字节到相同偏移，本地文件系统上
   竞态恰好无害。"测试时没出问题"正是它危险的原因。
   真实代价：① 冗余 I/O = world_size 倍（2 rank 各写 148 MB，共 296 MB，
   正确做法只需 148 MB）；② 潜在竞态——共享/网络文件系统（NFS）上并发写
   同一路径会真损坏，一旦保存 rank 相关状态（sampler/RNG）则 last-writer-wins
   会静默存下某一个 rank 的状态；③ 没有 barrier，resume 时可能有 rank 在
   文件写完前就去读。
   自证：inject.ckpt_writers = world（正确做法为 1）；ckpt.loadable 自检。
   正确做法：rank0 保存 + dist.barrier() + 全 rank 加载。

预期偏差速查表（parity 对照用）：

| 故障 | init_checksum | loss 轨迹 | final 参数 | 显性报错 |
|---|---|---|---|---|
| none | 一致 | 对齐（容差内） | 对齐 | 无 |
| no_sampler | 一致 | 逐步偏离 | 偏离 | 无（静默） |
| global_batch_misconfig | 一致 | 系统性偏离 | 偏离 | 无（静默） |
| sum_reduction | 一致 | 立刻发散/跑飞 | 大幅偏离 | 可能 NaN |
| no_set_epoch | 一致 | 短程难分辨 | 短程难分辨 | 无（静默） |
| all_rank_save | 一致 | 对齐 | 对齐 | 受控测试无损坏（见说明） |

五个故障里四个是静默的：程序不报错、loss 也在降，只有与单卡基线逐级
对齐才能暴露。all_rank_save 更隐蔽：受控测试里连文件都不损坏（各 rank
写相同字节），代价是 world_size 倍冗余 I/O 与共享文件系统上的潜在竞态。
这就是本篇核心判断的实验基础——"能跑起来"和"跑对了"之间隔着静默陷阱，
而"测试时没出问题"恰恰是某些陷阱最危险的伪装。
"""

from __future__ import annotations

FAULT_TABLE = {
    "no_sampler": {
        "what": "不用 DistributedSampler，各 rank 独立 shuffle 全池",
        "symptom": "样本重复训练 world 次，loss 轨迹逐步偏离单卡基线",
        "silent": True,
        "evidence": "inject.sample_duplication_factor / sampler_audit.no_sampler",
    },
    "global_batch_misconfig": {
        "what": "把全局 batch 当成每卡 batch，实际全局 = world × B",
        "symptom": "等效 batch 翻倍，与单卡 B 基线系统性偏离",
        "silent": True,
        "evidence": "inject.effective_global_batch",
    },
    "sum_reduction": {
        "what": "loss 用 sum reduction 且不除 world_size",
        "symptom": "梯度放大 (B/2) 倍，等效学习率爆炸，loss 发散",
        "silent": False,
        "evidence": "final_grad_norm / loss 轨迹",
    },
    "no_set_epoch": {
        "what": "多 epoch 从不调 sampler.set_epoch",
        "symptom": "每个 epoch 同一 shuffle 顺序，长训练才显形",
        "silent": True,
        "evidence": "sampler_audit.set_epoch.without_set_epoch_all_same",
    },
    "all_rank_save": {
        "what": "所有 rank 同时写同一 checkpoint 路径",
        "symptom": ("受控测试无损坏（各 rank 写相同字节）；真实代价是 "
                    "world_size 倍冗余 I/O + 共享文件系统上的潜在竞态"),
        "silent": True,
        "evidence": "inject.ckpt_writers=world / ckpt.loadable 自检",
    },
}


def describe_fault(name: str) -> dict:
    """按名字取故障说明。文章与报告用。"""
    if name not in FAULT_TABLE:
        raise KeyError(f"未知故障: {name}（可选 {list(FAULT_TABLE)}）")
    return FAULT_TABLE[name]
