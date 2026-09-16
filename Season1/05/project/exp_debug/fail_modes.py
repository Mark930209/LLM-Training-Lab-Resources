"""fail_modes.py —— 05 篇故障注入框架：每个故障一个独立开关。

设计原则（承接 04 篇的 mode 归因法）：
    1. 故障不靠改代码制造，全部走 fail_mode 开关，保证可复现、可对照。
    2. 每个故障只破坏训练循环的一个阶段（数据 / 前向 / 反向 / 更新 / 恢复），
       归因干净：现象 → 最小观测 → 根因 → 修法 → 复验。
    3. 故障分两类：
       - loud（会报错或指标显式异常）：NaN、lr=0、标签 shape 错位
       - silent（loss 看似正常但训练已经错了）：标签偏移、重复 batch、
         train/val 泄漏、伪续训。silent 类是本篇的主角。

用法：
    from exp_debug.fail_modes import apply_fault, FAULT_STAGES
    fault = apply_fault("label_shift", x, y)   # 返回注入后的 (x, y)
"""

from __future__ import annotations

import torch

# 每个故障破坏的阶段与类别（诊断决策树的输入）
FAULT_STAGES = {
    # ---- 数据阶段 ----
    "label_shift":   {"stage": "data",   "loud": False, "desc": "标签偏移 k 位：x[t] 预测 y[t+k]，任务被换掉但 loss 照常下降"},
    "label_shuffle": {"stage": "data",   "loud": False, "desc": "标签在 batch 内随机打乱：x 与 y 彻底脱钩"},
    "dup_batch":     {"stage": "data",   "loud": False, "desc": "每个 batch 复制前一半样本：有效数据量减半，loss 虚低"},
    "val_leak":      {"stage": "data",   "loud": False, "desc": "val 集混入 train 数据：val loss 失去意义"},
    # ---- 优化阶段 ----
    "lr_zero":       {"stage": "update", "loud": False, "desc": "学习率恒为 0：参数不动，loss 停在初始值"},
    "lr_huge":       {"stage": "update", "loud": True,  "desc": "学习率放大 100 倍：loss 爆炸 / NaN"},
    "no_clip":       {"stage": "update", "loud": False, "desc": "关闭梯度裁剪（本规模下单独无害，诚实记录）"},
    "lr_huge_noclip": {"stage": "update", "loud": True, "desc": "lr×100 且关闭裁剪：对照 lr_huge，看裁剪到底挡住了什么"},
    "amp_overflow":  {"stage": "update", "loud": False, "desc": "GradScaler 初始 scale 拉到 2^48：前几十步 inf 梯度全部跳步，参数纹丝不动"},
    # ---- 恢复阶段 ----
    "resume_opt":    {"stage": "resume", "loud": False, "desc": "续训只恢复权重，丢弃优化器状态"},
    "resume_scaler": {"stage": "resume", "loud": False, "desc": "续训只恢复权重，丢弃 GradScaler 状态"},
    "resume_rng":    {"stage": "resume", "loud": False, "desc": "续训只恢复权重，丢弃 RNG 状态"},
    "resume_sched":  {"stage": "resume", "loud": False, "desc": "续训只恢复权重，丢弃调度器进度"},
}

DATA_FAULTS = [f for f, m in FAULT_STAGES.items() if m["stage"] == "data"]
UPDATE_FAULTS = [f for f, m in FAULT_STAGES.items() if m["stage"] == "update"]
RESUME_FAULTS = [f for f, m in FAULT_STAGES.items() if m["stage"] == "resume"]


def apply_fault(fault: str, x: torch.Tensor, y: torch.Tensor,
                k: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    """在 batch 级别注入数据故障。返回注入后的 (x, y)。

    只处理 data 阶段的故障；update/resume 阶段的故障在训练循环里
    由 train_debug.py 按开关处理（它们破坏的不是 batch 本身）。
    """
    if fault == "label_shift":
        # x[t] 本应预测 y[t]=x[t+1]；偏移后预测 x[t+k+1]，任务变成"跳 k 个字预测"
        return x, torch.roll(y, shifts=k, dims=1)
    if fault == "label_shuffle":
        # batch 内每个样本的标签序列随机重排：x 与 y 的对应关系彻底破坏
        perm = torch.randperm(y.shape[0], device=y.device)
        return x, y[perm]
    if fault == "dup_batch":
        # 前一半样本覆盖后一半：模型反复见到同一批数据
        half = x.shape[0] // 2
        x = x.clone(); x[half:] = x[:half]
        y = y.clone(); y[half:] = y[:half]
        return x, y
    if fault == "val_leak":
        # val_leak 在数据集构建时注入（见 train_debug.py），batch 级无操作
        return x, y
    if fault in UPDATE_FAULTS or fault in RESUME_FAULTS:
        return x, y  # 非 batch 级故障
    raise ValueError(f"未知故障: {fault}（可选 {list(FAULT_STAGES)}）")


def fault_lr_multiplier(fault: str) -> float:
    """update 阶段故障的 lr 乘数。"""
    if fault == "lr_zero":
        return 0.0
    if fault in ("lr_huge", "lr_huge_noclip"):
        return 100.0
    return 1.0


def fault_clip_enabled(fault: str, default_clip: float) -> float:
    """update 阶段故障的裁剪设置。返回 clip 上限；关闭裁剪返回 inf。"""
    if fault in ("no_clip", "lr_huge_noclip"):
        return float("inf")
    return default_clip