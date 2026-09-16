"""diagnostics.py —— 05 篇最小诊断面板：健康基线该观测什么。

04 篇的训练程序只记录 loss / lr / grad_norm。本篇要回答"该观测什么"，
给训练循环补上五类证据，让数据、前向、反向、更新、恢复每个阶段
都有可检查的指标：

    数据    data_fingerprint   batch 内容哈希（检测重复/泄漏/错位）
    前向    initial_loss       理论值 ln(vocab)，偏离即结构/数据有 bug
    反向    grad_norm          梯度范数（爆炸/消失/NaN 的第一现场）
    更新    update_ratio       ||Δw|| / ||w||：参数到底动没动、动了多少
    恢复    state_checksum     权重/优化器/RNG 的恢复一致性

这些指标合起来就是 Training Correctness Harness 的核心。
"""

from __future__ import annotations

import hashlib
import math

import torch


def data_fingerprint(x: torch.Tensor, y: torch.Tensor) -> str:
    """batch 内容指纹：前 8 位十六进制。同一 batch 必得同一指纹。

    用途：跨 batch 的重复数据会让指纹集合变小；配合 batch_dup_rate
    与 dataset_overlap 覆盖三类数据故障。
    """
    h = hashlib.sha256()
    h.update(x.cpu().numpy().tobytes())
    h.update(y.cpu().numpy().tobytes())
    return h.hexdigest()[:8]


def batch_dup_rate(x: torch.Tensor) -> float:
    """batch 内样本重复率：1 - 去重行数 / 总行数。

    dup_batch 注入后接近 0.5（前一半覆盖后一半）；
    健康训练的滑动窗口几乎不可能产出完全相同的样本行，约 0。
    """
    rows = [hashlib.sha256(r.numpy().tobytes()).hexdigest()
            for r in x.cpu()]
    return 1 - len(set(rows)) / len(rows)


def dataset_overlap(train_ds, val_ds, n: int = 1000) -> int:
    """train/val 样本重叠数：各自取前 n 个样本算内容哈希，数交集。

    val_leak 注入后，val 的开头就是 train 的开头，重叠数接近 n；
    健康划分下为 0。这是泄漏最直接的证据。
    """
    def sample_hashes(ds):
        out = set()
        for i in range(min(n, len(ds))):
            x, y = ds[i]
            out.add(hashlib.sha256(x.numpy().tobytes() +
                                   y.numpy().tobytes()).hexdigest())
        return out
    return len(sample_hashes(train_ds) & sample_hashes(val_ds))


def expected_initial_loss(vocab_size: int) -> float:
    """随机初始化模型的初始 loss 理论值：均匀分布 = ln(vocab)。"""
    return math.log(vocab_size)


def param_checksum(model: torch.nn.Module) -> str:
    """模型权重校验和：恢复一致性测试的基准。"""
    h = hashlib.sha256()
    for p in model.parameters():
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:12]


def opt_checksum(opt: torch.optim.Optimizer) -> str:
    """优化器状态校验和（动量 m / v）。伪续训检测的关键：
    只恢复权重的续训，这里必然对不上。"""
    h = hashlib.sha256()
    for group in opt.param_groups:
        for p in group["params"]:
            state = opt.state.get(p, {})
            for key in ("exp_avg", "exp_avg_sq"):
                if key in state:
                    h.update(state[key].detach().cpu().numpy().tobytes())
    return h.hexdigest()[:12]


def update_ratio(model: torch.nn.Module, before: dict[str, torch.Tensor]) -> float:
    """参数更新量比 ||Δw|| / ||w||（对所有参数求和的标量版）。

    lr_zero 注入后这个值是 0（参数纹丝不动）；健康训练约 1e-3 量级。
    """
    delta_sq, w_sq = 0.0, 0.0
    for name, p in model.named_parameters():
        w_before = before[name]
        delta_sq += (p.detach() - w_before).pow(2).sum().item()
        w_sq += w_before.pow(2).sum().item()
    return math.sqrt(delta_sq) / max(math.sqrt(w_sq), 1e-12)


def snapshot_params(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """训练 step 前的参数快照（算 update_ratio 用）。"""
    return {name: p.detach().clone() for name, p in model.named_parameters()}


def rng_checksum() -> str:
    """当前 RNG 状态指纹（torch CPU + CUDA）。resume_rng 检测用。"""
    h = hashlib.sha256()
    h.update(torch.get_rng_state().numpy().tobytes())
    if torch.cuda.is_available():
        h.update(torch.cuda.get_rng_state().numpy().tobytes())
    return h.hexdigest()[:12]


def first_nonfinite_step(losses: list[float]) -> int | None:
    """loss 序列里第一个非有限值的位置（1-based step）。"""
    for i, v in enumerate(losses, 1):
        if not math.isfinite(v):
            return i
    return None