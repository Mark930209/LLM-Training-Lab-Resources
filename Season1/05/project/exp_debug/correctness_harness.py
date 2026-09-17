"""correctness_harness.py —— 五阶段训练诊断面板（05 篇核心交付物）。

把 diagnostics.py 的单项指标组织成一套带阈值的检查器，覆盖训练数据流的
五个阶段：数据 → 前向 → 反向 → 更新 → 恢复。

设计约束（重要）：
    Harness 只做观测与告警，不修改任何张量、不改变训练结果。
    所有 check_* 方法都是只读的，接入前后 loss 曲线必须逐位一致。
    这是它能当"回归门禁"的前提：门禁本身不能影响被测对象。

用法：
    harness = HarnessRunner(HarnessConfig(vocab_size=6015, freq_entropy=6.3680))
    harness.check_dataset(train_ds, val_ds)      # 训练开始前，一次
    harness.check_initial_loss(loss0)            # 第一个 batch，一次
    for x, y in loader:
        harness.check_batch(x)                   # 数据阶段
        before = snapshot_params(model)
        ... forward / backward ...
        harness.check_grad(model, grad_norm)     # 反向阶段
        ... optimizer.step() ...
        harness.check_update(model, before, scaler=scaler)   # 更新阶段
    harness.check_resume(model, opt, ref)        # 续训后，一次
    print(harness.summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from exp_debug.diagnostics import (
    batch_dup_rate, dataset_overlap, expected_initial_loss, opt_checksum,
    param_checksum, rng_checksum, update_ratio)


@dataclass
class HarnessConfig:
    """各阶段告警阈值。默认值来自 05 篇基线实测（10M × 大语料，300 步）。"""

    vocab_size: int
    # 数据阶段
    max_batch_dup_rate: float = 0.05      # 健康滑动窗口约 0；dup_batch 注入后 0.5
    max_dataset_overlap: int = 0          # train/val 重叠零容忍
    # 前向阶段
    max_init_loss_dev: float = 0.5        # 初始 loss 与 ln(vocab) 的允许偏差
    freq_entropy: float | None = None     # 字符频率熵，作为随机标签的 loss 下界
    entropy_tol: float = 0.15             # loss 落在 [entropy, entropy+tol] 视为可疑
    entropy_min_step: int = 200           # 给足步数再判熵下界，避免早期误报
    max_train_val_gap: float = 2.0        # train/val 剪刀差上限（基线实测 0.18）
    # 更新阶段
    zero_ratio_patience: int = 10         # 连续 N 步 update_ratio≈0 则告警
    min_update_ratio: float = 1e-8
    max_update_ratio: float = 1e-2        # 健康约 1e-3；lr_huge 实测 2.84e-02
    scaler_skip_patience: int = 10        # 连续 N 步 GradScaler 跳步则告警
    # 反向阶段
    max_grad_norm: float = 1e3            # 超过视为爆炸前兆


@dataclass
class HarnessState:
    """跨 step 的计数器与告警记录。

    告警按类别（key）去重：同一类问题无论触发多少步只记一条，
    另用 counts 累计次数。消息里带 step 号时若按全文去重会刷屏
    （label_shuffle 实测过 97 条）。
    """

    zero_ratio_streak: int = 0
    scaler_skip_streak: int = 0
    alerts: dict[str, str] = field(default_factory=dict)    # key -> 首次消息
    counts: dict[str, int] = field(default_factory=dict)    # key -> 触发次数
    checked: dict[str, bool] = field(default_factory=dict)

    def alert(self, key: str, msg: str) -> None:
        self.counts[key] = self.counts.get(key, 0) + 1
        self.alerts.setdefault(key, msg)

    def messages(self) -> list[str]:
        """去重后的告警文本，多次触发的附次数。"""
        out = []
        for key, msg in self.alerts.items():
            n = self.counts.get(key, 1)
            out.append(msg if n == 1 else f"{msg}（共 {n} 次）")
        return out


class HarnessRunner:
    """五阶段检查器。每个 check_* 对应训练循环里的一个位置。"""

    def __init__(self, cfg: HarnessConfig):
        self.cfg = cfg
        self.state = HarnessState()

    # ---- 阶段 1：数据 ----

    def check_dataset(self, train_ds, val_ds) -> int:
        """训练开始前调用一次：train/val 样本重叠检查。

        val_leak 注入后返回 1000（val 开头就是 train 开头）；健康划分为 0。
        这是唯一能在训练开始前就拦下泄漏的指标。
        """
        overlap = dataset_overlap(train_ds, val_ds)
        if overlap > self.cfg.max_dataset_overlap:
            self.state.alert(
                "dataset_overlap",
                f"[数据] train/val 重叠 {overlap} 条（阈值 "
                f"{self.cfg.max_dataset_overlap}）：val loss 在部分反映记忆而非泛化")
        self.state.checked["dataset"] = True
        return overlap

    def check_batch(self, x: torch.Tensor) -> float:
        """每个 batch 调用：batch 内样本重复率。

        dup_batch 注入后为 0.5（前一半覆盖后一半）；健康训练约 0。
        """
        dup = batch_dup_rate(x)
        if dup > self.cfg.max_batch_dup_rate:
            self.state.alert(
                "batch_dup",
                f"[数据] batch 内重复率 {dup:.3f}（阈值 "
                f"{self.cfg.max_batch_dup_rate}）：有效样本数被砍，检查采样逻辑")
        return dup

    # ---- 阶段 2：前向 ----

    def check_initial_loss(self, loss0: float) -> float:
        """第一个 batch 调用一次：初始 loss 应接近 ln(vocab)。

        随机初始化 + 均匀词表的理论值是 ln(vocab)。偏差过大说明模型结构、
        权重初始化或数据对接有 bug。这个检查启动后十秒就能做完。
        """
        expected = expected_initial_loss(self.cfg.vocab_size)
        dev = abs(loss0 - expected)
        if dev > self.cfg.max_init_loss_dev:
            self.state.alert(
                "init_loss",
                f"[前向] 初始 loss {loss0:.4f} 与理论值 ln({self.cfg.vocab_size})"
                f"={expected:.4f} 偏差 {dev:.4f}（阈值 {self.cfg.max_init_loss_dev}）")
        self.state.checked["initial_loss"] = True
        return dev

    def check_val_loss(self, val_loss: float, train_loss: float,
                       step: int) -> None:
        """每次评估后调用：val loss 对照 ln(vocab) 与 train loss。

        同一个理论值 ln(vocab) 有两种用法：初始 loss 应接近它（结构正确），
        而训练过的模型 val loss 应明显低于它。val loss 反而高于它，
        说明模型在被评估的任务上比没训练还差，几乎只能是 train/val
        任务不一致（label_shift 实测 val 9.85 > 8.70，train 却只有 0.45）。
        """
        expected = expected_initial_loss(self.cfg.vocab_size)
        if val_loss > expected:
            self.state.alert(
                "val_worse_than_random",
                f"[前向] step {step} 的 val loss {val_loss:.4f} 高于随机初始化"
                f"理论值 {expected:.4f}，而 train loss 只有 {train_loss:.4f}："
                f"train/val 任务不一致，优先查标签对齐")
            return
        gap = val_loss - train_loss
        if gap > self.cfg.max_train_val_gap:
            self.state.alert(
                "train_val_gap",
                f"[前向] step {step} 的 train/val 剪刀差 {gap:.2f}"
                f"（train {train_loss:.4f} / val {val_loss:.4f}，阈值 "
                f"{self.cfg.max_train_val_gap}）：模型在背训练集或任务已偏")

    def check_loss_floor(self, loss: float, step: int) -> bool:
        """loss 是否停在字符频率熵附近（随机标签的特征）。

        标签与输入脱钩时，模型只能学到字符边际频率，loss 会停在频率熵而
        不再下降。健康训练会突破这个下界。返回 True 表示可疑。
        """
        ent = self.cfg.freq_entropy
        if ent is None:
            return False
        suspicious = ent - self.cfg.entropy_tol <= loss <= ent + self.cfg.entropy_tol
        if suspicious and step >= self.cfg.entropy_min_step:
            self.state.alert(
                "loss_floor",
                f"[前向] step {step} 的 loss {loss:.4f} 停在字符频率熵 "
                f"{ent:.4f} 附近：标签可能已与输入脱钩")
        return suspicious

    # ---- 阶段 3：反向 ----

    def check_grad(self, grad_norm: float, step: int) -> None:
        """backward 之后调用：梯度范数是否有限、是否异常大。

        NaN/inf 在这里最先暴露，比等到 loss 变 NaN 早。
        """
        gn = float(grad_norm)
        if not torch.isfinite(torch.tensor(gn)):
            self.state.alert(
                "grad_nonfinite",
                f"[反向] step {step} grad_norm 为 {gn}：梯度 inf/nan")
        elif gn > self.cfg.max_grad_norm:
            self.state.alert(
                "grad_huge",
                f"[反向] step {step} grad_norm {gn:.2f} 超过 "
                f"{self.cfg.max_grad_norm:.0f}：爆炸前兆")

    # ---- 阶段 4：更新 ----

    def check_update(self, model: torch.nn.Module,
                     before: dict[str, torch.Tensor], step: int,
                     scaler=None) -> float:
        """optimizer.step() 之后调用：参数到底动没动、动了多少。

        lr_zero 注入后 update_ratio 恒为 0（唯一能抓到它的指标）；
        lr_huge 注入后达 2.84e-02，比健康值大两个量级。
        """
        ratio = update_ratio(model, before)

        if ratio < self.cfg.min_update_ratio:
            self.state.zero_ratio_streak += 1
            if self.state.zero_ratio_streak == self.cfg.zero_ratio_patience:
                self.state.alert(
                    "update_zero",
                    f"[更新] update_ratio 连续 {self.cfg.zero_ratio_patience} 步 "
                    f"< {self.cfg.min_update_ratio:.0e}：参数未更新，"
                    f"检查学习率是否为 0 或 AMP 是否连续跳步")
        else:
            self.state.zero_ratio_streak = 0

        if ratio > self.cfg.max_update_ratio:
            self.state.alert(
                "update_huge",
                f"[更新] step {step} update_ratio {ratio:.2e} 超过 "
                f"{self.cfg.max_update_ratio:.0e}：学习率可能过大")

        if scaler is not None:
            if self._scaler_skipped(scaler, step):
                self.state.scaler_skip_streak += 1
                if self.state.scaler_skip_streak == self.cfg.scaler_skip_patience:
                    self.state.alert(
                        "scaler_skip",
                        f"[更新] GradScaler 连续跳步 "
                        f"{self.cfg.scaler_skip_patience} 次：初始 scale 可能过大，"
                        f"这段时间的计算全部无效")
            else:
                self.state.scaler_skip_streak = 0

        return ratio

    @staticmethod
    def _scaler_skipped(scaler, step: int) -> bool:
        """本步 GradScaler 是否因 inf 梯度跳过了更新。

        读的是 PyTorch 内部记账（_per_optimizer_states 的 found_inf_per_device），
        只读不写。不同版本字段名可能变化，取不到时保守返回 False。
        """
        states = getattr(scaler, "_per_optimizer_states", None)
        if not states:
            return False
        for opt_state in states.values():
            found = opt_state.get("found_inf_per_device", {})
            for flag in found.values():
                try:
                    if bool(flag.item() if hasattr(flag, "item") else flag):
                        return True
                except Exception:
                    continue
        return False

    # ---- 阶段 5：恢复 ----

    def snapshot_state(self, model: torch.nn.Module,
                       opt: torch.optim.Optimizer) -> dict[str, str]:
        """存档时记录三类校验和，作为续训一致性的参考基准。"""
        return {"param": param_checksum(model),
                "opt": opt_checksum(opt),
                "rng": rng_checksum()}

    def check_resume(self, model: torch.nn.Module,
                     opt: torch.optim.Optimizer,
                     reference: dict[str, str] | None) -> list[str]:
        """续训后调用：逐组件比对校验和，定位丢了哪一份状态。

        不一致的组件就是没恢复完整的部分。注意 resume_sched 这类故障
        校验和可能全对（权重/优化器/RNG 都恢复了），只有调度器进度丢了，
        所以调度器要单独按 step 校验，见 check_scheduler_step。
        """
        mismatched: list[str] = []
        if not reference:
            return mismatched
        actual = self.snapshot_state(model, opt)
        for key in ("param", "opt", "rng"):
            ref = reference.get(key)
            if ref is not None and actual[key] != ref:
                mismatched.append(key)
        if mismatched:
            self.state.alert(
                "resume_checksum",
                f"[恢复] 校验和不一致的组件：{', '.join(mismatched)}；"
                f"续训状态不完整，即使指标看起来正常")
        return mismatched

    def check_scheduler_step(self, resumed_step: int,
                             expected_step: int) -> None:
        """续训后校验调度器进度：warmup 是否被重跑。

        resume_sched 故障下 resumed_step 会从 0 重新开始，学习率曲线与
        连续训练不一致。这是唯一能抓到它的检查。
        """
        if resumed_step != expected_step:
            self.state.alert(
                "resume_sched",
                f"[恢复] 调度器进度 {resumed_step} 与存档步数 {expected_step} "
                f"不一致：warmup 可能重跑，学习率轨迹已偏离")

    # ---- 汇总 ----

    def summary(self) -> str:
        msgs = self.state.messages()
        if not msgs:
            return "[Harness] 五阶段检查全部通过，训练状态健康。"
        lines = "\n".join(f"  - {m}" for m in msgs)
        return f"[Harness] 发现 {len(msgs)} 类问题：\n{lines}"

    @property
    def ok(self) -> bool:
        return not self.state.alerts