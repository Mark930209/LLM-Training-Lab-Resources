"""fail_modes_hf.py —— 接入 Hugging Face 时最容易配错的三处（06 篇失败实验）。

05 篇的 fail_modes 注入的是训练过程故障；本篇注入的是**接口配置故障**：
模型和循环都没坏，坏的是两者之间的约定。这类故障更隐蔽，
因为它们往往不报错，只是 loss 的数字悄悄变味。

三个开关（与 05 篇同样的用法：--fault 指定一个）：

    double_shift        target 在同长度内多 roll 一位：输入是到 t 为止的上下文，
                        标签却是 t+2。模型学的是"跳一个字预测"，
                        train loss 会降，但验证指标量的仍是 t+1 任务，
                        两条曲线剪刀差——和 05 篇 label_shift 同一族故障，
                        只是这次错在接口配置的 off-by-one，不是数据管线。
    vocab_mismatch      用 BPE 词表（8,192）的 id 喂 char 词表（6,015）的
                        embedding：id 越界直接 IndexError，
                        或者 id 不越界但指向错误的 token（静默版）。
    resume_weights_only 只 from_pretrained 权重、不读 sidecar 训练状态：
                        程序照常跑，但优化器动量与学习率进度全丢，
                        续训轨迹从断点处偏离（05 篇 resume_opt 的 HF 版）。
"""

from __future__ import annotations

import torch
import torch.nn as nn


FAULTS = {
    "double_shift": "label shift 做两遍：模型预测 t+2 而不是 t+1",
    "vocab_mismatch": "BPE 的 id 喂 char 词表的 embedding",
    "resume_weights_only": "只恢复权重，不读 sidecar 训练状态",
}


def apply_double_shift(x: torch.Tensor, y: torch.Tensor,
                       k: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """输入保持正确口径，target 在同长度内 roll k 位：制造 train/val 任务不一致。

    正常口径：输入 x（上下文到 t），标签 y（t+1）。
    double_shift：输入不变，标签变成 roll(y, k)（t+1+k）。
    训练任务被换掉，但验证指标仍按 t+1 口径量，剪刀差由此产生。
    与 05 篇 fail_modes.label_shift 同族，注入点在接口层。
    """
    return x, torch.roll(y, shifts=k, dims=1)


def check_vocab_match(ids_max: int, embedding_rows: int) -> str | None:
    """id 越界检查：越界返回错误描述，否则返回 None。

    越界是显性失败（IndexError 前的一步）；不越界但词表语义不同
    是静默失败——id 落在范围内，指向的却是另一个 token。
    """
    if ids_max >= embedding_rows:
        return (f"最大 id {ids_max} 超出 embedding 行数 {embedding_rows}，"
                f"tokenizer 与模型词表不一致")
    return None


def describe_fault(name: str) -> str:
    return FAULTS.get(name, f"未知故障: {name}（可选 {list(FAULTS)}）")
