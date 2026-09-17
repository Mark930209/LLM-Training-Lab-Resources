"""contract.py —— Mini Training Framework v1 的接口契约（06 篇核心概念）。

04 篇的训练循环直接 import SuperMiniGPT，模型和循环焊死在一起。
要换 Hugging Face 模型而不重写循环，就得先回答：循环到底依赖模型的哪几个接口？

把 04 篇 train.py 从头读一遍，循环碰模型的地方只有五处，这就是契约：

    1. 输入/输出   model(input_ids) -> logits，形状 (B, T, vocab)
    2. loss 口径   循环自己算 CE(logits, targets)，模型不算 loss；
                   shift 由数据集完成（dataset 返回的 (x, y) 已错一位），
                   循环不再二次 shift
                   （HF 的 forward(labels=...) 会在内部做 label shift，
                    和本循环的口径冲突，见 fail_modes_hf.double_shift）
    3. mask        因果掩码由模型内部保证，循环不传 attention_mask
                   （单序列 packed 训练，无 padding，mask 恒全 1）
    4. 序列化      权重走 save_pretrained/from_pretrained 标准格式；
                   优化器/调度器/步数等训练状态走 sidecar 文件（HF 格式不含它们）
    5. 配置        model.config 可序列化为 dict，重建模型只需要这个 dict

check_contract() 在接入任何新模型时先跑一遍，把不满足的接口列出来，
而不是等训练跑到一半才崩。
"""

from __future__ import annotations

import torch
import torch.nn as nn


CONTRACT_FIELDS = (
    "forward(input_ids) -> logits (B, T, vocab)",
    "loss 由循环计算：CE(logits[:, :-1], ids[:, 1:])",
    "因果掩码在模型内部，循环不传 attention_mask",
    "权重 save_pretrained/from_pretrained；训练状态走 sidecar",
    "config 可序列化为 dict，重建模型只需 config",
)


def _as_logits(out):
    """把模型输出统一成 logits 张量。

    手写模型直接返回张量；HF 模型返回 CausalLMOutput 对象（.logits）；
    也有返回 tuple 的旧接口。三种都接住，循环侧就只面对张量。
    """
    if isinstance(out, torch.Tensor):
        return out
    if hasattr(out, "logits"):
        return out.logits
    if isinstance(out, tuple):
        return out[0]
    raise TypeError(f"无法识别的模型输出类型：{type(out)}")


def check_contract(model: nn.Module, vocab_size: int,
                   require_hf: bool = True) -> list[str]:
    """对接入的模型做契约体检，返回不满足项（空列表 = 通过）。

    只做静态与前向检查，不训练：接入新模型的第一道门。
    require_hf=False 时只查前向与 loss 口径（裸模型参考实现路径），
    序列化与 config 检查只对 HF 路径生效。
    """
    problems: list[str] = []

    if not callable(getattr(model, "forward", None)):
        problems.append("模型没有 forward 方法")
        return problems

    # 1. 前向形状：小 batch 跑一次，检查 logits 维度
    dev = next(model.parameters()).device
    try:
        x = torch.zeros(2, 8, dtype=torch.long, device=dev)
        with torch.no_grad():
            out = model(x)
        out = _as_logits(out)
        if out.dim() != 3 or out.shape[0] != 2 or out.shape[1] != 8:
            problems.append(f"logits 形状应为 (B, T, V)，实测 {tuple(out.shape)}")
        elif out.shape[2] != vocab_size:
            problems.append(
                f"logits 最后一维 {out.shape[2]} 与词表 {vocab_size} 不一致"
                "（tokenizer 与 embedding 尺寸不匹配，见 fail_modes_hf.vocab_mismatch）")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"前向失败：{type(exc).__name__}: {exc}")
        return problems

    # 2. 循环口径的 loss 必须可算（模型自己不返回 loss 张量）
    try:
        y = torch.zeros(2, 8, dtype=torch.long, device=dev)
        with torch.no_grad():
            logits = _as_logits(model(x))
            nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
    except Exception as exc:  # noqa: BLE001
        problems.append(f"循环口径 loss 计算失败：{type(exc).__name__}: {exc}")

    if not require_hf:
        return problems

    # 5. config 可序列化
    cfg = getattr(model, "config", None)
    if cfg is None:
        problems.append("模型没有 config 属性")
    else:
        try:
            d = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)
            import json
            json.dumps(d, default=str)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"config 不可序列化：{type(exc).__name__}: {exc}")

    # 4. 标准序列化接口
    for name in ("save_pretrained",):
        if not hasattr(model, name):
            problems.append(f"模型缺少 {name}（无法走 HF 标准格式存权重）")
    return problems


def contract_loss(out, targets: torch.Tensor) -> torch.Tensor:
    """契约第 2 条的统一实现：循环侧唯一允许的 loss 口径。

    logits 与 targets 形状相同 (B, T)：shift 由数据集完成，
    这里只做 CE，不再二次 shift。
    所有 Adapter、所有模型都走这一个函数算 loss，
    保证"换模型"不会偷偷换掉 loss 的定义。
    输入可以是 logits 张量，也可以是 HF 的输出对象。
    """
    logits = _as_logits(out)
    return nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
