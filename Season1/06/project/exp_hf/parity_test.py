"""parity_test.py —— 接口改造的一致性证明（06 篇读者最终能力）。

Adapter 包了一层 HF 外壳，凭什么说它没有改变模型行为？凭这三个测试：

    1. logits parity   同一份权重、同一个输入，原模型与 Adapter 的 logits
                       最大绝对误差应为 0（外壳只转发，不加计算）
    2. loss parity     同一批数据，两种 loss 口径算出的标量一致
    3. grad parity     同一步 backward，两边参数梯度的最大绝对误差应为 0

    4. roundtrip       save_pretrained → from_pretrained 之后，
                       logits 与保存前逐位一致（标准格式不丢精度）

任何一项不为 0 / 不一致，说明"接入"这个动作本身改变了实验，
必须先修接入，再谈换模型。这就是 05 篇门禁思想在接口层的复用。
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exp_hf.adapters import SuperMiniGPTAdapter, SuperMiniGPTConfig  # noqa: E402
from exp_hf.contract import contract_loss  # noqa: E402
from exp_scale.model import SuperMiniGPT  # noqa: E402


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def run_parity(vocab_size: int = 6015, hidden: int = 384, layers: int = 6,
               heads: int = 6, seq_len: int = 256, seed: int = 7,
               device: str = "cpu") -> dict:
    torch.manual_seed(seed)
    core = SuperMiniGPT(vocab_size, hidden, layers, heads, seq_len).to(device)
    cfg = SuperMiniGPTConfig(vocab_size=vocab_size, hidden=hidden, layers=layers,
                             heads=heads, seq_len=seq_len)
    adapter = SuperMiniGPTAdapter.from_superminigpt(core, cfg).to(device)

    x = torch.randint(0, vocab_size, (4, 64), device=device)
    ids = torch.randint(0, vocab_size, (4, 65), device=device)

    # 1. logits parity
    with torch.no_grad():
        lg_core = core(ids[:, :-1])
        lg_adapt = adapter(ids[:, :-1]).logits
    logits_diff = _max_abs_diff(lg_core, lg_adapt)

    # 2. loss parity：contract_loss 对两者给出同一标量
    with torch.no_grad():
        loss_core = contract_loss(lg_core, ids[:, 1:])
        loss_adapt = contract_loss(adapter(ids[:, :-1]), ids[:, 1:])
    loss_diff = _max_abs_diff(loss_core, loss_adapt)

    # 3. grad parity：同一步 backward
    core.zero_grad(set_to_none=True)
    adapter.zero_grad(set_to_none=True)
    contract_loss(core(ids[:, :-1]), ids[:, 1:]).backward()
    contract_loss(adapter(ids[:, :-1]), ids[:, 1:]).backward()
    grad_diff = 0.0
    for (n1, p1), (n2, p2) in zip(core.named_parameters(),
                                  adapter.core.named_parameters()):
        if p1.grad is None or p2.grad is None:
            continue
        grad_diff = max(grad_diff, _max_abs_diff(p1.grad, p2.grad))

    # 4. roundtrip：save_pretrained → from_pretrained
    with tempfile.TemporaryDirectory() as td:
        adapter.save_pretrained(td)
        reloaded = SuperMiniGPTAdapter.from_pretrained(td).to(device)
        with torch.no_grad():
            lg_re = reloaded(ids[:, :-1]).logits
        roundtrip_diff = _max_abs_diff(lg_adapt, lg_re)
        files = sorted(p.name for p in Path(td).iterdir())

    return {
        "logits_max_abs_diff": logits_diff,
        "loss_max_abs_diff": loss_diff,
        "grad_max_abs_diff": grad_diff,
        "roundtrip_max_abs_diff": roundtrip_diff,
        "saved_files": files,
        "pass": all(v == 0.0 for v in
                    (logits_diff, loss_diff, grad_diff, roundtrip_diff)),
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    res = run_parity(device=args.device)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
