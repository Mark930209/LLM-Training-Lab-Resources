"""exp_smoke/run.py —— 统一实验入口 run_experiment(config) 的最小实现。

这是系列公共实验接口的样板：读 config → 固定 seed → 建模型 → benchmark 循环
→ 留痕 → 返回标准化结果。后续每篇专题的实验都按这个骨架扩展。

用法（在项目工程根目录，common/ 的上一级）：
    python -m exp_smoke.run --config exp_smoke/config.yaml
    python -m exp_smoke.run --config exp_smoke/config.yaml --override hardware.batch_size=8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

# 允许从工程根目录直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import benchmark, config as cfg_mod, logging as log_mod, metrics  # noqa: E402
from common.reproducibility import set_all_seeds  # noqa: E402


class TinyGPT(nn.Module):
    """最小因果语言模型：token+pos embedding → N 层 decoder → lm_head。"""

    def __init__(self, vocab: int, hidden: int, layers: int, heads: int, seq_len: int):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, hidden)
        self.pos_emb = nn.Embedding(seq_len, hidden)
        enc = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=hidden * 4,
            batch_first=True, norm_first=True,
        )
        # enable_nested_tensor=False：norm_first=True 下避免无害警告，保持输出干净
        self.blocks = nn.TransformerEncoder(enc, num_layers=layers, enable_nested_tensor=False)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        T = idx.shape[1]
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        # 因果掩码由 mask 提供；不要同时传 is_causal（新版 torch 二者互斥）
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=idx.device)
        return self.lm_head(self.blocks(x, mask=mask))


def run_experiment(cfg: cfg_mod.Config) -> dict:
    """公共实验接口：输入 config，返回标准化结果 dict。"""
    exp, hw = cfg.experiment, cfg.hardware
    device = cfg_mod.resolve_device(cfg)
    set_all_seeds(exp.seed, deterministic=False)

    model = TinyGPT(exp.vocab_size, exp.hidden_size, exp.num_layers,
                    exp.num_heads, exp.seq_len).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=exp.lr)
    tokens_per_step = hw.batch_size * exp.seq_len

    rl = log_mod.RunLogger("runs", exp.name)
    rl.write_env_card(log_mod.env_card())
    cfg_mod.snapshot(cfg, rl.run_dir)
    rl.log.info("参数量: %s", metrics.param_count_mb(model))

    loss_history: list[float] = []

    def train_step() -> float:
        x = torch.randint(0, exp.vocab_size, (hw.batch_size, exp.seq_len), device=device)
        y = torch.randint(0, exp.vocab_size, (hw.batch_size, exp.seq_len), device=device)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits.reshape(-1, exp.vocab_size), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        lv = loss.item()
        loss_history.append(lv)
        return lv

    result = benchmark.run_benchmark(
        train_step, steps=exp.steps, tokens_per_step=tokens_per_step,
        device=device, warmup=exp.warmup_steps,
    )

    rl.log_metrics(**result.to_dict())
    rl.log.info("结果: %s", result.to_dict())
    rl.log.info("显存对账: %s", metrics.memory_report())
    run_dir = rl.close()

    out = result.to_dict()
    out["run_dir"] = str(run_dir)
    out["perplexity"] = round(metrics.perplexity(result.loss), 2)
    out["初始loss"] = round(loss_history[0], 4)
    out["最终loss"] = round(loss_history[-1], 4)
    out["loss曲线"] = [round(v, 4) for v in loss_history]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="exp_smoke/config.yaml")
    ap.add_argument("--override", action="append", default=[],
                    help="形如 hardware.batch_size=8，可多次")
    ap.add_argument("--output", default=None, help="结果 JSON 落盘路径")
    args = ap.parse_args()

    cfg = cfg_mod.load_config(args.config, args.override)
    result = run_experiment(cfg)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                     encoding="utf-8")


if __name__ == "__main__":
    main()
