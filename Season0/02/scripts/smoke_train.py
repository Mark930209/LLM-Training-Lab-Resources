#!/usr/bin/env python3
"""smoke_train.py —— LLM Training Lab 通用 GPU 冒烟训练脚本

用途：验证环境真的能训练（GPU 在算、loss 在降、显存/步耗时可采集）。
特性：通用（模型规模/步数/批大小参数化）、非交互、结果落盘 JSON。

用法：
    python smoke_train.py [--steps 200] [--hidden 512] [--layers 4] [--batch-size 16]
                          [--seq-len 256] [--seed 42] [--output results.json] [--device cuda]

输出 JSON 字段：loss 曲线、step_time_ms、tokens_per_sec、peak_memory_mb、硬件信息。
"""

from __future__ import annotations

import argparse
import json
import time

import torch
import torch.nn as nn


class TinyGPT(nn.Module):
    """最小 Transformer 语言模型：embedding → N 层 decoder → lm_head。"""

    def __init__(self, vocab: int, hidden: int, layers: int, heads: int, seq_len: int):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, hidden)
        self.pos_emb = nn.Embedding(seq_len, hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=hidden * 4,
            batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        # 因果掩码由 mask 提供；不要同时传 is_causal（新版 torch 二者互斥）
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=idx.device)
        x = self.blocks(x, mask=mask)
        return self.lm_head(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", default="smoke_results.json")
    args = ap.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("警告: CUDA 不可用，回退 CPU（结果不具 GPU 参考性）")
        device = "cpu"

    torch.manual_seed(args.seed)

    model = TinyGPT(args.vocab, args.hidden, args.layers, args.heads, args.seq_len).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params/1e6:.2f}M  设备: {device}")

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    losses, step_times = [], []
    tokens_per_step = args.batch_size * args.seq_len
    t_start = time.perf_counter()

    for step in range(args.steps):
        # 随机 token（冒烟测试只验证训练机制，不需要真实数据）
        x = torch.randint(0, args.vocab, (args.batch_size, args.seq_len), device=device)
        y = torch.randint(0, args.vocab, (args.batch_size, args.seq_len), device=device)

        t0 = time.perf_counter()
        logits = model(x)
        loss = nn.functional.cross_entropy(logits.reshape(-1, args.vocab), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if device == "cuda":
            torch.cuda.synchronize()
        step_times.append((time.perf_counter() - t0) * 1000)
        losses.append(loss.item())

        if step % 50 == 0 or step == args.steps - 1:
            print(f"step {step:4d}  loss {loss.item():.4f}  step_time {step_times[-1]:.1f}ms")

    total_s = time.perf_counter() - t_start
    # 去掉前 10 步预热
    warm = step_times[10:] if len(step_times) > 10 else step_times
    avg_step_ms = sum(warm) / len(warm)

    result = {
        "硬件": torch.cuda.get_device_name(0) if device == "cuda" else "CPU",
        "torch版本": torch.__version__,
        "参数量M": round(n_params / 1e6, 2),
        "步数": args.steps,
        "批大小": args.batch_size,
        "序列长度": args.seq_len,
        "seed": args.seed,
        "初始loss": round(losses[0], 4),
        "最终loss": round(losses[-1], 4),
        "平均step_time_ms": round(avg_step_ms, 2),
        "tokens_per_sec": round(tokens_per_step / (avg_step_ms / 1000)),
        "总耗时s": round(total_s, 1),
        "loss曲线": [round(v, 4) for v in losses],
    }
    if device == "cuda":
        result["峰值显存MB"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n结果已写入 {args.output}")
    print(f"初始loss {result['初始loss']} → 最终loss {result['最终loss']}  "
          f"平均step {result['平均step_time_ms']}ms  "
          f"吞吐 {result['tokens_per_sec']} tok/s  "
          f"峰值显存 {result.get('峰值显存MB', 'N/A')}MB")


if __name__ == "__main__":
    main()
