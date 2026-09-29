"""probe_mem.py —— 单步显存/吞吐探针：为 config 的硬件选型给实测依据。

在目标卡上跑若干 (batch, seq, grad_ckpt) 组合，各 20 步 AdamW 取稳态峰值
显存与 tok/s。22 篇正文的"实测峰值显存 + 降级路线"即出自本脚本：

  RTX 3070 8GB（WSL2）实测：
    bs=2 seq=512 无 ckpt：6337 MiB / 1340 tok/s（正文采用档）
    bs=4 seq=512 无 ckpt：9783 MiB 超 8GB（WSL 显存溢出，tps 掉到 408，禁用）
    bs=4 seq=512 有 ckpt：7115 MiB / 1245 tok/s
    bs=2 seq=256 无 ckpt：4930 MiB / 1621 tok/s
    bs=1 seq=512 无 ckpt：4930 MiB / 1622 tok/s（4GB 卡降级路线）

用法：PYTHONPATH=. python probe_mem.py
（loss 值是随机 token 记忆的伪值，只用于确认训练在动，不作质量依据。）
"""

from __future__ import annotations

import time

import torch

from train_cpt import load_model

MODEL_DIR = "../../model/Qwen2.5-0.5B"


def probe(model, bs: int, seq: int, grad_ckpt: bool) -> bool:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    if grad_ckpt:
        model.gradient_checkpointing_enable()
    else:
        model.gradient_checkpointing_disable()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.1)
    ids = torch.randint(0, 151643, (bs, seq + 1), device="cuda")
    try:
        t0 = time.perf_counter()
        for i in range(20):
            out = model(ids, labels=ids)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            if i == 10:
                torch.cuda.reset_peak_memory_stats()   # 稳态峰值
        dt = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated() / 1024 ** 2
        tps = 10 * bs * seq / dt
        print(f"  bs={bs} seq={seq} ckpt={grad_ckpt}: "
              f"peak={peak:.0f} MiB  tps={tps:.0f}  loss={out.loss.item():.3f}")
        return True
    except torch.cuda.OutOfMemoryError:
        print(f"  bs={bs} seq={seq} ckpt={grad_ckpt}: OOM")
        return False


def main() -> None:
    model, _tok = load_model(MODEL_DIR)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params / 1e6:.1f}M")
    for bs, seq, ck in ((4, 512, False), (2, 512, False), (4, 512, True),
                        (2, 256, False), (1, 512, False)):
        probe(model, bs, seq, ck)


if __name__ == "__main__":
    main()
