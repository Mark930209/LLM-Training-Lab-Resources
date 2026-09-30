"""run_megatron.py —— Megatron-Core 迁移子集（E4）。

目标（REAL 尽力）：GPTModel（llama 风格配置）+ 映射后的冻结权重 + 200 步训练，
与基线 loss 曲线对账。跑不通的部分按 spec 标 REFERENCE 并做源码路径解读。

用法：
    python run_megatron.py --steps 200 --out results/e4_megatron.json
"""

from __future__ import annotations

import argparse
import os
import time

import torch

import parity_common as pc


def build_model():
    """megatron GPTModel（RMSNorm + swiglu + rope，对齐 llama 风格）。

    TE/Apex 缺席时用 get_gpt_layer_local_spec（官方本地降级路径）。
    """
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.transformer.transformer_config import TransformerConfig

    cfg = TransformerConfig(
        num_layers=6,
        hidden_size=256,
        num_attention_heads=16,
        num_query_groups=16,
        ffn_hidden_size=768,
        normalization="RMSNorm",
        activation_func=torch.nn.functional.silu,
        gated_linear_unit=True,      # silu(gate)*up = SwiGLU 语义
        add_bias_linear=False,
        init_method_std=0.02,
        params_dtype=torch.float32,
        use_cpu_initialization=False,
    )
    model = GPTModel(
        config=cfg,
        transformer_layer_spec=get_gpt_layer_local_spec(),
        pre_process=True,
        post_process=True,
        vocab_size=pc.VOCAB,
        max_sequence_length=pc.SEQ,
        position_embedding_type="rope",
        rotary_base=500000,
    )
    return model


def map_weights(llama_sd: dict, model: torch.nn.Module) -> dict:
    """ckpt_convert 的 llama3→megatron 映射 + 名字裁剪对齐。"""
    from ckpt_convert import llama3_to_megatron

    mapped = llama3_to_megatron(llama_sd)
    target = model.state_dict()
    used, skipped = {}, []
    for k, v in mapped.items():
        if k in target and target[k].shape == v.shape:
            used[k] = v
        else:
            skipped.append(k)
    return used, skipped, set(target) - set(used)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=pc.STEPS)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29561")

    t_start = time.perf_counter()
    import torch.distributed as dist
    dist.init_process_group("nccl")
    torch.cuda.set_device(0)
    device = "cuda:0"

    from megatron.core import parallel_state
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1,
                                             pipeline_model_parallel_size=1)
    # megatron tensor_parallel 层初始化要求 model-parallel RNG tracker
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    model_parallel_cuda_manual_seed(1234)

    model = build_model().to(device).train()

    used, skipped, missing = map_weights(pc.load_init(), model)
    missing = {m for m in missing if "inv_freq" not in m and "position" not in m}
    load_res = model.load_state_dict(used, strict=False)

    data = pc.load_data()
    import json as _json
    with open("out/e1_baseline.json", encoding="utf-8") as f:
        lr_table = _json.load(f)["lr_curve"]

    opt = torch.optim.AdamW(model.parameters(), lr=pc.LR, betas=pc.BETAS,
                            eps=pc.EPS, weight_decay=pc.WD)
    from torchtitan.components.loss import cross_entropy_loss

    loss_curve: list[float] = []
    t_first = None
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for i in range(args.steps):
        ids = data["input_ids"][i].to(device)
        labels = data["labels"][i].to(device)
        pos = torch.arange(pc.SEQ, device=device).unsqueeze(0) \
            .expand(ids.shape[0], -1)
        for g in opt.param_groups:
            g["lr"] = lr_table[i]
        opt.zero_grad()
        logits = model(ids, position_ids=pos, attention_mask=None)
        valid = (labels != -100).sum()
        loss = cross_entropy_loss(logits, labels) / valid
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), pc.MAX_NORM,
                                       foreach=True)
        opt.step()
        loss_curve.append(float(loss.detach()))
        if t_first is None:
            t_first = time.perf_counter() - t_start
        del logits
    train_s = time.perf_counter() - t0

    out = {
        "experiment": os.path.basename(args.out).replace(".json", ""),
        "framework": "megatron-core-subset",
        "megatron_core": __import__("megatron.core", fromlist=["__version__"])
        .__version__,
        "init": "file(mapped)",
        "map_used": len(used),
        "map_skipped": skipped,
        "map_missing_buffers": sorted(missing),
        "steps": args.steps,
        "loss_curve": loss_curve,
        "loss_first": loss_curve[0],
        "loss_last": loss_curve[-1],
        "param_checksum": pc.sd_checksum(model.state_dict()),
        "tokens_per_s": round(args.steps * pc.BATCH * pc.SEQ / train_s, 1),
        "train_seconds": round(train_s, 3),
        "startup_to_first_step_s": round(t_first, 3),
        "peak_mem_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "labels": {"REAL": "单机真实运行（子集）"},
    }
    pc.save_json(out, args.out)
    print(f"[megatron subset] loss {out['loss_first']:.4f} -> {out['loss_last']:.4f} "
          f"mapped {len(used)} skipped {len(skipped)}")


if __name__ == "__main__":
    main()
