"""run_baseline.py —— plain PyTorch 参照实现（parity 基线）。

用法：
    python run_baseline.py --out results/e1_baseline.json
    python run_baseline.py --init native --out results/e6a_baseline_native.json

训练语义（镜像 TorchTitan Trainer 的 step 顺序）：
    loss = cross_entropy(sum) / global_valid_tokens  →  backward
    →  clip_grad_norm_(max_norm=1.0, foreach=True)  →  AdamW.step  →  lr.step
"""

from __future__ import annotations

import argparse
import os
import time

import torch

import parity_common as pc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", choices=["file", "native"], default="file")
    ap.add_argument("--steps", type=int, default=pc.STEPS)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--diag", action="store_true", help="打印逐步 lr")
    ap.add_argument("--save-params", default="", help="落盘最终 state_dict")
    ap.add_argument("--stop-after", type=int, default=0,
                    help=">0 时跑满调度形状但在该步后停止（短跑对账用）")
    ap.add_argument("--max-norm", type=float, default=pc.MAX_NORM)
    ap.add_argument("--opt-impl", choices=["fused", "foreach", "for-loop"],
                    default="for-loop",
                    help="AdamW kernel 路径；三腿须同值否则单 ulp 差被混沌放大")
    ap.add_argument("--save-grads", default="", help="落盘首次反传梯度")
    args = ap.parse_args()

    t_start = time.perf_counter()
    device = args.device
    data = pc.load_data()

    # ---- 模型与初始化：默认从冻结文件加载（不信构建顺序的 RNG）
    torch.manual_seed(pc.SEED_MODEL)
    model = pc.build_model()
    if args.init == "file":
        missing, unexpected = model.load_state_dict(pc.load_init(), strict=True)
        assert not missing and not unexpected
    else:
        model.init_states()
    model = model.to(device).train()

    # ---- 优化器：与 TorchTitan default_adamw 同参数（torch.optim.AdamW）
    from torchtitan.components.optimizer import (
        LRSchedulersContainer,
        OptimizersContainer,
        ParamGroupConfig,
    )
    opt_kwargs = {"lr": pc.LR, "betas": pc.BETAS,
                  "eps": pc.EPS, "weight_decay": pc.WD}
    opt_cfg = OptimizersContainer.Config(
        implementation=args.opt_impl,
        param_groups=[
            ParamGroupConfig(pattern=r".*", optimizer_name="AdamW",
                             optimizer_kwargs=opt_kwargs),
        ])
    optimizers = opt_cfg.build(model_parts=[model])
    sched_cfg = LRSchedulersContainer.Config(warmup_steps=pc.WARMUP_STEPS)
    lr_schedulers = sched_cfg.build(optimizers=optimizers,
                                    training_steps=args.steps)

    # ---- loss 口径：sum 归约 / 全局有效 token 数（TorchTitan 口径）
    from torchtitan.components.loss import cross_entropy_loss

    def step_loss(pred, labels):
        valid = (labels != -100).sum()
        return cross_entropy_loss(pred, labels) / valid

    # ---- 训练循环
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    loss_curve: list[float] = []
    lr_curve: list[float] = []
    t_first = None
    t0 = time.perf_counter()
    for i in range(args.steps):
        if args.stop_after and i >= args.stop_after:
            break
        optimizers.zero_grad()   # 每步清零（对齐 Trainer/DS 的梯度生命周期）
        ids = data["input_ids"][i].to(device)
        labels = data["labels"][i].to(device)
        positions = torch.arange(pc.SEQ, device=device).unsqueeze(0) \
            .expand(ids.shape[0], -1)
        masks = model.get_attention_masks(positions=positions)
        pred = model(ids, positions=positions, attention_masks=masks)
        loss = step_loss(pred, labels)
        loss.backward()
        if args.save_grads and i in (0, 1):
            torch.save({k: p.grad.detach().cpu()
                        for k, p in model.named_parameters()},
                       args.save_grads.replace(".pt", f"_s{i+1}.pt"))
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm,
                                               foreach=True)
        # 本次 update 实际使用的 lr（scheduler 上一次 step 后的值）
        lr_used = float(list(lr_schedulers.get_metrics().values())[0])
        lr_curve.append(lr_used)
        optimizers.step()
        lr_schedulers.step()
        if args.diag:
            m = lr_schedulers.get_metrics()
            print(f"diag step={i+1} lr={m} gnorm={float(gnorm):.6f} "
                  f"loss={float(loss.detach()):.6f}")
        loss_curve.append(float(loss.detach()))
        if t_first is None:
            t_first = time.perf_counter() - t_start   # 启动到首 step 完成
        del pred
    train_s = time.perf_counter() - t0

    ntokens = args.steps * pc.BATCH * pc.SEQ
    sd = model.state_dict()
    if args.save_params:
        torch.save({k: v.detach().cpu() for k, v in sd.items()}, args.save_params)
    out = {
        "experiment": "e1_baseline" if args.init == "file" else "e6a_baseline_native",
        "framework": "plain-pytorch",
        "init": args.init,
        "device": device,
        "torch": torch.__version__,
        "steps": args.steps,
        "loss_curve": loss_curve,
        "lr_curve": lr_curve,
        "loss_first": loss_curve[0],
        "loss_last": loss_curve[-1],
        "param_checksum": pc.sd_checksum(sd),
        "tokens_per_s": round(ntokens / train_s, 1),
        "train_seconds": round(train_s, 3),
        "startup_to_first_step_s": round(t_first, 3),
        "peak_mem_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1)
        if device.startswith("cuda") else None,
        "labels": {"REAL": "本机真实运行"},
    }
    pc.save_json(out, args.out)
    print(f"[baseline] loss {out['loss_first']:.4f} -> {out['loss_last']:.4f} "
          f"checksum {out['param_checksum']} tokens/s {out['tokens_per_s']}")


if __name__ == "__main__":
    main()
