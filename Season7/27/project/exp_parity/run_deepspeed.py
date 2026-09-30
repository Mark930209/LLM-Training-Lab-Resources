"""run_deepspeed.py —— DeepSpeed 迁移腿：同一模型/数据/初始化的 parity 驱动。

用法（单机 world=1 或 torchrun 两机）：
    python run_deepspeed.py --zero 1 --out results/e2_ds_zero1.json
    python run_deepspeed.py --zero 3 --zero3-init --out results/e6b_ds_zero3_init.json

语义对齐点（与 run_baseline 同口径）：
    loss = CE(sum) / global_valid_tokens；AdamW 显式传入 torch.optim.AdamW
    （lr/betas/eps/wd 同 parity_common，绕开需 nvcc JIT 的 fused_adam）；
    gradient_clipping=1.0。zero3-init 腿在 zero.Init 下构造模型，初始化 RNG 与
    基线不同（失败案例 1 的病灶腿）。
"""

from __future__ import annotations

import argparse
import os
import time

import deepspeed
import torch

import parity_common as pc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--zero", type=int, default=1, choices=[0, 1, 2, 3])
    ap.add_argument("--steps", type=int, default=pc.STEPS)
    ap.add_argument("--sched", choices=["none", "warmup", "lrfile"],
                    default="none",
                    help="none=直跑默认（失败案例腿）；warmup=DS WarmupLR；"
                         "lrfile=吃基线 lr_curve（精确对齐腿）")
    ap.add_argument("--lr-file", default="", help="基线结果 JSON 路径")
    ap.add_argument("--zero3-init", action="store_true")
    ap.add_argument("--ckpt-dir", default="")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--resume-nopt", action="store_true",
                    help="恢复权重但不恢复优化器状态（失败案例 2：转换遗漏优化器）")
    ap.add_argument("--start-step", type=int, default=0,
                    help="恢复时的起始步（衔接 lr 表）")
    ap.add_argument("--diag", action="store_true", help="打印逐步 lr")
    ap.add_argument("--save-params", default="", help="落盘最终 state_dict")
    ap.add_argument("--stop-after", type=int, default=0,
                    help=">0 时跑满调度形状但在该步后停止（短跑对账用）")
    ap.add_argument("--max-norm", type=float, default=pc.MAX_NORM)
    ap.add_argument("--clip", choices=["builtin", "manual"], default="builtin",
                    help="builtin=DS 自研 clip；manual=torch clip（仅 zero0，bitwise 控腿）")
    ap.add_argument("--save-grads", default="", help="落盘首次反传梯度")
    args = ap.parse_args()

    # 单进程直跑时补齐 torchrun 环境（DeepSpeed init_distributed 否则走 MPI 探测）
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29511")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    deepspeed.init_distributed()
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    rank = int(os.environ.get("RANK", 0))

    t_start = time.perf_counter()
    data = pc.load_data()

    # ---- 模型：默认从冻结 init 文件加载；zero3-init 腿在 zero.Init 下现造
    # （zero.Init 与 torchtitan Configurable.build/.init_states 两处不兼容：
    #  .build() 报 "Config has no owner class"，.init_states() 撞分区空张量——
    #  绕过两处直接构造，权重由 zero.Init 的分区 RNG 现造 = 失败案例 1 病灶腿）
    if args.zero3_init:
        from torchtitan.models.llama3.model import Llama3Model
        with deepspeed.zero.Init(enabled=True):
            model = Llama3Model(config=pc.build_model_spec().model)
        init_mode = "zero3-init"
    else:
        torch.manual_seed(pc.SEED_MODEL)
        model = pc.build_model()
        model.load_state_dict(pc.load_init(), strict=True)
        init_mode = "file"
    model = model.to(device).train()

    ds_config = {
        "train_micro_batch_size_per_gpu": pc.BATCH,
        "gradient_accumulation_steps": 1,
        "steps_per_print": 10**9,
        "gradient_clipping": args.max_norm if args.clip == "builtin" else 1e9,
        "fp16": {"enabled": False},
        "bf16": {"enabled": False},
        "zero_optimization": {"stage": args.zero},
        "wall_clock_breakdown": False,
    }
    if args.clip == "manual":
        assert args.zero == 0, "manual clip 只支持 zero0（zero>=1 梯度在 flat buffer）"
    if args.sched == "warmup":
        # 对齐 torchtitan LRSchedulersContainer(warmup_steps=20)：线性 warmup 后恒定
        ds_config["scheduler"] = {
            "type": "WarmupLR",
            "params": {"warmup_min_lr": 0.0, "warmup_max_lr": pc.LR,
                       "warmup_num_steps": pc.WARMUP_STEPS},
        }
    lr_table = None
    if args.sched == "lrfile":
        assert args.lr_file, "--sched lrfile 需要 --lr-file"
        import json as _json
        with open(args.lr_file, encoding="utf-8") as f:
            lr_table = _json.load(f)["lr_curve"]

    # 优化器显式传入：与基线同一个 torch.optim.AdamW（fused_adam 需 nvcc JIT，
    # 本机无 CUDA_HOME 编不了——降级路线记入环境预检）。kernel 路径钉死 for-loop
    # （fused/foreach 与基线不一致会引入单 ulp 差并被混沌放大）。
    opt = torch.optim.AdamW(model.parameters(), lr=pc.LR, betas=pc.BETAS,
                            eps=pc.EPS, weight_decay=pc.WD,
                            fused=False, foreach=False)
    engine, optimizer, _, _ = deepspeed.initialize(
        model=model, optimizer=opt, config=ds_config)

    if (args.resume or args.resume_nopt) and args.ckpt_dir:
        if args.resume_nopt:
            # 只载权重，优化器动量/方差归零 = 转换脚本遗漏优化器状态的语义
            engine.load_checkpoint(args.ckpt_dir,
                                   load_optimizer_states=False,
                                   load_lr_scheduler_states=False)
        else:
            engine.load_checkpoint(args.ckpt_dir)

    from torchtitan.components.loss import cross_entropy_loss

    def step_loss(pred, labels):
        valid = (labels != -100).sum()
        return cross_entropy_loss(pred, labels) / valid

    loss_curve: list[float] = []
    t_first = None
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for i in range(args.start_step, args.steps):
        if args.stop_after and i >= args.stop_after:
            break
        if lr_table is not None:
            for g in engine.optimizer.param_groups:
                g["lr"] = lr_table[i]
        ids = data["input_ids"][i].to(device)
        labels = data["labels"][i].to(device)
        positions = torch.arange(pc.SEQ, device=device).unsqueeze(0) \
            .expand(ids.shape[0], -1)
        masks = engine.module.get_attention_masks(positions=positions)
        pred = engine(ids, positions=positions, attention_masks=masks)
        loss = step_loss(pred, labels)
        engine.backward(loss)
        if args.save_grads and i in (0, 1):
            torch.save({k: p.grad.detach().cpu()
                        for k, p in engine.module.named_parameters()
                        if p.grad is not None},
                       args.save_grads.replace(".pt", f"_s{i+1}.pt"))
        if args.clip == "manual":
            torch.nn.utils.clip_grad_norm_(engine.module.parameters(),
                                           args.max_norm, foreach=True)
        engine.step()
        if args.diag:
            lr = engine.get_lr()
            print(f"diag step={i+1} lr={lr} loss={float(loss.detach()):.6f}")
        loss_curve.append(float(loss.detach()))
        if t_first is None:
            t_first = time.perf_counter() - t_start
        del pred
    train_s = time.perf_counter() - t0

    if args.ckpt_dir and not args.resume:
        engine.save_checkpoint(args.ckpt_dir)

    # ZeRO-0/1 参数本就在每 rank 全量，可直接 checksum；2/3 由转换脚本另行合并
    param_checksum = None
    if args.zero <= 1:
        param_checksum = pc.sd_checksum(engine.module.state_dict())
    if args.save_params:
        torch.save({k: v.detach().cpu()
                    for k, v in engine.module.state_dict().items()},
                   args.save_params)

    ntokens = (args.steps - args.start_step) * pc.BATCH * pc.SEQ * world
    out = {
        "experiment": os.path.basename(args.out).replace(".json", ""),
        "framework": f"deepspeed-zero{args.zero}",
        "deepspeed": deepspeed.__version__,
        "init": init_mode,
        "world_size": world,
        "backend": os.environ.get("PARITY_BACKEND", "nccl"),
        "steps": args.steps,
        "loss_curve": loss_curve,
        "loss_first": loss_curve[0],
        "loss_last": loss_curve[-1],
        "param_checksum": param_checksum,
        "tokens_per_s": round(ntokens / train_s, 1),
        "train_seconds": round(train_s, 3),
        "start_step": args.start_step,
        "startup_to_first_step_s": round(t_first, 3),
        "peak_mem_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "labels": {"REAL": "两机真实运行" if world > 1 else "单机真实运行"},
    }
    if rank == 0:
        pc.save_json(out, args.out)
        print(f"[ds zero{args.zero} init={init_mode}] loss "
              f"{out['loss_first']:.4f} -> {out['loss_last']:.4f} "
              f"checksum {param_checksum} tokens/s {out['tokens_per_s']}")


if __name__ == "__main__":
    main()
