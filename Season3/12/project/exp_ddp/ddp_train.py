"""ddp_train.py —— torchrun 入口：single / ddp / gradsync / resume 四模式 + 故障注入。

用法（WSL 项目根目录）：
  # 单进程基线（同一套代码路径，world=1）
  ./.venv/bin/python -m exp_ddp.ddp_train --mode single --steps 30 --out <json>

  # 2-rank DDP（gloo，单卡 GPU 训练）
  ./.venv/bin/torchrun --nproc_per_node=2 --master_port=29520 \
      -m exp_ddp.ddp_train --mode ddp --steps 30 --out <json>

  # 梯度同步观测：backward 前后梯度对照
  ./.venv/bin/torchrun --nproc_per_node=2 --master_port=29520 \
      -m exp_ddp.ddp_train --mode gradsync --out <json>

  # 故障注入（详见 fail_modes_ddp.py）
  ... --mode ddp --fault no_sampler
  ... --mode ddp --fault global_batch_misconfig
  ... --mode ddp --fault sum_reduction
  ... --mode ddp --fault no_set_epoch
  ... --mode ddp --fault all_rank_save

等价判据：单卡与双卡用同一份固定样本池（FixedSampleDataset）、同一全局 batch、
同一初始化 seed、同一 shuffle 种子。DistributedSampler 的交错分片保证
"双卡第 k 步两个分片的并集"与"单卡第 k 步的 batch"是同一个样本集合，
mean reduction 的梯度因此逐位对齐（浮点求和顺序差异除外，容差 1e-4）。

10 篇的教训直接继承：
  - 每个输出带物理自检（checksum 长度、loss 有限性、梯度误差量级）
  - 故障注入必须自证生效（inject 字段记录注入的实际效果量）

AMP 说明：本篇全程 fp32。混合精度会引入 GradScaler 跳步等额外状态，
让"逐位对齐"的判据变模糊；正确性验证要在最干净的数值路径上做。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.reproducibility import set_all_seeds  # noqa: E402
from exp_hf.contract import contract_loss  # noqa: E402
from exp_scale.schedulers import cosine_with_warmup  # noqa: E402
from exp_ddp.ddp_common import (  # noqa: E402
    FixedSampleDataset, all_gather_values, all_reduce_mean, build_model,
    cleanup_dist, collect_grads, grad_max_abs_err, load_corpus_ids,
    param_checksum, param_l2, setup_dist)

ALL_FAULTS = ("none", "no_sampler", "global_batch_misconfig", "sum_reduction",
              "no_set_epoch", "all_rank_save")


# ---------------------------------------------------------------- 数据

def make_loaders(args, rank: int, world: int):
    """构造 train/val loader。

    正常路径：FixedSampleDataset（固定样本池）+ DistributedSampler（交错分片）。
    样本池大小 = epoch_steps × global_batch；epoch_steps < steps 时会跑多个
    epoch（no_set_epoch 故障需要多 epoch 才显形）。

    no_sampler 故障：不用 DistributedSampler，每个 rank 独立 shuffle 全池。
    两个 rank 都会遍历全部样本，每个样本每 epoch 被训练 world 次，
    等效全局 batch 语义被破坏。
    """
    ids, vocab = load_corpus_ids(args.data_dir)
    n_train = args.epoch_steps * args.global_batch
    train_ds = FixedSampleDataset(ids, args.seq, n_train, seed=args.data_seed)
    val_ds = FixedSampleDataset(ids, args.seq, args.val_samples,
                                seed=args.data_seed + 7)

    inject = {"fault": args.fault}
    per_rank_batch = args.global_batch // max(1, world)

    if args.fault == "global_batch_misconfig":
        # 最常见误配：把"全局 batch"当成"每卡 batch"，实际全局 = world × B。
        # 注入自证：记录实际的全局 batch 供核对。
        per_rank_batch = args.global_batch
        inject["effective_global_batch"] = args.global_batch * world
        inject["intended_global_batch"] = args.global_batch
    else:
        inject["effective_global_batch"] = per_rank_batch * world

    if args.fault == "no_sampler":
        g = torch.Generator().manual_seed(args.seed + rank)
        train_sampler = None
        train_loader = DataLoader(train_ds, batch_size=per_rank_batch,
                                  shuffle=True, drop_last=True,
                                  generator=g, num_workers=0)
        inject["sampler"] = "none (independent full-pool shuffle per rank)"
        inject["sample_duplication_factor"] = world
    else:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world, rank=rank,
            shuffle=True, seed=args.data_seed, drop_last=True)
        train_loader = DataLoader(train_ds, batch_size=per_rank_batch,
                                  sampler=train_sampler, drop_last=True,
                                  num_workers=0)
        inject["sampler"] = "DistributedSampler"

    val_loader = DataLoader(val_ds, batch_size=max(1, per_rank_batch),
                            shuffle=False, drop_last=True, num_workers=0)
    return train_loader, val_loader, train_sampler, vocab, inject


def loss_fn(model, x, y, reduction_fault_active: bool):
    """统一 loss 口径。sum_reduction 故障时改用 sum 且不除 world_size。

    正常：contract_loss = cross_entropy(mean)，与单卡逐位同口径。
    故障：sum reduction 让每 rank 梯度放大 (B/2) 倍，all-reduce mean 后
    仍比单卡大 (B/2) 倍——等效学习率爆炸，loss 轨迹立刻偏离。
    """
    out = model(x)
    if reduction_fault_active:
        lg = out.logits if hasattr(out, "logits") else out
        return nn.functional.cross_entropy(
            lg.reshape(-1, lg.shape[-1]), y.reshape(-1), reduction="sum")
    return contract_loss(out, y)


@torch.no_grad()
def estimate_val(model, val_loader, device: str, iters: int = 8) -> float:
    m = model.module if hasattr(model, "module") else model
    m.eval()
    losses = []
    for i, (x, y) in enumerate(val_loader):
        if i >= iters:
            break
        losses.append(contract_loss(m(x.to(device)), y.to(device)).item())
    m.train()
    return sum(losses) / max(1, len(losses))


# ---------------------------------------------------------------- 训练核心

def train_loop(args, rank: int, world: int, device: str):
    """single 与 ddp 共用同一训练循环。world=1 时即单进程基线。"""
    set_all_seeds(args.seed, deterministic=False)
    train_loader, val_loader, sampler, vocab, inject = make_loaders(
        args, rank, world)

    model = build_model(vocab, args.seq, device)
    pre_broadcast_checksum = param_checksum(model)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=0.1, betas=(0.9, 0.95))

    ddp_model = model
    if world > 1:
        ddp_model = nn.parallel.DistributedDataParallel(model)
    init_checksum = param_checksum(model)

    reduction_fault = args.fault == "sum_reduction"
    history = {"local_loss": [], "global_loss": [], "val_loss": [],
               "checksums": [], "per_rank_loss": []}
    t0 = time.perf_counter()
    step = 0
    epoch = 0
    if sampler is not None and args.fault != "no_set_epoch":
        sampler.set_epoch(epoch)
    data_iter = iter(train_loader)
    gn = torch.tensor(0.0)

    while step < args.steps:
        try:
            x, y = next(data_iter)
        except StopIteration:
            epoch += 1
            if sampler is not None and args.fault != "no_set_epoch":
                sampler.set_epoch(epoch)
            data_iter = iter(train_loader)
            x, y = next(data_iter)
        x, y = x.to(device), y.to(device)

        loss = loss_fn(ddp_model, x, y, reduction_fault)
        loss.backward()

        local_loss = loss.item()
        global_loss = all_reduce_mean(local_loss, device) if world > 1 else local_loss
        per_rank = all_gather_values(local_loss, device) if world > 1 else [local_loss]

        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr_now = cosine_with_warmup(step, args.schedule_total or args.steps,
                                    args.warmup, args.lr)
        for g in opt.param_groups:
            g["lr"] = lr_now
        opt.step()

        if step % args.record_every == 0:
            history["local_loss"].append({"step": step, "loss": round(local_loss, 6)})
            history["global_loss"].append({"step": step, "loss": round(global_loss, 6)})
            history["checksums"].append({"step": step,
                                         "checksum": param_checksum(model)})
            if world > 1:
                history["per_rank_loss"].append(
                    {"step": step, "losses": [round(v, 6) for v in per_rank]})

        opt.zero_grad(set_to_none=True)
        step += 1

        if args.eval_every > 0 and (step % args.eval_every == 0 or step == args.steps):
            vl = estimate_val(model, val_loader, device)
            vl_g = all_reduce_mean(vl, device) if world > 1 else vl
            history["val_loss"].append({"step": step, "loss": round(vl_g, 6)})

    wall = time.perf_counter() - t0

    # parity 用：把最终参数存盘，跨进程比对参数误差
    # （checksum 是哈希，差 1 ULP 就不同；误差量级才是诚实的判据）
    # 只由 rank0 写：所有 rank 同时写同一路径就是 all_rank_save 故障本身
    if args.save_params and rank == 0:
        pp = Path(args.save_params)
        pp.parent.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, pp)

    # checkpoint：rank0 保存是正确做法；all_rank_save 故障让所有 rank 同时写
    ckpt_info = {}
    if args.save_ckpt:
        ckpt_path = Path(args.ckpt_dir) / "ddp_ckpt.pt"
        # 目录先建好（多 rank 并发 mkdir + exist_ok 是安全的），
        # 否则 all_rank_save 会崩在"父目录不存在"而不是并发写冲突本身
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"model": model.state_dict(), "opt": opt.state_dict(),
                   "step": step, "checksum": param_checksum(model)}
        if args.fault == "all_rank_save":
            # 所有 rank 同时写同一路径：两个进程各自 open(truncate) 再写，
            # 写入流交错，文件可能半截损坏或混入两个 rank 的字节。
            # 不加 barrier、不分 rank，就是错误写法本身。
            torch.save(payload, ckpt_path)
            inject["ckpt_writers"] = world
        else:
            if rank == 0:
                torch.save(payload, ckpt_path)
            if world > 1:
                dist.barrier()   # 全 rank 等 rank0 写完再继续
            inject["ckpt_writers"] = 1
        ckpt_info = {"path": str(ckpt_path),
                     "exists": ckpt_path.exists(),
                     "size_bytes": ckpt_path.stat().st_size if ckpt_path.exists() else 0,
                     "loadable": _try_load(ckpt_path)}

    result = {
        "mode": args.mode,
        "fault": args.fault,
        "inject": inject,
        "backend": args.backend,
        "world_size": world,
        "rank": rank,
        "node_fingerprint": hashlib.sha256(socket.gethostname().encode()).hexdigest()[:16],
        "device": device,
        "gpu": torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu",
        "cuda_device_index": torch.cuda.current_device() if torch.cuda.is_available() else None,
        "corpus_sha256": hashlib.sha256(
            (Path(args.data_dir) / "corpus_large.txt").read_bytes()).hexdigest(),
        "torch": torch.__version__,
        "config": {"hidden": 384, "layers": 6, "heads": 6, "head_dim": 64,
                   "seq": args.seq, "global_batch": args.global_batch,
                   "steps": args.steps, "epoch_steps": args.epoch_steps,
                   "lr": args.lr, "warmup": args.warmup,
                   "data_seed": args.data_seed, "seed": args.seed},
        "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
        "pre_broadcast_checksum": pre_broadcast_checksum,
        "init_checksum": init_checksum,
        "final_checksum": param_checksum(model),
        "final_l2": round(param_l2(model), 4),
        "final_local_loss": history["local_loss"][-1]["loss"] if history["local_loss"] else None,
        "final_global_loss": history["global_loss"][-1]["loss"] if history["global_loss"] else None,
        "final_grad_norm": round(float(gn), 6),
        "epochs": epoch + 1,
        "wall_s": round(wall, 2),
        "history": history,
        "ckpt": ckpt_info,
    }
    if device == "cuda":
        result["peak_mb"] = round(
            torch.cuda.max_memory_allocated() / 1024 / 1024, 1)
    return result


def _try_load(path: Path) -> bool:
    """checkpoint 可加载性自检：半截文件在这里现形。"""
    try:
        torch.load(path, map_location="cpu", weights_only=False)
        return True
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------- gradsync 模式

def gradsync_probe(args, rank: int, world: int, device: str):
    """观测 backward 前后的梯度，证明同步发生在反向传播过程中。

    三条路径对照（ref 模型是独立实例，避免 DDP 的 autograd hook 干扰）：
      1. DDP backward：loss.backward() 返回后立刻抓梯度——若已等于全 batch
         梯度，说明 all-reduce 发生在 backward 过程中，不是 opt.step 时。
      2. 本 rank 分片的本地梯度（无 DDP）：all-reduce 前的样子，各 rank 不同。
      3. 单进程全 batch 梯度：等价目标。
    验证：manual all-reduce mean(路径2) == 路径3 == 路径1。
    """
    set_all_seeds(args.seed, deterministic=False)
    ids, vocab = load_corpus_ids(args.data_dir)
    B = args.global_batch
    ds = FixedSampleDataset(ids, args.seq, B, seed=args.data_seed)
    pairs = [ds[i] for i in range(B)]
    x_all = torch.stack([p[0] for p in pairs]).to(device)
    y_all = torch.stack([p[1] for p in pairs]).to(device)

    # 与 DistributedSampler 一致的交错分片：rank r 拿 indices[r::world]
    my_idx = list(range(rank, B, world))
    x_my, y_my = x_all[my_idx], y_all[my_idx]

    model = build_model(vocab, args.seq, device)
    ref = build_model(vocab, args.seq, device)
    # 两次 build_model 之间 RNG 已前进，ref 与 model 权重不同；
    # 必须显式复制再跨 rank 广播，否则三条路径的起点就不一致
    ref.load_state_dict(model.state_dict())
    if world > 1:
        for p in model.parameters():
            dist.broadcast(p.data, src=0)
        ref.load_state_dict(model.state_dict())

    # 路径 1：DDP backward，返回后立刻抓梯度
    ddp_model = nn.parallel.DistributedDataParallel(model) if world > 1 else model
    loss_local = contract_loss(ddp_model(x_my), y_my)
    loss_local.backward()
    grad_after_ddp = collect_grads(model)

    # 路径 2：本地分片梯度（无 DDP，all-reduce 前）
    ref.zero_grad(set_to_none=True)
    loss_ref_local = contract_loss(ref(x_my), y_my)
    loss_ref_local.backward()
    grad_local = collect_grads(ref)
    local_l2 = float(sum((g ** 2).sum() for g in grad_local.values()) ** 0.5)

    # 手工 all-reduce mean，验证等价数学
    grad_manual = {k: v.clone() for k, v in grad_local.items()}
    if world > 1:
        for k in grad_manual:
            dist.all_reduce(grad_manual[k], op=dist.ReduceOp.SUM)
            grad_manual[k] /= world

    # 路径 3：单进程全 batch 梯度（等价目标）
    ref.zero_grad(set_to_none=True)
    loss_full = contract_loss(ref(x_all), y_all)
    loss_full.backward()
    grad_single = collect_grads(ref)
    single_l2 = float(sum((g ** 2).sum() for g in grad_single.values()) ** 0.5)

    err_ddp = grad_max_abs_err(grad_after_ddp, grad_single)
    err_manual = grad_max_abs_err(grad_manual, grad_single)

    return {
        "mode": "gradsync",
        "rank": rank,
        "world_size": world,
        "backend": args.backend,
        "device": device,
        "shard_size": len(my_idx),
        "global_batch": B,
        "local_loss": round(loss_local.item(), 6),
        "full_batch_loss": round(loss_full.item(), 6),
        "grad_l2_local_only": round(local_l2, 6),
        "grad_l2_single_full": round(single_l2, 6),
        "grad_max_abs_err_ddp_vs_single": err_ddp,
        "grad_max_abs_err_manual_vs_single": err_manual,
        "ddp_equivalent": err_ddp < args.grad_tol,
        "manual_equivalent": err_manual < args.grad_tol,
        "grad_tol": args.grad_tol,
        "note": ("ddp_vs_single：DDP backward 返回后的梯度与单进程全 batch 梯度的"
                 "最大绝对误差，证明同步发生在 backward 中；manual_vs_single："
                 "手工 all-reduce mean 的对照。fp32 求和顺序不同，容差 1e-4。"),
    }


# ---------------------------------------------------------------- resume 模式

def resume_check(args, rank: int, world: int, device: str):
    """分布式恢复：从 rank0 存的 checkpoint 恢复，续跑并记录对齐证据。

    与"连续训练"的对照由 parity_check.py 统一比对，这里负责恢复与续跑。
    所有 rank 都加载同一份文件（模型与 optimizer 状态必须全 rank 一致，
    否则下一步 all-reduce 会把不一致放大）。
    """
    ckpt_path = Path(args.ckpt_dir) / "ddp_ckpt.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"checkpoint 不存在: {ckpt_path}（先跑 --mode ddp --save-ckpt）")

    st = torch.load(ckpt_path, map_location=device, weights_only=False)

    set_all_seeds(args.seed, deterministic=False)
    train_loader, val_loader, sampler, vocab, inject = make_loaders(
        args, rank, world)
    model = build_model(vocab, args.seq, device)
    model.load_state_dict(st["model"])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=0.1, betas=(0.9, 0.95))
    opt.load_state_dict(st["opt"])
    start_step = st["step"]
    loaded_checksum = param_checksum(model)

    ddp_model = nn.parallel.DistributedDataParallel(model) if world > 1 else model

    # 恢复数据位置：连续训练已经消耗了 start_step 个 batch，续跑必须
    # 从同一位置接着读，否则前 start_step 个 batch 会被重复训练，
    # 与连续训练的对照就不成立。sampler 的 shuffle 由 seed+epoch 决定，
    # 同 seed 同 epoch 下顺序确定，跳过 start_step 个 batch 即可对齐。
    schedule_total = args.schedule_total or (start_step + args.steps)
    step = 0
    epoch = 0
    if sampler is not None and args.fault != "no_set_epoch":
        sampler.set_epoch(epoch)
    data_iter = iter(train_loader)
    skipped = 0
    for _ in range(start_step):
        try:
            next(data_iter)
            skipped += 1
        except StopIteration:
            epoch += 1
            if sampler is not None and args.fault != "no_set_epoch":
                sampler.set_epoch(epoch)
            data_iter = iter(train_loader)
            next(data_iter)
            skipped += 1
    losses = []
    while step < args.steps:
        try:
            x, y = next(data_iter)
        except StopIteration:
            epoch += 1
            if sampler is not None and args.fault != "no_set_epoch":
                sampler.set_epoch(epoch)
            data_iter = iter(train_loader)
            x, y = next(data_iter)
        x, y = x.to(device), y.to(device)
        loss = contract_loss(ddp_model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr_now = cosine_with_warmup(start_step + step, schedule_total,
                                    args.warmup, args.lr)
        for g in opt.param_groups:
            g["lr"] = lr_now
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(round(loss.item(), 6))
        step += 1

    # parity 用：存最终参数，与连续训练做容差比对（checksum 差 1 ULP 就不同，
    # 误差量级才是诚实判据）。只由 rank0 写。
    if args.save_params and rank == 0:
        pp = Path(args.save_params)
        pp.parent.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, pp)

    return {
        "mode": "resume",
        "rank": rank,
        "world_size": world,
        "backend": args.backend,
        "ckpt_step": start_step,
        "ckpt_checksum": st.get("checksum"),
        "loaded_checksum": loaded_checksum,
        "load_matches_save": loaded_checksum == st.get("checksum"),
        "opt_state_loaded": True,
        "batches_skipped": skipped,
        "schedule_total": schedule_total,
        "resumed_steps": step,
        "final_checksum": param_checksum(model),
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "inject": inject,
    }


# ---------------------------------------------------------------- 入口

def run(args) -> dict:
    rank, world = 0, 1
    if args.mode != "single":
        rank, _local, world = setup_dist(args.backend)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.mode == "gradsync":
        out = gradsync_probe(args, rank, world, device)
    elif args.mode == "resume":
        out = resume_check(args, rank, world, device)
    else:   # single / ddp
        out = train_loop(args, rank, world, device)

    # 物理自检（10 篇教训：每个输出都要有边界检查）
    sanity = []
    ck = out.get("final_checksum")
    if ck and len(ck) != 16:
        sanity.append("checksum 长度异常")
    for k in ("final_local_loss", "final_global_loss", "loss_last"):
        v = out.get(k)
        if v is not None and v != v:   # NaN
            sanity.append(f"{k} 为 NaN")
    for k in ("grad_max_abs_err_ddp_vs_single", "grad_max_abs_err_manual_vs_single"):
        v = out.get(k)
        if v is not None and v > 1.0:
            sanity.append(f"{k}={v:.3g} 超过 1.0，等价性存疑")
    if out.get("ckpt", {}).get("exists") and not out["ckpt"].get("loadable"):
        sanity.append("checkpoint 存在但不可加载（半截文件）")
    if sanity:
        out["sanity_error"] = "; ".join(sanity)

    if args.out:
        p = Path(args.out)
        if rank != 0:
            p = p.with_name(f"{p.stem}_rank{rank}{p.suffix}")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                     encoding="utf-8")

    if args.mode != "single":
        cleanup_dist()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="ddp",
                    choices=("single", "ddp", "gradsync", "resume"))
    ap.add_argument("--backend", default="gloo", choices=("gloo", "nccl"))
    ap.add_argument("--fault", default="none", choices=ALL_FAULTS)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--epoch-steps", type=int, default=0,
                    help="一个 epoch 的步数（样本池 = epoch_steps × global_batch）；"
                         "0 表示等于 steps（单 epoch）。no_set_epoch 故障需要 "
                         "epoch_steps < steps 才显形")
    ap.add_argument("--global-batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-seed", type=int, default=1234)
    ap.add_argument("--val-samples", type=int, default=256)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--record-every", type=int, default=5)
    ap.add_argument("--grad-tol", type=float, default=1e-4)
    ap.add_argument("--schedule-total", type=int, default=0,
                    help="LR schedule 的总步数；0 表示用 --steps。分段跑与续跑"
                         "要用同一个总步数，否则与连续训练的 LR 轨迹对不上")
    ap.add_argument("--save-ckpt", action="store_true")
    ap.add_argument("--ckpt-dir", default="runs/ddp_ckpts")
    ap.add_argument("--save-params", default=None,
                    help="把最终参数存盘（.pt），parity_check 跨进程比对参数误差用")
    ap.add_argument("--data-dir", default="exp_scale/data")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.epoch_steps <= 0:
        args.epoch_steps = args.steps
    run(args)


if __name__ == "__main__":
    main()
