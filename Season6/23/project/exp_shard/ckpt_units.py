"""ckpt_units.py —— E8：sharded vs full checkpoint 的保存/恢复与世界大小可移植性。

同一模型、同一 run，四件事：

1. save            全量 checkpoint（rank0 汇总 flat 参数 + config + shapes）
                   与分片 checkpoint（每 rank 的 unit shards + manifest），
                   记录写入时间与体积（两类都只含权重；优化器状态沿用同一
                   unit 布局，本文不展开）；
2. load-full       全量恢复（任意 world size），校验 checksum；
3. load-sharded    分片恢复：manifest 记录 unit 形状与 world_size，
                   跨 world size（2→1）按 rank 顺序拼接即可重组；
4. load-sharded-nometa
                   失败案例 4：缺 manifest 时按参数形状直读 shard →
                   形状报错，证明元数据是分片 checkpoint 的一部分。

checksum = sha256(flat 参数原始字节)。实测环境为单机（序列化为 CPU
主导，跨机传输时间不计入），world=2 gloo 双进程真实分片。

用法（rank0 WSL）：
    PYTHONPATH=. torchrun --nproc_per_node=2 exp_shard/ckpt_units.py --mode save
    PYTHONPATH=. python exp_shard/ckpt_units.py --mode load-full
    PYTHONPATH=. python exp_shard/ckpt_units.py --mode load-sharded
    PYTHONPATH=. python exp_shard/ckpt_units.py --mode load-sharded-nometa
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exp_shard.wrap_units import CommLedger, WrapState  # noqa: E402


def build_model(vocab: int, args):
    from exp_hf.adapters import build_llama
    return build_llama(vocab, hidden=args.hidden, layers=args.layers,
                       heads=max(1, args.hidden // 64), head_dim=64,
                       intermediate=args.inter,
                       max_seq_len=max(args.seq, 512)
                       ).to("cpu").to(torch.bfloat16)


def flat_checksum(flat: torch.Tensor) -> str:
    raw = flat.contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()[:16]


def gather_full_flat(state: WrapState, world: int, device: str) -> torch.Tensor:
    """把各 rank 的 unit shards 拼成完整 flat 参数向量。"""
    parts = []
    for u in state.units:
        mine = torch.zeros(u.chunk, dtype=u.shard.dtype, device=device)
        mine[:u.hi - u.lo] = u.shard.detach()
        if world > 1:
            buf = torch.zeros(u.padded, dtype=u.shard.dtype, device=device)
            dist.all_gather_into_tensor(buf, mine)
            parts.append(buf[:u.n_elem].cpu())
        else:
            parts.append(mine[:u.n_elem].cpu())
        del mine
    return torch.cat(parts)


def unit_manifest(state: WrapState) -> list[dict]:
    return [{"name": u.name, "n_elem": u.n_elem, "shapes": u.shapes,
             "numels": u.numels, "chunk": u.chunk} for u in state.units]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=("save", "load-full", "load-sharded",
                             "load-sharded-nometa"))
    ap.add_argument("--ckpt-dir", default="runs/ckpt23")
    ap.add_argument("--steps", type=int, default=1, help="save 前的训练步数")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=1536)
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--inter", type=int, default=4096)
    ap.add_argument("--vocab", type=int, default=5120)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--backend", default="gloo")
    ap.add_argument("--out", default="results/Season6/23/ckpt_report.json")
    args = ap.parse_args()

    world, rank = 1, 0
    if args.mode == "save" and not dist.is_initialized():
        dist.init_process_group(backend=args.backend)
        world, rank = dist.get_world_size(), dist.get_rank()
    device = "cpu"          # 序列化实验以 CPU 为主导，避免设备噪声
    ckpt = Path(args.ckpt_dir)
    report: dict = {"mode": args.mode, "world_size": world,
                    "config": vars(args)}

    torch.manual_seed(args.seed)
    model = build_model(args.vocab, args)
    comm = CommLedger(world)
    state = WrapState(model, "block", True, 0, world, rank, 1e-4, device,
                      comm, [])

    if args.mode == "save":
        # 训练几步让 run 真实（loss 轨迹留档）
        losses = []
        x = torch.randint(0, args.vocab, (args.batch, args.seq))
        for _ in range(args.steps):
            state.begin_step()
            with torch.autograd.graph.saved_tensors_hooks(state.pack,
                                                          state.unpack):
                out = model(x, labels=x)
                out.loss.backward()
            state.end_step()
            losses.append(round(float(out.loss.item()), 4))

        t0 = time.perf_counter()
        full_flat = gather_full_flat(state, world, device)
        t_gather = time.perf_counter() - t0
        checksum = flat_checksum(full_flat)
        manifest = {"format": "exp_shard_ckpt_v1", "world_size": world,
                    "n_params": int(full_flat.numel()), "checksum": checksum,
                    "units": unit_manifest(state), "losses": losses}

        t0 = time.perf_counter()
        if rank == 0:
            ckpt.mkdir(parents=True, exist_ok=True)
            torch.save({"flat": full_flat, "manifest": manifest},
                       ckpt / "full.pt")
        t_full = time.perf_counter() - t0
        t0 = time.perf_counter()
        shard_payload = {"manifest": manifest,
                         "unit_shards": {u.name: u.shard.detach().cpu()
                                         for u in state.units}}
        torch.save(shard_payload, ckpt / f"shard_rank{rank}.pt")
        t_shard = time.perf_counter() - t0
        if rank == 0:
            (ckpt / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8")
        if world > 1:
            dist.barrier()      # 等 rank1 的 shard 文件写完再统计
        report.update({
            "checksum": checksum, "losses": losses,
            "gather_s": round(t_gather, 3), "save_full_s": round(t_full, 3),
            "save_shard_s": round(t_shard, 3),
            "n_units": len(state.units),
        })
        if rank == 0:
            report["size_full_bytes"] = (ckpt / "full.pt").stat().st_size
            report["size_shard_total_bytes"] = sum(
                (ckpt / f"shard_rank{r}.pt").stat().st_size
                for r in range(world))

    elif args.mode in ("load-full", "load-sharded", "load-sharded-nometa"):
        manifest = json.loads((ckpt / "manifest.json").read_text(
            encoding="utf-8"))
        try:
            t0 = time.perf_counter()
            if args.mode == "load-full":
                payload = torch.load(ckpt / "full.pt", weights_only=False)
                flat = payload["flat"]
            elif args.mode == "load-sharded":
                parts_by_unit: dict[str, list] = {}
                for r in range(manifest["world_size"]):
                    sd = torch.load(ckpt / f"shard_rank{r}.pt",
                                    weights_only=False)
                    for name, t in sd["unit_shards"].items():
                        parts_by_unit.setdefault(name, []).append(t)
                ushards = []
                for u in manifest["units"]:
                    vec = torch.cat(parts_by_unit[u["name"]])[:u["n_elem"]]
                    ushards.append(vec)
                flat = torch.cat(ushards)
            else:
                # 失败案例 4：没有 manifest 时按"这是完整参数"直读 shard
                sd = torch.load(ckpt / "shard_rank0.pt", weights_only=False)
                # 分片张量冒充完整参数向量 → 直接灌回模型时形状必然不匹配
                flat = torch.cat([t.reshape(-1)
                                  for t in sd["unit_shards"].values()])
                off = 0
                for p in model.parameters():
                    n = p.numel()
                    p.data = flat[off:off + n].view(p.shape)
                    off += n
            t_load = time.perf_counter() - t0
            checksum = flat_checksum(flat)
            report.update({
                "load_s": round(t_load, 3), "checksum": checksum,
                "checksum_match": checksum == manifest["checksum"],
                "world_from": manifest["world_size"], "world_to": world,
                "portable": True, "error": "",
            })
        except Exception:
            report.update({
                "load_s": 0.0, "checksum_match": False,
                "world_from": manifest["world_size"], "world_to": world,
                "portable": False,
                "error": traceback.format_exc(limit=4).strip().splitlines()[-1],
            })

    if rank == 0:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        prev = []
        if Path(args.out).exists():
            prev = json.loads(Path(args.out).read_text(encoding="utf-8"))
            if isinstance(prev, dict):
                prev = [prev]
        prev = [p for p in prev if p.get("mode") != args.mode] + [report]
        Path(args.out).write_text(json.dumps(prev, ensure_ascii=False,
                                             indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in report.items()
                          if k != "config"}, ensure_ascii=False))
        print(f"done -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
