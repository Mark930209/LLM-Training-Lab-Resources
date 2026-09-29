"""wrap_units.py —— E4/E5：FSDP 风格按单元分片的参数生命周期与 wrap sweep。

zero_levels 的 zero3 是"整模型一个单元"（最粗 wrap）。本模块把参数按单元
（embed / 每个 decoder block / final norm）分别分片，物化/释放按单元走
forward hook，从而 sweep 三个旋钮：

    --wrap model|block    分片单元粒度：整模型一个单元 vs 每 block 一个单元
    --reshard-after-forward 0|1
                          1：该单元 forward 后立即释放满参（FSDP 默认行为），
                             backward 用到时经 saved_tensors_hooks 重取；
                          0：满参保留到 step 结束（省一次 backward gather，
                             代价是 forward 结束时全部单元已物化 = 峰值更高）
    --prefetch N          在 unit i 的 pre-hook 提前物化 i+1..i+N（内存效果
                             等价；不做计算/通信 overlap，overlap 是 14 篇话题）

核心观测量：峰值显存、参数驻留时长（生命周期事件）、AllGather 次数。
E4 的生命周期日志：每个 unit 记录 gather/release/grad_reduced 事件与
local shape（lo,hi）/full numel，写进结果 JSON 的 `lifecycle` 字段。

梯度路径沿用 zero_levels 的生产形态：p.grad 预绑定 unit 梯度缓冲视图，
post_accumulate_grad_hook 计数，该单元梯度齐了立即 reduce_scatter 到属主。
reshard_after_forward=1 时 backward 的 saved tensor 只存（unit, offset, shape）
占位，unpack 时重取满参——这是"满参在 step 内何时存在"的完整生命周期。

用法（torchrun，两机 NCCL 或单机 gloo 预检）：
    PYTHONPATH=. torchrun --nproc_per_node=1 exp_shard/wrap_units.py \
        --wrap block --reshard-after-forward 1 --prefetch 1 \
        --steps 5 --out results/fsdp_sweep.json

容量实验两机各起一个 rank 并加 --mem-fraction 0.9（WSL 超配会把"放不下"
伪装成溢出降速，22 篇先例）。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exp_shard.shard_common import (  # noqa: E402
    StateAccount, StageProbe, count_params, run_meta)


# ---------------------------------------------------------------- 通信记账


class CommLedger:
    """collective 字节对账（13 篇算法口径，S=张量字节）。"""

    def __init__(self, world: int):
        self.world = max(1, world)
        self.rows: list[dict] = []

    def add(self, kind: str, nbytes: int) -> None:
        n = self.world
        factor = {"all_reduce": 2 * (n - 1) / n,
                  "reduce_scatter": (n - 1) / n,
                  "all_gather": (n - 1) / n}[kind]
        self.rows.append({"kind": kind, "bytes": int(nbytes * factor)})

    def as_dict(self) -> dict:
        total = sum(r["bytes"] for r in self.rows)
        by_kind: dict[str, int] = {}
        for r in self.rows:
            by_kind[r["kind"]] = by_kind.get(r["kind"], 0) + r["bytes"]
        return {"total_bytes": total, "by_kind": by_kind,
                "n_calls": len(self.rows)}


def shard_bounds(n_elem: int, world: int) -> list[tuple[int, int]]:
    chunk = math.ceil(n_elem / world)
    return [(r * chunk, min(n_elem, (r + 1) * chunk)) for r in range(world)]


# ---------------------------------------------------------------- 单元


class Unit:
    """一个分片单元：属主区间 [lo,hi) 的 shard 常驻，满参按需物化。"""

    def __init__(self, name: str, params: list[nn.Parameter],
                 shapes: list[tuple[int, ...]], world: int, rank: int,
                 device: str):
        self.name = name
        self.params = params
        self.shapes = shapes
        self.numels = [math.prod(s) for s in shapes]
        self.n_elem = sum(self.numels)
        self.bounds = shard_bounds(self.n_elem, world)
        self.chunk = self.bounds[0][1] - self.bounds[0][0]
        self.padded = self.chunk * world
        self.lo, self.hi = self.bounds[rank]
        self.world, self.rank = world, rank
        self.device = device
        uflat = torch.cat([p.detach().reshape(-1) for p in params])
        self.shard = nn.Parameter(uflat[self.lo:self.hi].clone())
        self.gathered: torch.Tensor | None = None
        self.grad_buf: torch.Tensor | None = None
        self.grad_pending = 0
        del uflat

    def materialized(self) -> bool:
        return self.gathered is not None

    def gather(self, comm) -> None:
        if self.materialized():
            return
        dtype = self.shard.dtype
        buf = torch.zeros(self.padded, dtype=dtype, device=self.device)
        mine = torch.zeros(self.chunk, dtype=dtype, device=self.device)
        mine[:self.hi - self.lo] = self.shard.detach()
        if self.world > 1:
            comm.add("all_gather", self.chunk * mine.element_size()
                     * self.world)
            dist.all_gather_into_tensor(buf, mine)
        else:
            buf[self.lo:self.hi] = mine[:self.hi - self.lo]
        self.gathered = buf[:self.n_elem]
        off = 0
        for p, shape in zip(self.params, self.shapes):
            n = math.prod(shape)
            p.data = self.gathered[off:off + n].view(shape)
            off += n
        if self.grad_buf is None:   # backward 重取时不重绑（会丢已累加梯度）
            self._bind_grads()
        del buf, mine

    def _bind_grads(self) -> None:
        self.grad_buf = torch.zeros(self.n_elem, dtype=self.shard.dtype,
                                    device=self.device)
        self.grad_pending = len(self.params)
        off = 0
        for p, shape in zip(self.params, self.shapes):
            n = math.prod(shape)
            p.grad = self.grad_buf[off:off + n].view(shape)
            off += n

    def release(self) -> None:
        """释放满参（saved tensor 是占位时才安全）。"""
        if not self.materialized():
            return
        for p in self.params:
            p.data = torch.empty(0, dtype=p.dtype, device=p.device)
        self.gathered = None

    def reduce_grads(self, comm) -> None:
        if self.grad_buf is None:
            return
        if self.world > 1:
            recv = torch.empty(self.chunk, dtype=self.grad_buf.dtype,
                               device=self.device)
            pad = self.padded - self.n_elem
            flat = (torch.cat([self.grad_buf,
                               self.grad_buf.new_zeros(pad)]) if pad > 0
                    else self.grad_buf)
            comm.add("reduce_scatter", self.n_elem
                     * self.grad_buf.element_size())
            dist.reduce_scatter_tensor(recv, flat, op=dist.ReduceOp.SUM)
            self.shard.grad = recv[:self.hi - self.lo] / self.world
            del recv, flat
        else:
            self.shard.grad = self.grad_buf[self.lo:self.hi].clone()

    def cleanup(self) -> None:
        for p in self.params:
            p.grad = None
        self.grad_buf = None
        self.release()


# ---------------------------------------------------------------- 状态管理


class WrapState:
    """按单元分片的训练状态与生命周期调度。"""

    def __init__(self, model: nn.Module, wrap: str, reshard: bool,
                 prefetch: int, world: int, rank: int, lr: float,
                 device: str, comm: "CommLedger", log: list):
        self.wrap, self.reshard, self.prefetch = wrap, reshard, prefetch
        self.world, self.rank = world, rank
        self.comm = comm
        self.log = log
        self.step_no = -1
        self.max_materialized_mb = 0.0

        # ---- 单元划分（tied embedding 归 embed 单元，lm_head 是释放边界）
        m = model.model if hasattr(model, "model") else model
        embed = m.embed_tokens
        layers = list(m.layers)
        norm = m.norm
        lm_head = model.lm_head if hasattr(model, "lm_head") else None
        groups: list[tuple[str, list[nn.Module]]] = [("embed", [embed])]
        for i, lyr in enumerate(layers):
            groups.append((f"layer{i}", [lyr]))
        groups.append(("final", [norm]))
        release_mod: dict[str, nn.Module] = {n: mods[-1] for n, mods in groups}
        if lm_head is not None:
            groups[0] = ("embed", [embed, lm_head])
            release_mod["embed"] = lm_head

        seen: set[int] = set()
        self.units: list[Unit] = []
        mod_unit: dict[int, Unit] = {}
        for name, mods in groups:
            params, shapes = [], []
            for mod in mods:
                for p in mod.parameters(recurse=True):
                    if id(p) not in seen:
                        seen.add(id(p))
                        params.append(p)
                        shapes.append(tuple(p.shape))
            if params:
                u = Unit(name, params, shapes, world, rank, device)
                self.units.append(u)
                for mod in mods:
                    mod_unit[id(mod)] = u
        self.by_name = {u.name: u for u in self.units}

        # ---- hook 注册（wrap=model 时不用模块 hook，step 前后整体调度）
        if wrap == "block":
            for name, mods in groups:
                u = self.by_name.get(name)
                if u is None:
                    continue
                mods[0].register_forward_pre_hook(self._make_pre(u))
                release_mod[name].register_forward_hook(self._make_post(u))
            for u in self.units:
                for p in u.params:
                    p.register_post_accumulate_grad_hook(
                        self._make_grad_hook(u))

        self.opt = torch.optim.AdamW([u.shard for u in self.units], lr=lr)

    # ---- hooks

    def _make_pre(self, unit: Unit):
        def pre(module, args):
            unit.gather(self.comm)
            self._record("gather", unit)
            if self.prefetch > 0:
                idx = self.units.index(unit)
                for j in range(idx + 1,
                               min(len(self.units), idx + 1 + self.prefetch)):
                    self.units[j].gather(self.comm)
                    self._record("prefetch_gather", self.units[j])
            return None
        return pre

    def _make_post(self, unit: Unit):
        def post(module, args, output):
            if self.reshard:
                unit.release()
                self._record("release", unit)
            return None
        return post

    def _make_grad_hook(self, unit: Unit):
        def hook(p):
            unit.grad_pending -= 1
            if unit.grad_pending <= 0 and unit.grad_buf is not None:
                unit.reduce_grads(self.comm)
                self._record("grad_reduced", unit)
                if self.reshard:
                    unit.release()
                    self._record("release_after_bwd", unit)
            return None
        return hook

    def _record(self, event: str, unit: Unit) -> None:
        mat = sum(u.n_elem for u in self.units
                  if u.materialized()) * 2 / 1024 / 1024
        self.max_materialized_mb = max(self.max_materialized_mb, mat)
        if self.step_no < 3:
            self.log.append({
                "step": self.step_no, "unit": unit.name, "event": event,
                "shard_lo": unit.lo, "shard_hi": unit.hi,
                "full_elems": unit.n_elem,
                "materialized_mb": round(mat, 1),
                "allocated_mb": round(
                    torch.cuda.memory_allocated() / 1024 / 1024, 1)
                    if torch.cuda.is_available() else 0.0,
            })

    # ---- step 生命周期

    def begin_step(self) -> None:
        self.step_no += 1
        if self.wrap == "model":
            for u in self.units:
                u.gather(self.comm)
                self._record("gather", u)

    def end_step(self) -> None:
        # wrap=model 无 grad hook；wrap=block 的 hook 可能因未使用参数未触发
        for u in self.units:
            if u.grad_buf is not None and u.shard.grad is None:
                u.reduce_grads(self.comm)
                self._record("grad_reduced", u)
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        for u in self.units:
            u.cleanup()
            self._record("cleanup", u)

    # ---- saved tensor 占位（reshard=1 时让满参真正离开显存）

    def pack(self, t: torch.Tensor):
        if self.reshard and self.wrap == "block":
            ptr = t.data_ptr()
            for i, u in enumerate(self.units):
                g = u.gathered
                if g is not None and g.data_ptr() <= ptr < g.data_ptr() \
                        + g.numel() * g.element_size():
                    off = (ptr - g.data_ptr()) // g.element_size()
                    # 必须记 stride：F.linear 保存的是 weight 的转置视图，
                    # 按连续张量重建会把元素顺序弄错（parity 教训）
                    return (i, off, tuple(t.shape), tuple(t.stride()))
        return t

    def unpack(self, o):
        if isinstance(o, tuple):
            i, off, shape, stride = o
            u = self.units[i]
            if not u.materialized():
                u.gather(self.comm)
                self._record("re_gather_bwd", u)
            return torch.as_strided(u.gathered, size=shape, stride=stride,
                                    storage_offset=off)
        return o


# ---------------------------------------------------------------- 主流程


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wrap", default="block", choices=("model", "block"))
    ap.add_argument("--reshard-after-forward", type=int, default=1)
    ap.add_argument("--prefetch", type=int, default=0)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=1536)
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--inter", type=int, default=4096)
    ap.add_argument("--vocab", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--backend", default="nccl")
    ap.add_argument("--data-dir", default="exp_scale/data")
    ap.add_argument("--mem-fraction", type=float, default=0.0)
    ap.add_argument("--out", default="results/fsdp_sweep.json")
    args = ap.parse_args()

    if args.mem_fraction > 0 and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.mem_fraction)
    if not dist.is_initialized():
        dist.init_process_group(backend=args.backend)
    rank = dist.get_rank()
    world = dist.get_world_size()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.set_device(
            int(os.environ.get("LOCAL_RANK", 0)) % torch.cuda.device_count())

    torch.manual_seed(args.seed)
    from exp_ddp.ddp_common import FixedSampleDataset, load_corpus_ids
    ids, vocab_real = load_corpus_ids(args.data_dir)
    vocab = vocab_real if args.vocab <= 0 else args.vocab
    ds = FixedSampleDataset(ids, args.seq,
                            n_samples=args.steps * args.batch * world,
                            seed=args.seed)
    subset = torch.utils.data.Subset(ds, list(range(rank, len(ds), world)))
    loader = torch.utils.data.DataLoader(subset, batch_size=args.batch,
                                         shuffle=False)

    from exp_hf.adapters import build_llama
    model = build_llama(vocab, hidden=args.hidden, layers=args.layers,
                        heads=max(1, args.hidden // 64), head_dim=64,
                        intermediate=args.inter,
                        max_seq_len=max(args.seq, 512)
                        ).to(device).to(torch.bfloat16)

    n_params = count_params(model)
    comm = CommLedger(world)
    lifecycle: list[dict] = []
    state = WrapState(model, args.wrap, bool(args.reshard_after_forward),
                      args.prefetch, world, rank, args.lr, device, comm,
                      lifecycle)

    probe = StageProbe(device=device)
    losses = []
    oom_info = None
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    probe.snap("model_ready")
    t0 = time.perf_counter()
    n_tokens = 0
    try:
        for step, (x, t) in enumerate(loader):
            x, t = x.to(device), t.to(device)
            state.begin_step()
            hooks = (torch.autograd.graph.saved_tensors_hooks(
                state.pack, state.unpack)
                if args.wrap == "block" and args.reshard_after_forward
                else None)
            ctx = hooks if hooks is not None else _NullCtx()
            with ctx:
                out = model(x, labels=t)
                out.loss.backward()
            state.end_step()
            losses.append(float(out.loss.item()))
            n_tokens += x.numel()
            if step == 0:
                probe.snap("after_first_step")
    except torch.cuda.OutOfMemoryError:
        oom_info = traceback.format_exc(limit=6)
        probe.snap("OOM")

    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    out = {
        "mode": "fsdp_wrap_sweep",
        "wrap": args.wrap,
        "reshard_after_forward": bool(args.reshard_after_forward),
        "prefetch": args.prefetch,
        "steps": args.steps,
        "world_size": world,
        "oom": oom_info is not None,
        "oom_traceback": oom_info or "",
        "config": vars(args),
        "n_params": n_params,
        "n_units": len(state.units),
        "losses": losses,
        "final_loss": losses[-1] if losses else None,
        "resident_mb": probe.records[-1]["allocated_mb"],
        "peak_mb": probe.records[-1]["peak_mb"],
        "max_materialized_mb": round(state.max_materialized_mb, 1),
        "comm": comm.as_dict(),
        "state_account": StateAccount(
            n_params, world, 8.0, True, True, True).as_dict(),
        "tok_per_s": round(n_tokens / dt, 1) if dt > 0 else 0,
        "stages": probe.records,
        "lifecycle": lifecycle,
        "meta": run_meta(rank, world, vars(args)),
    }
    out_path = (args.out if rank == 0 else
                args.out.replace(".json", f"_rank{rank}.json"))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    if rank == 0:
        tag = "OOM" if out["oom"] else f"loss={out['final_loss']:.4f}"
        print(f"[fsdp {args.wrap} r{args.reshard_after_forward} "
              f"p{args.prefetch}] world={world} {tag} "
              f"resident={out['resident_mb']} peak={out['peak_mb']} "
              f"mat={out['max_materialized_mb']} "
              f"comm={out['comm']['total_bytes'] / 1024 / 1024:.1f} MiB "
              f"tok/s={out['tok_per_s']}")
        print(f"done -> {out_path}")
    dist.destroy_process_group()
    return 2 if oom_info is not None else 0


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


if __name__ == "__main__":
    raise SystemExit(main())
