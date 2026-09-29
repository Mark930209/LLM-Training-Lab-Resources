"""zero_levels.py —— E3：ZeRO 逐级分片的实现与对照（每一级只多切一份）。

四个级别共用一个训练循环，差别只在三件事：梯度怎么归约、优化器状态在谁
手里、参数平时以什么形态存在：

    ddp     梯度全量 AllReduce（平均）后全量 AdamW；参数、梯度、优化器
            状态全量常驻（切 0 份）
    zero1   梯度全量 AllReduce，AdamW 只持有本地参数分片的 m/v（切优化器
            状态）；step 后 AllGather 更新分片（切 1 份）
    zero2   梯度逐 chunk 归约到属主 rank、其余 rank 即时释放（切梯度）；
            AdamW 同 zero1（切 2 份）
    zero3   参数平时只存分片；forward 前整模型 AllGather 满参、step 后
            满参失效（切 3 份）。**整模型粒度 gather 是最粗 wrap**，是
            E5 wrap/prefetch sweep 的对照组：常驻最低但瞬时峰值更高。

梯度归约的时机决定峰值形态（本实现的教学重点之一）：

- ddp/zero1 的梯度语义就是"全量"，用展平缓冲 flat_grad 预绑定 p.grad
  （学 DDP gradient_as_bucket_view），backward 直接累加进缓冲，每步
  不再分配、不再逐参数收集；
- zero2/zero3 的梯度语义是"分片"，用 post_accumulate_grad_hook 在每个
  参数梯度就绪时立即抄进所属 chunk 的暂存并释放 p.grad；chunk 齐了立刻
  dist.reduce 到属主 rank（SUM），属主除以 world 后存进 grad_shard。
  backward 期间只常驻 1~2 个 chunk 暂存 + 本地分片，而不是全量梯度
  ——生产 ZeRO-2 的峰值形态。collective 字节与整段 reduce_scatter 等价。

分片按展平参数向量的连续区间划分（每 rank 一个 chunk）。sharded AdamW
用 torch.optim.AdamW 作用在分片张量上：AdamW 更新逐元素独立，分片作用
与全量作用逐元素一致。

zero1/zero2 的 shard 是 flat_param 本地区间的视图（零拷贝，优化器原地
更新直接生效）；zero3 的 flat_param 每 step 临时存在（gather 生命期 =
一个 step），shard 用独立缓冲。

数据语义沿用 12 篇：FixedSampleDataset 固定样本池，indices[rank::world]
分片，并集 = 全局 batch；loss 本地 mean，梯度归约求和后除以 world。

collective 字节按 13 篇口径记账（S=通信张量字节，N=world）：
    all_reduce      ring 2(N-1)/N·S
    reduce_scatter  (N-1)/N·S（逐 chunk reduce 的字节总量与之等价）
    all_gather      (N-1)/N·S

用法（torchrun，两机 NCCL 或单机 gloo 预检）：
    PYTHONPATH=. torchrun --nproc_per_node=1 exp_shard/zero_levels.py \
        --level zero2 --steps 20 --out results/zero_zero2.json

容量实验在两机上各起一个 rank（--nnodes=2 --node_rank=$RANK），并加
--mem-fraction 0.9 把 CUDA 分配器预算钉在物理显存：WSL 超配会把"放不下"
伪装成溢出降速（22 篇 tps 1340→408 先例），钉预算才拿得到真实 OOM 对照。
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


# ---------------------------------------------------------------- 分片状态


def shard_bounds(n_elem: int, world: int) -> list[tuple[int, int]]:
    """每 rank 的 [lo, hi) 分片边界；pad 到 world 整除保证等长。"""
    chunk = math.ceil(n_elem / world)
    return [(r * chunk, min(n_elem, (r + 1) * chunk)) for r in range(world)]


class ShardedState:
    """分片训练状态与逐级通信路径。

    状态三件套：参数（flat_param 或独立 shard）、梯度（flat_grad 绑定或
    chunk 暂存 + grad_shard）、优化器（全量或分片 AdamW）。
    """

    def __init__(self, model: nn.Module, level: str, world: int, rank: int,
                 lr: float, device: str, comm: "CommLedger"):
        self.level, self.world, self.rank = level, world, rank
        self.comm = comm
        self.params = [p for p in model.parameters()]
        self._shapes = [tuple(p.shape) for p in self.params]
        self.n_elem = sum(p.numel() for p in self.params)
        self.bounds = shard_bounds(self.n_elem, world)
        self.chunk = self.bounds[0][1] - self.bounds[0][0]
        self.padded = self.chunk * world
        self.lo, self.hi = self.bounds[rank]
        dtype, self.device = self.params[0].dtype, device

        flat0 = torch.nn.utils.parameters_to_vector(
            [p.detach() for p in self.params]).to(dtype)

        # ---- 参数侧
        self._flat_param: torch.Tensor | None = None
        if level in ("zero1", "zero2"):
            # 持久满参：参数即缓冲视图；shard 是本地区间视图（零拷贝）
            self._flat_param = torch.zeros(
                self.padded, dtype=dtype, device=device)
            self._flat_param[:self.n_elem] = flat0
            torch.nn.utils.vector_to_parameters(
                self._flat_param[:self.n_elem], self.params)
            self.shard = nn.Parameter(self._flat_param[self.lo:self.hi])
        else:
            self.shard = nn.Parameter(flat0[self.lo:self.hi].clone())

        self.opt = (torch.optim.AdamW(self.params, lr=lr) if level == "ddp"
                    else torch.optim.AdamW([self.shard], lr=lr))

        # ---- 梯度侧
        self.grad_shard: torch.Tensor | None = None
        self._flat_grad: torch.Tensor | None = None
        self._chunk_buf: dict[int, torch.Tensor] = {}
        self._chunk_filled: dict[int, int] = {}
        self._chunk_expect: dict[int, int] = {}
        self._param_off: dict[int, int] = {}
        if level in ("zero2", "zero3"):
            self.grad_shard = torch.zeros(
                self.chunk, dtype=dtype, device=device)
            off = 0
            for p in self.params:
                self._param_off[id(p)] = off
                off += p.numel()
                p.register_post_accumulate_grad_hook(self._grad_hook)
            for c in range(math.ceil(self.n_elem / self.chunk)):
                lo_c = c * self.chunk
                self._chunk_expect[c] = min(self.chunk,
                                            self.n_elem - lo_c)
        else:
            self._flat_grad = torch.zeros(
                self.padded, dtype=dtype, device=device)
            self._bind_grads()
        del flat0

    # ---- chunk 归约（zero2/zero3）

    def _grad_hook(self, p: nn.Parameter) -> None:
        """参数梯度就绪：抄进所属 chunk 暂存并立即释放 p.grad。"""
        g = p.grad
        if g is None:
            return
        off = self._param_off[id(p)]
        flat = g.reshape(-1)
        for c in range(off // self.chunk,
                       (off + p.numel() - 1) // self.chunk + 1):
            lo_c = c * self.chunk
            hi_c = min(self.n_elem, lo_c + self.chunk)
            s, e = max(off, lo_c), min(off + p.numel(), hi_c)
            if e <= s:
                continue
            buf = self._chunk_buf.get(c)
            if buf is None:
                buf = torch.zeros(self.chunk, dtype=flat.dtype,
                                  device=flat.device)
                self._chunk_buf[c] = buf
            buf[s - lo_c:e - lo_c] = flat[s - off:e - off]
            self._chunk_filled[c] = self._chunk_filled.get(c, 0) + (e - s)
            if self._chunk_filled[c] >= self._chunk_expect[c]:
                self._finish_chunk(c)
        p.grad = None

    def _finish_chunk(self, c: int) -> None:
        buf = self._chunk_buf.pop(c)
        if self.world > 1:
            self.comm.add("reduce_scatter",
                          self._chunk_expect[c] * buf.element_size())
            dist.reduce(buf, dst=c, op=dist.ReduceOp.SUM)
        if self.rank == c:
            n = self._chunk_expect[c]
            self.grad_shard[:n] = buf[:n] / self.world
        del buf

    # ---- ddp/zero1 的全量梯度绑定

    def _bind_grads(self) -> None:
        self._flat_grad.zero_()
        off = 0
        for p in self.params:
            n = p.numel()
            p.grad = self._flat_grad[off:off + n].view_as(p)
            off += n

    def all_reduce_grads(self) -> None:
        g = self._flat_grad[:self.n_elem]
        self.comm.add("all_reduce", self.n_elem * g.element_size())
        dist.all_reduce(g, op=dist.ReduceOp.SUM)
        g /= self.world

    # ---- 参数侧 collective

    def all_gather_params(self) -> None:
        if self.world == 1:
            return
        mine = torch.zeros(self.chunk, dtype=self.shard.dtype,
                           device=self.device)
        mine[:self.hi - self.lo] = self.shard.detach()
        self.comm.add("all_gather", self.chunk * mine.element_size()
                      * self.world)
        dist.all_gather_into_tensor(self._flat_param, mine)
        del mine

    def materialize_params(self) -> None:
        """zero3：从分片 gather 满参并绑定到模型（整模型粒度 = 最粗 wrap）。"""
        self._flat_param = torch.zeros(
            self.padded, dtype=self.shard.dtype, device=self.device)
        if self.world > 1:
            mine = torch.zeros(self.chunk, dtype=self.shard.dtype,
                               device=self.device)
            mine[:self.hi - self.lo] = self.shard.detach()
            self.comm.add("all_gather", self.chunk * mine.element_size()
                          * self.world)
            dist.all_gather_into_tensor(self._flat_param, mine)
            del mine
        else:
            self._flat_param[self.lo:self.hi] = self.shard.detach()
        # 显式按形状绑定：release 后 param.data 形状已清空，
        # vector_to_parameters 的 view_as(param) 会失败
        flat = self._flat_param[:self.n_elem]
        off = 0
        for p, shape in zip(self.params, self._shapes):
            n = 1
            for s in shape:
                n *= s          # release 后 p.numel()==0，必须用保存的形状
            p.data = flat[off:off + n].view(shape)
            off += n

    def release_params(self) -> None:
        """zero3：step 后满参失效，只留分片（常驻口径的关键动作）。"""
        for p in self.params:
            p.data = torch.empty(0, dtype=p.dtype, device=p.device)
        self._flat_param = None

    # ---- step 与收尾

    def step(self) -> None:
        if self.level == "ddp":
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)
        else:
            if self.level == "zero1":
                self.shard.grad = self._flat_grad[self.lo:self.hi]
            else:
                self.shard.grad = self.grad_shard[:self.hi - self.lo]
            self.opt.step()
            self.shard.grad = None

    def finish_step(self) -> None:
        """step 间隙释放梯度侧暂存（常驻口径 = 参数 + 优化器状态）。"""
        for p in self.params:
            p.grad = None
        self._chunk_buf.clear()
        self._chunk_filled.clear()
        if self._flat_grad is not None:
            self._flat_grad.zero_()

    def release_aux(self) -> None:
        """训练结束释放全部辅助缓冲，留下纯训练态。"""
        self._flat_grad = None
        self._chunk_buf.clear()
        if self.level == "zero3":
            self.release_params()
        for p in self.params:
            p.grad = None

    def optim_state_bytes(self) -> int:
        total = 0
        for state in self.opt.state.values():
            for v in state.values():
                if torch.is_tensor(v):
                    total += v.numel() * v.element_size()
        return total


# ---------------------------------------------------------------- 主流程


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", required=True,
                    choices=("ddp", "zero1", "zero2", "zero3"))
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--batch", type=int, default=4,
                    help="每 rank 的 batch；全局 batch = batch × world")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=384)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--inter", type=int, default=1024)
    ap.add_argument("--vocab", type=int, default=0,
                    help="0 = 跟语料 CharTokenizer 走（默认）")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--backend", default="nccl")
    ap.add_argument("--data-dir", default="exp_scale/data")
    ap.add_argument("--mem-fraction", type=float, default=0.0,
                    help="把分配器预算钉在物理显存的比例，0=不钉（WSL 超配）")
    ap.add_argument("--out", default="results/zero_level.json")
    args = ap.parse_args()

    if args.mem_fraction > 0 and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.mem_fraction)
    if not dist.is_initialized():
        dist.init_process_group(backend=args.backend)
    rank = dist.get_rank()
    world = dist.get_world_size()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        # 12 篇口径：LOCAL_RANK 取模绑卡（单机 2 rank gloo 预检共卡；
        # 跨机 NCCL 时每进程一张卡，取模不变）。
        torch.cuda.set_device(
            int(os.environ.get("LOCAL_RANK", 0)) % torch.cuda.device_count())

    torch.manual_seed(args.seed)
    from exp_ddp.ddp_common import FixedSampleDataset, load_corpus_ids
    ids, vocab_real = load_corpus_ids(args.data_dir)
    vocab = vocab_real if args.vocab <= 0 else args.vocab
    # 每 rank 要有 steps 个 batch（每 batch 个样本），样本池 = steps*batch*world
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
    state = ShardedState(model, args.level, world, rank, args.lr, device, comm)

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
            if args.level in ("ddp", "zero1"):
                state._bind_grads()
            elif args.level == "zero3":
                state.materialize_params()
            out = model(x, labels=t)
            out.loss.backward()
            if args.level in ("ddp", "zero1"):
                state.all_reduce_grads()
            state.step()
            if args.level in ("zero1", "zero2"):
                state.all_gather_params()
            elif args.level == "zero3":
                state.release_params()
            state.finish_step()
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

    checksum = ""
    if oom_info is None:
        if args.level == "zero3":
            state.materialize_params()  # checksum 前重建满参
        state.release_aux()
        probe.snap("train_done")
        from exp_ddp.ddp_common import param_checksum
        checksum = param_checksum(model)

    out = {
        "level": args.level,
        "steps": args.steps,
        "world_size": world,
        "oom": oom_info is not None,
        "oom_traceback": oom_info or "",
        "config": vars(args),
        "n_params": n_params,
        "losses": losses,
        "final_loss": losses[-1] if losses else None,
        "param_checksum": checksum,
        "resident_mb": probe.records[-1]["allocated_mb"],
        "peak_mb": probe.records[-1]["peak_mb"],
        "comm": comm.as_dict(),
        "shard_opt_state_mb": round(
            state.optim_state_bytes() / 1024 / 1024, 1),
        "state_account": StateAccount(
            n_params, world, 8.0,
            shard_optimizer=args.level != "ddp",
            shard_gradient=args.level in ("zero2", "zero3"),
            shard_param=args.level == "zero3").as_dict(),
        "tok_per_s": round(n_tokens / dt, 1) if dt > 0 else 0,
        "stages": probe.records,
        "meta": run_meta(rank, world, vars(args)),
    }
    # 每 rank 都落盘（两机时各写本地文件；单机多 rank 加后缀避免覆盖）
    out_path = (args.out if rank == 0 else
                args.out.replace(".json", f"_rank{rank}.json"))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    if rank == 0:
        tag = "OOM" if out["oom"] else f"loss={out['final_loss']:.4f}"
        print(f"[{args.level}] world={world} {tag} "
              f"resident={out['resident_mb']} peak={out['peak_mb']} "
              f"comm={out['comm']['total_bytes'] / 1024 / 1024:.1f} MiB "
              f"tok/s={out['tok_per_s']}")
        print(f"done -> {args.out}")
    dist.destroy_process_group()
    return 2 if oom_info is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
