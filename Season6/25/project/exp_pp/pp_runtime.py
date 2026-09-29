"""pp_runtime.py —— 25 篇的 pipeline 运行时：GPipe / 1F1B + parity 验收。

死锁纪律（正文素材）：两个 rank 的 send/recv 必须交错匹配。GPipe 是
"全发全收"（rank0 先发 M 个激活再收 M 个梯度；rank1 先收后发）；1F1B 是
"发一个收一个"（rank0 warmup 1 个前向后进入交替）。任何一边改成
"先发后收"的对称写法就会 circular wait，见 pp_hang.py 的失败实证。

parity 验收（同 24 篇口径）：输出与梯度对单进程整模型容差对齐。
loss 定义：sum over micro-batch of (out_mb * gcoef_mb).sum()（不除 M，
参考实现同定义即可对齐）。
"""

from __future__ import annotations

import time

import torch

from pp_common import P2PBytes, Timeline
from pp_model import stack_forward


class Stage:
    """rank 的 stage 视图：本 rank 持有的 block 区间。"""

    def __init__(self, stack, blocks: list[int], rank: int, world: int):
        self.stack = stack
        self.blocks = blocks          # 本 stage 持有的 block 下标
        self.rank = rank
        self.world = world

    @property
    def first(self) -> bool:
        return self.rank == 0

    @property
    def last(self) -> bool:
        return self.rank == self.world - 1


def _fwd_mb(stage: Stage, x: torch.Tensor, tl: Timeline, mb: int) -> torch.Tensor:
    with tl.span("fwd", mb):
        y = stack_forward(stage.stack, x, stage.blocks)
    return y


def _bwd_mb(stage: Stage, y: torch.Tensor, g: torch.Tensor,
            tl: Timeline, mb: int) -> None:
    with tl.span("bwd", mb):
        torch.autograd.backward(y, g)


def _do_fwd(stage: Stage, xs, out_box, tl, p2p, i):
    """一个 micro-batch 的前向（含相邻收发）。out_box 存 (入张量, 出张量)。"""
    if stage.first:
        # 复用调用方传入的叶子：输入梯度留在 x.grad，供 parity 读取
        x = (xs[i] if (xs[i].is_leaf and xs[i].requires_grad)
             else xs[i].detach().requires_grad_(True))
        y = _fwd_mb(stage, x, tl, i)
        if not stage.last:
            p2p.send(y, stage.rank + 1, tl, i)
        out_box[i] = (x, y)
    else:
        buf = p2p.recv(xs[i], stage.rank - 1, tl, i)
        a = buf.detach().requires_grad_(True)
        y = _fwd_mb(stage, a, tl, i)
        if not stage.last:
            p2p.send(y, stage.rank + 1, tl, i)
        out_box[i] = (a, y)


def _do_bwd(stage: Stage, out_box, gcoef, tl, p2p, i) -> float:
    """一个 micro-batch 的反向（含相邻梯度收发）。返回末 stage 的 loss 贡献。"""
    a, y = out_box[i]
    if stage.last:
        l_i = (y.float() * gcoef[i]).sum()
        with tl.span("bwd", i):
            l_i.backward()
        p2p.send(a.grad, stage.rank - 1, tl, i)
        return float(l_i.detach())
    g = p2p.recv(y, stage.rank + 1, tl, i)
    _bwd_mb(stage, y, g, tl, i)
    if not stage.first:
        p2p.send(a.grad, stage.rank - 1, tl, i)   # 首 stage 无上游，不发输入梯度
    return 0.0


def run_gpipe(stage: Stage, xs: list[torch.Tensor],
              gcoef: torch.Tensor, tl: Timeline, p2p: P2PBytes) -> float:
    """GPipe：全前向（发）→ 全反向（收梯度、发梯度）。返回 loss（末 stage）。"""
    m = len(xs)
    out_box: dict = {}
    loss = 0.0
    for i in range(m):
        _do_fwd(stage, xs, out_box, tl, p2p, i)
    for i in range(m):
        loss += _do_bwd(stage, out_box, gcoef, tl, p2p, i)
    return loss


def run_1f1b(stage: Stage, xs: list[torch.Tensor],
             gcoef: torch.Tensor, tl: Timeline, p2p: P2PBytes) -> float:
    """1F1B：warmup(s−1−rank) 个前向后交替前向/反向，末 stage 直接交替。"""
    m = len(xs)
    s = stage.world
    warmup = min(m, s - 1 - stage.rank)
    out_box: dict = {}
    loss = 0.0
    fwd_i = bwd_i = 0
    for _ in range(warmup):
        _do_fwd(stage, xs, out_box, tl, p2p, fwd_i)
        fwd_i += 1
    while fwd_i < m:
        _do_fwd(stage, xs, out_box, tl, p2p, fwd_i)
        fwd_i += 1
        loss += _do_bwd(stage, out_box, gcoef, tl, p2p, bwd_i)
        bwd_i += 1
    while bwd_i < m:
        loss += _do_bwd(stage, out_box, gcoef, tl, p2p, bwd_i)
        bwd_i += 1
    return loss


def run_pp(stage: Stage, xs: list[torch.Tensor], gcoef: torch.Tensor,
           schedule: str, tl: Timeline, p2p: P2PBytes) -> float:
    if schedule == "gpipe":
        return run_gpipe(stage, xs, gcoef, tl, p2p)
    return run_1f1b(stage, xs, gcoef, tl, p2p)


def run_reference(stack, xs: list[torch.Tensor],
                  gcoef: torch.Tensor) -> tuple[float, dict, list]:
    """单进程整模型参考：同 xs、同 gcoef、同 loss 定义。
    返回 (loss, 参数梯度, 各 micro-batch 输入梯度)。"""
    for p in stack.parameters():
        p.grad = None
    loss = 0.0
    xleaves = []
    for i, x in enumerate(xs):
        x = x.detach().requires_grad_(True)
        xleaves.append(x)
        y = stack_forward(stack, x)
        loss = loss + (y.float() * gcoef[i]).sum()
    loss_t = loss
    loss_t.backward()
    grads = {n: p.grad.detach().clone() for n, p in stack.named_parameters()
             if p.grad is not None}
    xgrads = [x.grad.detach().clone() for x in xleaves]
    return float(loss_t.detach()), grads, xgrads


def timed_run(fn) -> tuple[float, object]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter() - t0, out
