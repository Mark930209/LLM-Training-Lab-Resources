"""mm_runtime.py —— Mini-Megatron v1 的模型与三维运行时。

设计（对应正文"组合难在映射与状态一致性"）：
- TP：column/row 成对（24 篇），collective 走 tp_group。
- PP：blocks 按 pp_idx 切段，激活沿 pp_chain 传（25 篇 isend 纪律）。
- DP：数据按 dp_idx 切片；反向后梯度在 dp_group 内 SUM 归约
  （SUM 而不是 MEAN：reference 定义是全批损失之和）。
- 随机数纪律：模型权重 rank0 生成后广播；输入由 rank0 生成后广播。
"""

from __future__ import annotations

import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- TP 层


class _CopyToTP(torch.autograd.Function):
    """前向恒等，反向 AllReduce(tp_group)。forward 返回新张量（25 篇教训）。"""

    group = None
    world = 1

    @staticmethod
    def forward(ctx, x):
        return x.clone()

    @staticmethod
    def backward(ctx, g):
        g = g.clone()
        if _CopyToTP.world > 1:
            dist.all_reduce(g, group=_CopyToTP.group)
        return g


class _ReduceFromTP(torch.autograd.Function):
    """前向 AllReduce(tp_group)，反向恒等。"""

    group = None
    world = 1

    @staticmethod
    def forward(ctx, y):
        out = y.clone()
        if _ReduceFromTP.world > 1:
            dist.all_reduce(out, group=_ReduceFromTP.group)
        return out

    @staticmethod
    def backward(ctx, g):
        return g


class ColLinear(nn.Module):
    def __init__(self, in_f, out_f, world, rank, full_w, full_b=None):
        super().__init__()
        self.world, self.rank = world, rank
        shard = out_f // world
        self.weight = nn.Parameter(
            full_w[rank * shard:(rank + 1) * shard].clone())
        if full_b is not None:
            self.bias = nn.Parameter(
                full_b[rank * shard:(rank + 1) * shard].clone())
        else:
            self.bias = None

    def forward(self, x):
        x = _CopyToTP.apply(x)
        return F.linear(x, self.weight, self.bias)


class RowLinear(nn.Module):
    def __init__(self, in_f, out_f, world, rank, full_w, full_b=None):
        super().__init__()
        self.world, self.rank = world, rank
        shard = in_f // world
        self.weight = nn.Parameter(
            full_w[:, rank * shard:(rank + 1) * shard].clone())
        self.bias = (nn.Parameter(full_b.clone()) if full_b is not None
                     else None)

    def forward(self, x):
        y = F.linear(x, self.weight)
        y = _ReduceFromTP.apply(y)
        return y + self.bias if self.bias is not None else y


# ---------------------------------------------------------------- 块与参考


def _ref_block_params(hidden: int, heads: int, seed: int) -> dict:
    torch.manual_seed(seed)
    shapes = {
        "ln1w": (hidden,), "ln1b": (hidden,),
        "qw": (hidden, hidden), "qb": (hidden,),
        "kw": (hidden, hidden), "kb": (hidden,),
        "vw": (hidden, hidden), "vb": (hidden,),
        "ow": (hidden, hidden), "ob": (hidden,),
        "ln2w": (hidden,), "ln2b": (hidden,),
        "fc1w": (hidden * 2, hidden), "fc1b": (hidden * 2,),
        "fc2w": (hidden, hidden * 2), "fc2b": (hidden,),
    }
    return {n: torch.randn(s) * 0.02 for n, s in shapes.items()}


class RefBlock(nn.Module):
    def __init__(self, hidden: int, heads: int, params: dict):
        super().__init__()
        self.hidden, self.heads = hidden, heads
        self.dh = hidden // heads
        self.ln1w = nn.Parameter(params["ln1w"])
        self.ln1b = nn.Parameter(params["ln1b"])
        self.qw = nn.Parameter(params["qw"])
        self.qb = nn.Parameter(params["qb"])
        self.kw = nn.Parameter(params["kw"])
        self.kb = nn.Parameter(params["kb"])
        self.vw = nn.Parameter(params["vw"])
        self.vb = nn.Parameter(params["vb"])
        self.ow = nn.Parameter(params["ow"])
        self.ob = nn.Parameter(params["ob"])
        self.ln2w = nn.Parameter(params["ln2w"])
        self.ln2b = nn.Parameter(params["ln2b"])
        self.fc1w = nn.Parameter(params["fc1w"])
        self.fc1b = nn.Parameter(params["fc1b"])
        self.fc2w = nn.Parameter(params["fc2w"])
        self.fc2b = nn.Parameter(params["fc2b"])

    def forward(self, x):
        b, s, h = x.shape
        t = F.layer_norm(x, (h,), self.ln1w, self.ln1b)
        hd = self.heads
        q = (t @ self.qw.t() + self.qb).view(b, s, hd, self.dh).transpose(1, 2)
        k = (t @ self.kw.t() + self.kb).view(b, s, hd, self.dh).transpose(1, 2)
        v = (t @ self.vw.t() + self.vb).view(b, s, hd, self.dh).transpose(1, 2)
        att = torch.softmax((q @ k.transpose(-2, -1)) / math.sqrt(self.dh),
                            dim=-1)
        o = (att @ v).transpose(1, 2).reshape(b, s, h)
        x = x + (o @ self.ow.t() + self.ob)
        t = F.layer_norm(x, (h,), self.ln2w, self.ln2b)
        return x + (F.gelu(t @ self.fc1w.t() + self.fc1b) @ self.fc2w.t()
                    + self.fc2b)


class TPBlock(nn.Module):
    def __init__(self, hidden: int, heads: int, params: dict,
                 world: int, rank: int):
        super().__init__()
        self.hidden, self.heads = hidden, heads
        self.dh = hidden // heads
        self.ln1w = nn.Parameter(params["ln1w"])
        self.ln1b = nn.Parameter(params["ln1b"])
        self.q = ColLinear(hidden, hidden, world, rank, params["qw"],
                           params["qb"])
        self.k = ColLinear(hidden, hidden, world, rank, params["kw"],
                           params["kb"])
        self.v = ColLinear(hidden, hidden, world, rank, params["vw"],
                           params["vb"])
        self.o = RowLinear(hidden, hidden, world, rank, params["ow"],
                           params["ob"])
        self.ln2w = nn.Parameter(params["ln2w"])
        self.ln2b = nn.Parameter(params["ln2b"])
        self.fc1 = ColLinear(hidden, hidden * 2, world, rank, params["fc1w"],
                             params["fc1b"])
        self.fc2 = RowLinear(hidden * 2, hidden, world, rank, params["fc2w"],
                             params["fc2b"])

    def forward(self, x):
        b, s, h = x.shape
        t = F.layer_norm(x, (h,), self.ln1w, self.ln1b)
        hd = self.heads
        dhr = self.q.weight.shape[0] // hd
        q = self.q(t).view(b, s, hd, dhr).transpose(1, 2)
        k = self.k(t).view(b, s, hd, dhr).transpose(1, 2)
        v = self.v(t).view(b, s, hd, dhr).transpose(1, 2)
        att = torch.softmax((q @ k.transpose(-2, -1)) / math.sqrt(self.dh),
                            dim=-1)
        # 注意力输出是本 rank 的头拼接 = hidden/tp（分片），不是全量 hidden
        o = (att @ v).transpose(1, 2).reshape(b, s, hd * dhr)
        x = x + self.o(o)
        t = F.layer_norm(x, (h,), self.ln2w, self.ln2b)
        return x + self.fc2(F.gelu(self.fc1(t)))


class TPStage(nn.Module):
    """本 pp stage 持有的 TP 块序列。"""

    def __init__(self, hidden: int, heads: int, params_list: list[dict],
                 world: int, rank: int):
        super().__init__()
        self.blocks = nn.ModuleList(
            [TPBlock(hidden, heads, p, world, rank) for p in params_list])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class RefStage(nn.Module):
    def __init__(self, hidden: int, heads: int, params_list: list[dict]):
        super().__init__()
        self.blocks = nn.ModuleList(
            [RefBlock(hidden, heads, p) for p in params_list])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


def broadcast_params(module: nn.Module) -> None:
    if not dist.is_initialized():
        return
    for p in module.parameters():
        dist.broadcast(p.data, src=0)
