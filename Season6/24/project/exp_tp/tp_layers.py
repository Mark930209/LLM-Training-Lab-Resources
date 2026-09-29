"""tp_layers.py —— 24 篇 Tensor Parallel Lab 的核心实现。

设计原则（正文"通信安排对"的落点）：

1. ColumnParallelLinear  按输出维切权重，输入复制、输出分片。
   前向零通信；反向对 grad_input 做 AllReduce（每个 rank 只算出部分贡献）。
2. RowParallelLinear    按输入维切权重，输入分片、输出求和。
   前向对 partial 输出 AllReduce；bias 必须在 AllReduce **之后**加一次。
3. column → row 成对使用：中间激活保持分片（gather_output=False），
   通信只出现在成对边界：每对前向 1 次 AllReduce（row 输出求和）、
   反向 1 次 AllReduce（column 输入梯度求和）。

两个故意留下的 bug 变体（E6 失败案例，正文实证）：
   RowParallelLinear(bias_before_reduce=True)
       bias 在 AllReduce 前加 → 输出 shape 对、bias 被加了 world_size 次。
   ColumnParallelLinear(split_bug=True)
       方阵权重下切错维度（按输入维切却当输出维用）→ matmul 仍可乘、
       shape 全对、数值静默错误。
"""

from __future__ import annotations

import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from tp_common import CommAccount


# ---------------------------------------------------------------- 并行区算子


class _CopyToTPRegion(torch.autograd.Function):
    """前向恒等（值不变），反向 AllReduce：把各 rank 的部分输入梯度求和。

    forward 不能直接 return input：输出与输入同一对象会让 leaf 输入的
    梯度累积被破坏（E0 实测 x.grad 既不是本地贡献也不是求和结果），
    必须返回新张量。反向同理：clone 后再 reduce，不改写上游缓冲。
    """

    comm: CommAccount | None = None

    @staticmethod
    def forward(ctx, x):
        return x.clone()

    @staticmethod
    def backward(ctx, g):
        if _CopyToTPRegion.comm is not None:
            g = g.clone()
            _CopyToTPRegion.comm.all_reduce_(g)
        return g


class _ReduceFromTPRegion(torch.autograd.Function):
    """前向 AllReduce（partial 输出求和），反向恒等（梯度天然复制）。

    同样返回新张量，不改写 F.linear 的输出缓冲。
    """

    comm: CommAccount | None = None

    @staticmethod
    def forward(ctx, y):
        out = y.clone()
        if _ReduceFromTPRegion.comm is not None:
            _ReduceFromTPRegion.comm.all_reduce_(out)
        return out

    @staticmethod
    def backward(ctx, g):
        return g


class _ScatterToTPRegion(torch.autograd.Function):
    """前向取本地分片（沿最后一维均匀切），反向 AllGather 拼回完整梯度。"""

    comm: CommAccount | None = None
    world: int = 1
    rank: int = 0

    @staticmethod
    def forward(ctx, x):
        if _ScatterToTPRegion.world == 1:
            return x
        w = _ScatterToTPRegion.world
        r = _ScatterToTPRegion.rank
        return x[..., r * (x.shape[-1] // w):(r + 1) * (x.shape[-1] // w)].contiguous()

    @staticmethod
    def backward(ctx, g):
        if _ScatterToTPRegion.comm is not None:
            return _ScatterToTPRegion.comm.all_gather_cat_(g)
        return g


def bind_comm(comm: CommAccount, world: int, rank: int) -> None:
    """把记账器注入并行区算子（进程内单例，实验脚本启动时调用一次）。"""
    _CopyToTPRegion.comm = comm
    _ReduceFromTPRegion.comm = comm
    _ScatterToTPRegion.comm = comm
    _ScatterToTPRegion.world = world
    _ScatterToTPRegion.rank = rank


# ---------------------------------------------------------------- 分片 Linear


class ColumnParallelLinear(nn.Module):
    """Y = X A + b，A 沿输出维（行）切：本 rank 持有 A[ho_i]，输出分片。

    split_bug=True 是 E6(b) 失败变体：方阵时改为沿输入维切却按输出维解释，
    shape 全对、数值静默错误。
    """

    def __init__(self, in_f: int, out_f: int, world: int, rank: int,
                 bias: bool = True, split_bug: bool = False,
                 full_weight: torch.Tensor | None = None,
                 full_bias: torch.Tensor | None = None):
        super().__init__()
        assert out_f % world == 0
        self.world, self.rank = world, rank
        self.split_bug = split_bug
        self.out_f, self.in_f = out_f, in_f
        shard_out = out_f // world
        if split_bug:
            assert in_f % world == 0, "split_bug 变体要求方阵可切"
            w = (torch.empty(in_f // world, in_f) if full_weight is None
                 else full_weight[:, rank * (in_f // world):
                                  (rank + 1) * (in_f // world)].t().contiguous())
        else:
            w = (torch.empty(shard_out, in_f) if full_weight is None
                 else full_weight[rank * shard_out:(rank + 1) * shard_out, :]
                     .contiguous())
        if full_weight is None:
            nn.init.normal_(w, std=0.02)   # 无参考切片时自行初始化
        self.weight = nn.Parameter(w)
        if bias:
            b = (torch.zeros(shard_out) if full_bias is None
                 else full_bias[rank * shard_out:(rank + 1) * shard_out].contiguous())
            self.bias = nn.Parameter(b)
        else:
            self.bias = None

    def forward(self, x):  # x 复制 [B, S, in_f]
        x = _CopyToTPRegion.apply(x)
        return F.linear(x, self.weight, self.bias)  # [B, S, out_f/world]


class RowParallelLinear(nn.Module):
    """Y = X A + b，A 沿输入维（列）切：本 rank 持有 A[:, in_i]，输入分片。

    bias_after_reduce=False 是 E6(a) 失败变体：bias 在 AllReduce 前加，
    输出 shape 对、bias 被加了 world_size 次。
    """

    def __init__(self, in_f: int, out_f: int, world: int, rank: int,
                 bias: bool = True, bias_before_reduce: bool = False,
                 full_weight: torch.Tensor | None = None,
                 full_bias: torch.Tensor | None = None):
        super().__init__()
        assert in_f % world == 0
        self.world, self.rank = world, rank
        self.bias_before_reduce = bias_before_reduce
        shard_in = in_f // world
        w = (torch.empty(out_f, shard_in) if full_weight is None
             else full_weight[:, rank * shard_in:(rank + 1) * shard_in].contiguous())
        if full_weight is None:
            nn.init.normal_(w, std=0.02)   # 无参考切片时自行初始化
        self.weight = nn.Parameter(w)
        if bias:
            b = (torch.zeros(out_f) if full_bias is None
                 else full_bias.contiguous())
            self.bias = nn.Parameter(b)
        else:
            self.bias = None

    def forward(self, x):  # x 分片 [B, S, in_f/world]
        y = F.linear(x, self.weight)  # partial
        if self.bias_before_reduce and self.bias is not None:
            y = y + self.bias
            return _ReduceFromTPRegion.apply(y)
        y = _ReduceFromTPRegion.apply(y)
        return y + self.bias if self.bias is not None else y


# ---------------------------------------------------------------- 组合模块


class TPMLP(nn.Module):
    """fc1 column → GELU → fc2 row。中间激活保持分片，边界各一次 AllReduce。"""

    def __init__(self, hidden: int, inter: int, world: int, rank: int,
                 bias: bool = True, bug: str = "none"):
        super().__init__()
        self.fc1 = ColumnParallelLinear(hidden, inter, world, rank, bias=bias)
        self.fc2 = RowParallelLinear(
            inter, hidden, world, rank, bias=bias,
            bias_before_reduce=(bug == "bias"))
        if bug == "splitdim":
            self.fc1 = ColumnParallelLinear(hidden, inter, world, rank,
                                            bias=bias, split_bug=True)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class TPAttention(nn.Module):
    """QKV column（按 head 切）→ 局部 head attention → out proj row。"""

    def __init__(self, hidden: int, heads: int, world: int, rank: int,
                 bug: str = "none"):
        super().__init__()
        assert heads % world == 0
        self.heads = heads
        self.per_rank = heads // world
        self.dh = hidden // heads
        self.q = ColumnParallelLinear(hidden, hidden, world, rank)
        self.k = ColumnParallelLinear(hidden, hidden, world, rank)
        self.v = ColumnParallelLinear(hidden, hidden, world, rank)
        self.proj = RowParallelLinear(
            hidden, hidden, world, rank,
            bias_before_reduce=(bug == "bias"))
        self.scale = 1.0 / math.sqrt(self.dh)

    def _split(self, t, b, s):
        return t.view(b, s, self.per_rank, self.dh).transpose(1, 2)

    def forward(self, x):
        b, s, _ = x.shape
        q = self._split(self.q(x), b, s)
        k = self._split(self.k(x), b, s)
        v = self._split(self.v(x), b, s)
        att = torch.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        out = (att @ v).transpose(1, 2).reshape(b, s, self.per_rank * self.dh)
        return self.proj(out)


class TPBlock(nn.Module):
    """LN(复制) → TP-Attn → 残差 → LN → TP-MLP → 残差。"""

    def __init__(self, hidden: int, inter: int, heads: int,
                 world: int, rank: int, bug: str = "none"):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden)
        self.attn = TPAttention(hidden, heads, world, rank, bug=bug)
        self.ln2 = nn.LayerNorm(hidden)
        self.mlp = TPMLP(hidden, inter, world, rank, bug=bug)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# ---------------------------------------------------------------- 单卡参考


class RefLinear(nn.Module):
    def __init__(self, in_f: int, out_f: int, bias: bool = True):
        super().__init__()
        self.fc = nn.Linear(in_f, out_f, bias=bias)

    def forward(self, x):
        return self.fc(x)


class RefMLP(nn.Module):
    def __init__(self, hidden: int, inter: int, bias: bool = True):
        super().__init__()
        self.fc1 = nn.Linear(hidden, inter, bias=bias)
        self.fc2 = nn.Linear(inter, hidden, bias=bias)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class RefAttention(nn.Module):
    def __init__(self, hidden: int, heads: int):
        super().__init__()
        self.heads = heads
        self.dh = hidden // heads
        self.q = nn.Linear(hidden, hidden)
        self.k = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, hidden)
        self.proj = nn.Linear(hidden, hidden)
        self.scale = 1.0 / math.sqrt(self.dh)

    def forward(self, x):
        b, s, _ = x.shape
        q = self.q(x).view(b, s, self.heads, self.dh).transpose(1, 2)
        k = self.k(x).view(b, s, self.heads, self.dh).transpose(1, 2)
        v = self.v(x).view(b, s, self.heads, self.dh).transpose(1, 2)
        att = torch.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        out = (att @ v).transpose(1, 2).reshape(b, s, self.heads * self.dh)
        return self.proj(out)


class RefBlock(nn.Module):
    def __init__(self, hidden: int, inter: int, heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden)
        self.attn = RefAttention(hidden, heads)
        self.ln2 = nn.LayerNorm(hidden)
        self.mlp = RefMLP(hidden, inter)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# ---------------------------------------------------------------- 权重对应


def tp_from_ref_linear_col(ref: nn.Linear, world: int, rank: int,
                           split_bug: bool = False) -> ColumnParallelLinear:
    return ColumnParallelLinear(
        ref.in_features, ref.out_features, world, rank,
        bias=ref.bias is not None, split_bug=split_bug,
        full_weight=ref.weight.data, full_bias=ref.bias.data if ref.bias is not None else None)


def tp_from_ref_linear_row(ref: nn.Linear, world: int, rank: int,
                           bug: str = "none") -> RowParallelLinear:
    return RowParallelLinear(
        ref.in_features, ref.out_features, world, rank,
        bias=ref.bias is not None, bias_before_reduce=(bug == "bias"),
        full_weight=ref.weight.data, full_bias=ref.bias.data if ref.bias is not None else None)


def tp_mlp_from_ref(ref: RefMLP, world: int, rank: int,
                    bug: str = "none") -> TPMLP:
    m = TPMLP.__new__(TPMLP)   # 不跑占位初始化，直接从参考权重切片构建
    nn.Module.__init__(m)
    m.fc1 = tp_from_ref_linear_col(ref.fc1, world, rank,
                                   split_bug=(bug == "splitdim"))
    m.fc2 = tp_from_ref_linear_row(ref.fc2, world, rank, bug=bug)
    return m


def tp_attn_from_ref(ref: RefAttention, world: int, rank: int,
                     bug: str = "none") -> TPAttention:
    heads = ref.heads
    a = TPAttention.__new__(TPAttention)
    nn.Module.__init__(a)
    a.heads = heads
    a.per_rank = heads // world
    a.dh = ref.dh
    a.scale = ref.scale
    a.q = tp_from_ref_linear_col(ref.q, world, rank)
    a.k = tp_from_ref_linear_col(ref.k, world, rank)
    a.v = tp_from_ref_linear_col(ref.v, world, rank)
    a.proj = tp_from_ref_linear_row(ref.proj, world, rank, bug=bug)
    return a


def tp_block_from_ref(ref: RefBlock, world: int, rank: int,
                      bug: str = "none") -> TPBlock:
    import copy
    b = TPBlock.__new__(TPBlock)
    nn.Module.__init__(b)
    b.ln1 = copy.deepcopy(ref.ln1)   # 独立副本：梯度对账不能共享参数对象
    b.ln2 = copy.deepcopy(ref.ln2)
    b.attn = tp_attn_from_ref(ref.attn, world, rank, bug=bug)
    b.mlp = tp_mlp_from_ref(ref.mlp, world, rank, bug=bug)
    return b
