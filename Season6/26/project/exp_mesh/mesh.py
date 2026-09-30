"""mesh.py —— 26 篇 Mini-Megatron v1 的三维 process mesh。

排布约定（Megatron 风格的简化版）：
    rank = dp_idx * (tp * pp) + pp_idx * tp + tp_idx

三类 group 从坐标导出，任何映射错误都会变成三种病：数据重复、
collective hang、checkpoint 无法重组（正文的诊断框架）。
"""

from __future__ import annotations

import torch.distributed as dist


class Mesh:
    def __init__(self, dp: int, tp: int, pp: int, world: int):
        assert dp * tp * pp == world, f"mesh {dp}x{tp}x{pp} != world {world}"
        self.dp, self.tp, self.pp = dp, tp, pp
        self.world = world

    # ---- 坐标
    def coords(self, rank: int) -> tuple[int, int, int]:
        dp_idx = rank // (self.tp * self.pp)
        rem = rank % (self.tp * self.pp)
        pp_idx = rem // self.tp
        tp_idx = rem % self.tp
        return dp_idx, pp_idx, tp_idx

    def rank_of(self, dp_idx: int, pp_idx: int, tp_idx: int) -> int:
        return dp_idx * (self.tp * self.pp) + pp_idx * self.tp + tp_idx

    # ---- group 成员（纯计算，不依赖 dist）
    def tp_members(self, rank: int) -> list[int]:
        """同 dp、同 pp、不同 tp：权重分片的 collective 组。"""
        d, p, _ = self.coords(rank)
        return [self.rank_of(d, p, t) for t in range(self.tp)]

    def pp_members(self, rank: int) -> list[int]:
        """同 dp、同 tp、不同 pp：流水线链。"""
        d, _, t = self.coords(rank)
        return [self.rank_of(d, k, t) for k in range(self.pp)]

    def dp_members(self, rank: int) -> list[int]:
        """同 tp、同 pp、不同 dp：梯度归约组。"""
        _, p, t = self.coords(rank)
        return [self.rank_of(d, p, t) for d in range(self.dp)]

    def all_tp_groups(self) -> list[list[int]]:
        seen, out = set(), []
        for r in range(self.world):
            g = tuple(sorted(self.tp_members(r)))
            if g not in seen:
                seen.add(g)
                out.append(list(g))
        return out

    def all_pp_chains(self) -> list[list[int]]:
        seen, out = set(), []
        for r in range(self.world):
            g = tuple(self.pp_members(r))
            if g not in seen:
                seen.add(g)
                out.append(list(g))
        return out

    def all_dp_groups(self) -> list[list[int]]:
        seen, out = set(), []
        for r in range(self.world):
            g = tuple(sorted(self.dp_members(r)))
            if g not in seen:
                seen.add(g)
                out.append(list(g))
        return out

    def table(self) -> list[dict]:
        """rank 映射表：每个 rank 的三个坐标与三类 group 成员。"""
        rows = []
        for r in range(self.world):
            d, p, t = self.coords(r)
            rows.append({
                "rank": r, "dp": d, "pp": p, "tp": t,
                "tp_group": self.tp_members(r),
                "pp_chain": self.pp_members(r),
                "dp_group": self.dp_members(r),
            })
        return rows

    def self_check(self) -> list[str]:
        """双向一致性自检：返回问题清单（空 = 通过）。"""
        errs = []
        for r in range(self.world):
            d, p, t = self.coords(r)
            if self.rank_of(d, p, t) != r:
                errs.append(f"rank {r} 坐标往返不一致")
            for name, mem in [("tp", self.tp_members(r)),
                              ("pp", self.pp_members(r)),
                              ("dp", self.dp_members(r))]:
                if r not in mem:
                    errs.append(f"rank {r} 不在自己的 {name} group 里")
                for m in mem:
                    if not (0 <= m < self.world):
                        errs.append(f"{name} group 越界: {m}")
        return errs


def build_groups(mesh: Mesh, rank: int, swap_tp: bool = False):
    """为所有 rank 建三类通信组。swap_tp=True 是映射错误变体。

    new_group 是集体操作：每个 rank 必须按**同一顺序**为**每一个**组调用，
    只把本 rank 所在的组挑出来返回。顺序不一致的写法本身就是 hang 病灶。
    """
    tp_list = mesh.all_tp_groups()
    if swap_tp and mesh.tp > 1:
        # 故意切错：把 tp 维当 pp 维切（例：0,2 / 1,3 而不是 0,1 / 2,3）
        tp_list = [[mesh.rank_of(d, k, t) for k in range(mesh.pp)]
                   for d in range(mesh.dp) for t in range(mesh.tp)]
    pp_list = mesh.all_pp_chains()
    dp_list = mesh.all_dp_groups()

    groups: dict = {"tp": None, "pp": None, "dp": None}
    for name, lst in (("tp", tp_list), ("pp", pp_list), ("dp", dp_list)):
        for members in lst:
            h = dist.new_group(members) if mesh.world > 1 else None
            if rank in members and groups[name] is None:
                groups[name] = (list(members), h)
    return groups
