"""tp_common.py —— 24 篇 Tensor Parallel Lab 的公共件。

三件职责（全部服务于"通信安排对了没有"这个判据）：

1. CommAccount     collective 记账：每一次 AllReduce / AllGather 记录名字、
                   payload 字节数与次数。TP 实现的对错由 collective 的位置决定，
                   性能由 collective 的次数与体积决定——这份账就是验收证据。
2. parity          容差 parity 对账：TP 求和顺序与单卡参考不同，只能容差对齐，
                   不能逐位（与 12 篇 DDP 参数逐位门禁的口径差异是正文要点）。
                   输出与梯度都用同一套 maxabs / maxrel 报告。
3. run_meta        两机留痕：rank、device、GPU 名、torch 版本、config 哈希
                   （沿用 12 篇 xnode_check 的 provenance 字段）。

口径说明（全文统一）：
    payload_bytes   collective 通信缓冲区的字节数（逻辑量，不做 ring 折算）
    ring_lower_bound 二卡 ring AllReduce 的理论搬运量 = 2 * payload * (n-1)/n，
                    用来和实测耗时对账，区分"实现慢"与"带宽顶死"
    parity          maxabs = max|a-b|，maxrel = max|a-b| / max(|b|, eps)
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.distributed as dist


# ---------------------------------------------------------------- collective 记账


@dataclass
class CommAccount:
    """collective 记账器。world=1 时不产生通信记录（本地自检路径）。"""

    world_size: int = 1
    events: list[dict] = field(default_factory=list)

    def add(self, name: str, payload_bytes: int) -> None:
        self.events.append({"name": name, "payload_bytes": int(payload_bytes)})

    def all_reduce_(self, t: torch.Tensor) -> None:
        if self.world_size > 1:
            self.add("all_reduce", t.numel() * t.element_size())
            dist.all_reduce(t)

    def all_gather_cat_(self, t: torch.Tensor) -> torch.Tensor:
        """沿最后一维 all_gather 并拼接，返回完整张量。"""
        if self.world_size == 1:
            return t
        self.add("all_gather", t.numel() * t.element_size() * self.world_size)
        parts = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(parts, t.contiguous())
        return torch.cat(parts, dim=-1)

    def summary(self) -> dict:
        agg: dict[str, dict] = {}
        for e in self.events:
            a = agg.setdefault(e["name"], {"count": 0, "payload_bytes": 0})
            a["count"] += 1
            a["payload_bytes"] += e["payload_bytes"]
        total = sum(a["payload_bytes"] for a in agg.values())
        ring = sum(
            2 * a["payload_bytes"] * (self.world_size - 1) / self.world_size
            for a in agg.values()
        ) if self.world_size > 1 else 0.0
        return {"by_op": agg, "total_payload_bytes": total,
                "ring_lower_bound_bytes": round(ring, 1),
                "events": list(self.events)}


def ring_lower_bound(payload_bytes: int, world_size: int) -> float:
    """二卡 ring AllReduce 理论搬运量（字节）。"""
    return 2.0 * payload_bytes * (world_size - 1) / max(world_size, 1)


# ---------------------------------------------------------------- parity 对账


def parity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6,
           atol: float = 1e-5, rtol: float = 1e-4) -> dict:
    """容差 parity 报告：a=TP 结果，b=单卡参考。

    判据用绝对+相对组合：maxabs <= atol + rtol*max|b|。只用相对尺度会在
    近零张量（如小梯度 bias）上把求和顺序噪声放大成假 FAIL；只用绝对尺度
    又会放过大幅值张量上的显著偏差。
    """
    a32 = a.detach().float()
    b32 = b.detach().float()
    diff = (a32 - b32).abs()
    maxabs = diff.max().item()
    scale = b32.abs().max().clamp_min(eps).item()
    maxrel = (diff / b32.abs().clamp_min(eps)).max().item()
    return {
        "maxabs": maxabs,
        "maxrel": maxrel,
        "norm_maxabs": maxabs / scale,
        "within_tol": bool(maxabs <= atol + rtol * scale),
        "shape_a": list(a32.shape),
        "shape_b": list(b32.shape),
        "shape_match": list(a32.shape) == list(b32.shape),
    }


# ---------------------------------------------------------------- 留痕


def run_meta(rank: int, world_size: int, config: dict,
             corpus_sha256: str = "") -> dict:
    # 每节点单卡：设备号是本地的，不能用全局 rank 当设备号
    gpu_name = (torch.cuda.get_device_name(torch.cuda.current_device())
                if torch.cuda.is_available() else "cpu")
    return {
        "rank": rank,
        "world_size": world_size,
        "gpu": gpu_name,
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "config": config,
        "config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()[:16],
        "corpus_sha256": corpus_sha256,
    }


def env_rank_world() -> tuple[int, int]:
    """从 torchrun 环境变量读 rank/world_size（单进程默认 1/1）。"""
    return int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))


def save_report(path: str | Path, report: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                 encoding="utf-8")
    print(f"[saved] {p}")
