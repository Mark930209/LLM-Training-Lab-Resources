"""pp_common.py —— 25 篇 Pipeline Parallel Lab 的公共件。

三件职责（全部服务于"bubble 是谁的产物"这个判据）：

1. Timeline        stage 时间线：每个 micro-batch 的 fwd/bwd/send/recv 起止
                   打点。bubble 就定义在 timeline 上：总墙钟 - 忙时占比。
2. parity          容差 parity（与 24 篇同判据）：PP 切层 + micro-batch 切批
                   之后，输出与梯度必须与单进程整模型对齐。
3. run_meta        两机留痕（沿用 12/23/24 篇口径）。

口径说明（全文统一）：
    busy_s     fwd+bwd 的累计时长（计算）
    comm_s     send/recv 等待的累计时长（通信阻塞）
    wall_s     首事件到末事件的墙钟
    util       busy_s / wall_s（阶段利用率）
    bubble     1 - util（空等比例）
    理论 bubble (s−1)/(M+s−1)：s 个 stage、M 个 micro-batch、时间均衡时
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.distributed as dist


# ---------------------------------------------------------------- 时间线


@dataclass
class Timeline:
    """stage 时间线：record(kind, mb, t0, t1)，kind ∈ fwd/bwd/send/recv。"""

    events: list[dict] = field(default_factory=list)

    def span(self, kind: str, mb: int):
        return _Span(self, kind, mb)

    def record(self, kind: str, mb: int, t0: float, t1: float) -> None:
        self.events.append({"kind": kind, "mb": mb, "t0": t0, "t1": t1})

    def summary(self) -> dict:
        if not self.events:
            return {"wall_s": 0.0, "busy_s": 0.0, "comm_s": 0.0,
                    "util": 0.0, "bubble": 0.0}
        t0 = min(e["t0"] for e in self.events)
        t1 = max(e["t1"] for e in self.events)
        busy = sum(e["t1"] - e["t0"] for e in self.events
                   if e["kind"] in ("fwd", "bwd"))
        comm = sum(e["t1"] - e["t0"] for e in self.events
                   if e["kind"] in ("send", "recv"))
        wall = t1 - t0
        util = busy / wall if wall > 0 else 0.0
        return {
            "wall_s": round(wall, 4),
            "busy_s": round(busy, 4),
            "comm_s": round(comm, 4),
            "util": round(util, 4),
            "bubble": round(1.0 - util, 4),
            "events": self.events,
        }


class _Span:
    def __init__(self, tl: Timeline, kind: str, mb: int):
        self.tl, self.kind, self.mb = tl, kind, mb

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.tl.record(self.kind, self.mb, self.t0, time.perf_counter())
        return False


def theoretical_bubble(stages: int, micro: int) -> float:
    """时间均衡下的理想 bubble 比例 (s−1)/(M+s−1)。"""
    return (stages - 1) / (micro + stages - 1)


# ---------------------------------------------------------------- P2P 收发


@dataclass
class P2PBytes:
    """send/recv 通信账（激活与梯度）。"""

    send_bytes: int = 0
    recv_bytes: int = 0
    n_send: int = 0
    n_recv: int = 0

    def send(self, t: torch.Tensor, dst: int, tl: Timeline, mb: int) -> None:
        with tl.span("send", mb):
            dist.send(t.contiguous(), dst)
        self.send_bytes += t.numel() * t.element_size()
        self.n_send += 1

    def recv(self, like: torch.Tensor, src: int, tl: Timeline, mb: int) -> torch.Tensor:
        buf = torch.empty_like(like)
        with tl.span("recv", mb):
            dist.recv(buf, src)
        self.recv_bytes += buf.numel() * buf.element_size()
        self.n_recv += 1
        return buf

    def summary(self) -> dict:
        return {"send_bytes": self.send_bytes, "recv_bytes": self.recv_bytes,
                "n_send": self.n_send, "n_recv": self.n_recv}


# ---------------------------------------------------------------- parity


def parity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6,
           atol: float = 1e-5, rtol: float = 1e-4) -> dict:
    """容差 parity 报告：a=PP 结果，b=单进程参考（同 24 篇判据）。"""
    a32 = a.detach().float()
    b32 = b.detach().float()
    diff = (a32 - b32).abs()
    maxabs = diff.max().item()
    scale = b32.abs().max().clamp_min(eps).item()
    return {
        "maxabs": maxabs,
        "norm_maxabs": maxabs / scale,
        "within_tol": bool(maxabs <= atol + rtol * scale),
    }


# ---------------------------------------------------------------- 留痕


def run_meta(rank: int, world_size: int, config: dict) -> dict:
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
    }


def env_rank_world() -> tuple[int, int]:
    return int(__import__("os").environ.get("RANK", 0)), \
        int(__import__("os").environ.get("WORLD_SIZE", 1))


def save_report(path: str | Path, report: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                 encoding="utf-8")
    print(f"[saved] {p}")
