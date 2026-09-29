"""shard_common.py —— 23 篇 State Sharding Lab 的公共件。

三件职责（全部服务于"常驻账 vs 峰值账"这个判据）：

1. state_account   混合精度 AdamW 的训练状态静态账：参数、梯度、优化器状态、
                   激活四项，按 bytes/param 与形状推算。分片省哪一项、省多少，
                   全部用这套账对账（口径来自 07 篇四件套）。
2. StageProbe      六阶段显存探针（07 篇 stage_probe 的分片版）：模型加载、
                   优化器初始化、forward、backward、step、zero_grad 打点，
                   记录 allocated / reserved / peak，还原 step 内显存时间线。
3. run_meta        两机留痕：rank、device、GPU 名、torch 版本、语料哈希、
                   config 哈希（沿用 12 篇 xnode_check 的 provenance 字段）。

口径说明（全文统一）：
    allocated   分配器已交给张量的字节数
    reserved    分配器向 CUDA 要到的字节数（含碎片）
    peak        max_memory_allocated 的运行至今峰值
    常驻显存    zero_grad 后、下一 step 前的 allocated（不含激活，含状态）
    峰值显存    一个 step 内的 peak
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch


# ---------------------------------------------------------------- 状态静态账


@dataclass
class StateAccount:
    """混合精度 AdamW 训练状态的静态账（bytes/param 口径）。

    bytes_per_param 由 probe 实测给出（参数 2 + 梯度 2 + 优化器状态实测值），
    不硬编码假设：不同 PyTorch 版本的 AdamW 状态 dtype 可能不同。
    DDP 下每 rank 持有全量；ZeRO-1 切优化器状态；ZeRO-2 再切梯度；
    ZeRO-3/FSDP 再切参数。world_size 张卡的分片每 rank 除以 world。
    """

    n_params: int
    world_size: int = 1
    bytes_per_param: float = 12.0
    shard_optimizer: bool = False   # ZeRO-1 及以上
    shard_gradient: bool = False    # ZeRO-2 及以上
    shard_param: bool = False       # ZeRO-3 / FSDP

    @property
    def param_bytes(self) -> int:
        b = self.n_params * 2  # bf16 参数
        return b // self.world_size if self.shard_param else b

    @property
    def grad_bytes(self) -> int:
        b = self.n_params * 2  # bf16 梯度
        return b // self.world_size if self.shard_gradient else b

    @property
    def optim_bytes(self) -> int:
        b = self.n_params * (self.bytes_per_param - 4)  # 扣除参数与梯度
        return int(b) // self.world_size if self.shard_optimizer else int(b)

    @property
    def state_bytes(self) -> int:
        """常驻状态合计（不含激活）。"""
        return self.param_bytes + self.grad_bytes + self.optim_bytes

    def as_dict(self) -> dict:
        return {
            "n_params": self.n_params,
            "world_size": self.world_size,
            "bytes_per_param": self.bytes_per_param,
            "shard_optimizer": self.shard_optimizer,
            "shard_gradient": self.shard_gradient,
            "shard_param": self.shard_param,
            "param_mb": _mb(self.param_bytes),
            "grad_mb": _mb(self.grad_bytes),
            "optim_mb": _mb(self.optim_bytes),
            "state_mb": _mb(self.state_bytes),
        }


def count_params(model: torch.nn.Module) -> int:
    """去重后的参数量（tied embedding 只计一次）。"""
    seen: set[int] = set()
    total = 0
    for p in model.parameters():
        if id(p) not in seen:
            seen.add(id(p))
            total += p.numel()
    return total


def _mb(n: int | float) -> float:
    return round(n / 1024 / 1024, 1)


# ---------------------------------------------------------------- 阶段探针


@dataclass
class StageProbe:
    """六阶段显存探针：一个 step 的显存时间线（07 篇 stage_probe 口径）。

    分片训练关注的额外字段：grad 与 param 在 step 边界前后的 allocated 落差，
    用来区分"状态常驻"与"step 内瞬时聚合"。
    """

    device: str = "cuda"
    records: list[dict] = field(default_factory=list)

    def snap(self, stage: str) -> dict:
        if not torch.cuda.is_available():
            rec = {"stage": stage, "allocated_mb": 0.0,
                   "reserved_mb": 0.0, "peak_mb": 0.0}
        else:
            rec = {
                "stage": stage,
                "allocated_mb": _mb(torch.cuda.memory_allocated(self.device)),
                "reserved_mb": _mb(torch.cuda.memory_reserved(self.device)),
                "peak_mb": _mb(torch.cuda.max_memory_allocated(self.device)),
            }
        self.records.append(rec)
        return rec

    def reset_peak(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

    def as_dict(self) -> dict:
        return {"device": self.device, "stages": self.records}


# ---------------------------------------------------------------- 留痕


def run_meta(rank: int, world_size: int, config: dict,
             corpus_sha256: str = "") -> dict:
    """两机留痕（12 篇 xnode_check 的 provenance 字段子集）。

    注意：不写主机名、IP 等本地标识；节点区分用 gpu 型号 + rank 号表达
    （隐私规则：发布材料用 FLOAT_3070 / FLOAT_4090 别名）。
    """
    gpu_name = (torch.cuda.get_device_name(0)
                if torch.cuda.is_available() else "cpu")
    return {
        "rank": rank,
        "world_size": world_size,
        "gpu": gpu_name,
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "config": config,
        "config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()).hexdigest()[:16],
        "corpus_sha256": corpus_sha256,
    }


def corpus_sha256(path: str | Path) -> str:
    """语料文件哈希，保证两机同一份数据（12 篇 same_corpus 判据）。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def env_rank_world() -> tuple[int, int]:
    """从 torchrun 环境变量读 rank/world_size（单进程默认 1/1）。"""
    return int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
