"""ddp_common.py —— 11 篇 DDP Lab 的公共件。

设计要点（都服务于"与单卡逐位对齐"这个唯一判据）：

1. 固定样本池 FixedSampleDataset：所有 rank 用同一个 seed 生成同一份样本序列，
   保证"双卡两个分片的并集"与"单卡看到的全局 batch"是同一个样本集合。
   等价数学只要求集合相同（梯度是求和，可交换），不要求 batch 内顺序相同。

2. 模型复用 10 篇的 build_llama（同一个 12.93M 小 Llama），char 分词，
   四大名著语料，保证从单卡基线平滑过渡。

3. param_checksum：对所有参数做确定性哈希，单卡/双卡跑完比对，
   是"训练等价"最硬的证据（比 loss 曲线更难蒙混）。

4. setup_dist：gloo/nccl 统一入口。gloo 用于单卡 2-rank（NCCL 拒绝同卡多 rank），
   nccl 用于跨机真双卡。
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import Dataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exp_scale.data import CharTokenizer  # noqa: E402


# ---------------------------------------------------------------- 进程组

def setup_dist(backend: str = "gloo") -> tuple[int, int, int]:
    """初始化进程组，返回 (rank, local_rank, world_size)。

    由 torchrun 注入 RANK / LOCAL_RANK / WORLD_SIZE 环境变量。
    backend 选择（实测）：
      - gloo：单卡也能起 2 rank，能驱动 CUDA 模型做 DDP（梯度走 CPU 中转），
        正确性与 nccl 一致。本篇主传输。
      - nccl：要求每 rank 独占一张 GPU，单卡 2 rank 会报 ncclInvalidUsage。
        跨机真双卡才用。
    """
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available() and backend == "nccl":
        torch.cuda.set_device(local_rank)
    elif torch.cuda.is_available():
        # gloo + CUDA 模型：两个 rank 都绑到 device 0（单卡场景），
        # DDP 内部把梯度搬到 CPU 做 gloo 集合通信再搬回。
        torch.cuda.set_device(local_rank % torch.cuda.device_count())
    return rank, local_rank, world


def cleanup_dist() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


def pick_device(backend: str) -> str:
    """gloo/nccl 都用 GPU 跑前向反向；只有无 CUDA 时退回 CPU。"""
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- 数据

class FixedSampleDataset(Dataset):
    """固定样本池：所有 rank 生成同一份样本序列。

    从真实语料切窗口，offsets 用固定 seed 生成，因此 rank0 与 rank1 的
    第 i 个样本完全相同。DistributedSampler 按 indices[rank::world] 分片，
    两个分片的并集就是整个池，与单卡顺序读全池是同一个样本集合。

    这是"双卡等价单卡"的数据基础：只要全局 batch 的样本集合一致，
    mean reduction 的梯度就和单卡逐位对齐（浮点求和顺序差异除外）。
    """

    def __init__(self, ids: torch.Tensor, seq_len: int, n_samples: int,
                 seed: int = 1234):
        self.ids = ids
        self.seq_len = seq_len
        g = torch.Generator().manual_seed(seed)
        self.offsets = torch.randint(
            0, max(1, len(ids) - seq_len - 1), (n_samples,), generator=g).tolist()

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, idx: int):
        o = self.offsets[idx]
        window = self.ids[o:o + self.seq_len + 1]
        return window[:-1], window[1:]


def load_corpus_ids(data_dir: str = "exp_scale/data") -> tuple[torch.Tensor, int]:
    """加载四大名著语料并 char 编码。与 10 篇同源。"""
    corpus_path = Path(data_dir) / "corpus_large.txt"
    text = corpus_path.read_text(encoding="utf-8")
    tok = CharTokenizer(text)
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    return ids, tok.vocab_size


# ---------------------------------------------------------------- 模型

def build_model(vocab_size: int, seq_len: int, device: str,
                hidden: int = 384, layers: int = 6, heads: int = 6,
                head_dim: int = 64, intermediate: int = 1024):
    """复用 10 篇的 build_llama：config 唯一输入，随机初始化，不下载权重。

    所有 rank 用同一个 seed 初始化，DDP 构造时再 broadcast rank0 权重，
    双保险保证起点一致。
    """
    from exp_hf.adapters import build_llama
    model = build_llama(vocab_size, hidden=hidden, layers=layers, heads=heads,
                        head_dim=head_dim, intermediate=intermediate,
                        max_seq_len=max(seq_len, 512))
    return model.to(device)


# ---------------------------------------------------------------- 校验和

def param_checksum(model: torch.nn.Module) -> str:
    """对所有参数做确定性 sha256，取前 16 位。

    单卡与双卡跑完同样步数后比对：checksum 一致 = 参数逐位对齐 = 训练等价。
    用 float32 的字节表示，避免不同 device 的浮点格式差异。
    DDP 包装后参数在 model.module 里，这里统一解包。
    """
    m = model.module if hasattr(model, "module") else model
    h = hashlib.sha256()
    for p in m.parameters():
        h.update(p.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def param_l2(model: torch.nn.Module) -> float:
    """参数 L2 范数，作为 checksum 之外的连续量对照。"""
    m = model.module if hasattr(model, "module") else model
    return float(sum((p.detach().float() ** 2).sum() for p in m.parameters()) ** 0.5)


def grad_max_abs_err(g_a: dict, g_b: dict) -> float:
    """两组梯度（按参数名索引）的最大绝对误差。parity_check 用。"""
    worst = 0.0
    for k in g_a:
        if k in g_b:
            d = (g_a[k] - g_b[k]).abs().max().item()
            worst = max(worst, d)
    return worst


def collect_grads(model: torch.nn.Module) -> dict:
    """抓取当前所有参数的梯度副本（解包 DDP）。"""
    m = model.module if hasattr(model, "module") else model
    return {n: (p.grad.detach().clone() if p.grad is not None
                else torch.zeros_like(p)) for n, p in m.named_parameters()}


# ---------------------------------------------------------------- 聚合

def all_reduce_mean(value: float, device: str) -> float:
    """跨 rank 求均值。日志聚合用：本地 loss 各 rank 不同，全局指标要聚合。"""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return value
    t = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item() / dist.get_world_size())


def all_gather_values(value: float, device: str) -> list:
    """收集所有 rank 的标量，rank0 用于核对各 rank 本地 loss 是否不同。"""
    if not dist.is_initialized():
        return [value]
    world = dist.get_world_size()
    t = torch.tensor([value], dtype=torch.float64, device=device)
    out = [torch.zeros(1, dtype=torch.float64, device=device) for _ in range(world)]
    dist.all_gather(out, t)
    return [float(x.item()) for x in out]
