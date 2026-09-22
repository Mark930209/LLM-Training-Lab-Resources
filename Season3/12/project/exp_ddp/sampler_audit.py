"""sampler_audit.py —— DistributedSampler 分片审计（不依赖训练，纯 CPU 可跑）。

审计三件事：
  1. 覆盖率与重复数：所有 rank 的分片并集是否恰好覆盖样本池、有无重复
  2. 交错分片语义：rank r 是否拿到 indices[r::world]
  3. set_epoch：不同 epoch 的 shuffle 顺序是否真的变化；漏调 set_epoch 时
     每个 epoch 顺序是否恒定（no_set_epoch 故障的静态证据）

用法：
  ./.venv/bin/python -m exp_ddp.sampler_audit --n-samples 64 --world 2 --epochs 3 \
      --out results/Season3/12/sampler_audit.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler


class _RangeDataset(Dataset):
    """审计专用：__getitem__ 返回自己的下标，分片结果一目了然。"""

    def __init__(self, n: int):
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> int:
        return i


def audit_shards(n_samples: int, world: int, seed: int, batch: int) -> dict:
    """静态审计：覆盖率、重复数、交错语义。不需要真的起进程组。"""
    ds = _RangeDataset(n_samples)
    shards = []
    for r in range(world):
        s = DistributedSampler(ds, num_replicas=world, rank=r, shuffle=True,
                               seed=seed, drop_last=True)
        s.set_epoch(0)
        shards.append(list(s))

    flat = [i for sh in shards for i in sh]
    cnt = Counter(flat)
    per_rank = [len(sh) for sh in shards]

    # 交错语义：DistributedSampler shuffle=True 时是"先全局 shuffle 再交错"，
    # 所以各 rank 分片应满足 shard_r[k] == perm[r + k*world]（perm 是全局
    # 打乱后的序列）。用 rank0..world-1 的分片交错重组，检查能否还原一个
    # 无重复无遗漏的排列。
    interleaved = [shards[r][k] for k in range(per_rank[0]) for r in range(world)]
    is_permutation = sorted(interleaved) == list(range(len(interleaved)))

    # 每步的全局 batch：第 k 步各 rank 取自己分片的第 k 个 batch，
    # 两个 rank 同一步的样本并集应当互不相交
    step_overlap = 0
    n_steps = per_rank[0] // batch
    for k in range(n_steps):
        a = set(shards[0][k * batch:(k + 1) * batch])
        b = set(shards[1][k * batch:(k + 1) * batch]) if world > 1 else set()
        step_overlap += len(a & b)

    return {
        "n_samples": n_samples,
        "world": world,
        "seed": seed,
        "per_rank_samples": per_rank,
        "total_distributed": len(flat),
        "duplicates": sum(v - 1 for v in cnt.values() if v > 1),
        "coverage": round(len(cnt) / n_samples, 4),
        "drop_last_note": ("样本池不能被 world×batch 整除时尾部被丢弃，"
                           "coverage < 1 是 drop_last 的预期行为"
                           if len(cnt) < n_samples else "全覆盖"),
        "interleaved_is_permutation": is_permutation,
        "same_step_cross_rank_overlap": step_overlap,
    }


def audit_set_epoch(n_samples: int, world: int, seed: int, epochs: int) -> dict:
    """set_epoch 审计：epoch 变了顺序是否变；不调 set_epoch 顺序是否恒定。"""
    ds = _RangeDataset(n_samples)
    s = DistributedSampler(ds, num_replicas=world, rank=0, shuffle=True,
                           seed=seed, drop_last=True)

    with_set_epoch = []
    for e in range(epochs):
        s.set_epoch(e)
        with_set_epoch.append(list(s)[:16])

    without_set_epoch = []
    s2 = DistributedSampler(ds, num_replicas=world, rank=0, shuffle=True,
                            seed=seed, drop_last=True)
    s2.set_epoch(0)
    for _ in range(epochs):
        without_set_epoch.append(list(s2)[:16])   # 从不更新 epoch

    return {
        "epochs": epochs,
        "with_set_epoch_all_differ": all(
            with_set_epoch[i] != with_set_epoch[j]
            for i in range(epochs) for j in range(i + 1, epochs)),
        "with_set_epoch_first3": [w[:8] for w in with_set_epoch[:3]],
        "without_set_epoch_all_same": all(
            without_set_epoch[i] == without_set_epoch[0] for i in range(epochs)),
        "without_set_epoch_first3": [w[:8] for w in without_set_epoch[:3]],
        "note": ("with：每个 epoch 调 set_epoch(e)，顺序逐 epoch 变化；"
                 "without：只在开头调一次，之后每个 epoch 顺序完全相同，"
                 "多 epoch 训练退化成反复过同一顺序"),
    }


def audit_no_sampler(n_samples: int, world: int, seed: int, steps: int,
                     batch: int) -> dict:
    """no_sampler 故障的静态证据：各 rank 独立 shuffle 全池。

    模拟 ddp_train 的 no_sampler 路径：每个 rank 用自己的 generator
    shuffle 整个池。统计 steps 步内每个样本被训练的总次数——正确分片时
    每个样本每 epoch 恰好 1 次；独立 shuffle 时平均 world 次且分布不均。
    """
    counts = Counter()
    for r in range(world):
        g = torch.Generator().manual_seed(seed + r)
        perm = torch.randperm(n_samples, generator=g).tolist()
        for k in range(steps * batch):
            counts[perm[k % n_samples]] += 1

    trained = sum(1 for v in counts.values() if v > 0)
    dup = sum(v - 1 for v in counts.values() if v > 1)
    return {
        "n_samples": n_samples,
        "world": world,
        "steps": steps,
        "batch_per_rank": batch,
        "samples_trained_at_least_once": trained,
        "duplicate_draws": dup,
        "max_times_one_sample_seen": max(counts.values()) if counts else 0,
        "correct_shard_reference": ("正确分片时每个样本每 epoch 恰好被训练 1 次"
                                    "（全局并集无重复）"),
        "note": ("独立 shuffle 下两个 rank 都会遍历全池：同一样本被训练约 "
                 "world 次，而另一些样本可能在窗口内 0 次。等效于全局 batch "
                 "里出现重复样本，梯度权重被扭曲"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=64)
    ap.add_argument("--world", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    result = {
        "shards": audit_shards(args.n_samples, args.world, args.seed, args.batch),
        "set_epoch": audit_set_epoch(args.n_samples, args.world, args.seed,
                                     args.epochs),
        "no_sampler": audit_no_sampler(args.n_samples, args.world, args.seed,
                                       args.steps, args.batch),
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
