"""longctx_common.py —— 长程关联检索任务 + 大海捞针评测 + 固定数据。

任务格式（合成，vocab 512）：
    k1 v1 <filler…> k2 v2 <filler…> … kN vN <query> kJ <ans> vJ
- key 词元：10..73；value 词元：200..263；filler：300..329；
- loss 只算 <ans> 后那个 value 位置（label=-100 其余）——命中率即精确匹配。
- 捞针网格：序列长度 S × 查询深度 d（配对块在序列中的相对位置）。
"""

from __future__ import annotations

import json
import os
import random

import torch

PAD, QUERY, ANS = 0, 1, 2
KEY0, N_KEYS = 10, 32
VAL0, N_VALS = 200, 32
FILL0, N_FILL = 300, 30
VOCAB = 512

TRAIN_SEED = 20260930
EVAL_SEED = 31337


def make_seq(rng: random.Random, seq_len: int, n_pairs: int,
             query_depth: float | None = 0.5) -> tuple[list[int], int]:
    """生成一条序列，返回 (ids, answer_pos)。query_depth=None 时随机深度。"""
    assert n_pairs >= 2
    body = seq_len - 4   # 留 <query> kJ <ans> vJ
    slots = [body // n_pairs] * n_pairs
    j = rng.randrange(n_pairs) if query_depth is None \
        else min(n_pairs - 1, int(query_depth * n_pairs))
    ids: list[int] = []
    keys, vals = rng.sample(range(KEY0, KEY0 + N_KEYS), n_pairs), \
        rng.sample(range(VAL0, VAL0 + N_VALS), n_pairs)
    for i in range(n_pairs):
        ids += [keys[i], vals[i]]
        fill = rng.choices(range(FILL0, FILL0 + N_FILL), k=slots[i] - 2)
        ids += fill
    ids = ids[:body]
    if len(ids) < body:                      # 槽位整除余数用 filler 补齐
        ids += rng.choices(range(FILL0, FILL0 + N_FILL), k=body - len(ids))
    ids += [QUERY, keys[j], ANS, vals[j]]
    return ids, len(ids) - 1


def make_batch(batch: int, seq_len: int, n_pairs: int,
               seed: int, query_depth: float | None = 0.5):
    rng = random.Random(seed)
    xs = torch.full((batch, seq_len), PAD, dtype=torch.long)
    ys = torch.full((batch, seq_len), -100, dtype=torch.long)
    for b in range(batch):
        ids, apos = make_seq(rng, seq_len, n_pairs, query_depth)
        xs[b] = torch.tensor(ids)
        # 标签写在 <ans> 位置：从 <ans> 的隐状态预测下一词 = vJ
        # （写在 vJ 自身位置会让模型抄输入——loss 归零但 eval 全错的自骗 bug）
        ys[b, apos - 1] = ids[apos]
    return xs, ys


def needle_eval(model, device: torch.Tensor.device, seq_len: int,
                depth: float, n_pairs: int, trials: int = 20,
                seed: int = EVAL_SEED) -> float:
    """大海捞针命中率：模型 argmax 是否答对被查配对的 value。"""
    model.eval()
    hit = 0
    with torch.no_grad():
        for t in range(trials):
            xs, ys = make_batch(1, seq_len, n_pairs, seed + t, query_depth=depth)
            logits = model(xs.to(device))
            apos = (ys != -100).nonzero()[0, 1].item()   # <ans> 位置
            pred = logits[0, apos].argmax().item()
            hit += int(pred == ys[0, apos].item())
    model.train()
    return hit / trials


def short_task_loss(model, device, trials: int = 20, seed: int = 7):
    """短文本回测：512 长度、少配对的 held-out 平均 loss。"""
    model.eval()
    tot = 0.0
    with torch.no_grad():
        for t in range(trials):
            xs, ys = make_batch(4, 512, 6, seed + t)   # 与训练同密度（~85 词/对）
            logits = model(xs.to(device))
            loss = torch.nn.functional.cross_entropy(
                logits.flatten(0, 1), ys.to(device).flatten(0, 1),
                ignore_index=-100)
            tot += float(loss)
    model.train()
    return tot / trials


def save_json(obj: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
