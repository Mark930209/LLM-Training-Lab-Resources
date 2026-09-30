"""run_pack.py —— E4 packing 与 document mask（失败案例 2）。

每条 2k 序列由 8 个 256-token 文档拼接：
- consistent 包：同一包内所有文档共用一套 key→value 映射（跨文档抄近路可行）；
- diverse 包：每文档独立映射（抄近路必错）。
两组训练：mask on（块对角 causal）vs mask off（跨文档可看）。
预期（失败案例 2）：mask off 在 consistent 数据上 loss 更低，但 diverse 评测崩塌。

用法：python run_pack.py --out out/e4_pack.json
"""

from __future__ import annotations

import argparse
import random
import time

import torch
import torch.nn.functional as F

import longctx_common as lc
from tiny_rope import TinyRoPE

DOC_LEN = 256
DOCS = 8
PAIRS_PER_DOC = 2


def make_pack(rng: random.Random, mode: str):
    """生成一个包：(ids [DOC_LEN*DOCS], ys, doc_ids)。

    泄漏设计：包内选一个"特殊对" (k*, v*) 只出现在文档 0；约一半文档是
    leaky 文档——它们的查询就是 k*，但自己的配对列表里没有 (k*, v*)。
    mask off 时模型可以跨文档抄文档 0 的答案 → 训练 loss 更低；
    mask on 时 leaky 文档的信息根本不可见（loss 有不可约下限）。
    diverse 评测全部用自包含文档 + 独立映射，抄近路必错。
    """
    S = DOC_LEN * DOCS
    ids = torch.full((S,), lc.PAD, dtype=torch.long)
    ys = torch.full((S,), -100, dtype=torch.long)
    doc_ids = torch.zeros(S, dtype=torch.long)
    keys = rng.sample(range(lc.KEY0, lc.KEY0 + lc.N_KEYS), 2 * DOCS + 2)
    vals = rng.sample(range(lc.VAL0, lc.VAL0 + lc.N_VALS), 2 * DOCS + 2)
    pack_map = dict(zip(keys, vals))
    k_star, v_star = keys[0], vals[0]

    def write_doc(d: int, pairs: list[tuple[int, int]], qkey: int):
        lo, hi = d * DOC_LEN, (d + 1) * DOC_LEN
        doc_ids[lo:hi] = d
        body = DOC_LEN - 4
        j = next(i for i, (k, _) in enumerate(pairs) if k == qkey) \
            if any(k == qkey for k, _ in pairs) else 0
        seg: list[int] = []
        per = body // len(pairs)
        for i, (k, v) in enumerate(pairs):
            seg += [k, v] + rng.choices(range(lc.FILL0, lc.FILL0 + lc.N_FILL),
                                        k=per - 2)
        seg = seg[:body]
        qpos_val = v_star if qkey == k_star else pack_map.get(qkey, pairs[0][1])
        seg += [lc.QUERY, qkey, lc.ANS, qpos_val]
        ids[lo:hi] = torch.tensor(seg)
        ys[hi - 2] = seg[-1]

    if mode == "consistent":
        # 文档 0 自包含 (k*, v*)；其余文档一半 leaky（查 k* 但自己没有）
        write_doc(0, [(k_star, v_star), (keys[1], vals[1])], k_star)
        for d in range(1, DOCS):
            other = [(keys[2 + d], vals[2 + d]), (keys[2 + DOCS + d], vals[2 + DOCS + d])]
            if rng.random() < 0.5:
                write_doc(d, other, k_star)                  # leaky：查 k*
            else:
                pair = (keys[2 + d], vals[2 + d])
                write_doc(d, [pair, other[1]], pair[0])      # 自包含
    else:
        # diverse：每文档独立映射、全部自包含
        for d in range(DOCS):
            pair = (keys[2 + d], vals[2 + d])
            write_doc(d, [pair, (keys[2 + DOCS + d], vals[2 + DOCS + d])],
                      pair[0])
    return ids, ys, doc_ids


def doc_mask(doc_ids: torch.Tensor) -> torch.Tensor:
    S = doc_ids.shape[0]
    same = doc_ids[:, None] == doc_ids[None, :]
    causal = torch.tril(torch.ones(S, S, dtype=torch.bool))
    return (same & causal).unsqueeze(0).unsqueeze(0)   # [1,1,S,S]


def train_variant(mask_on: bool, mode: str, steps: int, device: str):
    torch.manual_seed(20260930)
    model = TinyRoPE(vocab=lc.VOCAB).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4,
                            betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    curve = []
    t0 = time.perf_counter()
    for i in range(steps):
        rng = random.Random(5000 + i)
        ids, ys, doc_ids = make_pack(rng, mode)
        ids_b = ids.unsqueeze(0).to(device)
        ys_b = ys.unsqueeze(0).to(device)
        mask = doc_mask(doc_ids).to(device) if mask_on else None
        logits = model(ids_b, attn_mask=mask)
        loss = F.cross_entropy(logits.flatten(0, 1), ys_b.flatten(0, 1),
                               ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        curve.append(float(loss.detach()))
    return model, curve, round(time.perf_counter() - t0, 1)


def pack_eval(model, mode: str, mask_on: bool, device: str, trials: int = 16):
    """diverse 包上的逐文档捞针命中率。"""
    model.eval()
    hit = tot = 0
    with torch.no_grad():
        for t in range(trials):
            rng = random.Random(9000 + t)
            ids, ys, doc_ids = make_pack(rng, mode)
            mask = doc_mask(doc_ids).to(device) if mask_on else None
            logits = model(ids.unsqueeze(0).to(device), attn_mask=mask)
            for p in (ys != -100).nonzero():
                pos = p.item()
                hit += int(logits[0, pos].argmax().item() == ys[pos].item())
                tot += 1
    model.train()
    return hit / max(tot, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    results = {}
    for mask_on in (False, True):
        for mode in ("consistent",):
            tag = f"{'mask_on' if mask_on else 'mask_off'}_{mode}"
            model, curve, secs = train_variant(mask_on, mode, args.steps,
                                               args.device)
            results[tag] = {
                "train_loss_first": curve[0],
                "train_loss_last": curve[-1],
                "train_loss_curve": curve,
                "eval_consistent": pack_eval(model, "consistent", mask_on,
                                             args.device),
                "eval_diverse": pack_eval(model, "diverse", mask_on,
                                          args.device),
                "train_seconds": secs,
            }
            print(tag, "loss", round(curve[0], 3), "->", round(curve[-1], 3),
                  "eval_consistent", results[tag]["eval_consistent"],
                  "eval_diverse", results[tag]["eval_diverse"])

    out = {
        "experiment": "e4_pack_mask",
        "framework": "plain-pytorch(tiny_rope)",
        "results": results,
        "labels": {"REAL": "单卡真实运行"},
    }
    lc.save_json(out, args.out)


if __name__ == "__main__":
    main()
