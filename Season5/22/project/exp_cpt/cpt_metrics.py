"""exp_cpt.metrics —— 22 篇 Continual Pretraining Lab 纯函数指标层。

只依赖 numpy。覆盖提纲要求的量化口径：

1. 两段式学习率（re-warmup + cosine 衰减）与不 re-warmup 对照
2. replay 比例混合（领域/通用语料按比例组 batch）
3. 遗忘量化：参数相对底座的平均变化量 + 表示相似度（余弦）
4. 双评测口径：领域分数、通用分数、相对基线跌幅、权衡曲线
5. 达到目标领域分数所需 token 数（线性插值）
6. 扩词表三种 embedding 初始化：random / mean / subword_avg（18 篇方法）

全部为纯函数，CPU 上由 tests/ 单测覆盖，不碰 GPU。
"""

from __future__ import annotations

import math

import numpy as np


# ---------------------------------------------------------------- 两段式学习率


def lr_two_stage(step: int, total_steps: int, peak_lr: float,
                 rewarm_frac: float = 0.05, floor_frac: float = 0.1,
                 rewarm: bool = True) -> float:
    """两段式学习率：re-warmup（线性升温）+ cosine 衰减到 floor。

    rewarm=True：前 rewarm_frac·total 步从 0 线性升到峰值（CPT 惯例，
    避免一上来就用峰值撞坏底座），之后 cosine 衰减到 peak×floor_frac。
    rewarm=False：从第 0 步就取 cosine（起点即峰值），对照用。

    step 是 1-based（第 1 步训练前调用 lr(1)）。
    """
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if not 0.0 <= rewarm_frac < 1.0:
        raise ValueError("rewarm_frac must be in [0, 1)")
    if not 0.0 <= floor_frac <= 1.0:
        raise ValueError("floor_frac must be in [0, 1]")
    if step < 1:
        raise ValueError("step is 1-based (>=1)")
    floor = peak_lr * floor_frac

    if rewarm and rewarm_frac > 0:
        warm = max(1, int(round(rewarm_frac * total_steps)))
        if step <= warm:
            return peak_lr * step / warm
        decay_steps = total_steps - warm
        if decay_steps <= 0:
            return peak_lr
        t = min(1.0, (step - warm) / decay_steps)
    else:
        t = min(1.0, (step - 1) / max(1, total_steps - 1))

    cos = 0.5 * (1.0 + math.cos(math.pi * t))
    return floor + (peak_lr - floor) * cos


def lr_peak_ratio(peak_lr: float, base_pretrain_lr: float) -> float:
    """CPT 峰值相对底座原预训练学习率的比例（提纲要报的量）。"""
    if base_pretrain_lr <= 0:
        raise ValueError("base_pretrain_lr must be positive")
    return peak_lr / base_pretrain_lr


# ---------------------------------------------------------------- replay 混合


def replay_batch_plan(total_steps: int, replay_ratio: float) -> list[str]:
    """按 replay 比例规划每一步的 batch 来源（确定性、均匀交错）。

    每一步 batch 内部按比例混，还是整步整步交替？两种混合方式提纲都要看：
    这里给 step 级均匀交错（'domain' / 'replay' 交替，比例长期收敛到目标），
    batch 内混合由 train_cpt 的 collator 负责（token 级拼接）。
    返回长度 total_steps 的列表。
    """
    if not 0.0 <= replay_ratio <= 1.0:
        raise ValueError("replay_ratio must be in [0, 1]")
    plan: list[str] = []
    acc = 0.0
    for _ in range(total_steps):
        acc += replay_ratio
        if acc >= 0.5:
            plan.append("replay")
            acc -= 1.0
        else:
            plan.append("domain")
    return plan


def replay_realized_ratio(plan: list[str]) -> float:
    """实测 replay 步数占比（验收：与目标比例差 < 1 步）。"""
    if not plan:
        raise ValueError("plan must be non-empty")
    return sum(1 for s in plan if s == "replay") / len(plan)


def mix_token_ids(domain_ids: list[int], replay_ids: list[int],
                  replay_ratio: float, budget: int) -> list[int]:
    """token 级混合：按比例把通用语料 token 均匀撒进领域流（整数 Bresenham）。

    组成一条长度 budget 的训练流：domain 占 (1-ratio)，replay 占 ratio，
    数量精确（round 后 domain 补尾），位置均匀交错。不足则循环补齐
    （CPT 语料通常远小于预算）。
    """
    if budget <= 0:
        raise ValueError("budget must be positive")
    if not 0.0 <= replay_ratio <= 1.0:
        raise ValueError("replay_ratio must be in [0, 1]")
    n_replay = int(round(budget * replay_ratio))
    n_domain = budget - n_replay

    def take(ids: list[int], n: int) -> list[int]:
        if not ids and n > 0:
            raise ValueError("empty source ids")
        out: list[int] = []
        while len(out) < n:
            out.extend(ids)
        return out[:n]

    domain_part = take(domain_ids, n_domain)
    replay_part = take(replay_ids, n_replay)
    merged: list[int] = []
    di = ri = acc = 0
    for _ in range(budget):
        acc += n_replay
        if acc >= budget and ri < n_replay:
            merged.append(replay_part[ri])
            ri += 1
            acc -= budget
        else:
            merged.append(domain_part[di])
            di += 1
    return merged


# ---------------------------------------------------------------- 遗忘量化


def param_mean_change(base: np.ndarray, cur: np.ndarray) -> float:
    """参数相对底座的平均相对变化量：mean(|cur - base| / (|base| + eps))。

    base/cur 形状必须相同（同为某层权重展平或整体参数向量）。
    """
    if base.shape != cur.shape:
        raise ValueError("shape mismatch")
    base = np.asarray(base, dtype=np.float64)
    cur = np.asarray(cur, dtype=np.float64)
    return float(np.mean(np.abs(cur - base) / (np.abs(base) + 1e-8)))


def param_change_by_layer(base: dict[str, np.ndarray],
                          cur: dict[str, np.ndarray]) -> dict[str, float]:
    """分层参数变化量（找出遗忘主要发生在哪些层）。键集合必须一致。"""
    if set(base) != set(cur):
        raise ValueError("layer key mismatch")
    return {k: param_mean_change(base[k], cur[k]) for k in base}


def representation_similarity(base_states: np.ndarray,
                              cur_states: np.ndarray) -> float:
    """表示相似度：同一批输入在底座/续训后 hidden states 的平均余弦相似度。

    base_states/cur_states: [n_tokens, hidden]。逐 token 余弦后取均值。
    相似度下降 = 表示漂移（即使参数变化小，表示也可能漂）。
    """
    if base_states.shape != cur_states.shape:
        raise ValueError("shape mismatch")
    if base_states.ndim != 2:
        raise ValueError("must be 2-D [n_tokens, hidden]")
    a = np.asarray(base_states, dtype=np.float64)
    b = np.asarray(cur_states, dtype=np.float64)
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    valid = (na > 1e-12) & (nb > 1e-12)
    if not valid.any():
        raise ValueError("all-zero rows")
    cos = np.sum(a[valid] * b[valid], axis=1) / (na[valid] * nb[valid])
    return float(np.mean(cos))


# ---------------------------------------------------------------- 双评测口径


def relative_drop(base_score: float, cur_score: float) -> float:
    """相对基线跌幅（正 = 变差）：(base - cur) / base。

    分数约定：评测用 ppl（越低越好）或准确率（越高越好）时，
    统一先转换成"分数越高越好"再进来。ppl 请传 1/ppl 或用 ppl_drop。
    """
    if base_score == 0:
        raise ValueError("base_score must be non-zero")
    return (base_score - cur_score) / base_score


def ppl_drop(base_ppl: float, cur_ppl: float) -> float:
    """ppl 相对跌幅（正 = 变差 = ppl 上升）。"""
    if base_ppl <= 0 or cur_ppl <= 0:
        raise ValueError("ppl must be positive")
    return (cur_ppl - base_ppl) / base_ppl


def tradeoff_point(domain_drop: float, general_drop: float) -> dict:
    """一个配置的权衡点：领域跌幅（越负=领域提升越大越好）× 通用跌幅（越小越好）。

    domain_drop 用 relative_drop（领域分数上升时为负），
    general_drop 用 ppl_drop（通用 ppl 上升时为正）。
    """
    return {"domain_drop": float(domain_drop), "general_drop": float(general_drop)}


def tradeoff_dominates(a: dict, b: dict) -> bool:
    """a 是否帕累托优于 b（两轴都不差且至少一轴更好）。"""
    d_better = a["domain_drop"] <= b["domain_drop"]
    g_better = a["general_drop"] <= b["general_drop"]
    strict = (a["domain_drop"] < b["domain_drop"]
              or a["general_drop"] < b["general_drop"])
    return d_better and g_better and strict


# ---------------------------------------------------------------- token 效率


def tokens_to_target(tokens: list[int], domain_scores: list[float],
                     target: float) -> float | None:
    """达到目标领域分数所需 token 数（相邻评测点线性插值）。

    要求 domain_scores 随 tokens 单调上升段内插值；若从未达到目标返回 None；
    若第 1 点已超过目标返回第 1 点 token 数（不外推）。
    """
    if len(tokens) != len(domain_scores) or not tokens:
        raise ValueError("length mismatch or empty")
    if domain_scores[0] >= target:
        return float(tokens[0])
    for i in range(1, len(tokens)):
        if domain_scores[i] >= target:
            t0, t1 = tokens[i - 1], tokens[i]
            s0, s1 = domain_scores[i - 1], domain_scores[i]
            if s1 == s0:
                return float(t1)
            frac = (target - s0) / (s1 - s0)
            return float(t0 + frac * (t1 - t0))
    return None


# ---------------------------------------------------------------- 扩词表初始化


def init_new_embeddings(method: str, old_emb: np.ndarray,
                        new_subword_ids: list[list[int]],
                        rng: np.random.Generator) -> np.ndarray:
    """新 token 的 embedding 初始化（三种方法对照）。

    method:
      - 'random'：按旧 embedding 的 std 正态采样
      - 'mean'：全部旧 embedding 的均值
      - 'subword_avg'：新 token 文本切成旧子词后取子词 embedding 均值（18 篇方法）
    old_emb: [vocab, hidden]；new_subtoken_ids: 每个新 token 对应的旧子词 id 列表。
    返回 [n_new, hidden]。
    """
    if old_emb.ndim != 2:
        raise ValueError("old_emb must be [vocab, hidden]")
    n_new = len(new_subword_ids)
    if n_new == 0:
        return np.zeros((0, old_emb.shape[1]), dtype=np.float64)
    emb = np.asarray(old_emb, dtype=np.float64)

    if method == "random":
        std = float(np.std(emb))
        return rng.normal(0.0, std, size=(n_new, emb.shape[1]))
    if method == "mean":
        mu = emb.mean(axis=0)
        return np.tile(mu, (n_new, 1))
    if method == "subword_avg":
        out = np.empty((n_new, emb.shape[1]), dtype=np.float64)
        for i, ids in enumerate(new_subword_ids):
            if not ids:
                raise ValueError(f"new token {i} has no subword ids")
            bad = [j for j in ids if j < 0 or j >= emb.shape[0]]
            if bad:
                raise ValueError(f"subword id out of range: {bad}")
            out[i] = emb[ids].mean(axis=0)
        return out
    raise ValueError(f"unknown method: {method}")


def expanded_vocab_size(base_vocab: int, n_new: int, spare_rows: int) -> int:
    """扩词表后的 vocab 大小；超出底座余量行则报错（Qwen 余量 293）。"""
    if n_new < 0:
        raise ValueError("n_new must be >= 0")
    if n_new > spare_rows:
        raise ValueError(f"n_new={n_new} exceeds spare_rows={spare_rows}")
    return base_vocab + n_new


def eval_window_plan(n_ids: int, seq_len: int, score_tail: bool = False) -> dict:
    """评测切窗计划：完整窗与尾窗各预测多少 token。

    完整窗 i 覆盖 ids[i*seq_len : (i+1)*seq_len+1]（HF labels 内部移位
    一次，预测 seq_len 个），n_windows 个完整窗共预测 ids[1:n_windows*
    seq_len]。尾部 ids[n_windows*seq_len : ] 长度恒 ≥1：score_tail=True
    且长度 ≥2 时作短窗补计（首 token 作上下文，预测其余 len-1 个）；
    长度恰为 1 时该 token 已被最后一个完整窗预测，不重复计。

    score_tail=True 时 total_preds = n_ids - 1，即除全文首 token 外
    全覆盖——nats/char 的分子与分母（全文字符数）对应同一文本，
    跨 tokenizer 严格可比；False 时只计完整窗，与既有结果 JSON 的
    ppl 口径逐位一致。
    """
    if n_ids < seq_len + 1:
        raise ValueError(f"text too short: {n_ids} tokens < seq_len+1")
    n_windows = (n_ids - 1) // seq_len
    complete = n_windows * seq_len
    tail_len = n_ids - complete          # 恒 ≥ 1
    tail_preds = max(0, tail_len - 1) if score_tail else 0
    return {"n_windows": n_windows, "complete_preds": complete,
            "tail_preds": tail_preds, "total_preds": complete + tail_preds}
