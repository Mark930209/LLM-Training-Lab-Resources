"""exp_cpt 纯函数单测（CPU）。"""

import math

import numpy as np
import pytest

from cpt_metrics import (
    eval_window_plan,
    expanded_vocab_size,
    init_new_embeddings,
    lr_peak_ratio,
    lr_two_stage,
    mix_token_ids,
    param_change_by_layer,
    param_mean_change,
    ppl_drop,
    relative_drop,
    replay_batch_plan,
    replay_realized_ratio,
    representation_similarity,
    tokens_to_target,
    tradeoff_dominates,
    tradeoff_point,
)


# ------------------------------------------------------------ 两段式学习率


def test_lr_two_stage_reaches_peak_at_end_of_warmup():
    # total=100, rewarm 5%: 第 5 步恰好峰值
    lr5 = lr_two_stage(5, 100, 3e-5, rewarm_frac=0.05)
    assert abs(lr5 - 3e-5) < 1e-12
    # warmup 内单调升
    assert lr_two_stage(1, 100, 3e-5, rewarm_frac=0.05) < lr_two_stage(
        4, 100, 3e-5, rewarm_frac=0.05)


def test_lr_two_stage_decays_to_floor():
    lr_end = lr_two_stage(100, 100, 3e-5, rewarm_frac=0.05, floor_frac=0.1)
    assert abs(lr_end - 3e-6) < 1e-12
    # 中点介于峰值与 floor 之间且单调下降
    mid = lr_two_stage(52, 100, 3e-5, rewarm_frac=0.05, floor_frac=0.1)
    assert 3e-6 < mid < 3e-5


def test_lr_no_rewarm_starts_at_peak():
    lr1 = lr_two_stage(1, 100, 3e-5, rewarm_frac=0.05, rewarm=False)
    assert abs(lr1 - 3e-5) < 1e-12
    lr2 = lr_two_stage(2, 100, 3e-5, rewarm_frac=0.05, rewarm=False)
    assert lr2 < lr1


def test_lr_peak_ratio():
    assert abs(lr_peak_ratio(3e-5, 1e-3) - 0.03) < 1e-12
    with pytest.raises(ValueError):
        lr_peak_ratio(3e-5, 0.0)


# ------------------------------------------------------------ replay 混合


def test_replay_plan_ratio_converges():
    plan = replay_batch_plan(1000, 0.2)
    assert len(plan) == 1000
    r = replay_realized_ratio(plan)
    assert abs(r - 0.2) < 0.01  # 差 < 1 步量级
    plan0 = replay_batch_plan(50, 0.0)
    assert all(s == "domain" for s in plan0)


def test_mix_token_ids_ratio_and_budget():
    domain = list(range(100))
    replay = list(range(1000, 1100))
    mixed = mix_token_ids(domain, replay, 0.2, 1000)
    assert len(mixed) == 1000
    n_replay = sum(1 for x in mixed if x >= 1000)
    assert abs(n_replay - 200) <= 2
    # 纯领域 / 纯 replay 边界
    assert mix_token_ids(domain, replay, 0.0, 50) == list(range(50))
    pure = mix_token_ids(domain, replay, 1.0, 50)
    assert all(x >= 1000 for x in pure)


class _FakeTok:
    """1 字符 = 1 token 的玩具 tokenizer（测试流构造用）。"""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


def test_build_doc_stream_ratio_and_continuity():
    from cpt_data import build_doc_stream

    # 文档级混合：比例收敛（误差 < 一篇文档的 token 数）且文档内部连续
    dom = ["D" * 10] * 20      # 每篇 10 token
    rep = ["R" * 10] * 20
    s = build_doc_stream(dom, rep, _FakeTok(), budget=200, replay_ratio=0.2)
    assert len(s) == 200
    n_rep = sum(1 for x in s if x == ord("R"))
    assert abs(n_rep - 40) <= 10   # 误差上界 = 一篇文档长度
    # 文档内部连续：同一篇的 token 必须相邻成块（无逐 token 交错）
    runs = [1] + [1 if s[i] != s[i - 1] else 0 for i in range(1, len(s))]
    n_blocks = sum(runs)
    assert n_blocks <= 30          # 200 token / 10 token 每篇 ≤ 20 篇 + 余量
    assert all(len(v) == 1 for v in [set(s[i:i + 10]) for i in range(0, 20, 10)])


def test_build_doc_stream_edges():
    from cpt_data import build_doc_stream

    dom = ["D" * 10]
    rep = ["R" * 10]
    tok = _FakeTok()
    pure_dom = build_doc_stream(dom, rep, tok, budget=25, replay_ratio=0.0)
    assert all(x == ord("D") for x in pure_dom)
    pure_rep = build_doc_stream(dom, rep, tok, budget=25, replay_ratio=1.0)
    assert all(x == ord("R") for x in pure_rep)
    # 循环补齐（文档数少于预算所需）
    cyc = build_doc_stream(dom, rep, tok, budget=50, replay_ratio=0.5)
    assert len(cyc) == 50
    assert set(cyc) == {ord("D"), ord("R")}


# ------------------------------------------------------------ 遗忘量化


def test_param_mean_change():
    base = np.array([1.0, -2.0, 3.0])
    cur = np.array([1.1, -2.0, 2.7])
    got = param_mean_change(base, cur)
    expect = np.mean([0.1 / 1.0, 0.0, 0.3 / 3.0])
    assert abs(got - expect) < 1e-8  # 实现分母 +1e-8 防零，容差据此放宽
    with pytest.raises(ValueError):
        param_mean_change(np.zeros(3), np.zeros(4))


def test_param_change_by_layer():
    base = {"a": np.array([1.0, 1.0]), "b": np.array([2.0, 2.0])}
    cur = {"a": np.array([1.0, 1.0]), "b": np.array([3.0, 2.0])}
    got = param_change_by_layer(base, cur)
    assert got["a"] == 0.0
    assert abs(got["b"] - 0.25) < 1e-8


def test_representation_similarity():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(10, 8))
    assert abs(representation_similarity(a, a) - 1.0) < 1e-9
    b = a * 2.0  # 同方向缩放 → 相似度仍 1
    assert abs(representation_similarity(a, b) - 1.0) < 1e-9
    c = -a
    assert abs(representation_similarity(a, c) + 1.0) < 1e-9


# ------------------------------------------------------------ 双评测口径


def test_drops():
    assert abs(relative_drop(100.0, 90.0) - 0.1) < 1e-12
    assert abs(ppl_drop(10.0, 13.0) - 0.3) < 1e-12
    with pytest.raises(ValueError):
        relative_drop(0.0, 1.0)


def test_tradeoff_dominates():
    a = tradeoff_point(-0.10, 0.02)   # 领域涨 10% 通用 ppl +2%
    b = tradeoff_point(-0.05, 0.05)   # 领域涨 5% 通用 ppl +5%
    assert tradeoff_dominates(a, b)
    assert not tradeoff_dominates(b, a)
    assert not tradeoff_dominates(a, a)


# ------------------------------------------------------------ token 效率


def test_tokens_to_target_interpolates():
    got = tokens_to_target([0, 100, 200], [50.0, 60.0, 70.0], 65.0)
    assert abs(got - 150.0) < 1e-9
    assert tokens_to_target([0, 100], [50.0, 60.0], 80.0) is None
    assert tokens_to_target([0, 100], [70.0, 80.0], 65.0) == 0.0


# ------------------------------------------------------------ 扩词表初始化


def test_init_new_embeddings_random():
    rng = np.random.default_rng(1)
    old = rng.normal(0, 0.02, size=(10, 4))
    out = init_new_embeddings("random", old, [[0], [1]], rng)
    assert out.shape == (2, 4)
    assert not np.allclose(out[0], out[1])


def test_init_new_embeddings_mean():
    old = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = init_new_embeddings("mean", old, [[0]], np.random.default_rng(0))
    assert np.allclose(out[0], [2.0, 3.0])


def test_init_new_embeddings_subword_avg():
    old = np.array([[1.0, 0.0], [3.0, 2.0], [5.0, 4.0]])
    out = init_new_embeddings("subword_avg", old, [[0, 2], [1]],
                              np.random.default_rng(0))
    assert np.allclose(out[0], [3.0, 2.0])   # (1+5)/2, (0+4)/2
    assert np.allclose(out[1], [3.0, 2.0])
    with pytest.raises(ValueError):
        init_new_embeddings("subword_avg", old, [[]], np.random.default_rng(0))
    with pytest.raises(ValueError):
        init_new_embeddings("nope", old, [[0]], np.random.default_rng(0))


def test_expanded_vocab_size():
    assert expanded_vocab_size(151643, 256, 293) == 151899
    with pytest.raises(ValueError):
        expanded_vocab_size(151643, 300, 293)


# ------------------------------------------------------------ 评测切窗


def test_eval_window_plan_complete_only():
    plan = eval_window_plan(513, 512, score_tail=False)
    assert plan["n_windows"] == 1
    assert plan["complete_preds"] == 512
    assert plan["tail_preds"] == 0
    assert plan["total_preds"] == 512


def test_eval_window_plan_tail_covers_all_but_first():
    for n in (513, 600, 1024, 1025, 2048, 3000):
        plan = eval_window_plan(n, 512, score_tail=True)
        assert plan["total_preds"] == n - 1   # 除首 token 外全覆盖
    plan = eval_window_plan(600, 512, score_tail=True)
    assert plan["tail_preds"] == 87           # 尾段 88 token，首 token 作上下文


def test_eval_window_plan_single_token_tail():
    plan = eval_window_plan(513, 512, score_tail=True)
    assert plan["tail_preds"] == 0            # 尾部唯一 token 已被完整窗预测


def test_eval_window_plan_rejects_short_text():
    with pytest.raises(ValueError):
        eval_window_plan(512, 512)
