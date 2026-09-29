"""pipeline.py —— exp_hparam 统一入口（20 篇 HParam Lab）。

模式：
  plan          : 只算不训，离线估算各子实验格数与 GPU 时长（决策用）
  noise         : 基准配置 4 seeds，建立噪声带（判读标尺）
  lr_range      : LR range test，记录 loss-lr 曲线找可用区间
  sweep         : LR × batch（24 格），检验平方根 vs 线性缩放
  warmup_decay  : warmup 比例 × 衰减策略（9 格）
  grad_accum    : 梯度累积等效边界（4 格）
  scaling       : 4 档宽度拟合 L(N)=a·N^b 并外推到 1032
  compute_budget: 等算力 C≈6ND，4 档宽度变 N/D 比
  mup           : 宽度超参迁移简化验证（8 格）
  predict_larger: 用 sweep 最优 LR 按 1/width 预测更大一档（2 格）
  all           : 依次跑全部真实模式（noise→…→predict_larger）

架构固定 19 篇 modern；数据底座与 17/18/19 篇完全一致（同语料、同配方、
同词表 6120、同评测集，指纹可跨篇核对）。结果 JSON 一律 --out 直写 /mnt/d。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import yaml

from exp_recipe import corpus as corpus17
from exp_recipe import recipe as recipe17
from exp_recipe.train_eval import TrainConfig

from .hparam_metrics import (
    n_params_dense, non_emb_params, tok_per_sec, train_seconds,
    fit_power_law, predict_loss, rel_error, flops_6nd,
)

HERE = Path(__file__).resolve().parent


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _code_sha256() -> dict[str, str]:
    here = Path(__file__).parent
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (here / "pipeline.py", here / "hparam_metrics.py",
                      here / "train_hparam.py", here / "config.yaml")}


# ---------------- 数据装载（复用 19 篇 → 17 篇口径）----------------

def _load_corpus(cfg: dict) -> dict:
    ccfg = cfg["corpus"]
    nl_text = corpus17.load_nl_corpus(ccfg["nl_path"])
    code_docs = corpus17.load_code_docs()
    domains = corpus17.split_domains(nl_text, code_docs)
    nl_docs = corpus17.split_nl_train_docs(
        domains["nl_train"], int(ccfg.get("nl_chunk_chars", 4000)))
    return {
        "nl_train": domains["nl_train"],
        "code_train_text": "\n".join(t for _, t in domains["code_train"]),
        "nl_eval": domains["nl_eval"],
        "code_eval_text": "\n".join(t for _, t in domains["code_eval"]),
        "nl_docs": nl_docs,
        "code_texts": [t for _, t in domains["code_train"]],
        "fingerprint": corpus17.eval_fingerprint(domains),
    }


class DataBundle:
    """一次制备，全子实验共享：tokenizer、train_ids、评测文本、指纹。"""

    def __init__(self, cfg: dict):
        import torch
        from exp_scale.data import CharTokenizer
        self.cfg = cfg
        data = _load_corpus(cfg)
        self.fingerprint = data["fingerprint"]
        self.nl_eval = data["nl_eval"]
        self.code_eval = data["code_eval_text"]
        vocab_expect = cfg["base"]["vocab_expect"]
        self.tokenizer = CharTokenizer(data["nl_train"] + "\n" + data["code_train_text"])
        if self.tokenizer.vocab_size != vocab_expect:
            raise RuntimeError(
                f"词表口径不符：期望 {vocab_expect}（17 篇），实得 {self.tokenizer.vocab_size}")
        self.vocab = self.tokenizer.vocab_size
        rs = cfg["recipe"]
        spec_r = recipe17.RecipeSpec(
            name=rs["name"], nl_ratio=float(rs["nl_ratio"]), dedup=rs["dedup"],
            quality=bool(rs["quality"]), char_budget=int(rs["char_budget"]))
        prep = recipe17.prepare_recipe(spec_r, data["nl_docs"], data["code_texts"])
        self.recipe_text_sha256 = prep["report"]["text_sha256"]
        self.train_ids = torch.tensor(self.tokenizer.encode(prep["text"]), dtype=torch.long)
        self.n_train_tokens = len(self.train_ids)

    def epochs_for(self, token_budget: int) -> float:
        return token_budget / max(1, self.n_train_tokens)


def _base_cfg(cfg: dict, **over) -> TrainConfig:
    t = cfg["train"]
    b = cfg["base"]
    kw = dict(hidden=t["hidden"], layers=b["layers"], heads=b["heads"],
              seq_len=b["seq_len"], batch_size=t["batch_size"],
              token_budget=t.get("token_budget", 1_000_000), lr=t["lr"],
              weight_decay=t["weight_decay"], warmup_frac=t["warmup_frac"],
              seed=t["seed"], device=t["device"])
    kw.update(over)
    return TrainConfig(**kw)


def _run_cell(db: DataBundle, cfg: dict, *, hidden, lr, batch, warmup, decay,
              accum, seed, token_budget, diverge_thresh=None, tag=""):
    """训练 + 评测一个单元格，返回统一记录。"""
    import torch
    from .train_hparam import build_modern, train_hparam, eval_hparam
    tconf = _base_cfg(cfg, hidden=hidden, lr=lr, batch_size=batch,
                      warmup_frac=warmup, seed=seed, token_budget=token_budget)
    model = build_modern(hidden, cfg["base"], db.vocab)
    res = train_hparam(db.train_ids, model, tconf, accum=accum, decay=decay,
                       diverge_thresh=diverge_thresh)
    ev = eval_hparam(model, db.tokenizer, db.nl_eval, db.code_eval, tconf)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    rec = {
        "tag": tag, "hidden": hidden, "lr": lr, "batch_size": batch,
        "warmup_frac": warmup, "decay": decay, "accum": accum, "seed": seed,
        "token_budget": token_budget,
        "n_params": res["n_params"],
        "non_emb_params": non_emb_params(hidden, cfg["base"]["layers"]),
        "final_train_loss": res["final_train_loss"],
        "diverged": res["diverged"],
        "train_sec": res["train_sec"],
        "tokens_per_sec": res["tokens_per_sec"],
        "peak_memory_mib": res["peak_memory_mib"],
        "nl_ppl": ev["nl"]["ppl"], "code_ppl": ev["code"]["ppl"],
        "nl_loss": ev["nl"]["loss"], "code_loss": ev["code"]["loss"],
    }
    print(f"  {tag:26s} h={hidden:4d} lr={lr:<8g} b={batch:<3d} acc={accum} "
          f"loss={rec['final_train_loss']} nl={rec['nl_ppl']} code={rec['code_ppl']} "
          f"div={rec['diverged']}", flush=True)
    return rec


# ---------------- plan 模式（离线估算）----------------

def mode_plan(cfg: dict) -> dict:
    b = cfg["base"]
    ref = cfg["throughput_ref"]
    vocab = b["vocab_expect"]
    layers = b["layers"]
    base_hidden = cfg["train"]["hidden"]

    def sec(budget, hidden):
        return train_seconds(budget, hidden, ref["hidden"], ref["tok_per_sec"])

    bd = {}
    m = cfg

    # noise
    if m["noise"]["enabled"]:
        n = m["noise"]
        bd["noise"] = {"runs": len(n["seeds"]),
                       "sec": round(sec(n["token_budget"], base_hidden) * len(n["seeds"]), 1)}
    # lr_range
    if m["lr_range_test"]["enabled"]:
        n = m["lr_range_test"]
        bd["lr_range_test"] = {"runs": 1, "sec": round(sec(n["token_budget"], base_hidden), 1)}
    # sweep
    if m["lr_batch_sweep"]["enabled"]:
        n = m["lr_batch_sweep"]
        cells = len(n["batch_sizes"]) * len(n["lr_grid"])
        bd["lr_batch_sweep"] = {"runs": cells,
                                "sec": round(sec(n["token_budget"], base_hidden) * cells, 1)}
    # warmup_decay
    if m["warmup_decay"]["enabled"]:
        n = m["warmup_decay"]
        cells = len(n["warmup_ratios"]) * len(n["decay_modes"])
        bd["warmup_decay"] = {"runs": cells,
                              "sec": round(sec(n["token_budget"], base_hidden) * cells, 1)}
    # grad_accum
    if m["grad_accum"]["enabled"]:
        n = m["grad_accum"]
        bd["grad_accum"] = {"runs": len(n["cases"]),
                            "sec": round(sec(n["token_budget"], base_hidden) * len(n["cases"]), 1)}
    # scaling
    if m["scaling_law"]["enabled"]:
        n = m["scaling_law"]
        detail = [{"hidden": h, "params": n_params_dense(h, layers, vocab),
                   "non_emb": non_emb_params(h, layers),
                   "sec": round(sec(n["token_budget"], h), 1)} for h in n["hidden_sizes"]]
        bd["scaling_law"] = {"runs": len(detail),
                             "sec": round(sum(d["sec"] for d in detail), 1), "detail": detail}
    # compute_budget
    if m["compute_budget"]["enabled"]:
        n = m["compute_budget"]
        C = float(n["flops_budget"])
        detail = []
        for h in n["hidden_sizes"]:
            ne = non_emb_params(h, layers)
            d_opt = C / (6.0 * ne)
            detail.append({"hidden": h, "non_emb": ne, "d_opt": int(d_opt),
                           "nd_ratio": round(d_opt / ne, 1),
                           "sec": round(sec(int(d_opt), h), 1)})
        bd["compute_budget"] = {"runs": len(detail),
                                "sec": round(sum(x["sec"] for x in detail), 1), "detail": detail}
    # mup
    if m["mup"]["enabled"]:
        n = m["mup"]
        cells = len(n["hidden_sizes"]) * len(n["lr_grid"])
        sec_sum = sum(sec(n["token_budget"], h) for h in n["hidden_sizes"] for _ in n["lr_grid"])
        bd["mup"] = {"runs": cells, "sec": round(sec_sum, 1)}
    # predict_larger
    if m["predict_larger"]["enabled"]:
        n = m["predict_larger"]
        bd["predict_larger"] = {"runs": 2, "sec": round(sec(n["token_budget"], n["hidden"]) * 2, 1)}

    total_sec = round(sum(v["sec"] for v in bd.values()), 1)
    total_runs = sum(v["runs"] for v in bd.values())
    return {
        "mode": "plan", "truth_label": "ESTIMATE",
        "base_hidden": base_hidden, "params_at_base": n_params_dense(base_hidden, layers, vocab),
        "breakdown": bd,
        "total": {"runs": total_runs, "sec": total_sec, "hours": round(total_sec / 3600, 2)},
        "note": "sec 为 ESTIMATE（19 篇实测 tok/s 按 hidden² 缩放），真实时长以运行日志为准",
    }


# ---------------- 真实运行模式 ----------------

def _meta(db: DataBundle, cfg: dict, started: float, extra: dict | None = None) -> dict:
    md = {
        "python": platform.python_version(),
        "elapsed_sec": round(time.perf_counter() - started, 2),
        "code_sha256": _code_sha256(),
        "fingerprint": db.fingerprint,
        "vocab_size": db.vocab,
        "recipe_text_sha256": db.recipe_text_sha256,
        "n_train_tokens": db.n_train_tokens,
        "train_config": cfg["train"],
    }
    if extra:
        md.update(extra)
    return md


def mode_noise(db: DataBundle, cfg: dict) -> dict:
    started = time.perf_counter()
    n = cfg["noise"]
    t = cfg["train"]
    runs = []
    for seed in n["seeds"]:
        runs.append(_run_cell(db, cfg, hidden=t["hidden"], lr=t["lr"],
                              batch=t["batch_size"], warmup=t["warmup_frac"],
                              decay=t["decay"], accum=1, seed=seed,
                              token_budget=n["token_budget"], tag=f"noise_s{seed}"))
    nls = [r["nl_ppl"] for r in runs if r["nl_ppl"] is not None]
    codes = [r["code_ppl"] for r in runs if r["code_ppl"] is not None]

    def rng(v):
        return round((max(v) - min(v)) / (sum(v) / len(v)) * 100, 2)

    summary = {
        "nl_ppl_min": min(nls), "nl_ppl_max": max(nls), "nl_ppl_range_pct": rng(nls),
        "code_ppl_min": min(codes), "code_ppl_max": max(codes), "code_ppl_range_pct": rng(codes),
    }
    return {"mode": "noise", "truth_label": "REAL", "runs": runs, "summary": summary,
            "metadata": _meta(db, cfg, started),
            "note": "同配置多 seed 的极差/均值，作为单变量差异的判读标尺（19 篇纪律）"}


def mode_lr_range(db: DataBundle, cfg: dict) -> dict:
    started = time.perf_counter()
    import torch
    from .train_hparam import build_modern, lr_range_test
    n = cfg["lr_range_test"]
    t = cfg["train"]
    tconf = _base_cfg(cfg, token_budget=n["token_budget"])
    model = build_modern(t["hidden"], cfg["base"], db.vocab)
    res = lr_range_test(db.train_ids, model, tconf, n["lr_min"], n["lr_max"])
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # 找拐点：滑窗平均 loss 的最低点，之后 loss 持续上升即发散区
    curve = res["curve"]
    valid = [c for c in curve if c["loss"] is not None]
    best = min(valid, key=lambda c: c["loss"]) if valid else None
    return {"mode": "lr_range", "truth_label": "REAL",
            "lr_min": n["lr_min"], "lr_max": n["lr_max"],
            "total_steps": res["total_steps"], "train_sec": res["train_sec"],
            "best_point": best, "curve": curve,
            "metadata": _meta(db, cfg, started),
            "note": "LR 指数上升单次训练；best_point 为 loss 最低处，其后可用区间上界"}


def mode_sweep(db: DataBundle, cfg: dict) -> dict:
    started = time.perf_counter()
    n = cfg["lr_batch_sweep"]
    t = cfg["train"]
    thresh = t.get("diverge_thresh")
    runs = []
    for batch in n["batch_sizes"]:
        for lr in n["lr_grid"]:
            runs.append(_run_cell(db, cfg, hidden=t["hidden"], lr=lr, batch=batch,
                                  warmup=t["warmup_frac"], decay=t["decay"], accum=1,
                                  seed=t["seed"], token_budget=n["token_budget"],
                                  diverge_thresh=thresh, tag=f"sweep_b{batch}_lr{lr:g}"))
    return {"mode": "sweep", "truth_label": "REAL", "runs": runs,
            "metadata": _meta(db, cfg, started),
            "note": "LR×batch 网格；每 batch 档的最优 LR 用于检验平方根/线性缩放"}


def mode_warmup_decay(db: DataBundle, cfg: dict) -> dict:
    started = time.perf_counter()
    n = cfg["warmup_decay"]
    t = cfg["train"]
    runs = []
    for wu in n["warmup_ratios"]:
        for decay in n["decay_modes"]:
            runs.append(_run_cell(db, cfg, hidden=t["hidden"], lr=t["lr"],
                                  batch=t["batch_size"], warmup=wu, decay=decay,
                                  accum=1, seed=t["seed"], token_budget=n["token_budget"],
                                  tag=f"wd_w{wu}_{decay}"))
    return {"mode": "warmup_decay", "truth_label": "REAL", "runs": runs,
            "metadata": _meta(db, cfg, started)}


def mode_grad_accum(db: DataBundle, cfg: dict) -> dict:
    started = time.perf_counter()
    n = cfg["grad_accum"]
    t = cfg["train"]
    runs = []
    for case in n["cases"]:
        runs.append(_run_cell(db, cfg, hidden=t["hidden"], lr=t["lr"],
                              batch=case["batch_size"], warmup=t["warmup_frac"],
                              decay=t["decay"], accum=case["accum"], seed=t["seed"],
                              token_budget=n["token_budget"],
                              tag=f"accum_{case['name']}"))
    return {"mode": "grad_accum", "truth_label": "REAL", "runs": runs,
            "metadata": _meta(db, cfg, started),
            "note": "同有效 batch（b×accum）的不同拆法对照，检验梯度累积等效边界"}


def mode_scaling(db: DataBundle, cfg: dict) -> dict:
    started = time.perf_counter()
    n = cfg["scaling_law"]
    t = cfg["train"]
    layers = cfg["base"]["layers"]
    runs = []
    for h in n["hidden_sizes"]:
        runs.append(_run_cell(db, cfg, hidden=h, lr=t["lr"], batch=t["batch_size"],
                              warmup=t["warmup_frac"], decay=t["decay"], accum=1,
                              seed=t["seed"], token_budget=n["token_budget"],
                              tag=f"scaling_h{h}"))
    # 拟合 L(N) = a·N^b（用非 embedding 参数与 nl_loss）
    pts = [(r["non_emb_params"], r["nl_loss"]) for r in runs if r["nl_loss"] is not None]
    fit = None
    extrapolate = None
    if len(pts) >= 2:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        a, bexp = fit_power_law(xs, ys)
        fit = {"a": a, "b": bexp}
        # 外推到 predict_larger 的 hidden
        h_big = cfg["predict_larger"]["hidden"]
        n_big = non_emb_params(h_big, layers)
        extrapolate = {"hidden": h_big, "non_emb_params": n_big,
                       "predicted_nl_loss": round(predict_loss(a, bexp, n_big), 4)}
    return {"mode": "scaling", "truth_label": "REAL", "runs": runs,
            "fit": fit, "extrapolate": extrapolate,
            "metadata": _meta(db, cfg, started),
            "note": "L(N)=a·N^b 对数最小二乘；extrapolate 为拟合区间外的预测（SCALED）"}


def mode_compute_budget(db: DataBundle, cfg: dict) -> dict:
    started = time.perf_counter()
    n = cfg["compute_budget"]
    t = cfg["train"]
    C = float(n["flops_budget"])
    layers = cfg["base"]["layers"]
    runs = []
    for h in n["hidden_sizes"]:
        ne = non_emb_params(h, layers)
        d_opt = int(C / (6.0 * ne))          # D = C/(6N)
        runs.append(_run_cell(db, cfg, hidden=h, lr=t["lr"], batch=t["batch_size"],
                              warmup=t["warmup_frac"], decay=t["decay"], accum=1,
                              seed=t["seed"], token_budget=d_opt,
                              tag=f"cb_h{h}"))
        runs[-1]["d_opt"] = d_opt
        runs[-1]["nd_ratio"] = round(d_opt / ne, 1)
        runs[-1]["flops_6nd"] = flops_6nd(ne, d_opt)
    # 实测最优点（nl_loss 最低）与 Chinchilla 推导点（D/N=20）的距离
    valid = [r for r in runs if r["nl_loss"] is not None]
    best = min(valid, key=lambda r: r["nl_loss"]) if valid else None
    return {"mode": "compute_budget", "truth_label": "REAL",
            "flops_budget": C, "runs": runs,
            "best_nd_ratio": best["nd_ratio"] if best else None,
            "chinchilla_nd_ratio": 20,
            "metadata": _meta(db, cfg, started),
            "note": "等算力 C≈6ND；best_nd_ratio 为实测最优 D/N，与 Chinchilla 20 对照"}


def mode_mup(db: DataBundle, cfg: dict) -> dict:
    started = time.perf_counter()
    n = cfg["mup"]
    t = cfg["train"]
    runs = []
    for h in n["hidden_sizes"]:
        for lr in n["lr_grid"]:
            runs.append(_run_cell(db, cfg, hidden=h, lr=lr, batch=t["batch_size"],
                                  warmup=t["warmup_frac"], decay=t["decay"], accum=1,
                                  seed=t["seed"], token_budget=n["token_budget"],
                                  tag=f"mup_h{h}_lr{lr:g}"))
    # 每档宽度的最优 LR
    best_per_width = {}
    for h in n["hidden_sizes"]:
        rs = [r for r in runs if r["hidden"] == h and r["nl_loss"] is not None]
        if rs:
            best_per_width[h] = min(rs, key=lambda r: r["nl_loss"])["lr"]
    return {"mode": "mup", "truth_label": "REAL", "runs": runs,
            "best_lr_per_width": best_per_width,
            "metadata": _meta(db, cfg, started),
            "note": "宽度迁移：对照最优 LR 随 width 的变化与 1/w、1/sqrt(w) 规则"}


def mode_predict_larger(db: DataBundle, cfg: dict, sweep_result: dict | None = None) -> dict:
    started = time.perf_counter()
    n = cfg["predict_larger"]
    t = cfg["train"]
    h_big = n["hidden"]
    # 从 sweep（384 档）取最优 LR；没有就用基准 LR
    base_hidden = t["hidden"]
    best_lr_384 = t["lr"]
    if sweep_result:
        rs = [r for r in sweep_result["runs"]
              if r["batch_size"] == t["batch_size"] and r["nl_loss"] is not None]
        if rs:
            best_lr_384 = min(rs, key=lambda r: r["nl_loss"])["lr"]
    # 1/width 迁移规则预测大档 LR
    lr_pred = best_lr_384 * base_hidden / h_big
    runs = [
        _run_cell(db, cfg, hidden=h_big, lr=lr_pred, batch=t["batch_size"],
                  warmup=t["warmup_frac"], decay=t["decay"], accum=1, seed=t["seed"],
                  token_budget=n["token_budget"], tag=f"pred_h{h_big}_scaled"),
        _run_cell(db, cfg, hidden=h_big, lr=best_lr_384, batch=t["batch_size"],
                  warmup=t["warmup_frac"], decay=t["decay"], accum=1, seed=t["seed"],
                  token_budget=n["token_budget"], tag=f"pred_h{h_big}_direct"),
    ]
    return {"mode": "predict_larger", "truth_label": "REAL",
            "best_lr_384": best_lr_384, "lr_predicted_1_over_width": round(lr_pred, 8),
            "runs": runs, "metadata": _meta(db, cfg, started),
            "note": "scaled=按 1/width 迁移预测的 LR；direct=直搬 384 档最优 LR"}


def mode_all(db: DataBundle, cfg: dict) -> dict:
    started = time.perf_counter()
    out = {"mode": "all", "truth_label": "REAL"}
    out["noise"] = mode_noise(db, cfg)
    out["lr_range"] = mode_lr_range(db, cfg)
    out["sweep"] = mode_sweep(db, cfg)
    out["warmup_decay"] = mode_warmup_decay(db, cfg)
    out["grad_accum"] = mode_grad_accum(db, cfg)
    out["scaling"] = mode_scaling(db, cfg)
    out["compute_budget"] = mode_compute_budget(db, cfg)
    out["mup"] = mode_mup(db, cfg)
    out["predict_larger"] = mode_predict_larger(db, cfg, out["sweep"])
    out["metadata"] = _meta(db, cfg, started)
    return out


MODES = {
    "plan": None,   # 不需要数据
    "noise": mode_noise, "lr_range": mode_lr_range, "sweep": mode_sweep,
    "warmup_decay": mode_warmup_decay, "grad_accum": mode_grad_accum,
    "scaling": mode_scaling, "compute_budget": mode_compute_budget,
    "mup": mode_mup, "predict_larger": mode_predict_larger, "all": mode_all,
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="20 篇超参 pipeline")
    ap.add_argument("--mode", default="plan", choices=list(MODES.keys()))
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(Path(args.config))
    if args.mode == "plan":
        result = mode_plan(cfg)
    else:
        db = DataBundle(cfg)
        fn = MODES[args.mode]
        result = fn(db, cfg) if args.mode != "predict_larger" else fn(db, cfg, None)

    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"written: {args.out}", flush=True)
    if args.mode == "plan":
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
