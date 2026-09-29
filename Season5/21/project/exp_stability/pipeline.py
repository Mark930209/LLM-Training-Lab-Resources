"""pipeline.py —— exp_stability 统一入口（21 篇 Numerical Stability Lab）。

模式：
  plan      : 只算不训，离线估算各子实验格数与 GPU 时长（决策用）
  range     : 数值范围演示（纯函数 + 少量张量实测，秒级）
  precision : fp32/fp16/bf16 × clip on/off 训练对照（6 格）
  spike     : 坏 batch 注入 + 大 LR + 关 clip，复现 spike 并采集步级指标
  recovery  : 四种处置策略对照（skip_bad/lower_lr/rollback/tighten_clip）
  silent    : 静默退化（记忆化子集，train loss 降 / eval ppl 升）
  all       : 依次跑全部真实模式

架构固定 19 篇 modern；数据底座与 17/18/19/20 篇完全一致（指纹跨篇核对）。
结果 JSON 一律 --out 直写 /mnt/d（18 篇教训）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import yaml

from exp_recipe import corpus as corpus17
from exp_recipe import recipe as recipe17
from exp_recipe.train_eval import TrainConfig

from . import stability_metrics as sm
from .train_stability import build_modern, train_stability, eval_stability

HERE = Path(__file__).resolve().parent


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _code_sha256() -> dict[str, str]:
    here = Path(__file__).parent
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (here / "pipeline.py", here / "stability_metrics.py",
                      here / "train_stability.py", here / "config.yaml")}


# ---------------- 数据装载（与 20 篇同款 → 17 篇口径）----------------

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
    def __init__(self, cfg: dict):
        import torch
        from exp_scale.data import CharTokenizer
        self.cfg = cfg
        data = _load_corpus(cfg)
        self.fingerprint = data["fingerprint"]
        self.nl_eval = data["nl_eval"]
        self.code_eval = data["code_eval_text"]
        self.tokenizer = CharTokenizer(data["nl_train"] + "\n" + data["code_train_text"])
        if self.tokenizer.vocab_size != cfg["base"]["vocab_expect"]:
            raise RuntimeError(
                f"词表口径不符：期望 {cfg['base']['vocab_expect']}，实得 {self.tokenizer.vocab_size}")
        self.vocab = self.tokenizer.vocab_size
        rs = cfg["recipe"]
        spec_r = recipe17.RecipeSpec(
            name=rs["name"], nl_ratio=float(rs["nl_ratio"]), dedup=rs["dedup"],
            quality=bool(rs["quality"]), char_budget=int(rs["char_budget"]))
        prep = recipe17.prepare_recipe(spec_r, data["nl_docs"], data["code_texts"])
        self.recipe_text_sha256 = prep["report"]["text_sha256"]
        self.train_ids = torch.tensor(self.tokenizer.encode(prep["text"]), dtype=torch.long)
        self.n_train_tokens = len(self.train_ids)


def _base_cfg(cfg: dict, **over) -> TrainConfig:
    t, b = cfg["train"], cfg["base"]
    kw = dict(hidden=t["hidden"], layers=b["layers"], heads=b["heads"],
              seq_len=b["seq_len"], batch_size=t["batch_size"],
              token_budget=t["token_budget"], lr=t["lr"],
              weight_decay=t["weight_decay"], warmup_frac=t["warmup_frac"],
              seed=t["seed"], device=t["device"])
    kw.update(over)
    return TrainConfig(**kw)


def _meta(db: DataBundle, cfg: dict, started: float) -> dict:
    return {
        "python": platform.python_version(),
        "elapsed_sec": round(time.perf_counter() - started, 2),
        "code_sha256": _code_sha256(),
        "fingerprint": db.fingerprint,
        "vocab_size": db.vocab,
        "recipe_text_sha256": db.recipe_text_sha256,
        "n_train_tokens": db.n_train_tokens,
        "train_config": cfg["train"],
    }


# ---------------- plan 模式 ----------------

def mode_plan(cfg: dict) -> dict:
    t = cfg["train"]
    ref_tps = 54682.0        # 19 篇实测 h384 tok/s（fp32）
    def sec(budget, factor=1.0):
        return budget / ref_tps * factor

    bd = {}
    bd["range_demo"] = {"runs": 0, "sec": 1.0}
    p = cfg["precision"]
    n = len(p["dtypes"]) * len(p["clip"])
    bd["precision"] = {"runs": n, "sec": round(sec(t["token_budget"]) * n, 1)}
    s = cfg["spike"]
    n_sp = len(s["dtypes"]) + (1 if s.get("fp32_control") else 0)
    bd["spike"] = {"runs": n_sp, "sec": round(sec(s["token_budget"], 1.1) * n_sp, 1)}
    rp = cfg["lr_ramp"]
    bd["lr_ramp"] = {"runs": len(rp["dtypes"]),
                     "sec": round(sec(rp["token_budget"], 1.1) * len(rp["dtypes"]), 1)}
    r = cfg["recovery"]
    bd["recovery"] = {"runs": len(r["strategies"]),
                      "sec": round(sec(r["token_budget"], 1.15) * len(r["strategies"]), 1)}
    si = cfg["silent"]
    bd["silent"] = {"runs": 1, "sec": round(sec(si["token_budget"], 1.3), 1)}  # 含周期评测
    total = round(sum(v["sec"] for v in bd.values()), 1)
    runs = sum(v["runs"] for v in bd.values())
    return {
        "mode": "plan", "truth_label": "ESTIMATE",
        "breakdown": bd,
        "total": {"runs": runs, "sec": total, "hours": round(total / 3600, 2)},
        "note": "sec 为 ESTIMATE（19 篇实测 tok/s 基准，fp16/bf16 与注入/评测开销按系数放大）",
    }


# ---------------- range 模式（数值范围演示）----------------

def mode_range(cfg: dict) -> dict:
    import torch
    started = time.perf_counter()
    rows = []
    for dt in ("fp32", "fp16", "bf16", "fp8_e4m3", "fp8_e5m2"):
        rows.append({
            "dtype": dt,
            "min_normal": sm.min_normal(dt),
            "max_finite": sm.max_finite(dt),
            "mantissa_bits": sm.mantissa_bits(dt),
            "epsilon": sm.precision_epsilon(dt),
        })
    # 实测：fp16 上溢/下溢、bf16 精度损失（torch 张量验证，非纯解析）
    demos = {}
    if torch.cuda.is_available():
        v = torch.tensor([70000.0], device="cuda")
        demos["fp16_overflow_70000"] = v.half().item()          # → inf
        demos["bf16_holds_70000"] = v.bfloat16().item()          # → 70144（有损但有限）
        small = torch.tensor([1e-8], device="cuda")
        demos["fp16_underflow_1e-8"] = small.half().item()       # → 0
        demos["bf16_holds_1e-8"] = small.bfloat16().item()       # → ~1e-8
        # bf16 精度损失：1.0 + eps/2 应等于 1.0（尾数 8 位）
        one = torch.tensor([1.0], device="cuda")
        demos["bf16_1_plus_2^-9"] = (one + 2**-9).bfloat16().item()
        demos["fp16_1_plus_2^-12"] = (one + 2**-12).half().item()
    else:
        demos["note"] = "CPU 模式：仅解析范围，无张量实测"
    return {"mode": "range", "truth_label": "REAL",
            "ranges": rows, "tensor_demos": demos,
            "metadata": {"elapsed_sec": round(time.perf_counter() - started, 2),
                         "code_sha256": _code_sha256()},
            "note": "范围为 IEEE 754 解析值（REFERENCE）；tensor_demos 为本机实测（REAL）；"
                    "fp8 无 Ampere 硬件支持，只给解析范围"}


# ---------------- precision 模式 ----------------

def mode_precision(db: DataBundle, cfg: dict) -> dict:
    import torch
    started = time.perf_counter()
    p = cfg["precision"]
    t = cfg["train"]
    runs = []
    for dt in p["dtypes"]:
        for clip_on in p["clip"]:
            tconf = _base_cfg(cfg)
            model = build_modern(t["hidden"], cfg["base"], db.vocab)
            res = train_stability(db.train_ids, model, tconf, dtype=dt,
                                  clip=1.0 if clip_on else None, decay=t["decay"],
                                  log_every=0)
            ev = eval_stability(model, db.tokenizer, db.nl_eval, db.code_eval, tconf)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            rec = {"tag": f"{dt}_clip{'on' if clip_on else 'off'}", "dtype": dt,
                   "clip": clip_on, "n_params": None,
                   "final_train_loss": res["final_train_loss"], "diverged": res["diverged"],
                   "train_sec": res["train_sec"], "tokens_per_sec": res["tokens_per_sec"],
                   "peak_memory_mib": res["peak_memory_mib"],
                   "loss_scale_adjusts": res["loss_scale_adjusts"],
                   "nl_ppl": ev["nl"]["ppl"], "code_ppl": ev["code"]["ppl"]}
            runs.append(rec)
            print(f"  {rec['tag']:16s} loss={rec['final_train_loss']} nl={rec['nl_ppl']} "
                  f"code={rec['code_ppl']} tps={rec['tokens_per_sec']} "
                  f"peak={rec['peak_memory_mib']} scale_adj={rec['loss_scale_adjusts']} "
                  f"div={rec['diverged']}", flush=True)
    return {"mode": "precision", "truth_label": "REAL", "runs": runs,
            "metadata": _meta(db, cfg, started)}


# ---------------- spike 模式 ----------------

def mode_spike(db: DataBundle, cfg: dict) -> dict:
    import torch
    started = time.perf_counter()
    s = cfg["spike"]
    t = cfg["train"]
    mcfg = cfg["metrics"]
    acfg = cfg["alarm"]
    r_spike_k = cfg["recovery"]["spike_detect_k"]   # loss spike 判据（与 recovery 同口径）
    dtypes = list(s["dtypes"]) + (["fp32"] if s.get("fp32_control") else [])
    runs = []
    for dt in dtypes:
        tconf = _base_cfg(cfg, lr=t["lr"] * s["lr_mult"])
        model = build_modern(t["hidden"], cfg["base"], db.vocab)
        res = train_stability(db.train_ids, model, tconf, dtype=dt,
                              clip=s["clip"] if s["clip"] else None, decay=t["decay"],
                              inject_every=s["inject_every"], bad_factor=s["bad_factor"],
                              corrupt_frac=s.get("corrupt_frac", 0.0), vocab_size=db.vocab,
                              spike_k=r_spike_k, log_every=mcfg["log_every"])
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        metrics = res["metrics"]
        gnorms = [m["grad_norm"] for m in metrics]
        losses = [m["loss"] if m["loss"] is not None else float("nan") for m in metrics]
        # 基线窗口取首注入前的干净段 [warm, first_inject)：注入步的 grad_norm
        # （4.5+）混进基线会抬高阈值，且扫描起点 skip=warm+window=74 会跳过
        # 第一次注入（step 60），使 alarm=120/spike=60/lead=-60 成为假象——
        # 实际每次注入的 loss 与 grad_norm 是同步触发的。
        total_steps = max(1, s["token_budget"] // (t["batch_size"] * cfg["base"]["seq_len"]))
        warm = max(1, int(total_steps * t["warmup_frac"]))
        first_inject = s["inject_every"] if s.get("inject_every") else warm + acfg["window"]
        base_gn = gnorms[warm:min(first_inject, warm + acfg["window"])]
        thresh = sm.alarm_threshold(base_gn, acfg["grad_norm_k"])
        # 扫描从首次注入步开始（sustain=1：脏数据 spike 是孤立单步事件）
        a_step = sm.first_alarm_step(gnorms, thresh, skip=first_inject, sustain=1)
        sp_step = sm.first_spike_step(losses, r_spike_k, acfg["window"])
        lead = sm.lead_steps(a_step, sp_step)
        rec = {"tag": f"spike_{dt}", "dtype": dt, "lr": tconf.lr,
               "diverged": res["diverged"], "spike_count": res["spike_count"],
               "final_train_loss": res["final_train_loss"],
               "train_sec": res["train_sec"], "peak_memory_mib": res["peak_memory_mib"],
               "loss_scale_adjusts": res["loss_scale_adjusts"],
               "alarm": {"grad_norm_thresh": round(thresh, 4),
                         "first_alarm_step": a_step, "first_spike_step": sp_step,
                         "lead_steps": lead},
               "metrics": metrics}
        runs.append(rec)
        print(f"  spike_{dt:5s} spikes={res['spike_count']} div={res['diverged']} "
              f"alarm_step={a_step} spike_step={sp_step} lead={lead}", flush=True)
    return {"mode": "spike", "truth_label": "REAL", "runs": runs,
            "metadata": _meta(db, cfg, started),
            "note": "脏数据注入（污染目标 token）+ lr×5 + 关 clip；loss 与 grad_norm 同步飙升，"
                    "本场景验证 spike 的可复现与可定位；提前量由 lr_ramp 场景给出"}


def mode_lr_ramp(db: DataBundle, cfg: dict) -> dict:
    """渐进失稳场景：LR 从安全值线性攀升到失稳值（constant 衰减）。

    脏数据注入让 loss 与 grad_norm 同步飙升（lead≈0），验证不了核心判断
    "grad_norm 提前留下征兆"。渐进失稳才是提前量的正样本：LR 攀升过程中
    grad_norm 先逐步爬升（前兆），若干步后 loss 才爆炸（后果）。
    lr_to 取 20 篇 lr_range 实测的发散区（>1e-2）。
    """
    import torch
    started = time.perf_counter()
    rp = cfg["lr_ramp"]
    t = cfg["train"]
    mcfg = cfg["metrics"]
    acfg = cfg["alarm"]
    r_spike_k = cfg["recovery"]["spike_detect_k"]
    runs = []
    for dt in rp["dtypes"]:
        tconf = _base_cfg(cfg, lr=rp["lr_from"], token_budget=rp["token_budget"])
        model = build_modern(t["hidden"], cfg["base"], db.vocab)
        res = train_stability(db.train_ids, model, tconf, dtype=dt,
                              clip=rp["clip"] if rp["clip"] else None, decay="constant",
                              spike_k=r_spike_k, log_every=mcfg["log_every"],
                              lr_ramp={"from": rp["lr_from"], "to": rp["lr_to"],
                                       "start_frac": rp["start_frac"]})
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        metrics = res["metrics"]
        gnorms = [m["grad_norm"] for m in metrics]
        losses = [m["loss"] if m["loss"] is not None else float("nan") for m in metrics]
        total_steps = max(1, rp["token_budget"] // (t["batch_size"] * cfg["base"]["seq_len"]))
        warm = max(1, int(total_steps * t["warmup_frac"]))
        # 基线窗口取 ramp 前的稳定段 [ramp_start-8, ramp_start)：warmup 在 step 24 结束，
        # 但 [24,48) 仍有 warmup 尾部波动（gn 1.9~4.8），用它当基线会把阈值压得过低
        # （5.16），ramp 期正常波动（3~5）就误报（实测 step 58 孤立误报 → 虚高 lead=175）。
        # 稳定段 gn ~0.5~1.3，阈值 ~4~5，只有发散前的持续爬升（bf16 实测 9.6→24.7→139）才触发。
        ramp_start = int(total_steps * rp["start_frac"])
        base_lo, base_hi = max(warm, ramp_start - 8), ramp_start
        base_gn = gnorms[base_lo:base_hi]
        thresh = sm.alarm_threshold(base_gn, acfg["grad_norm_k"])
        # sustain=3：发散前兆是 grad_norm 持续爬升，要求连续 3 步超阈才算告警，
        # 过滤零星单步误报（fp16 突发溢出型发散前 gn 正常，alarm=None 是诚实结果）
        a_step = sm.first_alarm_step(gnorms, thresh, skip=base_hi, sustain=3)
        sp_step = sm.first_spike_step(losses, r_spike_k, acfg["window"])
        lead = sm.lead_steps(a_step, sp_step)
        rec = {"tag": f"ramp_{dt}", "dtype": dt,
               "lr_from": rp["lr_from"], "lr_to": rp["lr_to"],
               "diverged": res["diverged"], "spike_count": res["spike_count"],
               "final_train_loss": res["final_train_loss"],
               "train_sec": res["train_sec"], "peak_memory_mib": res["peak_memory_mib"],
               "loss_scale_adjusts": res["loss_scale_adjusts"],
               "alarm": {"grad_norm_thresh": round(thresh, 4),
                         "first_alarm_step": a_step, "first_spike_step": sp_step,
                         "lead_steps": lead},
               "metrics": metrics}
        runs.append(rec)
        print(f"  ramp_{dt:5s} spikes={res['spike_count']} div={res['diverged']} "
              f"alarm_step={a_step} spike_step={sp_step} lead={lead}", flush=True)
    return {"mode": "lr_ramp", "truth_label": "REAL", "runs": runs,
            "metadata": _meta(db, cfg, started),
            "note": "LR 渐进攀升（constant 衰减）：grad_norm 先爬升后 loss 爆炸，"
                    "提前量 = spike 步 − grad_norm 首超阈步"}


# ---------------- recovery 模式 ----------------

def mode_recovery(db: DataBundle, cfg: dict) -> dict:
    import torch
    started = time.perf_counter()
    r = cfg["recovery"]
    s = cfg["spike"]
    t = cfg["train"]
    runs = []
    for strat in r["strategies"]:
        tconf = _base_cfg(cfg, lr=t["lr"] * s["lr_mult"], token_budget=r["token_budget"])
        model = build_modern(t["hidden"], cfg["base"], db.vocab)
        res = train_stability(db.train_ids, model, tconf, dtype="fp16",
                              clip=r["clip"] if r.get("clip") else None,
                              decay=t["decay"],
                              inject_every=s["inject_every"], bad_factor=s["bad_factor"],
                              corrupt_frac=s.get("corrupt_frac", 0.0), vocab_size=db.vocab,
                              strategy=strat, spike_k=r["spike_detect_k"],
                              ckpt_every=r["ckpt_every"], log_every=1)
        ev = eval_stability(model, db.tokenizer, db.nl_eval, db.code_eval, tconf)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        rec = {"tag": f"recovery_{strat}", "strategy": strat,
               "diverged": res["diverged"], "spike_count": res["spike_count"],
               "skipped_batches": res["skipped_batches"], "rollbacks": res["rollbacks"],
               "final_train_loss": res["final_train_loss"],
               "nl_ppl": ev["nl"]["ppl"], "code_ppl": ev["code"]["ppl"],
               "train_sec": res["train_sec"], "peak_memory_mib": res["peak_memory_mib"]}
        runs.append(rec)
        print(f"  recovery_{strat:20s} spikes={res['spike_count']} div={res['diverged']} "
              f"skip={res['skipped_batches']} rb={res['rollbacks']} "
              f"nl={rec['nl_ppl']} loss={rec['final_train_loss']}", flush=True)
    return {"mode": "recovery", "truth_label": "REAL", "runs": runs,
            "metadata": _meta(db, cfg, started)}


# ---------------- silent 模式 ----------------

def mode_silent(db: DataBundle, cfg: dict) -> dict:
    import torch
    started = time.perf_counter()
    si = cfg["silent"]
    t = cfg["train"]
    tconf = _base_cfg(cfg, token_budget=si["token_budget"])
    total_steps = max(1, si["token_budget"] // (t["batch_size"] * cfg["base"]["seq_len"]))
    mem_from = int(total_steps * si["memorize_start_frac"])

    model = build_modern(t["hidden"], cfg["base"], db.vocab)
    nl_ids = torch.tensor(db.tokenizer.encode(db.nl_eval), dtype=torch.long)
    from exp_recipe.train_eval import evaluate_ppl

    def eval_fn(step, mdl):
        r = evaluate_ppl(mdl, nl_ids, tconf)
        return r["ppl"]

    res = train_stability(db.train_ids, model, tconf, dtype="fp32", clip=1.0,
                          decay=t["decay"], memorize_from=mem_from,
                          subset_frac=si["subset_frac"],
                          eval_fn=eval_fn, eval_every=si["eval_every"], log_every=10)
    ev = eval_stability(model, db.tokenizer, db.nl_eval, db.code_eval, tconf)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    curve = res["eval_curve"]
    train_losses = [c["train_loss"] for c in curve]
    eval_ppls = [c["eval_ppl"] for c in curve if c["eval_ppl"] is not None]
    # 退化判定用曲线上的相对位置（切换点在曲线 40% 处）
    degraded = sm.is_silent_degradation(train_losses, eval_ppls, si["memorize_start_frac"])
    gap = sm.degradation_gap_pct(eval_ppls, si["memorize_start_frac"])
    return {"mode": "silent", "truth_label": "REAL",
            "memorize_from_step": mem_from, "total_steps": total_steps,
            "subset_frac": si["subset_frac"],
            "final_train_loss": res["final_train_loss"],
            "final_nl_ppl": ev["nl"]["ppl"], "final_code_ppl": ev["code"]["ppl"],
            "silent_degradation": degraded, "degradation_gap_pct": round(gap, 2),
            "eval_curve": curve, "train_sec": res["train_sec"],
            "metadata": _meta(db, cfg, started),
            "note": "记忆化子集：40% 步后只用 10% 数据；train loss 降而 eval ppl 升 = 静默退化"}


def mode_all(db: DataBundle, cfg: dict) -> dict:
    started = time.perf_counter()
    out = {"mode": "all", "truth_label": "REAL"}
    out["range"] = mode_range(cfg)
    out["precision"] = mode_precision(db, cfg)
    out["spike"] = mode_spike(db, cfg)
    out["lr_ramp"] = mode_lr_ramp(db, cfg)
    out["recovery"] = mode_recovery(db, cfg)
    out["silent"] = mode_silent(db, cfg)
    out["metadata"] = _meta(db, cfg, started)
    return out


MODES = {
    "plan": None, "range": None,
    "precision": mode_precision, "spike": mode_spike, "lr_ramp": mode_lr_ramp,
    "recovery": mode_recovery, "silent": mode_silent, "all": mode_all,
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="21 篇数值稳定性 pipeline")
    ap.add_argument("--mode", default="plan", choices=list(MODES.keys()))
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(Path(args.config))
    if args.mode == "plan":
        result = mode_plan(cfg)
    elif args.mode == "range":
        result = mode_range(cfg)
    else:
        db = DataBundle(cfg)
        result = MODES[args.mode](db, cfg)

    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"written: {args.out}", flush=True)
    if args.mode == "plan":
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
