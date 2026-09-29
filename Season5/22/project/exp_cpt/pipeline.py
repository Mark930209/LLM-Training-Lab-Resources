"""pipeline.py —— 22 篇 Continual Pretraining Lab 全链路调度。

模式：
  plan      矩阵与耗时估计（不训练）
  baseline  底座 0 步的通用/领域起点（含 probe/参数快照的参照采集）
  core      核心矩阵：LR 峰值 × replay 比例（3×4=12 格，统一带 re-warmup）
  rewarm    re-warmup 必要性对照（rewarm=false 一格）
  mixmode   replay 混合方式对照（token 级交错 vs 文档级块拼接，一格）
  vocab     扩词表三种 embedding 初始化对照（3 格）
  failure   失败案例：过大 LR 擦除底座知识 + 两种恢复（3 格）
  all       以上全部（不含 plan）

每格独立重载底座（干净起点，杜绝跨格污染），seed 固定可复现。
结果 JSON 用 --out 直写 /mnt/d。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import cpt_data
import train_cpt
from train_cpt import CPTConfig, load_model


def load_config(path: str | Path) -> dict:
    import yaml
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _resolve(cfg_path: Path, rel: str) -> str:
    return str((cfg_path.parent / rel).resolve())


def _code_sha256() -> dict:
    import hashlib
    out = {}
    for f in ("cpt_metrics.py", "cpt_data.py", "train_cpt.py", "pipeline.py"):
        p = _HERE / f
        out[f] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    return out


def _base_cfg(tcfg: dict, sched: dict, tag: str, **over) -> CPTConfig:
    c = CPTConfig(
        rewarm_frac=sched["rewarm_frac"], floor_frac=sched["decay_floor_frac"],
        weight_decay=tcfg["weight_decay"], grad_clip=tcfg["grad_clip"],
        batch_size=tcfg["batch_size"], seq_len=tcfg["seq_len"],
        token_budget=tcfg["token_budget"], eval_every=tcfg["eval_every"],
        eval_max_tokens=tcfg["eval_max_tokens"], seed=tcfg["seed"],
        diverge_thresh=tcfg["diverge_thresh"], device=tcfg["device"], tag=tag)
    for k, v in over.items():
        setattr(c, k, v)
    return c


class DataBundle:
    """数据侧一次构建，多格复用（评测集/流源文档/probe 文本）。"""

    def __init__(self, cfg: dict, cfg_path: Path):
        self.nl_path = _resolve(cfg_path, cfg["corpus"]["nl_path"])
        self.domains = cpt_data.load_domains(self.nl_path)
        self.src_docs = cpt_data.stream_source_docs(self.domains)
        # base tokenizer 下的评测文本与 probe（probe 固定用 base tokenizer 的
        # token 序列，扩词表格同样喂这份，保证表示相似度逐格可比）
        from transformers import AutoTokenizer
        self.model_dir = _resolve(cfg_path, cfg["base_model"]["path"])
        self.tok = AutoTokenizer.from_pretrained(self.model_dir)
        self.eval_texts = cpt_data.eval_texts(self.domains)
        self.probe_ids = (self.tok.encode(self.eval_texts["code"][:256],
                                          add_special_tokens=False)
                          + self.tok.encode(self.eval_texts["nl"][:256],
                                            add_special_tokens=False))
        self.report = cpt_data.data_report(self.domains, self.src_docs,
                                           self.eval_texts, self.tok)


def _strip_checkpoints(result: dict) -> dict:
    return {k: v for k, v in result.items() if k != "checkpoints"}


def _run_cell(db: DataBundle, cfg: dict, sched: dict, tag: str,
              lr: float, replay: float, rewarm: bool = True,
              mix_mode: str = "docs", expand: dict | None = None,
              ckpt_every: int = 0, init_state: dict | None = None,
              budget: int | None = None,
              keep_final_state: bool = False):
    """一格实验：干净重载底座 →（可选）扩词表/恢复起点 → 训练。

    返回 (result, checkpoints, final_state)；checkpoints/final_state 是 CPU
    参数字典，不进 JSON，失败实验的恢复格用。
    """
    model, tok = load_model(db.model_dir, cfg["train"]["device"])
    rng = np.random.default_rng(cfg["train"]["seed"])
    if expand is not None:
        info = train_cpt.apply_vocab_expansion(
            model, tok, expand["new_tokens"], expand["subword_ids"],
            expand["method"], rng)
    else:
        info = None
    if init_state is not None:
        model.load_state_dict(init_state)

    stream_docs = db.src_docs
    tc = _base_cfg(cfg["train"], sched, tag,
                   lr_peak=lr, replay_ratio=replay, rewarm=rewarm,
                   checkpoint_every=ckpt_every,
                   token_budget=budget or cfg["train"]["token_budget"])
    if mix_mode == "docs":
        # 文档级混合（主方案，同真实 CPT 的 replay 混合方式）
        stream = cpt_data.build_doc_stream(stream_docs["code"], stream_docs["nl"],
                                           tok, tc.token_budget, replay)
    elif mix_mode == "blocks":
        # 极端块拼接：领域流在前、replay 整段在后（混合方式对照）
        n_rep = int(round(tc.token_budget * replay))
        n_dom = tc.token_budget - n_rep
        dom_ids = cpt_data.tokenize("\n".join(stream_docs["code"]), tok)[:n_dom]
        rep_ids = cpt_data.tokenize("\n".join(stream_docs["nl"]), tok)[:n_rep]
        stream = dom_ids + rep_ids
    else:
        # token 级交错（混合方式对照；首跑教训：非自然上下文放大遗忘）
        stream = cpt_data.build_stream(stream_docs["code"], stream_docs["nl"],
                                       tok, tc.token_budget, replay)

    base_probe = db.base_probe if hasattr(db, "base_probe") else None
    base_params = db.base_params if hasattr(db, "base_params") else None
    res = train_cpt.train_cpt(model, tok, stream, tc, db.eval_texts,
                              db.probe_ids, base_probe, base_params)
    res["vocab_expansion"] = info
    res["mix_mode"] = mix_mode
    ckpts = res.pop("checkpoints", None)
    final_state = train_cpt.snapshot_params(model) if keep_final_state else None
    del model
    import torch
    torch.cuda.empty_cache()
    return _strip_checkpoints(res), ckpts, final_state


def mode_plan(cfg: dict) -> dict:
    t = cfg["train"]
    row_len = t["seq_len"] + 1
    steps = t["token_budget"] // (t["batch_size"] * row_len)
    tps = 1340.0   # 探针实测（bs=2 seq=512）
    cells = {"core": 12, "rewarm": 1, "mixmode": 2, "vocab": 3, "failure": 4}
    per_cell = t["token_budget"] / tps + (steps // t["eval_every"]) * 3 + 25
    return {"steps_per_cell": steps, "seconds_per_cell_est": round(per_cell, 1),
            "cells": cells, "total_cells": sum(cells.values()),
            "total_minutes_est": round(sum(cells.values()) * per_cell / 60, 1)}


def mode_baseline(db: DataBundle, cfg: dict) -> dict:
    """0 步基线：底座通用/领域起点 + probe/参数快照参照。"""
    import torch
    model, tok = load_model(db.model_dir, cfg["train"]["device"])
    seq = cfg["train"]["seq_len"]
    out = {"nl": train_cpt.evaluate_text(model, tok, db.eval_texts["nl"], seq),
           "code": train_cpt.evaluate_text(model, tok, db.eval_texts["code"], seq)}
    db.base_probe = train_cpt.capture_probe(model, db.probe_ids)
    db.base_params = train_cpt.snapshot_params(model)
    n_params = sum(p.numel() for p in model.parameters())
    del model
    torch.cuda.empty_cache()
    return {"eval": out, "n_params": n_params,
            "probe_tokens": len(db.probe_ids)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["plan", "baseline", "core", "rewarm", "mixmode",
                             "vocab", "failure", "all"])
    ap.add_argument("--config", default=str(_HERE / "config.yaml"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    started = time.time()

    if args.mode == "plan":
        result = {"mode": "plan", "plan": mode_plan(cfg)}
        print(json.dumps(result["plan"], ensure_ascii=False, indent=2))
        return 0

    db = DataBundle(cfg, cfg_path)
    result = {"mode": args.mode,
              "meta": {"data_report": db.report, "code_sha256": _code_sha256(),
                       "base_model": cfg["base_model"], "started": started}}

    def dump():
        result["wall_s"] = round(time.time() - started, 1)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8")

    # baseline 先行：所有格都要它的 probe/参数参照
    result["baseline"] = mode_baseline(db, cfg)
    dump()
    print(f"[baseline] {result['baseline']['eval']}", flush=True)

    cells: list[dict] = []
    ckpt_bank: dict[str, dict] = {}

    def run(tag, **kw):
        t0 = time.time()
        res, ckpts, _final = _run_cell(db, cfg, cfg["schedule"], tag, **kw)
        cells.append(res)
        if ckpts:
            ckpt_bank[tag] = ckpts
        result["cells"] = cells
        dump()
        fe = res["final_eval"]
        print(f"[{tag}] code_ppl={fe['code']['ppl']:.2f} "
              f"nl_ppl={fe['nl']['ppl']:.2f} drift={res.get('param_drift', {}).get('mean_change')} "
              f"({time.time() - t0:.0f}s)", flush=True)

    modes = {args.mode} if args.mode != "all" else \
        {"core", "rewarm", "mixmode", "vocab", "failure"}

    if "core" in modes:
        for lr in cfg["core"]["lr_peaks"]:
            for rep in cfg["core"]["replay_ratios"]:
                run(f"core_lr{lr:g}_rep{rep:g}", lr=lr, replay=rep,
                    rewarm=cfg["core"]["rewarm"])
    if "rewarm" in modes:
        p = cfg["rewarm_probe"]
        run(f"rewarm_off_lr{p['lr']:g}", lr=p["lr"], replay=p["replay_ratio"],
            rewarm=False)
    if "mixmode" in modes:
        # 混合方式对照：token 级交错（run1 的全矩阵即此方式）vs 文档级（默认）
        p = cfg["rewarm_probe"]
        run(f"mixmode_interleave_rep{p['replay_ratio']:g}", lr=p["lr"],
            replay=p["replay_ratio"], mix_mode="interleave")
        run(f"mixmode_blocks_rep{p['replay_ratio']:g}", lr=p["lr"],
            replay=p["replay_ratio"], mix_mode="blocks")
    if "vocab" in modes:
        p = cfg["vocab_probe"]
        toks = cpt_data.extract_domain_tokens(
            [t for _, t in db.domains["code_train"]], db.tok,
            p["n_new_tokens"])
        result["vocab_stats"] = toks["stats"]
        for m in p["init_methods"]:
            run(f"vocab_{m}", lr=p["lr"], replay=p["replay_ratio"],
                expand={"new_tokens": toks["new_tokens"],
                        "subword_ids": toks["subword_ids"], "method": m})
    if "failure" in modes:
        f = cfg["failure"]
        res, ckpts, final_state = _run_cell(
            db, cfg, cfg["schedule"],
            f"fail_lr{f['lr']:g}_rep{f['replay_ratio']:g}",
            lr=f["lr"], replay=f["replay_ratio"],
            ckpt_every=f["checkpoint_every"], keep_final_state=True)
        cells.append(res)
        result["cells"] = cells
        dump()
        print(f"[fail] code_ppl={res['final_eval']['code']['ppl']:.2f} "
              f"nl_ppl={res['final_eval']['nl']['ppl']:.2f}", flush=True)
        # 恢复 A：降 LR 续训（从损坏终点继续，同预算）
        run("recover_low_lr", lr=f["recover_lr"], replay=f["replay_ratio"],
            init_state=final_state)
        # 恢复 B/C：回退 checkpoint 后低 LR 重训同预算。
        # B=最近的 ckpt（真实 ops 反应），C=最早的 ckpt（给恢复最有利条件）。
        # 冒烟教训：lr=1e-3 炸点附近的 ckpt 本身带毒（step4 ckpt 重训 step1
        # 即 loss 44 再崩），所以两个回退点都要看。
        if ckpts:
            for name, step_no in (("recover_rollback_last", max(ckpts)),
                                  ("recover_rollback_first", min(ckpts))):
                run(name, lr=f["recover_lr"], replay=f["replay_ratio"],
                    init_state=ckpts[step_no])
                print(f"[{name}] from ckpt step {step_no}", flush=True)
        else:
            print("[fail] no checkpoint available for rollback", flush=True)

    result["cells"] = cells
    dump()
    print(f"done: {len(cells)} cells -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
