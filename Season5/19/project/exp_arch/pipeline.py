"""pipeline.py —— exp_arch 统一入口。

模式：
  plan  : 只算不训。对消融矩阵每格做等参数量反解（solve_hidden_for_params），
          输出每格的 hidden/参数量分解/KV cache/每步 FLOPs 与等算力缩放系数。
          先看计划再决定跑不跑，避免白烧 GPU 时间。
  train : 按 plan 的 hidden 逐格训练（固定 token 预算），报两域困惑度、
          峰值显存、吞吐；stress 段用偏大 LR 重跑指定格，记录是否发散。

语料、评测集切分、固定词表（6120）与训练配方全部复用 17 篇（同一把尺子）。
结果 JSON 一律 --out 直写 /mnt/d 挂载路径（18 篇教训：WSL /tmp 不可靠）。
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

from . import arch_metrics as am
from .model_arch import kv_heads_for
from .train_arch import build_model, eval_arch, train_arch


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _code_sha256() -> dict[str, str]:
    here = Path(__file__).parent
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (here / "pipeline.py", here / "model_arch.py",
                      here / "arch_metrics.py", here / "train_arch.py")}


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


def _resolve_spec(cell: dict, base: dict, vocab: int, target_params: int) -> dict:
    """把矩阵格解析成完整 spec：hidden 给定则用给定值，null 则等参数量反解。"""
    heads = base["heads"]
    kv = kv_heads_for(cell.get("attn_type", "mha"), heads)
    spec = dict(cell)
    if cell.get("hidden") is None:
        spec["hidden"] = am.solve_hidden_for_params(
            target_params, vocab, base["layers"], heads, kv, base["seq_len"],
            cell.get("norm_type", "rms"), cell.get("ffn_type", "swiglu"),
            cell.get("pos_enc", "rope"), cell.get("ffn_expansion"))
        spec["hidden_source"] = "equal_params"
    else:
        spec["hidden_source"] = "given"
    spec["kv_heads"] = kv
    spec["params"] = am.count_params(
        vocab, spec["hidden"], base["layers"], heads, kv, base["seq_len"],
        cell.get("norm_type", "rms"), cell.get("ffn_type", "swiglu"),
        cell.get("pos_enc", "rope"), cell.get("ffn_expansion"))
    spec["kv_cache_mib_b1_s256"] = am.kv_cache_mib(
        base["layers"], kv, spec["hidden"] // heads, base["seq_len"], batch=1)
    return spec


def mode_plan(cfg: dict) -> dict:
    """只算不训：每格的 hidden/参数分解/KV cache/FLOPs 与等算力缩放系数。"""
    base, tcfg = cfg["base"], cfg["train"]
    vocab = base["vocab_expect"]
    heads, layers, seq = base["heads"], base["layers"], base["seq_len"]
    batch, budget = tcfg["batch_size"], tcfg["token_budget"]

    # 基准格（modern，hidden 给定 384）的参数量 = 等参数量目标
    modern = next(c for c in cfg["matrix"] if c["name"] == "modern")
    target_params = am.count_params(
        vocab, modern["hidden"], layers, heads,
        kv_heads_for(modern.get("attn_type", "mha"), heads), seq,
        modern.get("norm_type", "rms"), modern.get("ffn_type", "swiglu"),
        modern.get("pos_enc", "rope"), modern.get("ffn_expansion"))["total"]

    cells = []
    for cell in cfg["matrix"]:
        spec = _resolve_spec(cell, base, vocab, target_params)
        flops = am.step_flops(spec["hidden"], layers, vocab, seq, batch,
                              heads, spec["kv_heads"], cell.get("ffn_type", "swiglu"),
                              cell.get("ffn_expansion"))
        cells.append({
            "name": cell["name"],
            "axes": {k: cell.get(k) for k in
                     ("norm_type", "norm_pos", "ffn_type", "pos_enc", "attn_type",
                      "ffn_expansion")},
            "hidden": spec["hidden"],
            "hidden_source": spec["hidden_source"],
            "params": spec["params"],
            "kv_cache_mib_b1_s256": spec["kv_cache_mib_b1_s256"],
            "step_flops": flops,
        })

    # 等算力缩放：以 modern 的 step_flops × budget 为总预算，
    # 每格 token_budget' = 总预算 / 本格 step_flops（取整到步）
    base_flops = next(c["step_flops"] for c in cells if c["name"] == "modern")
    total_flops_budget = base_flops * budget
    tokens_per_step = batch * seq
    for c in cells:
        steps = max(1, round(total_flops_budget / c["step_flops"] / tokens_per_step))
        c["equal_flops_steps"] = steps
        c["equal_flops_budget_tokens"] = steps * tokens_per_step
        c["equal_flops_ratio"] = round(c["step_flops"] / base_flops, 4)

    return {
        "mode": "plan",
        "truth_label": "REAL",       # 解析式计算，输入是本仓库配置，可复算
        "target_params": target_params,
        "cells": cells,
        "metadata": {
            "python": platform.python_version(),
            "code_sha256": _code_sha256(),
            "base": base, "train": tcfg,
        },
        "note": "等参数量：hidden 反解到 target_params 以下最大 heads 倍数；"
                "等算力：总 FLOPs = modern 格 step_flops × token_budget，"
                "每格按 step_flops 反比缩放步数。",
    }


def mode_train(cfg: dict) -> dict:
    """逐格训练。等参数量口径（hidden 用 plan 的反解值）+ stress 大 LR 对照。"""
    import torch

    started = time.perf_counter()
    base, tcfg = cfg["base"], cfg["train"]
    vocab = base["vocab_expect"]

    data = _load_corpus(cfg)

    # 固定词表：17 篇口径（两域训练池并集 char 级）
    from exp_scale.data import CharTokenizer
    tokenizer = CharTokenizer(data["nl_train"] + "\n" + data["code_train_text"])
    if tokenizer.vocab_size != vocab:
        raise RuntimeError(
            f"词表口径不符：期望 {vocab}（17 篇），实得 {tokenizer.vocab_size}")

    # 训练语料：17 篇基准配方制备一次，全格共用
    rs = cfg["recipe"]
    spec_r = recipe17.RecipeSpec(
        name=rs["name"], nl_ratio=float(rs["nl_ratio"]), dedup=rs["dedup"],
        quality=bool(rs["quality"]), char_budget=int(rs["char_budget"]))
    prep = recipe17.prepare_recipe(spec_r, data["nl_docs"], data["code_texts"])
    train_ids = torch.tensor(tokenizer.encode(prep["text"]), dtype=torch.long)

    tconf = TrainConfig(hidden=tcfg["hidden"], layers=base["layers"],
                        heads=base["heads"], seq_len=base["seq_len"],
                        batch_size=tcfg["batch_size"],
                        token_budget=tcfg["token_budget"], lr=tcfg["lr"],
                        weight_decay=tcfg["weight_decay"],
                        warmup_frac=tcfg["warmup_frac"], seed=tcfg["seed"],
                        device=tcfg["device"])

    plan = mode_plan(cfg)
    target_params = plan["target_params"]

    runs = []
    for cell in cfg["matrix"]:
        spec = _resolve_spec(cell, base, vocab, target_params)
        model = build_model(spec, tokenizer.vocab_size, base)
        res = train_arch(train_ids, model, tconf)
        ev = eval_arch(model, tokenizer, data["nl_eval"], data["code_eval_text"], tconf)
        runs.append({
            "name": cell["name"],
            "axes": {k: cell.get(k) for k in
                     ("norm_type", "norm_pos", "ffn_type", "pos_enc", "attn_type",
                      "ffn_expansion")},
            "hidden": spec["hidden"],
            "hidden_source": spec["hidden_source"],
            "params_expected": spec["params"]["total"],
            "kv_cache_mib_b1_s256": spec["kv_cache_mib_b1_s256"],
            **res, "nl_eval": ev["nl"], "code_eval": ev["code"],
        })
        print(f"  {cell['name']:20s} hidden={spec['hidden']:4d} "
              f"params={res['n_params']:9d} peak={res['peak_memory_mib']} MiB "
              f"nl_ppl={ev['nl']['ppl']} code_ppl={ev['code']['ppl']}", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # stress：偏大 LR 重跑指定格，记录是否发散（loss NaN/爆炸）
    stress_runs = []
    if cfg.get("stress", {}).get("enabled"):
        sconf = TrainConfig(**{**tconf.__dict__, "lr": cfg["stress"]["lr"]})
        names = set(cfg["stress"]["cells"])
        for cell in cfg["matrix"]:
            if cell["name"] not in names:
                continue
            spec = _resolve_spec(cell, base, vocab, target_params)
            model = build_model(spec, tokenizer.vocab_size, base)
            res = train_arch(train_ids, model, sconf)
            ev = eval_arch(model, tokenizer, data["nl_eval"],
                           data["code_eval_text"], sconf)
            diverged = (res["final_train_loss"] is None
                        or math.isnan(res["final_train_loss"])
                        or res["final_train_loss"] > 20.0
                        or ev["nl"]["ppl"] is None)
            stress_runs.append({
                "name": cell["name"], "lr": sconf.lr,
                "diverged": diverged,
                "final_train_loss": res["final_train_loss"],
                "nl_ppl": ev["nl"]["ppl"], "code_ppl": ev["code"]["ppl"],
                "train_sec": res["train_sec"],
            })
            print(f"  [stress lr={sconf.lr}] {cell['name']:20s} "
                  f"loss={res['final_train_loss']} diverged={diverged}", flush=True)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    elapsed = round(time.perf_counter() - started, 2)
    return {
        "mode": "train",
        "truth_label": "REAL",
        "fingerprint": data["fingerprint"],
        "vocab_size": tokenizer.vocab_size,
        "target_params": target_params,
        "recipe": {"spec": spec_r.__dict__, "text_sha256": prep["report"]["text_sha256"]},
        "runs": runs,
        "stress": stress_runs,
        "metadata": {
            "python": platform.python_version(),
            "elapsed_sec": elapsed,
            "code_sha256": _code_sha256(),
            "train_config": tcfg,
        },
        "note": "等参数量口径：每格 hidden 反解到 target_params 以下；"
                "词表/语料/评测集/配方与 17 篇同口径，指纹可跨篇核对。"
                "stress 段用 lr_stress 重跑指定格，diverged 判据：loss NaN/>20 或 ppl 为空。",
    }


def mode_noise(cfg: dict) -> dict:
    """噪声底测量：modern 格固定配置，只换 seed 重复训练。

    核心判断"多数单变量质量差异在噪声带内"的前提是噪声带本身被实测过。
    GPU 矩阵乘非确定 + seed 差异共同构成 run-to-run 波动；本模式量出
    nl/code 困惑度的均值与极差，作为 §4 判读单变量差异的标尺。
    """
    import torch

    started = time.perf_counter()
    base, tcfg = cfg["base"], cfg["train"]
    vocab = base["vocab_expect"]
    seeds = cfg.get("noise", {}).get("seeds", [20260925, 1001, 1002, 1003])

    data = _load_corpus(cfg)
    from exp_scale.data import CharTokenizer
    tokenizer = CharTokenizer(data["nl_train"] + "\n" + data["code_train_text"])
    if tokenizer.vocab_size != vocab:
        raise RuntimeError(
            f"词表口径不符：期望 {vocab}（17 篇），实得 {tokenizer.vocab_size}")

    rs = cfg["recipe"]
    spec_r = recipe17.RecipeSpec(
        name=rs["name"], nl_ratio=float(rs["nl_ratio"]), dedup=rs["dedup"],
        quality=bool(rs["quality"]), char_budget=int(rs["char_budget"]))
    prep = recipe17.prepare_recipe(spec_r, data["nl_docs"], data["code_texts"])
    train_ids = torch.tensor(tokenizer.encode(prep["text"]), dtype=torch.long)

    modern = next(c for c in cfg["matrix"] if c["name"] == "modern")
    target_params = am.count_params(
        vocab, modern["hidden"], base["layers"], base["heads"],
        kv_heads_for(modern.get("attn_type", "mha"), base["heads"]),
        base["seq_len"], modern.get("norm_type", "rms"),
        modern.get("ffn_type", "swiglu"), modern.get("pos_enc", "rope"),
        modern.get("ffn_expansion"))["total"]
    spec = _resolve_spec(modern, base, vocab, target_params)

    runs = []
    for seed in seeds:
        tconf = TrainConfig(hidden=tcfg["hidden"], layers=base["layers"],
                            heads=base["heads"], seq_len=base["seq_len"],
                            batch_size=tcfg["batch_size"],
                            token_budget=tcfg["token_budget"], lr=tcfg["lr"],
                            weight_decay=tcfg["weight_decay"],
                            warmup_frac=tcfg["warmup_frac"], seed=seed,
                            device=tcfg["device"])
        model = build_model(spec, tokenizer.vocab_size, base)
        res = train_arch(train_ids, model, tconf)
        ev = eval_arch(model, tokenizer, data["nl_eval"],
                       data["code_eval_text"], tconf)
        runs.append({
            "seed": seed, "n_params": res["n_params"],
            "train_sec": res["train_sec"],
            "nl_ppl": ev["nl"]["ppl"], "code_ppl": ev["code"]["ppl"],
            "final_train_loss": res["final_train_loss"],
        })
        print(f"  [noise seed={seed}] nl_ppl={ev['nl']['ppl']} "
              f"code_ppl={ev['code']['ppl']} loss={res['final_train_loss']}",
              flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    nl = [r["nl_ppl"] for r in runs]
    code = [r["code_ppl"] for r in runs]
    summary = {
        "n_runs": len(runs),
        "nl_ppl_min": min(nl), "nl_ppl_max": max(nl),
        "nl_ppl_range_pct": round((max(nl) - min(nl)) / min(nl) * 100, 2),
        "code_ppl_min": min(code), "code_ppl_max": max(code),
        "code_ppl_range_pct": round((max(code) - min(code)) / min(code) * 100, 2),
    }
    return {
        "mode": "noise",
        "truth_label": "REAL",
        "fingerprint": data["fingerprint"],
        "cell": "modern",
        "runs": runs,
        "summary": summary,
        "metadata": {
            "python": platform.python_version(),
            "elapsed_sec": round(time.perf_counter() - started, 2),
            "code_sha256": _code_sha256(),
            "seeds": seeds,
        },
        "note": "噪声底：modern 格固定配置只换 seed。单变量格的困惑度差异"
                "小于本噪声带（极差%）时不得声称架构效应。",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["plan", "train", "noise"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.mode == "plan":
        payload = mode_plan(cfg)
    elif args.mode == "train":
        payload = mode_train(cfg)
    else:
        payload = mode_noise(cfg)

    text = json.dumps(payload, ensure_ascii=False, indent=1)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        n = len(payload.get("cells", payload.get("runs", [])))
        print(f"{args.mode} done: {n} cells, -> {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
