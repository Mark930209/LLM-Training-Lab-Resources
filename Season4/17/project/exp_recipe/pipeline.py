"""pipeline.py —— exp_recipe 统一入口。

模式：
  prepare : 制备全部配方（不训练），落盘配方文本与制备报告
  train   : 制备 + 训练 + 两域评测一个配方
  matrix  : 制备 + 训练 + 评测全部配方，汇总对照表（主实验）

所有数字来自真实运行；固定 seed 与 token 预算，跨配方可比。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import yaml

from . import corpus, recipe, train_eval


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _code_sha256() -> dict[str, str]:
    here = Path(__file__).parent
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (here / "pipeline.py", here / "corpus.py",
                  here / "recipe.py", here / "train_eval.py")
    }


def _load_domains(cfg: dict) -> dict:
    """加载并切分两域，返回 domains 与文档池（含重复注入）。"""
    ccfg = cfg["corpus"]
    nl_text = corpus.load_nl_corpus(ccfg["nl_path"])
    code_docs = corpus.load_code_docs()
    domains = corpus.split_domains(nl_text, code_docs)

    nl_docs = corpus.split_nl_train_docs(
        domains["nl_train"], int(ccfg.get("nl_chunk_chars", 4000)))
    code_texts = [t for _, t in domains["code_train"]]

    inject_stats = {"injected": False}
    if ccfg.get("inject_duplicates", True):
        nl_docs, nl_stats = corpus.inject_duplicates(nl_docs)
        code_texts, code_stats = corpus.inject_duplicates(code_texts)
        inject_stats = {"injected": True, "nl": nl_stats, "code": code_stats}

    domains["nl_docs"] = nl_docs
    domains["code_texts"] = code_texts
    domains["inject_stats"] = inject_stats
    domains["fingerprint"] = corpus.eval_fingerprint(domains)
    return domains


def _specs(cfg: dict) -> list[recipe.RecipeSpec]:
    return [recipe.RecipeSpec(**r) for r in cfg["recipes"]]


def mode_prepare(cfg: dict, out_dir: str) -> dict:
    started = time.perf_counter()
    domains = _load_domains(cfg)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    reports = {}
    for spec in _specs(cfg):
        res = recipe.prepare_recipe(spec, domains["nl_docs"],
                                    domains["code_texts"])
        (out / f"{spec.name}.txt").write_text(res["text"], encoding="utf-8")
        rep = res["report"]
        rep["diversity"] = recipe.diversity_metrics(res["text"])
        reports[spec.name] = rep
        recipe.save_report(out / f"{spec.name}_report.json", rep)

    payload = {
        "mode": "prepare",
        "fingerprint": domains["fingerprint"],
        "inject_stats": domains["inject_stats"],
        "recipes": reports,
        "metadata": {"python": platform.python_version(),
                     "elapsed_sec": round(time.perf_counter() - started, 2),
                     "code_sha256": _code_sha256()},
    }
    (out / "prepare_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return payload


def _train_one(cfg: dict, domains: dict, spec: recipe.RecipeSpec,
               tokenizer, work_dir: Path) -> dict:
    res = recipe.prepare_recipe(spec, domains["nl_docs"], domains["code_texts"])
    rep = res["report"]
    rep["diversity"] = recipe.diversity_metrics(res["text"])
    recipe.save_report(work_dir / f"{spec.name}_report.json", rep)

    tcfg = train_eval.TrainConfig(**cfg["train"])
    code_eval_text = "\n".join(t for _, t in domains["code_eval"])
    train_res = train_eval.train_recipe(
        res["text"], tokenizer, domains["nl_eval"], code_eval_text, tcfg)
    return {"recipe": spec.name, "prepare": rep, "train": train_res}


def mode_matrix(cfg: dict, out_dir: str) -> dict:
    started = time.perf_counter()
    domains = _load_domains(cfg)
    work_dir = Path(out_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # 固定 tokenizer：两域训练池并集（不含评测集、不含注入副本以外的内容）
    code_train_text = "\n".join(domains["code_texts"])
    tokenizer = train_eval.build_fixed_tokenizer(domains["nl_train"],
                                                 code_train_text)

    runs = []
    for spec in _specs(cfg):
        runs.append(_train_one(cfg, domains, spec, tokenizer, work_dir))

    elapsed = round(time.perf_counter() - started, 2)
    payload = {
        "mode": "matrix",
        "truth_label": "REAL",
        "fingerprint": domains["fingerprint"],
        "inject_stats": domains["inject_stats"],
        "tokenizer": {"vocab_size": tokenizer.vocab_size,
                      "note": "两域训练池并集建的固定词表，所有配方共用"},
        "train_config": cfg["train"],
        "runs": runs,
        "metadata": {"python": platform.python_version(),
                     "elapsed_sec": elapsed,
                     "code_sha256": _code_sha256()},
        "note": "固定 token 预算/seed/评测集；nl_eval 与 code_eval 困惑度"
                "越低越好，unique n-gram 比例是多样性代理（去重过激会拉低）。",
    }
    (work_dir / "matrix_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["prepare", "matrix"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.mode == "prepare":
        payload = mode_prepare(cfg, args.out_dir)
    else:
        payload = mode_matrix(cfg, args.out_dir)

    # 控制台只打摘要，完整结果已落盘
    if args.mode == "matrix":
        print(f"recipes={len(payload['runs'])} "
              f"vocab={payload['tokenizer']['vocab_size']} "
              f"elapsed={payload['metadata']['elapsed_sec']}s")
        for r in payload["runs"]:
            t = r["train"]
            print(f"  {r['recipe']:22s} nl_ppl={t['nl_eval']['ppl']} "
                  f"code_ppl={t['code_eval']['ppl']} "
                  f"u4={r['prepare']['diversity']['unique_4gram_ratio']}")
    else:
        print(f"recipes prepared={len(payload['recipes'])} "
              f"elapsed={payload['metadata']['elapsed_sec']}s")


if __name__ == "__main__":
    main()
