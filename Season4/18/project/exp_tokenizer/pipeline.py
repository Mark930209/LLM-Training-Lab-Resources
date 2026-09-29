"""pipeline.py —— exp_tokenizer 统一入口。

模式：
  sweep : 训练 BPE/Unigram × 词表档位 × 数字策略，算静态指标（压缩率、
          覆盖率、长尾分布、embedding 参数占比），产出词表 sweep 对照表
  train : 固定 token 预算训练小模型，对照词表档位的显存/吞吐/困惑度。
          两条控制口径：
            A 固定架构（hidden 384/6 层/6 头，与 17 篇同档）——词表越大
              总参数越多，embedding 占比越高，这是主軸；
            B 固定总参数（解析解，不训练）——把总参数锁在 17 篇 10M 档
              12,971,904，反解每档词表可用的 hidden，看注意力容量怎样
              被词表挤掉。
          跨 tokenizer 的困惑度不可直接比（切分粒度不同），统一换算成
          BPC（bits per char）：loss × tokens / chars / ln2。

语料与评测集切分复用 17 篇 exp_recipe.corpus（同一把尺子）。
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

# 17 篇的语料切分与训练循环（PYTHONPATH 需含 17/18 两个 project 目录）
from exp_recipe import corpus as corpus17
from exp_recipe import recipe as recipe17
from exp_recipe import train_eval as train_eval17

from . import tok_metrics, tok_train


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _code_sha256() -> dict[str, str]:
    here = Path(__file__).parent
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (here / "pipeline.py", here / "tok_train.py",
                  here / "tok_metrics.py")
    }


def _load_corpus(cfg: dict) -> dict:
    """加载两域语料并切分（复用 17 篇口径），返回 train 池与 eval 集。"""
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


def mode_sweep(cfg: dict) -> dict:
    started = time.perf_counter()
    scfg = cfg["sweep"]
    work_dir = Path(scfg["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)

    data = _load_corpus(cfg)
    corpus_file = tok_train.write_corpus_file(
        data["nl_train"], data["code_texts"], work_dir / "train_corpus.txt")

    sample_chars = int(scfg.get("metric_sample_chars", 200000))
    nl_sample = data["nl_eval"][:sample_chars]
    code_sample = data["code_eval_text"][:sample_chars]

    runs = []
    for algo in scfg["algos"]:
        for vocab in scfg["vocab_sizes"]:
            for digit in scfg["digit_strategies"]:
                train_fn = (tok_train.train_bpe if algo == "bpe"
                            else tok_train.train_unigram)
                info = train_fn(corpus_file, vocab, digit, work_dir)
                tok = tok_train.load_tokenizer(info)

                nl_comp = tok_metrics.compression_metrics(tok, info, nl_sample)
                code_comp = tok_metrics.compression_metrics(tok, info, code_sample)
                nl_cov = tok_metrics.coverage_metrics(tok, info, nl_sample)
                # 长尾分布用训练池的抽样（全量太慢），固定抽 300 篇文档
                freq_texts = (data["nl_docs"][:200] + data["code_texts"][:100])
                freq = tok_metrics.token_frequency_distribution(
                    tok, info, freq_texts,
                    low_freq_threshold=int(scfg.get("low_freq_threshold", 5)))
                emb = tok_metrics.embedding_param_share(
                    info["actual_vocab"], 384, 6, 6)

                runs.append({
                    "algo": algo, "vocab_target": vocab,
                    "vocab_actual": info["actual_vocab"],
                    "digit_strategy": digit,
                    "train_sec": info["train_sec"],
                    "nl": nl_comp, "code": code_comp,
                    "nl_coverage": nl_cov["roundtrip_similarity"],
                    "long_tail": freq,
                    "embedding_share": emb["embedding_share"],
                    "embedding_params": emb["embedding_params"],
                    "total_params_10m": emb["total_params"],
                })
                print(f"  {algo:8s} vocab={vocab:6d} digit={digit:6s} "
                      f"actual={info['actual_vocab']:6d} "
                      f"nl_bpt={nl_comp['bytes_per_token']} "
                      f"low_freq={freq['low_freq_ratio']} "
                      f"emb_share={emb['embedding_share']}", flush=True)

    elapsed = round(time.perf_counter() - started, 2)
    return {
        "mode": "sweep",
        "truth_label": "REAL",
        "fingerprint": data["fingerprint"],
        "runs": runs,
        "metadata": {
            "python": platform.python_version(),
            "elapsed_sec": elapsed,
            "code_sha256": _code_sha256(),
            "config": scfg,
        },
        "note": "词表 sweep REAL：bytes_per_token 越小压缩越好；"
                "low_freq_ratio 随词表增大而升高（长尾 token 占比）；"
                "embedding_share 是 embedding 参数占 10M 档总参数的比例。",
    }


class _TokenizerAdapter:
    """把 BPE/Unigram tokenizer 适配成 17 篇 CharTokenizer 的接口。

    train_eval.train_recipe 只用到两个成员：encode(text) -> list[int]
    与 vocab_size。适配后训练循环、学习率调度、评测口径与 17 篇完全一致。
    """

    def __init__(self, tokenizer, info: dict):
        self._tok = tokenizer
        self._info = info
        self.vocab_size = info["actual_vocab"]

    def encode(self, text: str) -> list[int]:
        return tok_train.encode_text(self._tok, self._info, text)


def _bpc(loss: float | None, n_tokens: int, n_chars: int) -> float | None:
    """把困惑度换算成 bits per char：跨 tokenizer 可比的唯一口径。

    困惑度是"每 token 的意外程度"，token 粒度随词表变化，不能横比；
    换算到每字符比特数后，比的是"压缩这段文本用了多少信息量"。
    """
    if loss is None or not n_chars:
        return None
    return round(loss * n_tokens / n_chars / math.log(2), 4)


def _non_emb_params(hidden: int, layers: int) -> int:
    """SuperMiniGPT 非 embedding 参数量（解析式，与 04 篇模型结构对齐）。

    每个 block：attn 4H²（q/k/v/o，无偏置）+ SwiGLU 8H²（gate/up/down，
    expansion 8/3 使中间维 = 8H/3，三矩阵共 3×H×8H/3 = 8H²）+ 两个
    RMSNorm 2H = 12H² + 2H；共 L 层；再加最终 RMSNorm H。
    lm_head 与 embedding 权重共享（tie），不另计。
    校验：vocab 6120 / H 384 / L 6 → 6120×384 + 本式 = 12,971,904，
    与 17 篇实测参数量一字不差。
    """
    return layers * (12 * hidden * hidden + 2 * hidden) + hidden


def _fixed_budget_hidden(vocab: int, target_total: int, layers: int) -> dict:
    """固定总参数预算下反解 hidden（解析解，不训练）。

    12·H²·L + H·(2L + 1 + V) = target 的正根取整。词表越大，同样的总
    参数预算里能留给注意力/FFN 的 hidden 越小——词表挤占模型容量。
    """
    a = 12.0 * layers
    b = float(2 * layers + 1 + vocab)
    c = -float(target_total)
    h = (-b + math.sqrt(b * b - 4 * a * c)) / (2 * a)
    h = int(h)
    total = vocab * h + _non_emb_params(h, layers)
    return {"hidden": h, "total_params": total}


def _load_or_train_tokenizer(algo: str, vocab: int, digit: str,
                             work_dir: Path, corpus_file: Path) -> tuple:
    """优先复用 sweep 已训好的词表产物，缺失才重训（省时间且口径一致）。"""
    if algo == "bpe":
        path = work_dir / f"bpe_{vocab // 1000}k_{digit}.json"
        if path.exists():
            info = {"algo": algo, "path": str(path), "vocab_size": vocab,
                    "digit_strategy": digit, "actual_vocab": None,
                    "train_sec": 0.0, "reused": True}
            tok = tok_train.load_tokenizer(info)
            info["actual_vocab"] = tok.get_vocab_size()
            return tok, info
        info = tok_train.train_bpe(corpus_file, vocab, digit, work_dir)
    else:
        path = work_dir / f"unigram_{vocab // 1000}k_{digit}.model"
        if path.exists():
            info = {"algo": algo, "path": str(path), "vocab_size": vocab,
                    "digit_strategy": digit, "actual_vocab": None,
                    "train_sec": 0.0, "reused": True}
            tok = tok_train.load_tokenizer(info)
            info["actual_vocab"] = tok.get_piece_size()
            return tok, info
        info = tok_train.train_unigram(corpus_file, vocab, digit, work_dir)
    info["reused"] = False
    return tok_train.load_tokenizer(info), info


def mode_train(cfg: dict) -> dict:
    started = time.perf_counter()
    tcfg_dict = cfg["train"]
    from exp_recipe.train_eval import TrainConfig
    tcfg = TrainConfig(**{k: v for k, v in tcfg_dict.items()
                          if k in TrainConfig.__dataclass_fields__})

    data = _load_corpus(cfg)
    work_dir = Path(cfg["sweep"]["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    corpus_file = tok_train.write_corpus_file(
        data["nl_train"], data["code_texts"], work_dir / "train_corpus.txt")

    # 训练侧语料：与 17 篇基准配方同口径（nl50 + MinHash0.8 + 开过滤），
    # 配方本身不是本篇变量，词表才是——固定配方隔离词表效应。
    rs = tcfg_dict.get("recipe", {})
    spec = recipe17.RecipeSpec(
        name=rs.get("name", "nl50_dedup08_q"),
        nl_ratio=float(rs.get("nl_ratio", 0.5)),
        dedup=rs.get("dedup", "minhash08"),
        quality=bool(rs.get("quality", True)),
        char_budget=int(rs.get("char_budget", 1500000)))
    prep = recipe17.prepare_recipe(spec, data["nl_docs"], data["code_texts"])
    recipe_text = prep["text"]

    nl_chars = len(data["nl_eval"])
    code_chars = len(data["code_eval_text"])
    target_total = int(tcfg_dict.get("fixed_param_target", 12971904))

    runs = []
    for item in tcfg_dict["matrix"]:
        algo, vocab, digit = item["algo"], int(item["vocab"]), item["digit"]
        tok, info = _load_or_train_tokenizer(algo, vocab, digit, work_dir,
                                             corpus_file)
        adapter = _TokenizerAdapter(tok, info)
        res = train_eval17.train_recipe(recipe_text, adapter,
                                        data["nl_eval"], data["code_eval_text"],
                                        tcfg)
        nl_bpc = _bpc(res["nl_eval"]["loss"], res["nl_eval"]["n_tokens"], nl_chars)
        code_bpc = _bpc(res["code_eval"]["loss"], res["code_eval"]["n_tokens"],
                        code_chars)
        fixed = _fixed_budget_hidden(info["actual_vocab"], target_total,
                                     tcfg.layers)
        runs.append({
            "algo": algo, "vocab_target": vocab,
            "vocab_actual": info["actual_vocab"],
            "digit_strategy": digit,
            "tokenizer_reused": info.get("reused", False),
            "n_params": res["n_params"],
            "embedding_share": round(info["actual_vocab"] * tcfg.hidden /
                                     res["n_params"], 4),
            "train_tokens": res["train_tokens"],
            "total_steps": res["total_steps"],
            "train_sec": res["train_sec"],
            "tokens_per_sec": round(res["train_tokens"] / res["train_sec"], 1),
            "peak_memory_mib": res["peak_memory_mib"],
            "device": res["device"],
            "final_train_loss": res["final_train_loss"],
            "nl": {"ppl": res["nl_eval"]["ppl"], "bpc": nl_bpc},
            "code": {"ppl": res["code_eval"]["ppl"], "bpc": code_bpc},
            "fixed_budget": fixed,
        })
        print(f"  {algo:8s} vocab={vocab:6d} params={res['n_params']:9d} "
              f"peak={res['peak_memory_mib']} MiB "
              f"nl_bpc={nl_bpc} code_bpc={code_bpc} "
              f"fixed_H={fixed['hidden']}", flush=True)

    elapsed = round(time.perf_counter() - started, 2)
    return {
        "mode": "train",
        "truth_label": "REAL",
        "fingerprint": data["fingerprint"],
        "recipe": {"spec": spec.__dict__, "report": prep["report"]},
        "runs": runs,
        "metadata": {
            "python": platform.python_version(),
            "elapsed_sec": elapsed,
            "code_sha256": _code_sha256(),
            "train_config": tcfg_dict,
            "fixed_param_target": target_total,
            "non_emb_params_formula": "12*H^2*L + H*(2L+1)，lm_head 与 embedding 权重共享",
        },
        "note": "训练侧 REAL：固定 token 预算与配方，只换词表。ppl 跨词表不可比，"
                "BPC（bits per char）才是可比口径；fixed_budget 是固定总参数"
                "（17 篇 10M 档 12,971,904）下反解的 hidden，词表越大 hidden 越小。",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["sweep", "train"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.mode == "sweep":
        payload = mode_sweep(cfg)
    elif args.mode == "train":
        payload = mode_train(cfg)
    else:  # pragma: no cover
        raise ValueError(f"未接线的模式: {args.mode}")

    text = json.dumps(payload, ensure_ascii=False, indent=1)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"{args.mode} done: {len(payload['runs'])} runs, "
              f"elapsed={payload['metadata']['elapsed_sec']}s -> {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
