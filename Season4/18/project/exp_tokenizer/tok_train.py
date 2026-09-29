"""tok_train.py —— 训练 BPE 与 Unigram tokenizer。

两条实现路线（都是工业界主流）：
  - BPE      : HuggingFace `tokenizers`（Rust 实现，GPT 系用的就是这条路线）
  - Unigram  : `sentencepiece`（SentencePiece 库，LLaMA/T5 系用的路线）

数字切分策略（digit_strategy）：
  - "merge"  : 连续数字合并成一个片段再进 BPE（GPT-2 风格，\\p{N}+）
  - "single" : 数字逐位切分（GPT-4 / LLaMA 3 风格，\\p{N}）
  算术任务上的差异由这两种策略直接体现。

语料复用 17 篇 exp_recipe.corpus 的两域切分：tokenizer 训练只用 train 池，
nl_eval / code_eval 永不参与训练——与 17 篇同一把尺子。
"""

from __future__ import annotations

import time
from pathlib import Path

# GPT-2 风格 pre-tokenize 正则：数字合并（\p{N}+）
_PAT_DIGIT_MERGE = (
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)
# GPT-4 风格：数字逐位（\p{N}）
_PAT_DIGIT_SINGLE = (
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)

SPECIAL_TOKENS = ["<pad>", "<eos>", "<unk>"]
EOS_TOKEN = "<eos>"


def write_corpus_file(nl_train: str, code_texts: list[str], path: str | Path) -> Path:
    """把两域训练池写成单个语料文件（tokenizer 训练的输入）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    parts = [nl_train]
    parts.extend(code_texts)
    p.write_text("\n".join(parts), encoding="utf-8")
    return p


def train_bpe(corpus_file: str | Path, vocab_size: int, digit_strategy: str,
              out_dir: str | Path) -> dict:
    """训练 ByteLevel BPE，返回 {path, vocab_size, train_sec, actual_vocab}。"""
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

    pattern = _PAT_DIGIT_SINGLE if digit_strategy == "single" else _PAT_DIGIT_MERGE
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    # GPT-2 风格预切分：按正则把文本切成片段，isolated 表示每个匹配独立成段
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(pattern), "isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False),
    ])
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        show_progress=False,
    )
    started = time.perf_counter()
    tok.train([str(corpus_file)], trainer)
    elapsed = time.perf_counter() - started

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"bpe_{vocab_size // 1000}k_{digit_strategy}.json"
    tok.save(str(path))
    return {"algo": "bpe", "path": str(path), "vocab_size": vocab_size,
            "digit_strategy": digit_strategy,
            "actual_vocab": tok.get_vocab_size(),
            "train_sec": round(elapsed, 2)}


def train_unigram(corpus_file: str | Path, vocab_size: int, digit_strategy: str,
                  out_dir: str | Path) -> dict:
    """训练 SentencePiece Unigram（byte fallback 开），返回同构 dict。"""
    import sentencepiece as spm

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / f"unigram_{vocab_size // 1000}k_{digit_strategy}"
    started = time.perf_counter()
    spm.SentencePieceTrainer.train(
        input=str(corpus_file),
        model_prefix=str(prefix),
        model_type="unigram",
        vocab_size=vocab_size,
        character_coverage=0.9995,
        split_digits=(digit_strategy == "single"),
        byte_fallback=True,
        bos_id=-1,
        eos_id=1,
        pad_id=0,
        unk_id=2,
        input_sentence_size=2_000_000,
        shuffle_input_sentence=True,
        max_sentence_length=8192,
        minloglevel=2,
    )
    elapsed = time.perf_counter() - started
    sp = spm.SentencePieceProcessor(model_file=str(prefix) + ".model")
    return {"algo": "unigram", "path": str(prefix) + ".model",
            "vocab_size": vocab_size, "digit_strategy": digit_strategy,
            "actual_vocab": sp.get_piece_size(),
            "train_sec": round(elapsed, 2)}


def load_tokenizer(info: dict):
    """按训练产物信息加载 tokenizer，统一 encode/decode 接口。"""
    if info["algo"] == "bpe":
        from tokenizers import Tokenizer
        return Tokenizer.from_file(info["path"])
    import sentencepiece as spm
    return spm.SentencePieceProcessor(model_file=info["path"])


def encode_text(tokenizer, info: dict, text: str) -> list[int]:
    """统一编码入口：两种实现返回同构的 id 列表。"""
    if info["algo"] == "bpe":
        return tokenizer.encode(text).ids
    return tokenizer.encode(text)


def decode_ids(tokenizer, info: dict, ids: list[int]) -> str:
    """统一解码入口。

    SentencePiece 必须用 detokenize：它的 decode 只把 id 拼回 piece，
    不还原 ▁ 空白标记，往返对比会虚报覆盖率损失（冒烟实测 0.07）。
    """
    if info["algo"] == "bpe":
        return tokenizer.decode(ids)
    return tokenizer.detokenize(ids)
