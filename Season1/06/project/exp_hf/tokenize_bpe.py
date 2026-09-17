"""tokenize_bpe.py —— 用四大名著语料训练一个 BPE tokenizer（06 篇交付物）。

03/04 篇用 char-level tokenizer：一个汉字一个 token，词表 6,015。
它的两个已知缺陷（03 篇实测过）：
    OOV：语料里没有的字（现代问句里的"着"）直接被跳过，信息丢失；
    长度：一句话的 token 数等于字数，seq_len 256 只能装 256 个字。

BPE 的思路是把高频字符组合并成子词："西游"出现得多就合成一个 token。
词表从 6,015 个字符变成 8,192 个子词后，同样的文本 token 数更少，
而且未登录字可以拆成已知子词，不再整字丢失。

训练用 tokenizers 库的 BPE trainer，语料就是 04 篇的四大名著文件，
不引入任何外部数据。产出 tokenizer.json，HF 的 PreTrainedTokenizerFast
可以直接加载，训练循环只需要换 encode/decode 两个调用。

注意：换 tokenizer 之后，旧 checkpoint 的 embedding 不能复用——
词表从 6,015 变 8,192，embedding 矩阵的行数变了，id 的含义也全变了
（同一个 id 在两套词表里指向不同的 token）。这一条在
fail_modes_hf.vocab_mismatch 里被做成显式失败实验。
"""

from __future__ import annotations

import json
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

VOCAB_SIZE = 8192
SPECIAL_TOKENS = ["<|pad|>", "<|bos|>", "<|eos|>"]


def train_bpe(corpus_path: str | Path, out_path: str | Path,
              vocab_size: int = VOCAB_SIZE) -> Path:
    """从语料训练 BPE，落盘 tokenizer.json。已存在则跳过（幂等）。"""
    corpus_path, out_path = Path(corpus_path), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return out_path

    tok = Tokenizer(models.BPE(unk_token=None))
    # 中文按字符预切分：BPE 在字符序列上做合并，
    # 不按字符切开的话，整句会被当成一个"词"，合并学不出子词
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        show_progress=False,
    )
    tok.train([str(corpus_path)], trainer=trainer)
    tok.save(str(out_path))
    return out_path


def load_fast_tokenizer(tokenizer_json: str | Path) -> PreTrainedTokenizerFast:
    """包成 HF 的 fast tokenizer：训练循环拿到的是 encode/decode 两个方法。"""
    base = Tokenizer.from_file(str(tokenizer_json))
    return PreTrainedTokenizerFast(
        tokenizer_object=base,
        pad_token="<|pad|>",
        bos_token="<|bos|>",
        eos_token="<|eos|>",
    )


def compare_tokenizers(char_tok, fast_tok, samples: list[str]) -> dict:
    """char vs BPE 的对照数据：token 数、OOV 行为、编码速度。

    返回写进结果文件的 dict，文章 §tokenizer 一节直接用。
    """
    import time

    rows = []
    for text in samples:
        n_char = len(char_tok.encode(text))
        t0 = time.perf_counter()
        for _ in range(20):
            char_tok.encode(text)
        t_char = (time.perf_counter() - t0) / 20

        ids_bpe = fast_tok.encode(text, add_special_tokens=False)
        n_bpe = len(ids_bpe)
        t0 = time.perf_counter()
        for _ in range(20):
            fast_tok.encode(text, add_special_tokens=False)
        t_bpe = (time.perf_counter() - t0) / 20

        rows.append({
            "chars": len(text),
            "char_tokens": n_char,
            "bpe_tokens": n_bpe,
            "ratio": round(n_bpe / max(n_char, 1), 3),
            "char_encode_s": round(t_char, 6),
            "bpe_encode_s": round(t_bpe, 6),
        })

    # OOV 行为：现代词汇里的字，四大名著语料没有（实测缺"咖""啡"），
    # char 词表整字跳过，BPE 拆成已知子词
    oov_probe = "咖啡、巧克力和披萨"
    char_ids = char_tok.encode(oov_probe)
    bpe_ids = fast_tok.encode(oov_probe, add_special_tokens=False)
    char_kept = len(char_ids)
    bpe_roundtrip = fast_tok.decode(bpe_ids)

    return {
        "samples": rows,
        "oov_probe": {
            "text": oov_probe,
            "char_input_chars": len(oov_probe),
            "char_kept_chars": char_kept,
            "char_dropped": len(oov_probe) - char_kept,
            "bpe_tokens": len(bpe_ids),
            "bpe_roundtrip_ok": bpe_roundtrip == oov_probe,
        },
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="exp_scale/data/corpus_large.txt")
    ap.add_argument("--out", default="exp_hf/tokenizer.json")
    args = ap.parse_args()
    p = train_bpe(args.corpus, args.out)
    meta = json.loads(Path(str(p)).read_text(encoding="utf-8"))
    print(f"tokenizer 已保存: {p}, 词表 {len(meta['model']['vocab'])}")
