"""data.py —— 语料与 tokenizer：小语料 / 大语料两档。

char-level tokenizer 与 03 篇完全一致（本篇不换分词器，那是 06 篇的事），
这样"数据规模"的对照只变一个变量：语料文件。

两档语料：
    small  西游记单书 73.8 万字符（与 03 篇对齐，作为 baseline）
    large  四大名著 307 万字符（西游记 + 红楼梦 + 三国演义 + 水浒传）

为什么大语料选四大名著：四部都是明清白话章回小说，与 03 同源，风格统一；
小语料是大语料的真子集，对照时只变"数据量"这一个变量，归因干净。
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import torch
from torch.utils.data import Dataset

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

# 语料档位 → 文件名（由 scripts/build_corpus.py 产出）
CORPUS_FILES = {
    "small": "corpus_small.txt",
    "large": "corpus_large.txt",
    "shakespeare": "tiny_shakespeare.txt",
}


def download_corpus(data_dir: str | Path) -> Path:
    """下载 Tiny Shakespeare；已存在则跳过（幂等）。"""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "tiny_shakespeare.txt"
    if not path.exists():
        urllib.request.urlretrieve(DATA_URL, path)
    return path


class CharTokenizer:
    """字符级 tokenizer：字符 ↔ id 双向映射。

    encode 对词表外字符跳过（OOV 容错）：明代白话语料里没有"着"，
    现代问句里有，直接 KeyError 会让推理崩溃；跳过是标准处理，
    代价是 OOV 字符携带的信息丢失（BPE 分词器就是为解决这个问题而生）。
    """

    def __init__(self, text: str):
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)


class LMDataset(Dataset):
    """滑动窗口取样本：从长文本里按块切出 (输入, 标签) 对。

    输入是 text[i:i+block_size]，标签是 text[i+1:i+block_size+1]：
    模型在第 t 个位置看到的字符，要预测第 t+1 个字符。错一位就是标签。
    """

    def __init__(self, data: torch.Tensor, block_size: int):
        self.data = data
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) - self.block_size - 1

    def __getitem__(self, idx: int):
        chunk = self.data[idx : idx + self.block_size + 1]
        return chunk[:-1], chunk[1:]


def load_corpus(data_dir: str | Path, block_size: int = 128,
                corpus: str = "small"):
    """加载语料并切分 train/val。返回 tokenizer、两个 Dataset。

    corpus="small"：西游记单书（与 03 篇对齐）
    corpus="large"：四大名著合并
    corpus="shakespeare"：英文对照（缺失时自动下载）
    """
    data_dir = Path(data_dir)
    if corpus == "shakespeare":
        path = download_corpus(data_dir)
    else:
        fname = CORPUS_FILES.get(corpus)
        if fname is None:
            raise ValueError(f"未知语料档位: {corpus}（可选 {list(CORPUS_FILES)}）")
        path = data_dir / fname
        if not path.exists():
            raise FileNotFoundError(
                f"未找到 {path}；先运行 scripts/build_corpus.py 构建语料")
    text = path.read_text(encoding="utf-8")
    tok = CharTokenizer(text)
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    n = len(ids)
    split = int(n * 0.9)
    train_ds = LMDataset(ids[:split], block_size)
    val_ds = LMDataset(ids[split:], block_size)
    return tok, train_ds, val_ds