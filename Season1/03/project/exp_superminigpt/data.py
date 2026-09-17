"""data.py —— 语料与 tokenizer：GPT 学习的第一件事。

char-level tokenizer：把文本拆成字符，每个字符一个 id。
为什么不用 BPE：本篇的主题是"让模型学会学习"，tokenizer 的复杂度
会淹没机制本身；char-level 词表小（~100）、无 OOV，是教学最优解。
04 篇继续用 char-level 把语料规模做大；06 篇接入 Hugging Face 时
会切换到 BPE，到时候讲 BPE 的机制与代价。

数据集：Tiny Shakespeare（karpathy char-rnn 用的那份，约 1.1MB，公开语料）。
训练/验证切分 90/10，验证集是判断"模型在学还是在背"的唯一标尺。
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import torch
from torch.utils.data import Dataset

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


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


def load_corpus(data_dir: str | Path, block_size: int = 128, corpus: str = "shakespeare"):
    """加载语料并切分 train/val。返回 tokenizer、两个 Dataset。

    corpus="shakespeare"：char-level 英文，65 词表，缺失时自动下载。
    corpus="xiyouji"：读 data_dir/xiyouji.txt（由 scripts/build_corpus.py 预先产出），
        char-level 中文，词表为语料去重汉字 + 标点，约数千。
    """
    data_dir = Path(data_dir)
    if corpus == "xiyouji":
        path = data_dir / "xiyouji.txt"
        if not path.exists():
            raise FileNotFoundError(
                f"未找到 {path}；先运行 scripts/build_corpus.py 构建西游记语料")
    else:
        path = download_corpus(data_dir)
    text = path.read_text(encoding="utf-8")
    tok = CharTokenizer(text)
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    n = len(ids)
    split = int(n * 0.9)
    train_ds = LMDataset(ids[:split], block_size)
    val_ds = LMDataset(ids[split:], block_size)
    return tok, train_ds, val_ds
