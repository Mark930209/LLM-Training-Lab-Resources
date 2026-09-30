"""NoopTokenizer —— 占位 tokenizer（本实验数据已是 token id，不走分词）。"""

from __future__ import annotations

from dataclasses import dataclass

from torchtitan.components.tokenizer import BaseTokenizer


class NoopTokenizer(BaseTokenizer):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseTokenizer.Config):
        pass

    def __init__(self, config: Config, *, tokenizer_path: str = ""):
        super().__init__()
        self.tokenizer_path = tokenizer_path

    def encode(self, text: str, **kwargs):
        return [0]

    def decode(self, ids, **kwargs):
        return ""

    def get_vocab_size(self):
        return 2048
