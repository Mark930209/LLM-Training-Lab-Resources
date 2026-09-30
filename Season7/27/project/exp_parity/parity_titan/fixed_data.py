"""FixedDataLoader —— 喂固定批次的数据管道（迁移自研数据管道的最小实现）。"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from torchtitan.components.dataloader import BaseDataLoader


class FixedDataLoader(BaseDataLoader):
    """迭代落盘固定批次；(input_dict, labels) 契约同 torchtitan。"""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseDataLoader.Config):
        data_file: str = "data_fixed.pt"

    def __init__(
        self,
        config: Config,
        *,
        dp_world_size: int,
        dp_rank: int,
        tokenizer=None,
        seq_len: int = 64,
        local_batch_size: int = 8,
        snapshot_every_n_steps=None,
    ):
        self.config = config
        blob = torch.load(config.data_file, weights_only=False)
        self.input_ids = blob["input_ids"]
        self.labels = blob["labels"]
        self.seq_len = seq_len
        self.local_batch_size = local_batch_size
        self.dp_world_size = dp_world_size
        self.dp_rank = dp_rank

    def __iter__(self):
        for i in range(len(self.input_ids)):
            ids = self.input_ids[i]
            labels = self.labels[i]
            positions = torch.arange(self.seq_len).unsqueeze(0) \
                .expand(ids.shape[0], -1)
            yield ({"input": ids, "positions": positions}, labels)

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass
