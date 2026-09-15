"""config —— 配置与代码分离：可迁移的第一件事。

换硬件时只改 config，不改代码。config.yaml 分两段：
    experiment: 实验本身的超参（模型、数据、训练）——迁移时不变
    hardware:   硬件相关项（设备、批大小、精度）——迁移时只改这里

支持 CLI 覆盖（--hardware.batch_size=8），方便 agent 与脚本化调参。
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """dict 子类，支持点号访问：cfg.hardware.batch_size。"""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError:
            raise AttributeError(key) from None
        return Config(value) if isinstance(value, dict) else value

    def deep_update(self, other: dict) -> "Config":
        for k, v in other.items():
            if isinstance(v, dict) and isinstance(self.get(k), dict):
                Config(self[k]).deep_update(v)
            else:
                self[k] = v
        return self


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    """加载 YAML 配置并应用 CLI 覆盖。

    参数:
        path: config.yaml 路径。
        overrides: 形如 ["hardware.batch_size=8", "experiment.lr=1e-4"] 的覆盖列表。
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = Config(yaml.safe_load(f))

    for item in overrides or []:
        key, _, raw = item.partition("=")
        if not _:
            raise ValueError(f"覆盖项格式应为 key=value: {item}")
        value: Any = yaml.safe_load(raw)  # 自动解析数字/布尔/null
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value

    return cfg


def resolve_device(cfg: Config) -> str:
    """按配置解析设备；配置要 cuda 但不可用时明确报错而不是静默回退。

    静默回退 CPU 会让 benchmark 数据失去意义，宁可失败。
    """
    want = cfg.get("hardware", {}).get("device", "cuda")
    if want == "cuda" and not __import__("torch").cuda.is_available():
        raise RuntimeError("config 要求 cuda 但 torch.cuda.is_available()=False；"
                           "如确需 CPU 运行，请显式设置 hardware.device=cpu")
    return want


def snapshot(cfg: Config, run_dir: str | Path) -> Path:
    """把本次运行实际使用的 config 快照写入运行目录（留痕，可对账）。"""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "config.snapshot.yaml"
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(copy.deepcopy(dict(cfg)), f, allow_unicode=True, sort_keys=False)
    return out
