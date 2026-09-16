"""logging —— 每次运行留痕：可对账的第一件事。

每次实验在 runs/ 下生成带时间戳的独立目录，内含：
    config.snapshot.yaml  本次实际配置（由 config.snapshot 写入）
    metrics.jsonl         逐行 JSON 指标（loss、step time 等，可流式追加）
    run.log               文本日志（stdout 同步落盘）
    env_card.txt          环境卡片快照（硬件/版本，迁移后对账用）

留痕的意义：三个月后回看某个数字，能确定它来自哪次运行、什么环境、什么配置。
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


class RunLogger:
    """一次实验运行的留痕器。用法:

        rl = RunLogger("runs", "smoke")
        rl.log_metrics(step=0, loss=2.31)
        rl.log.info("训练开始")
        rl.close()
    """

    def __init__(self, root: str | Path, name: str):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(root) / f"{name}_{ts}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self._metrics_path = self.run_dir / "metrics.jsonl"
        self._t0 = time.perf_counter()

        self.log = logging.getLogger(f"{name}.{ts}")
        self.log.setLevel(logging.INFO)
        self.log.handlers.clear()
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
        for handler in (logging.FileHandler(self.run_dir / "run.log", encoding="utf-8"),
                        logging.StreamHandler(sys.stdout)):
            handler.setFormatter(fmt)
            self.log.addHandler(handler)

    def log_metrics(self, **kv: Any) -> None:
        """追加一行 JSON 指标；自动带上相对运行时间。"""
        record = {"t_s": round(time.perf_counter() - self._t0, 3), **kv}
        with open(self._metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def write_env_card(self, text: str) -> Path:
        """写入环境卡片快照（内容由 env_card 函数生成）。"""
        path = self.run_dir / "env_card.txt"
        path.write_text(text, encoding="utf-8")
        return path

    def close(self) -> Path:
        self.log.info("运行结束，留痕目录: %s", self.run_dir)
        for h in self.log.handlers:
            h.close()
        return self.run_dir


def env_card() -> str:
    """采集当前环境卡片文本：硬件、驱动、torch/CUDA 版本。迁移后对账用。"""
    lines = [f"时间: {datetime.now().isoformat(timespec='seconds')}"]
    try:
        import torch
        lines += [
            f"torch: {torch.__version__}",
            f"cuda build: {torch.version.cuda}",
            f"cudnn: {torch.backends.cudnn.version()}",
            f"cuda available: {torch.cuda.is_available()}",
        ]
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            free_b, total_b = torch.cuda.mem_get_info(0)
            lines += [
                f"device: {torch.cuda.get_device_name(0)}",
                f"vram GB: {round(p.total_memory / 1e9, 2)}",
                f"实际可用显存 GB: {round(free_b / 1e9, 2)} / {round(total_b / 1e9, 2)}",
                f"capability: {p.major}.{p.minor}",
                f"sm count: {p.multi_processor_count}",
            ]
    except ImportError:
        lines.append("torch: 未安装")
    lines.append(f"python: {sys.version.split()[0]}")
    lines.append(f"platform: {sys.platform}")
    return "\n".join(lines)
