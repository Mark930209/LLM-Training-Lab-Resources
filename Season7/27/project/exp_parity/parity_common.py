"""parity_common.py —— 27 篇 Framework Parity Lab 共享层。

设计原则：三套框架的 parity 差异必须只剩"训练循环语义"，其余混淆变量全部固定：
- 模型：TorchTitan 的 Llama3Model(debugmodel, attn_backend=flex) 三处复用同一类；
- 数据：一次性生成落盘（data_fixed.pt），三套框架读同一份文件，不信 seed；
- 初始化：一次性生成落盘（init_state.pt），默认从文件加载，不信构建顺序；
- 优化器口径：AdamW(lr=8e-4, betas=(0.9,0.95), eps=1e-8, weight_decay=0.1)，
  grad clip max_norm=1.0，loss = CE(sum) / global_valid_tokens（TorchTitan 口径）。

真实性标签：全部实验产物由两机真实运行产生（REAL）；单机 world=1 对照亦为 REAL。
"""

from __future__ import annotations

import hashlib
import json
import os
import time

import torch

# ---- parity 口径（三套框架对齐的唯一真源） ----
STEPS = 200
BATCH = 8
SEQ = 64
VOCAB = 2048
LR = 8e-4
BETAS = (0.9, 0.95)
EPS = 1e-8
WD = 0.1
MAX_NORM = 1.0
WARMUP_STEPS = 20          # 200 步窗口内可观察 warmup+稳态，三处同值
SEED_MODEL = 20260930      # 初始化 seed（生成 init_state.pt 时用）
SEED_DATA = 31337          # 数据 seed（生成 data_fixed.pt 时用）
TOL = {"atol": 2e-4, "rtol": 1e-4}   # 24 篇容差判据

DATA_FILE = "data_fixed.pt"
INIT_FILE = "init_state.pt"


def build_model_spec(attn_backend: str = "flex"):
    from torchtitan.models.llama3 import model_registry

    return model_registry("debugmodel", attn_backend=attn_backend)


def build_model():
    """构造 Llama3Model(debugmodel) 并返回（不主动 init，由调用方决定）。"""
    spec = build_model_spec()
    return spec.model.build()


def make_data(path: str = DATA_FILE, seed: int = SEED_DATA) -> dict:
    """生成固定训练批次并落盘。input_ids/labels 均为 [STEPS, BATCH, SEQ]。"""
    g = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(0, VOCAB, (STEPS, BATCH, SEQ), generator=g)
    labels = torch.randint(0, VOCAB, (STEPS, BATCH, SEQ), generator=g)
    torch.save({"input_ids": input_ids, "labels": labels,
                "meta": {"steps": STEPS, "batch": BATCH, "seq": SEQ,
                         "vocab": VOCAB, "seed": seed}}, path)
    return {"input_ids": input_ids, "labels": labels}


def make_init(path: str = INIT_FILE, seed: int = SEED_MODEL) -> dict:
    """生成冻结初始权重并落盘（训练框架从文件加载，不各自初始化）。"""
    torch.manual_seed(seed)
    model = build_model()
    model.init_states()
    sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    torch.save({"state_dict": sd, "seed": seed,
                "checksum": sd_checksum(sd)}, path)
    return sd


def sd_checksum(sd: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(sd.keys()):
        v = sd[k]
        if v is None:
            continue
        if hasattr(v, "full_tensor"):      # DTensor（FSDP wrap 后）→ 全量张量
            v = v.full_tensor()
        h.update(k.encode())
        h.update(v.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def load_data(path: str = DATA_FILE) -> dict:
    return torch.load(path, weights_only=False)


def load_init(path: str = INIT_FILE) -> dict:
    obj = torch.load(path, weights_only=False)
    return obj["state_dict"]


def parity(a: torch.Tensor, b: torch.Tensor,
           atol: float = TOL["atol"], rtol: float = TOL["rtol"]) -> dict:
    a32, b32 = a.detach().float(), b.detach().float()
    diff = (a32 - b32).abs()
    maxabs = diff.max().item()
    scale = b32.abs().max().clamp_min(1e-6).item()
    return {"maxabs": maxabs, "norm_maxabs": round(maxabs / scale, 6),
            "within_tol": bool(maxabs <= atol + rtol * scale)}


def loss_parity(curve_a: list[float], curve_b: list[float]) -> dict:
    """loss 曲线 parity：逐点 maxabs + 首个分叉步（阈值 1e-5）。"""
    n = min(len(curve_a), len(curve_b))
    diffs = [abs(curve_a[i] - curve_b[i]) for i in range(n)]
    first_diverge = next((i + 1 for i, d in enumerate(diffs) if d > 1e-5), None)
    return {"steps": n, "maxabs": max(diffs) if diffs else 0.0,
            "meanabs": sum(diffs) / max(len(diffs), 1),
            "first_diverge_step": first_diverge,
            "within_tol": first_diverge is None}


def save_json(obj: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed_ms = (time.perf_counter() - self.t0) * 1000.0
