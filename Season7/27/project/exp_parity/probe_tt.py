"""probe_tt.py — 打印 TorchTitan 关键配置默认值（对齐用）。

用法：python probe_tt.py
"""
from torchtitan.components.optimizer import (
    LRSchedulersContainer,
    OptimizersContainer,
    ParamGroupConfig,
)
from torchtitan.config import TrainingConfig
from torchtitan.models.llama3 import model_registry
from torchtitan.models.common import compute_ffn_hidden_dim


def main():
    print("== ParamGroupConfig 默认 ==")
    pg = ParamGroupConfig(pattern=r".*", optimizer_name="AdamW")
    print(pg)
    print("== OptimizersContainer.Config ==")
    print(OptimizersContainer.Config())
    print("== TrainingConfig ==")
    tc = TrainingConfig()
    for f in ("local_batch_size", "seq_len", "steps", "max_norm", "warmup", "dtype"):
        print(f, "=", getattr(tc, f, "<无此字段>"))
    print("== 其余 TrainingConfig 字段 ==")
    print(tc)
    print("== LRSchedulersContainer.Config ==")
    print(LRSchedulersContainer.Config())
    print("== debugmodel 模型参数 ==")
    spec = model_registry("debugmodel", attn_backend="sdpa")
    m = spec.model
    for f in ("dim", "vocab_size", "n_layers", "n_heads", "n_kv_heads", "hidden_dim"):
        print(f, "=", getattr(m, f, "<无>"))
    print("ffn_hidden_dim(256, multiple_of=256) =", compute_ffn_hidden_dim(256, multiple_of=256))
    print("attn_backend =", getattr(m, "attn_backend", "<无>"))


if __name__ == "__main__":
    main()
