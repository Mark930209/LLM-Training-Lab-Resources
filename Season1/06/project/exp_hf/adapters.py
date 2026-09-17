"""adapters.py —— 把模型接入契约的两条路径（06 篇核心交付物）。

路径一：SuperMiniGPTAdapter
    把 04 篇手写的 SuperMiniGPT 包成 Hugging Face 的 PreTrainedModel。
    包一层不是为了好看：subclass PreTrainedModel 之后，
    save_pretrained / from_pretrained / config.json 这些标准件全部免费获得，
    04 篇的模型从此和 HF 生态的模型走同一套序列化与加载代码。

路径二：build_llama
    用 LlamaConfig 从零搭一个消费级 GPU 能短训的小 Llama-style 模型。
    不下载任何权重：config 是唯一输入，权重随机初始化后由 04 篇的循环训练。

两条路径对训练循环呈现的接口完全相同（见 contract.py 的五条契约），
所以循环一行不改就能在两者之间切换。

训练状态（优化器/调度器/步数）不进 HF 目录：save_pretrained 只管权重与配置，
把它们塞进去会污染标准格式。训练状态走 sidecar 文件（training_state.pt），
恢复时两边一起读，缺一个就报警（见 fail_modes_hf.resume_weights_only）。
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exp_scale.model import SuperMiniGPT  # noqa: E402


class SuperMiniGPTConfig(PretrainedConfig):
    """04 篇模型的配置序列化：把决定参数量的几个数存成 dict。

    继承 PretrainedConfig 是为了让 config 能 to_dict / to_json_file，
    from_pretrained 时只凭这个 dict 就能重建同结构的模型。
    """

    model_type = "superminigpt"

    def __init__(self, vocab_size: int = 6015, hidden: int = 384, layers: int = 6,
                 heads: int = 6, seq_len: int = 256, tie_weights: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden = hidden
        self.layers = layers
        self.heads = heads
        self.seq_len = seq_len
        self.tie_weights = tie_weights
        # HF 的 tie_weights() 认这个标准字段：from_pretrained 后自动重 tie
        self.tie_word_embeddings = tie_weights


class SuperMiniGPTAdapter(PreTrainedModel):
    """SuperMiniGPT 的 HF 外壳：forward 只转发，不加任何计算。

    外壳必须"透明"：adapter 与原模型的 logits 逐位一致，
    否则接入本身就成了一个偷偷改变实验的变量（见 parity_test.py 的验证）。
    """

    config_class = SuperMiniGPTConfig
    base_model_prefix = "superminigpt"
    supports_gradient_checkpointing = False
    # 04 篇模型 embedding 与 lm_head 共享权重；HF 保存时要去重，
    # 必须显式声明哪些键是 tied 的，否则 save_pretrained 直接报错。
    # transformers 5.x 里这是 dict：键是匹配完整参数名的模式
    _tied_weights_keys = {"core.lm_head.weight": "core.tok_emb.weight"}

    def __init__(self, config: SuperMiniGPTConfig):
        super().__init__(config)
        # 与 04 篇 train.py 的构造调用完全一致：(vocab, hidden, layers, heads, seq_len)
        self.core = SuperMiniGPT(
            config.vocab_size, config.hidden, config.layers,
            config.heads, config.seq_len)
        # post_init 负责初始化 all_tied_weights_keys 等静态属性，
        # 并触发 tie_weights()；不调它 save/load 都会缺属性
        self.post_init()
        # HF 的 from_pretrained 在 meta device 上实例化、加载后再物化：
        # 04 篇 RoPE 的 cos/sin 是 non-persistent buffer，不进 safetensors，
        # 物化时变成未初始化内存（或零填充），前向直接 NaN。
        # 修法：在 Adapter 里把它们重新注册成 persistent buffer，
        # 随权重一起存进 safetensors，加载时逐位还原。
        self._make_rope_persistent()

    @classmethod
    def from_superminigpt(cls, model: SuperMiniGPT, config: SuperMiniGPTConfig):
        """用已有实例构造 adapter（权重共享，不复制）：parity 测试用它。"""
        adapter = cls(config)
        adapter.core = model
        # __init__ 里的注册作用在随后被替换掉的 core 上，
        # 换入共享实例后必须对新的 core 再注册一次
        adapter._make_rope_persistent()
        return adapter

    def _make_rope_persistent(self) -> None:
        """把每个 block 的 RoPE cos/sin 重新注册为 persistent buffer。

        04 篇用 persistent=False（表是算出来的，不占存档体积）；
        但 HF 标准格式只序列化 persistent buffer 与参数，
        non-persistent 的表在 from_pretrained 后会丢失。
        重新注册不改数值，只改"是否进存档"这一个属性。
        """
        for blk in self.core.blocks:
            rope = blk.attn.rope
            if not getattr(rope, "use_rope", True):
                continue
            cos, sin = rope.cos, rope.sin
            rope.register_buffer("cos", cos, persistent=True)
            rope.register_buffer("sin", sin, persistent=True)

    def _init_weights(self, module: nn.Module) -> None:
        """no-op：core（SuperMiniGPT）在自己的 __init__ 里已完成初始化。

        post_init() 会对所有子模块调 _init_weights；如果这里再初始化一次，
        from_superminigpt 共享进来的权重会被覆盖，parity 测试立刻不为零。
        """
        return

    def forward(self, input_ids: torch.Tensor, **kwargs) -> CausalLMOutputWithPast:
        logits = self.core(input_ids)
        return CausalLMOutputWithPast(logits=logits)

    def get_input_embeddings(self) -> nn.Module:
        return self.core.tok_emb

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.core.tok_emb = value

    def get_output_embeddings(self) -> nn.Module:
        return self.core.lm_head

    def set_output_embeddings(self, value: nn.Module) -> None:
        self.core.lm_head = value


def build_llama(vocab_size: int, hidden: int = 384, layers: int = 6,
                heads: int = 6, head_dim: int = 64, intermediate: int = 1024,
                max_seq_len: int = 1024) -> LlamaForCausalLM:
    """从零搭小 Llama-style 模型：config 是唯一输入，不下载权重。

    选 Llama 架构是因为它是 HF 生态里 Causal LM 的事实标准接口：
    RoPE + RMSNorm + SwiGLU + GQA 可选，与 04 篇手写组件一一对应，
    读者能看清"自己写的每个组件，在 HF 里叫什么名字"。
    """
    cfg = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=heads,
        head_dim=head_dim,
        max_position_embeddings=max_seq_len,
        tie_word_embeddings=True,
        use_cache=False,
    )
    return LlamaForCausalLM(cfg)


def save_training_state(path: Path, opt, scaler, step: int, best_val: float,
                        sched_state: dict | None = None) -> None:
    """sidecar：训练状态单独存，不碰 HF 标准目录。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "optimizer": opt.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": step,
        "best_val": best_val,
        "sched": sched_state,
    }, path)


def load_training_state(path: Path, opt, scaler, device: str) -> tuple[int, float]:
    """读 sidecar；文件不存在直接抛错，由调用方决定是报错还是降级。"""
    st = torch.load(path, map_location=device, weights_only=False)
    opt.load_state_dict(st["optimizer"])
    if scaler is not None and st.get("scaler") is not None:
        scaler.load_state_dict(st["scaler"])
    return st.get("step", 0), st.get("best_val", float("inf"))


def describe_model(model: nn.Module) -> dict:
    """参数量与可训练参数：model swap 对照表用。"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"params_total": total, "params_trainable": trainable}


def config_summary(model: nn.Module) -> dict:
    """把 config 拍平成 dict，写进结果文件，方便读者对照两模型的差异。"""
    cfg = getattr(model, "config", None)
    if cfg is None:
        return {}
    d = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)
    keep = ("model_type", "vocab_size", "hidden_size", "hidden", "num_hidden_layers",
            "layers", "num_attention_heads", "heads", "intermediate_size",
            "max_position_embeddings", "seq_len", "tie_word_embeddings", "tie_weights")
    return {k: d[k] for k in keep if k in d}
