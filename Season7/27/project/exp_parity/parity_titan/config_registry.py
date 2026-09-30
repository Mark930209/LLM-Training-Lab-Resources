"""config_registry —— TorchTitan 配置入口（parity_tiny）。

环境变量：
    PARITY_OUT        结果 JSON 路径（必填）
    PARITY_INIT_FILE  冻结初始权重（缺省 = 框架原生初始化，失败案例腿）
"""

from __future__ import annotations

import dataclasses
import json
import os
import time

import torch

from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import (
    LRSchedulersContainer,
    OptimizersContainer,
    ParamGroupConfig,
)
from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.models.llama3 import model_registry
from torchtitan.trainer import Trainer

import parity_common as pc
from parity_titan.fixed_data import FixedDataLoader
from parity_titan.noop_tokenizer import NoopTokenizer

_T0 = time.perf_counter()
_MODEL_REF: list = []


def _patch_init_states():
    """冻结权重注入：PARITY_INIT_FILE 存在时用落盘权重替换框架初始化。"""
    from torchtitan.models.llama3.model import Llama3Model

    orig = Llama3Model.init_states

    def _init_states(self, **kwargs):
        _MODEL_REF.clear()
        _MODEL_REF.append(self)
        path = os.environ.get("PARITY_INIT_FILE", "")
        if path and os.path.exists(path):
            sd = pc.load_init(path)
            # DTensor 桥接：trainer 已 FSDP wrap（参数为 DTensor），
            # 冻结权重需 distribute 后才能 copy_（否则报 mixed Tensor/DTensor）
            from torch.distributed.tensor import DTensor, distribute_tensor
            cur = self.state_dict()
            bridged = {}
            for k, v in sd.items():
                p = cur.get(k)
                if isinstance(p, DTensor) and not isinstance(v, DTensor):
                    v = distribute_tensor(v, p.device_mesh, p.placements)
                bridged[k] = v
            self.load_state_dict(bridged, strict=True)
        else:
            orig(self, **kwargs)

    Llama3Model.init_states = _init_states


class MetricsCapture(MetricsProcessor):
    """逐步 loss 曲线落盘（Parity Harness 的取数层）。"""

    @dataclasses.dataclass(kw_only=True, slots=True)
    class Config(MetricsProcessor.Config):
        pass

    def log(self, step, global_avg_loss, global_max_loss, grad_norm,
            extra_metrics=None):
        if not hasattr(self, "_loss_curve"):
            self._loss_curve = []
        self._loss_curve.append(float(global_avg_loss))
        out = os.environ.get("PARITY_OUT", "")
        if out:
            entry = {
                "experiment": os.path.basename(out).replace(".json", ""),
                "framework": "torchtitan",
                "init": "file" if os.environ.get("PARITY_INIT_FILE") else "native",
                "world_size": int(os.environ.get("WORLD_SIZE", 1)),
                "steps_logged": step,
                "loss_curve": self._loss_curve,
                "loss_first": self._loss_curve[0],
                "loss_last": float(global_avg_loss),
                "param_checksum": pc.sd_checksum(_MODEL_REF[0].state_dict())
                if _MODEL_REF else None,
                "startup_to_first_step_s": getattr(self, "_t_first", None),
                "labels": {"REAL": "真实运行"},
            }
            if entry["startup_to_first_step_s"] is None:
                self._t_first = round(time.perf_counter() - _T0, 3)
                entry["startup_to_first_step_s"] = self._t_first
            pc.save_json(entry, out)
        return super().log(step, global_avg_loss, global_max_loss, grad_norm,
                           extra_metrics=extra_metrics)


def parity_tiny() -> Trainer.Config:
    _patch_init_states()
    spec = model_registry("debugmodel", attn_backend="flex")
    opt_cfg = OptimizersContainer.Config(
        implementation="for-loop",   # fused kernel 会引入单 ulp 差并被混沌放大
        param_groups=[
            ParamGroupConfig(pattern=r".*", optimizer_name="AdamW",
                             optimizer_kwargs={"lr": pc.LR, "betas": pc.BETAS,
                                               "eps": pc.EPS,
                                               "weight_decay": pc.WD}),
        ])
    cfg = Trainer.Config(
        loss=CrossEntropyLoss.Config(global_vocab_size=pc.VOCAB),
        model_spec=spec,
        optimizer=opt_cfg,
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=pc.WARMUP_STEPS),
        training=TrainingConfig(
            local_batch_size=pc.BATCH,
            seq_len=pc.SEQ,
            steps=pc.STEPS,
            max_norm=pc.MAX_NORM,
            disable_cuda_graphs=True,
            dtype="float32",
            mixed_precision_param="float32",   # 默认 bfloat16 会改写参数精度语义
            mixed_precision_reduce="float32",
        ),
        dataloader=FixedDataLoader.Config(data_file=pc.DATA_FILE),
        tokenizer=NoopTokenizer.Config(),
        metrics=MetricsCapture.Config(log_freq=1),
        # replicate 模式（DDP 语义）：FSDP wrap 会把参数变成 DTensor，
        # 冻结权重注入与 checksum 都要额外桥接——parity 腿先排除这个混淆变量
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=int(
                os.environ.get("PARITY_DP_REPLICATE", "1")),
            data_parallel_shard_degree=1,
        ),
        activation_checkpoint=None,   # 不做选择性重计算（若类型不允许会报错再调）
    )
    return cfg
