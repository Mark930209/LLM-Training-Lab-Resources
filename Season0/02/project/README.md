# 02 篇完整工程：可复现实验骨架

本目录是《搭建一套可复用的大模型基础训练环境》的配套完整工程，也是后续 37 篇共用的骨架。

## 结构

```text
project/
├── common/                  # 可复现四件事的最小实现
│   ├── reproducibility.py   # seed 管理（Python/NumPy/Torch CPU+CUDA/DataLoader worker）
│   ├── config.py            # 配置与代码分离（YAML + CLI 覆盖 + 运行快照）
│   ├── logging.py           # 每次运行留痕（时间戳目录 + metrics.jsonl + 环境卡片）
│   ├── benchmark.py         # 量化采集（StepTimer 预热丢弃 + 显存峰值 + 吞吐）
│   └── metrics.py           # 指标口径（困惑度、显存三方对账、参数体积）
└── exp_smoke/               # 冒烟实验：统一接口 run_experiment(config) 的样板
    ├── config.yaml          # experiment 段（迁移不变）+ hardware 段（换机只改这里）
    └── run.py               # 入口：读 config → 固定 seed → 训练 → benchmark → 留痕
```

## 使用

```bash
# 在 project/ 目录下，激活环境后：
pip install pyyaml   # 或 uv pip install pyyaml（config 模块依赖）

# 默认配置跑冒烟实验
python -m exp_smoke.run --config exp_smoke/config.yaml

# CLI 覆盖硬件项（换机器只改配置，不改代码）
python -m exp_smoke.run --config exp_smoke/config.yaml --override hardware.batch_size=8

# 结果落盘
python -m exp_smoke.run --config exp_smoke/config.yaml --output results.json
```

每次运行在 `runs/<name>_<时间戳>/` 下留痕：config 快照、metrics.jsonl、run.log、环境卡片。

## 设计目标：可复现 + 可迁移

- **同机复现**：`set_all_seeds` 固定全部随机源，同 seed 两次运行 loss 逐位一致（用 `scripts/reproducibility_test.py` 验证）。
- **跨机迁移**：config 分 experiment/hardware 两段，换到更大显存的卡只改 hardware 段；`requirements.lock.txt` 保证依赖一致；每次运行留痕的环境卡片让历史记录可对账。
- **口径统一**：所有专题的 benchmark 都走 `run_experiment(config)` 返回标准化 dict（loss / step_time_ms / tokens_per_sec / peak_memory_mb），跨篇可比。
