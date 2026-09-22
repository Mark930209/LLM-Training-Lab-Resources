# 02《搭建一套可复用的大模型基础训练环境》资源包

对应文章：`Articles/Season0/02_搭建一套可复用的大模型基础训练环境.md`

## 包内容

```text
02/
├── README.md                  # 本文件
├── scripts/
│   ├── setup_env.sh           # WSL2/Ubuntu 内一键初始化训练环境（uv + CUDA PyTorch + 验证）
│   ├── check_env.sh           # 环境状态体检：GPU/驱动/CUDA/torch/uv/磁盘，输出环境卡片
│   ├── smoke_train.py         # 最小 GPU 训练验证：loss 下降 + step time/显存/吞吐采集，结果落盘 JSON
│   ├── reproducibility_test.py# 同 seed 两次独立进程逐位一致 vs 异 seed 发散，验证环境可复现性
│   ├── collect_data.sh        # 一键采集全部实验数据（torch 验证/lock/环境卡片/冒烟/复现性/CPU 降级）落盘 results/
│   ├── migrate_backup.ps1     # Windows 侧 WSL2 环境整体导出/导入/瘦身（环境可迁移的直接证据）
│   ├── migrate_benchmark.ps1  # 迁移实测数据采集（导出/导入耗时与文件大小）
│   └── agent_diagnosis_prompt.md # 配合大模型 agent 排环境问题的 prompt 模板
└── project/                   # common/ 可复现骨架 + exp_smoke 冒烟实验（统一接口样板）
    ├── README.md              # 工程使用说明
    ├── requirements.lock.txt  # 完整依赖锁（57 包，全系列通用，含 06 篇起的 transformers）
    ├── common/                # reproducibility / config / logging / benchmark / metrics 五模块
    └── exp_smoke/             # config.yaml + run.py（run_experiment 公共接口最小实现）
```

## 依赖锁：一份管全系列

`project/requirements.lock.txt` 是**整个系列**的完整依赖锁（57 包），不只是本篇的。后续各篇引入新库时，锁文件会同步更新，读者始终按这一份装环境即可。

```bash
pip install -r project/requirements.lock.txt
# 或用 uv（本工程的 venv 就是 uv 建的，venv 里没有 pip）
uv pip install --python ./.venv/bin/python -r project/requirements.lock.txt
```

两个容易踩的点：

- **本篇只需要 torch + numpy + pyyaml**，锁里其余的包是后续篇用的。一次装全可以省掉后面反复补依赖。
- **06 篇起需要 `transformers` 与 `tokenizers`**。`results/Season0/02/requirements.lock.txt` 是本篇采集当时的历史快照（31 包，不含 transformers），只作为"环境可复现"的证据保留；装环境请用 `project/` 下这份完整的。

## 快速使用

### 环境初始化（WSL2 Ubuntu 内）

```bash
# 全部参数可覆盖，默认值即可用
bash setup_env.sh \
  --project-dir ~/llm-training-lab \
  --python-ver 3.12 \
  --mirror https://pypi.tuna.tsinghua.edu.cn/simple \
  --packages "torch numpy pyyaml"
```

脚本特性：幂等（重复运行安全）、非交互（可被 agent 直接调用）、自验证（结束打印 torch.cuda 验证结果与成功标志）。

### 环境体检

```bash
bash check_env.sh --project-dir ~/llm-training-lab
```

输出本机环境卡片（硬件/驱动/CUDA/torch/uv/磁盘），可直接粘贴给 agent 或附在问题报告里。

### 一键采集实验数据（WSL2 内）

```bash
bash collect_data.sh --project-dir ~/llm-training-lab
```

依次产出：torch 验证、依赖锁、环境卡片、冒烟训练（GPU + CPU 降级两组）、复现性对照，全部落盘到仓库 `results/Season0/02/`。

### 环境整体迁移（Windows 侧 PowerShell）

```powershell
.\migrate_backup.ps1 export  -Distro Ubuntu-24.04 -OutFile D:\backup\ubuntu-lab.tar
.\migrate_backup.ps1 import  -Backup D:\backup\ubuntu-lab.tar -Distro Ubuntu-Lab
.\migrate_backup.ps1 compact -Distro Ubuntu-24.04
```

换机器或切原生 Linux 时，项目目录与依赖锁原封不动，只有环境卡片的硬件行需要更新。

### agent 辅助排错

遇到报错时，按 `scripts/agent_diagnosis_prompt.md` 的模板，把"报错原文 + 环境体检输出"一起丢给 Copilot/Claude 等 agent 工具，比逐条搜索快得多。

## 前置条件

- Windows 10/11 + WSL2（或原生 Ubuntu 22.04/24.04）
- NVIDIA GPU + Windows 侧驱动已装（WSL2 内不需要单独装驱动）
- BIOS 已开启虚拟化（Intel VT-x / AMD SVM）

## 已知问题速查

| 现象 | 根因 | 修法 |
|---|---|---|
| `HypervisorPresent=False` | BIOS 虚拟化未开 | 进 BIOS 开 VT-x/SVM |
| `wsl --update` 403 | web-download 通道被禁 | 去掉 `--web-download` 走 Store 通道 |
| `ensurepip is not available` | Ubuntu 24.04 缺 python3.12-venv | `sudo apt install python3.12-venv` |
| uv 下载超时 | 直连 pypi.org | 配置 `~/.config/uv/uv.toml` 国内镜像 |
| `cuda available: False` | 装成 CPU 版 torch | 检查 `torch.version.cuda`，重装 CUDA 版 |

完整排查手册见文章第 4 章。
