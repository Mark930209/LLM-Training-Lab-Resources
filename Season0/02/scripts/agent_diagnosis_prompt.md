# 环境诊断提示词模板（配合大模型 Agent 排错）

环境搭建的报错排查是标准化劳动，推荐交给 Copilot / Claude 这类 agent 处理。人保留三件事：平台路线、包管理器、版本 pin 的决策，以及最终验证。

## 使用方法

1. 先跑 `check_env.sh`，复制全部输出。
2. 复制出错的命令和**完整报错原文**（不要截断，堆栈越全越好）。
3. 按下面模板拼好，丢给 agent。

## 模板一：环境报错诊断

```text
我在搭建 LLM 训练环境时遇到报错，请帮我定位根因并给出修复命令。

【环境体检输出】
（粘贴 check_env.sh 的全部输出）

【执行的命令】
（粘贴你运行的命令）

【完整报错】
（粘贴报错原文，含堆栈）

【我已尝试】
（如果已经试过某些方法，写在这里，避免 agent 重复建议）

请按这个格式回答：
1. 根因（一句话）
2. 修复命令（可直接复制执行）
3. 验证方式（修完怎么确认成功）
```

## 模板二：让 agent 直接执行修复（VS Code Copilot 等可执行命令的 agent）

```text
我的训练环境出问题了，报错如下。请你：
1. 先运行 DevResources/Season0/02/scripts/check_env.sh 收集环境状态
2. 根据报错定位根因
3. 直接执行修复命令
4. 修复后重新验证，把验证输出给我看

【报错原文】
（粘贴）

约束：不要升级我已 pin 的 torch 版本；涉及 sudo 的命令先列出来让我确认。
```

## 常见报错的关键词索引

把报错里的这些关键词直接给 agent，定位更快：

| 报错关键词 | 大概率方向 |
|---|---|
| `HypervisorPresent=False` / `Wsl/Service/CreateInstance` | BIOS 虚拟化未开 |
| `已禁止(403)` / `wsl --update` | 更新通道问题，去掉 `--web-download` |
| `ensurepip is not available` | 缺 `python3.x-venv` 系统包 |
| `Failed to fetch` / `operation timed out`（pypi.org） | 镜像源未配置 |
| `cuda available: False` 但 nvidia-smi 正常 | 装成 CPU 版 torch |
| `no CUDA-capable device is detected` | WSL GPU 直通或驱动版本问题 |
| `CUDA error: out of memory` | 显存不足或碎片化（先对账 nvidia-smi 与 torch） |
| `GLIBC_x.xx not found` | 系统版本过旧，与 wheel 不匹配 |

## 边界（哪些不要外包给 agent）

- **版本 pin 决策**：torch/CUDA 版本组合影响后续所有实验，人来定，agent 只执行。
- **硬件判断**：要不要升级显卡、要不要上云 GPU，是成本与路线决策，人来定。
- **最终验证**：agent 说"修好了"不算数，人以验证命令的真实输出为准。
