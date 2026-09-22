# DevResources —— 配套工程资源包

本目录存放《LLM Training Lab》系列每篇文章的**配套脚本、完整工程与说明文档**，是后续可独立分享的资源包。

## 设计原则

1. **按 Season + 文章索引号组织**：`Season{N}/{文章索引号}/`，与 `Articles/` 一一对应，避免错放。
2. **工程包内文件名一律英文**：目录、脚本、文档全部英文命名，避免跨平台、跨终端的编码问题；代码内可以有中文注释，文档内容可以是中文。
3. **脚本尽量通用**：同一脚本服务多篇文章，差异通过参数或环境变量注入，不为单篇写一次性脚本。
4. **说明文档与脚本同处**：每个资源包内放 `README.md`，读者不看文章正文也能确认前置模块、运行命令与预期输出。

## 目录结构

```text
DevResources/
├── README.md                  # 本文件：资源包总说明
├── Season0/
│   └── 02/                    # 《搭建一套可复用的大模型基础训练环境》
│       ├── README.md          # 本篇资源使用说明
│       ├── scripts/           # 环境初始化、体检、冒烟训练、复现性对照、数据采集、迁移
│       └── project/           # common/ 可复现骨架 + exp_smoke 统一接口样板
├── Season1/
│   ├── 03/                    # SuperMiniGPT v0（project/ 含 common + exp_superminigpt + 语料）
│   ├── 04/                    # 可复用训练程序（project/ 含 common + exp_scale + 四大名著语料）
│   ├── 05/                    # 训练正确性体检（project/ 含 exp_debug）
│   └── 06/                    # HF 接口契约（project/ 含 exp_hf）
├── Season2/
│   ├── 07/                    # 显存去哪了（project/ 含 exp_mem）
│   ├── 08/                    # 显存优化六项交换（project/ 含 exp_opt）
│   ├── 09/                    # Attention Kernel Lab（project/ 含 exp_attn）
│   └── 10/                    # 单卡性能（project/ 含 exp_perf）
└── Season3/
    └── 12/                    # DDP 正确性（project/ 含 exp_ddp）
```

> 11 篇（双卡训练环境搭建）的工程包待该篇定稿后补入 `Season3/11/`。
> 其跨机环境的实验档案已先行落在 `results/Season3/11/`。

## 各篇包是累积的，不是独立的

**这是使用本资源最重要的一条**。系列各篇的实验层层依赖：12 篇的 `exp_ddp` 要 import 06 篇的 `build_llama` 与 04 篇的 `CharTokenizer`，10 篇的 `exp_perf` 要复用 06 篇的 `contract_loss`。

所以每篇的 `project/` 只含**本篇新增**的模块，不是完整可独立运行的工程。正确用法是按篇号顺序，把各篇 `project/` 下的模块累积拷进同一个工作目录：

```bash
# 02 篇建立工作目录与共享骨架
mkdir -p ~/llm-training-lab && cd ~/llm-training-lab
cp -r <DevResources>/Season0/02/project/common .
cp -r <DevResources>/Season0/02/project/exp_smoke .

# 之后每篇只加自己的模块
cp -r <DevResources>/Season1/04/project/exp_scale .
cp -r <DevResources>/Season1/06/project/exp_hf .
cp -r <DevResources>/Season3/12/project/exp_ddp .
# ……依此类推
```

02/03/04 篇的包里各自带了一份 `common/`（内容相同），是为了让前三篇能独立起步；从 05 篇起不再重复携带，直接复用已累积的 `common/`。

每篇 README 的"前置"一节会列清本篇需要哪些前序模块。累积完成后，10 个 `exp_*` 模块全部可 import，各篇命令即可运行。

## 按学习路径累积

资源包按专题编号顺序累积：

```text
02 → 03 → ... → 32（毕业作品）→ 33 → ... → 43
```

每篇 README 的“前置”一节列明本篇真正需要哪几个前序模块，以它为准，不要默认复制全部前序包。后续模块不得反向 import 更高编号的模块；确需复用的底层能力应下沉到共享接口。

现有资源仍采用手工累积模块的方式，这次课程重组不改代码目录。把各阶段改成一个可 clone、可切换版本的学习工程，列在 `Docs/operations/curriculum-rollout-recommendations.md`，作为后续独立任务执行。

## 与文章的关系

- 文章正文只贴**关键片段**（10~40 行/处）并解释思路。
- 完整可运行版本在本资源包内，读者可直接执行。
- 文章脱离资源包也能读懂；资源包 README 脱离文章也能指导组装和运行，但代码仍需满足其中列出的前置模块。
- 文章里的命令形如 `python -m exp_perf.perf_bench ...`，都是在累积后的工作目录根下执行的。

## 通用脚本约定

所有脚本遵守：

- **幂等**：重复运行不报错、不产生副作用。
- **非交互**：不弹确认提示，可被 agent 或 CI 直接调用。
- **参数化**：项目目录、Python 版本、镜像源、依赖包通过参数或环境变量注入（参数名一律英文）。
- **自验证**：每步给出"预期输出"与"验证命令"，运行结束打印明确的成功标志。
