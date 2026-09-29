# DevResources/Season4/16 —— 从原始网页到可训练 token 的数据管线

对应文章：`Articles/Season4/16_怎样从原始网页做出训练语料：正文抽取、去重与分片存储.md`

## 本篇解决什么问题

17 篇的配方消融用的是已经清洗好的现成语料，真实情况往往是手里只有原始抓取。
本篇从 Common Crawl 的 WARC 出发，把"原始网页 → 带 manifest、可直接喂进
DataLoader 的 token 分片"这条链路的每道工序真实跑通，并量化每一阶段的留存率
与代价。

核心判断：数据管线的风险不在单点算法，而在阶段之间的口径对齐与规模的非线性。
去重是唯一随语料量超线性变贵的环节，其余工序基本线性；管线出错最常见的形态
不是某个过滤器写错，而是清洗端、tokenize 端与训练端对同一份数据的边界定义
不一致。

## 包内容

```text
16/
├── README.md                  # 本文件
└── project/
    └── exp_data/
        ├── __init__.py
        ├── config.yaml        # 全链路配置（source/extract/quality/dedup/shard/tokenize）
        ├── warc_sample.py     # WARC/WET 采样下载与流式解析
        ├── extract.py         # 正文抽取：WET 直用 vs HTML trafilatura 对照
        ├── boilerplate.py     # 导航/页脚/版权模板去除与高频 n-gram 暴露
        ├── quality.py         # 规则质量过滤（长度/符号率/重复行）
        ├── langid.py          # 语种识别与阈值
        ├── dedup.py           # 精确去重 + MinHash LSH 近重复去重（含签名验证）
        ├── synthetic.py       # 合成分布生成器（uniform/skewed/moderate，规模扫描用）
        ├── pii.py             # PII 检测与移除（邮箱/电话/证件号）
        ├── shard.py           # parquet / WebDataset 分片与流式读取
        ├── tokenize_par.py    # 并行 tokenize 与文档边界口径
        ├── manifest.py        # 各阶段留存率、配置哈希与来源清单
        ├── pipeline.py        # 统一入口：sample_extract / filter / dedup / dedup_scale
        ├── expected_results.md
        └── tests/
```

## 运行（本机 WSL，RTX 3070 节点；数据管线主体在 CPU）

```bash
cd ~/llm-training-lab
export PYTHONPATH="/mnt/d/DocProjects/LearnLLMFromDraft/DevResources/Season4/16/project"
PY="$HOME/llm-training-lab/.venv/bin/python"
CFG="/mnt/d/DocProjects/LearnLLMFromDraft/DevResources/Season4/16/project/exp_data/config.yaml"

# 增量 1：采样 + 三种正文抽取对照（REAL 留存率与 boilerplate 噪声比）
"$PY" -m exp_data.pipeline --mode sample_extract --config "$CFG" --out /tmp/sample_extract.json

# 增量 2：抽取 → 质量过滤 → 语种识别，逐阶段留存率（REAL）
"$PY" -m exp_data.pipeline --mode filter --config "$CFG" --out /tmp/filter.json

# 增量 3a：三阶段后接精确 + MinHash 去重（REAL，小样本）
"$PY" -m exp_data.pipeline --mode dedup --config "$CFG" --out /tmp/dedup.json

# 增量 3b：合成分布规模扫描，复现去重超线性与分桶倾斜（SIMULATED）
"$PY" -m exp_data.pipeline --mode dedup_scale --config "$CFG" --out /tmp/dedup_scale.json

# 增量 4：分片 + 并行 tokenize + 文档边界审计（REAL，含失败案例复现）
"$PY" -m exp_data.pipeline --mode shard_tokenize --config "$CFG" --out /tmp/shard_tokenize.json

# 增量 5：全链路收口（三阶段→去重→PII→分片→tokenize→审计→manifest 血缘）
"$PY" -m exp_data.pipeline --mode full --config "$CFG" --out /tmp/full.json
```

## 真实性标签

- 采样子集全链路：`REAL`
- 全量规模的耗时与留存率外推：`SCALED`
- TB 级去重分桶倾斜与内存压力（合成分布）：`SIMULATED`
- 工业级集群规模与处理时长：`REFERENCE`

## 依赖

`warcio`、`trafilatura`、`langdetect`、`datasketch`、`pyarrow`、`zstandard`
（已装入 `~/llm-training-lab/.venv`，uv 管理，清华源）。
