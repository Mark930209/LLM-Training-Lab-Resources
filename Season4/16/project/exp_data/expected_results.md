# expected_results.md —— exp_data 预期输出范围

供 LabRunner 判读异常。数值随采样的 Common Crawl 分段不同而波动，下面给的是
量级参考，不是精确断言。首个增量 `sample_extract` 的实测基线见
`results/Season4/16/sample_extract_20260924.json`（REAL）。

## sample_extract（WARC/WET 采样 + 三种正文抽取对照）

配置：单分段 Range 取 40 MB、每流最多 200 条记录、三种抽取方法。

| 指标 | 预期范围 | 说明 |
|---|---|---|
| common_docs | 150~200 | WARC 与 WET 同时命中的记录数；受 40 MB 截断影响，通常略低于 200 |
| wet.overall_retention | 0.04~0.10 | WET 已抽取文本字符数 / 原始 HTML 字符数 |
| trafilatura.overall_retention | 0.02~0.06 | 主正文抽取，留存率最低（去 boilerplate 最彻底） |
| naive_visible.overall_retention | 0.06~0.12 | 朴素去标签，留存率最高（含模板/导航噪声） |
| trafilatura.empty_docs | 0~20 | 部分页面 trafilatura 抽不出主正文（纯导航/JS 页） |
| boilerplate_noise_ratio | 0.4~0.7 | (naive − trafilatura) / naive；越高说明模板噪声越多 |
| metadata.elapsed_sec | 30~120 | 含两次 HTTP Range 下载与 trafilatura 逐条抽取 |

## 判读要点

- 留存率排序应稳定为 `trafilatura < wet < naive_visible`。若 trafilatura 留存率
  反而高于 naive，说明抽取方法接错或 HTML 解码异常。
- `common_docs=0` 表示 WARC↔WET 记录 ID 不相交，通常是 WET 路径取错分段
  （见 `warc_sample.wet_segment_from_warc` 的注释：必须同分段目录，不能按
  分段号去 wet.paths.gz 匹配）。
- `boilerplate_noise_ratio` 接近 0 说明朴素法与 trafilatura 输出几乎一样，
  要么页面本身干净，要么 trafilatura 没有真正去 boilerplate。

## filter（抽取 → 质量过滤 → 语种识别，逐阶段留存率）

配置：单分段 40 MB / 200 条，抽取用 trafilatura，质量与语种阈值见 config.yaml。
实测基线见 `results/Season4/16/filter_stages_20260924.json`（REAL）。

| 指标 | 预期范围 | 说明 |
|---|---|---|
| stages.extract.n_out | 150~200 | trafilatura 抽不出主正文的页面计为 empty_docs |
| stages.quality.doc_retention | 0.5~0.85 | 质量过滤条级留存；reason_counts 给出各规则删掉多少 |
| stages.langid.doc_retention | 0.2~0.6 | 目标语种留存；随分段语种构成波动大 |
| overall.doc_retention | 0.15~0.45 | 三阶段串联后的整体条级留存 |
| metadata.elapsed_sec | 20~90 | 含下载、trafilatura 抽取、langdetect 逐条判定 |

## 判读要点（filter）

- 三阶段条数必须单调不增：`extract.n_out ≥ quality.n_out ≥ langid.n_kept`。
  若某阶段输出大于输入，说明阶段间数据没有正确传递（口径错位）。
- `langid.status_counts` 里 `other` 占比高是正常的——Common Crawl 是多语种混合，
  目标语种（en/zh）只占一部分；`uncertain` 是置信度不足的，单独计数不直接丢。
- `quality.reason_counts` 三项（too_short / high_symbol_ratio / high_repeat_line）
  应都有命中；若某项恒为 0，检查对应阈值是否过松。

## dedup（三阶段过滤 + 精确去重 + MinHash 三档阈值）

配置：接 filter 的输出（约 50 条文档），精确去重 + MinHash（num_perm=128，
阈值 0.5/0.8/0.95）。实测基线见 `results/Season4/16/dedup_stages_20260924.json`（REAL）。

| 指标 | 预期范围 | 说明 |
|---|---|---|
| dedup.exact.duplicates | 0~5 | 真实网页随机样本几乎无完全重复 |
| dedup.minhash_by_threshold.*.duplicates | 0~5 | 阈值越低判重越多；小样本通常全 0 |
| exact vs minhash 耗时比 | 1:100 以上 | 同等规模 MinHash 远贵于精确哈希，这是"去重超线性"的第一层证据 |

## dedup_scale（合成分布规模扫描，SIMULATED）

配置：uniform/skewed/moderate 三种分布 × 500/1000/2000 三档规模，
MinHash num_perm=128、阈值 0.8。实测基线见
`results/Season4/16/dedup_scale_simulated_20260924.json`（SIMULATED）。

| 指标 | 预期行为 | 说明 |
|---|---|---|
| uniform.verifications | 恒为 0 | 随机文档不同桶，无碰撞 |
| skewed.max_bucket | 随规模线性增长（150→298→594） | 近重复组挤进同桶，但被早判重 |
| skewed.verifications | ≈ 近重复条数，线性 | 判重后不入桶，验证不放大 |
| moderate.verifications | 规模翻倍 → 验证×4（4011→18091→72941） | 同桶但不判重，桶内两两验证的平方项 |
| moderate.max_bucket | 随规模线性增长（367→703→1422） | 共享基底使签名趋同 |
| moderate 误判重率 | <5% | MinHash 概率估计噪声把个别对推过阈值，属正常 |

## 判读要点（dedup_scale）

- **超线性的直接证据是 moderate 的 verifications 增长倍率**：500→1000→2000
  应为 1:4.5:18 量级（平方项）；若接近 1:2:4 说明共享基底比例不够、
  没有形成同桶碰撞，检查 shared_frac。
- skewed 与 moderate 的对照是文章的关键机制解释：同样"挤进大桶"，
  近重复（skewed）被早判重所以验证线性，非重复同桶（moderate）验证平方。
  桶大不是问题本身，桶内两两验证才是。
- 三种分布的 exact 耗时都应保持线性且远小于 MinHash。

## shard_tokenize（分片 + 流式读取 + 并行 tokenize + 边界审计）

配置：接三阶段过滤 + 精确去重的输出（约 50 条文档），parquet 分片、
char 级并行 tokenize、两种边界口径各审计 20 个窗口。实测基线见
`results/Season4/16/shard_tokenize_20260924.json`（REAL）。

| 指标 | 预期范围 | 说明 |
|---|---|---|
| shard_write.n_shards | 1~2 | 53 条文档按 rows_per_shard=1000 只够 1 片 |
| shard_read.docs_per_sec | >5000 | parquet 列存 + zstd，小分片读取很快 |
| tokenize.tokens_per_sec | >10^6 | char 级 tokenize 是查表操作，8 进程并行 |
| boundary_audits.True.verdict | ok | EOS 分隔时窗口边界可识别 |
| boundary_audits.False.verdict | boundary_mismatch | **失败案例**：无 EOS 拼接时跨文档窗口无边界标记 |

## 判读要点（shard_tokenize）

- `False` 口径必须报 `boundary_mismatch`，这是提纲要求的失败案例：
  文档边界口径与训练端假设不一致时，loss 曲线完全正常，只有边界审计
  能暴露错位。若 `False` 也报 ok，检查 pack_stream 是否漏拼或窗口太短。
- `tokenize.vocab_size` 应等于采样文档的不同字符数 + 1（EOS）。
- 分片写入吞吐（write_mbps）远低于读取吞吐是正常的：写入含 zstd 压缩。

## full（全链路收口：三阶段 → 去重 → PII → 分片 → tokenize → 审计 → manifest）

实测基线见 `results/Season4/16/full_pipeline_20260924.json`（REAL）。

| 指标 | 预期范围 | 说明 |
|---|---|---|
| manifest.stages | 5 条 | extract/quality/langid/dedup_exact/pii 各一条，留存率单调不增 |
| pii.total_hits | 0~10 | 规则式检测（邮箱/手机号/身份证号），50 条样本命中个位数正常 |
| boundary_audit.verdict | ok | full 只用正确口径（EOS=True），必须通过 |
| manifest.config_sha256 | 64 位十六进制 | 配置即口径；改任何配置项哈希都会变 |
| metadata.elapsed_sec | 40~120 | 全链路含下载、抽取、过滤、去重、分片、tokenize |

## 判读要点（full）

- manifest.stages 的 n_in/n_out 必须首尾衔接（上一阶段 n_out = 下一阶段
  n_in）；衔接断了就是阶段间数据传递有 bug，这正是核心判断要防的口径错位。
- pii 命中样本的 context_before/after 窗口保存在 `pii_samples`，供人工
  核对误伤；规则式检测召回有限，命中 0 不代表没有 PII。
- `config_sha256` 变了而结果没重跑，说明结果与配置已经脱钩，数据作废。
