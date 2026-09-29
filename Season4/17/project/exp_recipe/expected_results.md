# expected_results.md —— exp_recipe 预期输出范围

供 LabRunner 判读异常。配方制备数据来自真实运行（REAL），训练困惑度随
seed 与硬件有小幅波动，下面给量级参考。

## prepare（配方制备，不训练）

配置：两域语料（四大名著 + Python 3.12 标准库），重复注入 10% 精确 + 10%
扰动，7 个配方。实测基线见 `results/Season4/17/prepare_summary_20260924.json`。

| 指标 | 预期范围 | 说明 |
|---|---|---|
| fingerprint.nl_eval_chars | 200,000 | 小说域评测集固定 200K 字符 |
| fingerprint.code_eval_docs | 20~30 | 代码域每 20 个文件取 1 个 |
| inject nl/code total | 约 822 / 542 | 原始文档数 × 1.2（注入 20%） |
| 去重留存（nl，dedup 档） | none 781 > exact 719 > mh08 690 > mh05 652 | **必须单调**：去重越狠留存越低 |
| actual_nl_ratio | 与设定 nl_ratio 差 <0.05 | 配比按字符预算控制 |
| unique_4gram_ratio | 0.2~0.7 | 代码域重的配方更低（代码语法重复度高） |

## 判读要点（prepare）

- **去重留存必须单调不增**：none ≥ exact ≥ minhash08 ≥ minhash05。若某档
  留存反而更高，说明去重模式接错或阈值解析错。
- **unique_4gram 随去重增强而升高是正常且反直觉的**：去掉重复副本后，
  剩余文本的 unique n-gram 比例上升（冗余被删）。它是"冗余度"代理而不是
  "覆盖度"代理，不能直接读成"多样性变好"。代码域配比越高，u4 越低
  （代码的 4-gram 重复度天然高于自然语言）。
- minhash05（过激档）的留存率应明显低于 mh08，这是失败案例一的数据基础。

## matrix（训练矩阵，主实验）

配置：10M 档 SuperMiniGPT（hidden 384 / 6 层），固定 2M token 预算
（约 488 步），固定 seed，两域留出困惑度。实测基线见
`results/Season4/17/matrix_results_20260924.json`（REAL）。

| 指标 | 预期范围 | 说明 |
|---|---|---|
| n_params | 约 12.9M | 10M 档（与 04 篇 10M 同构，词表不同） |
| total_steps | 约 488 | 2M token / (16 batch × 256 seq) |
| train_sec（每配方） | 32~35 | RTX 3070，10M 模型 488 步（实测基线） |
| peak_memory_mib | 约 1459 | 10M 模型 + batch16×seq256，远低于 8 GB |
| nl_eval.ppl | 157.89~317.97 | 小说域困惑度，nl 配比越高越低（nl80 最低 157.89，code80 最高 317.97） |
| code_eval.ppl | 4.27~8.38 | 代码域困惑度，code 配比越高越低（code80 最低 4.27，nl80 最高 8.38） |

## 判读要点（matrix）

- **配比的主效应**：nl80 配方的 nl_eval 困惑度应低于 code80 配方，反之
  code_eval 困惑度 code80 更低。这是"配比决定能力分布"的直接证据。
- **不是简单叠加**：nl50 的两域困惑度通常都介于 nl80 与 code80 之间，
  但不会正好是两者的平均，存在配比的非线性。
- **去重过激的代价（实测未触发）**：预设 mh05 困惑度会反弹，实测 nl 困惑度
  201.50 仍在去重轴 4.6% 窄带内，没有反弹。原因是 token 预算固定、删的是
  注入副本，有效内容差异小。这是如实记录的负结果，不是 bug。
- 所有配方的 peak_memory 与 train_sec 应接近（同模型同预算），差异大
  说明 token 预算或 batch 没对齐，对照不公平。
- 评测集指纹（nl_eval_sha256 / code_eval_sha256）在所有 run 中必须一致，
  否则评的不是同一份数据。
