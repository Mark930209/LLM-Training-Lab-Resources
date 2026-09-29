"""synthetic.py —— 去重规模扫描用的合成语料生成器。

真实采样子集只有几十到几百条文档，看不出去重的超线性代价；TB 级又跑不动。
按提纲约定，用合成分布在单机上复现两种行为（结果标 SIMULATED）：

- uniform 分布：每条文档由确定性随机字符序列生成，shingle 集合互不相同，
  两两 Jaccard 很低，桶内文档数接近均匀——这是"没有倾斜"的基线。
- skewed 分布：一部分文档是同一基底的近重复（基底 + 极小随机后缀），模拟
  整站雷同的登录页/分类页/错误页/SEO 模板页。这批文档两两 Jaccard 很高，
  签名在几乎所有 band 上都相同，挤进同几个桶。注意：近重复会被早判重、
  不插入 LSH，桶里只剩代表，验证次数仍线性——它展示的是桶占用倾斜本身。
- moderate 分布：每条文档 = 共享基底（约占 shared_frac）+ 各不相同的随机
  正文，两两 Jaccard 落在去重阈值下方不远处（如 0.65~0.75 对阈值 0.8）。
  签名足够相似 → band 碰撞频繁（同桶）；又不够相似 → 不判重、全部保留。
  于是每条新文档都要与桶内所有已保留文档逐一验证，验证次数随规模平方
  增长——这是去重超线性最直接的形态，也是 TB 级去重超时的先兆。

为什么"共享一小段模板"造不出倾斜（实测教训）：LSH 按 band（连续 r 个哈希）
分桶，只有签名在某个 band 完全相同才同桶。118 字模板对 400 字文档，两两
Jaccard 只有 ~0.2，band 碰撞概率 J^r 趋近于零，实测 128 个 band 键全部
不同、验证次数为 0。要造碰撞，共享比例必须高到把 Jaccard 抬进阈值的
S 曲线敏感区（threshold 附近下方），而不是"沾一点边"。

生成器只用标准库 random（固定 seed），不联网、不读语料，结果可复现。
"""

from __future__ import annotations

import random

_POOL = ("数据管线去重分片抽取过滤语种识别训练模型评测语料网页正文模板导航清洗留存"
         "字符哈希签名桶倾斜内存吞吐并行边界口径对齐血缘清单规则阈值")


def _random_doc(rng: random.Random, doc_chars: int) -> str:
    return "".join(rng.choice(_POOL) for _ in range(doc_chars))


def make_corpus(n_docs: int, distribution: str, doc_chars: int = 600,
                skew_ratio: float = 0.3, shared_frac: float = 0.80,
                seed: int = 20260924) -> list[str]:
    """生成 n_docs 条合成文档。

    distribution:
      - "uniform" : 全部为确定性随机序列，两两 Jaccard 低，桶均匀。
      - "skewed"  : 前 skew_ratio 比例的文档是同一基底的近重复
                    （基底 doc_chars 字 + 末尾 8 字随机后缀），其余为随机序列。
                    近重复组两两 Jaccard ≈ 0.98，会在几乎所有 band 上同桶，
                    但会被早判重，验证次数仍线性。
      - "moderate": 每条文档 = 共享基底（doc_chars × shared_frac 字）+
                    各不相同的随机正文。两两 Jaccard ≈ s/(2-s)（s 为共享
                    shingle 占比），shared_frac=0.80 时约 0.67，落在阈值
                    0.8 下方但足够引发 band 碰撞。
    """
    rng = random.Random(seed)
    texts: list[str] = []
    n_skew = int(n_docs * skew_ratio) if distribution == "skewed" else 0
    base = _random_doc(rng, doc_chars)
    shared_len = int(doc_chars * shared_frac)
    shared_base = base[:shared_len]
    for i in range(n_docs):
        if distribution == "moderate":
            body = _random_doc(rng, max(20, doc_chars - shared_len))
            texts.append(shared_base + body)
        elif i < n_skew:
            # 近重复：同一基底 + 极小随机后缀
            suffix = "".join(rng.choice(_POOL) for _ in range(8))
            texts.append(base + suffix)
        else:
            texts.append(_random_doc(rng, doc_chars))
    return texts
