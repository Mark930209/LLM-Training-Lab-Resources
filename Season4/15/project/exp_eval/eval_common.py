"""eval_common.py —— 15 篇 Eval Harness Lab 的公共件。

设计要点（服务于"分数可复算"这个唯一判据）：

1. 评测集是自建的小型基准：每条样本可人工核对，任务类型明确。
   不追公开榜单——公开数字标 REFERENCE，自建子集才是 REAL。

2. 两种判分方式都实现：
   - likelihood：对每个候选答案算 logprob，取最高者为预测。
     不需要生成，快，但只适合有固定选项的任务。
   - generation：让模型生成，再字符串匹配判对错。
     更接近"模型会不会"，但受采样参数影响，必须固定 seed。

3. few-shot 口径显式化：k=0/1/2，示例从评测集外抽取，
   每个任务的示例固定（写进报告），不允许运行时随机。

4. 多 seed 方差：同一 checkpoint 重复评测 n 次（generation 判分下
   采样 seed 不同），报告均值 ± 标准差。likelihood 判分是确定性的，
   方差应为 0——这本身是数据点。

5. 污染扫描：n-gram 重叠。评测样本与训练语料做 8-gram 重叠比对，
   命中即标记。清洗前后分数差 = 污染的贡献。
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import torch


# ---------------------------------------------------------------- 评测集

# 任务类型：每类样本可人工逐条核对
# 格式：{"task": ..., "prompt": ..., "options": [...], "answer": idx}
# generation 任务：{"task": ..., "prompt": ..., "expected": str}

def build_mini_eval() -> list[dict]:
    """自建小型评测集：分类、补全、算术三类，共 60 条。

    全部手工编写，可逐条核对；不来自任何公开榜单，避免许可与版本问题。
    """
    samples = []

    # --- 分类（20 条）：判断句子的类别，likelihood 判分 ---
    categories = [
        ("红楼梦里贾宝玉的身份是", ["公子", "农夫", "将军", "商人"], 0),
        ("水浒传中林冲的绰号是", ["豹子头", "及时雨", "智多星", "黑旋风"], 0),
        ("三国演义里诸葛亮的职务是", ["军师", "武将", "皇帝", "商人"], 0),
        ("西游记中孙悟空的本领是", ["七十二变", "医术", "算命", "烹饪"], 0),
        ("红楼梦里大观园的位置在", ["京城", "江南", "塞北", "岭南"], 0),
        ("水浒传中武松打死的动物是", ["老虎", "狮子", "熊", "狼"], 0),
        ("三国演义里关羽的兵器是", ["青龙偃月刀", "丈八蛇矛", "方天画戟", "双股剑"], 0),
        ("西游记里唐僧的徒弟数量是", ["三个", "两个", "四个", "五个"], 0),
        ("红楼梦里林黛玉的性格是", ["多愁善感", "豪爽", "狡诈", "迟钝"], 0),
        ("水浒传中鲁智深出家前的职业是", ["提辖", "教书", "务农", "行医"], 0),
        ("三国演义里曹操的身份是", ["丞相", "皇帝", "平民", "书生"], 0),
        ("西游记里猪八戒的兵器是", ["九齿钉耙", "金箍棒", "月牙铲", "大刀"], 0),
        ("红楼梦里薛宝钗的性格是", ["稳重", "急躁", "孤僻", "懒惰"], 0),
        ("水浒传中宋江的绰号是", ["及时雨", "豹子头", "行者", "神行太保"], 0),
        ("三国演义里张飞的字是", ["翼德", "云长", "孟德", "玄德"], 0),
        ("西游记里沙僧的原型是", ["卷帘大将", "天蓬元帅", "齐天大圣", "二郎神"], 0),
        ("红楼梦里王熙凤管理的是", ["荣国府", "皇宫", "学堂", "寺庙"], 0),
        ("水浒传中智取生辰纲的主谋是", ["吴用", "林冲", "武松", "李逵"], 0),
        ("三国演义里赤壁之战的胜者是", ["孙刘联军", "曹军", "袁绍", "董卓"], 0),
        ("西游记里白龙马的原型是", ["西海龙王之子", "东海龙王之子", "天马", "妖怪"], 0),
    ]
    for prompt, options, ans in categories:
        samples.append({"task": "classification", "prompt": prompt,
                        "options": options, "answer": ans})

    # --- 补全（20 条）：名著名句接龙，generation 判分 ---
    completions = [
        ("满纸荒唐言，", "一把辛酸泪"),
        ("都云作者痴，", "谁解其中味"),
        ("话说天下大势，", "分久必合"),
        ("滚滚长江东逝水，", "浪花淘尽英雄"),
        ("花谢花飞飞满天，", "红消香断有谁怜"),
        ("俺梁山泊好汉，", "替天行道"),
        ("身长八尺，", "豹头环眼"),
        ("皇帝轮流做，", "明年到我家"),
        ("宁教我负天下人，", "休教天下人负我"),
        ("既生瑜，", "何生亮"),
        ("三个臭皮匠，", "顶个诸葛亮"),
        ("周瑜打黄盖，", "一个愿打一个愿挨"),
        ("刘姥姥进大观园，", "眼花缭乱"),
        ("道高一尺，", "魔高一丈"),
        ("千里搭长棚，", "没有不散的筵席"),
        ("机关算尽太聪明，", "反误了卿卿性命"),
        ("世事洞明皆学问，", "人情练达即文章"),
        ("假作真时真亦假，", "无为有处有还无"),
        ("好风凭借力，", "送我上青云"),
        ("玉带林中挂，", "金簪雪里埋"),
    ]
    for prompt, expected in completions:
        samples.append({"task": "completion", "prompt": prompt, "expected": expected})

    # --- 算术（20 条）：两位数加法，generation 判分 ---
    import random
    rng = random.Random(1234)
    for _ in range(20):
        a = rng.randint(10, 89)
        b = rng.randint(10, 99 - a if a < 50 else 10)
        samples.append({"task": "arithmetic",
                        "prompt": f"{a} + {b} = ",
                        "expected": str(a + b)})

    return samples


# ---------------------------------------------------------------- 判分

@torch.no_grad()
def score_likelihood(model, tokenizer, sample: dict, device: str) -> int:
    """likelihood 判分：对每个选项算 P(option | prompt)，取最高。

    确定性：无采样，同 checkpoint 同输入必同结果。
    只适用于 classification（有固定选项）。
    """
    prompt_ids = tokenizer.encode(sample["prompt"])
    best_idx, best_lp = None, -math.inf
    for i, opt in enumerate(sample["options"]):
        opt_ids = tokenizer.encode(opt)
        ids = torch.tensor([prompt_ids + opt_ids], device=device)
        if ids.shape[1] < 2:
            continue
        logits = model(ids)
        logits = logits.logits if hasattr(logits, "logits") else logits
        # 只对 option 部分算 logprob
        log_probs = torch.log_softmax(logits[0, :-1].float(), dim=-1)
        tgt = ids[0, 1:]
        lp = log_probs[torch.arange(len(tgt)), tgt].sum().item()
        lp /= max(1, len(opt_ids))  # 长度归一：否则长选项天然吃亏
        if lp > best_lp:
            best_lp, best_idx = lp, i
    return 1 if best_idx == sample["answer"] else 0


@torch.no_grad()
def score_generation(model, tokenizer, sample: dict, device: str,
                     max_new: int = 16, seed: int = 42) -> int:
    """generation 判分：生成后字符串匹配。采样 seed 必须固定。

    SuperMiniGPT.generate 只有 temperature/top_k 采样参数（03 篇实现），
    与 HF generate 的 do_sample/top_p 不同——判分口径必须写清用的是哪个。
    """
    torch.manual_seed(seed)
    ids = torch.tensor([tokenizer.encode(sample["prompt"])], device=device)
    out = model.generate(ids, max_new_tokens=max_new, temperature=0.8, top_k=20)
    text = tokenizer.decode(out[0][ids.shape[1]:].tolist())
    expected = sample.get("expected", "")
    if not expected:
        return 0
    return 1 if expected[:4] in text else 0


# ---------------------------------------------------------------- 污染扫描

def ngram_contamination(samples: list[dict], corpus_text: str, n: int = 8) -> dict:
    """n-gram 重叠扫描：评测样本的 prompt 是否出现在训练语料里。

    中文按字符切 n-gram（本系列 char 分词）。
    返回每条样本的命中情况与总体命中率。
    """
    corpus_ngrams = set()
    for i in range(len(corpus_text) - n + 1):
        corpus_ngrams.add(corpus_text[i:i + n])

    hits = []
    for idx, s in enumerate(samples):
        prompt = s["prompt"]
        p_ngrams = [prompt[i:i + n] for i in range(max(1, len(prompt) - n + 1))]
        hit_count = sum(1 for g in p_ngrams if g in corpus_ngrams)
        hits.append({"idx": idx, "task": s["task"], "prompt": prompt,
                     "hit_ngrams": hit_count, "total_ngrams": len(p_ngrams),
                     "contaminated": hit_count > 0})
    rate = sum(1 for h in hits if h["contaminated"]) / len(hits)
    return {"n": n, "contamination_rate": rate, "samples": hits}


# ---------------------------------------------------------------- 报告

def write_report(path: str, payload: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"written {p}")
