#!/usr/bin/env bash
# gen_qa.sh —— 生成模型答事实问题实测：看它怎么"编"
# 前置：train.py 已保存权重（runs/superminigpt_main_*/model.pt，xiyouji 语料）
set -euo pipefail
cd ~/llm-training-lab
source .venv/bin/activate
export PYTHONPATH=.

OUT=/mnt/d/DocProjects/LearnLLMFromDraft/results/Season1/03

python - <<PY
import json, sys, torch
sys.path.insert(0, ".")
from pathlib import Path
from exp_superminigpt.data import load_corpus
from exp_superminigpt.model import SuperMiniGPT

tok, _, _ = load_corpus(Path("exp_superminigpt/data"), 128, corpus="xiyouji")

# 找最新的西游记训练权重（runs 目录按时间戳命名）
ckpts = sorted(Path("runs").glob("*minigpt_main_*/model.pt"),
               key=lambda p: p.stat().st_mtime, reverse=True)
assert ckpts, "未找到 model.pt；先跑 train_xiyouji.sh"
ckpt = ckpts[0]
state = torch.load(ckpt, map_location="cuda", weights_only=False)
cfg = state["config"]
model = SuperMiniGPT(state["vocab_size"], cfg["hidden"], cfg["layers"],
                cfg["heads"], cfg["seq_len"]).to("cuda")
model.load_state_dict(state["model"])
model.eval()
print(f"已加载权重: {ckpt}")

QUESTIONS = [
    "话说唐僧师徒行至狮驼岭，但见那",
    "却说那金角大王拿着宝葫芦，",
    "孙悟空一个筋斗云，径直来到",
    "那妖怪现出本相，原来是",
    "只见那芭蕉扇",
]

results = []
for q in QUESTIONS:
    ids = torch.tensor([tok.encode(q)], dtype=torch.long, device="cuda")
    out = model.generate(ids, 160, temperature=0.8, top_k=40)
    text = tok.decode(out[0].tolist())
    results.append({"prompt": q, "continuation": text})
    print("=" * 60)
    print("问句开头:", q)
    print(text)

Path("$OUT").mkdir(parents=True, exist_ok=True)
Path("$OUT/gen_qa_samples.json").write_text(
    json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print("GEN_QA_DONE")
PY