"""train_hf.py —— Mini Training Framework v1 的主循环（06 篇核心交付物）。

与 04 篇 train.py 的关系：循环骨架（AMP、梯度累积、调度、checkpoint、
吞吐统计）完全沿用，只把"模型从哪来"和"loss 怎么算"换成契约接口：

    04 篇：model = SuperMiniGPT(...)            # 模型与循环焊死
    06 篇：model = build_model(args.model, ...)  # 循环只认契约，不认具体类

loss 一律走 contract.contract_loss：换模型不换 loss 口径。
权重走 save_pretrained/from_pretrained；训练状态走 sidecar。

用法（在 WSL 项目根目录）：
    # 原模型 + char 词表（回归基线，应与 04 篇同配置结果一致）
    python -m exp_hf.train_hf --model superminigpt --tokenizer char --steps 300

    # 原模型 + HF 外壳（parity：logits 应与原模型逐位一致）
    python -m exp_hf.train_hf --model superminigpt-hf --tokenizer char --steps 300

    # 小 Llama + BPE（model swap 主实验）
    python -m exp_hf.train_hf --model llama --tokenizer bpe --steps 300

    # 标准格式保存/加载
    python -m exp_hf.train_hf --model llama --tokenizer bpe --steps 150 --save-hf /tmp/hf150
    python -m exp_hf.train_hf --model llama --tokenizer bpe --steps 300 --resume-hf /tmp/hf150

    # 故障注入
    python -m exp_hf.train_hf --model llama --tokenizer bpe --steps 300 --fault double_shift
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.benchmark import measure_peak_memory_mb  # noqa: E402
from common.reproducibility import set_all_seeds  # noqa: E402
from exp_hf.adapters import (  # noqa: E402
    SuperMiniGPTAdapter, SuperMiniGPTConfig, build_llama,
    load_training_state, save_training_state)
from exp_hf.contract import check_contract, contract_loss  # noqa: E402
from exp_hf.fail_modes_hf import (  # noqa: E402
    apply_double_shift, check_vocab_match, describe_fault)
from exp_hf.tokenize_bpe import load_fast_tokenizer, train_bpe  # noqa: E402
from exp_scale.data import CharTokenizer, LMDataset, load_corpus  # noqa: E402
from exp_scale.model import SuperMiniGPT  # noqa: E402
from exp_scale.schedulers import cosine_with_warmup  # noqa: E402


class TokenDataset(Dataset):
    """BPE 路径的数据集：整篇文本先编码成 id 序列，再滑窗切样本。

    与 04 篇 LMDataset 的区别只在"文本→id"这一步由谁做：
    char 路径在 load_corpus 里做，BPE 路径在这里做。切窗逻辑完全相同。
    """

    def __init__(self, ids: torch.Tensor, block_size: int):
        self.data = ids
        self.block_size = block_size

    def __len__(self) -> int:
        return max(0, len(self.data) - self.block_size - 1)

    def __getitem__(self, idx: int):
        chunk = self.data[idx: idx + self.block_size + 1]
        return chunk[:-1], chunk[1:]


def build_tokenizer(kind: str, corpus_text: str, data_dir: Path):
    """返回 (tokenizer, vocab_size)。char 走 04 篇，bpe 走本篇训练的 tokenizer。"""
    if kind == "char":
        tok = CharTokenizer(corpus_text)
        return tok, tok.vocab_size
    json_path = data_dir / "tokenizer_bpe.json"
    train_bpe(data_dir / "corpus_large.txt", json_path)
    fast = load_fast_tokenizer(json_path)
    return fast, fast.vocab_size


def encode_all(tok, kind: str, text: str) -> torch.Tensor:
    if kind == "char":
        return torch.tensor(tok.encode(text), dtype=torch.long)
    return torch.tensor(tok.encode(text, add_special_tokens=False), dtype=torch.long)


def build_model(name: str, vocab_size: int, seq_len: int, device: str):
    """循环只认契约：这里返回的三种模型对循环呈现同一接口。"""
    if name == "superminigpt":
        return SuperMiniGPT(vocab_size, 384, 6, 6, seq_len).to(device)
    if name == "superminigpt-hf":
        cfg = SuperMiniGPTConfig(vocab_size=vocab_size, hidden=384, layers=6,
                                 heads=6, seq_len=seq_len)
        return SuperMiniGPTAdapter(cfg).to(device)
    if name == "llama":
        return build_llama(vocab_size, hidden=384, layers=6, heads=6,
                           head_dim=64, intermediate=1024,
                           max_seq_len=max(seq_len, 512)).to(device)
    raise ValueError(f"未知模型: {name}（可选 superminigpt / superminigpt-hf / llama）")


@torch.no_grad()
def estimate_loss(model, loader, device, iters: int = 20, amp: bool = False) -> float:
    """验证集 loss：与训练完全同一个 loss 口径（contract_loss）。"""
    model.eval()
    losses = []
    for i, (x, y) in enumerate(loader):
        if i >= iters:
            break
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            losses.append(contract_loss(model(x), y).item())
    model.train()
    return sum(losses) / len(losses)


def run(args) -> dict:
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    set_all_seeds(args.seed, deterministic=False)

    data_dir = Path(args.data_dir)
    corpus_text = (data_dir / "corpus_large.txt").read_text(encoding="utf-8")
    tok, vocab_size = build_tokenizer(args.tokenizer, corpus_text, data_dir)

    ids = encode_all(tok, args.tokenizer, corpus_text)
    n = len(ids)
    cut = int(n * 0.9)
    train_ds = TokenDataset(ids[:cut], args.seq_len)
    val_ds = TokenDataset(ids[cut:], args.seq_len)
    bs = args.batch_size
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            drop_last=True, num_workers=0)

    model = build_model(args.model, vocab_size, args.seq_len, device)
    # 裸 SuperMiniGPT 是参考实现：只查前向与 loss 口径；
    # HF 路径（adapter / llama）才要求序列化与 config 契约
    contract_problems = check_contract(model, vocab_size,
                                       require_hf=args.model != "superminigpt")
    if contract_problems and args.fault != "vocab_mismatch":
        raise RuntimeError(f"契约体检不通过: {contract_problems}")

    # vocab_mismatch：tokenizer 与模型词表不配对，两个方向都测
    if args.fault == "vocab_mismatch":
        result = {"fault": "vocab_mismatch"}

        # 方向一（显性）：BPE 的 id（最大 8191）喂 char 词表模型（6015 行）
        # 在 CPU 上触发：CUDA 的 gather 越界是 device-side assert，
        # 异常虽能捕获但会污染整个 CUDA 上下文，后续实验全废
        char_model_cpu = SuperMiniGPT(6015, 384, 6, 6, args.seq_len)
        probe_ds = TokenDataset(ids[: bs * (args.seq_len + 1) * 2], args.seq_len)
        probe = torch.stack([probe_ds[i][0] for i in range(bs)])
        err = check_vocab_match(int(probe.max()), 6015)
        result["bpe_ids_to_char_model"] = {"vocab_check": err}
        if err is not None:
            try:
                with torch.no_grad():
                    char_model_cpu(probe)
                result["bpe_ids_to_char_model"]["raised"] = None
            except Exception as exc:  # noqa: BLE001
                result["bpe_ids_to_char_model"]["raised"] = \
                    f"{type(exc).__name__}: {exc}"

        # 方向二（静默）：char 的 id（最大 6014）喂 BPE 词表模型（8192 行）
        # id 全部在范围内，不报错，但每个 id 指向的是另一个 token
        if args.tokenizer == "bpe":
            char_tok = CharTokenizer(corpus_text)
            char_ids = encode_all(char_tok, "char", corpus_text)
            cds = TokenDataset(char_ids[: bs * (args.seq_len + 1) * 2], args.seq_len)
            cx = torch.stack([cds[i][0] for i in range(bs)]).to(device)
            cy = torch.stack([cds[i][1] for i in range(bs)]).to(device)
            bpe_model = build_llama(vocab_size, hidden=384, layers=6, heads=6,
                                    head_dim=64, intermediate=1024,
                                    max_seq_len=max(args.seq_len, 512)).to(device)
            err2 = check_vocab_match(int(cx.max()), vocab_size)
            with torch.no_grad():
                loss2 = contract_loss(bpe_model(cx), cy).item()
            result["char_ids_to_bpe_model"] = {
                "vocab_check": err2,
                "silent_mismatch": err2 is None,
                "loss_with_wrong_vocab": round(loss2, 4),
            }

        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
        return result

    scaler = torch.amp.GradScaler("cuda", enabled=args.amp) if device == "cuda" else None

    start_step, best_val = 0, float("inf")
    resumed_from_weights_only = False
    if args.resume_hf:
        if args.model == "superminigpt":
            raise ValueError("--resume-hf 只支持 HF 模型（superminigpt-hf / llama）")
        hf_dir = Path(args.resume_hf)
        model = model.__class__.from_pretrained(hf_dir).to(device)
        sidecar = hf_dir / "training_state.pt"
        if args.fault == "resume_weights_only":
            # 故障：只拿权重，优化器动量与学习率进度从零开始
            resumed_from_weights_only = True
        elif sidecar.exists():
            opt_tmp = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                        weight_decay=0.1, betas=(0.9, 0.95))
            start_step, best_val = load_training_state(sidecar, opt_tmp, scaler, device)
            del opt_tmp
    # opt 必须在 from_pretrained 之后建：它绑定的是最终模型的参数
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=0.1, betas=(0.9, 0.95))
    if args.resume_hf and not resumed_from_weights_only:
        sidecar = Path(args.resume_hf) / "training_state.pt"
        if sidecar.exists():
            start_step, best_val = load_training_state(sidecar, opt, scaler, device)

    history = {"train_loss": [], "val_loss": [], "step_ms": []}
    step = start_step
    micro = 0
    running_loss, loss_count = 0.0, 0
    t0 = time.perf_counter()
    nan_at = None

    while step < args.steps:
        for x, y in train_loader:
            if step >= args.steps:
                break
            x, y = x.to(device), y.to(device)

            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=args.amp):
                if args.fault == "double_shift":
                    # 输入不变，target 同长度 roll 一位：
                    # 训练任务被换成"跳一个字预测"，验证仍按 t+1 量
                    x_in, y_tgt = apply_double_shift(x, y)
                    loss = contract_loss(model(x_in), y_tgt)
                else:
                    loss = contract_loss(model(x), y)
                loss = loss / args.accum
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            running_loss += loss.item() * args.accum
            loss_count += 1
            micro += 1
            if micro < args.accum:
                continue
            micro = 0

            # AMP 下必须先 unscale 再 clip，grad_norm 才是真实量级
            if scaler is not None:
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            lr_now = cosine_with_warmup(step, args.steps, args.warmup, args.lr)
            for g in opt.param_groups:
                g["lr"] = lr_now
            if scaler is not None:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            opt.zero_grad(set_to_none=True)

            if step % args.log_every == 0:
                avg = running_loss / max(loss_count, 1)
                running_loss, loss_count = 0.0, 0
                history["train_loss"].append({"step": step, "loss": round(avg, 4)})
                if not torch.isfinite(torch.tensor(avg)):
                    nan_at = step
                    break
            step += 1
        if nan_at is not None:
            break

        if step % args.eval_every == 0 or step >= args.steps:
            vl = estimate_loss(model, val_loader, device, amp=args.amp)
            history["val_loss"].append({"step": step, "loss": round(vl, 4)})
            best_val = min(best_val, vl)

    wall = time.perf_counter() - t0
    steps_done = step - start_step
    result = {
        "model": args.model,
        "tokenizer": args.tokenizer,
        "fault": args.fault,
        "vocab_size": vocab_size,
        "steps": steps_done,
        "start_step": start_step,
        "wall_time_s": round(wall, 2),
        "step_ms": round(wall * 1000 / max(steps_done, 1), 2),
        "final_train_loss": history["train_loss"][-1]["loss"] if history["train_loss"] else None,
        "final_val_loss": history["val_loss"][-1]["loss"] if history["val_loss"] else None,
        "best_val_loss": round(best_val, 4) if best_val != float("inf") else None,
        "nan_at_step": nan_at,
        "contract_problems": contract_problems,
        "resumed_from_weights_only": resumed_from_weights_only,
        "params": sum(p.numel() for p in model.parameters()),
    }
    if device == "cuda":
        result["peak_mem_mb"] = round(measure_peak_memory_mb(), 1)

    if args.save_hf:
        hf_dir = Path(args.save_hf)
        hf_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(hf_dir)
        save_training_state(hf_dir / "training_state.pt", opt, scaler,
                            step, best_val)
        result["saved_hf_dir"] = str(hf_dir)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="superminigpt",
                    choices=["superminigpt", "superminigpt-hf", "llama"])
    ap.add_argument("--tokenizer", default="char", choices=["char", "bpe"])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fault", default=None,
                    choices=["double_shift", "vocab_mismatch", "resume_weights_only"])
    ap.add_argument("--save-hf", default=None)
    ap.add_argument("--resume-hf", default=None)
    ap.add_argument("--data-dir", default="exp_scale/data")
    ap.add_argument("--output", default="/tmp/train_hf.json")
    args = ap.parse_args()
    out = run(args)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
