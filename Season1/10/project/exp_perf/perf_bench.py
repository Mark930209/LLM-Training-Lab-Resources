"""perf_bench.py —— 单卡端到端性能 bench 与故障注入（10 篇核心交付物）。

回答一个问题：模型放得下、attention 也换了融合 kernel，GPU 为什么还是吃不满。

三种模式：
    baseline   固定模型/数据/token 预算，记录 step time 分布与阶段拆分
    inject     逐项注入瓶颈（慢 DataLoader、非 pinned、同步日志、频繁 eval/ckpt），
               看每一项在时间线上留下什么证据
    optimize   逐项打开优化（workers、pin_memory、prefetch、persistent、
               non_blocking H2D、fused optimizer、torch.compile、SDPA backend），
               测端到端收益，与 microbenchmark 收益对照

关键口径：
    wall_ms      一步墙上时间
    gpu_busy_ms  profiler 里 CUDA kernel 的 self device time 之和
    gap_ms       wall - gpu_busy，GPU 空洞。这是"吃不满"的量化定义
    阶段拆分     data / h2d / forward / backward / optimizer / periodic

用法（WSL 项目根目录）：
    python -m exp_perf.perf_bench --mode baseline --steps 60
    python -m exp_perf.perf_bench --mode inject --fault slow_data --slow-data-ms 8
    python -m exp_perf.perf_bench --mode optimize --opts workers,pin,prefetch
    python -m exp_perf.perf_bench --mode optimize --opts compile
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.reproducibility import set_all_seeds  # noqa: E402
from exp_hf.contract import contract_loss  # noqa: E402
from exp_perf.perf_harness import (  # noqa: E402
    PhaseTimer,
    classify_top_kernels_from_trace,
    export_trace_once,
    get,
    gpu_busy_ms_from_trace,
    mem_snapshot,
    summarize,
)

# 优化开关全集
ALL_OPTS = (
    "workers",       # DataLoader num_workers > 0
    "pin",           # pin_memory=True
    "prefetch",      # prefetch_factor 提高
    "persistent",    # persistent_workers=True
    "nonblock",      # H2D 用 non_blocking=True
    "fused",         # AdamW(fused=True)
    "compile",       # torch.compile 模型
    "sdpa",          # attn_implementation="sdpa"（09 篇结论）
    "setnone",       # zero_grad(set_to_none=True)
    "accum",         # 梯度累积（08 篇的省显存手段，本篇看它对吞吐的影响）
    "ckpt",          # activation checkpointing（同上）
)

# 08 篇已证 accum/ckpt 能省显存。本篇要问的是另一件事：
# 当显存不是约束时，它们对吞吐是正收益还是负收益。
# ckpt 用重算换激活，显存宽裕时重算就是纯亏损——这是本篇要的
# "microbenchmark 有效、端到端收益很小或为负"的失败案例载体。


# ---------------------------------------------------------------- 数据管道

def _calibrate_cpu_work(target_ms: float) -> dict:
    """标定一份目标量级的 CPU 负载（连续三次 softmax），返回缓冲与实测耗时。

    为什么必须标定（本篇踩过的坑，两个故障注入都栽在这）：
      heavy_cpu 第一版把矩阵大小写死成窗口长度（257 个 float），
      三次 softmax 只花几十微秒，与想注入的"8 ms 昂贵预处理"差三个
      数量级，注入组与 baseline 测出来一模一样——注入等于没注入。
      sync_log 第一版只有 synchronize 加一次 .item()，也测不出差异：
      本 harness 的阶段打点在每个阶段边界与每步末都已经 synchronize，
      额外一次同步是免费的。同步日志的真实代价是它后面跟着的 CPU
      串行工作（格式化、写盘期间 CPU 没法给 GPU 发 kernel）。
    故障注入必须先证明注入的量级就是想要的量级，所以构造时实测一次，
    把实测毫秒数写进结果文件，让读者能核对注入是否真的生效。
    """
    best: dict | None = None
    for n in (512, 1024, 2048, 4096):
        buf = torch.randn(n, n)
        for _ in range(2):
            c = buf
            for _ in range(3):
                c = torch.softmax(c, dim=-1)
        t0 = time.perf_counter()
        for _ in range(5):
            c = buf
            for _ in range(3):
                c = torch.softmax(c, dim=-1)
        ms = (time.perf_counter() - t0) / 5 * 1000
        if best is None or abs(ms - target_ms) < abs(best["measured_ms"] - target_ms):
            best = {"n": n, "buf": buf, "measured_ms": round(ms, 3)}
    return best or {}


def _run_cpu_work(state: dict) -> None:
    """跑一份已标定的 CPU 负载。算完丢掉，不碰数据本身。"""
    c = state["buf"]
    for _ in range(3):
        c = torch.softmax(c, dim=-1)
    del c


class CorpusWindowDataset(Dataset):
    """从真实语料切窗口。支持注入慢预处理来模拟数据管道瓶颈。

    slow_ms > 0 时在 __getitem__ 里 sleep，模拟 tokenize/解码/IO 慢。
    这是"GPU 等数据"最直接的故障注入：sleep 发生在 worker 进程里，
    workers=0 时会直接卡住主进程，workers>0 时可以被并行掩盖。
    heavy_cpu 则注入真占 CPU 的计算负载（标定量级），与 sleep 的区别是
    它吃掉的是 CPU 算力：workers=0 时卡主进程，workers>0 时可以被
    多 worker 分摊掩盖。
    """

    def __init__(self, ids: torch.Tensor, seq_len: int, n_samples: int,
                 seed: int = 1234, slow_ms: float = 0.0,
                 heavy_cpu: bool = False, heavy_ms: float = 8.0,
                 vocab_size: int | None = None):
        self.ids = ids
        self.seq_len = seq_len
        self.slow_ms = slow_ms
        self.heavy_cpu = heavy_cpu
        self.vocab_size = vocab_size
        # 标定量级：注入的每样本 CPU 耗时以实测值为准，写进结果文件
        self.heavy_state = _calibrate_cpu_work(heavy_ms) if heavy_cpu else None
        g = torch.Generator().manual_seed(seed)
        self.offsets = torch.randint(
            0, max(1, len(ids) - seq_len - 1), (n_samples,), generator=g).tolist()

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, idx: int) -> torch.Tensor:
        o = self.offsets[idx]
        window = self.ids[o:o + self.seq_len + 1]
        if self.slow_ms > 0:
            time.sleep(self.slow_ms / 1000.0)
        if self.heavy_state is not None:
            # 纯 CPU 计算负载，模拟昂贵的预处理（不是 sleep，是真占 CPU）。
            # 算完丢掉，返回原始合法 window。
            # 第一版把 softmax 结果 *1e4 再 .long() 当 token id 返回，
            # 值最大到 10000 而 char 词表只有约 6015，越界直接 IndexError——
            # 故障注入把数据本身污染了，测的就不是"CPU 慢"而是"数据错"。
            _run_cpu_work(self.heavy_state)
        return window


def load_corpus_ids(data_dir: str) -> tuple[torch.Tensor, int]:
    """加载真实语料并编码。沿用 08 篇 parity 的做法。"""
    from exp_scale.data import CharTokenizer

    corpus_path = Path(data_dir) / "corpus_large.txt"
    text = corpus_path.read_text(encoding="utf-8")
    tok = CharTokenizer(text)
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    return ids, tok.vocab_size


def make_loader(ids: torch.Tensor, seq_len: int, batch: int, n_samples: int,
                opts: set, fault: str, args, vocab_size: int | None = None):
    """按优化开关与故障注入构造 DataLoader。

    返回 (loader, inject_info)。inject_info 记录本次注入的实测量级，
    写进结果文件，让读者能核对注入是否真的生效：故障注入自己没量级，
    测出来的"对照"就是假的（heavy_cpu 第一版栽在这）。
    """
    slow_ms = args.slow_data_ms if fault == "slow_data" else 0.0
    heavy = fault == "heavy_cpu"
    ds = CorpusWindowDataset(ids, seq_len, n_samples, slow_ms=slow_ms,
                             heavy_cpu=heavy, heavy_ms=args.slow_data_ms,
                             vocab_size=vocab_size)
    inject_info = {"fault": fault}
    if fault == "slow_data":
        inject_info["sleep_ms_per_sample"] = slow_ms
    if heavy and ds.heavy_state:
        inject_info["cpu_work_matrix_n"] = ds.heavy_state["n"]
        inject_info["cpu_work_measured_ms"] = ds.heavy_state["measured_ms"]

    use_workers = "workers" in opts
    nw = args.workers if use_workers else 0
    kw = {
        "batch_size": batch,
        "num_workers": nw,
        "shuffle": False,
        "drop_last": True,
    }
    # pin_memory：默认关，opts 里有 pin 才开。故障 no_pin 显式关掉。
    kw["pin_memory"] = ("pin" in opts) and fault != "no_pin"
    if nw > 0:
        kw["persistent_workers"] = "persistent" in opts
        kw["prefetch_factor"] = args.prefetch if "prefetch" in opts else 2
    return DataLoader(ds, **kw), inject_info


def build_model(vocab: int, args, attn_impl: str, device: str, use_ckpt: bool = False):
    """自己构造 Llama，不用 exp_hf.build_llama。

    原因：attn_implementation 必须在 LlamaConfig 里给，模型构造时才会按它
    选 attention 类。先构造再改 config._attn_implementation 不会重建已经
    建好的 attention 模块，开关会静默失效——09 篇刚讲过这类错误，
    本篇的工具不能自己犯。构造后还会回读验证。
    """
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=vocab,
        hidden_size=args.hidden,
        intermediate_size=args.hidden * 8 // 3,
        num_hidden_layers=args.layers,
        num_attention_heads=args.heads,
        num_key_value_heads=args.heads,
        head_dim=args.head_dim,
        max_position_embeddings=max(args.seq, 512),
        tie_word_embeddings=True,
        use_cache=False,
        attn_implementation=attn_impl,
    )
    model = LlamaForCausalLM(cfg).to(device)

    # 验证开关真的生效：回读 config 与子模块的实际类型
    reported = getattr(model.config, "_attn_implementation", None)
    attn_cls = ""
    for m in model.modules():
        if type(m).__name__.endswith("Attention"):
            attn_cls = type(m).__name__
            break

    if use_ckpt:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})

    return model, {"requested": attn_impl, "config_reported": reported,
                   "attn_module": attn_cls, "ckpt": use_ckpt}


# ---------------------------------------------------------------- 周期任务

def run_eval(model, val_batches, device, use_amp: bool) -> float:
    """一次完整验证。周期任务污染 step benchmark 的主要来源。"""
    model.eval()
    tot = 0.0
    with torch.no_grad():
        for x, y in val_batches:
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                tot += float(contract_loss(model(x), y).item())
    model.train()
    return tot / max(1, len(val_batches))


def run_checkpoint(model, opt, path: Path) -> int:
    """存一次 checkpoint，返回字节数。IO 型周期任务。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "opt": opt.state_dict()}, path)
    return path.stat().st_size


# ---------------------------------------------------------------- 主循环

def run(args) -> dict:
    opts = set() if args.opts == "none" else set(args.opts.split(","))
    unknown = opts - set(ALL_OPTS)
    if unknown:
        raise SystemExit(f"未知优化开关: {sorted(unknown)}，可选 {ALL_OPTS}")

    fault = args.fault
    set_all_seeds(args.seed, deterministic=False)
    device = "cuda"

    ids, vocab = load_corpus_ids(args.data_dir)
    n_train = args.steps * args.batch
    n_val = args.eval_batches * args.batch
    loader, inject_info = make_loader(ids, args.seq, args.batch, n_train, opts, fault, args,
                                      vocab_size=vocab)

    # 同步日志故障：标定一份 CPU 负载，模拟日志格式化/写盘的串行 CPU 工作。
    # 第一版只有 synchronize 加一次 .item()，在本 harness 里是免费的：
    # 阶段打点在每个阶段边界与每步末都已经 synchronize 过，额外一次
    # 同步不增加任何等待。同步日志的真实代价是它占住的 CPU 时间——
    # 那段时间 CPU 没法给 GPU 继续发 kernel。量级同样以实测为准。
    sync_state = None
    if fault == "sync_log":
        sync_state = _calibrate_cpu_work(args.sync_log_ms)
        inject_info["sync_cpu_work_matrix_n"] = sync_state["n"]
        inject_info["sync_cpu_work_measured_ms"] = sync_state["measured_ms"]

    # 验证集固定，保证不同组合的 val loss 可比
    g = torch.Generator().manual_seed(args.seed + 7)
    val_batches = []
    for _ in range(args.eval_batches):
        o = torch.randint(0, len(ids) - args.seq - 1, (args.batch,), generator=g)
        b = torch.stack([ids[i:i + args.seq + 1] for i in o.tolist()]).to(device)
        val_batches.append((b[:, :-1], b[:, 1:]))

    attn_impl = "sdpa" if "sdpa" in opts else "eager"
    use_ckpt = "ckpt" in opts
    accum = args.accum if "accum" in opts else 1
    model, attn_check = build_model(vocab, args, attn_impl, device, use_ckpt)
    if attn_check["config_reported"] != attn_impl:
        # 开关没生效就不要默默跑下去，那会得出错误的优化收益
        raise SystemExit(f"attn_implementation 未生效: {attn_check}")

    if "compile" in opts:
        t0 = time.perf_counter()
        model = torch.compile(model)
        args._compile_wrap_s = round(time.perf_counter() - t0, 3)

    fused = "fused" in opts
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95),
                            weight_decay=0.1, fused=fused)
    use_amp = not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    nonblock = "nonblock" in opts
    set_none = "setnone" in opts

    n_params = sum(p.numel() for p in model.parameters())
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model.train()

    records = []
    data_iter = iter(loader)
    compile_first_step_s = None
    t_run0 = time.perf_counter()

    for step in range(args.steps + args.warmup):
        rec_t = PhaseTimer(device, sync_each=not args.no_phase_sync)
        # 真实 wall：只在步末同步一次，保留异步重叠。
        # 阶段之和会略大于它，因为 PhaseTimer 每段都同步；
        # 两者并列上报，差值就是同步开销与重叠量。
        torch.cuda.synchronize()
        t_wall0 = time.perf_counter()

        rec_t.start("data")
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        rec_t.stop("data")

        rec_t.start("h2d")
        batch = batch.to(device, non_blocking=nonblock)
        x, y = batch[:, :-1], batch[:, 1:]
        rec_t.stop("h2d")

        # 同步日志：故障注入项，强制每步 synchronize 再做一段串行 CPU 工作
        # （模拟日志格式化/写盘占住的 CPU）。只有 synchronize 的版本在本
        # harness 里测不出差异，原因见上面 sync_state 的注释。
        if fault == "sync_log" and step % max(1, args.log_every) == 0:
            torch.cuda.synchronize()
            if sync_state is not None:
                _run_cpu_work(sync_state)

        rec_t.start("forward")
        opt.zero_grad(set_to_none=set_none)
        # accum > 1 时一个"step"含多次小前向，micro_batch 相应缩小，
        # 等效 batch 不变。08 篇证过它省显存，本篇看它对吞吐的影响。
        micro = max(1, x.shape[0] // accum)
        if accum == 1:
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                loss = contract_loss(model(x), y)
            rec_t.stop("forward")
            rec_t.start("backward")
            scaler.scale(loss).backward()
            rec_t.stop("backward")
        else:
            # 累积路径：每次 micro 的前向与反向分别打点，
            # 否则 backward 会被算进 forward 段（第一版的归属错误）。
            loss = torch.zeros((), device=device)
            for i in range(accum):
                sl = slice(i * micro, (i + 1) * micro)
                if sl.start >= x.shape[0]:
                    break
                with torch.autocast("cuda", dtype=torch.float16,
                                    enabled=use_amp):
                    part = contract_loss(model(x[sl]), y[sl]) / accum
                rec_t.stop("forward")
                rec_t.start("backward")
                scaler.scale(part).backward()
                rec_t.stop("backward")
                loss = loss + part.detach()
                if i + 1 < accum:
                    rec_t.start("forward")

        rec_t.start("optimizer")
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        rec_t.stop("optimizer")

        # 周期任务：eval / checkpoint，单独计时并标记该步
        periodic_tag = []
        rec_t.start("periodic")
        is_measured = step >= args.warmup
        if args.eval_every > 0 and is_measured and step % args.eval_every == 0:
            run_eval(model, val_batches, device, use_amp)
            periodic_tag.append("eval")
        if args.ckpt_every > 0 and is_measured and step % args.ckpt_every == 0:
            run_checkpoint(model, opt,
                           Path(args.ckpt_dir) / f"step{step}.pt")
            periodic_tag.append("ckpt")
        rec_t.stop("periodic")

        if is_measured:
            torch.cuda.synchronize()
            wall_true_ms = (time.perf_counter() - t_wall0) * 1000
            phases = rec_t.sync_and_collect()
            phase_sum = sum(phases.values())
            r = {
                "step": step - args.warmup,
                "wall_ms": round(wall_true_ms, 3),
                "phase_sum_ms": round(phase_sum, 3),
                "sync_overhead_ms": round(phase_sum - wall_true_ms, 3),
                "phases": phases,
                "loss": round(float(loss.item()), 4),
                "grad_norm": round(float(gn), 4),
                "periodic": ",".join(periodic_tag),
            }
            records.append(r)
            if "compile" in opts and compile_first_step_s is None:
                compile_first_step_s = wall_true_ms

    run_s = time.perf_counter() - t_run0

    walls = [r["wall_ms"] for r in records]
    phase_names = ["data", "h2d", "forward", "backward", "optimizer", "periodic"]
    phase_mean = {
        p: round(sum(r["phases"].get(p, 0.0) for r in records) / max(1, len(records)), 3)
        for p in phase_names
    }
    phase_total = {
        p: round(sum(r["phases"].get(p, 0.0) for r in records), 2)
        for p in phase_names
    }
    # 阶段归属用阶段之和做分母（不是 wall），否则占比加起来不等于 1
    phase_denom = max(sum(phase_total.values()), 1e-9)
    sync_overhead_ms = round(
        sum(r.get("sync_overhead_ms", 0.0) for r in records), 2)

    # 长尾步：周期任务命中的步与未命中的步分开统计，
    # 这是"平均值骗人"的直接证据
    periodic_walls = [r["wall_ms"] for r in records if r["periodic"]]
    clean_walls = [r["wall_ms"] for r in records if not r["periodic"]]

    toks_per_step = args.batch * args.seq
    mean_ms = sum(walls) / len(walls) if walls else float("nan")

    result = {
        "mode": args.mode,
        "opts": sorted(opts),
        "fault": fault,
        "inject": inject_info,
        "config": {
            "hidden": args.hidden, "layers": args.layers, "heads": args.heads,
            "head_dim": args.head_dim, "seq": args.seq, "batch": args.batch,
            "steps": args.steps, "warmup": args.warmup, "amp": use_amp,
            "attn_impl": attn_impl, "fused": fused, "set_none": set_none,
            "non_blocking": nonblock,
            "num_workers": args.workers if "workers" in opts else 0,
            "pin_memory": ("pin" in opts) and fault != "no_pin",
            "prefetch": args.prefetch if "prefetch" in opts else None,
            "persistent": "persistent" in opts,
            "accum": accum,
            "ckpt": use_ckpt,
            "slow_data_ms": args.slow_data_ms if fault == "slow_data" else 0.0,
            "eval_every": args.eval_every, "ckpt_every": args.ckpt_every,
        },
        "params_m": round(n_params / 1e6, 2),
        "attn_check": attn_check,
        "device": torch.cuda.get_device_name(0),
        "capability": "%d.%d" % torch.cuda.get_device_capability(0),
        "torch": torch.__version__,
        "step_time": summarize(walls),
        "phase_mean_ms": phase_mean,
        "phase_total_ms": phase_total,
        "phase_share": {
            p: round(phase_total[p] / phase_denom, 4) for p in phase_names
        },
        "phase_sync_overhead_ms": sync_overhead_ms,
        "phase_sync_note": (
            "阶段打点每段都 synchronize，所以 phase_total 之和大于 wall；"
            "差值即本项。要保留异步重叠看 wall_ms，要看阶段归属看 phase_share。"
            if not args.no_phase_sync else "已关闭阶段同步"),
        "periodic_steps": {
            "n": len(periodic_walls),
            "mean_ms": round(sum(periodic_walls) / len(periodic_walls), 2)
            if periodic_walls else None,
            "max_ms": round(max(periodic_walls), 2) if periodic_walls else None,
        },
        "clean_steps": {
            "n": len(clean_walls),
            "mean_ms": round(sum(clean_walls) / len(clean_walls), 2)
            if clean_walls else None,
            "max_ms": round(max(clean_walls), 2) if clean_walls else None,
        },
        "tok_per_s": round(toks_per_step / (mean_ms / 1000)) if mean_ms == mean_ms else 0,
        "total_run_s": round(run_s, 2),
        "peak_mb": mem_snapshot("end")["peak_mb"],
        "reserved_mb": mem_snapshot("end")["reserved_mb"],
        "final_loss": records[-1]["loss"] if records else None,
        "compile_first_step_ms": compile_first_step_s,
        "compile_wrap_s": getattr(args, "_compile_wrap_s", None),
    }

    # 溢出防护（09 篇教训）：reserved 逼近整卡时数据不可用于速度对比
    total_mb = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
    ratio = result["reserved_mb"] / total_mb
    result["reserved_fraction_of_card"] = round(ratio, 3)
    if ratio > 0.60:
        result["spillover_suspect"] = True
        result["spillover_warning"] = (
            "reserved 占整卡 %.0f%%，WSL2 下可能已溢出到主机内存，"
            "step 时间含 PCIe 传输，本组数据不可用于速度对比。" % (ratio * 100))

    if args.profile:
        result["profile"] = profile_run(model, opt, scaler, loader, device,
                                        args, opts, use_amp, set_none, nonblock)

    if args.dump_steps:
        result["steps"] = records

    return result


def profile_run(model, opt, scaler, loader, device, args, opts, use_amp,
                set_none, nonblock) -> dict:
    """单独跑若干步并开 profiler，测 GPU 空洞与 top kernel。

    与主循环分开跑，因为 profiler 本身有开销，混进去会污染 step time。
    """
    from torch.profiler import ProfilerActivity, profile

    it = iter(loader)
    # 预热，避开首次编译与 autotune
    for _ in range(3):
        try:
            b = next(it)
        except StopIteration:
            it = iter(loader)
            b = next(it)
        b = b.to(device, non_blocking=nonblock)
        opt.zero_grad(set_to_none=set_none)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            loss = contract_loss(model(b[:, :-1]), b[:, 1:])
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    torch.cuda.synchronize()

    n = args.profile_steps
    walls = []
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(n):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            try:
                b = next(it)
            except StopIteration:
                it = iter(loader)
                b = next(it)
            b = b.to(device, non_blocking=nonblock)
            opt.zero_grad(set_to_none=set_none)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                loss = contract_loss(model(b[:, :-1]), b[:, 1:])
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            torch.cuda.synchronize()
            walls.append((time.perf_counter() - t0) * 1000)

    # export_chrome_trace 每个 profiler 只能调一次，busy 与 top kernel 共用
    trace, trace_path = export_trace_once(prof)
    keep_trace = os.environ.get("PERF_KEEP_TRACE") == "1"
    try:
        busy_info = gpu_busy_ms_from_trace(prof, trace)
        top_kernels = classify_top_kernels_from_trace(trace, k=args.top_k)
    finally:
        if keep_trace:
            # 诊断用：保留原始 trace 供离线分析（窗口与墙时不自洽时必看）
            kept = trace_path.with_name("_kept_trace.json")
            try:
                trace_path.replace(kept)
                trace_path = kept
            except OSError:
                pass
        else:
            try:
                trace_path.unlink()
            except OSError:
                pass

    busy = busy_info["gpu_busy_ms"]
    wall_total = sum(walls)
    out = {
        "steps": n,
        "wall_total_ms": round(wall_total, 2),
        "gpu_busy_ms": busy,
        "gpu_gap_ms": round(wall_total - busy, 2),
        "gpu_busy_fraction": round(busy / wall_total, 4) if wall_total > 0 else None,
        "gpu_event_count": busy_info["gpu_event_count"],
        # 多流并行时区间会重叠，重叠量单独报，说明并集与求和的差
        "kernel_overlap_ms": busy_info["overlap_ms"],
        # 窗口裁剪自证：profiler 自己的窗口、裁剪前后的 busy、被丢弃的事件量。
        # inject_no_pin 曾在不裁剪时报出 busy 351.3%，裁剪后必须回到 100% 以内
        "profiler_window_ms": busy_info["profiler_window_ms"],
        "busy_unclipped_ms": busy_info["busy_unclipped_ms"],
        "events_dropped_outside_window": busy_info["events_dropped_outside_window"],
        "ms_dropped_outside_window": busy_info["ms_dropped_outside_window"],
        "events_clipped_at_boundary": busy_info["events_clipped_at_boundary"],
        # 两个错误口径一并留下，文章里用它们做反面对照：
        # 累加 events 的 self_device_time 会把 CPU 注解与其启动的 kernel 重复计数
        "wrong_busy_events_all_ms": busy_info["wrong_events_all_ms"],
        "wrong_busy_events_cuda_ms": busy_info["wrong_events_cuda_only_ms"],
        "wrong_overcount_ratio": busy_info["overcount_ratio_cuda_only"],
        "top_kernels": top_kernels,
    }
    # 物理一致性自检：busy 不可能超过 wall，超了说明口径又错了。
    # 裁剪到 profiler 窗口后这条应当恒成立，它失败就说明窗口取错了。
    if wall_total > 0 and busy > wall_total * 1.02:
        out["sanity_error"] = (
            "gpu_busy %.2f ms 超过 wall %.2f ms，测量口径有误，"
            "本组 busy/gap 数据不可用" % (busy, wall_total))
    # 窗口自检：profiler 根事件窗口应当约等于 wall_total。瞬态 trace 损坏
    # 会让窗口变成 wall 的数倍（实测 heavy_cpu 一次给出 window 4385 ms 对
    # wall 1429 ms），此时 busy 仍可能低于 wall，上面那条查不出来，但
    # kernel 并集是在被撑大的窗口上算的，busy 被虚高。窗口撑大即判损坏。
    window_ms = busy_info.get("profiler_window_ms")
    if window_ms and wall_total > 0 and window_ms > wall_total * 1.5:
        out["sanity_error"] = (
            "profiler 窗口 %.2f ms 远超 wall %.2f ms，trace 损坏，"
            "本组 busy/gap 数据不可用" % (window_ms, wall_total))
    return out


def run_repeated(args) -> dict:
    """同一配置跑 repeats 轮，报中位数与极差。

    为什么必须有这个（本篇踩过的坑）：同一份代码、同一个配置连跑两轮，
    inject_heavy_cpu 的 step_ms 给出 56.189 与 103.835（差 1.85 倍），
    baseline 的 busy 给出 66.8% 与 55.8%（差 11 个百分点）。
    单次测量不足以支撑任何结论。

    聚合口径：
      step_ms / tok_per_s / peak_mb / busy 取各轮的中位数
      同时报 min、max 与极差比，让读者能判断这组数据稳不稳
      极差比超过 repeat_spread_warn 就打 unstable 标记
    """
    n = max(1, args.repeats)
    runs = []
    for i in range(n):
        r = run(args)
        r["_repeat_index"] = i
        runs.append(r)

    if n == 1:
        out = runs[0]
        out["repeats"] = 1
        return out

    def med(key_fn):
        vals = [key_fn(r) for r in runs]
        vals = [v for v in vals if v is not None]
        if not vals:
            return None
        s = sorted(vals)
        m = len(s) // 2
        return s[m] if len(s) % 2 else round((s[m - 1] + s[m]) / 2, 3)

    base = dict(runs[len(runs) // 2])   # 以中位那轮为骨架
    steps = [get(r, "step_time", "mean_ms") for r in runs]
    tps = [r.get("tok_per_s") for r in runs]
    peaks = [r.get("peak_mb") for r in runs]
    busys = [get(r, "profile", "gpu_busy_fraction") for r in runs]

    def spread(vals):
        vals = [v for v in vals if v is not None]
        if len(vals) < 2 or min(vals) == 0:
            return None
        return round(max(vals) / min(vals), 2)

    base["repeats"] = n
    base["repeat_step_ms"] = {
        "median": med(lambda r: get(r, "step_time", "mean_ms")),
        "min": min(v for v in steps if v is not None),
        "max": max(v for v in steps if v is not None),
        "all": steps,
        "spread_ratio": spread(steps),
    }
    base["repeat_tok_per_s"] = {
        "median": med(lambda r: r.get("tok_per_s")),
        "min": min(v for v in tps if v is not None),
        "max": max(v for v in tps if v is not None),
        "all": tps,
        "spread_ratio": spread(tps),
    }
    base["repeat_peak_mb"] = {
        "median": med(lambda r: r.get("peak_mb")),
        "all": peaks,
        "spread_ratio": spread(peaks),
    }
    if any(b is not None for b in busys):
        base["repeat_busy_fraction"] = {
            "median": med(lambda r: get(r, "profile", "gpu_busy_fraction")),
            "min": min(v for v in busys if v is not None),
            "max": max(v for v in busys if v is not None),
            "all": busys,
        }

    # 顶层字段用中位数覆盖，这样 perf_report 与审计脚本读到的就是中位口径。
    # 注意：step_time 里的 min/max/p99/spread_ratio 仍来自中位那一轮的
    # 逐步分布，不是跨轮分布。跨轮离散度看 repeat_step_ms。
    if base["repeat_step_ms"]["median"] is not None:
        st = dict(base.get("step_time") or {})
        st["mean_ms"] = base["repeat_step_ms"]["median"]
        st["note"] = ("mean_ms 已替换为 %d 轮中位数；min/max/p99/spread_ratio "
                      "仍是中位那一轮的逐步分布，跨轮离散度见 repeat_step_ms"
                      % n)
        base["step_time"] = st
    if base["repeat_tok_per_s"]["median"] is not None:
        base["tok_per_s"] = base["repeat_tok_per_s"]["median"]
    if base["repeat_peak_mb"]["median"] is not None:
        base["peak_mb"] = base["repeat_peak_mb"]["median"]

    # 稳定性标记：极差比超阈值即判不稳，文章引用时必须带上这个标记
    sr = base["repeat_step_ms"]["spread_ratio"]
    if sr is not None and sr > args.repeat_spread_warn:
        base["unstable"] = True
        base["unstable_warning"] = (
            "同一配置 %d 轮，step_ms 极差比 %.2f 超过阈值 %.2f，"
            "本组数字只能当量级判据，不能当精确值引用。"
            % (n, sr, args.repeat_spread_warn))

    # 任何一轮触发溢出或自检失败，整组都不可用于速度对比
    for r in runs:
        if r.get("spillover_suspect"):
            base["spillover_suspect"] = True
            base["spillover_warning"] = r.get("spillover_warning")
            break
        if get(r, "profile", "sanity_error"):
            base.setdefault("sanity_error", get(r, "profile", "sanity_error"))
            break

    base["repeat_runs"] = [
        {"i": r["_repeat_index"],
         "step_ms": get(r, "step_time", "mean_ms"),
         "tok_per_s": r.get("tok_per_s"),
         "peak_mb": r.get("peak_mb"),
         "busy": get(r, "profile", "gpu_busy_fraction")}
        for r in runs
    ]
    return base


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("baseline", "inject", "optimize"),
                    default="baseline")
    ap.add_argument("--opts", type=str, default="none")
    ap.add_argument("--fault", type=str, default="none",
                    choices=("none", "slow_data", "heavy_cpu", "no_pin",
                             "sync_log"))
    ap.add_argument("--data-dir", type=str, default="exp_scale/data")
    ap.add_argument("--hidden", type=int, default=384)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--accum", type=int, default=4,
                    help="opts 含 accum 时的累积步数")
    # DataLoader
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--prefetch", type=int, default=4)
    ap.add_argument("--slow-data-ms", type=float, default=8.0)
    # 周期任务
    ap.add_argument("--eval-every", type=int, default=0)
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--ckpt-every", type=int, default=0)
    ap.add_argument("--ckpt-dir", type=str, default="runs/perf_ckpts")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--sync-log-ms", type=float, default=20.0,
                    help="sync_log 故障注入的每步串行 CPU 工作量级（毫秒）；"
                         "同步本身在本 harness 里免费，代价全靠这段 CPU 工作，"
                         "量级必须大于跨轮噪声（约 ±5 ms）才测得出")
    # profiler
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--profile-steps", type=int, default=10)
    ap.add_argument("--top-k", type=int, default=12)
    ap.add_argument("--no-phase-sync", action="store_true",
                    help="关闭阶段边界的 synchronize（阶段和会更接近 wall，"
                         "但纯 CPU 阶段如 data 会被异步掩盖而测不准）")
    ap.add_argument("--dump-steps", action="store_true")
    ap.add_argument("--repeats", type=int, default=1,
                    help="同一配置跑几轮，报中位数与极差。单次测量的 run-to-run "
                         "方差可达 1.85 倍，关键组至少跑 3 轮")
    ap.add_argument("--repeat-spread-warn", type=float, default=1.15,
                    help="step_ms 极差比超过此值即标记 unstable")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    res = run_repeated(args)
    text = json.dumps(res, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
