"""perf_report.py —— 把 results/Season1/10/*.json 汇总成性能报告（10 篇交付物）。

outline 要求的最终交付之一是 Single GPU Performance Report。
本工具读全部结果 JSON，输出四张表：
    1. 阶段拆分与 GPU 空洞（含 busy 百分比）
    2. 故障注入对照（注入什么、时间线上留下什么证据）
    3. 单项优化收益（相对 baseline 的 tok/s 与 peak）
    4. 优化瀑布（逐项累加的收益）

只读，不改任何数据。所有数字直接来自 JSON，不做二次推算，
避免报告与结果文件不一致。

用法：
    python -m exp_perf.perf_report results/Season1/10
    python -m exp_perf.perf_report results/Season1/10 --baseline baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_all(d: Path) -> dict[str, dict]:
    """读目录下全部 JSON，键是文件名（不含扩展名）。"""
    out = {}
    for f in sorted(d.glob("*.json")):
        try:
            out[f.stem] = json.loads(f.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"!! 跳过 {f.name}: {exc}", file=sys.stderr)
    return out


def get(d: dict, *path, default=None):
    """安全取嵌套字段。"""
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def fmt_pct(v) -> str:
    return f"{v * 100:.1f}%" if isinstance(v, (int, float)) else "-"


def table_phase(reps: dict, names: list[str]) -> None:
    """表 1：阶段拆分与 GPU 空洞。"""
    print("\n" + "=" * 100)
    print("表 1  阶段拆分与 GPU 空洞")
    print("=" * 100)
    print("%-30s %8s %7s %7s %8s %8s %8s %8s %7s"
          % ("组", "step_ms", "data%", "h2d%", "fwd%", "bwd%", "opt%",
             "perio%", "busy%"))
    for n in names:
        r = reps.get(n)
        if not r:
            continue
        sh = r.get("phase_share") or {}
        busy = get(r, "profile", "gpu_busy_fraction")
        print("%-30s %8s %7s %7s %8s %8s %8s %8s %7s"
              % (n[:30],
                 get(r, "step_time", "mean_ms", default="-"),
                 fmt_pct(sh.get("data")), fmt_pct(sh.get("h2d")),
                 fmt_pct(sh.get("forward")), fmt_pct(sh.get("backward")),
                 fmt_pct(sh.get("optimizer")), fmt_pct(sh.get("periodic")),
                 fmt_pct(busy)))


def table_stepdist(reps: dict, names: list[str]) -> None:
    """表 2：step time 分布。平均值会把长尾藏起来，这张表专门暴露它。"""
    print("\n" + "=" * 100)
    print("表 2  step time 分布（看长尾，不只看平均值）")
    print("=" * 100)
    print("%-30s %8s %8s %8s %8s %8s %8s %8s"
          % ("组", "mean", "median", "min", "max", "p99", "极差比", "p99/p50"))
    for n in names:
        r = reps.get(n)
        if not r:
            continue
        st = r.get("step_time") or {}
        print("%-30s %8s %8s %8s %8s %8s %8s %8s"
              % (n[:30], st.get("mean_ms", "-"), st.get("median_ms", "-"),
                 st.get("min_ms", "-"), st.get("max_ms", "-"),
                 st.get("p99_ms", "-"), st.get("spread_ratio", "-"),
                 st.get("p99_over_p50", "-")))
        pe = r.get("periodic_steps") or {}
        cl = r.get("clean_steps") or {}
        if pe.get("n"):
            print("%-30s   周期任务命中 %d 步: mean %s ms / max %s ms"
                  % ("", pe["n"], pe.get("mean_ms"), pe.get("max_ms")))
        if cl.get("n") and pe.get("n"):
            print("%-30s   未命中 %d 步: mean %s ms / max %s ms"
                  % ("", cl["n"], cl.get("mean_ms"), cl.get("max_ms")))


def table_opts(reps: dict, names: list[str], base_name: str) -> None:
    """表 3：单项优化收益，相对 baseline。"""
    print("\n" + "=" * 100)
    print("表 3  单项优化收益（相对 baseline）")
    print("=" * 100)
    base = reps.get(base_name) or {}
    base_tps = base.get("tok_per_s") or 0
    base_ms = get(base, "step_time", "mean_ms") or 0
    base_peak = base.get("peak_mb") or 0
    print("baseline = %s: %s ms, %s tok/s, peak %s MB"
          % (base_name, base_ms, base_tps, base_peak))
    print()
    print("%-26s %9s %9s %9s %9s %9s"
          % ("组", "step_ms", "tok/s", "tok/s比", "peak_MB", "peak比"))
    for n in names:
        r = reps.get(n)
        if not r or n == base_name:
            continue
        tps = r.get("tok_per_s") or 0
        peak = r.get("peak_mb") or 0
        ms = get(r, "step_time", "mean_ms")
        tps_ratio = f"{tps / base_tps:.2f}x" if base_tps else "-"
        peak_ratio = f"{peak / base_peak:.2f}x" if base_peak else "-"
        flag = ""
        if base_tps and tps < base_tps:
            flag = "  <-- 负收益"
        print("%-26s %9s %9s %9s %9s %9s%s"
              % (n[:26], ms, tps, tps_ratio, peak, peak_ratio, flag))


# 优化瀑布的逻辑累加顺序。必须显式给定，不能按文件名字母序：
# 字母序下 combo_all 排在 combo_compute 前面，算出的"本步增益"
# 是排序假象（本篇实际踩过：报出 -36.1% / -24.8% 这种无意义的数）。
#
# 而且只有真正累加的组合才能进阶梯：
#   combo_data（只开数据管道）与 combo_compute（只开 compute）是两个独立分支，
#   不是阶梯上的两级，串起来算增量仍然是错的，归到分支表。
#   stability_combo 带周期任务（eval/ckpt 每 50 步），tok/s 与 combo_all 不可比，
#   归到稳定性表。
WATERFALL_ORDER = (
    "baseline",           # 未优化
    "combo_no_compile",   # 数据管道 + compute（sdpa/fused/setnone）
    "combo_all",          # 再加 torch.compile
)
BRANCH_ORDER = ("combo_data", "combo_compute")
STABILITY_ORDER = ("stability_combo",)


def _print_row(reps: dict, n: str, base_tps: int, prev: int | None) -> int:
    r = reps.get(n)
    if not r:
        return prev if prev is not None else 0
    tps = r.get("tok_per_s") or 0
    gain = (f"{(tps - prev) / prev * 100:+.1f}%"
            if prev else "-")
    busy = get(r, "profile", "gpu_busy_fraction")
    print("%-26s %9s %9s %11s %11s %9s"
          % (n[:26], tps,
             f"{tps / base_tps:.2f}x" if base_tps else "-",
             gain, r.get("peak_mb", "-"), fmt_pct(busy)))
    return tps


def table_waterfall(reps: dict, names: list[str], base_name: str) -> None:
    """表 4：优化瀑布。只列真正累加的组合，分支与稳定性跑分开列。"""
    print("\n" + "=" * 100)
    print("表 4  优化瀑布（按逻辑累加顺序）")
    print("=" * 100)
    base = reps.get(base_name) or {}
    base_tps = base.get("tok_per_s") or 0
    hdr = ("%-26s %9s %9s %11s %11s %9s"
           % ("组", "tok/s", "对base", "本步增益", "peak_MB", "busy%"))

    print("【累加阶梯】")
    print(hdr)
    prev = base_tps
    for n in WATERFALL_ORDER:
        prev = _print_row(reps, n, base_tps, prev)

    print("\n【独立分支（不是阶梯上的一级，各自只开一类优化）】")
    print(hdr)
    for n in BRANCH_ORDER:
        _print_row(reps, n, base_tps, base_tps)

    print("\n【长稳定性跑（含周期任务，tok/s 与上面不可直接比）】")
    print(hdr)
    for n in STABILITY_ORDER:
        _print_row(reps, n, base_tps, None)

    extra = [n for n in names
             if n not in WATERFALL_ORDER + BRANCH_ORDER + STABILITY_ORDER
             and (n.startswith("combo_") or n.startswith("stability_"))]
    if extra:
        print("\n  （未归类的组合组：%s）" % ", ".join(sorted(extra)))


def table_faults(reps: dict, names: list[str]) -> None:
    """表 5：故障注入对照。注入什么，时间线上留下什么证据。"""
    print("\n" + "=" * 100)
    print("表 5  故障注入对照")
    print("=" * 100)
    print("%-30s %-12s %9s %8s %8s %9s"
          % ("组", "fault", "step_ms", "data%", "busy%", "gap_ms"))
    for n in names:
        r = reps.get(n)
        if not r:
            continue
        print("%-30s %-12s %9s %8s %8s %9s"
              % (n[:30], r.get("fault", "-"),
                 get(r, "step_time", "mean_ms", default="-"),
                 fmt_pct(get(r, "phase_share", "data")),
                 fmt_pct(get(r, "profile", "gpu_busy_fraction")),
                 get(r, "profile", "gpu_gap_ms", default="-")))


def table_spill(reps: dict, names: list[str]) -> None:
    """表 6：溢出防护。09 篇的教训，reserved 超 60% 的数据不可用于速度对比。"""
    flagged = []
    for n in names:
        r = reps.get(n)
        if r and r.get("spillover_suspect"):
            flagged.append((n, r.get("reserved_fraction_of_card"),
                            r.get("spillover_warning", "")))
    print("\n" + "=" * 100)
    print("表 6  溢出防护（reserved 占整卡 > 60% 即标记）")
    print("=" * 100)
    if not flagged:
        print("无组被标记，全部数据可用于速度对比。")
        return
    for n, ratio, warn in flagged:
        print(f"  !! {n}: reserved 占整卡 {fmt_pct(ratio)}")
        print(f"     {warn[:120]}")


def table_sanity(reps: dict, names: list[str]) -> None:
    """表 7：测量口径自检。

    本篇踩了三次工具给出物理不可能数字的坑（busy 151.4%、493.1%、351.3%）。
    这张表把自检结果、窗口裁剪诊断和错误口径对照一起打出来。
    任何一组 busy > wall 都必须先修工具再看数据。
    """
    print("\n" + "=" * 108)
    print("表 7  测量口径自检（busy 不可能超 wall）")
    print("=" * 108)
    print("%-28s %9s %9s %7s %9s %9s %7s %6s %s"
          % ("组", "wall_ms", "busy_ms", "busy%", "未裁剪ms", "错口径cuda",
             "虚高", "丢弃", "自检"))
    bad = []
    for n in names:
        r = reps.get(n)
        p = (r or {}).get("profile") or {}
        if not p:
            continue
        wall = p.get("wall_total_ms")
        busy = p.get("gpu_busy_ms")
        sanity = p.get("sanity_error")
        if sanity:
            bad.append((n, sanity))
        dropped = p.get("events_dropped_outside_window")
        print("%-28s %9s %9s %7s %9s %9s %7s %6s %s"
              % (n[:28], wall, busy, fmt_pct(p.get("gpu_busy_fraction")),
                 p.get("busy_unclipped_ms", "-"),
                 p.get("wrong_busy_events_cuda_ms", "-"),
                 p.get("wrong_overcount_ratio", "-"),
                 dropped if dropped is not None else "-",
                 "!! 不可用" if sanity else "OK"))
    print()
    if bad:
        print(f"  !! {len(bad)} 组自检失败，这些组的 busy/gap 数据不得写进文章：")
        for n, msg in bad:
            print(f"     {n}: {msg[:100]}")
    else:
        print("  全部组自检通过。")
    print("  判读：")
    print("    虚高 = 错口径cuda / busy，即累加 events 的 self_device_time 比")
    print("           chrome trace 区间并集高出的倍数，本篇实测约 1.2~1.4 倍。")
    print("    丢弃 = 落在 profiler 窗口之外被裁掉的 GPU 事件数。为 0 说明")
    print("           trace 干净；不为 0 说明多 worker 下混进了窗口外事件，")
    print("           裁剪是必要的（inject_no_pin 曾因此报出 busy 351.3%）。")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", type=str)
    ap.add_argument("--baseline", type=str, default="baseline")
    args = ap.parse_args()

    d = Path(args.results_dir)
    if not d.is_dir():
        print(f"目录不存在: {d}")
        sys.exit(1)

    reps = load_all(d)
    if not reps:
        print("没有结果文件")
        sys.exit(1)

    names = list(reps.keys())
    print(f"读入 {len(names)} 个结果文件，来自 {d}")

    # 分组
    base_group = [n for n in names if n == args.baseline]
    inject = [n for n in names if n.startswith("inject_")]
    periodic = [n for n in names if n.startswith("periodic_")]
    single = [n for n in names if n.startswith("opt_")]
    combo = [n for n in names if n.startswith("combo_")]
    stability = [n for n in names if n.startswith("stability_")]

    table_phase(reps, base_group + inject + periodic + single + combo)
    table_stepdist(reps, base_group + periodic + combo + stability)
    table_faults(reps, base_group + inject)
    table_opts(reps, base_group + single, args.baseline)
    table_waterfall(reps, base_group + combo + stability, args.baseline)
    table_spill(reps, names)
    table_sanity(reps, names)

    # compile 的编译开销单列，因为它是一次性成本，不能摊进 tok/s 比较
    print("\n" + "=" * 100)
    print("torch.compile 的一次性成本")
    print("=" * 100)
    for n in names:
        r = reps[n]
        wrap = r.get("compile_wrap_s")
        first = r.get("compile_first_step_ms")
        if wrap or first:
            print(f"  {n}: wrap={wrap}s first_step={first}ms "
                  f"mean={get(r, 'step_time', 'mean_ms')}ms")
    print("  判读：first_step 若与 mean 接近，说明编译开销落在 warmup 里被吸收；")
    print("        若明显更大，短训练会被编译成本吃掉收益，要按总步数摊。")


if __name__ == "__main__":
    main()
