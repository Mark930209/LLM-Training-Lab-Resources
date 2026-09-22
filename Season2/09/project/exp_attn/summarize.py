"""summarize.py —— 把 exp_attn 各模式的 JSON 结果压成可读表格（09 篇辅助工具）。

跑完 sweep 后用它在终端看结论，不用翻几百行 JSON。
只读 results 文件，不改任何数据。

用法：
    python -m exp_attn.summarize results/Season2/09/backend_seq1024_fp16.json
    python -m exp_attn.summarize results/Season2/09/*.json
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path


def summarize_backend(rows: list[dict]) -> None:
    print("%-10s %-20s %8s %8s %9s %9s %s"
          % ("impl", "active_backend", "fwd_ms", "bwd_ms", "total_ms",
             "peak_mb", "honored"))
    for r in rows:
        if r.get("status") != "ok":
            print("%-10s %-20s %s"
                  % (r["impl"], r.get("status"), r.get("error", "")[:60]))
            continue
        print("%-10s %-20s %8s %8s %9s %9s %s"
              % (r["impl"], r.get("active_backend"), r["fwd_ms_median"],
                 r["bwd_ms_median"], r["total_ms_median"], r["peak_mb"],
                 r.get("forced_honored")))


def summarize_sweep(rows: list[dict], key: str) -> None:
    print("%-8s %-10s %-20s %9s %9s %s"
          % (key, "impl", "active_backend", "total_ms", "peak_mb", "honored"))
    for r in rows:
        if r.get("status") != "ok":
            print("%-8s %-10s %-20s %s"
                  % (r.get(key), r["impl"], r.get("status"),
                     r.get("error", "")[:50]))
            continue
        print("%-8s %-10s %-20s %9s %9s %s"
              % (r.get(key), r["impl"], r.get("active_backend"),
                 r["total_ms_median"], r["peak_mb"], r.get("forced_honored")))


def summarize_e2e(rows: list[dict]) -> None:
    print("%-18s %-8s %10s %10s %9s %9s %9s %8s %7s"
          % ("impl", "status", "step_ms", "tok/s", "peak_mb", "resv%",
             "loss", "speedup", "attn%"))
    for r in rows:
        if r.get("status") != "ok":
            print("%-18s %-8s %s"
                  % (r["impl"], r.get("status"), r.get("error", "")[:60]))
            continue
        share = r.get("attn_share") or {}
        attn_pct = share.get("attn_fraction")
        print("%-18s %-8s %10s %10s %9s %9s %9s %8s %7s"
              % (r["impl"], r["status"], r["step_ms_median"], r["tok_per_s"],
                 r["peak_mb"],
                 "%.0f%%" % (r.get("reserved_fraction_of_card", 0) * 100),
                 r.get("mean_loss"), r.get("speedup_vs_eager"),
                 "%.1f%%" % (attn_pct * 100) if attn_pct is not None else "-"))
        if r.get("spillover_suspect"):
            print("    !! 溢出嫌疑：%s" % r.get("spillover_warning", "")[:150])


def summarize_correctness(rep: dict) -> None:
    cfg = rep["config"]
    print("config: %(batch)s x %(heads)s heads, seq %(seq)s, head_dim "
          "%(head_dim)s, %(dtype)s, causal=%(causal)s" % cfg)
    print("tolerance: %s" % rep["tolerance"])
    print("%-10s %-12s %14s %14s %s"
          % ("backend", "status", "fwd_diff", "bwd_diff", "pass"))
    for name, r in rep["backends"].items():
        if r["status"] != "ok":
            print("%-10s %-12s %s"
                  % (name, r["status"], r.get("error", "")[:60]))
            continue
        print("%-10s %-12s %14s %14s %s"
              % (name, r["status"], r["fwd_max_abs_diff"],
                 r["bwd_max_abs_diff"], r["pass"]))
    print("summary: %s" % rep["summary"])


def summarize_probe(rep: dict) -> None:
    print("config: %s" % rep["config"])
    print("device: %s (sm%s), torch %s"
          % (rep["device"], rep["capability"], rep["torch"]))
    print("%-10s %-10s %s" % ("backend", "available", "error"))
    for name, r in rep["availability"].items():
        print("%-10s %-10s %s"
              % (name, r["available"], r.get("error", "")[:70]))
    dp = rep["default_path"]
    print("default path -> %s (evidence: %s)"
          % (dp["active_backend"], dp["evidence"]))
    for k in dp["top_kernels"][:3]:
        print("    %4d x %s" % (k["count"], k["name"][:96]))


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    paths = []
    for arg in sys.argv[1:]:
        paths.extend(sorted(glob.glob(arg)) or [arg])

    for p in paths:
        path = Path(p)
        if not path.exists():
            print("!! 文件不存在: %s" % p)
            continue
        rep = json.loads(path.read_text(encoding="utf-8"))
        print("=" * 78)
        print(path.name)
        print("=" * 78)

        if "backends" in rep and "tolerance" in rep:
            summarize_correctness(rep)
        elif "availability" in rep and "default_path" in rep:
            summarize_probe(rep)
        elif "rows" in rep:
            mode = rep.get("mode", "")
            rows = rep["rows"]
            if mode == "backend":
                summarize_backend(rows)
            elif mode in ("seq", "shape", "dtype", "causal"):
                key = {"seq": "seq", "shape": "head_dim", "dtype": "dtype",
                       "causal": "causal"}[mode]
                summarize_sweep(rows, key)
            else:
                # 没有 mode 字段但有 rows 的，按 e2e 处理
                summarize_e2e(rows)
        else:
            print(json.dumps(rep, indent=2, ensure_ascii=False)[:1500])
        print()


if __name__ == "__main__":
    main()
