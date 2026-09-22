"""perf_harness.py —— 单卡训练性能 harness（10 篇核心交付物）。

08 篇把显存压下来了，09 篇把 attention 换成了融合 kernel，但 GPU 利用率
仍然忽高忽低。本篇要回答：一个 step 里 GPU 到底在等什么。

与 common/benchmark.py 的 StepTimer 的区别（这是本篇的起点）：
    StepTimer 只报 avg_ms。平均值会把"偶发的长尾"和"稳定的慢"混成一件事，
    而单卡性能问题恰恰藏在长尾里（eval、checkpoint、日志、GC）。
    本 harness 记录每一步的完整分布与阶段拆分。

测量口径：
    wall_ms      一步的墙上时间（含 CPU 等待）
    gpu_busy_ms  该步内 CUDA kernel 的 self device time 之和
    gap_ms       wall - gpu_busy，即 GPU 空洞
    阶段拆分     data / h2d / forward / backward / optimizer / periodic

所有阶段都用 CUDA event 打点，避免把入队时间当成执行时间。
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass
class StepRecord:
    """一步的完整测量记录。"""

    step: int
    wall_ms: float
    phases: dict = field(default_factory=dict)
    loss: float = float("nan")
    periodic: str = ""          # 本步触发的周期任务：eval / ckpt / log
    gpu_busy_ms: float | None = None
    gap_ms: float | None = None


def pct(values: list[float], q: float) -> float:
    """分位数。q 取 0~100。空列表返回 nan。"""
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    idx = (len(s) - 1) * q / 100.0
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    frac = idx - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def summarize(times: list[float]) -> dict:
    """step time 分布，不只是平均值。

    极差比 max/min 与 p99/p50 是判断"有没有长尾污染"的两个关键量：
    接近 1 说明稳定，明显大于 1 说明有周期性任务或抖动混进来了。
    """
    if not times:
        return {"n": 0}
    p50 = pct(times, 50)
    return {
        "n": len(times),
        "mean_ms": round(statistics.mean(times), 3),
        "median_ms": round(p50, 3),
        "min_ms": round(min(times), 3),
        "max_ms": round(max(times), 3),
        "p90_ms": round(pct(times, 90), 3),
        "p99_ms": round(pct(times, 99), 3),
        "stdev_ms": round(statistics.stdev(times), 3) if len(times) > 1 else 0.0,
        "spread_ratio": round(max(times) / min(times), 2) if min(times) > 0 else None,
        "p99_over_p50": round(pct(times, 99) / p50, 2) if p50 > 0 else None,
    }


class PhaseTimer:
    """给一步内的各阶段打点。

    为什么不用 CUDA event（本篇踩过的坑）：event 记录的是流上两个点
    之间的时间，纯 CPU 阶段（如 next(data_iter)）不往流里排任何工作，
    两个 event 之间没有 kernel，elapsed_time 就接近 0——即使 CPU 实际
    阻塞了十几毫秒。而"GPU 在等数据"正是本篇要抓的东西，用 event 测
    会直接看不见。

    所以这里用 perf_counter + 每个边界 synchronize：同步会打断异步重叠，
    使阶段之和略大于真实 wall，但换来的是每个阶段的真实归属。
    真实 step 时间另用 wall_ms（只在步末同步一次）记录，两者并列上报，
    差值就是同步开销与异步重叠的量。
    """

    def __init__(self, device: str = "cuda", sync_each: bool = True):
        self.device = device
        self.sync_each = sync_each and device == "cuda"
        self.phases: dict[str, float] = {}
        self.order: list[str] = []
        self._open: tuple[str, float] | None = None

    def _sync(self) -> None:
        if self.sync_each and torch.cuda.is_available():
            torch.cuda.synchronize()

    def start(self, name: str) -> None:
        self._sync()
        self._open = (name, time.perf_counter())

    def stop(self, name: str) -> None:
        if self._open is None or self._open[0] != name:
            raise RuntimeError(f"phase {name} 未正确开启（当前 {self._open}）")
        self._sync()
        elapsed_ms = (time.perf_counter() - self._open[1]) * 1000
        # 同名阶段累加而不是覆盖：梯度累积下一步内有多次 micro-forward 与
        # micro-backward，要分别归属到 forward 与 backward。第一版把整个
        # accum 循环包在 forward 里，结果 opt_accum 报出 fwd 93.6% / bwd 0.0%，
        # 归属完全错了。
        self.phases[name] = round(self.phases.get(name, 0.0) + elapsed_ms, 3)
        if name not in self.order:
            self.order.append(name)
        self._open = None

    def sync_and_collect(self) -> dict:
        """返回各阶段毫秒数。接口保留，与旧调用方兼容。"""
        return dict(self.phases)


# chrome trace 里真正代表"GPU 在干活"的事件类别。
# 不包含 cpu_op / user_annotation：那些是 CPU 侧的算子与注解，
# 但它们也携带 self_device_time_total，累加会重复计数。
GPU_TRACE_CATEGORIES = ("kernel", "gpu_memcpy", "gpu_memset")


def _export_trace(prof, path) -> dict:
    prof.export_chrome_trace(str(path))
    return json.loads(Path(path).read_text(encoding="utf-8"))


def export_trace_once(prof, workdir=None) -> tuple[dict, Path | None]:
    """导出 chrome trace 并返回解析后的 dict。

    `export_chrome_trace` 每个 profiler 只能调一次，第二次抛
    `RuntimeError: Trace is already saved.`。所以 busy 与 top kernel
    必须共用一次导出，不能各自导。
    """
    import tempfile

    if workdir is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        trace_path = Path(tmp.name)
        tmp.close()
    else:
        trace_path = Path(workdir) / "_trace_tmp.json"
    return _export_trace(prof, trace_path), trace_path


def _profiler_window(trace: dict) -> tuple[float, float] | None:
    """取 profiler 自己的根事件窗口 [起, 止]（微秒）。

    chrome trace 里有一个 cat == "Trace" 的根事件，它的区间就是本次
    profiling 的窗口。实测与 perf_counter 量的 wall 几乎完全吻合
    （572.29 ms 对 568.52 ms），所以可以用它来裁剪。
    """
    for e in trace.get("traceEvents", []):
        if e.get("cat") == "Trace" and e.get("dur") is not None:
            ts = float(e["ts"])
            return ts, ts + float(e["dur"])
    return None


def _gpu_intervals(trace: dict, window: tuple[float, float] | None = None
                   ) -> tuple[list[tuple[float, float]], dict]:
    """从 chrome trace 取全部 GPU 侧事件的 [起, 止] 区间（微秒）。

    window 不为 None 时把每个区间裁剪到窗口内，并统计被丢弃/裁剪的量。
    裁剪是必要的（本篇踩过的坑）：多 worker 的 DataLoader 下，trace 里
    会混进计时窗口之外的事件，不裁剪时 inject_no_pin 算出 busy 2293 ms
    对 wall 652 ms（351.3%）。裁剪后 busy <= wall 由构造保证，
    不依赖 trace 是否干净。
    """
    iv = []
    dropped = 0
    dropped_us = 0.0
    clipped = 0
    for e in trace.get("traceEvents", []):
        if e.get("cat") not in GPU_TRACE_CATEGORIES:
            continue
        dur = e.get("dur")
        ts = e.get("ts")
        if dur is None or ts is None:
            continue
        s, t = float(ts), float(ts) + float(dur)
        if window is not None:
            ws, we = window
            if t <= ws or s >= we:      # 完全在窗口外，丢弃
                dropped += 1
                dropped_us += t - s
                continue
            ns, nt = max(s, ws), min(t, we)
            if ns != s or nt != t:      # 跨窗口边界，裁剪
                clipped += 1
            s, t = ns, nt
        if t > s:
            iv.append((s, t))
    iv.sort()
    stats = {"dropped_events": dropped,
             "dropped_ms": round(dropped_us / 1000.0, 3),
             "clipped_events": clipped}
    return iv, stats


def _union_us(iv: list[tuple[float, float]]) -> float:
    """区间并集总长（微秒）。重叠部分只计一次。"""
    total = 0.0
    cur_s = cur_e = None
    for s, e in iv:
        if cur_e is None:
            cur_s, cur_e = s, e
        elif s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def gpu_busy_ms_from_trace(prof, trace: dict) -> dict:
    """测"GPU 真正有 kernel 在跑的时间"（ms），以及它与各错误口径的对比。

    为什么不能用 events()/key_averages() 累加 self_device_time_total
    （本篇踩过的坑，连续两版都错）：
      第一版遍历 key_averages() 累加，算出 busy 918 ms 对 wall 606 ms，
        busy 151.4%、gap 为负。原因是表里同时含 CPU 算子条目
        （aten::mm）与 CUDA kernel 条目，device time 被两边各计一次。
      第二版改成只取 device_type == CUDA 的事件，仍然错：
        `Optimizer.step#AdamW.step` 这类 record_function 注解的
        device_type 就是 CUDA，它的 self_device_time_total 聚合了自己
        启动的全部 kernel，与 kernel 事件重复。实测 events CUDA 求和
        1.230 ms，而 chrome trace 里真实 kernel 区间并集只有 0.394 ms，
        虚高 3.1 倍。opt_prefetch 甚至报出 busy 493%。

    正确口径：只取 chrome trace 里 cat 为 kernel/gpu_memcpy/gpu_memset
    的事件，对时间区间求**并集**。并集而不是求和，因为多个流上的 kernel
    可以真正并行，求和会超过 wall。
    """
    window = _profiler_window(trace)
    iv, clip_stats = _gpu_intervals(trace, window)
    union_us = _union_us(iv)
    naive_sum_us = sum(e - s for s, e in iv)

    # 不裁剪的口径一并算出来，用于对照与自检
    iv_raw, _ = _gpu_intervals(trace, None)
    union_raw_us = _union_us(iv_raw)
    window_ms = round((window[1] - window[0]) / 1000.0, 3) if window else None

    # 两个错误口径一并算出来，文章里用它们做反面对照
    ev_all_us = sum(float(getattr(e, "self_device_time_total", 0.0) or 0.0)
                    for e in prof.events())
    from torch.profiler import DeviceType
    ev_cuda_us = sum(
        float(getattr(e, "self_device_time_total", 0.0) or 0.0)
        for e in prof.events()
        if getattr(e, "device_type", None) == DeviceType.CUDA)

    return {
        "gpu_busy_ms": round(union_us / 1000.0, 3),
        "gpu_event_count": len(iv),
        "overlap_ms": round((naive_sum_us - union_us) / 1000.0, 3),
        "profiler_window_ms": window_ms,
        "busy_unclipped_ms": round(union_raw_us / 1000.0, 3),
        "events_dropped_outside_window": clip_stats["dropped_events"],
        "ms_dropped_outside_window": clip_stats["dropped_ms"],
        "events_clipped_at_boundary": clip_stats["clipped_events"],
        "wrong_events_all_ms": round(ev_all_us / 1000.0, 3),
        "wrong_events_cuda_only_ms": round(ev_cuda_us / 1000.0, 3),
        "overcount_ratio_cuda_only": round(ev_cuda_us / union_us, 2)
        if union_us > 0 else None,
    }


def classify_top_kernels_from_trace(trace: dict, k: int = 12) -> list[dict]:
    """按真实 kernel 耗时排序取前 k 个。

    同样走 chrome trace，否则排在前面的会是 `Optimizer.step#AdamW.step`
    这种 CPU 侧注解名，而不是真正的 kernel。
    """
    from collections import defaultdict

    agg: dict[str, dict] = defaultdict(lambda: {"self_ms": 0.0, "count": 0})
    for e in trace.get("traceEvents", []):
        if e.get("cat") not in GPU_TRACE_CATEGORIES:
            continue
        dur = e.get("dur")
        if dur is None:
            continue
        name = str(e.get("name", ""))[:90]
        agg[name]["self_ms"] += float(dur) / 1000.0
        agg[name]["count"] += 1
    rows = [{"name": n, "self_ms": round(v["self_ms"], 3),
             "count": v["count"]} for n, v in agg.items()]
    rows.sort(key=lambda r: -r["self_ms"])
    return rows[:k]


def mb(n: float) -> float:
    return round(n / 1024 / 1024, 1)


def get(d, *path, default=None):
    """安全取嵌套字段。perf_bench 与 perf_report 共用。"""
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def mem_snapshot(tag: str) -> dict:
    """沿用 07 篇的口径：allocated / reserved / peak。"""
    return {
        "stage": tag,
        "allocated_mb": mb(torch.cuda.memory_allocated()),
        "reserved_mb": mb(torch.cuda.memory_reserved()),
        "peak_mb": mb(torch.cuda.max_memory_allocated()),
    }
