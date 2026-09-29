"""pp_hang.py —— 25 篇失败案例 2：send/recv 顺序错误造成 hang。

两个 rank 都用"先发后收"的对称写法：rank0 先发激活等梯度，rank1 先发梯度
等激活，两边都阻塞在 send 上，circular wait。P2P watchdog 会在 timeout 后
中止进程，捕获原始报错即为正文素材。

用法（torchrun 两机，timeout 建议 60s 内）：
    python pp_hang.py --out results/Season6/25/hang_report.json
"""

from __future__ import annotations

import argparse
import os
from datetime import timedelta

import torch
import torch.distributed as dist

from pp_common import env_rank_world, run_meta, save_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout-s", type=int, default=45)
    args = ap.parse_args()

    rank, world = env_rank_world()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    dev_idx = (local_rank if torch.cuda.is_available()
               and local_rank < torch.cuda.device_count() else 0)
    device = f"cuda:{dev_idx}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(dev_idx)
    if not dist.is_initialized():
        dist.init_process_group("nccl",
                                timeout=timedelta(seconds=args.timeout_s))

    report = {
        "meta": run_meta(rank, world, {**vars(args),
                                       "scenario": "both send before recv"}),
    }
    buf = torch.ones(1, 256, 1536, device=device)
    try:
        if rank == 0:
            dist.send(buf, 1)                       # 先发：等 rank1 收
            dist.recv(buf, 1)                       # 永远到不了
        else:
            dist.send(buf, 0)                       # 先发：等 rank0 收
            dist.recv(buf, 0)                       # 永远到不了
        report.update({"hang": False, "note": "unexpectedly completed"})
    except Exception as e:  # noqa: BLE001 —— 失败案例留档
        report.update({
            "hang": True,
            "error_type": type(e).__name__,
            "error_msg": str(e)[:600],
        })
    save_report(args.out.replace(".json", f".r{rank}.json"), report)


if __name__ == "__main__":
    main()
