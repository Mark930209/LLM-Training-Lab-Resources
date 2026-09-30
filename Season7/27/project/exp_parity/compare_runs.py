"""compare_runs.py —— 两次 run 结果 JSON 的 parity 对账。

用法：python compare_runs.py a.json b.json [--tol 1e-5]
输出：loss 曲线 maxabs / 首个分叉步 / 参数 checksum 是否一致。
"""

from __future__ import annotations

import argparse
import json

import parity_common as pc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    args = ap.parse_args()

    with open(args.a, encoding="utf-8") as f:
        ra = json.load(f)
    with open(args.b, encoding="utf-8") as f:
        rb = json.load(f)

    lp = pc.loss_parity(ra["loss_curve"], rb["loss_curve"])
    ca, cb = ra.get("param_checksum"), rb.get("param_checksum")
    out = {
        "a": {"file": args.a, "framework": ra.get("framework"),
              "init": ra.get("init"), "loss_first": ra.get("loss_first"),
              "loss_last": ra.get("loss_last")},
        "b": {"file": args.b, "framework": rb.get("framework"),
              "init": rb.get("init"), "loss_first": rb.get("loss_first"),
              "loss_last": rb.get("loss_last")},
        "loss_parity": lp,
        "checksum_a": ca,
        "checksum_b": cb,
        "checksum_match": (ca == cb) if (ca and cb) else None,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
