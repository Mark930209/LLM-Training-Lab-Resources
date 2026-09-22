"""dump_io_arithmetic.py —— 把 io_arithmetic 的估算写成 results JSON（09 篇）。

标签是 SCALED：按实现结构做的算术推导，不是实测。
实测峰值见 sweep_seq_fp16.json，两者口径不同（流量 vs 驻留峰值）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp_attn.io_arithmetic import flops, flash_hbm_mb, naive_hbm_mb

# 本脚本放在项目根目录（与 exp_attn/ 同级），results/ 就在它下面
PROJECT_ROOT = Path(__file__).resolve().parent
OUT = PROJECT_ROOT / "results" / "Season1" / "09" / "io_arithmetic.json"

B, H, D = 2, 8, 64


def main() -> None:
    rows = []
    for n in (256, 512, 1024, 2048, 4096):
        nv = naive_hbm_mb(B, H, n, D)
        fl = flash_hbm_mb(B, H, n, D)
        fp = flops(B, H, n, D)
        rows.append({
            "seq": n,
            "naive_hbm_mb": nv["total_mb"],
            "flash_hbm_mb": fl["total_mb"],
            "hbm_ratio": round(nv["total_mb"] / fl["total_mb"], 2),
            "naive_nn_matrix_mb": nv["nn_matrix_mb"],
            "naive_nn_passes": nv["nn_passes"],
            "flash_nn_passes": fl["nn_passes"],
            "naive_gflops": fp["naive_gflops"],
            "flash_gflops": fp["flash_gflops"],
            "flops_ratio_flash_over_naive": round(
                fp["flash_gflops"] / fp["naive_gflops"], 2),
            "naive_detail_mb": nv["detail_mb"],
            "flash_detail_mb": fl["detail_mb"],
        })

    out = {
        "config": f"B={B} H={H} D={D} fp16 causal",
        "label": "SCALED",
        "label_note": ("按 naive 与 flash 的实现结构做的 HBM 流量算术推导，"
                       "不是实测。实测驻留峰值见 sweep_seq_fp16.json；"
                       "流量比与峰值比口径不同，数值不同是正常的。"),
        "key_finding": ("HBM 流量比随 N 上升（2.67 → 37.59），"
                        "FLOPs 比恒为 1.33 且 flash 算得更多；"
                        "收益来自少搬数据，不是少算。"),
        "rows": rows,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("written %s (%d rows)" % (OUT, len(rows)))


if __name__ == "__main__":
    main()
