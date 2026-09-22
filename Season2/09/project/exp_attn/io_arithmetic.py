"""io_arithmetic.py —— naive 与 flash 的 HBM 读写量可复算估算（09 篇）。

核心判断"差距不在 FLOPs，在中间矩阵要不要反复进出显存"需要一个能复算的
算术支撑，不能只靠实测倍率反推。本脚本按公开的实现结构算两边的 HBM
流量，与 3.3 的实测显存倍率互相印证。

口径说明（写进文章，避免读者当成精确值）：
- 只算 attention 内部的主要张量流量，不含 QKV 投影与输出投影
- naive 按"每个中间张量写一次、读一次"计
- flash 按"分块在线计算，N×N 不落 HBM"计
- 单位 MB，1 MB = 1024*1024 字节

这是估算（SCALED 口径的算术推导），不是实测；实测峰值见 sweep_seq_fp16.json。
"""

from __future__ import annotations

import json


def naive_hbm_mb(b: int, h: int, n: int, d: int, itemsize: int = 2,
                 causal: bool = True) -> dict:
    """naive：物化 N×N，每个中间张量都要进出 HBM。

    前向：
      读 Q,K          -> 2 * B*H*N*D
      写 scores       -> B*H*N*N          （第一次落盘）
      读 scores       -> B*H*N*N          （softmax 要读回来）
      写 probs        -> B*H*N*N          （softmax 输出，第二份）
      读 probs, 读 V  -> B*H*N*N + B*H*N*D
      写 out          -> B*H*N*D
    反向：probs 必须留着，所以要再读一次
      读 probs        -> B*H*N*N
      （梯度链里 scores/probs 的读写还有若干次，这里只算主要项）
    """
    qkv = b * h * n * d * itemsize
    nn = b * h * n * n * itemsize

    fwd = {
        "read_qk": 2 * qkv,
        "write_scores": nn,
        "read_scores": nn,
        "write_probs": nn,
        "read_probs_v": nn + qkv,
        "write_out": qkv,
    }
    bwd = {
        "read_probs": nn,
        "write_dscores": nn,
        "read_dscores": nn,
    }
    total = sum(fwd.values()) + sum(bwd.values())
    return {
        "fwd_bytes": sum(fwd.values()),
        "bwd_bytes": sum(bwd.values()),
        "total_bytes": total,
        "total_mb": round(total / 1024 / 1024, 1),
        "nn_matrix_mb": round(nn / 1024 / 1024, 1),
        "nn_passes": 6,  # scores/probs 一共进出 HBM 六次
        "detail_mb": {k: round(v / 1024 / 1024, 1)
                      for k, v in {**fwd, **bwd}.items()},
    }


def flash_hbm_mb(b: int, h: int, n: int, d: int, itemsize: int = 2,
                 causal: bool = True) -> dict:
    """flash：分块在线计算，N×N 从不落 HBM。

    前向：
      读 Q,K,V        -> 3 * B*H*N*D
      写 out          -> B*H*N*D
      写 logsumexp    -> B*H*N      （反向重算 softmax 只需要这个统计量）
    反向：重算分数块，所以 Q,K,V 要再读一遍
      读 Q,K,V,out,dO -> 5 * B*H*N*D
      写 dQ,dK,dV     -> 3 * B*H*N*D
    N×N 项为 0，这是全部差别所在。
    """
    qkv = b * h * n * d * itemsize
    lse = b * h * n * 4  # logsumexp 通常存 fp32

    fwd = {
        "read_qkv": 3 * qkv,
        "write_out": qkv,
        "write_lse": lse,
    }
    bwd = {
        "read_qkv_out_do": 5 * qkv,
        "write_dqkv": 3 * qkv,
    }
    total = sum(fwd.values()) + sum(bwd.values())
    return {
        "fwd_bytes": sum(fwd.values()),
        "bwd_bytes": sum(bwd.values()),
        "total_bytes": total,
        "total_mb": round(total / 1024 / 1024, 1),
        "nn_matrix_mb": 0.0,
        "nn_passes": 0,
        "detail_mb": {k: round(v / 1024 / 1024, 1)
                      for k, v in {**fwd, **bwd}.items()},
    }


def flops(b: int, h: int, n: int, d: int, causal: bool = True) -> dict:
    """两边的 FLOPs。causal 下有效计算量约减半，但两边同比例，比值不变。

    QK^T:  2 * B*H*N*N*D
    P@V:   2 * B*H*N*N*D
    flash 反向要重算一次分数块，所以多一份 QK^T 与 P@V 的量级。
    """
    qk = 2 * b * h * n * n * d
    pv = 2 * b * h * n * n * d
    base = qk + pv
    if causal:
        base //= 2
    return {
        "naive_gflops": round(base * 3 / 1e9, 2),   # fwd 1 份 + bwd 2 份
        "flash_gflops": round(base * 4 / 1e9, 2),   # 反向重算，多 1 份
        "flash_does_more_flops": True,
    }


def main() -> None:
    rows = []
    for n in (256, 512, 1024, 2048, 4096):
        b, h, d = 2, 8, 64
        nv = naive_hbm_mb(b, h, n, d)
        fl = flash_hbm_mb(b, h, n, d)
        fp = flops(b, h, n, d)
        rows.append({
            "seq": n,
            "naive_hbm_mb": nv["total_mb"],
            "flash_hbm_mb": fl["total_mb"],
            "hbm_ratio": round(nv["total_mb"] / fl["total_mb"], 2),
            "naive_nn_matrix_mb": nv["nn_matrix_mb"],
            "naive_nn_passes": nv["nn_passes"],
            "naive_gflops": fp["naive_gflops"],
            "flash_gflops": fp["flash_gflops"],
            "flops_ratio_flash_over_naive": round(
                fp["flash_gflops"] / fp["naive_gflops"], 2),
        })

    print("B=2 H=8 D=64 fp16 causal")
    print("%-6s %12s %12s %8s %14s %10s %10s %8s"
          % ("seq", "naive_HBM_MB", "flash_HBM_MB", "ratio",
             "N×N_MB", "naive_GF", "flash_GF", "GF比"))
    for r in rows:
        print("%-6d %12.1f %12.1f %8.2f %14.1f %10.2f %10.2f %8.2f"
              % (r["seq"], r["naive_hbm_mb"], r["flash_hbm_mb"],
                 r["hbm_ratio"], r["naive_nn_matrix_mb"],
                 r["naive_gflops"], r["flash_gflops"],
                 r["flops_ratio_flash_over_naive"]))

    print()
    print("判读：HBM 流量比随 N 增大而上升（naive 含 N² 项，flash 只有 N 项），")
    print("      而 FLOPs 比恒为 1.33（flash 反向重算多算 1/3），与 N 无关。")
    print("      收益来自少搬数据，不是少算。")

    out = {"config": "B=2 H=8 D=64 fp16 causal", "rows": rows}
    print()
    print(json.dumps(out, indent=2, ensure_ascii=False)[:400])


if __name__ == "__main__":
    main()
