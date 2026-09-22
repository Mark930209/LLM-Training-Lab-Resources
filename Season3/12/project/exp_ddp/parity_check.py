"""parity_check.py —— 单卡与双卡三级对齐验证（驱动 torchrun 跑两条再比对）。

三级对齐（从松到严）：
  1. init_checksum：起点一致（同 seed 初始化，哈希必须逐位相同）
  2. loss 轨迹：每 record_every 步的 global_loss 逐步对齐（容差内）
  3. 最终参数：跨进程加载两份 state_dict，算 max abs err 与相对误差。
     注意不用 final_checksum 相等做硬判据：fp32 求和顺序不同会让参数差
     1 ULP，哈希就不同；误差量级才是诚实的判据（预期 1e-6~1e-5）。

为什么单卡与双卡能对齐（等价数学，详见 brief-and-structure.md）：
  DistributedSampler 交错分片保证"双卡第 k 步两个分片的并集"恰好等于
  "单卡第 k 步的 batch"（同一个 perm 切片）；contract_loss 是 mean reduction，
  DDP 在 backward 中对梯度 all-reduce mean，于是双卡梯度 = 单卡全 batch 梯度。
  唯一残差是 fp32 求和顺序不同，量级 1e-6~1e-5，用容差判定。

用法（WSL 项目根目录）：
  ./.venv/bin/python -m exp_ddp.parity_check --steps 30 --global-batch 16 \
      --out results/Season3/12/parity.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

PROJ = Path(__file__).resolve().parent.parent


def _run(cmd: list, out_path: Path) -> dict:
    """跑一条命令，读回它写的 JSON。"""
    print("  $ " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(PROJ), capture_output=True, text=True)
    if r.returncode != 0:
        print("  !! 命令失败 exit=%d" % r.returncode)
        print("  stderr 末尾：\n" + "\n".join(r.stderr.splitlines()[-15:]))
        raise SystemExit(f"parity 子命令失败: {' '.join(cmd)}")
    if not out_path.exists():
        raise SystemExit(f"未产出 {out_path}")
    return json.loads(out_path.read_text(encoding="utf-8"))


def compare(single: dict, ddp: dict, tol: float,
            single_params: Path, ddp_params: Path) -> dict:
    """三级对齐比对。第三级用参数误差量级，不用哈希相等。"""
    init_ok = single["init_checksum"] == ddp["init_checksum"]

    s_loss = [h["loss"] for h in single["history"]["global_loss"]]
    d_loss = [h["loss"] for h in ddp["history"]["global_loss"]]
    loss_pairs = list(zip(s_loss, d_loss))
    loss_max_err = max((abs(a - b) for a, b in loss_pairs), default=0.0)
    loss_ok = loss_max_err < tol and len(s_loss) == len(d_loss)

    # 第三级：跨进程加载两份最终参数，算误差量级
    sd_s = torch.load(single_params, map_location="cpu", weights_only=False)
    sd_d = torch.load(ddp_params, map_location="cpu", weights_only=False)
    worst_abs, worst_rel, worst_key = 0.0, 0.0, ""
    n_diff = 0
    for k in sd_s:
        a, b = sd_s[k].float(), sd_d[k].float()
        d_abs = (a - b).abs().max().item()
        denom = a.abs().max().item()
        d_rel = d_abs / denom if denom > 0 else 0.0
        if d_abs > 0:
            n_diff += 1
        if d_abs > worst_abs:
            worst_abs, worst_rel, worst_key = d_abs, d_rel, k
    params_ok = worst_rel < tol
    checksum_same = single["final_checksum"] == ddp["final_checksum"]

    l2_err = abs(single["final_l2"] - ddp["final_l2"])

    # 双卡各 rank 本地 loss 应当不同（数据不同），但聚合后与单卡一致
    per_rank = ddp["history"].get("per_rank_loss", [])
    ranks_differ = any(len(set(p["losses"])) > 1 for p in per_rank) if per_rank else None

    return {
        "level1_init_checksum": {
            "single": single["init_checksum"], "ddp": ddp["init_checksum"],
            "match": init_ok},
        "level2_loss_trajectory": {
            "n_points_single": len(s_loss), "n_points_ddp": len(d_loss),
            "max_abs_err": round(loss_max_err, 8), "tol": tol, "match": loss_ok,
            "first3_single": s_loss[:3], "first3_ddp": d_loss[:3]},
        "level3_final_params": {
            "n_tensors": len(sd_s),
            "n_tensors_with_any_diff": n_diff,
            "max_abs_err": worst_abs,
            "max_rel_err": worst_rel,
            "worst_tensor": worst_key,
            "tol": tol, "match": params_ok,
            "final_checksum_identical": checksum_same,
            "note": ("checksum 相同则参数逐位一致；不同但 rel err 在容差内，"
                     "是 fp32 求和顺序差异，等价仍成立")},
        "final_l2": {"single": single["final_l2"], "ddp": ddp["final_l2"],
                     "abs_err": round(l2_err, 6)},
        "per_rank_local_loss_differs": ranks_differ,
        "overall_pass": init_ok and loss_ok and params_ok,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--global-batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--backend", default="gloo")
    ap.add_argument("--nproc", type=int, default=2)
    ap.add_argument("--master-port", type=int, default=29521)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--workdir", default="runs/parity")
    ap.add_argument("--out", default=None)
    ap.add_argument("--data-dir", default="exp_scale/data")
    args = ap.parse_args()

    wd = PROJ / args.workdir
    wd.mkdir(parents=True, exist_ok=True)
    single_json = wd / "single.json"
    ddp_json = wd / "ddp.json"
    single_params = wd / "single_params.pt"
    ddp_params = wd / "ddp_params.pt"

    py = str(PROJ / ".venv" / "bin" / "python")
    torchrun = str(PROJ / ".venv" / "bin" / "torchrun")
    common = ["--steps", str(args.steps), "--global-batch", str(args.global_batch),
              "--seq", str(args.seq), "--data-dir", args.data_dir,
              "--record-every", "1", "--eval-every", "0"]

    print("=== 1/2 单进程基线 ===")
    single = _run([py, "-m", "exp_ddp.ddp_train", "--mode", "single",
                   "--save-params", str(single_params),
                   "--out", str(single_json)] + common, single_json)

    print("=== 2/2 双 rank DDP（%s）===" % args.backend)
    ddp = _run([torchrun, f"--nproc_per_node={args.nproc}",
                f"--master_port={args.master_port}",
                "-m", "exp_ddp.ddp_train", "--mode", "ddp",
                "--backend", args.backend,
                "--save-params", str(ddp_params),
                "--out", str(ddp_json)] + common,
               ddp_json)

    cmp = compare(single, ddp, args.tol, single_params, ddp_params)
    result = {
        "config": {"steps": args.steps, "global_batch": args.global_batch,
                   "seq": args.seq, "backend": args.backend,
                   "nproc": args.nproc, "tol": args.tol},
        "single": {"final_checksum": single["final_checksum"],
                   "final_l2": single["final_l2"],
                   "final_global_loss": single["final_global_loss"],
                   "params_m": single["params_m"], "wall_s": single["wall_s"]},
        "ddp": {"final_checksum": ddp["final_checksum"],
                "final_l2": ddp["final_l2"],
                "final_global_loss": ddp["final_global_loss"],
                "world_size": ddp["world_size"], "wall_s": ddp["wall_s"]},
        "comparison": cmp,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    print("\n=== 判定：%s ===" % ("PASS 三级对齐" if cmp["overall_pass"] else "FAIL"))
    sys.exit(0 if cmp["overall_pass"] else 1)


if __name__ == "__main__":
    main()
