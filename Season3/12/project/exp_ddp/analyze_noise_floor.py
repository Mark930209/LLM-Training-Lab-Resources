"""故障偏差 vs 噪声底：等价配置两次跑的参数差是噪声底，故障偏差必须显著高于它。

发现背景：continuous_30 与 ddp_none 配置完全等价（同 seed 同数据同 schedule），
final loss 六位小数一致，但 checksum 不同——GPU 原子归约的非确定性让参数差
ULP 量级，哈希就变。所以：
  1. checksum 相等不是等价判据（太严），容差参数比对才是（parity 已如此设计）
  2. 故障的"偏离"必须与噪声底对照才有意义

本脚本输出：
  A. 噪声底：ddp_none vs continuous_30 的参数 max_rel_err 与逐步 loss 差
  B. 各故障 vs ddp_none 的逐步 loss 轨迹偏差（首步、最大、末步）
  C. no_set_epoch vs ref_set_epoch 的逐步偏差（多 epoch 对照组）
"""
import json
import pathlib

import torch

R = pathlib.Path("results/Season3/12")
RUNS = pathlib.Path("runs/collect")


def L(n):
    return json.loads((R / f"{n}.json").read_text(encoding="utf-8"))


def loss_traj(d):
    return [h["loss"] for h in d["history"]["global_loss"]]


def param_err(p1, p2):
    a = torch.load(p1, map_location="cpu", weights_only=False)
    b = torch.load(p2, map_location="cpu", weights_only=False)
    worst_abs = worst_rel = 0.0
    for k in a:
        x, y = a[k].float(), b[k].float()
        da = (x - y).abs().max().item()
        den = x.abs().max().item()
        dr = da / den if den > 0 else 0.0
        worst_abs = max(worst_abs, da)
        worst_rel = max(worst_rel, dr)
    return worst_abs, worst_rel


print("=== A. 噪声底（等价配置两次跑：ddp_none vs continuous_30）===")
abs_err, rel_err = param_err(RUNS / "ddp_none_params.pt", RUNS / "continuous_params.pt")
print("param max_abs_err=%.3e max_rel_err=%.3e" % (abs_err, rel_err))
t0, t1 = loss_traj(L("ddp_none")), loss_traj(L("continuous_30"))
step_diffs = [abs(a - b) for a, b in zip(t0, t1)]
print("loss 逐步差 max=%.2e（六位小数下完全一致）" % max(step_diffs))
print("结论：checksum 不同但参数差在 1e-6 量级 = GPU 非确定性噪声底")

print("\n=== B. 各故障 vs ddp_none 的 loss 轨迹偏差 ===")
base = t0
for f in ("fault_no_sampler", "fault_global_batch", "fault_sum_reduction"):
    d = L(f)
    t = loss_traj(d)
    diffs = [abs(a - b) for a, b in zip(base, t)]
    print("%-24s 首步差=%.4f 最大差=%.4f 末步差=%.4f final_loss=%s"
          % (f, diffs[0], max(diffs), diffs[-1], d["final_global_loss"]))

print("\n=== C. no_set_epoch vs ref_set_epoch（多 epoch 对照）===")
ns, rf = loss_traj(L("fault_no_set_epoch")), loss_traj(L("ref_set_epoch"))
diffs = [round(abs(a - b), 5) for a, b in zip(ns, rf)]
first_div = next((i for i, x in enumerate(diffs) if x > 0), None)
print("首个非零差的 step =", first_div, "（epoch_steps=10，第 10 步进入第 2 个 epoch）")
print("偏差序列:", diffs)
print("末步差=%.5f final: no_set=%s ref=%s"
      % (diffs[-1], L("fault_no_set_epoch")["final_global_loss"],
         L("ref_set_epoch")["final_global_loss"]))

out = {
    "noise_floor": {"param_max_abs_err": abs_err, "param_max_rel_err": rel_err,
                    "loss_step_max_diff": max(step_diffs),
                    "note": "等价配置两次跑；checksum 不同但参数差在噪声底内"},
    "fault_traj_dev": {
        f: {"first": None, "max": None, "last": None}
        for f in ("fault_no_sampler", "fault_global_batch", "fault_sum_reduction")},
    "set_epoch_first_divergence_step": first_div,
}
for f in ("fault_no_sampler", "fault_global_batch", "fault_sum_reduction"):
    t = loss_traj(L(f))
    diffs = [abs(a - b) for a, b in zip(base, t)]
    out["fault_traj_dev"][f] = {"first": round(diffs[0], 6),
                                "max": round(max(diffs), 6),
                                "last": round(diffs[-1], 6),
                                "final_loss": L(f)["final_global_loss"]}
(R / "fault_deviation.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print("\n已写 results/Season3/12/fault_deviation.json")
