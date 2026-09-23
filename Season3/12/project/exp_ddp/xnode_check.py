"""Validate a two-host NCCL DDP run against a matching single-GPU baseline."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .parity_check import compare


def check(single: dict, rank0: dict, rank1: dict, single_params: Path,
          ddp_params: Path, tol: float) -> dict:
    runs = (single, rank0, rank1)
    required = ("corpus_sha256", "init_checksum", "final_checksum", "config",
                "history", "node_fingerprint", "gpu")
    if any(any(key not in run for key in required) for run in runs):
        raise ValueError("result is missing provenance or training fields")

    checks = {
        "ranks": (single.get("world_size"), single.get("rank"),
                  rank0.get("world_size"), rank0.get("rank"),
                  rank1.get("world_size"), rank1.get("rank")) == (1, 0, 2, 0, 2, 1),
        "nccl_cuda": all(run.get("backend") == "nccl" and run.get("device") == "cuda"
                         for run in (rank0, rank1)),
        "different_hosts": rank0["node_fingerprint"] != rank1["node_fingerprint"],
        "same_corpus": (len(single["corpus_sha256"]) == 64 and
                        len({run["corpus_sha256"] for run in runs}) == 1),
        "same_config": single["config"] == rank0["config"] == rank1["config"],
        "same_initial_weights": len({run["init_checksum"] for run in runs}) == 1,
        "ranks_finish_together": rank0["final_checksum"] == rank1["final_checksum"],
        "no_sanity_errors": all(not run.get("sanity_error") for run in runs),
    }
    trajectories = [run["history"]["global_loss"] for run in runs]
    checks["complete_loss_history"] = all(
        len(points) == single["config"]["steps"] and
        [point["step"] for point in points] == list(range(single["config"]["steps"])) and
        all(math.isfinite(point["loss"]) for point in points)
        for points in trajectories)
    checks["ranks_agree_on_loss"] = trajectories[1] == trajectories[2]

    if not all(checks.values()):
        return {"checks": checks, "overall_pass": False}

    parity = compare(single, rank0, tol, single_params, ddp_params)
    return {
        "checks": checks,
        "hardware": {"rank0": rank0["gpu"], "rank1": rank1["gpu"]},
        "software": {"rank0_torch": rank0["torch"], "rank1_torch": rank1["torch"]},
        "single_wall_s": single["wall_s"],
        "ddp_wall_s": {"rank0": rank0["wall_s"], "rank1": rank1["wall_s"]},
        "comparison": parity,
        "overall_pass": parity["overall_pass"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("single", "rank0", "rank1", "single-params", "ddp-params", "out"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--tol", type=float, default=1e-4)
    args = parser.parse_args()
    result = check(*(json.loads(path.read_text(encoding="utf-8"))
                     for path in (args.single, args.rank0, args.rank1)),
                   args.single_params, args.ddp_params, args.tol)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["overall_pass"] else 1)


if __name__ == "__main__":
    main()