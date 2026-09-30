"""ckpt_mesh.py —— 分片 checkpoint：按参数清单存取，支持换 mesh 重组。

manifest 记录每个参数的全量形状、切分维度与全量 checksum（23 篇结论的
组合推广：元数据是分片 checkpoint 的一部分）。换 mesh 重组：各源分片按
rank 顺序拼回全量，再按新网格的 tp/tp_idx 重切。
缺 manifest（--bug nometa）→ 加载失败：规格要求的失败案例 2。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import torch.distributed as dist


def shard_dim_of(name: str) -> int | None:
    """TP 参数的切分维度：col 层切行(0)、row 层切列(1)，其余复制。"""
    if name.endswith(("q.weight", "q.bias", "k.weight", "k.bias",
                      "v.weight", "v.bias", "fc1.weight", "fc1.bias")):
        return 0
    if name.endswith(("o.weight", "fc2.weight")):
        return 1
    return None


def _param_keys(module, my_blocks: list[int]) -> list[str]:
    keys = []
    for n, _ in module.named_parameters():
        # 形如 blocks.<stage 内序号>.<名字> → <全局块号>.<名字>
        parts = n.split(".", 2)
        keys.append(f"{my_blocks[int(parts[1])]}.{parts[2]}")
    return keys


def _assemble(files: list[dict], meta: dict) -> dict:
    """按坐标装配全量参数：DP 副本只取一份，TP 分片沿切分维拼接。

    rank 文件带 coords=(dp,pp,tp)。同一 (dp=0, pp) 的 tp 组拼接才是
    全量；把 DP 副本也拼进去是错的（E3 实测：view 形状错、数值悄悄变多）。
    """
    full_params = {}
    keys = set()
    for f in files:
        keys.update(f["params"].keys())
    for k in sorted(keys):
        dim = meta["shard_dims"].get(k)
        owners = []
        for f in files:
            d, p, t = f["coords"]
            if d == 0 and k in f["params"]:
                owners.append((p, t, f["params"][k]))
        pp0 = owners[0][0]
        parts = [t for (p, tt, t) in sorted(owners, key=lambda o: o[1])
                 if p == pp0]
        full_params[k] = parts[0] if dim is None else torch.cat(parts, dim=dim)
    return full_params


def save_ckpt(dirpath: str, module, mesh, rank: int, my_blocks: list[int],
              no_meta: bool = False) -> dict:
    d = Path(dirpath)
    d.mkdir(parents=True, exist_ok=True)
    keys = _param_keys(module, my_blocks)
    shard = {k: p.detach().float().cpu().clone()
             for k, (_, p) in zip(keys, module.named_parameters())}
    torch.save({"coords": mesh.coords(rank), "params": shard},
               d / f"shard.rank{rank}.pt")
    if mesh.world > 1:
        dist.barrier()

    if rank == 0:
        all_shards = [torch.load(d / f"shard.rank{r}.pt", map_location="cpu")
                      for r in range(mesh.world)]
        full_params = _assemble(all_shards,
                                {"mesh": {"tp": mesh.tp}, "shard_dims": {
                                    k: shard_dim_of(k) for k in shard}})
        shapes = {k: list(t.shape) for k, t in full_params.items()}
        dims = {k: shard_dim_of(k) for k in full_params}
        blob = b"".join(full_params[k].numpy().tobytes()
                        for k in sorted(full_params))
        sha = hashlib.sha256(blob).hexdigest()[:16]
        meta = {
            "mesh": {"dp": mesh.dp, "tp": mesh.tp, "pp": mesh.pp,
                     "world": mesh.world},
            "param_shapes": shapes,
            "shard_dims": dims,
            "full_checksum": sha,
            "note": "元数据是分片 checkpoint 的一部分（23 篇结论）",
        }
        if no_meta:
            (d / "manifest.json").unlink(missing_ok=True)
        else:
            (d / "manifest.json").write_text(
                json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    if mesh.world > 1:
        dist.barrier()
    return {"saved": True, "no_meta": no_meta,
            "n_params": len(shard), "rank": rank}


def load_ckpt(dirpath: str, module, mesh, rank: int,
              my_blocks: list[int]) -> dict:
    d = Path(dirpath)
    mf = d / "manifest.json"
    if not mf.exists():
        return {"loaded": False, "error_type": "MissingManifest",
                "error_msg": ("manifest.json missing: 参数形状与切分维度未知，"
                              "无法按 mesh 重组")}
    meta = json.loads(mf.read_text(encoding="utf-8"))
    src = meta["mesh"]
    all_shards = [torch.load(d / f"shard.rank{r}.pt", map_location="cpu")
                  for r in range(src["world"])]
    full_params = _assemble(all_shards, meta)
    blob = b"".join(full_params[k].numpy().tobytes()
                    for k in sorted(full_params))
    sha = hashlib.sha256(blob).hexdigest()[:16]

    keys = _param_keys(module, my_blocks)
    with torch.no_grad():
        for (n, p), k in zip(module.named_parameters(), keys):
            full = full_params[k]
            dim = meta["shard_dims"].get(k)
            if dim is None:
                p.copy_(full.view(p.shape))
            else:
                shard_w = full.shape[dim] // mesh.tp
                _, _, tp_idx = mesh.coords(rank)
                idx = [slice(None)] * full.dim()
                idx[dim] = slice(tp_idx * shard_w, (tp_idx + 1) * shard_w)
                p.copy_(full[tuple(idx)].contiguous().view(p.shape))
    return {"loaded": True, "checksum_match": sha == meta["full_checksum"],
            "full_checksum": sha, "src_mesh": src,
            "mesh_changed": not (src["dp"] == mesh.dp and src["tp"] == mesh.tp
                                 and src["pp"] == mesh.pp)}
