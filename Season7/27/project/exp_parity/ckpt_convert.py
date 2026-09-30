"""ckpt_convert.py —— 四套 checkpoint 格式的互转与往返校验（E5）。

格式：
    hub        ：本 Lab 的扁平 state_dict（init_state.pt 同构 {"state_dict": ...}）
    llama3     ：TorchTitan Llama3Model 参数名（hub 的默认命名）
    megatron   ：megatron.core GPTModel 参数名（含 qkv/fc1 融合布局转换）
    ds_zero1   ：DeepSpeed save_checkpoint 目录（含 optimizer 状态 + mp_rank 分片）
    tt_dcp     ：TorchTitan torch.distributed.checkpoint 目录

用法：
    python ckpt_convert.py --from hub --to megatron --src init_state.pt --dst mega_init.pt
    python ckpt_convert.py --check-roundtrip --src init_state.pt
    python ckpt_convert.py --from ds_zero1 --to hub --src runs/ds_ckpt --dst ds_hub.pt
    python ckpt_convert.py --from ds_zero1 --to hub --src runs/ds_ckpt --dst x.pt --bug drop_opt
    python ckpt_convert.py --from ds_zero1 --to hub --src runs/ds_ckpt --dst x.pt --bug nometa

失败变体（失败案例 2 的病灶）：
    drop_opt：丢弃优化器状态再恢复训练 → loss 跳变/训偏（消费侧实证见 e5）；
    nometa ：丢弃分片元信息（manifest）→ 换 mesh/合并时报缺元数据错误。
"""

from __future__ import annotations

import argparse
import os

import torch


# ---- llama3 ↔ megatron 名字与布局映射 ----
def llama3_to_megatron(sd: dict) -> dict:
    """Llama3Model 参数名/布局 → megatron GPTModel。

    布局差异（隐性语义，逐字段转换）：
    - qkv：llama3 [3, H, D, h] 拼接 → megatron [H, 3, D, h]（逐头交错）；
    - fc1：llama3 w1(gate)/w3(up) 两矩阵 → megatron linear_fc1 [gate; up] 行拼接。
    """
    out = {}
    layers = sorted({int(k.split(".")[1]) for k in sd if k.startswith("layers.")})
    out["embedding.word_embeddings.weight"] = sd["tok_embeddings.weight"].clone()
    out["decoder.final_layernorm.weight"] = sd["norm.weight"].clone()
    out["output_layer.weight"] = sd["lm_head.weight"].clone()
    for i in layers:
        p = f"layers.{i}"
        q = f"decoder.layers.{i}"
        # llama3 实际键集为分立 wq/wk/wv（trainer 平行化后才融合为 wqkv）；
        # 统一先拼 [q; k; v] 块，再转 megatron 逐头交错 [H,3,D,h]
        if f"{p}.attention.qkv_linear.wqkv.weight" in sd:
            wqkv = sd[f"{p}.attention.qkv_linear.wqkv.weight"]
        else:
            wqkv = torch.cat([
                sd[f"{p}.attention.qkv_linear.wq.weight"],
                sd[f"{p}.attention.qkv_linear.wk.weight"],
                sd[f"{p}.attention.qkv_linear.wv.weight"],
            ], dim=0)
        h3, hidden = wqkv.shape
        n_heads = 16                 # debugmodel：16 头、head_dim = hidden/16
        head_dim = hidden // n_heads
        # llama3 fused: [q; k; v] 各 [n_heads, head_dim, hidden] 展平
        qkv = wqkv.view(3, n_heads, head_dim, hidden)
        qkv = qkv.permute(1, 0, 2, 3).reshape(h3, hidden)   # 逐头交错 [q,k,v]
        out[f"{q}.self_attention.linear_qkv.weight"] = qkv.contiguous()
        out[f"{q}.self_attention.linear_proj.weight"] = \
            sd[f"{p}.attention.wo.weight"].clone()
        w1 = sd[f"{p}.feed_forward.w1.weight"]   # gate
        w3 = sd[f"{p}.feed_forward.w3.weight"]   # up
        out[f"{q}.mlp.linear_fc1.weight"] = torch.cat([w1, w3], dim=0).contiguous()
        out[f"{q}.mlp.linear_fc2.weight"] = \
            sd[f"{p}.feed_forward.w2.weight"].clone()
        # norm 位置两种布局：无 TE 为 input_layernorm/pre_mlp_layernorm，
        # TE 融合布局为 linear_qkv.layer_norm_weight——映射时两种都写（同值）
        out[f"{q}.input_layernorm.weight"] = \
            sd[f"{p}.attention_norm.weight"].clone()
        out[f"{q}.pre_mlp_layernorm.weight"] = \
            sd[f"{p}.ffn_norm.weight"].clone()
        out[f"{q}.self_attention.linear_qkv.layer_norm_weight"] = \
            sd[f"{p}.attention_norm.weight"].clone()
        out[f"{q}.mlp.linear_fc1.layer_norm_weight"] = \
            sd[f"{p}.ffn_norm.weight"].clone()
    return out


def megatron_to_llama3(sd: dict, ref: dict | None = None) -> dict:
    """反向转换（布局逆操作）。ref 提供 llama3 形状参考。"""
    out = {}
    out["tok_embeddings.weight"] = sd["embedding.word_embeddings.weight"].clone()
    out["norm.weight"] = sd["decoder.final_layernorm.weight"].clone()
    out["lm_head.weight"] = sd["output_layer.weight"].clone()
    layers = sorted({int(k.split(".")[2]) for k in sd if k.startswith("decoder.layers.")})
    for i in layers:
        p = f"layers.{i}"
        q = f"decoder.layers.{i}"
        qkv = sd[f"{q}.self_attention.linear_qkv.weight"]
        h3, hidden = qkv.shape
        n_heads = 16
        head_dim = hidden // n_heads
        # 逐头交错 [H,3,D,h] → [q; k; v] 块 → 分立 wq/wk/wv（hub 规范键）
        t = qkv.view(n_heads, 3, head_dim, hidden)
        t = t.permute(1, 0, 2, 3).reshape(h3, hidden)
        out[f"{p}.attention.qkv_linear.wq.weight"] = t[:hidden].contiguous()
        out[f"{p}.attention.qkv_linear.wk.weight"] = \
            t[hidden:2 * hidden].contiguous()
        out[f"{p}.attention.qkv_linear.wv.weight"] = \
            t[2 * hidden:].contiguous()
        out[f"{p}.attention.wo.weight"] = \
            sd[f"{q}.self_attention.linear_proj.weight"].clone()
        fc1 = sd[f"{q}.mlp.linear_fc1.weight"]
        half = fc1.shape[0] // 2
        out[f"{p}.feed_forward.w1.weight"] = fc1[:half].clone()
        out[f"{p}.feed_forward.w3.weight"] = fc1[half:].clone()
        out[f"{p}.feed_forward.w2.weight"] = \
            sd[f"{q}.mlp.linear_fc2.weight"].clone()
        out[f"{p}.attention_norm.weight"] = sd.get(
            f"{q}.input_layernorm.weight",
            sd.get(f"{q}.self_attention.linear_qkv.layer_norm_weight")).clone()
        out[f"{p}.ffn_norm.weight"] = sd.get(
            f"{q}.pre_mlp_layernorm.weight",
            sd.get(f"{q}.mlp.linear_fc1.layer_norm_weight")).clone()
    return out


def load_hub(path: str) -> dict:
    obj = torch.load(path, weights_only=False)
    return obj["state_dict"] if "state_dict" in obj else obj


def load_ds_zero1(ckpt_dir: str, bug: str = "") -> dict:
    """DeepSpeed save_checkpoint 目录 → hub 格式。

    含 model_states + optim_states；bug=drop_opt 时故意丢弃优化器状态
    （失败案例 2：恢复后 optimizer 动量清零，训练轨迹跳变）；
    bug=nometa 时丢弃 zero_pp_rank 元信息文件（模拟分片元信息缺失）。
    """
    import glob
    mp = sorted(glob.glob(os.path.join(ckpt_dir, "**", "mp_rank_*_model_states.pt"),
                          recursive=True))
    assert mp, f"no model_states in {ckpt_dir}"
    blob = torch.load(mp[0], weights_only=False)
    sd = blob["module"]
    out = {"state_dict": sd}
    if bug == "nometa":
        out["manifest"] = None
        return out
    fp = sorted(glob.glob(os.path.join(ckpt_dir, "**", "*_optim_states.pt"),
                          recursive=True))
    if bug == "drop_opt" or not fp:
        out["optimizer_state"] = None
    else:
        out["optimizer_state"] = "kept" if fp else None
        out["optim_files"] = [os.path.basename(x) for x in fp]
    out["manifest"] = {"format": "ds_zero1",
                       "param_count": sum(v.numel() for v in sd.values())}
    return out


def save_hub(obj: dict, path: str) -> None:
    torch.save(obj, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src_fmt", default="hub",
                    choices=["hub", "llama3", "megatron", "ds_zero1"])
    ap.add_argument("--to", dest="dst_fmt", default="hub",
                    choices=["hub", "llama3", "megatron"])
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default="")
    ap.add_argument("--bug", choices=["", "drop_opt", "nometa"], default="")
    ap.add_argument("--check-roundtrip", action="store_true")
    args = ap.parse_args()

    if args.check_roundtrip:
        sd = load_hub(args.src)
        mid = llama3_to_megatron(sd)
        back = megatron_to_llama3(mid)
        bad = []
        for k in sd:
            d = (sd[k].float() - back[k].float()).abs().max().item()
            if d != 0.0:
                bad.append((k, d))
        import parity_common as pc
        print("roundtrip hub→megatron→hub:")
        print("  checksum 原:", pc.sd_checksum(sd))
        print("  checksum 还:", pc.sd_checksum(back))
        print("  非零差异参数:", len(bad), bad[:5])
        return

    if args.src_fmt == "ds_zero1":
        obj = load_ds_zero1(args.src, bug=args.bug)
        sd = obj["state_dict"]
    else:
        sd = load_hub(args.src)
        obj = {"state_dict": sd}

    if args.dst_fmt == "megatron":
        obj = {"state_dict": llama3_to_megatron(sd)}
    elif args.dst_fmt == "llama3" and args.src_fmt == "megatron":
        obj = {"state_dict": megatron_to_llama3(sd)}

    if args.dst:
        save_hub(obj, args.dst)
        import parity_common as pc
        print(f"converted {args.src_fmt} -> {args.dst_fmt} "
              f"checksum {pc.sd_checksum(obj['state_dict'])}"
              + (f" bug={args.bug}" if args.bug else ""))


if __name__ == "__main__":
    main()
