"""量化 all_rank_save 代价与各故障的 checksum 偏离。只读 results。"""
import json
import pathlib

R = pathlib.Path("results/Season3/12")


def L(n):
    return json.loads((R / f"{n}.json").read_text(encoding="utf-8"))


ars = L("fault_all_rank_save")
ck = L("ckpt_first_half")
print("=== checkpoint 写法对照 ===")
print("rank0_save  writers=%s size=%d loadable=%s"
      % (ck["inject"].get("ckpt_writers"), ck["ckpt"]["size_bytes"], ck["ckpt"]["loadable"]))
print("all_rank    writers=%s size=%d loadable=%s"
      % (ars["inject"].get("ckpt_writers"), ars["ckpt"]["size_bytes"], ars["ckpt"]["loadable"]))
print("冗余写入倍数 = writers = %s（每个 rank 都写一份完整 %d MB）"
      % (ars["inject"].get("ckpt_writers"), ars["ckpt"]["size_bytes"] // 1024 // 1024))

base = L("ddp_none")["final_checksum"]
print("\n=== final_checksum 对齐（对 ddp_none 基线）===")
for f in ("ddp_none", "fault_no_sampler", "fault_global_batch",
          "fault_sum_reduction", "fault_no_set_epoch", "ref_set_epoch",
          "continuous_30"):
    d = L(f)
    tag = "== base" if d["final_checksum"] == base else "!= base (偏离)"
    print("%-24s %s  %s" % (f, d["final_checksum"], tag))

print("\n=== 各故障 final_global_loss（基线 ddp_none）===")
bl = L("ddp_none")["final_global_loss"]
for f in ("fault_no_sampler", "fault_global_batch", "fault_sum_reduction",
          "fault_no_set_epoch", "ref_set_epoch"):
    d = L(f)
    fl = d["final_global_loss"]
    dev = fl - bl
    print("%-24s loss=%-14s dev=%+.4f" % (f, fl, dev))
