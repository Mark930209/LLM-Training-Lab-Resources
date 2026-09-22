# DevResources/Season3/11 —— 跨机双卡环境 Lab（网络与 NCCL 验证）

对应文章：`Articles/Season3/11_如何在 Windows WSL 上进行多卡训练.md`

## 本篇解决什么问题

把两台各带一块消费级 GPU 的 Windows 机器配成一个可用的多卡训练环境。这是一篇动手教程：读者跟着操作把环境配齐，验收标准是跨机 `init_process_group(nccl)` 握手成功、`all_reduce` 数学结果正确、能测出可用带宽。

核心判断：多卡训练的门槛不在训练代码，而在网络地址。NCCL 的 bootstrap 阶段让各 rank 互相通告自己的真实 socket 地址，随后按这些地址直连建立数据面，端口由 NCCL 动态协商。只要 WSL2 处在 WinNAT 后面，出站流量就会被 SNAT 改写成宿主 Windows 的 LAN 地址，对端拿到的地址是假的，数据面必然建不起来。正解是 Win11 22H2+ 的 mirrored 网络模式，让 WSL 直接持有 LAN 地址。

## 包内容

```text
11/
└── scripts/
    ├── nccl_cross.py            # 跨机 NCCL 最小验证：all_reduce 正确性 + 双向 p2p + 分尺寸带宽（带宽计算已修正）
    ├── lan_throughput.py        # LAN TCP 吞吐探测（server / client 两端）
    ├── local_listen_probe.sh    # 本机 WSL 监听探针（29802）
    ├── local_connect_probe.sh   # 本机 WSL 主动连远端探针（29801）
    ├── remote_listen_probe.sh   # 远端 WSL 监听探针（29801）
    ├── remote_connect_probe.sh  # 远端 WSL 主动连本机探针（29802）
    ├── local_version_probe.sh   # 本机 torch/NCCL 运行时版本探测（ctypes ncclGetVersion）
    ├── remote_version_probe.sh  # 远端 torch/NCCL 运行时版本探测（含 LD_PRELOAD 对照）
    ├── local_rank0.sh           # rank0（本机）torchrun 启动脚本（mirrored + eth0）
    ├── remote_rank1.sh          # rank1（远端）torchrun 启动脚本（LD_PRELOAD 对齐 NCCL）
    ├── probe_remote_node.sh     # 远端节点能力探测：GPU / PyTorch / NCCL / 网络
    ├── remote_env_card.ps1      # 远端 Windows 侧环境档案采集（build / 网卡 / 防火墙 / portproxy）
    └── enable_wsl_lan_routing.ps1  # 尝试③（IP 转发 + 静态路由）配置脚本，含 -Undo；方案已证伪，保留作排查参考
```

## 四次尝试与脚本对应关系

| 尝试 | 结论 | 相关脚本 |
|---|---|---|
| ① SSH 反向隧道 | rendezvous 通、bootstrap 卡死 | `nccl_cross.py`（NAT 时代命令见 results 档案） |
| ② Tailscale VPN | 正确性通过，但走 DERP 中继，性能不可用 | `nccl_cross.py` + `remote_version_probe.sh`（NCCL 对齐） |
| ③ IP 转发 + 静态路由 | 包到达了，但源地址被 SNAT 改写 | `enable_wsl_lan_routing.ps1` |
| ④ Win11 mirrored 模式 | 全项通过，约 920 Mbps | 全部脚本（`.wslconfig` 配置见文章 §4.3） |

## 前置条件

- 两台带 NVIDIA GPU 的机器，同一局域网。
- 两侧 Windows 11 22H2+（mirrored 模式硬要求）；Win10 读者没有第④条路，可复现①~③的失败链。
- 两侧 WSL2 Ubuntu，各自装好 PyTorch（02 篇环境）。
- 两侧运行时 NCCL 版本必须一致；不一致时按文章 §6.2 的三个坑对齐（`LD_PRELOAD` 是唯一有效手段）。

## mirrored 配置（两侧 `.wslconfig`）

```ini
[wsl2]
networkingMode=mirrored

[experimental]
hostAddressLoopback=true
```

改完执行 `wsl --shutdown` 重启 WSL 生效。干净环境（防火墙开启）下还需放行 Hyper-V 防火墙入站，命令见文章 §4.5。

## 复现顺序

1. `probe_remote_node.sh`：确认远端 GPU / torch / NCCL 可用（`ssh <host> 'bash -s' < probe_remote_node.sh`）
2. `local_version_probe.sh` / `remote_version_probe.sh`：对齐两侧运行时 NCCL
3. 四个探针脚本：验证 WSL ↔ WSL 双向直连
4. `local_rank0.sh` + `remote_rank1.sh`：跨机 NCCL 全项验证
5. `lan_throughput.py`：裸 TCP 吞吐对照，定位带宽瓶颈归属

配完逐项对照文章 §9.1 的 9 项环境配齐检查清单自查；连不上时按 §9.2 的五步定位法排障。

## 注意事项

- 所有 `.sh` 脚本必须 **LF 换行**（经 `ssh 'bash -s'` 传入时 CRLF 会报 `$'\r': command not found`）。
- 所有 `.ps1` 脚本必须 **UTF-8 带 BOM**（含中文时 Windows PowerShell 5.1 按 GBK 解析会乱码）。
- `nccl_cross.py` 的带宽口径：`dt` 是 iters 次迭代总时间，`algbw = algo_bytes * iters / dt / 1e9`；world=2 时 algbw == busbw。
