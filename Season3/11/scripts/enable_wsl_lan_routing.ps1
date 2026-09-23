#Requires -RunAsAdministrator
<#
.SYNOPSIS
    让两台 Windows 各自的 WSL2 NAT 子网在 IP 层互相可路由。

.DESCRIPTION
    Win10 上 WSL2 只有 NAT 网络模式，两台机器的 WSL 各自躲在独立 NAT 后面，
    互相 100% 丢包。NCCL 的数据面地址由各 rank 自己通告、没有 NAT 改写机制，
    所以端口转发（netsh portproxy / SSH 隧道）盖不住它动态协商的数据通道端口。

    唯一可行的办法是在 IP 层打通：两台 Windows 都开启转发、各自加一条指向
    对端 WSL 子网的静态路由，让 WSL 通告的私有地址在对端真的可达。

    本脚本在**每台 Windows 上各跑一次**，参数不同。

.PARAMETER PeerWindowsIp
    对端 Windows 的 LAN IP（例如本机跑时填远端的 192.168.0.126）。

.PARAMETER PeerWslSubnet
    对端 WSL 的 NAT 子网 CIDR（例如 172.19.224.0/20）。
    在对端机器上用 `wsl hostname -I` 加前缀长度确认。

.PARAMETER Undo
    撤销本脚本做的全部改动。

.NOTES
    重要限制：WSL2 的 NAT 子网是**动态分配**的，重启 Windows 或 `wsl --shutdown`
    后可能变化。子网一变，静态路由就失效，需要重跑本脚本。
    Win11 22H2+ 的 mirrored 网络模式没有这个问题（WSL 直接拿 LAN 地址）。
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, ParameterSetName = 'Apply')]
    [string]$PeerWindowsIp,

    [Parameter(Mandatory = $true, ParameterSetName = 'Apply')]
    [string]$PeerWslSubnet,

    [Parameter(Mandatory = $true, ParameterSetName = 'Undo')]
    [switch]$Undo
)

$ErrorActionPreference = 'Stop'

$RuleName = 'LLM-Training-Lab WSL LAN routing'
$StateFile = Join-Path $PSScriptRoot 'wsl_lan_routing_state.json'
$RegPath = 'HKLM:\SYSTEM\CurrentControlSet\Services\Tcpip\Parameters'

function Write-Step($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Write-Ok($msg) { Write-Host "    $msg" -ForegroundColor Green }
function Write-Warn2($msg) { Write-Host "    $msg" -ForegroundColor Yellow }

# CIDR 前缀长度 -> 点分掩码（route.exe 只认掩码，不认 /20）
function ConvertTo-DottedMask([int]$PrefixLen) {
    $octets = @(0, 0, 0, 0)
    $remaining = $PrefixLen
    for ($i = 0; $i -lt 4; $i++) {
        $bits = [Math]::Min(8, $remaining)
        $octets[$i] = if ($bits -eq 0) { 0 } else { [int](256 - [Math]::Pow(2, 8 - $bits)) }
        $remaining -= $bits
    }
    return ($octets -join '.')
}

# ---------- 探测本机拓扑 ----------
function Get-Topology {
    $wslAdapter = Get-NetAdapter | Where-Object {
        $_.InterfaceDescription -like '*Hyper-V Virtual Ethernet*' -and
        $_.Name -like 'vEthernet (WSL*' -and
        $_.Status -eq 'Up'
    } | Select-Object -First 1

    if (-not $wslAdapter) {
        throw '找不到 vEthernet (WSL) 适配器。确认 WSL 正在运行（wsl -l -v 应显示 Running）。'
    }

    $wslRoute = Get-NetRoute -AddressFamily IPv4 -InterfaceIndex $wslAdapter.ifIndex |
        Where-Object { $_.DestinationPrefix -like '172.*' -and $_.DestinationPrefix -like '*/2*' } |
        Sort-Object { [int]($_.DestinationPrefix -split '/')[1] } |
        Select-Object -First 1

    if (-not $wslRoute) {
        throw "找不到 vEthernet (WSL) 上的 172.x 子网路由。WSL 可能未启动。"
    }

    return [pscustomobject]@{
        WslIfIndex   = $wslAdapter.ifIndex
        WslIfName    = $wslAdapter.Name
        WslSubnet    = $wslRoute.DestinationPrefix
    }
}

# ---------- 撤销 ----------
if ($Undo) {
    Write-Step '撤销 WSL LAN 路由改动'

    $saved = $null
    if (Test-Path $StateFile) {
        $saved = Get-Content $StateFile -Raw | ConvertFrom-Json
    } else {
        Write-Warn2 "未找到状态文件 $StateFile，将按保守策略还原。"
    }

    Get-NetFirewallRule -DisplayName $RuleName -ErrorAction SilentlyContinue | ForEach-Object {
        Remove-NetFirewallRule -InputObject $_
        Write-Ok "已删防火墙规则 [$($_.Direction) / $($_.DisplayName)]"
    }

    # 只删本脚本加的那条对端 WSL 子网路由，不碰其他 172.x 路由
    $targets = @()
    if ($saved -and $saved.PeerWslSubnet) { $targets += $saved.PeerWslSubnet }
    foreach ($prefix in $targets) {
        Get-NetRoute -AddressFamily IPv4 -DestinationPrefix $prefix -ErrorAction SilentlyContinue |
            ForEach-Object {
                try {
                    Remove-NetRoute -InputObject $_ -Confirm:$false -ErrorAction Stop
                    Write-Ok "已删路由: $($_.DestinationPrefix) via $($_.NextHop)"
                } catch {
                    Write-Warn2 "删路由 $($_.DestinationPrefix) 失败: $($_.Exception.Message)"
                }
            }

        # route -p add 写入的持久化路由必须用 route -p delete 清，
        # Remove-NetRoute 只动活动表。
        $net, $plen = $prefix.Split('/')
        $mask = ConvertTo-DottedMask ([int]$plen)

        $prevEAP = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        $delOut = (& route -p delete $net mask $mask 2>&1 | Out-String)
        $delExit = $LASTEXITCODE
        $ErrorActionPreference = $prevEAP

        if ($delExit -eq 0) {
            Write-Ok "已删持久化路由: $net mask $mask"
        } else {
            Write-Warn2 "route -p delete 未生效（可能本就不存在）: $($delOut.Trim())"
        }
    }

    if ($saved) {
        foreach ($item in $saved.Forwarding) {
            Set-NetIPInterface -InterfaceIndex $item.IfIndex -Forwarding $item.Before -ErrorAction SilentlyContinue
            Write-Ok "已还原接口 ifIndex=$($item.IfIndex) 转发状态为 $($item.Before)"
        }
        Set-ItemProperty -Path $RegPath -Name IPEnableRouter -Value $saved.IPEnableRouterBefore -Type DWord
        Write-Ok "已还原 IPEnableRouter 为 $($saved.IPEnableRouterBefore)"
        Remove-Item $StateFile -Force
        Write-Ok "已删状态文件"
    } else {
        Set-ItemProperty -Path $RegPath -Name IPEnableRouter -Value 0 -Type DWord
        Write-Warn2 'IPEnableRouter 已置 0；接口转发状态未还原（无记录），如需请手动 Set-NetIPInterface -Forwarding Disabled。'
    }

    Write-Host "`n撤销完成。" -ForegroundColor Green
    return
}

# ---------- 应用 ----------
$topo = Get-Topology

Write-Step '本机拓扑'
Write-Ok "WSL 适配器 : $($topo.WslIfName) (ifIndex $($topo.WslIfIndex))"
Write-Ok "本机 WSL 子网: $($topo.WslSubnet)"
Write-Ok "对端 Windows : $PeerWindowsIp"
Write-Ok "对端 WSL 子网: $PeerWslSubnet"

# 找到通往对端的物理网卡
$nr = Find-NetRoute -RemoteIPAddress $PeerWindowsIp | Select-Object -First 1
if (-not $nr) { throw "找不到通往 $PeerWindowsIp 的路由。确认两台机器在同一 LAN。" }
$physIfIndex = $nr.InterfaceIndex
$physIfAlias = (Get-NetAdapter -InterfaceIndex $physIfIndex).Name
$localIp = $nr.IPAddress

Write-Ok "本机 LAN IP : $localIp"
Write-Ok "物理网卡    : $physIfAlias (ifIndex $physIfIndex)"

# 记录改动前状态，供 Undo 还原
$before = [pscustomobject]@{
    IPEnableRouterBefore = (Get-ItemProperty -Path $RegPath -Name IPEnableRouter -ErrorAction SilentlyContinue).IPEnableRouter
    Forwarding           = @(
        [pscustomobject]@{ IfIndex = $physIfIndex; Before = (Get-NetIPInterface -InterfaceIndex $physIfIndex -AddressFamily IPv4).Forwarding }
        [pscustomobject]@{ IfIndex = $topo.WslIfIndex; Before = (Get-NetIPInterface -InterfaceIndex $topo.WslIfIndex -AddressFamily IPv4).Forwarding }
    )
    LocalWslSubnet       = $topo.WslSubnet
    PeerWslSubnet        = $PeerWslSubnet
    PeerWindowsIp        = $PeerWindowsIp
    Timestamp            = (Get-Date).ToString('o')
}
if (-not $before.IPEnableRouterBefore) { $before.IPEnableRouterBefore = 0 }
$before | ConvertTo-Json -Depth 5 | Set-Content $StateFile -Encoding UTF8
Write-Ok "改动前状态已存: $StateFile"

# 1. 开启 IP 转发
Write-Step '开启 IP 转发'
Set-NetIPInterface -InterfaceIndex $physIfIndex -Forwarding Enabled
Write-Ok "$physIfAlias -> Enabled"
Set-NetIPInterface -InterfaceIndex $topo.WslIfIndex -Forwarding Enabled
Write-Ok "$($topo.WslIfName) -> Enabled"
Set-ItemProperty -Path $RegPath -Name IPEnableRouter -Value 1 -Type DWord
Write-Ok 'IPEnableRouter = 1（重启后仍生效）'

# 2. 加静态路由：对端 WSL 子网 -> 对端 Windows
#
#    踩坑记录（两个都是真实踩到的）：
#    a) New-NetRoute 不接受 -PolicyStore PersistentStore，报
#       "Invalid parameter PolicyStore PersistentStore" / Windows System Error 87。
#    b) 先用 New-NetRoute 写 ActiveStore、再用 route -p add 补持久化，会冲突：
#       route.exe 看到同前缀路由已存在就拒绝，报"对象已存在"。
#    正确做法是只用 route -p add —— 它一次同时写入活动表和持久化表。
#
#    另一个坑：脚本顶部 ErrorActionPreference=Stop 时，route.exe 往 stderr 写的
#    中文提示会被 PowerShell 包成 NativeCommandError 终止性错误，脚本直接中断。
#    所以调用 route.exe 期间临时降级为 Continue，改用退出码判断成败。
Write-Step "添加静态路由 $PeerWslSubnet -> $PeerWindowsIp"

$net, $plen = $PeerWslSubnet.Split('/')
$mask = ConvertTo-DottedMask ([int]$plen)

$prevEAP = $ErrorActionPreference
$ErrorActionPreference = 'Continue'

# 先清同前缀旧路由（活动表 + 持久化表都清），保证脚本可重复执行
$existing = Get-NetRoute -AddressFamily IPv4 -DestinationPrefix $PeerWslSubnet -ErrorAction SilentlyContinue
if ($existing) {
    $existing | Remove-NetRoute -Confirm:$false -ErrorAction SilentlyContinue
    Write-Warn2 '已移除活动表中的同前缀路由'
}
$null = & route delete $net mask $mask 2>&1
$null = & route -p delete $net mask $mask 2>&1

$addOut = (& route -p add $net mask $mask $PeerWindowsIp metric 1 if $physIfIndex 2>&1 | Out-String)
$addExit = $LASTEXITCODE

$ErrorActionPreference = $prevEAP

if ($addExit -ne 0) {
    throw "route -p add 失败 (exit $addExit): $($addOut.Trim())"
}
Write-Ok "route -p add 成功: $net mask $mask -> $PeerWindowsIp (if $physIfIndex)"

# 复核：不轻信命令返回，直接查路由表
$verify = Get-NetRoute -AddressFamily IPv4 -DestinationPrefix $PeerWslSubnet -ErrorAction SilentlyContinue
if (-not $verify) {
    throw "路由写入后复核失败：活动路由表里找不到 $PeerWslSubnet"
}
Write-Ok "活动表复核通过: $($verify.DestinationPrefix) via $($verify.NextHop) on $($verify.InterfaceAlias)"

$ErrorActionPreference = 'Continue'
$persist = (& route print -4 2>&1 | Out-String)
$ErrorActionPreference = $prevEAP

if ($persist -match '(?s)Persistent Routes:.*?' + [regex]::Escape($net)) {
    Write-Ok '持久化表复核通过（route print 的 Persistent Routes 段可见，重启后仍在）'
} else {
    Write-Warn2 '持久化表复核未通过：route print 的 Persistent Routes 段找不到该网段'
}

# 3. 防火墙：放行对端 WSL 子网的双向流量
Write-Step '添加防火墙规则'
Get-NetFirewallRule -DisplayName $RuleName -ErrorAction SilentlyContinue | Remove-NetFirewallRule

New-NetFirewallRule -DisplayName $RuleName -Direction Inbound -Action Allow `
    -RemoteAddress $PeerWslSubnet -InterfaceAlias $physIfAlias -Profile Any | Out-Null
Write-Ok "入站: 允许来自 $PeerWslSubnet"

New-NetFirewallRule -DisplayName $RuleName -Direction Outbound -Action Allow `
    -RemoteAddress $PeerWslSubnet -InterfaceAlias $physIfAlias -Profile Any | Out-Null
Write-Ok "出站: 允许发往 $PeerWslSubnet"

# WSL 虚拟网卡侧也要放行，否则转发进 WSL 的包会被 Hyper-V 防火墙丢掉
New-NetFirewallRule -DisplayName $RuleName -Direction Inbound -Action Allow `
    -RemoteAddress $PeerWslSubnet -InterfaceAlias $topo.WslIfName -Profile Any | Out-Null
Write-Ok "入站($($topo.WslIfName)): 允许来自 $PeerWslSubnet"

New-NetFirewallRule -DisplayName $RuleName -Direction Outbound -Action Allow `
    -RemoteAddress $PeerWslSubnet -InterfaceAlias $topo.WslIfName -Profile Any | Out-Null
Write-Ok "出站($($topo.WslIfName)): 允许发往 $PeerWslSubnet"

# 4. 本机自检
Write-Step '本机自检'
$fwd = (Get-NetIPInterface -InterfaceIndex $physIfIndex -AddressFamily IPv4).Forwarding
Write-Ok "$physIfAlias 转发 = $fwd"
$fwd2 = (Get-NetIPInterface -InterfaceIndex $topo.WslIfIndex -AddressFamily IPv4).Forwarding
Write-Ok "$($topo.WslIfName) 转发 = $fwd2"
Get-NetRoute -DestinationPrefix $PeerWslSubnet -ErrorAction SilentlyContinue |
    Select-Object DestinationPrefix, NextHop, InterfaceAlias |
    Format-Table -AutoSize | Out-String | Write-Host

Write-Host "`n本机配置完成。" -ForegroundColor Green
Write-Host @"

下一步：
  1. 在**对端 Windows** 上用管理员权限跑同一个脚本，参数互换：
       -PeerWindowsIp $localIp -PeerWslSubnet $($topo.WslSubnet)
  2. 两边都跑完后，在任一 WSL 里 ping 对端 WSL 地址验证。

注意：对端 WSL 的 IP 用 `wsl hostname -I` 查，子网用 `ip -4 -brief addr show eth0` 的前缀推算。
"@ -ForegroundColor White
