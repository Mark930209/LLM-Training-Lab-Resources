# migrate_backup.ps1 —— WSL2 训练环境整体迁移脚本（Windows 侧运行）
#
# 用途：把整个 WSL2 发行版导出为 tar 备份，换机器/换系统时整体导入，
#       项目目录、uv 环境、依赖锁原封不动。这是"环境可迁移"的直接证据。
#
# 用法（PowerShell）：
#   .\migrate_backup.ps1 export  -Distro Ubuntu-24.04 -OutFile D:\backup\ubuntu-lab.tar
#   .\migrate_backup.ps1 import  -Backup D:\backup\ubuntu-lab.tar -Distro Ubuntu-Lab -InstallDir D:\WSL\Ubuntu-Lab
#   .\migrate_backup.ps1 compact -Distro Ubuntu-24.04     # 压缩 vhdx 磁盘（导出前建议先跑）
#
# 注意：导出文件包含环境内全部数据，体积等于已用磁盘；不要提交进 git。

param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet("export", "import", "compact")]
    [string]$Action,

    [string]$Distro = "Ubuntu-24.04",
    [string]$OutFile = "",
    [string]$Backup = "",
    [string]$InstallDir = ""
)

$ErrorActionPreference = "Stop"

switch ($Action) {
    "export" {
        if (-not $OutFile) { $OutFile = "$HOME\wsl-backup\$Distro-$(Get-Date -Format yyyyMMdd).tar" }
        $dir = Split-Path $OutFile -Parent
        if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }

        Write-Host "关闭发行版以保证一致性..."
        wsl --terminate $Distro

        Write-Host "导出 $Distro 到 $OutFile ..."
        wsl --export $Distro $OutFile

        $sizeGB = [math]::Round((Get-Item $OutFile).Length / 1GB, 2)
        Write-Host "导出完成: $OutFile ($sizeGB GB)"
        Write-Host "验证: 在新机器上执行 import 操作即可整体恢复"
    }

    "import" {
        if (-not $Backup) { throw "import 需要 -Backup 参数指定 tar 文件路径" }
        if (-not $InstallDir) { $InstallDir = "D:\WSL\$Distro" }
        if (-not (Test-Path $Backup)) { throw "备份文件不存在: $Backup" }

        Write-Host "导入 $Backup 为发行版 $Distro (目录 $InstallDir)..."
        wsl --import $Distro $InstallDir $Backup

        Write-Host "导入完成。启动验证:"
        Write-Host "  wsl -d $Distro -- nvidia-smi"
        Write-Host "  wsl -d $Distro -- bash -lc 'source ~/llm-training-lab/.venv/bin/activate && python -c `"import torch; print(torch.cuda.is_available())`"'"
    }

    "compact" {
        Write-Host "关闭发行版..."
        wsl --terminate $Distro
        Write-Host "压缩虚拟磁盘（新版本 WSL 支持 --manage）..."
        wsl --manage $Distro --set-sparse true
        Write-Host "瘦身完成。若 --manage 不可用，可用 diskpart 的 compact vdisk 手动压缩 ext4.vhdx"
    }
}
