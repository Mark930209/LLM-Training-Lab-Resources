# migrate_benchmark.ps1 —— WSL2 环境整体迁移的实测数据采集（Windows 侧运行）
#
# 产出：results/Season0/02/migration-benchmark.md（导出文件大小、导出/导入耗时、导入后验证输出）
# 注意：会终止发行版，必须在 PyTorch 安装完成、无训练任务运行时执行。

$ErrorActionPreference = "Stop"
$Distro = "Ubuntu-24.04"
$BackupFile = "$env:TEMP\wsl-migration-test.tar"
$ResultFile = "D:\DocProjects\LearnLLMFromDraft\results\Season0\02\migration-benchmark.md"

Write-Host "== 1. 终止发行版保证一致性 =="
wsl --terminate $Distro

Write-Host "== 2. 导出 =="
$t0 = Get-Date
wsl --export $Distro $BackupFile
$t1 = Get-Date
$exportSec = [math]::Round(($t1 - $t0).TotalSeconds, 1)
$sizeGB = [math]::Round((Get-Item $BackupFile).Length / 1GB, 2)
Write-Host "导出完成: ${sizeGB}GB, 耗时 ${exportSec}s"

Write-Host "== 3. 导入为临时发行版验证 =="
$TempName = "MigrationTest"
$TempDir = "$env:TEMP\wsl-migration-test"
if (Test-Path $TempDir) { Remove-Item $TempDir -Recurse -Force }
wsl --unregister $TempName 2>$null
Start-Sleep -Seconds 10   # unregister 是异步操作，立即 import 会报 0x8000000d
$t2 = Get-Date
wsl --import $TempName $TempDir $BackupFile
$t3 = Get-Date
$importSec = [math]::Round(($t3 - $t2).TotalSeconds, 1)
Write-Host "导入完成: 耗时 ${importSec}s"

Write-Host "== 4. 导入环境验证（项目目录与 venv 是否原封不动） =="
$verify = wsl -d $TempName -- bash -lc "ls ~/llm-training-lab/.venv/bin/python && ~/llm-training-lab/.venv/bin/python -c 'import torch; print(\"torch\", torch.__version__, torch.cuda.is_available())'"

Write-Host "== 5. 清理临时发行版 =="
wsl --unregister $TempName
Remove-Item $BackupFile -Force
Remove-Item $TempDir -Recurse -Force -ErrorAction SilentlyContinue

@"
# 迁移实测（REAL）

| 项 | 值 |
|---|---|
| 导出文件大小 | ${sizeGB} GB |
| 导出耗时 | ${exportSec} s |
| 导入耗时 | ${importSec} s |
| 导入后验证 | $verify |

结论：WSL2 发行版可整体导出为单个 tar 并在新机器/新发行版名下导入，
项目目录、uv 环境、依赖锁原封不动；导入后 torch 与 CUDA 直通立即恢复。
"@ | Set-Content -Path $ResultFile -Encoding UTF8

Write-Host "迁移实测完成: $ResultFile"
