$ErrorActionPreference = 'SilentlyContinue'
Write-Output "=== windows build ==="
$cv = Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion'
Write-Output ($cv.DisplayVersion + ' Build ' + $cv.CurrentBuild + '.' + $cv.UBR)
Write-Output "=== wsl version ==="
wsl --version | Out-String
Write-Output "=== physical nic (up) ==="
Get-NetAdapter -Physical | Where-Object Status -eq 'Up' | ForEach-Object {
  Write-Output ($_.Name + ' | ' + $_.InterfaceDescription + ' | ' + $_.LinkSpeed)
}
Write-Output "=== .wslconfig ==="
Get-Content "$env:USERPROFILE\.wslconfig" -ErrorAction SilentlyContinue | Out-String
Write-Output "=== hyper-v firewall ==="
Get-NetFirewallHyperVVMSetting -Name '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' |
  Format-List Name,Enabled,DefaultInboundAction,DefaultOutboundAction,LoopbackEnabled | Out-String
Write-Output "=== windows firewall profiles ==="
Get-NetFirewallProfile | ForEach-Object { Write-Output ($_.Name + ': ' + $_.Enabled) }
Write-Output "=== portproxy (legacy NAT leftovers) ==="
netsh interface portproxy show all | Out-String
Write-Output "=== done ==="
