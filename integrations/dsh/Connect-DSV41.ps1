[CmdletBinding()]
param(
    [ValidateRange(1, 65535)]
    [int]$LocalPort = 48241,
    [ValidateRange(1, 65535)]
    [int]$RelayPort = 49241,
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]*$')]
    [string]$SshHost = 'dmit',
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]*$')]
    [string]$ExpectedRelayHost,
    [string]$ConfigRoot = (Join-Path $HOME '.dsv41-connection')
)

$ErrorActionPreference = 'Stop'
$script:Child = $null
$script:StopRequested = $false

function Write-ConnectionLog {
    param([string]$Message)
    $stamp = (Get-Date).ToUniversalTime().ToString('o')
    Add-Content -LiteralPath (Join-Path $ConfigRoot 'windows-tunnel.log') -Value "$stamp $Message" -Encoding utf8
}

function Get-EffectiveSshConfig {
    param([string]$SshExe, [string]$HostName)
    $output = & $SshExe -G $HostName 2>$null
    if ($LASTEXITCODE -ne 0) { throw "无法读取 SSH 配置主机 $HostName。" }
    $result = @{}
    foreach ($line in $output) {
        $pair = $line -split '\s+', 2
        if ($pair.Count -eq 2) { $result[$pair[0].ToLowerInvariant()] = $pair[1] }
    }
    return $result
}

function Test-PreexistingListener {
    param([int]$Port)
    $probe = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $Port)
    $probe.Server.ExclusiveAddressUse = $true
    try { $probe.Start() } catch { throw "本机 127.0.0.1:$Port 无法绑定；本脚本不会结束占用它的进程。" } finally { $probe.Stop() }
}

function Stop-OwnSsh {
    if ($null -ne $script:Child -and -not $script:Child.HasExited) {
        try { Stop-Process -Id $script:Child.Id -ErrorAction Stop } catch { }
        try { $script:Child.WaitForExit(15000) } catch { }
        if (-not $script:Child.HasExited) {
            try { Stop-Process -Id $script:Child.Id -Force -ErrorAction Stop } catch { }
        }
    }
    $script:Child = $null
}

[System.IO.Directory]::CreateDirectory($ConfigRoot) | Out-Null

$mutex = [System.Threading.Mutex]::new($false, 'Local\DSV41-DSH-Tunnel')
$mutexOwned = $false
try { $mutexOwned = $mutex.WaitOne(0) }
catch [System.Threading.AbandonedMutexException] { $mutexOwned = $true }
if (-not $mutexOwned) {
    $mutex.Dispose()
    Write-Error '已有 DSV41 本地隧道监督进程在运行。'
    exit 3
}

try {
    $sshCommand = Get-Command ssh.exe -ErrorAction Stop
    $sshExe = $sshCommand.Source
    $effective = Get-EffectiveSshConfig -SshExe $sshExe -HostName $SshHost
    if ($effective['hostname'] -ne $ExpectedRelayHost) {
        throw "SSH 主机 $SshHost 当前解析为 $($effective['hostname'])，不是预期中转机 $ExpectedRelayHost。"
    }
    Test-PreexistingListener -Port $LocalPort
    Write-ConnectionLog "supervisor started; local=127.0.0.1:$LocalPort relay=127.0.0.1:$RelayPort"
    $backoff = 2
    while (-not $script:StopRequested) {
        $stdout = Join-Path $ConfigRoot 'windows-tunnel.stdout.log'
        $stderr = Join-Path $ConfigRoot 'windows-tunnel.stderr.log'
        $forward = "127.0.0.1:${LocalPort}:127.0.0.1:$RelayPort"
        $arguments = @(
            '-N', '-T',
            '-o', 'BatchMode=yes',
            '-o', 'StrictHostKeyChecking=yes',
            '-o', 'ExitOnForwardFailure=yes',
            '-o', 'ConnectTimeout=10',
            '-o', 'ServerAliveInterval=15',
            '-o', 'ServerAliveCountMax=3',
            '-o', 'LogLevel=ERROR',
            '-L', $forward,
            $SshHost
        )
        $script:Child = Start-Process -FilePath $sshExe -ArgumentList $arguments -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput $stdout -RedirectStandardError $stderr
        Write-ConnectionLog "ssh child started pid=$($script:Child.Id)"
        $script:Child.WaitForExit()
        $code = $script:Child.ExitCode
        $script:Child = $null
        if ($script:StopRequested) { break }
        Write-ConnectionLog "ssh child exited code=$code; retry in $backoff seconds"
        Start-Sleep -Seconds $backoff
        $backoff = [Math]::Min($backoff * 2, 60)
    }
} finally {
    Stop-OwnSsh
    try { Write-ConnectionLog 'supervisor stopped' } finally {
        if ($mutexOwned) { $mutex.ReleaseMutex() | Out-Null }
        $mutex.Dispose()
    }
}
