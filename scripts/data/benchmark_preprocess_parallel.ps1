param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $BenchmarkArgs
)

$ErrorActionPreference = "Stop"

$principal = New-Object Security.Principal.WindowsPrincipal(
    [Security.Principal.WindowsIdentity]::GetCurrent()
)
$isAdmin = $principal.IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator
)

if (-not $isAdmin) {
    Write-Warning "This PowerShell session is not running as Administrator. Cavernize may fail or benchmark results may be invalid."
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..\..")
$pythonScript = Join-Path $scriptDir "benchmark_preprocess_parallel.py"

Push-Location $repoRoot
try {
    python $pythonScript @BenchmarkArgs
}
finally {
    Pop-Location
}
