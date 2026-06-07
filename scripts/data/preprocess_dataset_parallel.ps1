param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $PreprocessArgs
)

$ErrorActionPreference = "Stop"

$principal = New-Object Security.Principal.WindowsPrincipal(
    [Security.Principal.WindowsIdentity]::GetCurrent()
)
$isAdmin = $principal.IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator
)

if (-not $isAdmin) {
    Write-Warning "This PowerShell session is not running as Administrator. Cavernize may fail or preprocessing may be incomplete."
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..\..")
$pythonScript = Join-Path $scriptDir "preprocess_dataset_parallel.py"

Push-Location $repoRoot
try {
    python $pythonScript @PreprocessArgs
}
finally {
    Pop-Location
}
