# engram install script — Windows (PowerShell 5.1+)
# Usage: .\install.ps1
#
# If you get "execution policy" errors, run first:
#   Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
#Requires -Version 5.1
$ErrorActionPreference = "Stop"

$InstallDir = Join-Path $env:USERPROFILE ".config\opencode"

Write-Host "engram installer"
Write-Host "================"
Write-Host ""

# ── Verify Python ──────────────────────────────────────────────────────────────
$python = $null
foreach ($candidate in @("python", "python3", "py")) {
    try {
        $ver = & $candidate --version 2>&1
        if ($ver -match "Python (\d+)\.(\d+)") {
            $major = [int]$Matches[1]
            $minor = [int]$Matches[2]
            if ($major -gt 3 -or ($major -eq 3 -and $minor -ge 10)) {
                $python = $candidate
                Write-Host "Python $major.$minor  OK"
                break
            }
        }
    } catch { }
}

if (-not $python) {
    Write-Error "Python 3.10+ not found. Install from https://python.org and try again."
    exit 1
}

# ── Locate install dir ─────────────────────────────────────────────────────────
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition

if ($ScriptDir -ne $InstallDir) {
    Write-Host ""
    Write-Host "Copying files to $InstallDir ..."
    if (-not (Test-Path $InstallDir)) {
        New-Item -ItemType Directory -Path $InstallDir | Out-Null
    }
    Copy-Item -Path "$ScriptDir\*" -Destination $InstallDir -Recurse -Force
}

Set-Location $InstallDir

# ── Create venv ────────────────────────────────────────────────────────────────
$venvPython = Join-Path $InstallDir ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "Creating virtualenv at $InstallDir\.venv ..."
    & $python -m venv .venv
} else {
    Write-Host "Virtualenv already exists, skipping creation."
}

# ── Install dependencies ───────────────────────────────────────────────────────
Write-Host "Installing dependencies ..."
& $venvPython -m pip install --upgrade pip --quiet
& $venvPython -m pip install -r requirements.txt --quiet

Write-Host ""
Write-Host "Done."
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Open preflight.config.json and adjust retrievalConfidenceThreshold / topN if needed."
Write-Host "  2. Restart opencode — it will discover the plugin automatically."
Write-Host "  3. Verify with:"
Write-Host "       $venvPython memory.py retrieve_facts test test `"hello`" 3 0.0"
Write-Host ""
