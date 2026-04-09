# EZ-CorridorKey Windows Build Script
# Usage: powershell -ExecutionPolicy Bypass -File scripts\build_windows.ps1
#
# Prerequisites:
#   pip install pyinstaller
#
# Output: dist/EZ-CorridorKey/EZ-CorridorKey.exe
#
# Checkpoints are NOT bundled — the setup wizard downloads them on first launch.

$ErrorActionPreference = "Stop"

$ROOT = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ROOT

# Extract version from pyproject.toml (single source of truth)
$VERSION = (Select-String -Path "pyproject.toml" -Pattern '^version\s*=\s*"(.+)"' | ForEach-Object { $_.Matches[0].Groups[1].Value })
if (-not $VERSION) {
    Write-Host "ERROR: Could not extract version from pyproject.toml" -ForegroundColor Red
    exit 1
}

Write-Host "=== EZ-CorridorKey Windows Build v$VERSION ===" -ForegroundColor Yellow
Write-Host "Project root: $ROOT"

# Check PyInstaller
if (-not (Get-Command pyinstaller -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: PyInstaller not found. Install with: pip install pyinstaller" -ForegroundColor Red
    exit 1
}

# Clean previous build
if (Test-Path "dist\EZ-CorridorKey") {
    Write-Host "Cleaning previous build..."
    Remove-Item -Recurse -Force "dist\EZ-CorridorKey"
}
if (Test-Path "build\EZ-CorridorKey") {
    Remove-Item -Recurse -Force "build\EZ-CorridorKey"
}

# Build
Write-Host "Building with PyInstaller..."
pyinstaller installers/corridorkey-windows.spec --noconfirm

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: PyInstaller build failed" -ForegroundColor Red
    exit 1
}

# Summary
$exePath = "dist\EZ-CorridorKey\EZ-CorridorKey.exe"
if (Test-Path $exePath) {
    $size = (Get-Item $exePath).Length / 1MB
    $distSize = (Get-ChildItem -Recurse "dist\EZ-CorridorKey" | Measure-Object -Property Length -Sum).Sum / 1MB
    Write-Host ""
    Write-Host "=== Build Complete (v$VERSION) ===" -ForegroundColor Green
    Write-Host "  Executable: $exePath ($([math]::Round($size, 1)) MB)"
    Write-Host "  Total dist: $([math]::Round($distSize, 0)) MB"
    Write-Host "  Version:    $VERSION"
    Write-Host ""
    Write-Host "  Checkpoints will be downloaded on first launch via setup wizard."
    Write-Host "  To run: .\dist\EZ-CorridorKey\EZ-CorridorKey.exe"
} else {
    Write-Host "ERROR: Build output not found" -ForegroundColor Red
    exit 1
}
