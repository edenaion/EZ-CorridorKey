# CorridorKey Windows Build Script
# Usage: powershell -ExecutionPolicy Bypass -File scripts\build_windows.ps1
#
# Prerequisites:
#   pip install pyinstaller
#
# Output: dist/CorridorKey/CorridorKey.exe
#
# Checkpoints are NOT bundled — the setup wizard downloads them on first launch.

$ErrorActionPreference = "Stop"

$ROOT = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ROOT

Write-Host "=== CorridorKey Windows Build ===" -ForegroundColor Yellow
Write-Host "Project root: $ROOT"

# Check PyInstaller
if (-not (Get-Command pyinstaller -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: PyInstaller not found. Install with: pip install pyinstaller" -ForegroundColor Red
    exit 1
}

# Clean previous build
if (Test-Path "dist\CorridorKey") {
    Write-Host "Cleaning previous build..."
    Remove-Item -Recurse -Force "dist\CorridorKey"
}
if (Test-Path "build\CorridorKey") {
    Remove-Item -Recurse -Force "build\CorridorKey"
}

# Build
Write-Host "Building with PyInstaller..."
pyinstaller corridorkey-windows.spec --noconfirm

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: PyInstaller build failed" -ForegroundColor Red
    exit 1
}

# Summary
$exePath = "dist\CorridorKey\CorridorKey.exe"
if (Test-Path $exePath) {
    $size = (Get-Item $exePath).Length / 1MB
    $distSize = (Get-ChildItem -Recurse "dist\CorridorKey" | Measure-Object -Property Length -Sum).Sum / 1MB
    Write-Host ""
    Write-Host "=== Build Complete ===" -ForegroundColor Green
    Write-Host "  Executable: $exePath ($([math]::Round($size, 1)) MB)"
    Write-Host "  Total dist: $([math]::Round($distSize, 0)) MB"
    Write-Host ""
    Write-Host "  Checkpoints will be downloaded on first launch via setup wizard."
    Write-Host "  To run: .\dist\CorridorKey\CorridorKey.exe"
} else {
    Write-Host "ERROR: Build output not found" -ForegroundColor Red
    exit 1
}
