# CorridorKey Windows Build Script
# Usage: powershell -ExecutionPolicy Bypass -File scripts\build_windows.ps1
#
# Prerequisites:
#   pip install pyinstaller
#
# Output: dist/CorridorKey/CorridorKey.exe
#
# Post-build: copy CorridorKeyModule/checkpoints/*.pth into
#   dist/CorridorKey/CorridorKeyModule/checkpoints/

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
pyinstaller corridorkey.spec --noconfirm

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: PyInstaller build failed" -ForegroundColor Red
    exit 1
}

# Create checkpoint directory placeholder
$ckptDir = "dist\CorridorKey\CorridorKeyModule\checkpoints"
if (-not (Test-Path $ckptDir)) {
    New-Item -ItemType Directory -Path $ckptDir -Force | Out-Null
}

# Copy checkpoint if available
$srcCkpt = "CorridorKeyModule\checkpoints"
if (Test-Path $srcCkpt) {
    $pthFiles = Get-ChildItem "$srcCkpt\*.pth"
    if ($pthFiles.Count -gt 0) {
        Write-Host "Copying checkpoint(s)..."
        Copy-Item "$srcCkpt\*.pth" $ckptDir
    } else {
        Write-Host "WARNING: No .pth checkpoint found in $srcCkpt" -ForegroundColor Yellow
        Write-Host "  You must manually place the checkpoint in: $ckptDir"
    }
} else {
    Write-Host "WARNING: Checkpoint directory not found: $srcCkpt" -ForegroundColor Yellow
}

# Summary
$exePath = "dist\CorridorKey\CorridorKey.exe"
if (Test-Path $exePath) {
    $size = (Get-Item $exePath).Length / 1MB
    Write-Host ""
    Write-Host "=== Build Complete ===" -ForegroundColor Green
    Write-Host "  Executable: $exePath ($([math]::Round($size, 1)) MB)"
    Write-Host "  Checkpoint: $ckptDir"
    Write-Host ""
    Write-Host "To run: .\dist\CorridorKey\CorridorKey.exe"
} else {
    Write-Host "ERROR: Build output not found" -ForegroundColor Red
    exit 1
}
