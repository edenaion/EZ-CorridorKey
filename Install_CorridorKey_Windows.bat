@echo off
setlocal enabledelayedexpansion
TITLE CorridorKey Setup Wizard
echo ===================================================
echo     CorridorKey - Windows Auto-Installer
echo ===================================================
echo.

:: 1. Check Python
echo [1/4] Checking Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo   [ERROR] Python not found. Install Python 3.10+ from https://python.org
    echo   Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)
for /f "tokens=2" %%V in ('python --version 2^>^&1') do echo   [OK] Python %%V

:: 2. Create Virtual Environment
echo.
echo [2/4] Setting up Python Virtual Environment (venv)...
if not exist "venv\Scripts\activate.bat" (
    python -m venv venv
) else (
    echo   Virtual environment already exists.
)

:: 3. Detect CUDA + Install Dependencies
echo.
echo [3/4] Installing Dependencies (This might take a while)...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip >nul 2>&1

set "INDEX_URL="
set "DRIVER="
set "CUDA_LINE="
set "CUDA_WHEEL_LABEL="
set "CUDA_NOTE="
for /f "usebackq tokens=1,* delims==" %%A in (`python scripts\detect_windows_torch_index.py --format env`) do set "%%A=%%B"
if not defined INDEX_URL set "INDEX_URL=https://download.pytorch.org/whl/cpu"
if not defined CUDA_NOTE set "CUDA_NOTE=Could not run CUDA detection helper; installing CPU-only PyTorch."
if defined DRIVER echo(  NVIDIA driver detected: !DRIVER!
if defined CUDA_LINE echo(  !CUDA_LINE!
if /i "!CUDA_DETECT_MODE!"=="nvidia" (
    echo(  Using !CUDA_WHEEL_LABEL!
) else (
    echo(  [WARN] !CUDA_NOTE!
    if /i "!CUDA_DETECT_REASON!"=="nvidia_smi_not_found" (
        echo(  If you have an NVIDIA GPU, ensure drivers are installed and nvidia-smi works.
    )
)

if defined INDEX_URL (
    pip install --extra-index-url !INDEX_URL! -r requirements.txt
) else (
    pip install -r requirements.txt
)

:: 4. Download Weights
echo.
echo [4/4] Downloading CorridorKey Model Weights...
if not exist "CorridorKeyModule\checkpoints" mkdir "CorridorKeyModule\checkpoints"

if not exist "CorridorKeyModule\checkpoints\CorridorKey.pth" (
    echo Downloading CorridorKey.pth...
    curl.exe -L -o "CorridorKeyModule\checkpoints\CorridorKey.pth" "https://huggingface.co/nikopueringer/CorridorKey_v1.0/resolve/main/CorridorKey_v1.0.pth"
) else (
    echo CorridorKey.pth already exists!
)

echo.
echo ===================================================
echo   Setup Complete! You are ready to key!
echo   Drag and drop folders onto CorridorKey_DRAG_CLIPS_HERE_local.bat
echo ===================================================
pause
