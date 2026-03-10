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
where nvidia-smi >nul 2>&1
if !errorlevel! equ 0 (
    echo   Detecting NVIDIA GPU...
    for /f "tokens=*" %%i in ('nvidia-smi 2^>nul ^| findstr "CUDA Version"') do set "CUDA_LINE=%%i"
    if defined CUDA_LINE (
        echo   !CUDA_LINE!
        echo !CUDA_LINE! | findstr "12.8 12.7 12.6 12.5 12.4" >nul
        if !errorlevel!==0 (
            set "INDEX_URL=https://download.pytorch.org/whl/cu128"
            echo   Using PyTorch CUDA 12.8 wheels
        ) else (
            echo !CUDA_LINE! | findstr "12.1 12.2 12.3" >nul
            if !errorlevel!==0 (
                set "INDEX_URL=https://download.pytorch.org/whl/cu121"
                echo   Using PyTorch CUDA 12.1 wheels
            ) else (
                echo !CUDA_LINE! | findstr "11.8 11.7" >nul
                if !errorlevel!==0 (
                    set "INDEX_URL=https://download.pytorch.org/whl/cu118"
                    echo   Using PyTorch CUDA 11.8 wheels
                )
            )
        )
    )
)

if not defined INDEX_URL (
    echo   [WARN] No NVIDIA GPU detected — installing CPU-only PyTorch
    echo   If you have an NVIDIA GPU, ensure drivers are installed and nvidia-smi works.
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
