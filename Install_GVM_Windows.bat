@echo off
setlocal enabledelayedexpansion
TITLE GVM Setup Wizard
echo ===================================================
echo     GVM (AlphaHint Generator) - Auto-Installer
echo ===================================================
echo.

if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found.
    echo Please run Install_CorridorKey_Windows.bat first!
    pause
    exit /b
)

:: 1. Install Requirements
echo [1/2] Installing GVM specific dependencies...
call venv\Scripts\activate.bat

if exist "gvm_core\requirements.txt" (
    REM Detect CUDA for correct PyTorch wheel (GVM requirements list torch)
    set "INDEX_URL="
    where nvidia-smi >nul 2>&1
    if !errorlevel! equ 0 (
        for /f "tokens=*" %%i in ('nvidia-smi 2^>nul ^| findstr "CUDA Version"') do set "CUDA_LINE=%%i"
        if defined CUDA_LINE (
            echo !CUDA_LINE! | findstr "12.8 12.7 12.6 12.5 12.4" >nul
            if !errorlevel!==0 (
                set "INDEX_URL=https://download.pytorch.org/whl/cu128"
            ) else (
                echo !CUDA_LINE! | findstr "12.1 12.2 12.3" >nul
                if !errorlevel!==0 (
                    set "INDEX_URL=https://download.pytorch.org/whl/cu121"
                ) else (
                    echo !CUDA_LINE! | findstr "11.8 11.7" >nul
                    if !errorlevel!==0 set "INDEX_URL=https://download.pytorch.org/whl/cu118"
                )
            )
        )
    )
    if defined INDEX_URL (
        pip install --extra-index-url !INDEX_URL! -r gvm_core\requirements.txt
    ) else (
        pip install -r gvm_core\requirements.txt
    )
) else (
    echo Using main project dependencies for GVM...
)

:: 2. Download Weights
echo.
echo [2/2] Downloading GVM Model Weights (WARNING: Massive 80GB+ Download)...
if not exist "gvm_core\weights" mkdir "gvm_core\weights"

echo Installing huggingface-cli...
pip install -U "huggingface_hub[cli]"

echo Downloading GVM weights from HuggingFace...
huggingface-cli download geyongtao/gvm --local-dir gvm_core\weights

echo.
echo ===================================================
echo   GVM Setup Complete!
echo ===================================================
pause
