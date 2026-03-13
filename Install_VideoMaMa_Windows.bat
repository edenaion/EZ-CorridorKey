@echo off
setlocal enabledelayedexpansion
TITLE VideoMaMa Setup Wizard
echo ===================================================
echo   VideoMaMa (AlphaHint Generator) - Auto-Installer
echo ===================================================
echo.

if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found.
    echo Please run Install_CorridorKey_Windows.bat first!
    pause
    exit /b
)

:: 1. Install Requirements
echo [1/2] Installing VideoMaMa specific dependencies...
call venv\Scripts\activate.bat

if exist "VideoMaMaInferenceModule\requirements.txt" (
    REM Detect CUDA for correct PyTorch wheel (VideoMaMa requirements list torch)
    set "INDEX_URL="
    set "CUDA_NOTE="
    set "CUDA_WHEEL_LABEL="
    for /f "usebackq tokens=1,* delims==" %%A in (`python scripts\detect_windows_torch_index.py --format env`) do set "%%A=%%B"
    if not defined INDEX_URL set "INDEX_URL=https://download.pytorch.org/whl/cpu"
    if not defined CUDA_NOTE set "CUDA_NOTE=Could not run CUDA detection helper; installing CPU-only PyTorch."
    if /i "!CUDA_DETECT_MODE!"=="nvidia" (
        echo(  Using !CUDA_WHEEL_LABEL!
    ) else (
        echo(  [WARN] !CUDA_NOTE!
    )
    if defined INDEX_URL (
        pip install --extra-index-url !INDEX_URL! -r VideoMaMaInferenceModule\requirements.txt
    ) else (
        pip install -r VideoMaMaInferenceModule\requirements.txt
    )
) else (
    echo Using main project dependencies for VideoMaMa...
)

:: 2. Download Weights
echo.
echo [2/2] Downloading VideoMaMa Model Weights...
if not exist "VideoMaMaInferenceModule\checkpoints" mkdir "VideoMaMaInferenceModule\checkpoints"

echo Installing huggingface-cli...
pip install -U "huggingface_hub[cli]"

echo Downloading VideoMaMa weights from HuggingFace...
huggingface-cli download SammyLim/VideoMaMa --local-dir VideoMaMaInferenceModule\checkpoints

echo.
echo ===================================================
echo   VideoMaMa Setup Complete!
echo ===================================================
pause
