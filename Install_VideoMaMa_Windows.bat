@echo off
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
    pip install -r VideoMaMaInferenceModule\requirements.txt
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
