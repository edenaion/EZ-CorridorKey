@echo off
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
    pip install -r gvm_core\requirements.txt
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
