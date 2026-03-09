@echo off
TITLE CorridorKey Setup Wizard
echo ===================================================
echo     CorridorKey - Windows Auto-Installer
echo ===================================================
echo.

:: 1. Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in your system PATH!
    echo Please install Python 3.10+ from python.org and ensure "Add Python to PATH" is checked.
    pause
    exit /b
)

:: 2. Create Virtual Environment
echo [1/3] Setting up Python Virtual Environment (venv)...
if not exist "venv\Scripts\activate.bat" (
    python -m venv venv
) else (
    echo Virtual environment already exists.
)

:: 3. Install Requirements
echo.
echo [2/3] Installing Dependencies (This might take a while)...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt

:: 4. Download Weights
echo.
echo [3/3] Downloading CorridorKey Model Weights...
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
