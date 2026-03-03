@echo off
TITLE CorridorKey
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] .venv not found. Run install.bat first!
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat
python main.py %*
