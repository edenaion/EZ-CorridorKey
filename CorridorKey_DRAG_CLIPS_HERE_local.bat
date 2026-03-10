@echo off
TITLE CorridorKey - CLI Wizard
cd /d "%~dp0"

REM SAFETY CHECK: Ensure a folder or file was dragged onto the script
if "%~1"=="" (
    echo [ERROR] No target provided.
    echo.
    echo USAGE:
    echo Please DRAG AND DROP a folder or video file onto this script.
    echo Do not double-click this script directly.
    echo.
    pause
    exit /b
)

REM Check for venv
if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] .venv not found. Run 1-install.bat first!
    pause
    exit /b 1
)

REM Activate venv
call .venv\Scripts\activate.bat

REM Add local ffmpeg to PATH if present
if exist "%~dp0tools\ffmpeg\bin\ffmpeg.exe" set "PATH=%~dp0tools\ffmpeg\bin;%PATH%"

set "WIN_PATH=%~1"

REM Strip trailing slash if present
if "%WIN_PATH:~-1%"=="\" set "WIN_PATH=%WIN_PATH:~0,-1%"

echo Starting Corridor Key CLI Wizard...
echo Target: "%WIN_PATH%"

python "%~dp0clip_manager.py" --action wizard --win_path "%WIN_PATH%"

pause
