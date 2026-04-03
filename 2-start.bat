@echo off
TITLE EZ-CorridorKey
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] .venv not found. Run 1-install.bat first!
    pause
    exit /b 1
)

REM Add local ffmpeg to PATH if present
if exist "%~dp0tools\ffmpeg\bin\ffmpeg.exe" set "PATH=%~dp0tools\ffmpeg\bin;%PATH%"

call .venv\Scripts\activate.bat
start "" pythonw main.py %*
exit
