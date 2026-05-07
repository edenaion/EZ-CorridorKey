@echo off
TITLE EZ-CorridorKey
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo [INFO] .venv not found. Running installer...
    call 1-install.bat
    if not exist ".venv\Scripts\activate.bat" (
        echo [ERROR] Installation failed. Please run 1-install.bat manually and check for errors.
        pause
        exit /b 1
    )
)

REM Add local ffmpeg to PATH if present
if exist "%~dp0tools\ffmpeg\bin\ffmpeg.exe" set "PATH=%~dp0tools\ffmpeg\bin;%PATH%"

call .venv\Scripts\activate.bat
start "" pythonw main.py %*
exit
