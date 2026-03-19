@echo off
TITLE EZ-CorridorKey (AMD Radeon Optimized)
:: Updated by Github toowyred - Instagram @wyred.on
cd /d "%~dp0"

echo --------------------------------------------------------
echo EZ-CorridorKey: AMD GPU & Portability Patch
echo Maintained by toowyred
echo --------------------------------------------------------

:: 1. FFmpeg Path Injection
:: Checks for local FFmpeg in tools folder to bypass system install requirements
if exist "%~dp0tools\ffmpeg\bin\ffmpeg.exe" (
    set "PATH=%~dp0tools\ffmpeg\bin;%PATH%"
    echo [INFO] Local FFmpeg detected and injected into PATH.
)

:: 2. Virtual Environment Check
if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment (.venv) not found. 
    echo Please run 1-install.bat before launching.
    pause
    exit /b 1
)

:: 3. AMD GPU / DirectML Optimization Flags
:: Forces Torch to look for DirectML devices and prevents CUDA conflict errors
set TORCH_DEVICE=directml
set CUDA_VISIBLE_DEVICES=-1

:: 4. Launch Application
echo [INFO] Launching with DirectML (AMD/Radeon) compatibility...
call .venv\Scripts\activate.bat
python main.py %*

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [SYSTEM] Application closed with an error. 
    echo If you have an AMD GPU, ensure torch-directml is installed in your .venv.
    pause
)

exit
