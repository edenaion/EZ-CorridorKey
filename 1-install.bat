@echo off
setlocal enabledelayedexpansion
TITLE EZ-CorridorKey Installer
cd /d "%~dp0"

echo.
echo  ========================================
echo   EZ-CorridorKey - One-Click Installer
echo  ========================================
echo.

REM ── Step 1: Check Python ──
echo [1/6] Checking Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo   [ERROR] Python not found. Install Python 3.10+ from https://python.org
    echo   Make sure to check "Add Python to PATH" during installation.
    goto :fail
)

for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
for /f "tokens=1,2 delims=." %%a in ("!PYVER!") do (
    set PYMAJOR=%%a
    set PYMINOR=%%b
)
if !PYMAJOR! LSS 3 (
    echo   [ERROR] Python 3.10+ required, found !PYVER!
    goto :fail
)
if !PYMAJOR!==3 if !PYMINOR! LSS 10 (
    echo   [ERROR] Python 3.10+ required, found !PYVER!
    goto :fail
)
echo   [OK] Python !PYVER!

REM ── Step 2: Check for old venv ──
if exist "venv\Scripts\activate.bat" (
    if not exist ".venv\Scripts\activate.bat" (
        echo.
        echo   [NOTE] Found old 'venv' directory from previous installer.
        echo   The new installer uses '.venv'. You can safely delete 'venv' later.
        echo.
    )
)

REM ── Step 3: Install/locate uv ──
echo [2/6] Setting up package manager...
set UV_AVAILABLE=0

where uv >nul 2>&1
if %errorlevel%==0 (
    set UV_AVAILABLE=1
    echo   [OK] uv found
    goto :uv_done
)

if exist "%USERPROFILE%\.local\bin\uv.exe" (
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
    set UV_AVAILABLE=1
    echo   [OK] uv found at %USERPROFILE%\.local\bin
    goto :uv_done
)

if exist "%LOCALAPPDATA%\uv\uv.exe" (
    set "PATH=%LOCALAPPDATA%\uv;%PATH%"
    set UV_AVAILABLE=1
    echo   [OK] uv found at %LOCALAPPDATA%\uv
    goto :uv_done
)

echo   Installing uv (fast Python package manager)...
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex" >nul 2>&1

if exist "%USERPROFILE%\.local\bin\uv.exe" (
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
    set UV_AVAILABLE=1
    echo   [OK] uv installed
    goto :uv_done
)

if exist "%LOCALAPPDATA%\uv\uv.exe" (
    set "PATH=%LOCALAPPDATA%\uv;%PATH%"
    set UV_AVAILABLE=1
    echo   [OK] uv installed
    goto :uv_done
)

echo   [WARN] uv install failed, falling back to pip (slower)

:uv_done

REM ── Step 4: Create venv + install dependencies ──
echo [3/6] Installing dependencies...

if !UV_AVAILABLE!==0 goto :use_pip

if not exist ".venv\Scripts\activate.bat" (
    echo   Creating virtual environment...
    uv venv .venv >nul 2>&1
)
echo   Installing packages (uv + auto CUDA detection)...
uv pip install --python .venv\Scripts\python.exe --torch-backend=auto -e . 2>&1
if %errorlevel% neq 0 (
    echo   [WARN] uv install failed, trying pip fallback...
    goto :use_pip
)
echo   [OK] Dependencies installed via uv
goto :deps_done

:use_pip
if not exist ".venv\Scripts\activate.bat" (
    echo   Creating virtual environment...
    python -m venv .venv
)
call .venv\Scripts\activate.bat

REM Detect CUDA version for correct PyTorch wheel
set INDEX_URL=
nvidia-smi >nul 2>&1
if !errorlevel! neq 0 goto :no_nvidia

for /f "tokens=*" %%i in ('nvidia-smi --query-gpu=driver_version --format=csv,noheader 2^>nul') do set DRIVER=%%i
echo   NVIDIA driver detected: !DRIVER!
for /f "tokens=*" %%i in ('nvidia-smi 2^>nul ^| findstr "CUDA Version"') do set CUDA_LINE=%%i
echo   !CUDA_LINE!

echo !CUDA_LINE! | findstr "12.8 12.7 12.6 12.5 12.4" >nul
if !errorlevel!==0 (
    set INDEX_URL=https://download.pytorch.org/whl/cu128
    echo   Using PyTorch CUDA 12.8 wheels
    goto :pip_install
)
echo !CUDA_LINE! | findstr "12.1 12.2 12.3" >nul
if !errorlevel!==0 (
    set INDEX_URL=https://download.pytorch.org/whl/cu121
    echo   Using PyTorch CUDA 12.1 wheels
    goto :pip_install
)
echo !CUDA_LINE! | findstr "11.8 11.7" >nul
if !errorlevel!==0 (
    set INDEX_URL=https://download.pytorch.org/whl/cu118
    echo   Using PyTorch CUDA 11.8 wheels
    goto :pip_install
)

:no_nvidia
echo   No NVIDIA GPU detected, installing CPU-only PyTorch
set INDEX_URL=https://download.pytorch.org/whl/cpu

:pip_install
echo   Installing packages via pip (this may take a few minutes)...
pip install --upgrade pip >nul 2>&1
pip install --extra-index-url !INDEX_URL! -e . 2>&1
if !errorlevel! neq 0 (
    echo   [ERROR] pip install failed
    goto :fail
)
echo   [OK] Dependencies installed via pip

:deps_done

REM ── Step 5: Check FFmpeg ──
echo [4/6] Checking FFmpeg...
where ffmpeg >nul 2>&1
if %errorlevel%==0 (
    echo   [OK] FFmpeg found
) else (
    echo   [WARN] FFmpeg not found. Video import requires FFmpeg.
    echo   Install via one of:
    echo     winget install ffmpeg
    echo     choco install ffmpeg
    echo     https://ffmpeg.org/download.html (add to PATH)
    echo.
)

REM ── Step 6: Download model weights ──
echo [5/6] Checking model weights...

.venv\Scripts\python.exe scripts\setup_models.py --check
.venv\Scripts\python.exe scripts\setup_models.py --corridorkey
if %errorlevel% neq 0 (
    echo   [WARN] CorridorKey model download failed. You can retry later:
    echo     .venv\Scripts\python scripts\setup_models.py --corridorkey
)

echo.
echo [6/6] Optional models (can be downloaded later)
echo.
set /p INSTALL_GVM="  Download GVM alpha generator? (~6GB) [y/N]: "
if /i "!INSTALL_GVM!"=="y" (
    .venv\Scripts\python.exe scripts\setup_models.py --gvm
)

set /p INSTALL_VM="  Download VideoMaMa alpha generator? (~37GB) [y/N]: "
if /i "!INSTALL_VM!"=="y" (
    .venv\Scripts\python.exe scripts\setup_models.py --videomama
)

REM ── Create desktop shortcut ──
echo.
set /p CREATE_SHORTCUT="  Create desktop shortcut? [Y/n]: "
if /i "!CREATE_SHORTCUT!"=="n" goto :skip_shortcut

set "SHORTCUT_PATH=%USERPROFILE%\Desktop\CorridorKey.lnk"
set "TARGET_PATH=%CD%\.venv\Scripts\pythonw.exe"
set "ICON_PATH=%CD%\ui\theme\corridorkey.ico"
set "WORK_DIR=%CD%"

echo $ws = New-Object -ComObject WScript.Shell > "%TEMP%\mk_shortcut.ps1"
echo $s = $ws.CreateShortcut("!SHORTCUT_PATH!") >> "%TEMP%\mk_shortcut.ps1"
echo $s.TargetPath = "!TARGET_PATH!" >> "%TEMP%\mk_shortcut.ps1"
echo $s.Arguments = "main.py" >> "%TEMP%\mk_shortcut.ps1"
echo $s.WorkingDirectory = "!WORK_DIR!" >> "%TEMP%\mk_shortcut.ps1"
echo $s.IconLocation = "!ICON_PATH!,0" >> "%TEMP%\mk_shortcut.ps1"
echo $s.WindowStyle = 7 >> "%TEMP%\mk_shortcut.ps1"
echo $s.Description = "CorridorKey - AI Green Screen" >> "%TEMP%\mk_shortcut.ps1"
echo $s.Save() >> "%TEMP%\mk_shortcut.ps1"
powershell -ExecutionPolicy ByPass -File "%TEMP%\mk_shortcut.ps1" >nul 2>&1
del "%TEMP%\mk_shortcut.ps1" >nul 2>&1

if exist "!SHORTCUT_PATH!" (
    echo   [OK] Desktop shortcut created (no console window)
    echo   Tip: right-click it to pin to taskbar
) else (
    echo   [WARN] Shortcut creation failed — you can pin 2-start.bat manually
)

:skip_shortcut

REM ── Done ──
echo.
echo  ========================================
echo   Installation complete!
echo  ========================================
echo.
echo   To launch: double-click 2-start.bat (or the desktop shortcut)
echo   Or run:    2-start.bat
echo.
echo   To download optional models later:
echo     .venv\Scripts\python scripts\setup_models.py --gvm
echo     .venv\Scripts\python scripts\setup_models.py --videomama
echo.
pause
exit /b 0

:fail
echo.
echo  Installation failed. See errors above.
echo.
pause
exit /b 1
