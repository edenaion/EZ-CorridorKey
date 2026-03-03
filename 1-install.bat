@echo off
setlocal enabledelayedexpansion
TITLE EZ-CorridorKey Installer
cd /d "%~dp0"

echo.
echo  ========================================
echo   EZ-CorridorKey - One-Click Installer
echo  ========================================
echo.

:: ── Step 1: Check Python ──
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

:: ── Step 2: Check for old venv ──
if exist "venv\Scripts\activate.bat" (
    if not exist ".venv\Scripts\activate.bat" (
        echo.
        echo   [NOTE] Found old 'venv' directory from previous installer.
        echo   The new installer uses '.venv'. You can safely delete 'venv' later.
        echo.
    )
)

:: ── Step 3: Install/locate uv ──
echo [2/6] Setting up package manager...
set UV_AVAILABLE=0

where uv >nul 2>&1
if %errorlevel%==0 (
    set UV_AVAILABLE=1
    echo   [OK] uv found
) else (
    :: Check common uv install locations
    if exist "%USERPROFILE%\.local\bin\uv.exe" (
        set "PATH=%USERPROFILE%\.local\bin;%PATH%"
        set UV_AVAILABLE=1
        echo   [OK] uv found at %USERPROFILE%\.local\bin
    ) else if exist "%LOCALAPPDATA%\uv\uv.exe" (
        set "PATH=%LOCALAPPDATA%\uv;%PATH%"
        set UV_AVAILABLE=1
        echo   [OK] uv found at %LOCALAPPDATA%\uv
    ) else (
        echo   Installing uv (fast Python package manager)...
        powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex" >nul 2>&1
        :: Check again after install
        if exist "%USERPROFILE%\.local\bin\uv.exe" (
            set "PATH=%USERPROFILE%\.local\bin;%PATH%"
            set UV_AVAILABLE=1
            echo   [OK] uv installed
        ) else if exist "%LOCALAPPDATA%\uv\uv.exe" (
            set "PATH=%LOCALAPPDATA%\uv;%PATH%"
            set UV_AVAILABLE=1
            echo   [OK] uv installed
        ) else (
            echo   [WARN] uv install failed, falling back to pip (slower)
        )
    )
)

:: ── Step 4: Create venv + install dependencies ──
echo [3/6] Installing dependencies...

if !UV_AVAILABLE!==1 (
    if not exist ".venv\Scripts\activate.bat" (
        echo   Creating virtual environment...
        uv venv .venv >nul 2>&1
    )
    echo   Installing packages (uv + auto CUDA detection)...
    uv pip install --python .venv\Scripts\python.exe --torch-backend=auto -e . 2>&1
    if %errorlevel% neq 0 (
        echo   [WARN] uv install failed, trying pip fallback...
        set UV_AVAILABLE=0
    ) else (
        echo   [OK] Dependencies installed via uv
    )
)

if !UV_AVAILABLE!==0 (
    if not exist ".venv\Scripts\activate.bat" (
        echo   Creating virtual environment...
        python -m venv .venv
    )
    call .venv\Scripts\activate.bat

    :: Detect CUDA version for correct PyTorch wheel
    set INDEX_URL=
    nvidia-smi >nul 2>&1
    if !errorlevel!==0 (
        for /f "tokens=*" %%i in ('nvidia-smi --query-gpu=driver_version --format=csv,noheader 2^>nul') do set DRIVER=%%i
        echo   NVIDIA driver detected: !DRIVER!
        :: Parse CUDA version from nvidia-smi
        for /f "tokens=*" %%i in ('nvidia-smi 2^>nul ^| findstr "CUDA Version"') do set CUDA_LINE=%%i
        echo   !CUDA_LINE!
        :: Map to PyTorch index URL
        echo !CUDA_LINE! | findstr "12.8 12.7 12.6 12.5 12.4" >nul
        if !errorlevel!==0 (
            set INDEX_URL=https://download.pytorch.org/whl/cu128
            echo   Using PyTorch CUDA 12.8 wheels
        ) else (
            echo !CUDA_LINE! | findstr "12.1 12.2 12.3" >nul
            if !errorlevel!==0 (
                set INDEX_URL=https://download.pytorch.org/whl/cu121
                echo   Using PyTorch CUDA 12.1 wheels
            ) else (
                echo !CUDA_LINE! | findstr "11.8 11.7" >nul
                if !errorlevel!==0 (
                    set INDEX_URL=https://download.pytorch.org/whl/cu118
                    echo   Using PyTorch CUDA 11.8 wheels
                )
            )
        )
    )

    if "!INDEX_URL!"=="" (
        echo   No NVIDIA GPU detected, installing CPU-only PyTorch
        set INDEX_URL=https://download.pytorch.org/whl/cpu
    )

    echo   Installing packages via pip (this may take a few minutes)...
    pip install --upgrade pip >nul 2>&1
    pip install --extra-index-url !INDEX_URL! -e . 2>&1
    if !errorlevel! neq 0 (
        echo   [ERROR] pip install failed
        goto :fail
    )
    echo   [OK] Dependencies installed via pip
)

:: ── Step 5: Check FFmpeg ──
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

:: ── Step 6: Download model weights ──
echo [5/6] Checking model weights...

:: CorridorKey is required
.venv\Scripts\python.exe scripts\setup_models.py --check
.venv\Scripts\python.exe scripts\setup_models.py --corridorkey
if %errorlevel% neq 0 (
    echo   [WARN] CorridorKey model download failed. You can retry later:
    echo     .venv\Scripts\python scripts\setup_models.py --corridorkey
)

:: Optional models
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

:: ── Done ──
echo.
echo  ========================================
echo   Installation complete!
echo  ========================================
echo.
echo   To launch: double-click 2-start.bat
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
