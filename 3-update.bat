@echo off
setlocal enabledelayedexpansion
TITLE EZ-CorridorKey Updater
cd /d "%~dp0"

echo.
echo  ========================================
echo   EZ-CorridorKey — Update
echo  ========================================
echo.

:: ── Step 1: Pull latest code ──
echo [1/3] Pulling latest changes...
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo   [ERROR] Git not found. Install Git from https://git-scm.com
    goto :fail
)

git pull --recurse-submodules 2>&1
if %errorlevel% neq 0 (
    echo   [WARN] Git pull had issues. You may have local changes.
    echo   If stuck, try: git stash ^&^& git pull ^&^& git stash pop
) else (
    echo   [OK] Code updated
)

:: ── Step 2: Update dependencies ──
echo [2/3] Updating dependencies...

set UV_AVAILABLE=0
where uv >nul 2>&1
if %errorlevel%==0 (
    set UV_AVAILABLE=1
) else (
    if exist "%USERPROFILE%\.local\bin\uv.exe" (
        set "PATH=%USERPROFILE%\.local\bin;%PATH%"
        set UV_AVAILABLE=1
    ) else if exist "%LOCALAPPDATA%\uv\uv.exe" (
        set "PATH=%LOCALAPPDATA%\uv;%PATH%"
        set UV_AVAILABLE=1
    )
)

if !UV_AVAILABLE!==1 (
    uv pip install --python .venv\Scripts\python.exe --torch-backend=auto -e . 2>&1
    if %errorlevel%==0 (
        echo   [OK] Dependencies updated via uv
    ) else (
        echo   [WARN] uv update failed, trying pip...
        set UV_AVAILABLE=0
    )
)

if !UV_AVAILABLE!==0 (
    if exist ".venv\Scripts\activate.bat" (
        call .venv\Scripts\activate.bat
        pip install -e . 2>&1
        echo   [OK] Dependencies updated via pip
    ) else (
        echo   [ERROR] No .venv found. Run 1-install.bat first.
        goto :fail
    )
)

:: ── Step 3: Check for new model weights ──
echo [3/3] Checking model weights...
.venv\Scripts\python.exe scripts\setup_models.py --check

:: ── Done ──
echo.
echo  ========================================
echo   Update complete!
echo  ========================================
echo.
echo   To launch: double-click 2-start.bat
echo.
pause
exit /b 0

:fail
echo.
echo  Update failed. See errors above.
echo.
pause
exit /b 1
