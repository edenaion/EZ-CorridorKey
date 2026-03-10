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

:: Check if this is a git repo AND git is available
set USE_GIT=0
git --version >nul 2>&1
if %errorlevel%==0 (
    if exist ".git" (
        set USE_GIT=1
    )
)

if !USE_GIT!==1 (
    git pull --recurse-submodules 2>&1
    if %errorlevel% neq 0 (
        echo   [WARN] Git pull had issues. You may have local changes.
        echo   If stuck, try: git stash ^&^& git pull ^&^& git stash pop
    ) else (
        echo   [OK] Code updated via git
    )
) else (
    echo   No git repo detected — downloading latest release as ZIP...
    set "UPDATE_URL=https://github.com/edenaion/EZ-CorridorKey/archive/refs/heads/main.zip"
    set "UPDATE_ZIP=%TEMP%\corridorkey-update.zip"
    set "UPDATE_EXTRACT=%TEMP%\corridorkey-update"

    powershell -ExecutionPolicy ByPass -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '!UPDATE_URL!' -OutFile '!UPDATE_ZIP!'" >nul 2>&1
    if not exist "!UPDATE_ZIP!" (
        echo   [ERROR] Download failed. Check your internet connection.
        goto :fail
    )

    echo   Extracting update...
    if exist "!UPDATE_EXTRACT!" rmdir /s /q "!UPDATE_EXTRACT!"
    powershell -ExecutionPolicy ByPass -Command "Expand-Archive -Path '!UPDATE_ZIP!' -DestinationPath '!UPDATE_EXTRACT!' -Force" >nul 2>&1

    :: The zip contains a top-level folder like EZ-CorridorKey-main
    set "UPDATE_INNER="
    for /d %%d in ("!UPDATE_EXTRACT!\EZ-CorridorKey-*") do set "UPDATE_INNER=%%d"
    if not defined UPDATE_INNER (
        echo   [ERROR] Unexpected archive structure.
        goto :update_cleanup
    )

    :: Copy new files over existing, skip user data dirs
    echo   Applying update (preserving .venv, tools, Projects)...
    robocopy "!UPDATE_INNER!" "%~dp0." /e /xd .venv venv tools Projects _BACKUPS __pycache__ .mypy_cache /xf *.pyc /njh /njs /ndl /nc /ns >nul 2>&1

    echo   [OK] Code updated via ZIP download

    :update_cleanup
    if exist "!UPDATE_ZIP!" del "!UPDATE_ZIP!" >nul 2>&1
    if exist "!UPDATE_EXTRACT!" rmdir /s /q "!UPDATE_EXTRACT!" >nul 2>&1
)

:: ── Step 1b: Ensure local tools are on PATH ──
if exist "%~dp0tools\ffmpeg\bin\ffmpeg.exe" set "PATH=%~dp0tools\ffmpeg\bin;%PATH%"

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

:: Auto-relaunch if called with --relaunch flag
if "%~1"=="--relaunch" (
    echo   Relaunching CorridorKey...
    call "%~dp02-start.bat"
    exit /b 0
)

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
