@echo off
setlocal enabledelayedexpansion
TITLE EZ-CorridorKey Updater
cd /d "%~dp0"

echo.
echo  ========================================
echo   EZ-CorridorKey - Update
echo  ========================================
echo.

REM ── Step 1: Pull latest code ──
echo [1/3] Pulling latest changes...

REM Check if this is a git repo AND git is available
set USE_GIT=0
git --version >nul 2>&1
if %errorlevel%==0 if exist ".git" set USE_GIT=1

if !USE_GIT!==1 (
    call :migrate_git_branch
    git pull --recurse-submodules 2>&1
    if %errorlevel% neq 0 (
        echo   [WARN] Git pull had issues. You may have local changes.
        echo   If stuck, try: git stash ^&^& git pull ^&^& git stash pop
    ) else (
        echo   [OK] Code updated via git
    )
) else (
    goto :zip_update
)
goto :after_update

:zip_update
echo   No git repo detected - downloading latest release as ZIP...
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

REM The zip contains a top-level folder like EZ-CorridorKey-main
set "UPDATE_INNER="
for /d %%d in ("!UPDATE_EXTRACT!\EZ-CorridorKey-*") do set "UPDATE_INNER=%%d"
if not defined UPDATE_INNER (
    echo   [ERROR] Unexpected archive structure.
    goto :update_cleanup
)

REM Copy new files over existing, skip user data dirs
echo.
echo   WARNING: No git detected. Updating via ZIP download.
echo.
echo   WILL OVERWRITE: all .py, .bat, .sh, .yaml, and other app files
echo   WILL NOT TOUCH: Projects folder, model weights, .venv, tools
echo.
echo   If you have local code changes, they will be lost.
echo.
set /p "CONFIRM=  Continue? [Y/n] "
if /i "!CONFIRM!"=="n" goto :update_cleanup
if /i "!CONFIRM!"=="no" goto :update_cleanup
echo   Applying update (preserving .venv, tools, Projects, model weights)...
robocopy "!UPDATE_INNER!" "%~dp0." /e /xd .venv venv tools Projects _BACKUPS __pycache__ .mypy_cache checkpoints weights /xf *.pyc *.pth *.safetensors *.bin *.pt /njh /njs /ndl /nc /ns >nul 2>&1

echo   [OK] Code updated via ZIP download

:update_cleanup
if exist "!UPDATE_ZIP!" del "!UPDATE_ZIP!" >nul 2>&1
if exist "!UPDATE_EXTRACT!" rmdir /s /q "!UPDATE_EXTRACT!" >nul 2>&1

:after_update

REM ── Step 1b: Ensure local tools are on PATH ──
if exist "%~dp0tools\ffmpeg\bin\ffmpeg.exe" set "PATH=%~dp0tools\ffmpeg\bin;%PATH%"

REM ── Step 2: Update dependencies ──
echo [2/3] Updating dependencies...

set UV_AVAILABLE=0
set "INSTALL_TARGET=-e ."
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

if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('sam2') else 1)" >nul 2>&1
    if !errorlevel! == 0 (
        set "INSTALL_TARGET=-e .[tracker]"
        echo   SAM2 tracker detected - updating tracker extras too
    )
)

if !UV_AVAILABLE!==1 (
    uv pip install --python .venv\Scripts\python.exe --torch-backend=auto !INSTALL_TARGET! 2>&1
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
        pip install !INSTALL_TARGET! 2>&1
        echo   [OK] Dependencies updated via pip
    ) else (
        echo   [ERROR] No .venv found. Run 1-install.bat first.
        goto :fail
    )
)

REM Clean up retired distributions that can survive editable upgrades.
if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe -m pip show pynvml >nul 2>&1
    if !errorlevel! == 0 (
        echo   Removing legacy pynvml package...
        .venv\Scripts\python.exe -m pip uninstall -y pynvml >nul 2>&1
    )
)

REM ── Step 3: Check for new model weights ──
echo [3/3] Checking model weights...
.venv\Scripts\python.exe scripts\setup_models.py --check

REM ── Done ──
echo.
echo  ========================================
echo   Update complete!
echo  ========================================
echo.

REM Auto-relaunch if called with --relaunch flag
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

:migrate_git_branch
set "CURRENT_BRANCH="
set "CURRENT_UPSTREAM="
for /f "tokens=*" %%b in ('git branch --show-current 2^>nul') do set "CURRENT_BRANCH=%%b"

if /i "!CURRENT_BRANCH!"=="master" (
    echo   [MIGRATE] Moving local branch from master to main...
    git fetch origin main --recurse-submodules >nul 2>&1
    git show-ref --verify --quiet refs/heads/main
    if !errorlevel! == 0 (
        git checkout main >nul 2>&1
    ) else (
        git branch -m master main >nul 2>&1
    )
    if !errorlevel! neq 0 (
        echo   [WARN] Automatic branch rename failed. Trying a fresh local main branch...
        git checkout -b main origin/main >nul 2>&1
    )
    if !errorlevel! neq 0 (
        echo   [WARN] Could not auto-migrate this checkout to main.
        echo   Finish this update, then run: git fetch origin ^&^& git checkout main
        exit /b 0
    )
    git branch --set-upstream-to=origin/main main >nul 2>&1
    if !errorlevel! == 0 echo   [OK] Now tracking origin/main
) else if /i "!CURRENT_BRANCH!"=="main" (
    for /f "tokens=*" %%u in ('git rev-parse --abbrev-ref --symbolic-full-name @{u} 2^>nul') do set "CURRENT_UPSTREAM=%%u"
    if /i "!CURRENT_UPSTREAM!"=="origin/master" (
        echo   [MIGRATE] Repointing main to track origin/main...
        git fetch origin main --recurse-submodules >nul 2>&1
        git branch --set-upstream-to=origin/main main >nul 2>&1
        if !errorlevel! == 0 echo   [OK] Now tracking origin/main
    )
)
exit /b 0
