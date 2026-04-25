@echo off
setlocal enabledelayedexpansion
TITLE EZ-CorridorKey Installer
cd /d "%~dp0"

echo.
echo  ========================================
echo   EZ-CorridorKey - One-Click Installer
echo  ========================================
echo.

set "PYTHON_SPEC=%CORRIDORKEY_PYTHON_VERSION%"
if not defined PYTHON_SPEC set "PYTHON_SPEC=3.11"
set "INSTALL_PYTHON="
set "PYVER="

echo [0/7] Installer runtime target...
echo   CorridorKey will provision and use managed Python !PYTHON_SPEC! for this install.
echo.

REM ── Step 1b: Check Visual Studio Build Tools (needed to compile OpenEXR etc.) ──
echo [1b/7] Checking C++ build tools...
where cl >nul 2>&1
if %errorlevel%==0 (
    echo   [OK] MSVC compiler found
    goto :buildtools_done
)

REM Check common VS Build Tools install locations via vswhere
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if exist "!VSWHERE!" (
    for /f "tokens=*" %%p in ('"!VSWHERE!" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2^>nul') do (
        if exist "%%p\VC\Auxiliary\Build\vcvarsall.bat" (
            echo   [OK] VS Build Tools found at %%p
            goto :buildtools_done
        )
    )
)

echo   [WARN] Visual Studio Build Tools not found.
echo   Some packages (OpenEXR) may need to compile from source.
echo.
set /p INSTALL_BT="  Install Visual Studio Build Tools now? (~2-7GB, requires admin) [Y/n]: "
if /i "!INSTALL_BT!"=="n" (
    echo   Skipping — if pip install fails later, you'll need Build Tools.
    echo   Manual install: https://visualstudio.microsoft.com/visual-cpp-build-tools/
    goto :buildtools_done
)

REM Launch VS Build Tools install in a SEPARATE window (fully detached)
REM so it cannot kill our installer window under any circumstances.
echo   Installing VS Build Tools (this may take 5-10 minutes)...
echo.
REM Write winget command to temp file to avoid nested-quote escaping hell
>"%TEMP%\_install_bt.bat" echo winget install Microsoft.VisualStudio.2022.BuildTools --override "--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --quiet --wait" --accept-source-agreements --accept-package-agreements
start "VS Build Tools Install" /wait cmd /c "%TEMP%\_install_bt.bat"
del "%TEMP%\_install_bt.bat" >nul 2>&1
echo.
REM Verify it actually installed
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if exist "!VSWHERE!" (
    for /f "tokens=*" %%p in ('"!VSWHERE!" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2^>nul') do (
        echo   [OK] VS Build Tools found at %%p
        goto :buildtools_done
    )
)
where cl >nul 2>&1
if %errorlevel%==0 (
    echo   [OK] MSVC compiler found
    goto :buildtools_done
)
echo   [WARN] VS Build Tools not detected after install.
echo   You can install manually later if pip fails:
echo     https://visualstudio.microsoft.com/visual-cpp-build-tools/
echo     Select "Desktop development with C++"

:buildtools_done
echo.

REM ── Step 1c: Check Git (needed for updates) ──
echo [1c/7] Checking Git...
where git >nul 2>&1
if %errorlevel%==0 (
    echo   [OK] Git found
    goto :git_done
)

echo   Git not found. Git is needed for updates (3-update.bat).
set /p INSTALL_GIT="  Install Git now? [Y/n]: "
if /i "!INSTALL_GIT!"=="n" (
    echo   Skipping — updates will use ZIP download fallback instead of git pull.
    goto :git_done
)

where winget >nul 2>&1
if %errorlevel%==0 (
    echo   Installing Git in a separate window...
    start "Git Install" /wait cmd /c "winget install Git.Git --accept-source-agreements --accept-package-agreements 2>&1 & pause"
    REM Check if git is now available (don't trust winget exit code)
    if exist "%ProgramFiles%\Git\cmd\git.exe" (
        echo   [OK] Git installed
        goto :git_done
    )
    if exist "%LOCALAPPDATA%\Programs\Git\cmd\git.exe" (
        echo   [OK] Git installed
        goto :git_done
    )
    echo   [WARN] Git install may not have completed.
)

echo   [INFO] Install Git manually from https://git-scm.com
echo   Without Git, 3-update.bat will download updates as a ZIP instead.

:git_done

REM If git is available but this isn't a repo (ZIP download), link it
REM After winget install, git may not be on PATH yet — check common locations
set "GIT_CMD="
where git >nul 2>&1
if %errorlevel%==0 (
    set "GIT_CMD=git"
) else (
    if exist "%ProgramFiles%\Git\cmd\git.exe" set "GIT_CMD=%ProgramFiles%\Git\cmd\git.exe"
    if exist "%ProgramFiles(x86)%\Git\cmd\git.exe" set "GIT_CMD=%ProgramFiles(x86)%\Git\cmd\git.exe"
    if exist "%LOCALAPPDATA%\Programs\Git\cmd\git.exe" set "GIT_CMD=%LOCALAPPDATA%\Programs\Git\cmd\git.exe"
)

if defined GIT_CMD (
    if not exist ".git" (
        echo   Linking to git repo for future updates...
        "!GIT_CMD!" init >nul 2>&1
        "!GIT_CMD!" remote add origin https://github.com/edenaion/EZ-CorridorKey.git >nul 2>&1
        "!GIT_CMD!" fetch origin >nul 2>&1
        "!GIT_CMD!" reset --mixed origin/main >nul 2>&1 || "!GIT_CMD!" reset --mixed origin/master >nul 2>&1
        if !errorlevel!==0 (
            echo   [OK] Linked to git — 3-update.bat will use git pull
        ) else (
            echo   [WARN] Git link failed — 3-update.bat will use ZIP fallback
        )
    )
)
echo.

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
echo [2/7] Setting up package manager...
set UV_AVAILABLE=0
set "UV_USER_EXE="

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

if exist "%USERPROFILE%\.cargo\bin\uv.exe" (
    set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
    set UV_AVAILABLE=1
    echo   [OK] uv found at %USERPROFILE%\.cargo\bin
    goto :uv_done
)

for /f "delims=" %%i in ('python -c "import os, site; print(os.path.join(site.USER_BASE, 'Scripts', 'uv.exe'))" 2^>nul') do set "UV_USER_EXE=%%i"
if defined UV_USER_EXE (
    if exist "!UV_USER_EXE!" (
        for %%d in ("!UV_USER_EXE!") do set "PATH=%%~dpd;%PATH%"
        set UV_AVAILABLE=1
        echo   [OK] uv found in Python user Scripts
        goto :uv_done
    )
)

echo   uv not found. Installing standalone uv...
powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; irm https://astral.sh/uv/install.ps1 | iex" >nul 2>&1

if defined UV_USER_EXE (
    if exist "!UV_USER_EXE!" (
        for %%d in ("!UV_USER_EXE!") do set "PATH=%%~dpd;%PATH%"
        set UV_AVAILABLE=1
        echo   [OK] uv installed to Python user Scripts
        goto :uv_done
    )
)

if exist "%USERPROFILE%\.local\bin\uv.exe" (
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
    set UV_AVAILABLE=1
    echo   [OK] uv installed
    goto :uv_done
)

if exist "%USERPROFILE%\.cargo\bin\uv.exe" (
    set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
    set UV_AVAILABLE=1
    echo   [OK] uv installed
    goto :uv_done
)

echo   [WARN] uv install failed.
echo   Manual install: https://docs.astral.sh/uv/getting-started/installation/

:uv_done

REM ── Step 4: Provision managed/system Python ──
echo [3/7] Provisioning Python...
if !UV_AVAILABLE!==1 (
    uv python install --managed-python !PYTHON_SPEC! >nul 2>&1
    if !errorlevel! equ 0 (
        for /f "usebackq delims=" %%i in (`uv python find --managed-python !PYTHON_SPEC! 2^>nul`) do set "INSTALL_PYTHON=%%i"
    ) else (
        echo   [WARN] Managed Python download failed, trying a supported system Python...
    )
)
if not defined INSTALL_PYTHON call :resolve_system_python
if not defined INSTALL_PYTHON (
    echo   [ERROR] Could not find a supported Python runtime.
    echo   CorridorKey needs Python 3.10-3.13, and the one-click installer targets managed Python !PYTHON_SPEC!.
    echo   Please install uv or Python 3.11 manually, then rerun this installer.
    goto :fail
)
for /f "tokens=2 delims= " %%v in ('"!INSTALL_PYTHON!" --version 2^>^&1') do set "PYVER=%%v"
echo   [OK] Using Python !PYVER!

REM ── Step 5: Create venv + install dependencies ──
echo [4/7] Installing dependencies...

if !UV_AVAILABLE!==0 goto :use_pip

echo   Creating virtual environment...
uv venv --clear --managed-python --python "!INSTALL_PYTHON!" .venv >nul 2>&1
if %errorlevel% neq 0 (
    echo   [WARN] uv venv creation failed, trying pip fallback...
    goto :use_pip
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
echo   Creating virtual environment...
if exist ".venv" rmdir /s /q ".venv" >nul 2>&1
"!INSTALL_PYTHON!" -m venv .venv
if %errorlevel% neq 0 (
    echo   [ERROR] Failed to create .venv with !INSTALL_PYTHON!
    goto :fail
)
call .venv\Scripts\activate.bat

REM Detect CUDA version for correct PyTorch wheel
set "INDEX_URL="
set "DRIVER="
set "CUDA_LINE="
set "CUDA_VERSION="
set "CUDA_WHEEL_LABEL="
set "CUDA_NOTE="
for /f "usebackq tokens=1,* delims==" %%A in (`.venv\Scripts\python.exe scripts\detect_windows_torch_index.py --format env`) do set "%%A=%%B"
if not defined INDEX_URL set "INDEX_URL=https://download.pytorch.org/whl/cpu"
if not defined CUDA_NOTE set "CUDA_NOTE=Could not run CUDA detection helper; installing CPU-only PyTorch."

if defined DRIVER echo(  NVIDIA driver detected: !DRIVER!
if defined CUDA_LINE echo(  !CUDA_LINE!
if /i "!CUDA_DETECT_MODE!"=="nvidia" (
    echo(  Using !CUDA_WHEEL_LABEL!
) else (
    echo(  !CUDA_NOTE!
    if /i "!CUDA_DETECT_REASON!"=="nvidia_smi_not_found" (
        echo(  If you have an NVIDIA GPU, ensure drivers are installed and nvidia-smi works.
    )
)

:pip_install
echo   Installing packages via pip (this may take a few minutes)...
.venv\Scripts\python.exe -m pip install --upgrade pip >nul 2>&1
.venv\Scripts\python.exe -m pip install --extra-index-url !INDEX_URL! -e . 2>&1
if !errorlevel! neq 0 (
    echo   [ERROR] pip install failed
    goto :fail
)
echo   [OK] Dependencies installed via pip

:deps_done

echo [4b/7] Verifying torch runtime...
set "DIAGNOSTICS_DIR=logs\diagnostics"
set "TORCH_RUNTIME_LOG=%DIAGNOSTICS_DIR%\install-torch-runtime.json"
set "SUPPORT_REPORT_LOG=%DIAGNOSTICS_DIR%\install-support-report.md"
if not exist "%DIAGNOSTICS_DIR%" mkdir "%DIAGNOSTICS_DIR%"
.venv\Scripts\python.exe scripts\verify_torch_runtime.py --log "%TORCH_RUNTIME_LOG%"
if %errorlevel% neq 0 (
    echo   [ERROR] Installed torch runtime did not validate.
    echo   Preparing a pre-filled GitHub issue to help you report this.
    .venv\Scripts\python.exe scripts\open_installer_issue.py --json "%TORCH_RUNTIME_LOG%" --body-out "%SUPPORT_REPORT_LOG%" --stage "windows-installer-runtime-verification"
    if %errorlevel% neq 0 (
        echo   [WARN] Could not open the GitHub issue helper automatically.
    )
    echo   See %TORCH_RUNTIME_LOG% and %SUPPORT_REPORT_LOG% for details.
    goto :fail
)
echo   [OK] Torch runtime verified

echo [4c/7] Verifying Triton...
.venv\Scripts\python.exe -c "import triton, torch; print(triton.__version__); print(torch.__version__)" >nul 2>&1
if %errorlevel%==0 (
    echo   [OK] Triton import verified
) else (
    echo   [WARN] Triton import check failed. CorridorKey will fall back to eager mode if compile is unavailable.
    echo   [WARN] You can retry manually with:
    echo     .venv\Scripts\python.exe -m pip install -U "triton-windows^>=3.5,^<3.6"
)

REM ── Step 4d: Performance accelerators (TensorRT / Rust native) ──
echo [4d/7] Installing performance accelerators...

where nvidia-smi >nul 2>&1
if !errorlevel! equ 0 (
    echo   NVIDIA GPU detected, installing torch-tensorrt...
    .venv\Scripts\python.exe -m pip install torch-tensorrt >nul 2>&1
    if !errorlevel! equ 0 (
        echo   [OK] torch-tensorrt installed
    ) else (
        echo   [WARN] torch-tensorrt install failed ^(will use default backend^)
    )
    REM SageAttention: 2x on RTX 30xx/40xx, 5x on RTX 50xx. Opt-in via CORRIDORKEY_USE_SAGE=1.
    echo   Installing SageAttention ^(quantized attention, opt-in via env var^)...
    .venv\Scripts\python.exe -m pip install "sageattention>=2.2.0" >nul 2>&1
    if !errorlevel! equ 0 (
        echo   [OK] SageAttention installed ^(enable with CORRIDORKEY_USE_SAGE=1^)
    ) else (
        echo   [WARN] SageAttention install failed ^(SDPA fallback will be used^)
    )
) else (
    echo   No NVIDIA GPU detected, skipping torch-tensorrt
)

where cargo >nul 2>&1
if !errorlevel! neq 0 (
    echo   Installing Rust toolchain...
    powershell -Command "Invoke-WebRequest -Uri https://win.rustup.rs/x86_64 -OutFile rustup-init.exe" >nul 2>&1
    if exist rustup-init.exe (
        rustup-init.exe -y --default-toolchain stable >nul 2>&1
        del rustup-init.exe >nul 2>&1
        set "PATH=%USERPROFILE%\.cargo\bin;!PATH!"
        where cargo >nul 2>&1
        if !errorlevel! equ 0 (
            echo   [OK] Rust installed
        ) else (
            echo   [WARN] Rust install failed ^(Python fallback will be used^)
        )
    ) else (
        echo   [WARN] Rust download failed ^(Python fallback will be used^)
    )
)
where cargo >nul 2>&1
if !errorlevel! equ 0 (
    echo   Building native Rust extension...
    where maturin >nul 2>&1
    if !errorlevel! equ 0 (
        pushd corridorkey_native
        maturin develop --release >nul 2>&1
        if !errorlevel! equ 0 (
            echo   [OK] Rust native extension installed
        ) else (
            echo   [WARN] Rust build failed ^(Python fallback will be used^)
        )
        popd
    ) else (
        echo   [WARN] maturin not found, skipping Rust build ^(pip install maturin^)
    )
) else (
    echo   Rust not available, skipping native extension ^(Python fallback will be used^)
)

echo [4e/7] Optional SAM2 tracker...
set INSTALL_SAM2=y
if defined CORRIDORKEY_INSTALL_SAM2 (
    set "INSTALL_SAM2_INPUT=%CORRIDORKEY_INSTALL_SAM2%"
) else (
    set /p INSTALL_SAM2_INPUT="  Install SAM2 tracking support? [Y/n]: "
)
if /i "!INSTALL_SAM2_INPUT!"=="n" set INSTALL_SAM2=n

if /i "!INSTALL_SAM2!"=="y" (
    set SAM2_OK=0
    echo   Installing SAM2 tracker package...
    if !UV_AVAILABLE!==1 (
        uv pip install --python .venv\Scripts\python.exe --torch-backend=auto -e ".[tracker]" 2>&1
        if !errorlevel! equ 0 set SAM2_OK=1
    ) else (
        if defined INDEX_URL (
            .venv\Scripts\python.exe -m pip install --extra-index-url !INDEX_URL! -e ".[tracker]" 2>&1
        ) else (
            .venv\Scripts\python.exe -m pip install -e ".[tracker]" 2>&1
        )
        if !errorlevel! equ 0 set SAM2_OK=1
    )

    if !SAM2_OK!==1 (
        echo   [OK] SAM2 tracker support installed
        if defined CORRIDORKEY_PREDOWNLOAD_SAM2 (
            set "DOWNLOAD_SAM2=%CORRIDORKEY_PREDOWNLOAD_SAM2%"
        ) else (
            set /p DOWNLOAD_SAM2="  Pre-download default SAM2 Base+ model? (324MB) [Y/n]: "
        )
        if /i not "!DOWNLOAD_SAM2!"=="n" (
            .venv\Scripts\python.exe scripts\setup_models.py --sam2
        )
    ) else (
        echo   [WARN] SAM2 tracker install failed. CorridorKey will still run without Track Mask.
        echo   [WARN] You can retry later with:
        echo     .venv\Scripts\python.exe -m pip install -e ".[tracker]"
    )
)

REM ── Step 5: Check/Install FFmpeg ──
echo [4/7] Checking FFmpeg...
set "FFMPEG_DEST=%~dp0tools\ffmpeg"
.venv\Scripts\python.exe scripts\check_ffmpeg.py
if %errorlevel%==0 goto :ffmpeg_done

echo   Existing FFmpeg install is missing FFprobe, older than 7, or not a full Windows build.
echo   Downloading a full FFmpeg build...
set "FFMPEG_URL=https://github.com/BtbN/FFmpeg-Builds/releases/latest/download/ffmpeg-master-latest-win64-gpl.zip"
set "FFMPEG_ZIP=%TEMP%\ffmpeg-master-latest-win64-gpl.zip"
set "FFMPEG_EXTRACT=%TEMP%\ffmpeg-extract"

REM Prefer curl to avoid PowerShell download heuristics
echo   Downloading ffmpeg (this may take a minute)...
where curl.exe >nul 2>&1
if %errorlevel%==0 (
    curl.exe -L --fail --output "%FFMPEG_ZIP%" "%FFMPEG_URL%" >nul 2>&1
) else (
    powershell -NoProfile -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%FFMPEG_URL%' -OutFile '%FFMPEG_ZIP%'" >nul 2>&1
)
if not exist "%FFMPEG_ZIP%" (
    echo   [WARN] FFmpeg download failed. Install manually:
    echo     winget install ffmpeg
    echo     choco install ffmpeg
    echo     Ensure both ffmpeg and ffprobe are available, version 7+.
    echo     https://ffmpeg.org/download.html
    goto :ffmpeg_done
)

REM Extract and move into tools\ffmpeg
echo   Extracting...
if exist "%FFMPEG_EXTRACT%" rmdir /s /q "%FFMPEG_EXTRACT%"
mkdir "%FFMPEG_EXTRACT%" >nul 2>&1
where tar.exe >nul 2>&1
if %errorlevel%==0 (
    tar.exe -xf "%FFMPEG_ZIP%" -C "%FFMPEG_EXTRACT%" >nul 2>&1
) else (
    powershell -NoProfile -Command "Expand-Archive -Path '%FFMPEG_ZIP%' -DestinationPath '%FFMPEG_EXTRACT%' -Force" >nul 2>&1
)

REM The zip contains a top-level folder like ffmpeg-master-latest-win64-gpl
set "FFMPEG_INNER="
for /d %%d in ("%FFMPEG_EXTRACT%\ffmpeg-*") do set "FFMPEG_INNER=%%d"
if not defined FFMPEG_INNER (
    echo   [WARN] FFmpeg extraction failed — unexpected archive structure.
    goto :ffmpeg_cleanup
)

if not exist "%~dp0tools" mkdir "%~dp0tools"
if exist "%FFMPEG_DEST%" rmdir /s /q "%FFMPEG_DEST%"
move "!FFMPEG_INNER!" "%FFMPEG_DEST%" >nul 2>&1

if exist "%FFMPEG_DEST%\bin\ffmpeg.exe" (
    set "PATH=%FFMPEG_DEST%\bin;%PATH%"
    .venv\Scripts\python.exe scripts\check_ffmpeg.py
    if !errorlevel! equ 0 (
        echo   [OK] FFmpeg installed to tools\ffmpeg
    ) else (
        echo   [WARN] Downloaded FFmpeg still failed validation.
    )
) else (
    echo   [WARN] FFmpeg install failed. Install manually:
    echo     winget install ffmpeg
    echo     choco install ffmpeg
    echo     Ensure both ffmpeg and ffprobe are available, version 7+.
    echo     https://ffmpeg.org/download.html
)

:ffmpeg_cleanup
if exist "%FFMPEG_ZIP%" del "%FFMPEG_ZIP%" >nul 2>&1
if exist "%FFMPEG_EXTRACT%" rmdir /s /q "%FFMPEG_EXTRACT%" >nul 2>&1

:ffmpeg_done

REM ── Step 6: Download model weights ──
echo [5/7] Checking model weights...

.venv\Scripts\python.exe scripts\setup_models.py --check
.venv\Scripts\python.exe scripts\setup_models.py --corridorkey
if %errorlevel% neq 0 (
    echo   [WARN] CorridorKey model download failed. You can retry later:
    echo     .venv\Scripts\python scripts\setup_models.py --corridorkey
)

echo.
echo [6/7] Optional models (can be downloaded later)
echo.
if defined CORRIDORKEY_INSTALL_GVM (
    set "INSTALL_GVM=%CORRIDORKEY_INSTALL_GVM%"
) else (
    set /p INSTALL_GVM="  Download GVM alpha generator? (~6GB) [y/N]: "
)
if /i "!INSTALL_GVM!"=="y" (
    .venv\Scripts\python.exe scripts\setup_models.py --gvm
)

if defined CORRIDORKEY_INSTALL_VIDEOMAMA (
    set "INSTALL_VM=%CORRIDORKEY_INSTALL_VIDEOMAMA%"
) else (
    set /p INSTALL_VM="  Download VideoMaMa alpha generator? (~37GB) [y/N]: "
)
if /i "!INSTALL_VM!"=="y" (
    .venv\Scripts\python.exe scripts\setup_models.py --videomama
)

REM ── Step 7: Create desktop shortcut ──
echo.
echo [7/7] Desktop shortcut
if defined CORRIDORKEY_CREATE_SHORTCUT (
    set "CREATE_SHORTCUT=%CORRIDORKEY_CREATE_SHORTCUT%"
) else (
    set /p CREATE_SHORTCUT="  Create desktop shortcut? [Y/n]: "
)
if /i "!CREATE_SHORTCUT!"=="n" goto :skip_shortcut

set "SHORTCUT_PATH=%USERPROFILE%\Desktop\CorridorKey.lnk"
set "TARGET_PATH=%CD%\.venv\Scripts\pythonw.exe"
set "ICON_PATH=%CD%\ui\theme\corridorkey.ico"
set "WORK_DIR=%CD%"

powershell -NoProfile -Command ^
    "$ws = New-Object -ComObject WScript.Shell; " ^
    "$s = $ws.CreateShortcut('!SHORTCUT_PATH!'); " ^
    "$s.TargetPath = '!TARGET_PATH!'; " ^
    "$s.Arguments = 'main.py'; " ^
    "$s.WorkingDirectory = '!WORK_DIR!'; " ^
    "$s.IconLocation = '!ICON_PATH!,0'; " ^
    "$s.WindowStyle = 7; " ^
    "$s.Description = 'CorridorKey - AI Green Screen'; " ^
    "$s.Save()" >nul 2>&1

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

:resolve_system_python
set "INSTALL_PYTHON="
python --version >nul 2>&1
if %errorlevel% neq 0 goto :eof
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set "SYSTEM_PYVER=%%v"
for /f "tokens=1,2 delims=." %%a in ("!SYSTEM_PYVER!") do (
    set SYSTEM_PYMAJOR=%%a
    set SYSTEM_PYMINOR=%%b
)
if !SYSTEM_PYMAJOR! LSS 3 goto :eof
if !SYSTEM_PYMAJOR!==3 if !SYSTEM_PYMINOR! LSS 10 goto :eof
if !SYSTEM_PYMAJOR!==3 if !SYSTEM_PYMINOR! GEQ 14 goto :eof
set "INSTALL_PYTHON=python"
goto :eof

:fail
echo.
echo  Installation failed. See errors above.
echo.
pause
exit /b 1
