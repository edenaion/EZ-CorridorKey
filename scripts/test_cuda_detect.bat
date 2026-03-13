@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0\.."

echo ---- Windows CUDA Detection Smoke Test ----

set PASS=0
set FAIL=0
set "TMPDIR=%TEMP%\corridorkey_cuda_detect_%RANDOM%_%RANDOM%"
mkdir "%TMPDIR%" >nul 2>&1

> "%TMPDIR%\cuda13.txt" echo ^| NVIDIA-SMI 581.57     Driver Version: 581.57     CUDA Version: 13.0     ^|
> "%TMPDIR%\cuda126.txt" echo ^| NVIDIA-SMI 551.61     Driver Version: 551.61     CUDA Version: 12.3     ^|
> "%TMPDIR%\cuda128.txt" echo ^| NVIDIA-SMI 572.16     Driver Version: 572.16     CUDA Version: 12.8     ^|
> "%TMPDIR%\cuda_es.txt" echo ^| NVIDIA-SMI 581.57     Version del controlador: 581.57     Version de CUDA: 12.3     ^|
> "%TMPDIR%\cuda_unsupported.txt" echo ^| NVIDIA-SMI 537.58     Driver Version: 537.58     CUDA Version: 11.8     ^|

for %%C in (437 850 1252 65001) do (
    chcp %%C >nul 2>&1
    if errorlevel 1 (
        echo [SKIP] Codepage %%C not available
    ) else (
        call :run_case %%C "%TMPDIR%\cuda13.txt" cu130 "English CUDA 13.0"
        call :run_case %%C "%TMPDIR%\cuda126.txt" cu126 "English CUDA 12.3"
        call :run_case %%C "%TMPDIR%\cuda128.txt" cu128 "English CUDA 12.8"
        call :run_case %%C "%TMPDIR%\cuda_es.txt" cu126 "Spanish-style CUDA 12.3"
        call :run_case %%C "%TMPDIR%\cuda_unsupported.txt" cpu "Unsupported CUDA 11.8"
    )
)

chcp 437 >nul 2>&1
rmdir /s /q "%TMPDIR%" >nul 2>&1

echo.
echo Results: !PASS! passed, !FAIL! failed
if !FAIL! gtr 0 exit /b 1
exit /b 0

:run_case
set "CP=%~1"
set "MOCK_FILE=%~2"
set "EXPECTED=%~3"
set "LABEL=%~4"
if /i "%EXPECTED%"=="cu128" set "EXPECTED=https://download.pytorch.org/whl/cu128"
if /i "%EXPECTED%"=="cu126" set "EXPECTED=https://download.pytorch.org/whl/cu126"
if /i "%EXPECTED%"=="cu130" set "EXPECTED=https://download.pytorch.org/whl/cu130"
if /i "%EXPECTED%"=="cpu" set "EXPECTED=https://download.pytorch.org/whl/cpu"
for %%V in (CUDA_DETECT_MODE CUDA_DETECT_REASON INDEX_URL CUDA_WHEEL_LABEL CUDA_NOTE DRIVER CUDA_VERSION CUDA_LINE NVIDIA_SMI_PATH) do set "%%V="
set "CORRIDORKEY_MOCK_NVIDIA_SMI_FILE=%MOCK_FILE%"
for /f "usebackq tokens=1,* delims==" %%A in (`python scripts\detect_windows_torch_index.py --format env`) do set "%%A=%%B"
if /i "!INDEX_URL!"=="%EXPECTED%" (
    echo [PASS] CP %CP%: %LABEL% URL !INDEX_URL!
    set /a PASS+=1
) else (
    echo [FAIL] CP %CP%: %LABEL% expected %EXPECTED%, got !INDEX_URL!
    set /a FAIL+=1
)
exit /b 0
