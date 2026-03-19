@echo off
TITLE AMD Installer - EZ-CorridorKey
cd /d "%~dp0"

echo.
echo =====================================================
echo   EZ-CorridorKey - AMD Radeon Setup
echo   github.com/toowyred  ^|  IG: @wyred.on
echo =====================================================
echo.

echo [1/5] Activating environment...
call .venv\Scripts\activate.bat

echo.
echo [2/5] Removing conflicting packages...
python -m pip uninstall onnxruntime onnxruntime-gpu -y 2>nul
echo     Done.

echo.
echo [3/5] Installing AMD AI support...
python -m pip install torch-directml torchvision numpy PySide6 timm transformers onnx onnxruntime-directml --upgrade
echo     Done.

echo.
echo [4/5] Exporting model to ONNX (required for AMD GPU acceleration)...
echo     This may take 1-2 minutes on first run...
python export_to_onnx.py
if %errorlevel% neq 0 (
    echo.
    echo [WARNING] ONNX export failed. You can try running it manually later:
    echo     python export_to_onnx.py
    echo     The app will still launch but may run slower.
) else (
    echo     ONNX export complete.
)

echo.
echo [5/5] Done!
echo.
echo =====================================================
echo   Your AMD card is ready to go.
echo   Launch the app with: 2-start.bat
echo.
echo   Note: The first inference pass after launch will
echo   be slow while DirectML compiles GPU shaders.
echo   Subsequent passes will be significantly faster.
echo =====================================================
echo.
echo   Follow @wyred.on on IG if this works for you :)
echo.
pause
