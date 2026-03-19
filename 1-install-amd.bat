@echo off
TITLE AMD Fixer - Github-toowyred IG-wyred.on
cd /d "%~dp0"

echo [1/2] Opening environment...
call .venv\Scripts\activate.bat

echo [2/2] Installing AMD AI support...
python -m pip install torch-directml torchvision numpy PySide6 timm transformers --upgrade

echo.
echo DONE! Your AMD card is ready. 
echo Use 2-start.bat to start.
echo Follow @wyred.on on IG if this works for you :)
pause
