@echo off
rem One-time setup: creates .venv and installs the Python packages.
cd /d "%~dp0"
py -3 -m venv .venv || python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
echo.
echo Done. Run "run.bat --windowed" and press F2 for the settings panel.
pause
