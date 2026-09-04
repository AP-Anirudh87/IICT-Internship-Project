@echo off
setlocal enabledelayedexpansion
title IICT Project - Automated Local Setup
echo =====================================================================
echo   IICT Internship Project: Automated Local Setup (.venv)
echo =====================================================================
echo.

REM 1. Auto-detect Python across any Windows computer
set "PY_CMD="

REM Check if 'python' is on PATH
where python >nul 2>nul
if %errorlevel% equ 0 (
    set "PY_CMD=python"
    goto :found_python
)

REM Check if 'py' launcher is available
where py >nul 2>nul
if %errorlevel% equ 0 (
    set "PY_CMD=py"
    goto :found_python
)

REM Check user AppData locations (works for any Windows username via %LOCALAPPDATA%)
for %%V in (Python313 Python312 Python311 Python310 Python39) do (
    if exist "%LOCALAPPDATA%\Programs\Python\%%V\python.exe" (
        set "PY_CMD=%LOCALAPPDATA%\Programs\Python\%%V\python.exe"
        goto :found_python
    )
)

REM Check system Program Files locations
for %%V in (Python313 Python312 Python311 Python310 Python39) do (
    if exist "C:\Program Files\Python\%%V\python.exe" (
        set "PY_CMD=C:\Program Files\Python\%%V\python.exe"
        goto :found_python
    )
)

echo [ERROR] No Python installation found on this system.
echo Please install Python 3.9+ from https://www.python.org/
echo (Make sure to check "Add Python to PATH" during installation)
pause
exit /b 1

:found_python
echo [INFO] Detected Python: %PY_CMD%
echo.

cd /d "%~dp0"

echo [1/3] Creating isolated virtual environment (.venv)...
if not exist ".venv" (
    "%PY_CMD%" -m venv .venv
) else (
    echo [INFO] .venv already exists. Skipping creation.
)

echo [2/3] Upgrading pip...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip --quiet

echo [3/3] Installing dependencies from requirements.txt...
pip install -r requirements.txt

echo.
echo =====================================================================
echo   SUCCESS! Environment setup is complete on this PC.
echo   You can now run:
echo     - run_truthguard.bat  (Project 1 on http://localhost:8501)
echo     - run_phishguard.bat   (Project 2 on http://localhost:8502)
echo =====================================================================
pause
