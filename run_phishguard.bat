@echo off
setlocal enabledelayedexpansion
title PhishGuard AI - Phishing Email SOC Dashboard (Port 8502)
echo =====================================================================
echo   Starting PhishGuard AI (Project 2: Phishing Email Detector)
echo   URL: http://localhost:8502
echo =====================================================================
echo.

cd /d "%~dp0"

REM 1. Prefer the local virtual environment if it exists
if exist ".venv\Scripts\streamlit.exe" (
    cd "Project 2"
    "..\.venv\Scripts\streamlit.exe" run app.py --server.port 8502
    goto :end
)

REM 2. Otherwise auto-detect Python
set "PY_CMD="
where python >nul 2>nul && set "PY_CMD=python"
if not defined PY_CMD (
    where py >nul 2>nul && set "PY_CMD=py"
)
if not defined PY_CMD (
    for %%V in (Python313 Python312 Python311 Python310 Python39) do (
        if exist "%LOCALAPPDATA%\Programs\Python\%%V\python.exe" (
            set "PY_CMD=%LOCALAPPDATA%\Programs\Python\%%V\python.exe"
        )
    )
)

if defined PY_CMD (
    cd "Project 2"
    "%PY_CMD%" -m streamlit run app.py --server.port 8502
) else (
    echo [ERROR] Virtual environment not found and Python could not be detected.
    echo Please run setup_local.bat first to set up the environment!
    pause
)

:end
