@echo off
title RTMDK Production Server
chcp 65001 >nul
echo.
echo ╔══════════════════════════════════════════════════╗
echo ║       RTMDK Production Server v8.0.0            ║
echo ║  OpenAI-compatible API (No SillyTavern)         ║
echo ╚══════════════════════════════════════════════════╝
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found! Please install Python 3.10+
    pause
    exit /b 1
)

REM Check dependencies
echo [*] Checking dependencies...
python -c "import fastapi" 2>nul
if errorlevel 1 (
    echo [!] Installing dependencies...
    pip install -r requirements-prod.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies!
        pause
        exit /b 1
    )
)
echo [+] Dependencies OK
echo.

REM Start production server
echo [*] Starting RTMDK Production Server...
echo.
python start_production.py %*
pause
