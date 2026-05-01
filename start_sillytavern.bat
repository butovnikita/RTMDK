@echo off
title RTMDK SillyTavern Launcher
chcp 65001 >nul
echo.
echo ╔══════════════════════════════════════════════════╗
echo ║       RTMDK SillyTavern Launcher v1.0.0         ║
echo ║  Starts RTMDK Server + SillyTavern Proxy        ║
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
    pip install -r requirements-home.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies!
        pause
        exit /b 1
    )
)
echo [+] Dependencies OK
echo.

REM Start launcher
echo [*] Starting RTMDK + SillyTavern Proxy...
echo.
python rtmdk_sillytavern_launcher.py %*
pause
