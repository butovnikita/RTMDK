@echo off
title RTMDK SillyTavern Proxy (Standalone)
chcp 65001 >nul
echo.
echo ╔══════════════════════════════════════════════════╗
echo ║       RTMDK SillyTavern Proxy v1.0.0            ║
echo ║  Standalone proxy for SillyTavern integration    ║
echo ╚══════════════════════════════════════════════════╝
echo.
echo  NOTE: Make sure RTMDK Server is running on port 8080
echo        and LM Studio is running on port 12345
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found! Please install Python 3.10+
    pause
    exit /b 1
)

REM Start proxy
python rtmdk_st_proxy.py %*
pause
