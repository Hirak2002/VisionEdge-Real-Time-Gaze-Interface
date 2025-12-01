@echo off
title Gaze Tracker - Python Demo
color 0A

echo.
echo ============================================================
echo           C++ Gaze Tracker - Python Demo
echo ============================================================
echo.
echo This will start the gaze tracking application.
echo.
echo CONTROLS:
echo   - SPACE : Toggle mouse control ON/OFF
echo   - ESC   : Exit application
echo.
echo TIPS:
echo   - Ensure good lighting
echo   - Position camera at eye level
echo   - Sit 50-70cm from camera
echo.
echo ============================================================
echo.
echo Starting in 3 seconds...
timeout /t 3 /nobreak >nul

echo.
echo [*] Launching gaze tracker...
echo.

python demo_python.py

if %errorLevel% neq 0 (
    echo.
    echo [ERROR] Failed to run demo. Make sure:
    echo   1. Python is installed
    echo   2. Required packages are installed:
    echo      pip install torch opencv-python pyautogui
    echo   3. Webcam is connected and working
    echo.
    pause
)
