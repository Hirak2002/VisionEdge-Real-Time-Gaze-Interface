@echo off
echo ============================================================
echo    C++ Gaze Tracker - Automated Setup Script
echo ============================================================
echo.

REM Check if running as administrator
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo [WARNING] Not running as Administrator. Some installations may fail.
    echo.
)

echo [1/5] Checking for Visual Studio...
where cl.exe >nul 2>&1
if %errorLevel% == 0 (
    echo [OK] Visual Studio C++ compiler found
) else (
    echo [!] Visual Studio not found
    echo Please install Visual Studio 2022 Community Edition with "Desktop development with C++"
    echo Download: https://visualstudio.microsoft.com/downloads/
    echo.
    echo Press any key to open download page...
    pause >nul
    start https://visualstudio.microsoft.com/downloads/
    exit /b 1
)

echo.
echo [2/5] Checking for CMake...
where cmake >nul 2>&1
if %errorLevel% == 0 (
    echo [OK] CMake found
    cmake --version
) else (
    echo [!] CMake not found
    echo Installing CMake via winget...
    winget install -e --id Kitware.CMake
)

echo.
echo [3/5] Checking for vcpkg...
if exist "C:\vcpkg\vcpkg.exe" (
    echo [OK] vcpkg found at C:\vcpkg
) else (
    echo [!] vcpkg not found. Installing...
    cd C:\
    git clone https://github.com/Microsoft/vcpkg.git
    cd vcpkg
    call bootstrap-vcpkg.bat
)

echo.
echo [4/5] Installing OpenCV and ONNX Runtime...
cd C:\vcpkg
echo Installing OpenCV (this may take 10-20 minutes)...
vcpkg install opencv:x64-windows
echo Installing ONNX Runtime...
vcpkg install onnxruntime:x64-windows
echo Integrating vcpkg...
vcpkg integrate install

echo.
echo [5/5] Building the project...
cd "%~dp0"
mkdir build 2>nul
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build . --config Release

echo.
echo ============================================================
echo    Setup Complete!
echo ============================================================
echo.
echo To run the application:
echo   cd "%~dp0build\Release"
echo   GazeTracker.exe
echo.
echo Press any key to exit...
pause >nul
