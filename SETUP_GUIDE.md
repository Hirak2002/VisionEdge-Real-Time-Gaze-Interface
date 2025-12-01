# C++ Gaze Tracking Project - Complete Setup Guide

## Project Overview
On-Device Gaze Tracking for Accessibility using C++ and ONNX Runtime

**Technologies:**
- C++ for inference engine
- OpenCV for webcam capture
- ONNX Runtime for model inference
- PyTorch for model training

---

## ✅ COMPLETED STEPS

### 1. Model Training ✓
The gaze tracking model has been successfully trained and exported to ONNX format:
- **File:** `gaze_model.onnx` (4.2 MB)
- **Input:** (1, 1, 36, 60) - Grayscale eye image
- **Output:** (1, 2) - [x, y] gaze coordinates

---

## 🔧 SETUP INSTRUCTIONS

### Prerequisites

You need to install the following tools to build and run this project:

1. **Visual Studio 2022 Community Edition** (Free)
   - Download: https://visualstudio.microsoft.com/downloads/
   - During installation, select:
     - ✅ "Desktop development with C++"
     - ✅ "CMake tools for Windows"
   - This includes MSVC compiler and CMake

2. **vcpkg** (Package Manager for C++)
   ```powershell
   # Install vcpkg
   cd C:\
   git clone https://github.com/Microsoft/vcpkg.git
   cd vcpkg
   .\bootstrap-vcpkg.bat
   ```

3. **Install OpenCV and ONNX Runtime via vcpkg**
   ```powershell
   cd C:\vcpkg
   .\vcpkg install opencv:x64-windows
   .\vcpkg install onnxruntime:x64-windows
   .\vcpkg integrate install
   ```

---

## 🏗️ BUILD THE PROJECT

### Option A: Using Visual Studio 2022 (Recommended)

1. **Open Visual Studio 2022**
2. **File → Open → Folder** → Select this project folder
3. VS will automatically detect `CMakeLists.txt`
4. Click **"Build → Build All"** or press `Ctrl+Shift+B`
5. The executable will be in: `out\build\x64-Debug\GazeTracker.exe`

### Option B: Using Command Line (CMake)

```powershell
# Open "x64 Native Tools Command Prompt for VS 2022" from Start Menu

cd "C:\Users\hirak\OneDrive\Desktop\C++ Gaze Tracker"

# Configure
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake

# Build
cmake --build build --config Release

# Run
.\build\Release\GazeTracker.exe
```

---

## 🚀 RUNNING THE APPLICATION

Once built, run the executable:

```powershell
.\build\Release\GazeTracker.exe
```

**Expected Behavior:**
1. Webcam will turn on
2. Application detects your face
3. Extracts eye regions
4. Predicts gaze direction in real-time
5. Mouse cursor moves based on gaze

**Controls:**
- Press `ESC` to quit
- Press `SPACE` to pause/resume

---

## 🔧 ALTERNATIVE: Quick Start with Pre-built Binaries

If you don't want to build from source, I can create a simplified version using MinGW-w64:

### Install MinGW-w64 + Dependencies

1. **Install MinGW-w64:**
   ```powershell
   winget install -e --id=MSYS2.MSYS2
   ```

2. **Open MSYS2 MINGW64 terminal and install packages:**
   ```bash
   pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake mingw-w64-x86_64-opencv mingw-w64-x86_64-onnxruntime
   ```

3. **Build:**
   ```bash
   cd "/c/Users/hirak/OneDrive/Desktop/C++ Gaze Tracker"
   mkdir build && cd build
   cmake .. -G "MinGW Makefiles"
   mingw32-make
   ./GazeTracker.exe
   ```

---

## 📁 Project Structure

```
C++ Gaze Tracker/
├── main.cpp                    # Main inference engine
├── train_gaze_model.py         # Model training script
├── gaze_model.onnx             # Trained ONNX model (4.2 MB)
├── gaze_model.pt               # TorchScript version
├── CMakeLists.txt              # Build configuration
└── README.md                   # This file
```

---

## 🐛 Troubleshooting

### Issue: "Cannot find OpenCV" or "Cannot find ONNX Runtime"
**Solution:** Make sure vcpkg is properly integrated:
```powershell
cd C:\vcpkg
.\vcpkg integrate install
```

### Issue: "Webcam not found"
**Solution:** 
- Check if other apps are using the webcam
- Try changing camera index in `main.cpp` (line ~20):
  ```cpp
  cv::VideoCapture cap(0);  // Change 0 to 1 or 2
  ```

### Issue: "Model file not found"
**Solution:** Ensure `gaze_model.onnx` is in the same directory as the executable

### Issue: Path too long errors during Python setup
**Solution:** We already handled this by using a temporary environment at `C:\temp_onnx`

---

## 📊 Model Details

**Architecture:** Lightweight CNN
- Conv layers: 3 (32, 64, 128 channels)
- Fully connected: 3 layers
- Parameters: ~3.7M
- Input: 36x60 grayscale eye patch
- Output: 2D gaze vector (x, y)

**Training:**
- Framework: PyTorch
- Dataset: Simulated MPIIGaze-style data
- Loss: MSE
- Optimizer: Adam

---

## 🎯 Next Steps

1. Install Visual Studio 2022 or MinGW-w64
2. Install dependencies via vcpkg or MSYS2
3. Build the project
4. Run and test!

For production use, replace the dummy dataset in `train_gaze_model.py` with actual MPIIGaze or other eye-tracking datasets.

---

## 📄 License

This project is for educational purposes. Microsoft's accessibility mission inspired this work.
