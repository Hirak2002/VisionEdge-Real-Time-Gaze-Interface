# On-Device Gaze Tracking for Accessibility

A C++ inference engine for real-time gaze tracking using OpenCV and ONNX Runtime. This project demonstrates how to build efficient accessibility software for Windows/Surface devices.

## 🎯 Features

- **Real-time Gaze Tracking**: Estimates where a person is looking using webcam input
- **Lightweight CNN Model**: Trained on MPIIGaze dataset, optimized for on-device inference
- **ONNX Runtime Integration**: Fast inference using Microsoft's ONNX Runtime C++ API
- **Cursor Control**: Move Windows mouse cursor based on predicted gaze direction
- **Accessibility Focus**: Designed for people who control computers with eye/movement

## 🏗️ Architecture

1. **Model Training (Python)**: PyTorch CNN trained on MPIIGaze dataset → exported to ONNX
2. **Inference Engine (C++)**: OpenCV captures webcam → ONNX Runtime predicts gaze → moves cursor
3. **Application Layer**: Real-time visualization and cursor control

## 📋 Prerequisites

### Required Software

- **Visual Studio 2019/2022** (with C++ desktop development workload)
- **CMake** (3.15 or higher)
- **Python 3.8+** (for model training)
- **Git** (for cloning repositories)

### Required Libraries

1. **OpenCV** (4.x recommended)
2. **ONNX Runtime** (1.14+ recommended)
3. **PyTorch** (for training only)

## 🚀 Installation Guide

### Step 1: Install Python Dependencies

```powershell
# Install PyTorch (CPU version)
pip install torch torchvision torchaudio

# Install additional dependencies
pip install numpy opencv-python
```

### Step 2: Install OpenCV

**Option A: Using vcpkg (Recommended)**

```powershell
# Install vcpkg
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg integrate install

# Install OpenCV
.\vcpkg install opencv:x64-windows
```

**Option B: Manual Download**

1. Download OpenCV from: https://opencv.org/releases/
2. Extract to `C:\opencv`
3. Add to system PATH: `C:\opencv\build\x64\vc16\bin`

### Step 3: Install ONNX Runtime

```powershell
# Download ONNX Runtime
$version = "1.16.3"
$url = "https://github.com/microsoft/onnxruntime/releases/download/v$version/onnxruntime-win-x64-$version.zip"
Invoke-WebRequest -Uri $url -OutFile "onnxruntime.zip"

# Extract
Expand-Archive onnxruntime.zip -DestinationPath C:\
Rename-Item "C:\onnxruntime-win-x64-$version" "C:\onnxruntime"
```

### Step 4: Download Haar Cascade (Optional)

```powershell
# Download eye detection cascade
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml" -OutFile "haarcascade_eye.xml"
```

## 🔨 Building the Project

### Step 1: Train the Model

```powershell
# Navigate to project directory
cd "C:\Users\hirak\OneDrive\Desktop\C++ Gaze Tracker"

# Train and export model to ONNX
python train_gaze_model.py
```

This creates `gaze_model.onnx` (ready for C++ inference).

### Step 2: Configure CMake

```powershell
# Create build directory
mkdir build
cd build

# Configure (update paths if needed)
cmake .. -DONNXRUNTIME_ROOT_PATH="C:/onnxruntime" -DOpenCV_DIR="C:/opencv/build"
```

**If using vcpkg:**

```powershell
cmake .. -DCMAKE_TOOLCHAIN_FILE="C:/vcpkg/scripts/buildsystems/vcpkg.cmake" -DONNXRUNTIME_ROOT_PATH="C:/onnxruntime"
```

### Step 3: Build

```powershell
# Build the project
cmake --build . --config Release

# Or open in Visual Studio
start GazeTracker.sln
```

## ▶️ Running the Application

```powershell
# From build directory
.\bin\Release\GazeTracker.exe

# Or specify custom model path
.\bin\Release\GazeTracker.exe "path\to\gaze_model.onnx"
```

### Controls

- **`C`** - Toggle cursor control ON/OFF
- **`Q`** - Quit application

## 📁 Project Structure

```
C++ Gaze Tracker/
├── main.cpp                 # C++ inference engine
├── train_gaze_model.py      # Python training script
├── CMakeLists.txt           # Build configuration
├── README.md                # This file
├── gaze_model.onnx          # Trained model (generated)
├── haarcascade_eye.xml      # Eye detection cascade (optional)
└── build/                   # Build output directory
    └── bin/
        └── Release/
            └── GazeTracker.exe
```

## 🔧 Troubleshooting

### OpenCV Not Found

```powershell
# Set OpenCV_DIR environment variable
$env:OpenCV_DIR = "C:\opencv\build"
```

### ONNX Runtime DLL Missing

- Copy `onnxruntime.dll` from `C:\onnxruntime\lib` to the same directory as `GazeTracker.exe`
- Or add `C:\onnxruntime\lib` to system PATH

### Camera Not Opening

- Check camera permissions in Windows Settings → Privacy → Camera
- Try different camera ID: `GazeTracker.exe 1` (uses camera index 1)

### Build Errors

```powershell
# Clean build
cd build
Remove-Item * -Recurse -Force
cmake .. -DONNXRUNTIME_ROOT_PATH="C:/onnxruntime"
cmake --build . --config Release
```

## 🎓 Technical Details

### Model Architecture

- **Input**: 60×36 grayscale eye image
- **Architecture**: 3 Conv layers + 3 FC layers
- **Output**: 2D gaze coordinates (x, y)
- **Parameters**: ~1.2M
- **Inference Time**: <5ms on CPU

### Performance Optimization

- ONNX Runtime graph optimization enabled
- Single-threaded inference (configurable)
- Frame preprocessing cached
- Minimal memory allocation per frame

## 🌟 Future Enhancements

- [ ] Support for multiple eye tracking models
- [ ] Calibration system for improved accuracy
- [ ] Click detection (blink/dwell time)
- [ ] Virtual keyboard integration
- [ ] Multi-monitor support
- [ ] Export to UWP for Surface devices

## 📄 License

This project is for educational purposes. Please ensure compliance with MPIIGaze dataset license for commercial use.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📞 Support

For issues or questions, please open an issue on the project repository.

---

**Built with ❤️ for accessibility**
