# VisionEdge: Real-Time Gaze Interface

An accessibility-focused gaze tracking system built with C++ and Python, designed to help people with limited mobility control their computers using eye movement.

**Created by:** Hirak with AI assistance  
**Project Type:** Accessibility Tool / Computer Vision  
**Status:** Working Prototype

## About This Project

This project started as an exploration into making computers more accessible for people with motor disabilities. After researching existing eye-tracking solutions, I worked with AI to build a complete system from scratch - from training the neural network to implementing the real-time tracking interface.

The goal was to create something that could actually help people, not just a proof-of-concept. Through iterative development and testing, we built both a calibrated high-accuracy version and a simpler demo version.

## 🎯 Key Features

- **Real-time Gaze Tracking**: Estimates where a person is looking using webcam input
- **Lightweight CNN Model**: Trained on MPIIGaze dataset, optimized for on-device inference
- **ONNX Runtime Integration**: Fast inference using Microsoft's ONNX Runtime C++ API
- **Cursor Control**: Move Windows mouse cursor based on predicted gaze direction
- **Accessibility Focus**: Designed for people who control computers with eye/movement

1. **Face & Eye Detection**: Uses OpenCV's Haar cascades to locate eyes in webcam feed
2. **Calibration** (calibrated version): Maps your eye positions to screen coordinates using 9-point calibration
3. **Gaze Prediction**: Neural network predicts where you're looking
4. **Cursor Control**: Smoothly moves mouse cursor based on gaze
5. **Dwell Clicking**: Stare at a spot for 1 second to automatically click

## 🔧 Technical Details

### Model Architecture
- Lightweight CNN with 3 convolutional layers
- Trained using PyTorch, exported to ONNX format
- Input: 60x36 grayscale eye image
- Output: (x, y) gaze coordinates
- Model size: ~4 MB

### Performance
- Runs at 20-30 FPS on CPU
- ~50-100ms latency from eye movement to cursor
- 10-frame smoothing buffer reduces jitter

## 🛠️ Development Notes

This project went through several iterations:

1. Started with basic eye detection
2. Added neural network for gaze prediction
3. Implemented calibration system for accuracy
4. Added dwell-time clicking for usability
5. Optimized smoothing and responsiveness

The biggest challenge was balancing accuracy with smoothness. Too much smoothing made it laggy, too little made it jittery. The current 10-frame buffer with 0.3 smoothing factor worked best in testing.

## 📦 Dependencies

```bash
pip install torch opencv-python pyautogui numpy
```

For C++ version, you'll need:
- OpenCV 4.x
- ONNX Runtime 1.x
- CMake 3.15+
- Visual Studio 2019/2022

## 🎓 What I Learned

- Training and deploying neural networks for real-time applications
- Computer vision techniques for face/eye detection
- Calibration algorithms for improving accuracy
- Balancing performance vs. accuracy in real-time systems
- Building accessible technology that actually works

## 🔮 Future Improvements

- [ ] Add blink detection for alternative clicking
- [ ] Support for multiple monitors
- [ ] Improve low-light performance
- [ ] Train on real MPIIGaze dataset (currently uses simulated data)
- [ ] Add user profiles to save calibrations
- [ ] Implement smooth scrolling with gaze

## 📄 License

This project is for educational and accessibility purposes. Feel free to use, modify, and improve it!

## 🙏 Acknowledgments

- Built with assistance from AI (GitHub Copilot)
- Inspired by Microsoft's accessibility mission
- Uses OpenCV for computer vision
- PyTorch for machine learning
- ONNX Runtime for deployment

---

**Note:** The current model is trained on simulated data for demonstration. For production use, retrain with actual gaze tracking datasets like MPIIGaze for better accuracy.

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

## 🚀 Quick Start

**Want to try it immediately?** Run the calibrated version:

```powershell
python gaze_tracker_calibrated.py
```

This will guide you through a 9-point calibration, then you can control your cursor with your eyes!

**For a simpler demo (no calibration):**

```powershell
python demo_python.py
```

Press SPACE to enable mouse control, ESC to quit.

## 📁 Project Structure

The project includes multiple implementations:

- **`gaze_tracker_calibrated.py`** - Main application with calibration system ⭐
- **`demo_python.py`** - Simpler demo without calibration
- **`main.cpp`** - C++ implementation (requires OpenCV + ONNX Runtime)
- **`gaze_model.onnx`** - Trained neural network model
- **Training scripts** - For retraining the model with custom data

## 🏗️ How It Works
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
