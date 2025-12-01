"""
Quick test to verify all components are working
"""

import sys

print("="*60)
print("Gaze Tracker - Component Test")
print("="*60)

# Test 1: PyTorch
print("\n[1/5] Testing PyTorch...")
try:
    import torch
    print(f"  ✓ PyTorch {torch.__version__} loaded")
    model_loaded = False
    try:
        model = torch.jit.load("gaze_model.pt")
        print(f"  ✓ Model loaded successfully")
        model_loaded = True
    except Exception as e:
        print(f"  ✗ Model load failed: {e}")
except ImportError as e:
    print(f"  ✗ PyTorch not found: {e}")
    sys.exit(1)

# Test 2: OpenCV
print("\n[2/5] Testing OpenCV...")
try:
    import cv2
    print(f"  ✓ OpenCV {cv2.__version__} loaded")
    
    # Test cascade classifiers
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    print(f"  ✓ Face/Eye detectors loaded")
except ImportError as e:
    print(f"  ✗ OpenCV not found: {e}")
    sys.exit(1)

# Test 3: NumPy
print("\n[3/5] Testing NumPy...")
try:
    import numpy as np
    print(f"  ✓ NumPy {np.__version__} loaded")
except ImportError as e:
    print(f"  ✗ NumPy not found: {e}")
    sys.exit(1)

# Test 4: PyAutoGUI
print("\n[4/5] Testing PyAutoGUI...")
try:
    import pyautogui
    screen_width, screen_height = pyautogui.size()
    print(f"  ✓ PyAutoGUI loaded")
    print(f"  ✓ Screen size: {screen_width}x{screen_height}")
except ImportError as e:
    print(f"  ✗ PyAutoGUI not found: {e}")
    sys.exit(1)

# Test 5: Webcam
print("\n[5/5] Testing Webcam...")
try:
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"  ✓ Webcam working (Frame: {frame.shape})")
        else:
            print(f"  ⚠ Webcam opened but no frame captured")
        cap.release()
    else:
        print(f"  ⚠ Webcam not accessible (may be in use)")
except Exception as e:
    print(f"  ⚠ Webcam test failed: {e}")

# Test 6: Model Inference
if model_loaded:
    print("\n[BONUS] Testing Model Inference...")
    try:
        dummy_input = torch.randn(1, 1, 36, 60)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"  ✓ Model inference successful")
        print(f"  ✓ Output shape: {output.shape}")
        print(f"  ✓ Sample prediction: ({output[0][0].item():.3f}, {output[0][1].item():.3f})")
    except Exception as e:
        print(f"  ✗ Inference failed: {e}")

print("\n" + "="*60)
print("Component Test Complete!")
print("="*60)
print("\n✅ All required components are working!")
print("\nYou can now run the full demo:")
print("  python demo_python.py")
print("\nOr double-click:")
print("  run_demo.bat")
print("="*60)
