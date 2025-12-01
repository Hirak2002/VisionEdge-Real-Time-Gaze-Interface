"""
Gaze Tracker - Python Demo Version
This demonstrates the full gaze tracking pipeline without requiring C++ compilation.
"""

import cv2
import numpy as np
import torch
import sys
import pyautogui
import time

print("="*60)
print("Gaze Tracker - Python Demo")
print("="*60)
print("\nControls:")
print("  - Press ESC to quit")
print("  - Press SPACE to toggle mouse control")
print("  - Look at a spot for 1.5 seconds to click")
print("\nInitializing...\n")

# Load the trained model
try:
    model = torch.jit.load("gaze_model.pt")
    model.eval()
    print("✓ Model loaded: gaze_model.pt")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    sys.exit(1)

# Load face and eye detectors
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    print("✓ Face/Eye detectors loaded")
except Exception as e:
    print(f"✗ Error loading cascades: {e}")
    sys.exit(1)

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("✗ Error: Cannot open webcam")
    sys.exit(1)

print("✓ Webcam initialized")
print("\n" + "="*60)
print("Starting gaze tracking...")
print("="*60 + "\n")

# Get screen size
screen_width, screen_height = pyautogui.size()

# State
mouse_control_enabled = False
smoothing_factor = 0.3
prev_x, prev_y = screen_width // 2, screen_height // 2

# Clicking state
dwell_time_threshold = 1.5  # seconds to dwell before clicking
dwell_start_time = None
dwell_position = None
dwell_radius = 50  # pixels - how close gaze must stay to trigger click
click_cooldown = 1.0  # seconds between clicks
last_click_time = 0

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        # Draw face rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Extract face ROI
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        # Detect eyes within face
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        
        for (ex, ey, ew, eh) in eyes[:1]:  # Use first detected eye
            # Draw eye rectangle
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            
            # Extract eye region
            eye_img = roi_gray[ey:ey+eh, ex:ex+ew]
            
            # Calculate eye position in frame
            absolute_eye_x = x + ex + ew//2
            absolute_eye_y = y + ey + eh//2
            
            # Map eye position to screen coordinates
            # Assume face is centered and map linearly with scaling
            frame_center_x = frame.shape[1] // 2
            frame_center_y = frame.shape[0] // 2
            
            # Calculate offset from center
            offset_x = absolute_eye_x - frame_center_x
            offset_y = absolute_eye_y - frame_center_y
            
            # Scale factor to amplify movement (adjust sensitivity)
            scale_x = 4.0  # Increase for more sensitive horizontal movement
            scale_y = 4.0  # Increase for more sensitive vertical movement
            
            # Map to screen coordinates
            screen_x = int(screen_width // 2 + offset_x * scale_x)
            screen_y = int(screen_height // 2 + offset_y * scale_y)
            
            # Also run model prediction for display (optional)
            try:
                eye_resized = cv2.resize(eye_img, (60, 36))
                eye_normalized = eye_resized.astype(np.float32) / 255.0
                eye_tensor = torch.from_numpy(eye_normalized).unsqueeze(0).unsqueeze(0)
                
                with torch.no_grad():
                    gaze_pred = model(eye_tensor)
                    gaze_x, gaze_y = gaze_pred[0].numpy()
            except Exception as e:
                gaze_x, gaze_y = 0, 0
                
            # Clamp to screen bounds
            screen_x = max(0, min(screen_width - 1, screen_x))
            screen_y = max(0, min(screen_height - 1, screen_y))
            
            # Smooth movement
            screen_x = int(prev_x * (1 - smoothing_factor) + screen_x * smoothing_factor)
            screen_y = int(prev_y * (1 - smoothing_factor) + screen_y * smoothing_factor)
            
            prev_x, prev_y = screen_x, screen_y
            
            # Move mouse if enabled
            if mouse_control_enabled:
                pyautogui.moveTo(screen_x, screen_y, duration=0.01)
                
                # Dwell-time clicking logic
                current_time = time.time()
                
                # Check if gaze is stable in one position
                if dwell_start_time is None:
                    # Start new dwell
                    dwell_start_time = current_time
                    dwell_position = (screen_x, screen_y)
                else:
                    # Check if still looking at same spot
                    distance = np.sqrt((screen_x - dwell_position[0])**2 + (screen_y - dwell_position[1])**2)
                    
                    if distance < dwell_radius:
                        # Still dwelling - check if ready to click
                        dwell_duration = current_time - dwell_start_time
                        
                        # Draw dwell progress indicator
                        progress = min(dwell_duration / dwell_time_threshold, 1.0)
                        circle_radius = int(30 * progress)
                        cv2.circle(frame, (ex + ew//2, ey + eh//2), circle_radius, (0, 255, 255), 2)
                        
                        if dwell_duration >= dwell_time_threshold:
                            # Perform click if cooldown elapsed
                            if current_time - last_click_time >= click_cooldown:
                                pyautogui.click()
                                last_click_time = current_time
                                print(f"CLICK at ({screen_x}, {screen_y})")
                                
                                # Visual feedback
                                cv2.circle(frame, (ex + ew//2, ey + eh//2), 40, (0, 255, 0), 3)
                            
                            # Reset dwell
                            dwell_start_time = None
                            dwell_position = None
                    else:
                        # Moved away - reset dwell
                        dwell_start_time = current_time
                        dwell_position = (screen_x, screen_y)
            
            # Display gaze info
            info_text = f"Eye Pos: ({absolute_eye_x}, {absolute_eye_y})"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 255), 2)
            
            screen_text = f"Cursor: ({screen_x}, {screen_y})"
            cv2.putText(frame, screen_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 255), 2)
            
            # Draw crosshair at eye center
            cv2.drawMarker(roi_color, (ex + ew//2, ey + eh//2), (255, 0, 255), 
                          cv2.MARKER_CROSS, 20, 2)
            
            try:
                cv2.putText(frame, f"Error: {str(e)[:30]}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            except:
                pass    # Display status
    status = "MOUSE ON" if mouse_control_enabled else "MOUSE OFF"
    color = (0, 255, 0) if mouse_control_enabled else (0, 0, 255)
    cv2.putText(frame, status, (10, frame.shape[0] - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    cv2.putText(frame, "Press SPACE to toggle mouse | ESC to quit", 
               (10, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 
               0.5, (255, 255, 255), 1)
    
    if mouse_control_enabled:
        cv2.putText(frame, "Look at a spot for 1.5s to click", 
                   (10, frame.shape[0] - 80), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (0, 255, 255), 1)
    
    # Show frame
    cv2.imshow('Gaze Tracker', frame)
    
    # Handle keyboard
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == 32:  # SPACE
        mouse_control_enabled = not mouse_control_enabled
        print(f"Mouse control: {'ENABLED' if mouse_control_enabled else 'DISABLED'}")

cap.release()
cv2.destroyAllWindows()

print("\n" + "="*60)
print("Gaze Tracker stopped")
print("="*60)
