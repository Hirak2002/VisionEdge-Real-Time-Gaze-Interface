"""
Calibrated Gaze Tracker with Smooth Performance
"""

import cv2
import numpy as np
import pyautogui
import time
from collections import deque

print("="*60)
print("Calibrated Gaze Tracker")
print("="*60)
print("\nCalibration Instructions:")
print("1. Look at the RED circle when it appears")
print("2. Press SPACE when looking at it")
print("3. Repeat for all 9 calibration points")
print("\nAfter calibration:")
print("  - Move your eyes to control cursor")
print("  - Stare at a spot for 1 second to click")
print("  - Press ESC to quit")
print("\n" + "="*60)

# Disable PyAutoGUI failsafe for smooth operation
pyautogui.FAILSAFE = False

# Get screen size
screen_width, screen_height = pyautogui.size()

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 60)

if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit(1)

# Load detectors
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Calibration points (9-point grid)
calibration_points = [
    (screen_width * 0.1, screen_height * 0.1),   # Top-left
    (screen_width * 0.5, screen_height * 0.1),   # Top-center
    (screen_width * 0.9, screen_height * 0.1),   # Top-right
    (screen_width * 0.1, screen_height * 0.5),   # Middle-left
    (screen_width * 0.5, screen_height * 0.5),   # Center
    (screen_width * 0.9, screen_height * 0.5),   # Middle-right
    (screen_width * 0.1, screen_height * 0.9),   # Bottom-left
    (screen_width * 0.5, screen_height * 0.9),   # Bottom-center
    (screen_width * 0.9, screen_height * 0.9),   # Bottom-right
]

calibration_data = []
current_calibration_point = 0
calibrating = True
calibration_samples = []

# Smoothing buffer for eye positions
eye_position_buffer = deque(maxlen=10)

# Click detection
dwell_threshold = 1.0  # 1 second to click
dwell_start_time = None
dwell_position = None
dwell_radius = 30
last_click_time = 0
click_cooldown = 0.5

def get_eye_center(frame, gray):
    """Extract eye center position from frame"""
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        return None
    
    # Use first face
    (x, y, w, h) = faces[0]
    roi_gray = gray[y:y+h, x:x+w]
    
    # Detect eyes
    eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
    
    if len(eyes) == 0:
        return None
    
    # Use first eye (or average if both detected)
    eye_centers = []
    for (ex, ey, ew, eh) in eyes[:2]:
        eye_center_x = x + ex + ew // 2
        eye_center_y = y + ey + eh // 2
        eye_centers.append((eye_center_x, eye_center_y))
        
        # Draw eye for debugging
        cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 2)
    
    # Return average of detected eyes
    if len(eye_centers) > 0:
        avg_x = sum(p[0] for p in eye_centers) // len(eye_centers)
        avg_y = sum(p[1] for p in eye_centers) // len(eye_centers)
        return (avg_x, avg_y)
    
    return None

def map_eye_to_screen(eye_pos, calibration_mapping):
    """Map eye position to screen coordinates using calibration data"""
    if calibration_mapping is None or len(calibration_mapping) < 4:
        # Fallback: direct mapping
        frame_width = 640
        frame_height = 480
        x_ratio = eye_pos[0] / frame_width
        y_ratio = eye_pos[1] / frame_height
        return (int(x_ratio * screen_width), int(y_ratio * screen_height))
    
    # Use inverse distance weighting with calibration points
    if len(calibration_mapping) < 4:
        return None
    
    weights = []
    total_weight = 0
    
    for (eye_x, eye_y), (screen_x, screen_y) in calibration_mapping:
        distance = np.sqrt((eye_pos[0] - eye_x)**2 + (eye_pos[1] - eye_y)**2)
        if distance < 1:
            return (int(screen_x), int(screen_y))
        weight = 1.0 / (distance ** 2)
        weights.append((weight, screen_x, screen_y))
        total_weight += weight
    
    # Weighted average
    screen_x = sum(w * sx for w, sx, sy in weights) / total_weight
    screen_y = sum(w * sy for w, sx, sy in weights) / total_weight
    
    return (int(screen_x), int(screen_y))

print("\nStarting calibration...")
print("Look at the RED circle and press SPACE")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)  # Mirror for natural interaction
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    if calibrating:
        # Calibration phase
        if current_calibration_point < len(calibration_points):
            target_x, target_y = calibration_points[current_calibration_point]
            
            # Draw calibration target
            cv2.circle(frame, (320, 240), 100, (0, 0, 255), -1)
            cv2.putText(frame, f"Look at RED circle on screen", 
                       (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Then press SPACE ({current_calibration_point + 1}/9)", 
                       (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show calibration point on separate window (full screen)
            calib_screen = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
            cv2.circle(calib_screen, (int(target_x), int(target_y)), 50, (0, 0, 255), -1)
            cv2.putText(calib_screen, "Look Here", (int(target_x) - 60, int(target_y) - 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow('Calibration', calib_screen)
            cv2.setWindowProperty('Calibration', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            
            # Get eye position
            eye_pos = get_eye_center(frame, gray)
            
            if eye_pos:
                calibration_samples.append(eye_pos)
                cv2.circle(frame, eye_pos, 5, (255, 0, 255), -1)
        else:
            # Calibration complete
            calibrating = False
            cv2.destroyWindow('Calibration')
            print("\n✓ Calibration complete!")
            print("You can now control the cursor with your eyes!")
    else:
        # Tracking phase
        eye_pos = get_eye_center(frame, gray)
        
        if eye_pos:
            # Add to buffer for smoothing
            eye_position_buffer.append(eye_pos)
            
            # Smooth position
            smoothed_x = sum(p[0] for p in eye_position_buffer) // len(eye_position_buffer)
            smoothed_y = sum(p[1] for p in eye_position_buffer) // len(eye_position_buffer)
            smoothed_pos = (smoothed_x, smoothed_y)
            
            # Map to screen
            screen_pos = map_eye_to_screen(smoothed_pos, calibration_data)
            
            if screen_pos:
                screen_x, screen_y = screen_pos
                
                # Clamp to screen
                screen_x = max(0, min(screen_width - 1, screen_x))
                screen_y = max(0, min(screen_height - 1, screen_y))
                
                # Move cursor
                pyautogui.moveTo(screen_x, screen_y, duration=0)
                
                # Dwell clicking
                current_time = time.time()
                
                if dwell_start_time is None:
                    dwell_start_time = current_time
                    dwell_position = (screen_x, screen_y)
                else:
                    distance = np.sqrt((screen_x - dwell_position[0])**2 + (screen_y - dwell_position[1])**2)
                    
                    if distance < dwell_radius:
                        dwell_duration = current_time - dwell_start_time
                        
                        # Draw progress
                        progress = min(dwell_duration / dwell_threshold, 1.0)
                        radius = int(40 * progress)
                        cv2.circle(frame, eye_pos, radius, (0, 255, 255), 3)
                        
                        if dwell_duration >= dwell_threshold:
                            if current_time - last_click_time >= click_cooldown:
                                pyautogui.click()
                                last_click_time = current_time
                                print(f"CLICK at ({screen_x}, {screen_y})")
                                cv2.circle(frame, eye_pos, 50, (0, 255, 0), 5)
                            
                            dwell_start_time = None
                            dwell_position = None
                    else:
                        dwell_start_time = current_time
                        dwell_position = (screen_x, screen_y)
                
                # Display info
                cv2.putText(frame, f"Cursor: ({screen_x}, {screen_y})", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.circle(frame, eye_pos, 3, (255, 0, 255), -1)
    
    cv2.imshow('Gaze Tracker', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == 32 and calibrating:  # SPACE
        if len(calibration_samples) > 0:
            # Average samples
            avg_x = sum(p[0] for p in calibration_samples) // len(calibration_samples)
            avg_y = sum(p[1] for p in calibration_samples) // len(calibration_samples)
            
            target_x, target_y = calibration_points[current_calibration_point]
            calibration_data.append(((avg_x, avg_y), (target_x, target_y)))
            
            print(f"✓ Point {current_calibration_point + 1}/9 calibrated")
            
            current_calibration_point += 1
            calibration_samples = []

cap.release()
cv2.destroyAllWindows()
print("\nGaze tracker stopped")
