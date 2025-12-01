/*
 * VisionEdge: Real-Time Gaze Interface
 * C++ Implementation with ONNX Runtime
 * 
 * Author: Hirak with AI assistance
 * Date: December 2025
 * 
 * This is the C++ version for production deployment.
 * For quick testing, use the Python version instead.
 */

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <windows.h>
#include <iostream>
#include <vector>
#include <array>

// Main class for gaze tracking functionality
class GazeTracker {
private:
    // ONNX Runtime components
    Ort::Env env;
    Ort::Session session;
    Ort::SessionOptions session_options;
    Ort::AllocatorWithDefaultOptions allocator;
    
    // Model I/O
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    std::vector<int64_t> input_shape;
    
    // OpenCV components
    cv::VideoCapture cap;
    cv::CascadeClassifier eye_cascade;
    
    // Screen dimensions
    int screen_width;
    int screen_height;
    
public:
    GazeTracker(const std::string& model_path) 
        : env(ORT_LOGGING_LEVEL_WARNING, "GazeTracker"),
          session(nullptr) {
        
        // Initialize ONNX Runtime session
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        
        std::wstring model_path_w(model_path.begin(), model_path.end());
        session = Ort::Session(env, model_path_w.c_str(), session_options);
        
        // Get input/output names and shapes
        input_names.push_back(session.GetInputName(0, allocator));
        output_names.push_back(session.GetOutputName(0, allocator));
        
        auto input_type_info = session.GetInputTypeInfo(0);
        auto tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        input_shape = tensor_info.GetShape();
        
        std::cout << "ONNX Model loaded successfully!" << std::endl;
        std::cout << "Input shape: [" << input_shape[0] << ", " 
                  << input_shape[1] << ", " << input_shape[2] << ", " 
                  << input_shape[3] << "]" << std::endl;
        
        // Get screen dimensions
        screen_width = GetSystemMetrics(SM_CXSCREEN);
        screen_height = GetSystemMetrics(SM_CYSCREEN);
        std::cout << "Screen resolution: " << screen_width << "x" << screen_height << std::endl;
        
        // Load Haar Cascade for eye detection (optional preprocessing)
        std::string cascade_path = "haarcascade_eye.xml";
        if (!eye_cascade.load(cascade_path)) {
            std::cerr << "Warning: Could not load eye cascade. Using full frame." << std::endl;
        }
    }
    
    bool initCamera(int camera_id = 0) {
        cap.open(camera_id);
        if (!cap.isOpened()) {
            std::cerr << "Error: Could not open camera!" << std::endl;
            return false;
        }
        
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        
        std::cout << "Camera initialized successfully!" << std::endl;
        return true;
    }
    
    cv::Mat preprocessFrame(const cv::Mat& frame) {
        cv::Mat gray, resized;
        
        // Convert to grayscale
        if (frame.channels() == 3) {
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = frame.clone();
        }
        
        // Resize to model input size (60x36)
        cv::resize(gray, resized, cv::Size(60, 36));
        
        // Normalize to [-1, 1]
        resized.convertTo(resized, CV_32F, 1.0 / 255.0);
        resized = (resized - 0.5) / 0.5;
        
        return resized;
    }
    
    std::array<float, 2> runInference(const cv::Mat& input_image) {
        // Prepare input tensor
        std::vector<float> input_tensor_values(1 * 1 * 36 * 60);
        
        // Copy data (HWC to CHW format, but we have single channel)
        for (int h = 0; h < 36; h++) {
            for (int w = 0; w < 60; w++) {
                input_tensor_values[h * 60 + w] = input_image.at<float>(h, w);
            }
        }
        
        // Create input tensor
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<int64_t> tensor_shape = {1, 1, 36, 60};
        
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            input_tensor_values.data(),
            input_tensor_values.size(),
            tensor_shape.data(),
            tensor_shape.size()   
        );
        
        // Run inference
        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            input_names.data(),
            &input_tensor,
            1,
            output_names.data(),
            1
        );
        
        // Get output
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        
        return {output_data[0], output_data[1]};
    }
    
    void moveCursor(float gaze_x, float gaze_y) {
        // Convert normalized coordinates [-1, 1] to screen coordinates
        int screen_x = static_cast<int>((gaze_x + 1.0f) * 0.5f * screen_width);
        int screen_y = static_cast<int>((gaze_y + 1.0f) * 0.5f * screen_height);
        
        // Clamp to screen bounds
        screen_x = std::max(0, std::min(screen_x, screen_width - 1));
        screen_y = std::max(0, std::min(screen_y, screen_height - 1));
        
        // Move cursor
        SetCursorPos(screen_x, screen_y);
    }
    
    void run() {
        std::cout << "\nStarting gaze tracking..." << std::endl;
        std::cout << "Press 'q' to quit, 'c' to toggle cursor control" << std::endl;
        
        cv::Mat frame;
        bool cursor_control_enabled = false;
        
        while (true) {
            // Capture frame
            cap >> frame;
            if (frame.empty()) {
                std::cerr << "Error: Empty frame!" << std::endl;
                break;
            }
            
            // Preprocess
            cv::Mat processed = preprocessFrame(frame);
            
            // Run inference
            auto gaze_coords = runInference(processed);
            
            // Move cursor if enabled
            if (cursor_control_enabled) {
                moveCursor(gaze_coords[0], gaze_coords[1]);
            }
            
            // Display results
            cv::Mat display_frame = frame.clone();
            
            // Draw gaze indicator
            int indicator_x = static_cast<int>((gaze_coords[0] + 1.0f) * 0.5f * display_frame.cols);
            int indicator_y = static_cast<int>((gaze_coords[1] + 1.0f) * 0.5f * display_frame.rows);
            cv::circle(display_frame, cv::Point(indicator_x, indicator_y), 10, 
                      cv::Scalar(0, 255, 0), 2);
            
            // Display coordinates
            std::string coords_text = "Gaze: (" + 
                std::to_string(gaze_coords[0]).substr(0, 5) + ", " + 
                std::to_string(gaze_coords[1]).substr(0, 5) + ")";
            cv::putText(display_frame, coords_text, cv::Point(10, 30),
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
            
            std::string status = cursor_control_enabled ? "Cursor: ON" : "Cursor: OFF";
            cv::putText(display_frame, status, cv::Point(10, 60),
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, 
                       cursor_control_enabled ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), 2);
            
            cv::imshow("Gaze Tracking", display_frame);
            
            // Handle key presses
            char key = cv::waitKey(1);
            if (key == 'q' || key == 'Q') {
                break;
            } else if (key == 'c' || key == 'C') {
                cursor_control_enabled = !cursor_control_enabled;
                std::cout << "Cursor control: " << (cursor_control_enabled ? "ENABLED" : "DISABLED") << std::endl;
            }
        }
        
        cap.release();
        cv::destroyAllWindows();
        std::cout << "Gaze tracking stopped." << std::endl;
    }
};

int main(int argc, char* argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "Gaze Tracking for Accessibility" << std::endl;
    std::cout << "C++ Inference Engine" << std::endl;
    std::cout << "========================================" << std::endl;
    
    try {
        // Model path
        std::string model_path = "gaze_model.onnx";
        if (argc > 1) {
            model_path = argv[1];
        }
        
        // Initialize tracker
        GazeTracker tracker(model_path);
        
        // Initialize camera
        if (!tracker.initCamera()) {
            return -1;
        }
        
        // Run tracking
        tracker.run();
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
