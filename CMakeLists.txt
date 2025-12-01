cmake_minimum_required(VERSION 3.15)
project(GazeTracker VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Find required packages
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# ONNX Runtime - You need to download and set the path
# Download from: https://github.com/microsoft/onnxruntime/releases
set(ONNXRUNTIME_ROOT_PATH "" CACHE PATH "Path to ONNX Runtime installation")

if(NOT ONNXRUNTIME_ROOT_PATH)
    message(STATUS "ONNXRUNTIME_ROOT_PATH not set. Trying to find it automatically...")
    
    # Common installation paths
    set(POSSIBLE_PATHS
        "C:/Program Files/onnxruntime"
        "C:/onnxruntime"
        "${CMAKE_SOURCE_DIR}/onnxruntime"
        "${CMAKE_SOURCE_DIR}/libs/onnxruntime"
    )
    
    foreach(path ${POSSIBLE_PATHS})
        if(EXISTS ${path})
            set(ONNXRUNTIME_ROOT_PATH ${path})
            message(STATUS "Found ONNX Runtime at: ${path}")
            break()
        endif()
    endforeach()
endif()

if(ONNXRUNTIME_ROOT_PATH)
    include_directories(${ONNXRUNTIME_ROOT_PATH}/include)
    link_directories(${ONNXRUNTIME_ROOT_PATH}/lib)
else()
    message(WARNING "ONNX Runtime path not found. Please set ONNXRUNTIME_ROOT_PATH")
    message(WARNING "Download from: https://github.com/microsoft/onnxruntime/releases")
endif()

# Add executable
add_executable(GazeTracker main.cpp)

# Link libraries
target_link_libraries(GazeTracker 
    ${OpenCV_LIBS}
    onnxruntime
)

# Copy ONNX Runtime DLL to output directory (Windows)
if(WIN32 AND ONNXRUNTIME_ROOT_PATH)
    add_custom_command(TARGET GazeTracker POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_ROOT_PATH}/lib/onnxruntime.dll"
        $<TARGET_FILE_DIR:GazeTracker>
        COMMENT "Copying ONNX Runtime DLL to output directory"
    )
endif()

# Installation rules
install(TARGETS GazeTracker DESTINATION bin)

# Print configuration summary
message(STATUS "")
message(STATUS "========================================")
message(STATUS "Configuration Summary:")
message(STATUS "  OpenCV version: ${OpenCV_VERSION}")
message(STATUS "  OpenCV libraries: ${OpenCV_LIBS}")
message(STATUS "  ONNX Runtime path: ${ONNXRUNTIME_ROOT_PATH}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "========================================")
message(STATUS "")
