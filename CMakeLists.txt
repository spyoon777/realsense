# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.18)

project ("realsense")

set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_STANDARD_REQUIRED ON)

set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -W1 -O2 /bigobj")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W1 -O2 /bigobj")

set (CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/build/install")

set (CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_SOURCE_DIR}/cmake_modules")

# EXT Lib
#set (opencv_LIBRARY_ROOT_DIR "${CMAKE_SOURCE_DIR}/ThirdParty/opencv-3.4.12")
#message ("opencv downloaded dir: ${opencv_LIBRARY_ROOT_DIR}")

# set (glfw_LIBRARY_ROOT_DIR "${CMAKE_SOURCE_DIR}/build/ThirdParty/glfw-3.3.2.bin.WIN64")
# message ("glfw downloaded dir: ${glfw_LIBRARY_ROOT_DIR}")

# set (glew_LIBRARY_ROOT_DIR "${CMAKE_SOURCE_DIR}/build/ThirdParty/glew-2.1.0")
# message ("glew downloaded dir: ${glew_LIBRARY_ROOT_DIR}")

#set (glm_LIBRARY_ROOT_DIR "${CMAKE_SOURCE_DIR}/ThirdParty/glm-0.9.9.8")
#message ("glm downloaded dir: ${glm_LIBRARY_ROOT_DIR}")
#

#set (OpenCV_DIR ${opencv_LIBRARY_ROOT_DIR})
#find_package(OpenCV REQUIRED)
#if (NOT OpenCV_FOUND)
#   message (FATAL_ERROR "opencv not found")
#endif ()

# Include sub-projects.
add_subdirectory ("eye")
