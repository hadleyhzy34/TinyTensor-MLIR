cmake_minimum_required(VERSION 3.15)
project(TinyTensor-MLIR CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default LLVM/MLIR build paths to the user's directory if not specified
if(NOT LLVM_DIR)
  set(LLVM_DIR "/home/hadley/Developments/llvm-project/build/lib/cmake/llvm" CACHE PATH "Path to LLVM CMake configuration")
endif()
if(NOT MLIR_DIR)
  set(MLIR_DIR "/home/hadley/Developments/llvm-project/build/lib/cmake/mlir" CACHE PATH "Path to MLIR CMake configuration")
endif()

# Find LLVM and MLIR
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVM_DIR: ${LLVM_DIR}")
message(STATUS "Using MLIR_DIR: ${MLIR_DIR}")

# Add LLVM and MLIR include paths
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Configure compiler flags
add_definitions(${LLVM_DEFINITIONS})

# Add GoogleTest
find_package(GTest REQUIRED)

# Declare subdirectories
add_subdirectory(tools/tt-opt)

# Testing
enable_testing()
add_subdirectory(tests)
