cmake_minimum_required(VERSION 3.15)
project(TinyTensor-MLIR CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(TINYTENSOR_BUILD_TT_OPT "Build the MLIR-based tt-opt skeleton" ON)

include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(tinytensor_support INTERFACE)
target_include_directories(tinytensor_support INTERFACE
  ${CMAKE_CURRENT_BINARY_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set(TINYTENSOR_HAS_MLIR OFF)

if(TINYTENSOR_BUILD_TT_OPT)
  # Default LLVM/MLIR build paths to the user's directory if not specified.
  if(NOT LLVM_DIR)
    set(LLVM_DIR "/home/hadley/Developments/llvm-project/build/lib/cmake/llvm" CACHE PATH "Path to LLVM CMake configuration")
  endif()
  if(NOT MLIR_DIR)
    set(MLIR_DIR "/home/hadley/Developments/llvm-project/build/lib/cmake/mlir" CACHE PATH "Path to MLIR CMake configuration")
  endif()

  find_package(LLVM CONFIG)
  find_package(MLIR CONFIG)

  if(LLVM_FOUND AND MLIR_FOUND)
    set(TINYTENSOR_HAS_MLIR ON)
    message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
    message(STATUS "Using LLVM_DIR: ${LLVM_DIR}")
    message(STATUS "Using MLIR_DIR: ${MLIR_DIR}")

    include_directories(${LLVM_INCLUDE_DIRS})
    include_directories(${MLIR_INCLUDE_DIRS})
    add_definitions(${LLVM_DEFINITIONS})
  else()
    message(STATUS "LLVM/MLIR not found; skipping tt-opt. Set TINYTENSOR_BUILD_TT_OPT=OFF to silence this path.")
  endif()
endif()

add_subdirectory(lib)

if(TINYTENSOR_HAS_MLIR)
  add_subdirectory(tools/tt-opt)
  add_subdirectory(tools/tt-emit)
endif()

# Add GoogleTest
find_package(GTest REQUIRED)

# Testing
enable_testing()
add_subdirectory(tests)
