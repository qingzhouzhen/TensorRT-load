cmake_minimum_required(VERSION 2.8)
project(trtNet)

set(CMAKE_BUILD_TYPE Release)

#include
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

#src
set(PLUGIN_SOURCES
  src/TrtNet.cpp
)

#
# CUDA Configuration
#
find_package(CUDA REQUIRED)

if(NOT CUDA_FOUND)
  return()
endif()

message(STATUS "CUDA detected: " ${CUDA_VERSION})

set(CUDA_VERBOSE_BUILD ON)


# Specify the cuda host compiler to use the same compiler as cmake.
set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})


set(TensorRT_ROOT_DIR "/data1/hhq/softinstall/TensorRT-5.1.5.0/")
set(TENSORRT_INCLUDE_DIR "${TensorRT_ROOT_DIR}/include")
set(TENSORRT_LIBRARY "${TensorRT_ROOT_DIR}/lib/libnvinfer.so" "${TensorRT_ROOT_DIR}/lib/libnvinfer_plugin.so" "${TensorRT_ROOT_DIR}/lib/libnvparsers.so")

MESSAGE(STATUS "Find TensorRT libs at ${TENSORRT_LIBRARY}")
find_package_handle_standard_args(
  TENSORRT DEFAULT_MSG TENSORRT_INCLUDE_DIR TENSORRT_LIBRARY)
if(NOT TENSORRT_FOUND)
  message(ERROR
    "Cannot find TensorRT library.")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")	# -std=gnu++11
set(BUILD_DEPS "YES" CACHE BOOL "If YES, will install dependencies into sandbox.  Automatically reset to NO after dependencies are installed.")

# if(NOT "${CUDA_NVCC_FLAGS}" MATCHES "-std=c\\+\\+11" )
#   list(APPEND CUDA_NVCC_FLAGS -std=c++11)
# endif()
list(APPEND CUDA_NVCC_FLAGS "-D_FORCE_INLINES -Xcompiler -fPIC")
CUDA_INCLUDE_DIRECTORIES(${CUDNN_INCLUDE_DIR} ${TENSORRT_INCLUDE_DIR})
CUDA_ADD_LIBRARY(TrtNet STATIC ${PLUGIN_SOURCES})

message(${CUDA_INCLUDE_DIRS})
target_include_directories(TrtNet PUBLIC ${CUDA_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR} ${CUDNN_INCLUDE_DIR})
target_link_libraries(TrtNet ${TENSORRT_LIBRARY})