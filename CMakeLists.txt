cmake_minimum_required(VERSION 2.8)
project(demo)

# set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors")

set(Z_LIBRARY "/home/hhq/anaconda3bak/lib/libz.so")
set(ZLIB_INCLUDE_DIR "/home/hhq/anaconda3/include")

#add lib
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/wrapper)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/wrapper/include)


include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

#add opencv
# set(OpenCV_DIR /data2/jinjiali/software/opencv3.4.1-cuda9.0/share/OpenCV)
FIND_PACKAGE(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARIES_DIRS})
message(OPENCV_LIBS: ${OpenCV_INCLUDE_DIRS})
message(OPENCV_LIBS: ${OpenCV_LIBRARIES_DIRS})

#build runYolov3
add_executable(classify main.cpp)
target_include_directories(classify PUBLIC ${ZLIB_INCLUDE_DIR})
target_link_libraries(classify TrtNet ${OpenCV_LIBS} ${Z_LIBRARY})

install(TARGETS classify DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/install/)
