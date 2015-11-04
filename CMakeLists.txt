cmake_minimum_required(VERSION 2.8)
project(gSLICr)
  
IF(MSVC_IDE)
  set(OpenCV_STATIC OFF)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  add_definitions(-DUSING_CMAKE=1)
ELSE(MSVC_IDE)
  set(CFLAGS_WARN "-Wall -Wextra -Wno-unused-parameter -Wno-strict-aliasing")
  set(CMAKE_CXX_FLAGS "-fPIC -O3 -march=native ${CFLAGS_WARN} ${CMAKE_CXX_FLAGS}")
ENDIF(MSVC_IDE)

if(APPLE)
  set(CUDA_HOST_COMPILER /usr/bin/clang)
endif(APPLE)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL 1.7 REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PCL_INCLUDE_DIRS})
add_subdirectory(ORUtils)

link_directories(${PCL_LIBRARY_DIRS})

set(GSLICR_LIB
gSLICr_Lib/engines/gSLICr_core_engine.h
gSLICr_Lib/engines/gSLICr_seg_engine.h
gSLICr_Lib/engines/gSLICr_seg_engine_GPU.h
gSLICr_Lib/engines/gSLICr_seg_engine_shared.h
gSLICr_Lib/engines/gSLICr_core_engine.cpp
gSLICr_Lib/engines/gSLICr_seg_engine.cpp
gSLICr_Lib/engines/gSLICr_seg_engine_GPU.cu
gSLICr_Lib/objects/gSLICr_settings.h
gSLICr_Lib/objects/gSLICr_spixel_info.h
gSLICr_Lib/gSLICr_defines.h
gSLICr_Lib/gSLICr.h
)

set(MASKING_LIB
    masking/masking.cpp
)

ADD_DEFINITIONS(${PCL_DEFINITIONS})

list(APPEND "-std=c++11 -ftree-vectorize")
SOURCE_GROUP(engines FILES ${GSLICR_LIB})

cuda_add_library(gSLICr_lib
			${GSLICR_LIB}
			NVTimer.h
			OPTIONS -gencode arch=compute_30,code=compute_30)
target_link_libraries(gSLICr_lib ${CUDA_LIBRARY})

add_library(masking ${MASKING_LIB})
target_link_libraries(masking ${OpenCV_LIBS} ${PCL_LIBS})

add_executable(demo demo.cpp)
target_link_libraries(demo gSLICr_lib ${OpenCV_LIBS})

# Build python wrapper
include(cmake/boost-python.cmake)

boost_python_module(gSLIC gSLIC.cpp)
target_link_libraries(gSLIC boost_system boost_python gSLICr_lib ${OpenCV_LIBS} masking ${PCL_LIBRARIES})
