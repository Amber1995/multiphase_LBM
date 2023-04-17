project(multiphaselbm LANGUAGES CUDA CXX)

enable_language(CUDA)
include(CheckLanguage)
check_language(CUDA)

set(CMAKE_CXX_STANDARD 14)

cmake_minimum_required(VERSION 3.12)

SET(CMAKE_COLOR_MAKEFILE ON)
SET(CMAKE_VERBOSE_MAKEFILE OFF)

# General compile settings
IF (NOT CMAKE_BUILD_TYPE)
 #SET(CMAKE_BUILD_TYPE "Debug")
 SET(CMAKE_BUILD_TYPE "Release")
ENDIF (NOT CMAKE_BUILD_TYPE)

# OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
  if (NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${OpenMP_CXX_FLAGS}") 
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}") 
  endif()
endif() 

# CMake Modules
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# CUDA PACKAGE
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS ON)
set(CUDA_HOST_COMPILER g++)


# Include
include_directories(BEFORE
  ${multiphaselbm_SOURCE_DIR}/include/
  ${multiphaselbm_SOURCE_DIR}/external/
 )

# COMPILE AND LINK
add_executable(multiphaselbm
 
  ${multiphaselbm_SOURCE_DIR}/src/iteration_host.cu
  ${multiphaselbm_SOURCE_DIR}/src/multiphase_lbm.cu
  ${multiphaselbm_SOURCE_DIR}/src/main.cc
  )

# Coverage
#find_package(codecov)
#if(ENABLE_COVERAGE)
#  add_executable(multiphaselbmtest_coverage ${multiphaselbm_src} ${test_src})
#  add_coverage(multiphaselbmtest_coverage)
#endif()
