# required cmake version
cmake_minimum_required(VERSION 3.21)
# project name
project(deepmd_op_rocm)
set(CMAKE_LINK_WHAT_YOU_USE TRUE)

# set c++ version c++11
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_HIP_STANDARD 14)
add_definitions("-DCUB_IGNORE_DEPRECATED_CPP_DIALECT")
add_definitions("-DCUB_IGNORE_DEPRECATED_CPP_DIALECT")

message(STATUS "HIP major version is " ${HIP_VERSION_MAJOR})

set(HIP_HIPCC_FLAGS -fno-gpu-rdc; -fPIC --std=c++14 ${HIP_HIPCC_FLAGS}
)# --amdgpu-target=gfx906
if(HIP_VERSION VERSION_LESS 3.5.1)
  set(HIP_HIPCC_FLAGS -hc; ${HIP_HIPCC_FLAGS})
endif()

file(GLOB SOURCE_FILES "*.hip.cu")

hip_add_library(deepmd_op_rocm SHARED ${SOURCE_FILES})
target_include_directories(
  deepmd_op_rocm
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include/>
         $<INSTALL_INTERFACE:include>)
target_precompile_headers(deepmd_op_rocm PUBLIC [["device.h"]])

install(TARGETS deepmd_op_rocm DESTINATION lib/)
if(BUILD_CPP_IF)
  install(
    TARGETS deepmd_op_rocm
    EXPORT ${CMAKE_PROJECT_NAME}Targets
    DESTINATION lib/)
endif(BUILD_CPP_IF)
if(BUILD_PY_IF)
  install(TARGETS deepmd_op_rocm DESTINATION deepmd/op/)
endif(BUILD_PY_IF)
