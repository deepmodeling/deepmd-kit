# required cmake version
cmake_minimum_required(VERSION 3.5)
# project name
project(deepmd_op_rocm)
set(CMAKE_LINK_WHAT_YOU_USE TRUE)
find_package(HIP REQUIRED)
add_compile_definitions(__HIP_PLATFORM_HCC__)
add_definitions("-DTENSORFLOW_USE_ROCM")

# set c++ version c++11
SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_HIP_STANDARD 11)

message(STATUS "HIP major version is " ${HIP_VERSION_MAJOR})


set (HIP_HIPCC_FLAGS -hc; -fno-gpu-rdc; --amdgpu-target=gfx906; -fPIC; -O3; --std=c++11; -D__HIP_PLATFORM_HCC__)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -DCUB_IGNORE_DEPRECATED_CPP_DIALECT -DCUB_IGNORE_DEPRECATED_CPP_DIALECT")

set (SOURCE_FILES
    prod_env_mat.hip.cu prod_force.hip.cu prod_virial.hip.cu gelu.hip.cu tabulate.hip.cu coord.hip.cu neighbor_list.hip.cu prod_force_grad.hip.cu prod_virial_grad.hip.cu region.hip.cu
)

hip_add_library(deepmd_op_rocm SHARED ${SOURCE_FILES})

install(TARGETS deepmd_op_rocm DESTINATION lib/)
if (BUILD_CPP_IF)
  install(TARGETS deepmd_op_rocm DESTINATION lib/)
endif (BUILD_CPP_IF)
if (BUILD_PY_IF)
  install(TARGETS deepmd_op_rocm DESTINATION deepmd/op/)
endif (BUILD_PY_IF)
