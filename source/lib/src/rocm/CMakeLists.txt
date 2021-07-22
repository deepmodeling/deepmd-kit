# required cmake version
cmake_minimum_required(VERSION 3.5)
# project name
project(deepmd_op_rocm)
set(CMAKE_LINK_WHAT_YOU_USE TRUE)

# set c++ version c++11
SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_HIP_STANDARD 11)
add_definitions("-DCUB_IGNORE_DEPRECATED_CPP_DIALECT")
add_definitions("-DCUB_IGNORE_DEPRECATED_CPP_DIALECT")

message(STATUS "HIP major version is " ${HIP_VERSION_MAJOR})

set (HIP_HIPCC_FLAGS -hc; -fno-gpu-rdc; --amdgpu-target=gfx906; -fPIC; -O3; --std=c++11)

file (GLOB SOURCE_FILES "*.hip.cu" )

function(_add_librocm_variant variant_name prec_def)
set (libname "deepmd_op_rocm${variant_name}")
hip_add_library(${libname} SHARED ${SOURCE_FILES} OPTIONS "-D${prec_def}")

install(TARGETS ${libname} DESTINATION lib/)
if (BUILD_CPP_IF)
  install(TARGETS ${libname} DESTINATION lib/)
endif (BUILD_CPP_IF)
if (BUILD_PY_IF)
  install(TARGETS ${libname} DESTINATION deepmd/op/)
endif (BUILD_PY_IF)
endfunction()

_add_librocm_variant("${HIGH_PREC_VARIANT}" "${HIGH_PREC_DEF}")
_add_librocm_variant("${LOW_PREC_VARIANT}" "${LOW_PREC_DEF}")
