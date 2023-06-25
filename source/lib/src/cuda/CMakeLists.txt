# required cmake version 3.23: CMAKE_CUDA_ARCHITECTURES all
cmake_minimum_required(VERSION 3.23)
# project name
project(deepmd_op_cuda)

set(CMAKE_CUDA_ARCHITECTURES all)
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 11)
add_compile_definitions(
  "$<$<COMPILE_LANGUAGE:CUDA>:_GLIBCXX_USE_CXX11_ABI=${OP_CXX_ABI}>")

find_package(CUDAToolkit REQUIRED)

# take dynamic open cudart library replace of static one so it's not required
# when using CPUs
add_subdirectory(cudart)

# nvcc -o libdeepmd_op_cuda.so -I/usr/local/cub-1.8.0 -rdc=true -DHIGH_PREC=true
# -gencode arch=compute_61,code=sm_61 -shared -Xcompiler -fPIC deepmd_op.cu
# -L/usr/local/cuda/lib64 -lcudadevrt very important here! Include path to cub.
# for searching device compute capability,
# https://developer.nvidia.com/cuda-gpus

# cub has been included in CUDA Toolkit 11, we do not need to include it any
# more see https://github.com/NVIDIA/cub
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_LESS "11")
  include_directories(cub)
endif()

message(STATUS "NVCC version is " ${CMAKE_CUDA_COMPILER_VERSION})

# arch will be configured by CMAKE_CUDA_ARCHITECTURES
set(CMAKE_CUDA_FLAGS
    "${CMAKE_CUDA_FLAGS} -DCUB_IGNORE_DEPRECATED_CPP_DIALECT -DCUB_IGNORE_DEPRECATED_CPP_DIALECT"
)

file(GLOB SOURCE_FILES "*.cu")

add_library(deepmd_op_cuda SHARED ${SOURCE_FILES})
target_link_libraries(deepmd_op_cuda PRIVATE deepmd_dyn_cudart)
target_include_directories(
  deepmd_op_cuda
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include/>
         $<INSTALL_INTERFACE:include>)
target_precompile_headers(deepmd_op_cuda PUBLIC [["device.h"]])
if(APPLE)
  set_target_properties(deepmd_op_cuda PROPERTIES INSTALL_RPATH @loader_path)
else()
  set_target_properties(deepmd_op_cuda PROPERTIES INSTALL_RPATH "$ORIGIN")
endif()

if(BUILD_CPP_IF AND NOT BUILD_PY_IF)
  install(
    TARGETS deepmd_op_cuda
    EXPORT ${CMAKE_PROJECT_NAME}Targets
    DESTINATION lib/)
endif(BUILD_CPP_IF AND NOT BUILD_PY_IF)
if(BUILD_PY_IF)
  install(TARGETS deepmd_op_cuda DESTINATION deepmd/op/)
endif(BUILD_PY_IF)
