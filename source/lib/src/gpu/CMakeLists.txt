if(USE_CUDA_TOOLKIT)
  # required cmake version 3.23: CMAKE_CUDA_ARCHITECTURES all
  cmake_minimum_required(VERSION 3.23)
  # project name
  project(deepmd_op_cuda)
  set(GPU_LIB_NAME deepmd_op_cuda)

  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES all)
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 11)
  if(DEFINED OP_CXX_ABI)
    add_compile_definitions(
      "$<$<COMPILE_LANGUAGE:CUDA>:_GLIBCXX_USE_CXX11_ABI=${OP_CXX_ABI}>")
  endif()

  find_package(CUDAToolkit REQUIRED)

  # take dynamic open cudart library replace of static one so it's not required
  # when using CPUs
  add_subdirectory(cudart)

  # nvcc -o libdeepmd_op_cuda.so -I/usr/local/cub-1.8.0 -rdc=true
  # -DHIGH_PREC=true -gencode arch=compute_61,code=sm_61 -shared -Xcompiler
  # -fPIC deepmd_op.cu -L/usr/local/cuda/lib64 -lcudadevrt very important here!
  # Include path to cub. for searching device compute capability,
  # https://developer.nvidia.com/cuda-gpus

  # cub has been included in CUDA Toolkit 11, we do not need to include it any
  # more see https://github.com/NVIDIA/cub
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_LESS "11")
    include(FetchContent)
    FetchContent_Declare(
      cub_download
      GIT_REPOSITORY https://github.com/NVIDIA/cub
      GIT_TAG b229817e3963fc942c7cc2c61715a6b2b2c49bed)
    FetchContent_GetProperties(cub_download)
    if(NOT cub_download_POPULATED)
      FetchContent_Populate(cub_download)
      set(CUB_SOURCE_ROOT ${cub_download_SOURCE_DIR})
    endif()
    include_directories(${CUB_SOURCE_ROOT})
  endif()
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_LESS "9")
    message(FATAL_ERROR "CUDA version must be >= 9.0")
  endif()

  message(STATUS "NVCC version is " ${CMAKE_CUDA_COMPILER_VERSION})

  # arch will be configured by CMAKE_CUDA_ARCHITECTURES
  set(CMAKE_CUDA_FLAGS
      "${CMAKE_CUDA_FLAGS} -DCUB_IGNORE_DEPRECATED_CPP_DIALECT -DCUB_IGNORE_DEPRECATED_CPP_DIALECT"
  )

  file(GLOB SOURCE_FILES "*.cu")

  add_library(${GPU_LIB_NAME} SHARED ${SOURCE_FILES})
  target_link_libraries(${GPU_LIB_NAME} PRIVATE deepmd_dyn_cudart)

elseif(USE_ROCM_TOOLKIT)

  # required cmake version
  cmake_minimum_required(VERSION 3.21)
  # project name
  project(deepmd_op_rocm)
  enable_language(HIP)
  set(GPU_LIB_NAME deepmd_op_rocm)
  set(CMAKE_LINK_WHAT_YOU_USE TRUE)

  # set c++ version c++11
  set(CMAKE_CXX_STANDARD 14)
  set(CMAKE_HIP_STANDARD 14)
  add_definitions("-DCUB_IGNORE_DEPRECATED_CPP_DIALECT")

  message(STATUS "HIP major version is " ${hip_VERSION_MAJOR})

  set(CMAKE_HIP_FLAGS "-fno-gpu-rdc ${CMAKE_HIP_FLAGS}"
  )# --amdgpu-target=gfx906
  if(hip_VERSION VERSION_LESS 3.5.1)
    set(CMAKE_HIP_FLAGS "-hc ${CMAKE_HIP_FLAGS}")
  endif()

  file(GLOB SOURCE_FILES "*.cu")

  add_library(${GPU_LIB_NAME} SHARED ${SOURCE_FILES})
  set_source_files_properties(${SOURCE_FILES} PROPERTIES LANGUAGE HIP)
  # -fpic
  set_property(TARGET ${GPU_LIB_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)
  target_link_libraries(${GPU_LIB_NAME} PRIVATE hip::hipcub)

endif()

target_include_directories(
  ${GPU_LIB_NAME}
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include/>
         $<INSTALL_INTERFACE:include>)
target_precompile_headers(${GPU_LIB_NAME} PUBLIC [["device.h"]])
if(APPLE)
  set_target_properties(${GPU_LIB_NAME} PROPERTIES INSTALL_RPATH @loader_path)
else()
  set_target_properties(${GPU_LIB_NAME} PROPERTIES INSTALL_RPATH "$ORIGIN")
endif()

if(BUILD_CPP_IF AND NOT BUILD_PY_IF)
  install(
    TARGETS ${GPU_LIB_NAME}
    EXPORT ${CMAKE_PROJECT_NAME}Targets
    DESTINATION lib/)
endif(BUILD_CPP_IF AND NOT BUILD_PY_IF)
if(BUILD_PY_IF)
  install(TARGETS ${GPU_LIB_NAME} DESTINATION deepmd/lib/)
endif(BUILD_PY_IF)
