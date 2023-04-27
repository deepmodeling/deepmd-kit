# required cmake version
cmake_minimum_required(VERSION 3.16)
# project name
project(deepmd_op_cuda)

# SET(CUDA_SEPARABLE_COMPILATION ON)
find_package(CUDA REQUIRED)
if(NOT CUDA_FOUND)
  message(STATUS "CUDA not found. Project will not be built.")
endif(NOT CUDA_FOUND)

# take dynamic open cudart library replace of static one so it's not required
# when using CPUs
add_subdirectory(cudart)
# important: it must be before cuda_add_library and any link target to cudart
set(CUDA_LIBRARIES deepmd_dyn_cudart)

# set c++ version c++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)
# nvcc -o libdeepmd_op_cuda.so -I/usr/local/cub-1.8.0 -rdc=true -DHIGH_PREC=true
# -gencode arch=compute_61,code=sm_61 -shared -Xcompiler -fPIC deepmd_op.cu
# -L/usr/local/cuda/lib64 -lcudadevrt very important here! Include path to cub.
# for searching device compute capability,
# https://developer.nvidia.com/cuda-gpus

# cub has been included in CUDA Toolkit 11, we do not need to include it any
# more see https://github.com/NVIDIA/cub
if(${CUDA_VERSION_MAJOR} LESS_EQUAL "10")
  include_directories(cub)
endif()

message(STATUS "CUDA major version is " ${CUDA_VERSION_MAJOR})

if(${CUDA_VERSION_MAJOR} GREATER "11"
   OR (${CUDA_VERSION_MAJOR} STREQUAL "11" AND ${CUDA_VERSION_MINOR}
                                               GREATER_EQUAL "5"))
  # nvcc flags
  set(CUDA_NVCC_FLAGS
      -arch=all; # embeds a compiled code image for all supported architectures
                 # (sm_*), and a PTX program for the highest major virtual
                 # architecture
      -O3;
      -Xcompiler
      -fPIC;
      ${CUDA_NVCC_FLAGS})
elseif(${CUDA_VERSION_MAJOR} STREQUAL "11" AND ${CUDA_VERSION_MINOR} GREATER
                                               "0")
  # nvcc flags
  set(CUDA_NVCC_FLAGS
      -gencode
      arch=compute_52,code=sm_52; # Tesla M40, Tesla M40, Quadro M6000...
      -gencode
      arch=compute_53,code=sm_53;
      -gencode
      arch=compute_60,code=sm_60; # Pascal – GP100/Tesla P100 – DGX-1 (Generic
                                  # Pascal)
      -gencode
      arch=compute_61,code=sm_61; # Pascal - GTX 1080, GTX 1070, GTX 1060, GTX
                                  # 1050, GTX 1030, Titan Xp, Tesla P40, Tesla
                                  # P4, Discrete GPU on the NVIDIA Drive PX2
      -gencode
      arch=compute_70,code=sm_70; # Volta  - GV100/Tesla V100, GTX 1180 (GV104)
      -gencode
      arch=compute_75,code=sm_75; # Turing - RTX 2080, Titan RTX, Quadro R8000
      -gencode
      arch=compute_80,code=sm_80; # Anpere - A100
      -gencode
      arch=compute_86,code=sm_86; # Anpere - RTX 3090
      -O3;
      -Xcompiler
      -fPIC;
      ${CUDA_NVCC_FLAGS})
elseif(${CUDA_VERSION_MAJOR} STREQUAL "11" AND ${CUDA_VERSION_MINOR} STREQUAL
                                               "0")
  # nvcc flags
  set(CUDA_NVCC_FLAGS
      -gencode
      arch=compute_52,code=sm_52; # Tesla M40, Tesla M40, Quadro M6000...
      -gencode
      arch=compute_53,code=sm_53;
      -gencode
      arch=compute_60,code=sm_60; # Pascal – GP100/Tesla P100 – DGX-1 (Generic
                                  # Pascal)
      -gencode
      arch=compute_61,code=sm_61; # Pascal - GTX 1080, GTX 1070, GTX 1060, GTX
                                  # 1050, GTX 1030, Titan Xp, Tesla P40, Tesla
                                  # P4, Discrete GPU on the NVIDIA Drive PX2
      -gencode
      arch=compute_70,code=sm_70; # Volta  - GV100/Tesla V100, GTX 1180 (GV104)
      -gencode
      arch=compute_75,code=sm_75; # Turing - RTX 2080, Titan RTX, Quadro R8000
      -gencode
      arch=compute_80,code=sm_80; # Anpere - A100
      -O3;
      -Xcompiler
      -fPIC;
      ${CUDA_NVCC_FLAGS})
elseif(${CUDA_VERSION_MAJOR} STREQUAL "10")
  set(CUDA_NVCC_FLAGS
      -gencode
      arch=compute_30,code=sm_30; # Tesla K10, Quadro K600 K420 K410,
      -gencode
      arch=compute_35,code=sm_35; # Tesla K20 K40, TITAN Z Black, GTX 780Ti 780
      -gencode
      arch=compute_37,code=sm_37; # Tesla K80
      -gencode
      arch=compute_50,code=sm_50; # Quadro 620 1200
      -gencode
      arch=compute_52,code=sm_52; # Tesla M40 M40, Quadro M6000 M5000 M4000
                                  # M2000, TITAN X, GTX 980Ti 980 970 960 950
      -gencode
      arch=compute_53,code=sm_53; # Jetson TX1, Tegra X1
      -gencode
      arch=compute_60,code=sm_60; # Pascal – GP100/Tesla P100 – DGX-1 (Generic
                                  # Pascal)
      -gencode
      arch=compute_61,code=sm_61; # Pascal - GTX 1080, GTX 1070, GTX 1060, GTX
                                  # 1050, GTX 1030, Titan Xp, Tesla P40, Tesla
                                  # P4, Discrete GPU on the NVIDIA Drive PX2
      -gencode
      arch=compute_70,code=sm_70; # Volta  - GV100/Tesla V100, GTX 1180 (GV104)
      -gencode
      arch=compute_75,code=sm_75; # Turing - RTX 2080, Titan RTX, Quadro R8000
      -O3;
      -Xcompiler
      -fPIC;
      ${CUDA_NVCC_FLAGS})
elseif(${CUDA_VERSION_MAJOR} STREQUAL "9")
  set(CUDA_NVCC_FLAGS
      -gencode
      arch=compute_30,code=sm_30;
      -gencode
      arch=compute_35,code=sm_35;
      -gencode
      arch=compute_37,code=sm_37;
      -gencode
      arch=compute_50,code=sm_50;
      -gencode
      arch=compute_52,code=sm_52; # Tesla M40, Tesla M40, Quadro M6000...
      -gencode
      arch=compute_53,code=sm_53;
      -gencode
      arch=compute_60,code=sm_60; # Pascal – GP100/Tesla P100 – DGX-1 (Generic
                                  # Pascal)
      -gencode
      arch=compute_61,code=sm_61; # Pascal - GTX 1080, GTX 1070, GTX 1060, GTX
                                  # 1050, GTX 1030, Titan Xp, Tesla P40, Tesla
                                  # P4, Discrete GPU on the NVIDIA Drive PX2
      -gencode
      arch=compute_70,code=sm_70; # Volta  - GV100/Tesla V100, GTX 1180 (GV104)
      -O3;
      -Xcompiler
      -fPIC;
      ${CUDA_NVCC_FLAGS})
elseif(${CUDA_VERSION_MAJOR} STREQUAL "8")
  set(CUDA_NVCC_FLAGS
      -gencode
      arch=compute_30,code=sm_30;
      -gencode
      arch=compute_35,code=sm_35;
      -gencode
      arch=compute_37,code=sm_37;
      -gencode
      arch=compute_50,code=sm_50;
      -gencode
      arch=compute_52,code=sm_52; # Tesla M40, Tesla M40, Quadro M6000...
      -gencode
      arch=compute_53,code=sm_53;
      -gencode
      arch=compute_60,code=sm_60; # Pascal – GP100/Tesla P100 – DGX-1 (Generic
                                  # Pascal)
      -gencode
      arch=compute_61,code=sm_61; # Pascal - GTX 1080, GTX 1070, GTX 1060, GTX
                                  # 1050, GTX 1030, Titan Xp, Tesla P40, Tesla
                                  # P4, Discrete GPU on the NVIDIA Drive PX2
      -O3;
      -Xcompiler
      -fPIC;
      ${CUDA_NVCC_FLAGS})
elseif(${CUDA_VERSION_MAJOR} STREQUAL "7")
  set(CUDA_NVCC_FLAGS
      -gencode
      arch=compute_30,code=sm_30;
      -gencode
      arch=compute_35,code=sm_35;
      -gencode
      arch=compute_37,code=sm_37;
      -gencode
      arch=compute_50,code=sm_50;
      -gencode
      arch=compute_52,code=sm_52; # Tesla M40, Tesla M40, Quadro M6000...
      -gencode
      arch=compute_53,code=sm_53;
      -O3;
      -Xcompiler
      -fPIC;
      ${CUDA_NVCC_FLAGS})
else()
  message(FATAL_ERROR "unsupported CUDA_VERSION " ${CUDA_VERSION}
                      ", please use a newer version (>=7.0) of CUDA toolkit!")
endif()

set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -std=c++11 -DCUB_IGNORE_DEPRECATED_CPP_DIALECT -DCUB_IGNORE_DEPRECATED_CPP_DIALECT"
)

if(${CUDA_VERSION_MAJOR} LESS_EQUAL "11")
  # check unsupported -std=c++17
  set(CMAKE_CXX_FLAGS_LIST "${CMAKE_CXX_FLAGS}")
  separate_arguments(CMAKE_CXX_FLAGS_LIST)
  if("-std=c++17" IN_LIST CMAKE_CXX_FLAGS_LIST)
    message(
      WARNING
        "Environment variable CXXFLAGS contains flag --std=c++17 which is unsupported by CUDA ${CUDA_VERSION}. Such flag will be removed automatically."
    )
    string(REPLACE "-std=c++17" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  endif()
endif()

file(GLOB SOURCE_FILES "*.cu")

cuda_add_library(deepmd_op_cuda SHARED ${SOURCE_FILES})
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
