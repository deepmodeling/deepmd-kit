# required cmake version
cmake_minimum_required(VERSION 3.0)
# project name
project(deepmd_op_cuda)

# SET(CUDA_SEPARABLE_COMPILATION ON)
find_package(CUDA REQUIRED)
if (NOT CUDA_FOUND)
    message(STATUS "CUDA not found. Project will not be built.")
endif(NOT CUDA_FOUND)

# set c++ version c++11
SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CUDA_STANDARD 11)
# nvcc -o libdeepmd_op_cuda.so -I/usr/local/cub-1.8.0 -rdc=true -DHIGH_PREC=true -gencode arch=compute_61,code=sm_61 -shared -Xcompiler -fPIC deepmd_op.cu -L/usr/local/cuda/lib64 -lcudadevrt
# very important here! Include path to cub.
# for searching device compute capability, https://developer.nvidia.com/cuda-gpus
include_directories(cub)

message(STATUS "CUDA major version is " ${CUDA_VERSION_MAJOR})

if (${CUDA_VERSION_MAJOR} GREATER "11")
    # nvcc flags
    set(CUDA_NVCC_FLAGS -gencode arch=compute_60,code=sm_60; # Pascal – GP100/Tesla P100 – DGX-1 (Generic Pascal)
                        -gencode arch=compute_61,code=sm_61; # Pascal - GTX 1080, GTX 1070, GTX 1060, GTX 1050, GTX 1030, Titan Xp, Tesla P40, Tesla P4, Discrete GPU on the NVIDIA Drive PX2
                        -gencode arch=compute_70,code=sm_70; # Volta  - GV100/Tesla V100, GTX 1180 (GV104)
                        -gencode arch=compute_75,code=sm_75; # Turing - RTX 2080, Titan RTX, Quadro R8000
                        -gencode arch=compute_80,code=sm_80; # Anpere - A100
                        -gencode arch=compute_86,code=sm_86; # Anpere - RTX 3090
                        -O3; -Xcompiler -fPIC;
        )
elseif (${CUDA_VERSION_MAJOR} STREQUAL "11" AND ${CUDA_VERSION_MINOR} GREATER "0")
    # nvcc flags
    set(CUDA_NVCC_FLAGS -gencode arch=compute_50,code=sm_50;
                        -gencode arch=compute_52,code=sm_52; # Tesla M40, Tesla M40, Quadro M6000...
                        -gencode arch=compute_53,code=sm_53; 
                        -gencode arch=compute_60,code=sm_60; # Pascal – GP100/Tesla P100 – DGX-1 (Generic Pascal)
                        -gencode arch=compute_61,code=sm_61; # Pascal - GTX 1080, GTX 1070, GTX 1060, GTX 1050, GTX 1030, Titan Xp, Tesla P40, Tesla P4, Discrete GPU on the NVIDIA Drive PX2
                        -gencode arch=compute_70,code=sm_70; # Volta  - GV100/Tesla V100, GTX 1180 (GV104)
                        -gencode arch=compute_75,code=sm_75; # Turing - RTX 2080, Titan RTX, Quadro R8000
                        -gencode arch=compute_80,code=sm_80; # Anpere - A100
                        -gencode arch=compute_86,code=sm_86; # Anpere - RTX 3090
                        -O3; -Xcompiler -fPIC;
        )
elseif (${CUDA_VERSION_MAJOR} STREQUAL "11" AND ${CUDA_VERSION_MINOR} STREQUAL "0")
    # nvcc flags
    set(CUDA_NVCC_FLAGS -gencode arch=compute_50,code=sm_50;
                        -gencode arch=compute_52,code=sm_52; # Tesla M40, Tesla M40, Quadro M6000...
                        -gencode arch=compute_53,code=sm_53; 
                        -gencode arch=compute_60,code=sm_60; # Pascal – GP100/Tesla P100 – DGX-1 (Generic Pascal)
                        -gencode arch=compute_61,code=sm_61; # Pascal - GTX 1080, GTX 1070, GTX 1060, GTX 1050, GTX 1030, Titan Xp, Tesla P40, Tesla P4, Discrete GPU on the NVIDIA Drive PX2
                        -gencode arch=compute_70,code=sm_70; # Volta  - GV100/Tesla V100, GTX 1180 (GV104)
                        -gencode arch=compute_75,code=sm_75; # Turing - RTX 2080, Titan RTX, Quadro R8000
                        -gencode arch=compute_80,code=sm_80; # Anpere - A100
                        -O3; -Xcompiler -fPIC;
        )
elseif (${CUDA_VERSION_MAJOR} STREQUAL "10")
    set(CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30; # Tesla K10, Quadro K600 K420 K410,
                        -gencode arch=compute_35,code=sm_35; # Tesla K20 K40, TITAN Z Black, GTX 780Ti 780
                        -gencode arch=compute_37,code=sm_37; # Tesla K80
                        -gencode arch=compute_50,code=sm_50; # Quadro 620 1200
                        -gencode arch=compute_52,code=sm_52; # Tesla M40 M40, Quadro M6000 M5000 M4000 M2000, TITAN X, GTX 980Ti 980 970 960 950
                        -gencode arch=compute_53,code=sm_53; # Jetson TX1, Tegra X1
                        -gencode arch=compute_60,code=sm_60; # Pascal – GP100/Tesla P100 – DGX-1 (Generic Pascal)
                        -gencode arch=compute_61,code=sm_61; # Pascal - GTX 1080, GTX 1070, GTX 1060, GTX 1050, GTX 1030, Titan Xp, Tesla P40, Tesla P4, Discrete GPU on the NVIDIA Drive PX2
                        -gencode arch=compute_70,code=sm_70; # Volta  - GV100/Tesla V100, GTX 1180 (GV104)
                        -gencode arch=compute_75,code=sm_75; # Turing - RTX 2080, Titan RTX, Quadro R8000
                        -O3; -Xcompiler -fPIC;
        )
elseif (${CUDA_VERSION_MAJOR} STREQUAL "9")
    set(CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30;
                        -gencode arch=compute_35,code=sm_35;
                        -gencode arch=compute_37,code=sm_37;
                        -gencode arch=compute_50,code=sm_50;
                        -gencode arch=compute_52,code=sm_52; # Tesla M40, Tesla M40, Quadro M6000...
                        -gencode arch=compute_53,code=sm_53; 
                        -gencode arch=compute_60,code=sm_60; # Pascal – GP100/Tesla P100 – DGX-1 (Generic Pascal)
                        -gencode arch=compute_61,code=sm_61; # Pascal - GTX 1080, GTX 1070, GTX 1060, GTX 1050, GTX 1030, Titan Xp, Tesla P40, Tesla P4, Discrete GPU on the NVIDIA Drive PX2
                        -gencode arch=compute_70,code=sm_70; # Volta  - GV100/Tesla V100, GTX 1180 (GV104)
                        -O3; -Xcompiler -fPIC;
        )
elseif (${CUDA_VERSION_MAJOR} STREQUAL "8")
    set(CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30;
                        -gencode arch=compute_35,code=sm_35;
                        -gencode arch=compute_37,code=sm_37;
                        -gencode arch=compute_50,code=sm_50;
                        -gencode arch=compute_52,code=sm_52; # Tesla M40, Tesla M40, Quadro M6000...
                        -gencode arch=compute_53,code=sm_53; 
                        -gencode arch=compute_60,code=sm_60; # Pascal – GP100/Tesla P100 – DGX-1 (Generic Pascal)
                        -gencode arch=compute_61,code=sm_61; # Pascal - GTX 1080, GTX 1070, GTX 1060, GTX 1050, GTX 1030, Titan Xp, Tesla P40, Tesla P4, Discrete GPU on the NVIDIA Drive PX2
                        -O3; -Xcompiler -fPIC;
        )
elseif (${CUDA_VERSION_MAJOR} STREQUAL "7")
    set(CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30;
                        -gencode arch=compute_35,code=sm_35;
                        -gencode arch=compute_37,code=sm_37;
                        -gencode arch=compute_50,code=sm_50;
                        -gencode arch=compute_52,code=sm_52; # Tesla M40, Tesla M40, Quadro M6000...
                        -gencode arch=compute_53,code=sm_53; 
                        -O3; -Xcompiler -fPIC;
        )
else () 
    message(FATAL_ERROR "unsupported CUDA_VERSION " ${CUDA_VERSION} ", please use a newer version (>=7.0) of CUDA toolkit!")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -DCUB_IGNORE_DEPRECATED_CPP_DIALECT -DCUB_IGNORE_DEPRECATED_CPP_DIALECT")

set (SOURCE_FILES
    prod_env_mat.cu prod_force.cu prod_force_grad.cu prod_virial.cu prod_virial_grad.cu gelu.cu tabulate.cu coord.cu neighbor_list.cu region.cu
)

cuda_add_library(deepmd_op_cuda SHARED ${SOURCE_FILES})

if (BUILD_CPP_IF)
    install(TARGETS deepmd_op_cuda DESTINATION lib/)
endif (BUILD_CPP_IF)
if (BUILD_PY_IF)
    install(TARGETS deepmd_op_cuda DESTINATION deepmd/op/)
endif (BUILD_PY_IF)
