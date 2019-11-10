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
include_directories(cub) 
# nvcc flags
set(CUDA_NVCC_FLAGS -gencode arch=compute_60,code=sm_60; # Pascal – GP100/Tesla P100 – DGX-1 (Generic Pascal)
                    -gencode arch=compute_61,code=sm_61; # Pascal - GTX 1080, GTX 1070, GTX 1060, GTX 1050, GTX 1030, Titan Xp, Tesla P40, Tesla P4, Discrete GPU on the NVIDIA Drive PX2
                    -gencode arch=compute_70,code=sm_70; # Volta  - GV100/Tesla V100, GTX 1180 (GV104)
                    -gencode arch=compute_75,code=sm_75; # Turing - RTX 2080, Titan RTX, Quadro R8000
                    -O3; -Xcompiler -fPIC;
    )

set (SOURCE_FILES
    descrpt_se_a.cu descrpt_se_r.cu prod_force_se_a.cu prod_force_se_r.cu prod_virial_se_a.cu prod_virial_se_r.cu 
)

cuda_add_library(deepmd_op_cuda SHARED ${SOURCE_FILES})

install(TARGETS deepmd_op_cuda DESTINATION lib/)
