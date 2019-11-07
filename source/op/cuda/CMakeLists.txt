# required cmake version
cmake_minimum_required(VERSION 3.0)
# project name
project(deepmd_op_cuda)

# SET(CUDA_SEPARABLE_COMPILATION ON)
find_package(CUDA REQUIRED)
# if (NOT CUDA_FOUND)
# message(STATUS "CUDA not found. Project will not be built.")
# endif(NOT CUDA_FOUND)

# set c++ version c++11
SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CUDA_STANDARD 11)
# nvcc -o libdeepmd_op_cuda.so -I/usr/local/cub-1.8.0 -rdc=true -DHIGH_PREC=true -gencode arch=compute_61,code=sm_61 -shared -Xcompiler -fPIC deepmd_op.cu -L/usr/local/cuda/lib64 -lcudadevrt
# very important here! path to cub
include_directories(cub-1.8.0)
# nvcc flags
set(CUDA_NVCC_FLAGS -gencode arch=compute_61,code=sm_61; -O3; -Xcompiler -fPIC;)

# file(GLOB_RECURSE CUDA_HEAD ${CMAKE_CURRENT_LIST_DIR}/../include/.h)
# file(GLOB CUDA_SRC ${CMAKE_CURRENT_LIST_DIR}/*.cu)

# file(GLOB CURRENT_HEADERS include/NNPInter.h include/NNPAtomMap.h ${CMAKE_CURRENT_BINARY_DIR}/version.h)
# file(GLOB CURRENT_SOURCES  *.cu)
# 
# source_group("Include" FILES ${CUDA_HEAD}) 
# source_group("Source" FILES ${CUDA_SRC}) 
set (SOURCE_FILES
    descrpt_se_a.cu descrpt_se_r.cu prod_force_se_a.cu prod_force_se_r.cu prod_virial_se_a.cu prod_virial_se_r.cu 
)
# cuda_add_library(gpu SHARED ${CURRENT_HEADERS} ${CURRENT_SOURCES})
cuda_add_library(deepmd_op_cuda SHARED ${SOURCE_FILES})
# file(GLOB CURRENT_SOURCES cuda/*.cu)
# file(GLOB CURRENT_HEADERS include/*.h ${CMAKE_CURRENT_BINARY_DIR}/version.h)
# 
# source_group("Include" FILES ${CURRENT_HEADERS}) 
# source_group("Source" FILES ${CURRENT_SOURCES}) 

# file(GLOB LIB_SRC cuda/*.cu)
# file(GLOB INC_SRC include/*.h ${CMAKE_CURRENT_BINARY_DIR}/version.h)

# add_library(cuNNPInter SHARED ${LIB_SRC})

# cuda_add_library(cuNNPInter SHARED ${LIB_SRC})

# add_library(${libname} SHARED ${LIB_SRC})
# target_link_libraries(${libname} -lcudart)

install(TARGETS deepmd_op_cuda DESTINATION lib/)
# install(
#   FILES		${INC_SRC}
#   DESTINATION	include/deepmd
# )
