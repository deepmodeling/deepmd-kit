cmake_minimum_required(VERSION 3.9)
project(deepmd_api_test)
set(CMAKE_LINK_WHAT_YOU_USE TRUE)

if (NOT DEFINED BUILD_CPP_IF) 
  set(BUILD_CPP_IF TRUE)
endif (NOT DEFINED BUILD_CPP_IF)
add_definitions ("-DHIGH_PREC")

enable_testing()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# model version
file(READ ${PROJECT_SOURCE_DIR}/../../config/MODEL_VER MODEL_VERSION)
string(REPLACE "\n" " " MODEL_VERSION ${MODEL_VERSION})
message(STATUS "Supported model version: ${MODEL_VERSION}")

set(libname "deepmd")
set(LIB_BASE_DIR ${CMAKE_SOURCE_DIR}/../../lib)
include_directories(${LIB_BASE_DIR}/include)
file(GLOB LIB_SRC ${LIB_BASE_DIR}/src/*.cc ${LIB_BASE_DIR}/src/*.cpp)
add_library(${libname} SHARED ${LIB_SRC})

set(apiname "deepmd_api")
set(API_BASE_DIR ${CMAKE_SOURCE_DIR}/../)
include_directories(${API_BASE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR})
file(GLOB API_SRC ${API_BASE_DIR}/src/*.cc ${API_BASE_DIR}/src/*.cpp)
add_library(${apiname} SHARED ${API_SRC})
configure_file(
  ${API_BASE_DIR}/include/version.h.in
  ${CMAKE_SOURCE_DIR}/version.h
  @ONLY
)

set(opname "deepmd_op")
set(OP_BASE_DIR ${CMAKE_SOURCE_DIR}/../../op)
# file(GLOB OP_SRC ${OP_BASE_DIR}/*.cc)
file(GLOB OP_SRC ${OP_BASE_DIR}/prod_force.cc ${OP_BASE_DIR}/prod_virial.cc ${OP_BASE_DIR}/descrpt.cc ${OP_BASE_DIR}/descrpt_se_a_ef.cc ${OP_BASE_DIR}/descrpt_se_a_ef.cc ${OP_BASE_DIR}/descrpt_se_a_ef_para.cc ${OP_BASE_DIR}/descrpt_se_a_ef_vert.cc ${OP_BASE_DIR}/pair_tab.cc ${OP_BASE_DIR}/prod_force_multi_device.cc ${OP_BASE_DIR}/prod_virial_multi_device.cc ${OP_BASE_DIR}/soft_min.cc ${OP_BASE_DIR}/soft_min_force.cc ${OP_BASE_DIR}/soft_min_virial.cc ${OP_BASE_DIR}/ewald_recp.cc ${OP_BASE_DIR}/gelu_multi_device.cc ${OP_BASE_DIR}/map_aparam.cc ${OP_BASE_DIR}/neighbor_stat.cc ${OP_BASE_DIR}/unaggregated_grad.cc ${OP_BASE_DIR}/tabulate_multi_device.cc ${OP_BASE_DIR}/prod_env_mat_multi_device.cc)
add_library(${opname} SHARED ${OP_SRC})

list (APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/../../cmake/)
find_package(tensorflow REQUIRED)
include_directories(${TensorFlow_INCLUDE_DIRS})

find_package(Threads)
# find openmp
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# define USE_CUDA_TOOLKIT
if (DEFINED USE_CUDA_TOOLKIT)
  if (USE_CUDA_TOOLKIT)
    find_package(CUDA REQUIRED)
  else()
    message(STATUS "Will not build nv GPU support")
  endif()
else()
  find_package(CUDA QUIET)  
  if (CUDA_FOUND)
    set(USE_CUDA_TOOLKIT TRUE)
    message(STATUS "Found CUDA in ${CUDA_TOOLKIT_ROOT_DIR}, build nv GPU support")
  else()
    set(USE_CUDA_TOOLKIT FALSE)
    message(STATUS "No cuda support found, will not build nv GPU support")
  endif()
endif()

if (USE_CUDA_TOOLKIT)
  add_definitions("-D GOOGLE_CUDA")
  include_directories(${CUDA_INCLUDE_DIRS})
  add_subdirectory(${LIB_BASE_DIR}/src/cuda cuda_binary_dir)
endif()

file(GLOB TEST_SRC test_*.cc)
add_executable( runUnitTests ${TEST_SRC} )

if (USE_CUDA_TOOLKIT)
  target_link_libraries(runUnitTests gtest gtest_main ${libname} pthread deepmd_op_cuda)
else()
  target_link_libraries(runUnitTests gtest gtest_main ${libname} ${apiname} ${opname} pthread ${TensorFlow_LIBRARY})
endif()
add_test( runUnitTests runUnitTests )

find_package(GTest)
if(NOT GTEST_LIBRARY)
  configure_file(../../cmake/googletest.cmake.in googletest-download/CMakeLists.txt)
  execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
  if(result)
    message(FATAL_ERROR "CMake step for googletest failed: ${result}")
  endif()
  execute_process(COMMAND ${CMAKE_COMMAND} --build .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
  if(result)
    message(FATAL_ERROR "Build step for googletest failed: ${result}")
  endif()
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src ${CMAKE_CURRENT_BINARY_DIR}/googletest-build EXCLUDE_FROM_ALL)
else ()
  include_directories(${GTEST_INCLUDE_DIRS})
endif ()
