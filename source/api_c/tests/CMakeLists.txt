cmake_minimum_required(VERSION 3.9)
project(deepmd_api_c_test)

if (NOT DEFINED BUILD_CPP_IF) 
  set(BUILD_CPP_IF TRUE)
endif (NOT DEFINED BUILD_CPP_IF)

enable_testing()
add_subdirectory(${CMAKE_SOURCE_DIR}/../../cmake/coverage_config coverage_config)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(DEEPMD_SOURCE_DIR ${PROJECT_SOURCE_DIR}/../../..)
list (APPEND CMAKE_MODULE_PATH ${DEEPMD_SOURCE_DIR}/source/cmake/)

# setup tensorflow libraries by python
if (USE_TF_PYTHON_LIBS)
  set(DEEPMD_SOURCE_DIR ${PROJECT_SOURCE_DIR}/../../..)
  find_package (Python COMPONENTS Interpreter Development REQUIRED)
endif(USE_TF_PYTHON_LIBS)

# model version
file(READ ${PROJECT_SOURCE_DIR}/../../config/MODEL_VER MODEL_VERSION)
string(REPLACE "\n" " " MODEL_VERSION ${MODEL_VERSION})
message(STATUS "Supported model version: ${MODEL_VERSION}")

set(libname "deepmd")
set(apiname "deepmd_api")

set(opname "deepmd_op")

find_package(tensorflow REQUIRED)

find_package(Threads)
# find openmp
find_package(OpenMP)
if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Devices that have both ROCM and CUDA are not currently supported
if (USE_ROCM_TOOLKIT AND USE_CUDA_TOOLKIT)
  message (FATAL_ERROR "Devices that have both ROCM and CUDA are not currently supported")
endif()

# define USE_CUDA_TOOLKIT
if (USE_CUDA_TOOLKIT)
  find_package(CUDA REQUIRED)
  add_definitions("-DGOOGLE_CUDA")
  message(STATUS "Found CUDA in ${CUDA_TOOLKIT_ROOT_DIR}, build nv GPU support")
else()
  message(STATUS "Will not build nv GPU support")
endif(USE_CUDA_TOOLKIT)

#define USE_ROCM_TOOLKIT
if (USE_ROCM_TOOLKIT)
  find_package(ROCM REQUIRED)
  add_definitions("-DTENSORFLOW_USE_ROCM")
  add_compile_definitions(__HIP_PLATFORM_HCC__)
else()
  message(STATUS "Will not build AMD GPU support")
endif (USE_ROCM_TOOLKIT)

set(LIB_DEEPMD ${libname})
add_subdirectory(${CMAKE_SOURCE_DIR}/../../lib lib)

set(HIGH_PREC_VARIANT "")
set(LOW_PREC_VARIANT "_low")

set(LIB_DEEPMD_CC ${apiname})
add_subdirectory(${CMAKE_SOURCE_DIR}/../../api_cc api_cc)

set(LIB_DEEPMD_OP ${opname})
add_subdirectory(${CMAKE_SOURCE_DIR}/../../op op)

set(cname "deepmd_c")
set(LIB_DEEPMD_C ${cname})
add_subdirectory(${CMAKE_SOURCE_DIR}/.. api_c)

file(GLOB TEST_SRC test_*.cc)

find_package(GTest)
if(NOT GTEST_LIBRARIES)
  configure_file(../../cmake/googletest.cmake.in googletest-download/CMakeLists.txt @ONLY)
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

set_target_properties(
  ${cname}
  PROPERTIES 
  INSTALL_RPATH "$ORIGIN;${TensorFlow_LIBRARY_PATH}"
)

add_executable( runUnitTests ${TEST_SRC} )
target_link_libraries(runUnitTests gtest gtest_main ${cname} rt coverage_config)
add_test( runUnitTests runUnitTests )
set_target_properties(
  runUnitTests
  PROPERTIES 
  INSTALL_RPATH "$ORIGIN/../lib"
)
install(TARGETS runUnitTests DESTINATION bin/)

