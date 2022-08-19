cmake_minimum_required(VERSION 3.9)
project(deepmd_api_test)

if (NOT DEFINED BUILD_CPP_IF) 
  set(BUILD_CPP_IF TRUE)
endif (NOT DEFINED BUILD_CPP_IF)
add_definitions ("-DHIGH_PREC")

enable_testing()

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
  # find tensorflow, I need tf abi info
  find_package(tensorflow REQUIRED)
  # auto op_cxx_abi
  try_run(
    CPP_CXX_ABI_RUN_RESULT_VAR CPP_CXX_ABI_COMPILE_RESULT_VAR
    ${CMAKE_CURRENT_BINARY_DIR}/tf_cxx_abi
    "${DEEPMD_SOURCE_DIR}/source/cmake/tf_cxx_abi.cpp"
    LINK_LIBRARIES ${TensorFlowFramework_LIBRARY}
    CMAKE_FLAGS "-DINCLUDE_DIRECTORIES:STRING=${TensorFlow_INCLUDE_DIRS}"
    RUN_OUTPUT_VARIABLE CPP_CXX_ABI
    COMPILE_OUTPUT_VARIABLE CPP_CXX_ABI_COMPILE_OUTPUT_VAR
  )
  if (NOT ${CPP_CXX_ABI_COMPILE_RESULT_VAR})
    message(FATAL_ERROR "Failed to compile: \n ${CPP_CXX_ABI_COMPILE_OUTPUT_VAR}" )
  endif()
  if (NOT ${CPP_CXX_ABI_RUN_RESULT_VAR} EQUAL "0")
    message(FATAL_ERROR "Failed to run, return code: ${CPP_CXX_ABI}" )
  endif()
  if (DEFINED PY_CXX_ABI)
    if (NOT (${CPP_CXX_ABI} EQUAL ${PY_CXX_ABI}))
	    message (WARNNING "NOT consistent CXX_ABIs: python interface of tf uses ${PY_CXX_ABI}, while c++ interface of tf uses ${CPP_CXX_ABI}, we follow c++ interface ")
    endif()
  endif()
  set(OP_CXX_ABI ${CPP_CXX_ABI})
  message (STATUS "Automatically determined OP_CXX_ABI=${OP_CXX_ABI} ")
  add_definitions(-D_GLIBCXX_USE_CXX11_ABI=${OP_CXX_ABI})
endif(USE_TF_PYTHON_LIBS)

# model version
file(READ ${PROJECT_SOURCE_DIR}/../../config/MODEL_VER MODEL_VERSION)
string(REPLACE "\n" " " MODEL_VERSION ${MODEL_VERSION})
message(STATUS "Supported model version: ${MODEL_VERSION}")

set(libname "deepmd")
set(apiname "deepmd_api")

set(opname "deepmd_op")

find_package(tensorflow REQUIRED)
if (TENSORFLOW_VERSION GREATER_EQUAL 2.7)
  set (CMAKE_CXX_STANDARD 14)
else()
  set (CMAKE_CXX_STANDARD 11)
endif()

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

set(HIGH_PREC_DEF "HIGH_PREC")
set(LOW_PREC_DEF "LOW_PREC")
set(HIGH_PREC_VARIANT "")
set(LOW_PREC_VARIANT "_low")
set(LIB_DEEPMD_CC ${apiname})
add_subdirectory(${CMAKE_SOURCE_DIR}/.. api_cc)

set(LIB_DEEPMD_OP ${opname})
add_subdirectory(${CMAKE_SOURCE_DIR}/../../op op)

file(GLOB TEST_SRC test_*.cc)
add_executable( runUnitTests ${TEST_SRC} )

add_library(coverage_config INTERFACE)
target_compile_options(coverage_config INTERFACE
  -O0        # no optimization
  -g         # generate debug info
  --coverage # sets all required flags
)
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.13)
  target_link_options(coverage_config INTERFACE --coverage)
else()
  target_link_libraries(coverage_config INTERFACE --coverage)
endif()

target_link_libraries(runUnitTests gtest gtest_main ${apiname} rt coverage_config)

add_test( runUnitTests runUnitTests )

find_package(GTest)
if(NOT GTEST_LIBRARIES)
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

set_target_properties(
  runUnitTests
  PROPERTIES 
  INSTALL_RPATH "$ORIGIN/../lib"
)
set_target_properties(
  ${apiname}
  PROPERTIES 
  INSTALL_RPATH "$ORIGIN;${TensorFlow_LIBRARY_PATH}"
)
install(TARGETS runUnitTests DESTINATION bin/)
