cmake_minimum_required(VERSION 3.9)
project(deepmd_api_test)

if (NOT DEFINED BUILD_CPP_IF) 
  set(BUILD_CPP_IF TRUE)
endif (NOT DEFINED BUILD_CPP_IF)
add_definitions ("-DHIGH_PREC")

enable_testing()

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

if (USE_TF_PYTHON_LIBS)
  target_link_libraries (${apiname} PUBLIC ${Python_LIBRARIES})
endif()

set(opname "deepmd_op")
set(OP_BASE_DIR ${CMAKE_SOURCE_DIR}/../../op)
# file(GLOB OP_SRC ${OP_BASE_DIR}/*.cc)
file(GLOB OP_SRC ${OP_BASE_DIR}/custom_op.cc ${OP_BASE_DIR}/prod_force.cc ${OP_BASE_DIR}/prod_virial.cc ${OP_BASE_DIR}/descrpt.cc ${OP_BASE_DIR}/descrpt_se_a_ef.cc ${OP_BASE_DIR}/descrpt_se_a_ef.cc ${OP_BASE_DIR}/descrpt_se_a_ef_para.cc ${OP_BASE_DIR}/descrpt_se_a_ef_vert.cc ${OP_BASE_DIR}/pair_tab.cc ${OP_BASE_DIR}/prod_force_multi_device.cc ${OP_BASE_DIR}/prod_virial_multi_device.cc ${OP_BASE_DIR}/soft_min.cc ${OP_BASE_DIR}/soft_min_force.cc ${OP_BASE_DIR}/soft_min_virial.cc ${OP_BASE_DIR}/ewald_recp.cc ${OP_BASE_DIR}/gelu_multi_device.cc ${OP_BASE_DIR}/map_aparam.cc ${OP_BASE_DIR}/neighbor_stat.cc ${OP_BASE_DIR}/unaggregated_grad.cc ${OP_BASE_DIR}/tabulate_multi_device.cc ${OP_BASE_DIR}/prod_env_mat_multi_device.cc)

find_package(tensorflow REQUIRED)
if (TENSORFLOW_VERSION GREATER_EQUAL 2.7)
  set (CMAKE_CXX_STANDARD 14)
else()
  set (CMAKE_CXX_STANDARD 11)
endif()
include_directories(${TensorFlow_INCLUDE_DIRS})
add_library(${opname} SHARED ${OP_SRC})

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
  include_directories(${CUDA_INCLUDE_DIRS})
  add_subdirectory(${LIB_BASE_DIR}/src/cuda cuda_binary_dir)
  target_link_libraries(${opname} PUBLIC deepmd_op_cuda)
else()
  message(STATUS "Will not build nv GPU support")
endif(USE_CUDA_TOOLKIT)

#define USE_ROCM_TOOLKIT
if (USE_ROCM_TOOLKIT)
  find_package(ROCM REQUIRED)
  add_definitions("-DTENSORFLOW_USE_ROCM")
  add_compile_definitions(__HIP_PLATFORM_HCC__)
  include_directories(${ROCM_INCLUDE_DIRS})
  add_subdirectory(${LIB_BASE_DIR}/src/rocm rocm_binary_dir)
  target_link_libraries(${opname} PUBLIC deepmd_op_rocm)
else()
  message(STATUS "Will not build AMD GPU support")
endif (USE_ROCM_TOOLKIT)

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

target_link_libraries(runUnitTests gtest gtest_main ${libname} ${apiname} pthread ${TensorFlow_LIBRARY} rt coverage_config)

find_package(Protobuf)
if(Protobuf_FOUND)
  include_directories(${Protobuf_INCLUDE_DIRS})
  target_link_libraries(runUnitTests ${Protobuf_LIBRARIES})
endif()

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
install(TARGETS ${libname} ${apiname} ${opname} DESTINATION lib/)
