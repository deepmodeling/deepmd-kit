# new in 3.16: GET_RUNTIME_DEPENDENCIES, target_precompile_headers
cmake_minimum_required(VERSION 3.16)
project(DeePMD)

option(ENABLE_TENSORFLOW "Enable TensorFlow interface" OFF)
option(ENABLE_PYTORCH "Enable PyTorch interface" OFF)
option(BUILD_TESTING "Build test and enable converage" OFF)
set(DEEPMD_C_ROOT
    ""
    CACHE PATH "Path to imported DeePMD-kit C library")

if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(${CMAKE_SOURCE_DIR}/cmake/coverage_config coverage_config)
endif()

# build cpp or python interfaces
if(NOT DEFINED BUILD_CPP_IF)
  set(BUILD_CPP_IF TRUE)
endif(NOT DEFINED BUILD_CPP_IF)
if(NOT DEFINED BUILD_PY_IF)
  set(BUILD_PY_IF FALSE)
endif(NOT DEFINED BUILD_PY_IF)
if((NOT BUILD_PY_IF) AND (NOT BUILD_CPP_IF))
  # nothing to do
  message(FATAL_ERROR "Nothing to do.")
endif()

if(BUILD_CPP_IF AND BUILD_TESTING)
  if(NOT GTEST_LIBRARIES)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/googletest.cmake.in
                   googletest-download/CMakeLists.txt @ONLY)
    execute_process(
      COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
      RESULT_VARIABLE result
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download)
    if(result)
      message(FATAL_ERROR "CMake step for googletest failed: ${result}")
    endif()
    execute_process(
      COMMAND ${CMAKE_COMMAND} --build .
      RESULT_VARIABLE result
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download)
    if(result)
      message(FATAL_ERROR "Build step for googletest failed: ${result}")
    endif()
    set(gtest_force_shared_crt
        ON
        CACHE BOOL "" FORCE)
    add_subdirectory(
      ${CMAKE_CURRENT_BINARY_DIR}/googletest-src
      ${CMAKE_CURRENT_BINARY_DIR}/googletest-build EXCLUDE_FROM_ALL)
  endif()
endif()

find_package(Git)
if(GIT_FOUND)
  execute_process(
    COMMAND git describe --tags --dirty
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_SUMM
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process(
    COMMAND git log -1 --format=%h
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process(
    COMMAND git rev-parse --abbrev-ref HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_BRANCH
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process(
    COMMAND git show -s --format=%ci ${GIT_HASH}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_DATE
    OUTPUT_STRIP_TRAILING_WHITESPACE)
endif(GIT_FOUND)

# global defines
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/)

# model version
file(READ ${PROJECT_SOURCE_DIR}/config/MODEL_VER MODEL_VERSION)
string(REPLACE "\n" " " MODEL_VERSION ${MODEL_VERSION})
message(STATUS "Supported model version: ${MODEL_VERSION}")

# Devices that have both ROCM and CUDA are not currently supported
if(USE_ROCM_TOOLKIT AND USE_CUDA_TOOLKIT)
  message(
    FATAL_ERROR
      "Devices that have both ROCM and CUDA are not currently supported")
endif()
set(DP_VARIANT "cpu")

# define USE_CUDA_TOOLKIT
if(USE_CUDA_TOOLKIT)
  cmake_minimum_required(VERSION 3.23)
  find_package(CUDAToolkit REQUIRED)
  if(NOT DEFINED CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER ${CUDAToolkit_NVCC_EXECUTABLE})
  endif()
  if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
    set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
  endif()
  add_definitions("-DGOOGLE_CUDA")
  message(STATUS "Found CUDA in ${CUDAToolkit_BIN_DIR}, build nv GPU support")
  set(DP_VARIANT "cuda")
else()
  message(STATUS "Will not build nv GPU support")
endif(USE_CUDA_TOOLKIT)

# define USE_ROCM_TOOLKIT
if(USE_ROCM_TOOLKIT)
  cmake_minimum_required(VERSION 3.21)
  include(CMakeDetermineHIPCompiler)
  list(APPEND CMAKE_PREFIX_PATH ${CMAKE_HIP_COMPILER_ROCM_ROOT})
  find_package(hip REQUIRED)
  find_package(hipcub REQUIRED)
  add_definitions("-DTENSORFLOW_USE_ROCM")
  message(
    STATUS
      "Found ROCM in ${CMAKE_HIP_COMPILER_ROCM_ROOT}, build AMD GPU support")
  set(DP_VARIANT "rocm")
else()
  message(STATUS "Will not build AMD GPU support")
endif(USE_ROCM_TOOLKIT)

set(DEEPMD_SOURCE_DIR ${PROJECT_SOURCE_DIR}/..)

# setup tensorflow libraries by python
if(INSTALL_TENSORFLOW)
  set(USE_TF_PYTHON_LIBS TRUE)
endif(INSTALL_TENSORFLOW)
if(USE_TF_PYTHON_LIBS)
  set(ENABLE_TENSORFLOW TRUE)
  if(NOT "$ENV{CIBUILDWHEEL}" STREQUAL "1")
    find_package(
      Python
      COMPONENTS Interpreter Development
      REQUIRED)
  else()
    set(Python_LIBRARIES ${Python_LIBRARY})
    set(PYTHON_INCLUDE_DIRS ${PYTHON_INCLUDE_DIR})
  endif()
endif(USE_TF_PYTHON_LIBS)
if(TENSORFLOW_ROOT)
  set(ENABLE_TENSORFLOW TRUE)
endif()

# find tensorflow, I need tf abi info
if(ENABLE_TENSORFLOW AND NOT DEEPMD_C_ROOT)
  find_package(tensorflow REQUIRED)
endif()
if(ENABLE_PYTORCH AND NOT DEEPMD_C_ROOT)
  find_package(Torch REQUIRED)
  string(REGEX MATCH "_GLIBCXX_USE_CXX11_ABI=([0-9]+)" CXXABI_PT_MATCH
               ${TORCH_CXX_FLAGS})
  if(CXXABI_PT_MATCH)
    message(STATUS "PyTorch CXX11 ABI: ${CMAKE_MATCH_1}")
    if(DEFINED OP_CXX_ABI)
      if(NOT ${CMAKE_MATCH_1} EQUAL ${OP_CXX_ABI})
        message(
          FATAL_ERROR
            "PyTorch CXX11 ABI mismatch TensorFlow: ${CMAKE_MATCH_1} != ${OP_CXX_ABI}"
        )
      endif()
    else()
      set(OP_CXX_ABI ${CMAKE_MATCH_1})
      add_definitions(-D_GLIBCXX_USE_CXX11_ABI=${OP_CXX_ABI})
    endif()
  endif()
endif()
# log enabled backends
if(NOT DEEPMD_C_ROOT)
  message(STATUS "Enabled backends:")
  if(ENABLE_TENSORFLOW)
    message(STATUS "- TensorFlow")
  endif()
  if(ENABLE_PYTORCH)
    message(STATUS "- PyTorch")
  endif()
  if(NOT ENABLE_TENSORFLOW
     AND NOT ENABLE_PYTORCH
     AND NOT BUILD_PY_IF)
    message(FATAL_ERROR "No backend is enabled.")
  endif()
endif()

# find threads
find_package(Threads)

# define build type
if((NOT DEFINED CMAKE_BUILD_TYPE) OR CMAKE_BUILD_TYPE STREQUAL "")
  set(CMAKE_BUILD_TYPE release)
endif()

# set op prec
set(HIGH_PREC_DEF "HIGH_PREC")
# this defination doesn't work, but leaving it empty will cause error
set(LOW_PREC_DEF "LOW_PREC")
set(HIGH_PREC_VARIANT "")
set(LOW_PREC_VARIANT "_low")

# find openmp
find_package(OpenMP)
if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# optimize flags
option(ENABLE_NATIVE_OPTIMIZATION "Enable native optimization" OFF)
if(ENABLE_NATIVE_OPTIMIZATION)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native -mtune=native")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mtune=native")
endif()

# define names of libs
set(LIB_DEEPMD "deepmd")
set(LIB_DEEPMD_OP "deepmd_op")
if(BUILD_CPP_IF)
  set(LIB_DEEPMD_CC "deepmd_cc")
  set(LIB_DEEPMD_C "deepmd_c")
  if(USE_CUDA_TOOLKIT)
    set(LIB_DEEPMD_OP_DEVICE "deepmd_op_cuda")
  elseif(USE_ROCM_TOOLKIT)
    set(LIB_DEEPMD_OP_DEVICE "deepmd_op_rocm")
  else()
    set(LIB_DEEPMD_OP_DEVICE "deepmd_op")
  endif()
  if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 4.8)
    set(LIB_DEEPMD_NATIVE "deepmd_native_md")
    set(LIB_DEEPMD_IPI "deepmd_ipi")
    set(LIB_DEEPMD_GROMACS "deepmd_gromacs")
  else()
    message(
      STATUS
        "Your gcc/g++ version is ${CMAKE_CXX_COMPILER_VERSION}, so native MD, ipi and gromacs plugin are disabled. To enable them, use gcc/g++ >= 4.8."
    )
  endif()
endif(BUILD_CPP_IF)

option(DP_USING_C_API "Build third-party interface with C API" ON)

if(DEEPMD_C_ROOT)
  # find c library
  find_path(
    DEEPMD_INCLUDE_C_DIR deepmd/deepmd.hpp deepmd/c_api.h
    HINTS ${DEEPMD_C_ROOT}
    PATH_SUFFIXES "include")
  find_library(
    deepmd_c
    NAMES deepmd_c
    HINTS ${DEEPMD_C_ROOT}
    PATH_SUFFIXES "lib")
  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(DEEPMD_C DEFAULT_MSG deepmd_c
                                    DEEPMD_INCLUDE_C_DIR)
  if(NOT DEEPMD_C_FOUND)
    message(
      FATAL_ERROR
        "DeePMD-kit C library not found. Download from https://github.com/deepmodeling/deepmd-kit/releases"
    )
  endif()

  add_library(${LIB_DEEPMD_C} SHARED IMPORTED GLOBAL)
  set_target_properties(
    ${LIB_DEEPMD_C}
    PROPERTIES IMPORTED_LINK_INTERFACE_LANGUAGES "C"
               IMPORTED_LOCATION "${deepmd_c}"
               INTERFACE_INCLUDE_DIRECTORIES "${DEEPMD_INCLUDE_C_DIR}/deepmd")
  # use variable for TF path to set deepmd_c path
  set(TENSORFLOW_ROOT "${DEEPMD_C_ROOT}")
  set(TensorFlow_LIBRARY_PATH "${DEEPMD_C_ROOT}/lib")
  set(TENSORFLOW_INCLUDE_DIRS "${DEEPMD_C_ROOT}/include")
  set(TORCH_LIBRARIES "${DEEPMD_C_ROOT}/lib/libtorch.so")
endif()

if(NOT DEEPMD_C_ROOT)
  if(ENABLE_TENSORFLOW)
    add_subdirectory(op/)
  endif()
  add_subdirectory(lib/)
endif()
if(BUILD_PY_IF)
  add_subdirectory(config/)
  # add_subdirectory (tests/)
endif(BUILD_PY_IF)
if(BUILD_CPP_IF)
  if(NOT DEEPMD_C_ROOT)
    add_subdirectory(api_cc/)
    add_subdirectory(api_c/)
  endif()
  if(LAMMPS_VERSION OR NOT BUILD_PY_IF)
    add_subdirectory(lmp/)
  endif()
  if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 4.8)
    # add_subdirectory (md/)
    if(ENABLE_IPI OR NOT BUILD_PY_IF)
      add_subdirectory(ipi/)
    endif()
    if(NOT BUILD_PY_IF)
      add_subdirectory(gmx/)
    endif()
  endif()
  if(BUILD_NODEJS_IF)
    add_subdirectory(nodejs/)
  endif()
endif(BUILD_CPP_IF)

# uninstall target
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake" IMMEDIATE @ONLY)

add_custom_target(
  uninstall COMMAND ${CMAKE_COMMAND} -P
                    ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)

# lammps target
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_lammps.cmake.in"
               "${CMAKE_CURRENT_BINARY_DIR}/cmake_lammps.cmake" IMMEDIATE @ONLY)

add_custom_target(lammps COMMAND ${CMAKE_COMMAND} -P
                                 ${CMAKE_CURRENT_BINARY_DIR}/cmake_lammps.cmake)

# add configure file
if(BUILD_CPP_IF
   AND NOT BUILD_PY_IF
   AND NOT DEEPMD_C_ROOT)
  include(CMakePackageConfigHelpers)
  set(targets_export_name
      ${CMAKE_PROJECT_NAME}Targets
      CACHE INTERNAL "")
  set(generated_dir
      "${CMAKE_CURRENT_BINARY_DIR}/generated"
      CACHE INTERNAL "")
  set(cmake_files_install_dir
      "${CMAKE_INSTALL_PREFIX}/lib/cmake/${CMAKE_PROJECT_NAME}")
  set(version_file "${generated_dir}/${CMAKE_PROJECT_NAME}ConfigVersion.cmake")
  write_basic_package_version_file(
    ${version_file}
    VERSION $<IF:${GIT_SUMM}?${GIT_SUMM}:"0.0.0">
    COMPATIBILITY AnyNewerVersion)
  install(
    EXPORT ${targets_export_name}
    NAMESPACE ${CMAKE_PROJECT_NAME}::
    DESTINATION ${cmake_files_install_dir})
  set(config_file "${generated_dir}/${CMAKE_PROJECT_NAME}Config.cmake")
  configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake.in" "${config_file}"
    INSTALL_DESTINATION ${cmake_files_install_dir})
  install(FILES ${version_file} ${config_file}
          DESTINATION ${cmake_files_install_dir})
endif(
  BUILD_CPP_IF
  AND NOT BUILD_PY_IF
  AND NOT DEEPMD_C_ROOT)
