# new in 3.16: GET_RUNTIME_DEPENDENCIES, target_precompile_headers
cmake_minimum_required(VERSION 3.16)
project(DeePMD)

macro(safe_set_static_flag)
  foreach(flag_var
          CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
          CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
    if(${flag_var} MATCHES "/MD")
      string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
    endif(${flag_var} MATCHES "/MD")
  endforeach(flag_var)
endmacro()

if(NOT DEFINED PADDLE_LIB)
  message(
    WARNING
      "Optional arg: 'PADDLE_LIB' is not set, skip all compilation of paddle code now. "
      "And do not forget to set 'PADDLE_LIB' with '-DPADDLE_LIB=/path/paddle/lib' before "
      "using paddle custom operators")
endif()

if(DEFINED PADDLE_LIB)
  set(PADDLE_LIB
      ${PADDLE_LIB}
      CACHE PATH "/path/paddle/lib")

  include_directories("${PADDLE_LIB}/")
  set(PADDLE_LIB_THIRD_PARTY_PATH "${PADDLE_LIB}/third_party/install/")

  include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}protobuf/include")
  include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}glog/include")
  include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}gflags/include")
  include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}xxhash/include")

  link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}protobuf/lib")
  link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}glog/lib")
  link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}gflags/lib")
  link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}xxhash/lib")
  link_directories("${PADDLE_LIB}/paddle/lib")
endif()

# add custom operators
option(USE_TENSORRT "Compile demo with TensorRT." OFF)

if(WITH_GPU)
  if(NOT WIN32)
    set(CUDA_LIB
        "/usr/local/cuda/lib64/"
        CACHE STRING "CUDA Library")
  else()
    if(CUDA_LIB STREQUAL "")
      set(CUDA_LIB
          "C:\\Program\ Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\lib\\x64"
      )
    endif()
  endif(NOT WIN32)
endif()

if(NOT WIN32)
  if(USE_TENSORRT AND WITH_GPU)
    include_directories("${TENSORRT_INCLUDE_DIR}")
    link_directories("${TENSORRT_LIB_DIR}")
  endif()
endif(NOT WIN32)

if(DEFINED PADDLE_LIB)
  if(WITH_STATIC_LIB)
    set(DEPS
        ${PADDLE_LIB}/paddle/lib/libpaddle_inference${CMAKE_STATIC_LIBRARY_SUFFIX}
    )
  else()
    if(WIN32)
      set(DEPS
          ${PADDLE_LIB}/paddle/lib/libpaddle_inference${CMAKE_STATIC_LIBRARY_SUFFIX}
      )
    else()
      set(DEPS
          ${PADDLE_LIB}/paddle/lib/libpaddle_inference${CMAKE_SHARED_LIBRARY_SUFFIX}
      )
    endif()
  endif()
endif()

if(NOT WIN32)
  set(EXTERNAL_LIB "-lrt -ldl -lpthread")
  set(DEPS
      ${DEPS}
      ${MATH_LIB}
      ${MKLDNN_LIB}
      glog
      gflags
      protobuf
      xxhash
      ${EXTERNAL_LIB})
else()
  set(DEPS
      ${DEPS}
      ${MATH_LIB}
      ${MKLDNN_LIB}
      glog
      gflags_static
      libprotobuf
      xxhash
      ${EXTERNAL_LIB})
  set(DEPS ${DEPS} shlwapi.lib)
endif(NOT WIN32)

if(WITH_GPU)
  if(NOT WIN32)
    if(USE_TENSORRT)
      set(DEPS ${DEPS}
               ${TENSORRT_LIB_DIR}/libnvinfer${CMAKE_SHARED_LIBRARY_SUFFIX})
      set(DEPS
          ${DEPS}
          ${TENSORRT_LIB_DIR}/libnvinfer_plugin${CMAKE_SHARED_LIBRARY_SUFFIX})
    endif()
    set(DEPS ${DEPS} ${CUDA_LIB}/libcudart${CMAKE_SHARED_LIBRARY_SUFFIX})
  else()
    if(USE_TENSORRT)
      set(DEPS ${DEPS}
               ${TENSORRT_LIB_DIR}/nvinfer${CMAKE_STATIC_LIBRARY_SUFFIX})
      set(DEPS ${DEPS}
               ${TENSORRT_LIB_DIR}/nvinfer_plugin${CMAKE_STATIC_LIBRARY_SUFFIX})
    endif()
    set(DEPS ${DEPS} ${CUDA_LIB}/cudart${CMAKE_STATIC_LIBRARY_SUFFIX})
    set(DEPS ${DEPS} ${CUDA_LIB}/cublas${CMAKE_STATIC_LIBRARY_SUFFIX})
    set(DEPS ${DEPS} ${CUDA_LIB}/cudnn${CMAKE_STATIC_LIBRARY_SUFFIX})
  endif()
endif()

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
  if(NOT INSTALL_TENSORFLOW)
    # some errors in conda packages...
    find_package(GTest)
  endif()
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
  set(CUDA_USE_STATIC_CUDA_RUNTIME
      OFF
      CACHE INTERNAL "")
  find_package(CUDA REQUIRED)
  add_definitions("-DGOOGLE_CUDA")
  message(STATUS "Found CUDA in ${CUDA_TOOLKIT_ROOT_DIR}, build nv GPU support")
  set(DP_VARIANT "cuda")
else()
  message(STATUS "Will not build nv GPU support")
endif(USE_CUDA_TOOLKIT)

# define USE_ROCM_TOOLKIT
if(USE_ROCM_TOOLKIT)
  find_package(ROCM REQUIRED)
  add_definitions("-DTENSORFLOW_USE_ROCM")
  add_compile_definitions(__HIP_PLATFORM_HCC__)
  message(STATUS "Found ROCM in ${ROCM_ROOT}, build AMD GPU support")
  set(DP_VARIANT "rocm")
else()
  message(STATUS "Will not build AMD GPU support")
endif(USE_ROCM_TOOLKIT)

set(DEEPMD_SOURCE_DIR ${PROJECT_SOURCE_DIR}/..)

# setup tensorflow libraries by python
if(USE_TF_PYTHON_LIBS)
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

# find tensorflow, I need tf abi info
if(NOT DEEPMD_C_ROOT)
  find_package(tensorflow REQUIRED)
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
    set(LIB_DEEPMD_OP_DEVICE "deepmd_paddle_op_cuda")
    set(LIB_DEEPMD_OP_DEVICE "deepmd_op_cuda")
  elseif(USE_ROCM_TOOLKIT)
    set(LIB_DEEPMD_OP_DEVICE "deepmd_op_rocm")
  else()
    set(LIB_DEEPMD_OP_DEVICE "deepmd_op")
  endif()
  if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 4.8)
    set(LIB_DEEPMD_NATIVE "deepmd_native_md")
    # set(LIB_DEEPMD_IPI "deepmd_ipi") set(LIB_DEEPMD_GROMACS "deepmd_gromacs")
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
  set(TensorFlow_LIBRARY_PATH "${DEEPMD_C_ROOT}/lib")
  set(TENSORFLOW_INCLUDE_DIRS "${DEEPMD_C_ROOT}/include")
endif()

if(NOT DEEPMD_C_ROOT)
  # add_subdirectory(op/)
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
    # add_subdirectory (md/) if(ENABLE_IPI OR NOT BUILD_PY_IF AND NOT
    # DEEPMD_C_ROOT) # ipi has a dependency on libdeepmd add_subdirectory(ipi/)
    # endif() if(NOT BUILD_PY_IF) add_subdirectory(gmx/) endif()
  endif()
  if(BUILD_NODEJS_IF)
    add_subdirectory(nodejs/)
  endif()
endif(BUILD_CPP_IF)

# if(WIN32) if(USE_TENSORRT) add_custom_command(TARGET ${DEMO_NAME} POST_BUILD
# COMMAND ${CMAKE_COMMAND} -E copy
# ${TENSORRT_LIB_DIR}/nvinfer${CMAKE_SHARED_LIBRARY_SUFFIX}
# ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE} COMMAND ${CMAKE_COMMAND} -E copy
# ${TENSORRT_LIB_DIR}/nvinfer_plugin${CMAKE_SHARED_LIBRARY_SUFFIX}
# ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE} ) endif() if(WITH_MKL)
# add_custom_command(TARGET ${DEMO_NAME} POST_BUILD COMMAND ${CMAKE_COMMAND} -E
# copy ${MATH_LIB_PATH}/lib/mklml.dll ${CMAKE_BINARY_DIR}/Release COMMAND
# ${CMAKE_COMMAND} -E copy ${MATH_LIB_PATH}/lib/libiomp5md.dll
# ${CMAKE_BINARY_DIR}/Release COMMAND ${CMAKE_COMMAND} -E copy
# ${MKLDNN_PATH}/lib/mkldnn.dll  ${CMAKE_BINARY_DIR}/Release ) else()
# add_custom_command(TARGET ${DEMO_NAME} POST_BUILD COMMAND ${CMAKE_COMMAND} -E
# copy ${OPENBLAS_LIB_PATH}/lib/openblas.dll ${CMAKE_BINARY_DIR}/Release )
# endif() if(NOT WITH_STATIC_LIB) add_custom_command(TARGET ${DEMO_NAME}
# POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy
# "${PADDLE_LIB}/paddle/lib/paddle_fluid.dll"
# ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE} ) endif() endif()

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
