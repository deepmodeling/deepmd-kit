# new in 3.16: GET_RUNTIME_DEPENDENCIES, target_precompile_headers
cmake_minimum_required(VERSION 3.16)
project(DeePMD)

option(ENABLE_TENSORFLOW "Enable TensorFlow interface" OFF)
option(ENABLE_PYTORCH "Enable PyTorch interface" OFF)
option(ENABLE_JAX "Enable JAX interface" OFF)
if(ENABLE_TENSORFLOW)
  # JAX requires TF C interface, contained in TF C++ library
  set(ENABLE_JAX ON)
endif()
option(ENABLE_PADDLE "Enable Paddle interface" OFF)
option(BUILD_TESTING "Build test and enable coverage" OFF)
set(DEEPMD_C_ROOT
    ""
    CACHE PATH "Path to imported DeePMD-kit C library")

if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 11)
endif()
macro(set_if_higher VARIABLE VALUE)
  # ${VARIABLE} is a variable name, not a string
  if(${VARIABLE} LESS "${VALUE}")
    set(${VARIABLE} ${VALUE})
  endif()
endmacro()
if(NOT DEEPMD_C_ROOT)
  # we can still allow C++ 11 for programs linked to the C library
  set_if_higher(CMAKE_CXX_STANDARD 14)
endif()

if(ENABLE_PADDLE)
  if(NOT DEFINED PADDLE_INFERENCE_DIR)
    # message(FATAL_ERROR "Make sure PADDLE_INFERENCE_DIR is set when
    # ENABLE_PADDLE=ON")
    if(USE_CUDA_TOOLKIT)
      find_package(CUDAToolkit REQUIRED)
      string(REGEX MATCH "^[0-9]+" CUDA_MAJOR_VERSION "${CUDAToolkit_VERSION}")
      message(STATUS "Find CUDAToolkit_VERSION: ${CUDAToolkit_VERSION}")
      # "6ed5dd3" is the commit id of paddle release/3.0, see:
      # https://github.com/PaddlePaddle/Paddle/tree/v3.0.0
      if(CUDA_MAJOR_VERSION VERSION_EQUAL "11")
        message(
          STATUS
            "PADDLE_INFERENCE_DIR is not defined, downloading CUDA11.8 inference lib to: ${CMAKE_BINARY_DIR}/"
        )
        set(DOWNLOAD_URL
            "https://paddle-qa.bj.bcebos.com/paddle-pipeline/GITHUB_Docker_Compile_Test_Cuda118_cudnn897_Trt8616_D1/6ed5dd3833c32c3b21e14b1fb1a71f5a535a0fcc/paddle_inference.tgz"
        )
      elseif(CUDA_MAJOR_VERSION VERSION_EQUAL "12")
        message(
          STATUS
            "PADDLE_INFERENCE_DIR is not defined, downloading CUDA12.6 inference lib to: ${CMAKE_BINARY_DIR}/"
        )
        set(DOWNLOAD_URL
            "https://paddle-qa.bj.bcebos.com/paddle-pipeline/GITHUB_Docker_Compile_Test_Cuda126_cudnn951_Trt105018_D1/6ed5dd3833c32c3b21e14b1fb1a71f5a535a0fcc/paddle_inference.tgz"
        )
      else()
        message(
          FATAL_ERROR
            "Paddle inference lib only support cuda 11 or 12, but your CUDA_MAJOR_VERSION is: ${CUDA_MAJOR_VERSION}"
        )
      endif()
    else()
      message(
        STATUS
          "PADDLE_INFERENCE_DIR is not defined, downloading CPU inference lib to: ${CMAKE_BINARY_DIR}/"
      )
      set(DOWNLOAD_URL
          "https://paddle-qa.bj.bcebos.com/paddle-pipeline/GITHUB_Docker_Compile_Test_Cpu_Mkl_Avx_D1/6ed5dd3833c32c3b21e14b1fb1a71f5a535a0fcc/paddle_inference.tgz"
      )
    endif()
    set(TGZ_FILE "${CMAKE_BINARY_DIR}/paddle_inference.tgz")
    set(EXTRACTED_DIR "${CMAKE_BINARY_DIR}/paddle_inference_install_dir")
    file(DOWNLOAD ${DOWNLOAD_URL} ${TGZ_FILE})
    message(STATUS "Downloading finished, extracting...")
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar -xzvf ${TGZ_FILE}
                    OUTPUT_QUIET)
    file(REMOVE ${TGZ_FILE})
    set(PADDLE_INFERENCE_DIR
        ${EXTRACTED_DIR}
        CACHE PATH
              "Path to 'paddle_inference_install_dir' or 'paddle_inference'")
  else()
    message(
      STATUS "PADDLE_INFERENCE_DIR is already defined: ${PADDLE_INFERENCE_DIR}")
  endif()

  link_directories(
    ${PADDLE_INFERENCE_DIR}/paddle/lib
    ${PADDLE_INFERENCE_DIR}/third_party/install/onednn/lib
    ${PADDLE_INFERENCE_DIR}/third_party/install/mklml/lib)

  # Check and append to existing CMAKE_BUILD_RPATH
  if(DEFINED CMAKE_BUILD_RPATH)
    list(APPEND CMAKE_BUILD_RPATH "${PADDLE_INFERENCE_DIR}/paddle/lib"
         "${PADDLE_INFERENCE_DIR}/third_party/install/onednn/lib"
         "${PADDLE_INFERENCE_DIR}/third_party/install/mklml/lib")
  else()
    set(CMAKE_BUILD_RPATH
        "${PADDLE_INFERENCE_DIR}/paddle/lib"
        "${PADDLE_INFERENCE_DIR}/third_party/install/onednn/lib"
        "${PADDLE_INFERENCE_DIR}/third_party/install/mklml/lib")
  endif()

  # Check and append to existing CMAKE_INSTALL_RPATH
  if(DEFINED CMAKE_INSTALL_RPATH)
    list(APPEND CMAKE_INSTALL_RPATH "${PADDLE_INFERENCE_DIR}/paddle/lib"
         "${PADDLE_INFERENCE_DIR}/third_party/install/onednn/lib"
         "${PADDLE_INFERENCE_DIR}/third_party/install/mklml/lib")
  else()
    set(CMAKE_INSTALL_RPATH
        "${PADDLE_INFERENCE_DIR}/paddle/lib"
        "${PADDLE_INFERENCE_DIR}/third_party/install/onednn/lib"
        "${PADDLE_INFERENCE_DIR}/third_party/install/mklml/lib")
  endif()

  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
  message(STATUS "Final PADDLE_INFERENCE_DIR is set to ${PADDLE_INFERENCE_DIR}")

  set(PADDLE_INFERENCE_DIR
      ${PADDLE_INFERENCE_DIR}
      CACHE PATH "Path to 'paddle_inference_install_dir' or 'paddle_inference'")

  # used in api_cc
  set(PADDLE_LIBRARIES
      "${PADDLE_INFERENCE_DIR}/paddle/lib/libpaddle_inference.so"
      CACHE PATH "Path to libpaddle_inference.so")

  include_directories("${PADDLE_INFERENCE_DIR}/")
  set(PADDLE_LIB_THIRD_PARTY_PATH
      "${PADDLE_INFERENCE_DIR}/third_party/install/")

  include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}protobuf/include")
  include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}glog/include")
  include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}gflags/include")
  include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}xxhash/include")
  list(APPEND BACKEND_INCLUDE_DIRS "${PADDLE_INFERENCE_DIR}/paddle/include")
  list(APPEND BACKEND_INCLUDE_DIRS
       "${PADDLE_LIB_THIRD_PARTY_PATH}protobuf/include")
  list(APPEND BACKEND_INCLUDE_DIRS "${PADDLE_LIB_THIRD_PARTY_PATH}glog/include")
  list(APPEND BACKEND_INCLUDE_DIRS
       "${PADDLE_LIB_THIRD_PARTY_PATH}gflags/include")
  list(APPEND BACKEND_INCLUDE_DIRS
       "${PADDLE_LIB_THIRD_PARTY_PATH}xxhash/include")

  link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}protobuf/lib")
  link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}glog/lib")
  link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}gflags/lib")
  link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}xxhash/lib")
  link_directories("${PADDLE_INFERENCE_DIR}/paddle/lib")
  list(APPEND BACKEND_LIBRARY_PATH "${PADDLE_INFERENCE_DIR}/paddle/lib")
  list(APPEND BACKEND_LIBRARY_PATH
       "${PADDLE_INFERENCE_DIR}/third_party/install/onednn/lib")
  list(APPEND BACKEND_LIBRARY_PATH
       "${PADDLE_INFERENCE_DIR}/third_party/install/mklml/lib")

  # if (USE_ROCM_TOOLKIT) add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1) endif()
endif(ENABLE_PADDLE)

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
  list(APPEND BACKEND_LIBRARY_PATH ${TensorFlow_LIBRARY_PATH})
  list(APPEND BACKEND_INCLUDE_DIRS ${TENSORFLOW_INCLUDE_DIRS})
endif()
if(BUILD_CPP_IF
   AND USE_PT_PYTHON_LIBS
   AND NOT CMAKE_CROSSCOMPILING
   AND NOT SKBUILD)
  find_package(
    Python
    COMPONENTS Interpreter
    REQUIRED)
  execute_process(
    COMMAND ${Python_EXECUTABLE} -c
            "import torch;print(torch.utils.cmake_prefix_path)"
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    OUTPUT_VARIABLE PYTORCH_CMAKE_PREFIX_PATH
    RESULT_VARIABLE PYTORCH_CMAKE_PREFIX_PATH_RESULT_VAR
    ERROR_VARIABLE PYTORCH_CMAKE_PREFIX_PATH_ERROR_VAR
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT ${PYTORCH_CMAKE_PREFIX_PATH_RESULT_VAR} EQUAL 0)
    message(
      FATAL_ERROR
        "Cannot determine PyTorch CMake prefix path, error code: $PYTORCH_CMAKE_PREFIX_PATH_RESULT_VAR}, error message: ${PYTORCH_CMAKE_PREFIX_PATH_ERROR_VAR}"
    )
  endif()
  list(APPEND CMAKE_PREFIX_PATH ${PYTORCH_CMAKE_PREFIX_PATH})
endif()
if(ENABLE_PYTORCH AND NOT DEEPMD_C_ROOT)
  find_package(Torch REQUIRED)
  if(NOT Torch_VERSION VERSION_LESS "2.1.0")
    set_if_higher(CMAKE_CXX_STANDARD 17)
  elseif(NOT Torch_VERSION VERSION_LESS "1.5.0")
    set_if_higher(CMAKE_CXX_STANDARD 14)
  endif()
  string(REGEX MATCH "_GLIBCXX_USE_CXX11_ABI=([0-9]+)" CXXABI_PT_MATCH
               "${TORCH_CXX_FLAGS}")
  if(CXXABI_PT_MATCH)
    set(OP_CXX_ABI_PT ${CMAKE_MATCH_1})
    message(STATUS "PyTorch CXX11 ABI: ${CMAKE_MATCH_1}")
    if(DEFINED OP_CXX_ABI)
      if(NOT ${CMAKE_MATCH_1} EQUAL ${OP_CXX_ABI})
        if(NOT BUILD_PY_IF)
          message(
            FATAL_ERROR
              "PyTorch CXX11 ABI mismatch TensorFlow: ${CMAKE_MATCH_1} != ${OP_CXX_ABI}"
          )
        else()
          if(NOT BUILD_CPP_IF)
            message(
              STATUS
                "PyTorch CXX11 ABI mismatch TensorFlow: ${CMAKE_MATCH_1} != ${OP_CXX_ABI}. "
                "Try to build libraries with both ABIs.")
          else()
            message(
              WARNING
                "PyTorch CXX11 ABI mismatch TensorFlow: ${CMAKE_MATCH_1} != ${OP_CXX_ABI}. "
                "PyTorch C++ OP will be built but PyTorch support for C++ libraries will be disabled. "
                "Note that we don't officially support building C++ libraries in the Python package, "
                "except for the wheels we officially release.")
          endif()
          set(DEEPMD_BUILD_COMPAT_CXXABI ON)
          set(OP_CXX_ABI_COMPAT ${OP_CXX_ABI_PT})
        endif()
      else()
        set(DEEPMD_BUILD_COMPAT_CXXABI OFF)
      endif()
    else()
      set(OP_CXX_ABI ${CMAKE_MATCH_1})
      add_definitions(-D_GLIBCXX_USE_CXX11_ABI=${OP_CXX_ABI})
    endif()
  else()
    # Maybe in macos/windows
    if(NOT DEFINED OP_CXX_ABI)
      set(OP_CXX_ABI 0)
    endif()
    set(OP_CXX_ABI_PT "${OP_CXX_ABI}")
  endif()
  # get torch directory get the directory of the target "torch"
  get_target_property(_TORCH_LOCATION torch LOCATION)
  get_filename_component(PyTorch_LIBRARY_PATH ${_TORCH_LOCATION} DIRECTORY)
  list(APPEND BACKEND_LIBRARY_PATH ${PyTorch_LIBRARY_PATH})
  list(APPEND BACKEND_INCLUDE_DIRS ${TORCH_INCLUDE_DIRS})
  if(USE_PT_PYTHON_LIBS OR BUILD_PY_IF)
    # when libtorch.so is in a wheel
    link_directories(${PyTorch_LIBRARY_PATH}/../../torch.libs)
  endif()
endif()
if(ENABLE_JAX
   AND BUILD_CPP_IF
   AND NOT DEEPMD_C_ROOT)
  # no way to find it using Python
  find_package(TensorFlowC REQUIRED MODULE)
  if(DEFINED TENSORFLOWC_LIBRARY)
    list(APPEND BACKEND_LIBRARY_PATH ${TENSORFLOWC_LIBRARY})
  endif()
  if(DEFINED TENSORFLOWC_INCLUDE_DIR)
    list(APPEND BACKEND_INCLUDE_DIRS ${TENSORFLOWC_INCLUDE_DIR})
  endif()
endif()
if(NOT DEFINED OP_CXX_ABI)
  # prevent setting an empty value; this is default on GCC>=5
  set(OP_CXX_ABI 1)
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
  if(ENABLE_JAX)
    message(STATUS "- JAX")
  endif()
  if(ENABLE_PADDLE)
    message(STATUS "- Paddle")
  endif()
  if(NOT ENABLE_TENSORFLOW
     AND NOT ENABLE_PYTORCH
     AND NOT ENABLE_JAX
     AND NOT ENABLE_PADDLE
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
# this definition doesn't work, but leaving it empty will cause error
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
  set(BACKEND_LIBRARY_PATH "${DEEPMD_C_ROOT}/lib")
  set(TENSORFLOW_INCLUDE_DIRS "${DEEPMD_C_ROOT}/include")
  set(BACKEND_INCLUDE_DIRS "${DEEPMD_C_ROOT}/include")
  set(TORCH_LIBRARIES "${DEEPMD_C_ROOT}/lib/libtorch.so")
endif()

if(NOT DEEPMD_C_ROOT)
  if(ENABLE_TENSORFLOW)
    add_subdirectory(op/tf/)
  endif()
  if(ENABLE_PYTORCH)
    add_subdirectory(op/pt/)
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
    VERSION $<IF:${GIT_SUMM}?${GIT_SUMM}:0.0.0>
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
