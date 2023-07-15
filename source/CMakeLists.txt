cmake_minimum_required(VERSION 3.12)
project(DeePMD)

# build cpp or python interfaces
if (NOT DEFINED BUILD_CPP_IF) 
  set(BUILD_CPP_IF TRUE)
endif (NOT DEFINED BUILD_CPP_IF)
if (NOT DEFINED BUILD_PY_IF) 
  set(BUILD_PY_IF FALSE)
endif (NOT DEFINED BUILD_PY_IF)
if ((NOT BUILD_PY_IF) AND (NOT BUILD_CPP_IF))
  # nothing to do
  message(FATAL_ERROR "Nothing to do.")
endif()

find_package(Git)
if(GIT_FOUND)
  execute_process(
    COMMAND git describe --tags --dirty
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_SUMM
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )
  execute_process(
    COMMAND git log -1 --format=%h
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )
  execute_process(
    COMMAND git rev-parse --abbrev-ref HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_BRANCH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )
  execute_process(
    COMMAND git show -s --format=%ci ${GIT_HASH}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_DATE
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )
endif(GIT_FOUND)

# global defines
list (APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/)

# model version
file(READ ${PROJECT_SOURCE_DIR}/config/MODEL_VER MODEL_VERSION)
string(REPLACE "\n" " " MODEL_VERSION ${MODEL_VERSION})
message(STATUS "Supported model version: ${MODEL_VERSION}")

# Devices that have both ROCM and CUDA are not currently supported
if (USE_ROCM_TOOLKIT AND USE_CUDA_TOOLKIT)
  message (FATAL_ERROR "Devices that have both ROCM and CUDA are not currently supported")
endif()
set(DP_VARIANT "cpu")

# define USE_CUDA_TOOLKIT
if (USE_CUDA_TOOLKIT)
  set(CUDA_USE_STATIC_CUDA_RUNTIME OFF CACHE INTERNAL "")
  find_package(CUDA REQUIRED)
  add_definitions("-DGOOGLE_CUDA")
  message(STATUS "Found CUDA in ${CUDA_TOOLKIT_ROOT_DIR}, build nv GPU support")
  set(DP_VARIANT "cuda")
else()
  message(STATUS "Will not build nv GPU support")
endif(USE_CUDA_TOOLKIT)

#define USE_ROCM_TOOLKIT
if (USE_ROCM_TOOLKIT)
  find_package(ROCM REQUIRED)
  add_definitions("-DTENSORFLOW_USE_ROCM")
  add_compile_definitions(__HIP_PLATFORM_HCC__)
  message(STATUS "Found ROCM in ${ROCM_ROOT}, build AMD GPU support")
  set(DP_VARIANT "rocm")
else()
  message(STATUS "Will not build AMD GPU support")  
endif (USE_ROCM_TOOLKIT)

# setup tensorflow libraries by python
if (USE_TF_PYTHON_LIBS)
  find_package (Python COMPONENTS Interpreter Development REQUIRED)
endif(USE_TF_PYTHON_LIBS)

#define USE_ASCEND, find tensorflow abi info
if (USE_ASCEND)
  add_definitions("-DHUAWEI_ASCEND")
  set(DP_VARIANT "ascend")
endif(USE_ASCEND)
if (USE_ASCEND AND BUILD_CPP_IF)
  if (NOT DEFINED ${PYTHON_EXECUTABLE})
    set(PYTHON_EXECUTABLE "python3.7")
  endif()
  set(FIND_TENSORFLOW_INC "import tensorflow as tf; print(tf.sysconfig.get_compile_flags()[0],end='')")
  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} "-c" "${FIND_TENSORFLOW_INC}"
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    OUTPUT_VARIABLE OUTPUT_VAR
    RESULT_VARIABLE RESULT_VAR
    ERROR_VARIABLE ERROR_VAR
  )
  string(REPLACE "-I" "" TensorFlow_INCLUDE_DIRS ${OUTPUT_VAR})
  if (NOT ${RESULT_VAR} EQUAL 0)
    message(FATAL_ERROR "Cannot determine tensorflow include dir, error message: ${ERROR_VAR}")
  endif()

  set(FIND_TENSORFLOW_LINK "import tensorflow as tf; print(tf.sysconfig.get_link_flags()[0],end='')")
  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} "-c" "${FIND_TENSORFLOW_LINK}"
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    OUTPUT_VARIABLE OUTPUT_VAR
    RESULT_VARIABLE RESULT_VAR
    ERROR_VARIABLE ERROR_VAR
  )
  string(REPLACE "-L" "" TensorFlow_LIBRARY_PATH ${OUTPUT_VAR})
  set (TensorFlowFramework_LIBRARY "${TensorFlow_LIBRARY_PATH}/libtensorflow_framework.so.1")
  set (TensorFlow_LIBRARY "${TensorFlow_LIBRARY_PATH}/python/_pywrap_tensorflow_internal.so")
  if (NOT ${RESULT_VAR} EQUAL 0)
    message(FATAL_ERROR "Cannot determine tensorflow link dir, error message: ${ERROR_VAR}")
  endif()

  find_package(Python REQUIRED COMPONENTS Interpreter Development)
else()
  message(STATUS "Will not build Ascend NPU support")
  # find tensorflow, I need tf abi info
  find_package(tensorflow REQUIRED)
endif(USE_ASCEND AND BUILD_CPP_IF)

set(DEEPMD_SOURCE_DIR ${PROJECT_SOURCE_DIR}/..)

if (TENSORFLOW_VERSION VERSION_GREATER_EQUAL 2.10)
  set (CMAKE_CXX_STANDARD 17)
elseif (TENSORFLOW_VERSION VERSION_GREATER_EQUAL 2.7)
  set (CMAKE_CXX_STANDARD 14)
else()
  set (CMAKE_CXX_STANDARD 11)
endif()

if (MSVC)
  # see TF .bazelrc
  add_compile_options(/W0 /Zc:__cplusplus /D_USE_MATH_DEFINES /d2ReducedOptimizeHugeFunctions)
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
endif()
if (TENSORFLOW_VERSION VERSION_GREATER_EQUAL 2.4 AND MSVC)
  # see TF 2.4 release notes
  add_compile_options(/Zc:preprocessor)
endif()

# find threads
find_package(Threads)

# auto op_cxx_abi
if (NOT DEFINED OP_CXX_ABI)
  if (BUILD_PY_IF) 
    if (DEFINED TENSORFLOW_ROOT)
      # note \ in windows path
      set(FIND_ABI_CMD "import sys,os; sys.path.insert(0, os.path.join(r'${TENSORFLOW_ROOT}', '..')); import tensorflow; print(tensorflow.CXX11_ABI_FLAG if 'CXX11_ABI_FLAG' in tensorflow.__dict__ else tensorflow.sysconfig.CXX11_ABI_FLAG, end = '')" )
    else()
      set(FIND_ABI_CMD "import tensorflow; print(tensorflow.CXX11_ABI_FLAG if 'CXX11_ABI_FLAG' in tensorflow.__dict__ else tensorflow.sysconfig.CXX11_ABI_FLAG, end = '')")
    endif()
    execute_process(
      COMMAND ${PYTHON_EXECUTABLE} "-c" "${FIND_ABI_CMD}"
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
      OUTPUT_VARIABLE PY_CXX_ABI
      RESULT_VARIABLE PY_CXX_ABI_RESULT_VAR
      ERROR_VARIABLE PY_CXX_ABI_ERROR_VAR
      )
    if (NOT ${PY_CXX_ABI_RESULT_VAR} EQUAL 0)
      message(FATAL_ERROR "Cannot determine cxx abi, error message: ${PY_CXX_ABI_ERROR_VAR}")
    endif()
    set(OP_CXX_ABI ${PY_CXX_ABI})
  endif()
  if (BUILD_CPP_IF)
    try_run(
      CPP_CXX_ABI_RUN_RESULT_VAR CPP_CXX_ABI_COMPILE_RESULT_VAR
      ${CMAKE_CURRENT_BINARY_DIR}/tf_cxx_abi
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/tf_cxx_abi.cpp"
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
  endif()
  message (STATUS "Automatically determined OP_CXX_ABI=${OP_CXX_ABI} ")
else()
  message (STATUS "User set OP_CXX_ABI=${OP_CXX_ABI} ")  
endif()    
# message the cxx_abi used during compiling
if (${OP_CXX_ABI} EQUAL 0) 
  message (STATUS "Set GLIBCXX_USE_CXX_ABI=0 when compiling ops")
else ()
  set (OP_CXX_ABI 1)
  message (STATUS "Set GLIBCXX_USE_CXX_ABI=1 when compiling ops")
endif ()

# set _GLIBCXX_USE_CXX11_ABI flag globally
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=${OP_CXX_ABI})

# define USE_TTM
if (NOT DEFINED USE_TTM)
  set(USE_TTM FALSE)
endif (NOT DEFINED USE_TTM)
if (USE_TTM)
  message(STATUS "Use TTM")
  set(TTM_DEF "-DUSE_TTM")
endif (USE_TTM)

# define build type
if ((NOT DEFINED CMAKE_BUILD_TYPE) OR CMAKE_BUILD_TYPE STREQUAL "")
   set (CMAKE_BUILD_TYPE release)
endif ()

# set op prec
set(HIGH_PREC_DEF "HIGH_PREC")
# this defination doesn't work, but leaving it empty will cause error
set(LOW_PREC_DEF "LOW_PREC")
set(HIGH_PREC_VARIANT "")
set(LOW_PREC_VARIANT "_low")

# find openmp
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# define names of libs
set (LIB_DEEPMD		"deepmd")
if (BUILD_CPP_IF)
  set (LIB_DEEPMD_OP		"deepmd_op")
  set (LIB_DEEPMD_CC		"deepmd_cc")
  if (USE_CUDA_TOOLKIT)
    set (LIB_DEEPMD_OP_DEVICE		"deepmd_op_cuda")
  elseif (USE_ROCM_TOOLKIT)
    set (LIB_DEEPMD_OP_DEVICE   "deepmd_op_rocm")
  else()
    set (LIB_DEEPMD_OP_DEVICE		"deepmd_op")
  endif()
  if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 4.8)
    set (LIB_DEEPMD_NATIVE	"deepmd_native_md")
    set (LIB_DEEPMD_IPI		  "deepmd_ipi")
    set (LIB_DEEPMD_GROMACS "deepmd_gromacs")
  else ()
    message (STATUS "Your gcc/g++ version is ${CMAKE_CXX_COMPILER_VERSION}, so native MD, ipi and gromacs plugin are disabled. To enable them, use gcc/g++ >= 4.8.")
  endif ()
endif (BUILD_CPP_IF)

add_subdirectory (op/)
add_subdirectory (lib/)
if (BUILD_PY_IF)
  add_subdirectory (config/)
  # add_subdirectory (tests/)
endif (BUILD_PY_IF)
if (BUILD_CPP_IF) 
  add_subdirectory (api_cc/)
  add_subdirectory (lmp/)
  if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 4.8)
    # add_subdirectory (md/)
    add_subdirectory (ipi/)
    add_subdirectory (gmx/)
  endif ()
endif (BUILD_CPP_IF)

# uninstall target
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
    IMMEDIATE @ONLY)

add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)

# lammps target
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_lammps.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake_lammps.cmake"
    IMMEDIATE @ONLY)

add_custom_target(lammps
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_lammps.cmake)

