cmake_minimum_required(VERSION 3.7)
project(DeePMD)
set(CMAKE_LINK_WHAT_YOU_USE TRUE)

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
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wno-ignored-attributes")

# model version
file(READ ${PROJECT_SOURCE_DIR}/config/MODEL_VER MODEL_VERSION)
string(REPLACE "\n" " " MODEL_VERSION ${MODEL_VERSION})
message(STATUS "Supported model version: ${MODEL_VERSION}")

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
endif()

# find tensorflow, I need tf abi info
find_package(tensorflow REQUIRED)

# find threads
find_package(Threads)

# auto op_cxx_abi
if (NOT DEFINED OP_CXX_ABI)
  if (BUILD_PY_IF) 
    if (DEFINED TENSORFLOW_ROOT)
      set(FIND_ABI_CMD "import sys,os; sys.path.insert(0, os.path.join('${TENSORFLOW_ROOT}', '..')); import tensorflow; print(tensorflow.CXX11_ABI_FLAG if 'CXX11_ABI_FLAG' in tensorflow.__dict__ else tensorflow.sysconfig.CXX11_ABI_FLAG, end = '')" )
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

# define USE_TTM
if (NOT DEFINED USE_TTM)
  set(USE_TTM FALSE)
endif (NOT DEFINED USE_TTM)
if (USE_TTM)
  message(STATUS "Use TTM")
  set(TTM_DEF "-DUSE_TTM")
endif (USE_TTM)

# old pppm interface
if(NOT DEFINED OLD_LMP_PPPM)
  set(OLD_LMP_PPPM FALSE)
endif(NOT DEFINED OLD_LMP_PPPM)
if (OLD_LMP_PPPM)
  set(OLD_LMP_PPPM_DEF "-DOLD_LMP_PPPM")
  message(STATUS "Use old lammps pppm interface")
endif()
add_definitions (${OLD_LMP_PPPM_DEF})

# define build type
if ((NOT DEFINED CMAKE_BUILD_TYPE) OR CMAKE_BUILD_TYPE STREQUAL "")
   set (CMAKE_BUILD_TYPE release)
endif ()

# set op prec
if (DEFINED FLOAT_PREC)
  string ( TOLOWER ${FLOAT_PREC} lower_float_prec )
  if (lower_float_prec STREQUAL "high")
    set(PREC_DEF "-DHIGH_PREC")
  else ()
    set(PREC_DEF "")    
  endif ()
else ()
    set(PREC_DEF "-DHIGH_PREC")
endif()
add_definitions (${PREC_DEF})

# find openmp
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# headers of lib
list (APPEND DeePMD_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/lib/include/)
list (APPEND DeePMD_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/3rdparty/)
if (BUILD_CPP_IF)
  list (APPEND DeePMD_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/api_cc/include/)
  include_directories(${CMAKE_BINARY_DIR}/api_cc/)
endif (BUILD_CPP_IF)

# include
include_directories(${DeePMD_INCLUDE_DIRS})
include_directories(${TensorFlow_INCLUDE_DIRS})

# define names of libs
set (LIB_DEEPMD		"deepmd")
if (BUILD_CPP_IF)
  set (LIB_DEEPMD_OP		"deepmd_op")
  set (LIB_DEEPMD_CC		"deepmd_cc")
  if (USE_CUDA_TOOLKIT)
    set (LIB_DEEPMD_OP_CUDA		"deepmd_op_cuda")
  else ()
    set (LIB_DEEPMD_OP_CUDA		"deepmd_op")
  endif()
  if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 4.9)
    set (LIB_DEEPMD_NATIVE	"deepmd_native_md")
    set (LIB_DEEPMD_IPI		"deepmd_ipi")
  else ()
    message (STATUS "Your gcc/g++ version is ${CMAKE_CXX_COMPILER_VERSION}, so native MD and ipi are disabled. To enable them, use gcc/g++ >= 4.9.")
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
  if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 4.9)
#   add_subdirectory (md/)
    add_subdirectory (ipi/)
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

