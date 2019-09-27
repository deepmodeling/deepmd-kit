cmake_minimum_required(VERSION 3.0)
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
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wno-ignored-attributes")

# find tensorflow, I need tf abi info
find_package(tensorflow REQUIRED)

# auto op_cxx_abi
if (NOT DEFINED OP_CXX_ABI)
  if (BUILD_PY_IF) 
    execute_process(
      COMMAND ${PYTHON_EXECUTABLE} "-c" "import tensorflow; print(tensorflow.CXX11_ABI_FLAG, end = '')"
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
      OUTPUT_VARIABLE PY_CXX_ABI
      )
    set(OP_CXX_ABI ${PY_CXX_ABI})
  endif()
  if (BUILD_CPP_IF)
    try_run(
      RUN_RES COMPILE_VAR
      ${CMAKE_CURRENT_BINARY_DIR}/tf_cxx_abi
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/tf_cxx_abi.cpp"
      LINK_LIBRARIES ${TensorFlowFramework_LIBRARY}
      CMAKE_FLAGS "-DINCLUDE_DIRECTORIES:STRING=${TensorFlow_INCLUDE_DIRS}"
      RUN_OUTPUT_VARIABLE CPP_CXX_ABI
      COMPILE_OUTPUT_VARIABLE OUTPUT
      )
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

# define USE_CUDA_TOOLKIT
if (USE_CUDA_TOOLKIT)
  add_definitions("-DUSE_CUDA_TOOLKIT")
endif()

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

# include
include_directories(${DeePMD_INCLUDE_DIRS})
include_directories(${TensorFlow_INCLUDE_DIRS})

# define names of libs
if (BUILD_CPP_IF)
  set (LIB_DEEPMD		"deepmd")
  set (LIB_DEEPMD_OP		"deepmd_op")
  if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 4.9)
    set (LIB_DEEPMD_NATIVE	"deepmd_native_md")
    set (LIB_DEEPMD_IPI		"deepmd_ipi")
  else ()
    message (STATUS "Your gcc/g++ version is ${CMAKE_CXX_COMPILER_VERSION}, so native MD and ipi are disabled. To enable them, use gcc/g++ >= 4.9.")
  endif ()
endif (BUILD_CPP_IF)

include_directories(${CMAKE_BINARY_DIR}/lib/)
add_subdirectory (op/)
if (BUILD_PY_IF)
  add_subdirectory (train/)
  add_subdirectory (scripts/)
  add_subdirectory (tests/)
endif (BUILD_PY_IF)
if (BUILD_CPP_IF) 
  add_subdirectory (lib/)
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

