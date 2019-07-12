cmake_minimum_required(VERSION 3.0)
project(DeePMD)

# build cpp or python interfaces
if (NOT DEFINED BUILD_CPP_IF) 
  set(BUILD_CPP_IF TRUE)
endif (NOT DEFINED BUILD_CPP_IF)
if (NOT DEFINED BUILD_PY_IF) 
  set(BUILD_PY_IF FALSE)
endif (NOT DEFINED BUILD_PY_IF)

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
list (APPEND CMAKE_CXX_FLAGS "-std=c++11 -Wno-ignored-attributes")

# define the abi
if (NOT DEFINED TF_GOOGLE_BIN) 
  set (TF_GOOGLE_BIN FALSE)
endif()
if (TF_GOOGLE_BIN) 
  set (OP_ABI 0)
  message (STATUS "Assuming google tensorflow binary, use abi = 0")
else ()
  set (OP_ABI 1)
  message (STATUS "Not assuming google tensorflow binary, use abi = 1")
endif ()

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

# find tensorflow
find_package(tensorflow REQUIRED)

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
  set (LIB_DEEPMD_NATIVE	"deepmd_native_md")
  set (LIB_DEEPMD_IPI		"deepmd_ipi")
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
# add_subdirectory (md/)
  add_subdirectory (ipi/)
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

