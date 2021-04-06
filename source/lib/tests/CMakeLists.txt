cmake_minimum_required(VERSION 3.9)
project(libdeepmd_test)

enable_testing()

set(libname "deepmd")
set(LIB_BASE_DIR ${CMAKE_SOURCE_DIR}/../)

include_directories(${LIB_BASE_DIR}/include)
file(GLOB LIB_SRC ${LIB_BASE_DIR}/src/*.cc ${LIB_BASE_DIR}/src/*.cpp)
message(status ${LIB_SRC})
# add_library(${libname} SHARED ${LIB_SRC})
add_library(${libname} ${LIB_SRC})

message(status "${CMAKE_SOURCE_DIR}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

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
  include_directories(${CUDA_INCLUDE_DIRS})
  add_subdirectory(${LIB_BASE_DIR}/src/cuda cuda_binary_dir)
endif()

file(GLOB TEST_SRC test_*.cc)
add_executable( runUnitTests ${TEST_SRC} )

find_package(Threads)
# find openmp
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

if (USE_CUDA_TOOLKIT)
  target_link_libraries(runUnitTests gtest gtest_main ${libname} pthread deepmd_op_cuda)
else()
  target_link_libraries(runUnitTests gtest gtest_main ${libname} pthread)
endif()
add_test( runUnitTests runUnitTests )

# include(GoogleTest)
# add_executable(FooTest tests/test_simulation_region.cc)
# gtest_add_tests(TARGET      FooTest
#                 TEST_SUFFIX .noArgs
#                 TEST_LIST   noArgsTests
# )

find_package(GTest)
if(NOT GTEST_LIBRARY)
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
