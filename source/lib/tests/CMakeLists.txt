cmake_minimum_required(VERSION 3.9)
project(libdeepmd_test)

enable_testing()

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(libname "deepmd")
set(BUILD_CPP_IF TRUE)
set(LIB_DEEPMD ${libname})
add_subdirectory(${CMAKE_SOURCE_DIR}/.. lib)

message(status "${CMAKE_SOURCE_DIR}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

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
  list (APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../../cmake/)
  find_package(ROCM REQUIRED)
  add_definitions("-DTENSORFLOW_USE_ROCM")
  add_compile_definitions(__HIP_PLATFORM_HCC__)
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

find_package(Threads)
# find openmp
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

target_link_libraries(runUnitTests gtest gtest_main ${libname} coverage_config)
add_test( runUnitTests runUnitTests )

set_target_properties(
  runUnitTests
  PROPERTIES 
  INSTALL_RPATH "$ORIGIN/../lib"
)

# include(GoogleTest)
# add_executable(FooTest tests/test_simulation_region.cc)
# gtest_add_tests(TARGET      FooTest
#                 TEST_SUFFIX .noArgs
#                 TEST_LIST   noArgsTests
# )

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

install(TARGETS runUnitTests DESTINATION bin/)

