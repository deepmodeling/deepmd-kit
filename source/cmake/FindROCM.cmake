# Input: ROCM_ROOT
#
# hip cmake path is added into CMAKE_MODULE_PATH

# define the search path
cmake_minimum_required(VERSION 3.21)
include(CMakeDetermineHIPCompiler)
set(ROCM_PATH ${CMAKE_HIP_COMPILER_ROCM_ROOT})
set(ROCM_search_PATHS ${CMAKE_HIP_COMPILER_ROCM_ROOT})

# FindHIP.cmake
find_path(
  HIP_CMAKE
  NAMES FindHIP.cmake
  PATHS ${ROCM_search_PATHS}
  # hip/cmake has been mirgrated to lib/cmake
  PATH_SUFFIXES "lib/cmake" "hip/cmake"
  NO_DEFAULT_PATH)

if(NOT HIP_CMAKE AND ROCM_FIND_REQUIRED)
  message(
    FATAL_ERROR "Not found 'FindHIP.cmake' file in path '${ROCM_search_PATHS}' "
                "You can manually set the ROCM install path by -DROCM_ROOT ")
endif()

list(APPEND CMAKE_MODULE_PATH ${HIP_CMAKE})
find_package(HIP)
find_package(hipCUB)

# print message
if(NOT ROCM_FIND_QUIETLY)
  message(STATUS "Found ROCM CMake Module ${HIP_CMAKE}"
                 " in ${ROCM_search_PATHS}, build AMD GPU support")
endif()

unset(ROCM_search_PATHS)
