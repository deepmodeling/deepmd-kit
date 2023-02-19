# Input: ROCM_ROOT
#
# Output: ROCM_FOUND ROCM_INCLUDE_DIRS ROCM_LIBRARIES

# define the search path
cmake_minimum_required(VERSION 3.21)
include(CMakeDetermineHIPCompiler)
set(ROCM_PATH ${CMAKE_HIP_COMPILER_ROCM_ROOT})
set(ROCM_search_PATHS ${CMAKE_HIP_COMPILER_ROCM_ROOT})

# includes
find_path(
  ROCM_INCLUDE_DIRS
  NAMES hip/hip_runtime.h rocprim/rocprim.hpp hipcub/hipcub.hpp
  PATHS ${ROCM_search_PATHS}
  PATH_SUFFIXES "include"
  NO_DEFAULT_PATH)
if(NOT ROCM_INCLUDE_DIRS AND ROCM_FIND_REQUIRED)
  message(
    FATAL_ERROR
      "Not found 'hip' or 'rocprim' or 'hipcub' directory in path '${ROCM_search_PATHS}' "
      "You can manually set the ROCM install path by -DROCM_ROOT ")
endif()

# FindHIP.cmake
find_path(
  HIP_CMAKE
  NAMES FindHIP.cmake
  PATHS ${ROCM_search_PATHS}
  PATH_SUFFIXES "hip/cmake"
  NO_DEFAULT_PATH)

if(NOT HIP_CMAKE AND ROCM_FIND_REQUIRED)
  message(
    FATAL_ERROR "Not found 'FindHIP.cmake' file in path '${ROCM_search_PATHS}' "
                "You can manually set the ROCM install path by -DROCM_ROOT ")
endif()

list(APPEND CMAKE_MODULE_PATH ${HIP_CMAKE})
find_package(HIP)

# define the libs to find
if(NOT ROCM_FIND_COMPONENTS)
  if(HIP_VERSION VERSION_GREATER_EQUAL 3.5.1)
    set(ROCM_FIND_COMPONENTS amd_comgr amdhip64)
  else()
    set(ROCM_FIND_COMPONENTS hip-hcc hiprtc)
  endif()
endif()

# libs
foreach(module ${ROCM_FIND_COMPONENTS})
  find_library(
    ROCM_LIBRARIES_${module}
    NAMES ${module}
    PATHS ${ROCM_search_PATHS}
    PATH_SUFFIXES "lib"
    NO_DEFAULT_PATH)
  if(ROCM_LIBRARIES_${module})
    list(APPEND ROCM_LIBRARIES ${ROCM_LIBRARIES_${module}})
  elseif(ROCM_FIND_REQUIRED)
    message(
      FATAL_ERROR "Not found lib/'${module}' in '${ROCM_search_PATHS}' "
                  "You can manually set the ROCM install path by -DROCM_ROOT ")
  endif()
endforeach()

# define the output variable
if(ROCM_INCLUDE_DIRS
   AND ROCM_LIBRARIES
   AND HIP_CMAKE)
  set(ROCM_FOUND TRUE)
else()
  set(ROCM_FOUND FALSE)
endif()

# print message
if(NOT ROCM_FIND_QUIETLY)
  message(
    STATUS "Found ROCM: ${ROCM_INCLUDE_DIRS}, ${ROCM_LIBRARIES}, ${HIP_CMAKE}"
           " in ${ROCM_search_PATHS}, build AMD GPU support")
endif()

unset(ROCM_search_PATHS)
