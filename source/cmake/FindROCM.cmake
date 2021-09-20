# Input:
# ROCM_ROOT 
#
# Output:
# ROCM_FOUND        
# ROCM_INCLUDE_DIRS 
# ROCM_LIBRARIES


if(ROCM_DTK AND ROCM_TWO)
  message(FATAL_ERROR "CANNOT SELECT TWO VERSIONS OF ROCM AT THA SAME TIME")
endif()

if(NOT ROCM_DTK AND ROCM_TWO)
  message(FATAL_ERROR "PLEASE USE -DROCM_DTK=TRUE OR -DROCM_TWO TO TELL ME THE VERSION OF ROCM")
endif()

# define the search path
if(ROCM_ROOT)
  list(APPEND ROCM_SEARCH_PATHS ${ROCM_ROOT})
else()
  if(ROCM_DTK)
    list(APPEND ROCM_SEARCH_PATHS "/public/software/compiler/dtk/dtk-21.04/")
    list(APPEND ROCM_SEARCH_PATHS "/public/software/compiler/rocm/dtk-21.04/")
  elseif(ROCM_TWO)
    list(APPEND ROCM_SEARCH_PATHS "/opt/rocm")
  endif(ROCM_DTK)
endif(ROCM_ROOT)

# define the libs to find
if (NOT ROCM_FIND_COMPONENTS and ROCM_DTK)
  set(ROCM_FIND_COMPONENTS amd_comgr amdhip64)
elseif(NOT ROCM_FIND_COMPONENTS and ROCM_TWO)
  set(ROCM_FIND_COMPONENTS hip_hcc hiprtc)
endif ()

# includes
find_path (ROCM_INCLUDE_DIRS
  NAMES 
  hip/hip_runtime.h
  rocprim/rocprim.hpp
  hipcub/hipcub.hpp
  PATHS ${ROCM_SEARCH_PATHS} 
  PATH_SUFFIXES "include"
  NO_DEFAULT_PATH
  )
if (NOT ROCM_INCLUDE_DIRS AND ROCM_FIND_REQUIRED)
  message(FATAL_ERROR 
    "Not found 'hip' or 'rocprim' or 'hipcub' directory in path '${ROCM_search_PATHS}' "
    "You can manually set the ROCM install path by -DROCM_ROOT ")
endif ()

# libs
foreach (module ${ROCM_FIND_COMPONENTS})
  find_library(ROCM_LIBRARIES_${module}
    NAMES ${module}
    PATHS ${ROCM_search_PATHS} PATH_SUFFIXES "lib64" "lib" NO_DEFAULT_PATH
    )
  if (ROCM_LIBRARIES_${module})
    list(APPEND ROCM_LIBRARIES ${ROCM_LIBRARIES_${module}})
  elseif (ROCM_FIND_REQUIRED)
    message(FATAL_ERROR 
      "Not found lib/'${module}' in '${ROCM_search_PATHS}' "
      "You can manually set the ROCM install path by -DROCM_ROOT ")
  endif ()
endforeach ()

# FindHIP.cmake
find_path (HIP_CMAKE
  NAMES 
  FindHIP.cmake
  PATHS ${ROCM_search_PATHS} 
  PATH_SUFFIXES "hip/cmake"
  NO_DEFAULT_PATH
  )

if (NOT HIP_CMAKE AND ROCM_FIND_REQUIRED)
  message(FATAL_ERROR 
    "Not found 'FindHIP.cmake' file in path '${ROCM_search_PATHS}' "
    "You can manually set the ROCM install path by -DROCM_ROOT ")
endif ()

list (APPEND CMAKE_MODULE_PATH ${HIP_CMAKE})
find_package(HIP) 

# define the output variable
if (ROCM_INCLUDE_DIRS AND ROCM_LIBRARIES AND HIP_CMAKE)
  set(ROCM_FOUND TRUE)
else ()
  set(ROCM_FOUND FALSE)
endif ()

# print message
if (NOT ROCM_FIND_QUIETLY)
  message(STATUS "Found ROCM: ${ROCM_INCLUDE_DIRS}, ${ROCM_LIBRARIES}, ${HIP_CMAKE}"
    " in ${ROCM_search_PATHS}, build AMD GPU support")
endif ()

unset(ROCM_SEARCH_PATHS)
