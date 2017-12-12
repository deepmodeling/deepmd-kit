# Input:
# XDRFILE_ROOT 
#
# Output:
# XDRFILE_FOUND        
# XDRFILE_INCLUDE_DIRS 
# XDRFILE_LIBRARIES    

# define the search path
list(APPEND XDRFILE_search_PATHS ${XDRFILE_ROOT})
list(APPEND XDRFILE_search_PATHS "/usr/")
list(APPEND XDRFILE_search_PATHS "/usr/local/")

# define the libs to find
if (NOT XDRFILE_FIND_COMPONENTS)
  set(XDRFILE_FIND_COMPONENTS xdrfile)
endif ()

# includes
find_path (XDRFILE_INCLUDE_DIRS
  NAMES 
  xdrfile/xdrfile.h
  xdrfile/xdrfile_xtc.h
  xdrfile/xdrfile_trr.h
  PATHS ${XDRFILE_search_PATHS} 
  PATH_SUFFIXES "/include"
  NO_DEFAULT_PATH
  )
if (NOT XDRFILE_INCLUDE_DIRS AND xdrfile_FIND_REQUIRED)
  message(FATAL_ERROR 
    "Not found 'include/xdrfile/xdrfile.h' directory in path '${XDRFILE_search_PATHS}' "
    "You can manually set the xdrfile install path by -DXDRFILE_ROOT ")
endif ()

# libs
foreach (module ${XDRFILE_FIND_COMPONENTS})
  find_library(XDRFILE_LIBRARIES_${module}
    NAMES ${module}
    PATHS ${XDRFILE_search_PATHS} PATH_SUFFIXES lib NO_DEFAULT_PATH
    )
  if (XDRFILE_LIBRARIES_${module})
    list(APPEND XDRFILE_LIBRARIES ${XDRFILE_LIBRARIES_${module}})
  elseif (xdrfile_FIND_REQUIRED)
    message(FATAL_ERROR 
      "Not found lib/'${module}' in '${XDRFILE_search_PATHS}' "
      "You can manually set the xdrfile install path by -DXDRFILE_ROOT ")
  endif ()
endforeach ()

# define the output variable
if (XDRFILE_INCLUDE_DIRS AND XDRFILE_LIBRARIES)
  set(XDRFILE_FOUND TRUE)
else ()
  set(XDRFILE_FOUND FALSE)
endif ()

# print message
if (NOT XDRFILE_FIND_QUIETLY)
  message(STATUS "Found XDRFILE: ${XDRFILE_INCLUDE_DIRS}, ${XDRFILE_LIBRARIES}"
    " in ${XDRFILE_search_PATHS}")
endif ()

unset(XDRFILE_search_PATHS)
