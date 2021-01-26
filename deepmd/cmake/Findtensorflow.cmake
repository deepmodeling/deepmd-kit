# Input:
# TENSORFLOW_ROOT 
# BUILD_CPP_IF
#
# Output:
# TensorFlow_FOUND        
# TensorFlow_INCLUDE_DIRS 
# TensorFlow_LIBRARY    
# TensorFlow_LIBRARY_PATH
# TensorFlowFramework_LIBRARY    
# TensorFlowFramework_LIBRARY_PATH


if(NOT DEFINED TENSORFLOW_ROOT)
  message(STATUS "TENSORFLOW_ROOT not set, finding in current python environment...")
  set(FIND_TENSORFLOW_ROOT_CMD "import os, tensorflow; print(os.path.dirname(tensorflow.__file__), end='')")
  execute_process(
          COMMAND "python" "-c" "${FIND_TENSORFLOW_ROOT_CMD}"  # Automatically set PYTHON_EXECUTABLE
          WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
          OUTPUT_VARIABLE TENSORFLOW_ROOT
          RESULT_VARIABLE TENSORFLOW_ROOT_RESULT_VAR
          ERROR_VARIABLE TENSORFLOW_ROOT_ERROR_VAR
  )
  message(STATUS "Set hypothetical TENSORFLOW_ROOT as: ${TENSORFLOW_ROOT}")
endif()

string(REPLACE "lib64" "lib" TENSORFLOW_ROOT_NO64 ${TENSORFLOW_ROOT})

# search path when using conda
if (NOT $ENV{CONDA_PREFIX} STREQUAL "")
    message(STATUS "Detected Conda being used, add corresponding search paths for TensorFlow")
    list(APPEND TensorFlow_search_PATHS $ENV{CONDA_PREFIX})
endif()

# define the search path
list(APPEND TensorFlow_search_PATHS ${TENSORFLOW_ROOT})
list(APPEND TensorFlow_search_PATHS "${TENSORFLOW_ROOT}/../tensorflow_core")
list(APPEND TensorFlow_search_PATHS ${TENSORFLOW_ROOT_NO64})
list(APPEND TensorFlow_search_PATHS "${TENSORFLOW_ROOT_NO64}/../tensorflow_core")
list(APPEND TensorFlow_search_PATHS "/usr/")
list(APPEND TensorFlow_search_PATHS "/usr/local/")


# includes
find_path(TensorFlow_INCLUDE_DIRS
  NAMES 
  tensorflow/core/public/session.h  # "Or" relations between NAMES
  tensorflow/core/platform/env.h
  tensorflow/core/framework/op.h
  tensorflow/core/framework/op_kernel.h
  tensorflow/core/framework/shape_inference.h
  PATHS ${TensorFlow_search_PATHS} 
  PATH_SUFFIXES "/include"
  NO_DEFAULT_PATH
  )
find_path(TensorFlow_INCLUDE_DIRS_GOOGLE
  NAMES 
  google/protobuf/type.pb.h
  PATHS ${TensorFlow_search_PATHS} 
  PATH_SUFFIXES "/include"
  NO_DEFAULT_PATH
  )
list(APPEND TensorFlow_INCLUDE_DIRS ${TensorFlow_INCLUDE_DIRS_GOOGLE})
  
if (NOT TensorFlow_INCLUDE_DIRS AND tensorflow_FIND_REQUIRED)
  message(FATAL_ERROR 
    "Not found 'tensorflow/core/public/session.h' directory in path '${TensorFlow_search_PATHS}' "
    "You can manually set the tensorflow install path by -DTENSORFLOW_ROOT ")
endif ()

# the lib
list(APPEND CMAKE_FIND_LIBRARY_SUFFIXES .so.1)
list(APPEND CMAKE_FIND_LIBRARY_SUFFIXES .so.2)

# tensorflow all
if (BUILD_CPP_IF)
  message (STATUS "Enabled cpp interface build, looking for tensorflow_cc and tensorflow_framework")
  # tensorflow_cc and tensorflow_framework
  if (NOT TensorFlow_FIND_COMPONENTS)
    set(TensorFlow_FIND_COMPONENTS tensorflow_cc tensorflow_framework)
  endif ()
  set (TensorFlow_LIBRARY_PATH "")
  foreach (module ${TensorFlow_FIND_COMPONENTS})
    find_library(TensorFlow_LIBRARY_${module}
      NAMES ${module}
      PATHS ${TensorFlow_search_PATHS}
      PATH_SUFFIXES lib
      NO_DEFAULT_PATH
      )
    if (TensorFlow_LIBRARY_${module})
      list(APPEND TensorFlow_LIBRARY ${TensorFlow_LIBRARY_${module}})
      get_filename_component(TensorFlow_LIBRARY_PATH_${module} ${TensorFlow_LIBRARY_${module}} DIRECTORY)
      list(APPEND TensorFlow_LIBRARY_PATH ${TensorFlow_LIBRARY_PATH_${module}})
    elseif (tensorflow_FIND_REQUIRED)
      message(FATAL_ERROR
	"Not found lib/'${module}' in '${TensorFlow_search_PATHS}' "
	"You can manually set the tensorflow install path by -DTENSORFLOW_ROOT ")
    endif ()
  endforeach ()
else ()
  message (STATUS "Disabled cpp interface build, looking for tensorflow_framework")
endif ()

# tensorflow_framework
if (NOT TensorFlowFramework_FIND_COMPONENTS)
  set(TensorFlowFramework_FIND_COMPONENTS tensorflow_framework)
endif ()
set (TensorFlowFramework_LIBRARY_PATH "")
foreach (module ${TensorFlowFramework_FIND_COMPONENTS})
  find_library(TensorFlowFramework_LIBRARY_${module}
    NAMES ${module}
    PATHS ${TensorFlow_search_PATHS}
    PATH_SUFFIXES lib
    NO_DEFAULT_PATH
    )
  if (TensorFlowFramework_LIBRARY_${module})
    list(APPEND TensorFlowFramework_LIBRARY ${TensorFlowFramework_LIBRARY_${module}})  # The one in py module dir if py version installed
    get_filename_component(TensorFlowFramework_LIBRARY_PATH_${module} ${TensorFlowFramework_LIBRARY_${module}} PATH)
    list(APPEND TensorFlowFramework_LIBRARY_PATH ${TensorFlowFramework_LIBRARY_PATH_${module}})
  elseif (tensorflow_FIND_REQUIRED)
    message(FATAL_ERROR 
      "Not found lib/'${module}' in '${TensorFlow_search_PATHS}' "
      "You can manually set the tensorflow install path by -DTENSORFLOW_ROOT ")
  endif ()
endforeach ()

if (BUILD_CPP_IF)
  # define the output variable
  if (TensorFlow_INCLUDE_DIRS AND TensorFlow_LIBRARY AND TensorFlowFramework_LIBRARY)
    set(TensorFlow_FOUND TRUE)
  else ()
    set(TensorFlow_FOUND FALSE)
  endif ()
else ()
  if (TensorFlow_INCLUDE_DIRS AND TensorFlowFramework_LIBRARY)
    set(TensorFlow_FOUND TRUE)
  else ()
    set(TensorFlow_FOUND FALSE)
  endif ()
endif ()

# print message
if (NOT TensorFlow_FIND_QUIETLY)
  message(STATUS "Found TensorFlow: TRUE")
  message(STATUS "Found TensorFlow: TensorFlow_INCLUDE_DIRS: ${TensorFlow_INCLUDE_DIRS}")
  message(STATUS "Found TensorFlow: TensorFlow_LIBRARY: ${TensorFlow_LIBRARY}")
  message(STATUS "Found TensorFlow: TensorFlowFramework_LIBRARY: ${TensorFlowFramework_LIBRARY}")
endif ()

unset(TensorFlow_search_PATHS)
