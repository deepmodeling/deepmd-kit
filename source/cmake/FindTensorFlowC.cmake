# Find TensorFlow C library (libtensorflow) Define target
# TensorFlow::tensorflow_c If TensorFlow::tensorflow_cc is not found, also
# define: - TENSORFLOWC_INCLUDE_DIR - TENSORFLOWC_LIBRARY

if(TARGET TensorFlow::tensorflow_cc)
  # since tensorflow_cc contain tensorflow_c, just use it
  add_library(TensorFlow::tensorflow_c ALIAS TensorFlow::tensorflow_cc)
  set(TensorFlowC_FOUND TRUE)
endif()

if(NOT TensorFlowC_FOUND)
  find_path(
    TENSORFLOWC_INCLUDE_DIR
    NAMES tensorflow/c/c_api.h
    PATH_SUFFIXES include
    DOC "Path to TensorFlow C include directory")

  find_library(
    TENSORFLOWC_LIBRARY
    NAMES tensorflow
    PATH_SUFFIXES lib
    DOC "Path to TensorFlow C library")

  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(
    TensorFlowC REQUIRED_VARS TENSORFLOWC_LIBRARY TENSORFLOWC_INCLUDE_DIR)

  if(TensorFlowC_FOUND)
    set(TensorFlowC_INCLUDE_DIRS ${TENSORFLOWC_INCLUDE_DIR})
    set(TensorFlowC_LIBRARIES ${TENSORFLOWC_LIBRARY})
  endif()

  add_library(TensorFlow::tensorflow_c SHARED IMPORTED GLOBAL)
  set_property(TARGET TensorFlow::tensorflow_c PROPERTY IMPORTED_LOCATION
                                                        ${TENSORFLOWC_LIBRARY})
  target_include_directories(TensorFlow::tensorflow_c
                             INTERFACE ${TENSORFLOWC_INCLUDE_DIR})

  mark_as_advanced(TENSORFLOWC_LIBRARY TENSORFLOWC_INCLUDE_DIR)
endif()
