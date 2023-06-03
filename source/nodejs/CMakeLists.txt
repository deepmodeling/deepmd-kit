find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})

set_property(SOURCE ../swig/deepmd.i PROPERTY CPLUSPLUS ON)
swig_add_library(
  deepmd_nodejs
  LANGUAGE javascript
  SOURCES ../swig/deepmd.i OUTFILE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
swig_link_libraries(deepmd_nodejs ${LIB_DEEPMD_C})
set_property(TARGET deepmd_nodejs PROPERTY SWIG_COMPILE_OPTIONS -node)
set_property(TARGET deepmd_nodejs PROPERTY SWIG_USE_TARGET_INCLUDE_DIRECTORIES
                                           ON)
target_include_directories(deepmd_nodejs PRIVATE ${NODEJS_INCLUDE_DIRS})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/binding.gyp.in
               ${CMAKE_CURRENT_SOURCE_DIR}/binding.gyp @ONLY)
