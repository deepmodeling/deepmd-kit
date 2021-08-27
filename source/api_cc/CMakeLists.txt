# libmd

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/version.h.in
  version.h
  @ONLY
)

file(GLOB LIB_SRC src/*.cc src/*.cpp)
file(GLOB INC_SRC include/*.h ${CMAKE_CURRENT_BINARY_DIR}/version.h)


function(_add_libapicc_variant variant_name prec_def)
set (libname "${LIB_DEEPMD_CC}${variant_name}")

add_library(${libname} SHARED ${LIB_SRC})

# link: libdeepmd libdeepmd_op libtensorflow_cc libtensorflow_framework
target_link_libraries (${libname} PUBLIC ${LIB_DEEPMD}	${TensorFlow_LIBRARY} ${TensorFlowFramework_LIBRARY})
target_link_libraries (${libname} PRIVATE	${LIB_DEEPMD_OP})
target_include_directories(${libname} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR} ${TensorFlow_INCLUDE_DIRS})

set_target_properties(
  ${libname} 
  PROPERTIES 
  COMPILE_DEFINITIONS ${prec_def}
  INSTALL_RPATH "$ORIGIN;${TensorFlow_LIBRARY_PATH}"
)

install(TARGETS ${libname} DESTINATION lib/)

install(
  FILES		${INC_SRC}
  DESTINATION	include/deepmd
)

endfunction()

_add_libapicc_variant("${HIGH_PREC_VARIANT}" "${HIGH_PREC_DEF}")
_add_libapicc_variant("${LOW_PREC_VARIANT}" "${LOW_PREC_DEF}")
