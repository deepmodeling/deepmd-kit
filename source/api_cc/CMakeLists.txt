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

if (USE_ASCEND)
  # link: libdeepmd libdeepmd_op libtensorflow_cc libtensorflow_framework
  target_link_libraries (${libname} PUBLIC ${LIB_DEEPMD} ${TensorFlow_LIBRARY} ${TensorFlowFramework_LIBRARY})
  # link: libdeepmd libpython3.7m dl for Ascend platfrom
  target_link_libraries (${libname} PUBLIC ${Python_LIBRARIES} dl)
else()
  # link: libdeepmd libdeepmd_op libtensorflow_cc libtensorflow_framework
  target_link_libraries (${libname} PUBLIC ${LIB_DEEPMD}	${TensorFlow_LIBRARY} ${TensorFlowFramework_LIBRARY})
endif(USE_ASCEND)
target_include_directories(${libname} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(${libname} PRIVATE ${TensorFlow_INCLUDE_DIRS})


# link: libpython3.x.so
if (USE_TF_PYTHON_LIBS)
  target_link_libraries (${libname} PUBLIC ${Python_LIBRARIES})
endif(USE_TF_PYTHON_LIBS)

set_target_properties(
  ${libname} 
  PROPERTIES 
  INSTALL_RPATH "$ORIGIN;${TensorFlow_LIBRARY_PATH}"
)
target_compile_definitions(${libname}
  PUBLIC ${prec_def}
  PRIVATE TF_PRIVATE
)

install(TARGETS ${libname} DESTINATION lib/)

install(
  FILES		${INC_SRC}
  DESTINATION	include/deepmd
)

endfunction()

_add_libapicc_variant("${HIGH_PREC_VARIANT}" "${HIGH_PREC_DEF}")
_add_libapicc_variant("${LOW_PREC_VARIANT}" "${LOW_PREC_DEF}")
