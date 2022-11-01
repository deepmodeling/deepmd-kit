# libmd

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/version.h.in
  version.h
  @ONLY
)

file(GLOB LIB_SRC src/*.cc src/*.cpp)
file(GLOB INC_SRC include/*.h ${CMAKE_CURRENT_BINARY_DIR}/version.h)


set (libname "${LIB_DEEPMD_CC}")

add_library(${libname} SHARED ${LIB_SRC})

# link: libdeepmd libdeepmd_op libtensorflow_cc libtensorflow_framework
target_link_libraries (${libname} PUBLIC ${LIB_DEEPMD}	${TensorFlow_LIBRARY} ${TensorFlowFramework_LIBRARY})
target_include_directories(${libname} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(${libname} PRIVATE ${TensorFlow_INCLUDE_DIRS})

if(Protobuf_LIBRARY)
  target_link_libraries(${libname} PRIVATE ${Protobuf_LIBRARY})
endif()

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
  PRIVATE TF_PRIVATE
)
if (CMAKE_TESTING_ENABLED)
  target_link_libraries(${libname} PRIVATE coverage_config)
endif()

install(TARGETS ${libname} DESTINATION lib/)

install(
  FILES		${INC_SRC}
  DESTINATION	include/deepmd
)

# make a link to libdeepmd_cc_low.so for compatibility
INSTALL(CODE "execute_process( \
COMMAND ${CMAKE_COMMAND} -E create_symlink \
${CMAKE_SHARED_LIBRARY_PREFIX}${libname}${CMAKE_SHARED_LIBRARY_SUFFIX} \
${CMAKE_INSTALL_PREFIX}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}${libname}${LOW_PREC_VARIANT}${CMAKE_SHARED_LIBRARY_SUFFIX}   \
)"
)
