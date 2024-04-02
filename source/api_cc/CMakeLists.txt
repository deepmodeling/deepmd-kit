# libmd

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/include/version.h.in version.h @ONLY)

file(GLOB LIB_SRC src/*.cc src/*.cpp)
file(GLOB INC_SRC include/*.h ${CMAKE_CURRENT_BINARY_DIR}/version.h)

set(libname "${LIB_DEEPMD_CC}")

add_library(${libname} SHARED ${LIB_SRC})

# link: libdeepmd libdeepmd_op libtensorflow_cc libtensorflow_framework
target_link_libraries(${libname} PUBLIC ${LIB_DEEPMD})
target_link_libraries(${libname} PRIVATE TensorFlow::tensorflow_cc
                                         TensorFlow::tensorflow_framework)
target_include_directories(
  ${libname}
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
         $<INSTALL_INTERFACE:include>)
target_precompile_headers(${libname} PUBLIC [["common.h"]])

if(Protobuf_LIBRARY)
  target_link_libraries(${libname} PRIVATE ${Protobuf_LIBRARY})
endif()

set_target_properties(
  ${libname}
  PROPERTIES INSTALL_RPATH "$ORIGIN;${BACKEND_LIBRARY_PATH}"
             INSTALL_RPATH_USE_LINK_PATH TRUE
             BUILD_RPATH "$ORIGIN/../op")
target_compile_definitions(${libname} PRIVATE TF_PRIVATE)
if(CMAKE_TESTING_ENABLED)
  target_link_libraries(${libname} PRIVATE coverage_config)
endif()
target_compile_features(${libname} PUBLIC cxx_std_11)

if(BUILD_PY_IF)
  install(TARGETS ${libname} DESTINATION deepmd/lib/)
else(BUILD_PY_IF)
  install(
    TARGETS ${libname}
    EXPORT ${CMAKE_PROJECT_NAME}Targets
    DESTINATION lib/)

  install(FILES ${INC_SRC} DESTINATION include/deepmd)

  # make a link to libdeepmd_cc_low.so for compatibility
  install(
    CODE "execute_process( \
COMMAND ${CMAKE_COMMAND} -E create_symlink \
${CMAKE_SHARED_LIBRARY_PREFIX}${libname}${CMAKE_SHARED_LIBRARY_SUFFIX} \
${CMAKE_INSTALL_PREFIX}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}${libname}${LOW_PREC_VARIANT}${CMAKE_SHARED_LIBRARY_SUFFIX}   \
)")

  if(CMAKE_TESTING_ENABLED)
    add_subdirectory(tests)
  endif()
endif(BUILD_PY_IF)
