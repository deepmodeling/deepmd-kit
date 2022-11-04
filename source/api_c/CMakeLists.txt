file(GLOB LIB_SRC src/*.cc src/*.cpp)
# only install public headers
file(GLOB INC_SRC include/c_api.h)

set (libname "${LIB_DEEPMD_C}")

add_library(${libname} SHARED ${LIB_SRC})

# link: libdeepmd libdeepmd_op libtensorflow_cc libtensorflow_framework
target_link_libraries (${libname} PRIVATE ${LIB_DEEPMD_CC})
target_include_directories(
  ${libname} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

set_target_properties(
  ${libname} 
  PROPERTIES 
  INSTALL_RPATH "$ORIGIN;${TensorFlow_LIBRARY_PATH}"
)
if (CMAKE_TESTING_ENABLED)
  target_link_libraries(${libname} PRIVATE coverage_config)
endif()

install(
  TARGETS ${libname}
  EXPORT ${CMAKE_PROJECT_NAME}Targets
  DESTINATION lib/
)

install(
  FILES		${INC_SRC}
  DESTINATION	include/deepmd
)
