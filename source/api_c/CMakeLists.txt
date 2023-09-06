file(GLOB LIB_SRC src/*.cc src/*.cpp)
# only install public headers
file(GLOB INC_SRC include/c_api.h include/deepmd.hpp)

set(libname "${LIB_DEEPMD_C}")

add_library(${libname} SHARED ${LIB_SRC})

# link: libdeepmd libdeepmd_op libtensorflow_cc libtensorflow_framework
target_link_libraries(${libname} PRIVATE ${LIB_DEEPMD_CC})
target_include_directories(
  ${libname} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                    $<INSTALL_INTERFACE:include>)

set_target_properties(
  ${libname} PROPERTIES INSTALL_RPATH "$ORIGIN;${TensorFlow_LIBRARY_PATH}")
if(CMAKE_TESTING_ENABLED)
  target_link_libraries(${libname} PRIVATE coverage_config)
endif()

if(BUILD_PY_IF)
  install(TARGETS ${libname} DESTINATION deepmd/lib/)
else(BUILD_PY_IF)
  install(
    TARGETS ${libname}
    EXPORT ${CMAKE_PROJECT_NAME}Targets
    DESTINATION lib/)

  install(FILES ${INC_SRC} DESTINATION include/deepmd)
endif(BUILD_PY_IF)

if(PACKAGE_C)
  message(STATUS "Packaging C API library")
  # follow pypa/auditwheel convention
  install(
    CODE [[
    file(GET_RUNTIME_DEPENDENCIES
      LIBRARIES $<TARGET_FILE:deepmd_c> $<TARGET_FILE:deepmd_op>
      RESOLVED_DEPENDENCIES_VAR _r_deps
      PRE_EXCLUDE_REGEXES "libgcc_s\\.so.*"
                          "libstdc\\+\\+\\.so.*"
                          "libm\\.so.*"
                          "libdl\\.so.*"
                          "librt\\.so.*"
                          "libc\\.so.*"
                          "libpthread\\.so.*"
                          "ld-.*\\.so.*"
                          "libgomp\\.so.*"
      )
    message(STATUS "Runtime dependencies: ${_r_deps}")
    foreach(_file ${_r_deps})
      file(INSTALL ${_file} DESTINATION libdeepmd_c/lib
        FOLLOW_SYMLINK_CHAIN
      )
    endforeach()
    ]])
  install(FILES ${INC_SRC}
          DESTINATION ${CMAKE_BINARY_DIR}/libdeepmd_c/include/deepmd)
  install(TARGETS ${libname} DESTINATION ${CMAKE_BINARY_DIR}/libdeepmd_c/lib)
  install(TARGETS ${LIB_DEEPMD_OP}
          DESTINATION ${CMAKE_BINARY_DIR}/libdeepmd_c/lib)
endif()

if(CMAKE_TESTING_ENABLED)
  add_subdirectory(tests)
endif()
