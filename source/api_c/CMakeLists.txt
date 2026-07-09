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

set_target_properties(${libname} PROPERTIES INSTALL_RPATH
                                            "$ORIGIN;${BACKEND_LIBRARY_PATH}")
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
  set(PACKAGE_C_RUNTIME_DEPENDENCY_PRE_EXCLUDE_REGEXES
      ""
      CACHE STRING
            "Additional file(GET_RUNTIME_DEPENDENCIES) pre-exclude regexes")
  set(PACKAGE_C_RUNTIME_DEPENDENCY_POST_EXCLUDE_REGEXES
      ""
      CACHE STRING
            "Additional file(GET_RUNTIME_DEPENDENCIES) post-exclude regexes")
  set(PACKAGE_C_RUNTIME_DEPENDENCY_PRE_EXCLUDE_REGEXES_DEFAULT
      "libgcc_s\\.so.*"
      "libstdc\\+\\+\\.so.*"
      "libm\\.so.*"
      "libdl\\.so.*"
      "librt\\.so.*"
      "libc\\.so.*"
      "libpthread\\.so.*"
      "ld-.*\\.so.*"
      "libgomp\\.so.*")
  set(PACKAGE_C_RUNTIME_DEPENDENCY_PRE_EXCLUDE_REGEXES_ALL
      ${PACKAGE_C_RUNTIME_DEPENDENCY_PRE_EXCLUDE_REGEXES_DEFAULT}
      ${PACKAGE_C_RUNTIME_DEPENDENCY_PRE_EXCLUDE_REGEXES})
  get_property(DEEPMD_BACKEND_PLUGIN_TARGETS GLOBAL
               PROPERTY DEEPMD_BACKEND_PLUGIN_TARGETS)
  set(PACKAGE_C_RUNTIME_LIBRARIES "$<TARGET_FILE:${libname}>")
  if(TARGET ${LIB_DEEPMD_OP})
    list(APPEND PACKAGE_C_RUNTIME_LIBRARIES "$<TARGET_FILE:${LIB_DEEPMD_OP}>")
  endif()
  foreach(_target deepmd_op_pt deepmd_op_pd)
    if(TARGET ${_target})
      list(APPEND PACKAGE_C_RUNTIME_LIBRARIES "$<TARGET_FILE:${_target}>")
    endif()
  endforeach()
  foreach(_target ${DEEPMD_BACKEND_PLUGIN_TARGETS})
    list(APPEND PACKAGE_C_RUNTIME_LIBRARIES "$<TARGET_FILE:${_target}>")
  endforeach()
  string(REPLACE ";" " " PACKAGE_C_RUNTIME_LIBRARIES_CODE
                 "${PACKAGE_C_RUNTIME_LIBRARIES}")
  # follow pypa/auditwheel convention
  install(CODE "set(_dp_runtime_libraries ${PACKAGE_C_RUNTIME_LIBRARIES_CODE})")
  install(
    CODE "set(_dp_package_c_lib_dir [==[${CMAKE_BINARY_DIR}/libdeepmd_c/lib]==])
set(_dp_runtime_pre_exclude_regexes [==[${PACKAGE_C_RUNTIME_DEPENDENCY_PRE_EXCLUDE_REGEXES_ALL}]==])
set(_dp_runtime_post_exclude_regexes [==[${PACKAGE_C_RUNTIME_DEPENDENCY_POST_EXCLUDE_REGEXES}]==])"
  )
  install(
    CODE [[
    file(GET_RUNTIME_DEPENDENCIES
      LIBRARIES ${_dp_runtime_libraries}
      RESOLVED_DEPENDENCIES_VAR _r_deps
      PRE_EXCLUDE_REGEXES ${_dp_runtime_pre_exclude_regexes}
      POST_EXCLUDE_REGEXES ${_dp_runtime_post_exclude_regexes}
      )
    message(STATUS "Runtime dependencies: ${_r_deps}")
    foreach(_file ${_r_deps})
      file(INSTALL ${_file} DESTINATION "${_dp_package_c_lib_dir}"
        FOLLOW_SYMLINK_CHAIN
      )
    endforeach()
    ]])
  install(FILES ${INC_SRC}
          DESTINATION ${CMAKE_BINARY_DIR}/libdeepmd_c/include/deepmd)
  install(TARGETS ${libname} DESTINATION ${CMAKE_BINARY_DIR}/libdeepmd_c/lib)
  if(TARGET ${LIB_DEEPMD_OP})
    install(TARGETS ${LIB_DEEPMD_OP}
            DESTINATION ${CMAKE_BINARY_DIR}/libdeepmd_c/lib)
  endif()
  foreach(_target deepmd_op_pt deepmd_op_pd)
    if(TARGET ${_target})
      install(TARGETS ${_target}
              DESTINATION ${CMAKE_BINARY_DIR}/libdeepmd_c/lib)
    endif()
  endforeach()
  foreach(_target ${DEEPMD_BACKEND_PLUGIN_TARGETS})
    install(TARGETS ${_target} DESTINATION ${CMAKE_BINARY_DIR}/libdeepmd_c/lib)
  endforeach()

  set(cmake_files_install_dir
      "${CMAKE_BINARY_DIR}/libdeepmd_c/lib/cmake/${CMAKE_PROJECT_NAME}")
  set(targets_export_name ${CMAKE_PROJECT_NAME}Targets)
  set(generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")
  set(version_file "${generated_dir}/${CMAKE_PROJECT_NAME}ConfigVersion.cmake")
  write_basic_package_version_file(
    ${version_file}
    VERSION $<IF:${GIT_SUMM}?${GIT_SUMM}:0.0.0>
    COMPATIBILITY AnyNewerVersion)
  set(config_file "${generated_dir}/${CMAKE_PROJECT_NAME}Config.cmake")
  configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/Config.cmake.in" "${config_file}"
    INSTALL_DESTINATION "lib/cmake/${CMAKE_PROJECT_NAME}")
  install(FILES ${version_file} ${config_file}
          DESTINATION "${cmake_files_install_dir}")
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/DeePMDPackagedTargets.cmake.in"
    "${generated_dir}/${targets_export_name}.cmake" @ONLY)
  install(FILES "${generated_dir}/${targets_export_name}.cmake"
          DESTINATION "${cmake_files_install_dir}")
endif()

if(CMAKE_TESTING_ENABLED)
  add_subdirectory(tests)
endif()
