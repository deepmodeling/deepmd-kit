# md

list(APPEND MD_INCLUDE_PATH "include")

file(GLOB IN_SRC src/*.cc src/*.c)

set(ipiname "dp_ipi")
set(libipiname "${LIB_DEEPMD_IPI}")
add_library(${libipiname} SHARED ${IN_SRC})
target_include_directories(${libipiname} PUBLIC ${MD_INCLUDE_PATH})

set(DRIVER_SOURCE_FILES driver.cc)
add_executable(${ipiname} ${DRIVER_SOURCE_FILES})
# link: libdeepmd_cc
if(DP_USING_C_API)
  # SimulationRegion.h
  target_link_libraries(${ipiname} PRIVATE ${LIB_DEEPMD_C})
  target_precompile_headers(${ipiname} PRIVATE [["deepmd.hpp"]])
  remove_definitions(-D_GLIBCXX_USE_CXX11_ABI=${OP_CXX_ABI})
else()
  target_link_libraries(${ipiname} PRIVATE ${LIB_DEEPMD_CC})
  target_compile_definitions(${ipiname} PRIVATE "DP_USE_CXX_API")
endif()
target_link_libraries(${ipiname} PRIVATE ${libipiname})
target_include_directories(${ipiname}
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/)
set(LIB_DIR lib)

if(BUILD_PY_IF AND TENSORFLOW_LINK_LIBPYTHON)
  # ignore undefined reference for libpython
  if(CMAKE_SYSTEM_NAME STREQUAL Darwin)
    set(extra_link_flags "-Wl,-undefined,dynamic_lookup")
  else()
    set(extra_link_flags "-Wl,--unresolved-symbols=ignore-in-shared-libs")
  endif()
else(BUILD_PY_IF AND TENSORFLOW_LINK_LIBPYTHON)
  if(CMAKE_SYSTEM_NAME STREQUAL Linux)
    set(extra_link_flags "-Wl,-z,defs")
  endif()
endif(BUILD_PY_IF AND TENSORFLOW_LINK_LIBPYTHON)

if(APPLE)
  set_target_properties(
    ${ipiname}
    PROPERTIES LINK_FLAGS "${extra_link_flags}"
               INSTALL_RPATH
               "@loader_path/../${LIB_DIR}:${TensorFlow_LIBRARY_PATH}")
else()
  set_target_properties(
    ${ipiname}
    PROPERTIES LINK_FLAGS
               "-Wl,-rpath,'$ORIGIN'/../${LIB_DIR} ${extra_link_flags}"
               INSTALL_RPATH "$ORIGIN/../${LIB_DIR}:${TensorFlow_LIBRARY_PATH}")
endif()

if(CMAKE_TESTING_ENABLED)
  target_link_libraries(${ipiname} PRIVATE coverage_config)
  target_link_libraries(${libipiname} PRIVATE coverage_config)
endif()

if(BUILD_PY_IF)
  install(TARGETS ${libipiname} DESTINATION deepmd/lib/)
  install(TARGETS ${ipiname} DESTINATION deepmd/lib/)
else(BUILD_PY_IF)
  install(TARGETS ${libipiname} DESTINATION lib/)
  install(TARGETS ${ipiname} DESTINATION bin/)
endif(BUILD_PY_IF)
