cmake_minimum_required(VERSION 3.9)
project(deepmd_api_c_test)

file(GLOB TEST_SRC test_*.cc)

find_package(GTest)
if(NOT GTEST_LIBRARIES)
  configure_file(../../cmake/googletest.cmake.in googletest-download/CMakeLists.txt @ONLY)
  execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
  if(result)
    message(FATAL_ERROR "CMake step for googletest failed: ${result}")
  endif()
  execute_process(COMMAND ${CMAKE_COMMAND} --build .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
  if(result)
    message(FATAL_ERROR "Build step for googletest failed: ${result}")
  endif()
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src ${CMAKE_CURRENT_BINARY_DIR}/googletest-build EXCLUDE_FROM_ALL)
else ()
  include_directories(${GTEST_INCLUDE_DIRS})
endif ()

set_target_properties(
  ${LIB_DEEPMD_C}
  PROPERTIES 
  INSTALL_RPATH "$ORIGIN;${TensorFlow_LIBRARY_PATH}"
)

add_executable( runUnitTests_c ${TEST_SRC} )
target_link_libraries(runUnitTests_c PRIVATE GTest::gtest_main ${LIB_DEEPMD_C} rt coverage_config)
target_link_libraries(runUnitTests_c PRIVATE ${LIB_DEEPMD})
target_precompile_headers(runUnitTests_c PRIVATE test_utils.h [["deepmd.hpp"]])
add_test( runUnitTests_c runUnitTests_c )
set_target_properties(
  runUnitTests_c
  PROPERTIES 
  INSTALL_RPATH "$ORIGIN/../lib"
)
install(TARGETS runUnitTests_c DESTINATION bin/)

