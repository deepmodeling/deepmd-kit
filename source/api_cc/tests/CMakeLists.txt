cmake_minimum_required(VERSION 3.9)
project(deepmd_api_test)

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

function(_add_apicc_test_variant variant_name prec_def)
add_executable( runUnitTests_cc${variant_name} ${TEST_SRC} )
target_link_libraries(runUnitTests_cc${variant_name} GTest::gtest_main ${LIB_DEEPMD_CC} rt coverage_config)
add_test( runUnitTest_cc${variant_name} runUnitTests_cc${variant_name} )
set_target_properties(
  runUnitTests_cc${variant_name}
  PROPERTIES 
  INSTALL_RPATH "$ORIGIN/../lib"
)
target_compile_definitions(
  runUnitTests_cc${variant_name}
  PUBLIC ${prec_def}
)
install(TARGETS runUnitTests_cc${variant_name} DESTINATION bin/)
endfunction()

_add_apicc_test_variant("${HIGH_PREC_VARIANT}" "${HIGH_PREC_DEF}")
_add_apicc_test_variant("${LOW_PREC_VARIANT}" "${LOW_PREC_DEF}")
