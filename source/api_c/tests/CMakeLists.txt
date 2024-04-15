cmake_minimum_required(VERSION 3.9)
project(deepmd_api_c_test)

file(GLOB TEST_SRC test_*.cc)

set_target_properties(
  ${LIB_DEEPMD_C} PROPERTIES INSTALL_RPATH "$ORIGIN;${BACKEND_LIBRARY_PATH}")

add_executable(runUnitTests_c ${TEST_SRC})
target_link_libraries(runUnitTests_c PRIVATE GTest::gtest_main ${LIB_DEEPMD_C}
                                             coverage_config)
target_link_libraries(runUnitTests_c PRIVATE ${LIB_DEEPMD} ${LIB_DEEPMD_CC})

if(UNIX AND NOT APPLE)
  find_library(RT_LIBRARY rt)
  if(RT_LIBRARY)
    target_link_libraries(runUnitTests_c PRIVATE ${RT_LIBRARY})
  endif()
endif()

target_precompile_headers(runUnitTests_c PRIVATE test_utils.h [["deepmd.hpp"]])
add_test(
  NAME runUnitTests_c
  COMMAND runUnitTests_c
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(runUnitTests_c PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")
install(TARGETS runUnitTests_c DESTINATION bin/)
