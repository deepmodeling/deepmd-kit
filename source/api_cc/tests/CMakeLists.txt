cmake_minimum_required(VERSION 3.25.2)
project(deepmd_api_test)

file(GLOB TEST_SRC test_*.cc)

add_executable(runUnitTests_cc ${TEST_SRC})
target_link_libraries(runUnitTests_cc GTest::gtest_main ${LIB_DEEPMD_CC}
                      coverage_config)
if(ENABLE_TENSORFLOW)
  target_compile_definitions(runUnitTests_cc PRIVATE BUILD_TENSORFLOW)
endif()
if(ENABLE_PYTORCH)
  target_compile_definitions(runUnitTests_cc PRIVATE BUILD_PYTORCH)
endif()
if(ENABLE_JAX)
  target_compile_definitions(runUnitTests_cc PRIVATE BUILD_JAX)
endif()
if(ENABLE_PADDLE)
  target_compile_definitions(runUnitTests_cc PRIVATE BUILD_PADDLE)
endif()

if(UNIX AND NOT APPLE)
  find_library(RT_LIBRARY rt)
  if(RT_LIBRARY)
    target_link_libraries(runUnitTests_cc ${RT_LIBRARY})
  endif()
endif()

target_precompile_headers(runUnitTests_cc PRIVATE test_utils.h)
add_test(
  NAME runUnitTest_cc
  COMMAND runUnitTests_cc
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
set_tests_properties(runUnitTest_cc PROPERTIES ENVIRONMENT
                                               "TF_DISABLE_MLIR_BRIDGE=1")
set_target_properties(runUnitTests_cc PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")
target_compile_definitions(runUnitTests_cc PUBLIC ${prec_def})
install(TARGETS runUnitTests_cc DESTINATION bin/)
