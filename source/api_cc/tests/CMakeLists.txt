cmake_minimum_required(VERSION 3.9)
project(deepmd_api_test)

file(GLOB TEST_SRC test_*.cc)

add_executable(runUnitTests_cc ${TEST_SRC})
target_link_libraries(runUnitTests_cc GTest::gtest_main ${LIB_DEEPMD_CC} rt
                      coverage_config)
target_precompile_headers(runUnitTests_cc PRIVATE test_utils.h)
add_test(runUnitTest_cc runUnitTests_cc)
set_target_properties(runUnitTests_cc PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")
target_compile_definitions(runUnitTests_cc PUBLIC ${prec_def})
install(TARGETS runUnitTests_cc DESTINATION bin/)
