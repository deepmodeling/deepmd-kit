cmake_minimum_required(VERSION 3.9)
project(libdeepmd_test)

file(GLOB TEST_SRC test_*.cc)
add_executable(runUnitTests_lib ${TEST_SRC})

target_link_libraries(runUnitTests_lib GTest::gtest_main GTest::gmock_main
                      ${LIB_DEEPMD} coverage_config)
add_test(runUnitTests_lib runUnitTests_lib)

set_target_properties(runUnitTests_lib PROPERTIES INSTALL_RPATH
                                                  "$ORIGIN/../lib")

install(TARGETS runUnitTests_lib DESTINATION bin/)
