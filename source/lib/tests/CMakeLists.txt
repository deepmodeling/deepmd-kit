cmake_minimum_required(VERSION 3.9)
project(libdeepmd_test)

enable_testing()

set(libname "deepmd")
set(LIB_BASE_DIR ${CMAKE_SOURCE_DIR}/../)

include_directories(${LIB_BASE_DIR}/include)
file(GLOB LIB_SRC ${LIB_BASE_DIR}/src/*.cc ${LIB_BASE_DIR}/src/*.cpp)
message(status ${LIB_SRC})
# add_library(${libname} SHARED ${LIB_SRC})
add_library(${libname} ${LIB_SRC})

message(status "${CMAKE_SOURCE_DIR}")

file(GLOB TEST_SRC test_*.cc)
add_executable( runUnitTests ${TEST_SRC} )
target_link_libraries(runUnitTests gtest gtest_main ${libname})
add_test( runUnitTests runUnitTests )

# include(GoogleTest)
# add_executable(FooTest tests/test_simulation_region.cc)
# gtest_add_tests(TARGET      FooTest
#                 TEST_SUFFIX .noArgs
#                 TEST_LIST   noArgsTests
# )

