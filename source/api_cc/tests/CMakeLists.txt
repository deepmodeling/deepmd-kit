cmake_minimum_required(VERSION 3.9)
project(deepmd_api_test)

file(GLOB TEST_SRC test_*.cc)

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
