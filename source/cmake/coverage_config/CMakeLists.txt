# include this directory before other modules
add_library(coverage_config INTERFACE)
target_compile_options(
  coverage_config
  INTERFACE -O0 # no optimization
            -g # generate debug info
            --coverage # sets all required flags
)
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.13)
  target_link_options(coverage_config INTERFACE --coverage)
else()
  target_link_libraries(coverage_config INTERFACE --coverage)
endif()
