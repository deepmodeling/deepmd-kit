cmake_minimum_required(VERSION 3.12.0)

project(dp_ipu_op)

include(GNUInstallDirs)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" )

list(APPEND POPC_ENVIRONMENT
"ASAN_OPTIONS=detect_leaks=0,halt_on_error=0"
"MSAN_OPTIONS=halt_on_error=0"
)

include(AddGPLibrary)

find_package(poplar REQUIRED)

if(NOT DEFINED POPLIBS_ENABLED_IPU_ARCH_NAMES)
  # by default build for all architectures that the Poplar we are using supports.
  get_target_property(POPC_EXECUTABLE popc_bin LOCATION)

  execute_process(COMMAND ${POPC_EXECUTABLE} --list-targets
                  OUTPUT_VARIABLE POPC_LIST_TARGETS_OUTPUT
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
                  string(REPLACE "," ";" TMP "${POPC_LIST_TARGETS_OUTPUT}")
  string(REPLACE "cpu;" "" TMP_REMOVE_CPU "${TMP}")
  set(TMP ${TMP_REMOVE_CPU})
  set(POPLIBS_ENABLED_IPU_ARCH_NAMES "${TMP}" CACHE STRING
      "The architectures to build the poplibs library objects for")
endif()
message("Enabled architectures: ${POPLIBS_ENABLED_IPU_ARCH_NAMES}")

# allow users to specify a subset of test variants to test on.
set(POPLIBS_ENABLED_IPU_ARCH_TEST_VARIANTS "${POPLIBS_ENABLED_IPU_ARCH_NAMES}" CACHE STRING
    "The architectures to run tests for, defaults to all enabled architectures")
message("Enabled test architectures: ${POPLIBS_ENABLED_IPU_ARCH_TEST_VARIANTS}")

# allow users easily to opt out from testing :cpp variants, this is off by
# default because this code is not packaged.
set(POPLIBS_ENABLE_CPP_TEST_VARIANTS OFF CACHE STRING
    "Whether to include cpp tests for codelets that otherwise have an assembly implentation")
message("cpp tests enabled: ${POPLIBS_ENABLE_CPP_TEST_VARIANTS}")

set(CPU_VARIANTS "")
set(IPUMODEL_VARIANTS "")
set(SIM_VARIANTS "")
set(SIMCPP_VARIANTS "")
foreach(IPU_ARCH_TEST_VARIANT ${POPLIBS_ENABLED_IPU_ARCH_TEST_VARIANTS})
  if (IPU_ARCH_TEST_VARIANT STREQUAL "cpu")
    list(APPEND CPU_VARIANTS "Cpu")
  elseif(IPU_ARCH_TEST_VARIANT MATCHES "^ipu")
    string(REPLACE "ipu" "IpuModel" IPUMODEL_VARIANT "${IPU_ARCH_TEST_VARIANT}")
    list(APPEND IPUMODEL_VARIANTS "${IPUMODEL_VARIANT}")

    string(REPLACE "ipu" "Sim" SIM_VARIANT "${IPU_ARCH_TEST_VARIANT}")
    list(APPEND SIM_VARIANTS "${SIM_VARIANT}")
    if (POPLIBS_ENABLE_CPP_TEST_VARIANTS)
      list(APPEND SIMCPP_VARIANTS "${SIM_VARIANT}:cpp")
    endif()
  else()
    message(FATAL_ERROR "Unrecognised architecture: ${IPU_ARCH_TEST_VARIANT}")
  endif()
endforeach()

set(DEFAULT_TEST_VARIANTS "${IPUMODEL_VARIANTS};${SIM_VARIANTS};${SIMCPP_VARIANTS};Hw" CACHE STRING
    "The device variants to run tests on when no variants are set explicitly")

set(POPC_FLAGS -O2)

add_gp_library(
  NAME dp_ipu_op
  CPP_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/init_value_codelets.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/prod_env_mat_codelets.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/prod_force_codelets.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/prod_virial_codelets.cpp
  ASM_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/init_mem.S
)

set(HOST_CPP_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/prod_env_mat.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/prod_force.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/prod_virial.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/jsoncpp.cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -fPIC")

add_library(${PROJECT_NAME} SHARED ${HOST_CPP_SOURCES})
target_link_libraries(${PROJECT_NAME} poplar)
target_link_libraries(${PROJECT_NAME} popops)
target_link_libraries(${PROJECT_NAME} poputil)
 
install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_LIBDIR})