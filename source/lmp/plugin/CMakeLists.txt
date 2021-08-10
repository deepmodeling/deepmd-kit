if (DEFINED LAMMPS_SOURCE_ROOT OR DEFINED LAMMPS_VERSION)
  add_library(lammps_interface INTERFACE)
  if (DEFINED LAMMPS_VERSION)
    include(ExternalProject)
    ExternalProject_Add(lammps_download
      GIT_REPOSITORY    https://github.com/lammps/lammps
      GIT_TAG           ${LAMMPS_VERSION}
      SOURCE_DIR        "${CMAKE_CURRENT_BINARY_DIR}/lammps-src"
      BINARY_DIR        "${CMAKE_CURRENT_BINARY_DIR}/lammps-build"
      CONFIGURE_COMMAND ""
      BUILD_COMMAND     ""
      INSTALL_COMMAND   ""
      TEST_COMMAND      ""
    )
    set(LAMMPS_SOURCE_ROOT ${CMAKE_CURRENT_BINARY_DIR}/lammps-src)
  endif()
  set(LAMMPS_HEADER_DIR ${LAMMPS_SOURCE_ROOT}/src)
  message(STATUS "LAMMPS_HEADER_DIR is ${LAMMPS_HEADER_DIR}")
  
  target_include_directories(lammps_interface INTERFACE ${LAMMPS_HEADER_DIR})

  find_package(MPI)
  if(MPI_FOUND)
    set(LAMMPS_MPI_INCLUDE_DIRS ${MPI_CXX_INCLUDE_DIRS})
    target_link_libraries(lammps_interface INTERFACE MPI::MPI_CXX)
  else()
    # Use LAMMPS serial mpi.h header
    target_include_directories(lammps_interface INTERFACE "${LAMMPS_HEADER_DIR}/STUBS")
  endif()

  configure_file("../pair_deepmd.h.in" "${CMAKE_CURRENT_BINARY_DIR}/pair_deepmd.h" @ONLY)

  file(GLOB LMP_SRC
    deepmdplugin.cpp
    ../*.cpp
    ${LAMMPS_SOURCE_ROOT}/src/kspace.cpp # for pppm_dplr
    ${LAMMPS_SOURCE_ROOT}/src/KSPACE/pppm.cpp
  )

  function(_add_lmp_plugin_variant variant_name prec_def)
  set (libname "deepmd_lmp${variant_name}")

  add_library(${libname} MODULE ${LMP_SRC})

  # link: libdeepmd libdeepmd_op libtensorflow_cc libtensorflow_framework
  target_link_libraries (${libname} PUBLIC
    lammps_interface
    ${LIB_DEEPMD_CC}${variant_name}
    ${TensorFlow_LIBRARY}
    ${TensorFlowFramework_LIBRARY}
  )
  target_include_directories(${libname} PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${TensorFlow_INCLUDE_DIRS}
    ${LAMMPS_SOURCE_ROOT}/src/PLUGIN
    ${LAMMPS_SOURCE_ROOT}/src/KSPACE
    ${LAMMPS_SOURCE_ROOT}/src
  )

  set_target_properties(
    ${libname} 
    PROPERTIES 
    COMPILE_DEFINITIONS ${prec_def}
    COMPILE_DEFINITIONS "LMPPLUGIN" # fix header path
    INSTALL_RPATH "$ORIGIN;${TensorFlow_LIBRARY_PATH}"
    LINK_FLAGS "-rdynamic"
  )

  install(TARGETS ${libname} DESTINATION lib/)

  endfunction()
  _add_lmp_plugin_variant("${HIGH_PREC_VARIANT}" "${HIGH_PREC_DEF}")
  _add_lmp_plugin_variant("${LOW_PREC_VARIANT}" "${LOW_PREC_DEF}")

endif()