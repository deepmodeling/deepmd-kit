if (DEFINED LAMMPS_SOURCE_ROOT OR DEFINED LAMMPS_VERSION)
  message(STATUS "enable LAMMPS plugin mode")
  add_library(lammps_interface INTERFACE)
  if (DEFINED LAMMPS_VERSION)
    cmake_minimum_required(VERSION 3.11)
    include(FetchContent)
    FetchContent_Declare(lammps_download
      GIT_REPOSITORY    https://github.com/lammps/lammps
      GIT_TAG           ${LAMMPS_VERSION}
    )
    FetchContent_GetProperties(lammps_download)
    if(NOT lammps_download_POPULATED)
      FetchContent_Populate(lammps_download)
      set(LAMMPS_SOURCE_ROOT ${lammps_download_SOURCE_DIR})
    endif()
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

  # get_lammps_version
  # https://github.com/lammps/lammps/blob/c2a12f97c5f665852fb38fdd4922f7dd2e77a0a1/cmake/Modules/LAMMPSUtils.cmake#L27-L46
  include(${LAMMPS_SOURCE_ROOT}/cmake/Modules/LAMMPSUtils.cmake)
  get_lammps_version(${LAMMPS_HEADER_DIR}/version.h LAMMPS_VERSION_NUMBER)
  set(LAMMPS_VERSION_NUMBER ${LAMMPS_VERSION_NUMBER} PARENT_SCOPE)

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

  # link: libdeepmd libtensorflow_cc libtensorflow_framework
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
    INSTALL_RPATH "$ORIGIN;${TensorFlow_LIBRARY_PATH}"
    LINK_FLAGS "-rdynamic"
  )
  target_compile_definitions(${libname}
    PUBLIC ${prec_def}
    PRIVATE "LMPPLUGIN" # fix header path
    PRIVATE "LAMMPS_VERSION_NUMBER=${LAMMPS_VERSION_NUMBER}"
  )

  install(TARGETS ${libname} DESTINATION lib/)

  endfunction()
  _add_lmp_plugin_variant("${HIGH_PREC_VARIANT}" "${HIGH_PREC_DEF}")
  _add_lmp_plugin_variant("${LOW_PREC_VARIANT}" "${LOW_PREC_DEF}")

else()
  message(STATUS "disable LAMMPS plugin mode")
endif()