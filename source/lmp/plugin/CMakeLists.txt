if(DEFINED LAMMPS_SOURCE_ROOT OR DEFINED LAMMPS_VERSION)
  message(STATUS "enable LAMMPS plugin mode")
  add_library(lammps_interface INTERFACE)
  if(DEFINED LAMMPS_VERSION)
    cmake_minimum_required(VERSION 3.11)
    include(FetchContent)
    FetchContent_Declare(
      lammps_download
      GIT_REPOSITORY https://github.com/lammps/lammps
      GIT_TAG ${LAMMPS_VERSION})
    FetchContent_GetProperties(lammps_download)
    if(NOT lammps_download_POPULATED)
      FetchContent_Populate(lammps_download)
      set(LAMMPS_SOURCE_ROOT ${lammps_download_SOURCE_DIR})
    endif()
  endif()
  set(LAMMPS_HEADER_DIR ${LAMMPS_SOURCE_ROOT}/src)
  message(STATUS "LAMMPS_HEADER_DIR is ${LAMMPS_HEADER_DIR}")

  target_include_directories(lammps_interface INTERFACE ${LAMMPS_HEADER_DIR})

  if("$ENV{DP_USE_MPICH2}" STREQUAL "1")
    # See https://stackoverflow.com/a/47976518/9567349
    set(MPI_EXECUTABLE_SUFFIX ".mpich")
  endif()

  find_package(MPI)
  if(MPI_FOUND)
    set(LAMMPS_MPI_INCLUDE_DIRS ${MPI_CXX_INCLUDE_DIRS})
    # LAMMPS has linked MPI; do not link twice
    # target_link_libraries(lammps_interface INTERFACE MPI::MPI_CXX)
    target_include_directories(lammps_interface
                               INTERFACE ${LAMMPS_MPI_INCLUDE_DIRS})
  else()
    # Use LAMMPS serial mpi.h header
    target_include_directories(lammps_interface
                               INTERFACE "${LAMMPS_HEADER_DIR}/STUBS")
  endif()

  # get_lammps_version
  # https://github.com/lammps/lammps/blob/c2a12f97c5f665852fb38fdd4922f7dd2e77a0a1/cmake/Modules/LAMMPSUtils.cmake#L27-L46
  include(${LAMMPS_SOURCE_ROOT}/cmake/Modules/LAMMPSUtils.cmake)
  get_lammps_version(${LAMMPS_HEADER_DIR}/version.h LAMMPS_VERSION_NUMBER)
  set(LAMMPS_VERSION_NUMBER
      ${LAMMPS_VERSION_NUMBER}
      PARENT_SCOPE)
  message(STATUS "LAMMPS version is ${LAMMPS_VERSION_NUMBER}")

  configure_file("../deepmd_version.h.in"
                 "${CMAKE_CURRENT_BINARY_DIR}/deepmd_version.h" @ONLY)

  file(
    GLOB
    LMP_SRC
    deepmdplugin.cpp
    ../*.cpp
    ${LAMMPS_SOURCE_ROOT}/src/KSPACE/pppm.cpp # for pppm_dplr
    ${LAMMPS_SOURCE_ROOT}/src/KSPACE/fft3d.cpp
    ${LAMMPS_SOURCE_ROOT}/src/KSPACE/fft3d_wrap.cpp
    ${LAMMPS_SOURCE_ROOT}/src/KSPACE/remap.cpp
    ${LAMMPS_SOURCE_ROOT}/src/KSPACE/remap_wrap.cpp)
  if(LAMMPS_VERSION_NUMBER GREATER 20210831)
    list(APPEND LMP_SRC ${LAMMPS_SOURCE_ROOT}/src/EXTRA-FIX/fix_ttm.cpp
    )# for ttm
  endif()

  set(libname "deepmd_lmp")
  if(BUILD_PY_IF)
    set(libname "${libname}plugin")
  endif()

  add_library(${libname} MODULE ${LMP_SRC})

  # link: libdeepmd
  if(DP_USING_C_API)
    target_link_libraries(${libname} PUBLIC ${LIB_DEEPMD_C})
    target_precompile_headers(${libname} PUBLIC [["deepmd.hpp"]])
    remove_definitions(-D_GLIBCXX_USE_CXX11_ABI=${OP_CXX_ABI})
    if("$ENV{CIBUILDWHEEL}" STREQUAL "1" OR "$ENV{LMP_CXX11_ABI_0}" STREQUAL
                                            "1")
      add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
    endif()
  else()
    target_link_libraries(${libname} PUBLIC ${LIB_DEEPMD_CC})
    target_compile_definitions(${libname} PUBLIC "DP_USE_CXX_API")
  endif()
  target_link_libraries(${libname} PUBLIC lammps_interface)
  target_include_directories(
    ${libname}
    PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/..
            ${LAMMPS_SOURCE_ROOT}/src/PLUGIN ${LAMMPS_SOURCE_ROOT}/src/KSPACE
            ${LAMMPS_SOURCE_ROOT}/src)
  if(LAMMPS_VERSION_NUMBER GREATER 20210831)
    target_include_directories(
      ${libname} PRIVATE ${LAMMPS_SOURCE_ROOT}/src/EXTRA-FIX) # for ttm
  endif()
  if(CMAKE_SYSTEM_NAME STREQUAL Darwin)
    set_target_properties(${libname} PROPERTIES LINK_FLAGS
                                                "-Wl,-undefined,dynamic_lookup")
  else()
    set_target_properties(
      ${libname} PROPERTIES INSTALL_RPATH "$ORIGIN;${TensorFlow_LIBRARY_PATH}"
                            LINK_FLAGS "-rdynamic")
  endif()
  target_compile_definitions(
    ${libname}
    PRIVATE "LMPPLUGIN" # fix header path
    PRIVATE "LAMMPS_VERSION_NUMBER=${LAMMPS_VERSION_NUMBER}")

  if(CMAKE_TESTING_ENABLED)
    target_link_libraries(${libname} PRIVATE coverage_config)
  endif()

  if(BUILD_PY_IF)
    install(TARGETS ${libname} DESTINATION deepmd/lib/)
  else(BUILD_PY_IF)
    install(TARGETS ${libname} DESTINATION lib/)

    if(${LAMMPS_VERSION_NUMBER} GREATER_EQUAL 20220324)
      set(PLUGINNAME "dpplugin.so")
      install(
        CODE "execute_process( \
        COMMAND ${CMAKE_COMMAND} -E make_directory \
		${CMAKE_INSTALL_PREFIX}/lib/${libname}/   \
        )")
      install(
        CODE "execute_process( \
        COMMAND ${CMAKE_COMMAND} -E create_symlink \
		../${CMAKE_SHARED_LIBRARY_PREFIX}${libname}${CMAKE_SHARED_LIBRARY_SUFFIX} \
        ${CMAKE_INSTALL_PREFIX}/lib/${libname}/${PLUGINNAME}   \
        )")
    endif()
  endif(BUILD_PY_IF)

else()
  message(STATUS "disable LAMMPS plugin mode")
endif()
