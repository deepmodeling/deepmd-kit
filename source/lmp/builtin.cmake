# This file should be included in the end of
# ${LAMMPS_SOURCE_DIR}/cmake/CMakeLists.txt
# include(/path/to/deepmd_source/source/lmp/builtin.cmake)

# assume LAMMPS CMake file has been executed, so these target/variables exist:
# lammps LAMMPS_SOURCE_DIR get_lammps_version

get_lammps_version(${LAMMPS_SOURCE_DIR}/version.h LAMMPS_VERSION_NUMBER)

configure_file("${CMAKE_CURRENT_LIST_DIR}/deepmd_version.h.in"
               "${CMAKE_CURRENT_BINARY_DIR}/deepmd_version.h" @ONLY)

file(GLOB DEEPMD_LMP_SRC ${CMAKE_CURRENT_LIST_DIR}/*.cpp)

find_package(DeePMD REQUIRED)
target_sources(
  lammps
  PRIVATE ${DEEPMD_LMP_SRC}
          ${LAMMPS_SOURCE_DIR}/KSPACE/pppm.cpp # for pppm_dplr
          ${LAMMPS_SOURCE_DIR}/KSPACE/fft3d.cpp
          ${LAMMPS_SOURCE_DIR}/KSPACE/fft3d_wrap.cpp
          ${LAMMPS_SOURCE_DIR}/KSPACE/remap.cpp
          ${LAMMPS_SOURCE_DIR}/KSPACE/remap_wrap.cpp
          ${LAMMPS_SOURCE_DIR}/EXTRA-FIX/fix_ttm.cpp # for ttm
)
target_link_libraries(lammps PUBLIC DeePMD::deepmd_c)
target_include_directories(
  lammps PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_LIST_DIR}
                 ${LAMMPS_SOURCE_DIR}/KSPACE ${LAMMPS_SOURCE_DIR}/EXTRA-FIX)
target_compile_definitions(
  lammps PRIVATE "LAMMPS_VERSION_NUMBER=${LAMMPS_VERSION_NUMBER}")

# register styles
registerstyles(${CMAKE_CURRENT_LIST_DIR})
generatestyleheaders(${LAMMPS_STYLE_HEADERS_DIR})
