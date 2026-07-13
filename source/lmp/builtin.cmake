# This file should be included in the end of
# ${LAMMPS_SOURCE_DIR}/cmake/CMakeLists.txt
# include(/path/to/deepmd_source/source/lmp/builtin.cmake)

# assume LAMMPS CMake file has been executed, so these target/variables exist:
# lammps LAMMPS_SOURCE_DIR get_lammps_version

# Since May 15, 2025, the output of get_lammps_version is changed. We vendor the
# old get_lammps_version
# https://github.com/lammps/lammps/commit/b3e7121535863df3db487cd3e6a68c080bf2a6b4#diff-1214db0d1c015a50103f61f8ff7896053dec7ebc1edb930d6ef8bb07282f52abR75
function(_get_lammps_version version_header variable)
  file(STRINGS ${version_header} line REGEX LAMMPS_VERSION)
  set(MONTHS
      x
      Jan
      Feb
      Mar
      Apr
      May
      Jun
      Jul
      Aug
      Sep
      Oct
      Nov
      Dec)
  string(REGEX
         REPLACE "#define LAMMPS_VERSION \"([0-9]+) ([A-Za-z]+) ([0-9]+)\""
                 "\\1" day "${line}")
  string(REGEX
         REPLACE "#define LAMMPS_VERSION \"([0-9]+) ([A-Za-z]+) ([0-9]+)\""
                 "\\2" month "${line}")
  string(REGEX
         REPLACE "#define LAMMPS_VERSION \"([0-9]+) ([A-Za-z]+) ([0-9]+)\""
                 "\\3" year "${line}")
  string(STRIP ${day} day)
  string(STRIP ${month} month)
  string(STRIP ${year} year)
  list(FIND MONTHS "${month}" month)
  string(LENGTH ${day} day_length)
  string(LENGTH ${month} month_length)
  if(day_length EQUAL 1)
    set(day "0${day}")
  endif()
  if(month_length EQUAL 1)
    set(month "0${month}")
  endif()
  set(${variable}
      "${year}${month}${day}"
      PARENT_SCOPE)
endfunction()

_get_lammps_version(${LAMMPS_SOURCE_DIR}/version.h LAMMPS_VERSION_NUMBER)

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
