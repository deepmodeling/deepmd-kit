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

function(_deepmd_lammps_link_torch target_name)
  if(NOT DeePMD_ENABLE_PYTORCH)
    return()
  endif()

  set(_deepmd_torch_from_recorded_dir FALSE)
  if(Torch_FOUND)
    # A parent project may already have loaded Torch. Validate it below rather
    # than attempting to redefine its imported targets from another package.
  elseif(DeePMD_TORCH_DIR)
    find_package(
      Torch
      CONFIG
      QUIET
      PATHS "${DeePMD_TORCH_DIR}"
      NO_DEFAULT_PATH)
    if(Torch_FOUND)
      set(_deepmd_torch_from_recorded_dir TRUE)
    endif()
  endif()

  if(NOT Torch_FOUND)
    # The recorded package may have moved or been removed. A caller-provided
    # installation is acceptable only after the compatibility checks below.
    unset(Torch_DIR CACHE)
    unset(Torch_DIR)
    find_package(Torch CONFIG QUIET)
  endif()

  if(NOT Torch_FOUND)
    message(
      WARNING
        "DeePMD-kit was built with the PyTorch backend, but Torch was not "
        "found while configuring LAMMPS. Install the same Torch package or "
        "set Torch_DIR to its CMake configuration directory.")
    return()
  endif()

  string(REGEX MATCH "_GLIBCXX_USE_CXX11_ABI=([0-9]+)"
               _deepmd_torch_abi_match "${TORCH_CXX_FLAGS}")
  set(_deepmd_torch_abi "")
  if(_deepmd_torch_abi_match)
    set(_deepmd_torch_abi "${CMAKE_MATCH_1}")
  elseif(UNIX AND NOT APPLE AND Torch_VERSION VERSION_GREATER_EQUAL "2.8.0")
    # Recent Linux Torch packages no longer publish the ABI in
    # TORCH_CXX_FLAGS and use the C++11 ABI unconditionally.
    set(_deepmd_torch_abi "1")
  else()
    # Match DeePMD's build-time fallback for platforms and older packages that
    # do not expose a libstdc++ ABI flag.
    set(_deepmd_torch_abi "0")
  endif()

  if(NOT _deepmd_torch_from_recorded_dir
     AND ("${DeePMD_TORCH_VERSION}" STREQUAL ""
          OR "${DeePMD_TORCH_CXX11_ABI}" STREQUAL ""))
    message(
      WARNING
        "DeePMD-kit did not record enough Torch compatibility metadata to "
        "validate a different installation; refusing to link it into LAMMPS.")
    return()
  endif()
  if(NOT "${DeePMD_TORCH_VERSION}" STREQUAL ""
     AND NOT "${Torch_VERSION}" STREQUAL "${DeePMD_TORCH_VERSION}")
    message(
      FATAL_ERROR
        "Torch version mismatch: DeePMD-kit was built with "
        "${DeePMD_TORCH_VERSION}, but LAMMPS found ${Torch_VERSION}.")
  endif()
  if(NOT "${DeePMD_TORCH_CXX11_ABI}" STREQUAL ""
     AND NOT "${_deepmd_torch_abi}" STREQUAL "${DeePMD_TORCH_CXX11_ABI}")
    message(
      FATAL_ERROR
        "Torch C++ ABI mismatch: DeePMD-kit was built with "
        "_GLIBCXX_USE_CXX11_ABI=${DeePMD_TORCH_CXX11_ABI}, but LAMMPS found "
        "${_deepmd_torch_abi}.")
  endif()

  # LAMMPS exports this target as LAMMPS::lammps. Torch is needed to resolve
  # the in-tree executable's transitive DeePMD symbols, but must not leak
  # imported targets or absolute paths into the installed LAMMPS interface.
  target_link_libraries(${target_name}
                        PUBLIC "$<BUILD_INTERFACE:${TORCH_LIBRARIES}>")
endfunction()

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
_deepmd_lammps_link_torch(lammps)
target_include_directories(
  lammps PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_LIST_DIR}
                 ${LAMMPS_SOURCE_DIR}/KSPACE ${LAMMPS_SOURCE_DIR}/EXTRA-FIX)
target_compile_definitions(
  lammps PRIVATE "LAMMPS_VERSION_NUMBER=${LAMMPS_VERSION_NUMBER}")

# register styles
registerstyles(${CMAKE_CURRENT_LIST_DIR})
generatestyleheaders(${LAMMPS_STYLE_HEADERS_DIR})
