#
# This file is part of the GROMACS molecular simulation package.
#
# Copyright (c) 2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019, by the GROMACS development team, led by
# Mark Abraham, David van der Spoel, Berk Hess, and Erik Lindahl,
# and including many others, as listed in the AUTHORS file in the
# top-level source directory and at http://www.gromacs.org.
#
# GROMACS is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License
# as published by the Free Software Foundation; either version 2.1
# of the License, or (at your option) any later version.
#
# GROMACS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with GROMACS; if not, see
# http://www.gnu.org/licenses, or write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
#
# If you want to redistribute modifications to GROMACS, please
# consider that scientific software is very special. Version
# control is crucial - bugs must be traceable. We will be happy to
# consider code for inclusion in the official distribution, but
# derived work must not be called official GROMACS. Details are found
# in the README & COPYING files - if they are missing, get the
# official version at http://www.gromacs.org.
#
# To help us fund GROMACS development, we humbly ask that you cite
# the research papers on the package. Check out http://www.gromacs.org.

cmake_minimum_required(VERSION 3.9.6)
if(POLICY CMP0074) #3.12
    cmake_policy(SET CMP0074 NEW)
endif()
cmake_policy(SET CMP0068 NEW) # From CMake-3.9

# CMake modules/macros are in a subdirectory to keep this file cleaner
# This needs to be set before project() in order to pick up toolchain files
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Platform)

if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    # Providing a default value >=10.9 helps to find modern C++ compatibility,
    # such as by defaulting to the Clang libc++ instead of libstdc++.
    set(CMAKE_OSX_DEPLOYMENT_TARGET 10.9 CACHE STRING
        "OS X deployment target affects default SDK version and compiler flags."
        FORCE)
    # By default, limit the binary architecture to a single 64-bit build.
    set(CMAKE_OSX_ARCHITECTURES x86_64 CACHE STRING
        "OS X architecture affects the compatibility of the (potentially fat) binaries produced."
        FORCE)
endif()

project(Gromacs)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(gmxManageCcache)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(LibStdCpp)

# Set up common version variables, as well as general information about
# the build tree (whether the build is from a source package or from a git
# repository).  Also declares a few functions that will be used for generating
# version info files later.
include(gmxBuildTreeInfo)
include(gmxVersionInfo)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT AND UNIX)
    set(CMAKE_INSTALL_PREFIX "/usr/local/gromacs" CACHE STRING "Installation prefix (installation will need write permissions here)" FORCE)
endif()
if("${CMAKE_INSTALL_PREFIX}" STREQUAL "${CMAKE_BINARY_DIR}")
    message(FATAL_ERROR "GROMACS cannot be installed into the build tree, choose a different location for CMAKE_INSTALL_PREFIX")
endif()

include(gmxBuildTypeReference)
include(gmxBuildTypeProfile)
include(gmxBuildTypeTSAN)
include(gmxBuildTypeASAN)
include(gmxBuildTypeMSAN)
include(gmxBuildTypeReleaseWithAssert)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel Reference RelWithAssert Profile TSAN ASAN MSAN." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
        "MinSizeRel" "RelWithDebInfo" "Reference" "RelWithAssert" "Profile" "TSAN" "ASAN" "MSAN")
endif()
if(CMAKE_CONFIGURATION_TYPES)
    # Add appropriate GROMACS-specific build types for the Visual
    # Studio generator (Debug, Release, MinSizeRel and RelWithDebInfo
    # are already present by default).
    list(APPEND CMAKE_CONFIGURATION_TYPES "RelWithAssert" "Reference")
    list(REMOVE_DUPLICATES CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_CONFIGURATION_TYPES "${CMAKE_CONFIGURATION_TYPES}" CACHE STRING
        "List of configuration types"
        FORCE)
endif()
set(build_types_with_explicit_flags RELEASE DEBUG RELWITHDEBINFO RELWITHASSERT MINSIZEREL PROFILE)

set_property(GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS ON)

include(gmxCTestUtilities)
gmx_ctest_init()

include(gmxCPackUtilities)
gmx_cpack_init()

# Variables that accumulate stuff influencing the installed headers
set(INSTALLED_HEADER_INCLUDE_DIRS "")
set(INSTALLED_HEADER_DEFINITIONS "")

########################################################################
# Global non-cache variables for implementing the build system
########################################################################

# These variables collect libraries that GROMACS requires for
# linking. They should be appended to with list(APPEND ${name}
# new-library) calls. They are:
#  - Libraries that are required for libgromacs (only)
set(GMX_EXTRA_LIBRARIES "")
#  - Libraries that are required for all code in the repository
set(GMX_COMMON_LIBRARIES "")
#  - Libraries that all code linked against libgromacs needs
#    (i.e., something that is exposed in installed headers).
set(GMX_PUBLIC_LIBRARIES "")

# DeepMD
message(STATUS "Compling with DeepMD...")
add_definitions(-w) # close warning
include_directories(${GMX_DEEPMD_ROOT}/include)
include_directories(${GMX_TENSORFLOW_ROOT}/include)
link_directories(${GMX_DEEPMD_ROOT}/lib)
link_directories(${GMX_TENSORFLOW_ROOT}/lib)
# define high precision
add_definitions("-D HIGH_PREC")
# ensure all dynamic libraries linked
set (CMAKE_EXE_LINKER_FLAGS "-Wl,--no-as-needed")
# add link libraries
list (APPEND GMX_PUBLIC_LIBRARIES deepmd_op)
list (APPEND GMX_PUBLIC_LIBRARIES deepmd_op_cuda)
list (APPEND GMX_PUBLIC_LIBRARIES deepmd)
list (APPEND GMX_PUBLIC_LIBRARIES deepmd_cc)
list (APPEND GMX_PUBLIC_LIBRARIES tensorflow_cc)
list (APPEND GMX_PUBLIC_LIBRARIES tensorflow_framework)
# DeepMD

########################################################################
# Check and warn if cache generated on a different host is being reused
########################################################################
if(CMAKE_HOST_UNIX)
    execute_process(COMMAND hostname
                    OUTPUT_VARIABLE TMP_HOSTNAME
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    # Only check for host name if not running in a CI environment, as the cache might
    # be reused there between different machines in different stages
    if(GMX_BUILD_HOSTNAME AND NOT "${GMX_BUILD_HOSTNAME}" STREQUAL "${TMP_HOSTNAME}"
            AND NOT DEFINED ENV{CI_JOB_ID})
        message(WARNING "
            The CMake cache, probably generated on a different host (${GMX_BUILD_HOSTNAME}),
            is being reused! This could lead to inconsistencies; therefore, it is
            recommended to regenerate the cache!")
    endif()
    set(GMX_BUILD_HOSTNAME "${TMP_HOSTNAME}" CACHE INTERNAL
            "Hostname of the machine where the cache was generated.")
endif()

########################################################################
# Detect architecture before setting options so we can alter defaults
########################################################################
# Detect the architecture the compiler is targetting, detect
# SIMD instructions possibilities on that hardware, suggest SIMD instruction set
# to use if none is specified, and populate the cache option for CPU
# SIMD.
include(gmxDetectTargetArchitecture)
gmx_detect_target_architecture()

########################################################################
# User input options                                                   #
########################################################################
include(gmxOptionUtilities)

set(CMAKE_PREFIX_PATH "" CACHE STRING "Extra locations to search for external libraries and tools (give directory without lib, bin, or include)")

# Fujitsu only has SIMD in double precision, so this will be faster
gmx_set_boolean(GMX_DOUBLE_DEFAULT GMX_TARGET_FUJITSU_SPARC64)
option(GMX_DOUBLE "Use double precision (much slower, use only if you really need it)" ${GMX_DOUBLE_DEFAULT})
option(GMX_RELAXED_DOUBLE_PRECISION "Accept single precision 1/sqrt(x) when using Fujitsu HPC-ACE SIMD" OFF)
mark_as_advanced(GMX_RELAXED_DOUBLE_PRECISION)

option(GMX_MPI    "Build a parallel (message-passing) version of GROMACS" OFF)
option(GMX_THREAD_MPI  "Build a thread-MPI-based multithreaded version of GROMACS (not compatible with MPI)" ON)
gmx_dependent_option(
    GMX_MPI_IN_PLACE
    "Enable MPI_IN_PLACE for MPIs that have it defined"
    ON
    GMX_MPI)
mark_as_advanced(GMX_MPI_IN_PLACE)

option(GMX_MIMIC "Enable MiMiC QM/MM interface (CPMD is required)" OFF)

option(GMX_FAHCORE "Build a library with mdrun functionality" OFF)
mark_as_advanced(GMX_FAHCORE)

option(GMX_COOL_QUOTES "Enable GROMACS cool quotes" ON)
mark_as_advanced(GMX_COOL_QUOTES)
gmx_add_cache_dependency(GMX_COOL_QUOTES BOOL "NOT GMX_FAHCORE" OFF)

option(GMX_USE_OPENCL "Enable OpenCL acceleration" OFF)

option(GMX_INSTALL_LEGACY_API "Install legacy headers" OFF)

# The earliest version of the CUDA toolkit that supports c++14 is 9.0
set(REQUIRED_CUDA_VERSION 9.0)
set(REQUIRED_CUDA_COMPUTE_CAPABILITY 3.0)

# OpenCL required version: 1.2 or newer
set(REQUIRED_OPENCL_MIN_VERSION_MAJOR 1)
set(REQUIRED_OPENCL_MIN_VERSION_MINOR 2)
set(REQUIRED_OPENCL_MIN_VERSION ${REQUIRED_OPENCL_MIN_VERSION_MAJOR}.${REQUIRED_OPENCL_MIN_VERSION_MINOR})

if(NOT GMX_USE_OPENCL)
    # CUDA detection is done only if GMX_USE_OPENCL is OFF.
    include(gmxManageGPU)
    set(GMX_USE_CUDA ${GMX_GPU})
    if(GMX_GPU)
        set(GMX_GPU_ACCELERATION_FRAMEWORK "GMX_GPU_CUDA")
    else()
        set(GMX_GPU_ACCELERATION_FRAMEWORK "GMX_GPU_NONE")
    endif()
else()
    #Now the OpenCL path (for both AMD and NVIDIA)
    if(GMX_GPU)
        include(gmxManageOpenCL)
        set(GMX_GPU_ACCELERATION_FRAMEWORK "GMX_GPU_OPENCL")
    else()
        message(FATAL_ERROR "OpenCL requested but GPU option is not enabled (try -DGMX_GPU=on) ")
    endif()
endif()

gmx_option_multichoice(
    GMX_SIMD
    "SIMD instruction set for CPU kernels and compiler optimization"
    "AUTO"
    AUTO None SSE2 SSE4.1 AVX_128_FMA AVX_256 AVX2_256 AVX2_128 AVX_512 AVX_512_KNL MIC ARM_NEON ARM_NEON_ASIMD IBM_VMX IBM_VSX Sparc64_HPC_ACE Reference)

if(GMX_TARGET_MIC)
    set(GMX_FFT_LIBRARY_DEFAULT "mkl")
else()
    set(GMX_FFT_LIBRARY_DEFAULT "fftw3")
endif()

gmx_option_multichoice(
    GMX_FFT_LIBRARY
    "FFT library"
    "${GMX_FFT_LIBRARY_DEFAULT}"
    fftw3 mkl "fftpack[built-in]")
gmx_dependent_option(
    GMX_BUILD_OWN_FFTW
    "Download and build FFTW 3 during the GROMACS build process, rather than fall back on the really slow fftpack."
    OFF
    "GMX_FFT_LIBRARY STREQUAL FFTW3")
gmx_dependent_option(
    GMX_DISABLE_FFTW_MEASURE
    "Do not optimize FFTW setups (not needed with SSE)"
    OFF
    "GMX_FFT_LIBRARY STREQUAL FFTW3")
mark_as_advanced(GMX_BUILD_OWN_FFTW)
mark_as_advanced(GMX_DISABLE_FFTW_MEASURE)

gmx_option_multichoice(
    GMX_QMMM_PROGRAM
    "QM package for QM/MM"
    None
    none gaussian mopac gamess orca)

gmx_dependent_cache_variable(GMX_SIMD_REF_FLOAT_WIDTH  "Reference SIMD single precision width" STRING "4" "GMX_SIMD STREQUAL REFERENCE")
gmx_dependent_cache_variable(GMX_SIMD_REF_DOUBLE_WIDTH "Reference SIMD double precision width" STRING "2" "GMX_SIMD STREQUAL REFERENCE")

# This should be moved to a separate NBNXN cmake module when that code is cleaned up and modularized

option(GMX_BROKEN_CALLOC "Work around broken calloc()" OFF)
mark_as_advanced(GMX_BROKEN_CALLOC)

option(GMX_OPENMP "Enable OpenMP-based multithreading" ON)

option(GMX_USE_TNG "Use the TNG library for trajectory I/O" ON)

option(GMX_BUILD_MDRUN_ONLY "Build and install only the mdrun binary" OFF)

option(GMX_CYCLE_SUBCOUNTERS "Enable cycle subcounters to get a more detailed cycle timings" OFF)
mark_as_advanced(GMX_CYCLE_SUBCOUNTERS)

option(GMX_SKIP_DEFAULT_CFLAGS "Don't automatically add suggested/required Compiler flags." OFF)
mark_as_advanced(GMX_SKIP_DEFAULT_CFLAGS)

option(GMX_BUILD_FOR_COVERAGE
       "Tune build for better code coverage metrics (e.g., disable asserts)"
       OFF)
mark_as_advanced(GMX_BUILD_FOR_COVERAGE)

option(GMX_DEVELOPER_BUILD
    "Enable Developer convenience features: always build unit-tests"
    OFF)
mark_as_advanced(GMX_DEVELOPER_BUILD)

gmx_set_boolean(GMX_COMPILER_WARNINGS_DEFAULT "NOT SOURCE_IS_SOURCE_DISTRIBUTION")
option(GMX_COMPILER_WARNINGS
    "Enable a default set of compiler warnings"
    ${GMX_COMPILER_WARNINGS_DEFAULT})
mark_as_advanced(GMX_COMPILER_WARNINGS)
# Always turn on compiler warnings with a developer build.
gmx_add_cache_dependency(GMX_COMPILER_WARNINGS BOOL "NOT GMX_DEVELOPER_BUILD" ON)

option(GMX_BUILD_SHARED_EXE
    "Build exectuables as shared binaries. If not set, this disables rpath and dynamic linker flags in an attempt to build a static binary, but this may require setting up the toolchain properly and making appropriate libraries available."
    ON)
mark_as_advanced(GMX_BUILD_SHARED_EXE)

option(GMX_PHYSICAL_VALIDATION
       "Include physical validation tests in ctest environment. These can then be called using 'make check-phys' or
       'make check-all'. Warning: Running the physical validation tests takes significantly more time than other tests!"
       OFF)
mark_as_advanced(GMX_PHYSICAL_VALIDATION)

######################################################################
# Detect OpenMP support
######################################################################
# The OpenMP detection _must_ come before tests for other CFLAGS.
include(gmxManageOpenMP)



######################################################################
# Compiler tests
# These need to be done early (before further tests).
#####################################################################

include(gmxCFlags)
gmx_c_flags()

# These variables should be used for CMake-style lists (ie. separated
# by semicolons) of additional compiler flags which are not generated
# in gmxCFlags nor are SIMD or MPI related.
#
# TODO These variables should be consolidated into
# EXTRA_COMPILER_FLAGS so that we we don't perpetrate bugs where
# things that work in C compilation (e.g. merging from old branches)
# might not also work for C++ compilation.
set(EXTRA_C_FLAGS "")
set(EXTRA_CXX_FLAGS "")

# Run through a number of tests for buggy compilers and other issues
include(gmxTestCompilerProblems)
gmx_test_compiler_problems()

# Implement double-precision option. This is complicated because we
# need installed headers to use the precision mode of the build that
# produced the library, but cannot use config.h in that case. We also
# want such variables to always have a definition, because #if is more
# robust than #ifdef. So, we put this value on the compiler command
# line in all cases.
#
# GMX_RELAXED_DOUBLE_PRECISION does not need to be handled here,
# because no installed header needs it
if(GMX_DOUBLE)
    set(GMX_DOUBLE_VALUE 1)
else()
    set(GMX_DOUBLE_VALUE 0)
endif()
add_definitions(-DGMX_DOUBLE=${GMX_DOUBLE_VALUE})
list(APPEND INSTALLED_HEADER_DEFINITIONS "-DGMX_DOUBLE=${GMX_DOUBLE_VALUE}")

option(GMX_IMD "Enable Interactive Molecular Dynamics (IMD) sessions, e.g. with VMD" ON)
if(GMX_IMD AND WIN32)
    list(APPEND GMX_EXTRA_LIBRARIES "wsock32")
endif()



########################################################################
# Basic system tests (standard libraries, headers, functions, types)   #
########################################################################
include(CheckIncludeFiles)
include(CheckIncludeFileCXX)
check_include_files(unistd.h     HAVE_UNISTD_H)
check_include_files(pwd.h        HAVE_PWD_H)
check_include_files(dirent.h     HAVE_DIRENT_H)
check_include_files(time.h       HAVE_TIME_H)
check_include_files(sys/time.h   HAVE_SYS_TIME_H)
check_include_files(io.h         HAVE_IO_H)
check_include_files(sched.h      HAVE_SCHED_H)
check_include_files(xmmintrin.h  HAVE_XMMINTRIN_H)

include(CheckCXXSymbolExists)
check_cxx_symbol_exists(gettimeofday      sys/time.h   HAVE_GETTIMEOFDAY)
check_cxx_symbol_exists(sysconf           unistd.h     HAVE_SYSCONF)
check_cxx_symbol_exists(nice              unistd.h     HAVE_NICE)
check_cxx_symbol_exists(fsync             unistd.h     HAVE_FSYNC)
check_cxx_symbol_exists(_fileno           stdio.h      HAVE__FILENO)
check_cxx_symbol_exists(fileno            stdio.h      HAVE_FILENO)
check_cxx_symbol_exists(_commit           io.h         HAVE__COMMIT)
check_cxx_symbol_exists(sigaction         signal.h     HAVE_SIGACTION)

# We cannot check for the __builtins as symbols, but check if code compiles
check_cxx_source_compiles("int main(){ return __builtin_clz(1);}"   HAVE_BUILTIN_CLZ)
check_cxx_source_compiles("int main(){ return __builtin_clzll(1);}" HAVE_BUILTIN_CLZLL)
if(MSVC)
    check_cxx_source_compiles("#include <intrin.h>\n int main(){unsigned long r;unsigned long i=1;_BitScanReverse(&r,i);return r;}" HAVE_BITSCANREVERSE)
    check_cxx_source_compiles("#include <intrin.h>\n int main(){unsigned long r;unsigned __int64 i=1;_BitScanReverse(&r,i);return r;}" HAVE_BITSCANREVERSE64)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "XL")
    check_cxx_source_compiles("int main(){ return __cntlz4(1);}" HAVE_CNTLZ4)
    check_cxx_source_compiles("int main(){ return __cntlz8(1);}" HAVE_CNTLZ8)
endif()

include(CheckLibraryExists)
find_library(HAVE_LIBM m)
mark_as_advanced(HAVE_LIBM)
check_library_exists(rt clock_gettime "" HAVE_CLOCK_GETTIME)
check_library_exists(m feenableexcept "" HAVE_FEENABLEEXCEPT)
check_library_exists(m fedisableexcept "" HAVE_FEDISABLEEXCEPT)

include(TestSchedAffinity)
test_sched_affinity(HAVE_SCHED_AFFINITY)

# Aligned memory allocation. We need to check for both mm_malloc(),
# posix_memalign(), memalign(), and on windows also _aligned_malloc()
include(gmxTestMMMalloc)
gmx_test_mm_malloc(HAVE__MM_MALLOC)
check_cxx_symbol_exists(posix_memalign    stdlib.h     HAVE_POSIX_MEMALIGN)
check_cxx_symbol_exists(memalign          stdlib.h     HAVE_MEMALIGN)
if(MSVC)
    # No need to waste time on this test on platforms where it will never be true
    check_cxx_symbol_exists(_aligned_malloc   stdlib.h     HAVE__ALIGNED_MALLOC)
endif()

include(TestBigEndian)
test_big_endian(GMX_INTEGER_BIG_ENDIAN)

gmx_set_boolean(GMX_USE_NICE "HAVE_UNISTD_H AND HAVE_NICE")

# Management of GROMACS options for specific toolchains should go
# here. Because the initial settings for some of the main options have
# already happened, but things like library detection and MPI compiler
# feature detection have not, the docstrings for any over-rides of
# GROMACS defaults or user settings will make sense. Also, any
# toolchain-related reasons for choosing whether to detect various
# things can be sorted out now, before the detection takes place.
if(GMX_TARGET_FUJITSU_SPARC64)
    include(gmxManageFujitsuSparc64)
endif()

########################################################################
#Process MPI settings
########################################################################
include(gmxManageMPI)

########################################################################
#Process MiMiC settings
########################################################################
include(gmxManageMimic)

########################################################################
#Process shared/static library settings
########################################################################
include(gmxManageSharedLibraries)


########################################################################
# Find external packages                                               #
########################################################################

# Unconditionally find the package, as it is also required for unit
# tests. This exports LIBXML2_FOUND, which we should not use because
# it does not tell us that linking will succeed. Instead, we test that
# next.
#if(DEFINED LIBXML2_LIBRARIES)
#  set(LibXml2_FIND_QUIETLY TRUE)
#endif()
#find_package(LibXml2)
#include(gmxTestLibXml2)
#gmx_test_libxml2(HAVE_LIBXML2)
#option(GMX_XML "Use libxml2 to parse xml files (currently has no effect)" ${HAVE_LIBXML2})
#set(PKG_XML "")
#mark_as_advanced(GMX_XML)
# Don't actually do anything, since libxml2 is currently not used by libgromacs
#if(GMX_XML AND NOT HAVE_LIBXML2)
#    message(FATAL_ERROR "libxml2 not found. Set GMX_XML=OFF to compile without XML support")
#endif()
#if(GMX_XML)
#    include_directories(${LIBXML2_INCLUDE_DIR})
#    set(PKG_XML libxml-2.0)
#    set(XML_LIBRARIES ${LIBXML2_LIBRARIES})
#endif()

gmx_option_trivalue(
    GMX_HWLOC
    "Use hwloc portable hardware locality library"
    "AUTO")

if (GMX_HWLOC)
    # Find quietly the second time.
    if (HWLOC_FIND_QUIETLY_AFTER_FIRST_RUN)
        set(HWLOC_FIND_QUIETLY TRUE)
    endif()
    find_package(HWLOC 1.5) 

    if (HWLOC_FOUND)
        if (HWLOC_LIBRARIES MATCHES ".a$")
            set(_STATIC_HWLOC TRUE)
        endif()

        gmx_check_if_changed(HWLOC_FOUND_CHANGED HWLOC_FOUND)
        if (_STATIC_HWLOC AND HWLOC_FOUND_CHANGED AND NOT GMX_HWLOC_FORCE)
            message(STATUS "Static hwloc library found, will not attempt using it as it could lead to link-time errors. To use the detected library, manually set GMX_HWLOC=ON and you will likely have to pass appropriate linker flags too to satisfy the link-time dependencies of your hwloc library. Try \"pkg-config --libs --static hwloc\" for suggestions on what you will need.")
            set(GMX_USE_HWLOC OFF)
        else()
            set(GMX_USE_HWLOC ON)
        endif()

        if (GMX_USE_HWLOC)
            include_directories(SYSTEM ${HWLOC_INCLUDE_DIRS})
            list(APPEND GMX_EXTRA_LIBRARIES ${HWLOC_LIBRARIES})
        endif()
    elseif(GMX_HWLOC_FORCE)
        message(FATAL_ERROR "HWLOC package support required, but not found.")
    endif()

    set(HWLOC_FIND_QUIETLY_AFTER_FIRST_RUN TRUE CACHE INTERNAL "Be quiet during future attempts to find HWLOC")
endif()

option(GMX_EXTERNAL_TINYXML2 "Use external TinyXML-2 instead of compiling the version bundled with GROMACS." OFF)
mark_as_advanced(GMX_EXTERNAL_TINYXML2)
if(GMX_EXTERNAL_TINYXML2)
    # Find an external TinyXML-2 library.
    find_package(TinyXML-2 3.0.0)
    set(HAVE_TINYXML2 ${TinyXML2_FOUND})
    if(NOT HAVE_TINYXML2)
        message(FATAL_ERROR "External TinyXML-2 could not be found, please adjust your search paths")
    endif()
endif()

option(GMX_EXTRAE "Add support for tracing using EXTRAE" OFF)
mark_as_advanced(GMX_EXTRAE)

if (GMX_EXTRAE)
  find_package(EXTRAE)
  if(EXTRAE_FOUND)
    include_directories(SYSTEM ${EXTRAE_INCLUDE_DIR})
    set(HAVE_EXTRAE 1)
  else()
    message(FATAL_ERROR "EXTRAE library was not found. Please add the correct path to CMAKE_PREFIX_PATH")
  endif()
endif()

option(GMX_X11 "Use X window system" OFF)
if (GMX_X11)
    find_package(X11)
    # X11 includes/libraries are only set in the ngmx subdirectory!
    if(NOT X11_FOUND)
        message(FATAL_ERROR
                "X11 include files and/or libraries were not found. "
                "Set GMX_X11=OFF to compile without X11 support. "
                "gmx view will not be available.")
    endif()
    include_directories(SYSTEM ${X11_INCLUDE_DIR})
endif()

include(ThreadMPI)
# Enable core threading facilities
tmpi_enable_core("${CMAKE_SOURCE_DIR}/src/external/thread_mpi/include")
if(GMX_THREAD_MPI)
    # enable MPI functions
    tmpi_enable()
    set(MPI_IN_PLACE_EXISTS 1)
endif()
# If atomics are manually disabled a define is needed because atomics.h doesn't depend on config.h
if (TMPI_ATOMICS_DISABLED)
   add_definitions(-DTMPI_ATOMICS_DISABLED)
endif()

include(gmxManageTNG)

include(gmxManageLmfit)

if(GMX_GPU)
    # now that we have detected the dependencies, do the second configure pass
    gmx_gpu_setup()
    if (GMX_CLANG_CUDA)
        list(APPEND GMX_EXTRA_LIBRARIES ${GMX_CUDA_CLANG_LINK_LIBS})
        link_directories("${GMX_CUDA_CLANG_LINK_DIRS}")
    endif()
endif()

if(CYGWIN)
    set(GMX_CYGWIN 1)
endif()

if(WIN32)
    set(GMX_NATIVE_WINDOWS 1)
    # This makes windows.h not declare min/max as macros that would break
    # C++ code using std::min/std::max.
    add_definitions(-DNOMINMAX)
endif()

option(GMX_BUILD_UNITTESTS "Build unit tests with BUILD_TESTING" ON)
mark_as_advanced(GMX_BUILD_UNITTESTS)
gmx_add_cache_dependency(GMX_BUILD_UNITTESTS BOOL BUILD_TESTING OFF)

########################################################################
# Our own GROMACS tests
########################################################################

include_directories(BEFORE ${CMAKE_SOURCE_DIR}/src)
# Required for config.h, maybe should only be set in src/CMakeLists.txt
include_directories(BEFORE ${CMAKE_BINARY_DIR}/src)

include(gmxTestInlineASM)
gmx_test_inline_asm_gcc_x86(GMX_X86_GCC_INLINE_ASM)

include(gmxSetBuildInformation)
gmx_set_build_information()

gmx_option_multichoice(
    GMX_USE_RDTSCP
    "Use low-latency RDTSCP instruction for CPU-based timers for mdrun execution; might need to be off when compiling for heterogeneous environments)"
    "AUTO"
    OFF ON AUTO DETECT)
mark_as_advanced(GMX_USE_RDTSCP)

macro(gmx_check_rdtscp)
    if (CPU_DETECTION_FEATURES MATCHES "rdtscp")
        set(HAVE_RDTSCP 1)
        set(RDTSCP_DETECTION_MESSAGE " - detected on the build host")
    else()
        set(RDTSCP_DETECTION_MESSAGE " - not detected on the build host")
    endif()
endmacro()

set(HAVE_RDTSCP 0)
if (GMX_USE_RDTSCP STREQUAL "ON")
    set(HAVE_RDTSCP 1)
elseif(GMX_USE_RDTSCP STREQUAL "DETECT")
    gmx_check_rdtscp()
elseif(GMX_USE_RDTSCP STREQUAL "AUTO")
    # If the user specified automated SIMD selection, that the choice
    # is made based on detection on the build host. If so, then RDTSCP
    # should be chosen the same way.
    #
    # If the user specified an AVX SIMD level (e.g. when
    # cross-compiling GROMACS) then they will get our best guess, ie
    # that in practice AVX mostly correlates with rdtscp (and anyway
    # is only relevant in rather old x86 hardware).
    if (GMX_SIMD STREQUAL "AUTO")
        gmx_check_rdtscp()
    elseif (GMX_SIMD MATCHES "AVX")
        set(HAVE_RDTSCP 1)
    endif()
endif()
gmx_check_if_changed(HAVE_RDTSCP_CHANGED HAVE_RDTSCP)
if (HAVE_RDTSCP_CHANGED)
    if (HAVE_RDTSCP)
        message(STATUS "Enabling RDTSCP support${RDTSCP_DETECTION_MESSAGE}")
    else()
        message(STATUS "Disabling RDTSCP support${RDTSCP_DETECTION_MESSAGE}")
    endif()
endif()

include(gmxTestLargeFiles)
gmx_test_large_files(GMX_LARGEFILES)

include(gmxTestSignal)
gmx_test_sigusr1(HAVE_SIGUSR1)

include(gmxTestPipes)
gmx_test_pipes(HAVE_PIPES)

include(gmxTestXDR)
gmx_test_xdr(GMX_SYSTEM_XDR)
# Darwin has system XDR, but it uses a three-argument flavour of
# xdrproc_t that it guarantees will still work if you pass the normal
# two-argument xdr filters, but gcc 8 warns about the cast necessary
# to do that, so it's simpler to just use our own XDR library.
#
# TODO It would be better to craft a cmake test which fails if such
# XDR operations cause warnings, and succeeds otherwise, because it is
# generally preferable to use system libraries where possible.
if(NOT GMX_SYSTEM_XDR OR CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(GMX_INTERNAL_XDR 1)
endif()


##################################################
# Process SIMD instruction settings
##################################################
# This checks what flags to add in order to
# support the SIMD instructions we need, it sets
# correct defines for the SIMD instructions supported,
# and adds advanced options to control accuracy
# for SIMD math operations.
include(gmxManageSimd)
gmx_manage_simd()

include(gmxManageCycleCounters)
gmx_manage_cycle_counters()

# Process QM/MM Settings
if(${GMX_QMMM_PROGRAM} STREQUAL "GAUSSIAN")
    set(GMX_QMMM_GAUSSIAN 1)
elseif(${GMX_QMMM_PROGRAM} STREQUAL "MOPAC")
    set(GMX_QMMM_MOPAC 1)
elseif(${GMX_QMMM_PROGRAM} STREQUAL "GAMESS")
    set(GMX_QMMM_GAMESS 1)
elseif(${GMX_QMMM_PROGRAM} STREQUAL "ORCA")
    set(GMX_QMMM_ORCA 1)
elseif(${GMX_QMMM_PROGRAM} STREQUAL "NONE")
    # nothing to do
else()
    gmx_invalid_option_value(GMX_QMMM_PROGRAM)
endif()


##################################################
# Process FFT library settings
##################################################
include(gmxManageFFTLibraries)


include(gmxManageLinearAlgebraLibraries)

include(gmxManagePluginSupport)

if (GMX_USE_PLUGINS)
    if(NOT GMX_VMD_PLUGIN_PATH)
        find_package(VMD)
    endif()
endif()

# Link real-time library for POSIX timers. The check for clock_gettime
# confirms the linkability of rt.
if(HAVE_TIME_H AND HAVE_UNISTD_H AND HAVE_CLOCK_GETTIME)
    list(APPEND GMX_EXTRA_LIBRARIES rt)
endif()

# Math and thread libraries must often come after all others when linking...
if (HAVE_LIBM)
    list(APPEND GMX_PUBLIC_LIBRARIES m)
endif()

option(GMX_NACL "Configure for Native Client builds" OFF)
if (GMX_NACL)
  list(APPEND GMX_EXTRA_LIBRARIES nosys)
  set(GMX_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lnosys")
  # TODO: Is this still necessary with the check for its presence?
  set(GMX_USE_NICE 0)
  set(GMX_NO_RENAME 1)
endif()
mark_as_advanced(GMX_NACL)

if(GMX_FAHCORE)
  set(COREWRAP_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/../corewrap" CACHE STRING
      "Path to swindirect.h")
  include_directories(${COREWRAP_INCLUDE_DIR})
endif()

# Value of GMX_BUILD_HELP=AUTO tries to generate things, but will only
# produce warnings if that fails.
set(build_help_default AUTO)
if (SOURCE_IS_SOURCE_DISTRIBUTION OR CMAKE_CROSSCOMPILING)
    set(build_help_default OFF)
endif()
gmx_option_trivalue(GMX_BUILD_HELP "Build completions automatically (requires that compiled binaries can be executed on the build host) and install man pages if built (requires building the 'man' target manually)" ${build_help_default})
mark_as_advanced(GMX_BUILD_HELP)
if (GMX_BUILD_HELP AND SOURCE_IS_SOURCE_DISTRIBUTION AND BUILD_IS_INSOURCE)
    message(FATAL_ERROR
        "Rebuilding shell completions or man pages is not supported for "
        "in-source builds from a source distribution. "
        "Set GMX_BUILD_HELP=OFF or do an out-of-source build to proceed.")
endif()

# # # # # # # # # # NO MORE TESTS AFTER THIS LINE! # # # # # # # # # # #
# these are set after everything else
if (NOT GMX_SKIP_DEFAULT_CFLAGS)
    set(CMAKE_EXE_LINKER_FLAGS "${FFT_LINKER_FLAGS} ${MPI_LINKER_FLAGS} ${CMAKE_EXE_LINKER_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS "${FFT_LINKER_FLAGS} ${MPI_LINKER_FLAGS} ${CMAKE_SHARED_LINKER_FLAGS}")
else()
    message("Recommended flags which are not added because GMX_SKIP_DEFAULT_CFLAGS=yes:")
    message("CMAKE_C_FLAGS: ${SIMD_C_FLAGS};${MPI_COMPILE_FLAGS};${EXTRA_C_FLAGS};${GMXC_CFLAGS}")
    foreach(build_type ${build_types_with_explicit_flags})
        message("CMAKE_C_FLAGS_${build_type}: ${GMXC_CFLAGS_${build_type}}")
    endforeach()
    message("CMAKE_CXX_FLAGS: ${SIMD_CXX_FLAGS};${MPI_COMPILE_FLAGS};${EXTRA_CXX_FLAGS};${GMXC_CXXFLAGS}")
    foreach(build_type ${build_types_with_explicit_flags})
        message("CMAKE_CXX_FLAGS_${build_type}: ${GMXC_CXXFLAGS_${build_type}}")
    endforeach()
    message("CMAKE_EXE_LINKER_FLAGS: ${FFT_LINKER_FLAGS} ${MPI_LINKER_FLAGS}")
    message("CMAKE_SHARED_LINKER_FLAGS: ${FFT_LINKER_FLAGS} ${MPI_LINKER_FLAGS}")
endif()

########################################################################
# Specify install locations
########################################################################
# Use GNUInstallDirs to set paths on multiarch systems.
include(GNUInstallDirs)

set(GMX_INSTALL_DATASUBDIR "gromacs" CACHE STRING "Subdirectory for GROMACS data under CMAKE_INSTALL_DATADIR")
mark_as_advanced(GMX_INSTALL_DATASUBDIR)

# Internal convenience so we do not have to join two path segments in the code
set(GMX_INSTALL_GMXDATADIR ${CMAKE_INSTALL_DATADIR}/${GMX_INSTALL_DATASUBDIR})

# If the nesting level wrt. the installation root is changed,
# gromacs-config.cmake.cmakein needs to be adapted.
set(GMX_INSTALL_CMAKEDIR  ${CMAKE_INSTALL_DATAROOTDIR}/cmake)

# TODO: Make GMXRC adapt if this is changed
set(GMX_INSTALL_PKGCONFIGDIR ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
set(GMX_INSTALL_OCLDIR       ${GMX_INSTALL_GMXDATADIR}/opencl)

list(APPEND INSTALLED_HEADER_INCLUDE_DIRS ${CMAKE_INSTALL_INCLUDEDIR})

# Binary and library suffix options
include(gmxManageSuffixes)

################################################################
# Shared library load path settings
################################################################
if(NOT GMX_BUILD_SHARED_EXE)
    # No rpath
    set(CMAKE_SKIP_RPATH TRUE)
    set(CMAKE_EXE_LINK_DYNAMIC_C_FLAGS) # remove -Wl,-Bdynamic
    set(CMAKE_EXE_LINK_DYNAMIC_CXX_FLAGS)
else()
    # The build folder always has bin/ and lib/; if we are also going to
    # install to lib/, then the installation RPATH works also in the build
    # tree.  This makes installation slightly faster (no need to rewrite the
    # RPATHs), and makes the binaries in the build tree relocatable.
    if(CMAKE_INSTALL_LIBDIR STREQUAL "lib")
        set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
        set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR TRUE)
    endif()
    # Set the RPATH as relative to the executable location to make the
    # binaries relocatable.
    if(CMAKE_SYSTEM_NAME STREQUAL "Darwin") #Assume OS X >=10.5
        set(CMAKE_INSTALL_RPATH "@loader_path/../${CMAKE_INSTALL_LIBDIR}")
        set(CMAKE_INSTALL_NAME_DIR ${CMAKE_INSTALL_RPATH})
    else()
        set(CMAKE_INSTALL_RPATH "\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
    endif()
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
    set(CMAKE_MACOSX_RPATH 1)
endif()

#COPYING file: Only necessary for binary distributions.
#Simpler to always install.
install(FILES COPYING DESTINATION ${GMX_INSTALL_GMXDATADIR} COMPONENT data)

if (GMX_BUILD_FOR_COVERAGE)
    # Code heavy with asserts makes conditional coverage close to useless metric,
    # as by design most of the false branches are impossible to trigger in
    # correctly functioning code.  And the benefit of testing those that could
    # be triggered by using an API against its specification isn't usually
    # worth the effort.
    add_definitions(-DNDEBUG -DGMX_DISABLE_ASSERTS)
endif()

if (BUILD_TESTING)
    include(tests/CheckTarget.cmake)
endif()

# TODO: Determine control flow and defaults for package installation and testing use cases.
# Ref: http://redmine.gromacs.org/issues/2896
option(GMX_PYTHON_PACKAGE "Configure gmxapi Python package" OFF)
mark_as_advanced(GMX_PYTHON_PACKAGE)

if (NOT GMX_BUILD_MDRUN_ONLY)
    # Note: Though only documented as an output variable, PYTHON_EXECUTABLE is
    # also effective as a CMake input variable to effectively hint the location
    # of the Python interpreter. This may be helpful in environments with both
    # Python 2 and Python 3 on the default PATH.
    # Ref: https://cmake.org/cmake/help/latest/module/FindPythonInterp.html
    if(FIND_PACKAGE_MESSAGE_DETAILS_PythonInterp)
        # Keep quiet on subsequent runs of cmake
        set(PythonInterp_FIND_QUIETLY ON)
    endif()
    # Older CMake versions might not search for Python newer than 3.7.
    set(Python_ADDITIONAL_VERSIONS 3.8)
    if(GMX_PYTHON_PACKAGE)
        find_package(PythonInterp 3.5 REQUIRED)
        # Note: PythonLibs will be found later by pybind11.
        # TODO: (issue #2998) When CMake >= 3.12 is required, update detection.
        # I.e.  find_package(Python3 3.5 COMPONENTS Interpreter Development REQUIRED)
    else()
        find_package(PythonInterp 3.5)
    endif()
    find_package(ImageMagick QUIET COMPONENTS convert)
    include(gmxTestImageMagick)
    GMX_TEST_IMAGEMAGICK(IMAGE_CONVERT_POSSIBLE)
    # TODO: Resolve circular dependency between docs, gromacs, and python_packaging
    add_subdirectory(docs)
    add_subdirectory(share)
    add_subdirectory(scripts)
endif()
add_subdirectory(src)

if (BUILD_TESTING)
    add_subdirectory(tests)
endif()

if(GMX_PYTHON_PACKAGE AND NOT GMX_BUILD_MDRUN_ONLY)
    add_subdirectory(python_packaging)
endif()

gmx_cpack_write_config()

# Issue a warning if NVIDIA GPUs were detected, but CUDA was not found.
# Don't bother the user after the first configure pass.
if ((CUDA_NOTFOUND_AUTO AND GMX_DETECT_GPU_AVAILABLE) AND NOT GMX_GPU_DETECTION_DONE)
    message(WARNING "${CUDA_NOTFOUND_MESSAGE}")
endif()
set(GMX_GPU_DETECTION_DONE TRUE CACHE INTERNAL "Whether GPU detection has already been done")

#######################
## uninstall target
#######################
CONFIGURE_FILE(   "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in"
                  "${CMAKE_CURRENT_BINARY_DIR}/cmake/cmake_uninstall.cmake"
                  IMMEDIATE @ONLY)
###########################
ADD_CUSTOM_TARGET(uninstall
                  "${CMAKE_COMMAND}" -P
                  "${CMAKE_CURRENT_BINARY_DIR}/cmake/cmake_uninstall.cmake")
###########################
set_directory_properties(PROPERTIES
            ADDITIONAL_MAKE_CLEAN_FILES "install_manifest.txt")
