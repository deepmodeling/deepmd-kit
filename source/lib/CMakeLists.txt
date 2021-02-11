# libmd

set (libname ${LIB_DEEPMD})

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/version.h.in
  version.h
  @ONLY
)

find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

if (USE_CUDA_TOOLKIT)
  include_directories("${CUDA_INCLUDE_DIRS}")
endif()

file(GLOB LIB_SRC src/*.cc src/*.cpp)
file(GLOB INC_SRC include/*.h ${CMAKE_CURRENT_BINARY_DIR}/version.h)

add_library(${libname} SHARED ${LIB_SRC})

if (USE_CUDA_TOOLKIT)
  target_link_libraries (${libname} ${CUDA_LIBRARIES})
endif()

target_link_libraries (${libname} ${MPI_C_LIBRARIES})

install(TARGETS ${libname} DESTINATION lib/)

install(
  FILES		${INC_SRC}
  DESTINATION	include/deepmd
)
