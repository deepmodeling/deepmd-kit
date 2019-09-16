# libmd

set (libname ${LIB_DEEPMD})

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/version.h.in
  version.h
  @ONLY
)

find_package(CUDA  REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")
link_directories("${CUDA_LIB_DIRS}")

file(GLOB LIB_SRC src/*.cc src/*.cpp)
file(GLOB INC_SRC include/*.h ${CMAKE_CURRENT_BINARY_DIR}/version.h)

add_library(${libname} SHARED ${LIB_SRC})

target_link_libraries (${libname} libcudart.so)

install(TARGETS ${libname} DESTINATION lib/)

install(
  FILES		${INC_SRC}
  DESTINATION	include/deepmd
)
