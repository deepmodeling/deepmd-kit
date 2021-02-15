# libmd

set (libname ${LIB_DEEPMD})

if (USE_CUDA_TOOLKIT)
  include_directories("${CUDA_INCLUDE_DIRS}")
endif()

file(GLOB LIB_SRC src/*.cc src/*.cpp)
file(GLOB INC_SRC include/*.h ${CMAKE_CURRENT_BINARY_DIR}/version.h)

add_library(${libname} SHARED ${LIB_SRC})

if (USE_CUDA_TOOLKIT)
  target_link_libraries (${libname} ${CUDA_LIBRARIES})
endif()

if(BUILD_PY_IF)
  install(TARGETS ${libname} DESTINATION deepmd/op/)
endif(BUILD_PY_IF)
if(BUILD_CPP_IF)
  install(TARGETS ${libname} DESTINATION lib/)
  install(
    FILES	${INC_SRC}
    DESTINATION	include/deepmd
    )
endif(BUILD_CPP_IF)

