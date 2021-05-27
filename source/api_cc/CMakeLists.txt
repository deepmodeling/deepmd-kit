# libmd
set (libname ${LIB_DEEPMD_CC})

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/version.h.in
  version.h
  @ONLY
)

# Devices that have both ROCM and CUDA are not currently supported
if (USE_ROCM_TOOLKIT AND USE_CUDA_TOOLKIT)
  message (FATAL_ERROR "Devices that have both ROCM and CUDA are not currently supported")
endif()

if (USE_CUDA_TOOLKIT)
  include_directories("${CUDA_INCLUDE_DIRS}")
endif()

file(GLOB LIB_SRC src/*.cc src/*.cpp)
file(GLOB INC_SRC include/*.h ${CMAKE_CURRENT_BINARY_DIR}/version.h)

add_library(${libname} SHARED ${LIB_SRC})

if (USE_CUDA_TOOLKIT)
  target_link_libraries (${libname} ${CUDA_LIBRARIES})
endif()

if (USE_ROCM_TOOLKIT)
  LINK_LIBRARIES(${HIP_ROOT_DIR}\lib) 
  target_link_libraries(${libname} hip_hcc)
endif()


install(TARGETS ${libname} DESTINATION lib/)

install(
  FILES		${INC_SRC}
  DESTINATION	include/deepmd
)

