# libmd

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/version.h.in
  version.h
  @ONLY
)

if (USE_CUDA_TOOLKIT)
  include_directories("${CUDA_INCLUDE_DIRS}")
endif()

if (USE_ROCM_TOOLKIT)
  include_directories("${ROCM_INCLUDE_DIRS}")
endif()

file(GLOB LIB_SRC src/*.cc src/*.cpp)
file(GLOB INC_SRC include/*.h ${CMAKE_CURRENT_BINARY_DIR}/version.h)


function(_add_libapicc_variant variant_name prec_def)
set (libname "${LIB_DEEPMD_CC}${variant_name}")

add_library(${libname} SHARED ${LIB_SRC})

if (USE_CUDA_TOOLKIT)
  target_link_libraries (${libname} ${CUDA_LIBRARIES})
endif()

if (USE_ROCM_TOOLKIT)
  target_link_libraries (${libname} ${ROCM_LIBRARIES})
endif()

set_target_properties(
  ${libname} 
  PROPERTIES 
  COMPILE_DEFINITIONS ${prec_def}
)

install(TARGETS ${libname} DESTINATION lib/)

install(
  FILES		${INC_SRC}
  DESTINATION	include/deepmd
)

endfunction()

_add_libapicc_variant("${HIGH_PREC_VARIANT}" "${HIGH_PREC_DEF}")
_add_libapicc_variant("${LOW_PREC_VARIANT}" "${LOW_PREC_DEF}")
