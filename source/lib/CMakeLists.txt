# libmd
set (libname ${LIB_DEEPMD})

if (USE_CUDA_TOOLKIT)
  include_directories("${CUDA_INCLUDE_DIRS}")
endif()

if (USE_ROCM_TOOLKIT)
  include_directories("${ROCM_INCLUDE_DIRS}")
endif()

file(GLOB LIB_SRC src/*.cc src/*.cpp)
file(GLOB INC_SRC include/*.h ${CMAKE_CURRENT_BINARY_DIR}/version.h)

if (USE_CUDA_TOOLKIT)
  add_definitions("-DGOOGLE_CUDA")
  add_subdirectory(src/cuda)
endif()

if (USE_ROCM_TOOLKIT)
  add_definitions("-DTENSORFLOW_USE_ROCM")
  add_subdirectory(src/rocm)
endif()

function(_add_libdeepmd_variant variant_name prec_def)
set (libname "${LIB_DEEPMD}${variant_name}")

add_library(${libname} SHARED ${LIB_SRC})

if (USE_CUDA_TOOLKIT)
  set (EXTRA_LIBS ${EXTRA_LIBS} "deepmd_op_cuda${variant_name}")
  target_link_libraries (${libname} ${CUDA_LIBRARIES} ${EXTRA_LIBS})
endif()

if (USE_ROCM_TOOLKIT)
  set (EXTRA_LIBS ${EXTRA_LIBS} "deepmd_op_rocm${variant_name}")
  target_link_libraries (${libname} ${ROCM_LIBRARIES} ${EXTRA_LIBS})
endif()

set_target_properties(
  ${libname} 
  PROPERTIES 
  COMPILE_DEFINITIONS ${prec_def}
)

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

endfunction()
_add_libdeepmd_variant("${HIGH_PREC_VARIANT}" "${HIGH_PREC_DEF}")
_add_libdeepmd_variant("${LOW_PREC_VARIANT}" "${LOW_PREC_DEF}")
