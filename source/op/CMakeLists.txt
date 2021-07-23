# libop

set(OP_LIB ${PROJECT_SOURCE_DIR}/lib/src/SimulationRegion.cpp ${PROJECT_SOURCE_DIR}/lib/src/neighbor_list.cc)

set (OP_CXX_FLAG -D_GLIBCXX_USE_CXX11_ABI=${OP_CXX_ABI} )
file(GLOB OP_SRC prod_force.cc prod_virial.cc descrpt.cc descrpt_se_a_ef.cc descrpt_se_a_ef.cc descrpt_se_a_ef_para.cc descrpt_se_a_ef_vert.cc pair_tab.cc prod_force_multi_device.cc prod_virial_multi_device.cc soft_min.cc soft_min_force.cc soft_min_virial.cc ewald_recp.cc gelu_multi_device.cc map_aparam.cc neighbor_stat.cc unaggregated_grad.cc tabulate_multi_device.cc prod_env_mat_multi_device.cc)
file(GLOB OP_CUDA_SRC prod_force.cc prod_virial.cc descrpt.cc prod_env_mat_multi_device.cc pair_tab.cc prod_force_multi_device.cc prod_virial_multi_device.cc soft_min.cc soft_min_force.cc soft_min_virial.cc gelu_multi_device.cc tabulate_multi_device.cc)
file(GLOB OP_ROCM_SRC prod_force.cc prod_virial.cc descrpt.cc prod_env_mat_multi_device.cc pair_tab.cc prod_force_multi_device.cc prod_virial_multi_device.cc soft_min.cc soft_min_force.cc soft_min_virial.cc gelu_multi_device.cc tabulate_multi_device.cc)
file(GLOB OP_GRADS_SRC prod_force_grad.cc prod_force_grad_multi_device.cc prod_virial_grad.cc prod_virial_grad_multi_device.cc soft_min_force_grad.cc soft_min_virial_grad.cc )
file(GLOB OP_PY *.py)

function(_add_libop_variant variant_name prec_def)
set(LIB_DEEPMD_OP_VARIANT "${LIB_DEEPMD_OP}${variant_name}")
set(LIB_OB_ABI_VARIANT "op_abi${variant_name}")
set(LIB_OB_GRADS_VARIANT "op_grads${variant_name}")

if (BUILD_CPP_IF) 
  if (USE_CUDA_TOOLKIT)
    add_library(${LIB_DEEPMD_OP_VARIANT} SHARED ${OP_CUDA_SRC})
    find_package(CUDA REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
    target_link_libraries (${LIB_DEEPMD_OP_VARIANT} ${CUDA_LIBRARIES})
  elseif (USE_ROCM_TOOLKIT)
    add_library(${LIB_DEEPMD_OP_VARIANT} SHARED ${OP_ROCM_SRC})
    find_package(ROCM REQUIRED)
    include_directories(${ROCM_INCLUDE_DIRS})
    target_link_libraries (${LIB_DEEPMD_OP_VARIANT} ${ROCM_LIBRARIES})
  else ()
    add_library(${LIB_DEEPMD_OP_VARIANT} SHARED ${OP_SRC})
  endif()
  set_target_properties(
    ${LIB_DEEPMD_OP_VARIANT} 
    PROPERTIES 
    COMPILE_DEFINITIONS ${prec_def}
  )
endif (BUILD_CPP_IF)

if (BUILD_PY_IF)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
  set(CMAKE_INSTALL_RPATH $ORIGIN)
  if (USE_CUDA_TOOLKIT)
    add_library(${LIB_OB_ABI_VARIANT} SHARED ${OP_SRC} ${OP_LIB})
    add_library(${LIB_OB_GRADS_VARIANT} SHARED ${OP_GRADS_SRC})
    find_package(CUDA REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
    target_link_libraries (${LIB_OB_ABI_VARIANT} ${LIB_DEEPMD_OP_DEVICE})
    target_link_libraries (${LIB_OB_GRADS_VARIANT} ${LIB_DEEPMD_OP_DEVICE})
  elseif(USE_ROCM_TOOLKIT)
    add_library(${LIB_OB_ABI_VARIANT} SHARED ${OP_SRC} ${OP_LIB})
    add_library(${LIB_OB_GRADS_VARIANT} SHARED ${OP_GRADS_SRC})
    find_package(ROCM REQUIRED)
    include_directories(${ROCM_INCLUDE_DIRS})
    target_link_libraries (${LIB_OB_ABI_VARIANT} ${LIB_DEEPMD_OP_DEVICE})
    target_link_libraries (${LIB_OB_GRADS_VARIANT} ${LIB_DEEPMD_OP_DEVICE})
  else()
    add_library(${LIB_OB_ABI_VARIANT} SHARED ${OP_SRC} ${OP_LIB})
    add_library(${LIB_OB_GRADS_VARIANT} SHARED ${OP_GRADS_SRC})
  endif()
  
  message(STATUS ${TensorFlowFramework_LIBRARY})
  target_link_libraries(${LIB_OB_ABI_VARIANT} ${LIB_DEEPMD})
  target_link_libraries(${LIB_OB_GRADS_VARIANT} ${LIB_DEEPMD})
  target_link_libraries(
    ${LIB_OB_ABI_VARIANT} ${TensorFlowFramework_LIBRARY}
    )
  target_link_libraries(
    ${LIB_OB_GRADS_VARIANT} ${TensorFlowFramework_LIBRARY}
    )
  set_target_properties(
    ${LIB_OB_ABI_VARIANT} 
    PROPERTIES 
    COMPILE_FLAGS ${OP_CXX_FLAG}
    COMPILE_DEFINITIONS ${prec_def}
    )
  set_target_properties(
    ${LIB_OB_GRADS_VARIANT}
    PROPERTIES
    COMPILE_FLAGS ${OP_CXX_FLAG}
    COMPILE_DEFINITIONS ${prec_def}
    )
endif (BUILD_PY_IF)

if (BUILD_CPP_IF)
  install(TARGETS ${LIB_DEEPMD_OP_VARIANT}		DESTINATION lib/)
endif (BUILD_CPP_IF)
if (BUILD_PY_IF)
  install(TARGETS ${LIB_OB_ABI_VARIANT}			DESTINATION deepmd/op/)
  install(TARGETS ${LIB_OB_GRADS_VARIANT}		DESTINATION deepmd/op/)
  install(FILES  ${OP_PY}			DESTINATION deepmd/op/)
endif (BUILD_PY_IF)

endfunction()
_add_libop_variant("${HIGH_PREC_VARIANT}" "${HIGH_PREC_DEF}")
_add_libop_variant("${LOW_PREC_VARIANT}" "${LOW_PREC_DEF}")
