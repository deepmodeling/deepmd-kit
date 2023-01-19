# libop

if (USE_ASCEND)
  # link: Ascend op HAS INCLUDED prod_force.cc prod_virial.cc prod_env_mat.cc
  file(GLOB OP_SRC custom_op.cc descrpt.cc descrpt_se_a_ef.cc descrpt_se_a_ef.cc descrpt_se_a_ef_para.cc descrpt_se_a_ef_vert.cc pair_tab.cc soft_min.cc soft_min_force.cc soft_min_virial.cc ewald_recp.cc map_aparam.cc neighbor_stat.cc unaggregated_grad.cc tabulate_multi_device.cc prod_env_mat_multi_device.cc)
else()
  file(GLOB OP_SRC prod_env_mat_multi_device_nvnmd.cc map_nvnmd.cc matmul_nvnmd.cc quantize_nvnmd.cc tanh2_nvnmd.cc tanh4_nvnmd.cc custom_op.cc prod_force.cc prod_virial.cc descrpt.cc descrpt_se_a_ef.cc descrpt_se_a_ef.cc descrpt_se_a_ef_para.cc descrpt_se_a_ef_vert.cc pair_tab.cc prod_force_multi_device.cc prod_virial_multi_device.cc soft_min.cc soft_min_force.cc soft_min_virial.cc ewald_recp.cc gelu_multi_device.cc map_aparam.cc neighbor_stat.cc unaggregated_grad.cc tabulate_multi_device.cc prod_env_mat_multi_device.cc)
endif (USE_ASCEND)
set(OP_LIB ${PROJECT_SOURCE_DIR}/lib/src/SimulationRegion.cpp ${PROJECT_SOURCE_DIR}/lib/src/neighbor_list.cc)

file(GLOB OP_GRADS_SRC custom_op.cc prod_force_grad.cc prod_force_grad_multi_device.cc prod_virial_grad.cc prod_virial_grad_multi_device.cc soft_min_force_grad.cc soft_min_virial_grad.cc )
file(GLOB OP_PY *.py)
file(GLOB OP_REMAPPER_SRC optimizer/parallel.cc)

add_library(${LIB_DEEPMD_OP} MODULE ${OP_SRC} ${OP_REMAPPER_SRC})
# link: libdeepmd libtensorflow_cc libtensorflow_framework
target_link_libraries (${LIB_DEEPMD_OP} PRIVATE TensorFlow::tensorflow_framework)
target_link_libraries (${LIB_DEEPMD_OP} PRIVATE ${LIB_DEEPMD})
if(APPLE)
  set_target_properties(${LIB_DEEPMD_OP} PROPERTIES INSTALL_RPATH "@loader_path;${TensorFlow_LIBRARY_PATH}")
else()
  set_target_properties(${LIB_DEEPMD_OP} PROPERTIES INSTALL_RPATH "$ORIGIN;${TensorFlow_LIBRARY_PATH}")
endif()
if (CMAKE_TESTING_ENABLED)
  target_link_libraries(${LIB_DEEPMD_OP} PRIVATE coverage_config)
endif()
target_precompile_headers(${LIB_DEEPMD_OP} PRIVATE custom_op.h)

if (BUILD_PY_IF)
  add_library(op_grads MODULE ${OP_GRADS_SRC})
  # link: libdeepmd libtensorflow_framework
  target_link_libraries(op_grads PRIVATE ${LIB_DEEPMD})
  target_link_libraries(
    op_grads PRIVATE TensorFlow::tensorflow_framework
    )
  if(APPLE)
    set_target_properties(
      op_grads
      PROPERTIES
      INSTALL_RPATH @loader_path
      )
  else()
  set_target_properties(
    op_grads
    PROPERTIES
    INSTALL_RPATH $ORIGIN
    )
  endif ()
  if (CMAKE_TESTING_ENABLED)
    target_link_libraries(op_grads PRIVATE coverage_config)
  endif()
  target_precompile_headers(op_grads PRIVATE custom_op.h)
endif (BUILD_PY_IF)

if (BUILD_PY_IF)
  install(TARGETS ${LIB_DEEPMD_OP}		DESTINATION deepmd/op/)
  install(TARGETS op_grads			DESTINATION deepmd/op/)
  install(FILES  ${OP_PY}			DESTINATION deepmd/op/)
else(BUILD_PY_IF)
  install(TARGETS ${LIB_DEEPMD_OP}		DESTINATION lib/)
endif (BUILD_PY_IF)

