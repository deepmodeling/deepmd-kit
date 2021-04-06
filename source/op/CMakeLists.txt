# libop

set(OP_LIB ${PROJECT_SOURCE_DIR}/lib/src/SimulationRegion.cpp ${PROJECT_SOURCE_DIR}/lib/src/neighbor_list.cc)

set (OP_CXX_FLAG -D_GLIBCXX_USE_CXX11_ABI=${OP_CXX_ABI} )
file(GLOB OP_SRC prod_force.cc prod_virial.cc descrpt.cc descrpt_se_a_ef.cc descrpt_se_a_ef.cc descrpt_se_a_ef_para.cc descrpt_se_a_ef_vert.cc pair_tab.cc prod_force_multi_device.cc prod_virial_multi_device.cc soft_min.cc soft_min_force.cc soft_min_virial.cc ewald_recp.cc gelu_multi_device.cc map_aparam.cc neighbor_stat.cc unaggregated_grad.cc tabulate_multi_device.cc prod_env_mat_multi_device.cc)
file(GLOB OP_CUDA_SRC prod_force.cc prod_virial.cc descrpt.cc prod_env_mat_multi_device.cc pair_tab.cc prod_force_multi_device.cc prod_virial_multi_device.cc soft_min.cc soft_min_force.cc soft_min_virial.cc gelu_multi_device.cc tabulate_multi_device.cc)
file(GLOB OP_GRADS_SRC prod_force_grad.cc prod_force_grad_multi_device.cc prod_virial_grad.cc prod_virial_grad_multi_device.cc soft_min_force_grad.cc soft_min_virial_grad.cc )
file(GLOB OP_PY *.py)

if (BUILD_CPP_IF) 
    if (USE_CUDA_TOOLKIT)
        add_library(${LIB_DEEPMD_OP} SHARED ${OP_CUDA_SRC})
        find_package(CUDA REQUIRED)
        include_directories(${CUDA_INCLUDE_DIRS})
        target_link_libraries (${LIB_DEEPMD_OP} ${CUDA_LIBRARIES})
    else (USE_CUDA_TOOLKIT)
        add_library(${LIB_DEEPMD_OP} SHARED ${OP_SRC})
    endif (USE_CUDA_TOOLKIT)
endif (BUILD_CPP_IF)

if (BUILD_PY_IF)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
  set(CMAKE_INSTALL_RPATH $ORIGIN)
  if (USE_CUDA_TOOLKIT)
    add_library(op_abi SHARED ${OP_SRC} ${OP_LIB})
    add_library(op_grads SHARED ${OP_GRADS_SRC})
    find_package(CUDA REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
    target_link_libraries (op_abi ${LIB_DEEPMD_OP_CUDA})
    target_link_libraries (op_grads ${LIB_DEEPMD_OP_CUDA})
  else (USE_CUDA_TOOLKIT)
    add_library(op_abi SHARED ${OP_SRC} ${OP_LIB})
    add_library(op_grads SHARED ${OP_GRADS_SRC})
  endif(USE_CUDA_TOOLKIT)
  message(STATUS ${TensorFlowFramework_LIBRARY})
  target_link_libraries(op_abi ${LIB_DEEPMD})
  target_link_libraries(op_grads ${LIB_DEEPMD})
  target_link_libraries(
    op_abi ${TensorFlowFramework_LIBRARY}
    )
  target_link_libraries(
    op_grads ${TensorFlowFramework_LIBRARY}
    )
  set_target_properties(
    op_abi 
    PROPERTIES 
    COMPILE_FLAGS ${OP_CXX_FLAG}
    )
  set_target_properties(
    op_grads
    PROPERTIES
    COMPILE_FLAGS ${OP_CXX_FLAG}
    )
endif (BUILD_PY_IF)

if (BUILD_CPP_IF)
  install(TARGETS ${LIB_DEEPMD_OP}		DESTINATION lib/)
endif (BUILD_CPP_IF)
if (BUILD_PY_IF)
  install(TARGETS op_abi			DESTINATION deepmd/op/)
  install(TARGETS op_grads			DESTINATION deepmd/op/)
  install(FILES  ${OP_PY}			DESTINATION deepmd/op/)
endif (BUILD_PY_IF)
