# libop

set(OP_LIB ${PROJECT_SOURCE_DIR}/lib/src/SimulationRegion.cpp ${PROJECT_SOURCE_DIR}/lib/src/NeighborList.cpp)

set (OP_CXX_FLAG -D_GLIBCXX_USE_CXX11_ABI=${OP_CXX_ABI} )
file(GLOB OP_SRC prod_force.cc prod_virial.cc descrpt.cc descrpt_se_a.cc descrpt_se_r.cc tab_inter.cc prod_force_se_a.cc prod_virial_se_a.cc prod_force_se_r.cc prod_virial_se_r.cc soft_min.cc soft_min_force.cc soft_min_virial.cc )
file(GLOB OP_CUDA_SRC prod_force.cc prod_virial.cc descrpt.cc descrpt_se_a_gpu.cc descrpt_se_r_gpu.cc tab_inter.cc prod_force_se_a_gpu.cc prod_virial_se_a_gpu.cc prod_force_se_r_gpu.cc prod_virial_se_r_gpu.cc soft_min.cc soft_min_force.cc soft_min_virial.cc )
file(GLOB OP_GRADS_SRC prod_force_grad.cc prod_force_se_a_grad.cc prod_force_se_r_grad.cc prod_virial_grad.cc prod_virial_se_a_grad.cc prod_virial_se_r_grad.cc soft_min_force_grad.cc soft_min_virial_grad.cc )
file(GLOB OP_PY *.py)

if (BUILD_CPP_IF) 
    if (USE_CUDA_TOOLKIT)
        add_library(${LIB_DEEPMD_OP} SHARED ${OP_CUDA_SRC})
        add_subdirectory(cuda)
        find_package(CUDA REQUIRED)
        include_directories(${CUDA_INCLUDE_DIRS})
        set (EXTRA_LIBS ${EXTRA_LIBS} deepmd_op_cuda)
        target_link_libraries (${LIB_DEEPMD_OP} ${EXTRA_LIBS})
        target_link_libraries (${LIB_DEEPMD_OP} ${CUDA_LIBRARIES})
    else (USE_CUDA_TOOLKIT)
        add_library(${LIB_DEEPMD_OP} SHARED ${OP_SRC})
    endif (USE_CUDA_TOOLKIT)
endif (BUILD_CPP_IF)

if (BUILD_PY_IF)
  add_library(op_abi SHARED ${OP_SRC} ${OP_LIB})
  add_library(op_grads SHARED ${OP_GRADS_SRC})
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
  install(TARGETS op_abi			DESTINATION deepmd)
  install(TARGETS op_grads			DESTINATION deepmd)
  install(FILES  ${OP_PY}			DESTINATION deepmd)
endif (BUILD_PY_IF)
