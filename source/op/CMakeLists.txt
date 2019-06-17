# libop

set(OP_LIB ${PROJECT_SOURCE_DIR}/lib/src/SimulationRegion.cpp ${PROJECT_SOURCE_DIR}/lib/src/NeighborList.cpp)

set (OP_CXX_FLAG -D_GLIBCXX_USE_CXX11_ABI=${OP_ABI} )
file(GLOB OP_SRC prod_force.cc prod_virial.cc descrpt.cc descrpt_norot.cc tab_inter.cc prod_force_norot.cc prod_virial_norot.cc soft_min.cc soft_min_force.cc soft_min_virial.cc )
file(GLOB OP_GRADS_SRC soft_min_force_grad.cc soft_min_virial_grad.cc)
file(GLOB OP_PY *.py)

if (BUILD_CPP_IF)
  add_library(${LIB_DEEPMD_OP} SHARED ${OP_SRC})
endif (BUILD_CPP_IF)
if (BUILD_PY_IF)
  add_library(op_abi SHARED ${OP_SRC} ${OP_LIB})
  add_library(op_grads SHARED ${OP_GRADS_SRC})
  add_library(prod_force_grad SHARED prod_force_grad.cc)
  add_library(prod_force_norot_grad SHARED prod_force_norot_grad.cc)
  add_library(prod_virial_grad SHARED prod_virial_grad.cc)
  add_library(prod_virial_norot_grad SHARED prod_virial_norot_grad.cc)
  target_link_libraries(
    op_abi ${TensorFlowFramework_LIBRARY}
    )
  target_link_libraries(
    op_grads ${TensorFlowFramework_LIBRARY}
    )
  target_link_libraries(
    prod_force_grad ${TensorFlowFramework_LIBRARY}
    )
  target_link_libraries(
    prod_force_norot_grad ${TensorFlowFramework_LIBRARY}
    )
  target_link_libraries(
    prod_virial_grad ${TensorFlowFramework_LIBRARY}
    )
  target_link_libraries(
    prod_virial_norot_grad ${TensorFlowFramework_LIBRARY}
    )
  set_target_properties(
    op_abi 
    PROPERTIES 
    COMPILE_FLAGS ${OP_CXX_FLAG}
    )
  set_target_properties(
    prod_force_grad  
    PROPERTIES 
    COMPILE_FLAGS ${OP_CXX_FLAG}
    )
  set_target_properties(
    prod_force_norot_grad  
    PROPERTIES 
    COMPILE_FLAGS ${OP_CXX_FLAG}
    )
  set_target_properties(
    prod_virial_grad 
    PROPERTIES 
    COMPILE_FLAGS ${OP_CXX_FLAG}
    )
  set_target_properties(
    prod_virial_norot_grad 
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
  install(TARGETS prod_force_grad		DESTINATION deepmd)
  install(TARGETS prod_force_norot_grad		DESTINATION deepmd)
  install(TARGETS prod_virial_grad		DESTINATION deepmd)
  install(TARGETS prod_virial_norot_grad	DESTINATION deepmd)
  install(FILES  ${OP_PY}			DESTINATION deepmd)
endif (BUILD_PY_IF)
