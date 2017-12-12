# libop

set(OP_LIB ${PROJECT_SOURCE_DIR}/lib/src/SimulationRegion.cpp ${PROJECT_SOURCE_DIR}/lib/src/NeighborList.cpp)

set (OP_CXX_FLAG -D_GLIBCXX_USE_CXX11_ABI=${OP_ABI} )
file(GLOB OP_SRC prod_force.cc prod_virial.cc descrpt.cc)
file(GLOB OP_PY  *.py)

add_library(${LIB_DEEPMD_OP} SHARED ${OP_SRC})
add_library(op_abi SHARED ${OP_SRC} ${OP_LIB})
add_library(prod_force_grad SHARED prod_force_grad.cc)
add_library(prod_virial_grad SHARED prod_virial_grad.cc)

target_link_libraries(
  op_abi ${TensorFlowFramework_LIBRARY}
)
target_link_libraries(
  prod_force_grad ${TensorFlowFramework_LIBRARY}
)
target_link_libraries(
  prod_virial_grad ${TensorFlowFramework_LIBRARY}
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
  prod_virial_grad 
  PROPERTIES 
  COMPILE_FLAGS ${OP_CXX_FLAG}
)

install(TARGETS ${LIB_DEEPMD_OP} DESTINATION lib/)
install(TARGETS op_abi DESTINATION lib/deepmd)
install(TARGETS prod_force_grad DESTINATION lib/deepmd)
install(TARGETS prod_virial_grad DESTINATION lib/deepmd)
install(FILES  ${OP_PY} DESTINATION lib/deepmd)
