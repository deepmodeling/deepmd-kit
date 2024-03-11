# libop

file(
  GLOB
  OP_PT_SRC
  comm.cc
)
add_library(${LIB_DEEPMD_OP_PT} MODULE ${OP_PT_SRC})
target_link_libraries(${libname} PRIVATE )
target_link_libraries(${LIB_DEEPMD_OP_PT} PRIVATE "${TORCH_LIBRARIES}")
target_link_libraries(${LIB_DEEPMD_OP_PT} PRIVATE ${LIB_DEEPMD})

if(CMAKE_TESTING_ENABLED)
  target_link_libraries(${LIB_DEEPMD_OP_PT} PRIVATE coverage_config)
endif()



if(BUILD_PY_IF)
  install(TARGETS ${LIB_DEEPMD_OP_PT} DESTINATION deepmd/lib/)
else(BUILD_PY_IF)
  install(TARGETS ${LIB_DEEPMD_OP_PT} DESTINATION lib/)
endif(BUILD_PY_IF)
