add_library(deepmd_dyn_cudart SHARED cudart_stub.cc)
target_include_directories(deepmd_dyn_cudart PUBLIC ${CUDA_INCLUDE_DIRS})
get_filename_component(CUDA_LIBRARY_DIR ${CUDA_cudart_static_LIBRARY} DIRECTORY)
set_target_properties(deepmd_dyn_cudart PROPERTIES INSTALL_RPATH
                                                   "${CUDA_LIBRARY_DIR}")
if(BUILD_CPP_IF AND NOT BUILD_PY_IF)
  install(
    TARGETS deepmd_dyn_cudart
    EXPORT ${CMAKE_PROJECT_NAME}Targets
    DESTINATION lib/)
endif(BUILD_CPP_IF AND NOT BUILD_PY_IF)
if(BUILD_PY_IF)
  install(TARGETS deepmd_dyn_cudart DESTINATION deepmd/op/)
endif(BUILD_PY_IF)
