enable_language(C ASM)
# Python is usually installed in every Linux distribution...
find_package(
  Python
  COMPONENTS Interpreter
  REQUIRED)
get_property(
  CUDART_LOCATION
  TARGET CUDA::cudart
  PROPERTY IMPORTED_LOCATION)
execute_process(
  COMMAND
    ${Python_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../3rdparty/implib/implib-gen.py
    ${CUDART_LOCATION} COMMAND_ERROR_IS_FATAL ANY
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

file(GLOB CUDA_STUB_SRC ${CMAKE_CURRENT_BINARY_DIR}/*.tramp.S
     ${CMAKE_CURRENT_BINARY_DIR}/*.init.c)

add_library(deepmd_dyn_cudart SHARED ${CUDA_STUB_SRC})
set_target_properties(deepmd_dyn_cudart PROPERTIES INSTALL_RPATH
                                                   "${CUDAToolkit_LIBRARY_DIR}")
if(BUILD_CPP_IF AND NOT BUILD_PY_IF)
  install(
    TARGETS deepmd_dyn_cudart
    EXPORT ${CMAKE_PROJECT_NAME}Targets
    DESTINATION lib/)
endif(BUILD_CPP_IF AND NOT BUILD_PY_IF)
if(BUILD_PY_IF)
  install(TARGETS deepmd_dyn_cudart DESTINATION deepmd/lib/)
endif(BUILD_PY_IF)
