enable_language(C ASM)
# Python is usually installed in every Linux distribution...
find_package(
  Python3
  COMPONENTS Interpreter
  REQUIRED)
get_property(
  CUDART_LOCATION
  TARGET CUDA::cudart
  PROPERTY IMPORTED_LOCATION)
execute_process(
  COMMAND
    ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../3rdparty/implib/implib-gen.py
    ${CUDART_LOCATION} --target ${CMAKE_SYSTEM_PROCESSOR} --dlopen-callback
    DP_cudart_dlopen --dlsym-callback DP_cudart_dlsym COMMAND_ERROR_IS_FATAL ANY
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

file(GLOB CUDA_STUB_SRC ${CMAKE_CURRENT_BINARY_DIR}/*.tramp.S
     ${CMAKE_CURRENT_BINARY_DIR}/*.init.c)

add_library(deepmd_dyn_cudart SHARED cudart_stub.cc ${CUDA_STUB_SRC})
target_include_directories(deepmd_dyn_cudart PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
set_target_properties(
  deepmd_dyn_cudart PROPERTIES INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}"
                               BUILD_RPATH "${CUDAToolkit_LIBRARY_DIR}")
target_compile_definitions(deepmd_dyn_cudart PRIVATE IMPLIB_EXPORT_SHIMS)
if(BUILD_CPP_IF AND NOT BUILD_PY_IF)
  install(
    TARGETS deepmd_dyn_cudart
    EXPORT ${CMAKE_PROJECT_NAME}Targets
    DESTINATION lib/)
endif(BUILD_CPP_IF AND NOT BUILD_PY_IF)
if(BUILD_PY_IF)
  install(TARGETS deepmd_dyn_cudart DESTINATION deepmd/lib/)
endif(BUILD_PY_IF)
