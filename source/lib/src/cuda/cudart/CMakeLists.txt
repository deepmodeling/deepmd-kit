add_library(deepmd_dyn_cudart SHARED cudart_stub.cc)
target_include_directories(
    deepmd_dyn_cudart
    PUBLIC
    ${CUDA_INCLUDE_DIRS}
)
install(
  TARGETS
  deepmd_dyn_cudart
  EXPORT ${CMAKE_PROJECT_NAME}Targets
  DESTINATION lib/)
