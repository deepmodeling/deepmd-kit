# libmd
set(libname ${LIB_DEEPMD})

file(GLOB LIB_SRC src/*.cc src/*.cpp)
file(GLOB INC_SRC include/*.h ${CMAKE_CURRENT_BINARY_DIR}/version.h)

add_library(${libname} SHARED ${LIB_SRC})
target_include_directories(
  ${libname} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                    $<INSTALL_INTERFACE:include>)

if(USE_CUDA_TOOLKIT)
  add_definitions("-DGOOGLE_CUDA")
  add_subdirectory(src/cuda)
  set(EXTRA_LIBS ${EXTRA_LIBS} deepmd_op_cuda)
  target_link_libraries(${libname} INTERFACE deepmd_dyn_cudart ${EXTRA_LIBS})
  # gpu_cuda.h
  target_include_directories(
    ${libname} PUBLIC $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
                      $<INSTALL_INTERFACE:include>)
endif()

if(USE_ROCM_TOOLKIT)
  add_definitions("-DTENSORFLOW_USE_ROCM")
  add_subdirectory(src/rocm)
  set(EXTRA_LIBS ${EXTRA_LIBS} deepmd_op_rocm)
  target_link_libraries(${libname} INTERFACE ${ROCM_LIBRARIES} ${EXTRA_LIBS})
  # gpu_rocm.h
  target_include_directories(
    ${libname} PUBLIC $<BUILD_INTERFACE:${ROCM_INCLUDE_DIRS}>
                      $<INSTALL_INTERFACE:include>)
endif()

set_target_properties(${libname} PROPERTIES INSTALL_RPATH $ORIGIN)

if(CMAKE_TESTING_ENABLED)
  target_link_libraries(${libname} PRIVATE coverage_config)
endif()

if(BUILD_PY_IF)
  install(TARGETS ${libname} DESTINATION deepmd/op/)
else(BUILD_PY_IF)
  install(
    TARGETS ${libname}
    EXPORT ${CMAKE_PROJECT_NAME}Targets
    DESTINATION lib/)
  install(FILES ${INC_SRC} DESTINATION include/deepmd)
endif(BUILD_PY_IF)

if(BUILD_CPP_IF AND CMAKE_TESTING_ENABLED)
  add_subdirectory(tests)
endif()
