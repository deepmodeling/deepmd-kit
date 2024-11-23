set(libname ${LIB_DEEPMD})

file(GLOB LIB_SRC src/*.cc src/*.cpp)
file(GLOB INC_SRC include/*.h ${CMAKE_CURRENT_BINARY_DIR}/version.h)

if(USE_CUDA_TOOLKIT)
  add_definitions("-DGOOGLE_CUDA")
  add_subdirectory(src/gpu)
endif()
if(USE_ROCM_TOOLKIT)
  add_definitions("-DTENSORFLOW_USE_ROCM")
  add_subdirectory(src/gpu)
endif()

function(create_library _suffix)
  set(libname_suffix "${libname}${_suffix}")
  add_library(${libname_suffix} SHARED ${LIB_SRC})
  target_include_directories(
    ${libname_suffix}
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
           $<INSTALL_INTERFACE:include>)

  if(USE_CUDA_TOOLKIT)
    target_link_libraries(${libname_suffix} PUBLIC deepmd_dyn_cudart)
    target_link_libraries(${libname_suffix} INTERFACE deepmd_op_cuda${_suffix})
    target_link_libraries(${libname_suffix} INTERFACE ${EXTRA_LIBS})
    # gpu_cuda.h
    target_include_directories(
      ${libname_suffix} PUBLIC $<BUILD_INTERFACE:${CUDAToolkit_INCLUDE_DIRS}>
                               $<INSTALL_INTERFACE:include>)
  endif()

  if(USE_ROCM_TOOLKIT)
    # to define __HIP_PLATFORM_AMD__ in hip_runtime.h
    target_link_libraries(${libname_suffix} PUBLIC hip::host)
    target_link_libraries(${libname_suffix} INTERFACE deepmd_op_rocm${_suffix})
    target_link_libraries(${libname_suffix} INTERFACE ${EXTRA_LIBS})
    # gpu_rocm.h
    target_include_directories(
      ${libname_suffix} PUBLIC $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
                               $<INSTALL_INTERFACE:include>)
  endif()

  set_target_properties(${libname_suffix} PROPERTIES INSTALL_RPATH $ORIGIN)

  if(CMAKE_TESTING_ENABLED)
    target_link_libraries(${libname_suffix} PRIVATE coverage_config)
  endif()

  if(BUILD_PY_IF)
    install(TARGETS ${libname_suffix} DESTINATION deepmd/lib/)
  else(BUILD_PY_IF)
    install(
      TARGETS ${libname_suffix}
      EXPORT ${CMAKE_PROJECT_NAME}Targets
      DESTINATION lib/)
  endif(BUILD_PY_IF)
endfunction()

remove_definitions(-D_GLIBCXX_USE_CXX11_ABI=${OP_CXX_ABI})
create_library("")
target_compile_definitions(
  ${libname}
  PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:_GLIBCXX_USE_CXX11_ABI=${OP_CXX_ABI}>")
if(DEEPMD_BUILD_COMPAT_CXXABI)
  create_library("_compat_cxxabi")
  target_compile_definitions(
    ${libname}_compat_cxxabi
    PUBLIC
      "$<$<COMPILE_LANGUAGE:CXX>:_GLIBCXX_USE_CXX11_ABI=${OP_CXX_ABI_COMPAT}>")
endif()

if(NOT BUILD_PY_IF)
  install(FILES ${INC_SRC} DESTINATION include/deepmd)
endif()

if(BUILD_CPP_IF AND CMAKE_TESTING_ENABLED)
  add_subdirectory(tests)
endif()
