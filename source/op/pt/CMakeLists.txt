file(GLOB OP_SRC print_summary.cc comm.cc tabulate_multi_device.cc)
option(
  DEEPMD_CUDA_PORTABLE_PTX
  "Embed a lowest-supported PTX fallback in the PyTorch CUDA operator library"
  ON)
# Fused graph-lower inference operators (CUDA / cuBLAS). They include ATen CUDA
# headers and link libtorch_cuda, so they build only against a CUDA-enabled
# PyTorch (DEEPMD_TORCH_HAS_CUDA); against a CPU-only torch they are omitted and
# the Python dispatch falls back to the reference path (see
# deepmd.kernels.cuda.*.op_available). The CUDA language is scoped per
# directory, so it must be enabled here: the sibling GPU library turns it on
# only within its own subtree, which does not cover this target.
if(USE_CUDA_TOOLKIT AND DEEPMD_TORCH_HAS_CUDA)
  find_package(CUDAToolkit REQUIRED)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # CUDA 12.9 CCCL fails to compile CUB/Thrust with -arch=all.
    if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.9" AND CUDAToolkit_VERSION
                                                            VERSION_LESS "13.0")
      set(CMAKE_CUDA_ARCHITECTURES all-major)
    else()
      set(CMAKE_CUDA_ARCHITECTURES all)
    endif()
  endif()
  enable_language(CUDA)
  list(
    APPEND
    OP_SRC
    dpa1_graph_descriptor.cu
    dpa1_graph_compress.cu
    graph_fitting.cu
    edge_force_virial.cu
    dpa1_graph_energy_force.cu)
endif()

add_library(deepmd_op_pt MODULE ${OP_SRC})
# link: libdeepmd libtorch
target_link_libraries(deepmd_op_pt PRIVATE ${TORCH_LIBRARIES})
if(USE_CUDA_TOOLKIT AND DEEPMD_TORCH_HAS_CUDA)
  if(DEEPMD_CUDA_PORTABLE_PTX AND NOT CMAKE_CUDA_ARCHITECTURES MATCHES
                                  "^(all|all-major)$")
    if(CMAKE_CUDA_ARCHITECTURES AND NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "OFF")
      set(DEEPMD_PT_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
    else()
      set(DEEPMD_PT_CUDA_ARCHITECTURES)
    endif()
    if(CUDAToolkit_VERSION VERSION_LESS "13.0")
      list(APPEND DEEPMD_PT_CUDA_ARCHITECTURES "70-virtual")
    else()
      list(APPEND DEEPMD_PT_CUDA_ARCHITECTURES "75-virtual")
    endif()
    list(REMOVE_DUPLICATES DEEPMD_PT_CUDA_ARCHITECTURES)
    set_property(TARGET deepmd_op_pt PROPERTY CUDA_ARCHITECTURES
                                              ${DEEPMD_PT_CUDA_ARCHITECTURES})
  endif()
  target_link_libraries(deepmd_op_pt PRIVATE CUDA::cublas)
  # libtorch headers require C++17; the CUDA sources must match.
  set_target_properties(deepmd_op_pt PROPERTIES CUDA_STANDARD 17
                                                CUDA_STANDARD_REQUIRED ON)
  set_source_files_properties(dpa1_graph_compress.cu
                              PROPERTIES COMPILE_OPTIONS "--use_fast_math")
endif()
if(${OP_CXX_ABI_PT} EQUAL ${OP_CXX_ABI})
  target_link_libraries(deepmd_op_pt PRIVATE ${LIB_DEEPMD})
else()
  target_link_libraries(deepmd_op_pt PRIVATE ${LIB_DEEPMD}_compat_cxxabi)
endif()
remove_definitions(-D_GLIBCXX_USE_CXX11_ABI=${OP_CXX_ABI})
target_compile_definitions(
  deepmd_op_pt
  PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:_GLIBCXX_USE_CXX11_ABI=${OP_CXX_ABI_PT}>"
         "$<$<COMPILE_LANGUAGE:CUDA>:_GLIBCXX_USE_CXX11_ABI=${OP_CXX_ABI_PT}>")
if(APPLE)
  set_target_properties(deepmd_op_pt PROPERTIES INSTALL_RPATH "@loader_path")
else()
  set_target_properties(deepmd_op_pt PROPERTIES INSTALL_RPATH "$ORIGIN")
endif()

find_package(MPI)
if(MPI_FOUND)
  include(CheckCXXSymbolExists)
  set(CMAKE_REQUIRED_INCLUDES ${MPI_CXX_INCLUDE_DIRS})
  set(CMAKE_REQUIRED_LIBRARIES ${MPI_CXX_LIBRARIES})
  check_cxx_symbol_exists(MPIX_Query_cuda_support "mpi.h" CUDA_AWARE)
  if(NOT CUDA_AWARE)
    check_cxx_symbol_exists(MPIX_Query_cuda_support "mpi.h;mpi-ext.h" OMP_CUDA)
    if(NOT OMP_CUDA)
      target_compile_definitions(deepmd_op_pt PRIVATE NO_CUDA_AWARE)
    endif()
  endif()
  target_link_libraries(deepmd_op_pt PRIVATE MPI::MPI_CXX)
  target_compile_definitions(deepmd_op_pt PRIVATE USE_MPI)
endif()
if(CMAKE_TESTING_ENABLED)
  target_link_libraries(deepmd_op_pt PRIVATE coverage_config)
endif()

if(BUILD_PY_IF)
  install(TARGETS deepmd_op_pt DESTINATION deepmd/lib/)
else(BUILD_PY_IF)
  install(TARGETS deepmd_op_pt DESTINATION lib/)
endif(BUILD_PY_IF)
