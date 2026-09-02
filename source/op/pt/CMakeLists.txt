file(GLOB OP_SRC print_summary.cc comm.cc tabulate_multi_device.cc)
option(
  DEEPMD_CUDA_PORTABLE_PTX
  "Embed a lowest-supported PTX fallback in the PyTorch CUDA operator library"
  ON)

# Graph-lower operators of the CPU backend. The schemas are declared in their
# own translation units so that an operator stays visible when the CUDA half is
# absent, and each instruction-set instantiation is compiled with only that
# level's flags: the library ships in a wheel and in the LAMMPS deployment tree,
# so it must load on a host older than the one that built it and select its
# kernels through CPUID at run time.
set(DPA4C_CPU_ISA_SRC
    dpa4c/graph_compress_cpu_scalar.cc dpa4c/graph_compress_cpu_avx2.cc
    dpa4c/graph_compress_cpu_avx512.cc)
list(
  APPEND
  OP_SRC
  graph_ops_schema.cc
  cpu/allocator_policy.cc
  cpu/edge_force_virial_cpu.cc
  cpu/graph_fitting_cpu.cc
  cpu/neighbor_search_cpu.cc
  dpa4c/ops.cc
  dpa4c/graph_compress_cpu.cc
  ${DPA4C_CPU_ISA_SRC})
# The architecture level is pinned per unit rather than added to it: the project
# may be configured with ENABLE_NATIVE_OPTIMIZATION, and a trailing -march wins,
# so this is what keeps the fallback units free of instructions the dispatcher
# promised the host would not need. Other targets compile the same bodies at
# their baseline ISA and select only the scalar body at runtime.
if(NOT MSVC
   AND CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$"
   AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$"
   AND NOT CMAKE_OSX_ARCHITECTURES MATCHES "arm64")
  set_source_files_properties(dpa4c/graph_compress_cpu_scalar.cc
                              PROPERTIES COMPILE_OPTIONS "-march=x86-64")
  set_source_files_properties(dpa4c/graph_compress_cpu_avx2.cc
                              PROPERTIES COMPILE_OPTIONS "-march=x86-64-v3")
  set_source_files_properties(
    dpa4c/graph_compress_cpu_avx512.cc
    PROPERTIES COMPILE_OPTIONS
               "-march=x86-64-v4;-mprefer-vector-width=512;-mtune=native")
endif()
# Fused graph-lower inference operators (CUDA / cuBLAS). They include ATen CUDA
# headers and link libtorch_cuda, so they build only against a CUDA-enabled
# PyTorch (DEEPMD_TORCH_HAS_CUDA); against a CPU-only torch they are omitted and
# the Python dispatch falls back to the reference path (see
# deepmd.pt_expt.kernels.cuda.*.op_available). The CUDA language is scoped per
# directory, so it must be enabled here: the sibling GPU library turns it on
# only within its own subtree, which does not cover this target.
if(USE_CUDA_TOOLKIT AND DEEPMD_TORCH_HAS_CUDA)
  find_package(CUDAToolkit REQUIRED)
  option(DEEPMD_ENABLE_DPA1_HIGH_LMAX
         "Instantiate experimental DPA1 CUDA kernels for lmax greater than one"
         OFF)
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
  # One translation unit per (degree, focus width) so the twelve specializations
  # of the fused SO(2) convolution compile in parallel; a single unit per width
  # serializes six degrees behind one nvcc invocation.
  set(DPA4_SO2_CONV_KERNEL_SRC
      dpa4/so2_conv_fwd_c32_l1.cu
      dpa4/so2_conv_fwd_c32_l2.cu
      dpa4/so2_conv_fwd_c32_l3.cu
      dpa4/so2_conv_fwd_c32_l4.cu
      dpa4/so2_conv_fwd_c32_l5.cu
      dpa4/so2_conv_fwd_c32_l6.cu
      dpa4/so2_conv_fwd_c64_l1.cu
      dpa4/so2_conv_fwd_c64_l2.cu
      dpa4/so2_conv_fwd_c64_l3.cu
      dpa4/so2_conv_fwd_c64_l4.cu
      dpa4/so2_conv_fwd_c64_l5.cu
      dpa4/so2_conv_fwd_c64_l6.cu
      dpa4/so2_conv_bwd_c32_l1.cu
      dpa4/so2_conv_bwd_c32_l2.cu
      dpa4/so2_conv_bwd_c32_l3.cu
      dpa4/so2_conv_bwd_c32_l4.cu
      dpa4/so2_conv_bwd_c32_l5.cu
      dpa4/so2_conv_bwd_c32_l6.cu
      dpa4/so2_conv_bwd_c64_l1.cu
      dpa4/so2_conv_bwd_c64_l2.cu
      dpa4/so2_conv_bwd_c64_l3.cu
      dpa4/so2_conv_bwd_c64_l4.cu
      dpa4/so2_conv_bwd_c64_l5.cu
      dpa4/so2_conv_bwd_c64_l6.cu)
  # The rotation training kernels form a compile-time grid over degree, radial
  # rank and scalar type. Generate one source per grid point so Ninja can
  # compile the expensive template instantiations independently, while the
  # maintained source remains one shard template and one instantiation header.
  set(DPA4_ROTATE_MIX_TRAIN_KERNEL_SRC)
  set(DPA4_RMT_GENERATED_DIR
      "${CMAKE_CURRENT_BINARY_DIR}/dpa4/rotate_mix_train")
  set(DPA4_RMT_INSTANTIATE_HEADER
      "${CMAKE_CURRENT_SOURCE_DIR}/dpa4/rotate_mix_train/instantiate.cuh")
  file(MAKE_DIRECTORY "${DPA4_RMT_GENERATED_DIR}")
  foreach(DPA4_RMT_L RANGE 1 6)
    foreach(DPA4_RMT_RANK RANGE 0 4)
      foreach(DPA4_RMT_DTYPE IN ITEMS f32 f64 bf16)
        if(DPA4_RMT_DTYPE STREQUAL "f32")
          set(DPA4_RMT_TYPE float)
        elseif(DPA4_RMT_DTYPE STREQUAL "f64")
          set(DPA4_RMT_TYPE double)
        else()
          set(DPA4_RMT_TYPE c10::BFloat16)
        endif()
        set(DPA4_RMT_SHARD
            "${DPA4_RMT_GENERATED_DIR}/l${DPA4_RMT_L}_r${DPA4_RMT_RANK}_${DPA4_RMT_DTYPE}.cu"
        )
        configure_file(dpa4/rotate_mix_train/shard.cu.in "${DPA4_RMT_SHARD}"
                       @ONLY)
        list(APPEND DPA4_ROTATE_MIX_TRAIN_KERNEL_SRC "${DPA4_RMT_SHARD}")
      endforeach()
    endforeach()
  endforeach()
  unset(DPA4_RMT_SHARD)
  unset(DPA4_RMT_TYPE)
  unset(DPA4_RMT_DTYPE)
  unset(DPA4_RMT_RANK)
  unset(DPA4_RMT_L)
  unset(DPA4_RMT_INSTANTIATE_HEADER)
  unset(DPA4_RMT_GENERATED_DIR)
  # The training forward has a smaller template grid over degree and scalar
  # type. Keep the same generated-shard layout as rotate_mix_train so the four
  # dtype instantiations of a high degree do not serialize behind one nvcc
  # invocation.
  set(DPA4_SO2_CONV_TRAIN_KERNEL_SRC)
  set(DPA4_SCT_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/dpa4/so2_conv_train")
  set(DPA4_SCT_INSTANTIATE_HEADER
      "${CMAKE_CURRENT_SOURCE_DIR}/dpa4/so2_conv_train/instantiate.cuh")
  file(MAKE_DIRECTORY "${DPA4_SCT_GENERATED_DIR}")
  foreach(DPA4_SCT_L RANGE 1 6)
    foreach(DPA4_SCT_DTYPE IN ITEMS f32 f64 f16 bf16)
      if(DPA4_SCT_DTYPE STREQUAL "f32")
        set(DPA4_SCT_TYPE float)
      elseif(DPA4_SCT_DTYPE STREQUAL "f64")
        set(DPA4_SCT_TYPE double)
      elseif(DPA4_SCT_DTYPE STREQUAL "f16")
        set(DPA4_SCT_TYPE c10::Half)
      else()
        set(DPA4_SCT_TYPE c10::BFloat16)
      endif()
      set(DPA4_SCT_SHARD
          "${DPA4_SCT_GENERATED_DIR}/l${DPA4_SCT_L}_${DPA4_SCT_DTYPE}.cu")
      configure_file(dpa4/so2_conv_train/shard.cu.in "${DPA4_SCT_SHARD}" @ONLY)
      list(APPEND DPA4_SO2_CONV_TRAIN_KERNEL_SRC "${DPA4_SCT_SHARD}")
    endforeach()
  endforeach()
  unset(DPA4_SCT_SHARD)
  unset(DPA4_SCT_TYPE)
  unset(DPA4_SCT_DTYPE)
  unset(DPA4_SCT_L)
  unset(DPA4_SCT_INSTANTIATE_HEADER)
  unset(DPA4_SCT_GENERATED_DIR)
  set(DPA4C_GRAPH_COMPRESS_KERNEL_SRC
      dpa4c/graph_compress.cu dpa4c/graph_compress_c8.cu
      dpa4c/graph_compress_c16.cu dpa4c/graph_compress_c32.cu
      dpa4c/graph_compress_c64.cu dpa4c/graph_compress_c128.cu)
  set(DPA1_GRAPH_COMPRESS_KERNEL_SRC
      dpa1_graph_compress_c8.cu dpa1_graph_compress_c16.cu
      dpa1_graph_compress_c32.cu dpa1_graph_compress_c64.cu
      dpa1_graph_compress_c128.cu dpa1_graph_compress_c256.cu)
  list(
    APPEND
    OP_SRC
    dpa1_graph_descriptor.cu
    dpa1_graph_compress.cu
    ${DPA1_GRAPH_COMPRESS_KERNEL_SRC}
    ${DPA4C_GRAPH_COMPRESS_KERNEL_SRC}
    graph_fitting.cu
    edge_force_virial.cu
    dpa1_graph_energy_force.cu
    dpa4/so2_conv.cu
    ${DPA4_SO2_CONV_KERNEL_SRC}
    dpa4/mixing_train.cu
    dpa4/rotate_mix_train.cu
    ${DPA4_ROTATE_MIX_TRAIN_KERNEL_SRC}
    dpa4/so2_conv_train.cu
    ${DPA4_SO2_CONV_TRAIN_KERNEL_SRC}
    dpa4/grid_pair.cu
    dpa4/zonal_scatter.cu
    dpa4/edge_radial.cu
    dpa4/wigner_dense.cu)
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
  target_link_libraries(deepmd_op_pt PRIVATE CUDA::cublas CUDA::cublasLt)
  # libtorch headers require C++17; the CUDA sources must match.
  set_target_properties(deepmd_op_pt PROPERTIES CUDA_STANDARD 17
                                                CUDA_STANDARD_REQUIRED ON)
  if(DEEPMD_ENABLE_DPA1_HIGH_LMAX)
    target_compile_definitions(deepmd_op_pt
                               PRIVATE DEEPMD_ENABLE_DPA1_HIGH_LMAX=1)
  endif()
  # The compressed DPA1 and DPA4C kernels are instantiated one translation unit
  # per channel width so their angular-degree and topology specializations
  # compile in parallel.
  set_source_files_properties(${DPA4_SO2_CONV_KERNEL_SRC}
                              PROPERTIES COMPILE_OPTIONS "--threads=2")
  set_source_files_properties(
    ${DPA1_GRAPH_COMPRESS_KERNEL_SRC} ${DPA4C_GRAPH_COMPRESS_KERNEL_SRC}
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
