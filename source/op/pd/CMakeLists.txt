file(GLOB OP_SRC comm.cc)
add_library(deepmd_op_pd MODULE ${OP_SRC})

# ========== 新增：Paddle 配置（路径由用户外部传入） ==========
if(NOT DEFINED PADDLE_LIB)
  message(
    FATAL_ERROR
      "please set PADDLE_LIB with -DPADDLE_LIB=/path/to/paddle_inference")
endif()

set(PADDLE_LIB_THIRD_PARTY_PATH ${PADDLE_LIB}/third_party/install)
include_directories(${PADDLE_LIB})
include_directories(${PADDLE_LIB}/paddle/include)
include_directories(${PADDLE_LIB_THIRD_PARTY_PATH}/protobuf/include)
include_directories(${PADDLE_LIB_THIRD_PARTY_PATH}/glog/include)
include_directories(${PADDLE_LIB_THIRD_PARTY_PATH}/gflags/include)
include_directories(${PADDLE_LIB_THIRD_PARTY_PATH}/xxhash/include)

# link_directories(${PADDLE_LIB_THIRD_PARTY_PATH}/protobuf/lib)
# link_directories(${PADDLE_LIB_THIRD_PARTY_PATH}/glog/lib)
# link_directories(${PADDLE_LIB_THIRD_PARTY_PATH}/gflags/lib)
# link_directories(${PADDLE_LIB_THIRD_PARTY_PATH}/xxhash/lib)
# link_directories(${PADDLE_LIB}/paddle/lib) target_link_libraries(deepmd_op_pd
# PRIVATE ${PADDLE_LIB_THIRD_PARTY_PATH}/glog/lib/libglog.a
# ${PADDLE_LIB_THIRD_PARTY_PATH}/gflags/lib/libgflags.a
# ${PADDLE_LIB_THIRD_PARTY_PATH}/xxhash/lib/libxxhash.a
# ${PADDLE_LIB_THIRD_PARTY_PATH}/protobuf/lib/libprotobuf.a dl pthread)

# Paddle lib
set(PADDLE_INFER_LIB ${PADDLE_LIB}/paddle/lib/libpaddle_inference.so)
# Math lib（默认用MKL）
set(MKLML_LIB ${PADDLE_LIB_THIRD_PARTY_PATH}/mklml/lib/libmklml_intel.so
              ${PADDLE_LIB_THIRD_PARTY_PATH}/mklml/lib/libiomp5.so)

# ========== 原有逻辑：保持不动 ==========
# target_link_libraries(deepmd_op_pd PRIVATE ${PADDLE_LIBRARIES})
# if(${OP_CXX_ABI_PT} EQUAL ${OP_CXX_ABI}) target_link_libraries(deepmd_op_pd
# PRIVATE ${LIB_DEEPMD}) else() target_link_libraries(deepmd_op_pd PRIVATE
# ${LIB_DEEPMD}_compat_cxxabi) endif()
# remove_definitions(-D_GLIBCXX_USE_CXX11_ABI=${OP_CXX_ABI})
# target_compile_definitions( deepmd_op_pd PUBLIC
# "$<$<COMPILE_LANGUAGE:CXX>:_GLIBCXX_USE_CXX11_ABI=${OP_CXX_ABI_PT}>")
# if(APPLE) set_target_properties(deepmd_op_pd PROPERTIES INSTALL_RPATH
# "@loader_path") else() set_target_properties(deepmd_op_pd PROPERTIES
# INSTALL_RPATH "$ORIGIN") endif()

find_package(MPI)
if(MPI_FOUND)
  include(CheckCXXSymbolExists)
  set(CMAKE_REQUIRED_INCLUDES ${MPI_CXX_INCLUDE_DIRS})
  set(CMAKE_REQUIRED_LIBRARIES ${MPI_CXX_LIBRARIES})
  check_cxx_symbol_exists(MPIX_Query_cuda_support "mpi.h" CUDA_AWARE)
  if(NOT CUDA_AWARE)
    check_cxx_symbol_exists(MPIX_Query_cuda_support "mpi.h;mpi-ext.h" OMP_CUDA)
    if(NOT OMP_CUDA)
      target_compile_definitions(deepmd_op_pd PRIVATE NO_CUDA_AWARE)
    endif()
  endif()
  target_link_libraries(deepmd_op_pd PRIVATE MPI::MPI_CXX)
  target_compile_definitions(deepmd_op_pd PRIVATE USE_MPI)
endif()
if(CMAKE_TESTING_ENABLED)
  target_link_libraries(deepmd_op_pd PRIVATE coverage_config)
endif()

# ========== 新增：链接 Paddle 所需库 ==========
target_link_libraries(
  deepmd_op_pd
  PRIVATE ${PADDLE_LIB_THIRD_PARTY_PATH}/glog/lib/libglog.a
          ${PADDLE_LIB_THIRD_PARTY_PATH}/gflags/lib/libgflags.a
          ${PADDLE_LIB_THIRD_PARTY_PATH}/xxhash/lib/libxxhash.a
          ${PADDLE_LIB_THIRD_PARTY_PATH}/protobuf/lib/libprotobuf.a
          ${PADDLE_INFER_LIB}
          dl
          pthread)

# ========== 保持原有安装路径逻辑 ==========
if(BUILD_PY_IF)
  install(TARGETS deepmd_op_pd DESTINATION deepmd/lib/)
else(BUILD_PY_IF)
  install(TARGETS deepmd_op_pd DESTINATION lib/)
endif(BUILD_PY_IF)
