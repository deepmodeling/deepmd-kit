file(GLOB OP_SRC comm.cc)

add_library(deepmd_op_pd MODULE ${OP_SRC})

if(NOT DEFINED PADDLE_INFERENCE_DIR)
  message(
    FATAL_ERROR
      "please set PADDLE_INFERENCE_DIR with -DPADDLE_INFERENCE_DIR=/path/to/paddle_inference"
  )
endif()

set(PADDLE_INFERENCE_DIR_THIRD_PARTY_PATH
    ${PADDLE_INFERENCE_DIR}/third_party/install)
include_directories(${PADDLE_INFERENCE_DIR})
include_directories(${PADDLE_INFERENCE_DIR}/paddle/include)
include_directories(${PADDLE_INFERENCE_DIR_THIRD_PARTY_PATH}/protobuf/include)
include_directories(${PADDLE_INFERENCE_DIR_THIRD_PARTY_PATH}/glog/include)
include_directories(${PADDLE_INFERENCE_DIR_THIRD_PARTY_PATH}/gflags/include)
include_directories(${PADDLE_INFERENCE_DIR_THIRD_PARTY_PATH}/xxhash/include)

# Paddle lib
set(PADDLE_INFER_LIB ${PADDLE_INFERENCE_DIR}/paddle/lib/libpaddle_inference.so)
# Math lib（默认用MKL）
set(MKLML_LIB
    ${PADDLE_INFERENCE_DIR_THIRD_PARTY_PATH}/mklml/lib/libmklml_intel.so
    ${PADDLE_INFERENCE_DIR_THIRD_PARTY_PATH}/mklml/lib/libiomp5.so)

# link: libdeepmd paddle - 添加 CXX ABI 兼容性处理 if(${OP_CXX_ABI_PT} EQUAL
# ${OP_CXX_ABI})
target_link_libraries(deepmd_op_pd PRIVATE ${LIB_DEEPMD})
# else() target_link_libraries(deepmd_op_pd PRIVATE ${LIB_DEEPMD}_compat_cxxabi)
# endif()

# 添加 CXX ABI 编译定义 remove_definitions(-D_GLIBCXX_USE_CXX11_ABI=${OP_CXX_ABI})
# target_compile_definitions( deepmd_op_pd PUBLIC
# "$<$<COMPILE_LANGUAGE:CXX>:_GLIBCXX_USE_CXX11_ABI=${OP_CXX_ABI_PT}>")

# 添加 RPATH 设置
if(APPLE)
  set_target_properties(deepmd_op_pd PROPERTIES INSTALL_RPATH "@loader_path")
else()
  set_target_properties(deepmd_op_pd PROPERTIES INSTALL_RPATH "$ORIGIN")
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
      target_compile_definitions(deepmd_op_pd PRIVATE NO_CUDA_AWARE)
    endif()
  endif()
  target_link_libraries(deepmd_op_pd PRIVATE MPI::MPI_CXX)
  target_compile_definitions(deepmd_op_pd PRIVATE USE_MPI)
endif()

if(CMAKE_TESTING_ENABLED)
  target_link_libraries(deepmd_op_pd PRIVATE coverage_config)
endif()

# 链接 Paddle 所需库
target_link_libraries(
  deepmd_op_pd
  PRIVATE ${PADDLE_INFERENCE_DIR_THIRD_PARTY_PATH}/glog/lib/libglog.a
          ${PADDLE_INFERENCE_DIR_THIRD_PARTY_PATH}/gflags/lib/libgflags.a
          ${PADDLE_INFERENCE_DIR_THIRD_PARTY_PATH}/xxhash/lib/libxxhash.a
          ${PADDLE_INFERENCE_DIR_THIRD_PARTY_PATH}/protobuf/lib/libprotobuf.a
          ${PADDLE_INFER_LIB}
          ${MKLML_LIB}
          dl
          pthread)

if(BUILD_PY_IF)
  install(TARGETS deepmd_op_pd DESTINATION deepmd/lib/)
else(BUILD_PY_IF)
  install(TARGETS deepmd_op_pd DESTINATION lib/)
endif(BUILD_PY_IF)
