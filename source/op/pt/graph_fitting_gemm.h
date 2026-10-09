// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <cublasLt.h>

#include <cstdint>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <tuple>

namespace deepmd_fitting {

inline void check_blas(cublasStatus_t status, const char* operation) {
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, operation,
              " failed with cuBLAS status ", static_cast<int>(status));
}

// This is a numerical reproducibility policy, not a precision switch. Read it
// once per process so a live model cannot silently change arithmetic policy.
inline bool shape_stable_enabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("DP_GRAPH_FITTING_GEMM_POLICY");
    const std::string policy = value ? value : "legacy";
    TORCH_CHECK(policy == "legacy" || policy == "shape_stable",
                "DP_GRAPH_FITTING_GEMM_POLICY must be legacy or shape_stable");
    return policy == "shape_stable";
  }();
  return enabled;
}

// cuBLASLt descriptors are host-side launch metadata. Keep them alive and
// update only the node-axis length; neither selecting an algorithm nor
// constructing descriptors belongs in the steady-state per-step path.
struct Layout {
  cublasLtMatrixLayout_t value = nullptr;
  Layout(uint64_t rows, uint64_t columns, int64_t leading) {
    check_blas(
        cublasLtMatrixLayoutCreate(&value, CUDA_R_32F, rows, columns, leading),
        "cublasLtMatrixLayoutCreate");
  }
  ~Layout() {
    if (value) {
      cublasLtMatrixLayoutDestroy(value);
    }
  }
  Layout(const Layout&) = delete;
  Layout& operator=(const Layout&) = delete;
  void columns(uint64_t columns) {
    check_blas(
        cublasLtMatrixLayoutSetAttribute(value, CUBLASLT_MATRIX_LAYOUT_COLS,
                                         &columns, sizeof(columns)),
        "cublasLtMatrixLayoutSetAttribute");
  }
};

struct Operation {
  cublasLtMatmulDesc_t value = nullptr;
  Operation() {
    check_blas(cublasLtMatmulDescCreate(&value, CUBLAS_COMPUTE_32F_PEDANTIC,
                                        CUDA_R_32F),
               "cublasLtMatmulDescCreate");
  }
  ~Operation() {
    if (value) {
      cublasLtMatmulDescDestroy(value);
    }
  }
  Operation(const Operation&) = delete;
  Operation& operator=(const Operation&) = delete;
};

struct Preference {
  cublasLtMatmulPreference_t value = nullptr;
  Preference() {
    check_blas(cublasLtMatmulPreferenceCreate(&value),
               "cublasLtMatmulPreferenceCreate");
  }
  ~Preference() {
    if (value) {
      cublasLtMatmulPreferenceDestroy(value);
    }
  }
  Preference(const Preference&) = delete;
  Preference& operator=(const Preference&) = delete;
};

// The fitting API supplies row-major node features. In column-major BLAS
// notation the variable node axis is N, while M and K depend only on weights.
// Pick a configuration using a fixed N=256, then submit the complete actual
// matrix. This does not tile windows or reduce the neural inference batch.
class Plan {
 public:
  static constexpr uint64_t canonical_nodes = 256;
  // Select using the canonical problem's budget, independently of the larger
  // scratch reservation needed by the same split-K algorithm on full batches.
  static constexpr size_t heuristic_workspace = 32 * 1024 * 1024;
  static constexpr size_t workspace_limit = 128 * 1024 * 1024;

  Plan(cublasLtHandle_t handle, int m, int k, bool transpose)
      : weight_(transpose ? k : m, transpose ? m : k, transpose ? k : m),
        features_(k, canonical_nodes, k),
        output_(m, canonical_nodes, m) {
    const cublasOperation_t trans = transpose ? CUBLAS_OP_T : CUBLAS_OP_N;
    check_blas(cublasLtMatmulDescSetAttribute(operation_.value,
                                              CUBLASLT_MATMUL_DESC_TRANSA,
                                              &trans, sizeof(trans)),
               "cublasLtMatmulDescSetAttribute");
    Preference preference;
    const size_t bytes = heuristic_workspace;
    check_blas(cublasLtMatmulPreferenceSetAttribute(
                   preference.value, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                   &bytes, sizeof(bytes)),
               "cublasLtMatmulPreferenceSetAttribute workspace");
    // Slices need only satisfy float alignment. Never let allocator addresses
    // choose a different reduction algorithm for otherwise identical rows.
    const uint32_t alignment = alignof(float);
    for (auto attribute : {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
                           CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
                           CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,
                           CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES}) {
      check_blas(
          cublasLtMatmulPreferenceSetAttribute(preference.value, attribute,
                                               &alignment, sizeof(alignment)),
          "cublasLtMatmulPreferenceSetAttribute alignment");
    }
    cublasLtMatmulHeuristicResult_t result{};
    int count = 0;
    check_blas(
        cublasLtMatmulAlgoGetHeuristic(
            handle, operation_.value, weight_.value, features_.value,
            output_.value, output_.value, preference.value, 1, &result, &count),
        "cublasLtMatmulAlgoGetHeuristic");
    TORCH_CHECK(count == 1 && result.state == CUBLAS_STATUS_SUCCESS,
                "No shape-stable FP32 fitting GEMM configuration for M=", m,
                ", K=", k, ", transpose=", transpose);
    algorithm_ = result.algo;
  }

  void run(cublasLtHandle_t handle,
           cudaStream_t stream,
           void* workspace,
           const float* weight,
           const float* features,
           float* output,
           int nodes,
           float beta) {
    if (nodes != last_nodes_) {
      // Invalidate before mutating either descriptor. If a layout update or
      // support check throws, the next call must restore and revalidate its
      // requested node count instead of trusting stale descriptor state.
      last_nodes_ = -1;
      features_.columns(nodes);
      output_.columns(nodes);
      cublasLtMatmulHeuristicResult_t support{};
      check_blas(cublasLtMatmulAlgoCheck(
                     handle, operation_.value, weight_.value, features_.value,
                     output_.value, output_.value, &algorithm_, &support),
                 "cublasLtMatmulAlgoCheck");
      TORCH_CHECK(support.state == CUBLAS_STATUS_SUCCESS &&
                      support.workspaceSize <= workspace_limit,
                  "Shape-stable fitting GEMM does not support this node count "
                  "within its 128 MiB workspace: ",
                  nodes, " (required bytes: ", support.workspaceSize,
                  ", status: ", static_cast<int>(support.state),
                  "). Refusing a shape-dependent algorithm fallback.");
      last_nodes_ = nodes;
    }
    const float alpha = 1.f;
    check_blas(cublasLtMatmul(handle, operation_.value, &alpha, weight,
                              weight_.value, features, features_.value, &beta,
                              output, output_.value, output, output_.value,
                              &algorithm_, workspace, workspace_limit, stream),
               "cublasLtMatmul shape-stable fitting");
  }

 private:
  Operation operation_;
  Layout weight_, features_, output_;
  cublasLtMatmulAlgo_t algorithm_{};
  int last_nodes_ = -1;
};

// A separate context per host thread, CUDA device, and stream prevents races
// in both launch metadata and scratch memory. A stream serializes repeated
// uses of its own workspace, with no device-wide synchronization in the loop.
class Context {
 public:
  explicit Context(int device) : device_(device) {
    workspace_ = torch::empty({static_cast<int64_t>(Plan::workspace_limit)},
                              torch::TensorOptions()
                                  .dtype(torch::kUInt8)
                                  .device(torch::kCUDA, device));
    check_blas(cublasLtCreate(&handle_), "cublasLtCreate");
  }
  ~Context() {
    // CUDA may already be shutting down. Destructors must not throw; ordinary
    // tensor destruction releases the allocator-owned scratch on its device.
    int previous = -1;
    if (cudaGetDevice(&previous) == cudaSuccess &&
        cudaSetDevice(device_) == cudaSuccess) {
      plans_.clear();
      if (handle_) {
        cublasLtDestroy(handle_);
      }
      cudaSetDevice(previous);
    }
  }
  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;

  void run(cudaStream_t stream,
           const float* features,
           const float* weight,
           float* output,
           int nodes,
           int width,
           int reduction,
           bool transpose,
           float beta) {
    TORCH_CHECK(nodes >= 0 && width > 0 && reduction > 0,
                "Invalid shape-stable fitting GEMM dimensions");
    if (nodes == 0) {
      return;
    }
    const auto key = std::make_tuple(width, reduction, transpose);
    auto& plan = plans_[key];
    if (!plan) {
      plan = std::make_unique<Plan>(handle_, width, reduction, transpose);
    }
    plan->run(handle_, stream, workspace_.data_ptr(), weight, features, output,
              nodes, beta);
  }

 private:
  int device_;
  cublasLtHandle_t handle_ = nullptr;
  torch::Tensor workspace_;
  std::map<std::tuple<int, int, bool>, std::unique_ptr<Plan>> plans_;
};

inline Context& context(cudaStream_t stream) {
  thread_local std::map<std::pair<int, std::uintptr_t>,
                        std::unique_ptr<Context>>
      contexts;
  const int device = c10::cuda::current_device();
  auto& result = contexts[std::make_pair(
      device, reinterpret_cast<std::uintptr_t>(stream))];
  if (!result) {
    result = std::make_unique<Context>(device);
  }
  return *result;
}

}  // namespace deepmd_fitting
