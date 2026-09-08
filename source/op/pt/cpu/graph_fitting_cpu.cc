// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Fused energy fitting network of the graph lower on the CPU.
//
//   h_0 = act(x @ W_0 + b_0)                 (+ identity residual when square)
//   h_l = act(h_{l-1} @ W_l + b_l)
//   e   = h_{L-1} @ w_head + b_head + bias_atom_e[atype]        (fp64 output)
//
// The layer products go to the BLAS the runtime already links, which is where
// the arithmetic belongs: a hand-written GEMM would have to beat a tuned
// library on its own ground. What the operator adds is the epilogue -- bias,
// activation and residual collapse into one pass over the layer output
// instead of three aten kernels each writing a node-scale tensor -- and the
// saved state, which is the pre-activation the GEMM already wrote, so the
// backward re-derives the activation derivative rather than storing it.

#include <ATen/Parallel.h>
#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <tuple>
#include <vector>

#include "../fitting_plan.h"
#include "activation.h"
#include "dispatch.h"

namespace {

/// Activation codes shared with the Triton and CUDA paths.
enum : int64_t { kTanh = 0, kSilu = 1 };

/// Nodes per parallel chunk of an elementwise epilogue.
///
/// The epilogues are bandwidth bound, so the chunk only has to be large
/// enough that the fork cost disappears against a row of a few hundred
/// floats.
constexpr int64_t kEpilogueGrain = 64;

/// Activation value.
///
/// The transcendentals come from ``activation.h`` rather than from the C
/// library, because a libm call is opaque to the vectorizer and would leave
/// every epilogue below running one channel at a time.
template <int64_t Act>
inline float activation(float z) {
  return Act == kTanh ? deepmd::fast_tanh(z) : z * deepmd::fast_sigmoid(z);
}

/// The state the forward leaves behind for the backward's derivative.
///
/// Tanh's derivative is algebraic in its own output, so storing the output
/// removes one transcendental evaluation per layer from the backward. Silu's
/// needs its argument, so the stored state is the biased pre-activation, which
/// also spares the backward the bias addition. Either way the forward leaves
/// exactly what ``derivative_from_state`` reads.
template <int64_t Act>
inline float activation_state(float biased, float value) {
  return Act == kTanh ? value : biased;
}

/// Activation derivative, from the state the forward stored.
template <int64_t Act>
inline float derivative_from_state(float state) {
  if (Act == kTanh) {
    return 1.0f - state * state;
  }
  const float s = deepmd::fast_sigmoid(state);
  return s * (1.0f + state * (1.0f - s));
}

/// Bias, activation and optional identity residual of one layer.
///
/// The pre-activation buffer is overwritten in place with the state the
/// backward needs, so the layer costs one pass whatever that state is.
template <int64_t Act>
void layer_epilogue(int64_t nodes,
                    int64_t width,
                    float* DEEPMD_RESTRICT pre,
                    const float* DEEPMD_RESTRICT bias,
                    const float* DEEPMD_RESTRICT residual,
                    float* DEEPMD_RESTRICT out) {
  at::parallel_for(0, nodes, kEpilogueGrain, [&](int64_t begin, int64_t end) {
    for (int64_t node = begin; node < end; ++node) {
      float* DEEPMD_RESTRICT row = pre + node * width;
      float* DEEPMD_RESTRICT target = out + node * width;
      if (residual != nullptr) {
        const float* DEEPMD_RESTRICT skip = residual + node * width;
        for (int64_t channel = 0; channel < width; ++channel) {
          const float biased = row[channel] + bias[channel];
          const float value = activation<Act>(biased);
          row[channel] = activation_state<Act>(biased, value);
          target[channel] = value + skip[channel];
        }
      } else {
        for (int64_t channel = 0; channel < width; ++channel) {
          const float biased = row[channel] + bias[channel];
          const float value = activation<Act>(biased);
          row[channel] = activation_state<Act>(biased, value);
          target[channel] = value;
        }
      }
    }
  });
}

/// Per-atom energy of the linear head, accumulated in double.
void head(int64_t nodes,
          int64_t width,
          const float* DEEPMD_RESTRICT activation_in,
          const float* DEEPMD_RESTRICT weight,
          float head_bias,
          const double* DEEPMD_RESTRICT atom_bias,
          const int64_t* DEEPMD_RESTRICT atype,
          double* DEEPMD_RESTRICT energy) {
  at::parallel_for(0, nodes, kEpilogueGrain, [&](int64_t begin, int64_t end) {
    for (int64_t node = begin; node < end; ++node) {
      const float* DEEPMD_RESTRICT row = activation_in + node * width;
      float total = 0.0f;
      for (int64_t channel = 0; channel < width; ++channel) {
        total += row[channel] * weight[channel];
      }
      energy[node] = static_cast<double>(total) +
                     static_cast<double>(head_bias) + atom_bias[atype[node]];
    }
  });
}

/// Seed the backward from the head cotangent, then convert it in place.
///
/// The last layer's output cotangent is the outer product of the per-node
/// energy cotangent with the head weight, which is cheaper to form here than
/// to materialize as a tensor.
template <int64_t Act>
void seed_epilogue(int64_t nodes,
                   int64_t width,
                   const double* DEEPMD_RESTRICT energy_cotangent,
                   const float* DEEPMD_RESTRICT head_weight,
                   const float* DEEPMD_RESTRICT state,
                   float* DEEPMD_RESTRICT pre_cotangent,
                   float* DEEPMD_RESTRICT residual_out) {
  at::parallel_for(0, nodes, kEpilogueGrain, [&](int64_t begin, int64_t end) {
    for (int64_t node = begin; node < end; ++node) {
      const float seed = static_cast<float>(energy_cotangent[node]);
      const float* DEEPMD_RESTRICT row = state + node * width;
      float* DEEPMD_RESTRICT target = pre_cotangent + node * width;
      if (residual_out != nullptr) {
        float* DEEPMD_RESTRICT skip = residual_out + node * width;
        for (int64_t channel = 0; channel < width; ++channel) {
          const float upstream = seed * head_weight[channel];
          skip[channel] = upstream;
          target[channel] = upstream * derivative_from_state<Act>(row[channel]);
        }
      } else {
        for (int64_t channel = 0; channel < width; ++channel) {
          target[channel] = seed * head_weight[channel] *
                            derivative_from_state<Act>(row[channel]);
        }
      }
    }
  });
}

/// Convert an output cotangent into a pre-activation cotangent in place.
template <int64_t Act>
void backward_epilogue(int64_t nodes,
                       int64_t width,
                       const float* DEEPMD_RESTRICT state,
                       float* DEEPMD_RESTRICT cotangent,
                       float* DEEPMD_RESTRICT residual_out) {
  at::parallel_for(0, nodes, kEpilogueGrain, [&](int64_t begin, int64_t end) {
    for (int64_t node = begin; node < end; ++node) {
      const float* DEEPMD_RESTRICT row = state + node * width;
      float* DEEPMD_RESTRICT target = cotangent + node * width;
      if (residual_out != nullptr) {
        float* DEEPMD_RESTRICT skip = residual_out + node * width;
        for (int64_t channel = 0; channel < width; ++channel) {
          skip[channel] = target[channel];
        }
      }
      for (int64_t channel = 0; channel < width; ++channel) {
        target[channel] *= derivative_from_state<Act>(row[channel]);
      }
    }
  });
}

/// Dispatch over the two supported activations.
template <typename Body>
void dispatch_activation(int64_t act, Body&& body) {
  if (act == kTanh) {
    body(std::integral_constant<int64_t, kTanh>{});
  } else {
    body(std::integral_constant<int64_t, kSilu>{});
  }
}

/// Wrap a raw buffer as a node-major matrix without copying.
torch::Tensor as_matrix(float* data,
                        int64_t nodes,
                        int64_t width,
                        const torch::TensorOptions& options) {
  return torch::from_blob(data, {nodes, width}, options);
}

/// Evaluate every layer of one node range and its per-atom energy.
void fitting_forward_range(const FittingLayerPlan& plan,
                           const float* input,
                           int64_t input_width,
                           const int64_t* atype,
                           const std::vector<torch::Tensor>& ws,
                           const std::vector<torch::Tensor>& bs,
                           const std::vector<int64_t>& resnets,
                           const torch::Tensor& w_head,
                           const torch::Tensor& b_head,
                           const torch::Tensor& bias_atom_e,
                           int64_t act,
                           int64_t nodes,
                           float* saved,
                           float* const activation_slot[2],
                           double* energy) {
  const at::NoGradGuard guard;
  const auto options = ws[0].options();
  const float* current = input;
  int64_t width_in = input_width;
  for (int layer = 0; layer < plan.n_layer; ++layer) {
    const int64_t width_out = ws[layer].size(1);
    float* pre = saved + plan.offset[layer] * nodes;
    float* out = activation_slot[layer & 1];
    auto pre_matrix = as_matrix(pre, nodes, width_out, options);
    torch::mm_out(pre_matrix,
                  torch::from_blob(const_cast<float*>(current),
                                   {nodes, width_in}, options),
                  ws[layer]);
    const bool residual = resnets[layer] && width_out == width_in;
    dispatch_activation(act, [&](auto tag) {
      layer_epilogue<decltype(tag)::value>(
          nodes, width_out, pre,
          bs[layer].numel() ? bs[layer].const_data_ptr<float>() : nullptr,
          residual ? current : nullptr, out);
    });
    current = out;
    width_in = width_out;
  }
  head(nodes, width_in, current, w_head.const_data_ptr<float>(),
       b_head.numel() ? b_head.const_data_ptr<float>()[0] : 0.0f,
       bias_atom_e.const_data_ptr<double>(), atype, energy);
}

/// Propagate the head cotangent of one node range back to the input.
void fitting_backward_range(const FittingLayerPlan& plan,
                            const double* energy_cotangent,
                            const float* saved,
                            const std::vector<torch::Tensor>& ws,
                            const std::vector<int64_t>& resnets,
                            const torch::Tensor& w_head,
                            int64_t act,
                            int64_t nodes,
                            float* cotangent,
                            float* cotangent_next,
                            float* input_cotangent) {
  const at::NoGradGuard guard;
  const auto options = ws[0].options();
  for (int layer = plan.n_layer - 1; layer >= 0; --layer) {
    const int64_t width_out = ws[layer].size(1);
    const int64_t width_in = ws[layer].size(0);
    const float* state = saved + plan.offset[layer] * nodes;
    float* out = layer > 0 ? cotangent_next : input_cotangent;
    const bool residual = resnets[layer] && width_out == width_in;
    dispatch_activation(act, [&](auto tag) {
      constexpr int64_t kAct = decltype(tag)::value;
      if (layer == plan.n_layer - 1) {
        seed_epilogue<kAct>(nodes, width_out, energy_cotangent,
                            w_head.const_data_ptr<float>(), state, cotangent,
                            residual ? out : nullptr);
      } else {
        backward_epilogue<kAct>(nodes, width_out, state, cotangent,
                                residual ? out : nullptr);
      }
    });
    auto out_matrix = as_matrix(out, nodes, width_in, options);
    auto cotangent_matrix = as_matrix(cotangent, nodes, width_out, options);
    if (residual) {
      out_matrix.addmm_(cotangent_matrix, ws[layer].t());
    } else {
      torch::mm_out(out_matrix, cotangent_matrix, ws[layer].t());
    }
    if (layer > 0) {
      std::swap(cotangent, cotangent_next);
    }
  }
}

/// Validate the inputs the operator's arithmetic assumes.
FittingLayerPlan validate(const char* operation,
                          const torch::Tensor& x,
                          const torch::Tensor& atype,
                          const std::vector<torch::Tensor>& ws,
                          const torch::Tensor& bias_atom_e) {
  TORCH_CHECK(x.dim() == 2 && x.device().is_cpu() && x.is_contiguous() &&
                  x.scalar_type() == torch::kFloat32,
              operation, ": x must be contiguous CPU fp32 with shape (N, D)");
  TORCH_CHECK(atype.dim() == 1 && atype.size(0) == x.size(0) &&
                  atype.device().is_cpu() && atype.is_contiguous() &&
                  atype.scalar_type() == torch::kInt64,
              operation,
              ": atype must be contiguous CPU int64 with shape (N,)");
  const FittingLayerPlan plan = fitting_layer_plan(ws);
  TORCH_CHECK(
      plan.n_layer > 0 && ws[0].dim() == 2 && ws[0].size(0) == x.size(1),
      operation, ": the first fitting weight must match the input width");
  TORCH_CHECK(bias_atom_e.dim() == 1 && bias_atom_e.device().is_cpu() &&
                  bias_atom_e.is_contiguous() &&
                  bias_atom_e.scalar_type() == torch::kFloat64,
              operation, ": bias_atom_e must be contiguous CPU fp64");
  // Validate every index before the tiled energy-gradient operator overwrites
  // descriptor rows with their cotangents. Checking in the head epilogue could
  // leave the input partially modified when a later tile contains an invalid
  // atom type.
  if (atype.numel() != 0) {
    const int64_t* begin = atype.const_data_ptr<int64_t>();
    const auto [min_type, max_type] =
        std::minmax_element(begin, begin + atype.numel());
    TORCH_CHECK_INDEX(*min_type >= 0 && *max_type < bias_atom_e.numel(),
                      operation, ": atype values must satisfy 0 <= atype < ",
                      bias_atom_e.numel(), ", but got range [", *min_type, ", ",
                      *max_type, "]");
  }
  return plan;
}

std::tuple<torch::Tensor, torch::Tensor> graph_fitting(
    torch::Tensor x,
    torch::Tensor atype,
    std::vector<torch::Tensor> ws,
    std::vector<torch::Tensor> bs,
    std::vector<int64_t> resnets,
    torch::Tensor w_head,
    torch::Tensor b_head,
    torch::Tensor bias_atom_e,
    int64_t act) {
  const FittingLayerPlan plan =
      validate("graph_fitting", x, atype, ws, bias_atom_e);
  const int64_t nodes = x.size(0);
  auto options = x.options();
  auto energy = torch::empty({nodes, 1}, options.dtype(torch::kFloat64));
  auto saved = torch::empty({nodes * plan.saved_width()}, options);
  const int slots = plan.n_layer > 1 ? 2 : 1;
  auto scratch = torch::empty({slots, nodes, plan.width_max}, options);
  if (nodes == 0) {
    return {energy, saved};
  }
  float* slot[2] = {
      scratch[0].data_ptr<float>(),
      slots > 1 ? scratch[1].data_ptr<float>() : scratch[0].data_ptr<float>()};
  fitting_forward_range(plan, x.const_data_ptr<float>(), x.size(1),
                        atype.const_data_ptr<int64_t>(), ws, bs, resnets,
                        w_head, b_head, bias_atom_e, act, nodes,
                        saved.data_ptr<float>(), slot,
                        energy.data_ptr<double>());
  return {energy, saved};
}

torch::Tensor graph_fitting_backward(torch::Tensor d_e,
                                     torch::Tensor saved,
                                     std::vector<torch::Tensor> ws,
                                     std::vector<torch::Tensor> bs,
                                     std::vector<int64_t> resnets,
                                     torch::Tensor w_head,
                                     int64_t act) {
  const FittingLayerPlan plan = fitting_layer_plan(ws);
  TORCH_CHECK(plan.saved_width() > 0 && saved.numel() % plan.saved_width() == 0,
              "graph_fitting_backward: the saved buffer does not match the "
              "layer widths");
  const int64_t nodes = saved.numel() / plan.saved_width();
  auto options = saved.options();
  auto input_cotangent = torch::empty({nodes, ws[0].size(0)}, options);
  if (nodes == 0) {
    return input_cotangent;
  }
  auto cotangent = torch::empty({nodes, plan.width_max}, options);
  auto cotangent_next = plan.n_layer > 1
                            ? torch::empty({nodes, plan.width_max}, options)
                            : torch::empty({0}, options);
  fitting_backward_range(
      plan, d_e.to(torch::kFloat64).contiguous().const_data_ptr<double>(),
      saved.const_data_ptr<float>(), ws, resnets, w_head, act, nodes,
      cotangent.data_ptr<float>(),
      plan.n_layer > 1 ? cotangent_next.data_ptr<float>() : nullptr,
      input_cotangent.data_ptr<float>());
  return input_cotangent;
}

// Energy and descriptor cotangent of one inference step, evaluated over runs
// of nodes so that the layer activations of a run retire before the next
// begins. The descriptor buffer is overwritten with its own cotangent, which
// the caller no longer needs in its forward form.
torch::Tensor graph_fitting_energy_gradient(torch::Tensor x,
                                            torch::Tensor atype,
                                            std::vector<torch::Tensor> ws,
                                            std::vector<torch::Tensor> bs,
                                            std::vector<int64_t> resnets,
                                            torch::Tensor w_head,
                                            torch::Tensor b_head,
                                            torch::Tensor bias_atom_e,
                                            int64_t act,
                                            torch::Tensor seed,
                                            int64_t tile) {
  const FittingLayerPlan plan =
      validate("graph_fitting_energy_gradient", x, atype, ws, bias_atom_e);
  const int64_t nodes = x.size(0);
  auto options = x.options();
  auto energy = torch::empty({nodes, 1}, options.dtype(torch::kFloat64));
  if (nodes == 0) {
    return energy;
  }
  auto seed_contiguous = seed.to(torch::kFloat64).contiguous();
  TORCH_CHECK(seed_contiguous.numel() == nodes,
              "graph_fitting_energy_gradient: seed must carry one entry per "
              "node");
  const int64_t run =
      tile > 0 ? std::max<int64_t>(1, std::min<int64_t>(tile, nodes)) : nodes;
  const int slots = plan.n_layer > 1 ? 2 : 1;
  auto saved = torch::empty({run * plan.saved_width()}, options);
  auto scratch = torch::empty({slots, run, plan.width_max}, options);
  auto cotangent = torch::empty({run, plan.width_max}, options);
  auto cotangent_next = plan.n_layer > 1
                            ? torch::empty({run, plan.width_max}, options)
                            : torch::empty({0}, options);
  float* slot[2] = {
      scratch[0].data_ptr<float>(),
      slots > 1 ? scratch[1].data_ptr<float>() : scratch[0].data_ptr<float>()};
  const int64_t width = x.size(1);
  float* descriptor = x.data_ptr<float>();
  for (int64_t begin = 0; begin < nodes; begin += run) {
    const int64_t count = std::min(run, nodes - begin);
    fitting_forward_range(plan, descriptor + begin * width, width,
                          atype.const_data_ptr<int64_t>() + begin, ws, bs,
                          resnets, w_head, b_head, bias_atom_e, act, count,
                          saved.data_ptr<float>(), slot,
                          energy.data_ptr<double>() + begin);
    fitting_backward_range(
        plan, seed_contiguous.const_data_ptr<double>() + begin,
        saved.const_data_ptr<float>(), ws, resnets, w_head, act, count,
        cotangent.data_ptr<float>(),
        plan.n_layer > 1 ? cotangent_next.data_ptr<float>() : nullptr,
        descriptor + begin * width);
  }
  return energy;
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(deepmd, library) {
  library.impl("graph_fitting", torch::kCPU, &graph_fitting);
  library.impl("graph_fitting_backward", torch::kCPU, &graph_fitting_backward);
  library.impl("graph_fitting_energy_gradient", torch::kCPU,
               &graph_fitting_energy_gradient);
}
