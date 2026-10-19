// SPDX-License-Identifier: LGPL-3.0-or-later
// Cases of the device-resident entry point of the compressed DPA4C archives
// frozen with the compact canonical lower
// (source/tests/infer/gen_dpa4c_canonical.py): an energy model, evaluated
// through a DeepPot interface, and a native-spin model, evaluated through a
// DeepSpin interface. The C++ suite instantiates them for deepmd::DeepPot and
// deepmd::DeepSpin, the C suite for deepmd::hpp::DeepPot and
// deepmd::hpp::DeepSpin and so for DP_DPA4ComputeCanonicalGraphGPU and
// DP_DPA4SpinComputeCanonicalGraphGPU. The including file provides
// _build_nlist through the test_utils.h of its suite.
//
// compute_canonical_graph_gpu evaluates a graph whose buffers, like its
// outputs, reside on the model device. The cases hand it the graph that the
// host entry points build from the same coordinates, so the two entry points
// evaluate one program on equal inputs and agree exactly. The graph of one
// rank has no halo nodes, so the repeated evaluations of a case run eagerly,
// are captured into a CUDA graph and are replayed. Every evaluation writes
// into outputs reset to NaN, and no case may fall back from the replay to
// eager evaluation, so the cases check the results of the replay itself: on
// the current stream and on a caller's stream, without the optional outputs
// and inputs, after its inputs change in place, across a change of its key,
// while another thread holds a capture of its own open, and under a charge
// state applied at run time. A graph with halo nodes, which domain
// decomposition hands over, is evaluated eagerly instead.
//
// The cases form a typed test suite in this namespace, which each suite
// instantiates in the same namespace.
#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "DeepPotPTExpt.h"
#include "expected_ref.h"

#if defined(BUILD_PYTORCH) && BUILD_PT_EXPT
#include <torch/torch.h>

#include "commonPT.h"
#ifdef DEEPMD_TORCH_HAS_CUDA
#include <ATen/cuda/Sleep.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>

#include <condition_variable>
#include <mutex>
#include <thread>
#endif
#endif

namespace deepmd_test {
namespace dpa4c_canonical {

// The charge state applied at run time; the archives are frozen with another.
const std::vector<double> kOtherChargeSpin = {1.0, 2.0};

// The fixtures are fp32 inside; the references come from the Python
// inference of the same archives on graphs of another edge order.
constexpr double kReferenceTolerance = 1e-5;

// Uniform strain of the system between two evaluations of one graph. Every
// atom keeps its fractional coordinates, and so its cell of the neighbor
// search, and every distance changes by far less than its margin to the
// cutoff, so the graph keeps its edges and their order while every edge
// vector changes.
constexpr double kStrain = 1e-4;

// Cycles a caller's stream is held back before an evaluation is enqueued on
// it: tens of milliseconds, far longer than the evaluation.
constexpr std::int64_t kHoldCycles = std::int64_t{1} << 27;

// The warning of a replay whose capture failed, which evaluates eagerly from
// then on.
constexpr const char* kCaptureWarning = "could not capture the evaluation";

/// The energy model, evaluated through the DeepPot interface ``DP``.
template <class DP>
struct Potential {
  using Evaluator = DP;
  static constexpr bool kSpin = false;
  static constexpr const char* kName = "Potential";
  static constexpr const char* kModelPath =
      "../../tests/infer/deeppot_dpa4c_canonical.pt2";
  static constexpr const char* kRefPath =
      "../../tests/infer/deeppot_dpa4c_canonical.expected";
};

/// The native-spin model, evaluated through the DeepSpin interface ``DS``.
template <class DS>
struct Spin {
  using Evaluator = DS;
  static constexpr bool kSpin = true;
  static constexpr const char* kName = "Spin";
  static constexpr const char* kModelPath =
      "../../tests/infer/deeppot_dpa4c_canonical_spin.pt2";
  static constexpr const char* kRefPath =
      "../../tests/infer/deeppot_dpa4c_canonical_spin.expected";
};

/// Names the instantiations of the cases by their model.
struct CaseNames {
  template <class Case>
  static std::string GetName(int) {
    return Case::kName;
  }
};

/// A configuration of the system: coordinates, magnetic moments (empty for
/// the energy model), atom types and cell.
struct Configuration {
  std::vector<double> coord;
  std::vector<double> spin;
  std::vector<int> atype;
  std::vector<double> box;
};

/// Results of one evaluation, on the host. An output the evaluation was not
/// asked for, or that the model does not produce, is empty.
struct Results {
  std::vector<double> atom_energy;
  std::vector<double> force;
  std::vector<double> force_mag;
  std::vector<double> virial;
  std::vector<double> atom_virial;
};

/// Largest absolute difference of two equally long arrays, NaN when an entry
/// of either is NaN.
inline double max_difference(const std::vector<double>& actual,
                             const std::vector<double>& expected) {
  EXPECT_EQ(actual.size(), expected.size());
  double difference = 0.0;
  for (std::size_t ii = 0; ii < std::min(actual.size(), expected.size());
       ++ii) {
    const double entry = std::abs(actual[ii] - expected[ii]);
    if (std::isnan(entry)) {
      return entry;
    }
    difference = std::max(difference, entry);
  }
  return difference;
}

/// Largest magnitude of an array.
inline double largest(const std::vector<double>& values) {
  double magnitude = 0.0;
  for (const double value : values) {
    magnitude = std::max(magnitude, std::abs(value));
  }
  return magnitude;
}

/// Sum the per-node rows of ``values``, three per node, onto the atoms that
/// own the nodes.
inline std::vector<double> fold_onto_owners(const std::vector<double>& values,
                                            const std::vector<int>& owner,
                                            const int natoms) {
  std::vector<double> folded(3 * static_cast<std::size_t>(natoms), 0.0);
  for (std::size_t node = 0; node < owner.size(); ++node) {
    for (int dd = 0; dd < 3; ++dd) {
      folded[3 * owner[node] + dd] += values[3 * node + dd];
    }
  }
  return folded;
}

#if defined(BUILD_PYTORCH) && BUILD_PT_EXPT && defined(DEEPMD_TORCH_HAS_CUDA)
/// A CUDA graph capture that another thread holds open on a stream of its own
/// from construction until ``close``.
class CaptureElsewhere {
 public:
  CaptureElsewhere() : thread_([this] { hold(); }) {
    std::unique_lock<std::mutex> lock(mutex_);
    changed_.wait(lock, [this] { return open_; });
  }
  ~CaptureElsewhere() { close(); }

  /// End the capture and return the status with which it ended:
  /// ``cudaSuccess`` unless something invalidated it while it was open.
  cudaError_t close() {
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      closing_ = true;
    }
    changed_.notify_all();
    if (thread_.joinable()) {
      thread_.join();
    }
    return status_;
  }

 private:
  void hold() {
    const c10::cuda::CUDAStream stream =
        c10::cuda::getStreamFromPool(/*isHighPriority=*/false, /*device=*/0);
    const cudaError_t begun = cudaStreamBeginCapture(
        stream.stream(), cudaStreamCaptureModeThreadLocal);
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      open_ = true;
    }
    changed_.notify_all();
    {
      std::unique_lock<std::mutex> lock(mutex_);
      changed_.wait(lock, [this] { return closing_; });
    }
    cudaGraph_t graph = nullptr;
    status_ = begun == cudaSuccess
                  ? cudaStreamEndCapture(stream.stream(), &graph)
                  : begun;
    if (graph != nullptr) {
      cudaGraphDestroy(graph);
    }
    // A failed capture leaves its error for the next query of this thread.
    cudaGetLastError();
  }

  std::mutex mutex_;
  std::condition_variable changed_;
  bool open_ = false;
  bool closing_ = false;
  cudaError_t status_ = cudaSuccess;
  // Declared last, so that the thread starts after the members it uses.
  std::thread thread_;
};
#endif

/// The device-resident entry point of the model of ``Case``.
template <class Case>
class TestInferDpa4cCanonicalPtExpt : public ::testing::Test {
 protected:
  using Evaluator = typename Case::Evaluator;

  Configuration system;
  int natoms = 0;
  Evaluator dp;
  ExpectedRef ref;
  Results host;
  bool capturing_stderr = false;

#if defined(BUILD_PYTORCH) && BUILD_PT_EXPT
  /// A compact canonical graph on the device, with the owner of each node
  /// and, for the spin model, the moment of each node.
  struct Graph {
    deepmd::CanonicalGraphTensorPack tensors;
    torch::Tensor spin;
    std::vector<int> owner;
  };

  // The compact canonical graph of the host entry points, on the device.
  torch::Tensor graph_atype;
  torch::Tensor graph_source;
  torch::Tensor graph_edge_vec;
  torch::Tensor graph_destination_row_ptr;
  torch::Tensor graph_source_row_ptr;
  torch::Tensor graph_source_order;
  torch::Tensor graph_spin;
  int nodes = 0;
  std::int64_t edge_storage = 0;
  // Outputs reused by the evaluations of a case, whose addresses the replay
  // of repeated evaluations keys on.
  torch::Tensor out_atom_energy;
  torch::Tensor out_force;
  torch::Tensor out_force_mag;
  torch::Tensor out_virial;
  torch::Tensor out_atom_virial;
#endif

  void SetUp() override {
#if !defined(BUILD_PYTORCH) || !BUILD_PT_EXPT
    GTEST_SKIP() << "Skip because PyTorch support is not enabled.";
#else
    if (!torch::cuda::is_available()) {
      GTEST_SKIP() << "Skip because the compact canonical path needs a CUDA "
                      "device.";
    }
    std::ifstream model_file(Case::kModelPath);
    if (!model_file.good()) {
      GTEST_SKIP() << "Skip because " << Case::kModelPath
                   << " was not generated (run "
                      "source/tests/infer/gen_dpa4c_canonical.py).";
    }
    dp.init(Case::kModelPath);
    ASSERT_TRUE(dp.uses_canonical_graph_inference());
    if constexpr (!Case::kSpin) {
      ASSERT_FALSE(dp.has_long_range());
    }
    ref.load(Case::kRefPath);
    system.coord = ref.get<double>("system", "coord");
    system.box = ref.get<double>("system", "box");
    if constexpr (Case::kSpin) {
      system.spin = ref.get<double>("system", "spin");
    }
    for (const double type : ref.get<double>("system", "atype")) {
      system.atype.push_back(static_cast<int>(type));
    }
    natoms = static_cast<int>(system.atype.size());
    host = compute_host(dp, system);

    use_graph(build_graph(system, /*fold=*/true));
    // One rank folds the ghost atoms onto their owners: no halo node.
    ASSERT_EQ(nodes, natoms);
    allocate_outputs();
    // Some cases evaluate on a caller's stream, which is ordered after none
    // of the work that built the graph.
    torch::cuda::synchronize();
    // A replay whose capture fails evaluates eagerly instead, with the same
    // results; only its warning tells the two apart.
    testing::internal::CaptureStderr();
    capturing_stderr = true;
#endif
  }

  void TearDown() override {
    if (!capturing_stderr) {
      return;
    }
    const std::string log = testing::internal::GetCapturedStderr();
    std::cerr << log;
    EXPECT_EQ(log.find(kCaptureWarning), std::string::npos)
        << "an evaluation fell back from the replay to eager evaluation";
  }

  /// Results of the host entry points of ``model`` for a configuration.
  static Results compute_host(Evaluator& model, const Configuration& config) {
    Results results;
    double energy = 0.0;
    if constexpr (Case::kSpin) {
      model.compute(energy, results.force, results.force_mag, results.virial,
                    results.atom_energy, results.atom_virial, config.coord,
                    config.spin, config.atype, config.box);
    } else {
      model.compute(energy, results.force, results.virial, results.atom_energy,
                    results.atom_virial, config.coord, config.atype,
                    config.box);
    }
    return results;
  }

  /// Agreement of the host entry points with the references of a section.
  void expect_reference(const Results& results, const std::string& section) {
    EXPECT_LT(max_difference(results.atom_energy,
                             ref.get<double>(section, "expected_e")),
              kReferenceTolerance)
        << section;
    EXPECT_LT(
        max_difference(results.force, ref.get<double>(section, "expected_f")),
        kReferenceTolerance)
        << section;
    EXPECT_LT(max_difference(results.virial,
                             ref.get<double>(section, "expected_tot_v")),
              kReferenceTolerance)
        << section;
    EXPECT_LT(max_difference(results.atom_virial,
                             ref.get<double>(section, "expected_atom_v")),
              kReferenceTolerance)
        << section;
    if constexpr (Case::kSpin) {
      EXPECT_LT(max_difference(results.force_mag,
                               ref.get<double>(section, "expected_fm")),
                kReferenceTolerance)
          << section;
    }
  }

  /// Exact agreement of an evaluation with the host entry points over the
  /// outputs it was asked for. NaN, the value of an unwritten output, never
  /// agrees.
  static void expect_equal(const Results& actual,
                           const Results& expected,
                           const std::string& what) {
    EXPECT_EQ(actual.atom_energy, expected.atom_energy)
        << what << ": atom energy";
    EXPECT_EQ(actual.force, expected.force) << what << ": force";
    EXPECT_EQ(actual.force_mag, expected.force_mag)
        << what << ": magnetic force";
    if (!actual.virial.empty()) {
      EXPECT_EQ(actual.virial, expected.virial) << what << ": virial";
    }
    if (!actual.atom_virial.empty()) {
      EXPECT_EQ(actual.atom_virial, expected.atom_virial)
          << what << ": atom virial";
    }
  }

#if defined(BUILD_PYTORCH) && BUILD_PT_EXPT
  /// The compact canonical graph that the host entry points build for a
  /// configuration, on the device. A folded graph holds the local atoms
  /// alone, as one rank builds it; an unfolded one gives every ghost atom a
  /// node of its own, as domain decomposition does, which carries the moment
  /// of its owner.
  Graph build_graph(const Configuration& config, const bool fold) const {
    std::vector<std::vector<int>> nlist;
    std::vector<double> coord_cpy;
    std::vector<int> atype_cpy;
    std::vector<int> mapping;
    _build_nlist<double>(nlist, coord_cpy, atype_cpy, mapping, config.coord,
                         config.atype, config.box,
                         static_cast<float>(dp.cutoff()));
    const std::vector<std::int64_t> mapping_64(mapping.begin(), mapping.end());
    const torch::Device device(torch::kCUDA, 0);
    deepmd::GraphTensorPack graph = deepmd::buildGraphTensors(
        nlist, coord_cpy, atype_cpy, mapping_64, natoms,
        static_cast<int>(coord_cpy.size() / 3), dp.cutoff(), device,
        /*row_centers=*/nullptr, fold);
    deepmd::canonicalizeGraphPayload(graph, graph.atype.size(0));
    graph.edge_vec = graph.edge_vec.to(torch::kFloat32);
    Graph result;
    result.tensors = deepmd::compactCanonicalGraph(graph);
    deepmd::CanonicalGraphTensorPack& compact = result.tensors;
    compact.atype = compact.atype.to(torch::kInt64).contiguous();
    compact.source = compact.source.contiguous();
    compact.edge_vec = compact.edge_vec.contiguous();
    compact.destination_row_ptr = compact.destination_row_ptr.contiguous();
    compact.source_row_ptr = compact.source_row_ptr.contiguous();
    compact.source_order = compact.source_order.contiguous();
    result.owner = mapping;
    result.owner.resize(compact.atype.size(0));
    if constexpr (Case::kSpin) {
      std::vector<float> moments(3 * result.owner.size());
      for (std::size_t node = 0; node < result.owner.size(); ++node) {
        for (int dd = 0; dd < 3; ++dd) {
          moments[3 * node + dd] =
              static_cast<float>(config.spin[3 * result.owner[node] + dd]);
        }
      }
      result.spin =
          torch::from_blob(moments.data(),
                           {static_cast<std::int64_t>(result.owner.size()), 3},
                           torch::TensorOptions().dtype(torch::kFloat32))
              .to(device);
    }
    return result;
  }

  /// Evaluate ``graph`` from here on.
  void use_graph(const Graph& graph) {
    graph_atype = graph.tensors.atype;
    graph_source = graph.tensors.source;
    graph_edge_vec = graph.tensors.edge_vec;
    graph_destination_row_ptr = graph.tensors.destination_row_ptr;
    graph_source_row_ptr = graph.tensors.source_row_ptr;
    graph_source_order = graph.tensors.source_order;
    graph_spin = graph.spin;
    nodes = static_cast<int>(graph_atype.size(0));
    edge_storage = graph_source.size(0);
  }

  /// Allocate outputs for the graph in use, at new addresses while the
  /// previous ones are held elsewhere.
  void allocate_outputs() {
    const auto f64 = torch::TensorOptions()
                         .dtype(torch::kFloat64)
                         .device(torch::Device(torch::kCUDA, 0));
    out_atom_energy = torch::empty({natoms}, f64);
    out_force = torch::empty({nodes, 3}, f64);
    if constexpr (Case::kSpin) {
      out_force_mag = torch::empty({nodes, 3}, f64);
    }
    out_virial = torch::empty({9}, f64);
    out_atom_virial = torch::empty({nodes, 9}, f64);
  }

  /// Rewrite the inputs of the graph in place with those of another
  /// configuration, whose graph has the same edges in the same order.
  void rewrite_in_place(const Configuration& config) {
    const Graph other = build_graph(config, /*fold=*/true);
    // The unsigned indices are compared as int64, which every comparison
    // kernel supports.
    const auto same = [](const torch::Tensor& left,
                         const torch::Tensor& right) {
      return torch::equal(left.to(torch::kInt64), right.to(torch::kInt64));
    };
    ASSERT_TRUE(same(other.tensors.source, graph_source));
    ASSERT_TRUE(
        same(other.tensors.destination_row_ptr, graph_destination_row_ptr));
    ASSERT_TRUE(same(other.tensors.source_row_ptr, graph_source_row_ptr));
    ASSERT_TRUE(same(other.tensors.source_order, graph_source_order));
    graph_edge_vec.copy_(other.tensors.edge_vec);
    graph_atype.copy_(other.tensors.atype);
    if constexpr (Case::kSpin) {
      graph_spin.copy_(other.spin);
    }
  }

  /// Reset every output to NaN on the current stream, so that an evaluation
  /// that leaves an output unwritten fails the comparison.
  void reset_outputs() {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    out_atom_energy.fill_(nan);
    out_force.fill_(nan);
    if constexpr (Case::kSpin) {
      out_force_mag.fill_(nan);
    }
    out_virial.fill_(nan);
    out_atom_virial.fill_(nan);
  }

  /// Enqueue an evaluation of the graph on ``stream``, or on the current
  /// stream when it is null. Without the source CSR the per-atom virial is
  /// declined.
  void run_device_entry(void* stream,
                        const bool source_csr = true,
                        const bool virial = true) {
    double* const atom_energy = out_atom_energy.data_ptr<double>();
    double* const force = out_force.data_ptr<double>();
    double* const total_virial =
        virial ? out_virial.data_ptr<double>() : nullptr;
    double* const atom_virial =
        source_csr ? out_atom_virial.data_ptr<double>() : nullptr;
    const auto* const atype = graph_atype.data_ptr<std::int64_t>();
    const auto* const source =
        static_cast<const std::uint32_t*>(graph_source.data_ptr());
    const float* const edge_vec = graph_edge_vec.data_ptr<float>();
    const auto* const destination_row_ptr =
        graph_destination_row_ptr.data_ptr<std::int64_t>();
    const std::int64_t* const source_row_ptr =
        source_csr ? graph_source_row_ptr.data_ptr<std::int64_t>() : nullptr;
    const std::uint32_t* const source_order =
        source_csr
            ? static_cast<const std::uint32_t*>(graph_source_order.data_ptr())
            : nullptr;
    if constexpr (Case::kSpin) {
      dp.compute_canonical_graph_gpu(
          atom_energy, force, out_force_mag.data_ptr<double>(), total_virial,
          atom_virial, atype, source, edge_vec, destination_row_ptr,
          source_row_ptr, source_order, graph_spin.data_ptr<float>(), natoms,
          nodes, edge_storage, stream);
    } else {
      dp.compute_canonical_graph_gpu(
          atom_energy, force, total_virial, atom_virial, atype, source,
          edge_vec, destination_row_ptr, source_row_ptr, source_order,
          /*d_coord=*/nullptr, /*world=*/nullptr, natoms, nodes, edge_storage,
          stream);
    }
  }

  /// The outputs an evaluation was asked for, copied to the host on the
  /// current stream.
  Results read_back(const bool source_csr = true,
                    const bool virial = true) const {
    const auto host_vector = [](const torch::Tensor& tensor) {
      const torch::Tensor values = tensor.cpu();
      return std::vector<double>(values.data_ptr<double>(),
                                 values.data_ptr<double>() + values.numel());
    };
    Results results;
    results.atom_energy = host_vector(out_atom_energy);
    results.force = host_vector(out_force);
    if constexpr (Case::kSpin) {
      results.force_mag = host_vector(out_force_mag);
    }
    if (virial) {
      results.virial = host_vector(out_virial);
    }
    if (source_csr) {
      results.atom_virial = host_vector(out_atom_virial);
    }
    return results;
  }

  /// Evaluate the graph on the current stream.
  Results evaluate(const bool source_csr = true, const bool virial = true) {
    reset_outputs();
    run_device_entry(nullptr, source_csr, virial);
    return read_back(source_csr, virial);
  }
#endif
};

TYPED_TEST_SUITE_P(TestInferDpa4cCanonicalPtExpt);

TYPED_TEST_P(TestInferDpa4cCanonicalPtExpt, host_entry_matches_reference) {
  this->expect_reference(this->host, "default_state");
}

#if defined(BUILD_PYTORCH) && BUILD_PT_EXPT
TYPED_TEST_P(TestInferDpa4cCanonicalPtExpt, device_entry_matches_host_entry) {
  // The first call runs eagerly, the second captures and replays, the third
  // replays.
  for (int call = 0; call < 3; ++call) {
    this->expect_equal(this->evaluate(), this->host,
                       "call " + std::to_string(call));
  }
}

TYPED_TEST_P(TestInferDpa4cCanonicalPtExpt,
             a_caller_stream_orders_the_evaluation) {
#ifdef DEEPMD_TORCH_HAS_CUDA
  // The outputs are reset and read back on the caller's stream alone, after
  // the stream is held back, while the current stream of the thread stays
  // another one. A replay enqueued anywhere but on the caller's stream
  // finishes before the reset and reads back as NaN.
  const c10::cuda::CUDAStream stream =
      c10::cuda::getStreamFromPool(/*isHighPriority=*/false, /*device=*/0);
  for (int call = 0; call < 3; ++call) {
    {
      const c10::cuda::CUDAStreamGuard guard(stream);
      at::cuda::sleep(kHoldCycles);
      this->reset_outputs();
    }
    this->run_device_entry(stream.stream());
    const c10::cuda::CUDAStreamGuard guard(stream);
    this->expect_equal(this->read_back(), this->host,
                       "call " + std::to_string(call) + " on a caller stream");
  }
#else
  GTEST_SKIP() << "Skip because deepmd-kit was built without the CUDA stream "
                  "API of PyTorch.";
#endif
}

TYPED_TEST_P(TestInferDpa4cCanonicalPtExpt, the_source_csr_is_optional) {
  if constexpr (!TypeParam::kSpin) {
    ASSERT_FALSE(this->dp.canonical_graph_requires_source_csr());
  }
  for (int call = 0; call < 3; ++call) {
    this->expect_equal(
        this->evaluate(/*source_csr=*/false), this->host,
        "call " + std::to_string(call) + " without the source CSR");
  }
}

TYPED_TEST_P(TestInferDpa4cCanonicalPtExpt, the_virial_is_optional) {
  for (int call = 0; call < 3; ++call) {
    this->expect_equal(this->evaluate(/*source_csr=*/true, /*virial=*/false),
                       this->host,
                       "call " + std::to_string(call) + " without the virial");
  }
}

TYPED_TEST_P(TestInferDpa4cCanonicalPtExpt,
             a_replay_reads_the_inputs_in_place) {
  // The system is strained, as a barostat rescales it between two neighbor
  // list rebuilds of an MD run, then the types of the atoms change, and for
  // the spin model then the moments. Each configuration is rewritten into the
  // graph in place, so that its evaluations replay the graph captured for
  // the first one.
  std::vector<Configuration> steps(1, this->system);
  for (double& value : steps[0].coord) {
    value *= 1.0 + kStrain;
  }
  for (double& value : steps[0].box) {
    value *= 1.0 + kStrain;
  }
  steps.push_back(steps[0]);
  for (int& type : steps[1].atype) {
    type = 1 - type;
  }
  if constexpr (TypeParam::kSpin) {
    steps.push_back(steps[1]);
    for (double& moment : steps[2].spin) {
      moment = -0.5 * moment;
    }
  }

  // The first call runs eagerly, the second captures and replays.
  for (int call = 0; call < 2; ++call) {
    this->expect_equal(this->evaluate(), this->host,
                       "call " + std::to_string(call));
  }
  Results previous = this->host;
  for (std::size_t step = 0; step < steps.size(); ++step) {
    const Results expected = this->compute_host(this->dp, steps[step]);
    // Each change must alter the results, or a replay that read the inputs
    // of the capture would still match.
    const double change =
        std::max({max_difference(expected.atom_energy, previous.atom_energy),
                  max_difference(expected.force, previous.force),
                  max_difference(expected.force_mag, previous.force_mag)});
    ASSERT_GT(change, 1e-6) << "step " << step;
    ASSERT_NO_FATAL_FAILURE(this->rewrite_in_place(steps[step]));
    for (int call = 0; call < 2; ++call) {
      this->expect_equal(this->evaluate(), expected,
                         "replay " + std::to_string(call) + " after step " +
                             std::to_string(step));
    }
    previous = expected;
  }
}

TYPED_TEST_P(TestInferDpa4cCanonicalPtExpt, a_new_key_releases_the_replay) {
  // Outputs at other addresses form another key: the graph captured for the
  // first outputs is released, and the evaluations under the new key run
  // eagerly, are captured and are replayed in turn, as they are again once
  // the first outputs return.
  for (int call = 0; call < 2; ++call) {
    this->expect_equal(this->evaluate(), this->host,
                       "call " + std::to_string(call));
  }
  const std::vector<torch::Tensor> first = {
      this->out_atom_energy, this->out_force, this->out_force_mag,
      this->out_virial, this->out_atom_virial};
  this->allocate_outputs();
  ASSERT_NE(this->out_force.data_ptr(), first[1].data_ptr());
  for (int call = 0; call < 3; ++call) {
    this->expect_equal(this->evaluate(), this->host,
                       "call " + std::to_string(call) + " on other outputs");
  }
  this->out_atom_energy = first[0];
  this->out_force = first[1];
  this->out_force_mag = first[2];
  this->out_virial = first[3];
  this->out_atom_virial = first[4];
  for (int call = 0; call < 3; ++call) {
    this->expect_equal(
        this->evaluate(), this->host,
        "call " + std::to_string(call) + " back on the first outputs");
  }
}

TYPED_TEST_P(TestInferDpa4cCanonicalPtExpt,
             a_graph_with_halo_nodes_matches_the_host_entry) {
  // The unfolded graph gives every ghost atom a node, whose forces the caller
  // adds to those of its owner, as the reverse communication of domain
  // decomposition does; the extents of such a graph change with every
  // neighbor rebuild, so it is evaluated eagerly. Its rows are those of the
  // folded graph, so its energies, folded forces and virial agree with the
  // host entry points up to the summation order of the run, which the node
  // count of the graph configures.
  const auto graph = this->build_graph(this->system, /*fold=*/false);
  this->use_graph(graph);
  ASSERT_GT(this->nodes, this->natoms);
  this->allocate_outputs();
  const Results& host = this->host;
  for (int call = 0; call < 2; ++call) {
    const Results halo = this->evaluate();
    const std::string what = "call " + std::to_string(call);
    EXPECT_LT(max_difference(halo.atom_energy, host.atom_energy),
              kReferenceTolerance * largest(host.atom_energy))
        << what << ": atom energy";
    EXPECT_LT(
        max_difference(fold_onto_owners(halo.force, graph.owner, this->natoms),
                       host.force),
        kReferenceTolerance * largest(host.force))
        << what << ": force";
    EXPECT_LT(max_difference(halo.virial, host.virial),
              kReferenceTolerance * largest(host.virial))
        << what << ": virial";
    if constexpr (TypeParam::kSpin) {
      EXPECT_LT(max_difference(
                    fold_onto_owners(halo.force_mag, graph.owner, this->natoms),
                    host.force_mag),
                kReferenceTolerance * largest(host.force_mag))
          << what << ": magnetic force";
    }
  }
}

TYPED_TEST_P(TestInferDpa4cCanonicalPtExpt, a_capture_elsewhere_stays_valid) {
#ifdef DEEPMD_TORCH_HAS_CUDA
  // Another thread holds a capture of its own open while this thread replays
  // the graph of one rank and evaluates a graph with halo nodes eagerly, both
  // without a caller's stream. Neither waits for the whole device, which is
  // invalid while a capture is under way and would invalidate that capture.
  for (int call = 0; call < 2; ++call) {
    this->expect_equal(this->evaluate(), this->host,
                       "call " + std::to_string(call));
  }
  const auto halo = this->build_graph(this->system, /*fold=*/false);
  torch::cuda::synchronize();
  CaptureElsewhere capture;
  this->expect_equal(this->evaluate(), this->host,
                     "a replay while another thread captures");
  this->use_graph(halo);
  this->allocate_outputs();
  const Results eager = this->evaluate();
  EXPECT_EQ(capture.close(), cudaSuccess)
      << "an evaluation invalidated the capture of another thread";
  EXPECT_LT(
      max_difference(fold_onto_owners(eager.force, halo.owner, this->natoms),
                     this->host.force),
      kReferenceTolerance * largest(this->host.force))
      << "an eager evaluation while another thread captures";
#else
  GTEST_SKIP() << "Skip because deepmd-kit was built without the CUDA stream "
                  "API of PyTorch.";
#endif
}

TYPED_TEST_P(TestInferDpa4cCanonicalPtExpt, a_charge_state_reaches_the_replay) {
  // A fresh instance evaluated through the host entry points is the
  // reference under the other state.
  typename TypeParam::Evaluator other;
  other.init(TypeParam::kModelPath);
  other.set_charge_spin(kOtherChargeSpin);
  const Results other_host = this->compute_host(other, this->system);
  ASSERT_GT(max_difference(other_host.atom_energy, this->host.atom_energy),
            1e-6)
      << "the two charge states must differ, or a replay that kept the "
         "constants of the first state would still match";
  this->expect_reference(other_host, "other_state");

  for (int call = 0; call < 3; ++call) {
    this->expect_equal(this->evaluate(), this->host,
                       "call " + std::to_string(call) + " before the state");
  }
  this->dp.set_charge_spin(kOtherChargeSpin);
  for (int call = 0; call < 3; ++call) {
    this->expect_equal(this->evaluate(), other_host,
                       "call " + std::to_string(call) + " after the state");
  }
}

REGISTER_TYPED_TEST_SUITE_P(TestInferDpa4cCanonicalPtExpt,
                            host_entry_matches_reference,
                            device_entry_matches_host_entry,
                            a_caller_stream_orders_the_evaluation,
                            the_source_csr_is_optional,
                            the_virial_is_optional,
                            a_replay_reads_the_inputs_in_place,
                            a_new_key_releases_the_replay,
                            a_graph_with_halo_nodes_matches_the_host_entry,
                            a_capture_elsewhere_stays_valid,
                            a_charge_state_reaches_the_replay);
#else
REGISTER_TYPED_TEST_SUITE_P(TestInferDpa4cCanonicalPtExpt,
                            host_entry_matches_reference);
#endif

}  // namespace dpa4c_canonical
}  // namespace deepmd_test
