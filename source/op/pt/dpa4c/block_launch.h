// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Host-side launch interface of the further message-passing blocks of the
// compressed DPA4C descriptor.
//
// Block k reads per-node features of the edge sources -- keys, an encoded
// state and the neighbor term of an edge hidden layer -- that the node maps
// form from the moments and the invariants of block k - 1, and corrects the
// moments of the first block over the edges inside its own cutoff (see
// deepmd/pt_expt/kernels/dpa4c/attention.py for the equations and the layout
// of the record of a block). Its stages are:
//
//   maps          node maps of block k from block k - 1: the source features,
//                 the null-mass logits of every centre and the divisor columns
//                 of the descriptor;
//   forward       the edge scan of block k over destination rows, which writes
//                 the moments of block k in the state layout of the first
//                 block together with the transport of every row, from which
//                 the readout of the first block forms the invariants of
//                 block k;
//   cotangents    the node part of the backward of block k: from the moment
//                 gradient of block k, the cotangents of every destination
//                 that its edges read and the moment gradient carried back to
//                 the first block;
//   backward      the edge scan of the backward over source rows: a canonical
//                 graph holds both directions of every bond, so the entries of
//                 the row of a source are the edges it emits, and the scan
//                 accumulates the cotangent of the source features in
//                 registers instead of scattering it;
//   map_backward  the backward of the node maps, into the descriptor
//                 cotangent of block k - 1 and the moment gradient of its keys.
//
// The node set may hold halo rows, which carry no edges; the caller refreshes
// the source features and the destination cotangents of every halo row from
// its owner between the stages (border_exchange.h).

#pragma once

#include <cuda_runtime.h>

#include "graph_compress_launch.h"

namespace deepmd_dpa4c {

/// Attention heads and hidden width of the state encoder of a further block,
/// widths the compiled operator specializes on.
constexpr int kBlockHeads = 2;
constexpr int kBlockStateHidden = 16;
/// Floats of the header of a block record: the cutoff of the block, the knees
/// of the invariant, key and amplitude saturations, the logit cap and the null
/// floor of the softmax, and two padding words.
constexpr int kBlockHeader = 8;

constexpr int aligned_floats(int count) { return (count + 3) / 4 * 4; }

/// Widths of a further block and the offsets of the sections of its record,
/// mirroring ``block_layout`` of deepmd/pt_expt/kernels/dpa4c/attention.py.
template <int Channels, int Lmax, bool HasSpin>
struct BlockProfile {
  using P = Profile<Channels, Lmax, HasSpin>;
  static constexpr int Heads = kBlockHeads;
  static constexpr int StateHidden = kBlockStateHidden;
  /// Non-scalar harmonic components, those of the keys of a node.
  static constexpr int Harmonics = (Lmax + 1) * (Lmax + 1) - 1;
  static constexpr int KeyWidth = Harmonics * Heads;
  /// Fields of an edge, one per non-scalar degree and head.
  static constexpr int Fields = Lmax * Heads;
  /// Contraction depth of both edge maps: the fields, then a channel vector.
  static constexpr int Operand = Fields + Channels;

  /// Source features of a node: the keys, harmonic-major with the heads
  /// innermost, the encoded state and the neighbor term of the hidden layer.
  /// The keys are padded to a 16-byte multiple, so that the channels of the
  /// state and of the hidden term start on 16-byte boundaries and load as
  /// vectors.
  static constexpr int SourceState = aligned_floats(KeyWidth);
  static constexpr int SourceHidden = SourceState + Channels;
  static constexpr int SourceWidth = SourceHidden + Channels;
  /// Moments of degrees zero through Lmax, which a further block corrects.
  static constexpr int GeometricWidth = P::SpinOffset;
  /// Cotangents of a destination that its edges read: the moment gradient of
  /// the geometric degrees, then that of the transported keys and of the
  /// softmax denominators. A row is padded to a 16-byte multiple, so that the
  /// channels of every component of degrees zero through two, which start on
  /// 16-byte boundaries, load as vectors.
  static constexpr int CotangentTransport = GeometricWidth;
  static constexpr int CotangentMass = GeometricWidth + KeyWidth;
  static constexpr int CotangentWidth =
      aligned_floats(GeometricWidth + KeyWidth + Heads);
  /// Transport of a row: the transported keys over their denominators, then
  /// the denominators.
  static constexpr int TransportWidth = KeyWidth + Heads;
  /// Columns of the input layer of the node maps.
  static constexpr int NodeHidden = 2 * StateHidden;
  static constexpr int NodeNull = NodeHidden + Channels;
  static constexpr int NodeWidth = aligned_floats(NodeNull + Heads);

  static constexpr int width_of(int degree) {
    return degree == 0 ? P::C0 : degree == 1 ? P::C1 : degree == 2 ? P::C2 : 1;
  }
  static constexpr int moment_offset(int degree) {
    return degree == 0   ? P::ScalarOffset
           : degree == 1 ? P::VectorOffset
           : degree == 2 ? P::TensorOffset
           : degree == 3 ? P::HighOffset
                         : P::HighOffset + 7;
  }
  /// First harmonic of a degree from one in the key layout.
  static constexpr int harmonic_offset(int degree) {
    return degree * degree - 1;
  }

  static constexpr int key_offset(int degree) {
    int offset = kBlockHeader;
    for (int l = 1; l < degree; ++l) {
      offset += aligned_floats(width_of(l) * Heads);
    }
    return offset;
  }
  static constexpr int NodeWeight = key_offset(Lmax + 1);
  static constexpr int NodeBias = NodeWeight + P::OutputWidth * NodeWidth;
  static constexpr int StateOutput = NodeBias + NodeWidth;
  static constexpr int EdgeField = StateOutput + StateHidden * Channels;
  static constexpr int EdgeAmplitude = EdgeField + Fields * Channels;
  static constexpr int EdgeGain = EdgeAmplitude + Channels * Channels;
  static constexpr int EdgeOutput = EdgeGain + Fields * Channels;
  static constexpr int EdgeAmplitudeT = EdgeOutput + Channels * Channels;
  static constexpr int EdgeOutputT = EdgeAmplitudeT + Channels * Channels;
  static constexpr int EdgeTransport = EdgeOutputT + Channels * Channels;
  static constexpr int output_offset(int degree) {
    int offset = EdgeTransport + Channels * Heads;
    for (int l = 1; l < degree; ++l) {
      offset += aligned_floats(Heads * width_of(l));
    }
    return offset;
  }
  static constexpr int Width = output_offset(Lmax + 1);
};

/// Arguments of one stage of a further block over one run of nodes.
///
/// ``edges`` carries the graph and the radial representation of the first
/// block exactly as the descriptor launches read them, with its node range
/// set to the run. Buffers of the whole node set are indexed by absolute node
/// id, like ``edges.atype``, because the edges of a run read the rows of
/// other runs; buffers of the run are offset by the caller, like
/// ``edges.descriptor``.
struct BlockArguments {
  Arguments edges;
  /// Record of the block, BlockProfile layout.
  const float* record = nullptr;
  /// Destination rows of a block of the edge scans, which their launch
  /// chooses from the node count of the whole system.
  int block_rows = 0;
  /// Rows of the run that the edge scans visit, ascending, and their number,
  /// both in device memory; null for every row of the run. The grid of a
  /// launch covers every row of the run, and its blocks past the visited rows
  /// retire.
  const int* scan_rows = nullptr;
  const int* scan_row_count = nullptr;

  // === Node set ===
  /// Moments and norms of the first block, (N, StateWidth).
  const float* first_state = nullptr;
  /// Moments of block k - 1, (N, StateWidth); the first state for k = 2.
  const float* previous_state = nullptr;
  /// Source features, (N, SourceWidth).
  float* sources = nullptr;
  /// Null-mass logits of every centre, (N, Heads).
  float* centers = nullptr;
  /// Calibrated divisor columns of the descriptor of the first block, which
  /// the descriptor of every further block carries, (N, 2).
  float* divisors = nullptr;
  /// Destination cotangents, (N, CotangentWidth).
  float* cotangents = nullptr;
  /// Moment gradient of the first block carried back from the further
  /// blocks, (N, StateWidth), in the layout of the node backward.
  float* carried = nullptr;
  /// Whether the cotangent stage adds to ``carried`` rather than writing it.
  bool accumulate_carried = false;

  // === Run ===
  /// Invariants of block k - 1, (run, OutputWidth), read by the node maps.
  const float* descriptor = nullptr;
  /// Moments and norms of block k, (run, StateWidth).
  float* state_out = nullptr;
  /// Transport of every row, (run, TransportWidth).
  float* transport = nullptr;
  /// Node-backward output at the moments of block k, (run, StateWidth).
  const float* moment_gradient = nullptr;
  /// Cotangent of the null-mass logits, (run, Heads).
  float* center_gradient = nullptr;
  /// Cotangent of the source features of every row, (run, SourceWidth).
  float* source_gradient = nullptr;
  /// Virial of the edges every row emits, which the backward adds, (run, 9).
  float* source_virial = nullptr;
  /// Gate totals of every node of the node set under a source gate
  /// (Arguments::source_gate), onto which the backward adds, per row, the sum
  /// over the edges the row emits of the envelope cotangent times the
  /// envelope; null without a source gate.
  float* gate_sums = nullptr;
  /// Cotangent of the invariants of block k - 1, (run, OutputWidth); it may
  /// take the place of ``descriptor``.
  float* descriptor_gradient = nullptr;
  /// Moment gradient of the keys of block k, in the node-backward layout of
  /// block k - 1, (run, StateWidth).
  float* keys_gradient = nullptr;
  /// Cotangent of the node input layer of every row, which the backward of the
  /// node maps passes between its two kernels, (run, NodeWidth).
  float* map_cotangent = nullptr;
};

/// Stages of a further block; see the head of this file.
enum class BlockStage : int {
  Maps = 0,
  Forward = 1,
  Cotangents = 2,
  Backward = 3,
  MapBackward = 4,
};

/// Stages of the first block that the composition of several blocks launches
/// one at a time, where a single block launches them together.
enum class FirstStage : int {
  /// The invariant readout of ``state_out`` into ``descriptor``.
  Readout = 0,
  /// The readout VJP of ``descriptor_gradient`` at ``state`` into
  /// ``moment_gradient``.
  NodeBackward = 1,
  /// The edge scans of the backward from ``moment_gradient``.
  EdgeBackward = 2,
};

/// One stage of a further block for one scalar width and angular degree. Every
/// width of DPA4C_FOR_EACH_CHANNEL and every degree from 2 to 4 is
/// instantiated in a generated block shard (``c<width>_l<degrees>_block.cu``,
/// configured from dpa4c/shard.cu.in).
template <int Channels, int Lmax>
void launch_block(BlockStage stage,
                  const BlockArguments& arguments,
                  cudaStream_t stream);

/// One stage of the first block, instantiated beside launch_forward.
template <int Channels, int Lmax>
void launch_first_stage(FirstStage stage,
                        const Arguments& arguments,
                        cudaStream_t stream);

}  // namespace deepmd_dpa4c
