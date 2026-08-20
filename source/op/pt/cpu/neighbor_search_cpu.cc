// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Threaded cell-list neighbor search for the host graph builders.
//
// The Python inference path rebuilds its neighbor graph on every call, so the
// search is the dominant term of an ASE-style evaluation: a single-threaded
// cell list over an 8000-atom cell spends about 90 ms finding 1.26 million
// pairs, against 4 to 18 ms for the model itself. The work is embarrassingly
// parallel over destination atoms and its arithmetic is three subtractions and
// a dot product per candidate, so the only structural requirements are that
// the candidate set stay small and that the output be written once.
//
// Two operators share the search. ``neighbor_search`` returns the pair list
// with the integer lattice image of each pair, which a differentiable caller
// needs because it recomputes the displacement from the coordinates it holds.
// ``neighbor_graph`` returns the whole destination-major payload -- endpoints,
// displacements, mask and both compressed-sparse-row views -- for a deployment
// caller that feeds a frozen artifact and takes its forces from the model's
// analytical backward. The second form exists because the search already
// computes every displacement it tests: handing them back turns a chain of
// gathers, a sort and a reordering of every edge field into nothing.
//
// Both emit pairs grouped by destination, which is the order the
// compressed-sparse-row views want and which makes the destination
// permutation the identity.

#include <ATen/Parallel.h>
#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "group.h"

namespace deepmd {
namespace {

/// Squared displacement below which a pair is treated as a self-image.
constexpr double kSelfPairTolerance = 1e-10;

/// Lattice geometry needed to bin atoms and to enumerate candidate images.
struct CellGrid {
  /// Cell divisions along each lattice direction.
  std::int64_t divisions[3] = {1, 1, 1};
  /// Image range searched along each lattice direction.
  std::int64_t reach[3] = {0, 0, 0};
  /// Row-major inverse of the lattice matrix, mapping Cartesian to fractional.
  double inverse[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  /// Row-major lattice matrix, rows being the lattice vectors.
  double lattice[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  bool periodic = false;

  std::int64_t count() const {
    return divisions[0] * divisions[1] * divisions[2];
  }
};

/// Invert a 3x3 row-major matrix; throws when the lattice is degenerate.
void invert3(const double* matrix, double* inverse) {
  const double a = matrix[0], b = matrix[1], c = matrix[2];
  const double d = matrix[3], e = matrix[4], f = matrix[5];
  const double g = matrix[6], h = matrix[7], i = matrix[8];
  const double determinant =
      a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  TORCH_CHECK(std::abs(determinant) > 0.0,
              "neighbor_search: the lattice matrix is singular");
  const double scale = 1.0 / determinant;
  inverse[0] = (e * i - f * h) * scale;
  inverse[1] = (c * h - b * i) * scale;
  inverse[2] = (b * f - c * e) * scale;
  inverse[3] = (f * g - d * i) * scale;
  inverse[4] = (a * i - c * g) * scale;
  inverse[5] = (c * d - a * f) * scale;
  inverse[6] = (d * h - e * g) * scale;
  inverse[7] = (b * g - a * h) * scale;
  inverse[8] = (a * e - b * d) * scale;
}

/**
 * @brief Choose the cell divisions and the image reach for one lattice.
 *
 * A direction is divided so that each slab is at least the cutoff wide, which
 * bounds the candidate set to the immediately neighbouring cells. When the
 * lattice is thinner than the cutoff the division saturates at one and the
 * reach grows instead, so a cell smaller than the cutoff is searched over as
 * many images as it takes.
 */
CellGrid make_grid(const double* lattice,
                   const bool periodic,
                   const double rcut) {
  CellGrid grid;
  grid.periodic = periodic;
  if (!periodic) {
    return grid;
  }
  std::copy(lattice, lattice + 9, grid.lattice);
  invert3(lattice, grid.inverse);
  // The perpendicular width along a direction is the volume divided by the
  // area of the opposite face, which the inverse already encodes: the norm of
  // its corresponding column is one over that width.
  for (int axis = 0; axis < 3; ++axis) {
    const double gx = grid.inverse[axis];
    const double gy = grid.inverse[3 + axis];
    const double gz = grid.inverse[6 + axis];
    const double width = 1.0 / std::sqrt(gx * gx + gy * gy + gz * gz);
    const auto divisions = static_cast<std::int64_t>(std::floor(width / rcut));
    grid.divisions[axis] = std::max<std::int64_t>(divisions, 1);
    const double slab = width / static_cast<double>(grid.divisions[axis]);
    grid.reach[axis] = std::max<std::int64_t>(
        static_cast<std::int64_t>(std::ceil(rcut / slab - 1e-12)), 1);
  }
  return grid;
}

/// Wrapped fractional coordinates and the integer image each atom came from.
struct Fractional {
  std::vector<double> position;
  std::vector<std::int32_t> image;
};

/// Map Cartesian coordinates into the primitive cell.
template <typename ScalarType>
Fractional to_fractional(const ScalarType* coord,
                         const std::int64_t atom_count,
                         const CellGrid& grid) {
  Fractional fractional;
  fractional.position.resize(static_cast<size_t>(atom_count) * 3);
  fractional.image.assign(static_cast<size_t>(atom_count) * 3, 0);
  at::parallel_for(
      0, atom_count, 1024, [&](std::int64_t begin, std::int64_t end) {
        for (std::int64_t atom = begin; atom < end; ++atom) {
          const double x = static_cast<double>(coord[atom * 3]);
          const double y = static_cast<double>(coord[atom * 3 + 1]);
          const double z = static_cast<double>(coord[atom * 3 + 2]);
          if (!grid.periodic) {
            fractional.position[atom * 3] = x;
            fractional.position[atom * 3 + 1] = y;
            fractional.position[atom * 3 + 2] = z;
            continue;
          }
          for (int axis = 0; axis < 3; ++axis) {
            const double raw = x * grid.inverse[axis] +
                               y * grid.inverse[3 + axis] +
                               z * grid.inverse[6 + axis];
            const double cell = std::floor(raw);
            fractional.position[atom * 3 + axis] = raw - cell;
            fractional.image[atom * 3 + axis] =
                -static_cast<std::int32_t>(cell);
          }
        }
      });
  return fractional;
}

/// Atoms bucketed by cell, in compressed-sparse-row form.
struct Buckets {
  std::vector<std::int64_t> start;
  std::vector<std::int32_t> atom;
};

/// Bucket atoms by cell index with a counting sort.
Buckets bucket_atoms(const Fractional& fractional,
                     const std::int64_t atom_count,
                     const CellGrid& grid) {
  const std::int64_t cell_count = grid.count();
  Buckets buckets;
  buckets.start.assign(cell_count + 1, 0);
  std::vector<std::int64_t> cell_of_atom(atom_count);
  for (std::int64_t atom = 0; atom < atom_count; ++atom) {
    std::int64_t index = 0;
    for (int axis = 0; axis < 3; ++axis) {
      auto bin =
          static_cast<std::int64_t>(fractional.position[atom * 3 + axis] *
                                    static_cast<double>(grid.divisions[axis]));
      bin = std::min(std::max<std::int64_t>(bin, 0), grid.divisions[axis] - 1);
      index = index * grid.divisions[axis] + bin;
    }
    cell_of_atom[atom] = index;
    ++buckets.start[index + 1];
  }
  for (std::int64_t cell = 0; cell < cell_count; ++cell) {
    buckets.start[cell + 1] += buckets.start[cell];
  }
  buckets.atom.resize(atom_count);
  std::vector<std::int64_t> cursor(buckets.start.begin(),
                                   buckets.start.end() - 1);
  for (std::int64_t atom = 0; atom < atom_count; ++atom) {
    buckets.atom[cursor[cell_of_atom[atom]]++] =
        static_cast<std::int32_t>(atom);
  }
  return buckets;
}

/// Enumerated candidate cell, with the lattice image its wrap implies.
struct CandidateCell {
  std::int64_t index;
  std::int32_t image[3];
};

/// Prepared search state, shared by the counting and the emitting pass.
struct PreparedSearch {
  CellGrid grid;
  Fractional fractional;
  Buckets buckets;
  double rcut_squared = 0.0;
  std::int64_t atom_count = 0;
  /// Destination offsets, one per atom plus the total.
  std::vector<std::int64_t> row_ptr;
};

/**
 * @brief Visit every neighbour of one destination atom within the cutoff.
 *
 * The visitor receives the neighbour index, the integer image relating the two
 * original coordinates, and the Cartesian displacement. Displacements are
 * formed in fractional space and mapped back through the lattice, which keeps
 * the periodic wrap exact for a triclinic cell.
 */
template <typename Visitor>
void visit_neighbors(const std::int64_t center,
                     const PreparedSearch& prepared,
                     std::vector<CandidateCell>& candidates,
                     Visitor&& visitor) {
  const CellGrid& grid = prepared.grid;
  const Fractional& fractional = prepared.fractional;
  const double* center_position = &fractional.position[center * 3];
  const std::int32_t* center_image = &fractional.image[center * 3];

  candidates.clear();
  if (!grid.periodic) {
    candidates.push_back({0, {0, 0, 0}});
  } else {
    std::int64_t home[3];
    for (int axis = 0; axis < 3; ++axis) {
      const auto bin = static_cast<std::int64_t>(
          center_position[axis] * static_cast<double>(grid.divisions[axis]));
      home[axis] =
          std::min(std::max<std::int64_t>(bin, 0), grid.divisions[axis] - 1);
    }
    for (std::int64_t da = -grid.reach[0]; da <= grid.reach[0]; ++da) {
      for (std::int64_t db = -grid.reach[1]; db <= grid.reach[1]; ++db) {
        for (std::int64_t dc = -grid.reach[2]; dc <= grid.reach[2]; ++dc) {
          const std::int64_t offset[3] = {da, db, dc};
          CandidateCell candidate{0, {0, 0, 0}};
          std::int64_t index = 0;
          for (int axis = 0; axis < 3; ++axis) {
            const std::int64_t raw = home[axis] + offset[axis];
            const std::int64_t divisions = grid.divisions[axis];
            // Floor division carries the wrap into the lattice image.
            std::int64_t wrap = raw / divisions;
            std::int64_t bin = raw % divisions;
            if (bin < 0) {
              bin += divisions;
              --wrap;
            }
            candidate.image[axis] = static_cast<std::int32_t>(wrap);
            index = index * divisions + bin;
          }
          candidate.index = index;
          candidates.push_back(candidate);
        }
      }
    }
  }

  for (const CandidateCell& candidate : candidates) {
    const std::int64_t begin = prepared.buckets.start[candidate.index];
    const std::int64_t end = prepared.buckets.start[candidate.index + 1];
    for (std::int64_t slot = begin; slot < end; ++slot) {
      const std::int64_t neighbor = prepared.buckets.atom[slot];
      double delta[3];
      for (int axis = 0; axis < 3; ++axis) {
        delta[axis] = fractional.position[neighbor * 3 + axis] -
                      center_position[axis] +
                      static_cast<double>(candidate.image[axis]);
      }
      double displacement[3];
      if (grid.periodic) {
        for (int axis = 0; axis < 3; ++axis) {
          displacement[axis] = delta[0] * grid.lattice[axis] +
                               delta[1] * grid.lattice[3 + axis] +
                               delta[2] * grid.lattice[6 + axis];
        }
      } else {
        std::copy(delta, delta + 3, displacement);
      }
      const double distance_squared = displacement[0] * displacement[0] +
                                      displacement[1] * displacement[1] +
                                      displacement[2] * displacement[2];
      if (distance_squared <= kSelfPairTolerance ||
          distance_squared > prepared.rcut_squared) {
        continue;
      }
      // The image relating the ORIGINAL coordinates absorbs the wrap that
      // brought each atom into the primitive cell.
      const std::int32_t image[3] = {
          fractional.image[neighbor * 3] + candidate.image[0] - center_image[0],
          fractional.image[neighbor * 3 + 1] + candidate.image[1] -
              center_image[1],
          fractional.image[neighbor * 3 + 2] + candidate.image[2] -
              center_image[2]};
      visitor(neighbor, image, displacement);
    }
  }
}

/// Bin the atoms and count each destination's neighbours.
template <typename ScalarType>
PreparedSearch prepare_search(const torch::Tensor& coord,
                              const torch::Tensor& cell,
                              const bool periodic,
                              const double rcut) {
  double lattice[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  if (periodic) {
    const auto host_cell = cell.to(torch::kFloat64).contiguous();
    std::copy(host_cell.const_data_ptr<double>(),
              host_cell.const_data_ptr<double>() + 9, lattice);
  }
  PreparedSearch prepared;
  prepared.atom_count = coord.size(0);
  prepared.rcut_squared = rcut * rcut;
  prepared.grid = make_grid(lattice, periodic, rcut);
  prepared.fractional = to_fractional(coord.const_data_ptr<ScalarType>(),
                                      prepared.atom_count, prepared.grid);
  prepared.buckets =
      bucket_atoms(prepared.fractional, prepared.atom_count, prepared.grid);

  prepared.row_ptr.assign(prepared.atom_count + 1, 0);
  std::int64_t* row_ptr = prepared.row_ptr.data();
  at::parallel_for(0, prepared.atom_count, 1,
                   [&](std::int64_t begin, std::int64_t end) {
                     std::vector<CandidateCell> candidates;
                     for (std::int64_t center = begin; center < end; ++center) {
                       std::int64_t found = 0;
                       visit_neighbors(center, prepared, candidates,
                                       [&](std::int64_t, const std::int32_t*,
                                           const double*) { ++found; });
                       row_ptr[center + 1] = found;
                     }
                   });
  for (std::int64_t center = 0; center < prepared.atom_count; ++center) {
    row_ptr[center + 1] += row_ptr[center];
  }
  return prepared;
}

/// Walk the candidates again, handing each survivor its output slot.
template <typename Emitter>
void emit_pairs(const PreparedSearch& prepared, Emitter&& emitter) {
  at::parallel_for(
      0, prepared.atom_count, 1, [&](std::int64_t begin, std::int64_t end) {
        std::vector<CandidateCell> candidates;
        for (std::int64_t center = begin; center < end; ++center) {
          std::int64_t cursor = prepared.row_ptr[center];
          visit_neighbors(center, prepared, candidates,
                          [&](std::int64_t neighbor, const std::int32_t* image,
                              const double* displacement) {
                            emitter(cursor, center, neighbor, image,
                                    displacement);
                            ++cursor;
                          });
        }
      });
}

}  // namespace

/**
 * @brief Find every pair within a cutoff, grouped by destination atom.
 *
 * @param coord Coordinates with shape ``(N, 3)``.
 * @param cell Lattice matrix with shape ``(3, 3)``, rows being the lattice
 *   vectors. Ignored when the system is not periodic.
 * @param periodic Whether the lattice wraps.
 * @param rcut Cutoff radius.
 *
 * @return ``(destination, source, image)``: the center of each pair, its
 *   neighbour, and the integer lattice image such that
 *   ``coord[source] + image @ cell - coord[destination]`` is the displacement.
 *   Pairs are grouped by destination and destinations appear in order.
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> neighbor_search(
    torch::Tensor coord, torch::Tensor cell, bool periodic, double rcut) {
  TORCH_CHECK(coord.device().is_cpu(), "neighbor_search: coord must be on CPU");
  TORCH_CHECK(coord.dim() == 2 && coord.size(1) == 3,
              "neighbor_search: coord must have shape (N, 3)");
  TORCH_CHECK(rcut > 0.0, "neighbor_search: rcut must be positive");
  if (periodic) {
    TORCH_CHECK(cell.dim() == 2 && cell.size(0) == 3 && cell.size(1) == 3,
                "neighbor_search: cell must have shape (3, 3)");
  }
  const auto contiguous = coord.contiguous();
  const auto index_options = torch::TensorOptions().dtype(torch::kInt64);
  if (contiguous.size(0) == 0) {
    return {torch::empty({0}, index_options), torch::empty({0}, index_options),
            torch::empty({0, 3}, index_options)};
  }
  TORCH_CHECK(contiguous.scalar_type() == torch::kFloat64 ||
                  contiguous.scalar_type() == torch::kFloat32,
              "neighbor_search: coord must be float32 or float64");
  const PreparedSearch prepared =
      contiguous.scalar_type() == torch::kFloat64
          ? prepare_search<double>(contiguous, cell, periodic, rcut)
          : prepare_search<float>(contiguous, cell, periodic, rcut);
  const std::int64_t edge_count = prepared.row_ptr[prepared.atom_count];

  torch::Tensor destination = torch::empty({edge_count}, index_options);
  torch::Tensor source = torch::empty({edge_count}, index_options);
  torch::Tensor image = torch::empty({edge_count, 3}, index_options);
  auto* destination_data = destination.data_ptr<std::int64_t>();
  auto* source_data = source.data_ptr<std::int64_t>();
  auto* image_data = image.data_ptr<std::int64_t>();
  emit_pairs(prepared,
             [&](std::int64_t slot, std::int64_t center, std::int64_t neighbor,
                 const std::int32_t* shift, const double*) {
               destination_data[slot] = center;
               source_data[slot] = neighbor;
               image_data[slot * 3] = shift[0];
               image_data[slot * 3 + 1] = shift[1];
               image_data[slot * 3 + 2] = shift[2];
             });
  return {destination, source, image};
}

/**
 * @brief Build the whole destination-major neighbor graph in one pass.
 *
 * The displacements come from the search rather than from a second gather
 * through the coordinates, and the destination grouping is structural, so the
 * destination permutation is the identity and its row pointers come from the
 * search's own counts. Only the source permutation costs a pass of its own.
 *
 * Two masked edges terminate the payload so that an exported graph never
 * observes an empty edge axis.
 *
 * @param coord Coordinates with shape ``(N, 3)``.
 * @param cell Lattice matrix with shape ``(3, 3)``.
 * @param periodic Whether the lattice wraps.
 * @param rcut Cutoff radius.
 * @param edge_dtype Scalar type of the returned displacements.
 *
 * @return ``(edge_index, edge_vec, edge_mask, destination_row_ptr,
 *   source_order, source_row_ptr)``.
 */
std::tuple<torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor>
neighbor_graph(torch::Tensor coord,
               torch::Tensor cell,
               bool periodic,
               double rcut,
               at::ScalarType edge_dtype) {
  TORCH_CHECK(coord.device().is_cpu(), "neighbor_graph: coord must be on CPU");
  TORCH_CHECK(coord.dim() == 2 && coord.size(1) == 3,
              "neighbor_graph: coord must have shape (N, 3)");
  TORCH_CHECK(rcut > 0.0, "neighbor_graph: rcut must be positive");
  TORCH_CHECK(edge_dtype == torch::kFloat32 || edge_dtype == torch::kFloat64,
              "neighbor_graph: edge_dtype must be float32 or float64");
  if (periodic) {
    TORCH_CHECK(cell.dim() == 2 && cell.size(0) == 3 && cell.size(1) == 3,
                "neighbor_graph: cell must have shape (3, 3)");
  }
  const auto contiguous = coord.contiguous();
  TORCH_CHECK(contiguous.scalar_type() == torch::kFloat64 ||
                  contiguous.scalar_type() == torch::kFloat32,
              "neighbor_graph: coord must be float32 or float64");
  const std::int64_t node_count = contiguous.size(0);
  const PreparedSearch prepared =
      contiguous.scalar_type() == torch::kFloat64
          ? prepare_search<double>(contiguous, cell, periodic, rcut)
          : prepare_search<float>(contiguous, cell, periodic, rcut);
  const std::int64_t real_edges = prepared.row_ptr[node_count];
  const std::int64_t edge_count = real_edges + 2;

  const auto index_options = torch::TensorOptions().dtype(torch::kInt64);
  torch::Tensor edge_index = torch::zeros({2, edge_count}, index_options);
  torch::Tensor edge_vec =
      torch::zeros({edge_count, 3}, torch::TensorOptions().dtype(edge_dtype));
  torch::Tensor edge_mask =
      torch::zeros({edge_count}, torch::TensorOptions().dtype(torch::kBool));
  auto* source_data = edge_index.data_ptr<std::int64_t>();
  auto* destination_data = source_data + edge_count;
  auto* mask_data = edge_mask.data_ptr<bool>();
  float* vec_f32 =
      edge_dtype == torch::kFloat32 ? edge_vec.data_ptr<float>() : nullptr;
  double* vec_f64 =
      edge_dtype == torch::kFloat32 ? nullptr : edge_vec.data_ptr<double>();
  emit_pairs(prepared,
             [&](std::int64_t slot, std::int64_t center, std::int64_t neighbor,
                 const std::int32_t*, const double* displacement) {
               source_data[slot] = neighbor;
               destination_data[slot] = center;
               mask_data[slot] = true;
               if (vec_f32 != nullptr) {
                 vec_f32[slot * 3] = static_cast<float>(displacement[0]);
                 vec_f32[slot * 3 + 1] = static_cast<float>(displacement[1]);
                 vec_f32[slot * 3 + 2] = static_cast<float>(displacement[2]);
               } else {
                 vec_f64[slot * 3] = displacement[0];
                 vec_f64[slot * 3 + 1] = displacement[1];
                 vec_f64[slot * 3 + 2] = displacement[2];
               }
             });

  torch::Tensor destination_row_ptr =
      torch::from_blob(const_cast<std::int64_t*>(prepared.row_ptr.data()),
                       {node_count + 1}, index_options)
          .clone();
  torch::Tensor source_row_ptr = torch::empty({node_count + 1}, index_options);
  torch::Tensor source_order = torch::empty({edge_count}, index_options);
  // Only the physical edges enter a source segment; the guard slots form the
  // suffix, outside every row, which is where a masked edge belongs.
  auto* order_data = source_order.data_ptr<std::int64_t>();
  group_by_node(source_data, real_edges, node_count,
                source_row_ptr.data_ptr<std::int64_t>(), order_data);
  for (std::int64_t slot = real_edges; slot < edge_count; ++slot) {
    order_data[slot] = slot;
  }
  return {edge_index,          edge_vec,     edge_mask,
          destination_row_ptr, source_order, source_row_ptr};
}

TORCH_LIBRARY_FRAGMENT(deepmd, library) {
  library.def(
      "neighbor_search(Tensor coord, Tensor cell, bool periodic, float rcut) "
      "-> (Tensor destination, Tensor source, Tensor image)");
  library.def(
      "neighbor_graph(Tensor coord, Tensor cell, bool periodic, float rcut, "
      "ScalarType edge_dtype) -> (Tensor edge_index, Tensor edge_vec, "
      "Tensor edge_mask, Tensor destination_row_ptr, Tensor source_order, "
      "Tensor source_row_ptr)");
}

TORCH_LIBRARY_IMPL(deepmd, CPU, library) {
  library.impl("neighbor_search", &neighbor_search);
  library.impl("neighbor_graph", &neighbor_graph);
}

}  // namespace deepmd
