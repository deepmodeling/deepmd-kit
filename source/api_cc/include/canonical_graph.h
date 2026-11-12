// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <stdint.h>

/** Width of node indices in bytes; edge counts and CSR positions are int64. */
typedef enum DP_CanonicalIndexWidth {
  DP_CANONICAL_INDEX32 = 4,
  DP_CANONICAL_INDEX64 = 8
} DP_CanonicalIndexWidth;

/**
 * @brief Halo exchange of the node set of one domain of a decomposed system,
 * the forward communication of a LAMMPS brick decomposition in model-node
 * space.
 *
 * The node set holds the ``nloc`` owned nodes first, then the halo nodes,
 * images of atoms that the swaps deliver. Swap ``s`` sends ``sendnum[s]``
 * nodes to rank ``sendproc[s]`` and receives ``recvnum[s]`` nodes from rank
 * ``recvproc[s]`` into the consecutive halo nodes from ``firstrecv[s]``; the
 * first swap receives into node ``nloc``, every further swap into the node
 * after those of the swap before it, and together they fill every halo node.
 * A swap whose two ranks are the caller's copies its nodes. The swaps run in
 * order, so a swap may send halo nodes that an earlier swap received. The
 * arrays of the swaps reside in host memory and the send list on the model
 * device.
 */
typedef struct DP_HaloExchange {
  /** Number of swaps. */
  int nswap;
  /** Rank each swap sends to, shape ``(nswap)``. */
  const int* sendproc;
  /** Rank each swap receives from, shape ``(nswap)``. */
  const int* recvproc;
  /** Nodes each swap sends, shape ``(nswap)``. */
  const int* sendnum;
  /** Nodes each swap receives, shape ``(nswap)``. */
  const int* recvnum;
  /** First halo node each swap receives into, shape ``(nswap)``. */
  const int* firstrecv;
  /** Nodes the swaps send, concatenated in swap order, on the device. */
  const int* d_sendlist;
} DP_HaloExchange;

#ifdef __cplusplus
namespace deepmd {
/** Select node-index width, preserving signed words at native host boundaries.
 */
inline DP_CanonicalIndexWidth canonical_index_width(int64_t node_count) {
  return node_count <= (int64_t{1} << 31) ? DP_CANONICAL_INDEX32
                                          : DP_CANONICAL_INDEX64;
}
}  // namespace deepmd
#endif
