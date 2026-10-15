# SPDX-License-Identifier: LGPL-3.0-or-later
"""Runtime selection and dispatch for pt_expt carry-all graph builders."""

import logging
from typing import (
    TYPE_CHECKING,
)

import torch

from deepmd.dpmodel.utils.lmdb_data import (
    PHANTOM_ATOM_TYPE,
)

if TYPE_CHECKING:
    from deepmd.dpmodel.utils.exclude_mask import (
        PairExcludeMask,
    )
    from deepmd.dpmodel.utils.neighbor_graph import (
        NeighborGraph,
    )

log = logging.getLogger(__name__)

# Warn once per process: resolve_auto_graph_builder runs per batch after
# call-time resolution, but "install nvalchemi-toolkit-ops" is a one-shot
# action for the user.
_warned_auto_no_nv = False


def resolve_auto_graph_builder(
    device: torch.device | str,
    nf: int = 1,
) -> str:
    """Resolve ``neighbor_graph_method="auto"`` to a concrete inference builder.

    Single owner of the inference / DeepEval auto ladder. Training uses
    :func:`resolve_neighbor_graph_method`, which never selects ``vesin``.

    Mirrors :func:`deepmd.pt.model.model.sezm_model._select_neighbor_builder`:
    ``vesin`` is eligible only for a single-frame batch (``nf == 1``), because
    its API loops frames in Python (~1 ms/frame). Multi-frame batches stay on
    ``nv`` (CUDA) or ``dense`` so ``auto_batch_size`` / ``dp test`` do not
    regress to the per-frame loop.

    Notes
    -----
    * CUDA + ``nvalchemiops``: ``nv`` (any ``nf``).
    * CPU + the operator library: ``cell``, whose search is threaded.
    * ``nf == 1`` + ``vesin.torch``: ``vesin``.
    * otherwise: ``dense``.

    ``ase`` is never chosen automatically. All builders emit the same carry-all
    neighbor set; the choice is performance-only. Builders run eagerly outside
    traced / compiled regions, so this does not change ``.pt2`` artifacts.

    Parameters
    ----------
    device : torch.device or str
        Device the coordinates live on (or will be moved to). Controls whether
        the CUDA-only ``nv`` builder is eligible.
    nf : int, default: 1
        Number of frames in the batch. ``vesin`` is selected only when
        ``nf == 1`` and ``vesin.torch`` is importable.

    Returns
    -------
    str
        One of ``"nv"``, ``"cell"``, ``"vesin"``, or ``"dense"``.

    Raises
    ------
    ValueError
        If ``nf`` is not a positive ``int`` (``bool`` is rejected).
    """
    global _warned_auto_no_nv

    from deepmd.pt.utils.nv_nlist import (
        is_nv_available,
    )
    from deepmd.pt_expt.utils.cell_graph_builder import (
        is_cell_search_available,
    )
    from deepmd.pt_expt.utils.vesin_neighbor_list import (
        is_vesin_torch_available,
    )

    # ``bool`` is a subclass of ``int``; reject it explicitly.
    if type(nf) is not int:
        raise ValueError(f"nf must be a positive int, got {nf!r}")
    if nf < 1:
        raise ValueError(f"nf must be >= 1, got {nf}")

    dev = torch.device(device)
    nv_available = is_nv_available()
    if dev.type == "cuda" and nv_available:
        return "nv"
    if dev.type == "cpu" and is_cell_search_available():
        return "cell"
    if nf == 1 and is_vesin_torch_available():
        return "vesin"
    if dev.type == "cuda" and not nv_available:
        if not _warned_auto_no_nv:
            _warned_auto_no_nv = True
            log.warning(
                "nvalchemi-toolkit-ops is unavailable; falling back from "
                "neighbor_graph_method='auto' to the dense graph builder"
                + (
                    ""
                    if nf == 1
                    else " (vesin is not used for nf>1; its API loops frames in Python)"
                )
                + ". Install it with `pip install nvalchemi-toolkit-ops` to enable "
                "the NV graph builder."
            )
    return "dense"


def resolve_neighbor_graph_method(
    requested: str,
    device: torch.device,
) -> str:
    """Resolve a training graph-builder policy to one concrete backend.

    Parameters
    ----------
    requested
        ``"auto"``, ``"dense"``, or ``"nv"``.
    device
        Device used by the model on the current rank.

    Returns
    -------
    str
        The concrete builder name, either ``"dense"`` or ``"nv"``.
        Training auto never selects ``vesin`` (per-frame Python loop); use
        :func:`resolve_auto_graph_builder` for inference auto selection.

    Raises
    ------
    ValueError
        If the requested method is unknown or NV is requested on a non-CUDA
        device.
    ImportError
        If NV is requested explicitly but nvalchemiops is unavailable.
    """
    if requested not in {"auto", "dense", "nv"}:
        raise ValueError(
            f"unknown training neighbor_graph_method {requested!r}; "
            "use 'auto', 'dense', or 'nv'"
        )
    if requested == "dense":
        return "dense"

    from deepmd.pt.utils.nv_nlist import (
        is_nv_available,
    )

    if requested == "auto":
        if device.type != "cuda":
            return "dense"
        if is_nv_available():
            return "nv"
        log.warning(
            "nvalchemi-toolkit-ops is unavailable; falling back from "
            "neighbor_graph_method='auto' to the dense graph builder. "
            "Install it with `pip install nvalchemi-toolkit-ops` to enable "
            "the NV graph builder."
        )
        return "dense"
    if device.type != "cuda":
        raise ValueError(
            "neighbor_graph_method='nv' requires a CUDA training device, "
            f"got {device!s}"
        )
    if not is_nv_available():
        raise ImportError(
            "neighbor_graph_method='nv' requires nvalchemi-toolkit-ops. "
            "Install the DeePMD-kit 'nvalchemi' extra or use 'auto'/'dense'."
        )
    return "nv"


def build_ragged_neighbor_graph(
    method: str,
    coord: torch.Tensor,
    atype: torch.Tensor,
    n_node: torch.Tensor,
    box: torch.Tensor | None,
    rcut: float,
    pair_excl: "PairExcludeMask | None",
    *,
    with_csr: bool = False,
) -> "NeighborGraph":
    """Build a carry-all graph over a batch whose node axis is already flat.

    The searches all take a rectangular tensor -- ``dense`` compares every pair
    of one, and the others derive their per-frame bounds from its shape -- so
    the frames are widened to a common width here and the resulting graph is
    narrowed back onto the flat axis. Widening is a scatter and narrowing a
    renumbering that drops no edge, since no builder draws one to a padded
    slot. ``nv`` additionally withholds the padded slots from the search
    itself, so for that backend the widening does not reach the geometry.

    Parameters
    ----------
    method : str
        The concrete builder, as resolved by :func:`resolve_neighbor_graph_method`.
    coord : torch.Tensor
        Local coordinates with shape ``(N, 3)``, frame-major over ``n_node``.
    atype : torch.Tensor
        Local atom types with shape ``(N,)``.
    n_node : torch.Tensor
        Atoms per frame with shape ``(nf,)``.
    box : torch.Tensor or None
        Simulation cell with shape ``(nf, 3, 3)``, or ``None`` for non-periodic.
    rcut : float
        Cutoff radius.
    pair_excl : PairExcludeMask or None
        Model-level pair exclusion, folded into the edge mask at build time.
    with_csr : bool, default: False
        Whether to attach destination/source CSR views.

    Returns
    -------
    NeighborGraph
        A graph whose node axis is the one described by ``n_node``.
    """
    from deepmd.dpmodel.utils.neighbor_graph import (
        compact_nodes,
    )

    nf = int(n_node.shape[0])
    width = int(n_node.max()) if nf else 0
    # Position of each flat node in the padded batch, built without a pass over
    # the frames: its frame times the width, plus its rank within that frame.
    frame = torch.repeat_interleave(
        torch.arange(nf, dtype=n_node.dtype, device=n_node.device), n_node
    )
    offset = torch.cumsum(n_node, 0) - n_node
    # The node total is the leading axis of the flat coordinates, so it is read
    # from the shape rather than from ``n_node.sum()``: summing on the device
    # and reading the result back synchronizes the stream once per step.
    slot = (
        torch.arange(coord.shape[0], dtype=n_node.dtype, device=n_node.device)
        - offset[frame]
    )
    padded_index = frame * width + slot

    padded_coord = coord.new_zeros((nf * width, 3))
    padded_atype = atype.new_full((nf * width,), PHANTOM_ATOM_TYPE)
    padded_coord[padded_index] = coord
    padded_atype[padded_index] = atype
    padded_coord = padded_coord.reshape(nf, width, 3)
    padded_atype = padded_atype.reshape(nf, width)

    graph = build_neighbor_graph_for_method(
        method, padded_coord, padded_atype, box, rcut, pair_excl, with_csr=with_csr
    )
    return compact_nodes(graph, padded_atype.reshape(-1) >= 0)[0]


def build_neighbor_graph_for_method(
    method: str,
    coord: torch.Tensor,
    atype: torch.Tensor,
    box: torch.Tensor | None,
    rcut: float,
    pair_excl: "PairExcludeMask | None",
    *,
    with_csr: bool = False,
) -> "NeighborGraph":
    """Build a carry-all graph with one concrete pt_expt backend."""
    if box is not None and torch.all(box == 0):
        box = None
    if method == "dense":
        from deepmd.dpmodel.utils.neighbor_graph import (
            build_neighbor_graph,
        )

        return build_neighbor_graph(
            coord,
            atype,
            box,
            rcut,
            with_csr=with_csr,
            pair_excl=pair_excl,
        )
    if method == "ase":
        from deepmd.dpmodel.utils.neighbor_graph import (
            build_neighbor_graph_ase,
        )

        return build_neighbor_graph_ase(
            coord,
            atype,
            box,
            rcut,
            with_csr=with_csr,
            pair_excl=pair_excl,
        )
    if method == "cell":
        from deepmd.pt_expt.utils.cell_graph_builder import (
            build_neighbor_graph_cell,
        )

        return build_neighbor_graph_cell(
            coord,
            atype,
            box,
            rcut,
            with_csr=with_csr,
            pair_excl=pair_excl,
        )
    if method == "vesin":
        from deepmd.pt_expt.utils.vesin_graph_builder import (
            build_neighbor_graph_vesin,
        )

        return build_neighbor_graph_vesin(
            coord,
            atype,
            box,
            rcut,
            with_csr=with_csr,
            pair_excl=pair_excl,
        )
    if method == "nv":
        from deepmd.pt_expt.utils.nv_graph_builder import (
            build_neighbor_graph_nv,
        )

        return build_neighbor_graph_nv(
            coord,
            atype,
            box,
            rcut,
            with_csr=with_csr,
            pair_excl=pair_excl,
        )
    raise ValueError(
        f"unknown neighbor_graph_method {method!r}; use 'dense', 'ase', "
        "'cell', 'vesin', or 'nv'"
    )


class AllPairsGraph:
    """Unordered intra-frame all-pairs graph for non-periodic LR terms.

    Only the fields consumed by the long-range consumers are populated:
    ``n_node``, ``edge_index`` (``[src, dst]``), ``edge_vec``
    (``r_src - r_dst``), and an all-true ``edge_mask``.  Each atom pair
    appears exactly once (``i < j``), so the energy contract is
    ``sum_{i < j}`` with NO ``0.5`` factor, and the force/virial assembly
    must scatter each pair to BOTH endpoints.

    Attributes
    ----------
    n_node
        Per-frame node counts with shape ``(nf,)``.
    edge_index
        ``[src, dst]`` node endpoints with shape ``(2, E)``, int64.
    edge_vec
        Per-edge displacement ``r_src - r_dst`` with shape ``(E, 3)``.
    edge_mask
        Valid-edge mask with shape ``(E,)``; always all-true (the builder
        emits exactly the real pairs, there is no padding).
    """

    def __init__(
        self,
        n_node: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vec: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> None:
        self.n_node = n_node
        self.edge_index = edge_index
        self.edge_vec = edge_vec
        self.edge_mask = edge_mask


def build_all_pairs_unordered_graph(
    coord: torch.Tensor,
    n_node: torch.Tensor,
) -> AllPairsGraph:
    """Build the unordered (``i < j``) intra-frame all-pairs graph directly.

    Every pair of atoms that shares a frame participates in a non-periodic
    all-pairs interaction, so no distance search, padding, or cutoff is
    involved: the pair indices are generated straight from the per-frame
    node counts and the displacements are gathered from them.  Storage is
    proportional to the true pair count ``sum_f n_f (n_f - 1) / 2`` -- half
    the directed all-pairs count, and without widening every frame to the
    batch maximum width.

    The builder runs eagerly (outside any compiled region), so the
    data-dependent pair count as a Python int is fine here.

    Parameters
    ----------
    coord
        Flat local coordinates with shape ``(N, 3)``, frame-major over
        ``n_node``.
    n_node
        Atoms per frame with shape ``(nf,)``.

    Returns
    -------
    AllPairsGraph
        The unordered all-pairs graph on the flat node axis.
    """
    nf = n_node.shape[0]
    device = n_node.device
    n_pairs = (n_node * (n_node - 1)) // 2  # (nf,)
    total_pairs = int(n_pairs.sum())
    if total_pairs == 0:
        return AllPairsGraph(
            n_node=n_node,
            edge_index=torch.zeros((2, 0), dtype=torch.int64, device=device),
            edge_vec=torch.zeros((0, 3), dtype=coord.dtype, device=coord.device),
            edge_mask=torch.zeros((0,), dtype=torch.bool, device=device),
        )

    pair_frame = torch.repeat_interleave(
        torch.arange(nf, dtype=torch.int64, device=device), n_pairs
    )
    pair_start = torch.cumsum(n_pairs, 0) - n_pairs  # (nf,)
    node_start = torch.cumsum(n_node, 0) - n_node  # (nf,)
    # Pair rank within its frame, in row-major strict-upper-triangle order
    # (0,1), (0,2), ..., (0,n-1), (1,2), ...
    k = (
        torch.arange(total_pairs, dtype=torch.int64, device=device)
        - pair_start[pair_frame]
    ).to(torch.float64)
    n = n_node[pair_frame].to(torch.float64)
    # Row i starts at S_i = i * (2n - 1 - i) / 2; solve S_i <= k < S_{i+1}.
    # float64 is exact for the sizes involved (n^2 << 2^53); one correction
    # step below guards the sqrt rounding regardless.
    discr = (2 * n - 1).square() - 8 * k
    ii = torch.floor(((2 * n - 1) - torch.sqrt(discr.clamp_min(0.0))) / 2)
    ii = ii.clamp_min(0.0)

    def row_start(i: torch.Tensor) -> torch.Tensor:
        return i * (2 * n - 1 - i) / 2

    ii = torch.where(k < row_start(ii), ii - 1, ii)
    ii = torch.where(k >= row_start(ii + 1), ii + 1, ii)
    jj = ii + 1 + (k - row_start(ii))

    src = node_start[pair_frame] + ii.to(torch.int64)
    dst = node_start[pair_frame] + jj.to(torch.int64)
    edge_index = torch.stack([src, dst])
    edge_vec = coord[src] - coord[dst]
    edge_mask = torch.ones(total_pairs, dtype=torch.bool, device=device)
    return AllPairsGraph(
        n_node=n_node,
        edge_index=edge_index,
        edge_vec=edge_vec,
        edge_mask=edge_mask,
    )
