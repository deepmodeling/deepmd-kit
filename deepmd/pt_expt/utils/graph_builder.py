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
    slot = (
        torch.arange(int(n_node.sum()), dtype=n_node.dtype, device=n_node.device)
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
        "'vesin', or 'nv'"
    )
