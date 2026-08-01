# SPDX-License-Identifier: LGPL-3.0-or-later
"""Runtime selection and dispatch for pt_expt carry-all graph builders."""

import logging
from typing import (
    TYPE_CHECKING,
)

import torch

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
