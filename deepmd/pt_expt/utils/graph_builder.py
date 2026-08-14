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

    Policy
    ------
    * CUDA + ``nvalchemiops``: ``nv`` (any ``nf``).
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
        One of ``"nv"``, ``"vesin"``, or ``"dense"``.

    Raises
    ------
    ValueError
        If ``nf`` is not a positive ``int`` (``bool`` is rejected).
    """
    global _warned_auto_no_nv

    from deepmd.pt.utils.nv_nlist import (
        is_nv_available,
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
