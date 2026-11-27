# SPDX-License-Identifier: LGPL-3.0-or-later
"""Parallel parity of a bridged SeZM model in the pt backend.

A bridged model's parallel (``comm_dict``) path must reproduce its
single-domain path. The analytical term reads the type of every edge source,
which is a ghost image row under domain decomposition, and the contact clamp
reads the types of both ends of every edge. Atoms 0 and 1 form a close pair
that STRADDLES the periodic x boundary, so the close contact's source is a
ghost row, and the gaps place the pair in the frozen zone and in the opening
zone of the clamp.
"""

import unittest

import numpy as np
import torch

from deepmd.pt.model.model import (
    get_model,
)

from .test_sezm_parallel import (
    _perturb_descriptor,
    _self_comm_dict,
    _tiny_parallel_model_params,
)

_L = 10.0  # box edge (= rcut * 2.5, matching the non-bridged parity file)


def _bridged_model_params() -> dict:
    params = _tiny_parallel_model_params()
    # ZBL bridging needs REAL element symbols (nuclear charges). The Ni-O
    # pair starts its data edge at its covalent bond length, 1.73 Å, so with
    # the default clamp fraction the read separation is frozen below 1.30 Å.
    params["type_map"] = ["Ni", "O"]
    params["bridging_method"] = "ZBL"
    return params


def _close_pair_coords(gap: float, nloc: int = 4) -> np.ndarray:
    """Atoms 0/1 straddle the x periodic boundary at distance ``gap``."""
    half = gap / 2.0
    coords = np.array(
        [
            [half, 5.0, 5.0],
            [_L - half, 5.0, 5.0],
            [4.0, 5.0, 5.0],
            [5.1, 5.2, 5.0],
        ],
        dtype=np.float64,
    )
    assert coords.shape[0] == nloc
    return coords.reshape(1, nloc, 3)


def _build_close_pair_system(
    model: torch.nn.Module, device: torch.device, gap: float
) -> dict[str, torch.Tensor]:
    """The non-bridged file's ``_build_extended_system`` with fixed coords."""
    from deepmd.dpmodel.utils.nlist import (
        build_neighbor_list,
        extend_coord_with_ghosts,
    )
    from deepmd.pt_expt.utils.edge_schema import (
        edge_schema_from_extended,
    )

    nloc = 4
    rcut = float(model.get_rcut())
    sel = list(model.get_sel())
    coord_np = _close_pair_coords(gap, nloc)
    ntypes = len(model.get_type_map())
    atype_np = (np.arange(nloc, dtype=np.int32) % ntypes).reshape(1, nloc)
    box = np.eye(3, dtype=np.float64) * _L

    extended_coord, extended_atype, mapping = extend_coord_with_ghosts(
        coord_np, atype_np, box.reshape(1, 9), rcut
    )
    nlist = build_neighbor_list(
        extended_coord,
        extended_atype,
        nloc,
        rcut,
        sel,
        distinguish_types=not model.mixed_types(),
    )
    extended_coord = np.asarray(extended_coord).reshape(1, -1, 3)

    ext_coord = torch.tensor(extended_coord, dtype=torch.float64, device=device)
    ext_atype = torch.tensor(
        np.asarray(extended_atype), dtype=torch.int64, device=device
    )
    nlist_t = torch.tensor(np.asarray(nlist), dtype=torch.int64, device=device)
    mapping_t = torch.tensor(np.asarray(mapping), dtype=torch.int64, device=device)

    formatted = model.format_nlist(ext_coord, ext_atype, nlist_t)
    schema = edge_schema_from_extended(
        ext_coord, ext_atype[:, :nloc], formatted, mapping_t
    )
    return {
        "coord": schema.coord,
        "atype": schema.atype,
        "extended_atype": ext_atype,
        "edge_index": schema.edge_index,
        "edge_vec": schema.edge_vec,
        "edge_scatter_index": schema.edge_scatter_index,
        "edge_mask": schema.edge_mask,
        "mapping": mapping_t,
        "nloc": nloc,
        "nall": ext_coord.shape[1],
    }


class TestSeZMBridgingSelfCommParity(unittest.TestCase):
    """The bridged parallel path equals the single-domain path."""

    @classmethod
    def setUpClass(cls) -> None:
        from deepmd.pt_expt.utils.comm import (
            ensure_comm_registered,
        )

        ensure_comm_registered()

    def _run_pair(self, gap: float, device: torch.device):
        model = get_model(_bridged_model_params())
        model.eval()
        model.to(device)
        _perturb_descriptor(model.atomic_model.descriptor)
        sysm = _build_close_pair_system(model, device, gap)
        comm = _self_comm_dict(sysm["mapping"], sysm["nloc"], sysm["nall"])

        ref = model.forward_lower(
            sysm["coord"],
            sysm["atype"],
            sysm["edge_index"],
            sysm["edge_vec"],
            sysm["edge_scatter_index"],
            sysm["edge_mask"],
            do_atomic_virial=True,
        )
        par = model.forward_lower(
            sysm["coord"],
            sysm["atype"],
            sysm["edge_scatter_index"],
            sysm["edge_vec"],
            sysm["edge_scatter_index"],
            sysm["edge_mask"],
            do_atomic_virial=True,
            comm_dict=comm,
            extended_atype=sysm["extended_atype"],
        )
        return ref, par

    def _assert_parity(self, gap: float) -> None:
        device = torch.device("cpu")
        ref, par = self._run_pair(gap, device)
        self.assertGreater(
            ref["extended_force"].abs().max().item(),
            1e-6,
            msg="reference forces are ~0; the parity check would be vacuous",
        )
        for key in ("energy", "extended_force", "virial"):
            # The perturbed weights give outputs of order 1e6 to 1e7, and the
            # two paths differ by round-off of about 1e-14 of the largest entry,
            # so the tolerance follows that entry rather than each one.
            torch.testing.assert_close(
                par[key],
                ref[key],
                rtol=0.0,
                atol=1e-11 * ref[key].abs().max().item(),
                msg=f"mismatch in {key}",
            )

    def test_parity_frozen_pair_cpu(self) -> None:
        """A pair below the frozen radius of the clamp crosses the boundary."""
        self._assert_parity(0.4)

    def test_parity_opening_pair_cpu(self) -> None:
        """A pair inside the opening zone of the clamp crosses the boundary."""
        self._assert_parity(1.5)


if __name__ == "__main__":
    unittest.main()
