# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for the ``NvNeighborList`` builder.

These cover the builder paths the DeepEval end-to-end equivalence test
(``test_nlist_backend.py``) cannot reach with its small ``batch_naive`` systems:
the ``batch_cell_list`` method and the over-capacity distance-trim path, plus the
periodic-box requirement.  Built neighbor lists are compared against the native
dense builder at the nlist level (edge topology + geometry).
"""

import contextlib
import unittest
from unittest.mock import (
    patch,
)

import torch

from deepmd.pt.utils import (
    nv_nlist,
)
from deepmd.pt.utils.nlist import (
    extend_input_and_build_neighbor_list,
)
from deepmd.pt.utils.nv_nlist import (
    NvNeighborList,
)

_NV_AVAILABLE = nv_nlist.is_nv_available()
_TEST_DEVICES = [torch.device("cpu")]
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    _TEST_DEVICES.append(torch.device("cuda:0"))


def _edge_topology_from_extended(
    mapping: torch.Tensor,
    nlist: torch.Tensor,
) -> torch.Tensor:
    """Convert an extended-coordinate nlist to sorted local edge topology rows."""
    nf, nloc, nsel = nlist.shape
    nall = mapping.shape[1]
    device = nlist.device
    dst = torch.arange(nf * nloc, dtype=torch.long, device=device).repeat_interleave(
        nsel
    )
    frame = dst // nloc
    center = dst % nloc
    neighbor = nlist.reshape(-1).to(dtype=torch.long)
    valid = neighbor >= 0
    neighbor_safe = torch.where(valid, neighbor, torch.zeros_like(neighbor))
    src_local = mapping.reshape(-1).index_select(0, frame * nall + neighbor_safe)
    valid = valid & (src_local >= 0) & (src_local < nloc)
    rows = torch.stack([frame[valid], src_local[valid], center[valid]], dim=1)
    key = rows[:, 0] * nloc * nloc + rows[:, 1] * nloc + rows[:, 2]
    return rows.index_select(0, torch.argsort(key))


def _edge_geometry_from_extended(
    extended_coord: torch.Tensor,
    mapping: torch.Tensor,
    nlist: torch.Tensor,
) -> torch.Tensor:
    """Convert an extended-coordinate nlist to sorted edge-vector rows."""
    nf, nloc, nsel = nlist.shape
    nall = extended_coord.shape[1]
    device = extended_coord.device
    dst = torch.arange(nf * nloc, dtype=torch.long, device=device).repeat_interleave(
        nsel
    )
    frame = dst // nloc
    center = dst % nloc
    neighbor = nlist.reshape(-1).to(dtype=torch.long)
    valid = neighbor >= 0
    neighbor_safe = torch.where(valid, neighbor, torch.zeros_like(neighbor))
    src_local = mapping.reshape(-1).index_select(0, frame * nall + neighbor_safe)
    src_valid = (src_local >= 0) & (src_local < nloc)

    coord_flat = extended_coord.reshape(nf * nall, 3)
    src_coord = coord_flat.index_select(0, frame * nall + neighbor_safe)
    dst_coord = coord_flat.index_select(0, frame * nall + center)
    edge_vec = src_coord - dst_coord
    valid = valid & src_valid & (torch.sum(edge_vec * edge_vec, dim=-1) > 1.0e-10)
    topo = torch.stack([frame[valid], src_local[valid], center[valid]], dim=1)
    key = topo[:, 0] * nloc * nloc + topo[:, 1] * nloc + topo[:, 2]
    return edge_vec[valid].index_select(0, torch.argsort(key)).to(torch.float64)


def _assert_extended_atype_matches_mapping(
    test_case: unittest.TestCase,
    local_atype: torch.Tensor,
    extended_atype: torch.Tensor,
    mapping: torch.Tensor,
) -> None:
    """Check that each real extended atom keeps the type of its mapped local atom."""
    nf, nall = extended_atype.shape
    nloc = local_atype.shape[1]
    frame = torch.arange(nf, dtype=torch.long, device=extended_atype.device)
    frame = frame.unsqueeze(1).expand(nf, nall)
    valid = extended_atype >= 0
    mapped = mapping.clamp(min=0, max=nloc - 1)
    expected = local_atype[frame, mapped]
    test_case.assertTrue(torch.equal(extended_atype[valid], expected[valid]))


@unittest.skipUnless(
    _NV_AVAILABLE,
    "NVIDIA Toolkit-Ops neighbor list is unavailable",
)
class TestNVNList(unittest.TestCase):
    def _build_case(
        self, nframes: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        coord_one = torch.tensor(
            [
                [0.2, 0.2, 0.2],
                [7.7, 0.2, 0.2],
                [0.2, 7.6, 0.2],
                [3.8, 3.9, 4.1],
            ],
            dtype=torch.float64,
            device=device,
        )
        coord = coord_one.unsqueeze(0).repeat(nframes, 1, 1)
        if nframes > 1:
            coord[1] = torch.tensor(
                [
                    [2.0, 2.0, 2.0],
                    [4.0, 2.0, 2.0],
                    [2.0, 4.0, 2.0],
                    [4.0, 4.0, 4.0],
                ],
                dtype=coord.dtype,
                device=coord.device,
            )
        atype = torch.tensor([[0, 1, 0, 1]], dtype=torch.int32, device=device)
        atype = atype.repeat(nframes, 1)
        box = torch.eye(3, dtype=torch.float64, device=device).reshape(1, 9) * 8.0
        box = box.repeat(nframes, 1)
        return coord, atype, box

    def _build_overfull_case(
        self, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        coord = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.9, 0.0, 0.0],
                [2.1, 0.0, 0.0],
                [3.4, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [2.2, 1.1, 0.0],
            ],
            dtype=torch.float64,
            device=device,
        ).unsqueeze(0)
        atype = torch.tensor([[0, 1, 0, 1, 0, 1]], dtype=torch.int32, device=device)
        box = torch.eye(3, dtype=torch.float64, device=device).reshape(1, 9) * 20.0
        return coord, atype, box

    def _assert_nv_matches_native(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        box: torch.Tensor,
        rcut: float,
        sel: list[int],
        force_cell_list: bool = False,
    ) -> None:
        # native: (extended_coord, extended_atype, mapping, nlist)
        native = extend_input_and_build_neighbor_list(
            coord,
            atype,
            rcut,
            sel,
            mixed_types=True,
            box=box,
        )
        # NeighborList strategy: (extended_coord, extended_atype, nlist, mapping)
        builder = NvNeighborList()
        # Pin the current CUDA device so the Toolkit-Ops backend launches there.
        device_ctx = (
            torch.cuda.device(coord.device)
            if coord.is_cuda
            else contextlib.nullcontext()
        )
        with device_ctx:
            if force_cell_list:
                with patch.object(nv_nlist, "NV_CELL_LIST_THRESHOLD", 1):
                    nv = builder.build(coord, atype, box, rcut, sel)
            else:
                nv = builder.build(coord, atype, box, rcut, sel)
        native_coord, _, native_mapping, native_nlist = native
        nv_coord, nv_atype, nv_nlist_out, nv_mapping = nv
        # The strategy trims to sum(sel) itself, so the width is fixed.
        self.assertEqual(nv_nlist_out.shape[-1], sum(sel))
        self.assertTrue(
            torch.equal(
                _edge_topology_from_extended(native_mapping, native_nlist),
                _edge_topology_from_extended(nv_mapping, nv_nlist_out),
            )
        )
        torch.testing.assert_close(
            _edge_geometry_from_extended(native_coord, native_mapping, native_nlist),
            _edge_geometry_from_extended(nv_coord, nv_mapping, nv_nlist_out),
            atol=1.0e-10,
            rtol=1.0e-10,
        )
        _assert_extended_atype_matches_mapping(self, atype, nv_atype, nv_mapping)

    def test_cell_list_matches_native(self) -> None:
        """The ``batch_cell_list`` method (forced via the threshold) matches the
        native builder over a multi-frame periodic batch.  End-to-end systems are
        always below the threshold and take ``batch_naive``.
        """
        for device in _TEST_DEVICES:
            with self.subTest(device=str(device)):
                coord, atype, box = self._build_case(2, device)
                self._assert_nv_matches_native(
                    coord=coord,
                    atype=atype,
                    box=box,
                    rcut=3.0,
                    sel=[8],
                    force_cell_list=True,
                )

    def test_overfull_truncates_to_sel(self) -> None:
        """A center with more real neighbors than ``sum(sel)`` is distance-sorted
        and trimmed to the nearest ``sum(sel)`` -- the path behind the
        compiled-graph width bug, which end-to-end systems never reach.
        """
        for device in _TEST_DEVICES:
            with self.subTest(device=str(device)):
                coord, atype, box = self._build_overfull_case(device)
                self._assert_nv_matches_native(
                    coord=coord,
                    atype=atype,
                    box=box,
                    rcut=4.0,
                    sel=[2],
                    force_cell_list=False,
                )

    def test_requires_periodic_box(self) -> None:
        """The cell list needs a periodic box; ``box=None`` is rejected."""
        for device in _TEST_DEVICES:
            with self.subTest(device=str(device)):
                coord, atype, _ = self._build_case(1, device)
                with self.assertRaises(ValueError):
                    NvNeighborList().build(coord, atype, None, 3.0, [8])
