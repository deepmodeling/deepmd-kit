# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for the ``NvNeighborList`` builder.

These cover the builder paths the DeepEval end-to-end equivalence test
(``test_nlist_backend.py``) cannot reach with its small ``batch_naive`` systems:
the ``batch_cell_list`` method and the over-capacity distance-trim path, plus the
non-periodic path.  Built neighbor lists are compared against the native dense
builder at the nlist level (edge topology + geometry).
"""

import unittest
from unittest.mock import (
    patch,
)

import numpy as np
import torch

from deepmd.pt.utils import (
    nv_nlist,
)
from deepmd.pt.utils.nlist import (
    extend_input_and_build_neighbor_list,
)
from deepmd.pt.utils.nv_nlist import (
    NvNeighborList,
    _input_device_context,
)
from deepmd.pt_expt.utils.edge_schema import (
    edge_schema_from_extended,
)
from deepmd.pt_expt.utils.vesin_neighbor_list import (
    VesinNeighborList,
    is_vesin_torch_available,
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
        box: torch.Tensor | None,
        rcut: float,
        sel: list[int],
        force_cell_list: bool = False,
    ) -> None:
        with _input_device_context(coord.device):
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
            if force_cell_list:
                with (
                    patch.object(nv_nlist, "NV_CELL_LIST_THRESHOLD", 1),
                    patch.object(nv_nlist, "NV_NONPERIODIC_CELL_LIST_THRESHOLD", 1),
                    patch.object(nv_nlist, "NV_CPU_CELL_LIST_THRESHOLD", 1),
                ):
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
                _edge_geometry_from_extended(
                    native_coord, native_mapping, native_nlist
                ),
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

    def test_nonperiodic_matches_native(self) -> None:
        """Non-periodic systems keep local atoms and match the native builder."""
        for device in _TEST_DEVICES:
            with self.subTest(device=str(device)):
                coord, atype, _ = self._build_case(1, device)
                self._assert_nv_matches_native(
                    coord=coord,
                    atype=atype,
                    box=None,
                    rcut=3.0,
                    sel=[8],
                    force_cell_list=False,
                )

    def test_nonperiodic_cell_list_matches_native(self) -> None:
        """The no-box ``batch_cell_list`` path returns zero shifts."""
        for device in _TEST_DEVICES:
            with self.subTest(device=str(device)):
                coord, atype, _ = self._build_case(1, device)
                self._assert_nv_matches_native(
                    coord=coord,
                    atype=atype,
                    box=None,
                    rcut=3.0,
                    sel=[8],
                    force_cell_list=True,
                )


def _to_numpy(x) -> np.ndarray:
    """Detach a tensor or pass a numpy array through to a numpy array."""
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def _canonical_edges(schema) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the real edges of an ``EdgeNeighborList`` in a builder-independent
    order as ``(src, dst, edge_vec)``.

    Edges are keyed by ``(src, dst, round(edge_vec))`` so that the multiple
    periodic images of one neighbor pair (distinct displacements that share a
    ``(src, dst)``) are disambiguated and ordered deterministically.
    """
    mask = _to_numpy(schema.edge_mask).astype(bool)
    edge_index = _to_numpy(schema.edge_index)
    src = edge_index[0][mask].astype(np.int64)
    dst = edge_index[1][mask].astype(np.int64)
    edge_vec = _to_numpy(schema.edge_vec)[mask].astype(np.float64)
    keys = np.round(edge_vec * 1.0e6).astype(np.int64)
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0], dst, src))
    return src[order], dst[order], edge_vec[order]


def _dense_keepall_edges(
    coord: torch.Tensor,
    atype: torch.Tensor,
    box: torch.Tensor | None,
    rcut: float,
):
    """Reference edge schema from the dense builder with no ``sel`` cap."""
    ext_coord, ext_atype, mapping, nlist = extend_input_and_build_neighbor_list(
        coord, atype, rcut, [1], mixed_types=True, box=box, cap_neighbors=False
    )
    return edge_schema_from_extended(
        ext_coord, atype, nlist, mapping, scatter_to_local=True
    )


class TestEdgeSchemaConsistency(unittest.TestCase):
    """``nv``, ``vesin``, and the sel-free dense builder must agree on the edge
    set, so that dropping ``sel`` and switching builders changes neither the
    neighbor topology nor the per-edge geometry.
    """

    def _dense_case(
        self, nframes: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # rcut > L / 2 makes each atom see several periodic images, exercising
        # the shifted-edge bookkeeping that differs most between builders.
        side = 5.0
        generator = torch.Generator(device="cpu").manual_seed(7)
        coord = (
            torch.rand(
                nframes, 8, 3, generator=generator, dtype=torch.float64, device="cpu"
            )
            * side
        )
        atype = torch.randint(
            0, 2, (nframes, 8), generator=generator, dtype=torch.int64, device="cpu"
        )
        box = (
            (torch.eye(3, dtype=torch.float64, device="cpu") * side)
            .reshape(1, 9)
            .repeat(nframes, 1)
        )
        return coord.to(device), atype.to(device), box.to(device)

    def _assert_builders_match(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        box: torch.Tensor | None,
        rcut: float,
        sel: list[int],
    ) -> None:
        ref_src, ref_dst, ref_vec = _canonical_edges(
            _dense_keepall_edges(coord, atype, box, rcut)
        )
        builders = []
        if _NV_AVAILABLE:
            builders.append(("nv", NvNeighborList()))
        if is_vesin_torch_available():
            builders.append(("vesin", VesinNeighborList()))
        self.assertTrue(builders, "no accelerated neighbor builder is available")
        for name, builder in builders:
            with self.subTest(builder=name):
                src, dst, vec = _canonical_edges(
                    builder.build(coord, atype, box, rcut, sel, return_mode="edges")
                )
                np.testing.assert_array_equal(src, ref_src)
                np.testing.assert_array_equal(dst, ref_dst)
                np.testing.assert_allclose(vec, ref_vec, atol=1.0e-9, rtol=1.0e-9)

    def test_periodic_single_frame(self) -> None:
        for device in _TEST_DEVICES:
            with self.subTest(device=str(device)):
                coord, atype, box = self._dense_case(1, device)
                self._assert_builders_match(coord, atype, box, 4.0, [64])

    def test_periodic_multi_frame(self) -> None:
        for device in _TEST_DEVICES:
            with self.subTest(device=str(device)):
                coord, atype, box = self._dense_case(3, device)
                self._assert_builders_match(coord, atype, box, 4.0, [64])

    def test_nonperiodic_single_frame(self) -> None:
        for device in _TEST_DEVICES:
            with self.subTest(device=str(device)):
                coord, atype, _ = self._dense_case(1, device)
                self._assert_builders_match(coord, atype, None, 4.0, [64])
