# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from importlib import (
    import_module,
)
from unittest.mock import (
    patch,
)

import torch

from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.nlist import (
    _nlist_distinguish_types_topk,
    build_directional_neighbor_list,
    build_multiple_neighbor_list,
    build_neighbor_list,
    extend_coord_with_ghosts,
    get_multiple_nlist_key,
    nlist_distinguish_types,
)
from deepmd.pt.utils.region import (
    inter2phys,
)

dtype = torch.float64


class TestNeighborTypeGrouping(unittest.TestCase):
    def test_cpu_keeps_stable_sort(self) -> None:
        nlist = torch.tensor([[[0, 1, 2, -1]]], device="cpu")
        atype = torch.tensor([[1, 0, 1]], device="cpu")
        expected = torch.tensor([[[1, -1, 0, 2]]], device="cpu")
        with patch.object(torch, "topk", wraps=torch.topk) as topk:
            torch.testing.assert_close(
                nlist_distinguish_types(nlist, atype, [2, 2]),
                expected,
                rtol=0,
                atol=0,
            )
            topk.assert_not_called()

    def test_stable_order(self) -> None:
        nlist = torch.tensor([[[1, 3, 0, -1, 2, 1]]], device=env.DEVICE)
        atype = torch.tensor([[0, 1, 0, -1]], device=env.DEVICE)
        tnlist = torch.tensor([[[1, -1, 0, -1, 0, 1]]], device=env.DEVICE)
        sel = [2, 3, 1]
        expected = torch.tensor([[[0, 2, 1, 1, -1, -1]]], device=env.DEVICE)
        torch.testing.assert_close(
            _nlist_distinguish_types_topk(nlist, tnlist, sel), expected, rtol=0, atol=0
        )
        torch.testing.assert_close(
            nlist_distinguish_types(nlist, atype, sel), expected, rtol=0, atol=0
        )

    def test_topk_matches_stable_sort(self) -> None:
        self._check_topk_matches_stable_sort(env.DEVICE)

    def test_npu_topk_matches_stable_sort(self) -> None:
        try:
            torch_npu = import_module("torch_npu")
        except ImportError:
            self.skipTest("torch_npu is unavailable")
        if not torch_npu.npu.is_available():
            self.skipTest("No Ascend NPU is available")

        device = torch.device("npu:0")
        # Exercise the public function, even when env.DEVICE is CPU or CUDA.
        nlist = torch.tensor([[[0, 1, 2, -1]]], device=device)
        atype = torch.tensor([[1, 0, 1]], device=device)
        expected = torch.tensor([[[1, -1, 0, 2]]], device=device)
        with patch.object(torch, "topk", wraps=torch.topk) as topk:
            torch.testing.assert_close(
                nlist_distinguish_types(nlist, atype, [2, 2]),
                expected,
                rtol=0,
                atol=0,
            )
            topk.assert_called()

        self._check_topk_matches_stable_sort(device)

    def _check_topk_matches_stable_sort(self, device: torch.device) -> None:
        generator = torch.Generator(device="cpu").manual_seed(20260914)
        scripted = torch.jit.script(_nlist_distinguish_types_topk)
        scripted_public = torch.jit.script(nlist_distinguish_types)
        shapes = [
            (1, 192, 138, 5184, [46, 92]),
            (2, 192, 138, 5184, [46, 92]),
            (1, 3, 7, 5, [3, 3, 1]),
            (2, 4, 17, 12, [0, 4, 13]),
            (1, 2, 6, 8, [10, 0, -1, -20]),
            (1, 1, 1, 3, [1, 1]),
            (1, 2, 0, 5, [0, 2]),
            (1, 0, 5, 4, [2, 3]),
            (0, 2, 5, 4, [2, 3]),
            (1, 2, 0, 0, [2, 3]),
        ]
        for nf, nloc, nnei, nall, sel in shapes:
            for variant in range(4):
                with self.subTest(shape=(nf, nloc, nnei, nall), variant=variant):
                    atype = torch.randint(
                        -1, len(sel) + 1, (nf, nall), generator=generator, device="cpu"
                    )
                    nlist = torch.randint(
                        -1,
                        max(nall, 1),
                        (nf, nloc, nnei),
                        generator=generator,
                        device="cpu",
                    )
                    if variant == 1:
                        nlist.fill_(-1)
                    elif variant == 2:
                        atype.fill_(0)
                    elif variant == 3:
                        atype.fill_(len(sel) + 1)
                    # Build the expected indices independently, keeping the
                    # input order and the old per-type slicing/padding rule.
                    width = sum(len(range(nnei)[:count]) for count in sel)
                    expected = torch.empty(
                        (nf, nloc, width), dtype=nlist.dtype, device="cpu"
                    )
                    tnlist = torch.empty_like(nlist)
                    for frame in range(nf):
                        for atom in range(nloc):
                            row = nlist[frame, atom].tolist()
                            types = [
                                int(atype[frame, idx]) if idx != -1 else -1
                                for idx in row
                            ]
                            tnlist[frame, atom] = torch.tensor(
                                types, dtype=tnlist.dtype, device="cpu"
                            )
                            grouped = []
                            for kind, count in enumerate(sel):
                                matched = [
                                    idx
                                    for idx, typ in zip(row, types, strict=True)
                                    if typ == kind
                                ]
                                grouped.extend(
                                    (matched + [-1] * (nnei - len(matched)))[:count]
                                )
                            expected[frame, atom] = torch.tensor(
                                grouped, dtype=nlist.dtype, device="cpu"
                            )
                    nlist = nlist.to(device)
                    atype = atype.to(device)
                    tnlist = tnlist.to(device)
                    expected = expected.to(device)
                    for function in (_nlist_distinguish_types_topk, scripted):
                        torch.testing.assert_close(
                            function(nlist, tnlist, sel), expected, rtol=0, atol=0
                        )
                    for function in (nlist_distinguish_types, scripted_public):
                        torch.testing.assert_close(
                            function(nlist, atype, sel), expected, rtol=0, atol=0
                        )


class TestNeighList(unittest.TestCase):
    def setUp(self) -> None:
        self.nf = 3
        self.nloc = 3
        self.ns = 5 * 5 * 3
        self.nall = self.ns * self.nloc
        self.cell = torch.tensor(
            [[1, 0, 0], [0.4, 0.8, 0], [0.1, 0.3, 2.1]], dtype=dtype, device=env.DEVICE
        )
        self.icoord = torch.tensor(
            [[0, 0, 0], [0, 0, 0], [0.5, 0.5, 0.1]], dtype=dtype, device=env.DEVICE
        )
        self.atype = torch.tensor([-1, 0, 1], dtype=torch.int, device=env.DEVICE)
        [self.cell, self.icoord, self.atype] = [
            ii.unsqueeze(0) for ii in [self.cell, self.icoord, self.atype]
        ]
        self.coord = inter2phys(self.icoord, self.cell).view([-1, self.nloc * 3])
        self.cell = self.cell.view([-1, 9])
        [self.cell, self.coord, self.atype] = [
            torch.tile(ii, [self.nf, 1]) for ii in [self.cell, self.coord, self.atype]
        ]
        self.rcut = 1.01
        self.prec = 1e-10
        self.nsel = [10, 10]
        # generated by preprocess.build_neighbor_list
        # ref_nlist, _, _ = legacy_build_neighbor_list(
        #   2, ecoord[0], eatype[0],
        #   self.rcut,
        #   torch.tensor([10,20], dtype=torch.long),
        #   mapping[0], type_split=True, )
        self.ref_nlist = torch.tensor(
            [
                [-1] * sum(self.nsel),
                [1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 2, 2, 2, 2, -1, -1, -1, -1, -1, -1],
                [1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 2, 2, 2, 2, 2, 2, -1, -1, -1, -1],
            ],
            device=env.DEVICE,
        )

    def test_build_notype(self) -> None:
        ecoord, eatype, mapping = extend_coord_with_ghosts(
            self.coord, self.atype, self.cell, self.rcut
        )
        # test normal sel
        nlist = build_neighbor_list(
            ecoord,
            eatype,
            self.nloc,
            self.rcut,
            sum(self.nsel),
            distinguish_types=False,
        )
        nlist_mask = nlist[0] == -1
        nlist_loc = mapping[0][nlist[0]]
        nlist_loc[nlist_mask] = -1
        torch.testing.assert_close(
            torch.sort(nlist_loc, dim=-1)[0],
            torch.sort(self.ref_nlist, dim=-1)[0],
        )
        # test a very large sel
        nlist = build_neighbor_list(
            ecoord,
            eatype,
            self.nloc,
            self.rcut,
            sum(self.nsel) + 300,  # +300, real nnei==224
            distinguish_types=False,
        )
        nlist_mask = nlist[0] == -1
        nlist_loc = mapping[0][nlist[0]]
        nlist_loc[nlist_mask] = -1
        torch.testing.assert_close(
            torch.sort(nlist_loc, descending=True, dim=-1)[0][:, : sum(self.nsel)],
            torch.sort(self.ref_nlist, descending=True, dim=-1)[0],
        )

    def test_build_type(self) -> None:
        ecoord, eatype, mapping = extend_coord_with_ghosts(
            self.coord, self.atype, self.cell, self.rcut
        )
        nlist = build_neighbor_list(
            ecoord,
            eatype,
            self.nloc,
            self.rcut,
            self.nsel,
            distinguish_types=True,
        )
        torch.testing.assert_close(nlist[0], nlist[1])
        nlist_mask = nlist[0] == -1
        nlist_loc = mapping[0][nlist[0]]
        nlist_loc[nlist_mask] = -1
        for ii in range(2):
            torch.testing.assert_close(
                torch.sort(torch.split(nlist_loc, self.nsel, dim=-1)[ii], dim=-1)[0],
                torch.sort(torch.split(self.ref_nlist, self.nsel, dim=-1)[ii], dim=-1)[
                    0
                ],
            )

    def test_build_multiple_nlist(self) -> None:
        rcuts = [1.01, 2.01]
        nsels = [20, 80]
        ecoord, eatype, mapping = extend_coord_with_ghosts(
            self.coord, self.atype, self.cell, max(rcuts)
        )
        nlist1 = build_neighbor_list(
            ecoord,
            eatype,
            self.nloc,
            rcuts[1],
            nsels[1] - 1,
            distinguish_types=False,
        )
        pad = -1 * torch.ones(
            [self.nf, self.nloc, 1], dtype=nlist1.dtype, device=nlist1.device
        )
        nlist2 = torch.cat([nlist1, pad], dim=-1)
        nlist0 = build_neighbor_list(
            ecoord,
            eatype,
            self.nloc,
            rcuts[0],
            nsels[0],
            distinguish_types=False,
        )
        nlists = build_multiple_neighbor_list(ecoord, nlist1, rcuts, nsels)
        for dd in range(2):
            self.assertEqual(
                nlists[get_multiple_nlist_key(rcuts[dd], nsels[dd])].shape[-1],
                nsels[dd],
            )

        # Since the nlist is created using unstable sort,
        # we check if the set of indices in the nlist matches,
        # regardless of the order
        torch.testing.assert_close(
            nlists[get_multiple_nlist_key(rcuts[0], nsels[0])].sort(dim=-1).values,
            nlist0.sort(dim=-1).values,
        )
        torch.testing.assert_close(
            nlists[get_multiple_nlist_key(rcuts[1], nsels[1])].sort(dim=-1).values,
            nlist2.sort(dim=-1).values,
        )

    def test_extend_coord(self) -> None:
        ecoord, eatype, mapping = extend_coord_with_ghosts(
            self.coord, self.atype, self.cell, self.rcut
        )
        # expected ncopy x nloc
        self.assertEqual(list(ecoord.shape), [self.nf, self.nall * 3])
        self.assertEqual(list(eatype.shape), [self.nf, self.nall])
        self.assertEqual(list(mapping.shape), [self.nf, self.nall])
        # check the nloc part is identical with original coord
        torch.testing.assert_close(
            ecoord[:, : self.nloc * 3], self.coord, rtol=self.prec, atol=self.prec
        )
        # check the shift vectors are aligned with grid
        shift_vec = (
            ecoord.view([-1, self.ns, self.nloc, 3])
            - self.coord.view([-1, self.nloc, 3])[:, None, :, :]
        )
        shift_vec = shift_vec.view([-1, self.nall, 3])
        # hack!!! assumes identical cell across frames
        shift_vec = torch.matmul(
            shift_vec, torch.linalg.inv(self.cell.view([self.nf, 3, 3])[0])
        )
        # nf x nall x 3
        shift_vec = torch.round(shift_vec)
        # check: identical shift vecs
        torch.testing.assert_close(
            shift_vec[0], shift_vec[1], rtol=self.prec, atol=self.prec
        )
        # check: shift idx aligned with grid
        mm, cc = torch.unique(shift_vec[0][:, 0], dim=-1, return_counts=True)
        torch.testing.assert_close(
            mm,
            torch.tensor([-2, -1, 0, 1, 2], dtype=dtype, device=env.DEVICE),
            rtol=self.prec,
            atol=self.prec,
        )
        torch.testing.assert_close(
            cc,
            torch.tensor(
                [self.ns * self.nloc // 5] * 5, dtype=torch.long, device=env.DEVICE
            ),
            rtol=self.prec,
            atol=self.prec,
        )
        mm, cc = torch.unique(shift_vec[1][:, 1], dim=-1, return_counts=True)
        torch.testing.assert_close(
            mm,
            torch.tensor([-2, -1, 0, 1, 2], dtype=dtype, device=env.DEVICE),
            rtol=self.prec,
            atol=self.prec,
        )
        torch.testing.assert_close(
            cc,
            torch.tensor(
                [self.ns * self.nloc // 5] * 5, dtype=torch.long, device=env.DEVICE
            ),
            rtol=self.prec,
            atol=self.prec,
        )
        mm, cc = torch.unique(shift_vec[1][:, 2], dim=-1, return_counts=True)
        torch.testing.assert_close(
            mm,
            torch.tensor([-1, 0, 1], dtype=dtype, device=env.DEVICE),
            rtol=self.prec,
            atol=self.prec,
        )
        torch.testing.assert_close(
            cc,
            torch.tensor(
                [self.ns * self.nloc // 3] * 3, dtype=torch.long, device=env.DEVICE
            ),
            rtol=self.prec,
            atol=self.prec,
        )

    def test_build_directional_nlist(self) -> None:
        """Directional nlist is tested against the standard nlist implementation."""
        ecoord, eatype, mapping = extend_coord_with_ghosts(
            self.coord, self.atype, self.cell, self.rcut
        )
        for distinguish_types, mysel in zip(
            [True, False], [sum(self.nsel), 300], strict=True
        ):
            # full neighbor list
            nlist_full = build_neighbor_list(
                ecoord,
                eatype,
                self.nloc,
                self.rcut,
                sum(self.nsel),
                distinguish_types=distinguish_types,
            )
            # central as part of the system
            nlist = build_directional_neighbor_list(
                ecoord[:, 3:6],
                eatype[:, 1:2],
                torch.concat(
                    [
                        ecoord[:, 0:3],
                        torch.zeros(
                            [self.nf, 3], dtype=dtype, device=env.DEVICE
                        ),  # placeholder
                        ecoord[:, 6:],
                    ],
                    dim=1,
                ),
                torch.concat(
                    [
                        eatype[:, 0:1],
                        -1
                        * torch.ones(
                            [self.nf, 1], dtype=int, device=env.DEVICE
                        ),  # placeholder
                        eatype[:, 2:],
                    ],
                    dim=1,
                ),
                self.rcut,
                mysel,
                distinguish_types=distinguish_types,
            )
            torch.testing.assert_close(nlist[0], nlist[1])
            torch.testing.assert_close(nlist[0], nlist[2])
            torch.testing.assert_close(
                torch.sort(nlist[0], descending=True, dim=-1)[0][:, : sum(self.nsel)],
                torch.sort(nlist_full[0][1:2], descending=True, dim=-1)[0],
            )
