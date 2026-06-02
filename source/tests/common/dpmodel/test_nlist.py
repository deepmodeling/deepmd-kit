# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from importlib.util import (
    find_spec,
)

import numpy as np

from deepmd.dpmodel.descriptor import (
    DescrptSeA,
)
from deepmd.dpmodel.fitting import (
    InvarFitting,
)
from deepmd.dpmodel.model.ener_model import (
    EnergyModel,
)
from deepmd.dpmodel.utils import (
    build_multiple_neighbor_list,
    build_neighbor_list,
    extend_coord_with_ghosts,
    get_multiple_nlist_key,
    inter2phys,
)
from deepmd.dpmodel.utils.nlist import (
    build_neighbor_list_vesin,
    extend_input_and_build_neighbor_list,
    is_vesin_available,
)


class TestDPModelFormatNlist(unittest.TestCase):
    def setUp(self) -> None:
        # nloc == 3, nall == 4
        self.nloc = 3
        self.nall = 5
        self.nf, self.nt = 1, 2
        self.coord_ext = np.array(
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0, -2, 0],
                [2.3, 0, 0],
            ],
            dtype=np.float64,
        ).reshape([1, self.nall * 3])
        # sel = [5, 2]
        self.sel = [5, 2]
        self.expected_nlist = np.array(
            [
                [1, 3, -1, -1, -1, 2, -1],
                [0, -1, -1, -1, -1, 2, -1],
                [0, 1, -1, -1, -1, -1, -1],
            ],
            dtype=int,
        ).reshape([1, self.nloc, sum(self.sel)])
        self.atype_ext = np.array([0, 0, 1, 0, 1], dtype=int).reshape([1, self.nall])
        self.rcut_smth = 0.4
        self.rcut = 2.1

        nf, nloc, nnei = self.expected_nlist.shape
        ds = DescrptSeA(
            self.rcut,
            self.rcut_smth,
            self.sel,
        )
        ft = InvarFitting(
            "energy",
            self.nt,
            ds.get_dim_out(),
            1,
            mixed_types=ds.mixed_types(),
        )
        type_map = ["foo", "bar"]
        self.md = EnergyModel(ds, ft, type_map=type_map)

    def test_nlist_eq(self) -> None:
        # n_nnei == nnei
        nlist = np.array(
            [
                [1, 3, -1, -1, -1, 2, -1],
                [0, -1, -1, -1, -1, 2, -1],
                [0, 1, -1, -1, -1, -1, -1],
            ],
            dtype=np.int64,
        ).reshape([1, self.nloc, -1])
        nlist1 = self.md.format_nlist(
            self.coord_ext,
            self.atype_ext,
            nlist,
        )
        np.testing.assert_allclose(self.expected_nlist, nlist1)

    def test_nlist_st(self) -> None:
        # n_nnei < nnei
        nlist = np.array(
            [
                [1, 3, -1, 2],
                [0, -1, -1, 2],
                [0, 1, -1, -1],
            ],
            dtype=np.int64,
        ).reshape([1, self.nloc, -1])
        nlist1 = self.md.format_nlist(
            self.coord_ext,
            self.atype_ext,
            nlist,
        )
        np.testing.assert_allclose(self.expected_nlist, nlist1)

    def test_nlist_lt(self) -> None:
        # n_nnei > nnei
        nlist = np.array(
            [
                [1, 3, -1, -1, -1, 2, -1, -1, 4],
                [0, -1, 4, -1, -1, 2, -1, 3, -1],
                [0, 1, -1, -1, -1, 4, -1, -1, 3],
            ],
            dtype=np.int64,
        ).reshape([1, self.nloc, -1])
        nlist1 = self.md.format_nlist(
            self.coord_ext,
            self.atype_ext,
            nlist,
        )
        np.testing.assert_allclose(self.expected_nlist, nlist1)


dtype = np.float64


class TestNeighList(unittest.TestCase):
    def setUp(self) -> None:
        self.nf = 3
        self.nloc = 3
        self.ns = 5 * 5 * 3
        self.nall = self.ns * self.nloc
        self.cell = np.array([[1, 0, 0], [0.4, 0.8, 0], [0.1, 0.3, 2.1]], dtype=dtype)
        self.icoord = np.array([[0, 0, 0], [0, 0, 0], [0.5, 0.5, 0.1]], dtype=dtype)
        self.atype = np.array([-1, 0, 1], dtype=np.int32)
        [self.cell, self.icoord, self.atype] = [
            np.expand_dims(ii, 0) for ii in [self.cell, self.icoord, self.atype]
        ]
        self.coord = inter2phys(self.icoord, self.cell).reshape([-1, self.nloc * 3])
        self.cell = self.cell.reshape([-1, 9])
        [self.cell, self.coord, self.atype] = [
            np.tile(ii, [self.nf, 1]) for ii in [self.cell, self.coord, self.atype]
        ]
        self.rcut = 1.01
        self.prec = 1e-10
        self.nsel = [10, 10]
        self.ref_nlist = np.array(
            [
                [-1] * sum(self.nsel),
                [1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 2, 2, 2, 2, -1, -1, -1, -1, -1, -1],
                [1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 2, 2, 2, 2, 2, 2, -1, -1, -1, -1],
            ]
        )

    def test_build_notype(self) -> None:
        ecoord, eatype, mapping = extend_coord_with_ghosts(
            self.coord, self.atype, self.cell, self.rcut
        )
        nlist = build_neighbor_list(
            ecoord,
            eatype,
            self.nloc,
            self.rcut,
            sum(self.nsel),
            distinguish_types=False,
        )
        np.testing.assert_allclose(nlist[0], nlist[1])
        nlist_mask = nlist[0] == -1
        nlist_loc = mapping[0][nlist[0]]
        nlist_loc[nlist_mask] = -1
        np.testing.assert_allclose(
            np.sort(nlist_loc, axis=-1),
            np.sort(self.ref_nlist, axis=-1),
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
        np.testing.assert_allclose(nlist[0], nlist[1])
        nlist_mask = nlist[0] == -1
        nlist_loc = mapping[0][nlist[0]]
        nlist_loc[nlist_mask] = -1
        for ii in range(2):
            np.testing.assert_allclose(
                np.sort(np.split(nlist_loc, self.nsel, axis=-1)[ii], axis=-1),
                np.sort(np.split(self.ref_nlist, self.nsel, axis=-1)[ii], axis=-1),
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
        pad = -1 * np.ones([self.nf, self.nloc, 1], dtype=nlist1.dtype)
        nlist2 = np.concatenate([nlist1, pad], axis=-1)
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
        np.testing.assert_allclose(
            nlists[get_multiple_nlist_key(rcuts[0], nsels[0])],
            nlist0,
        )
        np.testing.assert_allclose(
            nlists[get_multiple_nlist_key(rcuts[1], nsels[1])],
            nlist2,
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
        np.testing.assert_allclose(
            ecoord[:, : self.nloc * 3], self.coord, rtol=self.prec, atol=self.prec
        )
        # check the shift vectors are aligned with grid
        shift_vec = (
            ecoord.reshape([-1, self.ns, self.nloc, 3])
            - self.coord.reshape([-1, self.nloc, 3])[:, None, :, :]
        )
        shift_vec = shift_vec.reshape([-1, self.nall, 3])
        # hack!!! assumes identical cell across frames
        shift_vec = np.matmul(
            shift_vec, np.linalg.inv(self.cell.reshape([self.nf, 3, 3])[0])
        )
        # nf x nall x 3
        shift_vec = np.round(shift_vec)
        # check: identical shift vecs
        np.testing.assert_allclose(
            shift_vec[0], shift_vec[1], rtol=self.prec, atol=self.prec
        )
        # check: shift idx aligned with grid
        mm, cc = np.unique(shift_vec[0][:, 0], return_counts=True)
        np.testing.assert_allclose(
            mm,
            np.array([-2, -1, 0, 1, 2], dtype=dtype),
            rtol=self.prec,
            atol=self.prec,
        )
        np.testing.assert_allclose(
            cc,
            np.array([self.ns * self.nloc // 5] * 5, dtype=np.int32),
            rtol=self.prec,
            atol=self.prec,
        )
        mm, cc = np.unique(shift_vec[1][:, 1], return_counts=True)
        np.testing.assert_allclose(
            mm,
            np.array([-2, -1, 0, 1, 2], dtype=dtype),
            rtol=self.prec,
            atol=self.prec,
        )
        np.testing.assert_allclose(
            cc,
            np.array([self.ns * self.nloc // 5] * 5, dtype=np.int32),
            rtol=self.prec,
            atol=self.prec,
        )
        mm, cc = np.unique(shift_vec[1][:, 2], return_counts=True)
        np.testing.assert_allclose(
            mm,
            np.array([-1, 0, 1], dtype=dtype),
            rtol=self.prec,
            atol=self.prec,
        )
        np.testing.assert_allclose(
            cc,
            np.array([self.ns * self.nloc // 3] * 3, dtype=np.int32),
            rtol=self.prec,
            atol=self.prec,
        )

    @unittest.skipIf(find_spec("jax") is None, "JAX is not installed")
    def test_extend_coord_jax_matches_numpy(self) -> None:
        import jax.numpy as jnp

        ecoord_np, eatype_np, mapping_np = extend_coord_with_ghosts(
            self.coord, self.atype, self.cell, self.rcut
        )
        ecoord_jax, eatype_jax, mapping_jax = extend_coord_with_ghosts(
            jnp.asarray(self.coord),
            jnp.asarray(self.atype),
            jnp.asarray(self.cell),
            self.rcut,
        )

        np.testing.assert_allclose(np.asarray(ecoord_jax), ecoord_np, atol=1e-6)
        np.testing.assert_array_equal(np.asarray(eatype_jax), eatype_np)
        np.testing.assert_array_equal(np.asarray(mapping_jax), mapping_np)


def _per_atom_neighbor_dists(ext_coord, nlist, coord):
    """Sorted, rounded valid-neighbor distances for each local atom."""
    ext_coord = np.asarray(ext_coord).reshape(-1, 3)
    coord = np.asarray(coord).reshape(-1, 3)
    out = []
    for i in range(coord.shape[0]):
        ds = [
            round(float(np.linalg.norm(ext_coord[j] - coord[i])), 6)
            for j in nlist[i]
            if j >= 0
        ]
        out.append(sorted(ds))
    return out


@unittest.skipIf(not is_vesin_available(), "vesin is not installed")
class TestNeighListVesin(unittest.TestCase):
    """The O(N) ``vesin`` builder must produce the same neighbor relationships
    as the native all-pairs builder on the inference path.
    """

    def setUp(self) -> None:
        rng = np.random.default_rng(20240602)
        self.nloc = 40
        self.rcut = 2.5
        self.sel = [30, 30]
        self.box_len = 6.0
        self.box = (np.eye(3) * self.box_len).reshape(1, 9)
        self.coord = (rng.random((self.nloc, 3)) * self.box_len).reshape(
            1, self.nloc, 3
        )
        self.atype = rng.integers(0, 2, self.nloc).reshape(1, self.nloc)

    def _native(self, mixed_types, box):
        ec, ea, mp, nl = extend_input_and_build_neighbor_list(
            self.coord,
            self.atype,
            self.rcut,
            self.sel,
            mixed_types=mixed_types,
            box=box,
        )
        return np.asarray(ec).reshape(-1, 3), np.asarray(nl)[0]

    def test_pbc_matches_native_mixed(self) -> None:
        ec_n, nl_n = self._native(mixed_types=True, box=self.box)
        ec_v, ea_v, nl_v, mp_v = build_neighbor_list_vesin(
            self.coord, self.box, self.atype, self.rcut, self.sel, False
        )
        self.assertEqual(
            _per_atom_neighbor_dists(ec_n, nl_n, self.coord[0]),
            _per_atom_neighbor_dists(ec_v[0], nl_v[0], self.coord[0]),
        )
        # far fewer ghosts than the 27x tiling of the native builder
        self.assertLess(ec_v.shape[1], ec_n.shape[0])

    def test_nopbc_matches_native_mixed(self) -> None:
        ec_n, nl_n = self._native(mixed_types=True, box=None)
        ec_v, ea_v, nl_v, mp_v = build_neighbor_list_vesin(
            self.coord, None, self.atype, self.rcut, self.sel, False
        )
        self.assertEqual(
            _per_atom_neighbor_dists(ec_n, nl_n, self.coord[0]),
            _per_atom_neighbor_dists(ec_v[0], nl_v[0], self.coord[0]),
        )
        # no periodic images -> no ghosts
        self.assertEqual(ec_v.shape[1], self.nloc)

    def test_distinguish_types_matches_native(self) -> None:
        # large sel so no per-type truncation differs from the mixed list
        ec_n, nl_n = self._native(mixed_types=False, box=self.box)
        ec_v, ea_v, nl_v, mp_v = build_neighbor_list_vesin(
            self.coord, self.box, self.atype, self.rcut, self.sel, True
        )
        self.assertEqual(
            _per_atom_neighbor_dists(ec_n, nl_n, self.coord[0]),
            _per_atom_neighbor_dists(ec_v[0], nl_v[0], self.coord[0]),
        )

    def test_extended_coord_mapping_consistency(self) -> None:
        ec_v, ea_v, nl_v, mp_v = build_neighbor_list_vesin(
            self.coord, self.box, self.atype, self.rcut, self.sel, False
        )
        ec = ec_v[0]
        nl = nl_v[0]
        mp = mp_v[0]
        # real atoms map to themselves
        np.testing.assert_array_equal(mp[: self.nloc], np.arange(self.nloc))
        # every listed neighbor is within rcut and its type matches its owner
        for i in range(self.nloc):
            for j in nl[i]:
                if j >= 0:
                    self.assertLessEqual(
                        float(np.linalg.norm(ec[j] - self.coord[0, i])),
                        self.rcut + 1e-9,
                    )
                    self.assertEqual(ea_v[0, j], self.atype[0, mp[j]])

    def test_multiframe(self) -> None:
        coord2 = np.concatenate([self.coord, self.coord + 0.1], axis=0)
        atype2 = np.concatenate([self.atype, self.atype], axis=0)
        box2 = np.concatenate([self.box, self.box], axis=0)
        ec_v, ea_v, nl_v, mp_v = build_neighbor_list_vesin(
            coord2, box2, atype2, self.rcut, self.sel, False
        )
        self.assertEqual(ec_v.shape[0], 2)
        self.assertEqual(nl_v.shape, (2, self.nloc, sum(self.sel)))

    def test_isolated_atoms_have_no_neighbors(self) -> None:
        # tiny cutoff on a sparse box -> no neighbors at all
        ec_v, ea_v, nl_v, mp_v = build_neighbor_list_vesin(
            self.coord, self.box, self.atype, 0.01, self.sel, False
        )
        self.assertTrue(np.all(nl_v == -1))
        self.assertEqual(ec_v.shape[1], self.nloc)
