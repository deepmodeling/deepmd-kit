# SPDX-License-Identifier: LGPL-3.0-or-later
import itertools
import unittest

import numpy as np
import torch

from deepmd.dpmodel.atomic_model import DPAtomicModel as DPDPAtomicModel
from deepmd.dpmodel.descriptor import DescrptSeA as DPDescrptSeA
from deepmd.dpmodel.fitting import InvarFitting as DPInvarFitting
from deepmd.pt.model.atomic_model import (
    DPAtomicModel,
)
from deepmd.pt.model.descriptor.se_a import (
    DescrptSeA,
)
from deepmd.pt.model.task.ener import (
    InvarFitting,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.utils import (
    to_numpy_array,
    to_torch_tensor,
)

from .test_env_mat import (
    TestCaseSingleFrameWithNlist,
    TestCaseSingleFrameWithNlistWithVirtual,
)

dtype = env.GLOBAL_PT_FLOAT_PRECISION


class TestDPAtomicModel(unittest.TestCase, TestCaseSingleFrameWithNlist):
    def setUp(self) -> None:
        TestCaseSingleFrameWithNlist.setUp(self)

    def test_self_consistency(self) -> None:
        nf, nloc, nnei = self.nlist.shape
        ds = DescrptSeA(
            self.rcut,
            self.rcut_smth,
            self.sel,
        ).to(env.DEVICE)
        ft = InvarFitting(
            "energy",
            self.nt,
            ds.get_dim_out(),
            1,
            mixed_types=ds.mixed_types(),
        ).to(env.DEVICE)
        type_map = ["foo", "bar"]

        # test the case of exclusion
        for atom_excl, pair_excl in itertools.product([[], [1]], [[], [[0, 1]]]):
            md0 = DPAtomicModel(
                ds,
                ft,
                type_map=type_map,
            ).to(env.DEVICE)
            md0.reinit_atom_exclude(atom_excl)
            md0.reinit_pair_exclude(pair_excl)
            md1 = DPAtomicModel.deserialize(md0.serialize()).to(env.DEVICE)
            args = [
                to_torch_tensor(ii)
                for ii in [self.coord_ext, self.atype_ext, self.nlist]
            ]
            ret0 = md0.forward_common_atomic(*args)
            ret1 = md1.forward_common_atomic(*args)
            np.testing.assert_allclose(
                to_numpy_array(ret0["energy"]),
                to_numpy_array(ret1["energy"]),
            )

    def test_dp_consistency(self) -> None:
        nf, nloc, nnei = self.nlist.shape
        ds = DPDescrptSeA(
            self.rcut,
            self.rcut_smth,
            self.sel,
        )
        ft = DPInvarFitting(
            "energy",
            self.nt,
            ds.get_dim_out(),
            1,
            mixed_types=ds.mixed_types(),
        )
        type_map = ["foo", "bar"]
        md0 = DPDPAtomicModel(ds, ft, type_map=type_map)
        md1 = DPAtomicModel.deserialize(md0.serialize()).to(env.DEVICE)
        args0 = [self.coord_ext, self.atype_ext, self.nlist]
        args1 = [
            to_torch_tensor(ii) for ii in [self.coord_ext, self.atype_ext, self.nlist]
        ]
        ret0 = md0.forward_common_atomic(*args0)
        ret1 = md1.forward_common_atomic(*args1)
        np.testing.assert_allclose(
            ret0["energy"],
            to_numpy_array(ret1["energy"]),
        )

    def test_jit(self) -> None:
        nf, nloc, nnei = self.nlist.shape
        ds = DescrptSeA(
            self.rcut,
            self.rcut_smth,
            self.sel,
        ).to(env.DEVICE)
        ft = InvarFitting(
            "energy",
            self.nt,
            ds.get_dim_out(),
            1,
            mixed_types=ds.mixed_types(),
        ).to(env.DEVICE)
        type_map = ["foo", "bar"]
        md0 = DPAtomicModel(ds, ft, type_map=type_map).to(env.DEVICE)
        md0 = torch.jit.script(md0)
        self.assertEqual(md0.get_rcut(), self.rcut)
        self.assertEqual(md0.get_type_map(), type_map)

    def test_excl_consistency(self) -> None:
        type_map = ["foo", "bar"]

        # test the case of exclusion
        for atom_excl, pair_excl in itertools.product([[], [1]], [[], [[0, 1]]]):
            ds = DescrptSeA(
                self.rcut,
                self.rcut_smth,
                self.sel,
            ).to(env.DEVICE)
            ft = InvarFitting(
                "energy",
                self.nt,
                ds.get_dim_out(),
                1,
                mixed_types=ds.mixed_types(),
            ).to(env.DEVICE)
            md0 = DPAtomicModel(
                ds,
                ft,
                type_map=type_map,
            ).to(env.DEVICE)
            md1 = DPAtomicModel.deserialize(md0.serialize()).to(env.DEVICE)

            md0.reinit_atom_exclude(atom_excl)
            md0.reinit_pair_exclude(pair_excl)
            # hacking!
            md1.descriptor.reinit_exclude(pair_excl)
            md1.fitting_net.reinit_exclude(atom_excl)

            # check energy consistency
            args = [
                to_torch_tensor(ii)
                for ii in [self.coord_ext, self.atype_ext, self.nlist]
            ]
            ret0 = md0.forward_common_atomic(*args)
            ret1 = md1.forward_common_atomic(*args)
            np.testing.assert_allclose(
                to_numpy_array(ret0["energy"]),
                to_numpy_array(ret1["energy"]),
            )

            # check output def
            out_names = [vv.name for vv in md0.atomic_output_def().get_data().values()]
            self.assertEqual(out_names, ["energy", "mask"])
            if atom_excl != []:
                for ii in md0.atomic_output_def().get_data().values():
                    if ii.name == "mask":
                        self.assertEqual(ii.shape, [1])
                        self.assertFalse(ii.reducible)
                        self.assertFalse(ii.r_differentiable)
                        self.assertFalse(ii.c_differentiable)

            # check mask
            if atom_excl == []:
                pass
            elif atom_excl == [1]:
                self.assertIn("mask", ret0.keys())
                expected = np.array([1, 1, 0], dtype=int)
                expected = np.concatenate(
                    [expected, expected[self.perm[: self.nloc]]]
                ).reshape(2, 3)
                np.testing.assert_array_equal(to_numpy_array(ret0["mask"]), expected)
            else:
                raise ValueError(f"not expected atom_excl {atom_excl}")


class TestDPAtomicModelVirtualConsistency(unittest.TestCase):
    def setUp(self) -> None:
        self.case0 = TestCaseSingleFrameWithNlist()
        self.case1 = TestCaseSingleFrameWithNlistWithVirtual()
        self.case0.setUp()
        self.case1.setUp()

    def test_virtual_consistency(self) -> None:
        nf, _, _ = self.case0.nlist.shape
        ds = DescrptSeA(
            self.case0.rcut,
            self.case0.rcut_smth,
            self.case0.sel,
        )
        ft = InvarFitting(
            "energy",
            self.case0.nt,
            ds.get_dim_out(),
            1,
            mixed_types=ds.mixed_types(),
        )
        type_map = ["foo", "bar"]
        md1 = DPAtomicModel(ds, ft, type_map=type_map).to(env.DEVICE)

        args0 = [self.case0.coord_ext, self.case0.atype_ext, self.case0.nlist]
        args0 = [to_torch_tensor(ii) for ii in args0]
        args1 = [self.case1.coord_ext, self.case1.atype_ext, self.case1.nlist]
        args1 = [to_torch_tensor(ii) for ii in args1]

        ret0 = md1.forward_common_atomic(*args0)
        ret1 = md1.forward_common_atomic(*args1)

        for dd in range(self.case0.nf):
            np.testing.assert_allclose(
                to_numpy_array(ret0["energy"])[dd],
                to_numpy_array(ret1["energy"])[dd, self.case1.get_real_mapping[dd], :],
            )
        expected_mask = np.array(
            [
                [1, 0, 1, 1],
                [1, 1, 0, 1],
            ]
        )
        np.testing.assert_equal(to_numpy_array(ret1["mask"]), expected_mask)
