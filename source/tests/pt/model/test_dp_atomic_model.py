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
)

dtype = env.GLOBAL_PT_FLOAT_PRECISION


class TestDPAtomicModel(unittest.TestCase, TestCaseSingleFrameWithNlist):
    def setUp(self):
        TestCaseSingleFrameWithNlist.setUp(self)

    def test_self_consistency(self):
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

    def test_dp_consistency(self):
        rng = np.random.default_rng()
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

    def test_jit(self):
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

    def test_excl_consistency(self):
        nf, nloc, nnei = self.nlist.shape
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
