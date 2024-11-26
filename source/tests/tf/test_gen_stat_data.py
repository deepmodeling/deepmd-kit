# SPDX-License-Identifier: LGPL-3.0-or-later
import shutil
import unittest

import dpdata
import numpy as np

from deepmd.tf.descriptor import (
    DescrptSeA,
)
from deepmd.tf.fit import (
    EnerFitting,
)
from deepmd.tf.model.model_stat import (
    _make_all_stat_ref,
    make_stat_input,
    merge_sys_stat,
)
from deepmd.tf.utils import random as dp_random
from deepmd.tf.utils.data_system import (
    DeepmdDataSystem,
)

from ..seed import (
    GLOBAL_SEED,
)


def gen_sys(nframes, atom_types):
    rng = np.random.default_rng(GLOBAL_SEED)
    natoms = len(atom_types)
    data = {}
    data["coords"] = rng.random([nframes, natoms, 3])
    data["forces"] = rng.random([nframes, natoms, 3])
    data["cells"] = rng.random([nframes, 9])
    data["energies"] = rng.random([nframes, 1])
    types = list(set(atom_types))
    types.sort()
    data["atom_names"] = []
    data["atom_numbs"] = []
    for ii in range(len(types)):
        data["atom_names"].append(f"TYPE_{ii}")
        data["atom_numbs"].append(np.sum(atom_types == ii))
    data["atom_types"] = np.array(atom_types, dtype=int)
    return data


class TestGenStatData(unittest.TestCase):
    def setUp(self) -> None:
        data0 = gen_sys(20, [0, 1, 0, 2, 1])
        data1 = gen_sys(30, [0, 1, 0, 0])
        sys0 = dpdata.LabeledSystem()
        sys1 = dpdata.LabeledSystem()
        sys0.data = data0
        sys1.data = data1
        sys0.to_deepmd_npy("system_0", set_size=10)
        sys1.to_deepmd_npy("system_1", set_size=10)

    def tearDown(self) -> None:
        shutil.rmtree("system_0")
        shutil.rmtree("system_1")

    def _comp_data(self, d0, d1) -> None:
        np.testing.assert_almost_equal(d0, d1)

    def test_merge_all_stat(self) -> None:
        dp_random.seed(0)
        data0 = DeepmdDataSystem(["system_0", "system_1"], 5, 10, 1.0)
        data0.add("energy", 1, must=True)
        dp_random.seed(0)
        data1 = DeepmdDataSystem(["system_0", "system_1"], 5, 10, 1.0)
        data1.add("energy", 1, must=True)
        dp_random.seed(0)
        data2 = DeepmdDataSystem(["system_0", "system_1"], 5, 10, 1.0)
        data2.add("energy", 1, must=True)

        dp_random.seed(0)
        all_stat_0 = make_stat_input(data0, 10, merge_sys=False)
        dp_random.seed(0)
        all_stat_1 = make_stat_input(data1, 10, merge_sys=True)
        all_stat_2 = merge_sys_stat(all_stat_0)
        dp_random.seed(0)
        all_stat_3 = _make_all_stat_ref(data2, 10)

        ####################################
        # only check if the energy is concatenated correctly
        ####################################
        dd = "energy"
        # if 'find_' in dd: continue
        # if 'natoms_vec' in dd: continue
        # if 'default_mesh' in dd: continue
        # print(all_stat_2[dd])
        # print(dd, all_stat_1[dd])
        d1 = np.array(all_stat_1[dd])
        d2 = np.array(all_stat_2[dd])
        d3 = np.array(all_stat_3[dd])
        # print(dd)
        # print(d1.shape)
        # print(d2.shape)
        # self.assertEqual(all_stat_2[dd], all_stat_1[dd])
        self._comp_data(d1, d2)
        self._comp_data(d1, d3)


class TestEnerShift(unittest.TestCase):
    def setUp(self) -> None:
        data0 = gen_sys(30, [0, 1, 0, 2, 1])
        data1 = gen_sys(30, [0, 1, 0, 0])
        sys0 = dpdata.LabeledSystem()
        sys1 = dpdata.LabeledSystem()
        sys0.data = data0
        sys1.data = data1
        sys0.to_deepmd_npy("system_0", set_size=10)
        sys1.to_deepmd_npy("system_1", set_size=10)

    def tearDown(self) -> None:
        shutil.rmtree("system_0")
        shutil.rmtree("system_1")

    def test_ener_shift(self) -> None:
        dp_random.seed(0)
        data = DeepmdDataSystem(["system_0", "system_1"], 5, 10, 1.0)
        data.add("energy", 1, must=True)
        ener_shift0 = data.compute_energy_shift(rcond=1)
        all_stat = make_stat_input(data, 6, merge_sys=False)
        descrpt = DescrptSeA(6.0, 5.8, [46, 92], neuron=[25, 50, 100], axis_neuron=16)
        fitting = EnerFitting(
            descrpt.get_ntypes(),
            descrpt.get_dim_out(),
            neuron=[240, 240, 240],
            resnet_dt=True,
        )
        ener_shift1 = fitting._compute_output_stats(all_stat, rcond=1)
        np.testing.assert_almost_equal(ener_shift0, ener_shift1)

    def test_ener_shift_assigned(self) -> None:
        dp_random.seed(0)
        ae0 = dp_random.random()
        data = DeepmdDataSystem(["system_0"], 5, 10, 1.0)
        data.add("energy", 1, must=True)
        all_stat = make_stat_input(data, 6, merge_sys=False)
        descrpt = DescrptSeA(6.0, 5.8, [46, 92], neuron=[25, 50, 100], axis_neuron=16)
        fitting = EnerFitting(
            descrpt.get_ntypes(),
            descrpt.get_dim_out(),
            neuron=[240, 240, 240],
            resnet_dt=True,
            atom_ener=[ae0, None, None],
        )
        ener_shift1 = fitting._compute_output_stats(all_stat, rcond=1)
        # check assigned energy
        np.testing.assert_almost_equal(ae0, ener_shift1[0])
        # check if total energy are the same
        natoms = data.natoms_vec[0][2:]
        tot0 = np.dot(data.compute_energy_shift(rcond=1), natoms)
        tot1 = np.dot(ener_shift1, natoms)
        np.testing.assert_almost_equal(tot0, tot1)
