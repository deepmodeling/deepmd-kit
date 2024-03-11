# SPDX-License-Identifier: LGPL-3.0-or-later
import json
import os
import unittest
from abc import (
    ABC,
    abstractmethod,
)
from pathlib import (
    Path,
)

import dpdata
import numpy as np
import torch

from deepmd.pt.model.descriptor import (
    DescrptSeA,
)
from deepmd.pt.model.descriptor.dpa1 import (
    DescrptDPA1,
)
from deepmd.pt.model.task.ener import (
    EnergyFittingNet,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.dataloader import (
    DpLoaderSet,
)
from deepmd.pt.utils.stat import make_stat_input as my_make
from deepmd.tf.common import (
    expand_sys_str,
)
from deepmd.tf.descriptor.se_a import DescrptSeA as DescrptSeA_tf
from deepmd.tf.descriptor.se_atten import DescrptSeAtten as DescrptSeAtten_tf
from deepmd.tf.fit.ener import (
    EnerFitting,
)
from deepmd.tf.model.model_stat import make_stat_input as dp_make
from deepmd.tf.model.model_stat import merge_sys_stat as dp_merge
from deepmd.tf.utils import random as tf_random
from deepmd.tf.utils.data_system import (
    DeepmdDataSystem,
)
from deepmd.utils.data import (
    DataRequirementItem,
)

CUR_DIR = os.path.dirname(__file__)

energy_data_requirement = [
    DataRequirementItem(
        "energy",
        ndof=1,
        atomic=False,
        must=False,
        high_prec=True,
    ),
    DataRequirementItem(
        "force",
        ndof=3,
        atomic=True,
        must=False,
        high_prec=False,
    ),
    DataRequirementItem(
        "virial",
        ndof=9,
        atomic=False,
        must=False,
        high_prec=False,
    ),
    DataRequirementItem(
        "atom_ener",
        ndof=1,
        atomic=True,
        must=False,
        high_prec=False,
    ),
    DataRequirementItem(
        "atom_pref",
        ndof=1,
        atomic=True,
        must=False,
        high_prec=False,
        repeat=3,
    ),
]


def compare(ut, base, given):
    if isinstance(base, list):
        ut.assertEqual(len(base), len(given))
        for idx in range(len(base)):
            compare(ut, base[idx], given[idx])
    elif isinstance(base, np.ndarray):
        ut.assertTrue(np.allclose(base.reshape(-1), given.reshape(-1)))
    else:
        ut.assertEqual(base, given)


class DatasetTest(ABC):
    @abstractmethod
    def setup_data(self):
        pass

    @abstractmethod
    def setup_tf(self):
        pass

    @abstractmethod
    def setup_pt(self):
        pass

    @abstractmethod
    def tf_compute_input_stats(self):
        pass

    def setUp(self):
        with open(str(Path(__file__).parent / "water/se_e2_a.json")) as fin:
            content = fin.read()
        config = json.loads(content)
        data_file = [self.setup_data()]

        config["training"]["training_data"]["systems"] = data_file
        config["training"]["validation_data"]["systems"] = data_file
        model_config = config["model"]
        self.rcut = model_config["descriptor"]["rcut"]
        self.rcut_smth = model_config["descriptor"]["rcut_smth"]
        self.sel = model_config["descriptor"]["sel"]
        self.batch_size = config["training"]["training_data"]["batch_size"]
        self.systems = config["training"]["validation_data"]["systems"]
        if isinstance(self.systems, str):
            self.systems = expand_sys_str(self.systems)
        self.my_dataset = DpLoaderSet(
            self.systems,
            self.batch_size,
            model_config["type_map"],
            seed=10,
        )
        self.filter_neuron = model_config["descriptor"]["neuron"]
        self.axis_neuron = model_config["descriptor"]["axis_neuron"]
        self.data_stat_nbatch = 2
        self.filter_neuron = model_config["descriptor"]["neuron"]
        self.axis_neuron = model_config["descriptor"]["axis_neuron"]
        self.n_neuron = model_config["fitting_net"]["neuron"]
        self.my_dataset.add_data_requirement(energy_data_requirement)

        self.my_sampled = my_make(
            self.my_dataset.systems, self.my_dataset.dataloaders, self.data_stat_nbatch
        )

        tf_random.seed(10)
        dp_dataset = DeepmdDataSystem(self.systems, self.batch_size, 1, self.rcut)
        dp_dataset.add("energy", 1, atomic=False, must=False, high_prec=True)
        dp_dataset.add("force", 3, atomic=True, must=False, high_prec=False)
        self.dp_sampled = dp_make(dp_dataset, self.data_stat_nbatch, False)
        self.dp_merged = dp_merge(self.dp_sampled)
        self.dp_mesh = self.dp_merged.pop("default_mesh")
        self.dp_d = self.setup_tf()

    def test_stat_output(self):
        def my_merge(energy, natoms):
            energy_lst = []
            natoms_lst = []
            for i in range(len(energy)):
                for j in range(len(energy[i])):
                    energy_lst.append(torch.tensor(energy[i][j], device="cpu"))
                    natoms_lst.append(
                        torch.tensor(natoms[i][j], device="cpu")
                        .unsqueeze(0)
                        .expand(energy[i][j].shape[0], -1)
                    )
            energy_merge = torch.cat(energy_lst)
            natoms_merge = torch.cat(natoms_lst)
            return energy_merge, natoms_merge

        energy = self.dp_sampled["energy"]
        natoms = self.dp_sampled["natoms_vec"]
        energy, natoms = my_merge(energy, natoms)
        dp_fn = EnerFitting(
            self.dp_d.get_ntypes(), self.dp_d.get_dim_out(), self.n_neuron
        )
        dp_fn.compute_output_stats(self.dp_sampled, mixed_type=self.mixed_type)
        pt_fn = EnergyFittingNet(
            self.dp_d.get_ntypes(), self.dp_d.get_dim_out(), self.n_neuron
        )
        pt_fn.compute_output_stats(self.my_sampled)
        np.testing.assert_allclose(
            dp_fn.bias_atom_e, pt_fn.bias_atom_e.detach().cpu().numpy().ravel()
        )

    # temporarily delete this function for performance of seeds in tf and pytorch may be different
    """
    def test_stat_input(self):
        my_sampled = self.my_sampled
        # list of dicts, each dict contains samples from a system
        dp_keys = set(self.dp_merged.keys()) # dict of list of batches
        self.dp_merged['natoms'] = self.dp_merged['natoms_vec']
        for key in dp_keys:
            if not key in my_sampled[0] or key in 'coord':
                # coord is pre-normalized
                continue
            lst = []
            for item in my_sampled:
                bsz = item['energy'].shape[0]//self.data_stat_nbatch
                for j in range(self.data_stat_nbatch):
                    lst.append(item[key][j*bsz:(j+1)*bsz].cpu().numpy())
                compare(self, self.dp_merged[key], lst)
    """

    def test_descriptor(self):
        self.tf_compute_input_stats()

        my_en = self.setup_pt()
        sampled = self.my_sampled
        for sys in sampled:
            for key in [
                "coord",
                "atype",
                "natoms",
                "box",
            ]:
                if key in sys.keys():
                    sys[key] = sys[key].to(env.DEVICE)
        stat_dict = my_en.compute_input_stats(sampled)
        my_en.mean = my_en.mean
        my_en.stddev = my_en.stddev
        np.testing.assert_allclose(
            self.dp_d.davg.reshape([-1]),
            my_en.mean.cpu().reshape([-1]),
            rtol=1e-14,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            self.dp_d.dstd.reshape([-1]),
            my_en.stddev.cpu().reshape([-1]),
            rtol=1e-14,
            atol=1e-14,
        )


class TestDatasetNoMixed(DatasetTest, unittest.TestCase):
    def setup_data(self):
        original_data = str(Path(__file__).parent / "water/data/data_0")
        picked_data = str(Path(__file__).parent / "picked_data_for_test_stat")
        dpdata.LabeledSystem(original_data, fmt="deepmd/npy")[:2].to_deepmd_npy(
            picked_data
        )
        self.mixed_type = False
        return picked_data

    def setup_tf(self):
        return DescrptSeA_tf(
            rcut=self.rcut,
            rcut_smth=self.rcut_smth,
            sel=self.sel,
            neuron=self.filter_neuron,
            axis_neuron=self.axis_neuron,
        )

    def setup_pt(self):
        return DescrptSeA(
            self.rcut, self.rcut_smth, self.sel, self.filter_neuron, self.axis_neuron
        ).sea  # get the block who has stat as private vars

    def tf_compute_input_stats(self):
        coord = self.dp_merged["coord"]
        atype = self.dp_merged["type"]
        natoms = self.dp_merged["natoms_vec"]
        box = self.dp_merged["box"]
        self.dp_d.compute_input_stats(coord, box, atype, natoms, self.dp_mesh, {})


class TestDatasetMixed(DatasetTest, unittest.TestCase):
    def setup_data(self):
        original_data = str(Path(__file__).parent / "water/data/data_0")
        picked_data = str(Path(__file__).parent / "picked_data_for_test_stat")
        dpdata.LabeledSystem(original_data, fmt="deepmd/npy")[:2].to_deepmd_npy_mixed(
            picked_data
        )
        self.mixed_type = True
        return picked_data

    def setup_tf(self):
        return DescrptSeAtten_tf(
            ntypes=2,
            rcut=self.rcut,
            rcut_smth=self.rcut_smth,
            sel=sum(self.sel),
            neuron=self.filter_neuron,
            axis_neuron=self.axis_neuron,
            set_davg_zero=False,
        )

    def setup_pt(self):
        return DescrptDPA1(
            self.rcut,
            self.rcut_smth,
            sum(self.sel),
            2,
            self.filter_neuron,
            self.axis_neuron,
            set_davg_zero=False,
        ).se_atten

    def tf_compute_input_stats(self):
        coord = self.dp_merged["coord"]
        atype = self.dp_merged["type"]
        natoms = self.dp_merged["natoms_vec"]
        box = self.dp_merged["box"]
        real_natoms_vec = self.dp_merged["real_natoms_vec"]

        self.dp_d.compute_input_stats(
            coord,
            box,
            atype,
            natoms,
            self.dp_mesh,
            {},
            mixed_type=True,
            real_natoms_vec=real_natoms_vec,
        )


if __name__ == "__main__":
    unittest.main()
