# SPDX-License-Identifier: LGPL-3.0-or-later
import os
import unittest

import numpy as np
import tensorflow.compat.v1 as tf
import torch

tf.disable_eager_execution()
from copy import (
    deepcopy,
)
from pathlib import (
    Path,
)

from deepmd.pt.loss import (
    EnergySpinLoss,
    EnergyStdLoss,
)
from deepmd.pt.utils.dataset import (
    DeepmdDataSetForLoader,
)
from deepmd.tf.loss.ener import (
    EnerSpinLoss,
    EnerStdLoss,
)
from deepmd.utils.data import (
    DataRequirementItem,
)

from .model.test_embedding_net import (
    get_single_batch,
)
from .test_finetune import (
    energy_data_requirement,
)

CUR_DIR = os.path.dirname(__file__)


def get_batch(system, type_map, data_requirement):
    dataset = DeepmdDataSetForLoader(system, type_map)
    dataset.add_data_requirement(data_requirement)
    np_batch, pt_batch = get_single_batch(dataset)
    return np_batch, pt_batch


class TestEnerStdLoss(unittest.TestCase):
    def setUp(self):
        self.system = str(Path(__file__).parent / "water/data/data_0")
        self.type_map = ["H", "O"]
        self.start_lr = 1.1
        self.start_pref_e = 0.02
        self.limit_pref_e = 1.0
        self.start_pref_f = 1000.0
        self.limit_pref_f = 1.0
        self.start_pref_v = 0.02
        self.limit_pref_v = 1.0
        self.cur_lr = 1.2
        # data
        np_batch, pt_batch = get_batch(
            self.system, self.type_map, energy_data_requirement
        )
        natoms = np_batch["natoms"]
        self.nloc = natoms[0]
        l_energy, l_force, l_virial = (
            np_batch["energy"],
            np_batch["force"],
            np_batch["virial"],
        )
        p_energy, p_force, p_virial = (
            np.ones_like(l_energy),
            np.ones_like(l_force),
            np.ones_like(l_virial),
        )
        nloc = natoms[0]
        batch_size = pt_batch["coord"].shape[0]
        atom_energy = np.zeros(shape=[batch_size, nloc])
        atom_pref = np.zeros(shape=[batch_size, nloc * 3])
        # tf
        base = EnerStdLoss(
            self.start_lr,
            self.start_pref_e,
            self.limit_pref_e,
            self.start_pref_f,
            self.limit_pref_f,
            self.start_pref_v,
            self.limit_pref_v,
        )
        self.g = tf.Graph()
        with self.g.as_default():
            t_cur_lr = tf.placeholder(shape=[], dtype=tf.float64)
            t_natoms = tf.placeholder(shape=[None], dtype=tf.int32)
            t_penergy = tf.placeholder(shape=[None, 1], dtype=tf.float64)
            t_pforce = tf.placeholder(shape=[None, None], dtype=tf.float64)
            t_pvirial = tf.placeholder(shape=[None, 9], dtype=tf.float64)
            t_patom_energy = tf.placeholder(shape=[None, None], dtype=tf.float64)
            t_lenergy = tf.placeholder(shape=[None, 1], dtype=tf.float64)
            t_lforce = tf.placeholder(shape=[None, None], dtype=tf.float64)
            t_lvirial = tf.placeholder(shape=[None, 9], dtype=tf.float64)
            t_latom_energy = tf.placeholder(shape=[None, None], dtype=tf.float64)
            t_atom_pref = tf.placeholder(shape=[None, None], dtype=tf.float64)
            find_energy = tf.constant(1.0, dtype=tf.float64)
            find_force = tf.constant(1.0, dtype=tf.float64)
            find_virial = tf.constant(1.0, dtype=tf.float64)
            find_atom_energy = tf.constant(0.0, dtype=tf.float64)
            find_atom_pref = tf.constant(0.0, dtype=tf.float64)
            model_dict = {
                "energy": t_penergy,
                "force": t_pforce,
                "virial": t_pvirial,
                "atom_ener": t_patom_energy,
            }
            label_dict = {
                "energy": t_lenergy,
                "force": t_lforce,
                "virial": t_lvirial,
                "atom_ener": t_latom_energy,
                "atom_pref": t_atom_pref,
                "find_energy": find_energy,
                "find_force": find_force,
                "find_virial": find_virial,
                "find_atom_ener": find_atom_energy,
                "find_atom_pref": find_atom_pref,
            }
            self.base_loss_sess = base.build(
                t_cur_lr, t_natoms, model_dict, label_dict, ""
            )
        # torch
        self.feed_dict = {
            t_cur_lr: self.cur_lr,
            t_natoms: natoms,
            t_penergy: p_energy,
            t_pforce: p_force,
            t_pvirial: p_virial.reshape(-1, 9),
            t_patom_energy: atom_energy,
            t_lenergy: l_energy,
            t_lforce: l_force,
            t_lvirial: l_virial.reshape(-1, 9),
            t_latom_energy: atom_energy,
            t_atom_pref: atom_pref,
        }
        self.model_pred = {
            "energy": torch.from_numpy(p_energy),
            "force": torch.from_numpy(p_force),
            "virial": torch.from_numpy(p_virial),
        }
        self.label = {
            "energy": torch.from_numpy(l_energy),
            "find_energy": 1.0,
            "force": torch.from_numpy(l_force),
            "find_force": 1.0,
            "virial": torch.from_numpy(l_virial),
            "find_virial": 1.0,
        }
        self.label_absent = {
            "energy": torch.from_numpy(l_energy),
            "force": torch.from_numpy(l_force),
            "virial": torch.from_numpy(l_virial),
        }
        self.natoms = pt_batch["natoms"]

    def tearDown(self) -> None:
        tf.reset_default_graph()
        return super().tearDown()

    def test_consistency(self):
        with tf.Session(graph=self.g) as sess:
            base_loss, base_more_loss = sess.run(
                self.base_loss_sess, feed_dict=self.feed_dict
            )
        mine = EnergyStdLoss(
            self.start_lr,
            self.start_pref_e,
            self.limit_pref_e,
            self.start_pref_f,
            self.limit_pref_f,
            self.start_pref_v,
            self.limit_pref_v,
        )

        def fake_model():
            return self.model_pred

        _, my_loss, my_more_loss = mine(
            {},
            fake_model,
            self.label,
            self.nloc,
            self.cur_lr,
        )
        _, my_loss_absent, my_more_loss_absent = mine(
            {},
            fake_model,
            self.label_absent,
            self.nloc,
            self.cur_lr,
        )
        my_loss = my_loss.detach().cpu()
        my_loss_absent = my_loss_absent.detach().cpu()
        self.assertTrue(np.allclose(base_loss, my_loss.numpy()))
        self.assertTrue(np.allclose(0.0, my_loss_absent.numpy()))
        for key in ["ener", "force", "virial"]:
            self.assertTrue(
                np.allclose(
                    base_more_loss["l2_%s_loss" % key], my_more_loss["l2_%s_loss" % key]
                )
            )
            self.assertTrue(np.isnan(my_more_loss_absent["l2_%s_loss" % key]))


class TestEnerSpinLoss(unittest.TestCase):
    def setUp(self):
        self.system = str(Path(__file__).parent / "NiO/data/data_0")
        self.type_map = ["Ni", "O"]
        self.start_lr = 1.1
        self.start_pref_e = 0.02
        self.limit_pref_e = 1.0
        self.start_pref_fr = 1000.0
        self.limit_pref_fr = 1.0
        self.start_pref_fm = 1000.0
        self.limit_pref_fm = 1.0
        self.cur_lr = 1.2
        self.use_spin = [1, 0]
        # data
        spin_data_requirement = deepcopy(energy_data_requirement)
        spin_data_requirement.append(
            DataRequirementItem(
                "force_mag",
                ndof=3,
                atomic=True,
                must=False,
                high_prec=False,
            )
        )
        np_batch, pt_batch = get_batch(
            self.system, self.type_map, spin_data_requirement
        )
        natoms = np_batch["natoms"]
        self.nloc = natoms[0]
        nframes = np_batch["energy"].shape[0]
        l_energy, l_force_real, l_force_mag, l_virial = (
            np_batch["energy"],
            np_batch["force"],
            np_batch["force_mag"],
            np_batch["virial"],
        )
        # merged force for tf old implement
        l_force_merge_tf = np.concatenate(
            [
                l_force_real.reshape(nframes, self.nloc, 3),
                l_force_mag.reshape(nframes, self.nloc, 3)[
                    np_batch["atype"] == 0
                ].reshape(nframes, -1, 3),
            ],
            axis=1,
        ).reshape(nframes, -1)
        p_energy, p_force_real, p_force_mag, p_force_merge_tf, p_virial = (
            np.ones_like(l_energy),
            np.ones_like(l_force_real),
            np.ones_like(l_force_mag),
            np.ones_like(l_force_merge_tf),
            np.ones_like(l_virial),
        )
        virt_nloc = (np_batch["atype"] == 0).sum(-1)
        natoms_tf = np.concatenate([natoms, virt_nloc], axis=0)
        natoms_tf[:2] += virt_nloc
        nloc = natoms_tf[0]
        batch_size = pt_batch["coord"].shape[0]
        atom_energy = np.zeros(shape=[batch_size, nloc])
        atom_pref = np.zeros(shape=[batch_size, nloc * 3])
        self.nloc_tf = nloc
        # tf
        base = EnerSpinLoss(
            self.start_lr,
            self.start_pref_e,
            self.limit_pref_e,
            self.start_pref_fr,
            self.limit_pref_fr,
            self.start_pref_fm,
            self.limit_pref_fm,
            use_spin=self.use_spin,
        )
        self.g = tf.Graph()
        with self.g.as_default():
            t_cur_lr = tf.placeholder(shape=[], dtype=tf.float64)
            t_natoms = tf.placeholder(shape=[None], dtype=tf.int32)
            t_penergy = tf.placeholder(shape=[None, 1], dtype=tf.float64)
            t_pforce = tf.placeholder(shape=[None, None], dtype=tf.float64)
            t_pvirial = tf.placeholder(shape=[None, 9], dtype=tf.float64)
            t_patom_energy = tf.placeholder(shape=[None, None], dtype=tf.float64)
            t_lenergy = tf.placeholder(shape=[None, 1], dtype=tf.float64)
            t_lforce = tf.placeholder(shape=[None, None], dtype=tf.float64)
            t_lvirial = tf.placeholder(shape=[None, 9], dtype=tf.float64)
            t_latom_energy = tf.placeholder(shape=[None, None], dtype=tf.float64)
            t_atom_pref = tf.placeholder(shape=[None, None], dtype=tf.float64)
            find_energy = tf.constant(1.0, dtype=tf.float64)
            find_force = tf.constant(1.0, dtype=tf.float64)
            find_virial = tf.constant(0.0, dtype=tf.float64)
            find_atom_energy = tf.constant(0.0, dtype=tf.float64)
            find_atom_pref = tf.constant(0.0, dtype=tf.float64)
            model_dict = {
                "energy": t_penergy,
                "force": t_pforce,
                "virial": t_pvirial,
                "atom_ener": t_patom_energy,
            }
            label_dict = {
                "energy": t_lenergy,
                "force": t_lforce,
                "virial": t_lvirial,
                "atom_ener": t_latom_energy,
                "atom_pref": t_atom_pref,
                "find_energy": find_energy,
                "find_force": find_force,
                "find_virial": find_virial,
                "find_atom_ener": find_atom_energy,
                "find_atom_pref": find_atom_pref,
            }
            self.base_loss_sess = base.build(
                t_cur_lr, t_natoms, model_dict, label_dict, ""
            )
        # torch
        self.feed_dict = {
            t_cur_lr: self.cur_lr,
            t_natoms: natoms_tf,
            t_penergy: p_energy,
            t_pforce: p_force_merge_tf,
            t_pvirial: p_virial.reshape(-1, 9),
            t_patom_energy: atom_energy,
            t_lenergy: l_energy,
            t_lforce: l_force_merge_tf,
            t_lvirial: l_virial.reshape(-1, 9),
            t_latom_energy: atom_energy,
            t_atom_pref: atom_pref,
        }
        self.model_pred = {
            "energy": torch.from_numpy(p_energy),
            "force": torch.from_numpy(p_force_real).reshape(nframes, self.nloc, 3),
            "force_mag": torch.from_numpy(p_force_mag).reshape(nframes, self.nloc, 3),
            "mask_mag": torch.from_numpy(np_batch["atype"] == 0).reshape(
                nframes, self.nloc, 1
            ),
        }
        self.label = {
            "energy": torch.from_numpy(l_energy),
            "find_energy": 1.0,
            "force": torch.from_numpy(l_force_real).reshape(nframes, self.nloc, 3),
            "find_force": 1.0,
            "force_mag": torch.from_numpy(l_force_mag).reshape(nframes, self.nloc, 3),
            "find_force_mag": 1.0,
        }
        self.label_absent = {
            "energy": torch.from_numpy(l_energy),
            "force": torch.from_numpy(l_force_real).reshape(nframes, self.nloc, 3),
            "force_mag": torch.from_numpy(l_force_mag).reshape(nframes, self.nloc, 3),
        }
        self.natoms = pt_batch["natoms"]

    def tearDown(self) -> None:
        tf.reset_default_graph()
        return super().tearDown()

    def test_consistency(self):
        with tf.Session(graph=self.g) as sess:
            base_loss, base_more_loss = sess.run(
                self.base_loss_sess, feed_dict=self.feed_dict
            )
        mine = EnergySpinLoss(
            self.start_lr,
            self.start_pref_e,
            self.limit_pref_e,
            self.start_pref_fr,
            self.limit_pref_fr,
            self.start_pref_fm,
            self.limit_pref_fm,
        )

        def fake_model():
            return self.model_pred

        _, my_loss, my_more_loss = mine(
            {},
            fake_model,
            self.label,
            self.nloc_tf,  # use tf natoms pref
            self.cur_lr,
        )
        _, my_loss_absent, my_more_loss_absent = mine(
            {},
            fake_model,
            self.label_absent,
            self.nloc_tf,  # use tf natoms pref
            self.cur_lr,
        )
        my_loss = my_loss.detach().cpu()
        my_loss_absent = my_loss_absent.detach().cpu()
        self.assertTrue(np.allclose(base_loss, my_loss.numpy()))
        self.assertTrue(np.allclose(0.0, my_loss_absent.numpy()))
        for key in ["ener", "force_r", "force_m"]:
            self.assertTrue(
                np.allclose(
                    base_more_loss["l2_%s_loss" % key], my_more_loss["l2_%s_loss" % key]
                )
            )
            self.assertTrue(np.isnan(my_more_loss_absent["l2_%s_loss" % key]))


if __name__ == "__main__":
    unittest.main()
