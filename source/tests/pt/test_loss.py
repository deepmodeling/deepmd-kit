# SPDX-License-Identifier: LGPL-3.0-or-later
import json
import os
import unittest

import numpy as np
import tensorflow.compat.v1 as tf
import torch

tf.disable_eager_execution()
from pathlib import (
    Path,
)

from deepmd.pt.loss import (
    EnergyStdLoss,
)
from deepmd.pt.utils.dataset import (
    DeepmdDataSet,
)
from deepmd.tf.common import (
    expand_sys_str,
)
from deepmd.tf.loss.ener import (
    EnerStdLoss,
)

CUR_DIR = os.path.dirname(__file__)


def get_batch():
    with open(str(Path(__file__).parent / "water/se_e2_a.json")) as fin:
        content = fin.read()
    config = json.loads(content)
    data_file = [str(Path(__file__).parent / "water/data/data_0")]
    config["training"]["training_data"]["systems"] = data_file
    config["training"]["validation_data"]["systems"] = data_file
    model_config = config["model"]
    rcut = model_config["descriptor"]["rcut"]
    # self.rcut_smth = model_config['descriptor']['rcut_smth']
    sel = model_config["descriptor"]["sel"]
    batch_size = config["training"]["training_data"]["batch_size"]
    systems = config["training"]["validation_data"]["systems"]
    if isinstance(systems, str):
        systems = expand_sys_str(systems)
    dataset = DeepmdDataSet(systems, batch_size, model_config["type_map"], rcut, sel)
    np_batch, pt_batch = dataset.get_batch()
    return np_batch, pt_batch


class TestLearningRate(unittest.TestCase):
    def setUp(self):
        self.start_lr = 1.1
        self.start_pref_e = 0.02
        self.limit_pref_e = 1.0
        self.start_pref_f = 1000.0
        self.limit_pref_f = 1.0
        self.start_pref_v = 0.02
        self.limit_pref_v = 1.0
        self.cur_lr = 1.2
        # data
        np_batch, pt_batch = get_batch()
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
            "force": torch.from_numpy(l_force),
            "virial": torch.from_numpy(l_virial),
        }
        self.natoms = pt_batch["natoms"]

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
        my_loss, my_more_loss = mine(
            self.label,
            self.model_pred,
            self.nloc,
            self.cur_lr,
        )
        my_loss = my_loss.detach().cpu()
        self.assertTrue(np.allclose(base_loss, my_loss.numpy()))
        for key in ["ener", "force", "virial"]:
            self.assertTrue(
                np.allclose(
                    base_more_loss["l2_%s_loss" % key], my_more_loss["l2_%s_loss" % key]
                )
            )


if __name__ == "__main__":
    unittest.main()
