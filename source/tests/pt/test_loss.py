# SPDX-License-Identifier: LGPL-3.0-or-later
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
    EnergyHessianStdLoss,
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

from ..seed import (
    GLOBAL_SEED,
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


class LossCommonTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cur_lr = 1.2
        if not self.spin:
            self.system = str(Path(__file__).parent / "water/data/data_0")
            self.type_map = ["H", "O"]
            if self.hess:
                self.system = str(Path(__file__).parent / "hessian/data/H8C4N2O")
                self.type_map = ["C", "H", "N", "O"]
                energy_data_requirement.append(
                    DataRequirementItem(
                        "hessian",
                        ndof=1,
                        atomic=True,
                        must=False,
                        high_prec=False,
                    )
                )
        else:
            self.system = str(Path(__file__).parent / "NiO/data/data_0")
            self.type_map = ["Ni", "O"]
            energy_data_requirement.append(
                DataRequirementItem(
                    "force_mag",
                    ndof=3,
                    atomic=True,
                    must=False,
                    high_prec=False,
                )
            )
        # data
        np_batch, pt_batch = get_batch(
            self.system, self.type_map, energy_data_requirement
        )
        natoms = np_batch["natoms"]
        self.nloc = natoms[0]
        nframes = np_batch["energy"].shape[0]
        rng = np.random.default_rng(GLOBAL_SEED)

        if not self.spin:
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
            p_atom_energy = rng.random(size=[batch_size, nloc])
            l_atom_energy = rng.random(size=[batch_size, nloc])
            atom_pref = rng.random(size=[batch_size, nloc * 3])
            drdq = rng.random(size=[batch_size, nloc * 2 * 3])
            atom_ener_coeff = rng.random(size=[batch_size, nloc])
            # placeholders
            l_force_real = l_force
            l_force_mag = l_force
            p_force_real = p_force
            p_force_mag = p_force
        else:
            # data
            np_batch, pt_batch = get_batch(
                self.system, self.type_map, energy_data_requirement
            )
            natoms = np_batch["natoms"]
            self.nloc = natoms[0]
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
            p_atom_energy = rng.random(size=[batch_size, nloc])
            l_atom_energy = rng.random(size=[batch_size, nloc])
            atom_pref = rng.random(size=[batch_size, nloc * 3])
            drdq = rng.random(size=[batch_size, nloc * 2 * 3])
            atom_ener_coeff = rng.random(size=[batch_size, nloc])
            self.nloc_tf = nloc
            natoms = natoms_tf
            l_force = l_force_merge_tf
            p_force = p_force_merge_tf

        # tf
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
            t_atom_ener_coeff = tf.placeholder(shape=[None, None], dtype=tf.float64)
            t_drdq = tf.placeholder(shape=[None, None], dtype=tf.float64)
            find_energy = tf.constant(1.0, dtype=tf.float64)
            find_force = tf.constant(1.0, dtype=tf.float64)
            find_virial = tf.constant(1.0 if not self.spin else 0.0, dtype=tf.float64)
            find_atom_energy = tf.constant(1.0, dtype=tf.float64)
            find_atom_pref = tf.constant(1.0, dtype=tf.float64)
            find_drdq = tf.constant(1.0, dtype=tf.float64)
            find_atom_ener_coeff = tf.constant(1.0, dtype=tf.float64)
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
                "drdq": t_drdq,
                "atom_ener_coeff": t_atom_ener_coeff,
                "find_energy": find_energy,
                "find_force": find_force,
                "find_virial": find_virial,
                "find_atom_ener": find_atom_energy,
                "find_atom_pref": find_atom_pref,
                "find_drdq": find_drdq,
                "find_atom_ener_coeff": find_atom_ener_coeff,
            }
            self.tf_loss_sess = self.tf_loss.build(
                t_cur_lr, t_natoms, model_dict, label_dict, ""
            )

        self.feed_dict = {
            t_cur_lr: self.cur_lr,
            t_natoms: natoms,
            t_penergy: p_energy,
            t_pforce: p_force,
            t_pvirial: p_virial.reshape(-1, 9),
            t_patom_energy: p_atom_energy,
            t_lenergy: l_energy,
            t_lforce: l_force,
            t_lvirial: l_virial.reshape(-1, 9),
            t_latom_energy: l_atom_energy,
            t_atom_pref: atom_pref,
            t_drdq: drdq,
            t_atom_ener_coeff: atom_ener_coeff,
        }
        # pt
        if not self.spin:
            self.model_pred = {
                "energy": torch.from_numpy(p_energy),
                "force": torch.from_numpy(p_force),
                "virial": torch.from_numpy(p_virial),
                "atom_energy": torch.from_numpy(p_atom_energy),
            }
            self.label = {
                "energy": torch.from_numpy(l_energy),
                "find_energy": 1.0,
                "force": torch.from_numpy(l_force),
                "find_force": 1.0,
                "virial": torch.from_numpy(l_virial),
                "find_virial": 1.0,
                "atom_ener": torch.from_numpy(l_atom_energy),
                "find_atom_ener": 1.0,
                "atom_pref": torch.from_numpy(atom_pref),
                "find_atom_pref": 1.0,
                "drdq": torch.from_numpy(drdq),
                "find_drdq": 1.0,
                "atom_ener_coeff": torch.from_numpy(atom_ener_coeff),
                "find_atom_ener_coeff": 1.0,
            }
            self.label_absent = {
                "energy": torch.from_numpy(l_energy),
                "force": torch.from_numpy(l_force),
                "virial": torch.from_numpy(l_virial),
                "atom_ener": torch.from_numpy(l_atom_energy),
                "atom_pref": torch.from_numpy(atom_pref),
                "drdq": torch.from_numpy(drdq),
                "atom_ener_coeff": torch.from_numpy(atom_ener_coeff),
            }
            if self.hess:
                l_hessian = np_batch["hessian"]
                p_hessian = np.ones_like(l_hessian)
                self.model_pred["hessian"] = torch.from_numpy(p_hessian)
                self.label["hessian"] = torch.from_numpy(l_hessian)
                self.label["find_hessian"] = 1.0
                self.label_absent["hessian"] = torch.from_numpy(l_hessian)

        else:
            self.model_pred = {
                "energy": torch.from_numpy(p_energy),
                "force": torch.from_numpy(p_force_real).reshape(nframes, self.nloc, 3),
                "force_mag": torch.from_numpy(p_force_mag).reshape(
                    nframes, self.nloc, 3
                ),
                "mask_mag": torch.from_numpy(np_batch["atype"] == 0).reshape(
                    nframes, self.nloc, 1
                ),
                "atom_energy": torch.from_numpy(p_atom_energy),
            }
            self.label = {
                "energy": torch.from_numpy(l_energy),
                "find_energy": 1.0,
                "force": torch.from_numpy(l_force_real).reshape(nframes, self.nloc, 3),
                "find_force": 1.0,
                "force_mag": torch.from_numpy(l_force_mag).reshape(
                    nframes, self.nloc, 3
                ),
                "find_force_mag": 1.0,
                "atom_ener": torch.from_numpy(l_atom_energy),
                "find_atom_ener": 1.0,
                "atom_ener_coeff": torch.from_numpy(atom_ener_coeff),
                "find_atom_ener_coeff": 1.0,
            }
            self.label_absent = {
                "energy": torch.from_numpy(l_energy),
                "force": torch.from_numpy(l_force_real).reshape(nframes, self.nloc, 3),
                "force_mag": torch.from_numpy(l_force_mag).reshape(
                    nframes, self.nloc, 3
                ),
                "atom_ener": torch.from_numpy(l_atom_energy),
                "atom_ener_coeff": torch.from_numpy(atom_ener_coeff),
            }
        self.natoms = pt_batch["natoms"]

    def tearDown(self) -> None:
        tf.reset_default_graph()
        return super().tearDown()


class TestEnerStdLoss(LossCommonTest):
    def setUp(self) -> None:
        self.start_lr = 1.1
        self.start_pref_e = 0.02
        self.limit_pref_e = 1.0
        self.start_pref_f = 1000.0
        self.limit_pref_f = 1.0
        self.start_pref_v = 0.02
        self.limit_pref_v = 1.0
        # tf
        self.tf_loss = EnerStdLoss(
            self.start_lr,
            self.start_pref_e,
            self.limit_pref_e,
            self.start_pref_f,
            self.limit_pref_f,
            self.start_pref_v,
            self.limit_pref_v,
        )
        # pt
        self.pt_loss = EnergyStdLoss(
            self.start_lr,
            self.start_pref_e,
            self.limit_pref_e,
            self.start_pref_f,
            self.limit_pref_f,
            self.start_pref_v,
            self.limit_pref_v,
        )
        self.spin = False
        self.hess = False
        super().setUp()

    def test_consistency(self) -> None:
        with tf.Session(graph=self.g) as sess:
            tf_loss, tf_more_loss = sess.run(
                self.tf_loss_sess, feed_dict=self.feed_dict
            )

        def fake_model():
            return self.model_pred

        _, pt_loss, pt_more_loss = self.pt_loss(
            {},
            fake_model,
            self.label,
            self.nloc,
            self.cur_lr,
        )
        _, pt_loss_absent, pt_more_loss_absent = self.pt_loss(
            {},
            fake_model,
            self.label_absent,
            self.nloc,
            self.cur_lr,
        )
        pt_loss = pt_loss.detach().cpu()
        pt_loss_absent = pt_loss_absent.detach().cpu()
        self.assertTrue(np.allclose(tf_loss, pt_loss.numpy()))
        self.assertTrue(np.allclose(0.0, pt_loss_absent.numpy()))
        for key in ["ener", "force", "virial"]:
            self.assertTrue(
                np.allclose(
                    tf_more_loss[f"l2_{key}_loss"], pt_more_loss[f"l2_{key}_loss"]
                )
            )
            self.assertTrue(np.isnan(pt_more_loss_absent[f"l2_{key}_loss"]))


class TestEnerStdLossAePfGf(LossCommonTest):
    def setUp(self) -> None:
        self.start_lr = 1.1
        self.start_pref_e = 0.02
        self.limit_pref_e = 1.0
        self.start_pref_f = 1000.0
        self.limit_pref_f = 1.0
        self.start_pref_v = 0.02
        self.limit_pref_v = 1.0
        self.start_pref_ae = 0.02
        self.limit_pref_ae = 1.0
        self.start_pref_pf = 0.02
        self.limit_pref_pf = 1.0
        self.start_pref_gf = 0.02
        self.limit_pref_gf = 1.0
        self.numb_generalized_coord = 2
        # tf
        self.tf_loss = EnerStdLoss(
            self.start_lr,
            self.start_pref_e,
            self.limit_pref_e,
            self.start_pref_f,
            self.limit_pref_f,
            self.start_pref_v,
            self.limit_pref_v,
            self.start_pref_ae,
            self.limit_pref_ae,
            self.start_pref_pf,
            self.limit_pref_pf,
            start_pref_gf=self.start_pref_gf,
            limit_pref_gf=self.limit_pref_gf,
            numb_generalized_coord=self.numb_generalized_coord,
        )
        # pt
        self.pt_loss = EnergyStdLoss(
            self.start_lr,
            self.start_pref_e,
            self.limit_pref_e,
            self.start_pref_f,
            self.limit_pref_f,
            self.start_pref_v,
            self.limit_pref_v,
            self.start_pref_ae,
            self.limit_pref_ae,
            self.start_pref_pf,
            self.limit_pref_pf,
            start_pref_gf=self.start_pref_gf,
            limit_pref_gf=self.limit_pref_gf,
            numb_generalized_coord=self.numb_generalized_coord,
        )
        self.spin = False
        self.hess = False
        super().setUp()

    def test_consistency(self) -> None:
        with tf.Session(graph=self.g) as sess:
            tf_loss, tf_more_loss = sess.run(
                self.tf_loss_sess, feed_dict=self.feed_dict
            )

        def fake_model():
            return self.model_pred

        _, pt_loss, pt_more_loss = self.pt_loss(
            {},
            fake_model,
            self.label,
            self.nloc,
            self.cur_lr,
        )
        _, pt_loss_absent, pt_more_loss_absent = self.pt_loss(
            {},
            fake_model,
            self.label_absent,
            self.nloc,
            self.cur_lr,
        )
        pt_loss = pt_loss.detach().cpu()
        pt_loss_absent = pt_loss_absent.detach().cpu()
        self.assertTrue(np.allclose(tf_loss, pt_loss.numpy()))
        self.assertTrue(np.allclose(0.0, pt_loss_absent.numpy()))
        for key in ["ener", "force", "virial", "atom_ener", "pref_force", "gen_force"]:
            self.assertTrue(
                np.allclose(
                    tf_more_loss[f"l2_{key}_loss"], pt_more_loss[f"l2_{key}_loss"]
                )
            )
            self.assertTrue(np.isnan(pt_more_loss_absent[f"l2_{key}_loss"]))


class TestEnerStdLossAecoeff(LossCommonTest):
    def setUp(self) -> None:
        self.start_lr = 1.1
        self.start_pref_e = 0.02
        self.limit_pref_e = 1.0
        self.start_pref_f = 1000.0
        self.limit_pref_f = 1.0
        self.start_pref_v = 0.02
        self.limit_pref_v = 1.0
        # tf
        self.tf_loss = EnerStdLoss(
            self.start_lr,
            self.start_pref_e,
            self.limit_pref_e,
            self.start_pref_f,
            self.limit_pref_f,
            self.start_pref_v,
            self.limit_pref_v,
            enable_atom_ener_coeff=True,
        )
        # pt
        self.pt_loss = EnergyStdLoss(
            self.start_lr,
            self.start_pref_e,
            self.limit_pref_e,
            self.start_pref_f,
            self.limit_pref_f,
            self.start_pref_v,
            self.limit_pref_v,
            enable_atom_ener_coeff=True,
        )
        self.spin = False
        self.hess = False
        super().setUp()

    def test_consistency(self) -> None:
        with tf.Session(graph=self.g) as sess:
            tf_loss, tf_more_loss = sess.run(
                self.tf_loss_sess, feed_dict=self.feed_dict
            )

        def fake_model():
            return self.model_pred

        _, pt_loss, pt_more_loss = self.pt_loss(
            {},
            fake_model,
            self.label,
            self.nloc,
            self.cur_lr,
        )
        _, pt_loss_absent, pt_more_loss_absent = self.pt_loss(
            {},
            fake_model,
            self.label_absent,
            self.nloc,
            self.cur_lr,
        )
        pt_loss = pt_loss.detach().cpu()
        pt_loss_absent = pt_loss_absent.detach().cpu()
        self.assertTrue(np.allclose(tf_loss, pt_loss.numpy()))
        self.assertTrue(np.allclose(0.0, pt_loss_absent.numpy()))
        for key in ["ener", "force", "virial"]:
            self.assertTrue(
                np.allclose(
                    tf_more_loss[f"l2_{key}_loss"], pt_more_loss[f"l2_{key}_loss"]
                )
            )
            self.assertTrue(np.isnan(pt_more_loss_absent[f"l2_{key}_loss"]))


class TestEnerStdLossRelativeF(LossCommonTest):
    def setUp(self) -> None:
        self.start_lr = 1.1
        self.start_pref_e = 0.02
        self.limit_pref_e = 1.0
        self.start_pref_f = 1000.0
        self.limit_pref_f = 1.0
        self.start_pref_v = 0.02
        self.limit_pref_v = 1.0
        # tf
        self.tf_loss = EnerStdLoss(
            self.start_lr,
            self.start_pref_e,
            self.limit_pref_e,
            self.start_pref_f,
            self.limit_pref_f,
            self.start_pref_v,
            self.limit_pref_v,
            relative_f=0.1,
        )
        # pt
        self.pt_loss = EnergyStdLoss(
            self.start_lr,
            self.start_pref_e,
            self.limit_pref_e,
            self.start_pref_f,
            self.limit_pref_f,
            self.start_pref_v,
            self.limit_pref_v,
            relative_f=0.1,
        )
        self.spin = False
        self.hess = False
        super().setUp()

    def test_consistency(self) -> None:
        with tf.Session(graph=self.g) as sess:
            tf_loss, tf_more_loss = sess.run(
                self.tf_loss_sess, feed_dict=self.feed_dict
            )

        def fake_model():
            return self.model_pred

        _, pt_loss, pt_more_loss = self.pt_loss(
            {},
            fake_model,
            self.label,
            self.nloc,
            self.cur_lr,
        )
        _, pt_loss_absent, pt_more_loss_absent = self.pt_loss(
            {},
            fake_model,
            self.label_absent,
            self.nloc,
            self.cur_lr,
        )
        pt_loss = pt_loss.detach().cpu()
        pt_loss_absent = pt_loss_absent.detach().cpu()
        self.assertTrue(np.allclose(tf_loss, pt_loss.numpy()))
        self.assertTrue(np.allclose(0.0, pt_loss_absent.numpy()))
        for key in ["ener", "force", "virial"]:
            self.assertTrue(
                np.allclose(
                    tf_more_loss[f"l2_{key}_loss"], pt_more_loss[f"l2_{key}_loss"]
                )
            )
            self.assertTrue(np.isnan(pt_more_loss_absent[f"l2_{key}_loss"]))


class TestEnerHessStdLoss(LossCommonTest):
    def setUp(self):
        self.start_lr = 1.1
        self.start_pref_e = 0.02
        self.limit_pref_e = 1.0
        self.start_pref_f = 1000.0
        self.limit_pref_f = 1.0
        self.start_pref_v = 0.02
        self.limit_pref_v = 1.0
        self.start_pref_h = 10.0
        self.limit_pref_h = 1.0
        # tf
        self.tf_loss = EnerStdLoss(
            self.start_lr,
            self.start_pref_e,
            self.limit_pref_e,
            self.start_pref_f,
            self.limit_pref_f,
            self.start_pref_v,
            self.limit_pref_v,
        )
        # pt
        self.pt_loss = EnergyStdLoss(
            self.start_lr,
            self.start_pref_e,
            self.limit_pref_e,
            self.start_pref_f,
            self.limit_pref_f,
            self.start_pref_v,
            self.limit_pref_v,
        )
        # pt-hess
        self.pt_loss_h = EnergyHessianStdLoss(
            starter_learning_rate=self.start_lr,
            start_pref_e=self.start_pref_e,
            limit_pref_e=self.limit_pref_e,
            start_pref_f=self.start_pref_f,
            limit_pref_f=self.limit_pref_f,
            start_pref_v=self.start_pref_v,
            limit_pref_v=self.limit_pref_v,
            start_pref_h=self.start_pref_h,
            limit_pref_h=self.limit_pref_h,
        )
        self.spin = False
        self.hess = True
        super().setUp()

    def test_consistency(self):
        with tf.Session(graph=self.g) as sess:
            tf_loss, tf_more_loss = sess.run(
                self.tf_loss_sess, feed_dict=self.feed_dict
            )

        def fake_model():
            return self.model_pred

        _, pt_loss, pt_more_loss = self.pt_loss(
            {},
            fake_model,
            self.label,
            self.nloc,
            self.cur_lr,
        )
        _, pt_loss_absent, pt_more_loss_absent = self.pt_loss(
            {},
            fake_model,
            self.label_absent,
            self.nloc,
            self.cur_lr,
        )
        pt_loss = pt_loss.detach().cpu()
        pt_loss_absent = pt_loss_absent.detach().cpu()
        _, pt_loss_h, pt_more_loss_h = self.pt_loss_h(
            {},
            fake_model,
            self.label,
            self.nloc,
            self.cur_lr,
        )
        _, pt_loss_h_absent, pt_more_loss_h_absent = self.pt_loss_h(
            {},
            fake_model,
            self.label_absent,
            self.nloc,
            self.cur_lr,
        )
        pt_loss_h_absent = pt_loss_h_absent.detach().cpu()
        self.assertTrue(np.allclose(tf_loss, pt_loss.numpy()))
        self.assertTrue(np.allclose(0.0, pt_loss_absent.numpy()))
        self.assertTrue(np.allclose(0.0, pt_loss_h_absent.numpy()))
        for key in ["ener", "force", "virial"]:
            self.assertTrue(
                np.allclose(
                    tf_more_loss[f"l2_{key}_loss"], pt_more_loss[f"l2_{key}_loss"]
                )
            )
            self.assertTrue(
                np.allclose(
                    pt_more_loss[f"l2_{key}_loss"], pt_more_loss_h[f"l2_{key}_loss"]
                )
            )
            self.assertTrue(np.isnan(pt_more_loss_absent[f"l2_{key}_loss"]))
        for key in ["ener", "force", "virial", "hessian"]:
            self.assertTrue(np.isnan(pt_more_loss_h_absent[f"l2_{key}_loss"]))


class TestEnerSpinLoss(LossCommonTest):
    def setUp(self) -> None:
        self.start_lr = 1.1
        self.start_pref_e = 0.02
        self.limit_pref_e = 1.0
        self.start_pref_fr = 1000.0
        self.limit_pref_fr = 1.0
        self.start_pref_fm = 1000.0
        self.limit_pref_fm = 1.0
        self.cur_lr = 1.2
        self.use_spin = [1, 0]
        # tf
        self.tf_loss = EnerSpinLoss(
            self.start_lr,
            self.start_pref_e,
            self.limit_pref_e,
            self.start_pref_fr,
            self.limit_pref_fr,
            self.start_pref_fm,
            self.limit_pref_fm,
            use_spin=self.use_spin,
        )
        # pt
        self.pt_loss = EnergySpinLoss(
            self.start_lr,
            self.start_pref_e,
            self.limit_pref_e,
            self.start_pref_fr,
            self.limit_pref_fr,
            self.start_pref_fm,
            self.limit_pref_fm,
        )
        self.spin = True
        self.hess = False
        super().setUp()

    def test_consistency(self) -> None:
        with tf.Session(graph=self.g) as sess:
            tf_loss, tf_more_loss = sess.run(
                self.tf_loss_sess, feed_dict=self.feed_dict
            )

        def fake_model():
            return self.model_pred

        _, pt_loss, pt_more_loss = self.pt_loss(
            {},
            fake_model,
            self.label,
            self.nloc_tf,  # use tf natoms pref
            self.cur_lr,
        )
        _, pt_loss_absent, pt_more_loss_absent = self.pt_loss(
            {},
            fake_model,
            self.label_absent,
            self.nloc_tf,  # use tf natoms pref
            self.cur_lr,
        )
        pt_loss = pt_loss.detach().cpu()
        pt_loss_absent = pt_loss_absent.detach().cpu()
        self.assertTrue(np.allclose(tf_loss, pt_loss.numpy()))
        self.assertTrue(np.allclose(0.0, pt_loss_absent.numpy()))
        for key in ["ener", "force_r", "force_m"]:
            self.assertTrue(
                np.allclose(
                    tf_more_loss[f"l2_{key}_loss"], pt_more_loss[f"l2_{key}_loss"]
                )
            )
            self.assertTrue(np.isnan(pt_more_loss_absent[f"l2_{key}_loss"]))


class TestEnerSpinLossAe(LossCommonTest):
    def setUp(self) -> None:
        self.start_lr = 1.1
        self.start_pref_e = 0.02
        self.limit_pref_e = 1.0
        self.start_pref_fr = 1000.0
        self.limit_pref_fr = 1.0
        self.start_pref_fm = 1000.0
        self.limit_pref_fm = 1.0
        self.start_pref_ae = 0.02
        self.limit_pref_ae = 1.0
        self.cur_lr = 1.2
        self.use_spin = [1, 0]
        # tf
        self.tf_loss = EnerSpinLoss(
            self.start_lr,
            self.start_pref_e,
            self.limit_pref_e,
            self.start_pref_fr,
            self.limit_pref_fr,
            self.start_pref_fm,
            self.limit_pref_fm,
            start_pref_ae=self.start_pref_ae,
            limit_pref_ae=self.limit_pref_ae,
            use_spin=self.use_spin,
        )
        # pt
        self.pt_loss = EnergySpinLoss(
            self.start_lr,
            self.start_pref_e,
            self.limit_pref_e,
            self.start_pref_fr,
            self.limit_pref_fr,
            self.start_pref_fm,
            self.limit_pref_fm,
            start_pref_ae=self.start_pref_ae,
            limit_pref_ae=self.limit_pref_ae,
        )
        self.spin = True
        self.hess = False
        super().setUp()

    def test_consistency(self) -> None:
        with tf.Session(graph=self.g) as sess:
            tf_loss, tf_more_loss = sess.run(
                self.tf_loss_sess, feed_dict=self.feed_dict
            )

        def fake_model():
            return self.model_pred

        _, pt_loss, pt_more_loss = self.pt_loss(
            {},
            fake_model,
            self.label,
            self.nloc_tf,  # use tf natoms pref
            self.cur_lr,
        )
        _, pt_loss_absent, pt_more_loss_absent = self.pt_loss(
            {},
            fake_model,
            self.label_absent,
            self.nloc_tf,  # use tf natoms pref
            self.cur_lr,
        )
        pt_loss = pt_loss.detach().cpu()
        pt_loss_absent = pt_loss_absent.detach().cpu()
        self.assertTrue(np.allclose(tf_loss, pt_loss.numpy()))
        self.assertTrue(np.allclose(0.0, pt_loss_absent.numpy()))
        for key in ["ener", "force_r", "force_m", "atom_ener"]:
            self.assertTrue(
                np.allclose(
                    tf_more_loss[f"l2_{key}_loss"], pt_more_loss[f"l2_{key}_loss"]
                )
            )
            self.assertTrue(np.isnan(pt_more_loss_absent[f"l2_{key}_loss"]))


class TestEnerSpinLossAecoeff(LossCommonTest):
    def setUp(self) -> None:
        self.start_lr = 1.1
        self.start_pref_e = 0.02
        self.limit_pref_e = 1.0
        self.start_pref_fr = 1000.0
        self.limit_pref_fr = 1.0
        self.start_pref_fm = 1000.0
        self.limit_pref_fm = 1.0
        self.cur_lr = 1.2
        self.use_spin = [1, 0]
        # tf
        self.tf_loss = EnerSpinLoss(
            self.start_lr,
            self.start_pref_e,
            self.limit_pref_e,
            self.start_pref_fr,
            self.limit_pref_fr,
            self.start_pref_fm,
            self.limit_pref_fm,
            use_spin=self.use_spin,
            enable_atom_ener_coeff=True,
        )
        # pt
        self.pt_loss = EnergySpinLoss(
            self.start_lr,
            self.start_pref_e,
            self.limit_pref_e,
            self.start_pref_fr,
            self.limit_pref_fr,
            self.start_pref_fm,
            self.limit_pref_fm,
            enable_atom_ener_coeff=True,
        )
        self.spin = True
        self.hess = False
        super().setUp()

    def test_consistency(self) -> None:
        with tf.Session(graph=self.g) as sess:
            tf_loss, tf_more_loss = sess.run(
                self.tf_loss_sess, feed_dict=self.feed_dict
            )

        def fake_model():
            return self.model_pred

        _, pt_loss, pt_more_loss = self.pt_loss(
            {},
            fake_model,
            self.label,
            self.nloc_tf,  # use tf natoms pref
            self.cur_lr,
        )
        _, pt_loss_absent, pt_more_loss_absent = self.pt_loss(
            {},
            fake_model,
            self.label_absent,
            self.nloc_tf,  # use tf natoms pref
            self.cur_lr,
        )
        pt_loss = pt_loss.detach().cpu()
        pt_loss_absent = pt_loss_absent.detach().cpu()
        self.assertTrue(np.allclose(tf_loss, pt_loss.numpy()))
        self.assertTrue(np.allclose(0.0, pt_loss_absent.numpy()))
        for key in ["ener", "force_r", "force_m"]:
            self.assertTrue(
                np.allclose(
                    tf_more_loss[f"l2_{key}_loss"], pt_more_loss[f"l2_{key}_loss"]
                )
            )
            self.assertTrue(np.isnan(pt_more_loss_absent[f"l2_{key}_loss"]))


if __name__ == "__main__":
    unittest.main()
