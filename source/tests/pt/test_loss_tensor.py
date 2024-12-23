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

from deepmd.pt.loss import TensorLoss as PTTensorLoss
from deepmd.pt.utils import (
    dp_random,
    env,
)
from deepmd.pt.utils.dataset import (
    DeepmdDataSetForLoader,
)
from deepmd.tf.loss.tensor import TensorLoss as TFTensorLoss
from deepmd.utils.data import (
    DataRequirementItem,
)

from ..seed import (
    GLOBAL_SEED,
)

CUR_DIR = os.path.dirname(__file__)


def get_batch(system, type_map, data_requirement):
    dataset = DeepmdDataSetForLoader(system, type_map)
    dataset.add_data_requirement(data_requirement)
    np_batch, pt_batch = get_single_batch(dataset)
    return np_batch, pt_batch


def get_single_batch(dataset, index=None):
    if index is None:
        index = dp_random.choice(np.arange(len(dataset)))
    np_batch = dataset[index]
    pt_batch = {}

    for key in [
        "coord",
        "box",
        "atom_dipole",
        "dipole",
        "atom_polarizability",
        "polarizability",
        "atype",
        "natoms",
    ]:
        if key in np_batch.keys():
            np_batch[key] = np.expand_dims(np_batch[key], axis=0)
            pt_batch[key] = torch.as_tensor(np_batch[key], device=env.DEVICE)
            if key in ["coord", "atom_dipole"]:
                np_batch[key] = np_batch[key].reshape(1, -1)
    np_batch["natoms"] = np_batch["natoms"][0]
    return np_batch, pt_batch


class LossCommonTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cur_lr = 1.2
        self.type_map = ["H", "O"]

        # data
        tensor_data_requirement = [
            DataRequirementItem(
                "atomic_" + self.label_name,
                ndof=self.tensor_size,
                atomic=True,
                must=False,
                high_prec=False,
            ),
            DataRequirementItem(
                self.label_name,
                ndof=self.tensor_size,
                atomic=False,
                must=False,
                high_prec=False,
            ),
            DataRequirementItem(
                "atomic_weight",
                ndof=1,
                atomic=True,
                must=False,
                high_prec=False,
                default=1.0,
            ),
        ]
        np_batch, pt_batch = get_batch(
            self.system, self.type_map, tensor_data_requirement
        )
        natoms = np_batch["natoms"]
        self.nloc = natoms[0]
        self.nframes = np_batch["atom_" + self.label_name].shape[0]
        rng = np.random.default_rng(GLOBAL_SEED)

        l_atomic_tensor, l_global_tensor = (
            np_batch["atom_" + self.label_name],
            np_batch[self.label_name],
        )
        p_atomic_tensor, p_global_tensor = (
            np.ones_like(l_atomic_tensor),
            np.ones_like(l_global_tensor),
        )

        batch_size = pt_batch["coord"].shape[0]

        # atom_pref = rng.random(size=[batch_size, nloc * 3])
        # drdq = rng.random(size=[batch_size, nloc * 2 * 3])
        atom_weight = rng.random(size=[batch_size, self.nloc])

        # tf
        self.g = tf.Graph()
        with self.g.as_default():
            t_cur_lr = tf.placeholder(shape=[], dtype=tf.float64)
            t_natoms = tf.placeholder(shape=[None], dtype=tf.int32)
            t_patomic_tensor = tf.placeholder(shape=[None, None], dtype=tf.float64)
            t_pglobal_tensor = tf.placeholder(shape=[None, None], dtype=tf.float64)
            t_latomic_tensor = tf.placeholder(shape=[None, None], dtype=tf.float64)
            t_lglobal_tensor = tf.placeholder(shape=[None, None], dtype=tf.float64)
            t_atom_weight = tf.placeholder(shape=[None, None], dtype=tf.float64)
            find_atomic = tf.constant(1.0, dtype=tf.float64)
            find_global = tf.constant(1.0, dtype=tf.float64)
            find_atom_weight = tf.constant(1.0, dtype=tf.float64)
            model_dict = {
                self.tensor_name: t_patomic_tensor,
            }
            label_dict = {
                "atom_" + self.label_name: t_latomic_tensor,
                "find_atom_" + self.label_name: find_atomic,
                self.label_name: t_lglobal_tensor,
                "find_" + self.label_name: find_global,
                "atom_weight": t_atom_weight,
                "find_atom_weight": find_atom_weight,
            }
            self.tf_loss_sess = self.tf_loss.build(
                t_cur_lr, t_natoms, model_dict, label_dict, ""
            )

        self.feed_dict = {
            t_cur_lr: self.cur_lr,
            t_natoms: natoms,
            t_patomic_tensor: p_atomic_tensor,
            t_pglobal_tensor: p_global_tensor,
            t_latomic_tensor: l_atomic_tensor,
            t_lglobal_tensor: l_global_tensor,
            t_atom_weight: atom_weight,
        }
        # pt
        self.model_pred = {
            self.tensor_name: torch.from_numpy(p_atomic_tensor),
            "global_" + self.tensor_name: torch.from_numpy(p_global_tensor),
        }
        self.label = {
            "atom_" + self.label_name: torch.from_numpy(l_atomic_tensor),
            "find_" + "atom_" + self.label_name: 1.0,
            self.label_name: torch.from_numpy(l_global_tensor),
            "find_" + self.label_name: 1.0,
            "atom_weight": torch.from_numpy(atom_weight),
            "find_atom_weight": 1.0,
        }
        self.label_absent = {
            "atom_" + self.label_name: torch.from_numpy(l_atomic_tensor),
            self.label_name: torch.from_numpy(l_global_tensor),
            "atom_weight": torch.from_numpy(atom_weight),
        }
        self.natoms = pt_batch["natoms"]

    def tearDown(self) -> None:
        tf.reset_default_graph()
        return super().tearDown()


class TestAtomicDipoleLoss(LossCommonTest):
    def setUp(self) -> None:
        self.tensor_name = "dipole"
        self.tensor_size = 3
        self.label_name = "dipole"
        self.system = str(Path(__file__).parent / "water_tensor/dipole/O78H156")

        self.pref_atomic = 1.0
        self.pref = 0.0
        # tf
        self.tf_loss = TFTensorLoss(
            {
                "pref_atomic": self.pref_atomic,
                "pref": self.pref,
            },
            tensor_name=self.tensor_name,
            tensor_size=self.tensor_size,
            label_name=self.label_name,
        )
        # pt
        self.pt_loss = PTTensorLoss(
            self.tensor_name,
            self.tensor_size,
            self.label_name,
            self.pref_atomic,
            self.pref,
        )

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
        for key in ["local"]:
            self.assertTrue(
                np.allclose(
                    tf_more_loss[f"{key}_loss"],
                    pt_more_loss[f"l2_{key}_{self.tensor_name}_loss"],
                )
            )
            self.assertTrue(
                np.isnan(pt_more_loss_absent[f"l2_{key}_{self.tensor_name}_loss"])
            )


class TestAtomicDipoleAWeightLoss(LossCommonTest):
    def setUp(self) -> None:
        self.tensor_name = "dipole"
        self.tensor_size = 3
        self.label_name = "dipole"
        self.system = str(Path(__file__).parent / "water_tensor/dipole/O78H156")

        self.pref_atomic = 1.0
        self.pref = 0.0
        # tf
        self.tf_loss = TFTensorLoss(
            {
                "pref_atomic": self.pref_atomic,
                "pref": self.pref,
                "enable_atomic_weight": True,
            },
            tensor_name=self.tensor_name,
            tensor_size=self.tensor_size,
            label_name=self.label_name,
        )
        # pt
        self.pt_loss = PTTensorLoss(
            self.tensor_name,
            self.tensor_size,
            self.label_name,
            self.pref_atomic,
            self.pref,
            enable_atomic_weight=True,
        )

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
        for key in ["local"]:
            self.assertTrue(
                np.allclose(
                    tf_more_loss[f"{key}_loss"],
                    pt_more_loss[f"l2_{key}_{self.tensor_name}_loss"],
                )
            )
            self.assertTrue(
                np.isnan(pt_more_loss_absent[f"l2_{key}_{self.tensor_name}_loss"])
            )


class TestAtomicPolarLoss(LossCommonTest):
    def setUp(self) -> None:
        self.tensor_name = "polar"
        self.tensor_size = 9
        self.label_name = "polarizability"

        self.system = str(Path(__file__).parent / "water_tensor/polar/atomic_system")

        self.pref_atomic = 1.0
        self.pref = 0.0
        # tf
        self.tf_loss = TFTensorLoss(
            {
                "pref_atomic": self.pref_atomic,
                "pref": self.pref,
            },
            tensor_name=self.tensor_name,
            tensor_size=self.tensor_size,
            label_name=self.label_name,
        )
        # pt
        self.pt_loss = PTTensorLoss(
            self.tensor_name,
            self.tensor_size,
            self.label_name,
            self.pref_atomic,
            self.pref,
        )

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
        for key in ["local"]:
            self.assertTrue(
                np.allclose(
                    tf_more_loss[f"{key}_loss"],
                    pt_more_loss[f"l2_{key}_{self.tensor_name}_loss"],
                )
            )
            self.assertTrue(
                np.isnan(pt_more_loss_absent[f"l2_{key}_{self.tensor_name}_loss"])
            )


class TestAtomicPolarAWeightLoss(LossCommonTest):
    def setUp(self) -> None:
        self.tensor_name = "polar"
        self.tensor_size = 9
        self.label_name = "polarizability"

        self.system = str(Path(__file__).parent / "water_tensor/polar/atomic_system")

        self.pref_atomic = 1.0
        self.pref = 0.0
        # tf
        self.tf_loss = TFTensorLoss(
            {
                "pref_atomic": self.pref_atomic,
                "pref": self.pref,
                "enable_atomic_weight": True,
            },
            tensor_name=self.tensor_name,
            tensor_size=self.tensor_size,
            label_name=self.label_name,
        )
        # pt
        self.pt_loss = PTTensorLoss(
            self.tensor_name,
            self.tensor_size,
            self.label_name,
            self.pref_atomic,
            self.pref,
            enable_atomic_weight=True,
        )

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
        for key in ["local"]:
            self.assertTrue(
                np.allclose(
                    tf_more_loss[f"{key}_loss"],
                    pt_more_loss[f"l2_{key}_{self.tensor_name}_loss"],
                )
            )
            self.assertTrue(
                np.isnan(pt_more_loss_absent[f"l2_{key}_{self.tensor_name}_loss"])
            )


if __name__ == "__main__":
    unittest.main()
