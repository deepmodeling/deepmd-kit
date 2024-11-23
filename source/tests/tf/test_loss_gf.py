# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np
import tensorflow as tf

from deepmd.tf.loss import (
    EnerStdLoss,
)
from deepmd.utils.data import (
    DataRequirementItem,
)


class TestLossGf(tf.test.TestCase):
    def setUp(self) -> None:
        self.loss = EnerStdLoss(
            1e-3,
            start_pref_e=0,
            limit_pref_e=0,
            start_pref_f=0,
            limit_pref_f=0,
            start_pref_v=0,
            limit_pref_v=0,
            start_pref_ae=0,
            limit_pref_ae=0,
            start_pref_av=0,
            limit_pref_av=0,
            start_pref_gf=1,
            limit_pref_gf=1,
            numb_generalized_coord=2,
        )

    def test_label_requirements(self) -> None:
        """Test label_requirements are expected."""
        self.assertCountEqual(
            self.loss.label_requirement,
            [
                DataRequirementItem(
                    "energy",
                    1,
                    atomic=False,
                    must=False,
                    high_prec=True,
                    repeat=1,
                ),
                DataRequirementItem(
                    "force",
                    3,
                    atomic=True,
                    must=False,
                    high_prec=False,
                    repeat=1,
                ),
                DataRequirementItem(
                    "virial",
                    9,
                    atomic=False,
                    must=False,
                    high_prec=False,
                    repeat=1,
                ),
                DataRequirementItem(
                    "atom_pref",
                    1,
                    atomic=True,
                    must=False,
                    high_prec=False,
                    repeat=3,
                ),
                DataRequirementItem(
                    "atom_ener",
                    1,
                    atomic=True,
                    must=False,
                    high_prec=False,
                    repeat=1,
                ),
                DataRequirementItem(
                    "drdq",
                    2 * 3,
                    atomic=True,
                    must=False,
                    high_prec=False,
                    repeat=1,
                ),
            ],
        )

    def test_build_loss(self) -> None:
        natoms = tf.constant([6, 6])
        model_dict = {
            "energy": tf.zeros((1, 1), dtype=tf.float64),
            "force": tf.random.uniform((1, 6 * 3), dtype=tf.float64),
            "virial": tf.zeros((1, 9), dtype=tf.float64),
            "atom_ener": tf.zeros((1, 6), dtype=tf.float64),
            "atom_virial": tf.zeros((1, 6 * 9), dtype=tf.float64),
            "atom_pref": tf.zeros((1, 6), dtype=tf.float64),
        }
        label_dict = {
            "energy": tf.zeros((1, 1), dtype=tf.float64),
            "force": tf.random.uniform((1, 6 * 3), dtype=tf.float64),
            "virial": tf.zeros((1, 9), dtype=tf.float64),
            "atom_ener": tf.zeros((1, 6), dtype=tf.float64),
            "atom_virial": tf.zeros((1, 6 * 9), dtype=tf.float64),
            "atom_pref": tf.zeros((1, 6), dtype=tf.float64),
            "find_energy": 0.0,
            "find_force": 1.0,
            "find_virial": 0.0,
            "find_atom_ener": 0.0,
            "find_atom_virial": 0.0,
            "find_atom_pref": 0.0,
            "drdq": tf.random.uniform((1, 6 * 3 * 2), dtype=tf.float64),
            "find_drdq": 1.0,
        }
        t_total_loss, more_loss = self.loss.build(
            tf.constant(1e-3),
            natoms,
            model_dict,
            label_dict,
            "_test_gf_loss",
        )
        with self.cached_session() as sess:
            total_loss, gen_force_loss, force, force_hat, drdq = sess.run(
                [
                    t_total_loss,
                    more_loss["l2_gen_force_loss"],
                    model_dict["force"],
                    label_dict["force"],
                    label_dict["drdq"],
                ]
            )

        force = np.reshape(force, (1, 6 * 3))
        force_hat = np.reshape(force_hat, (1, 6 * 3))
        drdq = np.reshape(drdq, (1, 6 * 3, 2))
        gen_force = np.dot(force, drdq)
        gen_force_hat = np.dot(force_hat, drdq)
        np_loss = np.mean(np.square(gen_force - gen_force_hat))
        assert np.allclose(total_loss, np_loss, 6)
        assert np.allclose(gen_force_loss, np_loss, 6)
