# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np

from deepmd.tf.common import (
    add_data_requirement,
)
from deepmd.tf.env import (
    global_cvt_2_ener_float,
    global_cvt_2_tf_float,
    tf,
)
from deepmd.tf.utils.sess import (
    run_sess,
)

from .loss import (
    Loss,
)


class DOSLoss(Loss):
    """Loss function for DeepDOS models."""

    def __init__(
        self,
        starter_learning_rate: float,
        numb_dos: int = 500,
        start_pref_dos: float = 1.00,
        limit_pref_dos: float = 1.00,
        start_pref_cdf: float = 1000,
        limit_pref_cdf: float = 1.00,
        start_pref_ados: float = 0.0,
        limit_pref_ados: float = 0.0,
        start_pref_acdf: float = 0.0,
        limit_pref_acdf: float = 0.0,
        protect_value: float = 1e-8,
        log_fit: bool = False,
        **kwargs,
    ) -> None:
        self.starter_learning_rate = starter_learning_rate
        self.numb_dos = numb_dos
        self.protect_value = protect_value
        self.log_fit = log_fit

        self.start_pref_dos = start_pref_dos
        self.limit_pref_dos = limit_pref_dos
        self.start_pref_cdf = start_pref_cdf
        self.limit_pref_cdf = limit_pref_cdf

        self.start_pref_ados = start_pref_ados
        self.limit_pref_ados = limit_pref_ados
        self.start_pref_acdf = start_pref_acdf
        self.limit_pref_acdf = limit_pref_acdf

        self.has_dos = self.start_pref_dos != 0.0 or self.limit_pref_dos != 0.0
        self.has_cdf = self.start_pref_cdf != 0.0 or self.limit_pref_cdf != 0.0
        self.has_ados = self.start_pref_ados != 0.0 or self.limit_pref_ados != 0.0
        self.has_acdf = self.start_pref_acdf != 0.0 or self.limit_pref_acdf != 0.0
        # data required
        add_data_requirement(
            "dos", self.numb_dos, atomic=False, must=True, high_prec=True
        )
        add_data_requirement(
            "atom_dos", self.numb_dos, atomic=True, must=False, high_prec=True
        )

    def build(self, learning_rate, natoms, model_dict, label_dict, suffix):
        dos = model_dict["dos"]
        atom_dos = model_dict["atom_dos"]

        dos_hat = label_dict["dos"]
        atom_dos_hat = label_dict["atom_dos"]

        find_dos = label_dict["find_dos"]
        find_atom_dos = label_dict["find_atom_dos"]

        dos_reshape = tf.reshape(dos, [-1, self.numb_dos])
        dos_hat_reshape = tf.reshape(dos_hat, [-1, self.numb_dos])
        diff_dos = dos_hat_reshape - dos_reshape
        if self.has_dos:
            l2_dos_loss = tf.reduce_mean(tf.square(diff_dos), name="l2_dos_" + suffix)
        if self.has_cdf:
            cdf = tf.cumsum(dos_reshape, axis=1)
            cdf_hat = tf.cumsum(dos_hat_reshape, axis=1)
            diff_cdf = cdf_hat - cdf
            l2_cdf_loss = tf.reduce_mean(tf.square(diff_cdf), name="l2_cdf_" + suffix)

        atom_dos_reshape = tf.reshape(atom_dos, [-1, self.numb_dos])
        atom_dos_hat_reshape = tf.reshape(atom_dos_hat, [-1, self.numb_dos])
        diff_atom_dos = atom_dos_hat_reshape - atom_dos_reshape
        if self.has_ados:
            l2_atom_dos_loss = tf.reduce_mean(
                tf.square(diff_atom_dos), name="l2_ados_" + suffix
            )
        if self.has_acdf:
            atom_cdf = tf.cumsum(atom_dos_reshape, axis=1)
            atom_cdf_hat = tf.cumsum(atom_dos_hat_reshape, axis=1)
            diff_atom_cdf = atom_cdf_hat - atom_cdf
            l2_atom_cdf_loss = tf.reduce_mean(
                tf.square(diff_atom_cdf), name="l2_acdf_" + suffix
            )

        atom_norm = 1.0 / global_cvt_2_tf_float(natoms[0])
        atom_norm_ener = 1.0 / global_cvt_2_ener_float(natoms[0])
        pref_dos = global_cvt_2_ener_float(
            find_dos
            * (
                self.limit_pref_dos
                + (self.start_pref_dos - self.limit_pref_dos)
                * learning_rate
                / self.starter_learning_rate
            )
        )
        pref_cdf = global_cvt_2_tf_float(
            find_dos
            * (
                self.limit_pref_cdf
                + (self.start_pref_cdf - self.limit_pref_cdf)
                * learning_rate
                / self.starter_learning_rate
            )
        )
        pref_ados = global_cvt_2_tf_float(
            find_atom_dos
            * (
                self.limit_pref_ados
                + (self.start_pref_ados - self.limit_pref_ados)
                * learning_rate
                / self.starter_learning_rate
            )
        )
        pref_acdf = global_cvt_2_tf_float(
            find_atom_dos
            * (
                self.limit_pref_acdf
                + (self.start_pref_acdf - self.limit_pref_acdf)
                * learning_rate
                / self.starter_learning_rate
            )
        )

        l2_loss = 0
        more_loss = {}
        if self.has_dos:
            l2_loss += atom_norm_ener * (pref_dos * l2_dos_loss)
            more_loss["l2_dos_loss"] = self.display_if_exist(l2_dos_loss, find_dos)
        if self.has_cdf:
            l2_loss += atom_norm_ener * (pref_cdf * l2_cdf_loss)
            more_loss["l2_cdf_loss"] = self.display_if_exist(l2_cdf_loss, find_dos)
        if self.has_ados:
            l2_loss += global_cvt_2_ener_float(pref_ados * l2_atom_dos_loss)
            more_loss["l2_atom_dos_loss"] = self.display_if_exist(
                l2_atom_dos_loss, find_atom_dos
            )
        if self.has_acdf:
            l2_loss += global_cvt_2_ener_float(pref_acdf * l2_atom_cdf_loss)
            more_loss["l2_atom_cdf_loss"] = self.display_if_exist(
                l2_atom_cdf_loss, find_atom_dos
            )

        # only used when tensorboard was set as true
        self.l2_loss_summary = tf.summary.scalar("l2_loss_" + suffix, tf.sqrt(l2_loss))
        if self.has_dos:
            self.l2_loss_dos_summary = tf.summary.scalar(
                "l2_dos_loss_" + suffix,
                global_cvt_2_tf_float(tf.sqrt(l2_dos_loss))
                / global_cvt_2_tf_float(natoms[0]),
            )
        if self.has_cdf:
            self.l2_loss_cdf_summary = tf.summary.scalar(
                "l2_cdf_loss_" + suffix,
                global_cvt_2_tf_float(tf.sqrt(l2_cdf_loss))
                / global_cvt_2_tf_float(natoms[0]),
            )
        if self.has_ados:
            self.l2_loss_ados_summary = tf.summary.scalar(
                "l2_atom_dos_loss_" + suffix,
                global_cvt_2_tf_float(tf.sqrt(l2_atom_dos_loss))
                / global_cvt_2_tf_float(natoms[0]),
            )
        if self.has_acdf:
            self.l2_loss_acdf_summary = tf.summary.scalar(
                "l2_atom_cdf_loss_" + suffix,
                global_cvt_2_tf_float(tf.sqrt(l2_atom_cdf_loss))
                / global_cvt_2_tf_float(natoms[0]),
            )

        self.l2_l = l2_loss
        self.l2_more = more_loss
        return l2_loss, more_loss

    def eval(self, sess, feed_dict, natoms):
        placeholder = self.l2_l
        run_data = [
            self.l2_l,
            self.l2_more["l2_dos_loss"] if self.has_dos else placeholder,
            self.l2_more["l2_cdf_loss"] if self.has_cdf else placeholder,
            self.l2_more["l2_atom_dos_loss"] if self.has_ados else placeholder,
            self.l2_more["l2_atom_cdf_loss"] if self.has_acdf else placeholder,
        ]
        error, error_dos, error_cdf, error_ados, error_acdf = run_sess(
            sess, run_data, feed_dict=feed_dict
        )
        results = {"natoms": natoms[0], "rmse": np.sqrt(error)}
        if self.has_dos:
            results["rmse_dos"] = np.sqrt(error_dos) / natoms[0]
        if self.has_ados:
            results["rmse_ados"] = np.sqrt(error_ados)
        if self.has_cdf:
            results["rmse_cdf"] = np.sqrt(error_cdf) / natoms[0]
        if self.has_acdf:
            results["rmse_acdf"] = np.sqrt(error_acdf)

        return results
