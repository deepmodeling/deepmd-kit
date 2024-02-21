# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Optional,
)

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


class EnerStdLoss(Loss):
    r"""Standard loss function for DP models.

    Parameters
    ----------
    starter_learning_rate : float
        The learning rate at the start of the training.
    start_pref_e : float
        The prefactor of energy loss at the start of the training.
    limit_pref_e : float
        The prefactor of energy loss at the end of the training.
    start_pref_f : float
        The prefactor of force loss at the start of the training.
    limit_pref_f : float
        The prefactor of force loss at the end of the training.
    start_pref_v : float
        The prefactor of virial loss at the start of the training.
    limit_pref_v : float
        The prefactor of virial loss at the end of the training.
    start_pref_ae : float
        The prefactor of atomic energy loss at the start of the training.
    limit_pref_ae : float
        The prefactor of atomic energy loss at the end of the training.
    start_pref_pf : float
        The prefactor of atomic prefactor force loss at the start of the training.
    limit_pref_pf : float
        The prefactor of atomic prefactor force loss at the end of the training.
    relative_f : float
        If provided, relative force error will be used in the loss. The difference
        of force will be normalized by the magnitude of the force in the label with
        a shift given by relative_f
    enable_atom_ener_coeff : bool
        if true, the energy will be computed as \sum_i c_i E_i
    start_pref_gf : float
        The prefactor of generalized force loss at the start of the training.
    limit_pref_gf : float
        The prefactor of generalized force loss at the end of the training.
    numb_generalized_coord : int
        The dimension of generalized coordinates.
    **kwargs
        Other keyword arguments.
    """

    def __init__(
        self,
        starter_learning_rate: float,
        start_pref_e: float = 0.02,
        limit_pref_e: float = 1.00,
        start_pref_f: float = 1000,
        limit_pref_f: float = 1.00,
        start_pref_v: float = 0.0,
        limit_pref_v: float = 0.0,
        start_pref_ae: float = 0.0,
        limit_pref_ae: float = 0.0,
        start_pref_pf: float = 0.0,
        limit_pref_pf: float = 0.0,
        relative_f: Optional[float] = None,
        enable_atom_ener_coeff: bool = False,
        start_pref_gf: float = 0.0,
        limit_pref_gf: float = 0.0,
        numb_generalized_coord: int = 0,
        **kwargs,
    ) -> None:
        self.starter_learning_rate = starter_learning_rate
        self.start_pref_e = start_pref_e
        self.limit_pref_e = limit_pref_e
        self.start_pref_f = start_pref_f
        self.limit_pref_f = limit_pref_f
        self.start_pref_v = start_pref_v
        self.limit_pref_v = limit_pref_v
        self.start_pref_ae = start_pref_ae
        self.limit_pref_ae = limit_pref_ae
        self.start_pref_pf = start_pref_pf
        self.limit_pref_pf = limit_pref_pf
        self.relative_f = relative_f
        self.enable_atom_ener_coeff = enable_atom_ener_coeff
        self.start_pref_gf = start_pref_gf
        self.limit_pref_gf = limit_pref_gf
        self.numb_generalized_coord = numb_generalized_coord
        self.has_e = self.start_pref_e != 0.0 or self.limit_pref_e != 0.0
        self.has_f = self.start_pref_f != 0.0 or self.limit_pref_f != 0.0
        self.has_v = self.start_pref_v != 0.0 or self.limit_pref_v != 0.0
        self.has_ae = self.start_pref_ae != 0.0 or self.limit_pref_ae != 0.0
        self.has_pf = self.start_pref_pf != 0.0 or self.limit_pref_pf != 0.0
        self.has_gf = self.start_pref_gf != 0.0 or self.limit_pref_gf != 0.0
        if self.has_gf and self.numb_generalized_coord < 1:
            raise RuntimeError(
                "When generalized force loss is used, the dimension of generalized coordinates should be larger than 0"
            )
        # data required
        add_data_requirement("energy", 1, atomic=False, must=False, high_prec=True)
        add_data_requirement("force", 3, atomic=True, must=False, high_prec=False)
        add_data_requirement("virial", 9, atomic=False, must=False, high_prec=False)
        add_data_requirement("atom_ener", 1, atomic=True, must=False, high_prec=False)
        add_data_requirement(
            "atom_pref", 1, atomic=True, must=False, high_prec=False, repeat=3
        )
        # drdq: the partial derivative of atomic coordinates w.r.t. generalized coordinates
        # TODO: could numb_generalized_coord decided from the training data?
        if self.has_gf > 0:
            add_data_requirement(
                "drdq",
                self.numb_generalized_coord * 3,
                atomic=True,
                must=False,
                high_prec=False,
            )
        if self.enable_atom_ener_coeff:
            add_data_requirement(
                "atom_ener_coeff",
                1,
                atomic=True,
                must=False,
                high_prec=False,
                default=1.0,
            )

    def build(self, learning_rate, natoms, model_dict, label_dict, suffix):
        energy = model_dict["energy"]
        force = model_dict["force"]
        virial = model_dict["virial"]
        atom_ener = model_dict["atom_ener"]
        energy_hat = label_dict["energy"]
        force_hat = label_dict["force"]
        virial_hat = label_dict["virial"]
        atom_ener_hat = label_dict["atom_ener"]
        atom_pref = label_dict["atom_pref"]
        find_energy = label_dict["find_energy"]
        find_force = label_dict["find_force"]
        find_virial = label_dict["find_virial"]
        find_atom_ener = label_dict["find_atom_ener"]
        find_atom_pref = label_dict["find_atom_pref"]
        if self.has_gf:
            drdq = label_dict["drdq"]
            find_drdq = label_dict["find_drdq"]

        if self.enable_atom_ener_coeff:
            # when ener_coeff (\nu) is defined, the energy is defined as
            # E = \sum_i \nu_i E_i
            # instead of the sum of atomic energies.
            #
            # A case is that we want to train reaction energy
            # A + B -> C + D
            # E = - E(A) - E(B) + E(C) + E(D)
            # A, B, C, D could be put far away from each other
            atom_ener_coeff = label_dict["atom_ener_coeff"]
            atom_ener_coeff = tf.reshape(atom_ener_coeff, tf.shape(atom_ener))
            energy = tf.reduce_sum(atom_ener_coeff * atom_ener, 1)
        if self.has_e:
            l2_ener_loss = tf.reduce_mean(
                tf.square(energy - energy_hat), name="l2_" + suffix
            )

        if self.has_f or self.has_pf or self.relative_f or self.has_gf:
            force_reshape = tf.reshape(force, [-1])
            force_hat_reshape = tf.reshape(force_hat, [-1])
            diff_f = force_hat_reshape - force_reshape

        if self.relative_f is not None:
            force_hat_3 = tf.reshape(force_hat, [-1, 3])
            norm_f = tf.reshape(tf.norm(force_hat_3, axis=1), [-1, 1]) + self.relative_f
            diff_f_3 = tf.reshape(diff_f, [-1, 3])
            diff_f_3 = diff_f_3 / norm_f
            diff_f = tf.reshape(diff_f_3, [-1])

        if self.has_f:
            l2_force_loss = tf.reduce_mean(tf.square(diff_f), name="l2_force_" + suffix)

        if self.has_pf:
            atom_pref_reshape = tf.reshape(atom_pref, [-1])
            l2_pref_force_loss = tf.reduce_mean(
                tf.multiply(tf.square(diff_f), atom_pref_reshape),
                name="l2_pref_force_" + suffix,
            )

        if self.has_gf:
            drdq = label_dict["drdq"]
            force_reshape_nframes = tf.reshape(force, [-1, natoms[0] * 3])
            force_hat_reshape_nframes = tf.reshape(force_hat, [-1, natoms[0] * 3])
            drdq_reshape = tf.reshape(
                drdq, [-1, natoms[0] * 3, self.numb_generalized_coord]
            )
            gen_force_hat = tf.einsum(
                "bij,bi->bj", drdq_reshape, force_hat_reshape_nframes
            )
            gen_force = tf.einsum("bij,bi->bj", drdq_reshape, force_reshape_nframes)
            diff_gen_force = gen_force_hat - gen_force
            l2_gen_force_loss = tf.reduce_mean(
                tf.square(diff_gen_force), name="l2_gen_force_" + suffix
            )

        if self.has_v:
            virial_reshape = tf.reshape(virial, [-1])
            virial_hat_reshape = tf.reshape(virial_hat, [-1])
            l2_virial_loss = tf.reduce_mean(
                tf.square(virial_hat_reshape - virial_reshape),
                name="l2_virial_" + suffix,
            )

        if self.has_ae:
            atom_ener_reshape = tf.reshape(atom_ener, [-1])
            atom_ener_hat_reshape = tf.reshape(atom_ener_hat, [-1])
            l2_atom_ener_loss = tf.reduce_mean(
                tf.square(atom_ener_hat_reshape - atom_ener_reshape),
                name="l2_atom_ener_" + suffix,
            )

        atom_norm = 1.0 / global_cvt_2_tf_float(natoms[0])
        atom_norm_ener = 1.0 / global_cvt_2_ener_float(natoms[0])
        pref_e = global_cvt_2_ener_float(
            find_energy
            * (
                self.limit_pref_e
                + (self.start_pref_e - self.limit_pref_e)
                * learning_rate
                / self.starter_learning_rate
            )
        )
        pref_f = global_cvt_2_tf_float(
            find_force
            * (
                self.limit_pref_f
                + (self.start_pref_f - self.limit_pref_f)
                * learning_rate
                / self.starter_learning_rate
            )
        )
        pref_v = global_cvt_2_tf_float(
            find_virial
            * (
                self.limit_pref_v
                + (self.start_pref_v - self.limit_pref_v)
                * learning_rate
                / self.starter_learning_rate
            )
        )
        pref_ae = global_cvt_2_tf_float(
            find_atom_ener
            * (
                self.limit_pref_ae
                + (self.start_pref_ae - self.limit_pref_ae)
                * learning_rate
                / self.starter_learning_rate
            )
        )
        pref_pf = global_cvt_2_tf_float(
            find_atom_pref
            * (
                self.limit_pref_pf
                + (self.start_pref_pf - self.limit_pref_pf)
                * learning_rate
                / self.starter_learning_rate
            )
        )
        if self.has_gf:
            pref_gf = global_cvt_2_tf_float(
                find_drdq
                * (
                    self.limit_pref_gf
                    + (self.start_pref_gf - self.limit_pref_gf)
                    * learning_rate
                    / self.starter_learning_rate
                )
            )

        l2_loss = 0
        more_loss = {}
        if self.has_e:
            l2_loss += atom_norm_ener * (pref_e * l2_ener_loss)
            more_loss["l2_ener_loss"] = self.display_if_exist(l2_ener_loss, find_energy)
        if self.has_f:
            l2_loss += global_cvt_2_ener_float(pref_f * l2_force_loss)
            more_loss["l2_force_loss"] = self.display_if_exist(
                l2_force_loss, find_force
            )
        if self.has_v:
            l2_loss += global_cvt_2_ener_float(atom_norm * (pref_v * l2_virial_loss))
            more_loss["l2_virial_loss"] = self.display_if_exist(
                l2_virial_loss, find_virial
            )
        if self.has_ae:
            l2_loss += global_cvt_2_ener_float(pref_ae * l2_atom_ener_loss)
            more_loss["l2_atom_ener_loss"] = self.display_if_exist(
                l2_atom_ener_loss, find_atom_ener
            )
        if self.has_pf:
            l2_loss += global_cvt_2_ener_float(pref_pf * l2_pref_force_loss)
            more_loss["l2_pref_force_loss"] = self.display_if_exist(
                l2_pref_force_loss, find_atom_pref
            )
        if self.has_gf:
            l2_loss += global_cvt_2_ener_float(pref_gf * l2_gen_force_loss)
            more_loss["l2_gen_force_loss"] = self.display_if_exist(
                l2_gen_force_loss, find_drdq
            )

        # only used when tensorboard was set as true
        self.l2_loss_summary = tf.summary.scalar("l2_loss_" + suffix, tf.sqrt(l2_loss))
        if self.has_e:
            self.l2_loss_ener_summary = tf.summary.scalar(
                "l2_ener_loss_" + suffix,
                global_cvt_2_tf_float(tf.sqrt(l2_ener_loss))
                / global_cvt_2_tf_float(natoms[0]),
            )
        if self.has_f:
            self.l2_loss_force_summary = tf.summary.scalar(
                "l2_force_loss_" + suffix, tf.sqrt(l2_force_loss)
            )
        if self.has_v:
            self.l2_loss_virial_summary = tf.summary.scalar(
                "l2_virial_loss_" + suffix,
                tf.sqrt(l2_virial_loss) / global_cvt_2_tf_float(natoms[0]),
            )
        if self.has_ae:
            self.l2_loss_atom_ener_summary = tf.summary.scalar(
                "l2_atom_ener_loss_" + suffix, tf.sqrt(l2_atom_ener_loss)
            )
        if self.has_pf:
            self.l2_loss_pref_force_summary = tf.summary.scalar(
                "l2_pref_force_loss_" + suffix, tf.sqrt(l2_pref_force_loss)
            )
        if self.has_gf:
            self.l2_loss_gf_summary = tf.summary.scalar(
                "l2_gen_force_loss_" + suffix, tf.sqrt(l2_gen_force_loss)
            )

        self.l2_l = l2_loss
        self.l2_more = more_loss
        return l2_loss, more_loss

    def eval(self, sess, feed_dict, natoms):
        placeholder = self.l2_l
        run_data = [
            self.l2_l,
            self.l2_more["l2_ener_loss"] if self.has_e else placeholder,
            self.l2_more["l2_force_loss"] if self.has_f else placeholder,
            self.l2_more["l2_virial_loss"] if self.has_v else placeholder,
            self.l2_more["l2_atom_ener_loss"] if self.has_ae else placeholder,
            self.l2_more["l2_pref_force_loss"] if self.has_pf else placeholder,
            self.l2_more["l2_gen_force_loss"] if self.has_gf else placeholder,
        ]
        error, error_e, error_f, error_v, error_ae, error_pf, error_gf = run_sess(
            sess, run_data, feed_dict=feed_dict
        )
        results = {"natoms": natoms[0], "rmse": np.sqrt(error)}
        if self.has_e:
            results["rmse_e"] = np.sqrt(error_e) / natoms[0]
        if self.has_ae:
            results["rmse_ae"] = np.sqrt(error_ae)
        if self.has_f:
            results["rmse_f"] = np.sqrt(error_f)
        if self.has_v:
            results["rmse_v"] = np.sqrt(error_v) / natoms[0]
        if self.has_pf:
            results["rmse_pf"] = np.sqrt(error_pf)
        if self.has_gf:
            results["rmse_gf"] = np.sqrt(error_gf)
        return results


class EnerSpinLoss(Loss):
    def __init__(
        self,
        starter_learning_rate: float,
        start_pref_e: float = 0.02,
        limit_pref_e: float = 1.00,
        start_pref_fr: float = 1000,
        limit_pref_fr: float = 1.00,
        start_pref_fm: float = 10000,
        limit_pref_fm: float = 10.0,
        start_pref_v: float = 0.0,
        limit_pref_v: float = 0.0,
        start_pref_ae: float = 0.0,
        limit_pref_ae: float = 0.0,
        start_pref_pf: float = 0.0,
        limit_pref_pf: float = 0.0,
        relative_f: Optional[float] = None,
        enable_atom_ener_coeff: bool = False,
        use_spin: Optional[list] = None,
    ) -> None:
        self.starter_learning_rate = starter_learning_rate
        self.start_pref_e = start_pref_e
        self.limit_pref_e = limit_pref_e
        self.start_pref_fr = start_pref_fr
        self.limit_pref_fr = limit_pref_fr
        self.start_pref_fm = start_pref_fm
        self.limit_pref_fm = limit_pref_fm
        self.start_pref_v = start_pref_v
        self.limit_pref_v = limit_pref_v
        self.start_pref_ae = start_pref_ae
        self.limit_pref_ae = limit_pref_ae
        self.start_pref_pf = start_pref_pf
        self.limit_pref_pf = limit_pref_pf
        self.relative_f = relative_f
        self.enable_atom_ener_coeff = enable_atom_ener_coeff
        self.use_spin = use_spin
        self.has_e = self.start_pref_e != 0.0 or self.limit_pref_e != 0.0
        self.has_fr = self.start_pref_fr != 0.0 or self.limit_pref_fr != 0.0
        self.has_fm = self.start_pref_fm != 0.0 or self.limit_pref_fm != 0.0
        self.has_v = self.start_pref_v != 0.0 or self.limit_pref_v != 0.0
        self.has_ae = self.start_pref_ae != 0.0 or self.limit_pref_ae != 0.0
        # data required
        add_data_requirement("energy", 1, atomic=False, must=False, high_prec=True)
        add_data_requirement("force", 3, atomic=True, must=False, high_prec=False)
        add_data_requirement("virial", 9, atomic=False, must=False, high_prec=False)
        add_data_requirement("atom_ener", 1, atomic=True, must=False, high_prec=False)
        add_data_requirement(
            "atom_pref", 1, atomic=True, must=False, high_prec=False, repeat=3
        )
        if self.enable_atom_ener_coeff:
            add_data_requirement(
                "atom_ener_coeff",
                1,
                atomic=True,
                must=False,
                high_prec=False,
                default=1.0,
            )

    def build(self, learning_rate, natoms, model_dict, label_dict, suffix):
        energy_pred = model_dict["energy"]
        force_pred = model_dict["force"]
        virial_pred = model_dict["virial"]
        atom_ener_pred = model_dict["atom_ener"]
        energy_label = label_dict["energy"]
        force_label = label_dict["force"]
        virial_label = label_dict["virial"]
        atom_ener_label = label_dict["atom_ener"]
        atom_pref = label_dict["atom_pref"]
        find_energy = label_dict["find_energy"]
        find_force = label_dict["find_force"]
        find_virial = label_dict["find_virial"]
        find_atom_ener = label_dict["find_atom_ener"]

        if self.enable_atom_ener_coeff:
            # when ener_coeff (\nu) is defined, the energy is defined as
            # E = \sum_i \nu_i E_i
            # instead of the sum of atomic energies.
            #
            # A case is that we want to train reaction energy
            # A + B -> C + D
            # E = - E(A) - E(B) + E(C) + E(D)
            # A, B, C, D could be put far away from each other
            atom_ener_coeff = label_dict["atom_ener_coeff"]
            atom_ener_coeff = tf.reshape(atom_ener_coeff, tf.shape(atom_ener_pred))
            energy_pred = tf.reduce_sum(atom_ener_coeff * atom_ener_pred, 1)
        l2_ener_loss = tf.reduce_mean(
            tf.square(energy_pred - energy_label), name="l2_" + suffix
        )

        # split force to force_r and force_m, compute their respective loss
        real_nloc = tf.reduce_sum(natoms[2 : 2 + len(self.use_spin)])
        virt_nloc = natoms[0] - real_nloc
        force_pred_reshape = tf.reshape(force_pred, [-1, natoms[0] * 3])
        force_label_reshape = tf.reshape(force_label, [-1, natoms[0] * 3])
        force_r_pred = tf.reshape(
            tf.slice(force_pred_reshape, [0, 0], [-1, real_nloc * 3]), [-1]
        )
        force_m_pred = tf.reshape(
            tf.slice(force_pred_reshape, [0, real_nloc * 3], [-1, virt_nloc * 3]), [-1]
        )
        force_r_label = tf.reshape(
            tf.slice(force_label_reshape, [0, 0], [-1, real_nloc * 3]), [-1]
        )
        force_m_label = tf.reshape(
            tf.slice(force_label_reshape, [0, real_nloc * 3], [-1, virt_nloc * 3]), [-1]
        )
        l2_force_r_loss = tf.reduce_mean(
            tf.square(force_r_pred - force_r_label), name="l2_force_real_" + suffix
        )
        l2_force_m_loss = tf.reduce_mean(
            tf.square(force_m_pred - force_m_label), name="l2_force_mag_" + suffix
        )

        virial_pred_reshape = tf.reshape(virial_pred, [-1])
        virial_label_reshape = tf.reshape(virial_label, [-1])
        l2_virial_loss = tf.reduce_mean(
            tf.square(virial_pred_reshape - virial_label_reshape),
            name="l2_virial_" + suffix,
        )

        # need to change. can't get atom_ener_hat?
        atom_ener_pred_reshape = tf.reshape(atom_ener_pred, [-1])
        atom_ener_label_reshape = tf.reshape(atom_ener_label, [-1])
        l2_atom_ener_loss = tf.reduce_mean(
            tf.square(atom_ener_pred_reshape - atom_ener_label_reshape),
            name="l2_atom_ener_" + suffix,
        )

        atom_norm = 1.0 / global_cvt_2_tf_float(natoms[0])
        atom_norm_ener = 1.0 / global_cvt_2_ener_float(natoms[0])
        pref_e = global_cvt_2_ener_float(
            find_energy
            * (
                self.limit_pref_e
                + (self.start_pref_e - self.limit_pref_e)
                * learning_rate
                / self.starter_learning_rate
            )
        )
        pref_fr = global_cvt_2_tf_float(
            find_force
            * (
                self.limit_pref_fr
                + (self.start_pref_fr - self.limit_pref_fr)
                * learning_rate
                / self.starter_learning_rate
            )
        )
        pref_fm = global_cvt_2_tf_float(
            find_force
            * (
                self.limit_pref_fm
                + (self.start_pref_fm - self.limit_pref_fm)
                * learning_rate
                / self.starter_learning_rate
            )
        )
        pref_v = global_cvt_2_tf_float(
            find_virial
            * (
                self.limit_pref_v
                + (self.start_pref_v - self.limit_pref_v)
                * learning_rate
                / self.starter_learning_rate
            )
        )
        pref_ae = global_cvt_2_tf_float(
            find_atom_ener
            * (
                self.limit_pref_ae
                + (self.start_pref_ae - self.limit_pref_ae)
                * learning_rate
                / self.starter_learning_rate
            )
        )

        l2_loss = 0
        more_loss = {}
        if self.has_e:
            l2_loss += atom_norm_ener * (pref_e * l2_ener_loss)
        more_loss["l2_ener_loss"] = self.display_if_exist(l2_ener_loss, find_energy)
        if self.has_fr:
            l2_loss += global_cvt_2_ener_float(pref_fr * l2_force_r_loss)
        more_loss["l2_force_r_loss"] = self.display_if_exist(
            l2_force_r_loss, find_force
        )
        if self.has_fm:
            l2_loss += global_cvt_2_ener_float(pref_fm * l2_force_m_loss)
        more_loss["l2_force_m_loss"] = self.display_if_exist(
            l2_force_m_loss, find_force
        )
        if self.has_v:
            l2_loss += global_cvt_2_ener_float(atom_norm * (pref_v * l2_virial_loss))
        more_loss["l2_virial_loss"] = self.display_if_exist(l2_virial_loss, find_virial)
        if self.has_ae:
            l2_loss += global_cvt_2_ener_float(pref_ae * l2_atom_ener_loss)
        more_loss["l2_atom_ener_loss"] = self.display_if_exist(
            l2_atom_ener_loss, find_atom_ener
        )

        # only used when tensorboard was set as true
        self.l2_loss_summary = tf.summary.scalar("l2_loss", tf.sqrt(l2_loss))
        self.l2_loss_ener_summary = tf.summary.scalar(
            "l2_ener_loss",
            global_cvt_2_tf_float(tf.sqrt(l2_ener_loss))
            / global_cvt_2_tf_float(natoms[0]),
        )
        self.l2_loss_force_r_summary = tf.summary.scalar(
            "l2_force_r_loss", tf.sqrt(l2_force_r_loss)
        )
        self.l2_loss_force_m_summary = tf.summary.scalar(
            "l2_force_m_loss", tf.sqrt(l2_force_m_loss)
        )
        self.l2_loss_virial_summary = tf.summary.scalar(
            "l2_virial_loss", tf.sqrt(l2_virial_loss) / global_cvt_2_tf_float(natoms[0])
        )

        self.l2_l = l2_loss
        self.l2_more = more_loss
        return l2_loss, more_loss

    def eval(self, sess, feed_dict, natoms):
        placeholder = self.l2_l
        run_data = [
            self.l2_l,
            self.l2_more["l2_ener_loss"] if self.has_e else placeholder,
            self.l2_more["l2_force_r_loss"] if self.has_fr else placeholder,
            self.l2_more["l2_force_m_loss"] if self.has_fm else placeholder,
            self.l2_more["l2_virial_loss"] if self.has_v else placeholder,
            self.l2_more["l2_atom_ener_loss"] if self.has_ae else placeholder,
        ]
        error, error_e, error_fr, error_fm, error_v, error_ae = run_sess(
            sess, run_data, feed_dict=feed_dict
        )
        results = {"natoms": natoms[0], "rmse": np.sqrt(error)}
        if self.has_e:
            results["rmse_e"] = np.sqrt(error_e) / natoms[0]
        if self.has_ae:
            results["rmse_ae"] = np.sqrt(error_ae)
        if self.has_fr:
            results["rmse_fr"] = np.sqrt(error_fr)
        if self.has_fm:
            results["rmse_fm"] = np.sqrt(error_fm)
        if self.has_v:
            results["rmse_v"] = np.sqrt(error_v) / natoms[0]
        return results

    def print_header(self):  # depreciated
        prop_fmt = "   %11s %11s"
        print_str = ""
        print_str += prop_fmt % ("rmse_tst", "rmse_trn")
        if self.has_e:
            print_str += prop_fmt % ("rmse_e_tst", "rmse_e_trn")
        if self.has_ae:
            print_str += prop_fmt % ("rmse_ae_tst", "rmse_ae_trn")
        if self.has_fr:
            print_str += prop_fmt % ("rmse_fr_tst", "rmse_fr_trn")
        if self.has_fm:
            print_str += prop_fmt % ("rmse_fm_tst", "rmse_fm_trn")
        if self.has_v:
            print_str += prop_fmt % ("rmse_v_tst", "rmse_v_trn")
        return print_str

    def print_on_training(
        self, tb_writer, cur_batch, sess, natoms, feed_dict_test, feed_dict_batch
    ):  # depreciated
        run_data = [
            self.l2_l,
            self.l2_more["l2_ener_loss"],
            self.l2_more["l2_force_r_loss"],
            self.l2_more["l2_force_m_loss"],
            self.l2_more["l2_virial_loss"],
            self.l2_more["l2_atom_ener_loss"],
        ]

        # first train data
        train_out = run_sess(sess, run_data, feed_dict=feed_dict_batch)
        (
            error_train,
            error_e_train,
            error_fr_train,
            error_fm_train,
            error_v_train,
            error_ae_train,
        ) = train_out

        # than test data, if tensorboard log writter is present, commpute summary
        # and write tensorboard logs
        if tb_writer:
            summary_merged_op = tf.summary.merge(
                [
                    self.l2_loss_summary,
                    self.l2_loss_ener_summary,
                    self.l2_loss_force_r_summary,
                    self.l2_loss_force_m_summary,
                    self.l2_loss_virial_summary,
                ]
            )
            run_data.insert(0, summary_merged_op)

        test_out = run_sess(sess, run_data, feed_dict=feed_dict_test)

        if tb_writer:
            summary = test_out.pop(0)
            tb_writer.add_summary(summary, cur_batch)

        (
            error_test,
            error_e_test,
            error_fr_test,
            error_fm_test,
            error_v_test,
            error_ae_test,
        ) = test_out

        print_str = ""
        prop_fmt = "   %11.2e %11.2e"
        print_str += prop_fmt % (np.sqrt(error_test), np.sqrt(error_train))
        if self.has_e:
            print_str += prop_fmt % (
                np.sqrt(error_e_test) / natoms[0],
                np.sqrt(error_e_train) / natoms[0],
            )
        if self.has_ae:
            print_str += prop_fmt % (np.sqrt(error_ae_test), np.sqrt(error_ae_train))
        if self.has_fr:
            print_str += prop_fmt % (np.sqrt(error_fr_test), np.sqrt(error_fr_train))
        if self.has_fm:
            print_str += prop_fmt % (np.sqrt(error_fm_test), np.sqrt(error_fm_train))
        if self.has_v:
            print_str += prop_fmt % (
                np.sqrt(error_v_test) / natoms[0],
                np.sqrt(error_v_train) / natoms[0],
            )

        return print_str


class EnerDipoleLoss(Loss):
    def __init__(
        self,
        starter_learning_rate: float,
        start_pref_e: float = 0.1,
        limit_pref_e: float = 1.0,
        start_pref_ed: float = 1.0,
        limit_pref_ed: float = 1.0,
    ) -> None:
        self.starter_learning_rate = starter_learning_rate
        self.start_pref_e = start_pref_e
        self.limit_pref_e = limit_pref_e
        self.start_pref_ed = start_pref_ed
        self.limit_pref_ed = limit_pref_ed
        # data required
        add_data_requirement("energy", 1, atomic=False, must=True, high_prec=True)
        add_data_requirement(
            "energy_dipole", 3, atomic=False, must=True, high_prec=False
        )

    def build(self, learning_rate, natoms, model_dict, label_dict, suffix):
        coord = model_dict["coord"]
        energy = model_dict["energy"]
        atom_ener = model_dict["atom_ener"]
        nframes = tf.shape(atom_ener)[0]
        natoms = tf.shape(atom_ener)[1]
        # build energy dipole
        atom_ener0 = atom_ener - tf.reshape(
            tf.tile(
                tf.reshape(energy / global_cvt_2_ener_float(natoms), [-1, 1]),
                [1, natoms],
            ),
            [nframes, natoms],
        )
        coord = tf.reshape(coord, [nframes, natoms, 3])
        atom_ener0 = tf.reshape(atom_ener0, [nframes, 1, natoms])
        ener_dipole = tf.matmul(atom_ener0, coord)
        ener_dipole = tf.reshape(ener_dipole, [nframes, 3])

        energy_hat = label_dict["energy"]
        ener_dipole_hat = label_dict["energy_dipole"]
        find_energy = label_dict["find_energy"]
        find_ener_dipole = label_dict["find_energy_dipole"]

        l2_ener_loss = tf.reduce_mean(
            tf.square(energy - energy_hat), name="l2_" + suffix
        )

        ener_dipole_reshape = tf.reshape(ener_dipole, [-1])
        ener_dipole_hat_reshape = tf.reshape(ener_dipole_hat, [-1])
        l2_ener_dipole_loss = tf.reduce_mean(
            tf.square(ener_dipole_reshape - ener_dipole_hat_reshape),
            name="l2_" + suffix,
        )

        # atom_norm_ener  = 1./ global_cvt_2_ener_float(natoms[0])
        atom_norm_ener = 1.0 / global_cvt_2_ener_float(natoms)
        pref_e = global_cvt_2_ener_float(
            find_energy
            * (
                self.limit_pref_e
                + (self.start_pref_e - self.limit_pref_e)
                * learning_rate
                / self.starter_learning_rate
            )
        )
        pref_ed = global_cvt_2_tf_float(
            find_ener_dipole
            * (
                self.limit_pref_ed
                + (self.start_pref_ed - self.limit_pref_ed)
                * learning_rate
                / self.starter_learning_rate
            )
        )

        l2_loss = 0
        more_loss = {}
        l2_loss += atom_norm_ener * (pref_e * l2_ener_loss)
        l2_loss += global_cvt_2_ener_float(pref_ed * l2_ener_dipole_loss)
        more_loss["l2_ener_loss"] = self.display_if_exist(l2_ener_loss, find_energy)
        more_loss["l2_ener_dipole_loss"] = self.display_if_exist(
            l2_ener_dipole_loss, find_ener_dipole
        )

        self.l2_loss_summary = tf.summary.scalar("l2_loss_" + suffix, tf.sqrt(l2_loss))
        self.l2_loss_ener_summary = tf.summary.scalar(
            "l2_ener_loss_" + suffix,
            tf.sqrt(l2_ener_loss) / global_cvt_2_tf_float(natoms[0]),
        )
        self.l2_ener_dipole_loss_summary = tf.summary.scalar(
            "l2_ener_dipole_loss_" + suffix, tf.sqrt(l2_ener_dipole_loss)
        )

        self.l2_l = l2_loss
        self.l2_more = more_loss
        return l2_loss, more_loss

    def eval(self, sess, feed_dict, natoms):
        run_data = [
            self.l2_l,
            self.l2_more["l2_ener_loss"],
            self.l2_more["l2_ener_dipole_loss"],
        ]
        error, error_e, error_ed = run_sess(sess, run_data, feed_dict=feed_dict)
        results = {
            "natoms": natoms[0],
            "rmse": np.sqrt(error),
            "rmse_e": np.sqrt(error_e) / natoms[0],
            "rmse_ed": np.sqrt(error_ed),
        }
        return results
