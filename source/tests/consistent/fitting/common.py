# SPDX-License-Identifier: LGPL-3.0-or-later


from ..common import (
    INSTALLED_PD,
    INSTALLED_PT,
    INSTALLED_TF,
)

if INSTALLED_PT:
    pass
if INSTALLED_TF:
    from deepmd.tf.env import (
        GLOBAL_TF_FLOAT_PRECISION,
        tf,
    )
if INSTALLED_PD:
    pass


class FittingTest:
    """Useful utilities for descriptor tests."""

    def build_tf_fitting(self, obj, inputs, natoms, atype, fparam, aparam, suffix):
        t_inputs = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None], name="i_inputs")
        t_natoms = tf.placeholder(tf.int32, natoms.shape, name="i_natoms")
        t_atype = tf.placeholder(tf.int32, [None], name="i_atype")
        extras = {}
        feed_dict = {}
        if fparam is not None:
            t_fparam = tf.placeholder(
                GLOBAL_TF_FLOAT_PRECISION, [None], name="i_fparam"
            )
            extras["fparam"] = t_fparam
            feed_dict[t_fparam] = fparam
        if aparam is not None:
            t_aparam = tf.placeholder(
                GLOBAL_TF_FLOAT_PRECISION, [None, None], name="i_aparam"
            )
            extras["aparam"] = t_aparam
            feed_dict[t_aparam] = aparam
        t_out = obj.build(
            t_inputs,
            t_natoms,
            {"atype": t_atype, **extras},
            suffix=suffix,
        )
        return [t_out], {
            t_inputs: inputs,
            t_natoms: natoms,
            t_atype: atype,
            **feed_dict,
        }


class DipoleFittingTest:
    """Useful utilities for descriptor tests."""

    def build_tf_fitting(self, obj, inputs, rot_mat, natoms, atype, fparam, suffix):
        t_inputs = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None], name="i_inputs")
        t_rot_mat = tf.placeholder(
            GLOBAL_TF_FLOAT_PRECISION, rot_mat.shape, name="i_rot_mat"
        )
        t_natoms = tf.placeholder(tf.int32, natoms.shape, name="i_natoms")
        t_atype = tf.placeholder(tf.int32, [None], name="i_atype")
        extras = {}
        feed_dict = {}
        if fparam is not None:
            t_fparam = tf.placeholder(
                GLOBAL_TF_FLOAT_PRECISION, [None], name="i_fparam"
            )
            extras["fparam"] = t_fparam
            feed_dict[t_fparam] = fparam
        t_out = obj.build(
            t_inputs,
            t_rot_mat,
            t_natoms,
            {"atype": t_atype, **extras},
            suffix=suffix,
        )
        return [t_out], {
            t_inputs: inputs,
            t_rot_mat: rot_mat,
            t_natoms: natoms,
            t_atype: atype,
            **feed_dict,
        }
