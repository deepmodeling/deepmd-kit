# SPDX-License-Identifier: LGPL-3.0-or-later


from ..common import (
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


class FittingTest:
    """Useful utilities for descriptor tests."""

    def build_tf_fitting(self, obj, inputs, natoms, atype, suffix):
        t_inputs = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None], name="i_inputs")
        t_natoms = tf.placeholder(tf.int32, natoms.shape, name="i_natoms")
        t_atype = tf.placeholder(tf.int32, [None], name="i_atype")
        t_des = obj.build(
            t_inputs,
            t_natoms,
            {"atype": t_atype},
            suffix=suffix,
        )
        return [t_des], {
            t_inputs: inputs,
            t_natoms: natoms,
            t_atype: atype,
        }
