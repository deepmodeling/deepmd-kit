# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

from deepmd.common import (
    make_default_mesh,
)

from ..common import (
    INSTALLED_PT,
    INSTALLED_TF,
)

if INSTALLED_PT:
    from deepmd.pt.utils.utils import to_numpy_array as torch_to_numpy
    from deepmd.pt.utils.utils import to_torch_tensor as numpy_to_torch
if INSTALLED_TF:
    from deepmd.tf.env import (
        GLOBAL_TF_FLOAT_PRECISION,
        tf,
    )


class ModelTest:
    """Useful utilities for model tests."""

    def build_tf_model(self, obj, natoms, coords, atype, box, suffix):
        t_coord = tf.placeholder(
            GLOBAL_TF_FLOAT_PRECISION, [None, None, None], name="i_coord"
        )
        t_type = tf.placeholder(tf.int32, [None, None], name="i_type")
        t_natoms = tf.placeholder(tf.int32, natoms.shape, name="i_natoms")
        t_box = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None, 9], name="i_box")
        t_mesh = tf.placeholder(tf.int32, [None], name="i_mesh")
        ret = obj.build(
            t_coord,
            t_type,
            t_natoms,
            t_box,
            t_mesh,
            {},
            suffix=suffix,
        )
        return [ret["energy"], ret["atom_ener"]], {
            t_coord: coords,
            t_type: atype,
            t_natoms: natoms,
            t_box: box,
            t_mesh: make_default_mesh(True, False),
        }

    def eval_dp_model(self, dp_obj: Any, natoms, coords, atype, box) -> Any:
        return dp_obj(coords, atype, box=box)

    def eval_pt_model(self, pt_obj: Any, natoms, coords, atype, box) -> Any:
        return {
            kk: torch_to_numpy(vv)
            for kk, vv in pt_obj(
                numpy_to_torch(coords),
                numpy_to_torch(atype),
                box=numpy_to_torch(box),
            ).items()
        }
