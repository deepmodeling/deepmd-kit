# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

from deepmd.common import (
    make_default_mesh,
)
from deepmd.dpmodel.utils.nlist import (
    build_neighbor_list,
    extend_coord_with_ghosts,
)

from ..common import (
    INSTALLED_PT,
    INSTALLED_TF,
)

if INSTALLED_PT:
    import torch

    from deepmd.pt.utils.env import DEVICE as PT_DEVICE
    from deepmd.pt.utils.nlist import build_neighbor_list as build_neighbor_list_pt
    from deepmd.pt.utils.nlist import (
        extend_coord_with_ghosts as extend_coord_with_ghosts_pt,
    )
if INSTALLED_TF:
    from deepmd.tf.env import (
        GLOBAL_TF_FLOAT_PRECISION,
        tf,
    )


class DescriptorTest:
    """Useful utilities for descriptor tests."""

    def build_tf_descriptor(self, obj, natoms, coords, atype, box, suffix):
        t_coord = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None], name="i_coord")
        t_type = tf.placeholder(tf.int32, [None], name="i_type")
        t_natoms = tf.placeholder(tf.int32, natoms.shape, name="i_natoms")
        t_box = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [9], name="i_box")
        t_mesh = tf.placeholder(tf.int32, [None], name="i_mesh")
        t_des = obj.build(
            t_coord,
            t_type,
            t_natoms,
            t_box,
            t_mesh,
            {},
            suffix=suffix,
        )
        # ensure get_dim_out gives the correct shape
        t_des = tf.reshape(t_des, [1, natoms[0], obj.get_dim_out()])
        return [t_des], {
            t_coord: coords,
            t_type: atype,
            t_natoms: natoms,
            t_box: box,
            t_mesh: make_default_mesh(True, False),
        }

    def eval_dp_descriptor(
        self, dp_obj: Any, natoms, coords, atype, box, mixed_types: bool = False
    ) -> Any:
        ext_coords, ext_atype, mapping = extend_coord_with_ghosts(
            coords.reshape(1, -1, 3),
            atype.reshape(1, -1),
            box.reshape(1, 3, 3),
            dp_obj.get_rcut(),
        )
        nlist = build_neighbor_list(
            ext_coords,
            ext_atype,
            natoms[0],
            dp_obj.get_rcut(),
            dp_obj.get_sel(),
            distinguish_types=(not mixed_types),
        )
        return dp_obj(ext_coords, ext_atype, nlist=nlist, mapping=mapping)

    def eval_pt_descriptor(
        self, pt_obj: Any, natoms, coords, atype, box, mixed_types: bool = False
    ) -> Any:
        ext_coords, ext_atype, mapping = extend_coord_with_ghosts_pt(
            torch.from_numpy(coords).to(PT_DEVICE).reshape(1, -1, 3),
            torch.from_numpy(atype).to(PT_DEVICE).reshape(1, -1),
            torch.from_numpy(box).to(PT_DEVICE).reshape(1, 3, 3),
            pt_obj.get_rcut(),
        )
        nlist = build_neighbor_list_pt(
            ext_coords,
            ext_atype,
            natoms[0],
            pt_obj.get_rcut(),
            pt_obj.get_sel(),
            distinguish_types=(not mixed_types),
        )
        return [
            x.detach().cpu().numpy() if torch.is_tensor(x) else x
            for x in pt_obj(ext_coords, ext_atype, nlist=nlist, mapping=mapping)
        ]
