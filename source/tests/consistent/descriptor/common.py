# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

import numpy as np

from deepmd.common import (
    make_default_mesh,
)
from deepmd.dpmodel.common import (
    to_numpy_array,
)
from deepmd.dpmodel.utils.nlist import (
    build_neighbor_list,
    extend_coord_with_ghosts,
)

from ..common import (
    INSTALLED_ARRAY_API_STRICT,
    INSTALLED_JAX,
    INSTALLED_PD,
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
if INSTALLED_JAX:
    from deepmd.jax.env import (
        jnp,
    )
if INSTALLED_ARRAY_API_STRICT:
    import array_api_strict

if INSTALLED_PD:
    import paddle

    from deepmd.pd.utils.env import DEVICE as PD_DEVICE
    from deepmd.pd.utils.nlist import build_neighbor_list as build_neighbor_list_pd
    from deepmd.pd.utils.nlist import (
        extend_coord_with_ghosts as extend_coord_with_ghosts_pd,
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
        return [t_des, obj.get_rot_mat()], {
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

    def eval_jax_descriptor(
        self, jax_obj: Any, natoms, coords, atype, box, mixed_types: bool = False
    ) -> Any:
        ext_coords, ext_atype, mapping = extend_coord_with_ghosts(
            jnp.array(coords).reshape(1, -1, 3),
            jnp.array(atype).reshape(1, -1),
            jnp.array(box).reshape(1, 3, 3),
            jax_obj.get_rcut(),
        )
        nlist = build_neighbor_list(
            ext_coords,
            ext_atype,
            natoms[0],
            jax_obj.get_rcut(),
            jax_obj.get_sel(),
            distinguish_types=(not mixed_types),
        )
        return [
            np.asarray(x) if isinstance(x, jnp.ndarray) else x
            for x in jax_obj(ext_coords, ext_atype, nlist=nlist, mapping=mapping)
        ]

    def eval_pd_descriptor(
        self, pd_obj: Any, natoms, coords, atype, box, mixed_types: bool = False
    ) -> Any:
        ext_coords, ext_atype, mapping = extend_coord_with_ghosts_pd(
            paddle.to_tensor(coords).to(PD_DEVICE).reshape([1, -1, 3]),
            paddle.to_tensor(atype).to(PD_DEVICE).reshape([1, -1]),
            paddle.to_tensor(box).to(PD_DEVICE).reshape([1, 3, 3]),
            pd_obj.get_rcut(),
        )
        nlist = build_neighbor_list_pd(
            ext_coords,
            ext_atype,
            natoms[0],
            pd_obj.get_rcut(),
            pd_obj.get_sel(),
            distinguish_types=(not mixed_types),
        )
        return [
            x.detach().cpu().numpy() if paddle.is_tensor(x) else x
            for x in pd_obj(ext_coords, ext_atype, nlist=nlist, mapping=mapping)
        ]

    def eval_array_api_strict_descriptor(
        self,
        array_api_strict_obj: Any,
        natoms,
        coords,
        atype,
        box,
        mixed_types: bool = False,
    ) -> Any:
        ext_coords, ext_atype, mapping = extend_coord_with_ghosts(
            array_api_strict.asarray(coords.reshape(1, -1, 3)),
            array_api_strict.asarray(atype.reshape(1, -1)),
            array_api_strict.asarray(box.reshape(1, 3, 3)),
            array_api_strict_obj.get_rcut(),
        )
        nlist = build_neighbor_list(
            ext_coords,
            ext_atype,
            natoms[0],
            array_api_strict_obj.get_rcut(),
            array_api_strict_obj.get_sel(),
            distinguish_types=(not mixed_types),
        )
        return [
            to_numpy_array(x) if hasattr(x, "__array_namespace__") else x
            for x in array_api_strict_obj(
                ext_coords, ext_atype, nlist=nlist, mapping=mapping
            )
        ]
