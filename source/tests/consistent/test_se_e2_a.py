# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from typing import (
    Any,
    ClassVar,
    Tuple,
)

import numpy as np
import torch

from deepmd.common import (
    make_default_mesh,
)
from deepmd.dpmodel.descriptor.se_e2_a import DescrptSeA as DescrptSeADP
from deepmd.dpmodel.utils.nlist import (
    build_neighbor_list,
    extend_coord_with_ghosts,
)
from deepmd.env import (
    GLOBAL_NP_FLOAT_PRECISION,
)
from deepmd.pt.model.descriptor.se_a import DescrptSeA as DescrptSeAPT
from deepmd.pt.utils.env import DEVICE as PT_DEVICE
from deepmd.pt.utils.nlist import build_neighbor_list as build_neighbor_list_pt
from deepmd.pt.utils.nlist import (
    extend_coord_with_ghosts as extend_coord_with_ghosts_pt,
)
from deepmd.tf.descriptor.se_a import DescrptSeA as DescrptSeATF
from deepmd.tf.env import (
    GLOBAL_TF_FLOAT_PRECISION,
    tf,
)
from deepmd.utils.argcheck import (
    descrpt_se_a_args,
)

from .common import (
    CommonTest,
)


class CommonTestSeATest(CommonTest):
    tf_class = DescrptSeATF
    dp_class = DescrptSeADP
    pt_class = DescrptSeAPT
    args = descrpt_se_a_args()

    def setUp(self):
        CommonTest.setUp(self)

        self.ntypes = 2
        self.coords = np.array(
            [
                12.83,
                2.56,
                2.18,
                12.09,
                2.87,
                2.74,
                00.25,
                3.32,
                1.68,
                3.36,
                3.00,
                1.81,
                3.51,
                2.51,
                2.60,
                4.27,
                3.22,
                1.56,
            ],
            dtype=GLOBAL_NP_FLOAT_PRECISION,
        )
        self.atype = np.array([0, 1, 1, 0, 1, 1], dtype=np.int32)
        self.box = np.array(
            [13.0, 0.0, 0.0, 0.0, 13.0, 0.0, 0.0, 0.0, 13.0],
            dtype=GLOBAL_NP_FLOAT_PRECISION,
        )
        self.natoms = np.array([6, 6, 2, 4], dtype=np.int32)

    def build_tf(self, obj: Any, suffix: str) -> Tuple[list, dict]:
        t_coord = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None], name="i_coord")
        t_type = tf.placeholder(tf.int32, [None], name="i_type")
        t_natoms = tf.placeholder(tf.int32, [self.ntypes + 2], name="i_natoms")
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
        return [t_des], {
            t_coord: self.coords,
            t_type: self.atype,
            t_natoms: self.natoms,
            t_box: self.box,
            t_mesh: make_default_mesh(True, False),
        }

    def eval_dp(self, dp_obj: Any) -> Any:
        ext_coords, ext_atype, mapping = extend_coord_with_ghosts(
            self.coords.reshape(1, -1, 3),
            self.atype.reshape(1, -1),
            self.box.reshape(1, 3, 3),
            dp_obj.get_rcut(),
        )
        nlist = build_neighbor_list(
            ext_coords,
            ext_atype,
            self.natoms[0],
            dp_obj.get_rcut(),
            dp_obj.get_sel(),
            distinguish_types=True,
        )
        return dp_obj(ext_coords, ext_atype, nlist=nlist)

    def eval_pt(self, dp_obj: Any) -> Any:
        ext_coords, ext_atype, mapping = extend_coord_with_ghosts_pt(
            torch.from_numpy(self.coords).to(PT_DEVICE).reshape(1, -1, 3),
            torch.from_numpy(self.atype).to(PT_DEVICE).reshape(1, -1),
            torch.from_numpy(self.box).to(PT_DEVICE).reshape(1, 3, 3),
            dp_obj.get_rcut(),
        )
        nlist = build_neighbor_list_pt(
            ext_coords,
            ext_atype,
            self.natoms[0],
            dp_obj.get_rcut(),
            dp_obj.get_sel(),
            distinguish_types=True,
        )
        return [
            x.detach().cpu().numpy() if torch.is_tensor(x) else x
            for x in dp_obj(ext_coords, ext_atype, nlist=nlist)
        ]

    def extract_ret(self, ret: Any, backend) -> Any:
        return ret[0]


class TestSeATypeOneSide(CommonTestSeATest, unittest.TestCase):
    data: ClassVar[dict] = {
        "sel": [46, 92],
        "rcut_smth": 5.80,
        "rcut": 6.00,
        "neuron": [25, 50, 100],
        "axis_neuron": 16,
        "resnet_dt": True,
        "type_one_side": True,
        "seed": 1145141919810,
    }


class TestSeATypeTwoSide(CommonTestSeATest, unittest.TestCase):
    data: ClassVar[dict] = {
        "sel": [46, 92],
        "rcut_smth": 5.80,
        "rcut": 6.00,
        "neuron": [25, 50, 100],
        "axis_neuron": 16,
        "resnet_dt": False,
        "type_one_side": False,
        "seed": 1145141919810,
    }
    skip_dp = True
    skip_pt = True


class TestSeAExcludeTypeOneSide(CommonTestSeATest, unittest.TestCase):
    data: ClassVar[dict] = {
        "sel": [46, 92],
        "rcut_smth": 5.80,
        "rcut": 6.00,
        "neuron": [25, 50, 100],
        "axis_neuron": 16,
        "resnet_dt": False,
        "type_one_side": True,
        "exclude_types": [[0, 1]],
        "seed": 1145141919810,
    }
    unittest.skip("Unsupported by native model")
    skip_dp = True
    skip_pt = True


class TestSeAExcludeTypeTwoSide(CommonTestSeATest, unittest.TestCase):
    data: ClassVar[dict] = {
        "sel": [46, 92],
        "rcut_smth": 5.80,
        "rcut": 6.00,
        "neuron": [25, 50, 100],
        "axis_neuron": 16,
        "resnet_dt": False,
        "type_one_side": False,
        "exclude_types": [[0, 1]],
        "seed": 1145141919810,
    }
    skip_dp = True
    skip_pt = True
