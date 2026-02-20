# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
import unittest
from typing import (
    Any,
)

import numpy as np

from deepmd.dpmodel.model.model import get_model as get_model_dp
from deepmd.dpmodel.model.spin_model import SpinModel as SpinModelDP
from deepmd.dpmodel.utils.nlist import (
    build_neighbor_list,
    extend_coord_with_ghosts,
)
from deepmd.dpmodel.utils.region import (
    normalize_coord,
)
from deepmd.env import (
    GLOBAL_NP_FLOAT_PRECISION,
)

from ..common import (
    INSTALLED_PT,
    CommonTest,
)
from .common import (
    ModelTest,
)

if INSTALLED_PT:
    from deepmd.pt.model.model import get_model as get_model_pt
    from deepmd.pt.model.model.spin_model import SpinEnergyModel as SpinEnergyModelPT
    from deepmd.pt.utils.utils import to_numpy_array as torch_to_numpy
    from deepmd.pt.utils.utils import to_torch_tensor as numpy_to_torch
else:
    SpinEnergyModelPT = None

from deepmd.utils.argcheck import (
    model_args,
)

SPIN_DATA = {
    "type_map": ["O", "H", "B"],
    "descriptor": {
        "type": "se_e2_a",
        "sel": [20, 20, 20],
        "rcut_smth": 0.50,
        "rcut": 4.00,
        "neuron": [
            3,
            6,
        ],
        "resnet_dt": False,
        "axis_neuron": 2,
        "precision": "float64",
        "type_one_side": True,
        "seed": 1,
    },
    "fitting_net": {
        "neuron": [
            5,
            5,
        ],
        "resnet_dt": True,
        "precision": "float64",
        "seed": 1,
    },
    "spin": {
        "use_spin": [True, False, False],
        "virtual_scale": [0.3140],
    },
}


class TestSpinEner(CommonTest, ModelTest, unittest.TestCase):
    @property
    def data(self) -> dict:
        return SPIN_DATA

    tf_class = None
    dp_class = SpinModelDP
    pt_class = SpinEnergyModelPT
    pd_class = None
    pt_expt_class = None
    jax_class = None
    args = model_args()

    skip_tf = True
    skip_jax = True
    skip_pt_expt = True
    skip_pd = True

    def get_reference_backend(self):
        """Get the reference backend.

        We need a reference backend that can reproduce forces.
        """
        if not self.skip_pt:
            return self.RefBackend.PT
        if not self.skip_dp:
            return self.RefBackend.DP
        raise ValueError("No available reference")

    def pass_data_to_cls(self, cls, data) -> Any:
        """Pass data to the class."""
        data = copy.deepcopy(data)
        if cls is SpinModelDP:
            return get_model_dp(data)
        elif cls is SpinEnergyModelPT:
            return get_model_pt(data)
        return cls(**data, **self.additional_data)

    def setUp(self) -> None:
        CommonTest.setUp(self)

        self.ntypes = 3
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
        ).reshape(1, -1, 3)
        self.atype = np.array([0, 0, 1, 0, 1, 1], dtype=np.int32).reshape(1, -1)
        self.box = np.array(
            [13.0, 0.0, 0.0, 0.0, 13.0, 0.0, 0.0, 0.0, 13.0],
            dtype=GLOBAL_NP_FLOAT_PRECISION,
        ).reshape(1, 9)
        self.natoms = np.array([6, 6, 3, 3, 0], dtype=np.int32)
        self.spin = np.array(
            [
                0.50,
                0.30,
                0.20,
                0.40,
                0.25,
                0.15,
                0.10,
                0.05,
                0.08,
                0.12,
                0.07,
                0.09,
                0.45,
                0.35,
                0.28,
                0.11,
                0.06,
                0.03,
            ],
            dtype=GLOBAL_NP_FLOAT_PRECISION,
        ).reshape(1, -1, 3)

    def build_tf(self, obj: Any, suffix: str) -> tuple[list, dict]:
        raise NotImplementedError("no TF in this test")

    def eval_dp(self, dp_obj: Any) -> Any:
        return dp_obj(
            self.coords,
            self.atype,
            self.spin,
            box=self.box,
        )

    def eval_pt(self, pt_obj: Any) -> Any:
        return {
            kk: torch_to_numpy(vv)
            for kk, vv in pt_obj(
                numpy_to_torch(self.coords),
                numpy_to_torch(self.atype),
                numpy_to_torch(self.spin),
                box=numpy_to_torch(self.box),
            ).items()
        }

    def extract_ret(self, ret: Any, backend) -> tuple[np.ndarray, ...]:
        # shape not matched. ravel...
        from ..common import (
            SKIP_FLAG,
        )

        if backend is self.RefBackend.DP:
            return (
                ret["energy"].ravel(),
                ret["atom_energy"].ravel(),
                ret["mask_mag"].ravel(),
                SKIP_FLAG,
                SKIP_FLAG,
            )
        elif backend is self.RefBackend.PT:
            return (
                ret["energy"].ravel(),
                ret["atom_energy"].ravel(),
                ret["mask_mag"].ravel(),
                ret["force"].ravel(),
                ret["force_mag"].ravel(),
            )
        raise ValueError(f"Unknown backend: {backend}")


class TestSpinEnerLower(CommonTest, ModelTest, unittest.TestCase):
    @property
    def data(self) -> dict:
        return SPIN_DATA

    tf_class = None
    dp_class = SpinModelDP
    pt_class = SpinEnergyModelPT
    pd_class = None
    pt_expt_class = None
    jax_class = None
    args = model_args()

    skip_tf = True
    skip_jax = True
    skip_pt_expt = True
    skip_pd = True

    def get_reference_backend(self):
        """Get the reference backend.

        We need a reference backend that can reproduce forces.
        """
        if not self.skip_pt:
            return self.RefBackend.PT
        if not self.skip_dp:
            return self.RefBackend.DP
        raise ValueError("No available reference")

    def pass_data_to_cls(self, cls, data) -> Any:
        """Pass data to the class."""
        data = copy.deepcopy(data)
        if cls is SpinModelDP:
            return get_model_dp(data)
        elif cls is SpinEnergyModelPT:
            return get_model_pt(data)
        return cls(**data, **self.additional_data)

    def setUp(self) -> None:
        CommonTest.setUp(self)

        self.ntypes = 3
        coords = np.array(
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
        ).reshape(1, -1, 3)
        atype = np.array([0, 0, 1, 0, 1, 1], dtype=np.int32).reshape(1, -1)
        box = np.array(
            [13.0, 0.0, 0.0, 0.0, 13.0, 0.0, 0.0, 0.0, 13.0],
            dtype=GLOBAL_NP_FLOAT_PRECISION,
        ).reshape(1, 9)
        self.spin = np.array(
            [
                0.50,
                0.30,
                0.20,
                0.40,
                0.25,
                0.15,
                0.10,
                0.05,
                0.08,
                0.12,
                0.07,
                0.09,
                0.45,
                0.35,
                0.28,
                0.11,
                0.06,
                0.03,
            ],
            dtype=GLOBAL_NP_FLOAT_PRECISION,
        ).reshape(1, -1, 3)

        rcut = 4.0
        nframes, nloc = atype.shape[:2]
        coord_normalized = normalize_coord(
            coords.reshape(nframes, nloc, 3),
            box.reshape(nframes, 3, 3),
        )
        extended_coord, extended_atype, mapping = extend_coord_with_ghosts(
            coord_normalized, atype, box, rcut
        )
        nlist = build_neighbor_list(
            extended_coord,
            extended_atype,
            nloc,
            rcut,
            [20, 20, 20],
            distinguish_types=True,
        )
        extended_coord = extended_coord.reshape(nframes, -1, 3)
        self.nlist = nlist
        self.extended_coord = extended_coord
        self.extended_atype = extended_atype
        self.mapping = mapping

        # Build extended spin from mapping
        self.extended_spin = np.take_along_axis(
            self.spin,
            np.repeat(mapping[:, :, np.newaxis], 3, axis=2),
            axis=1,
        )

    def build_tf(self, obj: Any, suffix: str) -> tuple[list, dict]:
        raise NotImplementedError("no TF in this test")

    def eval_dp(self, dp_obj: Any) -> Any:
        return dp_obj.call_lower(
            self.extended_coord,
            self.extended_atype,
            self.extended_spin,
            self.nlist,
            self.mapping,
        )

    def eval_pt(self, pt_obj: Any) -> Any:
        return {
            kk: torch_to_numpy(vv)
            for kk, vv in pt_obj.forward_lower(
                numpy_to_torch(self.extended_coord),
                numpy_to_torch(self.extended_atype),
                numpy_to_torch(self.extended_spin),
                numpy_to_torch(self.nlist),
                numpy_to_torch(self.mapping),
            ).items()
        }

    def extract_ret(self, ret: Any, backend) -> tuple[np.ndarray, ...]:
        # shape not matched. ravel...
        from ..common import (
            SKIP_FLAG,
        )

        if backend is self.RefBackend.DP:
            return (
                ret["energy"].ravel(),
                ret["atom_energy"].ravel(),
                ret["extended_mask_mag"].ravel(),
                SKIP_FLAG,
                SKIP_FLAG,
            )
        elif backend is self.RefBackend.PT:
            return (
                ret["energy"].ravel(),
                ret["atom_energy"].ravel(),
                ret["extended_mask_mag"].ravel(),
                ret["extended_force"].ravel(),
                ret["extended_force_mag"].ravel(),
            )
        raise ValueError(f"Unknown backend: {backend}")
