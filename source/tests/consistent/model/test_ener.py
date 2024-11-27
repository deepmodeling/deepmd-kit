# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from typing import (
    Any,
)

import numpy as np

from deepmd.dpmodel.common import (
    to_numpy_array,
)
from deepmd.dpmodel.model.ener_model import EnergyModel as EnergyModelDP
from deepmd.dpmodel.model.model import get_model as get_model_dp
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
    INSTALLED_JAX,
    INSTALLED_PD,
    INSTALLED_PT,
    INSTALLED_TF,
    SKIP_FLAG,
    CommonTest,
    parameterized,
)
from .common import (
    ModelTest,
)

if INSTALLED_PT:
    from deepmd.pt.model.model import get_model as get_model_pt
    from deepmd.pt.model.model.ener_model import EnergyModel as EnergyModelPT
    from deepmd.pt.utils.utils import to_numpy_array as torch_to_numpy
    from deepmd.pt.utils.utils import to_torch_tensor as numpy_to_torch
else:
    EnergyModelPT = None
if INSTALLED_TF:
    from deepmd.tf.model.ener import EnerModel as EnergyModelTF
else:
    EnergyModelTF = None
if INSTALLED_PD:
    from deepmd.pd.model.model import get_model as get_model_pd
    from deepmd.pd.model.model.ener_model import EnergyModel as EnergyModelPD
    from deepmd.pd.utils.utils import to_numpy_array as paddle_to_numpy
    from deepmd.pd.utils.utils import to_paddle_tensor as numpy_to_paddle
else:
    EnergyModelPD = None
from deepmd.utils.argcheck import (
    model_args,
)

if INSTALLED_JAX:
    from deepmd.jax.common import (
        to_jax_array,
    )
    from deepmd.jax.model.ener_model import EnergyModel as EnergyModelJAX
    from deepmd.jax.model.model import get_model as get_model_jax
else:
    EnergyModelJAX = None


@parameterized(
    (
        [],
        [[0, 1]],
    ),
    (
        [],
        [1],
    ),
)
class TestEner(CommonTest, ModelTest, unittest.TestCase):
    @property
    def data(self) -> dict:
        pair_exclude_types, atom_exclude_types = self.param
        return {
            "type_map": ["O", "H"],
            "pair_exclude_types": pair_exclude_types,
            "atom_exclude_types": atom_exclude_types,
            "descriptor": {
                "type": "se_e2_a",
                "sel": [20, 20],
                "rcut_smth": 0.50,
                "rcut": 6.00,
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
        }

    tf_class = EnergyModelTF
    dp_class = EnergyModelDP
    pt_class = EnergyModelPT
    pd_class = EnergyModelPD
    jax_class = EnergyModelJAX
    pd_class = EnergyModelPD
    args = model_args()

    def get_reference_backend(self):
        """Get the reference backend.

        We need a reference backend that can reproduce forces.
        """
        if not self.skip_pt:
            return self.RefBackend.PT
        if not self.skip_tf:
            return self.RefBackend.TF
        if not self.skip_jax:
            return self.RefBackend.JAX
        if not self.skip_pd:
            return self.RefBackend.PD
        if not self.skip_dp:
            return self.RefBackend.DP
        raise ValueError("No available reference")

    @property
    def skip_tf(self):
        return (
            self.data["pair_exclude_types"] != []
            or self.data["atom_exclude_types"] != []
        )

    @property
    def skip_jax(self) -> bool:
        return not INSTALLED_JAX

    def pass_data_to_cls(self, cls, data) -> Any:
        """Pass data to the class."""
        data = data.copy()
        if cls is EnergyModelDP:
            return get_model_dp(data)
        elif cls is EnergyModelPT:
            model = get_model_pt(data)
            model.atomic_model.out_bias.uniform_()
            return model
        elif cls is EnergyModelJAX:
            return get_model_jax(data)
        elif cls is EnergyModelPD:
            return get_model_pd(data)
        return cls(**data, **self.additional_data)

    def setUp(self) -> None:
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
        ).reshape(1, -1, 3)
        self.atype = np.array([0, 1, 1, 0, 1, 1], dtype=np.int32).reshape(1, -1)
        self.box = np.array(
            [13.0, 0.0, 0.0, 0.0, 13.0, 0.0, 0.0, 0.0, 13.0],
            dtype=GLOBAL_NP_FLOAT_PRECISION,
        ).reshape(1, 9)
        self.natoms = np.array([6, 6, 2, 4], dtype=np.int32)

        # TF requires the atype to be sort
        idx_map = np.argsort(self.atype.ravel())
        self.atype = self.atype[:, idx_map]
        self.coords = self.coords[:, idx_map]

    def build_tf(self, obj: Any, suffix: str) -> tuple[list, dict]:
        return self.build_tf_model(
            obj,
            self.natoms,
            self.coords,
            self.atype,
            self.box,
            suffix,
        )

    def eval_dp(self, dp_obj: Any) -> Any:
        return self.eval_dp_model(
            dp_obj,
            self.natoms,
            self.coords,
            self.atype,
            self.box,
        )

    def eval_pt(self, pt_obj: Any) -> Any:
        return self.eval_pt_model(
            pt_obj,
            self.natoms,
            self.coords,
            self.atype,
            self.box,
        )

    def eval_jax(self, jax_obj: Any) -> Any:
        return self.eval_jax_model(
            jax_obj,
            self.natoms,
            self.coords,
            self.atype,
            self.box,
        )

    def eval_pd(self, pd_obj: Any) -> Any:
        return self.eval_pd_model(
            pd_obj,
            self.natoms,
            self.coords,
            self.atype,
            self.box,
        )

    def extract_ret(self, ret: Any, backend) -> tuple[np.ndarray, ...]:
        # shape not matched. ravel...
        if backend is self.RefBackend.DP:
            return (
                ret["energy_redu"].ravel(),
                ret["energy"].ravel(),
                SKIP_FLAG,
                SKIP_FLAG,
                SKIP_FLAG,
            )
        elif backend is self.RefBackend.PT:
            return (
                ret["energy"].ravel(),
                ret["atom_energy"].ravel(),
                ret["force"].ravel(),
                ret["virial"].ravel(),
                ret["atom_virial"].ravel(),
            )
        elif backend is self.RefBackend.TF:
            return (
                ret[0].ravel(),
                ret[1].ravel(),
                ret[2].ravel(),
                ret[3].ravel(),
                ret[4].ravel(),
            )
        elif backend is self.RefBackend.JAX:
            return (
                ret["energy_redu"].ravel(),
                ret["energy"].ravel(),
                ret["energy_derv_r"].ravel(),
                ret["energy_derv_c_redu"].ravel(),
                ret["energy_derv_c"].ravel(),
            )
        elif backend is self.RefBackend.PD:
            return (
                ret["energy"].flatten(),
                ret["atom_energy"].flatten(),
                ret["force"].flatten(),
                ret["virial"].flatten(),
                ret["atom_virial"].flatten(),
            )
        raise ValueError(f"Unknown backend: {backend}")


@parameterized(
    (
        [],
        [[0, 1]],
    ),
    (
        [],
        [1],
    ),
)
class TestEnerLower(CommonTest, ModelTest, unittest.TestCase):
    @property
    def data(self) -> dict:
        pair_exclude_types, atom_exclude_types = self.param
        return {
            "type_map": ["O", "H"],
            "pair_exclude_types": pair_exclude_types,
            "atom_exclude_types": atom_exclude_types,
            "descriptor": {
                "type": "se_e2_a",
                "sel": [20, 20],
                "rcut_smth": 0.50,
                "rcut": 6.00,
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
        }

    tf_class = EnergyModelTF
    dp_class = EnergyModelDP
    pt_class = EnergyModelPT
    jax_class = EnergyModelJAX
    pd_class = EnergyModelPD
    args = model_args()

    def get_reference_backend(self):
        """Get the reference backend.

        We need a reference backend that can reproduce forces.
        """
        if not self.skip_pt:
            return self.RefBackend.PT
        if not self.skip_jax:
            return self.RefBackend.JAX
        if not self.skip_dp:
            return self.RefBackend.DP
        if not self.skip_pd:
            return self.RefBackend.PD
        raise ValueError("No available reference")

    @property
    def skip_tf(self) -> bool:
        # TF does not have lower interface
        return True

    @property
    def skip_jax(self) -> bool:
        return not INSTALLED_JAX

    def pass_data_to_cls(self, cls, data) -> Any:
        """Pass data to the class."""
        data = data.copy()
        if cls is EnergyModelDP:
            return get_model_dp(data)
        elif cls is EnergyModelPT:
            return get_model_pt(data)
        elif cls is EnergyModelJAX:
            return get_model_jax(data)
        elif cls is EnergyModelPD:
            return get_model_pd(data)
        return cls(**data, **self.additional_data)

    def setUp(self) -> None:
        CommonTest.setUp(self)

        self.ntypes = 2
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
        atype = np.array([0, 1, 1, 0, 1, 1], dtype=np.int32).reshape(1, -1)
        box = np.array(
            [13.0, 0.0, 0.0, 0.0, 13.0, 0.0, 0.0, 0.0, 13.0],
            dtype=GLOBAL_NP_FLOAT_PRECISION,
        ).reshape(1, 9)

        rcut = 6.0
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
            6.0,
            [20, 20],
            distinguish_types=True,
        )
        extended_coord = extended_coord.reshape(nframes, -1, 3)
        self.nlist = nlist
        self.extended_coord = extended_coord
        self.extended_atype = extended_atype
        self.mapping = mapping

    def build_tf(self, obj: Any, suffix: str) -> tuple[list, dict]:
        raise NotImplementedError("no TF in this test")

    def eval_dp(self, dp_obj: Any) -> Any:
        return dp_obj.call_lower(
            self.extended_coord,
            self.extended_atype,
            self.nlist,
            self.mapping,
            do_atomic_virial=True,
        )

    def eval_pt(self, pt_obj: Any) -> Any:
        return {
            kk: torch_to_numpy(vv)
            for kk, vv in pt_obj.forward_lower(
                numpy_to_torch(self.extended_coord),
                numpy_to_torch(self.extended_atype),
                numpy_to_torch(self.nlist),
                numpy_to_torch(self.mapping),
                do_atomic_virial=True,
            ).items()
        }

    def eval_jax(self, jax_obj: Any) -> Any:
        return {
            kk: to_numpy_array(vv)
            for kk, vv in jax_obj.call_lower(
                to_jax_array(self.extended_coord),
                to_jax_array(self.extended_atype),
                to_jax_array(self.nlist),
                to_jax_array(self.mapping),
                do_atomic_virial=True,
            ).items()
        }

    def eval_pd(self, pd_obj: Any) -> Any:
        return {
            kk: paddle_to_numpy(vv)
            for kk, vv in pd_obj.forward_lower(
                numpy_to_paddle(self.extended_coord),
                numpy_to_paddle(self.extended_atype),
                numpy_to_paddle(self.nlist),
                numpy_to_paddle(self.mapping),
                do_atomic_virial=True,
            ).items()
        }

    def extract_ret(self, ret: Any, backend) -> tuple[np.ndarray, ...]:
        # shape not matched. ravel...
        if backend is self.RefBackend.DP:
            return (
                ret["energy_redu"].ravel(),
                ret["energy"].ravel(),
                SKIP_FLAG,
                SKIP_FLAG,
                SKIP_FLAG,
            )
        elif backend is self.RefBackend.PT:
            return (
                ret["energy"].ravel(),
                ret["atom_energy"].ravel(),
                ret["extended_force"].ravel(),
                ret["virial"].ravel(),
                ret["extended_virial"].ravel(),
            )
        elif backend is self.RefBackend.JAX:
            return (
                ret["energy_redu"].ravel(),
                ret["energy"].ravel(),
                ret["energy_derv_r"].ravel(),
                ret["energy_derv_c_redu"].ravel(),
                ret["energy_derv_c"].ravel(),
            )
        elif backend is self.RefBackend.PD:
            return (
                ret["energy"].flatten(),
                ret["atom_energy"].flatten(),
                ret["extended_force"].flatten(),
                ret["virial"].flatten(),
                ret["extended_virial"].flatten(),
            )
        raise ValueError(f"Unknown backend: {backend}")
