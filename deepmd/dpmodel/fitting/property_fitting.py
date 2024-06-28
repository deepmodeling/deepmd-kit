# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
from typing import (
    List,
    Optional,
    Union,
)

import numpy as np

from deepmd.dpmodel.common import (
    DEFAULT_PRECISION,
)
from deepmd.dpmodel.fitting.invar_fitting import (
    InvarFitting,
)
from deepmd.utils.version import (
    check_version_compatibility,
)


@InvarFitting.register("property")
class PropertyFittingNet(InvarFitting):
    def __init__(
        self,
        ntypes: int,
        dim_descrpt: int,
        task_dim: int = 1,
        neuron: List[int] = [128, 128, 128],
        bias_atom_p: Optional[np.ndarray] = None,
        rcond: Optional[float] = None,
        trainable: Union[bool, List[bool]] = True,
        intensive: bool = False,
        resnet_dt: bool = True,
        numb_fparam: int = 0,
        numb_aparam: int = 0,
        activation_function: str = "tanh",
        precision: str = DEFAULT_PRECISION,
        mixed_types: bool = True,
        exclude_types: List[int] = [],
        type_map: Optional[List[str]] = None,
        # not used
        seed: Optional[int] = None,
    ):
        self.task_dim = task_dim
        self.intensive = intensive
        super().__init__(
            var_name="property",
            ntypes=ntypes,
            dim_descrpt=dim_descrpt,
            dim_out=task_dim,
            neuron=neuron,
            bias_atom=bias_atom_p,
            resnet_dt=resnet_dt,
            numb_fparam=numb_fparam,
            numb_aparam=numb_aparam,
            rcond=rcond,
            trainable=trainable,
            activation_function=activation_function,
            precision=precision,
            mixed_types=mixed_types,
            exclude_types=exclude_types,
            type_map=type_map,
        )

    @classmethod
    def deserialize(cls, data: dict) -> "PropertyFittingNet":
        data = copy.deepcopy(data)
        check_version_compatibility(data.pop("@version", 1), 2, 1)
        data.pop("dim_out")
        data.pop("var_name")
        data.pop("tot_ener_zero")
        data.pop("layer_name")
        data.pop("use_aparam_as_mask", None)
        data.pop("spin", None)
        data.pop("atom_ener", None)
        obj = super().deserialize(data)

        return obj

    def serialize(self) -> dict:
        """Serialize the fitting to dict."""
        dd = {
            **InvarFitting.serialize(self),
            "type": "property",
            "task_dim": self.task_dim,
        }

        return dd
