# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
import logging
from typing import (
    List,
    Optional,
)

import torch

from deepmd.pt.model.task.ener import (
    InvarFitting,
)
from deepmd.pt.model.task.fitting import (
    Fitting,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env import (
    DEFAULT_PRECISION,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

dtype = env.GLOBAL_PT_FLOAT_PRECISION
device = env.DEVICE

log = logging.getLogger(__name__)


@Fitting.register("dos")
class DOSFittingNet(InvarFitting):
    def __init__(
        self,
        ntypes: int,
        dim_descrpt: int,
        numb_dos: int = 300,
        neuron: List[int] = [128, 128, 128],
        resnet_dt: bool = True,
        numb_fparam: int = 0,
        numb_aparam: int = 0,
        rcond: Optional[float] = None,
        bias_dos: Optional[torch.Tensor] = None,
        trainable: Optional[List[bool]] = None,
        seed: Optional[int] = None,
        activation_function: str = "tanh",
        precision: str = DEFAULT_PRECISION,
        exclude_types: List[int] = [],
        mixed_types: bool = True,
        **kwargs,
    ):
        super().__init__(
            var_name="dos",
            ntypes=ntypes,
            dim_descrpt=dim_descrpt,
            dim_out=numb_dos,
            neuron=neuron,
            bias_atom_e=bias_dos,
            resnet_dt=resnet_dt,
            numb_fparam=numb_fparam,
            numb_aparam=numb_aparam,
            activation_function=activation_function,
            precision=precision,
            mixed_types=mixed_types,
            rcond=rcond,
            seed=seed,
            exclude_types=exclude_types,
            trainable=trainable,
            **kwargs,
        )

    def __setitem__(self, key, value):
        if key in ["bias_dos"]:
            self.bias_dos = value
        else:
            super().__setitem__(key, value)

    def __getitem__(self, key):
        if key in ["bias_dos"]:
            return self.bias_atom_e
        else:
            return super().__getitem__(key)
        
    @classmethod
    def deserialize(cls, data: dict) -> "InvarFitting":
        data = copy.deepcopy(data)
        check_version_compatibility(data.pop("@version", 1), 1, 1)
        data.pop("var_name")
        data.pop("dim_out")
        return super().deserialize(data)

    def serialize(self) -> dict:
        """Serialize the fitting to dict."""
        dd = {
            **super().serialize(),
            "type": "dos",
        }
        dd["@variables"]["bias_dos"] = dd["@variables"].pop("bias_atom_e")
        return dd

    # make jit happy with torch 2.0.0
    exclude_types: List[int]
