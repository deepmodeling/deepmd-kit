# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from typing import (
    Optional,
    Union,
)

import torch

from deepmd.dpmodel import (
    FittingOutputDef,
    OutputVariableDef,
)
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
from deepmd.pt.utils.utils import (
    to_numpy_array,
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
        neuron: list[int] = [128, 128, 128],
        resnet_dt: bool = True,
        numb_fparam: int = 0,
        numb_aparam: int = 0,
        dim_case_embd: int = 0,
        rcond: Optional[float] = None,
        bias_dos: Optional[torch.Tensor] = None,
        trainable: Union[bool, list[bool]] = True,
        seed: Optional[Union[int, list[int]]] = None,
        activation_function: str = "tanh",
        precision: str = DEFAULT_PRECISION,
        exclude_types: list[int] = [],
        mixed_types: bool = True,
        type_map: Optional[list[str]] = None,
    ) -> None:
        if bias_dos is not None:
            self.bias_dos = bias_dos
        else:
            self.bias_dos = torch.zeros(
                (ntypes, numb_dos), dtype=dtype, device=env.DEVICE
            )
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
            dim_case_embd=dim_case_embd,
            activation_function=activation_function,
            precision=precision,
            mixed_types=mixed_types,
            rcond=rcond,
            seed=seed,
            exclude_types=exclude_types,
            trainable=trainable,
            type_map=type_map,
        )

    def output_def(self) -> FittingOutputDef:
        return FittingOutputDef(
            [
                OutputVariableDef(
                    self.var_name,
                    [self.dim_out],
                    reducible=True,
                    r_differentiable=False,
                    c_differentiable=False,
                ),
            ]
        )

    @classmethod
    def deserialize(cls, data: dict) -> "DOSFittingNet":
        data = data.copy()
        check_version_compatibility(data.pop("@version", 1), 3, 1)
        data.pop("@class", None)
        data.pop("var_name", None)
        data.pop("tot_ener_zero", None)
        data.pop("layer_name", None)
        data.pop("use_aparam_as_mask", None)
        data.pop("spin", None)
        data.pop("atom_ener", None)
        data["numb_dos"] = data.pop("dim_out")
        obj = super().deserialize(data)

        return obj

    def serialize(self) -> dict:
        """Serialize the fitting to dict."""
        # dd = super(InvarFitting, self).serialize()
        dd = {
            **InvarFitting.serialize(self),
            "type": "dos",
            "dim_out": self.dim_out,
        }
        dd["@variables"]["bias_atom_e"] = to_numpy_array(self.bias_atom_e)

        return dd

    # make jit happy with torch 2.0.0
    exclude_types: list[int]
