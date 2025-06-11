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
from deepmd.utils.version import (
    check_version_compatibility,
)

dtype = env.GLOBAL_PT_FLOAT_PRECISION
device = env.DEVICE

log = logging.getLogger(__name__)


@Fitting.register("property")
class PropertyFittingNet(InvarFitting):
    """Fitting the rotationally invariant properties of `task_dim` of the system.

    Parameters
    ----------
    ntypes : int
        Element count.
    dim_descrpt : int
        Embedding width per atom.
    task_dim : int
        The dimension of outputs of fitting net.
    property_name:
        The name of fitting property, which should be consistent with the property name in the dataset.
        If the data file is named `humo.npy`, this parameter should be "humo".
    neuron : list[int]
        Number of neurons in each hidden layers of the fitting net.
    bias_atom_p : torch.Tensor, optional
        Average property per atom for each element.
    intensive : bool, optional
        Whether the fitting property is intensive.
    resnet_dt : bool
        Using time-step in the ResNet construction.
    numb_fparam : int
        Number of frame parameters.
    numb_aparam : int
        Number of atomic parameters.
    dim_case_embd : int
        Dimension of case specific embedding.
    activation_function : str
        Activation function.
    precision : str
        Numerical precision.
    mixed_types : bool
        If true, use a uniform fitting net for all atom types, otherwise use
        different fitting nets for different atom types.
    seed : int, optional
        Random seed.
    """

    def __init__(
        self,
        ntypes: int,
        dim_descrpt: int,
        property_name: str,
        task_dim: int = 1,
        neuron: list[int] = [128, 128, 128],
        bias_atom_p: Optional[torch.Tensor] = None,
        intensive: bool = False,
        resnet_dt: bool = True,
        numb_fparam: int = 0,
        numb_aparam: int = 0,
        dim_case_embd: int = 0,
        activation_function: str = "tanh",
        precision: str = DEFAULT_PRECISION,
        mixed_types: bool = True,
        trainable: Union[bool, list[bool]] = True,
        seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        self.task_dim = task_dim
        self.intensive = intensive
        super().__init__(
            var_name=property_name,
            ntypes=ntypes,
            dim_descrpt=dim_descrpt,
            dim_out=task_dim,
            neuron=neuron,
            bias_atom_e=bias_atom_p,
            resnet_dt=resnet_dt,
            numb_fparam=numb_fparam,
            numb_aparam=numb_aparam,
            dim_case_embd=dim_case_embd,
            activation_function=activation_function,
            precision=precision,
            mixed_types=mixed_types,
            trainable=trainable,
            seed=seed,
            **kwargs,
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
                    intensive=self.intensive,
                ),
            ]
        )

    def get_intensive(self) -> bool:
        """Whether the fitting property is intensive."""
        return self.intensive

    @classmethod
    def deserialize(cls, data: dict) -> "PropertyFittingNet":
        data = data.copy()
        check_version_compatibility(data.pop("@version", 1), 4, 1)
        data.pop("dim_out")
        data["property_name"] = data.pop("var_name")
        obj = super().deserialize(data)

        return obj

    def serialize(self) -> dict:
        """Serialize the fitting to dict."""
        dd = {
            **InvarFitting.serialize(self),
            "type": "property",
            "task_dim": self.task_dim,
            "intensive": self.intensive,
        }
        dd["@version"] = 4

        return dd

    # make jit happy with torch 2.0.0
    exclude_types: list[int]
