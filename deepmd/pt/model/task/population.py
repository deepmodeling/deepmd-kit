# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from typing import (
    Any,
)

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


@Fitting.register("population")
class PopulationFittingNet(InvarFitting):
    """Fitting the rotationally invariant electronic population of the system.

    Parameters
    ----------
    ntypes : int
        Element count.
    dim_descrpt : int
        Embedding width per atom.
    neuron : list[int]
        Number of neurons in each hidden layers of the fitting net.
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
    trainable : bool or list[bool]
        If true, the fitting net is trainable. If a list, each element controls
        the corresponding layer.
    seed : int, optional
        Random seed.
    type_map : list[str], optional
        A list of strings that map atom type indices to element names.
    """

    def __init__(
        self,
        ntypes: int,
        dim_descrpt: int,
        neuron: list[int] | None = None,
        resnet_dt: bool = True,
        numb_fparam: int = 0,
        numb_aparam: int = 0,
        dim_case_embd: int = 0,
        activation_function: str = "tanh",
        precision: str = DEFAULT_PRECISION,
        mixed_types: bool = True,
        trainable: bool | list[bool] = True,
        seed: int | None = None,
        type_map: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the PopulationFittingNet."""
        if neuron is None:
            neuron = [128, 128, 128]
        super().__init__(
            var_name="population",
            ntypes=ntypes,
            dim_descrpt=dim_descrpt,
            neuron=neuron,
            resnet_dt=resnet_dt,
            dim_out=2,
            numb_fparam=numb_fparam,
            numb_aparam=numb_aparam,
            dim_case_embd=dim_case_embd,
            activation_function=activation_function,
            precision=precision,
            mixed_types=mixed_types,
            trainable=trainable,
            seed=seed,
            type_map=type_map,
            **kwargs,
        )

    def output_def(self) -> FittingOutputDef:
        """Return the output definition of the population fitting net."""
        return FittingOutputDef(
            [
                OutputVariableDef(
                    "population",
                    [2],
                    reducible=False,
                    r_differentiable=False,
                    c_differentiable=False,
                ),
            ]
        )

    @classmethod
    def deserialize(cls, data: dict) -> "PopulationFittingNet":
        """Deserialize the fitting from a dict."""
        data = data.copy()
        check_version_compatibility(data.pop("@version", 1), 4, 1)
        obj = super().deserialize(data)
        return obj

    def serialize(self) -> dict:
        """Serialize the fitting to dict."""
        dd = {
            **InvarFitting.serialize(self),
            "type": "population",
        }
        dd["@version"] = 4
        return dd

    # make jit happy with torch 2.0.0
    exclude_types: list[int]
