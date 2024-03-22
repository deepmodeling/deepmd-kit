# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
import logging
from typing import (
    Callable,
    List,
    Optional,
    Union,
)

import torch

from deepmd.dpmodel import (
    FittingOutputDef,
    OutputVariableDef,
)
from deepmd.pt.model.task.fitting import (
    Fitting,
    GeneralFitting,
)
from deepmd.pt.model.task.invar_fitting import (
    InvarFitting,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env import (
    DEFAULT_PRECISION,
)
from deepmd.utils.path import (
    DPPath,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

dtype = env.GLOBAL_PT_FLOAT_PRECISION
device = env.DEVICE

log = logging.getLogger(__name__)


@Fitting.register("property")
class PropertyFittingNet(InvarFitting):
    """Construct a fitting net for energy.

    Parameters
    ----------
    ntypes : int
        Element count.
    dim_descrpt : int
        Embedding width per atom.
    task_dim : int
        The dimension of outputs of fitting net
    neuron : List[int]
        Number of neurons in each hidden layers of the fitting net.
    bias_atom_p : torch.Tensor, optional
        Average property per atom for each element.
    resnet_dt : bool
        Using time-step in the ResNet construction.
    numb_fparam : int
        Number of frame parameters.
    numb_aparam : int
        Number of atomic parameters.
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
        task_dim: int = 1,
        neuron: List[int] = [128, 128, 128],
        bias_atom_p: Optional[torch.Tensor] = None,
        resnet_dt: bool = True,
        numb_fparam: int = 0,
        numb_aparam: int = 0,
        activation_function: str = "tanh",
        precision: str = DEFAULT_PRECISION,
        mixed_types: bool = True,
        seed: Optional[int] = None,
        **kwargs,
    ):
        self.task_dim = task_dim
        super().__init__(
            var_name="property",
            ntypes=ntypes,
            dim_descrpt=dim_descrpt,
            dim_out=task_dim,
            neuron=neuron,
            bias_atom_e=bias_atom_p,
            resnet_dt=resnet_dt,
            numb_fparam=numb_fparam,
            numb_aparam=numb_aparam,
            activation_function=activation_function,
            precision=precision,
            mixed_types=mixed_types,
            seed=seed,
            **kwargs,
        )

    @classmethod
    def deserialize(cls, data: dict) -> "GeneralFitting":
        data = copy.deepcopy(data)
        check_version_compatibility(data.pop("@version", 1), 1, 1)
        data.pop("dim_out")
        data.pop("var_name")
        return super().deserialize(data)

    def serialize(self) -> dict:
        """Serialize the fitting to dict."""
        return {**super().serialize(), "type": "property", "task_dim": self.task_dim}

    def output_def(self) -> FittingOutputDef:
        return FittingOutputDef(
            [
                OutputVariableDef(
                    self.var_name,
                    [self.dim_out],
                    reduciable=True,
                    r_differentiable=False,
                    c_differentiable=False,
                ),
            ]
        )

    def compute_output_stats(
        self,
        merged: Union[Callable[[], List[dict]], List[dict]],
        stat_file_path: Optional[DPPath] = None,
    ):
        """
        Compute the output statistics (e.g. energy bias) for the fitting net from packed data.

        Parameters
        ----------
        merged : Union[Callable[[], List[dict]], List[dict]]
            - List[dict]: A list of data samples from various data systems.
                Each element, `merged[i]`, is a data dictionary containing `keys`: `torch.Tensor`
                originating from the `i`-th data system.
            - Callable[[], List[dict]]: A lazy function that returns data samples in the above format
                only when needed. Since the sampling process can be slow and memory-intensive,
                the lazy function helps by only sampling once.
        stat_file_path : Optional[DPPath]
            The path to the stat file.

        """
        pass

    # make jit happy with torch 2.0.0
    exclude_types: List[int]
