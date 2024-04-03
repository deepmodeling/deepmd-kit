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
    fitting_check_output,
)
from deepmd.pt.model.task.fitting import (
    GeneralFitting,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env import (
    DEFAULT_PRECISION,
)
from deepmd.pt.utils.stat import (
    compute_output_stats,
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


@GeneralFitting.register("invar")
@fitting_check_output
class InvarFitting(GeneralFitting):
    """Construct a fitting net for energy.

    Parameters
    ----------
    var_name : str
        The atomic property to fit, 'energy', 'dipole', and 'polar'.
    ntypes : int
        Element count.
    dim_descrpt : int
        Embedding width per atom.
    dim_out : int
        The output dimension of the fitting net.
    neuron : List[int]
        Number of neurons in each hidden layers of the fitting net.
    bias_atom_e : torch.Tensor, optional
        Average enery per atom for each element.
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
    rcond : float, optional
        The condition number for the regression of atomic energy.
    seed : int, optional
        Random seed.
    exclude_types: List[int]
        Atomic contributions of the excluded atom types are set zero.
    atom_ener: List[float], optional
        Specifying atomic energy contribution in vacuum. The `set_davg_zero` key in the descrptor should be set.

    """

    def __init__(
        self,
        var_name: str,
        ntypes: int,
        dim_descrpt: int,
        dim_out: int,
        neuron: List[int] = [128, 128, 128],
        bias_atom_e: Optional[torch.Tensor] = None,
        resnet_dt: bool = True,
        numb_fparam: int = 0,
        numb_aparam: int = 0,
        activation_function: str = "tanh",
        precision: str = DEFAULT_PRECISION,
        mixed_types: bool = True,
        rcond: Optional[float] = None,
        seed: Optional[int] = None,
        exclude_types: List[int] = [],
        atom_ener: Optional[List[float]] = None,
        **kwargs,
    ):
        self.dim_out = dim_out
        self.atom_ener = atom_ener
        super().__init__(
            var_name=var_name,
            ntypes=ntypes,
            dim_descrpt=dim_descrpt,
            neuron=neuron,
            bias_atom_e=bias_atom_e,
            resnet_dt=resnet_dt,
            numb_fparam=numb_fparam,
            numb_aparam=numb_aparam,
            activation_function=activation_function,
            precision=precision,
            mixed_types=mixed_types,
            rcond=rcond,
            seed=seed,
            exclude_types=exclude_types,
            remove_vaccum_contribution=None
            if atom_ener is None or len([x for x in atom_ener if x is not None]) == 0
            else [x is not None for x in atom_ener],
            **kwargs,
        )

    def _net_out_dim(self):
        """Set the FittingNet output dim."""
        return self.dim_out

    def serialize(self) -> dict:
        data = super().serialize()
        data["type"] = "invar"
        data["dim_out"] = self.dim_out
        data["atom_ener"] = self.atom_ener
        return data

    @classmethod
    def deserialize(cls, data: dict) -> "GeneralFitting":
        data = copy.deepcopy(data)
        check_version_compatibility(data.pop("@version", 1), 1, 1)
        return super().deserialize(data)

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
        # [0] to get the mean (bias)
        bias_atom_e = compute_output_stats(
            merged,
            self.ntypes,
            keys=[self.var_name],
            stat_file_path=stat_file_path,
            rcond=self.rcond,
            atom_ener=self.atom_ener,
        )[0][self.var_name]
        self.bias_atom_e.copy_(bias_atom_e.view([self.ntypes, self.dim_out]))

    def output_def(self) -> FittingOutputDef:
        return FittingOutputDef(
            [
                OutputVariableDef(
                    self.var_name,
                    [self.dim_out],
                    reduciable=True,
                    r_differentiable=True,
                    c_differentiable=True,
                ),
            ]
        )

    def forward(
        self,
        descriptor: torch.Tensor,
        atype: torch.Tensor,
        gr: Optional[torch.Tensor] = None,
        g2: Optional[torch.Tensor] = None,
        h2: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
    ):
        """Based on embedding net output, alculate total energy.

        Args:
        - inputs: Embedding matrix. Its shape is [nframes, natoms[0], self.dim_descrpt].
        - natoms: Tell atom count and element count. Its shape is [2+self.ntypes].

        Returns
        -------
        - `torch.Tensor`: Total energy with shape [nframes, natoms[0]].
        """
        return self._forward_common(descriptor, atype, gr, g2, h2, fparam, aparam)

    # make jit happy with torch 2.0.0
    exclude_types: List[int]
