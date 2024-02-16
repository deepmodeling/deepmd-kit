# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from typing import (
    List,
    Optional,
)

import torch

from deepmd.pt.model.task.fitting import (
    GeneralFitting,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env import (
    DEFAULT_PRECISION,
)

log = logging.getLogger(__name__)


class DipoleFittingNet(GeneralFitting):
    """Construct a general fitting net.

    Parameters
    ----------
    var_name : str
        The atomic property to fit, 'dipole'.
    ntypes : int
        Element count.
    dim_descrpt : int
        Embedding width per atom.
    dim_out : int
        The output dimension of the fitting net.
    dim_rot_mat : int
        The dimension of rotation matrix, m1.
    neuron : List[int]
        Number of neurons in each hidden layers of the fitting net.
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
    distinguish_types : bool
        Neighbor list that distinguish different atomic types or not.
    rcond : float, optional
        The condition number for the regression of atomic energy.
    seed : int, optional
        Random seed.
    """

    def __init__(
        self,
        var_name: str,
        ntypes: int,
        dim_descrpt: int,
        dim_out: int,
        dim_rot_mat: int,
        neuron: List[int] = [128, 128, 128],
        resnet_dt: bool = True,
        numb_fparam: int = 0,
        numb_aparam: int = 0,
        activation_function: str = "tanh",
        precision: str = DEFAULT_PRECISION,
        distinguish_types: bool = False,
        rcond: Optional[float] = None,
        seed: Optional[int] = None,
        **kwargs,
    ):
        self.dim_rot_mat = dim_rot_mat
        super().__init__(
            var_name=var_name,
            ntypes=ntypes,
            dim_descrpt=dim_descrpt,
            dim_out=dim_out,
            neuron=neuron,
            resnet_dt=resnet_dt,
            numb_fparam=numb_fparam,
            numb_aparam=numb_aparam,
            activation_function=activation_function,
            precision=precision,
            distinguish_types=distinguish_types,
            rcond=rcond,
            seed=seed,
            **kwargs,
        )
        self.old_impl = False  # this only supports the new implementation.

    def _net_out_dim(self):
        """Set the FittingNet output dim."""
        return self.dim_rot_mat

    def serialize(self) -> dict:
        data = super().serialize()
        data["dim_rot_mat"] = self.dim_rot_mat
        data["old_impl"] = self.old_impl
        return data

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
        nframes, nloc, _ = descriptor.shape
        assert gr is not None, "Must provide the rotation matrix for dipole fitting."
        # (nframes, nloc, m1)
        out = self._forward_common(descriptor, atype, gr, g2, h2, fparam, aparam)[
            self.var_name
        ]
        # (nframes * nloc, 1, m1)
        out = out.view(-1, 1, self.dim_rot_mat)
        # (nframes * nloc, m1, 3)
        gr = gr.view(nframes * nloc, -1, 3)
        # (nframes, nloc, 3)
        out = torch.bmm(out, gr).squeeze(-2).view(nframes, nloc, 3)
        return {self.var_name: out.to(env.GLOBAL_PT_FLOAT_PRECISION)}
