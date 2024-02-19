# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from typing import (
    List,
    Optional,
)

import torch

from deepmd.dpmodel import (
    FittingOutputDef,
    OutputVariableDef,
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

log = logging.getLogger(__name__)


class PolarFittingNet(GeneralFitting):
    """Construct a polar fitting net.

    Parameters
    ----------
    var_name : str
        The atomic property to fit, 'polar'.
    ntypes : int
        Element count.
    dim_descrpt : int
        Embedding width per atom.
    dim_out : int
        The output dimension of the fitting net.
    embedding_width : int
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
    fit_diag : bool
        Fit the diagonal part of the rotational invariant polarizability matrix, which will be converted to normal polarizability matrix by contracting with the rotation matrix.
    """

    def __init__(
        self,
        var_name: str,
        ntypes: int,
        dim_descrpt: int,
        embedding_width: int,
        neuron: List[int] = [128, 128, 128],
        resnet_dt: bool = True,
        numb_fparam: int = 0,
        numb_aparam: int = 0,
        activation_function: str = "tanh",
        precision: str = DEFAULT_PRECISION,
        distinguish_types: bool = False,
        rcond: Optional[float] = None,
        seed: Optional[int] = None,
        exclude_types: List[int] = [],
        fit_diag: bool = True,
        **kwargs,
    ):
        self.embedding_width = embedding_width
        self.fit_diag = fit_diag
        super().__init__(
            var_name=var_name,
            ntypes=ntypes,
            dim_descrpt=dim_descrpt,
            neuron=neuron,
            resnet_dt=resnet_dt,
            numb_fparam=numb_fparam,
            numb_aparam=numb_aparam,
            activation_function=activation_function,
            precision=precision,
            distinguish_types=distinguish_types,
            rcond=rcond,
            seed=seed,
            exclude_types=exclude_types,
            **kwargs,
        )
        self.old_impl = False  # this only supports the new implementation.

    def _net_out_dim(self):
        """Set the FittingNet output dim."""
        return (
            self.embedding_width
            if self.fit_diag
            else self.embedding_width * self.embedding_width
        )

    def serialize(self) -> dict:
        data = super().serialize()
        data["embedding_width"] = self.embedding_width
        data["old_impl"] = self.old_impl
        data["fit_diag"] = self.fit_diag
        return data

    def output_def(self) -> FittingOutputDef:
        return FittingOutputDef(
            [
                OutputVariableDef(
                    self.var_name,
                    [3, 3],
                    reduciable=True,
                    r_differentiable=True,
                    c_differentiable=True,
                ),
            ]
        )

    @property
    def data_stat_key(self):
        """
        Get the keys for the data statistic of the fitting.
        Return a list of statistic names needed, such as "bias_atom_e".
        """
        return []

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
        assert (
            gr is not None
        ), "Must provide the rotation matrix for polarizability fitting."
        # (nframes, nloc, m1)
        out = self._forward_common(descriptor, atype, gr, g2, h2, fparam, aparam)[
            self.var_name
        ]
        if self.fit_diag:
            out = out.view(-1, self.embedding_width)  # (nframes * nloc, m1)
            out = torch.diag_embed(out)  # (nframes * nloc, m1, m1)
        else:
            out = out.view(
                -1, self.embedding_width, self.embedding_width
            )  # (nframes * nloc, m1, m1)

        out = out + out.transpose(1, 2)
        gr = gr.view(nframes * nloc, -1, 3)  # (nframes * nloc, m1, 3)
        out = torch.bmm(out, gr)  # (nframes * nloc, m1, 3)

        out = torch.bmm(gr.transpose(1, 2), out)  # (nframes * nloc, 3, 3)
        out = out.view(nframes, nloc, 3, 3)

        return {self.var_name: out.to(env.GLOBAL_PT_FLOAT_PRECISION)}
