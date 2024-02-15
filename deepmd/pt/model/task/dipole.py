# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from typing import (
    List,
    Optional,
)

import torch

from deepmd.pt.model.network.mlp import (
    FittingNet,
    NetworkCollection,
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

dtype = env.GLOBAL_PT_FLOAT_PRECISION
device = env.DEVICE

log = logging.getLogger(__name__)


class DipoleFittingNet(GeneralFitting):
    def __init__(
        self,
        var_name: str,
        ntypes: int,
        dim_descrpt: int,
        dim_out: int,
        neuron: List[int] = [128, 128, 128],
        resnet_dt: bool = True,
        numb_fparam: int = 0,
        numb_aparam: int = 0,
        activation_function: str = "tanh",
        precision: str = DEFAULT_PRECISION,
        distinguish_types: bool = False,
        **kwargs,
    ):
        """Construct a fitting net for dipole.

        Args:
        - ntypes: Element count.
        - embedding_width: Embedding width per atom.
        - neuron: Number of neurons in each hidden layers of the fitting net.
        - bias_atom_e: Average enery per atom for each element.
        - resnet_dt: Using time-step in the ResNet construction.
        """
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
            **kwargs,
        )
        self.old_impl = False # this only supports the new implementation.

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
        - inputs: Descriptor. Its shape is [nframes, nloc, self.embedding_width].
        - atype: Atom type. Its shape is [nframes, nloc].
        - atype_tebd: Atom type embedding. Its shape is [nframes, nloc, tebd_dim]
        - rot_mat: GR during descriptor calculation. Its shape is [nframes * nloc, m1, 3].

        Returns
        -------
        - vec_out: output vector. Its shape is [nframes, nloc, 3].
        """
        in_dim = self.dim_descrpt + self.numb_fparam + self.numb_aparam
        
        nframes, nloc, _ = descriptor.shape
        gr = gr.view(nframes, nloc, -1, 3)
        out_dim = gr.shape[2] # m1
        self.filter_layers = NetworkCollection(
            1 if self.distinguish_types else 0,
            self.ntypes,
            network_type="fitting_network",
            networks=[
                FittingNet(
                    in_dim,
                    out_dim,
                    self.neuron,
                    self.activation_function,
                    self.resnet_dt,
                    self.precision,
                    bias_out=True,
                )
                for ii in range(self.ntypes if self.distinguish_types else 1)
            ],
        )
        # (nframes, nloc, m1)
        out = self._forward_common(descriptor, atype, gr, g2, h2, fparam, aparam)[self.var_name]
        # (nframes * nloc, 1, m1)
        out = out.view(-1, 1, out_dim)
        # (nframes, nloc, 3)
        out = (
            torch.bmm(out, gr).squeeze(-2).view(nframes, nloc, 3)
        )  
        return {self.var_name: out.to(env.GLOBAL_PT_FLOAT_PRECISION)}