# SPDX-License-Identifier: LGPL-3.0-or-later
import logging

import torch

from deepmd.pt.model.network.network import (
    ResidualDeep,
)
from deepmd.pt.model.task.fitting import (
    Fitting,
)

log = logging.getLogger(__name__)


class DipoleFittingNetType(Fitting):
    def __init__(
        self, ntypes, embedding_width, neuron, out_dim, resnet_dt=True, **kwargs
    ):
        """Construct a fitting net for dipole.

        Args:
        - ntypes: Element count.
        - embedding_width: Embedding width per atom.
        - neuron: Number of neurons in each hidden layers of the fitting net.
        - bias_atom_e: Average enery per atom for each element.
        - resnet_dt: Using time-step in the ResNet construction.
        """
        super().__init__()
        self.ntypes = ntypes
        self.embedding_width = embedding_width
        self.out_dim = out_dim

        filter_layers = []
        one = ResidualDeep(
            0, embedding_width, neuron, 0.0, out_dim=self.out_dim, resnet_dt=resnet_dt
        )
        filter_layers.append(one)
        self.filter_layers = torch.nn.ModuleList(filter_layers)

        if "seed" in kwargs:
            log.info("Set seed to %d in fitting net.", kwargs["seed"])
            torch.manual_seed(kwargs["seed"])

    def forward(self, inputs, atype, atype_tebd, rot_mat):
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
        nframes, nloc, _ = inputs.size()
        if atype_tebd is not None:
            inputs = torch.concat([inputs, atype_tebd], dim=-1)
        vec_out = self.filter_layers[0](inputs)  # Shape is [nframes, nloc, m1]
        assert list(vec_out.size()) == [nframes, nloc, self.out_dim]
        vec_out = vec_out.view(-1, 1, self.out_dim)
        vec_out = (
            torch.bmm(vec_out, rot_mat).squeeze(-2).view(nframes, nloc, 3)
        )  # Shape is [nframes, nloc, 3]
        return vec_out
