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
    Fitting,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env import (
    DEFAULT_PRECISION,
    PRECISION_DICT,
)

dtype = env.GLOBAL_PT_FLOAT_PRECISION
device = env.DEVICE

log = logging.getLogger(__name__)


class DipoleFittingNet(Fitting):
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
        super().__init__()
        self.var_name = var_name
        self.ntypes = ntypes
        self.dim_descrpt = dim_descrpt
        self.dim_out = dim_out
        self.neuron = neuron
        self.distinguish_types = distinguish_types
        self.use_tebd = not self.distinguish_types
        self.resnet_dt = resnet_dt
        self.numb_fparam = numb_fparam
        self.numb_aparam = numb_aparam
        self.activation_function = activation_function
        self.precision = precision
        self.prec = PRECISION_DICT[self.precision]

        # init constants
        if self.numb_fparam > 0:
            self.register_buffer(
                "fparam_avg",
                torch.zeros(self.numb_fparam, dtype=self.prec, device=device),
            )
            self.register_buffer(
                "fparam_inv_std",
                torch.ones(self.numb_fparam, dtype=self.prec, device=device),
            )
        else:
            self.fparam_avg, self.fparam_inv_std = None, None
        if self.numb_aparam > 0:
            self.register_buffer(
                "aparam_avg",
                torch.zeros(self.numb_aparam, dtype=self.prec, device=device),
            )
            self.register_buffer(
                "aparam_inv_std",
                torch.ones(self.numb_aparam, dtype=self.prec, device=device),
            )
        else:
            self.aparam_avg, self.aparam_inv_std = None, None

        in_dim = self.dim_descrpt + self.numb_fparam + self.numb_aparam
        out_dim = 3

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

        if "seed" in kwargs:
            log.info("Set seed to %d in fitting net.", kwargs["seed"])
            torch.manual_seed(kwargs["seed"])

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
        xx = descriptor
        nframes, nloc, nd = xx.shape
        # check input dim
        if nd != self.dim_descrpt:
            raise ValueError(
                "get an input descriptor of dim {nd},"
                "which is not consistent with {self.dim_descrpt}."
            )
        # check fparam dim, concate to input descriptor
        if self.numb_fparam > 0:
            assert fparam is not None, "fparam should not be None"
            assert self.fparam_avg is not None
            assert self.fparam_inv_std is not None
            if fparam.shape[-1] != self.numb_fparam:
                raise ValueError(
                    "get an input fparam of dim {fparam.shape[-1]}, ",
                    "which is not consistent with {self.numb_fparam}.",
                )
            fparam = fparam.view([nframes, self.numb_fparam])
            nb, _ = fparam.shape
            t_fparam_avg = self._extend_f_avg_std(self.fparam_avg, nb)
            t_fparam_inv_std = self._extend_f_avg_std(self.fparam_inv_std, nb)
            fparam = (fparam - t_fparam_avg) * t_fparam_inv_std
            fparam = torch.tile(fparam.reshape([nframes, 1, -1]), [1, nloc, 1])
            xx = torch.cat(
                [xx, fparam],
                dim=-1,
            )
        # check aparam dim, concate to input descriptor
        if self.numb_aparam > 0:
            assert aparam is not None, "aparam should not be None"
            assert self.aparam_avg is not None
            assert self.aparam_inv_std is not None
            if aparam.shape[-1] != self.numb_aparam:
                raise ValueError(
                    "get an input aparam of dim {aparam.shape[-1]}, ",
                    "which is not consistent with {self.numb_aparam}.",
                )
            aparam = aparam.view([nframes, nloc, self.numb_aparam])
            nb, nloc, _ = aparam.shape
            t_aparam_avg = self._extend_a_avg_std(self.aparam_avg, nb, nloc)
            t_aparam_inv_std = self._extend_a_avg_std(self.aparam_inv_std, nb, nloc)
            aparam = (aparam - t_aparam_avg) * t_aparam_inv_std
            xx = torch.cat(
                [xx, aparam],
                dim=-1,
            )

        outs = torch.zeros_like(atype).unsqueeze(-1)  # jit assertion
        if self.use_tebd:
            atom_dipole = self.filter_layers.networks[0](xx)
            outs = outs + atom_dipole  # Shape is [nframes, natoms[0], 3]
        else:
            for type_i, ll in enumerate(self.filter_layers.networks):
                mask = (atype == type_i).unsqueeze(-1)
                mask = torch.tile(mask, (1, 1, self.dim_out))
                atom_dipole = ll(xx)
                atom_dipole = atom_dipole * mask
                outs = outs + atom_dipole  # Shape is [nframes, natoms[0], 3]
        outs = (
            torch.bmm(outs, gr).squeeze(-2).view(nframes, nloc, 3)
        )  # Shape is [nframes, nloc, 3]
        return {self.var_name: outs.to(env.GLOBAL_PT_FLOAT_PRECISION)}
