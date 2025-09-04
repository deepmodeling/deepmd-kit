# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from typing import (
    Any,
    Optional,
    Union,
)

import numpy as np
import torch

from deepmd.dpmodel import (
    FittingOutputDef,
    OutputVariableDef,
    fitting_check_output,
)
from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.pt.model.network.mlp import (
    FittingNet,
    NetworkCollection,
)
from deepmd.pt.model.network.network import (
    ResidualDeep,
)
from deepmd.pt.model.network.utils import (
    aggregate,
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
from deepmd.utils.version import (
    check_version_compatibility,
)

dtype = env.GLOBAL_PT_FLOAT_PRECISION
device = env.DEVICE

log = logging.getLogger(__name__)


@Fitting.register("ener")
class EnergyFittingNet(InvarFitting):
    def __init__(
        self,
        ntypes: int,
        dim_descrpt: int,
        neuron: list[int] = [128, 128, 128],
        bias_atom_e: Optional[torch.Tensor] = None,
        resnet_dt: bool = True,
        numb_fparam: int = 0,
        numb_aparam: int = 0,
        dim_case_embd: int = 0,
        activation_function: str = "tanh",
        precision: str = DEFAULT_PRECISION,
        mixed_types: bool = True,
        seed: Optional[Union[int, list[int]]] = None,
        type_map: Optional[list[str]] = None,
        default_fparam: Optional[list] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            "energy",
            ntypes,
            dim_descrpt,
            1,
            neuron=neuron,
            bias_atom_e=bias_atom_e,
            resnet_dt=resnet_dt,
            numb_fparam=numb_fparam,
            numb_aparam=numb_aparam,
            dim_case_embd=dim_case_embd,
            activation_function=activation_function,
            precision=precision,
            mixed_types=mixed_types,
            seed=seed,
            type_map=type_map,
            default_fparam=default_fparam,
            **kwargs,
        )

    @classmethod
    def deserialize(cls, data: dict) -> "GeneralFitting":
        data = data.copy()
        check_version_compatibility(data.pop("@version", 1), 4, 1)
        data.pop("var_name")
        data.pop("dim_out")
        return super().deserialize(data)

    def serialize(self) -> dict:
        """Serialize the fitting to dict."""
        return {
            **super().serialize(),
            "type": "ener",
        }

    # make jit happy with torch 2.0.0
    exclude_types: list[int]


@Fitting.register("direct_force")
@Fitting.register("direct_force_ener")
@fitting_check_output
class EnergyFittingNetDirect(Fitting):
    def __init__(
        self,
        ntypes,
        dim_descrpt,
        neuron,
        bias_atom_e=None,
        out_dim=1,
        resnet_dt=True,
        use_tebd=True,
        return_energy=False,
        **kwargs,
    ) -> None:
        """Construct a fitting net for energy.

        Args:
        - ntypes: Element count.
        - embedding_width: Embedding width per atom.
        - neuron: Number of neurons in each hidden layers of the fitting net.
        - bias_atom_e: Average energy per atom for each element.
        - resnet_dt: Using time-step in the ResNet construction.
        """
        super().__init__()
        self.ntypes = ntypes
        self.dim_descrpt = dim_descrpt
        self.use_tebd = use_tebd
        self.out_dim = out_dim
        if bias_atom_e is None:
            # place holder, dtype does not matter
            bias_atom_e = np.zeros([self.ntypes], dtype=np.float64)
        if not use_tebd:
            assert self.ntypes == len(bias_atom_e), "Element count mismatches!"
        bias_atom_e = torch.tensor(
            bias_atom_e, device=env.DEVICE, dtype=env.GLOBAL_PT_FLOAT_PRECISION
        )
        self.register_buffer("bias_atom_e", bias_atom_e)

        filter_layers_dipole = []
        for type_i in range(self.ntypes):
            one = ResidualDeep(
                type_i,
                dim_descrpt,
                neuron,
                0.0,
                out_dim=out_dim,
                resnet_dt=resnet_dt,
            )
            filter_layers_dipole.append(one)
        self.filter_layers_dipole = torch.nn.ModuleList(filter_layers_dipole)

        self.return_energy = return_energy
        filter_layers = []
        if self.return_energy:
            for type_i in range(self.ntypes):
                bias_type = 0.0 if self.use_tebd else bias_atom_e[type_i]
                one = ResidualDeep(
                    type_i, dim_descrpt, neuron, bias_type, resnet_dt=resnet_dt
                )
                filter_layers.append(one)
        self.filter_layers = torch.nn.ModuleList(filter_layers)

    def output_def(self):
        return FittingOutputDef(
            [
                OutputVariableDef(
                    "energy",
                    [1],
                    reducible=True,
                    r_differentiable=False,
                    c_differentiable=False,
                ),
                OutputVariableDef(
                    "dforce",
                    [3],
                    reducible=False,
                    r_differentiable=False,
                    c_differentiable=False,
                ),
            ]
        )

    def serialize(self) -> dict:
        raise NotImplementedError

    def deserialize(self) -> "EnergyFittingNetDirect":
        raise NotImplementedError

    def change_type_map(
        self, type_map: list[str], model_with_new_type_stat=None
    ) -> None:
        raise NotImplementedError

    def get_type_map(self) -> list[str]:
        raise NotImplementedError

    def forward(
        self,
        inputs: torch.Tensor,
        atype: torch.Tensor,
        gr: Optional[torch.Tensor] = None,
        g2: Optional[torch.Tensor] = None,
        h2: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, None]:
        """Based on embedding net output, alculate total energy.

        Args:
        - inputs: Embedding matrix. Its shape is [nframes, natoms[0], self.dim_descrpt].
        - natoms: Tell atom count and element count. Its shape is [2+self.ntypes].

        Returns
        -------
        - `torch.Tensor`: Total energy with shape [nframes, natoms[0]].
        """
        nframes, nloc, _ = inputs.size()
        if self.use_tebd:
            # if atype_tebd is not None:
            #     inputs = torch.concat([inputs, atype_tebd], dim=-1)
            vec_out = self.filter_layers_dipole[0](
                inputs
            )  # Shape is [nframes, nloc, m1]
            assert list(vec_out.size()) == [nframes, nloc, self.out_dim]
            # (nf x nloc) x 1 x od
            vec_out = vec_out.view(-1, 1, self.out_dim)
            assert gr is not None
            # (nf x nloc) x od x 3
            gr = gr.view(-1, self.out_dim, 3)
            vec_out = (
                torch.bmm(vec_out, gr).squeeze(-2).view(nframes, nloc, 3)
            )  # Shape is [nframes, nloc, 3]
        else:
            vec_out = torch.zeros_like(atype).unsqueeze(-1)  # jit assertion
            for type_i, filter_layer in enumerate(self.filter_layers_dipole):
                mask = atype == type_i
                vec_out_type = filter_layer(inputs)  # Shape is [nframes, nloc, m1]
                vec_out_type = vec_out_type * mask.unsqueeze(-1)
                vec_out = vec_out + vec_out_type  # Shape is [nframes, natoms[0], 1]

        outs = torch.zeros_like(atype).unsqueeze(-1)  # jit assertion
        if self.return_energy:
            if self.use_tebd:
                atom_energy = self.filter_layers[0](inputs) + self.bias_atom_e[
                    atype
                ].unsqueeze(-1)
                outs = outs + atom_energy  # Shape is [nframes, natoms[0], 1]
            else:
                for type_i, filter_layer in enumerate(self.filter_layers):
                    mask = atype == type_i
                    atom_energy = filter_layer(inputs)
                    if not env.ENERGY_BIAS_TRAINABLE:
                        atom_energy = atom_energy + self.bias_atom_e[type_i]
                    atom_energy = atom_energy * mask.unsqueeze(-1)
                    outs = outs + atom_energy  # Shape is [nframes, natoms[0], 1]
        return {
            "energy": outs.to(env.GLOBAL_PT_FLOAT_PRECISION),
            "dforce": vec_out,
        }


@Fitting.register("ener_readout")
@fitting_check_output
class EnergyFittingNetReadout(InvarFitting):
    def __init__(
        self,
        ntypes: int,
        dim_descrpt: int,
        neuron: list[int] = [128, 128, 128],
        bias_atom_e: Optional[torch.Tensor] = None,
        resnet_dt: bool = True,
        numb_fparam: int = 0,
        numb_aparam: int = 0,
        dim_case_embd: int = 0,
        embedding_width: int = 128,
        activation_function: str = "tanh",
        precision: str = DEFAULT_PRECISION,
        mixed_types: bool = True,
        seed: Optional[Union[int, list[int]]] = None,
        type_map: Optional[list[str]] = None,
        norm_fact: list[float] = [120.0],
        add_edge_readout: bool = True,
        slim_edge_readout: bool = False,
        **kwargs,
    ) -> None:
        """Construct a fitting net for energy.

        Args:
        - ntypes: Element count.
        - embedding_width: Embedding width per atom.
        - neuron: Number of neurons in each hidden layers of the fitting net.
        - bias_atom_e: Average energy per atom for each element.
        - resnet_dt: Using time-step in the ResNet construction.
        """
        self.add_edge_readout = add_edge_readout
        super().__init__(
            "energy",
            ntypes,
            dim_descrpt,
            1,
            neuron=neuron,
            bias_atom_e=bias_atom_e,
            resnet_dt=resnet_dt,
            numb_fparam=numb_fparam,
            numb_aparam=numb_aparam,
            dim_case_embd=dim_case_embd,
            activation_function=activation_function,
            precision=precision,
            mixed_types=mixed_types,
            seed=seed,
            type_map=type_map,
            **kwargs,
        )

        # embedding for edge readout
        self.embedding_width = embedding_width
        self.slim_edge_readout = slim_edge_readout
        self.norm_e_fact = norm_fact[0]

        if self.add_edge_readout:
            self.edge_embed = NetworkCollection(
                1 if not self.mixed_types else 0,
                self.ntypes,
                network_type="fitting_network",
                networks=[
                    FittingNet(
                        self.embedding_width,
                        1,
                        self.neuron if not self.slim_edge_readout else self.neuron[:1],
                        self.activation_function,
                        self.resnet_dt,
                        self.precision,
                        bias_out=True,
                        seed=child_seed(self.seed + 100, ii),
                    )
                    for ii in range(self.ntypes if not self.mixed_types else 1)
                ],
            )
        else:
            self.edge_embed = None

        # set trainable
        for param in self.parameters():
            param.requires_grad = self.trainable

    # make jit happy with torch 2.0.0
    exclude_types: list[int]

    def need_additional_input(self) -> bool:
        return True

    def serialize(self) -> dict:
        raise NotImplementedError

    @classmethod
    def deserialize(cls, data: dict) -> "EnergyFittingNetReadout":
        raise NotImplementedError

    def forward(
        self,
        descriptor: torch.Tensor,
        atype: torch.Tensor,
        gr: Optional[torch.Tensor] = None,
        g2: Optional[torch.Tensor] = None,
        h2: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
        sw: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
    ):
        """Based on embedding net output, alculate total energy.

        Args:
        - inputs: Embedding matrix. Its shape is [nframes, natoms[0], self.dim_descrpt].
        - natoms: Tell atom count and element count. Its shape is [2+self.ntypes].

        Returns
        -------
        - `torch.Tensor`: Total energy with shape [nframes, natoms[0]].
        """
        out = self._forward_common(descriptor, atype, gr, g2, h2, fparam, aparam)[
            self.var_name
        ]
        nf, nloc, _ = descriptor.shape

        if self.add_edge_readout:
            assert g2 is not None
            assert sw is not None
            assert self.edge_embed is not None
            # nf x nloc x nnei x d [OR] nedge x d
            edge_feature = g2
            # nf x nloc x nnei x 1 [OR] nedge x 1
            edge_atomic_contrib = self.edge_embed.networks[0](edge_feature)
            # nf x nloc x nnei x 1 [OR] nedge x 1
            edge_atomic_contrib = edge_atomic_contrib * sw.unsqueeze(-1)
            if edge_index is not None:
                # use dynamic sel
                n2e_index, n_ext2e_index = edge_index[0], edge_index[1]
                # nf x nloc x 1
                edge_energy = aggregate(
                    edge_atomic_contrib,
                    n2e_index,
                    average=False,
                    num_owner=nf * nloc,
                ).reshape(nf, nloc, 1)
            else:
                # nf x nloc x 1
                edge_energy = torch.sum(edge_atomic_contrib, dim=-2)
            # energy
            out = out + edge_energy / self.norm_e_fact
        return {self.var_name: out.to(env.GLOBAL_PT_FLOAT_PRECISION)}
