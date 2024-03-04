# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
import logging
from typing import (
    Callable,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import torch

from deepmd.dpmodel import (
    FittingOutputDef,
    OutputVariableDef,
    fitting_check_output,
)
from deepmd.pt.model.network.network import (
    ResidualDeep,
)
from deepmd.pt.model.task.fitting import (
    Fitting,
    GeneralFitting,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env import (
    DEFAULT_PRECISION,
)
from deepmd.pt.utils.utils import (
    to_numpy_array,
)
from deepmd.utils.out_stat import (
    compute_stats_from_redu,
)
from deepmd.utils.path import (
    DPPath,
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
    atom_ener
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
        if stat_file_path is not None:
            stat_file_path = stat_file_path / "bias_atom_e"
        if stat_file_path is not None and stat_file_path.is_file():
            bias_atom_e = stat_file_path.load_numpy()
        else:
            if callable(merged):
                # only get data for once
                sampled = merged()
            else:
                sampled = merged
            energy = [item["energy"] for item in sampled]
            data_mixed_type = "real_natoms_vec" in sampled[0]
            if data_mixed_type:
                input_natoms = [item["real_natoms_vec"] for item in sampled]
            else:
                input_natoms = [item["natoms"] for item in sampled]
            # shape: (nframes, ndim)
            merged_energy = to_numpy_array(torch.cat(energy))
            # shape: (nframes, ntypes)
            merged_natoms = to_numpy_array(torch.cat(input_natoms)[:, 2:])
            if self.atom_ener is not None and len(self.atom_ener) > 0:
                assigned_atom_ener = np.array(
                    [ee if ee is not None else np.nan for ee in self.atom_ener]
                )
            else:
                assigned_atom_ener = None
            bias_atom_e, _ = compute_stats_from_redu(
                merged_energy,
                merged_natoms,
                assigned_bias=assigned_atom_ener,
                rcond=self.rcond,
            )
            if stat_file_path is not None:
                stat_file_path.save_numpy(bias_atom_e)
        assert all(x is not None for x in [bias_atom_e])
        self.bias_atom_e.copy_(
            torch.tensor(bias_atom_e, device=env.DEVICE).view(
                [self.ntypes, self.dim_out]
            )
        )

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


@Fitting.register("ener")
class EnergyFittingNet(InvarFitting):
    def __init__(
        self,
        ntypes: int,
        dim_descrpt: int,
        neuron: List[int] = [128, 128, 128],
        bias_atom_e: Optional[torch.Tensor] = None,
        resnet_dt: bool = True,
        numb_fparam: int = 0,
        numb_aparam: int = 0,
        activation_function: str = "tanh",
        precision: str = DEFAULT_PRECISION,
        mixed_types: bool = True,
        **kwargs,
    ):
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
            activation_function=activation_function,
            precision=precision,
            mixed_types=mixed_types,
            **kwargs,
        )

    @classmethod
    def deserialize(cls, data: dict) -> "GeneralFitting":
        data = copy.deepcopy(data)
        data.pop("var_name")
        data.pop("dim_out")
        return super().deserialize(data)

    def serialize(self) -> dict:
        """Serialize the fitting to dict."""
        return {
            **super().serialize(),
            "type": "ener",
        }


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
    ):
        """Construct a fitting net for energy.

        Args:
        - ntypes: Element count.
        - embedding_width: Embedding width per atom.
        - neuron: Number of neurons in each hidden layers of the fitting net.
        - bias_atom_e: Average enery per atom for each element.
        - resnet_dt: Using time-step in the ResNet construction.
        """
        super().__init__()
        self.ntypes = ntypes
        self.dim_descrpt = dim_descrpt
        self.use_tebd = use_tebd
        self.out_dim = out_dim
        if bias_atom_e is None:
            bias_atom_e = np.zeros([self.ntypes])
        if not use_tebd:
            assert self.ntypes == len(bias_atom_e), "Element count mismatches!"
        bias_atom_e = torch.tensor(bias_atom_e, device=env.DEVICE)
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

        if "seed" in kwargs:
            torch.manual_seed(kwargs["seed"])

    def output_def(self):
        return FittingOutputDef(
            [
                OutputVariableDef(
                    "energy",
                    [1],
                    reduciable=True,
                    r_differentiable=False,
                    c_differentiable=False,
                ),
                OutputVariableDef(
                    "dforce",
                    [3],
                    reduciable=False,
                    r_differentiable=False,
                    c_differentiable=False,
                ),
            ]
        )

    def serialize(self) -> dict:
        raise NotImplementedError

    def deserialize(cls) -> "EnergyFittingNetDirect":
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
    ) -> Tuple[torch.Tensor, None]:
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
