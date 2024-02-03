# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
import logging
from typing import (
    List,
    Optional,
    Tuple,
)

import numpy as np
import torch

from deepmd.dpmodel import (
    FittingOutputDef,
    OutputVariableDef,
    fitting_check_output,
)
from deepmd.pt.model.network.mlp import (
    FittingNet,
    NetworkCollection,
)
from deepmd.pt.model.network.network import (
    ResidualDeep,
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
from deepmd.pt.utils.utils import (
    to_numpy_array,
    to_torch_tensor,
)

dtype = env.GLOBAL_PT_FLOAT_PRECISION
device = env.DEVICE


@fitting_check_output
class InvarFitting(Fitting):
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
        """Construct a fitting net for energy.

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
        if bias_atom_e is None:
            bias_atom_e = np.zeros([self.ntypes, self.dim_out])
        bias_atom_e = torch.tensor(bias_atom_e, dtype=self.prec, device=device)
        bias_atom_e = bias_atom_e.view([self.ntypes, self.dim_out])
        if not self.use_tebd:
            assert self.ntypes == bias_atom_e.shape[0], "Element count mismatches!"
        self.register_buffer("bias_atom_e", bias_atom_e)
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
        out_dim = 1

        self.old_impl = kwargs.get("old_impl", False)
        if self.old_impl:
            filter_layers = []
            for type_i in range(self.ntypes):
                bias_type = 0.0
                one = ResidualDeep(
                    type_i,
                    self.dim_descrpt,
                    self.neuron,
                    bias_type,
                    resnet_dt=self.resnet_dt,
                )
                filter_layers.append(one)
            self.filter_layers_old = torch.nn.ModuleList(filter_layers)
            self.filter_layers = None
        else:
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
            self.filter_layers_old = None

        # very bad design...
        if "seed" in kwargs:
            logging.info("Set seed to %d in fitting net.", kwargs["seed"])
            torch.manual_seed(kwargs["seed"])

    def output_def(self) -> FittingOutputDef:
        return FittingOutputDef(
            [
                OutputVariableDef(
                    self.var_name, [self.dim_out], reduciable=True, differentiable=True
                ),
            ]
        )

    def __setitem__(self, key, value):
        if key in ["bias_atom_e"]:
            # correct bias_atom_e shape. user may provide stupid  shape
            self.bias_atom_e = value
        elif key in ["fparam_avg"]:
            self.fparam_avg = value
        elif key in ["fparam_inv_std"]:
            self.fparam_inv_std = value
        elif key in ["aparam_avg"]:
            self.aparam_avg = value
        elif key in ["aparam_inv_std"]:
            self.aparam_inv_std = value
        else:
            raise KeyError(key)

    def __getitem__(self, key):
        if key in ["bias_atom_e"]:
            return self.bias_atom_e
        elif key in ["fparam_avg"]:
            return self.fparam_avg
        elif key in ["fparam_inv_std"]:
            return self.fparam_inv_std
        elif key in ["aparam_avg"]:
            return self.aparam_avg
        elif key in ["aparam_inv_std"]:
            return self.aparam_inv_std
        else:
            raise KeyError(key)

    def serialize(self) -> dict:
        """Serialize the fitting to dict."""
        return {
            "var_name": self.var_name,
            "ntypes": self.ntypes,
            "dim_descrpt": self.dim_descrpt,
            "dim_out": self.dim_out,
            "neuron": self.neuron,
            "resnet_dt": self.resnet_dt,
            "numb_fparam": self.numb_fparam,
            "numb_aparam": self.numb_aparam,
            "activation_function": self.activation_function,
            "precision": self.precision,
            "distinguish_types": self.distinguish_types,
            "nets": self.filter_layers.serialize(),
            "@variables": {
                "bias_atom_e": to_numpy_array(self.bias_atom_e),
                "fparam_avg": to_numpy_array(self.fparam_avg),
                "fparam_inv_std": to_numpy_array(self.fparam_inv_std),
                "aparam_avg": to_numpy_array(self.aparam_avg),
                "aparam_inv_std": to_numpy_array(self.aparam_inv_std),
            },
            # "rcond": self.rcond ,
            # "tot_ener_zero": self.tot_ener_zero ,
            # "trainable": self.trainable ,
            # "atom_ener": self.atom_ener ,
            # "layer_name": self.layer_name ,
            # "use_aparam_as_mask": self.use_aparam_as_mask ,
            # "spin": self.spin ,
            ## NOTICE:  not supported by far
            "rcond": None,
            "tot_ener_zero": False,
            "trainable": True,
            "atom_ener": None,
            "layer_name": None,
            "use_aparam_as_mask": False,
            "spin": None,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "InvarFitting":
        data = copy.deepcopy(data)
        variables = data.pop("@variables")
        nets = data.pop("nets")
        obj = cls(**data)
        for kk in variables.keys():
            obj[kk] = to_torch_tensor(variables[kk])
        obj.filter_layers = NetworkCollection.deserialize(nets)
        return obj

    def _extend_f_avg_std(self, xx: torch.Tensor, nb: int) -> torch.Tensor:
        return torch.tile(xx.view([1, self.numb_fparam]), [nb, 1])

    def _extend_a_avg_std(self, xx: torch.Tensor, nb: int, nloc: int) -> torch.Tensor:
        return torch.tile(xx.view([1, 1, self.numb_aparam]), [nb, nloc, 1])

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
        xx = descriptor
        nf, nloc, nd = xx.shape
        # NOTICE in tests/pt/test_model.py
        # it happens that the user directly access the data memeber self.bias_atom_e
        # and set it to a wrong shape!
        self.bias_atom_e = self.bias_atom_e.view([self.ntypes, self.dim_out])
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
            fparam = fparam.view([nf, self.numb_fparam])
            nb, _ = fparam.shape
            t_fparam_avg = self._extend_f_avg_std(self.fparam_avg, nb)
            t_fparam_inv_std = self._extend_f_avg_std(self.fparam_inv_std, nb)
            fparam = (fparam - t_fparam_avg) * t_fparam_inv_std
            fparam = torch.tile(fparam.reshape([nf, 1, -1]), [1, nloc, 1])
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
            aparam = aparam.view([nf, nloc, self.numb_aparam])
            nb, nloc, _ = aparam.shape
            t_aparam_avg = self._extend_a_avg_std(self.aparam_avg, nb, nloc)
            t_aparam_inv_std = self._extend_a_avg_std(self.aparam_inv_std, nb, nloc)
            aparam = (aparam - t_aparam_avg) * t_aparam_inv_std
            xx = torch.cat(
                [xx, aparam],
                dim=-1,
            )

        outs = torch.zeros_like(atype).unsqueeze(-1)  # jit assertion
        if self.old_impl:
            outs = torch.zeros_like(atype).unsqueeze(-1)  # jit assertion
            assert self.filter_layers_old is not None
            if self.use_tebd:
                atom_energy = self.filter_layers_old[0](xx) + self.bias_atom_e[
                    atype
                ].unsqueeze(-1)
                outs = outs + atom_energy  # Shape is [nframes, natoms[0], 1]
            else:
                for type_i, filter_layer in enumerate(self.filter_layers_old):
                    mask = atype == type_i
                    atom_energy = filter_layer(xx)
                    atom_energy = atom_energy + self.bias_atom_e[type_i]
                    atom_energy = atom_energy * mask.unsqueeze(-1)
                    outs = outs + atom_energy  # Shape is [nframes, natoms[0], 1]
            return {"energy": outs.to(env.GLOBAL_PT_FLOAT_PRECISION)}
        else:
            if self.use_tebd:
                atom_energy = (
                    self.filter_layers.networks[0](xx) + self.bias_atom_e[atype]
                )
                outs = outs + atom_energy  # Shape is [nframes, natoms[0], 1]
            else:
                for type_i, ll in enumerate(self.filter_layers.networks):
                    mask = (atype == type_i).unsqueeze(-1)
                    mask = torch.tile(mask, (1, 1, self.dim_out))
                    atom_energy = ll(xx)
                    atom_energy = atom_energy + self.bias_atom_e[type_i]
                    atom_energy = atom_energy * mask
                    outs = outs + atom_energy  # Shape is [nframes, natoms[0], 1]
            return {self.var_name: outs.to(env.GLOBAL_PT_FLOAT_PRECISION)}


@Fitting.register("ener")
class EnergyFittingNet(InvarFitting):
    def __init__(
        self,
        ntypes: int,
        embedding_width: int,
        neuron: List[int] = [128, 128, 128],
        bias_atom_e: Optional[torch.Tensor] = None,
        resnet_dt: bool = True,
        numb_fparam: int = 0,
        numb_aparam: int = 0,
        activation_function: str = "tanh",
        precision: str = DEFAULT_PRECISION,
        use_tebd: bool = True,
        **kwargs,
    ):
        super().__init__(
            "energy",
            ntypes,
            embedding_width,
            1,
            neuron=neuron,
            bias_atom_e=bias_atom_e,
            resnet_dt=resnet_dt,
            numb_fparam=numb_fparam,
            numb_aparam=numb_aparam,
            activation_function=activation_function,
            precision=precision,
            use_tebd=use_tebd,
            **kwargs,
        )


@Fitting.register("direct_force")
@Fitting.register("direct_force_ener")
@fitting_check_output
class EnergyFittingNetDirect(Fitting):
    def __init__(
        self,
        ntypes,
        embedding_width,
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
        self.dim_descrpt = embedding_width
        self.use_tebd = use_tebd
        self.out_dim = out_dim
        if bias_atom_e is None:
            bias_atom_e = np.zeros([self.ntypes])
        if not use_tebd:
            assert self.ntypes == len(bias_atom_e), "Element count mismatches!"
        bias_atom_e = torch.tensor(bias_atom_e)
        self.register_buffer("bias_atom_e", bias_atom_e)

        filter_layers_dipole = []
        for type_i in range(self.ntypes):
            one = ResidualDeep(
                type_i,
                embedding_width,
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
                    type_i, embedding_width, neuron, bias_type, resnet_dt=resnet_dt
                )
                filter_layers.append(one)
        self.filter_layers = torch.nn.ModuleList(filter_layers)

        if "seed" in kwargs:
            logging.info("Set seed to %d in fitting net.", kwargs["seed"])
            torch.manual_seed(kwargs["seed"])

    def output_def(self):
        return FittingOutputDef(
            [
                OutputVariableDef("energy", [1], reduciable=True, differentiable=False),
                OutputVariableDef(
                    "dforce", [3], reduciable=False, differentiable=False
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
