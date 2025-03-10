# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from typing import (
    Optional,
    Union,
)

import numpy as np
import torch

from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.pt.model.network.mlp import (
    FittingNet,
    NetworkCollection,
)
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
from deepmd.pt.model.task.invar_fitting import (
    InvarFitting,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env import (
    PRECISION_DICT,
)
from deepmd.pt.utils.exclude_mask import (
    AtomExcludeMask,
)
from deepmd.pt.utils.env import (
    DEFAULT_PRECISION,
)

dtype = env.GLOBAL_PT_FLOAT_PRECISION
device = env.DEVICE

@Fitting.register("denoise")
@fitting_check_output
class DenoiseNet(Fitting):
    def __init__(
        self,
        ntypes,
        dim_descrpt,
        neuron,
        bias_atom_e=None,
        out_dim=1,
        resnet_dt=True,
        numb_fparam: int = 0,
        numb_aparam: int = 0,
        dim_case_embd: int = 0,
        activation_function: str = "tanh",
        precision: str = DEFAULT_PRECISION,
        mixed_types: bool = True,
        rcond: Optional[float] = None,
        seed: Optional[Union[int, list[int]]] = None,
        exclude_types: list[int] = [],
        trainable: Union[bool, list[bool]] = True,
        type_map: Optional[list[str]] = None,
        use_aparam_as_mask: bool = False,
        **kwargs,
    ) -> None:
        """Construct a direct token, coordinate and cell fitting net.

        Parameters
        ----------
        ntypes : int
            Element count.
        dim_descrpt : int
            Embedding width per atom.
        neuron : list[int]
            Number of neurons in each hidden layers of the fitting net.
        bias_atom_e : torch.Tensor, optional
            Average energy per atom for each element.
        resnet_dt : bool
            Using time-step in the ResNet construction.
        out_dim : int
            The output dimension of the fitting net.
        numb_fparam : int
            Number of frame parameters.
        numb_aparam : int
            Number of atomic parameters.
        dim_case_embd : int
            Dimension of case specific embedding.
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
        exclude_types: list[int]
            Atomic contributions of the excluded atom types are set zero.
        trainable : Union[list[bool], bool]
            If the parameters in the fitting net are trainable.
            Now this only supports setting all the parameters in the fitting net at one state.
            When in list[bool], the trainable will be True only if all the boolean parameters are True.
        type_map: list[str], Optional
            A list of strings. Give the name to each type of atoms.
        use_aparam_as_mask: bool
            If True, the aparam will not be used in fitting net for embedding.
        """
        super().__init__()
        self.ntypes = ntypes
        self.dim_descrpt = dim_descrpt
        self.neuron = neuron
        self.mixed_types = mixed_types
        self.resnet_dt = resnet_dt
        self.out_dim = out_dim
        self.numb_fparam = numb_fparam
        self.numb_aparam = numb_aparam
        self.dim_case_embd = dim_case_embd
        self.activation_function = activation_function
        self.precision = precision
        self.prec = PRECISION_DICT[self.precision]
        self.rcond = rcond
        self.seed = seed
        self.type_map = type_map
        self.use_aparam_as_mask = use_aparam_as_mask
        # order matters, should be place after the assignment of ntypes
        self.reinit_exclude(exclude_types)
        self.trainable = trainable
        # need support for each layer settings
        self.trainable = (
            all(self.trainable) if isinstance(self.trainable, list) else self.trainable
        )

        # init constants
        if bias_atom_e is None:
            bias_atom_e = np.zeros([self.ntypes, 1], dtype=np.float64)
        bias_atom_e = torch.tensor(
            bias_atom_e, device=env.DEVICE, dtype=env.GLOBAL_PT_FLOAT_PRECISION
        )
        bias_atom_e = bias_atom_e.view([self.ntypes, 1])
        if not self.mixed_types:
            assert self.ntypes == bias_atom_e.shape[0], "Element count mismatches!"
        self.register_buffer("bias_atom_e", bias_atom_e)

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

        if self.dim_case_embd > 0:
            self.register_buffer(
                "case_embd",
                torch.zeros(self.dim_case_embd, dtype=self.prec, device=device),
                # torch.eye(self.dim_case_embd, dtype=self.prec, device=device)[0],
            )
        else:
            self.case_embd = None

        in_dim = (
            self.dim_descrpt
            + self.numb_fparam
            + (0 if self.use_aparam_as_mask else self.numb_aparam)
            + self.dim_case_embd
        )

        self.filter_layers_coord = NetworkCollection(
            1 if not self.mixed_types else 0,
            self.ntypes,
            network_type="fitting_network",
            networks=[
                FittingNet(
                    in_dim,
                    self.out_dim,
                    self.neuron,
                    self.activation_function,
                    self.resnet_dt,
                    self.precision,
                    bias_out=True,
                    seed=child_seed(self.seed, ii),
                )
                for ii in range(self.ntypes if not self.mixed_types else 1)
            ],
        )

        self.filter_layers_cell = NetworkCollection(
            1 if not self.mixed_types else 0,
            self.ntypes,
            network_type="fitting_network",
            networks=[
                FittingNet(
                    in_dim,
                    6,
                    self.neuron,
                    self.activation_function,
                    self.resnet_dt,
                    self.precision,
                    bias_out=True,
                    seed=child_seed(self.seed, ii),
                )
                for ii in range(self.ntypes if not self.mixed_types else 1)
            ],
        )
        
        # TODO: Type denoise

        # set trainable
        for param in self.parameters():
            param.requires_grad = self.trainable

    def reinit_exclude(
        self,
        exclude_types: list[int] = [],
    ) -> None:
        self.exclude_types = exclude_types
        self.emask = AtomExcludeMask(self.ntypes, self.exclude_types)

    def change_type_map(
        self, type_map: list[str], model_with_new_type_stat=None
    ) -> None:
        """Change the type related params to new ones, according to `type_map` and the original one in the model.
        If there are new types in `type_map`, statistics will be updated accordingly to `model_with_new_type_stat` for these new types.
        """
        assert self.type_map is not None, (
            "'type_map' must be defined when performing type changing!"
        )
        assert self.mixed_types, "Only models in mixed types can perform type changing!"
        remap_index, has_new_type = get_index_between_two_maps(self.type_map, type_map)
        self.type_map = type_map
        self.ntypes = len(type_map)
        self.reinit_exclude(map_atom_exclude_types(self.exclude_types, remap_index))
        if has_new_type:
            extend_shape = [len(type_map), *list(self.bias_atom_e.shape[1:])]
            extend_bias_atom_e = torch.zeros(
                extend_shape,
                dtype=self.bias_atom_e.dtype,
                device=self.bias_atom_e.device,
            )
            self.bias_atom_e = torch.cat([self.bias_atom_e, extend_bias_atom_e], dim=0)
        self.bias_atom_e = self.bias_atom_e[remap_index]

    def output_def(self):
        return FittingOutputDef(
            [
                OutputVariableDef(
                    "strain_components",
                    [6],
                    reducible=True,
                    r_differentiable=False,
                    c_differentiable=False,
                    intensive=True,
                ),
                OutputVariableDef(
                    "updated_coord",
                    [3],
                    reducible=False,
                    r_differentiable=False,
                    c_differentiable=False,
                ),
            ]
        )

    def serialize(self) -> dict:
        """Serialize the fitting to dict."""
        return {
            "@class": "Fitting",
            "@version": 3,
            "ntypes": self.ntypes,
            "out_dim": self.out_dim,
            "dim_descrpt": self.dim_descrpt,
            "neuron": self.neuron,
            "resnet_dt": self.resnet_dt,
            "numb_fparam": self.numb_fparam,
            "numb_aparam": self.numb_aparam,
            "dim_case_embd": self.dim_case_embd,
            "activation_function": self.activation_function,
            "precision": self.precision,
            "mixed_types": self.mixed_types,
            "cell_nets": self.filter_layers_cell.serialize(),
            "coord_nets": self.filter_layers_coord.serialize(),
            "rcond": self.rcond,
            "exclude_types": self.exclude_types,
            "@variables": {
                "bias_atom_e": to_numpy_array(self.bias_atom_e),
                "case_embd": to_numpy_array(self.case_embd),
                "fparam_avg": to_numpy_array(self.fparam_avg),
                "fparam_inv_std": to_numpy_array(self.fparam_inv_std),
                "aparam_avg": to_numpy_array(self.aparam_avg),
                "aparam_inv_std": to_numpy_array(self.aparam_inv_std),
            },
            "type_map": self.type_map,
            # "tot_ener_zero": self.tot_ener_zero ,
            # "trainable": self.trainable ,
            # "atom_ener": self.atom_ener ,
            # "layer_name": self.layer_name ,
            # "spin": self.spin ,
            ## NOTICE:  not supported by far
            "tot_ener_zero": False,
            "trainable": [self.trainable] * (len(self.neuron) + 1),
            "layer_name": None,
            "use_aparam_as_mask": self.use_aparam_as_mask,
            "spin": None,
        }


    def deserialize(self) -> "DenoiseNet":
        data = data.copy()
        variables = data.pop("@variables")
        cell_nets = data.pop("cell_nets")
        coord_nets = data.pop("coord_nets")
        obj = cls(**data)
        for kk in variables.keys():
            obj[kk] = to_torch_tensor(variables[kk])
        obj.filter_layers_cell = NetworkCollection.deserialize(cell_nets)
        obj.filter_layers_coord = NetworkCollection.deserialize(coord_nets)
        return obj

    def get_dim_fparam(self) -> int:
        """Get the number (dimension) of frame parameters of this atomic model."""
        return self.numb_fparam

    def get_dim_aparam(self) -> int:
        """Get the number (dimension) of atomic parameters of this atomic model."""
        return self.numb_aparam
    
    # make jit happy
    exclude_types: list[int]

    def get_sel_type(self) -> list[int]:
        """Get the selected atom types of this model.

        Only atoms with selected atom types have atomic contribution
        to the result of the model.
        If returning an empty list, all atom types are selected.
        """
        # make jit happy
        sel_type: list[int] = []
        for ii in range(self.ntypes):
            if ii not in self.exclude_types:
                sel_type.append(ii)
        return sel_type

    def get_type_map(self) -> list[str]:
        """Get the name to each type of atoms."""
        return self.type_map

    def set_case_embd(self, case_idx: int):
        """
        Set the case embedding of this fitting net by the given case_idx,
        typically concatenated with the output of the descriptor and fed into the fitting net.
        """
        self.case_embd = torch.eye(self.dim_case_embd, dtype=self.prec, device=device)[
            case_idx
        ]

    def __setitem__(self, key, value) -> None:
        if key in ["bias_atom_e"]:
            value = value.view([self.ntypes, 1])
            self.bias_atom_e = value
        elif key in ["fparam_avg"]:
            self.fparam_avg = value
        elif key in ["fparam_inv_std"]:
            self.fparam_inv_std = value
        elif key in ["aparam_avg"]:
            self.aparam_avg = value
        elif key in ["aparam_inv_std"]:
            self.aparam_inv_std = value
        elif key in ["case_embd"]:
            self.case_embd = value
        elif key in ["scale"]:
            self.scale = value
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
        elif key in ["case_embd"]:
            return self.case_embd
        elif key in ["scale"]:
            return self.scale
        else:
            raise KeyError(key)

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
    ) -> dict[str, torch.Tensor]:
        """Based on embedding net output, alculate total energy.

        Args:
        - inputs: Embedding matrix. Its shape is [nframes, natoms[0], self.dim_descrpt].
        - natoms: Tell atom count and element count. Its shape is [2+self.ntypes].

        Returns
        -------
        - `torch.Tensor`: Total energy with shape [nframes, natoms[0]].
        """
        # cast the input to internal precsion
        xx = descriptor.to(self.prec)
        fparam = fparam.to(self.prec) if fparam is not None else None
        aparam = aparam.to(self.prec) if aparam is not None else None

        xx_zeros = None
        nf, nloc, nd = xx.shape

        if nd != self.dim_descrpt:
            raise ValueError(
                f"get an input descriptor of dim {nd},"
                f"which is not consistent with {self.dim_descrpt}."
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
            if xx_zeros is not None:
                xx_zeros = torch.cat(
                    [xx_zeros, fparam],
                    dim=-1,
                )
        # check aparam dim, concate to input descriptor
        if self.numb_aparam > 0 and not self.use_aparam_as_mask:
            assert aparam is not None, "aparam should not be None"
            assert self.aparam_avg is not None
            assert self.aparam_inv_std is not None
            if aparam.shape[-1] != self.numb_aparam:
                raise ValueError(
                    f"get an input aparam of dim {aparam.shape[-1]}, ",
                    f"which is not consistent with {self.numb_aparam}.",
                )
            aparam = aparam.view([nf, -1, self.numb_aparam])
            nb, nloc, _ = aparam.shape
            t_aparam_avg = self._extend_a_avg_std(self.aparam_avg, nb, nloc)
            t_aparam_inv_std = self._extend_a_avg_std(self.aparam_inv_std, nb, nloc)
            aparam = (aparam - t_aparam_avg) * t_aparam_inv_std
            xx = torch.cat(
                [xx, aparam],
                dim=-1,
            )
            if xx_zeros is not None:
                xx_zeros = torch.cat(
                    [xx_zeros, aparam],
                    dim=-1,
                )

        if self.dim_case_embd > 0:
            assert self.case_embd is not None
            case_embd = torch.tile(self.case_embd.reshape([1, 1, -1]), [nf, nloc, 1])
            xx = torch.cat(
                [xx, case_embd],
                dim=-1,
            )
            if xx_zeros is not None:
                xx_zeros = torch.cat(
                    [xx_zeros, case_embd],
                    dim=-1,
                )

        outs = torch.zeros(
            (nf, nloc, 6),
            dtype=self.prec,
            device=descriptor.device,
        )  # jit assertion
        if self.mixed_types:
            # direct coord fitting
            vec_out = self.filter_layers_coord.networks[0](xx)
            assert list(vec_out.size()) == [nf, nloc, self.out_dim]
            # (nf x nloc) x 1 x od
            vec_out = vec_out.view(-1, 1, self.out_dim)
            assert gr is not None
            # (nf x nloc) x od x 3
            gr = gr.view(-1, self.out_dim, 3)
            vec_out = (
                torch.bmm(vec_out, gr).squeeze(-2).view(nf, nloc, 3)
            )  # Shape is [nf, nloc, 3]
            # direct cell fitting
            atom_strain_components = self.filter_layers_cell.networks[0](xx)
            outs = outs + atom_strain_components # Shape is [nframes, natoms[0], 6]
        else:
            vec_out = torch.zeros(
                (nf, nloc, 3),
                dtype=self.prec,
                device=descriptor.device,
            )  # jit assertion
            # direct coord fitting
            for type_i, ll in enumerate(self.filter_layers_coord.networks):
                mask = (atype == type_i).unsqueeze(-1)
                mask = torch.tile(mask, (1, 1, 1))
                vec_out_type = ll(xx)
                assert list(vec_out_type.size()) == [nf, nloc, self.out_dim]
                # (nf x nloc) x 1 x od
                vec_out_type = vec_out_type.view(-1, 1, self.out_dim)
                assert gr is not None
                # (nf x nloc) x od x 3
                gr = gr.view(-1, self.out_dim, 3)
                vec_out_type = (
                    torch.bmm(vec_out_type, gr).squeeze(-2).view(nf, nloc, 3)
                )  # Shape is [nf, nloc, 3]
                vec_out_type = torch.where(mask, vec_out_type, 0.0)
                vec_out = (
                    vec_out + vec_out_type
                )  # Shape is [nframes, natoms[0], 3]
            # direct cell fitting
            for type_i, ll in enumerate(self.filter_layers_cell.networks):
                mask = (atype == type_i).unsqueeze(-1)
                mask = torch.tile(mask, (1, 1, 1))
                atom_strain_components = ll(xx)
                atom_strain_components = torch.where(mask, atom_strain_components, 0.0)
                outs = (
                    outs + atom_strain_components
                )  # Shape is [nframes, natoms[0], 6]
        # nf x nloc
        mask = self.emask(atype).to(torch.bool)
        # nf x nloc x nod
        outs = torch.where(mask[:, :, None], outs, 0.0)
        vec_out = torch.where(mask[:, :, None], vec_out, 0.0)
        return {
            "strain_components": outs.to(env.GLOBAL_PT_FLOAT_PRECISION),
            "updated_coord": vec_out,
        }