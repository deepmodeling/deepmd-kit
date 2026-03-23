# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import Any, Optional, Union
from abc import abstractmethod

import numpy as np
import torch

from deepmd.dpmodel import (
    FittingOutputDef,
    OutputVariableDef,
)
from deepmd.dpmodel.utils.seed import (
    child_seed,
)
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
from deepmd.pt.utils.exclude_mask import (
    AtomExcludeMask,
)
from deepmd.pt.utils.utils import (
    to_numpy_array,
    to_torch_tensor,
)
from deepmd.utils.finetune import (
    get_index_between_two_maps,
    map_atom_exclude_types,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

dtype = env.GLOBAL_PT_FLOAT_PRECISION
device = env.DEVICE

@Fitting.register("lr")
class LRFittingNet(Fitting):
    """Construct a general sr+lr interactions fitting net.

    Parameters
    ----------
    var_name : str
        The atomic property to fit.
    ntypes : int
        Element count.
    dim_descrpt : int
        Embedding width per atom.
    dim_out_sr : int
        The output dimension of the sr fitting net.
    dim_out_lr : int
        The output dimension of the lr fitting net.
    neuron_sr : list[int]
        Number of neurons in each hidden layers of the sr fitting net.
    neuron_lr : list[int]
        Number of neurons in each hidden layers of the lr fitting net.
    bias_atom_e : torch.Tensor, optional
        Average energy per atom for each element.
    resnet_dt : bool
        Using time-step in the ResNet construction.
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
    remove_vaccum_contribution: list[bool], optional
        Remove vacuum contribution before the bias is added. The list assigned each
        type. For `mixed_types` provide `[True]`, otherwise it should be a list of the same
        length as `ntypes` signaling if or not removing the vacuum contribution for the atom types in the list.
    type_map: list[str], Optional
        A list of strings. Give the name to each type of atoms.
    use_aparam_as_mask: bool
        If True, the aparam will not be used in fitting net for embedding.
    default_fparam: list[float], optional
        The default frame parameter. If set, when `fparam.npy` files are not included in the data system,
        this value will be used as the default value for the frame parameter in the fitting net.
    """

    def __init__(
        self,
        var_name: str,
        ntypes: int,
        dim_descrpt: int,
        dim_out_sr: int,
        dim_out_lr: int,
        neuron_sr: list[int] = [128, 128, 128],
        neuron_lr: list[int] = [128, 128, 128],
        bias_atom_e: Optional[torch.Tensor] = None,
        resnet_dt: bool = True,
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
        remove_vaccum_contribution: Optional[list[bool]] = None,
        type_map: Optional[list[str]] = None,
        use_aparam_as_mask: bool = False,
        default_fparam: Optional[list[float]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.var_name = var_name
        self.ntypes = ntypes
        self.dim_descrpt = dim_descrpt
        self.dim_out_sr = dim_out_sr
        self.dim_out_lr = dim_out_lr
        self.neuron_sr = neuron_sr
        self.neuron_lr = neuron_lr
        self.mixed_types = mixed_types
        self.resnet_dt = resnet_dt
        self.numb_fparam = numb_fparam
        self.numb_aparam = numb_aparam
        self.default_fparam = default_fparam
        self.dim_case_embd = dim_case_embd
        self.activation_function = activation_function
        self.precision = precision
        self.prec = PRECISION_DICT[self.precision]
        self.rcond = rcond
        self.seed = seed
        self.type_map = type_map
        self.use_aparam_as_mask = use_aparam_as_mask
        self.reinit_exclude(exclude_types)
        self.trainable = trainable
        # need support for each layer settings
        self.trainable = (
            all(self.trainable) if isinstance(self.trainable, list) else self.trainable
        )
        self.remove_vaccum_contribution = remove_vaccum_contribution

        self.sr_net_dim_out = self._sr_net_out_dim()
        self.lr_net_dim_out = self._lr_net_out_dim()
        # init constants
        if bias_atom_e is None:
            bias_atom_e = np.zeros([self.ntypes, self.sr_net_dim_out], dtype=np.float64)
        bias_atom_e = torch.tensor(
            bias_atom_e, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=device
        )
        bias_atom_e = bias_atom_e.view([self.ntypes, self.sr_net_dim_out])
        if not self.mixed_types:
            assert self.ntypes == bias_atom_e.shape[0], "Element count mismatches!"
        self.register_buffer("bias_atom_e", bias_atom_e)

        if self.numb_fparam > 0:
            self.register_buffer(
                "fparam_avg",
                torch.zeros(self.numb_fparam, dtype=self.prec, device=env.DEVICE),
            )
            self.register_buffer(
                "fparam_inv_std",
                torch.ones(self.numb_fparam, dtype=self.prec, device=env.DEVICE),
            )
        else:
            self.fparam_avg, self.fparam_inv_std = None, None
        if self.numb_aparam > 0:
            self.register_buffer(
                "aparam_avg",
                torch.zeros(self.numb_aparam, dtype=self.prec, device=env.DEVICE),
            )
            self.register_buffer(
                "aparam_inv_std",
                torch.ones(self.numb_aparam, dtype=self.prec, device=env.DEVICE),
            )
        else:
            self.aparam_avg, self.aparam_inv_std = None, None

        if self.dim_case_embd > 0:
            self.register_buffer(
                "case_embd",
                torch.zeros(self.dim_case_embd, dtype=self.prec, device=env.DEVICE),
            )
        else:
            self.case_embd = None

        if self.default_fparam is not None:
            if self.numb_fparam > 0:
                assert len(self.default_fparam) == self.numb_fparam, (
                    "default_fparam length mismatch!"
                )
            self.register_buffer(
                "default_fparam_tensor",
                torch.tensor(
                    np.array(self.default_fparam),
                    dtype=self.prec,
                    device=env.DEVICE,
                ),
            )
        else:
            self.default_fparam_tensor = None

        in_dim = (
            self.dim_descrpt
            + self.numb_fparam
            + (0 if self.use_aparam_as_mask else self.numb_aparam)
            + self.dim_case_embd
        )

        net_count = self.ntypes if not self.mixed_types else 1
        self.filter_layers_lr = NetworkCollection(
            1 if not self.mixed_types else 0,
            self.ntypes,
            network_type="fitting_network",
            networks=[
                FittingNet(
                    in_dim,
                    self.lr_net_dim_out,
                    self.neuron_lr,
                    self.activation_function,
                    self.resnet_dt,
                    self.precision,
                    bias_out=True,
                    seed=child_seed(self.seed, ii * 2),
                    trainable=self.trainable,
                )
                for ii in range(net_count)
            ],
        )
        self.filter_layers_sr = NetworkCollection(
            1 if not self.mixed_types else 0,
            self.ntypes,
            network_type="fitting_network",
            networks=[
                FittingNet(
                    in_dim,
                    self.sr_net_dim_out,
                    self.neuron_sr,
                    self.activation_function,
                    self.resnet_dt,
                    self.precision,
                    bias_out=True,
                    seed=child_seed(self.seed, ii * 2 + 1),
                    trainable=self.trainable,
                )
                for ii in range(net_count)
            ],
        )

        for param in self.parameters():
            param.requires_grad = self.trainable

        self.eval_return_middle_output = False

    def reinit_exclude(
        self,
        exclude_types: list[int] = [],
    ) -> None:
        self.exclude_types = exclude_types
        self.emask = AtomExcludeMask(self.ntypes, self.exclude_types)

    def change_type_map(
        self,
        type_map: list[str],
        model_with_new_type_stat: Optional["LRFittingNet"] = None,
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

    def serialize(self) -> dict:
        """Serialize the fitting to dict."""
        return {
            "@class": "LRFitting",
            "@version": 1,
            "var_name": self.var_name,
            "ntypes": self.ntypes,
            "dim_descrpt": self.dim_descrpt,
            "dim_out_sr": self.dim_out_sr,
            "dim_out_lr": self.dim_out_lr,
            "neuron_sr": self.neuron_sr,
            "neuron_lr": self.neuron_lr,
            "resnet_dt": self.resnet_dt,
            "numb_fparam": self.numb_fparam,
            "numb_aparam": self.numb_aparam,
            "dim_case_embd": self.dim_case_embd,
            "default_fparam": self.default_fparam,
            "activation_function": self.activation_function,
            "precision": self.precision,
            "mixed_types": self.mixed_types,
            "nets_sr": self.filter_layers_sr.serialize(),
            "nets_lr": self.filter_layers_lr.serialize(),
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
            "trainable_sr": [self.trainable] * (len(self.neuron_sr) + 1),
            "trainable_lr": [self.trainable] * (len(self.neuron_lr) + 1),
            "layer_name": None,
            "use_aparam_as_mask": self.use_aparam_as_mask,
            "spin": None,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "LRFittingNet":
        data = data.copy()
        variables = data.pop("@variables")
        nets_sr = data.pop("nets_sr")
        nets_lr = data.pop("nets_lr")
        obj = cls(**data)
        for kk in variables.keys():
            obj[kk] = to_torch_tensor(variables[kk])
        obj.filter_layers_sr = NetworkCollection.deserialize(nets_sr)
        obj.filter_layers_lr = NetworkCollection.deserialize(nets_lr)
        return obj

    def get_dim_fparam(self) -> int:
        """Get the number (dimension) of frame parameters of this atomic model."""
        return self.numb_fparam

    def has_default_fparam(self) -> bool:
        """Check if the fitting has default frame parameters."""
        return self.default_fparam is not None

    def get_default_fparam(self) -> Optional[torch.Tensor]:
        return self.default_fparam_tensor

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

    def set_case_embd(self, case_idx: int) -> None:
        """
        Set the case embedding of this fitting net by the given case_idx,
        typically concatenated with the output of the descriptor and fed into the fitting net.
        """
        self.case_embd = torch.eye(self.dim_case_embd, dtype=self.prec, device=device)[
            case_idx
        ]

    def set_return_middle_output(self, return_middle_output: bool = True) -> None:
        self.eval_return_middle_output = return_middle_output

    def __setitem__(self, key: str, value: torch.Tensor) -> None:
        if key in ["bias_atom_e"]:
            value = value.view([self.ntypes, self._net_out_dim()])
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
        elif key in ["default_fparam_tensor"]:
            self.default_fparam_tensor = value
        else:
            raise KeyError(key)

    def __getitem__(self, key: str) -> torch.Tensor:
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
        elif key in ["default_fparam_tensor"]:
            return self.default_fparam_tensor
        else:
            raise KeyError(key)

    def _sr_net_out_dim(self) -> int:
        """Set the SRFittingNet output dim."""
        return self.dim_out_sr
    
    def _lr_net_out_dim(self) -> int:
        """Set the LR FittingNet output dim."""
        return self.dim_out_lr
    
    def _extend_f_avg_std(self, xx: torch.Tensor, nb: int) -> torch.Tensor:
        return torch.tile(xx.view([1, self.numb_fparam]), [nb, 1])

    def _extend_a_avg_std(self, xx: torch.Tensor, nb: int, nloc: int) -> torch.Tensor:
        return torch.tile(xx.view([1, 1, self.numb_aparam]), [nb, nloc, 1])

    def _forward_common(
        self,
        descriptor: torch.Tensor,
        atype: torch.Tensor,
        gr: Optional[torch.Tensor] = None,
        g2: Optional[torch.Tensor] = None,
        h2: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        xx = descriptor.to(self.prec)
        nf, nloc, nd = xx.shape

        if self.numb_fparam > 0 and fparam is None:
            assert self.default_fparam_tensor is not None
            fparam = torch.tile(self.default_fparam_tensor.unsqueeze(0), [nf, 1])

        fparam = fparam.to(self.prec) if fparam is not None else None
        aparam = aparam.to(self.prec) if aparam is not None else None

        if self.remove_vaccum_contribution is not None:
            xx_zeros = torch.zeros_like(xx)
        else:
            xx_zeros = None

        if nd != self.dim_descrpt:
            raise ValueError(
                f"get an input descriptor of dim {nd},"
                f"which is not consistent with {self.dim_descrpt}."
            )

        if self.numb_fparam > 0:
            assert fparam is not None, "fparam should not be None"
            assert self.fparam_avg is not None
            assert self.fparam_inv_std is not None
            if fparam.shape[-1] != self.numb_fparam:
                raise ValueError(
                    f"get an input fparam of dim {fparam.shape[-1]}, "
                    f"which is not consistent with {self.numb_fparam}."
                )
            fparam = fparam.view([nf, self.numb_fparam])
            nb, _ = fparam.shape
            t_fparam_avg = self._extend_f_avg_std(self.fparam_avg, nb)
            t_fparam_inv_std = self._extend_f_avg_std(self.fparam_inv_std, nb)
            fparam = (fparam - t_fparam_avg) * t_fparam_inv_std
            fparam = torch.tile(fparam.reshape([nf, 1, -1]), [1, nloc, 1])
            xx = torch.cat([xx, fparam], dim=-1)
            if xx_zeros is not None:
                xx_zeros = torch.cat([xx_zeros, fparam], dim=-1)

        if self.numb_aparam > 0 and not self.use_aparam_as_mask:
            assert aparam is not None, "aparam should not be None"
            assert self.aparam_avg is not None
            assert self.aparam_inv_std is not None
            if aparam.shape[-1] != self.numb_aparam:
                raise ValueError(
                    f"get an input aparam of dim {aparam.shape[-1]}, "
                    f"which is not consistent with {self.numb_aparam}."
                )
            aparam = aparam.view([nf, -1, self.numb_aparam])
            nb, nloc, _ = aparam.shape
            t_aparam_avg = self._extend_a_avg_std(self.aparam_avg, nb, nloc)
            t_aparam_inv_std = self._extend_a_avg_std(self.aparam_inv_std, nb, nloc)
            aparam = (aparam - t_aparam_avg) * t_aparam_inv_std
            xx = torch.cat([xx, aparam], dim=-1)
            if xx_zeros is not None:
                xx_zeros = torch.cat([xx_zeros, aparam], dim=-1)

        if self.dim_case_embd > 0:
            assert self.case_embd is not None
            case_embd = torch.tile(self.case_embd.reshape([1, 1, -1]), [nf, nloc, 1])
            xx = torch.cat([xx, case_embd], dim=-1)
            if xx_zeros is not None:
                xx_zeros = torch.cat([xx_zeros, case_embd], dim=-1)

        results: dict[str, torch.Tensor] = {}
        sr_out = self._apply_networks(
            self.filter_layers_sr,
            self.neuron_sr,
            self.sr_net_dim_out,
            xx,
            xx_zeros,
            atype,
            middle_output=results,
            bool_bias=True,
        )
        lr_out = self._apply_networks(
            self.filter_layers_lr,
            self.neuron_lr,
            self.lr_net_dim_out,
            xx,
            xx_zeros,
            atype,
            middle_output=results,
        )
        mask = self.emask(atype).to(torch.bool)
        sr_out = torch.where(mask[:, :, None], sr_out, 0.0)
        lr_out = torch.where(mask[:, :, None], lr_out, 0.0)
        results.update({"sr": sr_out, "lr": lr_out})
        return results
    
    def _apply_networks(
        self,
        layers: NetworkCollection,
        neuron: list[int],
        dim_out: int,
        xx: torch.Tensor,
        xx_zeros: Optional[torch.Tensor],
        atype: torch.Tensor,
        middle_output: Optional[dict[str, torch.Tensor]],
        bool_bias: bool = False,
    ) -> torch.Tensor:
        nf, nloc, _ = xx.shape
        outs = torch.zeros((nf, nloc, dim_out), dtype=self.prec, device=xx.device)
        if self.mixed_types:
            atom_property = layers.networks[0](xx)
            if self.eval_return_middle_output and middle_output is not None:
                middle_output["middle_output"] = layers.networks[0].call_until_last(
                    xx
                )
            if xx_zeros is not None:
                atom_property -= layers.networks[0](xx_zeros)
            outs = outs + atom_property
        else:
            if self.eval_return_middle_output and middle_output is not None:
                outs_middle = torch.zeros(
                    (nf, nloc, neuron[-1]),
                    dtype=self.prec,
                    device=xx.device,
                )
                for type_i, ll in enumerate(layers.networks):
                    mask = (atype == type_i).unsqueeze(-1)
                    mask = torch.tile(mask, (1, 1, dim_out))
                    middle_output_type = ll.call_until_last(xx)
                    middle_output_type = torch.where(
                        torch.tile(mask, (1, 1, neuron[-1])),
                        middle_output_type,
                        0.0,
                    )
                    outs_middle = outs_middle + middle_output_type
                middle_output["middle_output"] = outs_middle
            for type_i, ll in enumerate(layers.networks):
                mask = (atype == type_i).unsqueeze(-1)
                mask = torch.tile(mask, (1, 1, dim_out))
                atom_property = ll(xx)
                if xx_zeros is not None:
                    assert self.remove_vaccum_contribution is not None
                    if not (
                        len(self.remove_vaccum_contribution) > type_i
                        and not self.remove_vaccum_contribution[type_i]
                    ):
                        atom_property -= ll(xx_zeros)
                if bool_bias:
                    atom_property = atom_property + self.bias_atom_e[type_i].to(self.prec)
                else:
                    atom_property = atom_property
                atom_property = torch.where(mask, atom_property, 0.0)
                outs = outs + atom_property
        return outs