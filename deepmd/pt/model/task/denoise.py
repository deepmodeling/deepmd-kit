# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
import logging
import os
import tempfile
from abc import (
    abstractmethod,
)
from typing import (
    Callable,
    List,
    Optional,
    Union,
)

import numpy as np
import torch

from deepmd.infer.deep_eval import (
    DeepEval,
)
from deepmd.pt.model.network.mlp import (
    FittingNet,
    NetworkCollection,
)
from deepmd.pt.model.network.network import (
    ResidualDeep,
)
from deepmd.pt.model.task.base_fitting import (
    BaseFitting,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env import (
    DEFAULT_PRECISION,
    DEVICE,
    PRECISION_DICT,
)
from deepmd.pt.utils.exclude_mask import (
    AtomExcludeMask,
)
from deepmd.pt.utils.utils import (
    to_numpy_array,
    to_torch_tensor,
)
from deepmd.utils.data_system import (
    DeepmdDataSystem,
)
from deepmd.utils.finetune import (
    change_energy_bias_lower,
)
from deepmd.pt.model.task.fitting import (
    Fitting,
)
from deepmd.dpmodel import (
    FittingOutputDef,
    OutputVariableDef,
    fitting_check_output,
)
from deepmd.pt.model.network.network import (
    MaskLMHead,
    NonLinearHead,
)
from deepmd.utils.path import (
    DPPath,
)

dtype = env.GLOBAL_PT_FLOAT_PRECISION
device = env.DEVICE

log = logging.getLogger(__name__)

@Fitting.register("denoise")
class DenoiseFittingNet(Fitting):
    """Construct a denoise fitting net.

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
    trainable : Union[List[bool], bool]
        If the parameters in the fitting net are trainable.
        Now this only supports setting all the parameters in the fitting net at one state.
        When in List[bool], the trainable will be True only if all the boolean parameters are True.
    remove_vaccum_contribution: List[bool], optional
        Remove vaccum contribution before the bias is added. The list assigned each
        type. For `mixed_types` provide `[True]`, otherwise it should be a list of the same
        length as `ntypes` signaling if or not removing the vaccum contribution for the atom types in the list.
    """

    def __init__(
        self,
        ntypes: int,
        dim_descrpt: int,
        embedding_width: int,
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
        trainable: Union[bool, List[bool]] = True,
        remove_vaccum_contribution: Optional[List[bool]] = None,
        **kwargs,
    ):
        super().__init__()
        self.var_name = ["updated_coord","logits"]
        self.ntypes = ntypes
        self.dim_descrpt = dim_descrpt
        self.embedding_width = embedding_width
        self.neuron = neuron
        self.mixed_types = mixed_types
        self.resnet_dt = resnet_dt
        self.numb_fparam = numb_fparam
        self.numb_aparam = numb_aparam
        self.activation_function = activation_function
        self.precision = precision
        self.prec = PRECISION_DICT[self.precision]
        self.rcond = rcond
        # order matters, should be place after the assignment of ntypes
        self.reinit_exclude(exclude_types)
        self.trainable = trainable
        # need support for each layer settings
        self.trainable = (
            all(self.trainable) if isinstance(self.trainable, list) else self.trainable
        )
        self.remove_vaccum_contribution = remove_vaccum_contribution

        # in denoise task, net_dim_out is a list which has 2 elements: [3, ntypes]
        net_dim_out = self._net_out_dim()

        # init constants
        # TODO: actually bias is useless in denoise task
        if bias_atom_e is None:
            bias_atom_e = np.zeros([self.ntypes, net_dim_out[0]], dtype=np.float64)
        bias_atom_e = torch.tensor(bias_atom_e, dtype=self.prec, device=device)
        bias_atom_e = bias_atom_e.view([self.ntypes, net_dim_out[0]])
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

        in_dim_coord = self.embedding_width
        in_dim_logits = self.dim_descrpt + self.numb_fparam + self.numb_aparam

        # the first MLP is used to update coordinate
        self.filter_layers_coord = NetworkCollection(
            1 if not self.mixed_types else 0,
            self.ntypes,
            network_type="fitting_network",
            networks=[
                FittingNet(
                    in_dim_coord,
                    1,
                    self.neuron,
                    self.activation_function,
                    self.resnet_dt,
                    self.precision,
                    bias_out=True,
                )
                for ii in range(self.ntypes if not self.mixed_types else 1)
            ],
        )

        # the second MLP is used to update logits
        self.filter_layers_logits = NetworkCollection(
            1 if not self.mixed_types else 0,
            self.ntypes,
            network_type="fitting_network",
            networks=[
                FittingNet(
                    in_dim_logits,
                    net_dim_out[1],
                    self.neuron,
                    self.activation_function,
                    self.resnet_dt,
                    self.precision,
                    bias_out=True,
                )
                for ii in range(self.ntypes if not self.mixed_types else 1)
            ],
        )

        self.filter_layers_old = None

        if seed is not None:
            torch.manual_seed(seed)
        # set trainable
        for param in self.parameters():
            param.requires_grad = self.trainable

    def reinit_exclude(
        self,
        exclude_types: List[int] = [],
    ):
        self.exclude_types = exclude_types
        self.emask = AtomExcludeMask(self.ntypes, self.exclude_types)

    def serialize(self) -> dict:
        """Serialize the fitting to dict."""
        return {
            "@class": "Fitting",
            "@version": 1,
            "var_name": self.var_name,
            "ntypes": self.ntypes,
            "dim_descrpt": self.dim_descrpt,
            "neuron": self.neuron,
            "resnet_dt": self.resnet_dt,
            "numb_fparam": self.numb_fparam,
            "numb_aparam": self.numb_aparam,
            "activation_function": self.activation_function,
            "precision": self.precision,
            "mixed_types": self.mixed_types,
            "nets": self.filter_layers.serialize(),
            "rcond": self.rcond,
            "exclude_types": self.exclude_types,
            "@variables": {
                "bias_atom_e": to_numpy_array(self.bias_atom_e),
                "fparam_avg": to_numpy_array(self.fparam_avg),
                "fparam_inv_std": to_numpy_array(self.fparam_inv_std),
                "aparam_avg": to_numpy_array(self.aparam_avg),
                "aparam_inv_std": to_numpy_array(self.aparam_inv_std),
            },
            # "tot_ener_zero": self.tot_ener_zero ,
            # "trainable": self.trainable ,
            # "atom_ener": self.atom_ener ,
            # "layer_name": self.layer_name ,
            # "use_aparam_as_mask": self.use_aparam_as_mask ,
            # "spin": self.spin ,
            ## NOTICE:  not supported by far
            "tot_ener_zero": False,
            "trainable": [self.trainable] * (len(self.neuron) + 1),
            "layer_name": None,
            "use_aparam_as_mask": False,
            "spin": None,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "DenoiseFittingNet":
        data = copy.deepcopy(data)
        variables = data.pop("@variables")
        nets = data.pop("nets")
        obj = cls(**data)
        for kk in variables.keys():
            obj[kk] = to_torch_tensor(variables[kk])
        obj.filter_layers = NetworkCollection.deserialize(nets)
        return obj

    def get_dim_fparam(self) -> int:
        """Get the number (dimension) of frame parameters of this atomic model."""
        return self.numb_fparam

    def get_dim_aparam(self) -> int:
        """Get the number (dimension) of atomic parameters of this atomic model."""
        return self.numb_aparam

    # make jit happy
    exclude_types: List[int]

    def get_sel_type(self) -> List[int]:
        """Get the selected atom types of this model.

        Only atoms with selected atom types have atomic contribution
        to the result of the model.
        If returning an empty list, all atom types are selected.
        """
        # make jit happy
        sel_type: List[int] = []
        for ii in range(self.ntypes):
            if ii not in self.exclude_types:
                sel_type.append(ii)
        return sel_type

    def __setitem__(self, key, value):
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
        elif key in ["scale"]:
            return self.scale
        else:
            raise KeyError(key)

    def _net_out_dim(self):
        """Set the FittingNet output dim."""
        return [3, self.ntypes]
        #pass

    def output_def(self):
        return FittingOutputDef(
            [
                OutputVariableDef(
                    "updated_coord",
                    [3],
                    reduciable=False,
                    r_differentiable=False,
                    c_differentiable=False,
                ),
                OutputVariableDef(
                    "logits",
                    [-1],
                    reduciable=False,
                    r_differentiable=False,
                    c_differentiable=False,
                ),
            ]
        )

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
        sw: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
    ):
        xx = descriptor
        nnei = g2.shape[-2]
        if self.remove_vaccum_contribution is not None:
            # TODO: Idealy, the input for vaccum should be computed;
            # we consider it as always zero for convenience.
            # Needs a compute_input_stats for vaccum passed from the
            # descriptor.
            xx_zeros = torch.zeros_like(xx)
        else:
            xx_zeros = None
        nf, nloc, nd = xx.shape
        net_dim_out = self._net_out_dim()

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
            if xx_zeros is not None:
                xx_zeros = torch.cat(
                    [xx_zeros, fparam],
                    dim=-1,
                )
        # check aparam dim, concate to input descriptor
        if self.numb_aparam > 0:
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

        outs_coord = torch.zeros(
            (nf, nloc, net_dim_out[0]),
            dtype=env.GLOBAL_PT_FLOAT_PRECISION,
            device=descriptor.device,
        )  # jit assertion
        outs_logits = torch.zeros(
            (nf, nloc, net_dim_out[1]),
            dtype=env.GLOBAL_PT_FLOAT_PRECISION,
            device=descriptor.device,
        )
        if self.mixed_types:
            atom_updated_coord = ((self.filter_layers_coord.networks[0](g2)) * h2).sum(dim=-2) / (sw.sum(dim=-1).unsqueeze(-1)+1e-6)
            atom_logits = self.filter_layers_logits[0](xx)
            #Is xx_zeros useful in denoise task??????????????
            #if xx_zeros is not None:
            #    atom_property -= self.filter_layers.networks[0](xx_zeros)
            outs_coord = (
                outs_coord + atom_updated_coord
            )  # Shape is [nframes, natoms[0], net_dim_out]
            outs_logits = (
                outs_logits + atom_logits
            )
        # TODO:
        '''
        else:
            for type_i, ll in enumerate(self.filter_layers_coord.networks):
                mask = (atype == type_i).unsqueeze(-1)
                mask = torch.tile(mask, (1, 1, net_dim_out))
                atom_property = ll(xx)
                if xx_zeros is not None:
                    # must assert, otherwise jit is not happy
                    assert self.remove_vaccum_contribution is not None
                    if not (
                        len(self.remove_vaccum_contribution) > type_i
                        and not self.remove_vaccum_contribution[type_i]
                    ):
                        atom_property -= ll(xx_zeros)
                atom_property = atom_property + self.bias_atom_e[type_i]
                atom_property = atom_property * mask
                outs = (
                    outs + atom_property
                )  # Shape is [nframes, natoms[0], net_dim_out]
        '''
        # nf x nloc
        mask = self.emask(atype)
        # nf x nloc x nod
        outs_coord = outs_coord * mask[:, :, None]
        outs_logits = outs_logits * mask[:, :, None]
        return {self.var_name[0]: outs_coord.to(env.GLOBAL_PT_FLOAT_PRECISION),
                self.var_name[1]: outs_logits.to(env.GLOBAL_PT_FLOAT_PRECISION)}

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
        pass

@fitting_check_output
class DenoiseNet(Fitting):
    def __init__(
        self,
        feature_dim,
        ntypes,
        attn_head=8,
        prefactor=[0.5, 0.5],
        activation_function="gelu",
        **kwargs,
    ):
        """Construct a denoise net.

        Args:
        - ntypes: Element count.
        - embedding_width: Embedding width per atom.
        - neuron: Number of neurons in each hidden layers of the fitting net.
        - bias_atom_e: Average enery per atom for each element.
        - resnet_dt: Using time-step in the ResNet construction.
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.ntypes = ntypes
        self.attn_head = attn_head
        self.prefactor = torch.tensor(
            prefactor, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE
        )

        self.lm_head = MaskLMHead(
            embed_dim=self.feature_dim,
            output_dim=ntypes,
            activation_fn=activation_function,
            weight=None,
        )

        if not isinstance(self.attn_head, list):
            self.pair2coord_proj = NonLinearHead(
                self.attn_head, 1, activation_fn=activation_function
            )
        else:
            self.pair2coord_proj = []
            self.ndescriptor = len(self.attn_head)
            for ii in range(self.ndescriptor):
                _pair2coord_proj = NonLinearHead(
                    self.attn_head[ii], 1, activation_fn=activation_function
                )
                self.pair2coord_proj.append(_pair2coord_proj)
            self.pair2coord_proj = torch.nn.ModuleList(self.pair2coord_proj)

    def output_def(self):
        return FittingOutputDef(
            [
                OutputVariableDef(
                    "updated_coord",
                    [3],
                    reduciable=False,
                    r_differentiable=False,
                    c_differentiable=False,
                ),
                OutputVariableDef(
                    "logits",
                    [-1],
                    reduciable=False,
                    r_differentiable=False,
                    c_differentiable=False,
                ),
            ]
        )

    def forward(
        self,
        pair_weights,
        diff,
        nlist_mask,
        features,
        sw,
        masked_tokens: Optional[torch.Tensor] = None,
    ):
        """Calculate the updated coord.
        Args:
        - coord: Input noisy coord with shape [nframes, nloc, 3].
        - pair_weights: Input pair weights with shape [nframes, nloc, nnei, head].
        - diff: Input pair relative coord list with shape [nframes, nloc, nnei, 3].
        - nlist_mask: Input nlist mask with shape [nframes, nloc, nnei].

        Returns
        -------
        - denoised_coord: Denoised updated coord with shape [nframes, nloc, 3].
        """
        # [nframes, nloc, nnei, 1]
        logits = self.lm_head(features, masked_tokens=masked_tokens)
        if not isinstance(self.attn_head, list):
            attn_probs = self.pair2coord_proj(pair_weights)
            out_coord = (attn_probs * diff).sum(dim=-2) / (
                sw.sum(dim=-1).unsqueeze(-1) + 1e-6
            )
        else:
            assert len(self.prefactor) == self.ndescriptor
            all_coord_update = []
            assert len(pair_weights) == len(diff) == len(nlist_mask) == self.ndescriptor
            for ii in range(self.ndescriptor):
                _attn_probs = self.pair2coord_proj[ii](pair_weights[ii])
                _coord_update = (_attn_probs * diff[ii]).sum(dim=-2) / (
                    nlist_mask[ii].sum(dim=-1).unsqueeze(-1) + 1e-6
                )
                all_coord_update.append(_coord_update)
            out_coord = self.prefactor[0] * all_coord_update[0]
            for ii in range(self.ndescriptor - 1):
                out_coord += self.prefactor[ii + 1] * all_coord_update[ii + 1]
        return {
            "updated_coord": out_coord,
            "logits": logits,
        }
