# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
import logging
import os
import tempfile
from abc import (
    abstractmethod,
)
from typing import (
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
from deepmd.utils.version import (
    check_version_compatibility,
)

dtype = env.GLOBAL_PT_FLOAT_PRECISION
device = env.DEVICE

log = logging.getLogger(__name__)


class Fitting(torch.nn.Module, BaseFitting):
    # plugin moved to BaseFitting

    def __new__(cls, *args, **kwargs):
        if cls is Fitting:
            return BaseFitting.__new__(BaseFitting, *args, **kwargs)
        return super().__new__(cls)

    def share_params(self, base_class, shared_level, resume=False):
        """
        Share the parameters of self to the base_class with shared_level during multitask training.
        If not start from checkpoint (resume is False),
        some seperated parameters (e.g. mean and stddev) will be re-calculated across different classes.
        """
        assert (
            self.__class__ == base_class.__class__
        ), "Only fitting nets of the same type can share params!"
        if shared_level == 0:
            # link buffers
            if hasattr(self, "bias_atom_e"):
                self.bias_atom_e = base_class.bias_atom_e
            # the following will successfully link all the params except buffers, which need manually link.
            for item in self._modules:
                self._modules[item] = base_class._modules[item]
        elif shared_level == 1:
            # only not share the bias_atom_e
            # the following will successfully link all the params except buffers, which need manually link.
            for item in self._modules:
                self._modules[item] = base_class._modules[item]
        else:
            raise NotImplementedError

    def change_energy_bias(
        self,
        config,
        model,
        old_type_map: List[str],
        new_type_map: List[str],
        bias_shift="delta",
        ntest=10,
    ):
        """Change the energy bias according to the input data and the pretrained model.

        Parameters
        ----------
        config : Dict
            The configuration.
        model : EnergyModel
            Energy model loaded pre-trained model.
        new_type_map : List[str]
            The original type_map in dataset, they are targets to change the energy bias.
        old_type_map : List[str]
            The full type_map in pretrained model
        bias_shift : str
            The mode for changing energy bias : ['delta', 'statistic']
            'delta' : perform predictions on energies of target dataset,
                    and do least sqaure on the errors to obtain the target shift as bias.
            'statistic' : directly use the statistic energy bias in the target dataset.
        ntest : int
            The number of test samples in a system to change the energy bias.
        """
        log.info(
            "Changing energy bias in pretrained model for types {}... "
            "(this step may take long time)".format(str(new_type_map))
        )
        # data
        systems = config["training"]["training_data"]["systems"]
        finetune_data = DeepmdDataSystem(
            systems=systems,
            batch_size=config["training"]["training_data"].get("batch_size", "auto"),
            test_size=1,
        )
        finetune_data.add("energy", ndof=1, atomic=False, must=True, high_prec=True)
        model = torch.jit.script(model)
        if model.get_dim_fparam() > 0:
            finetune_data.add("fparam", model.get_dim_fparam(), atomic=False, must=True)
        if model.get_dim_aparam() > 0:
            finetune_data.add("aparam", model.get_dim_aparam(), atomic=True, must=True)
        tmp_model = tempfile.NamedTemporaryFile(delete=False, suffix=".pth")
        torch.jit.save(model, tmp_model.name)
        dp = DeepEval(tmp_model.name)
        os.unlink(tmp_model.name)
        bias = change_energy_bias_lower(
            finetune_data,
            dp,
            new_type_map,
            old_type_map,
            self.bias_atom_e.detach().cpu().numpy().reshape(-1),
            bias_shift=bias_shift,
            ntest=ntest,
        )
        self.bias_atom_e = (
            torch.from_numpy(bias)
            .type_as(self.bias_atom_e)
            .reshape(self.bias_atom_e.shape)
            .to(DEVICE)
        )


class GeneralFitting(Fitting):
    """Construct a general fitting net.

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
        var_name: str,
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
        rcond: Optional[float] = None,
        seed: Optional[int] = None,
        exclude_types: List[int] = [],
        trainable: Union[bool, List[bool]] = True,
        remove_vaccum_contribution: Optional[List[bool]] = None,
        **kwargs,
    ):
        super().__init__()
        self.var_name = var_name
        self.ntypes = ntypes
        self.dim_descrpt = dim_descrpt
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

        net_dim_out = self._net_out_dim()
        # init constants
        if bias_atom_e is None:
            bias_atom_e = np.zeros([self.ntypes, net_dim_out], dtype=np.float64)
        bias_atom_e = torch.tensor(bias_atom_e, dtype=self.prec, device=device)
        bias_atom_e = bias_atom_e.view([self.ntypes, net_dim_out])
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

        in_dim = self.dim_descrpt + self.numb_fparam + self.numb_aparam

        self.old_impl = kwargs.get("old_impl", False)
        if self.old_impl:
            filter_layers = []
            for type_i in range(self.ntypes if not self.mixed_types else 1):
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
                1 if not self.mixed_types else 0,
                self.ntypes,
                network_type="fitting_network",
                networks=[
                    FittingNet(
                        in_dim,
                        net_dim_out,
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
    def deserialize(cls, data: dict) -> "GeneralFitting":
        data = copy.deepcopy(data)
        check_version_compatibility(data.pop("@version", 1), 1, 1)
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

    def get_sel_type(self) -> List[int]:
        """Get the selected atom types of this model.

        Only atoms with selected atom types have atomic contribution
        to the result of the model.
        If returning an empty list, all atom types are selected.
        """
        return []

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

    @abstractmethod
    def _net_out_dim(self):
        """Set the FittingNet output dim."""
        pass

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
    ):
        xx = descriptor
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
            if xx_zeros is not None:
                xx_zeros = torch.cat(
                    [xx_zeros, aparam],
                    dim=-1,
                )

        outs = torch.zeros(
            (nf, nloc, net_dim_out),
            dtype=env.GLOBAL_PT_FLOAT_PRECISION,
            device=descriptor.device,
        )  # jit assertion
        if self.old_impl:
            assert self.filter_layers_old is not None
            assert xx_zeros is None
            if self.mixed_types:
                atom_property = self.filter_layers_old[0](xx) + self.bias_atom_e[atype]
                outs = outs + atom_property  # Shape is [nframes, natoms[0], 1]
            else:
                for type_i, filter_layer in enumerate(self.filter_layers_old):
                    mask = atype == type_i
                    atom_property = filter_layer(xx)
                    atom_property = atom_property + self.bias_atom_e[type_i]
                    atom_property = atom_property * mask.unsqueeze(-1)
                    outs = outs + atom_property  # Shape is [nframes, natoms[0], 1]
        else:
            if self.mixed_types:
                atom_property = (
                    self.filter_layers.networks[0](xx) + self.bias_atom_e[atype]
                )
                if xx_zeros is not None:
                    atom_property -= self.filter_layers.networks[0](xx_zeros)
                outs = (
                    outs + atom_property
                )  # Shape is [nframes, natoms[0], net_dim_out]
            else:
                for type_i, ll in enumerate(self.filter_layers.networks):
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
        # nf x nloc
        mask = self.emask(atype)
        # nf x nloc x nod
        outs = outs * mask[:, :, None]
        return {self.var_name: outs.to(env.GLOBAL_PT_FLOAT_PRECISION)}
